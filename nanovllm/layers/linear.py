"""
张量并行（Tensor Parallelism, TP）线性层。

一张卡放不下大模型时，vLLM/Megatron 风格的经典做法是把每个线性层切成两种形态，
两种配合使用，就能让一个 Transformer 块只付出**一次 AllReduce** 的通信代价：

1. ColumnParallelLinear —— 按**输出维**切：每张卡算一部分输出，本身不需要通信；
2. RowParallelLinear    —— 按**输入维**切：每张卡算出一部分和，最后 AllReduce 相加。

（本文件里的 MergedColumnParallelLinear / QKVParallelLinear 都是 Column 并行的特化版本。）

另外，每个线性层都实现了 weight_loader：读取 checkpoint 时，各 rank 只取属于
自己的那一片权重。于是同一份模型文件可以被任意 TP 度复用，加载逻辑也只有一份。
"""
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    """
    整除助手：TP 切分必须能均分。

    这里直接 assert 而不是静默取整，是为了让配置错误在启动时就暴露，
    而不是跑出莫名其妙的数值错误。
    """
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):
    """所有并行线性层的基类：负责建参数、把 weight_loader 挂到参数上、记录并行信息。"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        """
        input_size / output_size：**本 rank 上**这个层的形状（子类通常在调用前已除过 tp_size）
        bias                   ：是否带偏置
        tp_dim                 ：权重在哪个维度上被切分（0=输出维，1=输入维；None=不切分）
        """
        super().__init__()
        self.tp_dim = tp_dim
        # 当前进程（GPU）的编号与总并行度，由 ModelRunner 里的 init_process_group 决定
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        # 参数形状 [output_size, input_size]，与 F.linear 的权重布局一致
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        # 关键技巧：把加载函数"挂"在参数对象上。
        # 这样 utils/loader.py 拿到参数后，直接调 param.weight_loader(...) 就能按 rank 分片加载，
        # 不需要在加载器里写一堆 if/else 判断参数属于哪种层。
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            # 显式注册为 None，保证 state_dict / named_parameters 行为一致
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """由子类实现（不同并行方式的前向逻辑不同）。"""
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    """不切分的普通线性层：每张卡都保存**完整**权重（用于小矩阵，或不便切分的地方）。"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """完整拷贝即可（每张卡的内容都一样）。"""
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):
    """
    列并行：把权重沿**输出维**（dim 0）切成 tp_size 份。

    输入 x 每张卡都有完整的，输出各卡只持有自己那一部分（"列"）。
    好处：这一步不需要任何通信；坏处：输出被切开了，
    所以它通常与下面的 RowParallelLinear 配对（一个切、一个合）。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        # 本 rank 只保存 output_size / tp_size 行；tp_dim=0 表示"按输出维切"
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """从完整权重里取出属于本 rank 的那一段（第 tp_rank 片）。"""
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)          # 本 rank 该拿多少行
        start_idx = self.tp_rank * shard_size              # 起始偏移
        # narrow(dim, start, length)：在 tp_dim 上取连续的一段
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """直接算本卡的局部结果，不需要通信。"""
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    """
    "合并"的列并行层：一次矩阵乘同时算出多个逻辑上独立的投影。

    典型例子是 MLP 的 gate_proj 与 up_proj：checkpoint 里是两个独立张量，
    这里把它们拼成一个大权重一次算完——少一次 kernel 启动，也少读一遍 hidden_states。

    output_sizes 描述每一段有多宽，例如 [intermediate_size, intermediate_size]。
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        # 总输出宽 = 各段之和；交给父类再按 tp_size 去切
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        """
        loaded_shard_id 指明"这次加载的是第几段"（例如 0=gate_proj，1=up_proj）。

        因为整条权重是分段拼起来的，所以目标偏移要先把前面各段长度累加、
        再除以 tp_size（因为父类已经按输出维切过）。
        """
        param_data = param.data
        # 本段在"合并后的完整权重"里的起点（除 tp_size 是因为本地参数已被切小）
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        # 本段切给当前 rank 的大小
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        # 把 param_data 缩窄成"本段属于本 rank 的那一块"
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        # 再从 checkpoint 的完整权重里，取本段属于本 rank 的那一片
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):
    """
    QKV 的列并行实现：把 q_proj / k_proj / v_proj 三个投影合并成一个矩阵。

    输出被切成三段，每段宽度由头数决定：
      q: num_heads    * head_size
      k: num_kv_heads * head_size
      v: num_kv_heads * head_size
    GQA（分组查询注意力）下 kv 头数小于 q 头数，所以三段并不等宽。
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        # 不传 kv 头数时退化为 MHA（q、k、v 头数相同）
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        # 本 rank 分到的 q 头数 / kv 头数
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        # 合并权重的总输出宽度：(q + k + v) 三段头数之和 × 每头维度
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        """loaded_shard_id 取 "q" / "k" / "v"，决定这次加载到合并权重的哪一段。"""
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0                                   # q 在最前面
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size      # k 紧跟在 q 后面
        else:
            shard_size = self.num_kv_heads * self.head_size
            # v 排在 q、k 之后
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):
    """
    行并行：把权重沿**输入维**（dim 1）切成 tp_size 份。

    输入 x 已经是被上一层（通常是 Column 并行）切好的局部结果，
    所以每张卡算出来的只是"部分和"，最后必须 AllReduce 才能得到完整结果。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        # 输入维除以 tp_size；tp_dim=1 表示"按输入维切"
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """
        注意 weight 与 bias 的差别：

        - weight（二维）：按输入维切分，各 rank 拿不同的列；
        - bias（一维）   ：输出维没有被切分，所以每张卡都要完整的一份。
        """
        param_data = param.data
        if param_data.ndim == 1:
            # 一维参数就是 bias，不做切分
            param_data.copy_(loaded_weight)
            return
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        先算局部部分和，再 AllReduce 累加。

        注意 bias 只在 rank 0 上加一次：因为 AllReduce 是"求和"，
        如果每张卡都加一遍 bias，结果就会被加 tp_size 次。
        """
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
