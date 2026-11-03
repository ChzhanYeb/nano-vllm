import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,    # 整个模型的词表大小 (vocab_size)
        embedding_dim: int,     # 每个 token 对应的向量维度 (hidden_size)
    ):
        super().__init__()

        # 1. 获取当前进程（GPU）的排名和总进程数
        self.tp_rank = dist.get_rank()          # 例如：0, 1, 2, 3
        self.tp_size = dist.get_world_size()    # 例如：4

        # 2. 确保词表大小能被 GPU 数量整除
        assert num_embeddings % self.tp_size == 0        
        self.num_embeddings = num_embeddings

        # 3. 计算每张卡需要存储多少个词
        # 例如：总词表 32000 / 4张卡 = 每张卡存 8000 个词
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size

        # 4. 计算当前这张卡负责的“起始索引”和“结束索引”
        # Rank 0: [0, 8000)
        # Rank 1: [8000, 16000) ...以此类推
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition

        # 5. 定义权重参数
        # 形状是 (8000, embedding_dim)，而不是完整的 (32000, embedding_dim)
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))

        # 6. 绑定权重加载器（用于从 checkpoint 加载模型时切片）
        self.weight.weight_loader = self.weight_loader

    '''
        权重加载逻辑
    '''
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data             # 当前 GPU 上的空参数块 (8000, dim)
        shard_size = param_data.size(0)     # 分片大小 (8000)

        # 1. 计算在完整大权重中的起始位置
        start_idx = self.tp_rank * shard_size

        # 2. 从完整权重 loaded_weight 中切出对应片段
        # narrow(0, start, len) 表示在第0维从start开始取len个
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)

        # 3. 将切好的数据复制到当前 GPU 的参数中
        param_data.copy_(loaded_weight)

    '''
        前向传播与并行查表
    '''
    def forward(self, x: torch.Tensor):
        # x 是输入的 token ids，形状通常是 (batch_size, seq_len)

        if self.tp_size > 1:
            # 1. 构建掩码：判断输入的这个 token id 是否落在当前卡负责的范围内
            #   mask 是一个布尔矩阵，True 表示“这个词归我管”，False 表示“不归我管”
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            # 2. 调整索引
            # 如果我的范围是 [8000, 16000)，而输入是 9000
            # 我需要把它变成 1000 (9000 - 8000)，才能在我的局部权重表里查到
            # 对于不归我管的词（mask为False），这里减完可能是负数或无关值，但后面会被 mask 掉
            x = mask * (x - self.vocab_start_idx)

        # 3. 局部查表
        # 在当前 GPU 的局部权重 self.weight 中查找
        # 如果 tp_size=1，直接查；如果 >1，查的是调整过索引的 x
        y = F.embedding(x, self.weight)

        if self.tp_size > 1:
            # 4. 应用掩码清零
            # 对于那些“不归我管”的词，刚才查出来的结果其实是错误的（或者是随机值），必须把对应位置变成全 0 向量
            # mask.unsqueeze(1) 是为了匹配 embedding 输出的维度 (batch, seq, hidden)
            y = mask.unsqueeze(1) * y

            # 5. All-Reduce 求和
            # GPU 0 算出了它负责的那部分词的 Embedding（其他位置是0）
            # GPU 1 算出了它负责的那部分词的 Embedding（其他位置是0）
            # 通过 all_reduce(SUM)，大家把结果加起来，就得到了完整的、正确的 Embedding 矩阵
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,    # 词表大小 vocab_size
        embedding_dim: int,     # hidden_size
        bias: bool = False,     # 是否带 bias
    ):
        assert not bias         # 这里强制要求不带 bias，因为大多数 LLM 的 LM Head 就是纯线性投影

        # 复用父类的 self.weight，形状为 (vocab_per_partition, embedding_dim)
        # 这就是典型的 tie_word_embeddings：embedding 和 lm_head 共享同一张权重表
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        context = get_context()
        # 从全局上下文里拿到当前这一步的推理元信息
        # nano-vllm 用全局 context 避免每层都传一堆额外参数

        if context.is_prefill:
            # 当前处于 prefill 阶段：用户 prompt 被一次性前向传播
            # 但 LM Head 只需要对每条序列最后一个 token 打分，用来采样下一个 token

            last_indices = context.cu_seqlens_q[1:] - 1
            # cu_seqlens_q 是变长 batch 的序列边界前缀和
            # 例如 [0, 5, 9] 表示 batch 里有两条序列，长度分别是 5 和 4
            # cu_seqlens_q[1:] 得到 [5, 9]，再 -1 得到 [4, 8]
            # 也就是每条序列最后一个 token 在打平张量里的下标

            x = x[last_indices].contiguous()
            # 只保留每条序列最后一个位置的 hidden state
            # .contiguous() 保证索引后的张量内存连续，避免后续 linear 出错

        logits = F.linear(x, self.weight)
        # 做线性投影：hidden → logits
        # self.weight 形状是 (vocab_per_partition, embedding_dim)
        # F.linear 实际计算的是 x @ weight.T
        # 所以每张卡先算出“自己负责的那部分词表”的 logits
        
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
