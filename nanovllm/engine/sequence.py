"""
序列（Sequence）模块：一条请求在引擎内部的全部状态。

一条 Sequence 记录了：
- token_ids    ：prompt + 已经生成的 token
- 采样参数      ：temperature / max_tokens / ignore_eos
- 调度状态      ：WAITING / RUNNING / FINISHED
- KV Cache 信息 ：block_table（逻辑块 -> 物理块）、num_cached_tokens 等

另外要留意文件末尾的 __getstate__ / __setstate__：张量并行下，序列要被 pickle
后通过共享内存发给子进程，这里做了"只传必要信息"的优化。
"""
from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """序列在调度器里的三种状态。"""

    WAITING = auto()     # 在等待队列里：还没拿到 KV Cache 资源，或刚被抢占（preempt）
    RUNNING = auto()     # 正在运行：prompt 已全部处理完，进入逐 token 解码（decode）阶段
    FINISHED = auto()    # 已结束：生成了 EOS，或达到了 max_tokens


class Sequence:
    # 类属性：KV Cache 的分页大小。
    # 由 LLMEngine 启动时按 Config.kvcache_block_size 统一设置（写在类上而不是实例上，
    # 是因为一个引擎里所有序列的这个值都一样，而实例数量很多）。
    block_size = 256
    # 全局自增计数器：给每条序列分配唯一 id（next() 是原子的，天然线程安全）
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        """
        token_ids      ：prompt 的 token id 列表
        sampling_params：这条序列的采样参数（温度、最大生成长度、是否忽略 EOS）
        """
        # 唯一 id：对外返回结果时用它排序，也方便日志排查
        self.seq_id = next(Sequence.counter)
        # 新序列一律先进入等待状态
        self.status = SequenceStatus.WAITING
        # copy：不能和调用方共享同一个 list（外面还持有 prompt，后面还要用）
        self.token_ids = copy(token_ids)
        # 最后一个 token：decode 阶段每步只需要它作为模型输入
        self.last_token = token_ids[-1]
        # 当前总长度 = prompt + 已生成
        self.num_tokens = len(self.token_ids)
        # prompt 的长度，用来区分 prompt 与 completion
        self.num_prompt_tokens = len(token_ids)
        # 已经写进 KV Cache 并"登记为可缓存"的 token 数（前缀缓存命中数 + 历次调度累加）
        self.num_cached_tokens = 0
        # 本轮调度分配给这条序列的 token 数：
        # prefill 时可能是整段 prompt，也可能被 max_num_batched_tokens 截成一块（chunked prefill）
        self.num_scheduled_tokens = 0
        # 是否还需要 prefill；进入 decode 阶段会被置为 False
        self.is_prefill = True
        # 逻辑块号 -> 物理块号 的列表（PagedAttention 的页表），由 BlockManager 填充
        self.block_table = []
        # 把采样参数"摊平"成三个字段，避免每个序列都持有一整个 SamplingParams 对象
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        """len(seq) 就是序列当前的 token 总数（被 BlockManager 大量使用）。"""
        return self.num_tokens

    def __getitem__(self, key):
        """下标/切片访问：seq[i]、seq[start:end] 都直接作用于 token_ids。"""
        return self.token_ids[key]

    @property
    def is_finished(self):
        """是否已结束（Scheduler.is_finished / generate 的循环条件会用到）。"""
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        """已经生成了多少个 token（不含 prompt），用于判断是否达到 max_tokens。"""
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        """只取 prompt 部分（对外返回时可能用到）。"""
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        """只取生成部分 —— LLMEngine.step() 返回给用户的就是它。"""
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        """
        这条序列一共需要多少个 KV Cache 页。

        向上取整公式 (a + b - 1) // b：例如 300 个 token、页大小 256，
        需要 ceil(300/256) = 2 块。
        """
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        """
        最后一块里已经装了多少个 token（1..block_size）。

        用途：decode 时新 token 要写进"最后一块的第 last_block_num_tokens 个槽位"。
        """
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        """
        取出第 i 个逻辑块覆盖的 token id 列表（长度通常是 block_size）。

        注意：前缀缓存只对"整块"做哈希，所以最后一块（可能不满）不参与匹配。
        """
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        """
        把新采样出来的 token 追加到序列末尾。

        这是 decode 阶段每步都会做的事：长度 +1，并更新 last_token
        （下一个 iteration 的模型输入就是这个 last_token）。
        """
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        """
        自定义 pickle 序列化行为（张量并行时会用到）。

        rank 0 需要把本步要计算的序列通过共享内存 pickle 给其他 rank 的
        ModelRunner。如果每次都把整份 token_ids 传过去，通信量会随序列长度线性增长。

        优化：decode 阶段每条序列每步只新增 1 个 token，
        而模型前向只需要最后那个 token 作为输入，所以只传 last_token；
        prefill 阶段则需要完整的 token_ids，于是传 list。
        """
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)

    def __setstate__(self, state):
        """与 __getstate__ 对应：在子进程里把拿到的状态还原成一个可用的 Sequence。"""
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state
        if isinstance(last_state, list):
            # prefill：拿到的是完整 token_ids
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            # decode：只拿到 last_token。子进程只负责算模型前向，
            # 不需要完整的 token 列表（解码文本在 rank 0 做），所以这里留空列表。
            self.token_ids = []
            self.last_token = last_state
