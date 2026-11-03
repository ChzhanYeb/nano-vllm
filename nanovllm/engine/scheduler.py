"""
Scheduler：决定"每一轮（iteration）到底算哪些序列"。

nano-vllm 的调度策略刻意做得简单，规则只有三条：

  1. **prefill 优先**：先把等待队列里的 prompt 尽量塞满一个 batch 的 token 预算；
     只要挑到了序列，本轮就整批走 prefill 路径；
  2. prefill 一条都挑不出来时，才做 **decode**（每条运行中的序列各算 1 个 token）；
  3. decode 时如果 KV Cache 的物理页不够了，就 **抢占（preempt）** 队尾的序列，
     把它退回等待队列，稍后重新 prefill（代价是那段 KV 要重算）。

"prefill 与 decode 不混在一个 batch 里"是它比 vLLM 简化掉的地方：
换来的是实现简单、kernel 不需要区分两种模式，
代价是长 prompt 的 prefill 会拖延同批 decode 的响应时间。
"""
from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        # 调度约束：一个 batch 最多多少 token、最多多少条序列
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        # 结束符 token id（由 LLMEngine 从 tokenizer 里取到后回填）
        self.eos = config.eos
        # 页大小（与 BlockManager、Sequence 保持一致）
        self.block_size = config.kvcache_block_size
        # KV Cache 的分配器
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        # 两个队列：等待 prefill 的序列 + 正在 decode 的序列
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        """两个队列都空了，说明所有请求都处理完了（generate 的循环条件）。"""
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """新请求入队（只会进入 waiting，不直接进 running）。"""
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        挑选本轮要计算的序列。

        返回 (序列列表, is_prefill)：
          - is_prefill=True ：本轮做 prefill（可能包含多条序列、也可能是 chunked prefill 的一块）
          - is_prefill=False：本轮做 decode（每条序列 1 个 token）
        """
        scheduled_seqs = []
        num_batched_tokens = 0        # 本轮已累计的 token 数（预算 = max_num_batched_tokens）

        # ================= 第一优先级：prefill =================
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]                     # 只看队首，保证 FIFO 公平
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break                                 # token 预算已经用完
            if not seq.block_table:
                # 这条序列还没有分配过 KV Cache（首次调度，或者刚被抢占过）
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    # 空闲页不够。队首都排不下，后面的请求更排不下，直接结束 prefill 阶段
                    break
                # 还需要计算的 token = 总长度 - 命中前缀缓存的部分
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                # 已经分配过页表（chunked prefill 的后续块）：只需要算还没缓存的部分
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                # 剩余预算装不下整条 prompt，而本轮已经有别的序列了 → 收工。
                # 只有"本轮的第一条序列"才允许被切成小块（chunked prefill）分多轮算，
                # 否则一条超长 prompt 会把排在后面的短请求全部饿死。
                break
            if not seq.block_table:
                # 真正分配 KV Cache（命中的前缀缓存块会被共享进来，引用计数 +1）
                self.block_manager.allocate(seq, num_cached_blocks)
            # 本轮给这条序列分配：min(它还需要算的 token 数, 剩余预算)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                # 这条序列的 prompt 已经全部处理完 → 转入 RUNNING，并搬到运行队列
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True               # 有 prefill 可做，本轮就是 prefill

        # ================= 第二优先级：decode =================
        # 这里不需要判断 token 预算：每条序列只算 1 个 token，永远远小于 max_num_batched_tokens
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            # while ... else：只有当 while 条件自然变为 False（而不是被 break）时才执行 else。
            # 语义是"一直抢占别人，直到腾出足够空间让这条序列能继续 append"。
            while not self.block_manager.can_append(seq):
                if self.running:
                    # 还有别的序列可以牺牲：抢占队尾（最近加入运行队列的那条）
                    self.preempt(self.running.pop())
                else:
                    # 只剩它自己，且仍然放不下 → 连自己也抢占掉，本轮不再处理它
                    self.preempt(seq)
                    break
            else:
                # 空间够了：本轮它只算 1 个 token，并切换到 decode 模式
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)     # 必要时为这个新 token 分配新页
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        # 把本轮选中的序列放回运行队列的头部，并且保持相对顺序：
        # extendleft 是"逐个插到最左边"，效果上等于逆序，所以先用 reversed 还原
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        """
        抢占：把一条正在运行的序列退回等待队列。

        为什么需要它：显存里的物理页总数是固定的，当运行的序列变多（或变长）时，
        可能没有空闲页给新 token 用了。这时必须有人让位。

        代价：被抢占的序列会释放自己占用的页（KV 内容随之丢弃），
        之后重新调度时要再走一次 prefill —— 相当于把它的 prompt 重算一遍。
        所以这是一种"用计算换显存"的取舍。
        """
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)      # 插到队首：让被抢占的序列尽快重新被调度

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        """
        处理一轮前向的结果：登记缓存 → 更新计数 → 追加 token → 判断是否结束。

        seqs      ：本轮参与计算的序列
        token_ids ：每条序列采样出的新 token（只有 rank 0 会真正算出来）
        is_prefill：本轮是 prefill 还是 decode
        """
        for seq, token_id in zip(seqs, token_ids):
            # 1. 先把"刚刚写满的完整块"登记进前缀缓存表，后续请求就能复用它们
            self.block_manager.hash_blocks(seq)
            # 2. 本轮调度的 token 已经算完并写进 KV Cache，累加到"已缓存"计数
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                # chunked prefill：这条序列的 prompt 还没算完（本轮只是其中一块）。
                # 此时采样出来的 token 是"中间过程"的产物，必须丢弃，
                # 也不能计入 completion 长度，直接进入下一轮继续算 prefill。
                continue
            # 3. 把新 token 追加到序列末尾（下一轮 decode 就以它为输入）
            seq.append_token(token_id)
            # 4. 结束条件：采样到 EOS（且用户没要求忽略它），或者已经生成够 max_tokens 个 token
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)    # 立刻把显存页还给空闲队列
                self.running.remove(seq)              # 并从运行队列里摘掉
