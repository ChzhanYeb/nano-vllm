"""
BlockManager：KV Cache 的"分页内存管理器"（PagedAttention 的核心）。

要解决的问题：
  显存里预留了一大片 KV Cache，被切成固定大小的"页"（block，默认 256 个 token）。
  每条序列按需拿页、用完归还，类似操作系统的虚拟内存 + 页表：

      逻辑块号（序列自己看到的连续空间）  --block_table-->  物理块号（显存里的真实位置）

  好处是不要求每个序列的 KV 在显存里连续存放，于是显存碎片极少（浪费不超过 1 页）。

额外能力：前缀缓存（prefix caching）
  对所有"写满的块"按内容算哈希，存进 hash_to_block_id 表。
  如果新请求的前缀和之前某条请求完全一样（例如相同的 system prompt），
  就直接复用那些物理块——引用计数 +1，省掉一次 prefill 计算。

本文件还负责共享安全：同一块可能被多条序列引用，所以每块都有 ref_count，
只有减到 0 才真正归还到空闲队列。
"""
from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    """KV Cache 的一个物理页：能存 block_size 个 token 的 K/V。"""

    def __init__(self, block_id):
        self.block_id = block_id          # 物理块号（也就是它在 KV Cache 张量第 3 维上的下标）
        self.ref_count = 0                # 引用计数：多条序列可以共享同一块（前缀缓存）
        self.hash = -1                    # 内容哈希；-1 表示"未登记"（内容还会变）
        self.token_ids = []               # 该块内的 token，用于哈希冲突时的精确校验

    def update(self, hash: int, token_ids: list[int]):
        """登记块内容。块被写满、KV 已经落进显存之后才会调用（见 hash_blocks）。"""
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """把块重置为"刚分配给某条序列、内容待写入"的状态。"""
        self.ref_count = 1                # 分配出去就立刻有 1 个引用者
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        """
        num_blocks：一共多少页（由 ModelRunner 按实测剩余显存算出来，见 config.num_kvcache_blocks）
        block_size：每页装多少个 token
        """
        self.block_size = block_size
        # 所有物理页对象；初始化时全部空闲
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # 前缀缓存表：内容哈希 -> 物理块号
        self.hash_to_block_id: dict[int, int] = dict()
        # 空闲块队列：deque 是为了 O(1) 地从头部取块、从尾部归还块
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        # 已占用块集合：set 是为了 O(1) 判断"某块是否正在被使用"
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """
        计算一个块的内容哈希。

        用 xxhash64：非加密哈希，速度非常快、碰撞率足够低。

        prefix 是"前一个块的哈希"：把它混进来就构成链式哈希，
        于是这个块的哈希同时编码了"本块内容"和"它之前的所有内容"，
        这样不同序列只要前缀相同，逐块的哈希就都会相同，可以直接命中缓存。
        """
        h = xxhash.xxh64()
        if prefix != -1:
            # 把 64 位前缀哈希按小端序转成 8 字节再喂进去
            h.update(prefix.to_bytes(8, "little"))
        # token id 列表转成字节序列；注意 num_tokens 必须是同一个类型/字节序，
        # 否则同一段文本在不同情况下会算出不同哈希（因此这里统一走 numpy 的表示）
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        """从空闲队列里取一个块出来（内部方法，调用方负责把它挂到序列的页表上）。"""
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0        # 空闲块必须没有被引用
        # 这个块以前可能被登记进前缀缓存表：如果表里还指向它，必须清掉，
        # 否则后来的请求会命中一个"内容马上要被覆盖"的块，读出错误的 KV。
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()                      # ref_count=1，hash/token_ids 清空
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        """把一个块归还给空闲队列（只在引用计数减到 0 时调用）。"""
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        """
        判断现在能不能给 seq 的 prompt 分配 KV Cache，并顺便算出前缀缓存能命中几块。

        返回值：
            -1  ：空闲块不够，本次不能调度这条序列（Scheduler 会在这里停下来）
            >=0 ：可以分配，数值 = 命中的完整块数（这些块的 K/V 已在显存里，无需重算）
        """
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks        # 最坏情况：所有块都要新分配
        # 只检查到 num_blocks - 1：最后一块通常没填满，内容还会变，不参与前缀匹配
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            # 链式哈希：h 把前面所有块的内容也编码进来
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            # 查不到（-1）或者哈希撞了但内容不同，说明前缀只能匹配到这里，后面的都不能复用
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            # 该块已经被别人占用时，直接共享即可，不需要额外消耗空闲块；
            # 如果它是"空闲但哈希命中"的块，就还需要把它占用下来，所以不减 num_new_blocks
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1                          # 空闲块不足
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        """
        真正建立页表：前 num_cached_blocks 块复用缓存内容，其余全新分配。

        num_cached_blocks 来自 can_allocate 的返回值。
        """
        assert not seq.block_table             # 只应在序列第一次拿到资源时调用
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                # 已被别的序列占用：只需引用计数 +1，两个序列共享同一份 KV 显存
                block.ref_count += 1
            else:
                # 空闲但内容命中：把它标记为占用，并从空闲队列里摘掉
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        # 剩余部分全新分配
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        # 命中的这些 token 已经算过了，直接记为"已缓存"，
        # 于是本次 prefill 只需要算剩下的部分（这就是前缀缓存的收益）
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        """
        释放序列占用的全部块（从后往前遍历）。

        顺序本身不影响正确性：每个块都是独立地做一次引用计数递减。
        这里从后往前遍历，只是与 allocate() 的填充顺序相反，读起来更自然。
        引用计数减到 0 的块才真正回到空闲队列；被别的序列共享的块还要继续留着。
        """
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        # 清空页表与缓存计数：被抢占（preempt）的序列稍后要重新走一遍 prefill
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """
        decode 每步只新增 1 个 token，判断"这一步是否需要一个新的块"。

        关键观察：只有当 len(seq) % block_size == 1 时才需要新块——
        表示上一块刚好写满，这一步的 token 将落在下一块的第一个槽位。
        其余情况都写进当前最后一块的剩余槽位，不需要新块。

        括号里的比较结果是 bool，参与 >= 比较时 True 当成 1、False 当成 0，
        所以整句读作"空闲块数 >= (需要 1 块 ? 1 : 0)"。
        """
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """真正为新 token 分配块（判断条件必须与 can_append 完全一致）。"""
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        """
        把"本次刚刚算完的完整块"登记进前缀缓存表，供后续请求复用。

        调用时机：模型前向完成、K/V 已经写进显存之后（Scheduler.postprocess 里第一步）。
        只有写满的块才登记，因为只有它的内容不会再变了。
        """
        # 本次新增 token 覆盖到的块区间 [start, end)：
        # 用"已缓存 token 数"和"本次调度 token 数"分别除以页大小得到
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return                # 没有新增的完整块，直接返回
        # 链式哈希的起点是前一块的哈希（start=0 说明这是序列的第一块）
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            # 记录块内容（供冲突校验），并登记到全局哈希表
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
