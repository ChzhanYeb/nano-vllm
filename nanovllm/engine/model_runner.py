"""
ModelRunner：真正"跑模型"的那一层，每个 TP rank 一个实例。

它负责四件大事：
1. **分布式环境**：初始化 NCCL 进程组，rank 0 通过共享内存把命令广播给其他 rank；
2. **显存规划**：先用一个假 batch 跑一次 warmup（顺便触发 torch.compile 编译），
   再按"实测剩余显存"算出 KV Cache 能开多少页（config.num_kvcache_blocks）；
3. **输入打包**：把调度器给的一批 Sequence 拼成模型要的张量
   （input_ids、positions、cu_seqlens、slot_mapping、block_tables……）；
4. **执行与加速**：prefill 走 eager/编译路径，decode 走 CUDA Graph 重放，
   最后交给 Sampler 采样。

为什么 rank 0 在"主进程"里？因为采样只在 rank 0 做、结果也由主进程返回给用户，
放在同一进程里可以省掉一次跨进程通信。
"""
import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        """
        config：全局配置（含 hf_config、并行度、页大小等）
        rank  ：当前进程的 GPU 序号；0 号跑在主进程里
        event ：rank > 0 时是一个 Event（用来被 rank 0 唤醒）；
                rank 0 时是一个 Event 列表（每个 worker 一个，用于逐个唤醒）
        """
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        # 初始化 NCCL 进程组。地址写死成 localhost:2333 —— 说明 nano-vllm 只支持单机多卡，
        # 首卡负责建连（这行必须在所有 rank 上执行）。
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        # 把当前进程绑定到对应的 GPU 上（之后所有 tensor 默认都在这张卡上）
        torch.cuda.set_device(rank)
        # 下面两行是个实用技巧：临时把默认 dtype 设成模型的 dtype（如 bf16）、
        # 默认设备设成 cuda，这样构造模型时就不用给每个参数写 .to(device)。
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        # 建模型 + 把 checkpoint 权重灌进去（每个 rank 只加载自己负责的分片）
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        # warmup：触发编译 + 分配激活峰值显存，为后面算 KV Cache 页数做准备
        self.warmup_model()
        # 按实测剩余显存开辟 KV Cache，并把 cache 切片注入每一层 Attention
        self.allocate_kv_cache()
        if not self.enforce_eager:
            # 捕获 CUDA Graph（只有 decode 阶段会用到）
            self.capture_cudagraph()
        # 恢复默认设备/精度，避免影响后续代码（例如 tokenizer 相关张量默认落在 CPU）
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # ---------------- 张量并行的命令通道：共享内存 + Event ----------------
        if self.world_size > 1:
            if rank == 0:
                # rank 0 创建一块 1MB 的共享内存（"nanovllm" 是全局名字，多实例会冲突）
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()          # 等所有 worker attach 上去再继续
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                # worker rank 从此进入"待命循环"，__init__ 永远不会返回；
                # 它自己不主动做任何事，只等 rank 0 通过共享内存下命令。
                self.loop()

    def exit(self):
        """
        退出清理：释放共享内存、CUDA Graph，同步设备，销毁进程组。

        rank 0 负责 unlink（真正删除共享内存段），其他 rank 只 close（解除映射）。
        """
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            # 释放 CUDA Graph 对象与它们共用的显存池
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        """
        worker 进程的主循环（只有 rank > 0 会进来）：

        反复"从共享内存读一条命令 → 执行 → 等待下一条"，
        直到收到 "exit"。所以主进程每调用一次 call()，worker 就执行一次对应方法。
        """
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        """
        worker 侧：从共享内存读一条命令。

        协议很简单：[0:4] 是 4 字节小端长度，[4:4+n] 是 pickle 后的数据。
        """
        assert self.world_size > 1 and self.rank > 0
        # 阻塞等待 rank 0 发来的"有新命令"信号
        self.event.wait()
        # 前 4 字节是数据长度
        n = int.from_bytes(self.shm.buf[0:4], "little")
        # 后面是 [方法名, 参数...] 的 pickle 包
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        # 清掉事件，等待下一次唤醒
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        """
        rank 0 侧：把一条命令写进共享内存，并叫醒所有 worker。

        为什么用共享内存而不是消息队列？因为要传的参数里有 Sequence 列表
        （每次都不同、大小可变），pickle 到一段固定内存里最简单直接。
        """
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        # 先写长度（4 字节），再写数据本体
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        # 逐个 set 事件，把 worker 从 wait() 里唤醒（self.event 此时是 Event 列表）
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        """
        统一的"远程调用"入口：

        rank 0 调用时，先广播给所有 worker，然后自己也执行一遍（本地函数调用）；
        worker 通过 loop() 调用时，world_size/rank 条件不成立，于是只执行本地方法。
        """
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        # 用字符串反射出方法（例如 "run" -> self.run），再执行
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        """
        预热：用一个"尽可能大"的假 batch 跑一次 prefill。

        目的有两个：
        1. **触发编译**：模型里大量使用 @torch.compile 与 Triton kernel，
           第一次执行要现场编译（很慢）；先跑一次，真实请求就不用等编译了。
        2. **确定显存峰值**：这次前向会把激活值、cuBLAS workspace 等
           "运行期才需要"的显存真正分配出来，
           这样 allocate_kv_cache 里读到的 peak 才是可信的（否则 KV Cache 会开太多，
           真跑起来反而 OOM）。
        """
        torch.cuda.empty_cache()                 # 先归还缓存块，让下面统计更干净
        torch.cuda.reset_peak_memory_stats()     # 峰值统计清零
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        # 构造"填满预算"的假输入：优先用长序列，长度取 max_model_len 与预算的较小值
        seq_len = min(max_num_batched_tokens, max_model_len)
        # 能塞几条：受 token 预算与 max_num_seqs 双重限制
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        # 走一次 prefill 路径（这里不关心结果，只为"热身 + 占峰值"）
        self.run(seqs, True)
        torch.cuda.empty_cache()                 # 再把缓存块还回去，避免影响后续显存统计

    def allocate_kv_cache(self):
        """
        按"实测剩余显存"决定 KV Cache 开多少页，并把 cache 注入每一层 Attention。

        显存账本是本文件最容易看晕的地方，拆开看：

            total * gpu_memory_utilization        ：允许占用的显存上限
            used  = total - free                  ：当前进程已占用（含缓存分配器预留的块）
            peak  = allocated_bytes.all.peak      ：历史峰值"已分配"字节数
            current = allocated_bytes.all.current ：当前"已分配"字节数

            可用给 KV Cache 的 =
                total * util - used - (peak - current)

        最后那个 (peak - current) 是**为激活值预留的余量**：
        运行时激活值最大会长到 peak，而此刻只占用了 current，
        差额必须留出来，否则真跑起来就会 OOM。
        """
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()                      # 整卡的空闲/总显存
        used = total - free                                          # 整卡已被占用的显存（含本进程与其他进程）
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]     # 峰值已分配
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]  # 当前已分配
        # 张量并行会把 KV 头按卡切分，所以每卡只需要存 num_key_value_heads / world_size 个头
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        # head_dim 优先取 config 里的显式值，取不到就用 hidden_size / num_attention_heads 推
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        # 一个"页"在**所有层**上的字节数：
        #   2（K 和 V）× 层数 × 页大小（token 数）× kv 头数 × 每头维度 × 每元素字节数
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        # 可用字节数 // 每页字节数 = 能开多少页；结果回填到 config，Scheduler 建 BlockManager 时要用
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        # 至少要有 1 页，否则连一条最小请求都跑不了（通常是显存太小或利用率设太高）
        assert config.num_kvcache_blocks > 0
        # 一次性开好整块显存：[K/V, 层, 页号, 页内 token, kv 头, 每头维度]
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        # 按模型里模块的顺序，把第 layer_id 层对应的 K/V 视图注入到该层的 Attention 里。
        # 切片是"视图"而非拷贝：各层共享同一块底层显存，但索引互不重叠。
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        """
        把各序列的页表（block_table）打包成一个二维张量。

        难点：每条序列的页数不同，张量必须等长 —— 所以用 -1 补齐到最长的那条。
        -1 在 kernel 里表示"无效页，不要读"。

        pin_memory=True + non_blocking=True 是标准优化：
        主机内存固定在物理内存中（不会被换出），于是 GPU 可以用 DMA 异步拷贝，
        拷贝期间 CPU 可以继续准备下一批数据。
        """
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        """
        把一批待 prefill 的序列打包成模型输入（还有一堆注意力 kernel 需要的元信息）。

        这里是全项目最需要耐心的一段，几个概念先对齐：
        - **打平（flatten）**：变长 batch 不做 padding，而是把所有 token 首尾相接成一条长序列，
          再用 cu_seqlens（前缀和）标出每条序列的边界，FlashAttention 靠它区分序列；
        - **num_cached_tokens**：该序列开头已经有 num_cached_tokens 个 token 的 K/V 在缓存里
          （前缀缓存命中），所以本次只需要算从它开始的部分；
        - **slot_mapping**：本次算出的每个 token 的 K/V 该写进 KV Cache 的哪个物理槽位。
        """
        input_ids = []
        positions = []
        # cu_seqlens：前缀和数组，首元素恒为 0；q 是"本次要算的 token"，k 是"含缓存的全部 token"
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            # start：跳过已经缓存的前缀；seqlen_q：本次调度要算多少 token
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            # K/V 侧要覆盖到 end（历史 + 本次），所以长度是 end
            seqlen_k = end
            # 只把需要计算的这段 token 喂给模型
            input_ids.extend(seq[start:end])
            # 位置编码的位置就是它在序列里的下标
            positions.extend(range(start, end))
            # 累加前缀和，得到每条序列在打平张量里的边界
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                # warmup 时的假序列没有页表，跳过 slot_mapping 计算
                continue
            # 本次要写的槽位覆盖哪些物理块：从 start 所在的块到 end-1 所在的块
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                # 该物理块在 KV Cache 里的起点
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    # 第一块可能从中间开始（前面的 token 已缓存），要加上块内偏移
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    # 不是最后一块：整块都要写，到块尾为止
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    # 最后一块：只写到本次结束的位置（块内偏移 = end - i*block_size）
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            # K 比 Q 长，说明有历史 K/V 需要从缓存里读 —— 这时必须把页表传给 kernel
            block_tables = self.prepare_block_tables(seqs)
        # 统一转成 pinned + cuda 张量（注意索引类张量用 int32，token id 用 int64）
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        # 写进全局上下文，模型内部的 Attention 会 get_context() 取用
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        """
        decode 阶段的输入打包：每条序列只喂 1 个 token（上一轮生成的那个）。

        比 prefill 简单得多：不需要 cu_seqlens（每条序列长度都是 1），
        但需要 context_lens 告诉 kernel "该序列历史有多长"。
        """
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            # 喂进去的就是序列的最后一个 token
            input_ids.append(seq.last_token)
            # 它的位置下标是 len(seq) - 1
            positions.append(len(seq) - 1)
            # 历史总长度（KV Cache 里已有的 token 数）——Attention 靠它知道该读多长
            context_lens.append(len(seq))
            # 这个新 token 的 K/V 要写在最后一块的最后一个已用槽位的下一个位置：
            # 块号 × 页大小 + 块内已有 token 数 - 1
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        # decode 一定要按页表去读历史 K/V
        block_tables = self.prepare_block_tables(seqs)
        # 注意 decode 不需要 cu_seqlens，所以只传关键字参数
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        """把每条序列的温度收集成一个张量（每条序列可以有不同的采样温度）。"""
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """
        跑模型主干并算出 logits。

        两条路径：
        1. **eager 路径**：prefill、用户显式禁用 CUDA Graph、或 batch 太大（>512）时，
           老老实实前向一次；
        2. **CUDA Graph 路径**（只用于 decode）：
           把输入拷进预先分配好的静态缓冲区，然后 `graph.replay()` 重放整段计算。
           重放期间没有 Python 开销、没有 kernel launch 开销，对小 batch 的 decode 提升明显。

        注意 CUDA Graph 路径返回的是 graph 里固定的 outputs 缓冲区切片，
        logits 是在 graph 之外用 compute_logits 算的（LM Head 要做 TP gather，
        不方便放进 graph，而且采样也不需要进 graph）。
        """
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            # CUDA Graph 是按固定的 batch size 捕获的，只能"向上取整"到已捕获的档位：
            # 例如 bs=5 就重放 bs=8 的 graph，多出来的位置由 slot_mapping=-1 忽略掉
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            # 把真实输入拷进静态缓冲区（graph 捕获时记录的就是这些地址）
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            # slot_mapping 先整体填 -1（表示"这些位置不写 KV Cache"），再覆盖前 bs 个
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            # context_lens 同理：先清零，再写真实值
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            # 页表按实际宽度拷进静态缓冲区的对应列
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            # 重放：一次调用就把整个前向的 kernel 序列重新执行一遍
            graph.replay()
            # 只取前 bs 行，剩下的行是补齐用的垃圾数据
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """
        一轮完整执行：打包输入 → 前向 → 采样 → 清理上下文。

        返回每条序列采样出的 token id 列表（**只有 rank 0 返回真正的值**，
        其他 rank 返回 None —— 因为只有 rank 0 拿到完整词表的 logits）。
        """
        # 按阶段打包输入（同时把元信息写进全局 context）
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        # 温度只有 rank 0 需要（采样只在 rank 0 做）
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        # 采样：torch 张量转成 Python list，方便跨进程 pickle 回主进程
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        # 清空全局上下文，避免残留状态污染下一轮
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        """
        预捕获 CUDA Graph（decode 专用）。

        原理：decode 阶段每次的计算图完全一样（kernel 序列、张量形状都固定），
        只是输入数据不同。于是可以把整段前向"录"下来，之后用 replay() 直接重放，
        省掉每步几十上百次 kernel launch 的 CPU 开销。

        但 graph 的形状是固定的，所以这里做**档位（bucketing）**：
        只捕获 1、2、4、8、16、32…… 这些 batch size，
        实际 bs 落在两档之间时就用更大的那档，多出来的行用 -1 / 0 填掉。
        """
        config = self.config
        hf_config = config.hf_config
        # 最大档位：受 max_num_seqs 与 512 双重限制（再大就没必要用 graph 了）
        max_bs = min(self.config.max_num_seqs, 512)
        # 一条序列最多占多少页（按最大长度算），决定页表缓冲区的宽度
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        # 下面是 graph 的"静态输入缓冲区"：整个生命周期只分配一次，
        # 捕获与重放用的都是这些地址（CUDA Graph 本质上就是录下对这些地址的操作）
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        # 档位表：小 batch 单独列（1/2/4/8），之后每 16 一个档
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        # 所有 graph 共用一个显存池：不共享的话每个档位都要独占一份内存，很快就 OOM
        self.graph_pool = None

        # 从大到小捕获：先建最大的 graph，并拿它的 pool 作为公共池
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            # 用静态缓冲区的切片设置上下文（切片是视图，地址就是静态缓冲区的地址）
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            # 先跑一次热身：让 torch.compile / cuBLAS 在该形状下先完成初始化与内存分配，
            # 否则第一次捕获会把"分配内存"的动作也录进去，重放时会出问题
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            # 开始录制：从这一刻起的 CUDA 操作都被记录进 graph
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                # 第一个（也就是最大的）graph 的显存池作为公共池
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        # 把这些静态缓冲区挂到实例上，run_model 重放时要往里拷数据
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
