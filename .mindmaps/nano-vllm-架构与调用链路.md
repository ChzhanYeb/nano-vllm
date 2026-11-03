# 0. 项目速览

- 定位：教学版 vLLM，约 1200 行 Python，从零实现
- 已实现的关键能力
  - 连续批处理 continuous batching
  - PagedAttention 分页 KV Cache
  - 前缀缓存 prefix caching
  - chunked prefill 分块预填充
  - 张量并行 tensor parallelism
  - CUDA Graph + torch.compile
- 刻意简化的地方（与 vLLM 的差距）
  - prefill 与 decode 不混在同一个 batch 里
  - 只支持 Qwen3 一种模型结构，只支持单机多卡
  - 采样只支持随机采样，禁止 temperature=0 的贪心解码
- 两个可直接运行的脚本
  - `example.py`：最小可运行示例（两条 prompt）
  - `bench.py`：吞吐压测（256 条随机请求）

# 1. 分层架构（谁调用谁）

- 用户 API 层
  - `nanovllm/llm.py` → `LLM`：`LLMEngine` 的空子类，只是门面
  - `nanovllm/sampling_params.py` → `SamplingParams`
  - `nanovllm/__init__.py`：对外只导出 LLM 与 SamplingParams
- 引擎层 `nanovllm/engine/`
  - `llm_engine.py` → `LLMEngine`：`step()` 串起"调度 → 前向 → 后处理"
  - `scheduler.py` → `Scheduler`：prefill 优先、decode 抢占
  - `block_manager.py` → `BlockManager`：KV Cache 分页 + 前缀缓存
  - `sequence.py` → `Sequence`：一条请求的状态机与页表
  - `model_runner.py` → `ModelRunner`：打包输入、跑模型、CUDA Graph（每个 TP rank 一个）
- 模型层 `nanovllm/models/`
  - `qwen3.py` → `Qwen3ForCausalLM` / `Qwen3Model` / `Qwen3DecoderLayer` / `Qwen3Attention` / `Qwen3MLP`
- 算子层 `nanovllm/layers/`
  - `linear.py`：并行线性层（TP 的两种切法）
  - `attention.py`：FlashAttention + Triton kernel 写 KV Cache
  - `embed_head.py`：词表并行 Embedding 与 LMHead
  - `layernorm.py`：RMSNorm（含残差融合）
  - `rotary_embedding.py`：RoPE 旋转位置编码
  - `activation.py`：SwiGLU 门控激活
  - `sampler.py`：采样
- 基础设施 `nanovllm/utils/`
  - `context.py`：全局上下文，传递 cu_seqlens / slot_mapping / block_tables
  - `loader.py`：权重加载，把 weight_loader 挂到每个参数对象上

# 2. 一次 generate 的完整调用链

- ① 提交请求
  - `LLMEngine.generate()` → `add_request()`
  - `tokenizer.encode()` → `Sequence(...)` → `Scheduler.waiting`
- ② 主循环 `while not is_finished()`
  - `LLMEngine.step()`
- ③ 调度 `Scheduler.schedule()`
  - prefill 分支（只要 waiting 非空就优先做）
    - `BlockManager.can_allocate()`：算前缀缓存命中块数 + 查空闲页是否够
    - `BlockManager.allocate()`：命中页共享（ref_count+1），其余新分配
    - 记录 `num_scheduled_tokens`；预算不够时只切"本轮第一条"（chunked prefill）
    - prompt 全部算完 → 状态转 RUNNING，搬进 running 队列
  - decode 分支（prefill 一条都挑不出来时）
    - `BlockManager.can_append()` 不满足 → `preempt()` 抢占队尾序列
    - `may_append()`：只有当 token 跨页时才分配新页
    - 每条序列 `num_scheduled_tokens = 1`、`is_prefill = False`
  - 返回 `(seqs, is_prefill)`
- ④ 执行前向 `ModelRunner.call("run", seqs, is_prefill)`
  - TP 广播：rank0 `write_shm()` → worker 被 Event 唤醒 → `read_shm()` → `loop()`
  - `prepare_prefill()` / `prepare_decode()`：拼张量 + `set_context()`
  - `run_model()`
    - prefill / enforce_eager / bs>512 → eager 前向
    - 否则 → 选最小的可用档位重放 CUDA Graph
    - `compute_logits()` 算出 logits
  - `Sampler` 采样出下一个 token
  - `reset_context()`
- ⑤ 后处理 `Scheduler.postprocess()`
  - `BlockManager.hash_blocks()`：把本次新写满的块登记进前缀缓存
  - `append_token()`：新 token 追加到序列末尾
  - 命中 EOS 或达到 max_tokens → FINISHED + `deallocate()`
  - chunked prefill 还没算完 → 丢弃本次采样结果，下一轮继续
- ⑥ 输出
  - `step()` 返回本轮刚结束的序列
  - `generate()` 按 seq_id 排序 → `tokenizer.decode()` → `[{"text", "token_ids"}]`

# 3. 核心机制逐个拆

- PagedAttention 分页 KV Cache
  - 页大小 `kvcache_block_size`，默认 256 个 token 一页
  - `Sequence.block_table`：逻辑块号 → 物理块号，就是页表
  - `Block.ref_count`：多条序列可以共享同一页
  - `free_block_ids`(deque) + `used_block_ids`(set) 双结构，O(1) 分配与判断
  - KV Cache 张量形状 `[2, 层数, 页数, 页内token, kv头数, head_dim]`
- 前缀缓存 Prefix Caching
  - `compute_hash()`：链式哈希 = xxh64(前一块哈希 + 本块 token 字节)
  - 只有"写满的块"才登记（最后一页内容还会变，不参与匹配）
  - `can_allocate()` 返回命中块数 → `allocate()` 里共享而不是复制
  - 收益：相同 system prompt 的第二条请求几乎不用重算 prefill
- chunked prefill 分块预填充
  - `max_num_batched_tokens` 是"一个 batch 的 token 预算"
  - 只允许把本轮第一条序列切块，避免超长 prompt 饿死后面的请求
  - 没算完时，本次采样出的 token 必须丢弃
- 抢占 preempt
  - 物理页不够时，把 running 队尾的序列退回 waiting
  - 代价：已算的 KV 全部丢弃，之后要重新 prefill（用算力换显存）
- 张量并行 TP
  - `ColumnParallelLinear`：按输出维切，本身不需要通信
  - `RowParallelLinear`：按输入维切 + all_reduce；bias 只在 rank0 加一次
  - 两者配对使用，一个 Transformer 块只付一次 all_reduce 的代价
  - 词表并行：Embedding 各查一段再 all_reduce；LMHead gather 到 rank0
  - 控制通道：1MB 共享内存 + Event；命令 = `pickle([方法名, 参数...])`
  - `Sequence.__getstate__`：decode 只传 last_token，通信量不随序列变长而增大
- CUDA Graph（decode 加速）
  - 档位：1、2、4、8、16、32……（实际 bs 向上取整到已捕获档位）
  - 静态输入缓冲区复用地址；所有 graph 共享同一个 `graph_pool`
  - 补齐出来的行：`slot_mapping = -1`、`context_lens = 0`
  - 捕获前先 warmup 一次，让编译与显存分配发生在 graph 之外
  - prefill 与 bs>512 不走 graph（形状变化太大，不划算）
- 显存规划：KV Cache 到底能开多少页
  - warmup 跑一次最大 batch，拿到 `allocated_bytes.all.peak`
  - 可用 = `total × util − used − (peak − current)`（差额是给激活值留的余量）
  - 每页字节 = 2 × 层数 × 页大小 × kv头数 × head_dim × dtype 字节数
  - 结果回填 `config.num_kvcache_blocks`，Scheduler 建 BlockManager 时使用
- 采样
  - 要求 `temperature > 0`；`logits / temperature` → softmax
  - 指数竞速：`argmax(probs / Exp(1))` 等价于按 probs 做多项式采样
- Pre-Norm 残差融合
  - 第一层 `residual=None`；之后 `add_rms_forward` 返回新的 residual
  - 一次读写完成"加残差 + 归一化"，每层都要做，收益明显

# 4. 建议阅读路线（新手向）

- 第 1 天：骨架
  - `example.py` → `config.py` → `llm.py` → `llm_engine.py`（先看 generate / step 的流程）
- 第 2 天：调度与显存（最核心）
  - `sequence.py` → `block_manager.py`（前缀缓存是精髓）→ `scheduler.py`
- 第 3 天：执行与加速
  - `model_runner.py`（显存账本 + CUDA Graph）→ `utils/context.py`
- 第 4 天：模型与算子
  - `layers/linear.py`（TP 两种切法）→ `layers/attention.py` → `models/qwen3.py`
- 第 5 天：细节算子
  - `layernorm.py` / `rotary_embedding.py` / `activation.py` / `sampler.py` / `embed_head.py` / `loader.py`
- 一句话主线
  - 调度决定"算什么" → 打包决定"怎么喂" → 分页决定"KV 放哪" → 内核决定"怎么算快"

# 5. 动手实验清单

- 在前缀缓存处打印 `num_cached_blocks`，观察命中情况
- 把 `kvcache_block_size` 从 256 改成 512，对比吞吐与可容纳并发
- `enforce_eager=True/False` 对比 decode 吞吐（CUDA Graph 的收益）
- `tensor_parallel_size=1/2` 对比速度（all_reduce 的代价）
- 调小 `max_num_batched_tokens`，观察 chunked prefill 被触发
- 调大 `max_num_seqs`，观察抢占 preempt 何时被触发

# 6. 本仓库注释现状

- 已加中文注释：`nanovllm/` 下 19 个 .py + 根目录 `example.py`、`bench.py`
- 每个类/函数都有 docstring，关键代码行有行内注释
- `layers/embed_head.py` 原有的中文注释已保留，并在其上补全
- 校验方式：加注释前后做 AST 比对（剥掉 docstring 后语法树一致）
- 回退方式：`git diff` 审阅、`git checkout -- .` 撤销
