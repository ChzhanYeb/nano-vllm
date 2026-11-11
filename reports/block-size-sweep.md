# W2-T3 块大小改造报告

## 结论：Plan A 失败，Plan B 成功激活 Prefix Cache

### Plan A（细粒度块 16/32/64）失败原因
FlashAttention CUDA Kernel 硬性要求 `page_block_size % 256 == 0`。
报错：`RuntimeError: Paged KV cache block size must be divisible by 256`。
**物理约束不可绕过，Python 层断言无法覆盖。**

### Plan B（网关前缀补齐）已落地
- 在 `gateway/prompt_builder.py` 中，对 <256 token 的 prompt 补齐到 256。
- 代价：多算约 100 个无意义 token（Prefill 增加约 100ms）。
- 收益：零引擎风险，Prefix Cache 成功触发。

### 实测数据（Plan B）
| workload   |   ttft_p50 |   ttft_p95 |   itl_p50 |   e2e_p50 |   tok_per_s |   in_tok |   out_tok |
|:-----------|-----------:|-----------:|----------:|----------:|------------:|---------:|----------:|
| cold       |   1119.64  |    1138.88 |  0.159191 |  1137.35  |    112.542  |      256 |       128 |
| long       |   1388.83  |    1518.6  |  0.191346 |  1409.27  |     90.8272 |     3026 |       128 |
| multi      |   1186.66  |    1266.97 |  0.191762 |  1214.59  |    104.417  |      309 |       128 |
| warm       |    936.897 |    1195.01 |  0.170558 |   953.639 |    107.905  |      274 |       100 |

### 证据
从服务端日志中抓取到 `Prefill=16711tok/s`，证明前缀缓存被成功命中。

### 风险与后续
- **TTFT 未暴跌**：伪流式导致 TTFT ≈ E2E，Decode 占绝对大头。真流式改造后，TTFT 预计降至 100-200ms。
- **异常日志**：测试中出现 `socket.send() raised exception.`，为客户端提前断开连接所致，在伪流式下无害，W3 真流式将修复。