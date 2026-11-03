# ADR-001 Prompt 拼装位置：网关侧 vs Adapter 侧

- **状态**：Accepted
- **日期**：2026-11-03
- **关联任务**：W1-T2
- **模型**：Qwen/Qwen3-0.6B
- **Tokenizer 来源**：/home/chzhanyeb/huggingface/Qwen3-0.6B
- **Transformers 版本**：4.57.0
- **Chat Template SHA256**：a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8

## 背景

工具调用需要把 OpenAI 格式的 `messages` 和 `tools` schema 渲染成模型可读的单一 prompt。如果网关侧和 Adapter 侧都调用 `apply_chat_template`，可能导致重复拼装或格式漂移；如果两边都不拼，模型无法正确识别工具。

## 决策

Prompt 拼装放在 **网关侧 Python 服务**。
- 网关负责接收 `messages` 和 `tools`，调用 `tokenizer.apply_chat_template`，并将最终 prompt/input_ids 传给推理引擎。
- 网关需基于固定前缀长度（System + Tool Schema）估算 `cached_tokens`。
- Adapter/推理引擎仅负责执行，**严禁**二次应用 chat template。

## 实测数据 (基于 Qwen3-0.6B)

- 无 tools 完整 prompt token 数：30
- 有 tools 完整 prompt token 数：169
- Tools 带来的 token 增量：139
- 系统提示 + 工具 schema 固定前缀 token 数：156
- **固定前缀占完整 prompt 比例：92.31%**
- 工具 schema JSON 单独 token 数：63
- 渲染样例：见 `baseline/chat-template-probe.log`

## 理由

1. **Token 与缓存计算**：网关侧是 Python，有 tokenizer，能直接算 token 长度并估算 `cached_tokens`（前缀占比高达 92.31%，缓存收益极大）。
2. **职责单一**：统一拼装便于日志审计和格式稳定，避免引擎层与网关层职责重叠。
3. **适配多模型**：网关侧可根据不同模型的 `chat_template` 动态切换，推理引擎保持黑盒。

## 后果与风险

- 网关使用的 tokenizer 必须与推理模型严格一致；模型升级需重跑本探针（确保 SHA256 不变）。
- Adapter 不得再调用 `apply_chat_template`。
- 工具 schema 的 JSON 顺序应稳定，避免前缀变化导致 KV 缓存失效。

## 验收

- [x] `baseline/chat-template-probe.log` 已生成
- [x] 本文包含实测 token 数和前缀占比