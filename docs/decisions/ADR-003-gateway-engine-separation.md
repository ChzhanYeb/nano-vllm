# ADR-003: 网关层与引擎层分离架构

## 背景
在 W1-T2 阶段确认，Prompt 的 Chat Template 拼装必须在网关侧完成，且需要精确计算 Prefix Token 数以支持未来的 KV Cache 优化。早期的实现将 FastAPI 网关代码直接写在了 `nanovllm/server/` 中，导致引擎层与网络协议、Tokenizer 强耦合。

## 决策
实施严格的网关与引擎分离架构：
1. **网关层 (`gateway/`)**：负责 HTTP 路由、Pydantic 校验、SSE 流式、Chat Template 拼装、Token 计数统计。
2. **引擎层 (`nanovllm/`)**：作为纯净的推理库，不包含任何 Chat Template 概念，只接收 `prompt_str` 或 `prompt_token_ids`，返回生成结果。
3. 依赖方向固定为：`gateway` -> `nanovllm`。

## 后果
- **优势**：引擎可以独立作为库被其他项目引用；网关可以随时替换底层引擎（例如后续做性能对照实验时换成真实 vLLM）而不影响上层协议；实现了职责单一。
- **文件变化**：原 `nanovllm/server/api.py` 废弃，拆分至 `gateway/main.py`、`gateway/schemas.py` 与 `gateway/prompt_builder.py`。