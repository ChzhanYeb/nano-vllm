# nano-vllm & Gateway

> 一个轻量级、高性能的 LLM 推理引擎，提供标准的 OpenAI 兼容 API 网关。

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-brightgreen.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)

---

## ✨ 核心特性

- **🚀 高性能推理**：基于 PagedAttention 的 KV Cache 显存管理，支持高效的 Prefill 与 Decode 阶段。
- **⚡ 前缀缓存 (Prefix Caching)**：基于 xxhash 的链式哈希实现，自动复用相同的 Prompt 前缀（如 System Prompt + Tools），显著降低首 Token 延迟 (TTFT)。
- **🌐 OpenAI 兼容网关**：内置标准的 HTTP 接口，提供 `/v1/models` 与 `/v1/chat/completions`，全量支持 `stream=true` 的 SSE 流式响应。
- **🧩 架构解耦设计**：
  - `gateway/` 专注协议解析、数据校验与 Prompt 模板拼装。
  - `nanovllm/` 专注底层张量计算与核心调度，职责边界清晰。

---

## 🛠️ 架构设计

```
[ Client / SDK ]
       │
       ▼ (OpenAI Protocol)
┌──────────────┐
│   gateway/   │  FastAPI / Schemas / Prompt Builder
└──────┬───────┘
       │ (Tokens & Config)
       ▼
┌──────────────┐
│  nanovllm/   │  Engine / BlockManager (PagedAttention) / Model Execution
└──────────────┘
```

---

## 📂 目录结构

```text
nano-vllm/
├── gateway/              # OpenAI 兼容网关 (FastAPI)
│   ├── main.py           # 路由与服务入口
│   ├── prompt_builder.py # Chat Template 拼装与 Tokenizer 封装
│   └── schemas.py        # Pydantic 请求/响应数据模型
├── nanovllm/             # 核心推理引擎
│   ├── engine/           # 调度器、BlockManager (KV Cache Management) 与模型执行器
│   ├── layers/           # 自定义算子层 (PagedAttention, Linear 等)
│   └── models/           # 模型结构定义 (如 Qwen3 等)
├── bench/                # 性能与压力测试脚本 (client.py)
├── baseline/             # 环境配置与基准测试记录
└── docs/decisions/       # ADR (Architecture Decision Records) 架构决策文档
```

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆仓库
git clone https://github.com/your-repo/nano-vllm.git
cd nano-vllm

# 安装依赖
pip install -r requirements.txt
```

### 2. 启动服务

```bash
# 启动 API 网关
python -m gateway.main --host 0.00.0.0 --port 8000
```

### 3. 调用测试

#### 使用 Curl 测试

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3",
    "messages": [
      {"role": "user", "content": "你好，请介绍一下你自己。"}
    ],
    "stream": true
  }'
```

#### 使用 OpenAI Python SDK 测试

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="none"
)

response = client.chat.completions.create(
    model="qwen3",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

---

## 📊 性能基准测试

你可以使用 `bench/` 目录下的脚本对系统进行基准测试和压测：

```bash
python -m bench.client --concurrency 10 --total-requests 100
```

详细的测试结果及对比数据请查阅 [baseline/ README](baseline/README.md)。

---

## 📄 架构决策记录 (ADR)

关于项目的核心技术选型及设计变更，请参考 [docs/decisions/](docs/decisions/) 目录。

---

## 📜 开源许可

本项目基于 [MIT License](LICENSE) 开源。