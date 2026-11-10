# gateway/main.py
import time
import uuid
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

# 引入我们刚写的模块
from .schemas import ChatCompletionRequest, ChatCompletionResponse, ChatMessage, ChatCompletionChoice, Usage
from .prompt_builder import PromptBuilder

# 引入引擎（依赖方向：gateway -> nanovllm）
from nanovllm.llm import LLM
from nanovllm.sampling_params import SamplingParams

app = FastAPI(title="nano-vllm OpenAI Gateway")

# ================= 初始化 =================
MODEL_NAME = "Qwen3-0.6B"
MODEL_PATH = "/home/chzhanyeb/huggingface/Qwen3-0.6B"

print("正在加载引擎与 Tokenizer...")
llm = LLM(model=MODEL_PATH)
builder = PromptBuilder(MODEL_PATH)
print("加载完成！")

# ================= API 路由 =================

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": MODEL_NAME, "object": "model", "created": int(time.time()), "owned_by": "nano-vllm"}]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # 1 & 2. 网关侧负责拼装和算 Token（落实 T2 决策）
    messages_dict = [msg.model_dump() for msg in request.messages]
    prompt_str, prompt_token_ids = builder.build(messages_dict, request.tools)
    prompt_tokens = len(prompt_token_ids)

    # 3. 构造采样参数
    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens or 512
    )
    
    # 4. 调用引擎（TODO: W2/W3 改为 asyncio.to_thread 或引擎级异步）
    # 注意：目前传的 prompt_str，未来做极致 KV Cache 优化时，可以考虑改传 prompt_token_ids
    outputs = llm.generate([prompt_str], sampling_params)
    raw_output = outputs[0] 
    
    # 兼容提取结果
    if isinstance(raw_output, dict):
        generated_text = raw_output.get('text', '')
        completion_token_ids = raw_output.get('token_ids', [])
    elif hasattr(raw_output, 'text'):
        generated_text = raw_output.text
        completion_token_ids = getattr(raw_output, 'token_ids', [])
    else:
        generated_text = str(raw_output)
        completion_token_ids = builder.tokenizer.encode(generated_text)

    completion_tokens = len(completion_token_ids)
    response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    
    # ================= T4: 伪流式分支 =================
    if request.stream:
        async def event_generator():
            # TODO: W3-T1 替换为引擎级真流式
            for token_id in completion_token_ids:
                chunk_text = builder.tokenizer.decode([token_id], skip_special_tokens=True)
                if not chunk_text:
                    continue
                
                chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {"content": chunk_text}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

            # 发送结束块
            final_chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            }
            yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
            
            # === 新增：针对 stream_options={"include_usage": True} 的兼容 ===
            usage_chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [], # 必须为空
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            }
            yield f"data: {json.dumps(usage_chunk, ensure_ascii=False)}\n\n"
            # ==========================================================

            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # ================= T3: 非流式分支 =================
    return ChatCompletionResponse(
        id=response_id,
        created=int(time.time()),
        model=request.model,
        choices=[ChatCompletionChoice(
            index=0,
            message=ChatMessage(role="assistant", content=generated_text),
            finish_reason="stop" if completion_tokens < (request.max_tokens or 512) else "length"
        )],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )
    )