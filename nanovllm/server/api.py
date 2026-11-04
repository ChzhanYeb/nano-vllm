import time
import uuid
from typing import List, Optional, Union
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

# 导入你项目中的 LLM 和采样参数
from nanovllm.llm import LLM
from nanovllm.sampling_params import SamplingParams

app = FastAPI(title="nano-vllm OpenAI Gateway")

# 初始化模型和 Tokenizer (实际应用中建议放到 startup event 中懒加载)
MODEL_NAME = "Qwen3-0.6B"
MODEL_PATH = "/home/chzhanyeb/huggingface/Qwen3-0.6B" # 请替换为你 T1 下载的真实路径

llm = LLM(model=MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# ================= 定义 OpenAI 兼容的 Pydantic 模型 =================

class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str = MODEL_NAME
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: Optional[int] = 512
    stream: bool = False # T3 先只做 False
    tools: Optional[List[dict]] = None # 支持工具调用透传

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage

# ================= API 路由 =================

@app.get("/v1/models")
async def list_models():
    """兼容 OpenAI GET /v1/models"""
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "nano-vllm"
            }
        ]
    }

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """兼容 OpenAI POST /v1/chat/completions (v0 同步阻塞版本)"""
    
    # 1. 将 messages 通过 Chat Template 拼接为 prompt (基于 T2 的结论放在网关侧)
    # 如果 T2 决策是支持 tools，这里需要把 tools 传进去
    messages_dict = [msg.model_dump() for msg in request.messages]
    prompt = tokenizer.apply_chat_template(
        messages_dict, 
        tools=request.tools,
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # 2. 计算 prompt tokens
    prompt_token_ids = tokenizer.encode(prompt)
    prompt_tokens = len(prompt_token_ids)

    # 3. 构造采样参数并调用引擎
    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens or 512
    )
    
    outputs = llm.generate([prompt], sampling_params)
    raw_output = outputs[0] 
    
    # 修复：因为 raw_output 实际上是一个字典 (包含 'text' 和 'token_ids')
    if isinstance(raw_output, dict):
        generated_text = raw_output.get('text', '')
        completion_token_ids = raw_output.get('token_ids', [])
    elif hasattr(raw_output, 'text'):
        generated_text = raw_output.text
        completion_token_ids = getattr(raw_output, 'token_ids', [])
    else:
        generated_text = str(raw_output)
        completion_token_ids = tokenizer.encode(generated_text)

    # 4. 直接使用引擎返回的 token_ids 长度，避免二次 tokenize 误差
    completion_tokens = len(completion_token_ids)

    # 5. 构造符合 OpenAI 规范的返回体
    response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    
    return ChatCompletionResponse(
        id=response_id,
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=generated_text),
                finish_reason="stop" if len(completion_token_ids) < (request.max_tokens or 512) else "length"
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )
    )