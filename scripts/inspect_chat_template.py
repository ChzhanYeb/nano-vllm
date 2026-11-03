import json
import hashlib
import transformers
from transformers import AutoTokenizer

MODEL_ID = "/home/chzhanyeb/huggingface/Qwen3-0.6B" 
tok = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True, trust_remote_code=True)

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询指定城市天气",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名"}
            },
            "required": ["city"]
        }
    }
}]

messages = [
    {"role": "system", "content": "你是一个严谨的助手。需要时调用工具。"},
    {"role": "user", "content": "北京今天天气怎么样？"},
]

def render(msgs, tools=None, add_generation_prompt=True):
    kwargs = dict(tokenize=False, add_generation_prompt=add_generation_prompt)
    if tools is not None:
        kwargs["tools"] = tools
    # Qwen3 模板若支持 enable_thinking，建议固定并记录，避免长度漂移。
    # 如果你的模板不支持会报错，先注释掉：
    # kwargs["enable_thinking"] = False
    return tok.apply_chat_template(msgs, **kwargs)

def ntok(text):
    return len(tok.encode(text, add_special_tokens=False))

full_with_tools = render(messages, tools=tools, add_generation_prompt=True)
full_without_tools = render(messages, tools=None, add_generation_prompt=True)

# 固定前缀：完整渲染中第一个 user 消息标记之前的部分 = system + tool schema
marker = "<|im_start|>user"
pos = full_with_tools.find(marker)
if pos == -1:
    prefix_text = render([messages[0]], tools=tools, add_generation_prompt=False)
else:
    prefix_text = full_with_tools[:pos]

n_full = ntok(full_with_tools)
n_without = ntok(full_without_tools)
n_prefix = ntok(prefix_text)
n_tools_json = ntok(json.dumps(tools, ensure_ascii=False))
template_sha = hashlib.sha256((tok.chat_template or "").encode("utf-8")).hexdigest()

print("=== WITHOUT TOOLS ===")
print(full_without_tools)

print("\n=== WITH TOOLS ===")
print(full_with_tools)

print("\n=== PREFIX: system + tool schema ===")
print(prefix_text)

print("\n=== METRICS ===")
metrics = {
    "model_id": MODEL_ID,
    "transformers_version": transformers.__version__,
    "tokenizer_name_or_path": tok.name_or_path,
    "chat_template_sha256": template_sha,
    "tokens_without_tools": n_without,
    "tokens_with_tools_full_prompt": n_full,
    "tokens_tools_delta": n_full - n_without,
    "tokens_prefix_system_plus_tools": n_prefix,
    "tokens_tools_json_alone": n_tools_json,
    "prefix_ratio_vs_full_prompt": round(n_prefix / n_full, 4),
    "prefix_percent_vs_full_prompt": f"{n_prefix / n_full * 100:.2f}%",
}
print(json.dumps(metrics, ensure_ascii=False, indent=2))
