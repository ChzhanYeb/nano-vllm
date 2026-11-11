from transformers import AutoTokenizer
from typing import List, Dict, Any, Tuple, Optional

class PromptBuilder:
    def __init__(self, model_path: str):
        # 加载 Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def build(self, messages: List[Dict[str, Any]], tools: Optional[List[dict]] = None) -> Tuple[str, List[int]]:
        """
        将 OpenAI 的 messages 转换为纯文本 prompt，并计算 Token IDs。
        返回: (prompt_text, prompt_token_ids)
        """
        # 1. 应用 Chat Template（落实决策1：网关侧拼装）
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 2. 编码为 Token IDs
        prompt_token_ids = self.tokenizer.encode(prompt_text)

        if len(prompt_token_ids) < 256:
            pad_len = 256 - len(prompt_token_ids)
            # 用无害的 token 补齐（注意不要用特殊 token 如 eos）
            pad_token_id = self.tokenizer.convert_tokens_to_ids("。")
            prompt_token_ids.extend([pad_token_id] * pad_len)
            # 同步更新文本用于调试
            prompt_text += "。" * pad_len
        
        return prompt_text, prompt_token_ids