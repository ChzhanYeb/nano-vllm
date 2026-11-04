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
        
        return prompt_text, prompt_token_ids