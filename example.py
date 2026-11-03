"""
最小可运行示例：加载 Qwen3-0.6B 生成两条回答。

它同时也是理解整个项目的最佳入口，运行链路是：

    tokenizer 把 prompt 编码成 token id
      -> LLM(...) 内部：建 Config、拉起张量并行子进程、建 Scheduler
      -> llm.generate(...) 循环调用 step()：调度 -> 前向 -> 后处理，直到所有序列结束
      -> 生成出的 token id 被解码回文本

这里用 enforce_eager=True 关掉了 CUDA Graph，方便调试；
想看 CUDA Graph 的加速效果，把它去掉（改成 False）即可。
"""
import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch


def main():
    torch._dynamo.config.cache_size_limit = 64
    # 模型权重目录：按 README 的 huggingface-cli download 命令下载后的默认位置
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    # tokenizer 负责 文本 <-> token id 互转。
    # 注意引擎内部自己也会建一份（见 LLMEngine.__init__），这里这份只用于套对话模板。
    tokenizer = AutoTokenizer.from_pretrained(path)
    # 建引擎：
    #   enforce_eager=True      不用 CUDA Graph（decode 走 eager，便于调试/排错）
    #   tensor_parallel_size=1  单卡；改成 2/4 就会 spawn 出对应数量的子进程做张量并行
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    # 采样参数：temperature 越小越"确定"，max_tokens 限制每条最多生成多少 token
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    # apply_chat_template：套用 Qwen 的对话模板，把 user 消息包成
    # "<|im_start|>user ... <|im_end|>" 这种模型认识的格式。
    # tokenize=False 表示只返回字符串（编码交给引擎内部的 tokenizer 做）。
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,   # 末尾追加"轮到 assistant 说话"的提示前缀
        )
        for prompt in prompts
    ]
    # 一次调用完成全部生成：两条 prompt 会一起被调度（连续批处理），
    # 返回列表的顺序与传入的 prompts 一致
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        # output 形如 {"text": "解码后的文本", "token_ids": [...]}
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
