"""
对外的门面（facade）模块。

LLM 是 LLMEngine 的空子类：
- 用户 `from nanovllm import LLM` 拿到一个语义清晰的入口名；
- 真正的实现全部在 LLMEngine 里，这样"用户 API"与"引擎实现"解耦，
  以后想让 LLM 只暴露部分方法，也完全不用改动引擎。
"""
from nanovllm.engine.llm_engine import LLMEngine


class LLM(LLMEngine):
    """
    vLLM 风格的入口类，典型用法：

        llm = LLM("/path/to/Qwen3-0.6B", enforce_eager=True)
        outputs = llm.generate(["你好"], SamplingParams(temperature=0.6))

    它自身没有任何新逻辑，能力全部继承自 LLMEngine：
      - __init__    ：建 Config、拉起 TP 子进程、建 tokenizer / Scheduler
      - add_request ：把一条 prompt 变成 Sequence 放进等待队列
      - step        ：调度 + 模型前向 + 后处理，推进"一个 iteration"
      - generate    ：循环 step 直到所有序列结束，返回解码后的文本
    """

    pass
