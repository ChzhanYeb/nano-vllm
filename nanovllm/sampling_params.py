"""
采样参数模块。

用户通过 SamplingParams 告诉引擎"怎么从模型输出的 logits 里挑下一个 token"。
nanovllm 为了教学刻意做得极简：只保留温度采样必需的三个参数，
也因此**只支持随机采样**，不支持 temperature=0 的贪心解码。
"""
from dataclasses import dataclass


# slots=True：实例字段固定，省内存、访问略快。
@dataclass(slots=True)
class SamplingParams:
    # 温度：>1 更随机、<1 更保守（概率分布更尖，趋近于总取概率最大的词）；禁止为 0。
    temperature: float = 1.0
    # 每条序列最多生成多少个新 token（不含 prompt 部分）。
    max_tokens: int = 64
    # True = 即使采样到结束符（EOS）也继续生成，直到 max_tokens 用完。
    # benchmark 时常用它固定输出长度，方便比较吞吐。
    ignore_eos: bool = False

    def __post_init__(self):
        """dataclass 构造完成后自动调用的校验钩子（这里只做参数合法性检查）。"""
        # nanovllm 的采样实现是 logits / temperature 之后再 softmax，
        # temperature=0 会除零；想要贪心得另写 argmax 分支，所以这里直接禁止。
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
