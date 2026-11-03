"""
采样模块：从模型输出的 logits 里按温度采样出下一个 token。

这里用的是"指数竞速（exponential race）"技巧，它和 Gumbel-max 是同一类做法：
    设 e_i ~ Exp(1)，则 e_i / p_i ~ Exp(p_i)（速率参数为 p_i）；
    而"一组独立指数分布随机变量，谁最小"这件事，
    正好以 p_i / Σp 的概率选出第 i 个——也就是 Categorical(p)。
    又因为 argmin_i(e_i/p_i) == argmax_i(p_i/e_i)，
    所以直接算 argmax(probs / e) 就等价于从 probs 里做多项式采样。

好处：整个过程都是逐元素运算，能在 GPU 上一条龙算完，
不需要"前缀和 + 二分查找"那类带数据相关分支的采样写法（那样会很慢）。
"""
import torch
from torch import nn


class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        """
        logits      ：[num_seqs, vocab_size]，每条序列对词表的原始打分
        temperatures：[num_seqs]，每条序列各自的温度
        返回        ：[num_seqs]，每条序列采样出的 token id
        """
        # 1. 升到 fp32 后除以温度；unsqueeze(dim=1) 让温度沿词表维度广播。
        #    温度越低分布越尖（越接近贪心），越高越平（越随机）。
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        # 2. softmax 变成概率分布 probs
        probs = torch.softmax(logits, dim=-1)
        # 3. 指数竞速采样：每个位置除以一个 Exp(1) 随机数（clamp_min_ 防止除零），
        #    再取 argmax。div_ 是原地除法，省一次显存分配。
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
