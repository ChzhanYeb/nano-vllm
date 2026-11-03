"""
激活函数模块：SwiGLU 的"门控 + 激活"部分。

Qwen3 的 MLP 采用 SwiGLU：先用一个合并的线性层同时算出 gate 与 up 两路，
再做 silu(gate) * up。这个文件就是最后那步逐元素运算。
"""
import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    """
    SiLU 门控单元（SwiGLU 的核心），数学上等价于：

        out = silu(x[..., :d]) * x[..., d:]        # 输入最后一维是 2d，输出是 d

    它本身没有可学习参数，之所以做成 nn.Module，是为了能挂进模型里统一管理，
    并且可以用 @torch.compile 把这几步融合成少量 kernel，
    减少 kernel 启动次数与显存往返。
    """

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., 2d] -> [..., d]。"""
        # 沿最后一维一分为二：前半是 gate（要过 SiLU），后半是 up（直接相乘）
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
