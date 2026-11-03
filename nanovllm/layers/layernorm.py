"""
RMSNorm 模块（Qwen3 使用的归一化层，比 LayerNorm 少了"减均值"这一步）。

两个值得注意的工程细节：
1. add_rms_forward 把"残差相加"和"归一化"融合在一次读写里完成，
   Transformer 每层都要做这件事，省一次显存往返收益明显；
2. 用 @torch.compile 把这段小算子序列编译融合。
"""
import torch
from torch import nn


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        """
        hidden_size：归一化的维度（即模型的 hidden_size）
        eps        ：防止除以 0 的小常数
        """
        super().__init__()
        self.eps = eps
        # RMSNorm 只有一个可学习的缩放向量：没有 bias，也不减均值
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """最基础的 RMSNorm：y = x / sqrt(mean(x^2) + eps) * weight。"""
        # 记下原始精度（通常是 bf16），算完再转回去
        orig_dtype = x.dtype
        # 升到 fp32 做归约：bf16 下"平方再求和"的精度不够，容易损失有效位数
        x = x.float()
        # 最后一维的均方值（keepdim=True 便于按最后一维广播）
        var = x.pow(2).mean(dim=-1, keepdim=True)
        # 原地除以 sqrt(var + eps)；rsqrt 比 "1/sqrt" 更快，原地操作省一次显存分配
        x.mul_(torch.rsqrt(var + self.eps))
        # 转回原精度，再乘上可学习缩放（同样原地完成）
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        "残差相加 + RMSNorm"的融合版本（Pre-Norm Transformer 的常见优化）。

        返回 (归一化后的输出, 新的残差)。注意返回的 residual 是**未归一化**的
        x + residual，它要继续沿残差支路传给下一层，这样就不必为了做加法
        而重复保存/读取一份完整张量。
        """
        orig_dtype = x.dtype
        # 在 fp32 下做加法，精度更稳（add_ 是原地加法）
        x = x.float().add_(residual.float())
        # 残差支路的输出：高精度加完之后转回原精度
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        统一入口，按是否传 residual 分派：

        - residual 为 None：只做普通 RMSNorm，返回一个张量
          （用在模型最后一层之后，见 Qwen3Model.forward）；
        - 传了 residual    ：做融合版本，返回 (输出, 新残差)
          （用在每个 Decoder 层里，见 Qwen3DecoderLayer.forward）。
        """
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
