"""
旋转位置编码（RoPE, Rotary Position Embedding）模块。

核心思想：把 q / k 的相邻两维看成一个复平面上的向量，按 token 的位置旋转一个角度。
这样两个 token 做内积时，结果只依赖它们的**相对位置差**，
天然具备相对位置编码的性质，对长文本外推也更友好。

本文件做了两处工程优化：
1. 预先算好所有位置的 cos/sin 表（cos_sin_cache），前向时只需按位置索引查表；
2. 用 @torch.compile 把旋转运算编译融合。
"""
from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    对单个张量（query 或 key）施加旋转。

    x 的最后一维是 head_dim，被对半切成 x1、x2，然后按二维旋转公式计算：
        y1 = x1 * cos - x2 * sin
        y2 = x2 * cos + x1 * sin
    """
    # 先升到 fp32 保证精度；chunk(2, dim=-1) 即沿最后一维对半切
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    # 拼回原形状，并转回输入精度（如 bf16）
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        """
        head_size              ：每个注意力头的维度
        rotary_dim             ：参与旋转的维度（本实现要求它等于 head_size）
        max_position_embeddings：预计算到多长的位置（越长的表占显存越多）
        base                   ：旋转基频 theta，Qwen3 默认 1000000（越大频率越高）
        """
        super().__init__()
        self.head_size = head_size
        # 本实现做了简化：整个 head_dim 都参与旋转
        assert rotary_dim == head_size
        # inv_freq：每一对维度对应的角频率，形状 [rotary_dim/2]。
        # arange(0, rotary_dim, 2) 取 0,2,4,... 是因为每两个维度共用同一个频率。
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        # t：所有可能的位置下标 0 .. max_position_embeddings-1
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        # 外积得到"每个位置 × 每个频率"的旋转角度 freqs[pos, i]
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        # 把 cos 与 sin 拼在同一维（最后一维变成 rotary_dim），前向时一次索引就能取回两者；
        # unsqueeze_(1) 补出 head 维（大小为 1），方便与 q / k 自动广播。
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        # persistent=False：这是可以现算出来的缓存，不必写进 state_dict
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        positions：每个 token 的位置下标，形状 [num_tokens]
        query/key：形状 [num_tokens, num_heads, head_dim]
        返回旋转后的 query、key（形状不变）。
        """
        # 按位置查表：得到 [num_tokens, 1, rotary_dim]
        cos_sin = self.cos_sin_cache[positions]
        # 沿最后一维对半切回 cos 与 sin
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
):
    """
    带缓存的工厂函数：同一种配置只创建一次 RotaryEmbedding。

    lru_cache(1) 表示只缓存最近一次结果——cos/sin 表本身可能几十 MB，
    而一次运行中所有层用的是同一份表，缓存 1 份就足够。
    """
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
