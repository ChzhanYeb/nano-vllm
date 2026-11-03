"""
注意力层：把"KV Cache 写入 + Prefill / Decode 两种注意力计算"封装在一起。

三个关键点：
1. store_kvcache：用一个小的 Triton kernel，把本步算出的 K/V 按 slot_mapping
   散写到"分页"的 KV Cache 中（PagedAttention 的写路径）；
2. Prefill 走 flash_attn_varlen_func：把变长 batch 一次算完；
   如果命中了前缀缓存（prefix cache），还要带上 block_table 去读历史 K/V；
3. Decode 走 flash_attn_with_kvcache：q 只有 1 个 token，
   但需要按页表去 KV Cache 里读该序列的全部历史 K/V。
"""
import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    """
    把每个 token 的 K/V 向量写进 KV Cache 的对应槽位。

    每个 program 处理一个 token（idx 是它在展开张量里的行号）：
      - key/value 是连续的 [N, num_heads * head_dim] 两块内存，按行读取；
      - slot_mapping[idx] 给出该 token 在 KV Cache 里的物理槽位号；
      - 目标地址 = slot * D + 行内偏移，所以写入是"随机行、连续行内"的模式。

    D 声明为 tl.constexpr：它是编译期常量，kernel 才能展开成固定宽度的向量操作。
    """
    # 当前 program 处理第几个 token
    idx = tl.program_id(0)
    # 该 token 要写入的物理槽位；-1 表示不写（CUDA Graph 补齐出来的空位就是 -1）
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    # 源张量第 idx 行的地址偏移（key_stride 是行间步长，通常等于 num_heads*head_dim）
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    # 目标：KV Cache 里第 slot 个槽位的连续 D 个元素
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    """
    启动上面的 Triton kernel 的 Python 包装。

    key/value    ：[num_tokens, num_heads, head_dim]
    k_cache/v_cache：某一层的 KV Cache，形状 [num_blocks, block_size, num_kv_heads, head_dim]；
                    它在显存里是连续的，所以 kernel 可以直接按"槽位号 × D"寻址，
                    其中 D = num_kv_heads * head_dim 就是"一个 token 的 KV 向量长度"
    slot_mapping ：[num_tokens]，每个 token 对应的物理槽位（= 物理块号 × block_size + 块内偏移）
    """
    N, num_heads, head_dim = key.shape
    # D：一个 token 的 K（或 V）向量长度 = 头数 × 每头维度
    D = num_heads * head_dim
    # 下面几条断言是 kernel 能这样写的前提：
    # 最后一维与 head 维都连续，于是可以把 [num_heads, head_dim] 当成一段连续内存
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    # 网格大小 = token 数（一个 program 一个 token）；
    # 把 stride(0) 作为参数传进去，避免 kernel 内部再做乘法
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        """
        num_heads   ：本 rank 负责的 query 头数
        head_dim    ：每个注意力头的维度
        scale       ：softmax 的缩放系数（通常等于 head_dim ** -0.5）
        num_kv_heads：本 rank 负责的 KV 头数（GQA 下小于 num_heads）
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        # 真正的 KV Cache 稍后由 ModelRunner.allocate_kv_cache() 注入每个层，
        # 这里先放一个空张量占位（后面就用 "numel() 是否为 0" 来判断是否已分配）
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        q/k/v 形状都是 [num_tokens, num_heads(或 num_kv_heads), head_dim]，
        其中 num_tokens 是本步所有序列的 token 数之和（变长 batch 打平后的结果）。
        返回 o，形状 [num_tokens, num_heads, head_dim]。
        """
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        # 只要 KV Cache 已经分配，就先把本步新算出的 K/V 写进去（prefill 与 decode 都要写）
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            # 命中前缀缓存时，本次只需算新增部分的 Q，
            # 但 K/V 必须包含全部历史 —— 直接从 KV Cache 取（页表由 block_tables 描述）
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            # varlen：batch 内各序列长度不同，用 cu_seqlens 描述每条序列的边界；
            # causal=True 保证每个位置只看它自己和之前的 token。
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            # decode 时每条序列只有 1 个 query token，所以要 unsqueeze(1) 补出 seqlen 维；
            # cache_seqlens 告诉 kernel 每条序列历史有多长，
            # block_table 则告诉它这些历史位置分别落在哪个物理块上。
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o
