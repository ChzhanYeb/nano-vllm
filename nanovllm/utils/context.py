"""
全局上下文模块：给"当前这一次前向"传递元信息。

为什么要这么设计？
  Attention、LM Head 这些层深埋在模型内部，如果每次都要把 cu_seqlens、
  slot_mapping 之类的张量一层层当参数传下去，代码会非常啰嗦。
  nano-vllm 的做法是：ModelRunner 在调用模型前用 set_context() 把本次前向
  需要的元信息写进一个全局单例，层内部用 get_context() 取用，
  一次前向结束后 reset_context() 清空。

  代价：它是进程级全局状态，所以一个进程同一时刻只能有一次前向在进行
  （这也解释了为什么 nano-vllm 是单线程、同步执行的）。
"""
from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    """一次前向（prefill 或 decode）所需的全部元信息。"""
    # 当前是 prefill 阶段（True）还是 decode 阶段（False）；Attention 据此选择不同 kernel。
    is_prefill: bool = False
    # 【prefill 用】变长 batch 中每条序列 query 长度的前缀和（cumulative sequence lengths）。
    # 例如 [0, 5, 9] 表示这批有两条序列，长度分别是 5 和 4，
    # 用来把一个"打平"的大张量重新切回各自的序列。
    cu_seqlens_q: torch.Tensor | None = None
    # 【prefill 用】key/value 侧的前缀和。带前缀缓存（prefix cache）时它会大于 cu_seqlens_q，
    # 因为 Q 只算新增部分，K/V 还要包含命中缓存的历史部分。
    cu_seqlens_k: torch.Tensor | None = None
    # 【prefill 用】本批中最长的 query / key 长度，FlashAttention 用它决定 kernel 的分块大小。
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    # 【两阶段共用】把"本次算出的 K/V"写进 KV Cache 时，每个 token 对应的物理槽位下标。
    slot_mapping: torch.Tensor | None = None
    # 【decode 用】每条序列当前的总长度（也就是 KV Cache 里已有的 token 数）。
    context_lens: torch.Tensor | None = None
    # 【decode / 带前缀缓存的 prefill 用】逻辑块号 -> 物理块号的映射表，
    # 即 PagedAttention 的"页表"。
    block_tables: torch.Tensor | None = None


# 模块级全局单例：外面只能通过下面三个函数访问它（保持"私有"约定）。
_CONTEXT = Context()

def get_context():
    """取出当前的上下文对象（模型各层内部调用，如 Attention / ParallelLMHead）。"""
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    """
    在调用模型之前写入上下文。

    参数顺序与 Context 的字段顺序一一对应，所以这里直接用位置参数构造新对象，
    相当于"整份替换"——保证不会残留上一次前向的值。
    """
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context():
    """一次前向结束后清空上下文，避免残留状态影响下一轮（也顺便释放张量引用）。"""
    global _CONTEXT
    _CONTEXT = Context()
