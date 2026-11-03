"""
Qwen3 模型定义：把 layers/ 下的各种并行算子组装成一个完整的 Transformer。

结构自下而上：
    Qwen3Attention    —— 单个注意力块（GQA + QK-Norm + RoPE）
    Qwen3MLP          —— 前馈网络（SwiGLU）
    Qwen3DecoderLayer —— 一层 Decoder = Attention + MLP + 两个 RMSNorm（Pre-Norm 结构）
    Qwen3Model        —— 词嵌入 + N 层 Decoder + 末尾 RMSNorm
    Qwen3ForCausalLM  —— 再加上 LM Head，并提供 compute_logits 供采样使用

一个工程细节：整个模型的前向**只接收** (input_ids, positions) 两个参数。
像 KV Cache 页表、cu_seqlens、slot_mapping 这些推理元信息，
都通过 nanovllm/utils/context.py 的全局上下文传递，
好处是模型代码本身保持干净，不需要到处都是"推理专用"的参数。
"""
import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3Attention(nn.Module):
    """Qwen3 的注意力块：QKV 投影 -> QK-Norm -> RoPE -> 注意力 -> 输出投影。"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
    ) -> None:
        """
        hidden_size  ：隐藏层维度
        num_heads    ：query 头数（**全局**数量，这里会按 TP 切分）
        num_kv_heads ：key/value 头数；小于 num_heads 时即为 GQA
        max_position ：RoPE 预计算的最大位置数
        head_dim     ：每头维度（Qwen3 可以单独指定，不指定就按 hidden_size/num_heads 推）
        rms_norm_eps ：QK-Norm 的 eps
        qkv_bias     ：QKV 投影是否带 bias
        rope_theta   ：RoPE 基频
        rope_scaling ：RoPE 外推配置（可能覆盖 rope_theta）
        """
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        # 头数必须能被 TP 均分，否则没法按头切分
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size      # 本卡负责的 q 头数
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size  # 本卡负责的 kv 头数
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        # q_size / kv_size：本卡上 q、kv 投影输出的宽度
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        # 注意力缩放：1/sqrt(head_dim)，等价于把点积的方差归一化
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

        # 三个投影合并成一个并行线性层（见 layers/linear.py 的 QKVParallelLinear）
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        # 输出投影：行并行（输入维按头切开，需要 AllReduce）
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,   # 注意这里是"全局"头数×维度，内部会再按 TP 除
            hidden_size,
            bias=False,
        )
        if isinstance(rope_scaling, dict):
            # 某些模型的 config 会把 theta 放在 rope_scaling 里
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
        # 所有层共享同一份 cos/sin 表（get_rope 内部有 lru_cache）
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        # 真正做注意力计算的模块（内部含 KV Cache 写入 + FlashAttention 调用）
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        if not self.qkv_bias:
            # 不带 qkv bias 时启用 QK-Norm（Qwen3 的设计：对每个头的 q/k 先做 RMSNorm，
            # 用来稳定训练、抑制 attention logits 爆炸）；
            # 这里用 head_dim 作为归一化维度，因为是对"每个头"独立归一化。
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        positions     ：[num_tokens] 每个 token 的位置
        hidden_states ：[num_tokens, hidden_size]（变长 batch 打平后的形状）
        返回           ：[num_tokens, hidden_size] 的注意力输出
        """
        # 1. 一次矩阵乘算出 q、k、v（合并权重）
        qkv = self.qkv_proj(hidden_states)
        # 2. 按 [q_size, kv_size, kv_size] 切回三段
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # 3. reshape 成 [num_tokens, num_heads, head_dim]（-1 自动推出 token 数）
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            # 4. QK-Norm：逐头归一化
            q = self.q_norm(q)
            k = self.k_norm(k)
        # 5. 施加旋转位置编码（只旋转 q/k，v 不动）
        q, k = self.rotary_emb(positions, q, k)
        # 6. 注意力（内部会先写 KV Cache，再选 prefill / decode 路径）
        o = self.attn(q, k, v)
        # 7. 多头结果拼回 hidden_size，再做输出投影
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen3MLP(nn.Module):
    """Qwen3 的前馈网络：gate_up 合并投影 -> SwiGLU 激活 -> down 投影。"""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        """
        hidden_size       ：隐藏层维度
        intermediate_size ：FFN 中间层维度
        hidden_act        ：激活函数名（本实现只支持 "silu"）
        """
        super().__init__()
        # gate_proj 与 up_proj 合并成一个权重（两个 [intermediate_size] 段）
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        # down_proj 是行并行：输入维（intermediate 维）按 TP 切分
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        # 本实现只实现了 SwiGLU
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        """x: [num_tokens, hidden_size] -> [num_tokens, hidden_size]。"""
        # 一次算出 gate 和 up 两路（合并权重的好处）
        gate_up = self.gate_up_proj(x)
        # 门控激活：silu(gate) * up
        x = self.act_fn(gate_up)
        # 再投影回 hidden_size
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):
    """一个标准 Pre-Norm Decoder 层：输入归一化 -> Attention -> 归一化 -> MLP。"""

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            # getattr 带默认值：不同版本的 config 字段名可能不同，这里做兼容
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        # Pre-Norm：两个归一化层分别放在 Attention 之前和 MLP 之前
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        为了让"残差相加 + 归一化"能融合，这里用了一种特殊的传参方式：

        - 第一层传 residual=None  ：此时还没有上一层的输出，
          所以直接把 hidden_states 当作待加的残差，做一次普通 RMSNorm；
        - 后续层传 residual 张量  ：RMSNorm 内部完成 (x + residual) 与归一化，
          同时把"未归一化的和"作为新的 residual 返回，继续往下传。

        返回 (本层输出, 新的残差)。
        """
        if residual is None:
            # 第一层：只做归一化，原输入整体作为残差支路
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            # 其余层：残差相加与归一化一次完成
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        # MLP 之前的归一化（同样带残差相加）
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):
    """不含 LM Head 的主体：词嵌入 + N 层 Decoder + 末尾归一化。"""

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        # 词嵌入也是按词表切分的（每卡只存一段）
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        # ModuleList：按 config 指定的层数堆叠 Decoder 层
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 所有层之后的最终归一化
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        input_ids：[num_tokens]（变长 batch 打平后的 token id）
        positions：[num_tokens]（每个 token 的位置）
        返回      ：[num_tokens, hidden_size] 的最终隐藏状态
        """
        # 1. token id -> embedding 向量
        hidden_states = self.embed_tokens(input_ids)
        # 2. residual 初始为 None，交给第一层特殊处理（见 Qwen3DecoderLayer.forward 的说明）
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        # 3. 最后一层之后再做一次归一化；这里同样带残差相加，
        #    所以传 residual 进去，返回值只取归一化结果（残差支路到此结束）
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    """完整的因果语言模型：Qwen3Model + LM Head，就是用户实际加载的那个模型。"""

    # checkpoint 里是分开的 q/k/v、gate/up 权重，而我们的模型把它们合并了。
    # 这张表告诉 utils/loader.py：遇到 "q_proj" 这类名字时，
    # 应该写进合并参数的哪一段（第二项就是 weight_loader 的 shard_id）。
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            # 权重绑定：让 lm_head 与词嵌入共享同一块显存/参数
            # （Qwen3-0.6B 这类小模型常这么做，可以省下一份 [vocab, hidden] 的参数）
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """只跑主干网络，返回 hidden_states（logits 交给 compute_logits 单独算）。"""
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        把隐藏状态投影成词表 logits。

        单独拆出来是因为 CUDA Graph 场景下，"主干前向"和"LM Head + 采样"
        需要分开处理（见 model_runner.py 的 run_model）。
        """
        return self.lm_head(hidden_states)
