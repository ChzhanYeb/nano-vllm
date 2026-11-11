"""
全局配置模块：定义推理引擎启动时需要的所有超参数。

Config 就是"引擎的启动参数集合"：模型在哪、显存怎么划分、并行怎么切、
一个 batch 最多处理多少 token、KV Cache 的页有多大……全部在这里声明。

真正读它的人只有两个：
- Scheduler   ：关心 max_num_batched_tokens / max_num_seqs / kvcache_block_size
- ModelRunner ：关心 gpu_memory_utilization / tensor_parallel_size / kvcache_block_size

注意：字段的声明顺序就是 dataclass 生成 __init__ 时形参的顺序，
      所以这里只加注释，绝不可以调整字段之间的先后次序。
"""
import os
from dataclasses import dataclass
from transformers import AutoConfig


# slots=True 会让 dataclass 给实例生成 __slots__：
# 字段固定、不能临时添加新属性，内存更省、属性访问也略快（在推理热路径上有意义）。
@dataclass(slots=True)
class Config:
    # 模型权重所在的本地目录（HuggingFace 格式，例如 ~/huggingface/Qwen3-0.6B/）。
    # 目前只支持本地路径，不做联网下载。
    model: str

    # ---------------- 调度容量：Scheduler 每次挑选一个 batch 时的约束 ----------------
    # 单次前向（一个 batch）最多喂进去多少 token；
    # prefill 阶段用它把超长 prompt 切成多块（chunked prefill / 分块预填充）。
    max_num_batched_tokens: int = 16384
    # 一个 batch 里最多同时装多少条序列（上限同时受 CUDA Graph 与显存约束）。
    max_num_seqs: int = 512

    # ---------------- 序列长度、显存与并行：ModelRunner 初始化时的依据 ----------------
    # 单条序列允许的最大长度（prompt + 生成），最终会与模型自身的位置编码上限取小值。
    max_model_len: int = 4096
    # 允许占用的显存比例：0.9 表示留 10% 给 CUDA Graph、临时张量、通信 buffer 等。
    gpu_memory_utilization: float = 0.9
    # 张量并行度：用几张 GPU 一起跑同一个模型（1 表示单卡）。
    tensor_parallel_size: int = 1
    # True = 禁用 CUDA Graph，全部走 eager 执行（调试/排错时用）。
    enforce_eager: bool = False

    # ---------------- 运行期回填：不需要用户传，由引擎自己补齐 ----------------
    # HuggingFace 的模型配置对象（层数、hidden_size、头数……），在 __post_init__ 里加载。
    hf_config: AutoConfig | None = None
    # 结束符（end of sentence）的 token id，由 LLMEngine 从 tokenizer 读出后回填。
    eos: int = -1

    # ---------------- KV Cache 分页：PagedAttention 的核心参数 ----------------
    # KV Cache 的分页大小：一"页"（block）存多少个 token 的 K/V。
    # 必须能整除模型最大长度，且是 256 的倍数（见下面的断言）。
    kvcache_block_size: int = 256
    # KV Cache 一共多少页；-1 是占位符，表示"还没算"，
    # 稍后由 ModelRunner.allocate_kv_cache() 按实测剩余显存回填。
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        """
        dataclass 生成的 __init__ 执行完之后会自动调用本方法。

        它的职责是"校验用户传参 + 加载派生信息"，这样用户在构造 Config 时
        只需要关心很少的几个参数，其余都自动补齐。
        """
        # 1. 模型目录必须真实存在
        assert os.path.isdir(self.model)

        # 2. 页大小必须是 256 的整数倍：Attention 的 kernel 与显存对齐都依赖这个约定
        assert self.max_num_batched_tokens % 256 == 0, "max_num_batched_tokens 必须能被块大小整除"

        # 3. 张量并行度限制在 1~8 张卡以内
        assert 1 <= self.tensor_parallel_size <= 8

        # 4. 读取模型自带的 config.json：建模型、算 KV Cache 显存都要用它
        self.hf_config = AutoConfig.from_pretrained(self.model)

        # 5. 用户给的 max_model_len 可能比模型位置编码上限还大，取更小的那个才安全
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
