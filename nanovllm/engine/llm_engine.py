"""
LLMEngine：把调度器、模型执行器、tokenizer 串起来的"总入口"。

一次 step（一个 iteration）的数据流：

    add_request(prompt)                  -> 生成 Sequence，进入 Scheduler.waiting
      │
      ▼
    Scheduler.schedule()                 -> 选出本轮要算的序列（prefill 或 decode）
      │
      ▼
    ModelRunner.call("run", seqs, ...)   -> 真正跑模型；
                                            张量并行时 rank 0 会把任务广播给其他 rank
      │
      ▼
    Scheduler.postprocess(...)           -> 登记前缀缓存 / 追加 token / 判断结束
      │
      ▼
    step() 返回"本轮结束的序列"          -> generate() 收集结果，最后 decode 成文本
"""
import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        """
        model ：本地模型目录
        kwargs：透传给 Config 的参数（如 enforce_eager / tensor_parallel_size / max_num_seqs）

        这里只把 Config 认识的字段挑出来，其它参数被忽略——
        这样即使用户按 vLLM 的习惯多传了参数，也不会因为类型不匹配而报错。
        """
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        # Sequence 的页大小必须与 KV Cache 的分页大小一致（它是类属性，全局设置一次）
        Sequence.block_size = config.kvcache_block_size
        self.ps = []          # 张量并行的子进程
        self.events = []      # 每个子进程一个事件，rank 0 用它们通知"有新任务/新数据"
        # spawn 模式：为子进程启动全新的解释器。
        # CUDA 多进程必须用 spawn（fork 会复制已初始化的 CUDA 上下文，导致各种诡异错误）。
        ctx = mp.get_context("spawn")
        # rank 1..N-1 放在子进程里（每张卡一个进程）
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        # rank 0 直接建在当前进程里，这样主进程可以同步地驱动它
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        # 结束符：Scheduler 判断"生成该停了"要用它，所以回填进 config
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        # 注册退出钩子：即使脚本异常结束、解释器关闭，也能优雅释放 GPU 显存与子进程
        atexit.register(self.exit)

    def exit(self):
        """
        优雅退出：通知所有 rank 收工，然后回收子进程。

        - call("exit") 会在 rank 0 上执行 exit()，同时把命令广播给其他 rank；
        - del self.model_runner 触发析构，把 rank 0 的模型权重与 KV Cache 显存释放掉；
        - 最后 join 等所有子进程真正结束。
        """
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """
        提交一条请求。

        prompt 既可以是字符串（内部用 tokenizer 编码），
        也可以直接是 token id 列表（跳过编码，方便做实验/对齐测试）。
        """
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        """
        推进"一个 iteration"：调度 → 前向 → 后处理。

        返回 (outputs, num_tokens)：
          - outputs   ：[(seq_id, 生成出的 token 列表), ...]，只包含**本轮刚结束**的序列
          - num_tokens：本轮实际处理的 token 数。这里用了一个符号约定：
                        prefill 返回正数（本轮真正喂进模型的 token 数；
                        若触发 chunked prefill，那就只是 prompt 中的一块），
                        decode 返回负数（-序列条数，每条 1 个 token）。
                        这样 generate() 一眼就能分辨这轮是哪种阶段，用来统计吞吐。
        """
        seqs, is_prefill = self.scheduler.schedule()
        # prefill：本轮 token 数 = 各序列被调度 token 之和；decode：每序列 1 个 token → -len(seqs)
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        # 真正跑模型（TP 时这行内部会把任务广播给其他 rank；
        # 只有 rank 0 会返回采样出的 token id）
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        # 后处理：登记缓存、追加 token、把结束的序列标记为 FINISHED
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        # 只把"本轮刚刚结束"的序列作为结果返回
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        """所有请求都处理完了吗（调度器里两个队列都为空）。"""
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        """
        离线批量推理的主入口（vLLM 风格）。

        prompts         ：一组 prompt（字符串或 token id 列表）
        sampling_params ：一组采样参数，或一份"所有 prompt 共用"的参数
        use_tqdm        ：是否显示进度条

        返回一个列表，元素形如 {"text": 解码后的文本, "token_ids": token id 列表}，
        顺序与传入的 prompts 一致。
        """
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        # 只传了一份采样参数时，复制成与 prompt 等长的一份（每条请求独立持有）
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        # 把所有请求一次性提交进调度器的等待队列（真正开始生成是在下面的循环里）
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        # 两种吞吐统计：prefill 看"每秒吃进多少 prompt token"，
        # decode 看"每秒生成多少 token"（这才是用户最关心的指标）
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                # 正数 = prefill：token 数 / 本轮耗时
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                # 负数 = decode，取相反数就是本轮生成的 token 数
                decode_throughput = -num_tokens / (perf_counter() - t)
            # 把两个吞吐实时显示在进度条右侧
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            # 收集本轮结束的序列，并推进进度条
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        # seq_id 是按提交顺序自增的，按它排序即可恢复成输入顺序
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        # 把 token id 解码成人类可读文本（同时保留 token_ids 方便调试）
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
