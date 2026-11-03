"""
吞吐基准脚本（README 里那张 vLLM / Nano-vLLM 对比表的来源）。

思路：固定随机种子，造 256 条请求（输入 100~1024 token、输出 100~1024 token），
一次性全部提交给引擎，测"总输出 token 数 / 总耗时"。

两处和真实使用不同，值得留意：
- ignore_eos=True：不管模型是否已经说完，都一直生成到 max_tokens 为止。
  这样每条请求的输出长度是确定的，压测结果才稳定、可复现；
- prompt 直接给随机 token id（不是真实文本）：绕开 tokenizer 与对话模板，
  纯粹压测引擎的调度 + 前向性能。
"""
import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams     # 想跑 vLLM 做对比时，换成这一行导入，
                                           # 并把下面第 20 行那句注释打开（vLLM 的入参格式不同）


def main():
    seed(0)                      # 固定随机种子：每次生成的负载完全一样
    num_seqs = 64                # 请求总数（README 的测试配置）
    max_input_len = 512          # 输入长度上限
    max_ouput_len = 512          # 输出长度上限（变量名里的拼写保持了原样，未改动）

    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    # enforce_eager=False：启用 CUDA Graph，压测跑的就是最快路径；
    # max_model_len=4096：输入最长 1024 + 输出最长 1024，4096 有充足余量
    llm = LLM(path, enforce_eager=False, max_model_len=1024)

    # 随机长度的随机 token id 列表，直接当作 prompt（跳过 tokenizer）
    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    # 每条请求一份采样参数：温度 0.6、忽略 EOS、生成 100~1024 个 token
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    # 先跑一个小请求热身：让 torch.compile 编译、CUDA Graph 捕获、显存池都进入稳态，
    # 否则第一次请求的编译开销会被算进下面的计时里
    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    # use_tqdm=False：关掉进度条，避免进度条自身的开销干扰计时
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = (time.time() - t)
    # 总输出 token 数：因为 ignore_eos=True，每条都会恰好生成 max_tokens 个，
    # 所以直接求和即可
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
