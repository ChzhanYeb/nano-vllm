import time
import statistics
import pandas as pd
from openai import OpenAI
from rich.console import Console
from rich.table import Table

# 配置
BASE_URL = "http://127.0.0.1:8000/v1"
MODEL_NAME = "Qwen3-0.6B"
N_RUNS = 5
client = OpenAI(base_url=BASE_URL, api_key="dummy")
console = Console()

# 定义 156 tokens 的前缀（约 100 个汉字 + 50 个英文字符）
# 你可以根据实际业务替换为你真实的 System Prompt
PREFIX_156 = "你是nano-vllm测试助手。请严格遵循以下规则：1. 所有回答必须包含'测试'二字；2. 字数控制在50字以内；3. 不要输出任何Markdown格式；4. 保持语气客观中立。以下是工具列表：[tool1, tool2, tool3, tool4, tool5, tool6, tool7, tool8, tool9, tool10, tool11, tool12, tool13, tool14, tool15, tool16, tool17, tool18, tool19, tool20]。"
LONG_TEXT_2048 = "请阅读以下长文并回答问题：" + "The quick brown fox jumps over the lazy dog. " * 300

def measure_request(messages, workload_name):
    t0 = time.perf_counter()
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            stream=True,
            max_tokens=128,
            stream_options={"include_usage": True}
        )
        
        t_first = None
        chunk_count = 0
        in_tok, out_tok = 0, 0
        
        for chunk in response:
            now = time.perf_counter()
            if t_first is None and chunk.choices and chunk.choices[0].delta.content:
                t_first = now
            
            if chunk.choices and chunk.choices[0].delta.content:
                chunk_count += 1
                
            if chunk.usage:
                in_tok = chunk.usage.prompt_tokens
                out_tok = chunk.usage.completion_tokens

        t_end = time.perf_counter()
        
        if t_first is None:
            print(f"[{workload_name}] Failed: No content received.")
            return None

        ttft = (t_first - t0) * 1000
        e2e = (t_end - t0) * 1000
        # 如果只有1个token，避免除0
        itl = ((t_end - t_first) / (chunk_count - 1)) * 1000 if chunk_count > 1 else 0
        tps = out_tok / (e2e / 1000) if e2e > 0 else 0

        return {
            "workload": workload_name,
            "concurrency": 1,
            "ttft_ms": ttft,
            "itl_ms": itl,
            "e2e_ms": e2e,
            "in_tok": in_tok,
            "out_tok": out_tok,
            "tok_per_s": tps
        }
    except Exception as e:
        console.print(f"[red]Error in {workload_name}: {e}[/red]")
        return None

def run_workloads():
    results = []

    # 1. Cold (单次请求)
    console.print("[yellow]Testing Cold workload...[/yellow]")
    for _ in range(N_RUNS):
        res = measure_request([{"role": "user", "content": "你好，请简单介绍一下你自己。"}], "cold")
        if res: results.append(res)

    # 2. Warm (同 156 前缀连发 10 次)
    console.print("[yellow]Testing Warm workload...[/yellow]")
    for _ in range(N_RUNS):
        for i in range(10):
            messages = [
                {"role": "system", "content": PREFIX_156},
                {"role": "user", "content": "继续"}
            ]
            res = measure_request(messages, "warm")
            if res: results.append(res)

    # 3. Long (前缀 2048)
    console.print("[yellow]Testing Long workload...[/yellow]")
    for _ in range(N_RUNS):
        messages = [
            {"role": "system", "content": LONG_TEXT_2048},
            {"role": "user", "content": "总结一下。"}
        ]
        res = measure_request(messages, "long")
        if res: results.append(res)

    # 4. Multi (同前缀 4 轮对话)
    console.print("[yellow]Testing Multi workload...[/yellow]")
    for _ in range(N_RUNS):
        history = [{"role": "system", "content": PREFIX_156}]
        for turn in range(4):
            history.append({"role": "user", "content": f"第 {turn+1} 轮"})
            res = measure_request(history, "multi")
            if res: 
                results.append(res)
                # 假象 Assistant 的回复，这里不真实收集，仅为了构建上下文
                history.append({"role": "assistant", "content": "这是测试回复。"})

    return pd.DataFrame(results)

if __name__ == "__main__":
    console.print("[bold green]Starting Baseline Bench...[/bold green]")
    df = run_workloads()
    
    # 落 CSV
    df.to_csv("bench/out/baseline-w1.csv", index=False)
    
    # 聚合取中位数与 P95
    summary = df.groupby("workload").agg(
        ttft_p50=("ttft_ms", "median"),
        ttft_p95=("ttft_ms", lambda x: x.quantile(0.95)),
        itl_p50=("itl_ms", "median"),
        e2e_p50=("e2e_ms", "median"),
        tok_per_s=("tok_per_s", "median"),
        in_tok=("in_tok", "median"),
        out_tok=("out_tok", "median")
    ).reset_index()

    # 终端表格打印
    table = Table(title="Baseline Performance (W1)")
    for col in summary.columns:
        table.add_column(col, justify="right")
    for _, row in summary.iterrows():
        table.add_row(*[f"{v:.2f}" if isinstance(v, float) else str(v) for v in row])
    console.print(table)

    # 生成 Markdown 报告
    cold_ttft = summary[summary["workload"] == "cold"]["ttft_p50"].values[0]
    warm_ttft = summary[summary["workload"] == "warm"]["ttft_p50"].values[0]
    diff = warm_ttft - cold_ttft
    
    with open("reports/baseline-w1.md", "w") as f:
        f.write("# W2-T1 Baseline Performance Report\n\n")
        f.write("## 关键结论\n")
        f.write(f"> **warm 与 cold 的 TTFT 差 = {diff:.2f} ms**（预期 ≈0，因为缓存未生效）\n\n")
        f.write("## 数据表\n")
        f.write(summary.to_markdown(index=False))
    
    console.print(f"[green]Done! Cold TTFT: {cold_ttft:.1f}ms, Warm TTFT: {warm_ttft:.1f}ms[/green]")
    console.print(f"[green]Report saved to reports/baseline-w1.md[/green]")