# vLLM Serving Benchmark

**Notebook:** `vllm_serving_benchmark.ipynb` · **Tier:** Comparable · **Workload:** LLM serving · **GPU:** T4+ (any CUDA GPU)

Runs a real vLLM OpenAI server driven by vLLM's `bench serve` load generator over ShareGPT, across a request-rate sweep. Comparable to public vLLM numbers.

## What it measures
- **TTFT** (mean/median/p99)
- **TPOT / ITL**
- **Output & request throughput**
- latency-vs-throughput curve

## Configuration

| Knob | Default | Notes |
|------|---------|-------|
| `MODEL` | Qwen2.5-1.5B/7B | VRAM tier |
| `DTYPE` | auto/half | fp16 on pre-Ampere |
| `DATASET` | sharegpt | or random |
| `NUM_PROMPTS` | 300 | per sweep point |
| `REQUEST_RATES` | `[4, 16, "inf"]` | req/s sweep (`"inf"` = saturate / max throughput) |
| `TP_SIZE` | 1 | tensor-parallel; recorded for the cost model |

## How to run
1. GPU runtime, run top to bottom.
2. Installs `vllm[bench]` (may need a Colab restart).
3. Downloads ShareGPT, launches server, waits for `/health`+`/v1/models`, sweeps, tears down.

## Output

Writes `vllm_bench_results/...json` with schema **`vllm-serving-bench/1.0`** (field reference in `../DOCS.md` §5) and prints a summary table.

## How to read the results

The rate where p99 TTFT crosses your SLA = max sustainable load; the `inf` row is the max-throughput ceiling. Compare with `compare_results.py`; cost with `cost_model.py`.

## Caveats
- vLLM upgrades torch (Colab restart).
- Won't install on locked-down Kaggle/SageMaker.

**References:** vLLM `bench serve`; ShareGPT_V3.
