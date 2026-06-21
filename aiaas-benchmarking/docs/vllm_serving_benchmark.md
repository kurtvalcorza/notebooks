# vLLM Serving Benchmark

**Notebook:** `vllm_serving_benchmark.ipynb` · **Tier:** Comparable · **Workload:** LLM serving · **GPU:** T4+ (any CUDA GPU)

Runs a real vLLM OpenAI server and drives it with vLLM's `bench serve` load generator over the ShareGPT request distribution, across a request-rate sweep. Produces the serving metrics the industry reports, comparable to public vLLM numbers.

## What it measures
- **TTFT** — time-to-first-token (mean/median/p99)
- **TPOT / ITL** — per-output-token / inter-token latency
- **Output throughput** (tokens/s) and **request throughput** (req/s)
- Latency-vs-throughput curve across the rate sweep (the capacity 'knee')

## Configuration

| Knob | Default | Notes |
|------|---------|-------|
| `MODEL` | Qwen2.5-1.5B (T4) / 7B (A100) | auto-selected by VRAM tier |
| `DTYPE` | auto / half | fp16 forced on pre-Ampere (T4) since they lack bf16 |
| `DATASET` | sharegpt | or `random` for fixed input/output lengths (no download) |
| `RANDOM_INPUT_LEN / RANDOM_OUTPUT_LEN` | 1024 / 1024 | used only when DATASET=random |
| `NUM_PROMPTS` | 300 | requests per sweep point |
| `REQUEST_RATES` | [4, 16, "inf"] | req/s sweep; `inf` = fire all at once (max throughput) |
| `TP_SIZE` | 1 | tensor-parallel GPUs; recorded so the cost model charges only those |
| `GPU_MEM_UTIL / MAX_MODEL_LEN` | 0.90 / 4096 | vLLM server knobs |

## How to run
1. Open on a GPU runtime; run top to bottom.
2. Cell 1 installs `vllm[bench]` (may upgrade torch → restart runtime, re-run from cell 2).
3. It downloads ShareGPT (~200 MB once), launches the server, waits for `/health` + confirms `/v1/models`, runs the sweep, writes the JSON, then tears the server down.

## Output

Writes `vllm_bench_results/<name>_<platform>_<gpu>.json` with schema **`vllm-serving-bench/1.0`** (see `../DOCS.md` §5 for the field reference) and prints a summary table.

## How to read the results

Find the request rate where p99 TTFT crosses your SLA — that's max sustainable load. The `inf` row is the raw max-throughput ceiling. Compare across platforms with `compare_results.py`; cost it with `cost_model.py`.

## Caveats
- vLLM pins its own CUDA-matched torch; on Colab expect a restart.
- On locked-down envs (Kaggle/SageMaker) vLLM may not install — use proxy notebooks.

**References:** vLLM `bench serve`; ShareGPT_V3 dataset.
