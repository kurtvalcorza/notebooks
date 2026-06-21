# TensorRT-LLM Peak-Ceiling Benchmark

**Notebook:** `tensorrt_llm_benchmark.ipynb` · **Tier:** Peak HW · **Workload:** LLM serving · **GPU:** A100 / Hopper (Ampere+)

Runs TensorRT-LLM's own `trtllm-bench` (PyTorch backend, CUDA graphs on; no engine build) over a concurrency sweep with `--streaming`. Same workload as the vLLM notebook, so the gap between the two is the optimized-engine headroom on the GPU.

## What it measures
- **TTFT / TPOT** and **output throughput** at each concurrency
- Concurrency=1 = best single-stream latency; max concurrency = peak ceiling

## Configuration

| Knob | Default | Notes |
|------|---------|-------|
| `MODEL` | Qwen2.5-1.5B / 7B | VRAM tier |
| `BACKEND` | pytorch | PyTorch flow — no separate engine build |
| `TP_SIZE` | 1 | tensor-parallel size; `--tp` flag may be `--tp_size` on some releases |
| `STREAMING` | True | required for TTFT/TPOT |
| `INPUT_LEN / OUTPUT_LEN` | 1024 / 1024 | token-norm-dist synthetic lengths (stdev 0) |
| `NUM_REQUESTS` | 1000 | dataset size |
| `CONCURRENCIES` | [1, 16, 256] | in-flight requests per run → the curve |
| `REPO_URL` | kurtvalcorza/TensorRT-LLM | fork shallow-cloned for `prepare_dataset.py` |

## How to run
1. **A100/Hopper only** — recent TensorRT-LLM drops Turing/T4 (the notebook warns).
2. Installs `tensorrt-llm` from NVIDIA's index (large), shallow-clones the fork for `prepare_dataset.py`, builds a synthetic dataset, then sweeps `trtllm-bench throughput`.

## Output

Writes `trtllm_bench_results/<name>_<platform>_<gpu>.json` with schema **`trtllm-bench/1.0`** (see `../DOCS.md` §5 for the field reference) and prints a summary table.

## How to read the results

Compare to the vLLM notebook at a matched input/output length (use vLLM's `random` dataset with the same lengths). The throughput delta is the TRT-LLM headroom.

## Caveats
- `trtllm-bench` flag names drift between releases — check `trtllm-bench throughput -h`.
- Heavy install + first-run compile.

**References:** NVIDIA/TensorRT-LLM `trtllm-bench`; `benchmarks/cpp/prepare_dataset.py`.
