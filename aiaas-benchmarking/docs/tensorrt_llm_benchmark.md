# TensorRT-LLM Peak-Ceiling Benchmark

**Notebook:** `tensorrt_llm_benchmark.ipynb` · **Tier:** Peak HW · **Workload:** LLM serving · **GPU:** A100 / Hopper (Ampere+)

Runs TensorRT-LLM's `trtllm-bench` (PyTorch backend, `--streaming`) over a concurrency sweep. Same workload as the vLLM notebook, so the delta is the optimized-engine headroom.

## What it measures
- **TTFT / TPOT**
- **output throughput** per concurrency

## Configuration

| Knob | Default | Notes |
|------|---------|-------|
| `MODEL` | Qwen2.5-1.5B/7B | VRAM tier |
| `BACKEND` | pytorch | no engine build |
| `TP_SIZE` | 1 | `--tp` may be `--tp_size` |
| `INPUT_LEN/OUTPUT_LEN` | 1024/1024 |  |
| `CONCURRENCIES` | [1,16,256] | the curve |
| `REPO_URL` | kurtvalcorza/TensorRT-LLM | fork for prepare_dataset.py |

## How to run
1. **A100/Hopper only** (warns on pre-Ampere).
2. Installs tensorrt-llm, shallow-clones the fork for `prepare_dataset.py`, sweeps `trtllm-bench`.

## Output

Writes `trtllm_bench_results/...json` with schema **`trtllm-bench/1.0`** (field reference in `../DOCS.md` §5) and prints a summary table.

## How to read the results

Compare to the vLLM notebook at a matched input/output length; the throughput delta is the headroom.

## Caveats
- `trtllm-bench` flags drift across releases.
- Heavy install + first-run compile.

**References:** NVIDIA/TensorRT-LLM `trtllm-bench`; `prepare_dataset.py`.
