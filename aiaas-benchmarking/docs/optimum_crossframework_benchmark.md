# Cross-Framework Benchmark (optimum-benchmark)

**Notebook:** `optimum_crossframework_benchmark.ipynb` · **Tier:** Cross-framework · **Workload:** LLM · **GPU:** T4+ (TensorRT EP wants A100-class)

Runs one model through PyTorch, ONNX Runtime (CUDA EP), and ONNX Runtime (TensorRT EP) with HuggingFace `optimum-benchmark` — one harness, identical inputs, only the backend varies.

## What it measures
- **decode throughput** (tokens/s) per backend
- **prefill / per-token latency**
- **peak VRAM**

## Configuration

| Knob | Default | Notes |
|------|---------|-------|
| `MODEL` | gpt2 | exports to ONNX cleanly; swap for a small ungated LLM once confirmed |
| `TASK` | text-generation | passed to each backend |
| `DTYPE` | float16 |  |
| `BATCH_SIZE / SEQ_LEN / GEN_TOKENS` | 1 / 128 / 128 | identical across backends |
| `WARMUP_RUNS` | 10 |  |
| `BACKENDS` | pytorch, onnxruntime-cuda [, onnxruntime-tensorrt] | TensorRT EP added only if ORT exposes it |

## How to run
1. Installs `optimum-benchmark[onnxruntime-gpu]`.
2. Runs each backend in its own subprocess; a backend that fails to export/build is recorded as an error and skipped.

## Output

Writes `optimum_bench_results/<name>_<platform>_<gpu>.json` with schema **`optimum-bench/1.0`** (see `../DOCS.md` §5 for the field reference) and prints a summary table.

## How to read the results

Decode throughput is the headline: ORT (and especially the TensorRT EP) usually beats eager PyTorch; the gap is the optimized-runtime win. A `status` row = that backend failed.

## Caveats
- optimum-benchmark's report schema drifts; extraction is defensive and the full report is saved.
- ONNX export reliability varies by architecture — gpt2 is the safe default.

**References:** huggingface/optimum-benchmark (`Benchmark.launch`, `ONNXRuntimeConfig(provider=…)`).
