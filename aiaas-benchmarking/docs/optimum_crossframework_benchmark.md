# Cross-Framework Benchmark (optimum-benchmark)

**Notebook:** `optimum_crossframework_benchmark.ipynb` · **Tier:** Cross-framework · **Workload:** LLM · **GPU:** T4+ (TensorRT EP wants A100-class)

One model through PyTorch, ONNX Runtime (CUDA EP), and ONNX Runtime (TensorRT EP) with HuggingFace `optimum-benchmark` — only the backend varies.

## What it measures
- **decode throughput** per backend
- **prefill / per-token latency**
- **peak VRAM**

## Configuration

| Knob | Default | Notes |
|------|---------|-------|
| `MODEL` | gpt2 | exports cleanly; swap for a small LLM |
| `TASK` | text-generation |  |
| `DTYPE` | float16 |  |
| `BATCH_SIZE/SEQ_LEN/GEN_TOKENS` | 1/128/128 | identical across backends |
| `BACKENDS` | pytorch, onnxruntime-cuda[, -tensorrt] | TensorRT EP added only if available |

## How to run
1. Installs `optimum-benchmark[onnxruntime-gpu]`.
2. Each backend runs in its own subprocess; failures are recorded and skipped.

## Output

Writes `optimum_bench_results/...json` with schema **`optimum-bench/1.0`** (field reference in `../DOCS.md` §5) and prints a summary table.

## How to read the results

Decode throughput is the headline — ORT/TensorRT EP usually beat eager PyTorch; the gap is the optimized-runtime win.

## Caveats
- Report schema drifts; extraction is defensive.
- ONNX export reliability varies — gpt2 is safe.

**References:** huggingface/optimum-benchmark.
