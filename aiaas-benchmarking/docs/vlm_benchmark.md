# Vision-Language Model (VLM) Throughput Benchmark

**Notebook:** `vlm_benchmark.ipynb` · **Tier:** Proxy · **Workload:** Multimodal (image+text) · **GPU:** T4+ (7B wants A100)

A Qwen2.5-VL answers a fixed prompt about a synthetic image; sweeps batch size and reports decode throughput, latency, and VRAM.

## What it measures
- **output tokens/s** (decode throughput)
- **end-to-end latency** per call
- **peak VRAM**

## Configuration

| Knob | Default | Notes |
|------|---------|-------|
| `MODEL` | Qwen2.5-VL-3B (T4) / 7B (A100) | VRAM tier |
| `DTYPE` | bfloat16 / float16 |  |
| `BATCH_SIZES` | [1, 4] | OOM-guarded |
| `GEN_TOKENS` | 64 | fixed answer length |
| `NUM_ITERS / IMG_SIZE` | 3 / 512 |  |

## How to run
1. Installs transformers/accelerate/pillow; builds a synthetic image + chat-templated prompt; sweeps batch sizes.

## Output

Writes `vlm_bench_results/<name>_<platform>_<gpu>.json` with schema **`vlm-bench/1.0`** (see `../DOCS.md` §5 for the field reference) and prints a summary table.

## How to read the results

output tokens/s at the largest batch that fits is multimodal serving throughput; latency at batch 1 is best-case. VLMs spend prefill on image tokens, so latency exceeds a text-only LLM.

## Caveats
- Proxy tier; targets recent `transformers` (`AutoModelForImageTextToText`).

**References:** transformers VLM (`AutoProcessor` + `AutoModelForImageTextToText`); Qwen2.5-VL.
