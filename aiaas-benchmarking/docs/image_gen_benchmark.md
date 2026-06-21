# Image-Generation Throughput Benchmark

**Notebook:** `image_gen_benchmark.ipynb` · **Tier:** Proxy · **Workload:** Image generation · **GPU:** T4+ (SDXL wants A100)

Sweeps batch size with `diffusers` and reports images/s, seconds/image, and VRAM. VRAM-tiered: SDXL on A100, SD-Turbo on a T4.

## What it measures
- **images/s** and **seconds/image**
- **peak VRAM**

## Configuration

| Knob | Default | Notes |
|------|---------|-------|
| `MODEL` | SDXL base 1.0 (A100) / sd-turbo (T4) | VRAM tier |
| `STEPS` | 30 (SDXL) / 4 (turbo) | denoise steps |
| `HEIGHT / WIDTH` | 1024 / 512 | by tier |
| `BATCH_SIZES` | [1, 2, 4] | images per call; OOM-guarded |
| `NUM_ITERS` | 3 | timed iterations after warmup |
| `DTYPE` | bfloat16 / float16 |  |

## How to run
1. Installs diffusers/transformers/accelerate; loads the pipeline; sweeps batch sizes (catches OOM).

## Output

Writes `image_gen_bench_results/<name>_<platform>_<gpu>.json` with schema **`image-gen-bench/1.0`** (see `../DOCS.md` §5 for the field reference) and prints a summary table.

## How to read the results

images/s at the largest batch that fits is the generation ceiling; s/image at batch 1 is best-case latency. Feeds a $/image estimate.

## Caveats
- Proxy tier. For credible numbers use MLPerf stable-diffusion-xl.

**References:** diffusers `AutoPipelineForText2Image`; Stability SDXL / SD-Turbo.
