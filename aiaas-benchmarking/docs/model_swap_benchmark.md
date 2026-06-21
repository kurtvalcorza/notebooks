# Model-Swap / Cold-Start Benchmark

**Notebook:** `model_swap_benchmark.ipynb` · **Tier:** Systems · **Workload:** Multi-tenant capacity · **GPU:** T4+ (any CUDA GPU)

Measures, per workload (LLM / embeddings / image-gen), the cost of moving a model in and out of the GPU, then gives a co-residency verdict for the 40 GB constraint.

## What it measures
- **load** and **unload** time
- **cold-start tax** = cold first-inference − warm inference
- **resident** and **peak VRAM**
- **co-residency verdict** + **swap-in/out cost** when they don't all fit

## Configuration

| Knob | Default | Notes |
|------|---------|-------|
| `WORKLOADS` | llm / emb / img (VRAM-tiered models) | edit to match what you actually serve |
| `DTYPE` | bfloat16 / float16 |  |
| `IMG_STEPS / IMG_SIZE` | by tier | for the image-gen workload's inference call |

## How to run
1. Installs transformers/sentence-transformers/diffusers; loads → infers → unloads each workload; computes the verdict.

## Output

Writes `model_swap_results/<name>_<platform>_<gpu>.json` with schema **`model-swap-bench/1.0`** (see `../DOCS.md` §5 for the field reference) and prints a summary table.

## How to read the results

Sum of resident footprints vs ~90% of VRAM = the co-residency test. If they don't fit, swap_in_out_s (+ cold-start tax) is the latency a request pays when its model isn't loaded — feed it into your keep-warm vs load-on-demand policy.

## Caveats
- Systems measurement (no accuracy). The companion the throughput notebooks don't cover.

**References:** —
