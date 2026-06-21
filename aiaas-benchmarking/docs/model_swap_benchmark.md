# Model-Swap / Cold-Start Benchmark

**Notebook:** `model_swap_benchmark.ipynb` · **Tier:** Systems (tooling) · **Workload:** Multi-tenant capacity · **GPU:** T4+ (any CUDA GPU)

Measures per-workload load/unload time, cold-start tax, and resident/peak VRAM, then gives a co-residency verdict for the 40 GB constraint. Capacity tooling, not a comparable benchmark.

## What it measures
- **load / unload** time
- **cold-start tax** (cold-warm)
- **resident / peak VRAM**
- **co-residency verdict** + **swap-in/out cost**

## Configuration

| Knob | Default | Notes |
|------|---------|-------|
| `WORKLOADS` | llm/emb/img (VRAM-tiered) | edit to what you serve |
| `DTYPE` | bf16/fp16 |  |

## How to run
1. Installs transformers/sentence-transformers/diffusers.
2. Loads→infers→unloads each workload; computes the verdict.

## Output

Writes `model_swap_results/...json` with schema **`model-swap-bench/1.0`** (field reference in `../DOCS.md` §5) and prints a summary table.

## How to read the results

Sum of resident footprints vs ~90% of VRAM = the co-residency test. If they don't fit, swap_in_out_s (+ cold-start tax) is the latency a request pays when its model isn't loaded.

## Caveats
- Systems measurement (no accuracy), kept as supporting tooling.

**References:** —
