# Model-Swap / Cold-Start Benchmark

**Notebook:** `model_swap_benchmark.ipynb` · **Tier:** Systems (tooling) · **Workload:** Multi-tenant capacity · **GPU:** T4+ (any CUDA GPU)

Measures per-workload load/unload time, cold-start tax, and resident/peak VRAM, then gives a co-residency verdict against the card's detected VRAM (no fixed card size assumed). Capacity tooling, not a comparable benchmark.

## What it measures
- **load / unload** time
- **cold-start tax** (cold-warm)
- **resident / peak VRAM**
- **co-residency verdict** + **swap-in/out cost**

## Configuration

| Knob | Default | Notes |
|------|---------|-------|
| `WORKLOADS` | llm / emb / img | tier-selected models — large: Qwen2.5-7B + SDXL; small: Qwen2.5-1.5B + sd-turbo; emb: bge-small-en-v1.5 |
| `DTYPE` | bf16 (Ampere+) / fp16 | applied to all three workloads |
| `IMG_STEPS` / `IMG_SIZE` | 20 / 1024 (large) · 2 / 512 (small) | image-gen diffusion steps and resolution |

## How to run
1. Installs transformers/sentence-transformers/diffusers.
2. An **untimed warmup pass** loads→infers→unloads each workload once (so the timed
   pass measures steady-state GPU swap latency, not first-time download / import /
   kernel warmup).
3. The timed pass loads→infers→unloads each workload and computes the verdict.

## Output

Writes `model_swap_results/...json` with schema **`model-swap-bench/1.0`** (field reference in `../DOCS.md` §5) and prints a summary table.

## How to read the results

The co-residency test is **peak-aware**: for each workload it checks
`resident_total − resident_w + peak_w` (all models resident, one active at its
measured inference peak) against ~90% of VRAM. The verdict is:

- **CO-RESIDE** — that worst-case need fits, no swapping required;
- **swap required** — it doesn't fit; `worst_swap_s` (evict one model's `unload_s`
  + the requested model's `load_s` + its cold-start tax, worst over workload pairs)
  is the latency a request pays when its model isn't already resident;
- **INCONCLUSIVE** — a workload failed to load/run, or >0.1 GB was left unreclaimed
  after unloads, so the resident accounting can't be trusted.

## Caveats
- Systems measurement (no accuracy), kept as supporting tooling.

**References:** —
