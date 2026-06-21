# MLPerf Training Runner

**Notebook:** `mlperf_training_benchmark.ipynb` · **Tier:** Standard (training) · **Workload:** Reference-model training · **GPU:** cluster-scale (smoke on 1 GPU)

Portable runner for **MLPerf Training** (references the `kurtvalcorza/training` fork): trains a reference model to a target quality under MLPerf rules. Replaces the dropped LoRA proxy (which had no industry-comparable harness).

## What it measures
- MLLog `throughput`, `eval_accuracy`, `run_start`/`run_stop`
- wall-time

## Configuration

| Knob | Default | Notes |
|------|---------|-------|
| `BENCHMARK_DIR` | image_classification | pick from the fork's benchmark dirs |
| `SMOKE` | True | short pipeline/throughput run, not to-target |
| `DATA_DIR` | '' | REQUIRED for a real run (per-benchmark dataset) |
| `REPO_URL` | kurtvalcorza/training | the training fork |

## How to run
1. Clone fork, pick a benchmark, install its requirements.
2. Launch its reference script (smoke by default), parse MLLog.

## Output

Writes `mlperf_training_results/...json` with schema **`mlperf-training/1.0`** (field reference in `../DOCS.md` §5) and prints a summary table.

## How to read the results

A to-target MLPerf Training run needs the reference world size (often 8+ GPUs) and runs hours-to-days; on one 40 GB A100 use this for a **smoke/throughput** signal only.

## Caveats
- Cluster-scale + large per-benchmark datasets.
- Single 40 GB card = bridge/smoke, not a to-target result.
- Runs belong in a `training`-fork session on adequate hardware.

**References:** mlcommons/training reference implementations; MLLog.
