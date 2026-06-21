# MLPerf Inference Runner

**Notebook:** `mlperf_inference_benchmark.ipynb` · **Tier:** Standard · **Workload:** Vision · SDXL · whisper · **GPU:** A100 (heavy)

Portable bridge to MLPerf Inference. Two paths: (a) the `kurtvalcorza/inference` fork's LoadGen reference app for vision classification/detection (modeled on its `GettingStarted.ipynb`); and (b) an **MLCFlow** path for any model — incl. **sdxl** (comparable image-gen) and **whisper** (comparable ASR).

## What it measures
- Official `mlperf_log_summary.txt` — QPS/throughput, latency percentiles, **VALID**
- **Accuracy** (Top-1 / mAP / etc., the gate)
- LoadGen **scenario** (Offline/Server/SingleStream)

## Configuration

| Knob | Default | Notes |
|------|---------|-------|
| `MODEL (path a)` | mobilenet/resnet50/retinanet | smoke=mobilenet+fake imagenet |
| `MLC_MODEL (path b)` | sdxl / whisper / … | MLCFlow universal runner |
| `SCENARIO` | Offline |  |
| `ACCURACY / execution_mode` | True / test | `valid` for submission-grade |
| `DATASET_MODE / DATA_DIR` | fake / '' | real ImageNet/OpenImages for credible |
| `REPO_URL` | kurtvalcorza/inference | the Standard-tier fork |

## How to run
1. Path a: clone fork, build LoadGen+app, `run_local.sh ... --accuracy`.
2. Path b: `pip install cm4mlops`; `cmr run-mlperf,inference --model=sdxl|whisper ...`.

## Output

Writes `mlperf_results/...json` with schema **`mlperf-inference/1.0`** (field reference in `../DOCS.md` §5) and prints a summary table.

## How to read the results

Check `Result is VALID`, QPS, latency percentiles, accuracy vs the gate. Smoke runs are not VALID — use real datasets + `execution_mode=valid` (or the MLCFlow path) for credible numbers.

## Caveats
- Heavy: clones/builds the fork, needs datasets; wants the A100.
- MLPerf/MLCFlow field names drift — parsing is defensive.
- Credible/submission runs belong in an `inference`-scoped session.

**References:** mlcommons/inference `GettingStarted.ipynb`; LoadGen; MLCFlow `run-mlperf,inference`.
