# MLPerf Inference Runner

**Notebook:** `mlperf_inference_benchmark.ipynb` · **Tier:** Standard · **Workload:** Vision (class./detection) · **GPU:** A100 (heavy)

Portable bridge to MLPerf Inference: clones the `kurtvalcorza/inference` fork, builds LoadGen + the reference app, and runs the vision classification/detection benchmark under a LoadGen scenario with the accuracy gate. Modeled on the fork's `GettingStarted.ipynb`.

## What it measures
- Official **`mlperf_log_summary.txt`** — QPS/throughput, latency percentiles, VALID
- **Accuracy** (Top-1 for resnet50, mAP for retinanet)
- LoadGen **scenario** (Offline / Server / SingleStream / MultiStream)

## Configuration

| Knob | Default | Notes |
|------|---------|-------|
| `MODEL` | mobilenet (smoke) / resnet50 / retinanet | smoke default = mobilenet + fake imagenet |
| `BACKEND / DEVICE` | onnxruntime / gpu | ORT-GPU on a GPU |
| `SCENARIO` | Offline | Offline/Server datacenter; SingleStream/MultiStream edge |
| `ACCURACY` | True | also run the accuracy pass (the gate) |
| `DATASET_MODE / DATA_DIR` | fake / '' | set 'real' + DATA_DIR for ImageNet/OpenImages |
| `EXTRA_OPS` | --time 10 … (smoke) | drop for a full VALID run |
| `REPO_URL` | kurtvalcorza/inference | the Standard-tier fork |

## How to run
1. Clones the fork (`--recurse-submodules`), builds LoadGen + the app, installs the backend.
2. Downloads the model, prepares the dataset (fake smoke / real), runs `run_local.sh`, parses the LoadGen summary.

## Output

Writes `mlperf_results/<name>_<platform>_<gpu>.json` with schema **`mlperf-inference/1.0`** (see `../DOCS.md` §5 for the field reference) and prints a summary table.

## How to read the results

Check `Result is VALID`, the QPS/throughput, the latency percentiles, and the accuracy vs the gate. Smoke runs are not VALID (shortened) — for credible numbers use real datasets + drop EXTRA_OPS, or the MLCFlow path (`mlcr run-mlperf-inference-app …`).

## Caveats
- Heavy: clones+builds the fork, needs a dataset; really wants the A100.
- MLPerf field names drift across versions — parsing is defensive.
- Credible/submission-grade runs belong in an `inference`-scoped session.

**References:** mlcommons/inference vision class./detection `GettingStarted.ipynb`; LoadGen; MLCFlow `run-mlperf-inference-app`.
