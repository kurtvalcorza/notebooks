# AIaaS Benchmarking — Full Documentation

Portable, cross-platform benchmarks for sizing an **AI-as-a-Service** platform on a
single **A100 (40 GB)** server and comparing it against cloud notebook services
(Colab, SageMaker Studio Lab, Kaggle, …).

This is the reference manual. For the *why* (tiers, the five reasons a homemade
timing notebook isn't industry-comparable, the 40 GB fit table, phases), read
**`BENCHMARKING_STRATEGY.md`** first. For deep per-notebook pages see **`docs/`**.

> **Design principle:** every benchmark notebook in this package is
> **industry/community-comparable** — it runs the *same standardized harness the
> published numbers use* (vLLM, TensorRT-LLM, optimum-benchmark, MLPerf, MTEB).
> Proxy-tier timings with no comparable harness were dropped; the NAIRA PoC v2
> remains as the explicitly-labeled cross-platform proxy.

---

## 1. What this package is

Self-contained notebooks plus aggregation/economics tooling. Each benchmark
notebook auto-detects platform + GPU, tiers itself by VRAM, uses ungated models,
writes a normalized result JSON to a `*_results/` folder, and prints a summary.

### Two-repo split
| Repo | Holds |
|------|-------|
| **`kurtvalcorza/notebooks`** (here, `aiaas-benchmarking/`) | the portable comparable benchmarks + tooling |
| **`kurtvalcorza/inference`**, **`training`**, **`optimum(-benchmark)`**, **`TensorRT(-LLM)`** | upstream forks the notebooks drive / reference for the heavy runs |

---

## 2. Benchmark tiers → harness

| Tier | Harness (industry/community standard) | Notebook |
|------|----------------------------------------|----------|
| **Comparable** (LLM serving) | vLLM `bench serve` + ShareGPT | `vllm_serving_benchmark` |
| **Peak HW** (LLM serving) | TensorRT-LLM `trtllm-bench` | `tensorrt_llm_benchmark` |
| **Cross-framework** (LLM) | optimum-benchmark | `optimum_crossframework_benchmark` |
| **Standard** (vision / image-gen / ASR) | MLPerf Inference (LoadGen + MLCFlow) | `mlperf_inference_benchmark` |
| **Standard** (retrieval) | MTEB leaderboard | `mteb_benchmark` |
| **Standard** (training) | MLPerf Training | `mlperf_training_benchmark` |
| **Systems / economics** (kept tooling) | — | `model_swap_benchmark`, `cost_model.py`, `report.ipynb`, `compare_results.py` |

---

## 3. Notebook & script catalog

| File | Workload | Tier | Measures | Result schema | GPU |
|------|----------|------|----------|---------------|-----|
| `vllm_serving_benchmark.ipynb` | LLM serving | Comparable | TTFT, TPOT/ITL, throughput (rate sweep) | `vllm-serving-bench/1.0` | T4+ |
| `tensorrt_llm_benchmark.ipynb` | LLM serving | Peak HW | TTFT, TPOT, throughput (concurrency sweep) | `trtllm-bench/1.0` | A100/Hopper |
| `optimum_crossframework_benchmark.ipynb` | LLM | Cross-framework | decode throughput, latency, VRAM per backend | `optimum-bench/1.0` | T4+ |
| `mlperf_inference_benchmark.ipynb` | vision · sdxl · whisper | Standard | LoadGen QPS / latency / accuracy (VALID) | `mlperf-inference/1.0` | A100 |
| `mteb_benchmark.ipynb` | embeddings + reranking | Standard | MAP/MRR, Spearman, accuracy (leaderboard) | `mteb-bench/1.0` | any |
| `mlperf_training_benchmark.ipynb` | reference-model training | Standard | MLLog throughput / eval_accuracy / time | `mlperf-training/1.0` | cluster (smoke on 1) |
| `model_swap_benchmark.ipynb` | multi-tenant systems | Systems | load/unload, cold-start tax, resident/peak VRAM, co-residency | `model-swap-bench/1.0` | T4+ |
| `cost_model.py` | — | Systems | `$/M-tokens` (energy + amortized HW) | `vllm-cost-model/1.0` | CPU |
| `compare_results.py` | — | — | cross-platform comparison table | — | CPU |
| `report.ipynb` | — | — | combined charted report over all schemas | — | CPU |

### Dropped (no industry-comparable harness)
Embeddings-throughput, image-gen, ASR, and VLM **proxy** notebooks were removed.
Their comparable replacements: image-gen → MLPerf **sdxl**, ASR → MLPerf
**whisper** (both in `mlperf_inference_benchmark` via MLCFlow), embeddings →
**MTEB**. VLM has no standardized perf harness, so it's out (the PoC covers VLM
as a labeled proxy if needed).

The **LoRA/QLoRA training proxy** is being replaced, not kept: it has no
industry-comparable harness, so it is dropped in favor of the **MLPerf Training**
runner (`mlperf_training_benchmark.ipynb`, the comparable harness) — this lands
via **PR #23**. Until #23 merges, `lora_qlora_train_benchmark.ipynb` (schema
`lora-train-bench/1.0`, read by `compare_results.flatten_train`) still ships on
`main`; #23 removes the notebook and that code path together. The catalog and
schema tables above describe the post-#23 state.

### Build status (PRs into `main`)
| PR | Adds / changes |
|----|----------------|
| #10, #11 | initial package (vLLM serving, compare_results) — **merged** |
| #12 | `cost_model.py` (+ serving records `tensor_parallel_size`) |
| #13 | `optimum_crossframework_benchmark.ipynb` |
| #14 | `tensorrt_llm_benchmark.ipynb` |
| #15 | `report.ipynb` |
| #16 | `model_swap_benchmark.ipynb` |
| #18 | `DOCS.md`, `SESSION_HANDOFF.md` |
| #19 | `mlperf_inference_benchmark.ipynb` (LoadGen app + MLCFlow sdxl/whisper) |
| #20 | `docs/` per-notebook pages |
| #22 | `mteb_benchmark.ipynb` |
| #23 | `mlperf_training_benchmark.ipynb` (drops the LoRA proxy) |

---

## 4. Quick start

1. Open a notebook on a **GPU** runtime; run top to bottom. It installs deps,
   captures the environment, picks the model by VRAM, runs, and writes
   `*_results/<name>_<platform>_<gpu>.json`.
2. Repeat per platform, collect the JSONs, then aggregate:
   ```bash
   python compare_results.py 'vllm_bench_results/*.json'
   python cost_model.py     'vllm_bench_results/*.json' --price 0.12 --util 0.5
   ```
   …or open **`report.ipynb`** from `aiaas-benchmarking/` for a combined report.

### Platform notes
- **Colab:** vLLM/TensorRT-LLM may upgrade torch → *Runtime → Restart*, re-run from install.
- **T4 (Turing):** runs the small tiers; **not** supported by TensorRT-LLM (Ampere+).
- **Kaggle / SageMaker:** locked-down; the heavy installs may fail — use the PoC there.
- **MLPerf / MLPerf-Training:** heavy (datasets, LoadGen); really want the A100 + a fork session.

---

## 5. Result JSON schemas

All carry an `env` block (platform, gpu_name, gpu_count, vram_total_gb,
compute_capability, cuda, driver, torch, python) and a `schema` string.

- **`vllm-serving-bench/1.0`** — `model`, `tensor_parallel_size`, `request_rates`,
  `sweep`: {rate → `output_throughput`, `request_throughput`, `p99_ttft_ms`,
  `p99_tpot_ms` (or `percentiles_*_ms`)}.
- **`trtllm-bench/1.0`** — `engine`, `backend`, `tensor_parallel_size`,
  `input_len`/`output_len`, `summary`: {concurrency → out tok/s, req/s, TTFT, TPOT}.
- **`optimum-bench/1.0`** — `task`, `dtype`, `summary`: {backend → decode throughput,
  latency, VRAM}, `reports` (raw).
- **`mlperf-inference/1.0`** — `model`/`MLC_MODEL`, `scenario`, `device`, `via`
  (loadgen-app | mlcflow), `loadgen_summary` (QPS, latency percentiles, VALID),
  `app_results`.
- **`mteb-bench/1.0`** — `model`, `results`: [{task, type, main_score}],
  `average_main_score`.
- **`mlperf-training/1.0`** — `benchmark`, `smoke`, `metrics` (MLLog throughput /
  eval_accuracy / run_start / run_stop), `result`.
- **`model-swap-bench/1.0`** — `results`: [{name, load_s, cold_start_tax_s,
  resident_vram_gb, peak_vram_gb, unload_s, swap_in_out_s}], `analysis` (verdict).
- **`vllm-cost-model/1.0`** — `assumptions`, `results`: per-run `$/M` energy /
  hardware / total.

`compare_results.py` reads the serving, MLPerf-training, and PoC schemas;
`report.ipynb` reads the comparable schemas + model-swap.

---

## 6. Aggregation & reporting

- **`compare_results.py`** — `python compare_results.py 'vllm_bench_results/*.json'`;
  one table (platforms × metrics), skips non-result JSONs, merges per platform label.
- **`cost_model.py`** — serving throughput → `$/M-tokens` (energy + amortized HW).
  Flags: `--price --pue --util --amortization-years --server-overhead --power-watts
  --gpu-capex --gpus` (`--gpus` = GPUs vLLM used; defaults to recorded
  `tensor_parallel_size`, else 1 — never every visible GPU). `--json` writes a file.
- **`report.ipynb`** — imports the two scripts, summarizes the comparable schemas +
  model-swap, draws charts, writes `aiaas_report.md` / `.json`.

---

## 7. Metrics glossary

- **TTFT** — time to first token. **TPOT/ITL** — per-output-token latency.
- **Output / request throughput** — tokens/s, requests/s.
- **MAP/MRR** (MTEB reranking), **Spearman** (STS), **main_score** (MTEB headline).
- **VALID** — MLPerf LoadGen run met the minimum-duration/query rules.
- **Accuracy gate** — MLPerf's quality target (Top-1, mAP, eval_accuracy).
- **Resident / peak VRAM**, **cold-start tax**, **$/M-tokens** — as in the systems/cost tooling.

---

## 8. Fork workflow (heavy / submission-grade runs)

A Claude session is scoped to specific repos; to run the heavy paths, start a
session scoped to the fork, then:
- **MLPerf Inference** (vision, sdxl, whisper, llama) → `kurtvalcorza/inference`
  (LoadGen app or MLCFlow `run-mlperf,inference`).
- **MLPerf Training** → `kurtvalcorza/training` (reference impls; multi-GPU).
- **Cross-framework runs** → `kurtvalcorza/optimum-benchmark`.
- **Peak LLM runs** → `kurtvalcorza/TensorRT-LLM`.

---

## 9. Capacity & the 40 GB constraint

The serving workloads don't all fit resident in 40 GB at once.
`model_swap_benchmark.ipynb` gives the co-residency verdict + swap cost; combine
with the throughput notebooks (capacity at SLA) and `cost_model.py` ($/M-tokens) to
decide what stays warm vs. loads on demand.

---

## 10. Troubleshooting

- **vLLM upgraded torch** → restart runtime, re-run from install.
- **TensorRT-LLM on a T4** → unsupported (Turing); use the vLLM notebook.
- **`trtllm-bench` flag error** → check `trtllm-bench throughput -h` (`--tp` vs `--tp_size`).
- **ONNX export fails (optimum)** → use `gpt2`; that backend row is recorded as an error.
- **MLPerf smoke run "not VALID"** → expected; use real datasets + `execution_mode=valid`.
- **MTEB task name unknown** → it's skipped; browse `mteb.get_tasks(task_types=[...])`.

---

## 11. Roadmap / open items

See `BENCHMARKING_STRATEGY.md` → *Open items*. Outstanding:
- **Run** the notebooks on the A100 + Colab to collect numbers (all are code-complete, un-run).
- **Confirm the A100 SKU** (SXM vs PCIe, 40 vs 80 GB).
- **Credible heavy runs** in the forks: MLPerf inference/training, optimum, TensorRT-LLM.
- Polish: measured DCGM power capture in the serving notebook; `$/image` cost extension.
