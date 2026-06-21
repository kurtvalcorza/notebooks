# AIaaS Benchmarking — portable / cross-platform notebooks

Portable benchmarks for sizing an **AI-as-a-Service** platform on a single
**A100 (40 GB)** server and comparing it against cloud notebook services
(Colab, SageMaker Studio Lab, Kaggle, …).

## Where things live (two-repo split)

| Repo | Holds |
|------|-------|
| **`kurtvalcorza/notebooks`** (here) | portable / cross-platform benchmark notebooks |
| **`kurtvalcorza/inference`** (MLPerf fork) | MLPerf reference runs — leaderboard-grade, accuracy-gated (belongs with the suite) |

## Contents

- **`BENCHMARKING_STRATEGY.md`** — the master plan. Proxy vs comparable vs
  standard benchmark tiers, why a homemade `transformers`-timing notebook is
  **not** industry-comparable, the A100 40 GB fit table, per-workload run
  matrix, phases, and the open-items tracker. **Read this first.**
- **`DOCS.md`** — full reference manual: notebook catalog, quick start, result
  JSON schemas, aggregation/reporting, cost methodology, metrics glossary, the
  fork workflow, and troubleshooting.
- **`SESSION_HANDOFF.md`** — the original kickoff plan (goal, tiers, forks).
- **`vllm_serving_benchmark.ipynb`** — the *comparable* tier. Runs a real vLLM
  OpenAI server + `bench serve` over ShareGPT with a request-rate sweep, and
  reports standardized **TTFT / TPOT / throughput** percentiles you can line up
  against public vLLM numbers. VRAM-tiered (T4-anchor model vs A100 model),
  ungated Qwen2.5 models, graceful teardown.
- **`mlperf_inference_benchmark.ipynb`** — the *standard* tier. Portable bridge to
  **MLPerf Inference**. Two paths: (a) the fork's LoadGen reference app for vision
  classification/detection (mobilenet/resnet50/retinanet), modeled on the fork's
  `GettingStarted.ipynb`; and (b) an **MLCFlow** path that runs the **whole MLPerf
  Inference suite** as a configurable model list — resnet50, retinanet, bert,
  3d-unet, dlrm, gptj, **sdxl**, **whisper**, llama3.1-8b, … (incl. the comparable
  upgrades for the dropped image-gen and ASR proxies). Runs under a LoadGen
  scenario with the accuracy gate; smoke defaults, real datasets /
  `execution_mode=valid` for credible runs.
- **`lora_qlora_train_benchmark.ipynb`** — the *training* companion. Runs a
  fixed-budget LoRA / QLoRA supervised fine-tune of Qwen2.5 and reports **train
  tokens/s, samples/s, peak VRAM, and wall-time**. Same VRAM-tiered, ungated,
  fixed-step design so a Colab T4 and an A100 are directly comparable (QLoRA is
  the portable anchor that fits a T4).
- **`tensorrt_llm_benchmark.ipynb`** — the *peak-ceiling* tier. Runs
  TensorRT-LLM's own `trtllm-bench` (PyTorch backend) over a concurrency sweep and
  reports **TTFT / TPOT / output throughput**, so the gap vs the vLLM notebook is
  the optimized-engine headroom. Generates its dataset with the
  `kurtvalcorza/TensorRT-LLM` fork's `prepare_dataset.py`. **A100/Hopper only**
  (recent TensorRT-LLM doesn't support Turing/T4).
- **`optimum_crossframework_benchmark.ipynb`** — the *cross-framework* tier. Runs
  one model through **PyTorch vs ONNX Runtime (CUDA)** (plus the **ONNX Runtime
  TensorRT EP when the platform exposes it**) with HuggingFace `optimum-benchmark`,
  reporting decode throughput / latency / VRAM per backend so the only variable is
  the runtime — **precision matched at fp16** (PyTorch fp16, ORT auto-optimization
  O4). (Runs belong in a fork session scoped to `optimum-benchmark`; the notebook is portable.)
- **`model_swap_benchmark.ipynb`** — the *systems* companion. Measures per-workload
  **load / unload time, cold-start tax, and resident/peak VRAM**, then gives a
  **co-residency verdict** (do LLM + embeddings + image-gen fit in the GPU at once?)
  and the **swap cost** when they don't. Targets the 40 GB binding constraint.
- **`compare_results.py`** — side-by-side table across platforms; reads the vLLM
  serving JSONs, the LoRA/QLoRA training JSONs, the cross-framework and
  TensorRT-LLM JSONs, the MLPerf inference JSONs, and the PoC proxy notebook JSONs.
- **`cost_model.py`** — turns the vLLM serving throughput into **$/M-tokens**,
  split into energy (power draw × PUE × electricity price) and amortized hardware
  (capex over a utilized lifetime). Per-GPU power/capex defaults, all overridable.
- **`report.ipynb`** — combined, charted report: imports `compare_results.py`
  (and `cost_model.py` when present) and summarizes the result schemas into one
  place (`aiaas_report.md` / `.json`).

> Your existing **NAIRA PoC v2** notebook is the *proxy* tier — a good
> cross-platform hardware feel, but not comparable to industry numbers (see the
> strategy doc for the five structural reasons). Keep it for breadth; use the
> vLLM notebook for any number that leaves the building.

## Quick start

1. Open `vllm_serving_benchmark.ipynb` on a **GPU** runtime (A100 / Colab T4·L4·A100).
2. Run top to bottom. It auto-selects the model by VRAM, downloads ShareGPT,
   launches the server, runs the sweep, and writes
   `vllm_bench_results/vllm_serving_<platform>_<gpu>.json`.
3. Repeat on each platform, collect the JSONs, then compare and cost them:
   ```bash
   python compare_results.py 'vllm_bench_results/*.json'
   python cost_model.py 'vllm_bench_results/*.json' --price 0.12 --util 0.5
   ```

> vLLM brings its own CUDA-matched torch; on Colab the install may require a
> runtime restart. On locked-down environments (SageMaker Studio Lab, Kaggle)
> vLLM may not install — use the PoC proxy notebook there instead.

## Next rungs (tracked in the strategy doc)

- MLPerf `llama3.1-8b` / `resnet50` / `whisper` / `sdxl` runs → in the
  `inference` repo (accuracy-gated, leaderboard-comparable).
- ~~TensorRT-LLM (peak A100 ceiling)~~ — added (`tensorrt_llm_benchmark.ipynb`).
- ~~optimum-benchmark (cross-framework)~~ — added (`optimum_crossframework_benchmark.ipynb`).
- ~~Cost model ($/M-tokens from power + amortized HW)~~ — added (`cost_model.py`).
- ~~LoRA/QLoRA training benchmark~~ — added (`lora_qlora_train_benchmark.ipynb`).
