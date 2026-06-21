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
- **`compare_results.py`** — side-by-side table across platforms; reads the vLLM
  serving JSONs, the LoRA/QLoRA training JSONs, and the PoC proxy notebook JSONs.

> Your existing **NAIRA PoC v2** notebook is the *proxy* tier — a good
> cross-platform hardware feel, but not comparable to industry numbers (see the
> strategy doc for the five structural reasons). Keep it for breadth; use the
> vLLM notebook for any number that leaves the building.

## Quick start

1. Open `vllm_serving_benchmark.ipynb` on a **GPU** runtime (A100 / Colab T4·L4·A100).
2. Run top to bottom. It auto-selects the model by VRAM, downloads ShareGPT,
   launches the server, runs the sweep, and writes
   `vllm_bench_results/vllm_serving_<platform>_<gpu>.json`.
3. Repeat on each platform, collect the JSONs, then:
   ```bash
   python compare_results.py 'vllm_bench_results/*.json'
   ```

> vLLM brings its own CUDA-matched torch; on Colab the install may require a
> runtime restart. On locked-down environments (SageMaker Studio Lab, Kaggle)
> vLLM may not install — use the PoC proxy notebook there instead.

## Next rungs (tracked in the strategy doc)

- MLPerf `llama3.1-8b` / `resnet50` / `whisper` / `sdxl` runs → in the
  `inference` repo (accuracy-gated, leaderboard-comparable).
- TensorRT-LLM (peak A100 ceiling) and optimum-benchmark (cross-framework).
- Cost model ($/M-tokens from power + amortized HW).
- ~~LoRA/QLoRA training benchmark~~ — added (`lora_qlora_train_benchmark.ipynb`).
