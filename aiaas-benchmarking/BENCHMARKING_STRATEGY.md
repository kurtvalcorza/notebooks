# AIaaS Benchmarking Strategy

Goal: benchmark an on-prem **A100 (40 GB)** server (and compare it against cloud
notebook services — Colab, SageMaker Studio Lab, Kaggle, etc.) to plan an
inference + training "AI-as-a-Service" platform. We want both **internal
capacity-planning** numbers and **externally-credible / industry-comparable**
numbers.

This document is the source of truth for *how* we benchmark and *why*. It also
tracks the open items.

---

## TL;DR

There are three distinct kinds of benchmark, and they are **not**
interchangeable:

| Tier | Purpose | Tool(s) | Comparable to public numbers? |
|------|---------|---------|-------------------------------|
| **Proxy** | quick "does it run, rough hardware feel" across every platform | the portable PoC notebook (transformers/diffusers eager timing) | ❌ No — different runtime, synthetic data, no accuracy gate |
| **Comparable** | serving numbers you can put next to vendor blogs & community runs | **vLLM `benchmark_serving`** + ShareGPT | ✅ Yes (same harness everyone uses) |
| **Standard** | leaderboard-grade, accuracy-gated, audited | **MLPerf Inference** (this repo) | ✅ Yes (official rules) |
| **Cross-framework** | same model across PyTorch / ONNX Runtime / TensorRT | **optimum-benchmark** | ✅ Yes (one standardized harness) |
| **Peak HW** | best-case A100 ceiling | **TensorRT-LLM** | ✅ Yes (vendor-grade) |

Keep the proxy notebook for cross-platform breadth; use the comparable/standard
tiers for any number that leaves the building.

---

## Why a homemade transformers-timing notebook is NOT industry-comparable

A clean, careful timing harness (warmup, percentiles, cold-start separated,
batch sweep) is still **structurally** non-comparable to MLPerf / optimum /
TensorRT numbers, for five independent reasons:

1. **Runtime = the real ceiling.** Raw `transformers` eager `generate()` is the
   slowest inference path. Published numbers use vLLM / TensorRT-LLM / TGI
   (PagedAttention, continuous batching, CUDA graphs, fused kernels). On the
   same GPU, eager transformers is commonly **5–20× slower**. The number
   measures the *framework*, not the *hardware*.
2. **No standardized workload.** A single hard-coded prompt, synthetic sine-wave
   audio, or random-noise images are not reproducible by anyone else. Industry
   runs use defined datasets with realistic input/output length *distributions*
   (ShareGPT / OpenOrca, LibriSpeech, COCO, ImageNet).
3. **No accuracy gate.** MLPerf forbids trading quality for speed — you must hit
   an accuracy target (e.g. 99% of reference FP32). A homemade speed test would
   reward a fast-but-degraded config. Industry numbers are *accuracy-constrained
   throughput*.
4. **No standardized load generator / scenarios.** MLPerf's LoadGen defines
   exact traffic: Offline (max throughput), Server (Poisson arrivals under a p99
   latency bound), SingleStream, MultiStream. A batch sweep is a rough proxy,
   not an SLA-bound Server measurement.
5. **Self-reported & different model class.** Official results pass compliance
   tests + review. And datacenter LLM tasks are 8B / 70B / 405B — a 3B model
   chosen for VRAM reasons isn't in the same category as the published rows.

**Conclusion:** the PoC notebook is a legitimate *cross-platform hardware
proxy*. To compare against a vendor blog or the MLPerf leaderboard, run the same
harness they ran.

---

## Hardware reality (single A100 40 GB)

40 GB is the binding constraint. Rough single-card fit:

| Workload | Fits | Tight (quant/offload) | Won't fit |
|----------|------|-----------------------|-----------|
| LLM chat | ≤8B fp16 (~16 GB) | ~14B fp16; 70B 4-bit (slow) | 70B+ fp16, Mixtral fp16 |
| Embeddings/RAG | all (bge/e5/gte <2 GB) | — | — |
| Vision/VLM | 7B VLMs | larger VLMs | 70B+ VLMs |
| Image/video gen | SDXL | video (Wan2.2, CogVideoX) | large video fp16 |
| Training | LoRA/QLoRA on 7–8B | full FT of small models | full FT ≥13B |

Serving *all four workload types resident at once* does **not** fit in 40 GB —
so part of the benchmark is measuring **model-swap / cold-start cost** and
deciding what stays resident vs. loaded on demand. (Flag for the scale phase.)

---

## Recommended run matrix

For each platform (on-prem A100, Colab, SageMaker Studio Lab, …):

| Workload | Proxy (everywhere) | Comparable / Standard (A100, capable GPUs) |
|----------|--------------------|--------------------------------------------|
| LLM chat | PoC notebook (Qwen 3B) | **vLLM benchmark_serving** (ShareGPT) → MLPerf **llama3.1-8b** |
| Embeddings/RAG | PoC notebook | optimum-benchmark (bge/e5) |
| ASR | PoC notebook (faster-whisper) | MLPerf **whisper** (LibriSpeech + WER gate) |
| CV detection | PoC notebook (YOLO) | MLPerf **resnet50 / retinanet** (ImageNet/COCO + mAP gate) |
| Image gen | PoC notebook (SDXL) | MLPerf **stable-diffusion-xl** |

Always capture alongside every run: GPU name, VRAM, CUDA/driver, torch/engine
versions, **GPU power (DCGM / nvidia-smi)**, and lock clocks where possible
(`nvidia-smi -lgc`) for repeatability. Normalize to **tokens/sec**, **ms/token**,
and **tokens/sec-per-dollar-hour** (the last decides hosting).

---

## Phases

- **Phase 0 — Environment baseline.** Pin driver/CUDA/engine versions; record
  peak FP16/FP8 FLOPS & memory BW; confirm A100 SKU (SXM vs PCIe, 40 vs 80 GB);
  stand up DCGM/nvidia-smi power capture.
- **Phase 1 — Comparable serving (this PR's notebook).** vLLM benchmark_serving
  on A100 + Colab; latency-vs-throughput curve; max users at SLA.
- **Phase 2 — Standard/credible.** MLPerf Inference subset that fits 40 GB:
  `resnet50`, `retinanet`, `bert`, `3d-unet`, `stable-diffusion-xl`,
  `llama3.1-8b`, `whisper` (Offline + Server scenarios).
- **Phase 3 — Capacity + cost model.** Combine curves into users-per-model at
  SLA; derive $/M-tokens and $/image; document model-swap latency.
- **Training track.** **MLPerf Training** reference runs (the comparable harness)
  via `mlperf_training_benchmark.ipynb`. On one 40 GB card this is a
  smoke/throughput signal only — full to-target runs are cluster-scale. (The
  earlier LoRA/QLoRA proxy was dropped as non-comparable.)

---

## Tooling repos (forked under `kurtvalcorza`)

- `inference` (this repo) — MLPerf Inference, Phase 2 standard runs.
- `optimum` / `optimum-benchmark` — cross-framework comparable runs.
- `TensorRT` (+ TensorRT-LLM) — peak A100 ceiling after baseline.
- `training` — reference only; use lightweight fine-tune benchmarks instead.

> Note: these forks are **not** in the current session's repo scope (scoped to
> `kurtvalcorza/inference`). To work on them, start a session scoped to those
> repos.

---

## Open items

- [ ] **Colab-ready portable notebook** — superseded by the existing NAIRA PoC
      v2 notebook (good proxy). Re-evaluate if we want a cleaned/merged version.
- [x] **vLLM serving benchmark notebook** — added
      (`vllm_serving_benchmark.ipynb`). Still to run on A100 + Colab.
- [ ] **Confirm A100 SKU** — SXM vs PCIe, 40 vs 80 GB (changes the fit table and
      expected numbers).
- [~] **MLPerf subset run** on the A100 (Phase 2) — portable runner notebook
      added (`mlperf_inference_benchmark.ipynb`): vision class/detection via the
      `inference` fork's LoadGen app, plus an MLCFlow path for any model incl.
      **sdxl** (image-gen comparable) and **whisper** (ASR comparable). Credible
      runs still need real datasets + the A100 (fork session).
- [~] **Image-gen / ASR comparability** — MLCFlow runner added for the MLPerf
      equivalents (sdxl / whisper); smoke (`execution_mode=test`) by default, so
      credible numbers still need real datasets + the A100. The PoC proxies stay
      until these are actually run.
- [x] **optimum-benchmark** cross-framework notebook added
      (`optimum_crossframework_benchmark.ipynb`). Runs still need a fork session.
- [x] **TensorRT-LLM** peak-ceiling notebook added (`tensorrt_llm_benchmark.ipynb`,
      `trtllm-bench` PyTorch backend). Runs need the `TensorRT-LLM` fork session
      on an A100/Hopper GPU.
- [x] **Cost model** — `$/M-tokens` from power + amortized HW added
      (`cost_model.py`). Refine with measured DCGM power via `--power-watts`.
- [~] **Training track** — upgraded from the LoRA/QLoRA proxy (dropped, not
      industry-comparable) to a **MLPerf Training** runner
      (`mlperf_training_benchmark.ipynb`, references the `training` fork). Credible
      to-target runs are cluster-scale; one 40 GB card = smoke/throughput only.
- [x] **Model-swap / cold-start cost** — `model_swap_benchmark.ipynb` measures
      load/unload/cold-start + resident VRAM and gives a co-residency verdict for
      the 40 GB card. Still to run on the A100.
- [x] **Combined report** — `report.ipynb` aggregates the result schemas and
      reuses `compare_results.py` / `cost_model.py`. Each section populates only
      once its producing notebook (and `cost_model.py`) is present; degrades
      gracefully otherwise.
