# Session handoff — AIaaS benchmarking

> Historical kickoff plan for this package. For the current reference manual see
> **`DOCS.md`**; for the strategy/rationale see **`BENCHMARKING_STRATEGY.md`**.

## Goal
Benchmark an on-prem **A100 (40 GB)** server to plan an **AIaaS platform**
(inference **and** training, multi-tenant web service). Run the *same* benchmark
across **on-prem + cloud notebook services** (Colab, SageMaker Studio Lab,
Kaggle, etc.) to compare hardware and price/performance. Scale is a later
concern. Want **both** internal capacity-planning numbers **and**
externally-credible / industry-comparable numbers.

> Hardware believed to be a single A100 **40 GB**, but confirm the exact SKU
> (**SXM vs PCIe, 40 vs 80 GB**) — it changes the fit table and expected numbers.

## Workloads to serve
LLM chat/completions, Embeddings/RAG, Vision/multimodal (VLM), Image/video
generation. (ASR + CV also appear in the existing PoC notebook.)

## Key insight
A homemade `transformers`-timing notebook (the existing **NAIRA PoC v2**) is a
fine **cross-platform proxy** but is **structurally NOT comparable** to MLPerf /
optimum-benchmark / TensorRT numbers, because: (1) eager runtime ≠ optimized
serving (vLLM/TRT) — measures the framework not the hardware, often 5–20× off;
(2) synthetic/single-prompt workloads aren't reproducible; (3) no accuracy gate;
(4) no standardized load generator/scenarios; (5) self-reported + wrong model
class. → For comparable numbers, **run the same harness the published numbers
use.**

## Benchmark tiers (the plan)
- **Proxy** = existing NAIRA PoC v2 notebook → keep for breadth, runs everywhere.
- **Comparable** = **vLLM `benchmark_serving`** + ShareGPT → TTFT/TPOT/throughput.
- **Standard** = **MLPerf Inference** subset (accuracy-gated, leaderboard-grade).
- **Cross-framework** = **optimum-benchmark** (PyTorch vs ONNX vs TRT).
- **Peak HW** = **TensorRT-LLM**.

## Two-repo split (agreed)
- **`kurtvalcorza/notebooks`** → portable/cross-platform notebooks (this package).
- **`kurtvalcorza/inference`** (MLPerf fork) → MLPerf reference runs.

## Forks (under `kurtvalcorza`)
| Fork | Upstream | Use |
|------|----------|-----|
| `inference` | mlcommons/inference | MLPerf Inference — Standard tier runs |
| `training` | mlcommons/training | MLPerf Training — reference only (overkill for 1×40 GB) |
| `optimum` | huggingface/optimum | export/optimize models (ONNX/TRT/OpenVINO) |
| `optimum-benchmark` | huggingface/optimum-benchmark | cross-framework benchmark harness |
| `TensorRT` | NVIDIA/TensorRT | OSS TensorRT components + `trtexec` |
| `TensorRT-LLM` | NVIDIA/TensorRT-LLM | peak A100 LLM ceiling |

> Working on a fork needs a Claude session **scoped to that repo**.

## Other tools/repos discussed
- **vLLM** — serving + `bench serve` (the Comparable tier)
- **NVIDIA GenAI-Perf** — perf_analyzer / genai-perf; alt LLM serving load test
- **EleutherAI lm-evaluation-harness** — quality/accuracy (pairs with MLPerf gates)
- **ONNX Runtime**, **PyTorch `torch.utils.benchmark`**
- **MLCommons family** — MLPerf Tiny / Mobile/Client, **MLCube + CM / Collective
  Mind** (easiest reproducible way to drive MLPerf Inference)

## Open items / next steps
1. ✅ Commit this package into `notebooks`.
2. Run `vllm_serving_benchmark.ipynb` on the **A100** (first comparable numbers),
   then on Colab to compare.
3. Confirm A100 SKU (SXM/PCIe, 40/80 GB).
4. ✅ **MLPerf runner** notebook (`mlperf_inference_benchmark.ipynb`): vision via the
   `inference` fork's LoadGen app, plus an MLCFlow path for any model incl.
   **sdxl** / **whisper** / llama. Credible runs need real datasets + the fork session.
5. ✅ **optimum-benchmark** cross-framework notebook (runs need the fork session).
6. ✅ **TensorRT-LLM** peak-ceiling notebook (runs need the fork session).
7. ✅ **Cost model** ($/M-tokens). Refine with measured DCGM power; lock clocks
   (`nvidia-smi -lgc`).
8. ✅ **Retrieval quality**: MTEB notebook (embeddings + reranking, leaderboard).
9. ✅ **Training track**: upgraded from the LoRA/QLoRA proxy (dropped — not
   industry-comparable) to a **MLPerf Training** runner (`mlperf_training_benchmark.ipynb`).
10. ✅ Serving all workload types resident at once won't fit 40 GB → model-swap /
    cold-start cost measured (`model_swap_benchmark.ipynb`).

> **Methodology note:** the suite keeps only **industry/community-comparable**
> benchmarks. Proxy-tier notebooks with no standardized harness (embeddings
> throughput, image-gen, ASR, VLM, LoRA fine-tune) were dropped; image-gen/ASR are
> covered comparably by MLPerf (sdxl/whisper) and embeddings by MTEB.

> Status ✅ reflects work completed in `notebooks/aiaas-benchmarking/`
> (some via open PRs). See `DOCS.md` → *Build status* for the PR mapping.
