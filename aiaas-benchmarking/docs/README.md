# Notebook Documentation

Deep per-notebook reference for the AIaaS benchmarking suite. Every notebook here is an
**industry/community-comparable** benchmark (or the kept systems tooling). For the high-level
manual see `../DOCS.md`; for the *why* see `../BENCHMARKING_STRATEGY.md`.

| Notebook | What it is |
|----------|-----------|
| [vllm_serving_benchmark.ipynb](vllm_serving_benchmark.md) | Comparable · LLM serving · TTFT/TPOT/throughput |
| [tensorrt_llm_benchmark.ipynb](tensorrt_llm_benchmark.md) | Peak HW · LLM serving · TTFT/TPOT/throughput |
| [optimum_crossframework_benchmark.ipynb](optimum_crossframework_benchmark.md) | Cross-framework · LLM · decode/latency/VRAM per backend |
| [mlperf_inference_benchmark.ipynb](mlperf_inference_benchmark.md) | Standard · vision/SDXL/whisper · LoadGen QPS/latency/accuracy |
| [mteb_benchmark.ipynb](mteb_benchmark.md) | Standard · embeddings+reranking · MTEB quality (leaderboard) |
| [mlperf_training_benchmark.ipynb](mlperf_training_benchmark.md) | Standard · training · MLPerf Training (throughput/to-target) |
| [model_swap_benchmark.ipynb](model_swap_benchmark.md) | Systems tooling · load/unload/cold-start/co-residency |

**Dropped** (no industry-comparable harness): LoRA-training, embeddings-throughput,
image-gen, ASR, and VLM proxies — image-gen/ASR are covered comparably by MLPerf
(`sdxl`/`whisper`) and embeddings by MTEB. The NAIRA PoC v2 stays as the labeled cross-platform proxy.

> **LoRA transition (PR #23, merged).** The LoRA/QLoRA proxy was *replaced* by the
> MLPerf Training runner above (the comparable harness), so it has no page here.
> `lora_qlora_train_benchmark.ipynb` and its `lora-train-bench/1.0` reader were
> removed from `main` together; this index describes that post-#23 state.

Aggregation/economics tooling (not benchmarks): `../compare_results.py`, `../cost_model.py`,
`../report.ipynb` — see `../DOCS.md` §6.
