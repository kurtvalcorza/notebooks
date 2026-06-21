# Notebook Documentation

Deep per-notebook reference for the AIaaS benchmarking suite. For the high-level
manual (schemas, cost methodology, fork workflow, glossary) see `../DOCS.md`; for
the *why* see `../BENCHMARKING_STRATEGY.md`.

| Notebook | Tier | Workload | Measures |
|----------|------|----------|----------|
| [vllm_serving_benchmark.ipynb](vllm_serving_benchmark.md) | Comparable | LLM serving | TTFT / TPOT / throughput |
| [mlperf_inference_benchmark.ipynb](mlperf_inference_benchmark.md) | Standard | Vision (class./detection) | LoadGen QPS / latency / accuracy (VALID) |
| [tensorrt_llm_benchmark.ipynb](tensorrt_llm_benchmark.md) | Peak HW | LLM serving | TTFT / TPOT / throughput (peak) |
| [optimum_crossframework_benchmark.ipynb](optimum_crossframework_benchmark.md) | Cross-framework | LLM | decode throughput / latency / VRAM per backend |
| [lora_qlora_train_benchmark.ipynb](lora_qlora_train_benchmark.md) | Training | LLM fine-tune | train tokens/s / VRAM / wall-time |
| [embeddings_benchmark.ipynb](embeddings_benchmark.md) | Proxy | Embeddings/RAG | sentences/s / tokens/s / VRAM |
| [image_gen_benchmark.ipynb](image_gen_benchmark.md) | Proxy | Image generation | images/s / s-per-image / VRAM |
| [asr_benchmark.ipynb](asr_benchmark.md) | Proxy | Speech-to-text | RTF / ×real-time / GPU mem |
| [vlm_benchmark.ipynb](vlm_benchmark.md) | Proxy | Multimodal (image+text) | output tokens/s / latency / VRAM |
| [model_swap_benchmark.ipynb](model_swap_benchmark.md) | Systems | Multi-tenant capacity | load/unload / cold-start / co-residency |

Aggregation & reporting (not notebooks): `../compare_results.py`, `../cost_model.py`,
and `../report.ipynb` — documented in `../DOCS.md` §6.
