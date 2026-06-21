# AIaaS Benchmarking — Full Documentation

Portable, cross-platform benchmarks for sizing an **AI-as-a-Service** platform on a
single **A100 (40 GB)** server and comparing it against cloud notebook services
(Colab, SageMaker Studio Lab, Kaggle, …).

This is the reference manual. For the *why* (tiers, the five reasons a homemade
timing notebook isn't industry-comparable, the 40 GB fit table, phases), read
**`BENCHMARKING_STRATEGY.md`** first. For a one-screen overview, see `README.md`.

---

## 1. What this package is

A set of **self-contained notebooks** plus two **aggregation scripts**. Each
notebook:

- runs on a GPU runtime (Colab / A100 / Kaggle / SageMaker Studio Lab),
- auto-detects the platform and GPU and **tiers itself by VRAM** (a small,
  T4-friendly model vs. a larger A100 model),
- uses **ungated** models (Qwen2.5 family, BAAI/bge, Stability SD) so no Hugging
  Face token is needed,
- writes a **normalized result JSON** into a `*_results/` folder,
- prints a summary table.

The scripts then read those JSONs to produce a cross-platform comparison and a
cost estimate, and `report.ipynb` rolls everything into one charted report.

### Two-repo split

| Repo | Holds |
|------|-------|
| **`kurtvalcorza/notebooks`** (here, `aiaas-benchmarking/`) | portable proxy/comparable/peak/training/systems notebooks |
| **`kurtvalcorza/inference`** (MLPerf fork) | MLPerf reference runs — the *Standard* tier, accuracy-gated |

---

## 2. Benchmark tiers

| Tier | Question it answers | In this package |
|------|---------------------|-----------------|
| **Proxy** | "does it run, rough hardware feel" everywhere | the existing PoC notebook + the proxy-tier benchmarks below |
| **Comparable** | serving numbers you can line up with public results | `vllm_serving_benchmark.ipynb` |
| **Cross-framework** | how much an optimized runtime buys on this GPU | `optimum_crossframework_benchmark.ipynb` |
| **Peak HW** | best-case optimized-engine ceiling | `tensorrt_llm_benchmark.ipynb` |
| **Standard** | leaderboard-grade, accuracy-gated | MLPerf — in the `inference` fork (not here) |
| **Training** | fine-tune throughput / fit | `lora_qlora_train_benchmark.ipynb` |
| **Systems** | co-residency, swap cost, cost-per-token | `model_swap_benchmark.ipynb`, `cost_model.py` |

> Proxy-tier notebooks are honest *systems timings*, not accuracy-gated. For any
> number that "leaves the building," use the Comparable / Peak / Standard tiers.

---

## 3. Notebook & script catalog

| File | Workload | Tier | Measures | Result schema | GPU |
|------|----------|------|----------|---------------|-----|
| `vllm_serving_benchmark.ipynb` | LLM serving | Comparable | TTFT, TPOT/ITL, output & request throughput (rate sweep) | `vllm-serving-bench/1.0` | T4+ |
| `tensorrt_llm_benchmark.ipynb` | LLM serving | Peak HW | TTFT, TPOT, output throughput (concurrency sweep) | `trtllm-bench/1.0` | A100/Hopper |
| `optimum_crossframework_benchmark.ipynb` | LLM | Cross-framework | decode throughput, prefill/per-token latency, VRAM per backend | `optimum-bench/1.0` | T4+ |
| `lora_qlora_train_benchmark.ipynb` | LLM fine-tune | Training | train tokens/s, samples/s, peak VRAM, wall-time | `lora-train-bench/1.0` | T4+ |
| `embeddings_benchmark.ipynb` | Embeddings/RAG | Proxy | sentences/s, tokens/s, latency, VRAM (batch sweep) | `embeddings-bench/1.0` | any |
| `image_gen_benchmark.ipynb` | Image gen | Proxy | images/s, s/image, VRAM (batch sweep) | `image-gen-bench/1.0` | T4+ |
| `asr_benchmark.ipynb` | Speech-to-text | Proxy | RTF, ×real-time speedup, audio-s/wall-s, GPU mem | `asr-bench/1.0` | T4+ |
| `vlm_benchmark.ipynb` | Vision-language | Proxy | output tokens/s, latency, peak VRAM (batch sweep) | `vlm-bench/1.0` | T4+ |
| `model_swap_benchmark.ipynb` | Multi-tenant systems | Systems | load/unload time, cold-start tax, resident/peak VRAM, co-residency verdict | `model-swap-bench/1.0` | T4+ |
| `cost_model.py` | — | Systems | `$/M-tokens` (energy + amortized HW) from serving JSONs | `vllm-cost-model/1.0` | CPU |
| `compare_results.py` | — | — | cross-platform comparison table | — | CPU |
| `report.ipynb` | — | — | combined charted report over all schemas | — | CPU |

### Build status

The original package (`vllm_serving_benchmark.ipynb`, `compare_results.py`,
`lora_qlora_train_benchmark.ipynb`, plus `README` / `BENCHMARKING_STRATEGY`) is
**merged to `main`**. The remaining notebooks land via open PRs:

| PR | Adds |
|----|------|
| #12 | `cost_model.py` (+ serving notebook records `tensor_parallel_size`) |
| #13 | `optimum_crossframework_benchmark.ipynb` |
| #14 | `tensorrt_llm_benchmark.ipynb` |
| #15 | `report.ipynb`, `embeddings_benchmark.ipynb`, `image_gen_benchmark.ipynb` |
| #16 | `model_swap_benchmark.ipynb` |
| #17 | `asr_benchmark.ipynb`, `vlm_benchmark.ipynb` |

---

## 4. Quick start

1. Open a notebook on a **GPU** runtime.
2. **Run top to bottom.** It installs deps, captures the environment, auto-selects
   the model by VRAM, runs, and writes `*_results/<name>_<platform>_<gpu>.json`.
3. Repeat on each platform, collect the JSONs in one place, then aggregate:
   ```bash
   python compare_results.py 'vllm_bench_results/*.json' 'lora_train_results/*.json'
   python cost_model.py     'vllm_bench_results/*.json' --price 0.12 --util 0.5
   ```
   …or open **`report.ipynb`** from the `aiaas-benchmarking/` directory for a
   combined, charted report (`aiaas_report.md` / `aiaas_report.json`).

### Platform notes
- **Colab:** vLLM/TensorRT-LLM may upgrade `torch` and require *Runtime → Restart*,
  then re-run from the install cell.
- **T4 (Turing):** runs the small tier of most notebooks. **Not** supported by
  recent **TensorRT-LLM** (Ampere+); use the vLLM notebook there.
- **SageMaker Studio Lab / Kaggle:** locked-down; vLLM/TRT-LLM may not install —
  use the proxy notebooks (embeddings/image-gen/ASR/VLM/PoC).
- **Pre-Ampere GPUs** lack bf16; the notebooks auto-pick fp16 there.

---

## 5. Result JSON schemas

All result files share an `env` block (platform, gpu_name, gpu_count,
vram_total_gb, compute_capability, cuda, driver, torch, python) and a `schema`
string. Workload-specific fields:

- **`vllm-serving-bench/1.0`** — `model`, `tier`, `dataset`, `num_prompts`,
  `tensor_parallel_size`, `request_rates`, `sweep`: {rate → vLLM `bench serve`
  metrics incl. `output_throughput`, `request_throughput`, `p99_ttft_ms`,
  `p99_tpot_ms` (or `percentiles_*_ms` arrays)}.
- **`trtllm-bench/1.0`** — `engine: tensorrt-llm`, `backend`, `tensor_parallel_size`,
  `input_len`, `output_len`, `summary`: {concurrency → out tok/s, req/s, TTFT,
  TPOT}, `sweep`: raw reports.
- **`optimum-bench/1.0`** — `task`, `dtype`, `input_shapes`, `summary`: {backend →
  decode throughput, prefill/per-token latency, VRAM}, `reports`: raw.
- **`lora-train-bench/1.0`** — `method`, `compute_dtype`, `max_steps`,
  `effective_batch`, `metrics`: {train_tokens_per_second, train_samples_per_second,
  peak_vram_gb, final_train_loss, trainable_pct, …}.
- **`embeddings-bench/1.0`** — `avg_tokens_per_sentence`, `results`: [{batch_size,
  sentences_per_s, tokens_per_s, ms_per_batch, peak_vram_gb}].
- **`image-gen-bench/1.0`** — `steps`, `height`, `width`, `results`: [{batch_size,
  images_per_s, s_per_image, peak_vram_gb}].
- **`asr-bench/1.0`** — `compute_type`, `beam_size`, `clip_seconds`, `result`:
  {rtf, speedup_x_realtime, audio_s_per_wall_s, gpu_mem_used_mb}.
- **`vlm-bench/1.0`** — `gen_tokens`, `img_size`, `results`: [{batch_size,
  output_tokens_per_s, latency_s_per_call, peak_vram_gb}].
- **`model-swap-bench/1.0`** — `results`: [{name, load_s, cold_infer_s,
  warm_infer_s, cold_start_tax_s, resident_vram_gb, peak_vram_gb, unload_s,
  swap_in_out_s}], `analysis`: {usable_vram_gb, resident_total_gb,
  all_co_resident, …}.
- **`vllm-cost-model/1.0`** — `assumptions`, `results`: per-run `$/M` energy /
  hardware / total for output and total tokens.

`compare_results.py` reads the serving / training / PoC schemas; `report.ipynb`
reads all of them.

---

## 6. Aggregation & reporting

### `compare_results.py`
```bash
python compare_results.py 'vllm_bench_results/*.json' 'lora_train_results/*.json'
```
Prints one table; columns are platforms (`gpu (platform)`), rows are metrics.
Skips non-result JSONs (e.g. the ShareGPT dataset file) and merges runs that
share a platform label.

### `cost_model.py`
Converts serving throughput to `$/M-tokens`:
```
$/M-tok = power_kW·PUE·hours_per_M·$/kWh            (energy)
        + capex/(lifetime_h·util)·hours_per_M        (hardware)
  hours_per_M = 1e6 / throughput_tok_s / 3600
```
Per-GPU power/capex defaults (T4…H100) with overrides:
`--price`, `--pue`, `--util`, `--amortization-years`, `--server-overhead`,
`--power-watts` (measured DCGM power), `--gpu-capex`, `--gpus`
(GPUs vLLM actually used / tensor-parallel size — defaults to the recorded
`tensor_parallel_size`, else 1; **not** every visible GPU). `--json` writes a
report file.

### `report.ipynb`
Run from the `aiaas-benchmarking/` directory. Imports `compare_results` and
`cost_model` (the latter guarded — it lights up once PR #12 is on `main`),
summarizes every schema, draws bar charts, and writes `aiaas_report.md` /
`aiaas_report.json`. `matplotlib`/`tabulate` degrade gracefully if missing.

---

## 7. Metrics glossary

- **TTFT** — time to first token (prompt → first output token). Interactivity.
- **TPOT / ITL** — time per output token / inter-token latency (steady-state
  decode). `1/TPOT` is per-stream tokens/s.
- **Output throughput** — output tokens/s across all concurrent requests.
- **Request throughput** — completed requests/s.
- **RTF** (ASR) — processing-time ÷ audio-seconds; `< 1` = faster than real time.
  **×real-time speedup** = audio-seconds ÷ processing-time ≈ concurrent live
  streams one GPU can keep up with.
- **Resident VRAM** — steady-state memory a loaded model holds.
- **Peak VRAM** — max during inference (resident + activations/KV cache).
- **Cold-start tax** — cold first-inference minus warm inference (kernel/compile
  warmup).
- **$/M-tokens** — dollars per million tokens (energy + amortized hardware).

---

## 8. Fork workflow (Standard / Peak runs)

Some work lives in other forks (all under `kurtvalcorza`): `inference`,
`training`, `optimum`, `optimum-benchmark`, `TensorRT`, `TensorRT-LLM`. A Claude
session is **scoped to specific repos**; to develop or run in a fork, start a
session scoped to it, then:

- **MLPerf (Standard tier)** → build/run the runners in `inference`
  (`llama3.1-8b`, `resnet50`, `retinanet`, `whisper`, `stable-diffusion-xl`;
  Offline + Server). This is also where the **accuracy-gated, non-LLM** numbers
  come from.
- **Cross-framework runs** → drop `optimum_crossframework_benchmark.ipynb` into
  the `optimum-benchmark` fork session.
- **Peak LLM runs** → run `tensorrt_llm_benchmark.ipynb` in the `TensorRT-LLM`
  fork session (its `prepare_dataset.py` is what the notebook references).

---

## 9. Capacity & the 40 GB constraint

All four serving workloads (LLM + embeddings + VLM + image-gen) **do not** fit
resident in 40 GB at once. `model_swap_benchmark.ipynb` measures the co-residency
verdict and the swap cost; combine it with the throughput notebooks (capacity at
SLA) and `cost_model.py` ($/M-tokens) to decide what stays warm vs. loads on
demand.

---

## 10. Troubleshooting

- **vLLM install upgraded torch** → restart the runtime, re-run from the install
  cell.
- **vLLM/TRT-LLM won't install** (Kaggle/SageMaker) → expected on locked-down
  envs; use the proxy notebooks.
- **TensorRT-LLM errors on a T4** → unsupported (Turing); use the vLLM notebook.
- **`trtllm-bench` flag error** → flag names drift; check `trtllm-bench
  throughput -h` (e.g. `--tp` vs `--tp_size`) and adjust the config cell.
- **ONNX export fails (optimum)** → swap `MODEL` to `gpt2`; that backend row is
  recorded as an error and the others still report.
- **OOM on a batch point** (image-gen / VLM) → recorded as `OOM`; lower the
  batch size or use the small tier.
- **A model is gated** → all defaults are ungated; if you swap one in, set
  `HF_TOKEN`.

---

## 11. Roadmap / open items

See `BENCHMARKING_STRATEGY.md` → *Open items* for the live tracker. Outstanding:

- **MLPerf runners** in the `inference` fork (Standard tier; also covers
  accuracy-gated non-LLM workloads).
- **Actual GPU runs** on the A100 + Colab to collect numbers (all notebooks are
  code-complete but un-run).
- **Confirm the A100 SKU** (SXM vs PCIe, 40 vs 80 GB) — changes the fit table.
- Polish: measured DCGM power capture in the serving notebook; unify
  `compare_results.py` across every schema; `$/image` cost extension.
