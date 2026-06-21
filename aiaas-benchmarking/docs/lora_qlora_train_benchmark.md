# LoRA / QLoRA Fine-Tune Benchmark

**Notebook:** `lora_qlora_train_benchmark.ipynb` · **Tier:** Training · **Workload:** LLM fine-tune · **GPU:** T4+ (any CUDA GPU)

Runs a fixed-step LoRA/QLoRA supervised fine-tune (TRL `SFTTrainer`) of Qwen2.5 on an Alpaca slice, with packing. Fixed `MAX_STEPS` + packed length make wall-time comparable across platforms.

## What it measures
- **train tokens/s** and **samples/s**
- **peak VRAM** (fit / headroom)
- **wall-time** for the fixed step budget
- trainable-parameter %

## Configuration

| Knob | Default | Notes |
|------|---------|-------|
| `MODEL` | Qwen2.5-1.5B / 7B | VRAM tier |
| `METHOD` | qlora (<24 GB) / lora | QLoRA 4-bit is the portable T4 anchor; set `qlora` everywhere for strict comparison |
| `COMPUTE_DTYPE` | bfloat16 / float16 | fp16 on pre-Ampere |
| `DATASET` | yahma/alpaca-cleaned | ungated instruction set |
| `MAX_SEQ_LEN` | 1024 | packed; set via `max_length`/`max_seq_length` per trl version |
| `NUM_TRAIN_SAMPLES` | 2000 | mapped before packing |
| `MICRO_BATCH / GRAD_ACCUM` | 1–2 / 16 | effective batch = product |
| `MAX_STEPS` | 60 | fixed budget → comparable wall-time |
| `LORA_R / ALPHA / DROPOUT` | 16 / 32 / 0.05 | adapter config |

## How to run
1. Installs transformers/peft/trl/bitsandbytes. Loads model (4-bit for QLoRA), attaches LoRA, trains, records metrics.

## Output

Writes `lora_train_results/<name>_<platform>_<gpu>.json` with schema **`lora-train-bench/1.0`** (see `../DOCS.md` §5 for the field reference) and prints a summary table.

## How to read the results

train tokens/s is throughput; peak VRAM tells you if the method/model fits and the headroom. On a fixed step budget the wall-time compares directly across platforms.

## Caveats
- Targets recent `trl` (SFTConfig; `max_length` length field). bitsandbytes needs a CUDA GPU.

**References:** TRL `SFTTrainer`/`SFTConfig`; PEFT LoRA; bitsandbytes 4-bit (QLoRA).
