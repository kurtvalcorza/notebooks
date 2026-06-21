# ASR Throughput Benchmark (faster-whisper)

**Notebook:** `asr_benchmark.ipynb` · **Tier:** Proxy · **Workload:** Speech-to-text · **GPU:** T4+ (any CUDA GPU)

Transcribes synthetic audio with faster-whisper (CTranslate2) and reports real-time factor and throughput.

## What it measures
- **RTF** = processing-time ÷ audio-seconds (lower is better; <1 = faster than real time)
- **×real-time speedup** ≈ concurrent live streams one GPU can keep up with
- **audio-seconds per wall-second**; **GPU memory used** (via nvidia-smi)

## Configuration

| Knob | Default | Notes |
|------|---------|-------|
| `MODEL` | large-v3 (A100) / small (T4) | faster-whisper model size |
| `COMPUTE_TYPE` | float16 | int8_float16 is lighter |
| `BEAM_SIZE` | 1 | greedy = throughput-oriented |
| `CLIP_SECONDS / NUM_CLIPS` | 30 / 10 | whisper processes 30s windows |

## How to run
1. Installs faster-whisper; generates synthetic 16 kHz audio; warms up; transcribes the clips.

## Output

Writes `asr_bench_results/<name>_<platform>_<gpu>.json` with schema **`asr-bench/1.0`** (see `../DOCS.md` §5 for the field reference) and prints a summary table.

## How to read the results

×real-time speedup is the headline capacity number (e.g. 20× ≈ 20 concurrent real-time streams).

## Caveats
- Synthetic audio → **timing only, not accuracy** (no WER).
- Memory is read via nvidia-smi (CTranslate2 isn't on the torch allocator).
- Credible numbers = MLPerf whisper on LibriSpeech.

**References:** faster-whisper (CTranslate2).
