# Embeddings / RAG Throughput Benchmark

**Notebook:** `embeddings_benchmark.ipynb` · **Tier:** Proxy · **Workload:** Embeddings/RAG · **GPU:** any CUDA GPU

Sweeps batch size with `sentence-transformers` over a synthetic fixed-length corpus and reports encode throughput, latency, and VRAM.

## What it measures
- **sentences/s** and **tokens/s**
- **ms per batch** (latency)
- **peak VRAM**

## Configuration

| Knob | Default | Notes |
|------|---------|-------|
| `MODEL` | BAAI/bge-small-en-v1.5 | ungated, ~130 MB, runs anywhere |
| `BATCH_SIZES` | [1, 8, 32, 128] | throughput-vs-batch sweep |
| `SEQ_WORDS` | 64 | ≈ tokens/sentence via repeated words |
| `NUM_SENTENCES` | 2000 | corpus encoded per batch point |
| `DTYPE` | float16 (GPU) |  |

## How to run
1. Installs sentence-transformers; builds a synthetic corpus; sweeps encode batch sizes.

## Output

Writes `embeddings_bench_results/<name>_<platform>_<gpu>.json` with schema **`embeddings-bench/1.0`** (see `../DOCS.md` §5 for the field reference) and prints a summary table.

## How to read the results

Throughput rises with batch size until the GPU saturates — the knee is the efficient serving batch. sentences/s sizes a RAG ingest/query pipeline.

## Caveats
- Proxy tier (clean timing, not accuracy-gated). For credible numbers use optimum-benchmark / MLPerf BERT.

**References:** sentence-transformers; BAAI/bge.
