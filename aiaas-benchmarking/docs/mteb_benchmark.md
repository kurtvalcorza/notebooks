# Retrieval Quality Benchmark (MTEB)

**Notebook:** `mteb_benchmark.ipynb` · **Tier:** Standard (retrieval) · **Workload:** Embeddings + reranking · **GPU:** any (GPU just speeds it)

Runs the **Massive Text Embedding Benchmark (MTEB)** — the industry leaderboard — for embeddings + reranking **quality**, so scores line up with the public MTEB leaderboard. The comparable counterpart to the (dropped) embeddings throughput proxy.

## What it measures
- **Reranking** (MAP/MRR)
- **STS** (Spearman)
- **Classification** (accuracy)
- average main_score

## Configuration

| Knob | Default | Notes |
|------|---------|-------|
| `MODEL` | BAAI/bge-small-en-v1.5 | ungated; on the leaderboard |
| `TASKS` | AskUbuntuDupQuestions, STSBenchmark, Banking77Classification | quick representative trio; add task names (e.g. `SciFact` for retrieval) to broaden — the full leaderboard suite is many hours |

## How to run
1. Installs `mteb` + sentence-transformers.
2. Resilient task loading; `MTEB(tasks).run(model)`; defensive score extraction.

## Output

Writes `mteb_bench_results/...json` with schema **`mteb-bench/1.0`** (field reference in `../DOCS.md` §5) and prints a summary table.

## How to read the results

`main_score` per task ranks on the MTEB leaderboard — compare your model directly. Quality is GPU-independent; pair with a throughput run to pick a model that's accurate AND fast.

## Caveats
- Quality benchmark (not throughput).
- Full leaderboard suite is many hours.

**References:** embeddings-benchmark/mteb; MTEB leaderboard.
