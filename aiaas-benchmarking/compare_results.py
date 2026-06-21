#!/usr/bin/env python3
"""Compare benchmark result JSONs side-by-side across platforms.

Reads result files produced by any of:
  - the vLLM serving notebook (schema "vllm-serving-bench/1.0"),
  - the LoRA/QLoRA training notebook (schema "lora-train-bench/1.0"), or
  - the PoC proxy notebook (has top-level "tests"/"env").

Usage:
    python compare_results.py 'vllm_bench_results/*.json' 'lora_train_results/*.json'
"""
import sys
import glob
import json


def label(run):
    env = run.get("env", {})
    gpu = env.get("gpu_name") or env.get("gpu") or "?"
    plat = env.get("platform") or env.get("environment") or "?"
    return f"{gpu} ({plat})"


def p99(point, metric):
    """p99 for a metric from a vLLM bench-serve point.

    Current vLLM writes flat keys (`p99_ttft_ms`); older / customized runs may
    instead carry `percentiles_ttft_ms` as a list of [percentile, value] pairs.
    Support both so the SLA columns don't render blank.
    """
    flat = point.get(f"p99_{metric}_ms")
    if flat is not None:
        return flat
    for p, v in point.get(f"percentiles_{metric}_ms", []) or []:
        if int(p) == 99:
            return v
    return None


def flatten_vllm(run):
    """Pick the max-throughput ('inf') sweep point if present, else the last."""
    sweep = run.get("sweep", {})
    point = sweep.get("inf") or (list(sweep.values())[-1] if sweep else {})
    if not isinstance(point, dict):
        point = {}
    return {
        "model": run.get("model"),
        "out tok/s (max)": point.get("output_throughput"),
        "req/s (max)": point.get("request_throughput"),
        "TTFT p99 ms": p99(point, "ttft"),
        "TPOT p99 ms": p99(point, "tpot"),
    }


def flatten_train(run):
    m = run.get("metrics", {})
    return {
        "model": run.get("model"),
        "method": run.get("method"),
        "train tok/s": m.get("train_tokens_per_second"),
        "train samp/s": m.get("train_samples_per_second"),
        "peak VRAM GB": m.get("peak_vram_gb"),
        "final loss": m.get("final_train_loss"),
    }


def flatten_trtllm(run):
    """Best output throughput across concurrency points for a TensorRT-LLM run."""
    best, bt = None, None
    for c, m in run.get("summary", {}).items():
        if isinstance(m, dict):
            t = m.get("out tok/s")
            if isinstance(t, (int, float)) and (bt is None or t > bt):
                bt, best = t, c
    return {
        "model": run.get("model"),
        "trtllm out tok/s (best)": bt,
        "trtllm best concurrency": best,
    }


def flatten_poc(run):
    t = run.get("tests", {})
    out = {}
    llm = t.get("llm", {})
    if "ttft" in llm:
        out["LLM TTFT p50 s"] = llm["ttft"].get("p50_s")
        out["LLM tok/s p50"] = llm.get("tokens_per_s_p50")
    if "rtf_p50" in t.get("asr", {}):
        out["ASR RTF p50"] = t["asr"]["rtf_p50"]
    if "ms_per_image_p50" in t.get("cv", {}):
        out["CV ms/img p50"] = t["cv"]["ms_per_image_p50"]
    return out


def main(argv):
    paths = []
    for a in argv:
        paths.extend(sorted(glob.glob(a)))
    if not paths:
        print(__doc__)
        return

    cols = {}
    for p in paths:
        try:
            run = json.load(open(p))
        except Exception as e:
            print(f"skip {p}: {e}")
            continue
        if not isinstance(run, dict):
            # e.g. the ShareGPT dataset file, which decodes to a list and the
            # README glob also matches — skip before any dict access.
            print(f"skip {p}: not a benchmark result (top-level {type(run).__name__})")
            continue
        if run.get("schema", "").startswith("vllm-serving-bench"):
            cols.setdefault(label(run), {}).update(flatten_vllm(run))
        elif run.get("schema", "").startswith("lora-train-bench"):
            cols.setdefault(label(run), {}).update(flatten_train(run))
        elif "tests" in run:
            cols.setdefault(label(run), {}).update(flatten_poc(run))
        else:
            # e.g. the notebook's raw per-rate bench_rate_*.json files, which
            # the README glob also matches — skip rather than emit a junk column.
            print(f"skip {p}: unrecognized result schema")
            continue

    if not cols:
        print("no usable result files")
        return

    metrics = []
    for c in cols.values():
        for k in c:
            if k not in metrics:
                metrics.append(k)

    names = list(cols)
    w = max([len(m) for m in metrics] + [6])
    print(f"{'metric':{w}} " + " ".join(f"{n[:26]:>26}" for n in names))
    for m in metrics:
        print(f"{m:{w}} " + " ".join(f"{str(cols[n].get(m, '-')):>26}" for n in names))


if __name__ == "__main__":
    main(sys.argv[1:])
