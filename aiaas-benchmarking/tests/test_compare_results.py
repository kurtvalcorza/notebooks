"""Unit tests for compare_results.py — the cross-platform aggregator.

One fixture per supported schema asserts the flattener picks the right point and
emits the expected columns, plus an integration check that main() dispatches
every known schema (and skips unknown ones) instead of silently dropping them.
"""
import json

import compare_results as CR


def test_p99_flat_and_percentile_forms():
    assert CR.p99({"p99_ttft_ms": 80}, "ttft") == 80
    assert CR.p99({"percentiles_ttft_ms": [[50, 10], [99, 80]]}, "ttft") == 80
    assert CR.p99({}, "ttft") is None


def test_flatten_vllm_picks_inf_point():
    run = {
        "schema": "vllm-serving-bench/1.0", "model": "m",
        "sweep": {
            "4": {"output_throughput": 100, "request_throughput": 1,
                  "p99_ttft_ms": 50, "p99_tpot_ms": 10},
            "inf": {"output_throughput": 500, "request_throughput": 5,
                    "p99_ttft_ms": 80, "p99_tpot_ms": 15},
        },
    }
    out = CR.flatten_vllm(run)
    assert out["out tok/s (max)"] == 500       # the saturation point, not "4"
    assert out["req/s (max)"] == 5
    assert out["TTFT p99 ms"] == 80
    assert out["TPOT p99 ms"] == 15


def test_flatten_optimum_best_backend():
    run = {"model": "gpt2", "summary": {
        "pytorch": {"decode throughput": 100.0},
        "onnxruntime": {"decode throughput": 250.0},
    }}
    out = CR.flatten_optimum(run)
    assert out["optimum best backend"] == "onnxruntime"
    assert out["optimum best tok/s"] == 250.0


def test_flatten_trtllm_best_concurrency():
    run = {"model": "m", "summary": {
        "8": {"out tok/s": 3000.0},
        "32": {"out tok/s": 5000.0},
    }}
    out = CR.flatten_trtllm(run)
    assert out["trtllm best concurrency"] == "32"
    assert out["trtllm out tok/s (best)"] == 5000.0


def test_flatten_mlperf_inference_keys():
    run = {"model": "resnet50", "scenario": "Offline",
           "loadgen_summary": {"Result is": "VALID", "Samples per second": 4200.0}}
    out = CR.flatten_mlperf(run)
    assert out["mlperf valid"] == "VALID"
    assert out["mlperf QPS"] == 4200.0
    assert out["mlperf scenario"] == "Offline"


def test_flatten_mlperf_training_keys():
    run = {"benchmark": "image_classification", "smoke": True,
           "result": {"status": "ok"},
           "metrics": {"throughput": 1234.0, "eval_accuracy": 0.76, "run_time_s": 42.0}}
    out = CR.flatten_mlperf_training(run)
    assert out["train benchmark"] == "image_classification"
    assert out["throughput"] == 1234.0
    assert out["eval_accuracy"] == 0.76
    assert out["run_time s"] == 42.0


def test_flatten_poc_keys():
    run = {"tests": {"llm": {"ttft": {"p50_s": 0.5}, "tokens_per_s_p50": 40},
                     "asr": {"rtf_p50": 0.1}, "cv": {"ms_per_image_p50": 20}}}
    out = CR.flatten_poc(run)
    assert out["LLM tok/s p50"] == 40
    assert out["ASR RTF p50"] == 0.1
    assert out["CV ms/img p50"] == 20


# Distinct env per fixture so each lands in its own column (a shared label would
# merge them and hide a schema that failed to dispatch).
KNOWN = {
    "vllm": {"schema": "vllm-serving-bench/1.0", "env": {"gpu_name": "A100", "platform": "vllm"},
             "model": "m", "sweep": {"inf": {"output_throughput": 1}}},
    "optimum": {"schema": "optimum-bench/1.0", "env": {"gpu_name": "A100", "platform": "optimum"},
                "model": "m", "summary": {"pytorch": {"decode throughput": 1}}},
    "trtllm": {"schema": "trtllm-bench/1.0", "env": {"gpu_name": "A100", "platform": "trtllm"},
               "model": "m", "summary": {"8": {"out tok/s": 1}}},
    "mlpinf": {"schema": "mlperf-inference/1.0", "env": {"gpu_name": "A100", "platform": "mlpinf"},
               "model": "m", "scenario": "Offline", "loadgen_summary": {"Result is": "VALID"}},
    "mlptrain": {"schema": "mlperf-training/1.0", "env": {"gpu_name": "A100", "platform": "mlptrain"},
                 "benchmark": "image_classification", "result": {"status": "ok"},
                 "metrics": {"throughput": 1}},
    "poc": {"env": {"gpu_name": "A100", "platform": "poc"},
            "tests": {"llm": {"ttft": {"p50_s": 0.5}, "tokens_per_s_p50": 1}}},
}


def test_main_recognizes_every_known_schema(tmp_path, capsys):
    for name, obj in KNOWN.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(obj))
    CR.main([str(tmp_path / "*.json")])
    out = capsys.readouterr().out
    assert "unrecognized result schema" not in out
    assert "no usable result files" not in out
    # each schema produced its own column (distinct platform labels)
    for plat in ("vllm", "optimum", "trtllm", "mlpinf", "mlptrain", "poc"):
        assert plat in out


def test_main_skips_unknown_schema(tmp_path, capsys):
    (tmp_path / "weird.json").write_text(json.dumps({"schema": "weird/9.9", "env": {}}))
    CR.main([str(tmp_path / "*.json")])
    assert "unrecognized result schema" in capsys.readouterr().out
