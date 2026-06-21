"""Unit tests for cost_model.py — the $/M-token estimator.

Covers the longest-match GPU lookup, the cost formula (including the
energy+hardware split and the inverse-throughput scaling), and the end-to-end
analyze() path (max-throughput point selection + tensor-parallel charging).
"""
from argparse import Namespace

import pytest

import cost_model as CM


def _args(**kw):
    d = dict(price=0.12, pue=1.4, amortization_years=3.0, util=0.5,
             server_overhead=1.3, power_watts=None, gpu_capex=None, gpus=None)
    d.update(kw)
    return Namespace(**d)


def test_gpu_spec_longest_match():
    assert CM.gpu_spec("NVIDIA A100-SXM4-40GB") == (400, 10000.0)  # a100 beats a10
    assert CM.gpu_spec("NVIDIA A10") == (150, 3500.0)
    assert CM.gpu_spec("NVIDIA L40S") == (300, 9000.0)            # l40 beats l4
    assert CM.gpu_spec("NVIDIA L4") == (72, 2500.0)
    assert CM.gpu_spec("some unknown gpu") == CM.DEFAULT_SPEC


def test_cost_per_m_zero_or_negative_throughput_is_none():
    assert CM.cost_per_m(0, 400, 10000, _args()) is None
    assert CM.cost_per_m(-5, 400, 10000, _args()) is None


def test_cost_per_m_split_and_known_value():
    r = CM.cost_per_m(1000, 1000, 10000, _args())
    # total is exactly energy + hardware
    assert r["total"] == pytest.approx(r["energy"] + r["hardware"])
    # independent recompute of the documented formula
    hpm = (1e6 / 1000) / 3600.0
    energy = (1000 * 1.3 / 1000.0) * 1.4 * hpm * 0.12
    hw = 10000 / (3 * 365 * 24 * 0.5) * hpm
    assert r["energy"] == pytest.approx(energy)
    assert r["hardware"] == pytest.approx(hw)


def test_cost_scales_inversely_with_throughput():
    a = CM.cost_per_m(1000, 1000, 10000, _args())
    b = CM.cost_per_m(2000, 1000, 10000, _args())
    assert b["total"] == pytest.approx(a["total"] / 2)


def test_analyze_picks_max_point_and_honors_tensor_parallel():
    run = {
        "schema": "vllm-serving-bench/1.0", "model": "m", "tensor_parallel_size": 2,
        "env": {"gpu_name": "NVIDIA A100-SXM4-40GB", "platform": "local"},
        "sweep": {
            "4": {"output_throughput": 100, "total_token_throughput": 150},
            "inf": {"output_throughput": 500, "total_token_throughput": 700},
        },
    }
    r = CM.analyze(run, _args())
    assert r["gpus_used"] == 2
    assert r["output_throughput"] == 500          # the max-throughput point
    assert r["power_w"] == 400 * 2                 # A100 TDP * tp size
    assert r["total_capex"] == 10000 * 2 * 1.3     # capex * tp * server-overhead
    assert r["out"]["total"] > 0


def test_gpus_override_beats_recorded_tp():
    run = {"schema": "vllm-serving-bench/1.0", "model": "m", "tensor_parallel_size": 4,
           "env": {"gpu_name": "A100", "platform": "p"},
           "sweep": {"inf": {"output_throughput": 100}}}
    r = CM.analyze(run, _args(gpus=1))
    assert r["gpus_used"] == 1
