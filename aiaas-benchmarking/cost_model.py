#!/usr/bin/env python3
"""Cost model: turn vLLM serving throughput into $/M-tokens.

Reads the serving result JSONs produced by `vllm_serving_benchmark.ipynb`
(schema "vllm-serving-bench/1.0") and estimates the cost to generate a million
tokens, split into two parts:

  - **energy**   — GPU power draw over the time to produce the tokens, scaled by
                   datacenter PUE and the electricity price.
  - **hardware** — server capex amortized over its lifetime, charged only for the
                   productive (utilized) hours.

Throughput drives everything: the faster the box, the fewer GPU-seconds (and
joules) each token costs.

    $/M-tok = (power_kW * PUE * hours_per_M * $/kWh)            # energy
            + (capex / (lifetime_h * utilization) * hours_per_M) # hardware
    where hours_per_M = 1e6 / throughput_tok_s / 3600

Power and capex default to a per-GPU table (overridable). If you measured real
power (DCGM / `nvidia-smi`), pass --power-watts to override the TDP estimate.

Usage:
    python cost_model.py 'vllm_bench_results/*.json'
    python cost_model.py 'vllm_bench_results/*.json' --price 0.15 --util 0.6 --json costs.json
    python cost_model.py run.json --power-watts 320   # measured per-GPU power
"""
import sys
import glob
import json
import argparse

# Rough per-GPU defaults: (board TDP watts, street capex USD per GPU).
# Override with --power-watts / --gpu-capex for your actual hardware and pricing.
GPU_DEFAULTS = {
    "h100": (700, 30000.0),
    "a100": (400, 10000.0),   # SXM 400W; PCIe is ~250W — use --power-watts to refine
    "l40":  (300, 9000.0),
    "l4":   (72,  2500.0),
    "a10":  (150, 3500.0),
    "v100": (300, 8000.0),
    "t4":   (70,  2000.0),
}
DEFAULT_SPEC = (300, 10000.0)  # unknown GPU fallback


def gpu_spec(name):
    """(tdp_w, capex_usd) for a GPU by case-insensitive substring match."""
    n = (name or "").lower()
    for key, spec in GPU_DEFAULTS.items():
        if key in n:
            return spec
    return DEFAULT_SPEC


def pick_point(run):
    """Max-throughput ('inf') sweep point if present, else the last one."""
    sweep = run.get("sweep", {})
    point = sweep.get("inf") or (list(sweep.values())[-1] if sweep else {})
    return point if isinstance(point, dict) else {}


def cost_per_m(throughput_tok_s, power_w, capex_usd, args):
    """$/M-tokens broken into (energy, hardware, total) for one throughput."""
    if not throughput_tok_s or throughput_tok_s <= 0:
        return None
    hours_per_m = (1e6 / throughput_tok_s) / 3600.0
    energy = (power_w / 1000.0) * args.pue * hours_per_m * args.price
    lifetime_h = args.amortization_years * 365 * 24
    hw_per_hour = capex_usd / (lifetime_h * args.util)
    hardware = hw_per_hour * hours_per_m
    return {"energy": energy, "hardware": hardware, "total": energy + hardware}


def analyze(run, args):
    env = run.get("env", {})
    gpu = env.get("gpu_name") or env.get("gpu") or "?"
    plat = env.get("platform") or "?"
    count = int(env.get("gpu_count", 1) or 1)

    tdp, capex = gpu_spec(gpu)
    power_w = (args.power_watts if args.power_watts is not None else tdp) * count
    total_capex = (args.gpu_capex if args.gpu_capex is not None else capex) * count * args.server_overhead

    point = pick_point(run)
    out_tps = point.get("output_throughput")
    tot_tps = point.get("total_token_throughput")

    return {
        "label": f"{gpu} x{count} ({plat})",
        "model": run.get("model"),
        "power_w": power_w,
        "total_capex": total_capex,
        "output_throughput": out_tps,
        "total_throughput": tot_tps,
        "out": cost_per_m(out_tps, power_w, total_capex, args),
        "tot": cost_per_m(tot_tps, power_w, total_capex, args),
    }


def fmt(x, nd=4):
    return f"${x:.{nd}f}" if isinstance(x, (int, float)) else "-"


def main(argv):
    ap = argparse.ArgumentParser(description="Estimate $/M-tokens from vLLM serving JSONs.")
    ap.add_argument("paths", nargs="+", help="globs of vllm-serving-bench JSONs")
    ap.add_argument("--price", type=float, default=0.12, help="electricity $/kWh (default 0.12)")
    ap.add_argument("--pue", type=float, default=1.4, help="datacenter PUE (default 1.4)")
    ap.add_argument("--amortization-years", type=float, default=3.0, help="HW amortization period (default 3)")
    ap.add_argument("--util", type=float, default=0.5,
                    help="duty cycle: fraction of lifetime the box is productive (default 0.5)")
    ap.add_argument("--server-overhead", type=float, default=1.3,
                    help="capex multiplier for CPU/RAM/chassis/networking on top of GPUs (default 1.3)")
    ap.add_argument("--power-watts", type=float, default=None,
                    help="measured per-GPU power (W); overrides the TDP estimate")
    ap.add_argument("--gpu-capex", type=float, default=None,
                    help="per-GPU capex (USD); overrides the default table")
    ap.add_argument("--json", default=None, help="also write results to this JSON file")
    args = ap.parse_args(argv)

    if not 0 < args.util <= 1:
        ap.error("--util must be in (0, 1]")

    paths = []
    for a in args.paths:
        paths.extend(sorted(glob.glob(a)))
    if not paths:
        print("no matching files")
        return

    rows = []
    for p in paths:
        try:
            run = json.load(open(p))
        except Exception as e:
            print(f"skip {p}: {e}")
            continue
        if not isinstance(run, dict) or not run.get("schema", "").startswith("vllm-serving-bench"):
            print(f"skip {p}: not a vllm-serving-bench result")
            continue
        rows.append(analyze(run, args))

    if not rows:
        print("no usable vLLM serving results")
        return

    print("Assumptions: "
          f"${args.price}/kWh, PUE {args.pue}, {args.amortization_years}y amortization, "
          f"util {args.util:.0%}, server-overhead x{args.server_overhead}")
    print("(power/capex from per-GPU defaults unless overridden)\n")

    metrics = [
        ("model", lambda r: r["model"]),
        ("power (W)", lambda r: f"{r['power_w']:.0f}"),
        ("capex ($)", lambda r: f"{r['total_capex']:.0f}"),
        ("out tok/s", lambda r: f"{r['output_throughput']:.0f}" if r["output_throughput"] else "-"),
        ("$/M out — energy", lambda r: fmt(r["out"]["energy"]) if r["out"] else "-"),
        ("$/M out — hardware", lambda r: fmt(r["out"]["hardware"]) if r["out"] else "-"),
        ("$/M out — TOTAL", lambda r: fmt(r["out"]["total"]) if r["out"] else "-"),
        ("$/M total-tok TOTAL", lambda r: fmt(r["tot"]["total"]) if r["tot"] else "-"),
    ]
    w = max(len(m) for m, _ in metrics)
    print(f"{'metric':{w}} " + " ".join(f"{r['label'][:30]:>30}" for r in rows))
    for name, getter in metrics:
        print(f"{name:{w}} " + " ".join(f"{str(getter(r)):>30}" for r in rows))

    if args.json:
        payload = {
            "schema": "vllm-cost-model/1.0",
            "assumptions": {
                "electricity_usd_per_kwh": args.price, "pue": args.pue,
                "amortization_years": args.amortization_years, "utilization": args.util,
                "server_overhead": args.server_overhead,
                "power_watts_override": args.power_watts, "gpu_capex_override": args.gpu_capex,
            },
            "results": rows,
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main(sys.argv[1:])
