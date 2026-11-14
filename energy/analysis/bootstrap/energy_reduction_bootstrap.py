#!/usr/bin/env python3
"""
Energy reduction vs. the exact baseline, with 95% bootstrap confidence intervals.

Reads the per-trace CSV written by the energy pipeline
(``<workdir>/drampower_results/drampower_trace_results.csv``) and writes one row
per error rate with the point reduction and its 95% percentile CI.

For every scenario r != dataset_error_rate_0::

    reduction = 100 * (1 - mean(E_r) / mean(E_0))

The CI comes from an independent (unpaired) bootstrap: in each resample the
baseline traces and the scenario traces are drawn with replacement separately,
and the reduction is recomputed; the CI is the 2.5 and 97.5 percentiles of the
resampled reductions. Traces are not paired across scenarios because the trace
names are PID based and do not identify the input that produced them.

Only traces with status OK or SKIP and a numeric value are used. Scenarios are
ordered by mem_read_prob. Plotting is left to the user; this script only
produces the numbers.

Example::

    python3 scripts/analysis/energy_bootstrap/energy_reduction_bootstrap.py \\
        experiments/jpeg/drampower_results/drampower_trace_results.csv
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

BASELINE = "dataset_error_rate_0"

METRICS = {
    "energy": "total_trace_energy_pj",
    "power": "average_power_mw",
}

OUTPUT_FIELDS = [
    "scenario",
    "mem_read_prob",
    "voltage_v",
    "n_baseline",
    "n_scenario",
    "mean_baseline",
    "mean_scenario",
    "reduction_percent",
    "ci95_low_percent",
    "ci95_high_percent",
]


def to_float(value: str):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_traces(path: Path, column: str) -> dict[str, dict]:
    """Group the valid values of `column` by scenario."""
    scenarios: dict[str, dict] = {}
    seen = set()

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = {"scenario", "trace_id", "status", column} - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"{path} is missing columns: {sorted(missing)}")

        for row in reader:
            if row["status"] not in {"OK", "SKIP"}:
                continue
            value = to_float(row[column])
            if value is None:
                continue

            key = (row["scenario"], row["trace_id"])
            if key in seen:
                raise RuntimeError(f"Duplicate scenario/trace_id in {path}: {key}")
            seen.add(key)

            entry = scenarios.setdefault(row["scenario"], {
                "values": [],
                "mem_read_prob": row.get("mem_read_prob", ""),
                "voltage_v": row.get("voltage_v", ""),
            })
            entry["values"].append(value)

    return scenarios


def percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolation percentile (same as numpy's default)."""
    pos = (len(sorted_values) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def bootstrap_reduction(baseline: list[float], scenario: list[float],
                        n_boot: int, rng: random.Random) -> tuple[float, float]:
    n_b, n_s = len(baseline), len(scenario)
    boot = []
    for _ in range(n_boot):
        mean_b = sum(rng.choices(baseline, k=n_b)) / n_b
        mean_s = sum(rng.choices(scenario, k=n_s)) / n_s
        boot.append(100.0 * (1.0 - mean_s / mean_b))
    boot.sort()
    return percentile(boot, 2.5), percentile(boot, 97.5)


def summarize(scenarios: dict[str, dict], n_boot: int, seed: int) -> list[dict]:
    if BASELINE not in scenarios:
        raise RuntimeError(f"Baseline scenario '{BASELINE}' not found (no valid traces).")

    baseline = scenarios[BASELINE]["values"]
    mean_b = sum(baseline) / len(baseline)
    if mean_b <= 0:
        raise RuntimeError("Baseline mean must be positive.")

    order = sorted(scenarios, key=lambda s: (to_float(scenarios[s]["mem_read_prob"]) or 0.0, s))
    rows = []

    for index, name in enumerate(order):
        entry = scenarios[name]
        values = entry["values"]
        mean_s = sum(values) / len(values)
        point = 100.0 * (1.0 - mean_s / mean_b)

        if name == BASELINE:
            low, high = 0.0, 0.0
        elif len(baseline) < 2 or len(values) < 2:
            print(f"[WARN] {name}: fewer than 2 traces in a sample, CI left empty")
            low, high = "", ""
        else:
            low, high = bootstrap_reduction(baseline, values, n_boot, random.Random(seed + index))

        rows.append({
            "scenario": name,
            "mem_read_prob": entry["mem_read_prob"],
            "voltage_v": entry["voltage_v"],
            "n_baseline": len(baseline),
            "n_scenario": len(values),
            "mean_baseline": mean_b,
            "mean_scenario": mean_s,
            "reduction_percent": point,
            "ci95_low_percent": low,
            "ci95_high_percent": high,
        })

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Energy/power reduction vs. dataset_error_rate_0 with independent-bootstrap 95% CIs."
    )
    parser.add_argument("traces", type=Path, help="drampower_trace_results.csv from the energy pipeline")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output CSV (default: energy_reduction_bootstrap.csv next to the input)")
    parser.add_argument("--metric", choices=sorted(METRICS), default="energy",
                        help="energy (total_trace_energy_pj) or power (average_power_mw)")
    parser.add_argument("--bootstrap", type=int, default=10_000, help="Number of bootstrap resamples")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.bootstrap < 100:
        parser.error("--bootstrap must be at least 100")

    traces = args.traces.expanduser().resolve()
    if not traces.exists():
        raise FileNotFoundError(f"Trace CSV not found: {traces}")

    output = args.output or traces.parent / f"{args.metric}_reduction_bootstrap.csv"
    output = output.expanduser().resolve()

    scenarios = load_traces(traces, METRICS[args.metric])
    rows = summarize(scenarios, args.bootstrap, args.seed)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Metric: {args.metric} | bootstrap: {args.bootstrap} | seed: {args.seed}")
    for r in rows:
        ci = (f"[{r['ci95_low_percent']:.2f}, {r['ci95_high_percent']:.2f}]"
              if r["ci95_low_percent"] != "" else "[n/a]")
        print(f"  {r['scenario']:<26} n={r['n_scenario']:<5} "
              f"reduction={r['reduction_percent']:7.2f}%  CI95={ci}")
    print(f"CSV: {output}")


if __name__ == "__main__":
    main()
