#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import csv
import math
import re
import statistics
import subprocess
import time


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TOTAL_ENERGY_RE = re.compile(r"Total Trace Energy:\s*([0-9.eE+-]+)\s*pJ")
AVG_POWER_RE = re.compile(r"Average Power:\s*([0-9.eE+-]+)\s*mW")

METRIC_PATTERNS = {
    "act_commands": (r"#ACT commands:\s*([0-9]+)", int),
    "read_commands": (r"#RD \+ #RDA commands:\s*([0-9]+)", int),
    "write_commands": (r"#WR \+ #WRA commands:\s*([0-9]+)", int),
    "pre_commands": (r"#PRE \(\+ PREA\) commands:\s*([0-9]+)", int),
    "ref_commands": (r"#REF commands:\s*([0-9]+)", int),
    "total_trace_cycles": (r"Total Trace Length \(clock cycles\):\s*([0-9]+)", int),

    "act_cmd_energy_pj": (r"ACT Cmd Energy:\s*([0-9.eE+-]+)\s*pJ", float),
    "pre_cmd_energy_pj": (r"PRE Cmd Energy:\s*([0-9.eE+-]+)\s*pJ", float),
    "rd_cmd_energy_pj": (r"RD Cmd Energy:\s*([0-9.eE+-]+)\s*pJ", float),
    "wr_cmd_energy_pj": (r"WR Cmd Energy:\s*([0-9.eE+-]+)\s*pJ", float),
    "act_stdby_energy_pj": (r"ACT Stdby Energy:\s*([0-9.eE+-]+)\s*pJ", float),
    "pre_stdby_energy_pj": (r"PRE Stdby Energy:\s*([0-9.eE+-]+)\s*pJ", float),
    "idle_energy_pj": (r"Total Idle Energy \(Active \+ Precharged\):\s*([0-9.eE+-]+)\s*pJ", float),
    "refresh_energy_pj": (r"Auto-Refresh Energy:\s*([0-9.eE+-]+)\s*pJ", float),
}


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def enabled_value(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def read_voltage_map(csv_path: Path):
    scenarios = {}

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        required = {
            "scenario",
            "mem_read_prob",
            "mapped_error_percent",
            "voltage_v",
            "memspec_path",
            "enabled",
        }

        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"CSV missing required columns: {sorted(missing)}")

        for row in reader:
            scenario = row["scenario"].strip()

            if scenario in scenarios:
                raise RuntimeError(f"Duplicated scenario in CSV: {scenario}")

            memspec_path = resolve_project_path(row["memspec_path"])

            scenarios[scenario] = {
                "scenario": scenario,
                "mem_read_prob": row["mem_read_prob"].strip(),
                "mapped_error_percent": row["mapped_error_percent"].strip(),
                "mapped_error_percent_order": row.get("mapped_error_percent_order", "").strip(),
                "voltage_v": row["voltage_v"].strip(),
                "memspec_path": str(memspec_path),
                "enabled": enabled_value(row["enabled"]),
                "notes": row.get("notes", "").strip(),
            }

    return scenarios


def validate_scenarios(scenarios):
    issues = []

    for scenario, row in scenarios.items():
        if not row["enabled"]:
            continue

        memspec = Path(row["memspec_path"])

        if not memspec.exists():
            issues.append({
                "type": "missing_memspec",
                "scenario": scenario,
                "message": f"Missing memspec: {memspec}",
            })
            row["valid"] = False
            continue

        text = memspec.read_text(encoding="utf-8", errors="ignore")

        if "<!DOCTYPE memspec SYSTEM \"memspec.dtd\">" not in text:
            issues.append({
                "type": "missing_doctype",
                "scenario": scenario,
                "message": f"Missing expected DOCTYPE in {memspec}",
            })
            row["valid"] = False
            continue

        if 'id="vdd"' not in text:
            issues.append({
                "type": "missing_vdd",
                "scenario": scenario,
                "message": f"Missing vdd field in {memspec}",
            })
            row["valid"] = False
            continue

        if not (memspec.parent / "memspec.dtd").exists():
            issues.append({
                "type": "missing_dtd",
                "scenario": scenario,
                "message": f"Missing memspec.dtd beside {memspec}",
            })
            row["valid"] = False
            continue

        row["valid"] = True

    return issues


def parse_output(text: str):
    data = {}

    m_energy = TOTAL_ENERGY_RE.search(text)
    m_power = AVG_POWER_RE.search(text)

    data["total_trace_energy_pj"] = float(m_energy.group(1)) if m_energy else ""
    data["average_power_mw"] = float(m_power.group(1)) if m_power else ""

    for name, (pattern, cast) in METRIC_PATTERNS.items():
        m = re.search(pattern, text)
        data[name] = cast(m.group(1)) if m else ""

    return data


def safe_trace_id(trace: Path, scenario_dir: Path):
    rel_parent = trace.parent.relative_to(scenario_dir)

    if rel_parent.parts:
        raw = "__".join(rel_parent.parts)
    else:
        raw = trace.stem

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


def run_one(task, drampower_bin: Path, output_root: Path, force: bool, timeout: int):
    scenario = task["scenario"]
    scenario_dir = task["scenario_dir"]
    trace = task["trace"]
    memspec = Path(task["memspec_path"])

    trace_id = safe_trace_id(trace, scenario_dir)

    run_dir = output_root / "runs" / scenario / trace_id
    stdout_path = run_dir / "drampower.out"
    stderr_path = run_dir / "drampower.err"

    run_dir.mkdir(parents=True, exist_ok=True)

    base_result = {
        "scenario": scenario,
        "trace_id": trace_id,
        "trace_path": str(trace),
        "trace_relative": str(trace.relative_to(scenario_dir)),
        "voltage_v": task["voltage_v"],
        "mem_read_prob": task["mem_read_prob"],
        "mapped_error_percent": task["mapped_error_percent"],
        "mapped_error_percent_order": task["mapped_error_percent_order"],
        "memspec_path": str(memspec),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "returncode": "",
        "elapsed_sec": "",
    }

    if stdout_path.exists() and not force:
        previous = stdout_path.read_text(encoding="utf-8", errors="ignore")
        parsed = parse_output(previous)

        if parsed["total_trace_energy_pj"] != "" and parsed["average_power_mw"] != "":
            return {
                **base_result,
                **parsed,
                "status": "SKIP",
                "returncode": 0,
                "elapsed_sec": 0,
            }

    cmd = [
        str(drampower_bin),
        "-m",
        str(memspec),
        "-c",
        str(trace),
    ]

    t0 = time.time()

    try:
        completed = subprocess.run(
            cmd,
            cwd=drampower_bin.parent,
            capture_output=True,
            text=True,
            timeout=timeout if timeout > 0 else None,
        )
        elapsed = time.time() - t0

        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")

        combined = completed.stdout + "\n" + completed.stderr
        parsed = parse_output(combined)

        if completed.returncode != 0:
            status = "ERROR"
        elif parsed["total_trace_energy_pj"] == "":
            status = "NO_ENERGY"
        else:
            status = "OK"

        return {
            **base_result,
            **parsed,
            "status": status,
            "returncode": completed.returncode,
            "elapsed_sec": round(elapsed, 4),
        }

    except subprocess.TimeoutExpired as exc:
        elapsed = time.time() - t0

        stdout_path.write_text(exc.stdout or "", encoding="utf-8")
        stderr_path.write_text(exc.stderr or "TIMEOUT", encoding="utf-8")

        empty_metrics = parse_output("")

        return {
            **base_result,
            **empty_metrics,
            "status": "TIMEOUT",
            "returncode": "",
            "elapsed_sec": round(elapsed, 4),
        }


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stat_block(values):
    values = [float(v) for v in values if v != ""]

    n = len(values)

    if n == 0:
        return {
            "n": 0,
            "mean": "",
            "std": "",
            "ci95_low": "",
            "ci95_high": "",
            "ci95_half_width": "",
        }

    mean = statistics.mean(values)

    if n == 1:
        std = 0.0
        half = 0.0
    else:
        std = statistics.stdev(values)
        half = 1.96 * std / math.sqrt(n)

    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "ci95_half_width": half,
    }


def build_summary(results, scenarios):
    scenario_names = sorted({r["scenario"] for r in results})
    rows = []

    ok_results = [
        r for r in results
        if r["status"] in {"OK", "SKIP"}
        and r["total_trace_energy_pj"] != ""
        and r["average_power_mw"] != ""
    ]

    baseline_energy_mean = None
    baseline_power_mean = None

    baseline_rows = [r for r in ok_results if r["scenario"] == "dataset_error_rate_0"]

    if baseline_rows:
        baseline_energy_mean = stat_block([r["total_trace_energy_pj"] for r in baseline_rows])["mean"]
        baseline_power_mean = stat_block([r["average_power_mw"] for r in baseline_rows])["mean"]

    for scenario in scenario_names:
        rows_all = [r for r in results if r["scenario"] == scenario]
        rows_ok = [r for r in ok_results if r["scenario"] == scenario]

        energy = stat_block([r["total_trace_energy_pj"] for r in rows_ok])
        power = stat_block([r["average_power_mw"] for r in rows_ok])

        meta = scenarios[scenario]

        if baseline_energy_mean and energy["mean"] != "":
            energy_reduction = (1.0 - float(energy["mean"]) / float(baseline_energy_mean)) * 100.0
        else:
            energy_reduction = ""

        if baseline_power_mean and power["mean"] != "":
            power_reduction = (1.0 - float(power["mean"]) / float(baseline_power_mean)) * 100.0
        else:
            power_reduction = ""

        rows.append({
            "scenario": scenario,
            "n_total": len(rows_all),
            "n_ok": energy["n"],
            "n_failed": len(rows_all) - energy["n"],
            "voltage_v": meta["voltage_v"],
            "mem_read_prob": meta["mem_read_prob"],
            "mapped_error_percent": meta["mapped_error_percent"],
            "mapped_error_percent_order": meta["mapped_error_percent_order"],

            "energy_mean_pj": energy["mean"],
            "energy_std_pj": energy["std"],
            "energy_ci95_low_pj": energy["ci95_low"],
            "energy_ci95_high_pj": energy["ci95_high"],
            "energy_ci95_half_width_pj": energy["ci95_half_width"],
            "energy_reduction_vs_dataset_error_rate_0_percent": energy_reduction,

            "power_mean_mw": power["mean"],
            "power_std_mw": power["std"],
            "power_ci95_low_mw": power["ci95_low"],
            "power_ci95_high_mw": power["ci95_high"],
            "power_ci95_half_width_mw": power["ci95_half_width"],
            "power_reduction_vs_dataset_error_rate_0_percent": power_reduction,
        })

    return rows


def discover_unmapped(input_root: Path, scenarios):
    rows = []

    if not input_root.exists():
        return rows

    for child in sorted(input_root.iterdir()):
        if not child.is_dir():
            continue

        count = sum(1 for _ in child.rglob("cmd-trace-*.cmdtrace"))

        if child.name not in scenarios:
            rows.append({
                "scenario_dir": child.name,
                "status": "not_in_csv_ignored",
                "cmdtrace_count": count,
                "path": str(child),
            })
        elif not scenarios[child.name]["enabled"]:
            rows.append({
                "scenario_dir": child.name,
                "status": "disabled_in_csv_ignored",
                "cmdtrace_count": count,
                "path": str(child),
            })

    return rows


def run_batch(input_root: Path, output_root: Path, voltage_map: Path, drampower_bin: Path,
              jobs: int = 1, limit_per_scenario: int | None = None,
              scenarios_filter: list[str] | None = None, force: bool = False,
              dry_run: bool = False, timeout: int = 0) -> dict:
    """Run DRAMPower on every cmd trace of the enabled scenarios in the voltage map and write the CSVs."""

    input_root = Path(input_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    voltage_map = Path(voltage_map).expanduser().resolve()
    drampower_bin = Path(drampower_bin).expanduser().resolve()

    if not voltage_map.exists():
        raise FileNotFoundError(f"Voltage map not found: {voltage_map}")

    if not drampower_bin.exists():
        raise FileNotFoundError(f"DRAMPower binary not found: {drampower_bin}")

    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    scenarios = read_voltage_map(voltage_map)
    issues = validate_scenarios(scenarios)

    selected = set(scenarios_filter) if scenarios_filter else None

    if selected:
        unknown = selected - set(scenarios.keys())
        if unknown:
            raise RuntimeError(f"Requested scenarios not found in CSV: {sorted(unknown)}")

    unmapped = discover_unmapped(input_root, scenarios)

    tasks = []

    for scenario, meta in scenarios.items():
        if selected and scenario not in selected:
            continue

        if not meta["enabled"]:
            continue

        if not meta.get("valid", False):
            continue

        scenario_dir = input_root / scenario

        if not scenario_dir.exists():
            issues.append({
                "type": "missing_scenario_dir",
                "scenario": scenario,
                "message": f"Missing scenario directory: {scenario_dir}",
            })
            continue

        traces = sorted(scenario_dir.rglob("cmd-trace-*.cmdtrace"))

        if limit_per_scenario is not None:
            traces = traces[:limit_per_scenario]

        if not traces:
            issues.append({
                "type": "no_cmdtraces",
                "scenario": scenario,
                "message": f"No cmd-trace files found in {scenario_dir}",
            })
            continue

        for trace in traces:
            tasks.append({
                **meta,
                "trace": trace,
                "scenario_dir": scenario_dir,
            })

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Input root: {input_root}")
    print(f"Output root: {output_root}")
    print(f"Voltage map: {voltage_map}")
    print(f"DRAMPower: {drampower_bin}")
    print(f"Enabled valid scenarios: {sum(1 for s in scenarios.values() if s.get('enabled') and s.get('valid', False))}")
    print(f"Tasks: {len(tasks)}")
    print(f"Jobs: {jobs}")

    if unmapped:
        print()
        print("Unmapped/disabled scenario folders will be ignored:")
        for row in unmapped:
            print(f"  - {row['scenario_dir']} ({row['status']}, cmdtraces={row['cmdtrace_count']})")

    if issues:
        print()
        print("Issues detected:")
        for issue in issues:
            print(f"  - [{issue['type']}] {issue.get('scenario', '')}: {issue['message']}")

    output_root.mkdir(parents=True, exist_ok=True)

    issue_fields = ["type", "scenario", "message"]
    write_csv(output_root / "drampower_run_issues.csv", issues, issue_fields)

    unmapped_fields = ["scenario_dir", "status", "cmdtrace_count", "path"]
    write_csv(output_root / "drampower_unmapped_scenarios.csv", unmapped, unmapped_fields)

    if dry_run:
        print()
        print("Dry-run mode: DRAMPower was not executed.")
        return {"tasks": tasks, "issues": issues, "unmapped": unmapped, "results": [], "summary": []}

    results = []

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = [
            executor.submit(
                run_one,
                task,
                drampower_bin,
                output_root,
                force,
                timeout,
            )
            for task in tasks
        ]

        for i, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            results.append(row)

            print(
                f"[{i}/{len(tasks)}] {row['status']} | "
                f"{row['scenario']} | V={row['voltage_v']} | "
                f"E={row['total_trace_energy_pj']} pJ | "
                f"P={row['average_power_mw']} mW | "
                f"{row['trace_id']}"
            )

    results.sort(key=lambda r: (r["scenario"], r["trace_id"]))

    trace_fields = [
        "scenario",
        "trace_id",
        "trace_relative",
        "trace_path",
        "status",
        "returncode",
        "elapsed_sec",
        "voltage_v",
        "mem_read_prob",
        "mapped_error_percent",
        "mapped_error_percent_order",
        "memspec_path",
        "total_trace_energy_pj",
        "average_power_mw",
        "act_commands",
        "read_commands",
        "write_commands",
        "pre_commands",
        "ref_commands",
        "total_trace_cycles",
        "act_cmd_energy_pj",
        "pre_cmd_energy_pj",
        "rd_cmd_energy_pj",
        "wr_cmd_energy_pj",
        "act_stdby_energy_pj",
        "pre_stdby_energy_pj",
        "idle_energy_pj",
        "refresh_energy_pj",
        "stdout_path",
        "stderr_path",
    ]

    write_csv(output_root / "drampower_trace_results.csv", results, trace_fields)

    summary = build_summary(results, scenarios)

    summary_fields = [
        "scenario",
        "n_total",
        "n_ok",
        "n_failed",
        "voltage_v",
        "mem_read_prob",
        "mapped_error_percent",
        "mapped_error_percent_order",
        "energy_mean_pj",
        "energy_std_pj",
        "energy_ci95_low_pj",
        "energy_ci95_high_pj",
        "energy_ci95_half_width_pj",
        "energy_reduction_vs_dataset_error_rate_0_percent",
        "power_mean_mw",
        "power_std_mw",
        "power_ci95_low_mw",
        "power_ci95_high_mw",
        "power_ci95_half_width_mw",
        "power_reduction_vs_dataset_error_rate_0_percent",
    ]

    write_csv(output_root / "drampower_scenario_summary.csv", summary, summary_fields)

    print()
    print("Done.")
    print(f"Trace CSV:   {output_root / 'drampower_trace_results.csv'}")
    print(f"Summary CSV: {output_root / 'drampower_scenario_summary.csv'}")
    print(f"Issues CSV:  {output_root / 'drampower_run_issues.csv'}")
    print(f"Unmapped CSV:{output_root / 'drampower_unmapped_scenarios.csv'}")

    return {"tasks": tasks, "issues": issues, "unmapped": unmapped, "results": results, "summary": summary}


def main():
    parser = argparse.ArgumentParser(
        description="Safely run DRAMPower using pre-defined memspec XMLs from a scenario CSV."
    )

    parser.add_argument(
        "--input-root",
        default=str(Path.home() / "Documents" / "ramulator_results"),
        help="Root containing Ramulator results organized as <scenario>/<trace>/cmd-trace-*.cmdtrace",
    )

    parser.add_argument(
        "--output-root",
        default=str(Path.home() / "Documents" / "drampower_results_vendor_b"),
        help="Root where DRAMPower outputs and CSV summaries will be stored",
    )

    parser.add_argument(
        "--voltage-map",
        default=str(PROJECT_ROOT / "configs" / "drampower" / "voltage_map_vendor_b.csv"),
        help="CSV mapping scenario -> voltage -> memspec_path",
    )

    parser.add_argument(
        "--drampower",
        default=str(PROJECT_ROOT / "DRAMPower-4.1" / "drampower"),
        help="Path to DRAMPower executable",
    )

    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Parallel DRAMPower processes",
    )

    parser.add_argument(
        "--limit-per-scenario",
        type=int,
        default=None,
        help="For testing: limit number of traces per scenario",
    )

    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=None,
        help="Optional list of scenario names to run",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run traces even if drampower.out already exists",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be processed; do not run DRAMPower",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Timeout in seconds per trace; 0 disables timeout",
    )

    args = parser.parse_args()

    run_batch(
        args.input_root,
        args.output_root,
        args.voltage_map,
        args.drampower,
        jobs=args.jobs,
        limit_per_scenario=args.limit_per_scenario,
        scenarios_filter=args.scenarios,
        force=args.force,
        dry_run=args.dry_run,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
