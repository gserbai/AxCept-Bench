#!/usr/bin/env python3
"""Single entry point for the energy pipeline: AxRAM logs -> Ramulator -> DRAMPower.

Tools, Ramulator config and voltage map (memspecs) come from configs/pipeline.toml,
so only the experiment directory has to be given. Inside it the stages use:

    <workdir>/uy_filtered/         filtered AxRAM traces      (extract)
    <workdir>/ramulator_results/   Ramulator stats + cmdtraces (ramulator)
    <workdir>/drampower_results/   DRAMPower outputs + CSVs    (drampower)

Examples:
    python3 scripts/axcept.py config
    python3 scripts/axcept.py run -w experiments/jpeg
    python3 scripts/axcept.py ramulator -w experiments/jpeg --limit 1 --jobs 1
    python3 scripts/axcept.py drampower -w experiments/jpeg --dry-run

It can also be used from Python:
    from axcept import load_settings, stage_extract, stage_ramulator, stage_drampower
"""
from __future__ import annotations

from pathlib import Path
import argparse
import sys
import tomllib

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
DEFAULT_PIPELINE_CONFIG = PROJECT_ROOT / "configs" / "pipeline.toml"

for sub in ("axram", "ramulator", "drampower"):
    sys.path.insert(0, str(SCRIPTS_DIR / sub))

import extract_uy_structure  # noqa: E402
import run_ramulator_batch  # noqa: E402
import run_drampower_batch  # noqa: E402

UY_DIR = "uy_filtered"
RAMULATOR_DIR = "ramulator_results"
DRAMPOWER_DIR = "drampower_results"


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def load_settings(profile: str | None = None, pipeline_config: str | Path = DEFAULT_PIPELINE_CONFIG,
                  ramulator_config: str | Path | None = None,
                  voltage_map: str | Path | None = None) -> dict:
    """Read pipeline.toml and return absolute paths for the selected profile (explicit args override it)."""
    pipeline_config = resolve(pipeline_config)
    with pipeline_config.open("rb") as f:
        cfg = tomllib.load(f)

    defaults = cfg.get("defaults", {})
    profiles = cfg.get("profiles", {})
    profile = profile or defaults.get("profile")

    if profile not in profiles:
        raise KeyError(f"Profile '{profile}' not in {pipeline_config}. Available: {sorted(profiles)}")

    prof = profiles[profile]
    tools = cfg.get("tools", {})

    return {
        "pipeline_config": pipeline_config,
        "profile": profile,
        "description": prof.get("description", ""),
        "ramulator_bin": resolve(tools["ramulator"]),
        "drampower_bin": resolve(tools["drampower"]),
        "ramulator_config": resolve(ramulator_config or prof["ramulator_config"]),
        "voltage_map": resolve(voltage_map or prof["voltage_map"]),
        "jobs": int(defaults.get("jobs", 1)),
    }


def stage_extract(workdir: Path, logs_root: Path | None = None) -> dict:
    workdir = Path(workdir).resolve()
    logs_root = Path(logs_root).resolve() if logs_root else workdir
    result = extract_uy_structure.extract(logs_root, workdir / UY_DIR, verbose=False)

    print(f"Logs root: {logs_root}")
    print(f"AXRAM logs filtered: {result['files']} | U Y lines: {result['uy_lines']}")
    print(f"Output: {result['output_root']}")
    for path in result["empty_files"]:
        print(f"  [WARN] no U Y lines: {path}")

    if result["files"] == 0:
        raise RuntimeError(f"No AXRAM_log_pid*_hart*.log found under {logs_root}")

    return result


def stage_ramulator(settings: dict, workdir: Path, jobs: int | None = None,
                    limit: int | None = None, force: bool = False) -> list[dict]:
    workdir = Path(workdir).resolve()
    return run_ramulator_batch.run_batch(
        workdir / UY_DIR,
        workdir / RAMULATOR_DIR,
        settings["ramulator_bin"],
        settings["ramulator_config"],
        jobs=jobs or settings["jobs"],
        limit=limit,
        force=force,
    )


def stage_drampower(settings: dict, workdir: Path, jobs: int | None = None,
                    limit: int | None = None, scenarios: list[str] | None = None,
                    force: bool = False, dry_run: bool = False, timeout: int = 0) -> dict:
    workdir = Path(workdir).resolve()
    return run_drampower_batch.run_batch(
        workdir / RAMULATOR_DIR,
        workdir / DRAMPOWER_DIR,
        settings["voltage_map"],
        settings["drampower_bin"],
        jobs=jobs or settings["jobs"],
        limit_per_scenario=limit,
        scenarios_filter=scenarios,
        force=force,
        dry_run=dry_run,
        timeout=timeout,
    )


def show_config(settings: dict) -> bool:
    """Print the resolved settings and the scenarios of the voltage map. Returns False if anything is broken."""
    ok = True

    def status(path: Path) -> str:
        nonlocal ok
        if path.exists():
            return "ok"
        ok = False
        return "MISSING"

    print(f"Pipeline config: {settings['pipeline_config']}")
    print(f"Profile:         {settings['profile']}  {settings['description']}")
    print(f"Default jobs:    {settings['jobs']}")
    print()
    for key in ("ramulator_bin", "drampower_bin", "ramulator_config", "voltage_map"):
        print(f"  {key:<17} [{status(settings[key])}] {settings[key]}")

    if not settings["voltage_map"].exists():
        return False

    scenarios = run_drampower_batch.read_voltage_map(settings["voltage_map"])
    issues = run_drampower_batch.validate_scenarios(scenarios)

    print()
    print(f"  {'scenario':<26} {'mem_read_prob':>13} {'voltage_v':>9}  enabled  memspec")
    for name, row in scenarios.items():
        enabled = "yes" if row["enabled"] else "no"
        print(f"  {name:<26} {row['mem_read_prob']:>13} {row['voltage_v']:>9}  {enabled:<7}  "
              f"{Path(row['memspec_path']).name}")

    if issues:
        ok = False
        print()
        print("Issues:")
        for issue in issues:
            print(f"  - [{issue['type']}] {issue['scenario']}: {issue['message']}")

    return ok


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--pipeline-config", default=str(DEFAULT_PIPELINE_CONFIG),
                        help="Static pipeline config (default: configs/pipeline.toml)")
    common.add_argument("--profile", default=None,
                        help="Profile from the pipeline config (default: [defaults].profile)")
    common.add_argument("--ramulator-config", default=None, help="Override the profile's Ramulator config")
    common.add_argument("--voltage-map", default=None, help="Override the profile's voltage map CSV")

    workdir = argparse.ArgumentParser(add_help=False)
    workdir.add_argument("-w", "--workdir", required=True,
                         help="Experiment directory, e.g. experiments/jpeg")

    execution = argparse.ArgumentParser(add_help=False)
    execution.add_argument("--jobs", type=int, default=None, help="Parallel runs (default: [defaults].jobs)")
    execution.add_argument("--limit", type=int, default=None,
                           help="Test mode: Ramulator runs N traces, DRAMPower N traces per scenario")
    execution.add_argument("--force", action="store_true", help="Re-run traces that already have output")

    drampower = argparse.ArgumentParser(add_help=False)
    drampower.add_argument("--scenarios", nargs="*", default=None, help="Only these voltage map scenarios")
    drampower.add_argument("--timeout", type=int, default=0, help="DRAMPower timeout per trace in s (0 = none)")

    logs = argparse.ArgumentParser(add_help=False)
    logs.add_argument("--logs", default=None,
                      help="Root with the raw AXRAM_log_pid*_hart*.log files (default: the workdir)")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("config", parents=[common], help="Show the resolved config and validate memspecs")
    sub.add_parser("extract", parents=[common, workdir, logs], help="Filter U Y lines of the AxRAM logs")
    sub.add_parser("ramulator", parents=[common, workdir, execution], help="Run Ramulator on the filtered traces")
    p = sub.add_parser("drampower", parents=[common, workdir, execution, drampower],
                       help="Run DRAMPower on the Ramulator cmd traces")
    p.add_argument("--dry-run", action="store_true", help="Only list the tasks, do not run DRAMPower")
    sub.add_parser("run", parents=[common, workdir, logs, execution, drampower],
                   help="extract + ramulator + drampower")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    settings = load_settings(args.profile, args.pipeline_config, args.ramulator_config, args.voltage_map)

    if args.command == "config":
        return 0 if show_config(settings) else 1

    if args.command in ("extract", "run"):
        print("== extract")
        stage_extract(args.workdir, args.logs)
        print()

    if args.command in ("ramulator", "run"):
        print("== ramulator")
        stage_ramulator(settings, args.workdir, args.jobs, args.limit, args.force)
        print()

    if args.command in ("drampower", "run"):
        print("== drampower")
        stage_drampower(settings, args.workdir, args.jobs, args.limit, args.scenarios,
                        args.force, getattr(args, "dry_run", False), args.timeout)

    return 0


if __name__ == "__main__":
    sys.exit(main())
