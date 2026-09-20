#!/usr/bin/env python3
"""
fft_output_classifier.py
=========================

Classifies AxCept-Bench FFT (dominant-frequency) execution outputs and
produces an aggregated CSV summary per memory error rate.

WHAT IT DOES
------------
For each error rate, this script scans the raw ".bin" outputs produced by
the AxPike/AxRAM batch runner and classifies every file directly from its
bytes (parsing the AXDFREQ1 binary header, detecting crash signatures, and
checking for missing/empty outputs). Classification is fully reconstructed
from the .bin outputs themselves.

Output classes:
    OK_RESULT                 valid, finite, in-range frequency result
    RESULT_WITH_CRASH_DUMP    valid result, but a crash signature was
                               also present in the same output
    AXPIKE_CRASH_DUMP         crash signature present, no valid AXDFREQ1
                               header
    APPLICATION_ERROR         header present, status field != 0
    INVALID_NUMERIC           header present, frequency out of range /
                               not finite
    INVALID_OUTPUT            unrecognized output format
    TIMEOUT_OR_EMPTY          zero-byte output
    MISSING_OUTPUT            expected file was never produced

These are aggregated into the three categories used in the execution
stability chart:
    Successes (Survivors)    = OK_RESULT
    Data Crash (Crash Dump)  = RESULT_WITH_CRASH_DUMP + AXPIKE_CRASH_DUMP
                                + APPLICATION_ERROR + INVALID_OUTPUT
                                + INVALID_NUMERIC
    Flow / Timeout (0 bytes) = TIMEOUT_OR_EMPTY + MISSING_OUTPUT

EXPECTED DIRECTORY LAYOUT
--------------------------
Reference input dataset (.bin files, one execution's worth):
    <base-dir>/dataset_iowa_music_bin/**/*.bin

Per-rate outputs, one folder per evaluated error rate:
    <base-dir>/dataset_iowa_music_<rate>/src/dataset_audio_error_rate_<rate>/**/*.bin

USAGE
-----
Basic run, using the default layout under ~/Music:
    $ python3 fft_output_classifier.py

Custom base directory:
    $ python3 fft_output_classifier.py --base-dir /path/to/Music

Only a subset of error rates:
    $ python3 fft_output_classifier.py --rates 1e-3 1e-2 1e-1

Custom output CSV path:
    $ python3 fft_output_classifier.py --out results/fft_summary.csv

OUTPUT
------
A single CSV with one row per evaluated error rate, containing counts and
percentages for every class above, plus the aggregated
successes/data-crash/flow-timeout figures used for plotting.
"""

from __future__ import annotations

import argparse
import csv
import math
import struct
from collections import Counter
from pathlib import Path

RATES_DEFAULT = ["1e-5", "1e-4", "1e-3", "1e-2", "1e-1"]

CRASH_SIGNATURES = [
    b"z  ",
    b"User load segfault",
    b"User store segfault",
    b"User fetch segfault",
    b"segfault",
]

AXDFREQ_HEADER = b"AXDFREQ1"
AXDFREQ_HEADER_SIZE = 24  # 8s (magic) + i (status) + i (num_samples) + i (num_frames) + f (freq_hz)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="fft_output_classifier.py",
        description=(
            "Classify AxCept-Bench FFT (.bin) outputs per error rate and "
            "export an aggregated CSV summary for the execution-stability chart."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 fft_output_classifier.py\n"
            "  python3 fft_output_classifier.py --base-dir /path/to/Music\n"
            "  python3 fft_output_classifier.py --rates 1e-3 1e-2 1e-1\n"
            "  python3 fft_output_classifier.py --out results/fft_summary.csv\n"
        ),
    )

    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path.home() / "Music",
        help="Base directory containing dataset_iowa_music_<rate> folders. "
             "Default: ~/Music",
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="Reference input dataset (.bin files used to determine the "
             "expected file list). Default: <base-dir>/dataset_iowa_music_bin",
    )

    parser.add_argument(
        "--rates",
        nargs="+",
        default=RATES_DEFAULT,
        help=f"Error rates to evaluate. Default: {' '.join(RATES_DEFAULT)}",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path. Default: <base-dir>/fft_direct_output_count_by_rate.csv",
    )

    return parser.parse_args()


# =============================================================================
# Classification
# =============================================================================

def has_crash_dump(data: bytes) -> bool:
    """Return True if any known AxPike crash signature appears in the output."""
    return any(sig in data for sig in CRASH_SIGNATURES)


def classify_output_bytes(data: bytes) -> str:
    """
    Classify a single raw .bin output.

    Mirrors the classification logic used during batch execution, but
    applied retroactively directly on the saved bytes.
    """
    if len(data) == 0:
        return "TIMEOUT_OR_EMPTY"

    crash_dump = has_crash_dump(data)

    if data.startswith(AXDFREQ_HEADER) and len(data) >= AXDFREQ_HEADER_SIZE:
        try:
            _, status, _num_samples, _num_frames, freq_hz = struct.unpack(
                "<8siiif", data[:AXDFREQ_HEADER_SIZE]
            )
        except struct.error:
            return "INVALID_OUTPUT"

        if status != 0:
            return "APPLICATION_ERROR"

        if not math.isfinite(freq_hz) or freq_hz <= 0.0 or freq_hz > 22050.0:
            return "INVALID_NUMERIC"

        if crash_dump:
            return "RESULT_WITH_CRASH_DUMP"

        return "OK_RESULT"

    if crash_dump:
        return "AXPIKE_CRASH_DUMP"

    return "INVALID_OUTPUT"


# =============================================================================
# Directory helpers
# =============================================================================

def get_expected_relative_paths(input_root: Path) -> list[Path]:
    """List every expected .bin file, relative to the reference input dataset."""
    return sorted(p.relative_to(input_root) for p in input_root.rglob("*.bin"))


def output_root_for_rate(base_dir: Path, rate: str) -> Path:
    return (
        base_dir
        / f"dataset_iowa_music_{rate}"
        / "src"
        / f"dataset_audio_error_rate_{rate}"
    )


def pct(count: int, total: int) -> float:
    return 100.0 * count / total if total else 0.0


# =============================================================================
# Per-rate analysis
# =============================================================================

def analyze_rate(base_dir: Path, rate: str, expected_rel_paths: list[Path]) -> dict:
    out_root = output_root_for_rate(base_dir, rate)

    if not out_root.exists():
        raise FileNotFoundError(f"Output folder not found for rate {rate}: {out_root}")

    class_counts: Counter = Counter()
    total = len(expected_rel_paths)

    for rel in expected_rel_paths:
        out_path = out_root / rel

        if not out_path.exists():
            class_counts["MISSING_OUTPUT"] += 1
            continue

        data = out_path.read_bytes()
        class_counts[classify_output_bytes(data)] += 1

    ok_result = class_counts["OK_RESULT"]

    zero_bytes = class_counts["TIMEOUT_OR_EMPTY"] + class_counts["MISSING_OUTPUT"]

    crash_before_data = class_counts["AXPIKE_CRASH_DUMP"]
    crash_with_data = class_counts["RESULT_WITH_CRASH_DUMP"]

    invalid_or_app_error = (
        class_counts["APPLICATION_ERROR"]
        + class_counts["INVALID_OUTPUT"]
        + class_counts["INVALID_NUMERIC"]
    )

    successes_survivors = ok_result
    flow_timeout_zero_bytes = zero_bytes
    data_crash_crash_dump = crash_before_data + crash_with_data + invalid_or_app_error

    classified_total = successes_survivors + flow_timeout_zero_bytes + data_crash_crash_dump

    return {
        "error_rate": rate,
        "output_root": str(out_root),
        "total_files_evaluated": total,
        "classified_total": classified_total,

        "successes_survivors": successes_survivors,
        "data_crash_crash_dump": data_crash_crash_dump,
        "flow_timeout_zero_bytes": flow_timeout_zero_bytes,

        "successes_survivors_pct": pct(successes_survivors, total),
        "data_crash_crash_dump_pct": pct(data_crash_crash_dump, total),
        "flow_timeout_zero_bytes_pct": pct(flow_timeout_zero_bytes, total),

        "OK_RESULT": class_counts["OK_RESULT"],
        "RESULT_WITH_CRASH_DUMP": class_counts["RESULT_WITH_CRASH_DUMP"],
        "AXPIKE_CRASH_DUMP": class_counts["AXPIKE_CRASH_DUMP"],
        "TIMEOUT_OR_EMPTY": class_counts["TIMEOUT_OR_EMPTY"],
        "APPLICATION_ERROR": class_counts["APPLICATION_ERROR"],
        "INVALID_OUTPUT": class_counts["INVALID_OUTPUT"],
        "INVALID_NUMERIC": class_counts["INVALID_NUMERIC"],
        "MISSING_OUTPUT": class_counts["MISSING_OUTPUT"],
    }


def write_csv(rows: list[dict], out_csv: Path) -> None:
    if not rows:
        return

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# Reporting
# =============================================================================

def print_rate_summary(row: dict) -> None:
    print(f"{'Rate':22}: {row['error_rate']}")
    print(f"{'Files evaluated':22}: {row['classified_total']} / {row['total_files_evaluated']}")
    print(
        f"{'Successes':22}: {row['successes_survivors']:>5}"
        f" ({row['successes_survivors_pct']:5.2f}%)"
    )
    print(
        f"{'Data crash':22}: {row['data_crash_crash_dump']:>5}"
        f" ({row['data_crash_crash_dump_pct']:5.2f}%)"
    )
    print(
        f"{'Flow / timeout':22}: {row['flow_timeout_zero_bytes']:>5}"
        f" ({row['flow_timeout_zero_bytes_pct']:5.2f}%)"
    )
    print(
        "  breakdown -> "
        f"OK_RESULT={row['OK_RESULT']} "
        f"RESULT_WITH_CRASH_DUMP={row['RESULT_WITH_CRASH_DUMP']} "
        f"AXPIKE_CRASH_DUMP={row['AXPIKE_CRASH_DUMP']} "
        f"APPLICATION_ERROR={row['APPLICATION_ERROR']} "
        f"INVALID_NUMERIC={row['INVALID_NUMERIC']} "
        f"INVALID_OUTPUT={row['INVALID_OUTPUT']} "
        f"TIMEOUT_OR_EMPTY={row['TIMEOUT_OR_EMPTY']} "
        f"MISSING_OUTPUT={row['MISSING_OUTPUT']}"
    )
    print("-" * 72)


def main() -> None:
    args = parse_args()

    base_dir = args.base_dir.expanduser().resolve()
    input_root = (
        args.input_root.expanduser().resolve()
        if args.input_root
        else base_dir / "dataset_iowa_music_bin"
    )
    out_csv = (
        args.out.expanduser().resolve()
        if args.out
        else base_dir / "fft_direct_output_count_by_rate.csv"
    )

    if not input_root.exists():
        raise SystemExit(f"Input dataset not found: {input_root}")

    expected_rel_paths = get_expected_relative_paths(input_root)

    if not expected_rel_paths:
        raise SystemExit(f"No .bin files found in: {input_root}")

    print("AxCept-Bench FFT output classifier")
    print("=" * 72)
    print(f"Base directory     : {base_dir}")
    print(f"Reference dataset  : {input_root}")
    print(f"Files per rate     : {len(expected_rel_paths)}")
    print(f"Rates to evaluate  : {', '.join(args.rates)}")
    print(f"Output CSV         : {out_csv}")
    print("=" * 72)
    print()

    rows = []
    skipped_rates = []

    for rate in args.rates:
        try:
            row = analyze_rate(base_dir, rate, expected_rel_paths)
        except FileNotFoundError as e:
            print(f"[SKIPPED] {e}")
            print("-" * 72)
            skipped_rates.append(rate)
            continue

        rows.append(row)
        print_rate_summary(row)

    if not rows:
        raise SystemExit("No rate could be evaluated. Check --base-dir and folder names.")

    write_csv(rows, out_csv)

    total_files_evaluated = sum(r["classified_total"] for r in rows)

    print()
    print("Done.")
    print(
        f"Rates evaluated    : {len(rows)}/{len(args.rates)}"
        + (f" (skipped: {', '.join(skipped_rates)})" if skipped_rates else "")
    )
    print(f"Total files scanned: {total_files_evaluated}")
    print(f"Summary CSV        : {out_csv}")


if __name__ == "__main__":
    main()