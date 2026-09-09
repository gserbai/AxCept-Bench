#!/usr/bin/env python3
"""
Analyze AxCept-Bench FFT outputs directly from AXDFREQ1 binary files.

The exact output tree defines the expected samples. For every error rate, an
approximate output is paired with the exact output only when both have the same
full relative path, for example ``piano/note_a4.bin``. Matching only by basename
is intentionally forbidden because equal filenames may exist in different
classes.

Expected default layout::

    <base-dir>/dataset_iowa_music_exact/
        src/dataset_audio_error_rate_exact/<class>/<audio>.bin

    <base-dir>/dataset_iowa_music_<rate>/
        src/dataset_audio_error_rate_<rate>/<class>/<audio>.bin

The exact tree may alternatively use ``dataset_iowa_music_0`` and
``dataset_audio_error_rate_0``. A custom exact output directory can be supplied
with ``--exact-dir``.

Each binary record starts with 24 little-endian bytes::

    [8-byte magic] [int32 status] [int32 num_samples]
    [int32 num_frames] [float32 freq_hz]

Input CSV summaries are not required. CSV files produced by this script are
analysis outputs only.
"""

from __future__ import annotations

import argparse
import csv
import math
import struct
from collections import Counter
from pathlib import Path
from statistics import mean, median


RATES_DEFAULT = ["1e-5", "1e-4", "1e-3", "1e-2", "1e-1"]

CRASH_SIGNATURES = [
    b"z  ",
    b"User load segfault",
    b"User store segfault",
    b"User fetch segfault",
    b"segfault",
]

AXDFREQ_HEADER = b"AXDFREQ1"
AXDFREQ_STRUCT = struct.Struct("<8siiif")
AXDFREQ_HEADER_SIZE = AXDFREQ_STRUCT.size

USABLE_CLASSES = {
    "OK_RESULT",
    "RESULT_WITH_CRASH_DUMP",
}

PAIR_FIELDNAMES = [
    "rate",
    "relative_path",
    "exact_path",
    "approx_path",
    "exact_class",
    "approx_class",
    "exact_status",
    "approx_status",
    "exact_num_samples",
    "approx_num_samples",
    "exact_num_frames",
    "approx_num_frames",
    "freq_exact",
    "freq_approx",
    "abs_error_hz",
    "mape",
    "mape_percent",
    "midi_exact",
    "midi_approx",
    "is_acceptable",
    "compared",
    "skip_reason",
]

UNMATCHED_FIELDNAMES = [
    "rate",
    "relative_path",
    "issue",
    "path",
]


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compara diretamente os .bin exatos e aproximados do workload FFT, "
            "calcula MAPE/MIDI e gera CSVs e gráficos."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemplo:\n"
            "  python3 analyze_fft_acceptability_two_plots_one_figure.py \\\n"
            "    --base-dir /caminho/para/Music \\\n"
            "    --rates 1e-5 1e-4 1e-3 1e-2 1e-1 \\\n"
            "    --out-dir /caminho/para/fft_acceptability\n"
        ),
    )

    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path.home() / "Music",
        help="Pasta que contém dataset_iowa_music_exact e as pastas de cada taxa.",
    )
    parser.add_argument(
        "--exact-dir",
        type=Path,
        default=None,
        help=(
            "Pasta que contém diretamente os .bin exatos organizados como "
            "<classe>/<audio>.bin. Se omitida, o script descobre a pasta em --base-dir."
        ),
    )
    parser.add_argument(
        "--rates",
        nargs="+",
        default=RATES_DEFAULT,
        help=f"Taxas analisadas. Padrão: {' '.join(RATES_DEFAULT)}",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("fft_acceptability_analysis"),
        help="Pasta de saída dos CSVs e gráficos.",
    )
    parser.add_argument(
        "--exact-summary",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )

    args = parser.parse_args()
    if args.exact_summary is not None:
        parser.error(
            "--exact-summary não é mais usado: os .bin são lidos diretamente. "
            "Use --exact-dir para informar a pasta dos outputs exatos."
        )

    parsed_rates = []
    seen_rates = set()
    for rate in args.rates:
        try:
            numeric_rate = float(rate)
        except ValueError:
            parser.error(f"taxa inválida: {rate}")
        if not math.isfinite(numeric_rate) or numeric_rate <= 0.0:
            parser.error(f"a taxa deve ser finita e maior que zero: {rate}")
        if numeric_rate in seen_rates:
            parser.error(f"taxa duplicada: {rate}")
        seen_rates.add(numeric_rate)
        parsed_rates.append((numeric_rate, rate))

    args.rates = [rate for _, rate in sorted(parsed_rates)]
    return args


# =============================================================================
# Binary parsing and classification
# =============================================================================

def has_crash_dump(data: bytes) -> bool:
    """Return True when a known AxPike crash signature occurs in the output."""
    return any(signature in data for signature in CRASH_SIGNATURES)


def empty_record(output_class: str, byte_count: int = 0) -> dict:
    return {
        "class": output_class,
        "status": None,
        "num_samples": None,
        "num_frames": None,
        "freq_hz": None,
        "crash": False,
        "bytes": byte_count,
    }


def parse_output_bytes(data: bytes) -> dict:
    """Parse and classify one saved FFT output using the benchmark wire format."""
    if not data:
        return empty_record("TIMEOUT_OR_EMPTY")

    crash_dump = has_crash_dump(data)

    if data.startswith(AXDFREQ_HEADER) and len(data) >= AXDFREQ_HEADER_SIZE:
        try:
            magic, status, num_samples, num_frames, freq_hz = AXDFREQ_STRUCT.unpack(
                data[:AXDFREQ_HEADER_SIZE]
            )
        except struct.error:
            return empty_record("INVALID_OUTPUT", len(data))

        record = {
            "class": "",
            "status": status,
            "num_samples": num_samples,
            "num_frames": num_frames,
            "freq_hz": float(freq_hz),
            "crash": crash_dump,
            "bytes": len(data),
        }

        if magic != AXDFREQ_HEADER:
            record["class"] = "INVALID_OUTPUT"
        elif status != 0:
            record["class"] = "APPLICATION_ERROR"
        elif not valid_frequency(record["freq_hz"]):
            record["class"] = "INVALID_NUMERIC"
        elif crash_dump:
            record["class"] = "RESULT_WITH_CRASH_DUMP"
        else:
            record["class"] = "OK_RESULT"

        return record

    if crash_dump:
        record = empty_record("AXPIKE_CRASH_DUMP", len(data))
        record["crash"] = True
        return record

    return empty_record("INVALID_OUTPUT", len(data))


def read_output(path: Path) -> dict:
    try:
        return parse_output_bytes(path.read_bytes())
    except OSError as exc:
        record = empty_record("READ_ERROR")
        record["read_error"] = str(exc)
        return record


def missing_output_record() -> dict:
    return empty_record("MISSING_OUTPUT")


def valid_frequency(freq: float | None) -> bool:
    return freq is not None and math.isfinite(freq) and 0.0 < freq <= 22050.0


def is_usable(record: dict) -> bool:
    return record["class"] in USABLE_CLASSES and valid_frequency(record["freq_hz"])


def hz_to_midi(freq: float | None) -> int | None:
    if not valid_frequency(freq):
        return None
    midi_float = 69.0 + 12.0 * math.log2(freq / 440.0)
    return int(math.floor(midi_float + 0.5))


# =============================================================================
# Directory discovery and exact/approximate pairing
# =============================================================================

def discover_exact_root(base_dir: Path, exact_dir: Path | None) -> Path:
    if exact_dir is not None:
        root = exact_dir.expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Pasta exata não encontrada: {root}")
        return root

    candidates = [
        base_dir
        / "dataset_iowa_music_exact"
        / "src"
        / "dataset_audio_error_rate_exact",
        base_dir
        / "dataset_iowa_music_exact"
        / "src"
        / "dataset_audio_error_rate_0",
        base_dir
        / "dataset_iowa_music_0"
        / "src"
        / "dataset_audio_error_rate_0",
        base_dir
        / "dataset_iowa_music_0"
        / "src"
        / "dataset_audio_error_rate_exact",
    ]

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    attempted = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "Pasta de outputs exatos não encontrada. Caminhos testados:\n"
        f"  {attempted}\n"
        "Use --exact-dir se o baseline estiver em outro local."
    )


def output_root_for_rate(base_dir: Path, rate: str) -> Path:
    return (
        base_dir
        / f"dataset_iowa_music_{rate}"
        / "src"
        / f"dataset_audio_error_rate_{rate}"
    )


def relative_bin_paths(root: Path) -> list[Path]:
    """Return full relative paths, such as class/audio.bin."""
    return sorted(path.relative_to(root) for path in root.rglob("*.bin") if path.is_file())


def format_optional(value: object) -> object:
    return "" if value is None else value


def pct(part: int | float, total: int | float) -> float:
    return 100.0 * part / total if total else 0.0


def skip_reason_for_pair(exact: dict, approx: dict) -> str:
    if approx["class"] == "MISSING_OUTPUT":
        return "MISSING_OUTPUT"
    if not is_usable(exact):
        return f"EXACT_{exact['class']}"
    if not is_usable(approx):
        return f"APPROX_{approx['class']}"
    return ""


def build_pair_row(
    rate: str,
    relative_path: Path,
    exact_path: Path,
    approx_path: Path,
    exact: dict,
    approx: dict,
) -> dict:
    row = {
        "rate": rate,
        "relative_path": relative_path.as_posix(),
        "exact_path": str(exact_path),
        "approx_path": str(approx_path),
        "exact_class": exact["class"],
        "approx_class": approx["class"],
        "exact_status": format_optional(exact["status"]),
        "approx_status": format_optional(approx["status"]),
        "exact_num_samples": format_optional(exact["num_samples"]),
        "approx_num_samples": format_optional(approx["num_samples"]),
        "exact_num_frames": format_optional(exact["num_frames"]),
        "approx_num_frames": format_optional(approx["num_frames"]),
        "freq_exact": format_optional(exact["freq_hz"]),
        "freq_approx": format_optional(approx["freq_hz"]),
        "abs_error_hz": "",
        "mape": "",
        "mape_percent": "",
        "midi_exact": "",
        "midi_approx": "",
        "is_acceptable": "",
        "compared": False,
        "skip_reason": skip_reason_for_pair(exact, approx),
    }

    if row["skip_reason"]:
        return row

    freq_exact = exact["freq_hz"]
    freq_approx = approx["freq_hz"]
    midi_exact = hz_to_midi(freq_exact)
    midi_approx = hz_to_midi(freq_approx)

    abs_error_hz = abs(freq_approx - freq_exact)
    mape = abs_error_hz / freq_exact

    row.update({
        "abs_error_hz": abs_error_hz,
        "mape": mape,
        "mape_percent": mape * 100.0,
        "midi_exact": midi_exact,
        "midi_approx": midi_approx,
        "is_acceptable": midi_exact == midi_approx,
        "compared": True,
    })
    return row


# =============================================================================
# Per-rate analysis
# =============================================================================

def analyze_one_rate(
    rate: str,
    exact_root: Path,
    approx_root: Path,
    exact_paths: list[Path],
    exact_records: dict[Path, dict],
) -> tuple[dict, list[dict], list[dict]]:
    if not approx_root.is_dir():
        raise FileNotFoundError(f"Pasta de outputs não encontrada para a taxa {rate}: {approx_root}")

    expected = set(exact_paths)
    approx_paths = set(relative_bin_paths(approx_root))
    extra_paths = sorted(approx_paths - expected)

    class_counts: Counter = Counter()
    pair_rows: list[dict] = []
    unmatched_rows: list[dict] = []

    for relative_path in exact_paths:
        exact_path = exact_root / relative_path
        approx_path = approx_root / relative_path
        exact = exact_records[relative_path]

        if approx_path.is_file():
            approx = read_output(approx_path)
        else:
            approx = missing_output_record()
            unmatched_rows.append({
                "rate": rate,
                "relative_path": relative_path.as_posix(),
                "issue": "MISSING_OUTPUT",
                "path": str(approx_path),
            })

        class_counts[approx["class"]] += 1
        pair_rows.append(
            build_pair_row(
                rate,
                relative_path,
                exact_path,
                approx_path,
                exact,
                approx,
            )
        )

    for relative_path in extra_paths:
        unmatched_rows.append({
            "rate": rate,
            "relative_path": relative_path.as_posix(),
            "issue": "EXTRA_OUTPUT",
            "path": str(approx_root / relative_path),
        })

    compared_rows = [row for row in pair_rows if row["compared"]]
    usable_count = sum(
        class_counts[name]
        for name in USABLE_CLASSES
    )
    exact_usable_count = sum(is_usable(exact_records[path]) for path in exact_paths)
    acceptable_count = sum(row["is_acceptable"] for row in compared_rows)
    mape_values = [row["mape"] for row in compared_rows]
    abs_values = [row["abs_error_hz"] for row in compared_rows]

    total = len(exact_paths)
    compared_count = len(compared_rows)
    mape_mean = mean(mape_values) if mape_values else math.nan
    mape_median = median(mape_values) if mape_values else math.nan

    metrics = {
        "rate": rate,
        "exact_root": str(exact_root),
        "approx_root": str(approx_root),
        "total_expected_from_exact": total,
        "exact_usable": exact_usable_count,
        "ok_result": class_counts["OK_RESULT"],
        "result_with_crash_dump": class_counts["RESULT_WITH_CRASH_DUMP"],
        "axpike_crash_dump": class_counts["AXPIKE_CRASH_DUMP"],
        "timeout_or_empty": class_counts["TIMEOUT_OR_EMPTY"],
        "application_error": class_counts["APPLICATION_ERROR"],
        "invalid_output": class_counts["INVALID_OUTPUT"],
        "invalid_numeric": class_counts["INVALID_NUMERIC"],
        "read_error": class_counts["READ_ERROR"],
        "missing_output": class_counts["MISSING_OUTPUT"],
        "extra_output": len(extra_paths),
        "usable_frequency": usable_count,
        "excluded_from_quality_calc": total - compared_count,
        "pairs_compared_vs_exact": compared_count,
        "acceptable_midi": acceptable_count,
        "not_acceptable_midi": compared_count - acceptable_count,
        "usable_frequency_rate_total": pct(usable_count, total),
        "midi_match_rate_compared": pct(acceptable_count, compared_count),
        "midi_match_rate_total": pct(acceptable_count, total),
        "mape_mean": mape_mean,
        "mape_median": mape_median,
        "mape_mean_percent": mape_mean * 100.0 if math.isfinite(mape_mean) else math.nan,
        "mape_median_percent": mape_median * 100.0 if math.isfinite(mape_median) else math.nan,
        "abs_error_mean_hz": mean(abs_values) if abs_values else math.nan,
        "abs_error_median_hz": median(abs_values) if abs_values else math.nan,
        "skipped_bad_exact": sum(
            1 for row in pair_rows if str(row["skip_reason"]).startswith("EXACT_")
        ),
        "skipped_bad_approx": sum(
            1 for row in pair_rows if str(row["skip_reason"]).startswith("APPROX_")
        ),
    }

    return metrics, pair_rows, unmatched_rows


# =============================================================================
# Plotting and CSV output
# =============================================================================

def setup_plot_style(mpl) -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 10,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 8.8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
    })


def finish_log_x_axis(ax, x_values, x_labels, fixed_locator, fixed_formatter, null_locator) -> None:
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(fixed_locator(x_values))
    ax.xaxis.set_major_formatter(fixed_formatter(x_labels))
    ax.xaxis.set_minor_locator(null_locator())
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.5, alpha=0.3)


def plot_two_panels_one_figure(metrics_rows: list[dict], out_dir: Path) -> None:
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Dependências dos gráficos não instaladas. Execute: "
            "python3 -m pip install numpy matplotlib"
        ) from exc

    setup_plot_style(mpl)

    rates = [row["rate"] for row in metrics_rows]
    x = np.array([float(rate) for rate in rates], dtype=float)
    mean_mape = np.array([row["mape_mean"] for row in metrics_rows], dtype=float)
    usable = np.array(
        [row["usable_frequency_rate_total"] / 100.0 for row in metrics_rows],
        dtype=float,
    )
    midi_total = np.array(
        [row["midi_match_rate_total"] / 100.0 for row in metrics_rows],
        dtype=float,
    )

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(3.8, 4.8),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.16},
    )

    colors = plt.get_cmap("tab10").colors
    ax1.plot(
        x,
        mean_mape,
        label="Mean MAPE",
        color=colors[0],
        linestyle="-",
        marker="o",
        markersize=5.5,
        markeredgewidth=0.8,
    )
    finish_log_x_axis(ax1, x, rates, FixedLocator, FixedFormatter, NullLocator)
    ax1.set_ylabel("Mean MAPE")
    ax1.legend(loc="best", frameon=False, handlelength=2.4)

    finite_mape = mean_mape[np.isfinite(mean_mape)]
    max_mape = float(np.max(finite_mape)) if finite_mape.size else 1.0
    if max_mape <= 0:
        max_mape = 1.0
    ax1.set_ylim(0.0, max_mape * 1.10)

    ax2.plot(
        x,
        usable,
        label="Usable Frequencies",
        color=colors[1],
        linestyle="--",
        marker="s",
        markersize=5.5,
        markeredgewidth=0.8,
    )
    ax2.plot(
        x,
        midi_total,
        label="MIDI Match",
        color=colors[2],
        linestyle="-.",
        marker="^",
        markersize=5.5,
        markeredgewidth=0.8,
    )
    finish_log_x_axis(ax2, x, rates, FixedLocator, FixedFormatter, NullLocator)
    ax2.set_xlabel("Error Rate")
    ax2.set_ylabel("Metric Value")
    ax2.set_ylim(0.0, 1.05)
    ax2.set_yticks(np.linspace(0.0, 1.0, 6))
    ax2.legend(loc="best", frameon=False, handlelength=2.4)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / "fft_two_plots_one_figure_publication.pdf"
    fig.tight_layout(pad=0.35)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        if not rows:
            return
        fieldnames = list(rows[0].keys())

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()

    try:
        exact_root = discover_exact_root(base_dir, args.exact_dir)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    exact_paths = relative_bin_paths(exact_root)
    if not exact_paths:
        raise SystemExit(f"Nenhum .bin exato encontrado em: {exact_root}")

    exact_records = {
        relative_path: read_output(exact_root / relative_path)
        for relative_path in exact_paths
    }
    exact_class_counts = Counter(record["class"] for record in exact_records.values())

    print("AxCept-Bench FFT acceptability analysis")
    print("=" * 72)
    print(f"Base directory       : {base_dir}")
    print(f"Exact output root    : {exact_root}")
    print(f"Exact files          : {len(exact_paths)}")
    print(f"Exact usable files   : {sum(is_usable(record) for record in exact_records.values())}")
    print(f"Exact classes        : {dict(sorted(exact_class_counts.items()))}")
    print(f"Rates                : {', '.join(args.rates)}")
    print(f"Output directory     : {out_dir}")
    print("Pairing key          : full relative path (<class>/<audio>.bin)")
    print("=" * 72)

    all_metrics: list[dict] = []
    all_pairs: list[dict] = []
    all_unmatched: list[dict] = []
    skipped_rates: list[str] = []

    for rate in args.rates:
        approx_root = output_root_for_rate(base_dir, rate)
        print(f"\nRate {rate}")
        print(f"  Approx output root : {approx_root}")

        try:
            metrics, pairs, unmatched = analyze_one_rate(
                rate,
                exact_root,
                approx_root,
                exact_paths,
                exact_records,
            )
        except FileNotFoundError as exc:
            print(f"  [SKIP] {exc}")
            skipped_rates.append(rate)
            continue

        all_metrics.append(metrics)
        all_pairs.extend(pairs)
        all_unmatched.extend(unmatched)

        print(
            f"  Compared           : {metrics['pairs_compared_vs_exact']} / "
            f"{metrics['total_expected_from_exact']}"
        )
        print(
            f"  Usable approximate : {metrics['usable_frequency']} "
            f"({metrics['usable_frequency_rate_total']:.2f}%)"
        )
        print(f"  Missing outputs    : {metrics['missing_output']}")
        print(f"  Extra outputs      : {metrics['extra_output']}")
        print(f"  MIDI match / total : {metrics['midi_match_rate_total']:.2f}%")
        print(f"  MIDI match / pairs : {metrics['midi_match_rate_compared']:.2f}%")
        print(
            f"  Mean MAPE          : {metrics['mape_mean']:.6f} "
            f"({metrics['mape_mean_percent']:.4f}%)"
        )

    if not all_metrics:
        raise SystemExit("Nenhuma taxa foi analisada. Verifique as pastas de outputs.")

    write_csv(out_dir / "fft_metrics_by_rate.csv", all_metrics)
    write_csv(
        out_dir / "fft_all_comparisons_vs_exact.csv",
        all_pairs,
        PAIR_FIELDNAMES,
    )
    write_csv(
        out_dir / "fft_unmatched_files.csv",
        all_unmatched,
        UNMATCHED_FIELDNAMES,
    )
    plot_two_panels_one_figure(all_metrics, out_dir)

    print("\n" + "=" * 72)
    print("FILES GENERATED")
    print("=" * 72)
    print(f"Metrics by rate      : {out_dir / 'fft_metrics_by_rate.csv'}")
    print(f"Pair audit           : {out_dir / 'fft_all_comparisons_vs_exact.csv'}")
    print(f"Unmatched files      : {out_dir / 'fft_unmatched_files.csv'}")
    print(f"Figure PDF           : {out_dir / 'fft_two_plots_one_figure_publication.pdf'}")
    print(f"Figure PNG           : {out_dir / 'fft_two_plots_one_figure_publication.png'}")
    if skipped_rates:
        print(f"Skipped rates        : {', '.join(skipped_rates)}")


if __name__ == "__main__":
    main()
