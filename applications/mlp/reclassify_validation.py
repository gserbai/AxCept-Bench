#!/usr/bin/env python3
######################################################################
# AxCept-Bench (GENANN) -- re-classify validation runs that were already
# executed, using the CURRENT parser in
# train_exact_and_validate/run_validate_axram.py, without running AxPike
# again.
#
# Why: an older parser discarded samples whose kernel wrote a complete
# AXGENN01 record and only then crashed (AxPike appends its crash dump
# after the record). The current parser keeps those samples
# (class OK_AFTER_CRASH), the same rule JPEG and FFT already follow:
# bytes written before a crash are still evaluated. Re-running AxPike
# would inject new random faults and overwrite the energy traces, so this
# script re-reads the raw records that were saved instead.
#
# What it rewrites (after backing up the originals with a
# .bak_before_reclassify suffix):
#   digits_summary_err<rate>_<tag>.csv   one per validated (model, rate)
#   validate_matrix_results.csv          rebuilt, one row per (model, rate)
#
# Author: Guilherme Saides Serbai 
# Year: 2026
#
# Run from the project root:
#   python3 reclassify_validation.py
######################################################################

import csv
import importlib.util
import shutil
import struct
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

VALIDATE_SCRIPT = Path("train_exact_and_validate/run_validate_axram.py")
TEST_BIN = Path("dataset_digits/test.bin")
MATRIX_CSV = Path("validate_matrix_results.csv")
RECORD_PREFIX = "digits_records_err"
BACKUP_SUFFIX = ".bak_before_reclassify"


def load_validate_module():
    spec = importlib.util.spec_from_file_location("run_validate_axram", VALIDATE_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_labels(path: Path) -> list:
    d = path.read_bytes()
    (n,) = struct.unpack("<i", d[:4])
    off, labels = 4, []
    for _ in range(n):
        off += 64 * 8
        (y,) = struct.unpack("<i", d[off:off + 4])
        off += 4
        labels.append(y)
    return labels


def backup(path: Path) -> None:
    if path.exists():
        bak = path.with_name(path.name + BACKUP_SUFFIX)
        if not bak.exists():  # never overwrite the first, original backup
            shutil.copyfile(path, bak)


def split_dir_name(name: str):
    """digits_records_err1e-2_model_trained_axram_err0 -> ("1e-2", "model_trained_axram_err0")"""
    rest = name[len(RECORD_PREFIX):]
    rate, sep, tag = rest.partition("_model_")
    if not sep:
        return None
    return rate, "model_" + tag


def main():
    if not VALIDATE_SCRIPT.exists() or not TEST_BIN.exists():
        print("ERRO: rode a partir da raiz do projeto (precisa de "
              f"{VALIDATE_SCRIPT} e {TEST_BIN}).")
        sys.exit(1)

    mod = load_validate_module()
    labels = load_labels(TEST_BIN)

    old_matrix = {}
    if MATRIX_CSV.exists():
        with open(MATRIX_CSV, newline="") as f:
            for r in csv.DictReader(f):
                old_matrix[(r["tag"], r["error_rate"])] = r

    new_rows = []
    for rec_dir in sorted(Path(".").glob(RECORD_PREFIX + "*_model_trained_axram_err*")):
        parsed = split_dir_name(rec_dir.name)
        if not parsed or not rec_dir.is_dir():
            continue
        rate, tag = parsed
        summary_csv = Path(f"digits_summary_err{rate}_{tag}.csv")

        old_elapsed = {}
        if summary_csv.exists():
            with open(summary_csv, newline="") as f:
                for r in csv.DictReader(f):
                    old_elapsed[r["index"]] = r.get("elapsed_sec", "")

        rows = []
        for i, true_label in enumerate(labels):
            rec = rec_dir / f"sample_{i:04d}.bin"
            data = rec.read_bytes() if rec.exists() else b""
            info = mod.parse_genn_record(data)
            row = {"index": i, "true_label": true_label,
                   "predicted": info.get("predicted", ""),
                   "correct": "", "class": info["class"],
                   "recoverable": info["recoverable"],
                   "elapsed_sec": old_elapsed.get(str(i), "")}
            if info["recoverable"]:
                row["correct"] = info["predicted"] == true_label
            rows.append(row)

        backup(summary_csv)
        fields = ["index", "true_label", "predicted", "correct", "class",
                  "recoverable", "elapsed_sec"]
        with open(summary_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

        total = len(rows)
        recoverable = [r for r in rows if r["recoverable"]]
        correct = sum(1 for r in recoverable if r["correct"])
        after_crash = sum(1 for r in rows if r["class"] == "OK_AFTER_CRASH")
        old = old_matrix.get((tag, rate), {})
        model_file = old.get("model_file") or f"{tag}.bin"
        train_mode = old.get("train_mode") or ("exact" if tag.endswith("_err0") else "axram")
        new_rows.append({
            "timestamp": old.get("timestamp") or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "line": old.get("line") or (2 if train_mode == "exact" else 4),
            "train_mode": train_mode,
            "tag": tag,
            "validate_mode": "axram",
            "error_rate": rate,
            "hidden_units": old.get("hidden_units", ""),
            "hidden_layers": old.get("hidden_layers", ""),
            "model_file": model_file,
            "accuracy_all_pct": f"{100 * correct / total:.4f}" if total else "",
            "accuracy_recoverable_pct": f"{100 * correct / len(recoverable):.4f}" if recoverable else "",
            "n_recoverable": len(recoverable),
            "n_total": total,
            "summary_csv": str(summary_csv),
            "n_recovered_after_crash": after_crash,
        })

        classes = dict(Counter(r["class"] for r in rows))
        old_acc = old.get("accuracy_all_pct", "?")
        print(f"{tag} @ {rate}: {classes} | accuracy {old_acc} -> "
              f"{new_rows[-1]['accuracy_all_pct']}")

    if not new_rows:
        print("Nenhuma pasta digits_records_err*_model_trained_axram_err* encontrada.")
        return

    backup(MATRIX_CSV)
    with open(MATRIX_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=mod.RESULTS_CSV_FIELDS)
        w.writeheader()
        w.writerows(new_rows)
    print(f"\n{MATRIX_CSV} reescrito ({len(new_rows)} linhas). "
          f"Originais salvos com sufixo {BACKUP_SUFFIX}.")


if __name__ == "__main__":
    main()
