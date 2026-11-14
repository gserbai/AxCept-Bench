#!/usr/bin/env python3
######################################################################
# AxCept-Bench (GENANN) -- validate EVERY available model against EVERY
# configured error rate under AxPike/AxRAM, so all of them can be compared
# on the same axis (accuracy vs. error rate).
# Author: Guilherme Saides Serbai 
# Year: 2026
#
# Run this from the project root:
#
#   python3 validate_matrix.py
#   python3 validate_matrix.py --force              # re-runs everything even if an OK row already exists
#   python3 validate_matrix.py --error-rates 1e-4,1e-2
#
# What it does: AUTOMATICALLY discovers which models exist in the project
# (no need to list anything by hand) and runs
# train_exact_and_validate/run_validate_axram.py once for each
# (model, error rate) combination, writing everything to ONE single CSV
# (VALIDATE_RESULTS_CSV) so all models can be compared side by side.
#
# Discovered models -- ONLY model_trained_axram_err<rate>.bin (train_sweep.py),
# does NOT use model.bin:
#   model_trained_axram_err0.bin      -- the accurate model (zero error
#                                         rate -- no bit-flip happens at
#                                         this rate, but it really goes
#                                         through axpike), labeled "exact"
#   model_trained_axram_err<rate>.bin -- the others, one per rate that
#                                         finished training with status OK
#                                         (train_sweep.py) -- rates that
#                                         crashed or have not run yet
#                                         simply do not show up
#
# Each (model, validation rate) combination runs the 360 samples of
# test.bin (one axpike invocation per sample) and writes its own
# accuracy to VALIDATE_RESULTS_CSV, along with the usual detail files
# (digits_summary_err<rate>_<tag>.csv, digits_records_err<rate>_<tag>/
# etc. -- see run_validate_axram.py and its new --tag).
######################################################################

import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ===============================
# Configuration -- EDIT HERE
# ===============================

# Same 6 rates used in the rest of the project (train_sweep.py) -- includes "0"
# as the noise-free baseline, useful both for the reference accuracy and
# for a comparable energy trace (see train_sweep.py).
VALIDATE_ERROR_RATES = ["0", "1e-5", "1e-4", "1e-3", "1e-2", "1e-1"]

VALIDATE_SCRIPT = Path("train_exact_and_validate/run_validate_axram.py")
VALIDATE_RESULTS_CSV = Path("./validate_matrix_results.csv")

PYTHON = sys.executable


# ===============================
# Automatic model discovery
# ===============================

def discover_models() -> list[tuple[str, str, Path]]:
    """Returns a list of (tag, train_mode, path), one item per model
    found in the project. tag uniquely identifies the model (used to
    name output files/directories via --tag, see run_validate_axram.py);
    train_mode is just "exact" or "axram", for the row label in the CSV.

    Does NOT use model.bin -- only the model_trained_axram_err<rate>.bin
    files that actually exist (that is, only the rates whose training finished
    with status OK in train_sweep.py). err0 (zero error rate) is the accurate
    model -- labeled "exact" because no bit-flip happens at that
    rate; the other rates (err1e-5, err1e-4, etc.) are "axram"."""
    models = []

    for p in sorted(Path(".").glob("model_trained_axram_err*.bin")):
        train_mode = "exact" if p.name == "model_trained_axram_err0.bin" else "axram"
        models.append((p.stem, train_mode, p))

    return models


# ===============================
# Resume support
# ===============================

def already_done(tag: str, error_rate: str) -> bool:
    if not VALIDATE_RESULTS_CSV.exists():
        return False
    with open(VALIDATE_RESULTS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("tag") == tag and row.get("error_rate") == error_rate:
                return True
    return False


# ===============================
# One (model, rate) combination
# ===============================

def run_one(tag: str, train_mode: str, model_path: Path, error_rate: str, force: bool) -> None:
    if not force and already_done(tag, error_rate):
        print(f"[{tag} @ {error_rate}] ja tem linha em {VALIDATE_RESULTS_CSV} -- pulando "
              f"(use --force pra re-rodar)")
        return

    print(f"\n=== validando '{tag}' ({model_path}) @ error_rate={error_rate} ===")
    start = time.monotonic()
    proc = subprocess.run([
        PYTHON, str(VALIDATE_SCRIPT),
        "--model", str(model_path),
        "--error-rate", error_rate,
        "--train-mode", train_mode,
        "--tag", tag,
        "--results-csv", str(VALIDATE_RESULTS_CSV),
    ])
    elapsed = time.monotonic() - start

    if proc.returncode != 0:
        print(f"[{tag} @ {error_rate}] run_validate_axram.py saiu com codigo "
              f"{proc.returncode} apos {elapsed:.0f}s -- ver output acima.")
    else:
        print(f"[{tag} @ {error_rate}] OK em {elapsed:.0f}s")


# ===============================
# Final summary
# ===============================

def print_final_summary() -> None:
    if not VALIDATE_RESULTS_CSV.exists():
        print("\nNenhum resultado gravado ainda.")
        return

    with open(VALIDATE_RESULTS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    tags = sorted({r["tag"] for r in rows})
    rates = VALIDATE_ERROR_RATES

    print(f"\n=== Resumo: acuracia (%) por modelo x taxa de erro ({VALIDATE_RESULTS_CSV}) ===")
    header = "tag".ljust(28) + "".join(rate.rjust(10) for rate in rates)
    print(header)
    for tag in tags:
        by_rate = {r["error_rate"]: r.get("accuracy_all_pct", "") for r in rows if r["tag"] == tag}
        line = tag.ljust(28)
        for rate in rates:
            val = by_rate.get(rate, "")
            line += (val if val else "-").rjust(10)
        print(line)


# ===============================
# Main
# ===============================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true",
                    help="re-roda mesmo que ja tenha linha gravada")
    p.add_argument("--error-rates", default=None,
                    help="lista separada por virgula, ex: 1e-4,1e-2 (default: todas as 6)")
    return p.parse_args()


def main():
    args = parse_args()
    rates = args.error_rates.split(",") if args.error_rates else VALIDATE_ERROR_RATES

    models = discover_models()
    if not models:
        print("ERRO: nenhum modelo encontrado (model_trained_axram_err*.bin) -- "
              "treine primeiro, por exemplo:\n"
              "  python3 train_sweep.py --error-rates 0   # modelo acurado\n"
              "  python3 train_sweep.py                    # demais taxas")
        sys.exit(1)

    total = len(models) * len(rates)
    print(f"=== Matriz de validacao: {len(models)} modelo(s) x {len(rates)} taxa(s) "
          f"= {total} combinacoes ===")
    for tag, train_mode, path in models:
        print(f"  modelo: {tag:28s} ({train_mode:5s})  {path}")
    print(f"Taxas: {rates}")
    print(f"Resultados em: {VALIDATE_RESULTS_CSV}\n")

    for tag, train_mode, path in models:
        for rate in rates:
            run_one(tag, train_mode, path, rate, args.force)

    print_final_summary()
    print(f"\nMatriz completa (ou retomada) -- ver {VALIDATE_RESULTS_CSV} pra todos os pontos.")


if __name__ == "__main__":
    main()
