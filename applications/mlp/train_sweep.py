#!/usr/bin/env python3
######################################################################
# AxCept-Bench (GENANN) -- training under AxRAM, training only, keeping
# the final accuracy + logs of each run.
# Author: Guilherme Saides Serbai 
# Year: 2026
#
# Stage 1 of the experiment -- training ONLY, no validation under AxRAM yet
# (that comes later, as a separate stage). Runs `train_axram/run_train_axram.py`
# once per error rate, and for each run:
#
#   1. trains under AxRAM (one long axpike invocation, all EPOCHS)
#   2. if it succeeded, evaluates the resulting model with eval_exact_host
#      (host, exact) to get the final accuracy of that model
#   3. records everything (accuracy, time, status, paths of the logs/energy
#      artifacts) in one CSV row -- never overwrites
#
# Usage:
#
#   python3 train_sweep.py
#   python3 train_sweep.py --force              # re-runs even if an OK row already exists
#   python3 train_sweep.py --error-rates 1e-4,1e-2
######################################################################

import argparse
import csv
import math
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ===============================
# Configuration -- EDIT HERE
# ===============================

# "0" was added (2026-09) as the energy baseline -- it runs through the SAME
# axpike/train_kernel_axpike (mem_read_prob:0, no bit-flip injected at
# all), to have an energy trace comparable with the real rates.
# train_exact_host (pure host, no axpike) does NOT work for this -- it does
# not go through the simulator, so it produces no AxPIKE_energy_*.csv.
ERROR_RATES = ["0", "1e-5", "1e-4", "1e-3", "1e-2", "1e-1"]

N_TRAIN_SAMPLES = 1437  # full dataset
EPOCHS = 10             # fixed at 10 -- 50 epochs over 5 rates in a row made the
                        # full sweep impractical. See Makefile: the NaN assert is
                        # disabled (-DNDEBUG) only on the riscv side, so training
                        # no longer aborts on the first NaN caused by an AxRAM bit-flip.

INIT_WEIGHTS_BIN = Path("./init_weights.bin")
EVAL_BIN = "./eval_exact_host"   # string, not Path -- Path("./eval_exact_host") strips
                                   # the "./" and subprocess.run() then searches the
                                   # $PATH instead of the cwd, breaking the call.

RESULTS_CSV = Path("./train_results.csv")
RESULTS_FIELDS = [
    "timestamp", "error_rate", "status", "elapsed_sec",
    "model_file", "accuracy_exact_pct",
    "weights_total", "weights_nan_or_inf", "nan_fraction_pct",
    "train_log", "train_stdout_raw", "axpike_artifacts_dir",
]

PYTHON = sys.executable
TRAIN_SCRIPT = Path("train_axram/run_train_axram.py")
TRAIN_RUNS_CSV = Path("./genann_train_axram_runs.csv")  # written by run_train_axram.py itself

# Statuses worth automatically retrying ONCE: these are cases where the
# guest process exited CLEANLY (returncode 0, no crash signature, no
# timeout) but the captured output was still short/unparseable. Observed
# empirically (2026-09): a PARTIAL_OUTPUT with returncode=0, missing
# exactly 1024 bytes of an otherwise-complete record, on a run that took
# ~14 minutes -- doesn't look like AxRAM noise (that shows up as NaN
# weights or a real crash, not a clean exit with truncated stdout), looks
# more like a transient axpike/pk quirk on long runs. Genuine crashes
# (CRASHED, negative returncode = killed by signal, AX_STATUS_N) are NOT
# retried here -- those are treated as legitimate experimental outcomes,
# not something to paper over by re-rolling until it passes.
RETRIABLE_STATUSES = {"PARTIAL_OUTPUT", "MISSING_OUTPUT", "SHORT_HEADER", "INVALID_OUTPUT"}
MAX_TRAIN_ATTEMPTS = 2


def read_last_train_run(error_rate: str) -> dict | None:
    """Last row of TRAIN_RUNS_CSV matching this error_rate, or None."""
    if not TRAIN_RUNS_CSV.exists():
        return None
    last = None
    with open(TRAIN_RUNS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("error_rate") == error_rate:
                last = row
    return last


# ===============================
# Resume support
# ===============================

def already_done(error_rate: str) -> bool:
    if not RESULTS_CSV.exists():
        return False
    with open(RESULTS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row["error_rate"] == error_rate and row.get("status") == "OK":
                return True
    return False


def append_result(row: dict) -> None:
    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in RESULTS_FIELDS})


# ===============================
# NaN/Inf check -- model.bin format is 16 bytes of header (4x int32 dims:
# inputs, hidden_layers, hidden, outputs) followed by total_weights x
# float64 (8 bytes each), same layout genann_train_axram_err<rate>.py's
# parse_train_record() writes out. Reading it back and counting NaN/Inf is
# pure file parsing -- no need to run the network at all.
# ===============================

def count_nan_inf(model_file: Path) -> dict:
    data = model_file.read_bytes()
    inputs, hidden_layers, hidden, outputs = struct.unpack("<iiii", data[:16])
    weights_bytes = data[16:]
    n_weights = len(weights_bytes) // 8
    weights = struct.unpack(f"<{n_weights}d", weights_bytes[:n_weights * 8])
    bad = sum(1 for w in weights if math.isnan(w) or math.isinf(w))
    return {
        "dims": (inputs, hidden_layers, hidden, outputs),
        "total": n_weights,
        "bad": bad,
        "fraction_pct": (100.0 * bad / n_weights) if n_weights else 0.0,
    }


# ===============================
# One error rate
# ===============================

def run_one(error_rate: str, force: bool) -> None:
    if not force and already_done(error_rate):
        print(f"[{error_rate}] ja tem linha OK em {RESULTS_CSV} -- pulando (use --force pra re-rodar)")
        return

    model_file = Path(f"./model_trained_axram_err{error_rate}.bin")
    log_file = Path(f"./genann_train_axram_err{error_rate}.log")
    stdout_raw = Path(f"./genann_train_axram_err{error_rate}.stdout.raw")
    artifacts_dir = Path(f"./axpike_artifacts_train_err{error_rate}")

    total_elapsed = 0.0
    for attempt in range(1, MAX_TRAIN_ATTEMPTS + 1):
        print(f"\n=== treinando sob AxRAM @ {error_rate} "
              f"(N_TRAIN_SAMPLES={N_TRAIN_SAMPLES}, EPOCHS={EPOCHS}, "
              f"tentativa {attempt}/{MAX_TRAIN_ATTEMPTS}) ===")
        start = time.monotonic()
        proc = subprocess.run([
            PYTHON, str(TRAIN_SCRIPT),
            "--error-rate", error_rate,
            "--n-train-samples", str(N_TRAIN_SAMPLES),
            "--epochs", str(EPOCHS),
            "--init-weights", str(INIT_WEIGHTS_BIN),
            "--output", str(model_file),
        ])
        elapsed = time.monotonic() - start
        total_elapsed += elapsed

        if proc.returncode == 0:
            break  # success -- leave the attempts loop

        last_run = read_last_train_run(error_rate)
        status = last_run.get("status") if last_run else None
        crashed = (last_run or {}).get("crashed") == "True"
        # Str compare on purpose -- CSV values are text ("0", "-9", ""),
        # and we only want to retry a CLEAN exit (returncode "0") that
        # still came back short -- a negative returncode (killed by
        # signal) or a detected crash signature is a real outcome, not a
        # fluke worth re-rolling.
        returncode_str = (last_run or {}).get("returncode")
        clean_exit = returncode_str in ("0", "0.0")
        retriable = status in RETRIABLE_STATUSES and not crashed and clean_exit

        if retriable and attempt < MAX_TRAIN_ATTEMPTS:
            print(f"[{error_rate}] treino saiu com status={status}, sem crash, "
                  f"returncode limpo -- parece anomalia transitoria (nao ruido da "
                  f"AxRAM). Tentando de novo (tentativa {attempt + 1}/{MAX_TRAIN_ATTEMPTS})...")
            continue

        print(f"[{error_rate}] treino FALHOU (codigo {proc.returncode}, {elapsed:.0f}s, "
              f"status={status}) -- ver {log_file} / {stdout_raw} / "
              f"genann_train_axram_runs.csv pro motivo exato.")
        append_result({
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "error_rate": error_rate, "status": "CRASHED_OR_FAILED",
            "elapsed_sec": f"{total_elapsed:.2f}",
            "model_file": "", "accuracy_exact_pct": "",
            "train_log": str(log_file), "train_stdout_raw": str(stdout_raw),
            "axpike_artifacts_dir": str(artifacts_dir),
        })
        return

    print(f"[{error_rate}] treino OK em {total_elapsed:.0f}s -> {model_file}")

    # Check NaN/Inf BEFORE calling eval_exact_host -- eval_exact_host is
    # compiled with plain gcc (no -DNDEBUG), so the assert(!isnan(a)) in
    # lib/genann.c is still enabled there and kills the process if any weight
    # turned into NaN during training under AxRAM (see conversation 2026-09).
    # Reading the .bin and counting NaN/Inf first avoids that crash AND gives
    # a quantitative number (fraction of corrupted weights) instead of just "failed".
    nan_info = count_nan_inf(model_file)
    weights_total = nan_info["total"]
    weights_bad = nan_info["bad"]
    nan_fraction = nan_info["fraction_pct"]

    if weights_bad > 0:
        print(f"[{error_rate}] AVISO: {weights_bad}/{weights_total} pesos "
              f"({nan_fraction:.2f}%) sao NaN/Inf -- modelo numericamente "
              f"corrompido, NAO vou chamar {EVAL_BIN} (ia crashar no assert "
              f"do host). Salvando como TRAINED_NAN_WEIGHTS.")
        append_result({
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "error_rate": error_rate, "status": "TRAINED_NAN_WEIGHTS",
            "elapsed_sec": f"{total_elapsed:.2f}",
            "model_file": str(model_file), "accuracy_exact_pct": "",
            "weights_total": weights_total, "weights_nan_or_inf": weights_bad,
            "nan_fraction_pct": f"{nan_fraction:.4f}",
            "train_log": str(log_file), "train_stdout_raw": str(stdout_raw),
            "axpike_artifacts_dir": str(artifacts_dir),
        })
        return

    print(f"[{error_rate}] pesos limpos (0/{weights_total} NaN/Inf) -- "
          f"avaliando acuracia final (host, exato): {EVAL_BIN} {model_file}")
    eval_proc = subprocess.run([EVAL_BIN, str(model_file)], capture_output=True, text=True)
    if eval_proc.returncode != 0:
        print(f"[{error_rate}] ERRO ao avaliar: {EVAL_BIN} saiu com codigo "
              f"{eval_proc.returncode}: {eval_proc.stderr}")
        accuracy = ""
        status = "TRAINED_BUT_EVAL_FAILED"
    else:
        accuracy = eval_proc.stdout.strip().splitlines()[-1]
        status = "OK"
        print(f"[{error_rate}] acuracia final: {accuracy}%")

    append_result({
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "error_rate": error_rate, "status": status,
        "elapsed_sec": f"{total_elapsed:.2f}",
        "model_file": str(model_file), "accuracy_exact_pct": accuracy,
        "weights_total": weights_total, "weights_nan_or_inf": weights_bad,
        "nan_fraction_pct": f"{nan_fraction:.4f}",
        "train_log": str(log_file), "train_stdout_raw": str(stdout_raw),
        "axpike_artifacts_dir": str(artifacts_dir),
    })


# ===============================
# Main
# ===============================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true",
                    help="re-roda mesmo que ja tenha linha OK em train_results.csv")
    p.add_argument("--error-rates", default=None,
                    help="lista separada por virgula, ex: 1e-4,1e-2 (default: todas as 5)")
    return p.parse_args()


def main():
    args = parse_args()
    rates = args.error_rates.split(",") if args.error_rates else ERROR_RATES

    if not INIT_WEIGHTS_BIN.exists():
        print(f"ERRO: {INIT_WEIGHTS_BIN} nao existe -- rode `make init_weights` primeiro.")
        sys.exit(1)

    print(f"=== Sweep de treino sob AxRAM ({len(rates)} taxas) ===")
    print(f"Taxas: {rates}")
    print(f"Resultados em: {RESULTS_CSV}\n")

    for rate in rates:
        run_one(rate, args.force)

    print(f"\nSweep completo (ou retomado) -- ver {RESULTS_CSV} pra todos os pontos.")


if __name__ == "__main__":
    main()
