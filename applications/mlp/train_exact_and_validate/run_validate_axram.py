#!/usr/bin/env python3
######################################################################
# AxCept-Bench (MLP digits inference under AxRAM)
# Author: Guilherme Saides Serbai 
# Year: 2026
#
# Run this from the project root (it reads/writes paths like ./model_*.bin,
# ./dataset_digits/, ./validate_kernel_axpike relative to the current
# directory, not to this script's own location):
#
#   python3 train_exact_and_validate/run_validate_axram.py
#
# Optional CLI overrides (for sweeping error rates / models without editing
# the constants below by hand):
#
#   python3 train_exact_and_validate/run_validate_axram.py \
#       --error-rate 1e-3 --model model_trained_axram_err0.bin --train-mode exact
#
# When --train-mode is given, this run also appends ONE summary row to
# --results-csv (default ./experiment_results.csv) -- a master, cumulative
# table meant for the final accuracy-vs-error-rate plot. This is separate
# from SUMMARY_CSV below, which keeps the full per-sample detail for a
# single run.
#
# Pipeline per test sample:
#   test.bin sample (pixels + true label, from dataset/export_digits.py)
#     -> stdin payload = 64-byte pk padding + model.bin (dims+weights,
#        SAME for every sample) + this sample's 64 pixels
#     -> validate_kernel_axpike running INSIDE AxPike/AxRAM (approximate --
#        the only step exposed to memory errors: one forward pass)
#     -> AXGENN01 record captured from stdout
#     -> argmax the 10 output values -> predicted digit
#     -> compare against the true label
#
# One validate_kernel_axpike run is a single small forward pass -- expect
# each invocation to take well under a second even under full
# instruction-level simulation, so running the whole test set (hundreds of
# samples) per error rate is tractable.
######################################################################

import argparse
import csv
import math
import re
import struct
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# ===============================
# Experiment configuration
# ===============================
#
# These are the defaults for a plain manual run (`python3
# run_validate_axram.py`, no arguments). Every value below can be
# overridden per-invocation with a CLI flag -- see parse_args() -- so the
# same script works both for a one-off run and for an automated sweep.

ERROR_RATE = "1e-4"

MODEL_BIN = Path("./model_trained_axram_err0.bin")    # dims + weights; reference model from train_sweep.py (rate 0)
TEST_BIN = Path("./dataset_digits/test.bin")           # pixels + true label, from dataset/export_digits.py
APP_BIN = "./validate_kernel_axpike"          # riscv64 cross build -- ADJUST path if needed

# Which model is being validated: "exact" = trained exact/host, "axram" =
# trained under AxRAM. Only used to (a) keep this run's output
# folders/files from colliding with a different model validated at the
# SAME error rate, and (b) label the row appended to RESULTS_CSV. Purely a
# label -- doesn't change how the validation itself runs.
TRAIN_MODE = "exact"

# Optional finer-grained label for output naming, on top of TRAIN_MODE --
# needed when there is more than one model of the same TRAIN_MODE (e.g.
# several models trained under AxRAM at different error rates: TAG lets
# "axram_err1e-5" and "axram_err1e-4" get separate output folders instead
# of colliding under a shared "..._axram" suffix). Defaults to TRAIN_MODE
# when not set explicitly (see --tag below), which keeps single-model runs
# unchanged. Set automatically by validate_matrix.py when validating
# several models across the same set of error rates.
TAG = None

# Master, all-lines results table -- only written to when --train-mode is
# explicitly passed (or left at the "exact" default), never required for a
# quick manual run.
RESULTS_CSV = Path("./experiment_results.csv")

N_INPUTS = 64
N_OUTPUTS = 10


def _tagged(base: str) -> str:
    # "exact" (no TAG set) keeps the original (untagged) file/folder
    # names, so existing single-run workflows and already-collected data
    # aren't disturbed. Any other tag/train-mode gets suffixed, so e.g.
    # validating several AxRAM-trained models at the same error rate
    # doesn't silently overwrite each other's summary CSV/records/logs.
    label = TAG if TAG else TRAIN_MODE
    return base if label == "exact" else f"{base}_{label}"


OUTPUT_RECORD_DIR = Path(f"./{_tagged(f'digits_records_err{ERROR_RATE}')}")
LOG_DIR = Path(f"./{_tagged(f'digits_logs_err{ERROR_RATE}')}")
SUMMARY_CSV = Path(f"./{_tagged(f'digits_summary_err{ERROR_RATE}')}.csv")

# axpike drops its own per-run counter/energy CSVs and AXRAM log straight
# into the current directory (one set per invocation, named with its PID --
# not something we control). With 360 invocations per error rate that
# clutters the project dir fast, so we sweep them into their own folder
# right after each run instead of leaving them loose.
AXPIKE_ARTIFACT_DIR = Path(f"./{_tagged(f'axpike_artifacts_err{ERROR_RATE}')}")
AXPIKE_ARTIFACT_GLOBS = (
    "AxPIKE_counters_*.csv",
    "AxPIKE_energy_*.csv",
    "AXRAM_log_*.log",
)

ADD_PK_STDIN_PADDING = True
# 64 is correct for this pk build/invocation, same as every other
# AxCept-Bench kernel (see validate_kernel_axpike.cpp).
PK_STDIN_PADDING_BYTES = 64

# A single forward pass through a ~2400-weight network is tiny compared to
# a full image -- this is generous headroom, not an expected runtime.
TIMEOUT_SEC = 300

MAX_WORKERS = 1  # bump once you've confirmed memory/CPU headroom per run


def build_cmd():
    return [
        "axpike",
        f"--adele=mem_read_prob:{ERROR_RATE},linesz:32",
        "--adele-activate=0:AXRAM",
        "--dc=128:8:32",
        "--ic=256:4:32",
        "--l2=1024:4:32",
        "pk",
        APP_BIN,
    ]


# ===============================
# test.bin reader
# ===============================

def load_test_samples(path: Path):
    samples = []
    with open(path, "rb") as f:
        (n,) = struct.unpack("<i", f.read(4))
        for _ in range(n):
            pixels = struct.unpack(f"<{N_INPUTS}d", f.read(8 * N_INPUTS))
            (label,) = struct.unpack("<i", f.read(4))
            samples.append((pixels, label))
    return samples


def load_model_bytes(path: Path) -> bytes:
    # dims (4 int32) + weights (total_weights float64) -- read as one blob,
    # identical bytes get reused as-is for every sample's stdin payload.
    return path.read_bytes()


def make_stdin_payload(model_bytes: bytes, pixels) -> bytes:
    payload = bytearray()
    if ADD_PK_STDIN_PADDING:
        payload.extend(b"\x00" * PK_STDIN_PADDING_BYTES)
    payload.extend(model_bytes)
    payload.extend(struct.pack(f"<{N_INPUTS}d", *pixels))
    return bytes(payload)


# ===============================
# AXGENN01 record -> prediction
# ===============================

# AxPike crash signatures -- same convention as
# train_axram/run_train_axram.py's detect_crash() and the JPEG/FFT
# kernels' crash-dump handling.
REGISTER_DUMP_PATTERN = re.compile(rb"z\s+[0-9a-fA-F]{16}\s+ra\s+[0-9a-fA-F]{16}")
ASSERT_FAILURE_PATTERN = re.compile(rb'assertion\s+"[^"]*"\s+failed')
CRASH_TEXT_MARKERS = (b"User load segfault", b"User store segfault",
                      b"User fetch segfault", b"segfault")


def find_crash_offset(data: bytes, start: int = 0):
    """Byte offset of the first AxPike crash signature at or after `start`,
    or None if there is none."""
    hits = []
    for pat in (REGISTER_DUMP_PATTERN, ASSERT_FAILURE_PATTERN):
        m = pat.search(data, start)
        if m:
            hits.append(m.start())
    for marker in CRASH_TEXT_MARKERS:
        k = data.find(marker, start)
        if k != -1:
            hits.append(k)
    return min(hits) if hits else None


def parse_genn_record(data: bytes) -> dict:
    if not data:
        return {"class": "MISSING_OUTPUT", "recoverable": False}
    if len(data) < 12 or data[0:8] != b"AXGENN01":
        return {"class": "INVALID_OUTPUT", "recoverable": False}

    (status,) = struct.unpack("<i", data[8:12])
    if status != 0:
        return {"class": f"AX_STATUS_{status}", "recoverable": False}

    if len(data) < 20:
        return {"class": "SHORT_HEADER", "recoverable": False}

    inputs, outputs = struct.unpack("<ii", data[12:20])
    expected_payload = outputs * 8
    record_end = 20 + expected_payload
    payload = data[20:]

    if len(payload) == expected_payload:
        klass = "OK"
    elif len(payload) > expected_payload:
        # Same rule as JPEG/FFT: bytes written before a crash are still
        # evaluated. Accept the record only if it is complete and the extra
        # bytes after it start with an AxPike crash signature (the kernel
        # wrote its result, then crashed, e.g. while exiting).
        crash_at = find_crash_offset(data, 20)
        if crash_at is None or crash_at < record_end:
            return {"class": "PARTIAL_OUTPUT", "recoverable": False,
                    "payload_len": len(payload), "expected_payload_len": expected_payload}
        klass = "OK_AFTER_CRASH"
    else:
        return {"class": "PARTIAL_OUTPUT", "recoverable": False,
                "payload_len": len(payload), "expected_payload_len": expected_payload}

    values = struct.unpack(f"<{outputs}d", data[20:record_end])
    if not all(math.isfinite(v) for v in values):
        return {"class": "NONFINITE_OUTPUT", "recoverable": False}
    predicted = max(range(outputs), key=lambda i: values[i])
    return {"class": klass, "recoverable": True, "predicted": predicted, "values": values}


# ===============================
# One-sample execution
# ===============================

def sweep_axpike_artifacts() -> None:
    """Move any AxPIKE_counters_*/AxPIKE_energy_*/AXRAM_log_* files that
    just landed in the current directory into AXPIKE_ARTIFACT_DIR. Called
    right after every axpike invocation, so they never pile up loose in
    the project folder even mid-batch."""
    found = []
    for pattern in AXPIKE_ARTIFACT_GLOBS:
        found.extend(Path(".").glob(pattern))

    if not found:
        return

    AXPIKE_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    for f in found:
        try:
            f.rename(AXPIKE_ARTIFACT_DIR / f.name)
        except OSError:
            pass  # best-effort housekeeping -- never fail the batch over this


def run_one(index: int, pixels, true_label: int, model_bytes: bytes,
            record_out: Path, log_path: Path) -> dict:
    start = time.monotonic()
    stdin_payload = make_stdin_payload(model_bytes, pixels)
    cmd = build_cmd()

    timed_out = False
    try:
        proc = subprocess.run(
            cmd, input=stdin_payload,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=TIMEOUT_SEC,
        )
        output_data = proc.stdout or b""
        stderr_data = proc.stderr or b""
    except subprocess.TimeoutExpired as e:
        timed_out = True
        output_data = e.stdout if isinstance(e.stdout, (bytes, bytearray)) else b""
        stderr_data = b""
    finally:
        sweep_axpike_artifacts()

    elapsed_sec = time.monotonic() - start

    record_out.parent.mkdir(parents=True, exist_ok=True)
    record_out.write_bytes(output_data)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_bytes(stderr_data)

    info = parse_genn_record(output_data)
    if timed_out and info["class"] == "MISSING_OUTPUT":
        info["class"] = "TIMEOUT_OR_EMPTY"
    info["elapsed_sec"] = elapsed_sec
    info["true_label"] = true_label
    info["index"] = index
    if info.get("recoverable"):
        info["correct"] = (info["predicted"] == true_label)
    return info


# ===============================
# Batch execution
# ===============================

def run_batch():
    for d in (OUTPUT_RECORD_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    model_bytes = load_model_bytes(MODEL_BIN)
    samples = load_test_samples(TEST_BIN)
    total = len(samples)

    print(f"Loaded {MODEL_BIN} ({len(model_bytes)} bytes) and {total} test samples.")
    print(f"Error rate: {ERROR_RATE}")
    print(f"Application: {APP_BIN}")
    print()

    rows = []
    completed = 0
    for i, (pixels, true_label) in enumerate(samples):
        completed += 1
        record_out = OUTPUT_RECORD_DIR / f"sample_{i:04d}.bin"
        log_path = LOG_DIR / f"sample_{i:04d}.log"
        info = run_one(i, pixels, true_label, model_bytes, record_out, log_path)
        rows.append(info)
        status = "OK" if info.get("correct") else info["class"]
        pred = info.get("predicted", "-")
        print(f"[{completed}/{total}] sample={i:04d} true={true_label} "
              f"pred={pred} -> {status}")

    write_summary(rows)
    print_summary(rows)


def write_summary(rows: list):
    fields = ["index", "true_label", "predicted", "correct", "class",
              "recoverable", "elapsed_sec"]
    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields})


def print_summary(rows: list):
    total = len(rows)
    by_class = Counter(r["class"] for r in rows)
    recoverable = [r for r in rows if r.get("recoverable")]
    correct = sum(1 for r in recoverable if r["correct"])

    print()
    print("=== Summary ===")
    print(f"Total samples: {total}")
    for cls, count in sorted(by_class.items(), key=lambda kv: -kv[1]):
        print(f"  {cls}: {count}")
    print()
    print(f"Recoverable outputs: {len(recoverable)}/{total} "
          f"({100*len(recoverable)/total:.2f}%)")
    if recoverable:
        print(f"Accuracy over recoverable outputs: {correct}/{len(recoverable)} "
              f"({100*correct/len(recoverable):.2f}%)")
    print(f"Accuracy over ALL samples (unrecoverable = wrong): "
          f"{correct}/{total} ({100*correct/total:.2f}%)")
    print(f"CSV written to: {SUMMARY_CSV}")

    log_experiment_result(rows, total, recoverable, correct)


# ===============================
# Master results table (all 4 lines of the experiment matrix)
# ===============================

RESULTS_CSV_FIELDS = [
    "timestamp", "line", "train_mode", "tag", "validate_mode", "error_rate",
    "hidden_units", "hidden_layers", "model_file",
    "accuracy_all_pct", "accuracy_recoverable_pct",
    "n_recoverable", "n_total", "summary_csv",
    "n_recovered_after_crash",
]


def _model_dims(model_bytes: bytes):
    """dims are the first 4 int32s of model.bin's own layout -- see README
    'Formatos binarios'. Best-effort only: logged as blank if it can't be
    parsed, never fatal."""
    try:
        import struct as _struct
        inputs, hidden_layers, hidden, outputs = _struct.unpack("<iiii", model_bytes[:16])
        return hidden, hidden_layers
    except Exception:
        return "", ""


def _upgrade_results_header() -> None:
    """If RESULTS_CSV was written by an older version of this script (fewer
    columns), rewrite it once with the current header so new rows stay
    aligned. Missing columns are left blank in the old rows."""
    if not RESULTS_CSV.exists():
        return
    with open(RESULTS_CSV, newline="") as f:
        reader = csv.DictReader(f)
        old_fields = reader.fieldnames or []
        old_rows = list(reader)
    if old_fields == RESULTS_CSV_FIELDS:
        return
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_CSV_FIELDS)
        writer.writeheader()
        for r in old_rows:
            writer.writerow({k: r.get(k, "") for k in RESULTS_CSV_FIELDS})


def log_experiment_result(rows: list, total: int, recoverable: list, correct: int) -> None:
    """Appends ONE row to RESULTS_CSV summarizing this whole run -- a
    cumulative master table for plotting accuracy vs. error rate. The
    "line" field just labels which model was validated: 2 = an
    exact-host-trained model (--train-mode exact), 4 = a model trained
    under AxRAM (--train-mode axram). Never overwrites -- one line per
    invocation, same cumulative-log convention as
    train_axram/run_train_axram.py's genann_train_axram_runs.csv."""
    try:
        model_bytes = load_model_bytes(MODEL_BIN)
        hidden, hidden_layers = _model_dims(model_bytes)
    except Exception:
        hidden, hidden_layers = "", ""

    accuracy_all = 100 * correct / total if total else ""
    accuracy_recoverable = 100 * correct / len(recoverable) if recoverable else ""

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "line": 4 if TRAIN_MODE == "axram" else 2,
        "train_mode": TRAIN_MODE,
        "tag": TAG if TAG else TRAIN_MODE,
        "validate_mode": "axram",
        "error_rate": ERROR_RATE,
        "hidden_units": hidden,
        "hidden_layers": hidden_layers,
        "model_file": str(MODEL_BIN),
        "accuracy_all_pct": f"{accuracy_all:.4f}" if accuracy_all != "" else "",
        "accuracy_recoverable_pct": f"{accuracy_recoverable:.4f}" if accuracy_recoverable != "" else "",
        "n_recoverable": len(recoverable),
        "n_total": total,
        "summary_csv": str(SUMMARY_CSV),
        "n_recovered_after_crash": sum(1 for r in rows if r.get("class") == "OK_AFTER_CRASH"),
    }

    _upgrade_results_header()
    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"Result logged to: {RESULTS_CSV} (line={row['line']}, train_mode={TRAIN_MODE})")


# ===============================
# CLI
# ===============================

def parse_args():
    p = argparse.ArgumentParser(
        description="Validate a GENANN model under AxPike/AxRAM, one forward "
                     "pass per test sample. Run from the project root."
    )
    p.add_argument("--error-rate", default=ERROR_RATE,
                    help=f"mem_read_prob passed to axpike (default: {ERROR_RATE})")
    p.add_argument("--model", default=str(MODEL_BIN),
                    help=f"path to the model.bin-format file to validate (default: {MODEL_BIN})")
    p.add_argument("--train-mode", choices=("exact", "axram"), default=TRAIN_MODE,
                    help="how --model was trained -- used to label the row in "
                         "--results-csv, and (unless --tag is given) to tag "
                         f"output files/folders (default: {TRAIN_MODE})")
    p.add_argument("--tag", default=None,
                    help="finer-grained label for output file/folder naming, "
                         "overriding --train-mode for that purpose only -- use "
                         "when validating several models of the same "
                         "--train-mode (e.g. several AxRAM-trained models) so "
                         "their outputs don't collide (default: --train-mode)")
    p.add_argument("--results-csv", default=str(RESULTS_CSV),
                    help=f"master all-lines results table to append to (default: {RESULTS_CSV})")
    p.add_argument("--app-bin", default=APP_BIN,
                    help=f"path to the validate_kernel_axpike binary (default: {APP_BIN})")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ERROR_RATE = args.error_rate
    MODEL_BIN = Path(args.model)
    TRAIN_MODE = args.train_mode
    TAG = args.tag
    RESULTS_CSV = Path(args.results_csv)
    APP_BIN = args.app_bin
    # Re-derive the per-run output paths now that ERROR_RATE/TRAIN_MODE may
    # have changed -- they were computed once at import time from the
    # module-level defaults above.
    OUTPUT_RECORD_DIR = Path(f"./{_tagged(f'digits_records_err{ERROR_RATE}')}")
    LOG_DIR = Path(f"./{_tagged(f'digits_logs_err{ERROR_RATE}')}")
    SUMMARY_CSV = Path(f"./{_tagged(f'digits_summary_err{ERROR_RATE}')}.csv")
    AXPIKE_ARTIFACT_DIR = Path(f"./{_tagged(f'axpike_artifacts_err{ERROR_RATE}')}")

    run_batch()
