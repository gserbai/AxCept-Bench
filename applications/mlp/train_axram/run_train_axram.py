#!/usr/bin/env python3
######################################################################
# AxCept-Bench (MLP -- TRAINING under AxRAM, not just inference)
# Author: Guilherme Saides Serbai 
# Year: 2026
#
# Run this from the project root (it reads/writes paths like
# ./dataset_digits/, ./init_weights.bin, ./train_kernel_axpike relative to
# the current directory, not to this script's own location):
#
#   python3 train_axram/run_train_axram.py
#
# Unlike train_exact_and_validate/run_validate_axram.py (one axpike
# invocation PER TEST SAMPLE, doing a single forward pass each), this
# script makes exactly ONE axpike invocation for the WHOLE training run:
# n_train_samples * epochs forward+backward passes, all touching the same
# weight buffer inside approximate memory, entirely inside
# train_kernel_axpike.cpp. See that file's header comment for why training
# has to be one long-running guest process rather than one-process-per-step.
#
# PRACTICAL WARNING: AxPike is a detailed simulator. epochs * n_train_samples
# forward+backward passes in a row, all under simulation, with no
# parallelism, can take a LOT longer than you'd expect from the host
# training time. Start with small N_TRAIN_SAMPLES / EPOCHS (the defaults
# below) to get a wall-clock feel before scaling up to the full dataset --
# there is no way to checkpoint mid-run, a killed/timed-out run produces no
# usable output at all.
#
# Pipeline:
#   1. Load a FIXED initial-weights file (gen_init_weights_host.c) -- same
#      starting point an exact-host baseline run should use, so the
#      comparison isn't confounded by random-init variance.
#   2. Load the first N_TRAIN_SAMPLES samples of dataset_digits/train.bin.
#   3. Build ONE stdin payload: padding + header (dims/epochs/lr) + initial
#      weights + training samples.
#   4. Run axpike ONCE, with AxRAM active for the whole training loop.
#   5. Parse the AXTRN001 record -> final trained weights -> write as
#      model_trained_axram_err<rate>.bin, same layout as model.bin, directly
#      usable as MODEL_BIN in train_exact_and_validate/run_validate_axram.py
#      or as the argument to eval_exact_host.
######################################################################

import argparse
import csv
import re
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ===============================
# Experiment configuration
# ===============================
#
# These are the defaults for a plain manual run (`python3
# run_train_axram.py`, no arguments). Every value below can be overridden
# per-invocation with a CLI flag -- see parse_args() -- so the same script
# works both for a one-off run and for an automated sweep (train_sweep.py,
# project root) without hand-editing this file between rates.

ERROR_RATE = "1e-4"

# Small on purpose -- see the PRACTICAL WARNING above. Bump these once
# you've seen how long a small run actually takes under axpike.
N_TRAIN_SAMPLES = 50 # 1437 digits
EPOCHS = 5
LEARNING_RATE = 1.0  # matches train_exact_host.c's default

INIT_WEIGHTS_BIN = Path("./init_weights.bin")     # from gen_init_weights_host
TRAIN_BIN = Path("./dataset_digits/train.bin")     # from dataset/export_digits.py
APP_BIN = "./train_kernel_axpike"                  # riscv64 cross build -- ADJUST path if needed

OUTPUT_MODEL = Path(f"./model_trained_axram_err{ERROR_RATE}.bin")
LOG_FILE = Path(f"./genann_train_axram_err{ERROR_RATE}.log")
# Raw stdout, always saved (small: a few 100s KB at most for realistic
# architectures) -- the only way to see what actually came back when the
# result ISN'T OK: a crash dump, empty output, or something that doesn't
# even look like an AXTRN001 record shows up on stdout, not stderr.
RAW_STDOUT_FILE = Path(f"./genann_train_axram_err{ERROR_RATE}.stdout.raw")

# Cumulative log across EVERY run of this script, whatever ERROR_RATE was
# set to at the time -- one row per invocation, never overwritten. Unlike
# LOG_FILE/RAW_STDOUT_FILE (one per error rate, overwritten on rerun), this
# is meant to accumulate a full history across a sweep of error rates
# and/or repeated runs, so crash/timeout rate as a function of error rate
# can be read back later without having tracked it by hand.
RUNS_CSV = Path("./genann_train_axram_runs.csv")

# axpike drops its own per-run counter/energy CSVs and AXRAM log straight
# into the current directory -- swept into their own folder after the run,
# same convention as the other AxCept-Bench batch scripts.
AXPIKE_ARTIFACT_DIR = Path(f"./axpike_artifacts_train_err{ERROR_RATE}")
AXPIKE_ARTIFACT_GLOBS = (
    "AxPIKE_counters_*.csv",
    "AxPIKE_energy_*.csv",
    "AXRAM_log_*.log",
)

ADD_PK_STDIN_PADDING = True
PK_STDIN_PADDING_BYTES = 64  # same pk-loader-level padding as every other
                              # AxCept-Bench kernel (see validate_kernel_axpike.cpp)

# One long-running invocation instead of many short ones -- give it real
# headroom. This is a ceiling, not an expected runtime; a training run that
# hits this is probably not going to finish reasonably and should be
# killed and retried with smaller N_TRAIN_SAMPLES/EPOCHS instead of a
# bigger timeout.
TIMEOUT_SEC = 3600 * 6  # 6 hours


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
# axpike artifact housekeeping
# ===============================

def sweep_axpike_artifacts() -> None:
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
            pass


# ===============================
# Input loading
# ===============================

def load_init_weights(path: Path):
    """dims (4 int32) + weights (total_weights float64) -- same layout as
    model.bin. Returns (dims_tuple, weights_bytes)."""
    data = path.read_bytes()
    inputs, hidden_layers, hidden, outputs = struct.unpack("<iiii", data[:16])
    weights_bytes = data[16:]
    return (inputs, hidden_layers, hidden, outputs), weights_bytes


def load_train_subset(path: Path, n_samples: int, n_inputs: int, n_outputs: int) -> bytes:
    """Reads the first n_samples samples of train.bin's body (pixels +
    one_hot, interleaved per sample -- matches train_kernel_axpike.cpp's
    expected layout exactly) and returns them as raw bytes, ready to append
    to the stdin payload as-is."""
    with open(path, "rb") as f:
        (total_in_file,) = struct.unpack("<i", f.read(4))
        if n_samples > total_in_file:
            raise ValueError(
                f"requested N_TRAIN_SAMPLES={n_samples} but {path} only has "
                f"{total_in_file} samples"
            )
        sample_bytes = 8 * (n_inputs + n_outputs)
        return f.read(sample_bytes * n_samples)


# ===============================
# Payload construction
# ===============================

def build_stdin_payload(dims, weights_bytes: bytes, train_bytes: bytes) -> bytes:
    inputs, hidden_layers, hidden, outputs = dims
    payload = bytearray()

    if ADD_PK_STDIN_PADDING:
        payload.extend(b"\x00" * PK_STDIN_PADDING_BYTES)

    payload.extend(struct.pack("<i", N_TRAIN_SAMPLES))
    payload.extend(struct.pack("<iiii", inputs, hidden_layers, hidden, outputs))
    payload.extend(struct.pack("<i", EPOCHS))
    payload.extend(struct.pack("<d", LEARNING_RATE))
    payload.extend(weights_bytes)
    payload.extend(train_bytes)

    return bytes(payload)


# ===============================
# AXTRN001 record -> trained model
# ===============================

def parse_train_record(data: bytes) -> dict:
    if not data:
        return {"status": "MISSING_OUTPUT", "ok": False}

    # Observed empirically: axpike/pk appears to funnel the guest's stderr
    # (train_kernel_axpike.cpp's per-epoch progress fprintf(stderr, ...)
    # calls) into the same stream this script captures as the child's
    # stdout, even though the kernel writes progress and the final record
    # through two different FILE*s. Rather than depend on pk's exact fd
    # semantics (undocumented, possibly build-specific), locate the magic
    # wherever it actually landed instead of assuming it's at offset 0 --
    # same "search, don't assume position" approach the JPEG kernel's
    # crash-recovery already uses for AxPike's own crash-dump text.
    magic_idx = data.find(b"AXTRN001")
    if magic_idx == -1:
        return {"status": "INVALID_OUTPUT", "ok": False}
    if magic_idx > 0:
        print(f"  (note: {magic_idx} bytes of non-record output -- likely "
              f"guest stderr merged into stdout -- appeared before the "
              f"AXTRN001 record and were skipped)")
    data = data[magic_idx:]

    if len(data) < 12:
        return {"status": "SHORT_HEADER", "ok": False}

    (status,) = struct.unpack("<i", data[8:12])
    if status != 0:
        return {"status": f"AX_STATUS_{status}", "ok": False}

    if len(data) < 28:
        return {"status": "SHORT_HEADER", "ok": False}

    inputs, hidden_layers, hidden, outputs = struct.unpack("<iiii", data[12:28])
    total_weights = _total_weights(inputs, hidden_layers, hidden, outputs)
    expected_payload = total_weights * 8
    payload = data[28:]
    if len(payload) != expected_payload:
        return {
            "status": "PARTIAL_OUTPUT", "ok": False,
            "payload_len": len(payload), "expected_payload_len": expected_payload,
        }

    dims_and_weights = data[12:28] + payload  # exactly model.bin's own layout
    return {
        "status": "OK", "ok": True,
        "dims": (inputs, hidden_layers, hidden, outputs),
        "model_bytes": dims_and_weights,
    }


# ===============================
# Crash detection
# ===============================

# Same convention as the other AxCept-Bench kernels' crash-dump detection
# (classify_jpeg_output() in axcept_batch_jpeg.py, r50t.py's PADRAO_CRASH):
# "z  " at the very start of a stream, or the AxPike register-dump pattern
# "z <16 hex> ra <16 hex>" anywhere, or the plain text markers AxPike's own
# crash log uses. This is checked on the RAW stdout, independent of
# whether an AXTRN001 record was also found -- a crash can happen with or
# without a partial/prior record present.
#
# ALSO covers a second, GENANN-specific crash mode this kernel is exposed
# to that AxPike itself never sees: lib/genann.c's sigmoid activation has
# `assert(!isnan(a))` (line 75). A single AxRAM bit-flip on a weight can
# turn it into an astronomically large (or outright NaN) double, which
# then blows up the sigmoid's accumulated input across the network and
# hits that assert -- the guest process aborts via newlib/picolibc's
# `assertion "..." failed: file "...", line N, function: ...` message
# (observed empirically: 2026-09, a single epoch over the full 1437-sample
# dataset at mem_read_prob=1e-3 hit this on the FIRST calibration run).
# This is a genuine numerical consequence of training under approximate
# memory, not a pipeline bug -- worth counting as a crash (not lumped into
# the generic INVALID_OUTPUT bucket) so crash-rate-vs-error-rate analysis
# actually reflects it.
REGISTER_DUMP_PATTERN = re.compile(rb"z\s+[0-9a-fA-F]{16}\s+ra\s+[0-9a-fA-F]{16}")
ASSERT_FAILURE_PATTERN = re.compile(rb'assertion\s+"[^"]*"\s+failed')
CRASH_TEXT_MARKERS = (
    b"User load segfault",
    b"User store segfault",
    b"User fetch segfault",
    b"segfault",
)


def detect_crash(data: bytes) -> bool:
    """True if the raw stdout contains an AxPike crash signature anywhere
    -- distinguishes "the guest process actually crashed" from "the output
    just doesn't parse as an AXTRN001 record for some other reason"
    (SHORT_HEADER/PARTIAL_OUTPUT truncation, or a build/protocol mismatch)."""
    if not data:
        return False
    if data.startswith(b"z  "):
        return True
    if REGISTER_DUMP_PATTERN.search(data):
        return True
    if ASSERT_FAILURE_PATTERN.search(data):
        return True
    return any(marker in data for marker in CRASH_TEXT_MARKERS)


# ===============================
# Cumulative run log
# ===============================

RUNS_CSV_FIELDS = [
    "timestamp", "error_rate", "n_train_samples", "epochs", "learning_rate",
    "elapsed_sec", "timed_out", "crashed", "returncode", "status",
    "model_written", "output_model", "raw_stdout_bytes",
]


def log_run(row: dict) -> None:
    """Appends one row to RUNS_CSV -- one row per invocation of this
    script, across every error rate and every attempt, never overwritten.
    Writes the header only the first time the file is created."""
    write_header = not RUNS_CSV.exists()
    with open(RUNS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUNS_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in RUNS_CSV_FIELDS})


def _total_weights(inputs, hidden_layers, hidden, outputs):
    # Mirrors genann_init()'s own formula (lib/genann.c) -- needed here
    # only to know how many bytes of payload to expect.
    if hidden_layers == 0:
        return (inputs + 1) * outputs
    hidden_weights = (inputs + 1) * hidden + (hidden_layers - 1) * (hidden + 1) * hidden
    output_weights = (hidden + 1) * outputs
    return hidden_weights + output_weights


# ===============================
# Main
# ===============================

def run_training() -> dict:
    dims, weights_bytes = load_init_weights(INIT_WEIGHTS_BIN)
    inputs, hidden_layers, hidden, outputs = dims
    train_bytes = load_train_subset(TRAIN_BIN, N_TRAIN_SAMPLES, inputs, outputs)

    payload = build_stdin_payload(dims, weights_bytes, train_bytes)
    cmd = build_cmd()

    print("=== GENANN training under AxPike/AxRAM ===")
    print(f"Error rate:      {ERROR_RATE}")
    print(f"Architecture:    {inputs} -> " + " -> ".join([str(hidden)] * hidden_layers) + f" -> {outputs}")
    print(f"Train samples:   {N_TRAIN_SAMPLES}")
    print(f"Epochs:          {EPOCHS}")
    print(f"Learning rate:   {LEARNING_RATE}")
    print(f"Init weights:    {INIT_WEIGHTS_BIN}")
    print(f"App:             {APP_BIN}")
    print(f"Timeout:         {TIMEOUT_SEC}s ({TIMEOUT_SEC / 3600:.1f}h ceiling)")
    print()
    print("Running -- this streams epoch progress to stderr/LOG_FILE as it goes.")
    print("(one long invocation, no per-sample output until it's all done)")
    print()

    start = time.monotonic()
    timed_out = False
    returncode = None  # None only if TimeoutExpired -- the child was killed by
                        # subprocess.run() itself after TIMEOUT_SEC and no exit
                        # code is available for that case.
    try:
        proc = subprocess.run(
            cmd, input=payload,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=TIMEOUT_SEC,
        )
        output_data = proc.stdout or b""
        stderr_data = proc.stderr or b""
        returncode = proc.returncode
    except subprocess.TimeoutExpired as e:
        timed_out = True
        output_data = e.stdout if isinstance(e.stdout, (bytes, bytearray)) else b""
        stderr_data = e.stderr if isinstance(e.stderr, (bytes, bytearray)) else b""
    finally:
        sweep_axpike_artifacts()

    elapsed_sec = time.monotonic() - start
    LOG_FILE.write_bytes(stderr_data)
    RAW_STDOUT_FILE.write_bytes(output_data)
    crashed = detect_crash(output_data) or detect_crash(stderr_data)

    print(f"Elapsed: {elapsed_sec:.1f}s ({elapsed_sec / 60:.1f} min)")
    # A negative returncode means the OS killed axpike with that signal
    # (e.g. -9 = SIGKILL, often the OOM killer) -- something OUTSIDE this
    # script's own TIMEOUT_SEC/6h ceiling terminated it early. This is
    # invisible unless we print/log it: subprocess.run() happily returns
    # whatever partial stdout came through the pipe before the process
    # died, which then just looks like a generic PARTIAL_OUTPUT/garbage
    # parse failure with no clue about WHY it's partial.
    if returncode is not None and returncode < 0:
        import signal as _signal
        try:
            sig_name = _signal.Signals(-returncode).name
        except ValueError:
            sig_name = f"signal {-returncode}"
        print(f"WARNING: axpike was killed by {sig_name} (returncode={returncode}) -- "
              f"NOT this script's own timeout ({TIMEOUT_SEC}s ceiling, {timed_out=}). "
              f"Something external killed it (OOM killer is the usual suspect -- "
              f"check `dmesg | grep -i oom` / `free -h`).")
    elif returncode not in (None, 0):
        print(f"NOTE: axpike exited with non-zero returncode={returncode} "
              f"(not a signal, not a timeout).")

    base_row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "error_rate": ERROR_RATE,
        "n_train_samples": N_TRAIN_SAMPLES,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "elapsed_sec": f"{elapsed_sec:.2f}",
        "timed_out": timed_out,
        "crashed": crashed,
        "returncode": returncode,
        "raw_stdout_bytes": len(output_data),
    }

    if timed_out:
        print(f"TIMEOUT after {TIMEOUT_SEC}s -- no usable output. "
              f"Retry with smaller N_TRAIN_SAMPLES/EPOCHS. Stderr saved to {LOG_FILE}.")
        log_run({**base_row, "status": "TIMEOUT", "model_written": False, "output_model": ""})
        return {"ok": False, "status": "TIMEOUT", "output_model": None}

    result = parse_train_record(output_data)
    status = result["status"]
    # A crash signature overrides a generic parse failure -- "the record
    # doesn't parse" and "the record doesn't parse BECAUSE the guest
    # crashed" are different findings, and the crash case is the more
    # specific/useful one to have logged.
    if crashed and not result["ok"]:
        status = "CRASHED"
    print(f"Result: {status}" + (" (crash signature detected)" if crashed and result["ok"] else ""))

    if not result["ok"]:
        print(f"No usable trained model -- see {LOG_FILE} (stderr) and "
              f"{RAW_STDOUT_FILE} (raw stdout, {len(output_data)} bytes) for the run.")
        # Quick hint printed straight to the terminal -- no need to open
        # the raw file just to tell "empty" from "crash dump" from
        # "garbage that isn't even an AXTRN001 record" apart.
        if not output_data:
            print("  stdout was completely empty.")
        elif crashed:
            print("  stdout/stderr contain an AxPike crash signature -- the guest process")
            print("  likely crashed (segfault/exception) partway through the run. All")
            print("  progress up to that point is lost -- this kernel only writes its")
            print("  output once, at the very end (see README.md, 'Detecção de crash e")
            print("  política de resultados parciais').")
        else:
            preview = output_data[:64]
            print(f"  first {len(preview)} bytes of stdout: {preview!r}")
        log_run({**base_row, "status": status, "model_written": False, "output_model": ""})
        return {"ok": False, "status": status, "output_model": None}

    OUTPUT_MODEL.write_bytes(result["model_bytes"])
    print(f"Trained model written to: {OUTPUT_MODEL}")
    log_run({**base_row, "status": status, "model_written": True, "output_model": str(OUTPUT_MODEL)})
    print()
    print("Next steps:")
    print(f"  ./eval_exact_host {OUTPUT_MODEL}")
    print("      -> exact host accuracy of the AxRAM-trained network")
    print(f"  ./train_exact_host {hidden} {EPOCHS} {hidden_layers} {INIT_WEIGHTS_BIN} {N_TRAIN_SAMPLES} > model_exact_baseline.bin")
    print("      -> exact-host baseline: SAME initial weights, SAME samples, SAME epochs --")
    print("         the only difference left is AxRAM noise. Without the last arg")
    print("         (n_train_samples) the baseline would train on the full dataset instead")
    print("         of just these samples, and the comparison wouldn't isolate anything.")
    print()
    print(f"Run logged to: {RUNS_CSV}")
    return {"ok": True, "status": status, "output_model": str(OUTPUT_MODEL)}


# ===============================
# CLI
# ===============================

def parse_args():
    p = argparse.ArgumentParser(
        description="Train a GENANN model under AxPike/AxRAM in a single "
                     "long invocation (whole training loop, all epochs/"
                     "samples). Run from the project root."
    )
    p.add_argument("--error-rate", default=ERROR_RATE,
                    help=f"mem_read_prob passed to axpike (default: {ERROR_RATE})")
    p.add_argument("--n-train-samples", type=int, default=N_TRAIN_SAMPLES,
                    help=f"how many samples from train.bin to use (default: {N_TRAIN_SAMPLES})")
    p.add_argument("--epochs", type=int, default=EPOCHS,
                    help=f"how many epochs to train (default: {EPOCHS})")
    p.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                    help=f"default: {LEARNING_RATE}")
    p.add_argument("--init-weights", default=str(INIT_WEIGHTS_BIN),
                    help=f"default: {INIT_WEIGHTS_BIN}")
    p.add_argument("--output", default=None,
                    help="output model path (default: derived from --error-rate, "
                         "same as the module default OUTPUT_MODEL)")
    p.add_argument("--timeout-sec", type=int, default=TIMEOUT_SEC,
                    help=f"default: {TIMEOUT_SEC}s")
    p.add_argument("--app-bin", default=APP_BIN,
                    help=f"path to the train_kernel_axpike binary (default: {APP_BIN})")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ERROR_RATE = args.error_rate
    N_TRAIN_SAMPLES = args.n_train_samples
    EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    INIT_WEIGHTS_BIN = Path(args.init_weights)
    TIMEOUT_SEC = args.timeout_sec
    APP_BIN = args.app_bin
    # Re-derive the per-run output paths now that ERROR_RATE may have
    # changed -- they were computed once at import time from the
    # module-level defaults above. --output overrides OUTPUT_MODEL only
    # (the .log/.stdout.raw/artifact-dir names stay keyed by error rate,
    # same convention as run_validate_axram.py).
    OUTPUT_MODEL = Path(args.output) if args.output else Path(f"./model_trained_axram_err{ERROR_RATE}.bin")
    LOG_FILE = Path(f"./genann_train_axram_err{ERROR_RATE}.log")
    RAW_STDOUT_FILE = Path(f"./genann_train_axram_err{ERROR_RATE}.stdout.raw")
    AXPIKE_ARTIFACT_DIR = Path(f"./axpike_artifacts_train_err{ERROR_RATE}")

    result = run_training()
    sys.exit(0 if result["ok"] else 1)
