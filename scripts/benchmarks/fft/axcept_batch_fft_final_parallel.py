#!/usr/bin/env python3
######################################################################
# AxCept-Bench (Audio FFT / Dominant Frequency)
# Author: Guilherme Saides Serbai
# Year: 2026
######################################################################

import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# ===============================
# Experiment configuration
# ===============================

ERROR_RATE = "1e-4"

DATASET_DIR = Path("/home/guilherme/Music/dataset_iowa_music_bin")

OUTPUT_BIN_DIR = Path(f"./src/dataset_audio_error_rate_{ERROR_RATE}")
LOG_DIR = Path(f"./src/logs_audio_error_rate_{ERROR_RATE}")

APP_BIN = "./dominant_freq"

INPUT_SUFFIX = ".bin"

# FFT input is raw float32 binary.
# 64 zero bytes = 16 float32 samples equal to 0.0.
ADD_PK_STDIN_PADDING = True
PK_STDIN_PADDING_BYTES = 64

# 1200 seconds = 20 minutes per file.
TIMEOUT_SEC = 1200

# Number of AxPike executions to run in parallel.
MAX_WORKERS = 4


# ===============================
# AxPike command
# ===============================

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
# One-file execution
# ===============================

def make_stdin_payload(input_path: Path) -> bytes:
    payload = bytearray()

    if ADD_PK_STDIN_PADDING:
        payload.extend(b"\x00" * PK_STDIN_PADDING_BYTES)

    payload.extend(input_path.read_bytes())
    return bytes(payload)


def run_one(input_path: Path, output_path: Path, log_path: Path):
    cmd = build_cmd()
    stdin_payload = make_stdin_payload(input_path)

    start = time.monotonic()

    try:
        proc = subprocess.run(
            cmd,
            input=stdin_payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=TIMEOUT_SEC,
        )

        elapsed_sec = time.monotonic() - start

        output_data = proc.stdout if proc.stdout is not None else b""
        log_data = proc.stderr if proc.stderr is not None else b""

        output_path.write_bytes(output_data)
        log_path.write_bytes(log_data)

        return {
            "returncode": proc.returncode,
            "elapsed_sec": elapsed_sec,
            "timeout": False,
            "output_bytes": len(output_data),
        }

    except subprocess.TimeoutExpired as e:
        elapsed_sec = time.monotonic() - start

        output_path.write_bytes(b"")

        stderr_data = e.stderr if isinstance(e.stderr, (bytes, bytearray)) else b""
        with open(log_path, "wb") as f_log:
            if stderr_data:
                f_log.write(stderr_data)
            f_log.write(b"\nTIMEOUT\n")

        return {
            "returncode": 124,
            "elapsed_sec": elapsed_sec,
            "timeout": True,
            "output_bytes": 0,
        }


# ===============================
# Batch execution
# ===============================

def process_one_file(input_bin: Path):
    relative_path = input_bin.relative_to(DATASET_DIR)

    output_bin = OUTPUT_BIN_DIR / relative_path.with_suffix(".bin")
    output_log = LOG_DIR / relative_path.with_suffix(".log")

    output_bin.parent.mkdir(parents=True, exist_ok=True)
    output_log.parent.mkdir(parents=True, exist_ok=True)

    info = run_one(input_bin, output_bin, output_log)

    return relative_path, info


def run_conversion():
    OUTPUT_BIN_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    input_files = sorted(DATASET_DIR.rglob(f"*{INPUT_SUFFIX}"))

    if not input_files:
        print(f"No {INPUT_SUFFIX} files found in {DATASET_DIR}")
        return

    total = len(input_files)

    print(f"Found {total} audio files to process.")
    print(f"Error rate: {ERROR_RATE}")
    print(f"PK stdin padding: {ADD_PK_STDIN_PADDING} ({PK_STDIN_PADDING_BYTES} zero bytes)")
    print(f"Application: {APP_BIN}")
    print(f"Timeout: {TIMEOUT_SEC} seconds per file")
    print(f"Parallel workers: {MAX_WORKERS}")
    print(f"Outputs in: {OUTPUT_BIN_DIR}")
    print(f"Logs in:    {LOG_DIR}")
    print()

    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_one_file, input_bin): input_bin
            for input_bin in input_files
        }

        for future in as_completed(futures):
            input_bin = futures[future]
            completed += 1
            relative_path = input_bin.relative_to(DATASET_DIR)

            try:
                relative_path, info = future.result()

                timeout_msg = " TIMEOUT" if info["timeout"] else ""
                print(
                    f"[{completed}/{total}] done: {relative_path} "
                    f"returncode={info['returncode']} "
                    f"bytes={info['output_bytes']} "
                    f"time={info['elapsed_sec']:.2f}s"
                    f"{timeout_msg}"
                )

            except Exception as e:
                output_log = LOG_DIR / relative_path.with_suffix(".log")
                output_log.parent.mkdir(parents=True, exist_ok=True)

                with open(output_log, "ab") as f_log:
                    f_log.write(f"\nSCRIPT_ERROR: {e}\n".encode("utf-8", errors="replace"))

                print(f"[{completed}/{total}] SCRIPT_ERROR: {relative_path} ({e})")

    print()
    print("Completed.")
    print(f"Outputs in: {OUTPUT_BIN_DIR}")
    print(f"Logs in:    {LOG_DIR}")


if __name__ == "__main__":
    run_conversion()
