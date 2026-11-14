#!/usr/bin/env python3
######################################################################
# AxCept-Bench (JPEG Encoder)
# Author: Guilherme Saides Serbai
# Year: 2026
######################################################################

import shutil
import subprocess
import time
from pathlib import Path


# ===============================
# Experiment configuration
# ===============================

ERROR_RATE = "1e-1"
QUALITY = "100"
TIMEOUT_SEC = 1200

# PK/stdin workaround:
# AxPike pk may consume the first 64 bytes from stdin before the application
# receives the stream.
#
# IMPORTANT:
# For the JPEG workload, the input is textual CSV. Therefore, the 64-byte
# prefix is made of ASCII spaces (0x20), not NUL bytes. This keeps the input
# textual and harmless for CSV parsers that skip whitespace before width/height.
ADD_PK_STDIN_PADDING = True
PK_STDIN_PADDING_BYTES = 64
PK_STDIN_PADDING_BYTE = b" "

# Directory configuration
DATASET_DIR = Path("/home/guilherme/UC_MERCED_LAND_USE/Datas/dataset_csv")
APP_BIN = "./src/toojpeg_encoder"

OUTPUT_JPEG_DIR = Path(f"./src/dataset_error_rate_{ERROR_RATE}")
LOG_DIR = Path(f"./src/logs_error_rate_{ERROR_RATE}")

# AxPike/AxRAM configuration
AXPIKE_CMD = [
    "axpike",
    f"--adele=mem_read_prob:{ERROR_RATE},linesz:32",
    "--adele-activate=0:AXRAM",
    "--dc=128:8:32",
    "--ic=256:4:32",
    "--l2=1024:4:32",
    "pk",
    APP_BIN,
    QUALITY,
]


# ===============================
# Output classification
# ===============================

def classify_jpeg_output(output_path: Path, timed_out: bool) -> dict:
    """
    Classifica apenas para imprimir no terminal.
    Nada daqui é salvo em CSV.
    """

    if not output_path.exists():
        return {
            "class": "TIMEOUT_OR_EMPTY" if timed_out else "MISSING_OUTPUT",
            "crash": False,
            "valid_jpeg": False,
        }

    data = output_path.read_bytes()

    if len(data) == 0:
        return {
            "class": "TIMEOUT_OR_EMPTY",
            "crash": False,
            "valid_jpeg": False,
        }

    crash = (
        data.startswith(b"z  ")
        or b"User load segfault" in data
        or b"User store segfault" in data
        or b"User fetch segfault" in data
        or b"segfault" in data
    )

    jpeg_soi = data.startswith(b"\xff\xd8")
    jpeg_jfif = b"JFIF" in data[:256]
    jpeg_exif = b"Exif" in data[:256]

    # Minimal validity for this benchmark:
    # it must start as JPEG and contain a recognizable JPEG header marker.
    valid_jpeg = jpeg_soi and (jpeg_jfif or jpeg_exif or b"\xff\xdb" in data[:512])

    if crash and valid_jpeg:
        cls = "JPEG_WITH_CRASH_DUMP"
    elif crash:
        cls = "AXPIKE_CRASH_DUMP"
    elif timed_out and valid_jpeg:
        cls = "PARTIAL_JPEG_TIMEOUT"
    elif timed_out:
        cls = "TIMEOUT_OR_EMPTY"
    elif valid_jpeg:
        cls = "VALID_JPEG"
    elif jpeg_soi:
        cls = "PARTIAL_JPEG"
    else:
        cls = "INVALID_OUTPUT"

    return {
        "class": cls,
        "crash": crash,
        "valid_jpeg": valid_jpeg,
    }


# ===============================
# One-file execution
# ===============================

def feed_input_with_optional_padding(proc, input_path: Path) -> None:
    """
    Stream CSV data to guest stdin.

    For JPEG CSV input, the pk padding uses ASCII spaces. This avoids injecting
    NUL bytes into a textual CSV stream.
    """
    try:
        if ADD_PK_STDIN_PADDING:
            proc.stdin.write(PK_STDIN_PADDING_BYTE * PK_STDIN_PADDING_BYTES)

        with open(input_path, "rb") as f_in:
            shutil.copyfileobj(f_in, proc.stdin, length=1024 * 1024)

    except BrokenPipeError:
        # The guest may crash before consuming the full input. This is expected
        # under aggressive approximation and should not stop the batch.
        pass

    except OSError as e:
        # Broken pipe can also appear as OSError errno 32.
        if getattr(e, "errno", None) != 32:
            raise

    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass


def run_one(input_csv: Path, output_jpeg: Path, output_log: Path) -> dict:
    """Run one CSV image through AxPike and save stdout/stderr artifacts."""
    timed_out = False
    return_code = None
    start = time.monotonic()

    with open(output_jpeg, "wb") as f_out, open(output_log, "wb") as f_log:
        proc = subprocess.Popen(
            AXPIKE_CMD,
            stdin=subprocess.PIPE,
            stdout=f_out,
            stderr=f_log,
        )

        feed_input_with_optional_padding(proc, input_csv)

        try:
            return_code = proc.wait(timeout=TIMEOUT_SEC)

        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            return_code = proc.wait()
            f_log.write(b"\n[HOST] TIMEOUT: process killed by batch runner.\n")

    elapsed_sec = time.monotonic() - start
    output_bytes = output_jpeg.stat().st_size if output_jpeg.exists() else 0
    log_bytes = output_log.stat().st_size if output_log.exists() else 0

    classification = classify_jpeg_output(output_jpeg, timed_out)

    return {
        "returncode": return_code,
        "timed_out": timed_out,
        "elapsed_sec": elapsed_sec,
        "output_bytes": output_bytes,
        "log_bytes": log_bytes,
        **classification,
    }


# ===============================
# Batch execution
# ===============================

def print_outcome(info: dict) -> None:
    cls = info["class"]

    if cls == "VALID_JPEG":
        print(
            f"  -> Finished: VALID_JPEG "
            f"bytes={info['output_bytes']} "
            f"time={info['elapsed_sec']:.2f}s"
        )

    elif cls == "AXPIKE_CRASH_DUMP":
        print(f"  -> Finished: AXPIKE_CRASH_DUMP time={info['elapsed_sec']:.2f}s")

    elif cls == "JPEG_WITH_CRASH_DUMP":
        print(
            f"  -> Finished: JPEG_WITH_CRASH_DUMP "
            f"bytes={info['output_bytes']} "
            f"time={info['elapsed_sec']:.2f}s"
        )

    elif cls == "TIMEOUT_OR_EMPTY":
        print(f"  -> Finished: TIMEOUT_OR_EMPTY time={info['elapsed_sec']:.2f}s")

    else:
        print(
            f"  -> Finished: {cls} "
            f"returncode={info['returncode']} "
            f"bytes={info['output_bytes']} "
            f"time={info['elapsed_sec']:.2f}s"
        )


def run_conversion() -> None:
    OUTPUT_JPEG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(DATASET_DIR.rglob("*.csv"))

    if not csv_files:
        print(f"No .csv files found in {DATASET_DIR}")
        return

    total = len(csv_files)

    print(f"Found {total} files to process.")
    print(f"Error rate: {ERROR_RATE}")
    print(f"Quality:    {QUALITY}")
    print(f"Padding:    {ADD_PK_STDIN_PADDING} ({PK_STDIN_PADDING_BYTES} ASCII spaces)")
    print(f"Output:     {OUTPUT_JPEG_DIR}")
    print(f"Logs:       {LOG_DIR}")
    print()

    for idx, input_csv in enumerate(csv_files, start=1):
        relative_path = input_csv.relative_to(DATASET_DIR)
        output_jpeg = OUTPUT_JPEG_DIR / relative_path.with_suffix(".jpeg")
        output_log = LOG_DIR / relative_path.with_suffix(".log")

        output_jpeg.parent.mkdir(parents=True, exist_ok=True)
        output_log.parent.mkdir(parents=True, exist_ok=True)

        print(f"[{idx}/{total}] processing: {relative_path}")
        print(f"  -> Image: {output_jpeg.relative_to(OUTPUT_JPEG_DIR)}")
        print(f"  -> Log:   {output_log.relative_to(LOG_DIR)}")

        try:
            info = run_one(input_csv, output_jpeg, output_log)

        except Exception as exc:
            info = {
                "class": "SCRIPT_ERROR",
                "returncode": "EXCEPTION",
                "timed_out": False,
                "elapsed_sec": 0.0,
                "output_bytes": output_jpeg.stat().st_size if output_jpeg.exists() else 0,
                "log_bytes": output_log.stat().st_size if output_log.exists() else 0,
                "crash": False,
                "valid_jpeg": False,
            }

            with open(output_log, "ab") as f_log:
                f_log.write(f"\n[HOST] SCRIPT_ERROR: {exc}\n".encode("utf-8", errors="replace"))

        print_outcome(info)

    print()
    print("Completed!")
    print(f"Images in: {OUTPUT_JPEG_DIR}")
    print(f"Logs in:   {LOG_DIR}")


if __name__ == "__main__":
    run_conversion()
