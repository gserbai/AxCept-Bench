#!/usr/bin/env python3
######################################################################
# AxCept-Bench (Sobel)
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

ERROR_RATE = "1e-3"
TIMEOUT_SEC = 1200

# PK/stdin workaround (mesmo esquema do JPEG):
# o pk pode consumir os primeiros 64 bytes do stdin. Usamos espaços (0x20);
# o parser PGM do sobel.cpp pula whitespace antes do "P5", então funciona
# com ou sem o consumo.
ADD_PK_STDIN_PADDING = True
PK_STDIN_PADDING_BYTES = 64
PK_STDIN_PADDING_BYTE = b" "

DATASET_DIR = Path("/home/guilherme/UFPR-ALPR/dataset_pgm")
APP_BIN = "./applications/sobel/src/sobel"

OUTPUT_DIR = Path(f"./applications/sobel/src/dataset_error_rate_{ERROR_RATE}")
LOG_DIR = Path(f"./applications/sobel/src/logs_error_rate_{ERROR_RATE}")

AXPIKE_CMD = [
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
# Output classification
# ===============================

def classify_pgm_output(output_path: Path, timed_out: bool) -> str:
    if not output_path.exists() or output_path.stat().st_size == 0:
        return "TIMEOUT_OR_EMPTY" if timed_out else "MISSING_OUTPUT"

    data = output_path.read_bytes()
    crash = b"segfault" in data or data.startswith(b"z  ")

    valid = False
    head = data[:64].split(maxsplit=4)
    if len(head) >= 4 and head[0] == b"P5":
        try:
            w, h = int(head[1]), int(head[2])
            hdr_len = len(b"P5\n%d %d\n255\n" % (w, h))
            valid = len(data) >= hdr_len + w * h
        except ValueError:
            pass

    if crash and valid:
        return "PGM_WITH_CRASH_DUMP"
    if crash:
        return "AXPIKE_CRASH_DUMP"
    if timed_out:
        return "PARTIAL_PGM_TIMEOUT" if data.startswith(b"P5") else "TIMEOUT_OR_EMPTY"
    if valid:
        return "VALID_PGM"
    return "PARTIAL_PGM" if data.startswith(b"P5") else "INVALID_OUTPUT"


# ===============================
# One-file execution
# ===============================

def feed_input(proc, input_path: Path) -> None:
    try:
        if ADD_PK_STDIN_PADDING:
            proc.stdin.write(PK_STDIN_PADDING_BYTE * PK_STDIN_PADDING_BYTES)
        with open(input_path, "rb") as f_in:
            shutil.copyfileobj(f_in, proc.stdin, length=1024 * 1024)
    except BrokenPipeError:
        pass  # guest pode crashar antes de consumir tudo
    except OSError as e:
        if getattr(e, "errno", None) != 32:
            raise
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass


def run_one(input_pgm: Path, output_pgm: Path, output_log: Path) -> dict:
    timed_out = False
    start = time.monotonic()

    with open(output_pgm, "wb") as f_out, open(output_log, "wb") as f_log:
        proc = subprocess.Popen(AXPIKE_CMD, stdin=subprocess.PIPE,
                                stdout=f_out, stderr=f_log)
        feed_input(proc, input_pgm)
        try:
            rc = proc.wait(timeout=TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            rc = proc.wait()
            f_log.write(b"\n[HOST] TIMEOUT: process killed by batch runner.\n")

    return {
        "returncode": rc,
        "elapsed_sec": time.monotonic() - start,
        "class": classify_pgm_output(output_pgm, timed_out),
    }


# ===============================
# Batch execution
# ===============================

def main() -> None:
    inputs = sorted(DATASET_DIR.rglob("*.pgm"))
    if not inputs:
        raise SystemExit(f"Nenhum .pgm em {DATASET_DIR}")

    print(f"[INFO] {len(inputs)} imagens | mem_read_prob={ERROR_RATE}")
    counts = {}

    for i, inp in enumerate(inputs, 1):
        rel = inp.relative_to(DATASET_DIR)
        out = OUTPUT_DIR / rel
        log = (LOG_DIR / rel).with_suffix(".log")
        out.parent.mkdir(parents=True, exist_ok=True)
        log.parent.mkdir(parents=True, exist_ok=True)

        info = run_one(inp, out, log)
        counts[info["class"]] = counts.get(info["class"], 0) + 1
        print(f"[{i}/{len(inputs)}] {rel} -> {info['class']} "
              f"(rc={info['returncode']}, {info['elapsed_sec']:.1f}s)")

    print("\n[RESUMO]")
    for cls, c in sorted(counts.items()):
        print(f"  {cls}: {c}")


if __name__ == "__main__":
    main()
