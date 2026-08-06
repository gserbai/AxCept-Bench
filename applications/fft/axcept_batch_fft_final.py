######################################################################
# AxCept-Bench (Audio FFT / Dominant Frequency)
# Author: Guilherme Saides Serbai
# Year: 2026
#
# Input expected by the RISC-V application:
#   raw float32 little-endian audio samples, mono, 44100 Hz, via stdin.
#
# Output written by the RISC-V application to stdout:
#   Normal result, 24 bytes:
#     [8 bytes  magic: AXDFREQ1]
#     [int32 LE status]
#     [int32 LE num_samples]
#     [int32 LE num_frames]
#     [float32 LE freq_hz]
#
# Crash behavior:
#   If AxPike crashes, the output .bin may contain the AxPike register dump,
#   usually starting with "z  000000..." and containing "User ... segfault".
#
# Timeout behavior:
#   If the run exceeds TIMEOUT_SEC, this script leaves the output .bin empty.
######################################################################

import shutil
import subprocess
from pathlib import Path

# =========================
# Configuration
# =========================

ERROR_RATE = "1e-1"

# Folder containing raw float32 mono .bin files.
# These should be WITHOUT the 64-byte pk padding if ADD_PK_STDIN_PADDING=True.
DATASET_DIR = Path("/home/guilherme/Music/dataset_iowa_music_bin")

# Output and log folders.
OUTPUT_BIN_DIR = Path(f"./src/dataset_audio_error_rate_{ERROR_RATE}")
LOG_DIR = Path(f"./src/logs_audio_error_rate_{ERROR_RATE}")

# Final RISC-V binary compiled from dominant_freq_axpike_final_nosignal.cpp.
APP_BIN = "./src/dominant_freq"

# pk consumes the first 64 bytes of stdin in your setup.
# Keep True if DATASET_DIR contains clean raw .bin files.
# Set False only if your dataset files are already .pad64.bin.
ADD_PK_STDIN_PADDING = True
PK_STDIN_PADDING_BYTES = 64

# Prevent stuck runs from blocking the whole batch.
# Timeout => output .bin is kept empty.
TIMEOUT_SEC = 1200

# File selection.
INPUT_SUFFIX = ".bin"


def build_axpike_cmd():
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


def stream_input_to_process(proc, input_path: Path):
    """Send optional 64-byte padding and then the raw audio file to proc.stdin."""
    assert proc.stdin is not None

    if ADD_PK_STDIN_PADDING:
        proc.stdin.write(b"\x00" * PK_STDIN_PADDING_BYTES)

    with open(input_path, "rb") as f_in:
        shutil.copyfileobj(f_in, proc.stdin, length=1024 * 1024)

    proc.stdin.close()


def run_one(input_bin: Path, output_bin: Path, output_log: Path):
    cmd = build_axpike_cmd()

    with open(output_bin, "wb") as f_out, open(output_log, "wb") as f_log:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=f_out,
            stderr=f_log,
        )

        try:
            stream_input_to_process(proc, input_bin)
            returncode = proc.wait(timeout=TIMEOUT_SEC)
            return returncode

        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

            # Timeout convention: empty output file.
            output_bin.write_bytes(b"")

            with open(output_log, "ab") as f:
                f.write(b"\nTIMEOUT\n")

            return 124


def run_conversion():
    OUTPUT_BIN_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    input_files = sorted(DATASET_DIR.rglob(f"*{INPUT_SUFFIX}"))

    if not input_files:
        print(f"No {INPUT_SUFFIX} files found in {DATASET_DIR}")
        return

    print(f"Found {len(input_files)} audio files to process.")
    print(f"Error rate: {ERROR_RATE}")
    print(f"PK stdin padding: {ADD_PK_STDIN_PADDING} ({PK_STDIN_PADDING_BYTES} bytes)")
    print(f"Application: {APP_BIN}")

    for input_bin in input_files:
        relative_path = input_bin.relative_to(DATASET_DIR)

        # Avoid duplicating ".pad64" in output names if your input dataset already has it.
        output_bin = OUTPUT_BIN_DIR / relative_path.with_suffix(".bin")
        output_log = LOG_DIR / relative_path.with_suffix(".log")

        output_bin.parent.mkdir(parents=True, exist_ok=True)
        output_log.parent.mkdir(parents=True, exist_ok=True)

        print(f"processing: {relative_path}")
        print(f"  -> Output: {output_bin.relative_to(OUTPUT_BIN_DIR)}")
        print(f"  -> Log:    {output_log.relative_to(LOG_DIR)}")

        try:
            rc = run_one(input_bin, output_bin, output_log)
            if rc != 0:
                print(f"  [!] Non-zero return code: {rc}")

        except Exception as e:
            print(f"  [!] Critical failure on {input_bin.name}: {e}")

    print("\nCompleted!")
    print(f"Outputs in: {OUTPUT_BIN_DIR}")
    print(f"Logs in:    {LOG_DIR}")


if __name__ == "__main__":
    run_conversion()
