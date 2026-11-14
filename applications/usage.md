# Execution and Compilation Guide: AxPike Applications

This guide covers the steps to compile and execute the JPEG and FFT applications using the **AxPike** simulator with **AxRAM**, and how to organize the runs so that their energy can be computed afterwards. Sobel and MLP have their own guides: [sobel/README.md](sobel/README.md) and [mlp/README.md](mlp/README.md).

## 1. Important: Pre-Execution Steps

To ensure the output is not corrupted by system logs, you must handle the Proxy Kernel (PK) output.

### Option A: Modify the Proxy Kernel (Recommended)

You need to silence the bootloader messages in the `axpike-pk` source:

1. Open the file: `axpike-pk/machine/minit.c`
2. **Comment out** the following line:
```c
// printm("bbl loader");
```
3. Recompile the `axpike-pk`.

---

## 2. Compiling the Application

Use the RISC-V cross-compiler to generate a static binary with high-level optimization. From the repository root:

```bash
# JPEG
riscv64-unknown-elf-g++ -O3 -static -o applications/jpeg/src/toojpeg_encoder \
  applications/jpeg/src/main.cpp applications/jpeg/src/toojpeg.cpp

# FFT
riscv64-unknown-elf-g++ -O3 -static -o applications/fft/dominant_freq \
  applications/fft/src/dominant_freq.cpp applications/fft/src/kiss_fft.c -lm
```

* **`-O3`**: High optimization level for performance.
* **`-static`**: Required for bare-metal/simulator execution to include all libraries.

The output paths above match the default `APP_BIN` of each batch runner when the runner is started from its application directory (`applications/jpeg` or `applications/fft`).

---

## 3. Executing a Single Input (manual)

Use this to test one input by hand. To process a whole dataset, use the batch runners in [section 4](#4-running-a-whole-dataset-batch-runners).

Run the simulation using **AxPike** with the following parameters for cache and approximate memory (AXRAM). You can alter the parameters:

```bash
# JPEG: quality factor 100, input is a CSV (width, height, RGB values)
axpike --adele=mem_read_prob:1e-2,linesz:32 --adele-activate=0:AXRAM --dc=128:8:32 --ic=256:4:32 --l2=1024:4:32 \
  pk applications/jpeg/src/toojpeg_encoder 100 < input.csv > output.jpeg

# FFT: input is raw float32 audio (.bin), output is a 24-byte record
axpike --adele=mem_read_prob:1e-3,linesz:32 --adele-activate=0:AXRAM --dc=128:8:32 --ic=256:4:32 --l2=1024:4:32 \
  pk applications/fft/dominant_freq < input.bin > output.bin
```

### Parameter Breakdown:

* **`--adele`**: Sets memory read error probability and line size.
* **`--adele-activate`**: Activates **AXRAM** for the specified memory region.
* **`--dc / --ic / --l2`**: Configures Data, Instruction, and L2 caches (Size:Ways:LineSize).
* **`pk`**: The Proxy Kernel used to load and run the binary.
* **`100 < input.csv`** (JPEG): Passes the quality factor (100) and redirects the input image data.
* **`> output.jpeg`** / **`> output.bin`**: Redirects the application output to a file.

---

## 4. Running a Whole Dataset (batch runners)

Each application has a batch runner in its own directory. It runs the application over an entire dataset, so you do not need to repeat section 3 for each file.

| Application | Runner |
|---|---|
| JPEG | `applications/jpeg/axcept_batch_jpeg.py` (sequential) |
| FFT | `applications/fft/axcept_batch_fft.py` (parallel, `MAX_WORKERS`) |
| Sobel | `applications/sobel/axcept_batch_sobel.py` (see [sobel/README.md](sobel/README.md)) |
| MLP (GENANN) | `applications/mlp/train_exact_and_validate/run_validate_axram.py`, `applications/mlp/train_axram/run_train_axram.py` (see [mlp/README.md](mlp/README.md)) |

For each input file found in the dataset directory, a JPEG/FFT runner:

1. calls `axpike` with AxRAM enabled and the configured `mem_read_prob`;
2. saves the application output (`stdout`) and the execution `stderr` in separate directories, keeping the dataset's folder structure;
3. kills runs that exceed `TIMEOUT_SEC` and records the timeout in the log;
4. applies the 64-byte stdin padding that works around `pk` consuming the first bytes of stdin;
5. prints the progress and a per-file result in the terminal (for JPEG, e.g. `VALID_JPEG`, `AXPIKE_CRASH_DUMP`, `TIMEOUT_OR_EMPTY`). No summary CSV is written; use the analysis scripts of each application for that (`applications/<app>/output_analyzer/`, `quality_metric/`, `acceptability_semantic/`).

### Configuration

The runners are configured by constants at the top of each script. Open the script and set at least:

| Constant | Meaning |
|---|---|
| `ERROR_RATE` | read error probability (`mem_read_prob`), e.g. `"1e-4"`; use `"0"` for the exact baseline |
| `DATASET_DIR` | directory with the input files (`.csv` for JPEG, `.bin` for FFT) |
| `APP_BIN` | path to the RISC-V executable built in section 2 |
| `OUTPUT_JPEG_DIR` / `OUTPUT_BIN_DIR` | where the outputs are saved |
| `LOG_DIR` | where the `stderr` of each run is saved |
| `TIMEOUT_SEC` | per-file time limit |
| `QUALITY` (JPEG) / `MAX_WORKERS` (FFT) | JPEG quality factor / number of parallel runs |

`APP_BIN` and the output directories are relative paths, resolved from the directory you run the script in. If you run a runner from another directory (as in section 5), set `APP_BIN` to an absolute path.

### What each runner writes

All paths are relative to the directory the runner is started from.

| | JPEG | FFT |
|---|---|---|
| Application outputs | `src/dataset_error_rate_<rate>/` | `src/dataset_audio_error_rate_<rate>/` |
| `stderr` logs | `src/logs_error_rate_<rate>/` | `src/logs_audio_error_rate_<rate>/` |
| AxRAM traces (`AXRAM_log_pid*_hart*.log`) | current directory | moved to `src/axpike_artifacts_error_rate_<rate>/` |

---

## 5. Organizing the Runs for the Energy Pipeline

The energy pipeline (`energy/axcept.py`) needs the AxRAM traces of each error rate in its own directory, named after the scenarios of the voltage map, inside one experiment directory:

```text
experiments/jpeg/
├── dataset_error_rate_0/            ← AXRAM_log_pid*_hart*.log of the run with ERROR_RATE = "0"
│   └── src/                         ← outputs and stderr logs (ignored by the pipeline)
├── dataset_error_rate_1e-5/
└── ...
```

Valid names: `dataset_error_rate_0`, `dataset_error_rate_1e-5`, `dataset_error_rate_1e-4`, `dataset_error_rate_1e-3`, `dataset_error_rate_1e-2`, `dataset_error_rate_1e-1` (`python3 energy/axcept.py config` lists them). The pipeline skips every directory named `src`, so **the AxRAM traces must not stay inside `src/`**.

For each error rate, set `ERROR_RATE` in the runner and `APP_BIN` to an absolute path, then run it from the directory of that rate.

**JPEG** (the traces are written straight to the rate directory):

```bash
mkdir -p experiments/jpeg/dataset_error_rate_1e-5
cd experiments/jpeg/dataset_error_rate_1e-5
python3 ../../../applications/jpeg/axcept_batch_jpeg.py      # ERROR_RATE = "1e-5"
cd ../../..
```

**FFT** (the runner moves the traces into `src/`, so move them one level up after the run):

```bash
mkdir -p experiments/fft/dataset_error_rate_1e-5
cd experiments/fft/dataset_error_rate_1e-5
python3 ../../../applications/fft/axcept_batch_fft.py        # ERROR_RATE = "1e-5"
mv src/axpike_artifacts_error_rate_1e-5/AXRAM_log_*.log .
cd ../../..
```

Repeat for every error rate, always including `0` (the baseline). Then compute the energy:

```bash
python3 energy/axcept.py run -w experiments/jpeg
```

See the [energy pipeline guide](../energy/README_pipeline.md) for the output files.

---

## 6. Troubleshooting

If the `output.jpeg` is not opening, ensure that:

1. The `printm` line was successfully commented out.
2. You are using the correct `xxx.csv` format for the encoder.

If the energy pipeline finds no traces for a rate, check that the `AXRAM_log_*.log` files are directly in the rate directory and not inside `src/`.
