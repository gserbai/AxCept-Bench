<p align="center">
  <img src="docs/assets/axcept-bench-banner.svg" alt="AxCept-Bench — Approximate Computing Benchmark" width="900">
</p>

# AxCept-Bench

<p align="center">
  <strong>English</strong> ·
  <a href="README.pt-BR.md">Português (Brasil)</a>
</p>

**AxCept-Bench** is an experimental infrastructure for studying the trade-off between output quality and energy savings in approximate computing.

The project runs RISC-V applications on **AxPike**, injects read faults using the **AxRAM** model, evaluates output degradation, and converts memory accesses into energy estimates using **Ramulator** and **DRAMPower**.

There are currently two main workloads:

- **JPEG:** image compression with crash analysis, SSIM assessment, and semantic quality evaluation;
- **FFT / dominant frequency:** audio processing with crash analysis, frequency-error measurement, and MIDI note matching.

This repository provides the benchmark components and supporting scripts. It does not provide a turnkey experiment. Researchers must choose datasets, error rates, memory configurations, and metrics, and organize results to suit their study.

> Datasets are not included in the repository. Some scripts use hard-coded constants and must be configured before execution.

## Workflow Overview

```text
Dataset
   |
   v
RISC-V Application (JPEG or FFT)
   |
   v
AxPike + AxRAM
   |---------------------------|
   v                           v
Approximate Output        Memory Logs
   |                           |
   v                           v
Quality Analysis          U Y Access Filter
                               |
                               v
                           Ramulator
                               |
                               v
                           DRAMPower
                               |
                               v
                        Energy and Power
```

In a typical experiment:

1. build the workload for RISC-V;
2. choose a dataset and a read error rate;
3. run the workload on AxPike with AxRAM enabled;
4. store workload outputs, standard error logs, and AxRAM traces separately;
5. repeat the execution for all error rates under study;
6. evaluate output stability and quality;
7. process the traces with Ramulator and DRAMPower;
8. compare output quality and energy consumption against the exact baseline.

A common set of probabilities is:

| Scenario | `mem_read_prob` |
|---|---:|
| Exact | `0` |
| Approximate | `1e-5` |
| Approximate | `1e-4` |
| Approximate | `1e-3` |
| Approximate | `1e-2` |
| Approximate | `1e-1` |

For the exact baseline, AxRAM may remain enabled with probability `0`. This makes it possible to generate traces that are comparable to those from the approximate scenarios.

## Repository Layout

```text
applications/           JPEG and FFT workloads
axpike-isa-sim/         AxPike simulator
axpike-pk/              RISC-V Proxy Kernel
ramulator/              memory simulator
DRAMPower-4.1/          energy model used by the current workflow
drampower/              newer DRAMPower version
scripts/benchmarks/     workload runners
scripts/analysis/       stability and quality analyses
scripts/axram/          AxRAM log processing
scripts/ramulator/      Ramulator batch execution
scripts/drampower/      DRAMPower batch execution
scripts/utils/          converters and utilities
```

Keep source code, executables, and results separate. One possible layout is:

```text
applications/jpeg/src/       JPEG sources
applications/fft/src/        FFT sources
bin/jpeg/                     JPEG executable
bin/fft/                      FFT executable
experiments/jpeg/             JPEG results
experiments/fft/              FFT results
```

These names are only suggestions. If you use a different layout, update the paths in the scripts accordingly.

## Dependencies

The workflow was developed for Linux and requires:

- Python 3.10 or newer;
- a C/C++ compiler and Make;
- the `riscv64-unknown-elf` toolchain;
- AxPike;
- Proxy Kernel (`pk`);
- Ramulator;
- DRAMPower 4.1;
- Python libraries required by some analyses.

Basic packages for Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install build-essential autoconf automake libtool pkg-config \
  gawk bison flex texinfo gperf patchutils bc device-tree-compiler \
  libboost-regex-dev libboost-system-dev libmpc-dev libmpfr-dev \
  libgmp-dev zlib1g-dev libexpat1-dev libxerces-c-dev \
  python3 python3-pip python3-venv ffmpeg
```

Libraries used by the analyses:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install numpy pillow scikit-image tqdm matplotlib \
  pandas seaborn torch torchvision
```

The AxRAM trace-extraction and Ramulator/DRAMPower batch scripts use only the Python standard library.

## Submodules

After cloning the project:

```bash
git submodule update --init --recursive
```

## Building the Infrastructure

The commands below are provided for reference. Installation prefixes and toolchain options may vary depending on the laboratory environment.

### Proxy Kernel

```bash
mkdir -p axpike-pk/build
cd axpike-pk/build
../configure --prefix=/path/to/riscv --host=riscv64-unknown-elf
make -j4
make install
cd ../..
```

> Before running the workloads, see the [application build and execution guide](applications/usage.md).
> It explains how to suppress the `bbl loader` message in the Proxy Kernel so that it does not contaminate binary output, and also provides the commands for building and running applications on AxPike.

### AxPike

```bash
mkdir -p axpike-isa-sim/build
cd axpike-isa-sim/build
../configure --prefix=/path/to/riscv
make -j4
make install
cd ../..
```

If the programs are not installed system-wide, add their local paths to the experiment's runtime environment.

### Ramulator

```bash
make -C ramulator -j4
```

### DRAMPower 4.1

```bash
make -C DRAMPower-4.1 -j4 drampower
```

The current batch workflow uses `DRAMPower-4.1/drampower`. The submodule under `drampower/` provides a different command-line interface.

## Building the Workloads

Create a directory for executables outside `src`:

```bash
mkdir -p bin/jpeg bin/fft
```

### JPEG

```bash
riscv64-unknown-elf-g++ -O3 -static \
  -o bin/jpeg/toojpeg_encoder \
  applications/jpeg/src/main.cpp \
  applications/jpeg/src/toojpeg.cpp
```

### FFT

```bash
riscv64-unknown-elf-g++ -O3 -static \
  -o bin/fft/dominant_freq \
  applications/fft/src/dominant_freq.cpp \
  applications/fft/src/kiss_fft.c -lm
```

After building, set `APP_BIN` in the corresponding runner to the executable path. The current runners do not expose every configuration option through the command line.

## JPEG Benchmark

### Input Format

The JPEG workload accepts CSV files. Each file must contain the width and height, followed by `width × height × 3` RGB values.

Minimal example for a 2 × 1 pixel image:

```csv
2,1
255,0,0,0,255,0
```

One possible dataset layout is:

```text
dataset_csv/
└── class/
    └── image.csv
```

The repository does not include a general-purpose image converter for this format.

### Configuration

Before running the benchmark, open:

```text
scripts/benchmarks/jpeg/axcept_batch_jpeg.py
```

Configure at least:

- `ERROR_RATE`;
- `QUALITY`;
- `TIMEOUT_SEC`;
- `DATASET_DIR`;
- `APP_BIN`;
- `OUTPUT_JPEG_DIR`;
- `LOG_DIR`.

The executable may be located at `bin/jpeg/toojpeg_encoder`. Use either an absolute path or a path that is correct relative to the working directory.

### Running

```bash
python3 scripts/benchmarks/jpeg/axcept_batch_jpeg.py
```

The application's stdout is saved as a JPEG file. `stderr` from each run is saved separately. The `AXRAM_log_pid*_hart*.log` files are generated by AxRAM in the working directory.

## FFT Benchmark

### Input Format

The input must be raw `.bin` audio with:

- one channel;
- a 44,100 Hz sample rate;
- little-endian `float32` samples;
- no header.

The AIFF conversion utility is located at:

```text
scripts/utils/convert_iowa_aif_to_bin.py
```

Configure the directories inside the script and run:

```bash
python3 scripts/utils/convert_iowa_aif_to_bin.py
```

### Configuration

Before running the benchmark, open:

```text
scripts/benchmarks/fft/axcept_batch_fft_final_parallel.py
```

Configure at least:

- `ERROR_RATE`;
- `DATASET_DIR`;
- `APP_BIN`;
- `OUTPUT_BIN_DIR`;
- `LOG_DIR`;
- `TIMEOUT_SEC`;
- `MAX_WORKERS`.

The executable may be located at `bin/fft/dominant_freq`.

### Running

```bash
python3 scripts/benchmarks/fft/axcept_batch_fft_final_parallel.py
```

A valid output record is 24 bytes long and starts with the `AXDFREQ1` identifier. The record contains the status, number of samples, number of frames, and dominant frequency in Hz.

## Experiment Organization

Run each error rate in an isolated directory. For example:

```text
experiments/
├── jpeg/
│   ├── dataset_error_rate_0/
│   ├── dataset_error_rate_1e-5/
│   ├── dataset_error_rate_1e-4/
│   ├── dataset_error_rate_1e-3/
│   ├── dataset_error_rate_1e-2/
│   └── dataset_error_rate_1e-1/
└── fft/
    ├── dataset_error_rate_0/
    ├── dataset_error_rate_1e-5/
    ├── dataset_error_rate_1e-4/
    ├── dataset_error_rate_1e-3/
    ├── dataset_error_rate_1e-2/
    └── dataset_error_rate_1e-1/
```

Within each scenario, keep the following separate:

```text
outputs/                     workload output
logs/                        execution stderr
AXRAM_log_pid*_hart*.log     raw AxRAM traces
```

This separation prevents results from being mixed with `applications/*/src` and makes the experiment easier to reproduce.

## Energy Pipeline

### 1. Filter AxRAM Accesses

The extractor selects lines beginning with `U Y` and preserves the scenario structure.

```bash
python3 scripts/axram/extract_uy_structure.py \
  experiments/jpeg \
  --output-dir experiments/jpeg/uy_filtered
```

For FFT, replace the paths:

```bash
python3 scripts/axram/extract_uy_structure.py \
  experiments/fft \
  --output-dir experiments/fft/uy_filtered
```

Empty filtered files indicate traces without `U Y` accesses and must be investigated before simulation.

### 2. Run Ramulator

Example for JPEG:

```bash
python3 scripts/ramulator/run_ramulator_batch.py \
  --input-root experiments/jpeg/uy_filtered \
  --out-root experiments/jpeg/ramulator_results \
  --ramulator ramulator/ramulator \
  --config configs/ramulator/DDR3_2Gb_x16.cfg \
  --jobs 4
```

Test a single input first:

```bash
python3 scripts/ramulator/run_ramulator_batch.py \
  --input-root experiments/jpeg/uy_filtered \
  --out-root experiments/jpeg/ramulator_test \
  --ramulator ramulator/ramulator \
  --config configs/ramulator/DDR3_2Gb_x16.cfg \
  --limit 1 --jobs 1
```

A successful run should produce `DDR3.stats` and at least one `cmd-trace-*.cmdtrace` file.

### 3. Run DRAMPower

The voltage map associates each scenario name and error rate with a voltage and memory specification file.

```bash
python3 scripts/drampower/run_drampower_batch.py \
  --input-root experiments/jpeg/ramulator_results \
  --output-root experiments/jpeg/drampower_results \
  --voltage-map configs/drampower/voltage_map_vendor_b.csv \
  --drampower DRAMPower-4.1/drampower \
  --jobs 4
```

Validate task discovery before running the simulation:

```bash
python3 scripts/drampower/run_drampower_batch.py \
  --input-root experiments/jpeg/ramulator_results \
  --output-root experiments/jpeg/drampower_results \
  --voltage-map configs/drampower/voltage_map_vendor_b.csv \
  --drampower DRAMPower-4.1/drampower \
  --dry-run
```

DRAMPower does not inject faults. It calculates energy from the command traces generated for each scenario.

The main output files are:

| File | Contents |
|---|---|
| `drampower_trace_results.csv` | energy and power per trace |
| `drampower_scenario_summary.csv` | statistics aggregated by scenario |
| `drampower_run_issues.csv` | issues found during execution |
| `drampower_unmapped_scenarios.csv` | scenarios without a corresponding map entry |

The Ramulator configuration, DRAMPower memory specifications (`memspecs`), and voltage map must represent the same memory.

## Quality Analyses

Quality analyses can be run independently of the energy analysis stage.

### JPEG: Stability

```bash
python3 'scripts/analysis/crashes&survivors/crash_analyzer_jpeg.py' \
  /path/to/jpeg/outputs
```

The script classifies empty outputs, crash dumps, and successful (surviving) outputs.

### JPEG: SSIM

```bash
python3 scripts/analysis/structuraljpeg/structuraljpeg/compute_jpeg_ssim.py \
  --perfect-dir /path/to/original_images \
  --error-dir /path/to/jpeg/outputs
```

The computation matches images by relative path and considers only decodable files.

### JPEG: Semantic Quality

The scripts are located in:

```text
scripts/analysis/semanticjpeg/acceptabilityjpeg/
```

The workflow includes baseline training, separate training for each error rate, validation, and plot generation. Dataset, model, and output paths are defined within the files themselves.

Main commands:

```bash
python3 scripts/analysis/semanticjpeg/acceptabilityjpeg/codetrain/r50t_tif.py
python3 scripts/analysis/semanticjpeg/acceptabilityjpeg/codetrain/r50t.py
```

For validation:

```bash
cd scripts/analysis/semanticjpeg/acceptabilityjpeg/codeval
python3 val.py
python3 geradorgraficonn.py
```

The current scripts assume datasets are organized by class. Training can use a GPU, but one is not required.

### FFT: Stability

```bash
python3 'scripts/analysis/crashes&survivors/fft_output_classifier_crash.py' \
  --base-dir /path/to/fft_experiments \
  --rates 1e-5 1e-4 1e-3 1e-2 1e-1 \
  --out /path/to/fft_crash_summary.csv
```

The CSV aggregates valid results, crash dumps, errors, timeouts, invalid outputs, and missing files.

### FFT: Structural Quality and Semantic Acceptability

The analysis reads the `.bin` files produced by the benchmark directly. There is
no need to generate `summary.csv` before this stage.

By default, the exact baseline and the runs for each approximate error rate must be under the
same base directory:

```text
fft_experiments/
├── dataset_iowa_music_exact/
│   └── src/
│       └── dataset_audio_error_rate_exact/
│           └── <class>/<audio>.bin
├── dataset_iowa_music_1e-5/
│   └── src/
│       └── dataset_audio_error_rate_1e-5/
│           └── <class>/<audio>.bin
└── dataset_iowa_music_1e-4/
    └── src/
        └── dataset_audio_error_rate_1e-4/
            └── <class>/<audio>.bin
```

The baseline can also use the names `dataset_iowa_music_0` and
`dataset_audio_error_rate_0`.

Run:

```bash
python3 'scripts/analysis/acceptability&qualitystructural_fft/analyze_fft_acceptability_two_plots_one_figure.py' \
  --base-dir /path/to/fft_experiments \
  --rates 1e-5 1e-4 1e-3 1e-2 1e-1 \
  --out-dir /path/to/fft_acceptability
```

If the exact-output directory is located elsewhere, specify it directly:

```bash
python3 'scripts/analysis/acceptability&qualitystructural_fft/analyze_fft_acceptability_two_plots_one_figure.py' \
  --base-dir /path/to/fft_experiments \
  --exact-dir /path/to/dataset_audio_error_rate_exact \
  --out-dir /path/to/fft_acceptability
```

The script interprets the first 24 bytes of each file using the
`<8siiif` format: magic value `AXDFREQ1`, status, number of samples, number of frames,
and frequency in Hz. Each approximate output is matched with its exact output
using the full relative path, such as `piano/audio.bin`. Files with the same name
in different classes are not mixed.

Only pairs in which both outputs are classified as `OK_RESULT` or `RESULT_WITH_CRASH_DUMP` and have
valid frequencies are included in the MAPE and MIDI calculations. Missing
files, empty files, error statuses, and invalid outputs are recorded but excluded
from these calculations. Approximate files without an exact match are also
reported.

The analysis generates:

```text
fft_metrics_by_rate.csv
fft_all_comparisons_vs_exact.csv
fft_unmatched_files.csv
fft_two_plots_one_figure_publication.pdf
fft_two_plots_one_figure_publication.png
```

The CSV files are analysis results, not inputs. The figures require `numpy` and
`matplotlib`.

## Main Scripts

| Script | Function |
|---|---|
| `scripts/benchmarks/jpeg/axcept_batch_jpeg.py` | runs the JPEG workload |
| `scripts/benchmarks/fft/axcept_batch_fft_final_parallel.py` | runs the FFT workload in parallel |
| `scripts/utils/convert_iowa_aif_to_bin.py` | converts AIFF files to raw audio |
| `scripts/axram/extract_uy_structure.py` | filters `U Y` accesses |
| `scripts/ramulator/run_ramulator_batch.py` | runs traces through Ramulator |
| `scripts/drampower/run_drampower_batch.py` | computes and aggregates energy consumption |
| `scripts/analysis/structuraljpeg/structuraljpeg/compute_jpeg_ssim.py` | computes SSIM |
| `scripts/analysis/crashes&survivors/crash_analyzer_jpeg.py` | analyzes JPEG stability |
| `scripts/analysis/crashes&survivors/fft_output_classifier_crash.py` | analyzes FFT stability |
| `scripts/analysis/acceptability&qualitystructural_fft/analyze_fft_acceptability_two_plots_one_figure.py` | analyzes FFT acceptability and structural quality and generates CSV files and plots |
| `scripts/analysis/semanticjpeg/acceptabilityjpeg/codetrain/r50t.py` | trains and evaluates a ResNet-50 using approximate JPEG outputs |
| `scripts/analysis/semanticjpeg/acceptabilityjpeg/codetrain/r50t_tif.py` | trains and evaluates the reference ResNet-50 using TIFF images |
| `scripts/analysis/semanticjpeg/acceptabilityjpeg/codeval/config.py` | configures models, datasets, and outputs for JPEG semantic validation |
| `scripts/analysis/semanticjpeg/acceptabilityjpeg/codeval/val.py` | validates the ResNet-50 models on the JPEG datasets and consolidates the results |
| `scripts/analysis/semanticjpeg/acceptabilityjpeg/codeval/geradorgraficonn.py` | generates plots for JPEG semantic validation |

For scripts that provide a command-line interface, view the available options with:

```bash
python3 path/to/script.py --help
```

## Experimental Guidelines

- Use the same dataset across all error rates.
- Record the code version, parameters, and random seeds when applicable.
- Do not mix outputs from different error rates.
- Preserve the raw AxRAM traces.
- Validate a sample before starting large batches.
- Compare all results against the exact baseline.
- Record timeouts and crashes; do not silently discard failures.
- Keep the Ramulator configuration, DRAMPower memspecs, and voltage map consistent.
- Increase parallelism only after validating a small run.

## Common Issues

### `axpike` or `pk` Not Found

Check the installation or the binary paths in the environment used to run the benchmark.

### Dataset or Application Not Found

Check `DATASET_DIR`, `APP_BIN`, and the directory from which the runner was started. Relative paths are resolved from that directory.

### Output Contains Proxy Kernel Text

The application's stdout contains binary data. Check the Proxy Kernel configuration and the handling of the 64-byte padding used by the runners.

### No AxRAM Log Was Generated

Confirm that:

- AxRAM is enabled with `--adele-activate=0:AXRAM`;
- the error rate is set in `mem_read_prob`;
- filenames follow the `AXRAM_log_pid*_hart*.log` format;
- the execution directory is writable.

### The Extractor Does Not Find Any Accesses

Check whether the traces contain lines beginning with `U Y` and whether the specified directory contains the raw logs.

### Ramulator Does Not Generate Command Traces

Check `stdout.txt`, `stderr.txt`, `DDR3.stats`, and the configuration used. Command trace recording must be enabled.

### DRAMPower Does Not Find Any Tasks

Check:

- `cmd-trace-*.cmdtrace` files;
- scenario names;
- entries enabled in the voltage map;
- whether the specified memory specification files exist;
- the path to the `DRAMPower-4.1/drampower` executable.

## Scope

AxCept-Bench is a foundation for experiments, not a package of ready-made results. Each study should document its experimental decisions, datasets, hardware configurations, metrics, and acceptability criteria.
