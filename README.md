# AxCept-Bench

**AxCept-Bench** is an experimental infrastructure for studying the trade-off between output quality, both structural and semantic, and energy savings in approximate computing.

RISC-V applications run on **AxPike**, read faults are injected by the **AxRAM** model, output degradation is evaluated, and memory accesses are converted into energy estimates using **Ramulator** and **DRAMPower**.

This repository provides the benchmark components and supporting scripts. It does not provide a turnkey experiment: datasets are not included, and some scripts use hard-coded constants that must be configured before execution.

## Applications

| Application | Location | Batch runner | Quality evaluation |
|---|---|---|---|
| JPEG | `applications/jpeg` | `applications/jpeg/axcept_batch_jpeg.py` | crash analysis, SSIM, semantic quality, plate recognition |
| FFT (dominant frequency) | `applications/fft` | `applications/fft/axcept_batch_fft.py` | crash analysis, frequency error, MIDI note matching |
| Sobel (edge detection) | `applications/sobel` | `applications/sobel/axcept_batch_sobel.py` | plate recognition (OpenALPR) |
| MLP (GENANN neural network) | `applications/mlp` | `run_validate_axram.py`, `run_train_axram.py` | classification accuracy |

To build and run JPEG and FFT on AxPike, see the [application build and execution guide](applications/usage.md). Sobel and MLP have their own guides: [applications/sobel/README.md](applications/sobel/README.md) and [applications/mlp/README.md](applications/mlp/README.md).

## Workflow

```text
Dataset
   |
   v
RISC-V Application
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

Each application is run once per read error rate (`mem_read_prob`). A common set is `0` (exact baseline), `1e-5`, `1e-4`, `1e-3`, `1e-2` and `1e-1`. The baseline can keep AxRAM enabled with probability `0`, so its traces are comparable to the approximate ones.

## Repository Layout

```text
applications/           workloads (JPEG, FFT, Sobel, MLP/GENANN): batch runner, sources and analyses of each
axpike-isa-sim/         AxPike simulator
axpike-pk/              RISC-V Proxy Kernel
ramulator/              memory simulator
DRAMPower-4.1/          energy model used by the current workflow
drampower/              newer DRAMPower version (different CLI, not used by the scripts)
configs/                Ramulator configuration, DRAMPower memspecs / voltage map, pipeline.toml
energy/axcept.py        energy pipeline entry point (AxRAM logs -> Ramulator -> DRAMPower)
energy/axram/           AxRAM log processing
energy/ramulator/       Ramulator batch execution
energy/drampower/       DRAMPower batch execution
energy/analysis/        energy analyses (bootstrap confidence intervals)
```

Each application keeps its own analyses next to its runner. Not every application has every folder, and MLP has its own layout (see [its README](applications/mlp/README.md)):

```text
applications/<app>/
├── README.md                  how to build and run the application
├── axcept_batch_<app>.py      batch runner (one run per error rate)
├── src/                       application sources
├── utils/                     dataset converters
├── output_analyzer/           crash / timeout / valid output classification
├── quality_metric/            quality metrics (e.g. SSIM)
└── acceptability_semantic/    semantic acceptability (e.g. classifier, plate recognition)
```

## Setting Up the Environment

The workflow was developed for Linux and requires:

- Python 3.11 or newer;
- a C/C++ compiler and Make;
- the [RISC-V GNU toolchain](https://github.com/riscv-collab/riscv-gnu-toolchain) (`riscv64-unknown-elf`);
- AxPike, the Proxy Kernel (`pk`), Ramulator and DRAMPower 4.1 (built below).

### System packages (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install build-essential autoconf automake libtool pkg-config \
  gawk bison flex texinfo gperf patchutils bc device-tree-compiler \
  libboost-regex-dev libboost-system-dev libmpc-dev libmpfr-dev \
  libgmp-dev zlib1g-dev libexpat1-dev libxerces-c-dev \
  python3 python3-pip python3-venv ffmpeg
```

### Python libraries

Only needed by the quality analyses. The AxRAM, Ramulator and DRAMPower scripts use the standard library.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install numpy pillow scikit-image tqdm matplotlib \
  pandas seaborn torch torchvision
```

### Submodules

```bash
git submodule update --init --recursive
```

### Building the infrastructure

Set `RISCV` to the toolchain installation prefix. Prefixes and options may vary depending on your environment.

**Proxy Kernel**

```bash
mkdir -p axpike-pk/build
cd axpike-pk/build
../configure --prefix=$RISCV --host=riscv64-unknown-elf
make -j
make install
cd ../..
```

> Applications write binary data to stdout, so the `bbl loader` message printed by the Proxy Kernel would corrupt it. Silence it before building `pk`; the steps are in the [application guide](applications/usage.md).

**AxPike**

```bash
mkdir -p axpike-isa-sim/build
cd axpike-isa-sim/build
../configure --prefix=$RISCV
make -j4
make install
cd ../..
```

**Ramulator**

```bash
make -C ramulator -j4
```

**DRAMPower 4.1**

```bash
make -C DRAMPower-4.1 -j4 drampower
```

## Energy Pipeline

This stage is the same for every application: it only reads the `AXRAM_log_pid*_hart*.log` traces written by AxPike. Put the traces of each error rate in its own directory, named after the scenarios of the voltage map (`dataset_error_rate_0`, `dataset_error_rate_1e-5`, ...), inside one experiment directory. The [application guide](applications/usage.md#5-organizing-the-runs-for-the-energy-pipeline) shows how to do this with the batch runners.

### Single command

`energy/axcept.py` runs the three steps below with the tools, Ramulator config and voltage map taken from [`configs/pipeline.toml`](configs/pipeline.toml), so only the experiment directory has to be given:

```bash
python3 energy/axcept.py config                          # show config, validate memspecs, list scenarios
python3 energy/axcept.py run -w experiments/<app>        # extract + ramulator + drampower
python3 energy/axcept.py ramulator -w experiments/<app> --limit 1 --jobs 1   # one stage only
python3 energy/axcept.py drampower -w experiments/<app> --dry-run
```

Results go to `experiments/<app>/{uy_filtered,ramulator_results,drampower_results}`. Use `--logs PATH` if the raw AxRAM logs are not inside the workdir, and `--profile NAME` to select another memory profile defined in `pipeline.toml` (a profile pairs a Ramulator config with a voltage map). The stages can also be called from Python: `load_settings`, `stage_extract`, `stage_ramulator`, `stage_drampower`.

See the [energy pipeline guide](energy/README_pipeline.md) for the expected input layout (one `dataset_error_rate_*` directory per error rate), the output files and a worked example.

### Step by step

**1. Filter AxRAM accesses** (keeps lines starting with `U Y`; empty filtered files must be investigated):

```bash
python3 energy/axram/extract_uy_structure.py experiments/<app> \
  --output-dir experiments/<app>/uy_filtered
```

**2. Run Ramulator** (add `--limit 1 --jobs 1` to test a single input first). A successful run produces `DDR3.stats` and at least one `cmd-trace-*.cmdtrace`:

```bash
python3 energy/ramulator/run_ramulator_batch.py \
  --input-root experiments/<app>/uy_filtered \
  --out-root experiments/<app>/ramulator_results \
  --ramulator ramulator/ramulator \
  --config configs/ramulator/DDR3_2Gb_x16.cfg \
  --jobs 4
```

**3. Run DRAMPower** (add `--dry-run` to validate task discovery first):

```bash
python3 energy/drampower/run_drampower_batch.py \
  --input-root experiments/<app>/ramulator_results \
  --output-root experiments/<app>/drampower_results \
  --voltage-map configs/drampower/voltage_map_vendor_b.csv \
  --drampower DRAMPower-4.1/drampower \
  --jobs 4
```

DRAMPower does not inject faults; it computes energy from the command traces of each scenario. Results are written to `drampower_trace_results.csv` (per trace) and `drampower_scenario_summary.csv` (per scenario), with problems listed in `drampower_run_issues.csv` and `drampower_unmapped_scenarios.csv`.

The Ramulator configuration, DRAMPower memspecs and voltage map must describe the same memory.

## Troubleshooting

- **Output contains Proxy Kernel text:** the `bbl loader` message was not silenced. See the [application guide](applications/usage.md).
- **No AxRAM log generated:** confirm AxRAM is enabled with `--adele-activate=0:AXRAM`, that `mem_read_prob` is set, and that the execution directory is writable.
- **DRAMPower finds no tasks:** check that `cmd-trace-*.cmdtrace` files exist, that scenario names match the voltage map, and that the memspec files exist.
- **Unsure about a script's options:** run `python3 path/to/script.py --help`.
