# AxCept-Bench

**AxCept-Bench** is an experimental infrastructure for studying the trade-off between output acceptability and energy savings when an application runs on approximate DRAM.

Each RISC-V application runs on the **AxPike** simulator, where the **AxRAM** model injects read errors at a given rate (`mem_read_prob`). The approximate outputs are then judged for acceptability, and the memory accesses recorded by AxRAM are converted into DRAM energy with **Ramulator** and **DRAMPower**, using a lower supply voltage for each error rate.

This repository provides the benchmark components and supporting scripts. It does not provide a turnkey experiment: datasets are not included, and some scripts use hard-coded constants that must be configured before execution.

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
Approximate Output        AxRAM Logs
   |                           |
   v                           v
Acceptability Analysis    Energy Pipeline (energy/axcept.py)
                          U Y filter -> Ramulator -> DRAMPower
                               |
                               v
                        Energy and Power
```

Each application is run once per read error rate. A common set is `0` (exact baseline), `1e-5`, `1e-4`, `1e-3`, `1e-2` and `1e-1`. The baseline keeps AxRAM enabled with probability `0`, so its traces are comparable to the approximate ones.

## Repository Layout

```text
applications/       workloads (JPEG, FFT, Sobel, MLP): sources, batch runner and acceptability analyses of each
configs/            Ramulator config, DRAMPower memspecs and voltage map, pipeline.toml
energy/             energy pipeline (axcept.py) and energy analyses
axpike-isa-sim/     AxPike simulator
axpike-pk/          RISC-V Proxy Kernel
ramulator/          memory simulator
DRAMPower-4.1/      energy model used by the pipeline
drampower/          newer DRAMPower version (different CLI, not used by the scripts)
```

How to build and run each application: [JPEG and FFT](applications/usage.md), [Sobel](applications/sobel/README.md), [MLP](applications/mlp/README.md).

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

Only needed by the acceptability analyses. The energy pipeline uses only the standard library.

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

The pipeline is the same for every application: it only reads the `AXRAM_log_pid*_hart*.log` files written by AxPike. Put the logs of each error rate in its own directory, named after the scenarios of the voltage map (`dataset_error_rate_0`, `dataset_error_rate_1e-5`, ...), inside one experiment directory. The [application guide](applications/usage.md#5-organizing-the-runs-for-the-energy-pipeline) shows how to do this with the batch runners.

```bash
python3 energy/axcept.py config                     # check tools, memspecs and scenarios
python3 energy/axcept.py run -w experiments/<app>   # extract + ramulator + drampower
```

Results are written inside the experiment directory; the main one is `drampower_results/drampower_scenario_summary.csv` (one row per error rate). The [energy pipeline guide](energy/README_pipeline.md) describes the input layout, every option, each stage on its own and the output files.

## Troubleshooting

- **Output contains Proxy Kernel text:** the `bbl loader` message was not silenced. See the [application guide](applications/usage.md).
- **No AxRAM log generated:** confirm AxRAM is enabled with `--adele-activate=0:AXRAM`, that `mem_read_prob` is set, and that the execution directory is writable.
- **An error rate is missing from the energy results:** check that its directory name matches the voltage map and that the logs are not inside a `src/` directory.
- **Unsure about a script's options:** run `python3 path/to/script.py --help`.
