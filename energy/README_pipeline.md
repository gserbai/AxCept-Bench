# Energy Pipeline (`axcept.py`)

`energy/axcept.py` computes the DRAM energy of an application at each error rate. It runs three steps in order:

1. **extract**: keeps only the `U Y` lines of the AxRAM logs ([`axram/extract_uy_structure.py`](axram/extract_uy_structure.py)).
2. **ramulator**: simulates the memory accesses and produces DRAM command traces ([`ramulator/run_ramulator_batch.py`](ramulator/run_ramulator_batch.py)).
3. **drampower**: computes the energy of each command trace at the voltage of its error rate ([`drampower/run_drampower_batch.py`](drampower/run_drampower_batch.py)).

The pipeline works with any application (JPEG, FFT, ...). It only reads the `AXRAM_log_pid*_hart*.log` files that AxPike writes when AxRAM is enabled.

## 1. Prepare the input

Run the application once per error rate, each rate from its own directory, and put these directories inside one experiment directory. [applications/usage.md, section 5](../applications/usage.md#5-organizing-the-runs-for-the-energy-pipeline) shows the commands for JPEG and FFT. Note that the FFT runner moves the traces into `src/`, so they must be moved back to the rate directory.

```text
experiments/jpeg/                         ← experiment directory (passed with -w)
├── dataset_error_rate_0/                 ← run with mem_read_prob:0 (baseline, required)
│   ├── AXRAM_log_pid100_hart0.log        ← one log per dataset input
│   ├── AXRAM_log_pid101_hart0.log
│   └── ...
│   └── src/                              ← application outputs / stderr logs (ignored)
├── dataset_error_rate_1e-5/              ← run with mem_read_prob:1e-5
│   └── AXRAM_log_pid*_hart*.log
└── ...
```

Only files named `AXRAM_log_pid*_hart*.log` are read. Other files (application outputs, stderr logs) can stay next to them. Directories named `src` are skipped entirely, so do not put the AxRAM logs inside one.

Rules:

- **The directory names must match the `scenario` column of the voltage map** ([`configs/drampower/voltage_map_vendor_b.csv`](../configs/drampower/voltage_map_vendor_b.csv)). The name decides which voltage and memspec are used:

  | Directory | `mem_read_prob` | Voltage | Memspec |
  |---|---|---|---|
  | `dataset_error_rate_0` | 0 | 1.350 V | `..._1p350V.xml` |
  | `dataset_error_rate_1e-5` | 1e-5 | 1.150 V | `..._1p150V.xml` |
  | `dataset_error_rate_1e-4` | 1e-4 | 1.125 V | `..._1p125V.xml` |
  | `dataset_error_rate_1e-3` | 1e-3 | 1.100 V | `..._1p100V.xml` |
  | `dataset_error_rate_1e-2` | 1e-2 | 1.075 V | `..._1p075V.xml` |
  | `dataset_error_rate_1e-1` | 1e-1 | 1.050 V | `..._1p050V.xml` |

  Run `python3 energy/axcept.py config` to print this table from the current configuration. The match is exact text: variations such as `dataset_audio_error_rate_1e-5`, `error_rate_1e-5` or `dataset_error_rate_1E-5` are not recognized. Directories with other names are ignored and listed in `drampower_unmapped_scenarios.csv`; `python3 energy/axcept.py drampower -w <dir> --dry-run` shows which directories were found and which were ignored before anything runs.
- **Always include `dataset_error_rate_0`.** It is the baseline for the energy and power reduction columns. Without it those columns are empty.
- You do not need every rate. Missing rates are listed in `drampower_run_issues.csv` and the others are processed normally.
- If the logs are somewhere else, do not move them: pass that directory with `--logs PATH`. It must have the same `dataset_error_rate_*` subdirectories.

## 2. Run

```bash
python3 energy/axcept.py config                                # check tools, memspecs and scenarios
python3 energy/axcept.py run -w experiments/jpeg --limit 1     # quick test: 1 trace per stage/scenario
python3 energy/axcept.py run -w experiments/jpeg               # full run
```

Each step can also be run on its own: `extract`, `ramulator`, `drampower` (with `-w`). Runs can be resumed: traces that already have output are skipped, so running the same command again after an interruption continues where it stopped. Use `--force` to recompute everything, for example after changing a config.

| Option | Meaning |
|---|---|
| `-w, --workdir PATH` | experiment directory (input and output) |
| `--logs PATH` | raw AxRAM logs, if not inside the workdir |
| `--jobs N` | parallel Ramulator/DRAMPower runs (default from `pipeline.toml`) |
| `--limit N` | test mode: Ramulator runs N traces, DRAMPower N traces per scenario |
| `--force` | re-run traces that already have output |
| `--scenarios A B` | DRAMPower: only these scenarios |
| `--dry-run` | DRAMPower: list the tasks without running |
| `--profile NAME` | memory profile from `pipeline.toml` |

## 3. Output

```text
experiments/jpeg/
├── dataset_error_rate_*/            ← input, not modified
├── uy_filtered/                     ← step 1: "U Y" lines, same structure as the input
├── ramulator_results/               ← step 2
│   ├── ramulator_batch_summary.csv  ← status of each trace
│   └── dataset_error_rate_*/AXRAM_log_pid*_hart*/
│       ├── DDR3.stats
│       └── cmd-trace-chan-0-rank-0.cmdtrace
└── drampower_results/               ← step 3: energy
    ├── drampower_scenario_summary.csv   ← one row per error rate (main result)
    ├── drampower_trace_results.csv      ← one row per log
    ├── drampower_run_issues.csv         ← problems found
    ├── drampower_unmapped_scenarios.csv ← directories not in the voltage map
    └── runs/<scenario>/<trace>/drampower.out   ← raw DRAMPower output
```

**`drampower_scenario_summary.csv`** is the main result, with one row per error rate. Example (main columns):

| scenario | voltage_v | n_ok | energy_mean_pj | power_mean_mw | energy_reduction_vs_dataset_error_rate_0_percent |
|---|---|---|---|---|---|
| dataset_error_rate_0 | 1.350 | 1 | 113569909727.62 | 552.45 | 0.0 |
| dataset_error_rate_1e-5 | 1.150 | 1 | 96744737916.12 | 470.61 | 14.81 |

- `n_ok` / `n_failed`: logs of that rate that were or were not processed successfully.
- `energy_*_pj` / `power_*_mw`: mean, standard deviation and 95% confidence interval over the logs of that rate.
- `*_reduction_vs_dataset_error_rate_0_percent`: reduction relative to the baseline.

**`drampower_trace_results.csv`** has one row per log: status, voltage, total energy, average power, command counts (ACT, RD, WR, PRE, REF) and the energy of each component.

## 4. Configuration

Tools and memory configuration come from [`configs/pipeline.toml`](../configs/pipeline.toml). A profile pairs a Ramulator config with a voltage map, and the voltage map points to the memspecs. Both must describe the same memory. To use another memory, add a profile and select it with `--profile`. One-off overrides: `--ramulator-config PATH`, `--voltage-map PATH`.

## 5. Troubleshooting

- **`n_ok` is 0 or a status is `ERROR`:** read `drampower_results/runs/<scenario>/<trace>/drampower.err`, or `stderr.txt` in `ramulator_results`.
- **`error while loading shared libraries`:** the Ramulator or DRAMPower binary is missing a system library. Rebuild it or install the library (for example `libxerces-c`).
- **A rate is missing from the summary:** check the directory name against the voltage map and look at `drampower_unmapped_scenarios.csv` and `drampower_run_issues.csv`.
- **Empty filtered file (`[WARN] no U Y lines`):** that run produced no AxRAM accesses. Check that it used `--adele-activate=0:AXRAM`.

The stages can also be called from Python: `load_settings`, `stage_extract`, `stage_ramulator`, `stage_drampower` in `energy/axcept.py`.
