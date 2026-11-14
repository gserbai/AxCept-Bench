# MLP Workload for AxCept-Bench

**Evaluating a neural network trained and executed under approximate DRAM (AxRAM) on RISC-V.**

**Author:** Guilherme Saides Serbai (UTFPR Campo Mourão) · **Advisor:** Prof. Dr. João Fabrício Filho · **Year:** 2026

This is the MLP (multilayer perceptron) workload of **AxCept-Bench**. The network uses the [GENANN](https://github.com/codeplea/genann) C library and is used to study two independent questions:

1. **Inference under AxRAM.** How does the accuracy of a trained network degrade when its forward pass reads weights and activations from approximate DRAM?
2. **Training under AxRAM.** What happens when the network is *trained* entirely inside approximate memory, and how does that model behave when validated under the same or other fault rates?

Every experiment runs the RISC-V kernel inside **AxPIKE** (Spike-based simulator with the **AxRAM** fault model), classifies how each execution ended, and keeps the AxRAM traces used to estimate DRAM energy with Ramulator and DRAMPower.

## Requirements

| Tool | Used for |
|---|---|
| `gcc`, `make` | Host (exact) binaries |
| `riscv64-unknown-elf-g++` | Kernels that run under AxPIKE |
| `axpike` and `pk` (on `PATH`) | Running the kernels with AxRAM fault injection |
| Python 3 + `scikit-learn` | Dataset export (installed by `make` if missing) |
| Ramulator, DRAMPower 4.1 | Energy estimation, through the AxCept-Bench energy pipeline |

## Quick start

Run everything from `applications/mlp/`:

```bash
make all                     # dataset, host and RISC-V binaries
make init_weights            # reproducible initial weights (needed for training under AxRAM)
python3 train_sweep.py       # train one model per fault probability under AxRAM
python3 validate_matrix.py   # validate every model at every fault probability under AxRAM
python3 genann_report.py     # regenerate every reported number (read-only)
```

Validation runs one short AxPIKE invocation per test sample; training runs the whole training loop in a single long invocation (see [Training under AxRAM](#training-under-axram) before launching long runs).

## Configuration

| Parameter | Value |
|---|---|
| Dataset | scikit-learn `digits` (handwritten digits 0–9, 8×8 pixels, scaled to [0, 1]) |
| Split | 1,437 training / 360 test samples (80/20, stratified, seed 42) |
| Architecture | 64 inputs → 2 hidden layers × 80 units → 10 outputs (12,490 weights, ~98 KB) |
| Activation | GENANN cached sigmoid (hidden and output layers) |
| Learning rate | 1.0 |
| Epochs | 10 (`train_sweep.py`) |
| Fault probabilities | `0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1` per DRAM read |
| Caches | L1-D 32 KB (128 sets × 8 ways) · L1-I 32 KB (256 × 4) · L2 128 KB (1024 × 4) · 32-byte lines |

The architecture (`HIDDEN_UNITS`, `HIDDEN_LAYERS`) and the initial-weights seed (`INIT_SEED`) are set at the top of the `Makefile`. Every kernel is launched as:

```bash
axpike --adele=mem_read_prob:<rate>,linesz:32 --adele-activate=0:AXRAM \
       --dc=128:8:32 --ic=256:4:32 --l2=1024:4:32 pk <kernel>
```

## Host vs. RISC-V

| | Runs on | Compiler | Exact or approximate |
|---|---|---|---|
| **host** (`*_host.c`) | local machine | `gcc` | always exact, never touches AxPIKE/AxRAM |
| **riscv** (`*_axpike.cpp`) | `axpike` + `pk` | `riscv64-unknown-elf-g++` | under AxRAM: what the experiment measures |

RISC-V build flags: `RV_FLAGS = -O3 -static -fpermissive -DNDEBUG`.

- `-fpermissive` is required: `g++` compiles `genann.c` as C++, and the library has two `malloc()` calls without a cast. The flag turns this into a warning without modifying the library.
- `-DNDEBUG` disables `assert()` **only on the RISC-V side**. The GENANN sigmoid has `assert(!isnan(a))`; an AxRAM bit flip can turn a weight into NaN or an extreme value, and the assertion would abort every faulty training run on the first NaN. With the flag, the run completes and the model is saved; a numerically unusable model (NaN/Inf weights, accuracy near chance) is the experimental outcome for that probability, not a bug. Genuine segfaults are not affected. The host build keeps assertions enabled. After changing `RV_FLAGS`, rebuild with `make clean && make riscv`.

## File map

```
applications/mlp/
├── lib/                          GENANN (codeplea/genann, zlib, unmodified)
├── dataset/export_digits.py      exports dataset_digits/train.bin and test.bin (host)
├── train_exact_and_validate/     validation under AxRAM (+ optional host training)
│   ├── train_exact_host.c        optional: trains exactly on the host (model.bin)
│   ├── eval_exact_host.c         exact accuracy of any model file on test.bin
│   ├── validate_kernel_axpike.cpp  RISC-V: one forward pass for one sample
│   └── run_validate_axram.py     driver: one invocation per test sample
├── train_axram/                  PATH 2 — training under AxRAM
│   ├── gen_init_weights_host.c   reproducible init_weights.bin
│   ├── train_kernel_axpike.cpp   RISC-V: whole training loop in one invocation
│   └── run_train_axram.py        driver: builds the payload, runs axpike once
├── train_sweep.py                training under AxRAM for every fault probability
├── validate_matrix.py            validation of every trained model at every fault probability
├── reclassify_validation.py      re-classifies saved validation records (no new AxPIKE runs)
├── genann_report.py              read-only report with every number used in the paper
├── inspect_train_record.py       manual diagnosis of one training .stdout.raw
└── Makefile
```

Binaries and runtime artifacts (`dataset_digits/`, models, CSVs, logs, records) are written to `applications/mlp/`; the subfolders only hold source code. **Run every script and binary from `applications/mlp/`.**

`make all` runs every step except AxPIKE. Individual targets: `setup`, `dataset`, `host` (`train_exact_host`, `eval_exact_host`, `gen_init_weights_host`), `riscv` (`validate_kernel_axpike`, `train_kernel_axpike`), `init_weights`, and the optional `train` (host-trained `model.bin`, not used by the experiment). `make clean` removes the binaries; `make distclean` also removes the dataset, models and initial weights.

## Inference under AxRAM

`validate_matrix.py` runs this for every model and rate; a single run is:

```bash
python3 train_exact_and_validate/run_validate_axram.py --error-rate 1e-3 --model model_trained_axram_err0.bin
```

The driver runs **one `axpike` invocation per test sample** (360 per fault probability). Each is a single forward pass, so samples are independent: a crash on one does not affect the others.

| Option | Meaning |
|---|---|
| `--error-rate` | Per-read fault probability (`0` = fault-free run through the simulator, useful as energy baseline) |
| `--model` | Model to validate (default `model_trained_axram_err0.bin`; any file with the `model.bin` layout) |
| `--train-mode {exact,axram}` | Labels the row appended to `--results-csv` |
| `--tag` | Suffix for output names, so different models at the same rate do not overwrite each other |
| `--results-csv` | Cumulative summary table (default `experiment_results.csv`) |

Outputs (`[_<tag>]` is omitted when no tag is given and `--train-mode` is `exact`):

| Output | Content |
|---|---|
| `digits_summary_err<rate>[_<tag>].csv` | Per sample: `index`, `true_label`, `predicted`, `correct`, `class`, `recoverable`, `elapsed_sec` |
| `digits_records_err<rate>[_<tag>]/` · `digits_logs_err<rate>[_<tag>]/` | Raw stdout / stderr of every invocation |
| `axpike_artifacts_err<rate>[_<tag>]/` | AXRAM logs, counters and energy CSVs (moved automatically) |
| `experiment_results.csv` | One row per run: `error_rate`, `train_mode`, `tag`, `model_file`, `accuracy_all_pct`, `accuracy_recoverable_pct`, `n_recoverable`, `n_recovered_after_crash`, `n_total`, `summary_csv` |

The driver reports the accuracy over recoverable outputs and over **all** samples (unrecoverable = wrong). The latter is the number to compare against the baseline.

## Training under AxRAM

```bash
python3 train_sweep.py                          # all configured fault probabilities
python3 train_sweep.py --error-rates 1e-4,1e-2  # a subset
python3 train_sweep.py --force                  # re-run rates that already finished
```

Training uses a **single** `axpike` invocation for the whole run. It is sequential and stateful (the weights updated on sample *N* feed the gradient of sample *N+1*), so the weights must live continuously in approximate memory, as they would on real AxRAM hardware; one process per step would reload them from the host between steps.

For each fault probability, `train_sweep.py` calls `train_axram/run_train_axram.py` and:

1. trains under AxRAM (full training set, 10 epochs, from `init_weights.bin`);
2. counts NaN/Inf weights in the resulting model;
3. if all weights are finite, evaluates the model exactly with `eval_exact_host`;
4. appends one row to `train_results.csv` (never overwritten).

Final statuses: `OK`, `TRAINED_NAN_WEIGHTS` (finished with non-finite weights), `TRAINED_BUT_EVAL_FAILED`, `CRASHED_OR_FAILED`. A run that exits cleanly with truncated output (an observed `axpike`/`pk` anomaly unrelated to AxRAM noise) is retried once; genuine crashes and corrupted models are experimental outcomes and are not re-rolled.

The rate `0` run goes through the same kernel with no faults injected and is the **reference model** (`model_trained_axram_err0.bin`, labeled `exact`) for the other rates.

A full run has no checkpointing. Calibrate the run time with one epoch first:

```bash
python3 train_axram/run_train_axram.py --n-train-samples 1437 --epochs 1 --error-rate <rate>
```

| Output | Content |
|---|---|
| `model_trained_axram_err<rate>.bin` | Final weights (same layout as `model.bin`) |
| `genann_train_axram_err<rate>.log` · `.stdout.raw` | stderr and raw stdout of the run (stdout always kept for diagnosis) |
| `genann_train_axram_runs.csv` | Cumulative log, one row per invocation |
| `axpike_artifacts_train_err<rate>/` | AXRAM logs, counters and energy CSVs |
| `train_results.csv` | One row per training run from `train_sweep.py` |

## Full experiment and analysis

```bash
make all && make init_weights
python3 train_sweep.py
python3 validate_matrix.py              # --force / --error-rates 1e-4,1e-2 also accepted
python3 reclassify_validation.py        # only if runs were produced by an older parser
python3 genann_report.py
```

| Script | What it does | Writes |
|---|---|---|
| `validate_matrix.py` | Validates every `model_trained_axram_err<rate>.bin` at every fault probability under AxRAM. `model.bin` is not used. | `validate_matrix_results.csv` + the inference outputs above per (model, rate) |
| `reclassify_validation.py` | Re-classifies saved validation records with the current parser without running AxPIKE again (a re-run would inject new random faults and overwrite the energy traces). Originals are kept as `*.bak_before_reclassify`. | rewrites `digits_summary_*.csv`, `validate_matrix_results.csv` |
| `genann_report.py` | Read-only. Regenerates every MLP number of the paper: architecture and dataset, training outcome per rate, extreme weights and saturated neurons, full validation matrix. | `genann_report.txt`, `genann_report_training.csv`, `genann_report_validation.csv` |
| `inspect_train_record.py` | Diagnoses one training `.stdout.raw`, including failed runs: header, dimensions, weights written, NaN/Inf per layer. | stdout |

| Training | Validation | How |
|---|---|---|
| Rate 0 (reference) | Every rate | row `err0` of `validate_matrix.py`: inference-time exposure of an exactly trained model |
| Every rate | Fault-free | `train_sweep.py` (evaluates each model with `eval_exact_host`): training-time exposure |
| Every rate | Every rate | full `validate_matrix.py` matrix |

**Energy.** The AXRAM logs are in the `axpike_artifacts_*` folders. To estimate DRAM energy, put the logs of each rate in a `dataset_error_rate_<rate>/` directory inside one experiment directory and run `python3 energy/axcept.py run -w <dir>` from the repository root (see the [energy pipeline guide](../../energy/README_pipeline.md)).

## Reproducible initial weights

Comparing *trained under AxRAM* with *trained exact* only isolates the effect of memory noise if both start from the **same** weights. `gen_init_weights_host` writes the same `init_weights.bin`, byte for byte, for the same `HIDDEN_UNITS`, `HIDDEN_LAYERS` and `INIT_SEED`. `run_train_axram.py` uses it automatically. For a host baseline, pass it to `train_exact_host` with the same epochs and number of samples as the AxRAM run:

```bash
./train_exact_host <hidden_units> <epochs> <hidden_layers> init_weights.bin <n_train_samples> > model_exact_baseline.bin
```

## Output classification

**Validation** (each `validate_kernel_axpike` execution, from its raw stdout):

| Class | Meaning | Recoverable |
|---|---|---|
| `OK` | Complete `AXGENN01` record with finite outputs | yes |
| `OK_AFTER_CRASH` | Complete record followed by an AxPIKE crash signature | yes |
| `AX_STATUS_<n>` | Kernel exited normally with an error status | no |
| `SHORT_HEADER` / `PARTIAL_OUTPUT` | Record started but is incomplete | no |
| `NONFINITE_OUTPUT` | Complete record with NaN/Inf outputs | no |
| `INVALID_OUTPUT` | No `AXGENN01` magic at the start of the stream | no |
| `MISSING_OUTPUT` / `TIMEOUT_OR_EMPTY` | Nothing written (clean exit or timeout) | no |

`OK_AFTER_CRASH` follows the JPEG/FFT rule: bytes written before a crash are still evaluated, but only if the record is complete and what follows starts with a crash signature. The prediction is the argmax of the 10 outputs.

**Training** (`genann_train_axram_runs.csv`). The kernel writes its record once, after the last epoch, so any run that does not end with `status == AX_OK` is discarded entirely:

| Class | Meaning |
|---|---|
| `OK` | Valid `AXTRN001` record; model written |
| `AX_STATUS_1` / `_2` / `_3` | Clean exit with `AX_SHORT_INPUT`, `AX_INVALID_DIMS` or `AX_ALLOC_ERROR` |
| `CRASHED` | Crash signature (register dump `z <hex> ra <hex>`, segfault or assertion text) in stdout or stderr |
| `TIMEOUT` | Exceeded `TIMEOUT_SEC` (6 h) |
| `MISSING_OUTPUT` · `SHORT_HEADER` / `PARTIAL_OUTPUT` · `INVALID_OUTPUT` | Empty stdout · incomplete record · no `AXTRN001` magic |

`axpike`/`pk` may deliver the guest's per-epoch progress lines (stderr) in the stdout stream, so `parse_train_record()` searches for the `AXTRN001` magic anywhere in the stream and reports how many bytes were skipped.

## Binary formats

`pk` consumes the first 64 bytes of stdin before `main()`, so the drivers prefix every payload with 64 zero bytes; the kernels do not discard them again. Integers are little-endian `int32`, reals `float64`.

| File / stream | Layout |
|---|---|
| `validate_kernel_axpike` stdin | `inputs, hidden_layers, hidden, outputs` · `total_weights × float64` · `inputs × float64` (sample) |
| `validate_kernel_axpike` stdout | `AXGENN01` · `status` · if 0: `inputs, outputs` · `outputs × float64` |
| `train_kernel_axpike` stdin | `n_train_samples` · `inputs, hidden_layers, hidden, outputs` · `epochs` · `float64 learning_rate` · `total_weights × float64` · `n_train_samples × (inputs + outputs) × float64` |
| `train_kernel_axpike` stdout | `AXTRN001` · `status` · if 0: `inputs, hidden_layers, hidden, outputs` · `total_weights × float64` |
| `model.bin`, `init_weights.bin`, `model_trained_axram_err<rate>.bin` | `inputs, hidden_layers, hidden, outputs` · `total_weights × float64` |
| `dataset_digits/train.bin` | `n_samples` · per sample `64 × float64` pixels + `10 × float64` one-hot label |
| `dataset_digits/test.bin` | `n_samples` · per sample `64 × float64` pixels + `int32` label |

Status codes (both kernels): `0 = AX_OK`, `1 = AX_SHORT_INPUT`, `2 = AX_INVALID_DIMS`, `3 = AX_ALLOC_ERROR`.

## What to compare

- Reference accuracy vs. accuracy at each fault probability: the inference degradation curve.
- A model trained under AxRAM vs. the reference trained from the same initial weights and samples: the effect of training under noise.
- Training failure rate (`genann_train_axram_runs.csv`, `train_results.csv`) vs. fault probability.
- The full training × validation matrix (`validate_matrix_results.csv`).
- DRAM energy of each configuration.

Both kernels expose weight and activation buffers to AxRAM together, like the rest of the benchmark; exposing only one of them is a possible extension, not implemented.

## License and acknowledgments

`lib/genann.c` and `lib/genann.h` are from [codeplea/genann](https://github.com/codeplea/genann), used unmodified under the zlib license (see [`LICENSE`](LICENSE)). Every other file is original work by Guilherme Saides Serbai for AxCept-Bench (Universidade Tecnológica Federal do Paraná, Campo Mourão, Brazil, 2026).

Developed with support from a PIBIT scholarship (Fundação Araucária, 2025/2026).
