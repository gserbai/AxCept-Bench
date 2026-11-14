# Ramulator Batch Runner

> Usually run through `energy/axcept.py`, which fills in all the paths below. See the [energy pipeline guide](../README_pipeline.md).

`run_ramulator_batch.py` runs filtered AxRAM traces (`U Y` accesses only) through Ramulator, one isolated output directory per trace. The traces come from [`energy/axram/extract_uy_structure.py`](../axram/extract_uy_structure.py).

## Input

One directory per scenario, each containing the filtered traces:

```text
uy_filtered/
├── dataset_error_rate_0/           # exact baseline (mem_read_prob:0)
│   └── AXRAM_log_pid*_hart*.log
├── dataset_error_rate_1e-5/
└── ...
```

The baseline is collected with AxRAM enabled and `mem_read_prob:0`, so its traces are comparable to the approximate ones.

## Usage

Pass `--input-root` and `--out-root` explicitly: their defaults point to `~/Documents/...`. `--ramulator` and `--config` default to `ramulator/ramulator` and `configs/ramulator/DDR3_2Gb_x16.cfg` in the repository.

```bash
python3 energy/ramulator/run_ramulator_batch.py \
  --input-root experiments/<app>/uy_filtered \
  --out-root experiments/<app>/ramulator_results \
  --ramulator ramulator/ramulator \
  --config configs/ramulator/DDR3_2Gb_x16.cfg \
  --jobs 4
```

| Option | Meaning |
|---|---|
| `--input-root PATH` | directory with the filtered traces |
| `--out-root PATH` | where Ramulator results are stored |
| `--ramulator PATH` | Ramulator executable |
| `--config PATH` | Ramulator config file |
| `--jobs N` | parallel Ramulator runs |
| `--limit N` | run only `N` traces (for testing) |
| `--force` | re-run traces that already have output |

Test with `--limit 1 --jobs 1` first. A successful run prints `OK | cmdtraces=1 | stats=... bytes`.

The script skips traces that already have a non-empty `DDR3.stats`, so an interrupted run can be resumed by running the same command again. Use `--force` after changing the config.

## Output

```text
ramulator_results/
├── ramulator_batch_summary.csv
└── dataset_error_rate_0/
    └── AXRAM_log_pid10003_hart0/
        ├── trace.cpu                      # normalized CPU trace
        ├── DDR3-config-used.cfg           # exact config used (cmd trace forced on)
        ├── DDR3.stats                     # Ramulator statistics
        ├── stdout.txt / stderr.txt
        └── cmd-trace-chan-0-rank-0.cmdtrace   # input of the DRAMPower stage
```

If `cmd-trace-*.cmdtrace` is missing, check `stderr.txt`, `stdout.txt` and `DDR3-config-used.cfg`.

## Compatibility with DRAMPower

The Ramulator config and the DRAMPower memspec must describe the same memory.

```text
Ramulator : DDR3_1600K + DDR3_2Gb_x16   (configs/ramulator/DDR3_2Gb_x16.cfg)
DRAMPower : MICRON_2Gb_DDR3-1600_16bit_D.xml
```

If you change the DRAM organization, change the memspec too. Editing the config does not require recompiling Ramulator, only changes to its source do.
