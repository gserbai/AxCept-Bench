# Ramulator Batch Pipeline

Batch runner for executing filtered AxRAM traces in Ramulator.

The script takes `U Y` traces generated from AxRAM logs, normalizes them into
Ramulator CPU traces, runs Ramulator, and stores one isolated output directory
per input trace.

Script:

```text
scripts/run_ramulator_batch.py
```

Filtering script:

```text
scripts/ramulator/extract_uy_structure.py
```

---

## 1. Input expected by the batch script

The Ramulator batch script expects filtered AxRAM traces under:

```text
~/Documents/uy_filtrado/
```

Expected structure:

```text
~/Documents/uy_filtrado/
├── dataset_error_rate_0/
├── dataset_error_rate_1e-5/
├── dataset_error_rate_1e-4/
├── dataset_error_rate_1e-3/
├── dataset_error_rate_1e-2/
└── dataset_error_rate_1e-1/
```

Each scenario directory must contain files named like:

```text
AXRAM_log_pid*_hart*.log
```

Example:

```text
AXRAM_log_pid10003_hart0.log
```

Check the number of filtered traces:

```bash
find ~/Documents/uy_filtrado -name "AXRAM_log_pid*_hart*.log" | wc -l
```

With 2100 images per scenario, the expected count is:

```text
5 approximate scenarios       -> 10500 traces
5 approximate scenarios + 0   -> 12600 traces
```

---

## 2. Filtering AxRAM logs

Raw AxRAM logs contain lines like:

```text
U Y 102 80428850 8042e840
```

The filtering script keeps only `U Y` accesses and removes the prefix:

```text
102 80428850 8042e840
```

Run from the project root:

```bash
cd ~/AxCept-Bench
```

For approximate datasets:

```bash
python3 scripts/ramulator/extract_uy_structure.py \
  ~/Documents/DataSetsWithlogsAxPike \
  -o ~/Documents/uy_filtrado
```

For the non-approximate baseline:

```bash
python3 scripts/ramulator/extract_uy_structure.py \
  ~/Documents/non_approx \
  -o ~/Documents/uy_filtrado/dataset_error_rate_0
```

The baseline should be collected with AxRAM enabled and bit-flip injection
disabled:

```text
mem_read_prob:0
```

So `U Y` still means “user-space access exposed to approximation,” not
“corrupted access.”

---

## 3. Ramulator configuration

The batch script uses this config by default:

```text
~/AxCept-Bench/ramulator/configs/DDR3-config.cfg
```

Current DDR3 backend:

```text
standard = DDR3
channels = 1
ranks = 1
speed = DDR3_1600K
org = DDR3_2Gb_x16
```

This configuration matches the DRAMPower memory specification used later:

```text
DRAMPower-4.1/memspecs/MICRON_2Gb_DDR3-1600_16bit_D.xml
```

Compatibility summary:

```text
Ramulator : DDR3_1600K + DDR3_2Gb_x16
DRAMPower : MICRON_2Gb_DDR3-1600_16bit_D.xml
```

If you change the DRAM organization in Ramulator, also change the DRAMPower
memspec to a compatible one.

Changing `DDR3-config.cfg` does **not** require recompiling Ramulator.
Recompile only if Ramulator source code is changed.

---

## 4. Main script options

The script already has these defaults:

```text
--input-root  ~/Documents/uy_filtrado
--out-root    ~/Documents/ramulator_results
--ramulator   ~/AxCept-Bench/ramulator/ramulator
--config      ~/AxCept-Bench/ramulator/configs/DDR3-config.cfg
```

Normal run:

```bash
cd ~/AxCept-Bench
python3 scripts/run_ramulator_batch.py --jobs 5
```

Equivalent explicit run:

```bash
python3 scripts/run_ramulator_batch.py \
  --input-root ~/Documents/uy_filtrado \
  --out-root ~/Documents/ramulator_results \
  --config ~/AxCept-Bench/ramulator/configs/DDR3-config.cfg \
  --jobs 5
```

Useful options:

| Option | Meaning |
|---|---|
| `--jobs N` | Number of parallel Ramulator runs |
| `--limit N` | Run only `N` traces for testing |
| `--force` | Re-run traces even if output already exists |
| `--input-root PATH` | Directory containing filtered traces |
| `--out-root PATH` | Directory where Ramulator results are stored |
| `--config PATH` | Ramulator config file to use |

---

## 5. Test before full execution

Run one trace:

```bash
python3 scripts/run_ramulator_batch.py --limit 1 --jobs 1
```

Expected result:

```text
OK | cmdtraces=1 | stats=... bytes
```

Run a small batch:

```bash
python3 scripts/run_ramulator_batch.py --limit 5 --jobs 1
```

Then run all traces:

```bash
python3 scripts/run_ramulator_batch.py --jobs 5
```

---

## 6. Running in the background

```bash
cd ~/AxCept-Bench
mkdir -p ~/Documents/ramulator_results

nohup python3 scripts/run_ramulator_batch.py --jobs 5 \
  > ~/Documents/ramulator_results/run.log 2>&1 &
```

Monitor:

```bash
tail -f ~/Documents/ramulator_results/run.log
```

---

## 7. Output

Results are written to:

```text
~/Documents/ramulator_results/
```

Example:

```text
~/Documents/ramulator_results/
├── ramulator_batch_summary.csv
├── dataset_error_rate_0/
│   └── AXRAM_log_pid10003_hart0/
│       ├── trace.cpu
│       ├── DDR3-config-used.cfg
│       ├── DDR3.stats
│       ├── stdout.txt
│       ├── stderr.txt
│       └── cmd-trace-chan-0-rank-0.cmdtrace
├── dataset_error_rate_1e-5/
├── dataset_error_rate_1e-4/
├── dataset_error_rate_1e-3/
├── dataset_error_rate_1e-2/
└── dataset_error_rate_1e-1/
```

Generated files:

| File | Purpose |
|---|---|
| `trace.cpu` | Normalized CPU trace used by Ramulator |
| `DDR3-config-used.cfg` | Exact config used in that run |
| `DDR3.stats` | Ramulator statistics |
| `cmd-trace-*.cmdtrace` | DRAM command trace for energy estimation |
| `stdout.txt` | Ramulator output |
| `stderr.txt` | Ramulator errors/warnings |

---

## 8. Progress checks

Count completed simulations:

```bash
find ~/Documents/ramulator_results -name "DDR3.stats" | wc -l
```

Count generated command traces:

```bash
find ~/Documents/ramulator_results -name "cmd-trace-*.cmdtrace" | wc -l
```

Check baseline only:

```bash
find ~/Documents/ramulator_results/dataset_error_rate_0 \
  -name "DDR3.stats" | wc -l
```

Check the config actually used in a run:

```bash
find ~/Documents/ramulator_results -name "DDR3-config-used.cfg" \
  | head -n 1 \
  | xargs grep -E "standard|speed|org|record_cmd_trace"
```

Expected:

```text
standard = DDR3
speed = DDR3_1600K
org = DDR3_2Gb_x16
record_cmd_trace = on
```

---

## 9. Resume or re-run

The script skips traces that already have a non-empty `DDR3.stats`.

Resume:

```bash
python3 scripts/run_ramulator_batch.py --jobs 5
```

Force regeneration:

```bash
python3 scripts/run_ramulator_batch.py --jobs 5 --force
```

Use `--force` when the Ramulator config changed or the previous output must be
discarded.

---

## 10. Output used by DRAMPower

The downstream energy stage uses:

```text
cmd-trace-*.cmdtrace
```

Example DRAMPower command:

```bash
cd ~/AxCept-Bench/DRAMPower-4.1

./drampower \
  -m memspecs/MICRON_2Gb_DDR3-1600_16bit_D.xml \
  -c ~/Documents/ramulator_results/dataset_error_rate_0/AXRAM_log_pid10003_hart0/cmd-trace-chan-0-rank-0.cmdtrace
```

Keep the Ramulator config and the DRAMPower memspec compatible.

---

## 11. Common checks

No traces found:

```bash
find ~/Documents/uy_filtrado -name "AXRAM_log_pid*_hart*.log" | head
```

Wrong number of scenarios:

```bash
find ~/Documents/uy_filtrado -maxdepth 1 -type d | sort
```

Wrong DRAM configuration:

```bash
grep -E "standard|speed|org" \
  ~/AxCept-Bench/ramulator/configs/DDR3-config.cfg
```

Missing command trace:

```bash
cat stderr.txt
cat stdout.txt
cat DDR3-config-used.cfg
```

---

## Minimal command sequence

```bash
cd ~/AxCept-Bench

find ~/Documents/uy_filtrado -name "AXRAM_log_pid*_hart*.log" | wc -l

python3 scripts/run_ramulator_batch.py --limit 1 --jobs 1

python3 scripts/run_ramulator_batch.py --jobs 5
```

Keep the setup explicit. Keep the outputs isolated. Keep the configuration
documented.
