# JPEG Benchmark — Memory Fault Susceptibility

Single and batch execution of the JPEG encoder under memory approximation via AxPike,
using probabilistic error models to simulate low-voltage scenarios.

---

## Single Execution

```bash
$ axpike --adele=mem_read_prob:1e-4,linesz:32 \
       --adele-activate=0:AXRAM \
       --mem_log=log_mem.mem \
       --dc=128:8:32 \
       --ic=256:4:32 \
       --l2=1024:4:32 \
       pk applications/jpeg/src/toojpeg_encoder 100 < input.csv > output.jpeg
```

---

## Approximation Parameters

| Parameter | Description |
|---|---|
| `mem_read_prob` | probability of memory error in read |
| `linesz` | cache line size |

```
1e-4  → low
1e-3  → medium
1e-1  → high
```

---

## Batch Execution

```bash
cd applications/jpeg
python3 axcept_batch_jpeg.py
```

The runner scans `DATASET_DIR` recursively for `.csv` files and writes, relative to the
directory it is started from, the images to `src/dataset_error_rate_<rate>/` and the
`stderr` logs to `src/logs_error_rate_<rate>/`, preserving the dataset's directory structure.
The AxRAM traces (`AXRAM_log_pid*_hart*.log`) are written to the current directory.

To also compute energy, see section 5 of the [application guide](../usage.md#5-organizing-the-runs-for-the-energy-pipeline).

---

## Changing Error Rate

In the batch runner, set `ERROR_RATE` at the top of the script. For a single execution, edit `mem_read_prob` in the `--adele` flag:

```
--adele=mem_read_prob:1e-4
--adele=mem_read_prob:1e-3
--adele=mem_read_prob:1e-1
--adele=mem_read_prob:1.4e-1
```