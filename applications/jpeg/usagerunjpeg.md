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
       pk /home/user/AxCept-Bench/applications/jpeg/src/toojpeg_encoder 100 < input.csv > output.jpeg
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

**Python:**
```bash
python3 scripts/benchmarks/jpeg/axcept_batch_jpeg.py
```

The runner scans `src/dataset_csv/` recursively and writes output to
`src/dataset_error_rate_<rate>/`, preserving the original directory structure.

---

## Changing Error Rate

Edit `mem_read_prob` in the `--adele` flag:

```
--adele=mem_read_prob:1e-4
--adele=mem_read_prob:1e-3
--adele=mem_read_prob:1e-1
--adele=mem_read_prob:1.4e-1
```