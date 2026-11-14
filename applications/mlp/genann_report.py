#!/usr/bin/env python3
######################################################################
# AxCept-Bench (MLP) -- regenerate every MLP number reported in the
# paper directly from the project's own files, without running AxPIKE.
#
# What it reports (and where it appears in the paper):
#   1. Architecture and dataset: dimensions, weight count, split sizes,
#      input pixels that are zero in every training sample.
#   2. Training outcome per fault probability (Table 2 / Figure 8):
#      model or failure type (malformed record, crash, ...), weights
#      changed vs. the fault-free model, median/max |dw|, extreme finite
#      weights, NaN, Inf, fault-free validation accuracy.
#   3. Where the extreme weights are and how many validation samples
#      saturate each affected first-layer neuron.
#   4. Validation matrix (Table 3 / Figure 9): accuracy of every model at
#      every validation fault probability, with the outcome of each
#      execution (OK, OK_AFTER_CRASH, AX_STATUS_1, ...).
#
# Read-only: it never modifies models, records, logs or CSVs. It writes
# only its own outputs:
#   genann_report.txt              the full human-readable report
#   genann_report_training.csv     one row per training fault probability
#   genann_report_validation.csv   one row per (model, validation rate)
#
# Author: Guilherme Saides Serbai 
# Year: 2026
#
# Run from the project root, after train_sweep.py and validate_matrix.py:
#   python3 genann_report.py
######################################################################

import csv
import importlib.util
import math
import re
import statistics
import struct
import sys
from collections import Counter
from pathlib import Path

RATES = ["0", "1e-5", "1e-4", "1e-3", "1e-2", "1e-1"]
REFERENCE_RATE = "0"
EXTREME_THRESHOLD = 1e3          # finite |w| above this counts as "extreme"
SIGMOID_DOMAIN = 15.0            # genann's cached sigmoid saturates outside [-15, 15]

TRAIN_BIN = Path("dataset_digits/train.bin")
TEST_BIN = Path("dataset_digits/test.bin")
VALIDATE_SCRIPT = Path("train_exact_and_validate/run_validate_axram.py")
TRAIN_RESULTS_CSV = Path("train_results.csv")

OUT_TXT = Path("genann_report.txt")
OUT_TRAIN_CSV = Path("genann_report_training.csv")
OUT_VAL_CSV = Path("genann_report_validation.csv")

# Same crash-signature convention as train_axram/run_train_axram.py.
REGISTER_DUMP_PATTERN = re.compile(rb"z\s+[0-9a-fA-F]{16}\s+ra\s+[0-9a-fA-F]{16}")
ASSERT_FAILURE_PATTERN = re.compile(rb'assertion\s+"[^"]*"\s+failed')
CRASH_TEXT_MARKERS = (b"User load segfault", b"User store segfault",
                      b"User fetch segfault", b"segfault")

# validate_kernel_axpike.cpp status codes
AX_STATUS_NAMES = {1: "AX_SHORT_INPUT", 2: "AX_INVALID_DIMS", 3: "AX_ALLOC_ERROR"}

_lines = []


def out(text: str = "") -> None:
    print(text)
    _lines.append(text)


# ===============================
# Network helpers (same layout as lib/genann.c)
# ===============================

def total_weights(inputs, hidden_layers, hidden, outputs) -> int:
    if hidden_layers == 0:
        return (inputs + 1) * outputs
    return ((inputs + 1) * hidden
            + (hidden_layers - 1) * (hidden + 1) * hidden
            + (hidden + 1) * outputs)


def layer_blocks(inputs, hidden_layers, hidden, outputs):
    """[(name, start, n_neurons, fan_in)] in the order genann stores them:
    per neuron, the bias first, then one weight per input."""
    blocks, pos = [], 0
    if hidden_layers == 0:
        return [("input -> output", 0, outputs, inputs)]
    blocks.append(("input -> hidden 1", pos, hidden, inputs))
    pos += (inputs + 1) * hidden
    for h in range(1, hidden_layers):
        blocks.append((f"hidden {h} -> hidden {h + 1}", pos, hidden, hidden))
        pos += (hidden + 1) * hidden
    blocks.append((f"hidden {hidden_layers} -> output", pos, outputs, hidden))
    return blocks


def locate(idx, blocks):
    """Weight index -> (layer name, neuron, 'bias' or input index)."""
    for name, start, n, fan_in in blocks:
        end = start + n * (fan_in + 1)
        if start <= idx < end:
            neuron, k = divmod(idx - start, fan_in + 1)
            return name, neuron, ("bias" if k == 0 else k - 1)
    return "beyond expected range", None, None


def load_model(path: Path):
    d = path.read_bytes()
    dims = struct.unpack("<iiii", d[:16])
    n = (len(d) - 16) // 8
    return dims, list(struct.unpack(f"<{n}d", d[16:16 + n * 8]))


# genann_act_sigmoid_cached, reproduced exactly (lookup table, 4096 entries)
_LOOKUP_SIZE, _LO, _HI = 4096, -15.0, 15.0
_F = (_HI - _LO) / _LOOKUP_SIZE
_INTERVAL = _LOOKUP_SIZE / (_HI - _LO)


def _exact_sigmoid(a):
    if a < -45.0:
        return 0.0
    if a > 45.0:
        return 1.0
    return 1.0 / (1 + math.exp(-a))


_LOOKUP = [_exact_sigmoid(_LO + _F * i) for i in range(_LOOKUP_SIZE)]


def sigmoid_cached(a):
    if a < _LO:
        return _LOOKUP[0]
    if a >= _HI:
        return _LOOKUP[-1]
    j = int((a - _LO) * _INTERVAL + 0.5)
    return _LOOKUP[-1] if j >= _LOOKUP_SIZE else _LOOKUP[j]


def forward(weights, dims, x):
    """genann_run(). Returns (outputs, first-layer pre-activation sums)."""
    inputs, hidden_layers, hidden, outputs = dims
    pos, inp, first_sums = 0, list(x), None
    sizes = [hidden] * hidden_layers + [outputs]
    for layer, n_out in enumerate(sizes):
        nxt, sums = [], []
        for _ in range(n_out):
            s = weights[pos] * -1.0
            pos += 1
            for v in inp:
                s += weights[pos] * v
                pos += 1
            sums.append(s)
            nxt.append(sigmoid_cached(s))
        if layer == 0:
            first_sums = sums
        inp = nxt
    return inp, first_sums


# ===============================
# Dataset
# ===============================

def load_train(path: Path):
    d = path.read_bytes()
    (n,) = struct.unpack("<i", d[:4])
    off, xs = 4, []
    for _ in range(n):
        xs.append(struct.unpack("<64d", d[off:off + 512]))
        off += 512 + 80
    return xs


def load_test(path: Path):
    d = path.read_bytes()
    (n,) = struct.unpack("<i", d[:4])
    off, samples = 4, []
    for _ in range(n):
        x = struct.unpack("<64d", d[off:off + 512])
        off += 512
        (y,) = struct.unpack("<i", d[off:off + 4])
        off += 4
        samples.append((x, y))
    return samples


# ===============================
# Raw training output (for runs that produced no model)
# ===============================

def crash_signature(data: bytes):
    if data.startswith(b"z  "):
        return "register dump"
    if REGISTER_DUMP_PATTERN.search(data):
        return "register dump"
    if ASSERT_FAILURE_PATTERN.search(data):
        return "assertion failure"
    for m in CRASH_TEXT_MARKERS:
        if m in data:
            return m.decode()
    return None


def inspect_raw_training(path: Path, dims_ref):
    if not path.exists():
        return {"outcome": "no output file"}
    data = path.read_bytes()
    idx = data.find(b"AXTRN001")
    if idx == -1:
        sig = crash_signature(data)
        if sig:
            return {"outcome": "crash", "detail": f"{sig}, no record written ({len(data)} bytes)"}
        return {"outcome": "missing/invalid output", "detail": f"{len(data)} bytes, no record"}
    rec = data[idx:]
    if len(rec) < 28:
        return {"outcome": "short header"}
    (status,) = struct.unpack("<i", rec[8:12])
    if status != 0:
        return {"outcome": f"kernel status {status}"}
    dims = struct.unpack("<iiii", rec[12:28])
    expected = total_weights(*dims)
    n = (len(rec) - 28) // 8
    vals = struct.unpack(f"<{n}d", rec[28:28 + n * 8])
    within = vals[:expected]
    beyond = vals[expected:]
    res = {
        "outcome": "malformed record" if n != expected else "record OK but no model file",
        "dims": dims, "values_found": n, "values_expected": expected,
        "nan": sum(1 for v in within if math.isnan(v)),
        "inf": sum(1 for v in within if math.isinf(v)),
        "extreme": sum(1 for v in within if math.isfinite(v) and abs(v) > EXTREME_THRESHOLD),
        "beyond": len(beyond),
        "beyond_nan": sum(1 for v in beyond if math.isnan(v)),
        "crash_signature": crash_signature(data),
    }
    blocks = layer_blocks(*dims)
    res["per_layer"] = []
    for name, start, nn, fan_in in blocks:
        seg = within[start:start + nn * (fan_in + 1)]
        res["per_layer"].append((name, len(seg),
                                 sum(1 for v in seg if math.isnan(v)),
                                 sum(1 for v in seg if math.isinf(v))))
    return res


# ===============================
# Main report
# ===============================

def main():
    missing = [p for p in (TRAIN_BIN, TEST_BIN) if not p.exists()]
    if missing:
        print("ERRO: rode a partir da raiz do projeto; faltam: " + ", ".join(map(str, missing)))
        sys.exit(1)

    train_x = load_train(TRAIN_BIN)
    test = load_test(TEST_BIN)

    ref_path = Path(f"model_trained_axram_err{REFERENCE_RATE}.bin")
    if not ref_path.exists():
        print(f"ERRO: {ref_path} nao existe (modelo de referencia, taxa 0).")
        sys.exit(1)
    dims, ref = load_model(ref_path)
    blocks = layer_blocks(*dims)
    expected = total_weights(*dims)

    # ---------------- 1. architecture and dataset
    out("=" * 72)
    out("GENANN report -- numbers reported in the paper, regenerated from files")
    out("=" * 72)
    out()
    out("1. Architecture and dataset")
    inputs, hl, hidden, outputs = dims
    arch = " -> ".join([str(inputs)] + [str(hidden)] * hl + [str(outputs)])
    out(f"   network: {arch}  ({expected} weights)")
    for name, start, n, fan_in in blocks:
        out(f"     {name:22s} {n} neurons x ({fan_in} inputs + bias) = {n * (fan_in + 1)} weights")
    out(f"   training samples: {len(train_x)}   validation samples: {len(test)}")
    zero_pixels = [j for j in range(64) if all(x[j] == 0.0 for x in train_x)]
    out(f"   input pixels zero in every training sample: {zero_pixels}")
    out(f"   largest |w| of the reference model: {max(abs(w) for w in ref):.2f}")
    out()

    # accuracy recorded by train_sweep.py, for cross-checking
    recorded = {}
    if TRAIN_RESULTS_CSV.exists():
        with open(TRAIN_RESULTS_CSV, newline="") as f:
            for r in csv.DictReader(f):
                if r.get("accuracy_exact_pct"):
                    recorded[r["error_rate"]] = r["accuracy_exact_pct"]

    # ---------------- 2. training outcome per rate
    out("2. Training outcome per fault probability (paper: Table 2, Figure 8)")
    header = f"   {'rate':6s} {'outcome':18s} {'changed':>8s} {'median|dw|':>10s} {'max|dw|':>9s} {'extreme':>7s} {'NaN':>6s} {'Inf':>5s} {'accuracy':>9s}"
    out(header)
    train_rows, extreme_details = [], {}
    for rate in RATES:
        mpath = Path(f"model_trained_axram_err{rate}.bin")
        row = {"error_rate": rate}
        if mpath.exists():
            mdims, w = load_model(mpath)
            finite = all(math.isfinite(v) for v in w)
            diff = [abs(a - b) if (math.isfinite(a) and math.isfinite(b)) else math.inf
                    for a, b in zip(ref, w)]
            changed = sum(1 for d in diff if d != 0)
            unchanged = [i for i, d in enumerate(diff) if d == 0]
            extreme = [i for i, v in enumerate(w) if math.isfinite(v) and abs(v) > EXTREME_THRESHOLD]
            acc = ""
            if finite:
                ok = 0
                sat = Counter()
                ext_neurons = sorted({locate(i, blocks)[1] for i in extreme
                                      if locate(i, blocks)[0] == blocks[0][0]})
                for x, y in test:
                    o, s1 = forward(w, mdims, x)
                    ok += (o.index(max(o)) == y)
                    for k in ext_neurons:
                        sat[k] += abs(s1[k]) > SIGMOID_DOMAIN
                acc = f"{100 * ok / len(test):.2f}"
                if extreme:
                    extreme_details[rate] = (extreme, w, sat)
            row.update({
                "outcome": "reference" if rate == REFERENCE_RATE else "model",
                "weights_changed": "" if rate == REFERENCE_RATE else changed,
                "median_abs_dw": "" if rate == REFERENCE_RATE else f"{statistics.median(diff):.4f}",
                "max_abs_dw": "" if rate == REFERENCE_RATE else f"{max(diff):.4g}",
                "extreme": len(extreme),
                "nan": sum(1 for v in w if math.isnan(v)),
                "inf": sum(1 for v in w if math.isinf(v)),
                "accuracy_pct": acc,
                "accuracy_recorded_pct": recorded.get(rate, ""),
                "unchanged_inputs": "" if rate == REFERENCE_RATE else sorted(
                    {locate(i, blocks)[2] for i in unchanged
                     if locate(i, blocks)[0] == blocks[0][0] and locate(i, blocks)[2] != "bias"}),
                "n_unchanged": "" if rate == REFERENCE_RATE else len(unchanged),
            })
        else:
            raw = inspect_raw_training(Path(f"genann_train_axram_err{rate}.stdout.raw"), dims)
            row.update({"outcome": raw["outcome"], "weights_changed": "", "median_abs_dw": "",
                        "max_abs_dw": "", "extreme": raw.get("extreme", ""),
                        "nan": raw.get("nan", ""), "inf": raw.get("inf", ""),
                        "accuracy_pct": "", "accuracy_recorded_pct": "",
                        "raw": raw})
        train_rows.append(row)
        out(f"   {rate:6s} {row['outcome']:18s} {str(row['weights_changed']):>8s} "
            f"{str(row['median_abs_dw']):>10s} {str(row['max_abs_dw']):>9s} {str(row['extreme']):>7s} "
            f"{str(row['nan']):>6s} {str(row['inf']):>5s} {row['accuracy_pct']:>9s}")
    out(f"   (extreme = finite |w| > {EXTREME_THRESHOLD:g}; accuracy = fault-free, on the "
        f"{len(test)} validation samples, recomputed here with genann's cached sigmoid)")
    for r in train_rows:
        if r.get("accuracy_pct") and r.get("accuracy_recorded_pct"):
            same = abs(float(r["accuracy_pct"]) - float(r["accuracy_recorded_pct"])) < 0.01
            out(f"   check {r['error_rate']}: recomputed {r['accuracy_pct']}% vs train_results.csv "
                f"{float(r['accuracy_recorded_pct']):.2f}% -> {'OK' if same else 'MISMATCH'}")
    out()

    # details: unchanged weights, malformed records, crashes
    out("   Details")
    for r in train_rows:
        rate = r["error_rate"]
        if r["outcome"] == "model":
            out(f"   {rate}: {r['n_unchanged']} unchanged weights; first-layer ones connect "
                f"to input pixels {r['unchanged_inputs']}")
        elif "raw" in r:
            raw = r["raw"]
            if raw["outcome"] == "malformed record":
                out(f"   {rate}: malformed record -- {raw['values_found']} values instead of "
                    f"{raw['values_expected']}; {raw['beyond']} beyond the expected count "
                    f"({raw['beyond_nan']} NaN); crash signature: {raw['crash_signature'] or 'none'}")
                for name, n, nn, ni in raw["per_layer"]:
                    out(f"         {name:22s} NaN {nn}/{n}   Inf {ni}/{n}")
            else:
                out(f"   {rate}: {raw['outcome']}" + (f" ({raw['detail']})" if raw.get("detail") else ""))
    out()

    # ---------------- 3. extreme weights
    out("3. Extreme weights and neuron saturation")
    if not extreme_details:
        out("   no model with extreme finite weights")
    for rate, (extreme, w, sat) in extreme_details.items():
        groups = Counter()
        for i in extreme:
            name, neuron, _ = locate(i, blocks)
            groups[(name, neuron)] += 1
        mags = [abs(w[i]) for i in extreme]
        out(f"   {rate}: {len(extreme)} extreme weights, |w| from {min(mags):.1e} to {max(mags):.1e}")
        for (name, neuron), c in sorted(groups.items(), key=lambda kv: -kv[1]):
            fan_in = next(fi for nm, _, _, fi in blocks if nm == name)
            line = f"      {name}, neuron {neuron}: {c} of {fan_in + 1} weights"
            if name == blocks[0][0] and neuron in sat:
                line += f"; saturated on {sat[neuron]} of {len(test)} validation samples"
            if name != blocks[0][0]:
                srcs = sorted({locate(i, blocks)[2] for i in extreme
                               if locate(i, blocks)[:2] == (name, neuron)}, key=str)
                line += f"; input(s) from previous layer: {srcs}"
            out(line)
    out()

    # ---------------- 4. validation matrix
    out("4. Validation matrix (paper: Table 3, Figure 9)")
    parse = None
    if VALIDATE_SCRIPT.exists():
        spec = importlib.util.spec_from_file_location("run_validate_axram", VALIDATE_SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        parse = mod.parse_genn_record
    labels = [y for _, y in test]
    val_rows = []
    def training_rate(tag):
        r = tag.rsplit("_err", 1)[-1]
        return RATES.index(r) if r in RATES else len(RATES)

    tags = sorted({p.name[p.name.index("_model_") + 1:]
                   for p in Path(".").glob("digits_records_err*_model_trained_axram_err*")
                   if p.is_dir()}, key=training_rate)
    for tag in tags:
        for rate in RATES:
            rec_dir = Path(f"digits_records_err{rate}_{tag}")
            if not rec_dir.is_dir():
                continue
            classes, correct = Counter(), 0
            for i, y in enumerate(labels):
                f = rec_dir / f"sample_{i:04d}.bin"
                data = f.read_bytes() if f.exists() else b""
                info = parse(data) if parse else {"class": "?", "recoverable": False}
                cls = info["class"]
                classes[cls] += 1
                if info.get("recoverable") and info.get("predicted") == y:
                    correct += 1
            val_rows.append({"model": tag, "validation_rate": rate,
                             "accuracy_pct": f"{100 * correct / len(labels):.2f}",
                             **{f"n_{k}": v for k, v in classes.items()}})
    if not val_rows:
        out("   no digits_records_err*_model_trained_axram_err* folders found")
    else:
        models = list(dict.fromkeys(r["model"] for r in val_rows))
        rates_present = [r for r in RATES if any(v["validation_rate"] == r for v in val_rows)]
        out("   accuracy (%) -- rows: model (training rate), columns: validation rate")
        out("   " + f"{'model':30s}" + "".join(f"{r:>9s}" for r in rates_present))
        for m in models:
            cells = []
            for r in rates_present:
                v = next((x for x in val_rows if x["model"] == m and x["validation_rate"] == r), None)
                cells.append(v["accuracy_pct"] if v else "-")
            out("   " + f"{m:30s}" + "".join(f"{c:>9s}" for c in cells))
        out()
        out("   outcome of each execution (360 per cell)")
        for v in val_rows:
            counts = {k[2:]: n for k, n in v.items() if k.startswith("n_")}
            text = ", ".join(f"{k}={n}" for k, n in sorted(counts.items(), key=lambda kv: -kv[1]))
            for code, name in AX_STATUS_NAMES.items():
                text = text.replace(f"AX_STATUS_{code}=", f"AX_STATUS_{code} ({name})=")
            out(f"   {v['model']:30s} @ {v['validation_rate']:5s}: {text}")
    out()

    # ---------------- write outputs
    OUT_TXT.write_text("\n".join(_lines) + "\n")
    tfields = ["error_rate", "outcome", "weights_changed", "median_abs_dw", "max_abs_dw",
               "extreme", "nan", "inf", "accuracy_pct", "accuracy_recorded_pct"]
    with open(OUT_TRAIN_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=tfields)
        w.writeheader()
        for r in train_rows:
            w.writerow({k: r.get(k, "") for k in tfields})
    if val_rows:
        vfields = ["model", "validation_rate", "accuracy_pct"] + sorted(
            {k for r in val_rows for k in r if k.startswith("n_")})
        with open(OUT_VAL_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=vfields)
            w.writeheader()
            for r in val_rows:
                w.writerow({k: r.get(k, 0 if k.startswith("n_") else "") for k in vfields})
    print(f"\nwritten: {OUT_TXT}, {OUT_TRAIN_CSV}" + (f", {OUT_VAL_CSV}" if val_rows else ""))


if __name__ == "__main__":
    main()
