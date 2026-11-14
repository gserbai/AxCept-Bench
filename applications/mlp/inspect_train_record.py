#!/usr/bin/env python3
######################################################################
# AxCept-Bench (GENANN) -- inspects a .stdout.raw from a training run
# (even one that failed) and reports what really happened to the
# AXTRN001 record: header, dimensions, how many weights were
# actually written vs. expected, and exactly where NaN/Inf show up
# (with the range of each layer, to say "the whole input layer"
# instead of just "there is NaN somewhere").
#
# Does not replace the official parser (train_axram/run_train_axram.py's
# parse_train_record()) -- this one is only for manual diagnosis of a
# run that already failed (PARTIAL_OUTPUT, SHORT_HEADER, etc.) and that the
# normal pipeline (count_nan_inf() in train_sweep.py) never gets to
# check, because it only runs on an already validated model.bin.
#
# Author: Guilherme Saides Serbai 
# Year: 2026
#
# Usage:
#   python3 inspect_train_record.py genann_train_axram_err1e-3.stdout.raw
######################################################################

import argparse
import math
import re
import struct
import sys
from pathlib import Path

MAGIC = b"AXTRN001"

# Same crash detection convention as run_train_axram.py.
REGISTER_DUMP_PATTERN = re.compile(rb"z\s+[0-9a-fA-F]{16}\s+ra\s+[0-9a-fA-F]{16}")
ASSERT_FAILURE_PATTERN = re.compile(rb'assertion\s+"[^"]*"\s+failed')
CRASH_TEXT_MARKERS = (
    b"User load segfault",
    b"User store segfault",
    b"User fetch segfault",
    b"segfault",
)


def detect_crash(data: bytes) -> str | None:
    if not data:
        return None
    if data.startswith(b"z  "):
        return "register dump at start of stream"
    m = REGISTER_DUMP_PATTERN.search(data)
    if m:
        return f"register dump pattern at byte {m.start()}"
    m = ASSERT_FAILURE_PATTERN.search(data)
    if m:
        return f"assertion failure at byte {m.start()}: {data[m.start():m.start()+80]!r}"
    for marker in CRASH_TEXT_MARKERS:
        idx = data.find(marker)
        if idx != -1:
            return f"crash text marker {marker!r} at byte {idx}"
    return None


def total_weights(inputs: int, hidden_layers: int, hidden: int, outputs: int) -> int:
    # Same formula as genann/train_sweep.py.
    if hidden_layers <= 1:
        return (inputs + 1) * hidden + (hidden + 1) * outputs
    return (inputs + 1) * hidden + (hidden + 1) * hidden * (hidden_layers - 1) + (hidden + 1) * outputs


def layer_ranges(inputs: int, hidden_layers: int, hidden: int, outputs: int):
    """List of (name, start_idx, end_idx) inside the weight vector, in the
    same order the kernel writes them: input->hidden1, hidden_i->hidden_{i+1}
    (if there is more than one hidden layer), hidden_last->output."""
    ranges = []
    pos = 0
    n = (inputs + 1) * hidden
    ranges.append(("input -> hidden_1", pos, pos + n))
    pos += n
    for i in range(1, hidden_layers):
        n = (hidden + 1) * hidden
        ranges.append((f"hidden_{i} -> hidden_{i+1}", pos, pos + n))
        pos += n
    n = (hidden + 1) * outputs
    ranges.append((f"hidden_{hidden_layers} -> output", pos, pos + n))
    pos += n
    return ranges


def inspect(path: Path) -> None:
    data = path.read_bytes()
    print(f"=== {path} ({len(data)} bytes) ===\n")

    idx = data.find(MAGIC)
    if idx == -1:
        print("Nenhum registro AXTRN001 encontrado -- procurando assinatura de crash...")
        reason = detect_crash(data)
        if reason:
            print(f"CRASH detectado: {reason}")
        else:
            print("Nenhuma assinatura de crash conhecida encontrada tambem "
                  "(pode ser MISSING_OUTPUT/INVALID_OUTPUT -- saida vazia ou "
                  "protocolo diferente do esperado).")
        return

    if idx > 0:
        print(f"(nota: {idx} bytes antes do magic -- provavelmente stderr do "
              f"guest, ex.: progresso por epoca)\n")

    rec = data[idx:]
    if len(rec) < 12:
        print(f"SHORT_HEADER: registro tem so {len(rec)} bytes (precisa de "
              f"pelo menos 12 pro header).")
        return

    (status,) = struct.unpack("<i", rec[8:12])
    print(f"status: {status} ({'OK' if status == 0 else f'AX_STATUS_{status} (erro reportado pelo proprio kernel)'})")
    if status != 0:
        return

    if len(rec) < 28:
        print(f"SHORT_HEADER: registro tem so {len(rec)} bytes (precisa de "
              f"pelo menos 28 pra ter as dimensoes).")
        return

    inputs, hidden_layers, hidden, outputs = struct.unpack("<iiii", rec[12:28])
    print(f"dims: inputs={inputs} hidden_layers={hidden_layers} hidden={hidden} outputs={outputs}")

    expected = total_weights(inputs, hidden_layers, hidden, outputs)
    payload = rec[28:]
    n_actual = len(payload) // 8
    remainder = len(payload) % 8
    print(f"pesos esperados: {expected} ({expected * 8} bytes)")
    print(f"pesos encontrados: {n_actual} ({len(payload)} bytes"
          + (f", + {remainder} bytes sobrando (nao formam um float64 completo)" if remainder else "")
          + ")")

    diff = n_actual - expected
    if diff == 0:
        print("-> tamanho bate com o esperado (isso normalmente seria status OK "
              "no parser oficial).")
    elif diff > 0:
        print(f"-> {diff} valores A MAIS que o esperado (registro maior que deveria).")
    else:
        print(f"-> faltam {-diff} valores (registro menor/truncado).")

    vals = struct.unpack(f"<{n_actual}d", payload[:n_actual * 8])
    nan_count = sum(1 for v in vals if math.isnan(v))
    inf_count = sum(1 for v in vals if math.isinf(v))
    print(f"\nNaN: {nan_count}/{n_actual} ({100*nan_count/n_actual:.2f}%)")
    print(f"Inf: {inf_count}/{n_actual} ({100*inf_count/n_actual:.2f}%)")

    print("\n--- por camada (dentro do range esperado, 0..{}) ---".format(expected - 1))
    for name, start, end in layer_ranges(inputs, hidden_layers, hidden, outputs):
        seg = vals[start:min(end, n_actual)]
        seg_nan = sum(1 for v in seg if math.isnan(v))
        seg_inf = sum(1 for v in seg if math.isinf(v))
        print(f"  {name:24s} idx [{start:6d}:{end:6d}) "
              f"-- NaN: {seg_nan}/{len(seg)}  Inf: {seg_inf}/{len(seg)}")

    if n_actual > expected:
        extra = vals[expected:]
        extra_nan = sum(1 for v in extra if math.isnan(v))
        extra_inf = sum(1 for v in extra if math.isinf(v))
        print(f"\n--- {len(extra)} valores ALEM do esperado (idx {expected}:{n_actual}) ---")
        print(f"  NaN: {extra_nan}/{len(extra)}  Inf: {extra_inf}/{len(extra)}")
        print(f"  primeiros 10: {[round(v, 4) if not (math.isnan(v) or math.isinf(v)) else v for v in extra[:10]]}")

    reason = detect_crash(data)
    print(f"\nassinatura de crash no stream: {reason if reason else 'nenhuma encontrada'}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stdout_raw", type=Path, help="arquivo .stdout.raw a inspecionar")
    args = p.parse_args()
    if not args.stdout_raw.exists():
        print(f"ERRO: {args.stdout_raw} nao existe", file=sys.stderr)
        sys.exit(1)
    inspect(args.stdout_raw)


if __name__ == "__main__":
    main()
