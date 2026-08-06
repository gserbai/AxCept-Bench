#!/usr/bin/env python3
from pathlib import Path
import argparse
import subprocess
import shutil
import re
import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


LOG_RE = re.compile(r"^AXRAM_log_pid\d+_hart\d+\.log$")


def patch_config(src_config: Path, dst_config: Path):
    text = src_config.read_text(encoding="utf-8", errors="ignore")

    if re.search(r"^[ \t]*record_cmd_trace[ \t]*=", text, flags=re.MULTILINE):
        text = re.sub(
            r"^([ \t]*)record_cmd_trace[ \t]*=.*$",
            r"\1record_cmd_trace = on",
            text,
            flags=re.MULTILINE,
        )
    else:
        text += "\nrecord_cmd_trace = on\n"

    if re.search(r"^[ \t]*print_cmd_trace[ \t]*=", text, flags=re.MULTILINE):
        text = re.sub(
            r"^([ \t]*)print_cmd_trace[ \t]*=.*$",
            r"\1print_cmd_trace = off",
            text,
            flags=re.MULTILINE,
        )
    else:
        text += "\nprint_cmd_trace = off\n"

    dst_config.write_text(text, encoding="utf-8")


def normalize_addr(addr: str) -> str:
    addr = addr.strip()

    if addr.startswith(("0x", "0X")):
        raw = addr[2:]
    else:
        raw = addr

    # Seus endereços vêm em hexadecimal sem 0x.
    int(raw, 16)  # valida
    return "0x" + raw.lower()


def normalize_cpu_trace(input_file: Path, output_file: Path):
    total = 0
    malformed = 0

    with input_file.open("r", encoding="utf-8", errors="ignore") as fin, \
         output_file.open("w", encoding="utf-8") as fout:

        for line_no, line in enumerate(fin, start=1):
            parts = line.strip().split()

            if not parts:
                continue

            # Formato esperado:
            # num_cpu_instr read_addr [writeback_addr]
            if len(parts) < 2:
                malformed += 1
                continue

            try:
                cpu_instr = int(parts[0], 10)
                addrs = [normalize_addr(a) for a in parts[1:]]
                fout.write(str(cpu_instr) + " " + " ".join(addrs) + "\n")
                total += 1
            except Exception:
                malformed += 1

    return total, malformed


def run_one(trace_file: Path, input_root: Path, out_root: Path,
            ramulator_bin: Path, base_config: Path, force: bool):
    rel = trace_file.relative_to(input_root)

    # Exemplo:
    # dataset_error_rate_1e-1/AXRAM_log_pid3065_hart0.log
    # vira:
    # dataset_error_rate_1e-1/AXRAM_log_pid3065_hart0/
    run_dir = out_root / rel.parent / trace_file.stem

    stats_file = run_dir / "DDR3.stats"
    stdout_file = run_dir / "stdout.txt"
    stderr_file = run_dir / "stderr.txt"
    normalized_trace = run_dir / "trace.cpu"
    used_config = run_dir / "DDR3-config-used.cfg"

    if stats_file.exists() and stats_file.stat().st_size > 0 and not force:
        cmdtraces = list(run_dir.glob("cmd-trace-*.cmdtrace"))
        return {
            "input": str(trace_file),
            "output": str(run_dir),
            "status": "SKIP",
            "trace_lines": "",
            "malformed_lines": "",
            "stats_bytes": stats_file.stat().st_size,
            "cmdtrace_count": len(cmdtraces),
            "elapsed_sec": 0,
        }

    run_dir.mkdir(parents=True, exist_ok=True)

    # Limpa restos antigos dessa execução
    for f in run_dir.glob("cmd-trace-*.cmdtrace"):
        f.unlink()
    for f in [stats_file, stdout_file, stderr_file, normalized_trace, used_config]:
        if f.exists():
            f.unlink()

    patch_config(base_config, used_config)
    trace_lines, malformed_lines = normalize_cpu_trace(trace_file, normalized_trace)

    cmd = [
        str(ramulator_bin),
        str(used_config),
        "--mode=cpu",
        "--stats",
        str(stats_file),
        str(normalized_trace),
    ]

    t0 = time.time()

    with stdout_file.open("w") as out, stderr_file.open("w") as err:
        result = subprocess.run(
            cmd,
            cwd=run_dir,
            stdout=out,
            stderr=err,
            text=True,
        )

    elapsed = time.time() - t0
    cmdtraces = list(run_dir.glob("cmd-trace-*.cmdtrace"))

    if result.returncode != 0:
        status = "ERROR"
    elif not stats_file.exists() or stats_file.stat().st_size == 0:
        status = "NO_STATS"
    else:
        status = "OK"

    return {
        "input": str(trace_file),
        "output": str(run_dir),
        "status": status,
        "trace_lines": trace_lines,
        "malformed_lines": malformed_lines,
        "stats_bytes": stats_file.stat().st_size if stats_file.exists() else 0,
        "cmdtrace_count": len(cmdtraces),
        "elapsed_sec": round(elapsed, 3),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Executa traces filtrados do AxCept-Bench no Ramulator mantendo estrutura por cenário."
    )

    parser.add_argument(
        "--input-root",
        default=str(Path.home() / "Documents" / "uy_filtrado"),
        help="Raiz dos traces filtrados"
    )

    parser.add_argument(
        "--out-root",
        default=str(Path.home() / "Documents" / "ramulator_results"),
        help="Raiz da saída organizada"
    )

    parser.add_argument(
        "--ramulator",
        default=str(Path.home() / "AxCept-Bench" / "ramulator" / "ramulator"),
        help="Caminho para o binário ramulator"
    )

    parser.add_argument(
        "--config",
        default=str(Path.home() / "AxCept-Bench" / "ramulator" / "configs" / "DDR3-config.cfg"),
        help="Config base do Ramulator"
    )

    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Execuções paralelas. Comece com 1 ou 2."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limita a quantidade de traces para teste."
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Reexecuta mesmo se DDR3.stats já existir."
    )

    args = parser.parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    ramulator_bin = Path(args.ramulator).expanduser().resolve()
    base_config = Path(args.config).expanduser().resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"input-root não existe: {input_root}")

    if not ramulator_bin.exists():
        raise FileNotFoundError(f"ramulator não encontrado: {ramulator_bin}")

    if not base_config.exists():
        raise FileNotFoundError(f"config não encontrada: {base_config}")

    traces = sorted(
        p for p in input_root.rglob("*.log")
        if LOG_RE.match(p.name)
    )

    if args.limit is not None:
        traces = traces[:args.limit]

    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Input root: {input_root}")
    print(f"Output root: {out_root}")
    print(f"Ramulator: {ramulator_bin}")
    print(f"Config: {base_config}")
    print(f"Traces encontrados: {len(traces)}")
    print(f"Jobs: {args.jobs}")
    print()

    summary_path = out_root / "ramulator_batch_summary.csv"

    results = []

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futures = [
            ex.submit(
                run_one,
                trace,
                input_root,
                out_root,
                ramulator_bin,
                base_config,
                args.force,
            )
            for trace in traces
        ]

        for i, fut in enumerate(as_completed(futures), start=1):
            res = fut.result()
            results.append(res)
            print(
                f"[{i}/{len(traces)}] {res['status']} | "
                f"cmdtraces={res['cmdtrace_count']} | "
                f"stats={res['stats_bytes']} bytes | "
                f"{res['input']}"
            )

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "input",
            "output",
            "status",
            "trace_lines",
            "malformed_lines",
            "stats_bytes",
            "cmdtrace_count",
            "elapsed_sec",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print()
    print(f"Resumo salvo em: {summary_path}")


if __name__ == "__main__":
    main()
