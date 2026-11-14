#!/usr/bin/env python3
######################################################################
# AxCept-Bench (Sobel) - qualidade estrutural (SSIM / PSNR) vs. baseline
# Author: Guilherme Saides Serbai
# Year: 2026
######################################################################
# O QUE ESTE SCRIPT FAZ
#   Mede a QUALIDADE das saídas do Sobel em cada taxa de erro comparando
#   com a BASELINE: a saída do próprio AxPike com taxa de erro 0.
#   Mesmo esquema do script de análise do FFT. Separado do
#   evaluate_sobel.py, que mede crashes e ACEITABILIDADE (OpenALPR).
#
# ESTRUTURA ESPERADA (saídas do axcept_batch_sobel.py)
#   <base-dir>/dataset_error_rate_0/<arquivo>.pgm        baseline (taxa 0)
#   <base-dir>/dataset_error_rate_<taxa>/<arquivo>.pgm   uma pasta por taxa
#   (a baseline também é aceita como dataset_error_rate_exact; ou use
#    --exact-dir para informar outra pasta)
#
# PAREAMENTO
#   A árvore da baseline define as amostras esperadas. Uma saída
#   aproximada só é comparada com a baseline que tem o MESMO CAMINHO
#   RELATIVO COMPLETO (ex.: track0001[01].pgm, ou testing/track0091/...).
#   Se não existir correspondente, NÃO calcula:
#     MISSING_OUTPUT  existe na baseline, falta na taxa
#     EXTRA_OUTPUT    existe na taxa, não existe na baseline
#   (listados em sobel_quality_unmatched.csv)
#
# VALIDAÇÃO (mesmas regras do evaluate_sobel.py, função classify importada)
#   Avaliáveis: SURVIVOR, SALVAGED_CRASH
#   Não avaliáveis: DATA_CRASH, FLOW_TIMEOUT, INVALID
#   O par só é comparado se a baseline E a saída aproximada forem
#   avaliáveis; senão, skip_reason = EXACT_<classe> ou APPROX_<classe>.
#   Saídas SALVAGED_CRASH têm as linhas faltantes em preto, então o SSIM
#   delas cai bastante (a classe fica registrada em cada par).
#
# MÉTRICAS
#   SSIM  (principal) - skimage.metrics.structural_similarity, parâmetros
#         padrão (mesma função do structuraljpeg/compute_jpeg_ssim.py).
#   PSNR  (complementar) - dB; saída idêntica à baseline = PSNR_CAP (100).
#   Duas regiões:
#     *_full   imagem inteira
#     *_plate  só a placa (bbox do labels.csv + PLATE_MARGIN px), com
#              --labels; é a parte que importa para a leitura.
#
# USO
#   python3 quality_sobel.py --base-dir applications/sobel/src \
#       --rates 1e-5 1e-4 1e-3 1e-2 1e-1 \
#       --labels /caminho/ufpr_sobel_set/labels.csv \
#       --out-dir /caminho/sobel_quality
#
# SAÍDAS (em --out-dir)
#   sobel_quality_by_rate.csv   uma linha por taxa: classes, pares
#                               comparados, média/mediana de SSIM e PSNR
#   sobel_quality_all_pairs.csv uma linha por arquivo e taxa (auditoria)
#   sobel_quality_unmatched.csv arquivos sem correspondente
######################################################################

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median

import numpy as np
from skimage.metrics import structural_similarity as ssim

# Mesma classificação do evaluate_sobel.py (arquivo na mesma pasta)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_sobel import classify  # noqa: E402

RATES_DEFAULT = ["1e-5", "1e-4", "1e-3", "1e-2", "1e-1"]
USABLE_CLASSES = {"SURVIVOR", "SALVAGED_CRASH"}
PSNR_CAP = 100.0   # dB, para saídas idênticas à baseline
PLATE_MARGIN = 4   # px em volta da bbox da placa (contexto para a janela do SSIM)

PAIR_FIELDNAMES = [
    "rate", "relative_path", "exact_class", "approx_class",
    "ssim_full", "psnr_full", "ssim_plate", "psnr_plate",
    "compared", "skip_reason",
]
UNMATCHED_FIELDNAMES = ["rate", "relative_path", "issue", "path"]


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="SSIM/PSNR das saídas do Sobel vs. baseline (taxa 0).")
    ap.add_argument("--base-dir", type=Path, default=Path("applications/sobel/src"),
                    help="pasta com dataset_error_rate_0 e dataset_error_rate_<taxa>")
    ap.add_argument("--exact-dir", type=Path, default=None,
                    help="pasta da baseline, se não estiver em --base-dir")
    ap.add_argument("--rates", nargs="+", default=RATES_DEFAULT,
                    help=f"taxas analisadas (padrão: {' '.join(RATES_DEFAULT)})")
    ap.add_argument("--labels", type=Path, default=None,
                    help="labels.csv do crop_plates.py (ativa as métricas da placa)")
    ap.add_argument("--out-dir", type=Path, default=Path("sobel_quality_analysis"))
    args = ap.parse_args()
    args.rates = sorted(args.rates, key=float)
    return args


# =============================================================================
# Métricas
# =============================================================================

def psnr(ref: np.ndarray, img: np.ndarray) -> float:
    mse = np.mean((ref.astype(np.float64) - img.astype(np.float64)) ** 2)
    if mse == 0:
        return PSNR_CAP
    return min(PSNR_CAP, 10 * math.log10(255.0 ** 2 / mse))


def plate_region(arr: np.ndarray, box):
    """Recorta a bbox da placa (+ margem), limitada às bordas da imagem."""
    h, w = arr.shape
    x0, y0, x1, y1 = box
    x0, y0 = max(0, x0 - PLATE_MARGIN), max(0, y0 - PLATE_MARGIN)
    x1, y1 = min(w, x1 + PLATE_MARGIN), min(h, y1 + PLATE_MARGIN)
    return arr[y0:y1, x0:x1]


# =============================================================================
# Baseline e pareamento
# =============================================================================

def discover_exact_root(base_dir: Path, exact_dir: Path | None) -> Path:
    if exact_dir is not None:
        if not exact_dir.is_dir():
            raise SystemExit(f"Pasta da baseline não encontrada: {exact_dir}")
        return exact_dir
    for name in ("dataset_error_rate_0", "dataset_error_rate_exact"):
        cand = base_dir / name
        if cand.is_dir():
            return cand
    raise SystemExit(
        f"Baseline não encontrada em {base_dir} "
        "(dataset_error_rate_0 ou dataset_error_rate_exact). Use --exact-dir.")


def relative_pgm_paths(root: Path) -> list[Path]:
    """Caminhos relativos completos (ex.: track0001[01].pgm)."""
    return sorted(p.relative_to(root) for p in root.rglob("*.pgm") if p.is_file())


def compare_pair(exact_pix, approx_pix, box) -> dict:
    out = {"ssim_full": round(float(ssim(exact_pix, approx_pix)), 6),
           "psnr_full": round(psnr(exact_pix, approx_pix), 4),
           "ssim_plate": "", "psnr_plate": ""}
    if box is not None:
        ep, ap_ = plate_region(exact_pix, box), plate_region(approx_pix, box)
        if min(ep.shape) >= 7:  # janela padrão do SSIM é 7x7
            out["ssim_plate"] = round(float(ssim(ep, ap_)), 6)
        out["psnr_plate"] = round(psnr(ep, ap_), 4)
    return out


# =============================================================================
# Análise por taxa
# =============================================================================

def stats(rows, col):
    vals = [r[col] for r in rows if r[col] != ""]
    if not vals:
        return math.nan, math.nan
    return mean(vals), median(vals)


def analyze_one_rate(rate, exact_root, approx_root, exact_paths, exact_recs, boxes):
    expected = set(exact_paths)
    extra = sorted(set(relative_pgm_paths(approx_root)) - expected)

    counts, pairs, unmatched = Counter(), [], []
    for rel in exact_paths:
        e_cls, e_pix = exact_recs[rel]
        a_path = approx_root / rel
        row = {"rate": rate, "relative_path": rel.as_posix(),
               "exact_class": e_cls, "approx_class": "",
               "ssim_full": "", "psnr_full": "", "ssim_plate": "", "psnr_plate": "",
               "compared": False, "skip_reason": ""}

        if not a_path.is_file():
            row["approx_class"] = "MISSING_OUTPUT"
            row["skip_reason"] = "MISSING_OUTPUT"
            unmatched.append({"rate": rate, "relative_path": rel.as_posix(),
                              "issue": "MISSING_OUTPUT", "path": str(a_path)})
        else:
            a_cls, a_pix = classify(a_path)
            row["approx_class"] = a_cls
            if e_cls not in USABLE_CLASSES:
                row["skip_reason"] = f"EXACT_{e_cls}"
            elif a_cls not in USABLE_CLASSES:
                row["skip_reason"] = f"APPROX_{a_cls}"
            elif e_pix.shape != a_pix.shape:
                row["skip_reason"] = "SHAPE_MISMATCH"
            else:
                row.update(compare_pair(e_pix, a_pix, boxes.get(rel.as_posix())))
                row["compared"] = True

        counts[row["approx_class"]] += 1
        pairs.append(row)

    for rel in extra:
        unmatched.append({"rate": rate, "relative_path": rel.as_posix(),
                          "issue": "EXTRA_OUTPUT", "path": str(approx_root / rel)})

    comp = [r for r in pairs if r["compared"]]
    total = len(exact_paths)
    m = {"rate": rate, "total_expected_from_exact": total}
    for cls in ("SURVIVOR", "SALVAGED_CRASH", "DATA_CRASH", "FLOW_TIMEOUT",
                "INVALID", "MISSING_OUTPUT"):
        m[cls.lower()] = counts[cls]
    m["extra_output"] = len(extra)
    m["pairs_compared"] = len(comp)
    m["pairs_compared_rate_total"] = 100.0 * len(comp) / total if total else 0.0
    for col in ("ssim_full", "ssim_plate", "psnr_full", "psnr_plate"):
        m[f"{col}_mean"], m[f"{col}_median"] = stats(comp, col)
    surv = [r for r in comp if r["approx_class"] == "SURVIVOR"]
    m["ssim_full_mean_survivor_only"], _ = stats(surv, "ssim_full")
    m["ssim_plate_mean_survivor_only"], _ = stats(surv, "ssim_plate")
    return m, pairs, unmatched


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None):
    if not rows and fieldnames is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames or list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    exact_root = discover_exact_root(args.base_dir, args.exact_dir)
    exact_paths = relative_pgm_paths(exact_root)
    if not exact_paths:
        raise SystemExit(f"Nenhum .pgm na baseline: {exact_root}")

    exact_recs = {rel: classify(exact_root / rel) for rel in exact_paths}
    exact_counts = Counter(c for c, _ in exact_recs.values())

    boxes = {}
    if args.labels:
        boxes = {r["file"]: tuple(int(r[k]) for k in ("x0", "y0", "x1", "y1"))
                 for r in csv.DictReader(open(args.labels))}

    print("AxCept-Bench Sobel quality analysis (SSIM/PSNR vs. baseline)")
    print("=" * 72)
    print(f"Baseline (taxa 0)    : {exact_root}")
    print(f"Arquivos na baseline : {len(exact_paths)}  {dict(exact_counts)}")
    print(f"Taxas                : {', '.join(args.rates)}")
    print(f"Métricas da placa    : {'sim' if boxes else 'não (sem --labels)'}")
    print("Pareamento           : caminho relativo completo")
    print("=" * 72)

    all_m, all_pairs, all_unm, skipped = [], [], [], []
    for rate in args.rates:
        approx_root = args.base_dir / f"dataset_error_rate_{rate}"
        print(f"\nTaxa {rate}  ({approx_root})")
        if not approx_root.is_dir():
            print("  [SKIP] pasta não encontrada")
            skipped.append(rate)
            continue
        m, pairs, unm = analyze_one_rate(rate, exact_root, approx_root,
                                         exact_paths, exact_recs, boxes)
        all_m.append(m)
        all_pairs.extend(pairs)
        all_unm.extend(unm)
        print(f"  Comparados         : {m['pairs_compared']} / {m['total_expected_from_exact']}")
        print(f"  Survivor / Salvaged: {m['survivor']} / {m['salvaged_crash']}")
        print(f"  Crash/Timeout/Inv. : {m['data_crash']} / {m['flow_timeout']} / {m['invalid']}")
        print(f"  Missing / Extra    : {m['missing_output']} / {m['extra_output']}")
        print(f"  SSIM full  (média) : {m['ssim_full_mean']:.4f}")
        print(f"  SSIM placa (média) : {m['ssim_plate_mean']:.4f}")
        print(f"  PSNR full  (média) : {m['psnr_full_mean']:.2f} dB")

    if not all_m:
        raise SystemExit("Nenhuma taxa analisada. Verifique as pastas.")

    out = args.out_dir
    write_csv(out / "sobel_quality_by_rate.csv", all_m)
    write_csv(out / "sobel_quality_all_pairs.csv", all_pairs, PAIR_FIELDNAMES)
    write_csv(out / "sobel_quality_unmatched.csv", all_unm, UNMATCHED_FIELDNAMES)

    print("\n" + "=" * 72)
    print(f"Por taxa   : {out / 'sobel_quality_by_rate.csv'}")
    print(f"Pares      : {out / 'sobel_quality_all_pairs.csv'}")
    print(f"Sem par    : {out / 'sobel_quality_unmatched.csv'}")
    if skipped:
        print(f"Taxas puladas: {', '.join(skipped)}")


if __name__ == "__main__":
    main()
