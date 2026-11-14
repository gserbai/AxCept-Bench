#!/usr/bin/env python3
######################################################################
# AxCept-Bench (Sobel) - validação das saídas + aceitabilidade (OpenALPR)
# Author: Guilherme Saides Serbai
# Year: 2026
######################################################################
# O QUE ESTE SCRIPT FAZ
#   Último passo do workload Sobel: pega as saídas do AxPike e responde
#   (1) a execução sobreviveu? e (2) a placa continua legível?
#
# ---------------------------------------------------------------------
# ETAPA 1 - CLASSIFICAÇÃO DAS SAÍDAS (mesmo critério do crash_analyzer_jpeg.py)
# ---------------------------------------------------------------------
#   FLOW_TIMEOUT    arquivo vazio (0 bytes)                         -> fatal
#   DATA_CRASH      segfault com < MIN_PAYLOAD bytes de pixels      -> fatal
#   SALVAGED_CRASH  segfault com >= MIN_PAYLOAD bytes de pixels     -> avaliado
#   INVALID         header corrompido ou truncado sem segfault      -> fatal
#   SURVIVOR        header válido + w*h pixels, sem dump            -> avaliado
#
#   Por que o dump aparece no arquivo: o pk/AxPike imprime o dump do
#   crash ("z  0000... ra ...", "User load segfault @ ...") no MESMO
#   stdout da imagem, que é redirecionado para o arquivo de saída.
#
#   Regra dos 1024 bytes (igual ao JPEG): o sobel.cpp escreve linha a
#   linha com fflush, então num crash as linhas já calculadas ficam no
#   arquivo ANTES do dump. Se houver >= 1024 bytes de pixels, a saída é
#   aproveitada (SALVAGED_CRASH): mantém as linhas completas e preenche
#   o resto com 0 (preto). O início do dump é detectado pela linha
#   "z  <16 dígitos hex>"; se não for encontrado, classifica como
#   DATA_CRASH (lado seguro).
#
# ---------------------------------------------------------------------
# ETAPA 2 - ACEITABILIDADE COM OpenALPR (opção --alpr)
# ---------------------------------------------------------------------
#   SURVIVOR e SALVAGED_CRASH viram PNG (o CLI `alpr` só aceita
#   .png/.jpg/.tif/.bmp/.jpeg/.gif, não .pgm - ver is_supported_image()
#   em openalpr/src/main.cpp) e passam pelo OpenALPR.
#   ACCEPTABLE   = placa lida igual à do labels.csv (ignora hífen/maiúsculas)
#   UNACCEPTABLE = leu outra placa ou nenhuma
#
#   OpenALPR via Docker (imagem compilada do submódulo, commit 736ab0e):
#     --alpr-cmd "docker run --rm -v {dir}:/data:ro openalpr -c br -j {name}"
#
# ---------------------------------------------------------------------
# POR QUE O FILTRO DE FORMATO LLL#### (e não "-p <estado>")
# ---------------------------------------------------------------------
#   O OpenALPR tem dois parâmetros diferentes:
#     -c br       PAÍS: tamanho físico da placa, fonte, OCR treinado para
#                 caracteres brasileiros, detector. Faz ele ACHAR e LER a
#                 placa, mas NÃO impõe o formato -> pode confundir O/0,
#                 I/1, B/8 (ex.: leu "AXV88O4" para a placa "AXV8804").
#     -p <estado> PADRÃO: aplica as regras de runtime_data/postprocess/
#                 br.patterns, que são POR ESTADO (ex.: "pr A@@####",
#                 "sp B[F-Z]@####"). Só aí ele corrige O->0 etc.
#                 Na saída JSON, "matches_template": 0 indica que nenhum
#                 padrão foi aplicado.
#   O UFPR-ALPR tem placas de VÁRIOS estados e só dá para passar UM -p por
#   execução; um estado errado "corrigiria" placas para o formato errado.
#   Solução (genérica, vale para todos os estados): entre os top-N
#   candidatos do alpr, escolhe o PRIMEIRO no formato comum a todas as
#   placas brasileiras antigas, LLL#### (3 letras + 4 dígitos). É o mesmo
#   efeito do -p, sem precisar saber o estado de cada placa.
#   --no-format-filter usa o top-1 cru. O results.csv guarda as duas
#   leituras (plate_read = com filtro, plate_top1_raw = cru).
#   Texto para o artigo: "the first top-10 OpenALPR candidate matching the
#   Brazilian plate format (LLL####) was taken as the reading".
#
# ---------------------------------------------------------------------
# USO
# ---------------------------------------------------------------------
#   python3 evaluate_sobel.py <dir_saidas_axpike> <labels.csv> <dir_png> \
#       --alpr --alpr-cmd "docker run --rm -v {dir}:/data:ro openalpr -c br -j {name}"
#   Sem --alpr: só classifica e gera os PNGs.
#
# SAÍDA
#   <dir_png>/results.csv: file, class, plate_gt, plate_read,
#                          plate_top1_raw, verdict
#   Resumo no terminal (use "| tee resumo.txt" para salvar).
######################################################################

import argparse
import csv
import json
import re
import shlex
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


# ===============================
# Etapa 1: classificação
# ===============================

def parse_pgm_header(data: bytes):
    """Retorna (w, h, offset_pixels) ou None se o header for inválido."""
    tokens, i = [], 0
    while len(tokens) < 4:
        while i < len(data) and data[i:i + 1].isspace():
            i += 1
        if i >= len(data):
            return None
        if data[i:i + 1] == b"#":
            while i < len(data) and data[i:i + 1] != b"\n":
                i += 1
            continue
        j = i
        while j < len(data) and not data[j:j + 1].isspace():
            j += 1
        tokens.append(data[i:j])
        i = j
    try:
        if tokens[0] != b"P5" or int(tokens[3]) != 255:
            return None
        w, h = int(tokens[1]), int(tokens[2])
    except ValueError:
        return None
    if not (0 < w <= 8192 and 0 < h <= 8192):
        return None
    return w, h, i + 1


MIN_PAYLOAD = 1024  # bytes de pixels antes do dump (mesma regra do JPEG)

# Início do dump do pk/AxPike: linha "z  <16 dígitos hex> ra ..."
DUMP_START = re.compile(rb"z  [0-9a-fA-F]{16}")


def classify(path: Path):
    data = path.read_bytes()
    if len(data) == 0:
        return "FLOW_TIMEOUT", None

    if b"segfault" in data.lower():
        hdr = parse_pgm_header(data)
        if hdr is None:
            return "DATA_CRASH", None
        w, h, off = hdr
        m = DUMP_START.search(data, off)
        if m is None:  # não achou onde o dump começa: não arrisca
            return "DATA_CRASH", None
        avail = m.start() - off
        if avail < MIN_PAYLOAD:
            return "DATA_CRASH", None
        nrows = min(avail // w, h)  # só linhas completas
        pix = np.zeros((h, w), dtype=np.uint8)
        pix[:nrows] = np.frombuffer(data[off:off + nrows * w],
                                    dtype=np.uint8).reshape(nrows, w)
        return "SALVAGED_CRASH", pix

    hdr = parse_pgm_header(data)
    if hdr is None:
        return "INVALID", None
    w, h, off = hdr
    if len(data) - off < w * h:
        return "INVALID", None
    pix = np.frombuffer(data[off:off + w * h], dtype=np.uint8).reshape(h, w)
    return "SURVIVOR", pix


# ===============================
# Etapa 2: OpenALPR
# ===============================

def norm(plate: str) -> str:
    return "".join(c for c in plate.upper() if c.isalnum())


# Comando do OpenALPR. Placeholders: {file} caminho do PNG,
# {dir} pasta do PNG, {name} nome do arquivo.
#   instalado no sistema: alpr -c br -j {file}
#   via Docker:           docker run --rm -v {dir}:/data:ro openalpr -c br -j {name}
DEFAULT_ALPR_CMD = "alpr -c br -j {file}"


# Formato das placas brasileiras antigas (o mesmo do UFPR-ALPR): LLL####
# O alpr só aplica o br.patterns quando recebe o estado (-p), e o dataset
# tem placas de vários estados; sem isso ele confunde O/0, I/1, B/8...
# Por isso escolhemos, entre os top-N candidatos, o primeiro que tem o
# formato LLL#### (o mesmo que o pós-processamento do OpenALPR faria).
BR_FORMAT = re.compile(r"^[A-Z]{3}[0-9]{4}$")


def run_alpr(png: Path, cmd_tpl: str, use_format: bool):
    """Retorna (placa escolhida, placa top-1 crua). '' se nenhuma placa."""
    png = png.resolve()
    cmd = [t.format(file=png, dir=png.parent, name=png.name)
           for t in shlex.split(cmd_tpl)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    try:
        res = json.loads(r.stdout).get("results", [])
    except json.JSONDecodeError:
        return "", ""
    if not res:
        return "", ""
    top1 = res[0]["plate"]
    if use_format:
        for c in res[0].get("candidates", []):
            if BR_FORMAT.match(norm(c["plate"])):
                return c["plate"], top1
    return top1, top1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outputs", type=Path, help="pasta com os .pgm do AxPike")
    ap.add_argument("labels", type=Path, help="labels.csv do crop_plates.py")
    ap.add_argument("png_dir", type=Path, help="onde salvar PNGs + results.csv")
    ap.add_argument("--alpr", action="store_true", help="rodar o OpenALPR")
    ap.add_argument("--alpr-cmd", default=DEFAULT_ALPR_CMD,
                    help="comando do alpr (ver DEFAULT_ALPR_CMD)")
    ap.add_argument("--no-format-filter", action="store_true",
                    help="usa o top-1 cru do alpr (sem filtrar por LLL####)")
    args = ap.parse_args()

    gt = {r["file"]: r["plate"] for r in csv.DictReader(open(args.labels))}
    files = sorted(args.outputs.rglob("*.pgm"))
    if not files:
        raise SystemExit(f"Nenhum .pgm em {args.outputs}")

    args.png_dir.mkdir(parents=True, exist_ok=True)
    counts, rows = Counter(), []

    for f in files:
        rel = f.relative_to(args.outputs).as_posix()
        cls, pix = classify(f)
        plate_gt = gt.get(rel, "")
        read, top1, verdict = "", "", cls

        if cls in ("SURVIVOR", "SALVAGED_CRASH"):
            png = args.png_dir / Path(rel).with_suffix(".png")
            png.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pix, "L").save(png)
            if args.alpr:
                read, top1 = run_alpr(png, args.alpr_cmd,
                                      not args.no_format_filter)
                ok = read and norm(read) == norm(plate_gt)
                verdict = "ACCEPTABLE" if ok else "UNACCEPTABLE"

        counts[verdict] += 1
        rows.append([rel, cls, plate_gt, read, top1, verdict])
        print(f"[{verdict}] {rel} {('-> ' + read) if read else ''}")

    with open(args.png_dir / "results.csv", "w", newline="") as fo:
        wr = csv.writer(fo)
        wr.writerow(["file", "class", "plate_gt", "plate_read",
                     "plate_top1_raw", "verdict"])
        wr.writerows(rows)

    n = len(files)
    fatal = counts["DATA_CRASH"] + counts["FLOW_TIMEOUT"] + counts["INVALID"]
    print("\n" + "=" * 50)
    print(f" Total                      : {n}")
    salv = sum(1 for r in rows if r[1] == "SALVAGED_CRASH")
    print(f" [Data Crash] (segfault)    : {counts['DATA_CRASH']}")
    print(f" [Salvaged] (crash >=1024B) : {salv}")
    print(f" [Flow/Timeout] (0 bytes)   : {counts['FLOW_TIMEOUT']}")
    print(f" [Invalid] (header/truncado): {counts['INVALID']}")
    print(f" Fatal rate                 : {100 * fatal / n:.2f}%")
    if args.alpr:
        print(f" Acceptable (alpr = GT)     : {counts['ACCEPTABLE']}")
        print(f" Unacceptable               : {counts['UNACCEPTABLE']}")
        print(f" Acceptability (sobre total): {100 * counts['ACCEPTABLE'] / n:.2f}%")
    else:
        print(f" Survivors                  : {counts['SURVIVOR']}")
        print(f" Salvaged (avaliáveis)      : {counts['SALVAGED_CRASH']}")
    print("=" * 50)
    print(f"[INFO] {args.png_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
