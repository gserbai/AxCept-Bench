#!/usr/bin/env python3
######################################################################
# AxCept-Bench (Sobel) - recorte de janela fixa centrada na placa
# Author: Guilherme Saides Serbai
# Year: 2026
######################################################################
# O QUE ESTE SCRIPT FAZ
#   Gera o conjunto de entrada do workload Sobel a partir do UFPR-ALPR:
#   para cada imagem selecionada, recorta uma janela de tamanho fixo
#   centrada na placa, converte para tons de cinza e grava em PGM (P5),
#   que é o formato que o binário `sobel` lê no AxPike.
#
# REGRAS (aplicadas nesta ordem)
#   1. Lê a anotação (.txt ao lado de cada imagem): placa ("plate:"),
#      tipo de veículo ("type:") e posição da placa ("corners:" ou
#      "position_plate:"). Imagem sem .txt ou sem posição é ignorada.
#   2. --exclude-type motorcycle: tira as motos. A placa antiga de moto
#      tem DUAS linhas, e o "-c br" do OpenALPR só lê placa de UMA linha
#      (carros, ônibus e caminhões ficam).
#   3. --frames largest: cada track do UFPR-ALPR tem 30 frames do mesmo
#      veículo; fica só o frame em que a placa tem a MAIOR área. Regra
#      objetiva (usa só a anotação, não o resultado), escolhendo o frame
#      com mais chance de leitura.
#   4. --min-plate-width 70: descarta o veículo se a placa desse frame
#      tiver menos de 70 px de largura. O OpenALPR (br.conf,
#      min_plate_size_width_px = 65) ignora placas < 65 px; os 5 px a
#      mais são margem. Esses casos falhariam mesmo SEM falha de memória,
#      então não dizem nada sobre a AxRAM.
#   5. --size 320x240: recorta a janela centrada na placa. Se encostar na
#      borda da imagem, a janela é EMPURRADA para dentro (não cortada):
#      todas as saídas têm exatamente 320x240, a placa nunca é cortada e
#      nenhum pixel é redimensionado (resolução original preservada).
#   6. Converte para cinza (1 canal, 8 bits) e salva em PGM (P5).
#   7. --flat: junta training/validation/testing numa pasta só. Não há
#      treino (o OpenALPR é usado pronto), então todos viram teste.
#   8. Gera labels.csv: arquivo, placa, tipo e posição da placa DENTRO do
#      recorte (ground truth usado pelo evaluate_sobel.py).
#
# Todas as regras são opcionais: sem as opções, recorta todas as imagens
# mantendo a estrutura de pastas original.
#
# Uso (configuração do experimento):
#   python3 crop_plates.py <dir_dataset> <dir_saida> \
#       --size 320x240 --frames largest --exclude-type motorcycle \
#       --min-plate-width 70 --flat
#
# Outras opções:
#   --frames 15 / 1,15,30   frames fixos por track em vez do maior
#   --type car              só um tipo de veículo
######################################################################

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

IMG_EXTS = (".png", ".jpg", ".jpeg")

# Nome dos frames no UFPR-ALPR: "track0091[15]" -> track "track0091", frame 15
FRAME_RE = re.compile(r"^(.*)\[(\d+)\]$")


# ======================================================================
# REGRA 1 - leitura da anotação
# ======================================================================
def parse_annotation(txt: Path):
    """Retorna (placa, tipo, (x0, y0, x1, y1)) em coordenadas da imagem.

    Exemplo de .txt do UFPR-ALPR:
        type: car
        plate: MLS5511
        corners: 856,504 919,504 919,525 856,525
    Os 4 cantos viram o retângulo que envolve a placa (min/max de x e y).
    """
    plate, vtype, box = None, None, None
    for line in txt.read_text(errors="ignore").splitlines():
        line = line.strip()
        if line.startswith("plate:"):
            plate = line.split(":", 1)[1].strip()
        elif line.startswith("type:"):
            vtype = line.split(":", 1)[1].strip().lower()
        elif line.startswith("position_plate:"):      # formato x y w h
            x, y, w, h = map(int, line.split(":", 1)[1].split())
            box = (x, y, x + w, y + h)
        elif line.startswith("corners:"):             # formato de 4 cantos
            pts = [tuple(map(int, p.split(",")))
                   for p in re.findall(r"-?\d+,-?\d+", line)]
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            box = (min(xs), min(ys), max(xs), max(ys))
    return plate, vtype, box


# ======================================================================
# REGRA 5 - posição da janela de recorte
# ======================================================================
def crop_window(img_w, img_h, box, cw, ch):
    """Canto superior esquerdo (x0, y0) da janela cw x ch.

    1) centro da placa:   cx, cy
    2) janela centrada:   x0 = cx - cw/2,  y0 = cy - ch/2
    3) se sair da imagem, limita x0 a [0, img_w - cw] e y0 a [0, img_h - ch]
       (a janela é empurrada para dentro; a placa continua inteira)
    """
    cx = (box[0] + box[2]) // 2
    cy = (box[1] + box[3]) // 2
    x0 = min(max(cx - cw // 2, 0), img_w - cw)
    y0 = min(max(cy - ch // 2, 0), img_h - ch)
    return x0, y0


# ======================================================================
# REGRA 3 - identificação da track e do frame
# ======================================================================
def track_key(img: Path):
    """(track, frame) para nomes trackXXXX[NN].

    Se o nome não seguir esse padrão (ex.: RodoSol), cada imagem é
    tratada como uma track própria.
    """
    m = FRAME_RE.match(img.stem)
    if m:
        return (img.parent / m.group(1)).as_posix(), int(m.group(2))
    return img.as_posix(), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path, help="pasta do dataset original")
    ap.add_argument("dst", type=Path, help="pasta de saída (criada se não existir)")
    ap.add_argument("--size", default="640x480", help="LxA do recorte (regra 5)")
    ap.add_argument("--frames", default=None,
                    help="'largest' ou lista de frames, ex.: 15 ou 1,15,30 (regra 3)")
    ap.add_argument("--type", default=None, help="só esse tipo, ex.: car")
    ap.add_argument("--exclude-type", default=None,
                    help="tira esse tipo, ex.: motorcycle (regra 2)")
    ap.add_argument("--min-plate-width", type=int, default=0,
                    help="largura mínima da placa em px (regra 4)")
    ap.add_argument("--flat", action="store_true",
                    help="grava todos os recortes direto em <dst>/ (regra 7)")
    args = ap.parse_args()

    cw, ch = map(int, args.size.lower().split("x"))
    images = sorted(p for p in args.src.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not images:
        sys.exit(f"Nenhuma imagem em {args.src}")

    # Contador de imagens ignoradas por motivo (aparece no resumo final)
    skip = defaultdict(int)

    # ==================================================================
    # REGRAS 1 e 2 - lê anotações e filtra por tipo de veículo
    # ==================================================================
    cand = []  # (imagem, placa, tipo, bbox)
    for img in images:
        txt = img.with_suffix(".txt")
        if not txt.exists():
            skip["sem anotação"] += 1
            continue
        plate, vtype, box = parse_annotation(txt)
        if box is None:
            skip["sem posição da placa"] += 1
            continue
        if args.type and vtype != args.type.lower():
            skip[f"tipo != {args.type}"] += 1
            continue
        # Regra 2: placa de moto (2 linhas) não é lida com "-c br"
        if args.exclude_type and vtype == args.exclude_type.lower():
            skip[f"tipo = {args.exclude_type}"] += 1
            continue
        cand.append((img, plate, vtype, box))

    # ==================================================================
    # REGRA 3 - um frame por veículo
    # ==================================================================
    if args.frames == "largest":
        # Para cada track, guarda o frame com a maior área de placa
        best = {}
        for c in cand:
            k, _ = track_key(c[0])
            b = c[3]
            area = (b[2] - b[0]) * (b[3] - b[1])
            if k not in best or area > best[k][0]:
                best[k] = (area, c)
        n_before = len(cand)
        cand = [v[1] for v in best.values()]
        skip["outros frames da track"] += n_before - len(cand)
    elif args.frames:
        # Alternativa: frames fixos (ex.: o 15 de todas as tracks)
        wanted = {int(f) for f in args.frames.split(",")}
        keep = [c for c in cand if track_key(c[0])[1] in wanted]
        skip["frame fora da lista"] += len(cand) - len(keep)
        cand = keep

    # ==================================================================
    # REGRA 4 - descarta placas menores que o mínimo do OpenALPR
    # (aplicada DEPOIS da regra 3: se nem a maior placa da track passa,
    #  o veículo sai do experimento)
    # ==================================================================
    if args.min_plate_width:
        keep = [c for c in cand if c[3][2] - c[3][0] >= args.min_plate_width]
        skip[f"placa < {args.min_plate_width}px"] += len(cand) - len(keep)
        cand = keep

    # ==================================================================
    # REGRAS 5, 6, 7 e 8 - recorta, converte, salva e gera labels.csv
    # ==================================================================
    args.dst.mkdir(parents=True, exist_ok=True)
    ok = 0
    with open(args.dst / "labels.csv", "w", newline="") as fcsv:
        wr = csv.writer(fcsv)
        wr.writerow(["file", "plate", "type", "x0", "y0", "x1", "y1"])

        for img_path, plate, vtype, box in sorted(cand):
            img = Image.open(img_path)
            W, H = img.size
            if cw > W or ch > H:
                sys.exit(f"Recorte {cw}x{ch} maior que a imagem {W}x{H}: {img_path}")
            if box[2] - box[0] > cw or box[3] - box[1] > ch:
                print(f"[WARN] placa maior que o recorte: {img_path}")

            # Regra 5: janela fixa centrada na placa (sem redimensionar)
            x0, y0 = crop_window(W, H, box, cw, ch)
            # Regra 6: tons de cinza (1 canal, 8 bits)
            crop = img.convert("L").crop((x0, y0, x0 + cw, y0 + ch))

            # Regra 7: pasta única ou estrutura original
            if args.flat:
                rel = Path(img_path.stem + ".pgm")
            else:
                rel = img_path.relative_to(args.src).with_suffix(".pgm")
            out = args.dst / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            crop.save(out)  # extensão .pgm -> PIL grava P5 (binário)

            # Regra 8: posição da placa convertida para coordenadas do
            # recorte (subtrai o canto da janela: x - x0, y - y0)
            wr.writerow([rel.as_posix(), plate, vtype or "",
                         box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0])
            ok += 1
            print(f"[CROP] {rel} ({plate}, {box[2] - box[0]}x{box[3] - box[1]}px)")

    # Resumo: quantos recortes e quantas imagens cada regra descartou
    print(f"\n[RESUMO] {ok} recortes {cw}x{ch}")
    for reason, n in skip.items():
        if n:
            print(f"  ignoradas ({reason}): {n}")
    print(f"[INFO] labels em {args.dst / 'labels.csv'}")


if __name__ == "__main__":
    main()