# Sobel Benchmark — Memory Fault Susceptibility

Filtro de Sobel 3x3 (magnitude do gradiente) executado no AxPike/AxRAM via `pk`.
Entrada e saída em PGM binário (P5, tons de cinza).

## 1. Compilar

```bash
$ riscv64-unknown-elf-g++ -O3 -static -o ./sobel sobel.cpp -lm
```

(Lembre de usar o `pk` com o `printm("bbl loader")` comentado, como nos outros workloads.)

## 2. Preparar o dataset (PNG -> PGM)

O [`utils/crop_plates.py`](utils/crop_plates.py) recorta uma janela fixa centrada na placa de cada imagem do UFPR-ALPR, converte para tons de cinza e grava em PGM (P5). As regras de seleção estão no cabeçalho do script.

```bash
$ python3 applications/sobel/utils/crop_plates.py /caminho/UFPR-ALPR /caminho/UFPR-ALPR/dataset_pgm \
    --size 320x240 --frames largest --exclude-type motorcycle \
    --min-plate-width 70 --flat
```

## 3. Execução única

```bash
$ { printf '%64s' ''; cat input.pgm; } | \
  axpike --adele=mem_read_prob:1e-3,linesz:32 \
         --adele-activate=0:AXRAM \
         --dc=128:8:32 --ic=256:4:32 --l2=1024:4:32 \
         pk ./sobel > output.pgm
```

O prefixo de 64 espaços cobre o consumo de stdin do `pk`; o parser ignora
whitespace antes do `P5`, então funciona com ou sem esse consumo.

## 4. Execução em lote

```bash
$ python3 applications/sobel/axcept_batch_sobel.py
```

Ajuste `ERROR_RATE`, `DATASET_DIR` e `APP_BIN` no topo do script. Para calcular a energia, organize as execuções como na [seção 5 do guia das aplicações](../usage.md#5-organizing-the-runs-for-the-energy-pipeline).

## 5. Avaliação das saídas (crashes + OpenALPR)

O [`utils/evaluate_sobel.py`](utils/evaluate_sobel.py) classifica cada saída do AxPike, converte para PNG as que podem ser avaliadas e, com `--alpr`, roda o OpenALPR. As regras estão no cabeçalho do script.

```bash
$ python3 applications/sobel/utils/evaluate_sobel.py \
    applications/sobel/src/dataset_error_rate_1e-3 \
    /caminho/UFPR-ALPR/dataset_pgm/labels.csv \
    out_png/ --alpr \
    --alpr-cmd "docker run --rm -v {dir}:/data:ro openalpr -c br -j {name}"
```

| Classe | Condição | Vai para o OpenALPR |
|---|---|---|
| `FLOW_TIMEOUT` | saída vazia (0 bytes) | não (fatal) |
| `DATA_CRASH` | segfault com < 1024 bytes de pixels | não (fatal) |
| `INVALID` | header corrompido ou truncado, sem segfault | não (fatal) |
| `SALVAGED_CRASH` | segfault com ≥ 1024 bytes de pixels (linhas faltantes = 0) | sim |
| `SURVIVOR` | imagem completa, sem dump | sim |

O `sobel.cpp` escreve a saída linha a linha, então num crash as linhas já calculadas ficam antes do dump (mesma regra de 1024 bytes do JPEG). Uma saída é `ACCEPTABLE` quando a placa lida pelo OpenALPR (primeiro candidato no formato `LLL####`) é igual à do `labels.csv`. Os resultados por imagem ficam em `out_png/results.csv`.

O reconhecimento de placas com OpenALPR está em [`acceptability_semantic/platerecognition/howtouse.md`](acceptability_semantic/platerecognition/howtouse.md).