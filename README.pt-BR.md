<p align="center">
  <img src="docs/assets/axcept-bench-banner.svg" alt="AxCept-Bench — Approximate Computing Benchmark" width="900">
</p>

# AxCept-Bench

<p align="center">
  <a href="README.md">English</a> ·
  <strong>Português (Brasil)</strong>
</p>

O **AxCept-Bench** é uma infraestrutura experimental para estudar o compromisso entre qualidade de resultado e economia de energia em computação aproximada.

O projeto executa aplicações RISC-V no **AxPike**, injeta falhas de leitura com o modelo **AxRAM**, avalia a degradação das saídas e transforma os acessos de memória em estimativas de energia usando **Ramulator** e **DRAMPower**.

Atualmente existem dois workloads principais:

- **JPEG:** compressão de imagens com análise de crashes, avaliação por SSIM e avaliação de qualidade semântica;
- **FFT / frequência dominante:** processamento de áudio com análise de crashes, medição do erro de frequência e correspondência de notas MIDI.

Este repositório fornece os componentes do benchmark e scripts de apoio. Ele não entrega um experimento fechado. O pesquisador deve escolher datasets, taxas de erro, configurações de memória e métricas, e organizar os resultados de acordo com seu estudo.

> Os datasets não fazem parte do repositório. Alguns scripts usam constantes internas e precisam ser configurados antes da execução.

## Visão geral do fluxo

```text
Dataset
   |
   v
Aplicação RISC-V (JPEG ou FFT)
   |
   v
AxPike + AxRAM
   |---------------------------|
   v                           v
Saída aproximada          Logs de memória
   |                           |
   v                           v
Análise dos Resultados    Filtro de acessos U Y
                               |
                               v
                           Ramulator
                               |
                               v
                           DRAMPower
                               |
                               v
                       Energia e potência
```

Em um experimento típico:

1. compile o workload para RISC-V;
2. escolha um dataset e uma taxa de erro de leitura;
3. execute o workload no AxPike com o AxRAM ativo;
4. guarde separadamente saídas, stderr e traces AxRAM;
5. repita a execução para todas as taxas estudadas;
6. avalie estabilidade e qualidade das saídas;
7. processe os traces no Ramulator e no DRAMPower;
8. compare a qualidade das saídas e o consumo de energia com o baseline exato.

Uma série comum de probabilidades é:

| Cenário | `mem_read_prob` |
|---|---:|
| Exato | `0` |
| Aproximado | `1e-5` |
| Aproximado | `1e-4` |
| Aproximado | `1e-3` |
| Aproximado | `1e-2` |
| Aproximado | `1e-1` |

No baseline exato, o AxRAM pode continuar ativo com probabilidade `0`. Isso permite gerar traces comparáveis aos cenários aproximados.

## Organização do repositório

```text
applications/           workloads JPEG e FFT
axpike-isa-sim/         simulador AxPike
axpike-pk/              Proxy Kernel RISC-V
ramulator/              simulador de memória
DRAMPower-4.1/          modelo de energia usado pelo fluxo atual
drampower/              versão mais nova do DRAMPower
scripts/benchmarks/     runners dos workloads
scripts/analysis/       análises de estabilidade e qualidade
scripts/axram/          tratamento dos logs AxRAM
scripts/ramulator/      execução em lote do Ramulator
scripts/drampower/      execução em lote do DRAMPower
scripts/utils/          conversores e utilitários
```

Mantenha código-fonte, executáveis e resultados separados. Uma organização possível é:

```text
applications/jpeg/src/       fontes JPEG
applications/fft/src/        fontes FFT
bin/jpeg/                     executável JPEG
bin/fft/                      executável FFT
experiments/jpeg/             resultados JPEG
experiments/fft/              resultados FFT
```

Os nomes são apenas uma sugestão. Se usar outra estrutura, ajuste os caminhos nos scripts.

## Dependências

O fluxo foi desenvolvido para Linux e requer:

- Python 3.10 ou mais recente;
- compilador C/C++ e Make;
- toolchain `riscv64-unknown-elf`;
- AxPike;
- Proxy Kernel (`pk`);
- Ramulator;
- DRAMPower 4.1;
- bibliotecas Python específicas para algumas análises.

Pacotes básicos no Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install build-essential autoconf automake libtool pkg-config \
  gawk bison flex texinfo gperf patchutils bc device-tree-compiler \
  libboost-regex-dev libboost-system-dev libmpc-dev libmpfr-dev \
  libgmp-dev zlib1g-dev libexpat1-dev libxerces-c-dev \
  python3 python3-pip python3-venv ffmpeg
```

Bibliotecas usadas pelas análises:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install numpy pillow scikit-image tqdm matplotlib \
  pandas seaborn torch torchvision
```

Os scripts de extração de traces, Ramulator e DRAMPower usam apenas a biblioteca padrão do Python.

## Submódulos

Após clonar o projeto:

```bash
git submodule update --init --recursive
```

## Compilação da infraestrutura

Os comandos abaixo são referências. Prefixos de instalação e opções do toolchain podem mudar conforme o ambiente do laboratório.

### Proxy Kernel

```bash
mkdir -p axpike-pk/build
cd axpike-pk/build
../configure --prefix=/caminho/riscv --host=riscv64-unknown-elf
make -j4
make install
cd ../..
```

> Antes de executar os workloads, consulte o [guia de compilação e execução das aplicações](applications/usage.md).
> Ele explica como silenciar a mensagem `bbl loader` no Proxy Kernel para não contaminar a saída binária, além de reunir os comandos de compilação e execução no AxPike.

### AxPike

```bash
mkdir -p axpike-isa-sim/build
cd axpike-isa-sim/build
../configure --prefix=/caminho/riscv
make -j4
make install
cd ../..
```

Se os programas não forem instalados globalmente, informe os caminhos locais no ambiente de execução do experimento.

### Ramulator

```bash
make -C ramulator -j4
```

### DRAMPower 4.1

```bash
make -C DRAMPower-4.1 -j4 drampower
```

O batch atual usa `DRAMPower-4.1/drampower`. O submódulo em `drampower/` possui outra interface de linha de comando.

## Compilação dos workloads

Crie uma pasta para executáveis fora de `src`:

```bash
mkdir -p bin/jpeg bin/fft
```

### JPEG

```bash
riscv64-unknown-elf-g++ -O3 -static \
  -o bin/jpeg/toojpeg_encoder \
  applications/jpeg/src/main.cpp \
  applications/jpeg/src/toojpeg.cpp
```

### FFT

```bash
riscv64-unknown-elf-g++ -O3 -static \
  -o bin/fft/dominant_freq \
  applications/fft/src/dominant_freq.cpp \
  applications/fft/src/kiss_fft.c -lm
```

Depois da compilação, configure `APP_BIN` no runner correspondente com o caminho do executável. Os runners atuais não expõem todas as opções de configuração pela linha de comando.

## Benchmark JPEG

### Formato da entrada

O workload JPEG recebe arquivos CSV. Cada arquivo deve conter largura e altura, seguidas de `largura × altura × 3` valores RGB.

Exemplo mínimo para uma imagem de 2 × 1 pixels:

```csv
2,1
255,0,0,0,255,0
```

Uma organização de dataset possível é:

```text
dataset_csv/
└── classe/
    └── imagem.csv
```

O repositório não inclui um conversor geral de imagens para esse formato.

### Configuração

Antes da execução, abra:

```text
scripts/benchmarks/jpeg/axcept_batch_jpeg.py
```

Configure pelo menos:

- `ERROR_RATE`;
- `QUALITY`;
- `TIMEOUT_SEC`;
- `DATASET_DIR`;
- `APP_BIN`;
- `OUTPUT_JPEG_DIR`;
- `LOG_DIR`.

O executável pode ficar em `bin/jpeg/toojpeg_encoder`. Use um caminho absoluto ou um caminho correto em relação ao diretório de execução.

### Execução

```bash
python3 scripts/benchmarks/jpeg/axcept_batch_jpeg.py
```

O stdout da aplicação é salvo como JPEG. O stderr de cada execução é salvo separadamente. Os arquivos `AXRAM_log_pid*_hart*.log` são produzidos pelo AxRAM no diretório de execução.

## Benchmark FFT

### Formato da entrada

A entrada deve ser áudio raw `.bin` com:

- um canal;
- taxa de 44.100 Hz;
- amostras `float32` little-endian;
- nenhum cabeçalho.

O utilitário de conversão de AIFF está em:

```text
scripts/utils/convert_iowa_aif_to_bin.py
```

Configure os diretórios dentro do script e execute:

```bash
python3 scripts/utils/convert_iowa_aif_to_bin.py
```

### Configuração

Antes da execução, abra:

```text
scripts/benchmarks/fft/axcept_batch_fft_final_parallel.py
```

Configure pelo menos:

- `ERROR_RATE`;
- `DATASET_DIR`;
- `APP_BIN`;
- `OUTPUT_BIN_DIR`;
- `LOG_DIR`;
- `TIMEOUT_SEC`;
- `MAX_WORKERS`.

O executável pode ficar em `bin/fft/dominant_freq`.

### Execução

```bash
python3 scripts/benchmarks/fft/axcept_batch_fft_final_parallel.py
```

Um registro de saída válido possui 24 bytes e começa com o identificador `AXDFREQ1`. O registro contém status, número de amostras, número de frames e frequência dominante em Hz.

## Organização dos experimentos

Execute cada taxa em um diretório isolado. Por exemplo:

```text
experiments/
├── jpeg/
│   ├── dataset_error_rate_0/
│   ├── dataset_error_rate_1e-5/
│   ├── dataset_error_rate_1e-4/
│   ├── dataset_error_rate_1e-3/
│   ├── dataset_error_rate_1e-2/
│   └── dataset_error_rate_1e-1/
└── fft/
    ├── dataset_error_rate_0/
    ├── dataset_error_rate_1e-5/
    ├── dataset_error_rate_1e-4/
    ├── dataset_error_rate_1e-3/
    ├── dataset_error_rate_1e-2/
    └── dataset_error_rate_1e-1/
```

Em cada cenário, separe:

```text
outputs/                     saída do workload
logs/                        stderr das execuções
AXRAM_log_pid*_hart*.log     traces brutos do AxRAM
```

Essa separação evita misturar resultados com `applications/*/src` e facilita a reprodução do experimento.

## Pipeline de energia

### 1. Filtrar os acessos AxRAM

O extrator seleciona linhas iniciadas por `U Y` e preserva a estrutura dos cenários.

```bash
python3 scripts/axram/extract_uy_structure.py \
  experiments/jpeg \
  --output-dir experiments/jpeg/uy_filtered
```

Para FFT, troque os caminhos:

```bash
python3 scripts/axram/extract_uy_structure.py \
  experiments/fft \
  --output-dir experiments/fft/uy_filtered
```

Arquivos filtrados vazios indicam traces sem acessos `U Y` e devem ser investigados antes da simulação.

### 2. Executar o Ramulator

Exemplo para JPEG:

```bash
python3 scripts/ramulator/run_ramulator_batch.py \
  --input-root experiments/jpeg/uy_filtered \
  --out-root experiments/jpeg/ramulator_results \
  --ramulator ramulator/ramulator \
  --config configs/ramulator/DDR3_2Gb_x16.cfg \
  --jobs 4
```

Teste primeiro uma entrada:

```bash
python3 scripts/ramulator/run_ramulator_batch.py \
  --input-root experiments/jpeg/uy_filtered \
  --out-root experiments/jpeg/ramulator_test \
  --ramulator ramulator/ramulator \
  --config configs/ramulator/DDR3_2Gb_x16.cfg \
  --limit 1 --jobs 1
```

Uma execução válida deve produzir `DDR3.stats` e pelo menos um arquivo `cmd-trace-*.cmdtrace`.

### 3. Executar o DRAMPower

O mapa de tensão relaciona o nome do cenário, a taxa de erro, a tensão e o arquivo de especificação de memória.

```bash
python3 scripts/drampower/run_drampower_batch.py \
  --input-root experiments/jpeg/ramulator_results \
  --output-root experiments/jpeg/drampower_results \
  --voltage-map configs/drampower/voltage_map_vendor_b.csv \
  --drampower DRAMPower-4.1/drampower \
  --jobs 4
```

Valide a descoberta das tarefas antes da simulação:

```bash
python3 scripts/drampower/run_drampower_batch.py \
  --input-root experiments/jpeg/ramulator_results \
  --output-root experiments/jpeg/drampower_results \
  --voltage-map configs/drampower/voltage_map_vendor_b.csv \
  --drampower DRAMPower-4.1/drampower \
  --dry-run
```

O DRAMPower não injeta falhas. Ele calcula energia a partir dos command traces gerados para cada cenário.

Os principais arquivos de saída são:

| Arquivo | Conteúdo |
|---|---|
| `drampower_trace_results.csv` | energia e potência por trace |
| `drampower_scenario_summary.csv` | estatísticas agregadas por cenário |
| `drampower_run_issues.csv` | problemas encontrados na execução |
| `drampower_unmapped_scenarios.csv` | cenários sem entrada no mapa |

As configurações do Ramulator, as especificações de memória do DRAMPower (`memspecs`) e o mapa de tensão devem representar a mesma memória.

## Análises de qualidade

As análises de qualidade podem ser executadas independentemente da etapa de energia.

### JPEG: estabilidade

```bash
python3 'scripts/analysis/crashes&survivors/crash_analyzer_jpeg.py' \
  /caminho/para/outputs/jpeg
```

O script classifica saídas vazias, dumps de falha e resultados bem-sucedidos (sobreviventes).

### JPEG: SSIM

```bash
python3 scripts/analysis/structuraljpeg/structuraljpeg/compute_jpeg_ssim.py \
  --perfect-dir /caminho/para/imagens_originais \
  --error-dir /caminho/para/outputs/jpeg
```

O cálculo associa imagens pelo caminho relativo e considera apenas arquivos decodificáveis.

### JPEG: qualidade semântica

Os scripts estão em:

```text
scripts/analysis/semanticjpeg/acceptabilityjpeg/
```

O fluxo inclui treinamento do baseline, treinamento separado para cada taxa de erro, validação e geração de gráficos. Caminhos de datasets, modelos e saídas são definidos nos próprios arquivos.

Comandos principais:

```bash
python3 scripts/analysis/semanticjpeg/acceptabilityjpeg/codetrain/r50t_tif.py
python3 scripts/analysis/semanticjpeg/acceptabilityjpeg/codetrain/r50t.py
```

Para validação:

```bash
cd scripts/analysis/semanticjpeg/acceptabilityjpeg/codeval
python3 val.py
python3 geradorgraficonn.py
```

Os scripts atuais assumem datasets organizados por classes. O treinamento pode usar GPU, mas ela não é obrigatória.

### FFT: estabilidade

```bash
python3 'scripts/analysis/crashes&survivors/fft_output_classifier_crash.py' \
  --base-dir /caminho/para/experimentos_fft \
  --rates 1e-5 1e-4 1e-3 1e-2 1e-1 \
  --out /caminho/para/fft_crash_summary.csv
```

O CSV agrega resultados válidos, dumps de crash, erros, timeouts, saídas inválidas e arquivos ausentes.

### FFT: qualidade estrutural e aceitabilidade semântica

A análise lê diretamente os arquivos `.bin` produzidos pelo benchmark. Não é
necessário gerar `summary.csv` antes dessa etapa.

Por padrão, o baseline exato e as execuções de cada taxa de erro aproximada devem estar no mesmo
diretório base:

```text
experimentos_fft/
├── dataset_iowa_music_exact/
│   └── src/
│       └── dataset_audio_error_rate_exact/
│           └── <classe>/<audio>.bin
├── dataset_iowa_music_1e-5/
│   └── src/
│       └── dataset_audio_error_rate_1e-5/
│           └── <classe>/<audio>.bin
└── dataset_iowa_music_1e-4/
    └── src/
        └── dataset_audio_error_rate_1e-4/
            └── <classe>/<audio>.bin
```

O baseline também pode usar os nomes `dataset_iowa_music_0` e
`dataset_audio_error_rate_0`.

Execute:

```bash
python3 'scripts/analysis/acceptability&qualitystructural_fft/analyze_fft_acceptability_two_plots_one_figure.py' \
  --base-dir /caminho/para/experimentos_fft \
  --rates 1e-5 1e-4 1e-3 1e-2 1e-1 \
  --out-dir /caminho/para/fft_acceptability
```

Se a pasta de outputs exatos estiver em outro local, informe diretamente:

```bash
python3 'scripts/analysis/acceptability&qualitystructural_fft/analyze_fft_acceptability_two_plots_one_figure.py' \
  --base-dir /caminho/para/experimentos_fft \
  --exact-dir /caminho/para/dataset_audio_error_rate_exact \
  --out-dir /caminho/para/fft_acceptability
```

O script interpreta os primeiros 24 bytes de cada arquivo no formato
`<8siiif`: magic `AXDFREQ1`, status, número de amostras, número de frames e
frequência em Hz. Cada saída aproximada é pareada com a saída exata pelo caminho
relativo completo, por exemplo `piano/audio.bin`. Arquivos com o mesmo nome em
classes diferentes não são misturados.

Somente pares nos quais ambas as saídas são classificadas como `OK_RESULT` ou
`RESULT_WITH_CRASH_DUMP` e têm frequências válidas entram nos cálculos de MAPE e
MIDI. Ausências, arquivos vazios, status de erro e saídas inválidas são
registrados, mas ficam fora desses cálculos. Arquivos aproximados sem par exato
também são reportados.

A análise gera:

```text
fft_metrics_by_rate.csv
fft_all_comparisons_vs_exact.csv
fft_unmatched_files.csv
fft_two_plots_one_figure_publication.pdf
fft_two_plots_one_figure_publication.png
```

Os CSVs são resultados da análise, não entradas. As figuras requerem `numpy` e
`matplotlib`.

## Scripts principais

| Script | Função |
|---|---|
| `scripts/benchmarks/jpeg/axcept_batch_jpeg.py` | executa o workload JPEG |
| `scripts/benchmarks/fft/axcept_batch_fft_final_parallel.py` | executa o workload FFT em paralelo |
| `scripts/utils/convert_iowa_aif_to_bin.py` | converte AIFF para áudio raw |
| `scripts/axram/extract_uy_structure.py` | filtra acessos `U Y` |
| `scripts/ramulator/run_ramulator_batch.py` | executa traces no Ramulator |
| `scripts/drampower/run_drampower_batch.py` | calcula e agrega energia |
| `scripts/analysis/structuraljpeg/structuraljpeg/compute_jpeg_ssim.py` | calcula SSIM |
| `scripts/analysis/crashes&survivors/crash_analyzer_jpeg.py` | analisa estabilidade JPEG |
| `scripts/analysis/crashes&survivors/fft_output_classifier_crash.py` | analisa estabilidade FFT |
| `scripts/analysis/acceptability&qualitystructural_fft/analyze_fft_acceptability_two_plots_one_figure.py` | analisa aceitabilidade e qualidade estrutural FFT e gera CSVs e gráficos |
| `scripts/analysis/semanticjpeg/acceptabilityjpeg/codetrain/r50t.py` | treina e avalia uma ResNet-50 com saídas JPEG aproximadas |
| `scripts/analysis/semanticjpeg/acceptabilityjpeg/codetrain/r50t_tif.py` | treina e avalia a ResNet-50 de referência com imagens TIFF |
| `scripts/analysis/semanticjpeg/acceptabilityjpeg/codeval/config.py` | configura modelos, datasets e saídas da validação semântica JPEG |
| `scripts/analysis/semanticjpeg/acceptabilityjpeg/codeval/val.py` | valida os modelos ResNet-50 nos datasets JPEG e consolida os resultados |
| `scripts/analysis/semanticjpeg/acceptabilityjpeg/codeval/geradorgraficonn.py` | gera gráficos da validação semântica JPEG |

Nos scripts que possuem interface de linha de comando, consulte as opções com:

```bash
python3 caminho/do/script.py --help
```

## Cuidados experimentais

- Use o mesmo dataset em todas as taxas de erro.
- Registre versão do código, parâmetros e sementes quando aplicável.
- Não misture saídas de taxas diferentes.
- Preserve os traces brutos do AxRAM.
- Valide uma amostra antes de iniciar lotes grandes.
- Compare todos os resultados com o baseline exato.
- Registre timeouts e crashes; não descarte falhas silenciosamente.
- Mantenha coerência entre Ramulator, memspecs e mapa de tensão.
- Aumente o paralelismo somente após validar uma execução pequena.

## Problemas comuns

### `axpike` ou `pk` não encontrado

Confira a instalação ou o caminho dos binários no ambiente usado para executar o benchmark.

### Dataset ou aplicação não encontrados

Revise `DATASET_DIR`, `APP_BIN` e o diretório no qual o runner foi iniciado. Caminhos relativos são resolvidos a partir desse diretório.

### Saída com texto do Proxy Kernel

O stdout da aplicação contém dados binários. Confira a configuração do Proxy Kernel e o tratamento do padding de 64 bytes usado pelos runners.

### Nenhum log AxRAM foi gerado

Confirme:

- AxRAM ativo com `--adele-activate=0:AXRAM`;
- taxa definida em `mem_read_prob`;
- nomes no formato `AXRAM_log_pid*_hart*.log`;
- diretório de execução com permissão de escrita.

### O extrator não encontra acessos

Verifique se os traces possuem linhas iniciadas por `U Y` e se o diretório informado contém os logs brutos.

### Ramulator não gera command traces

Confira `stdout.txt`, `stderr.txt`, `DDR3.stats` e a configuração utilizada. A gravação de command trace precisa estar ativa.

### DRAMPower não encontra tarefas

Confira:

- arquivos `cmd-trace-*.cmdtrace`;
- nomes dos cenários;
- entradas habilitadas no mapa de tensão;
- existência dos memspecs indicados;
- caminho do executável `DRAMPower-4.1/drampower`.

## Escopo

O AxCept-Bench é uma base para experimentos, não um pacote de resultados prontos. Cada trabalho deve documentar suas decisões experimentais, datasets, configurações de hardware, métricas e critérios de aceitabilidade.
