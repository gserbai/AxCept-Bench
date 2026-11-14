# OpenALPR (Docker)

License plate recognition used to evaluate acceptability in AxCept-Bench
(JPEG and Sobel workloads). OpenALPR is included as a submodule
(`openalpr/`, commit `736ab0e`, July 2020) and runs through Docker.

## Setup (once)

```bash
# Docker (skip if already installed)
sudo apt install -y docker.io
sudo usermod -aG docker $USER      # log out and back in afterwards

# Build the OpenALPR image from the submodule
cd applications/sobel/acceptability_semantic/platerecognition/openalpr
docker build -t openalpr .
```

## Usage

```bash
# One image
docker run --rm -v /path/to/folder:/data:ro openalpr -c br image.png

# JSON output (for scripts)
docker run --rm -v /path/to/folder:/data:ro openalpr -c br -j image.png

# Best candidate only
docker run --rm -v /path/to/folder:/data:ro openalpr -c br -n 1 image.png

# Every image in a folder
docker run --rm -v /path/to/folder:/data:ro openalpr -c br -n 1 /data
```

`-v /path/to/folder:/data:ro` mounts your folder (absolute path, e.g. `$(pwd)`)
as `/data` inside the container, so pass only the **file name** after `openalpr`.

## Notes

- `-c br`: Brazilian single-row plates (`LLL####`). Use `-c br2` for two-row
  (motorcycle) plates. Mercosur plates are not supported.
- Accepted formats: `.png .jpg .jpeg .tif .bmp .gif` (not `.pgm`).
- Plates narrower than 65 px are ignored (`runtime_data/config/br.conf`).