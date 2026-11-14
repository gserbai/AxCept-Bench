import subprocess
from pathlib import Path

SRC_DIR = Path("/home/guilherme/Music/dataset_iowa_music")
DST_DIR = Path("/home/guilherme/Music/dataset_iowa_music_bin")

SAMPLE_RATE = 44100

def convert_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(src),
        "-ac", "1",              # mono
        "-ar", str(SAMPLE_RATE), # 44100 Hz
        "-f", "f32le",           # float32 little-endian raw
        str(dst),
    ]

    subprocess.run(cmd, check=True)

def main():
    files = []
    files.extend(SRC_DIR.rglob("*.aif"))
    files.extend(SRC_DIR.rglob("*.aiff"))
    files.extend(SRC_DIR.rglob("*.AIF"))
    files.extend(SRC_DIR.rglob("*.AIFF"))

    files = sorted(set(files))

    if not files:
        print(f"Nenhum .aif/.aiff encontrado em {SRC_DIR}")
        return

    print(f"Encontrados {len(files)} arquivos AIFF.")

    for i, src in enumerate(files, 1):
        rel = src.relative_to(SRC_DIR)
        dst = DST_DIR / rel.with_suffix(".bin")

        print(f"[{i}/{len(files)}] {rel} -> {dst.relative_to(DST_DIR)}")

        try:
            convert_file(src, dst)
        except subprocess.CalledProcessError as e:
            print(f"  [ERRO] Falha ao converter {src}: {e}")

    print("\nConversão concluída.")
    print(f"Dataset binário salvo em: {DST_DIR}")

if __name__ == "__main__":
    main()
