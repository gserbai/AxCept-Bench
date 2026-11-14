#!/usr/bin/env python3
from pathlib import Path
import argparse
import os
import re

LOG_RE = re.compile(r"^AXRAM_log_pid\d+_hart\d+\.log$")


def get_documents_dir() -> Path:
    home = Path.home()

    # Linux em português geralmente usa Documentos
    documentos = home / "Documentos"
    if documentos.exists():
        return documentos

    # Fallback para sistemas em inglês
    documents = home / "Documents"
    if documents.exists():
        return documents

    # Se nenhuma existir, cria Documentos
    documentos.mkdir(parents=True, exist_ok=True)
    return documentos


def extract_uy_from_file(input_file: Path, output_file: Path) -> int:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    count = 0

    with input_file.open("r", encoding="utf-8", errors="ignore") as fin, \
         output_file.open("w", encoding="utf-8") as fout:

        for line in fin:
            parts = line.strip().split()

            if len(parts) >= 3 and parts[0] == "U" and parts[1] == "Y":
                fout.write(" ".join(parts[2:]) + "\n")
                count += 1

    return count


def extract(root: Path, output_root: Path, verbose: bool = True) -> dict:
    """Filtra as linhas U Y de todos os logs AXRAM sob root, espelhando a estrutura em output_root."""
    root = Path(root).resolve()
    output_root = Path(output_root).resolve()

    if not root.exists():
        raise FileNotFoundError(f"Pasta raiz não encontrada: {root}")

    total_files = 0
    total_uy_lines = 0
    empty_files = []

    for current_dir, dirs, files in os.walk(root):
        current_path = Path(current_dir)

        # Impede de entrar recursivamente em qualquer pasta chamada src
        # e na própria saída (quando ela fica dentro de root)
        dirs[:] = [
            d for d in dirs
            if d != "src" and (current_path / d).resolve() != output_root
        ]

        for filename in files:
            if not LOG_RE.match(filename):
                continue

            input_file = current_path / filename

            relative_dir = current_path.relative_to(root)
            output_file = output_root / relative_dir / filename

            count = extract_uy_from_file(input_file, output_file)

            total_files += 1
            total_uy_lines += count
            if count == 0:
                empty_files.append(str(output_file))

            if verbose:
                print(f"[OK] {input_file} -> {output_file} | linhas U Y: {count}")

    return {
        "files": total_files,
        "uy_lines": total_uy_lines,
        "empty_files": empty_files,
        "output_root": output_root,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Filtra linhas U Y dos logs AXRAM mantendo a estrutura de pastas."
    )

    parser.add_argument(
        "root",
        help="Pasta raiz onde estão os logs"
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Pasta de saída. Se não passar, salva em ~/Documentos/uy_filtrado/"
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()

    if args.output_dir:
        output_root = Path(args.output_dir).resolve()
    else:
        output_root = get_documents_dir() / "uy_filtrado" / root.name

    result = extract(root, output_root)

    print()
    print(f"Arquivos processados: {result['files']}")
    print(f"Linhas U Y extraídas: {result['uy_lines']}")
    print(f"Saída salva em: {result['output_root']}")


if __name__ == "__main__":
    main()
