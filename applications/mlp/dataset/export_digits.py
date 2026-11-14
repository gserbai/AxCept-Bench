#!/usr/bin/env python3
######################################################################
# AxCept-Bench -- MLP digits workload -- dataset export (HOST ONLY)
# Author: Guilherme Saides Serbai 
# Year: 2026
#
# Loads sklearn's `digits` dataset (8x8 grayscale handwritten digits,
# 1797 samples, 10 classes -- the small/classic MNIST-style benchmark) and
# splits it into a train file and a test file, in the raw binary format the
# host trainer (train_exact_and_validate/train_exact_host.c) and the AxPike
# batch drivers (train_exact_and_validate/run_validate_axram.py,
# train_axram/run_train_axram.py) all expect. Nothing here touches AxPike --
# this is pure host-side data prep.
#
# Run this from the project root (it writes to ./dataset_digits/, relative
# to the current directory, not to this script's own location):
#
#   python3 dataset/export_digits.py
#
# Pixel values are scaled to [0, 1] (sklearn's digits are 0..16 integers) --
# unscaled inputs that large make the sigmoid activation saturate and slow
# down training a lot.
#
# train.bin format:
#   int32 n_samples
#   per sample: 64 * float64 pixels, 10 * float64 one-hot label
#
# test.bin format:
#   int32 n_samples
#   per sample: 64 * float64 pixels, int32 true_label (0-9)
######################################################################

import struct
from pathlib import Path

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

OUTPUT_DIR = Path("dataset_digits")
TEST_FRACTION = 0.2
SPLIT_SEED = 42


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    digits = load_digits()
    X = digits.data / 16.0          # 1797 x 64, scaled 0..1
    y = digits.target                # 1797, 0..9

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_FRACTION, random_state=SPLIT_SEED, stratify=y
    )

    train_path = OUTPUT_DIR / "train.bin"
    with open(train_path, "wb") as f:
        f.write(struct.pack("<i", len(X_train)))
        for pixels, label in zip(X_train, y_train):
            f.write(struct.pack("<64d", *pixels))
            one_hot = [0.0] * 10
            one_hot[label] = 1.0
            f.write(struct.pack("<10d", *one_hot))

    test_path = OUTPUT_DIR / "test.bin"
    with open(test_path, "wb") as f:
        f.write(struct.pack("<i", len(X_test)))
        for pixels, label in zip(X_test, y_test):
            f.write(struct.pack("<64d", *pixels))
            f.write(struct.pack("<i", int(label)))

    print(f"train.bin: {len(X_train)} samples -> {train_path}")
    print(f"test.bin:  {len(X_test)} samples -> {test_path}")


if __name__ == "__main__":
    main()
