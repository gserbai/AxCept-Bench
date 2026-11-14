/*
 * eval_exact_host.c -- HOST ONLY. Exact, no AxPike.
 *
 * AxCept-Bench -- MLP workload
 * Author: Guilherme Saides Serbai 
 * Year: 2026
 *
 * Loads any model.bin-format file (dims + weights -- same layout produced
 * by train_exact_host.c, train_axram/gen_init_weights_host.c, and the
 * "final trained weights" payload of train_axram/train_kernel_axpike.cpp /
 * AXTRN001) and reports its EXACT accuracy on dataset_digits/test.bin.
 *
 * Point of this tool: after training a network under AxRAM (via
 * train_axram/run_train_axram.py), you have a weights file that was
 * PRODUCED under memory noise. This measures what that network is actually
 * worth -- clean, deterministic, host-side inference -- without needing a
 * second AxPike run per test sample. Compare its number against
 * train_exact_host's own "FINAL exact (host) test accuracy" line (the
 * baseline trained from the SAME initial weights, exact throughout) to see
 * what training-under-noise cost, if anything.
 *
 * Usage (from the project root, so dataset_digits/ is found):
 *   ./eval_exact_host <model.bin>
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "genann.h"

#define N_INPUTS 64
#define N_OUTPUTS 10

static genann *load_model(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }

    int32_t inputs, hidden_layers, hidden, outputs;
    if (fread(&inputs, sizeof(int32_t), 1, f) != 1 ||
        fread(&hidden_layers, sizeof(int32_t), 1, f) != 1 ||
        fread(&hidden, sizeof(int32_t), 1, f) != 1 ||
        fread(&outputs, sizeof(int32_t), 1, f) != 1) {
        fprintf(stderr, "failed to read dims from %s\n", path);
        exit(1);
    }

    genann *ann = genann_init(inputs, hidden_layers, hidden, outputs);
    if (!ann) {
        fprintf(stderr, "genann_init() rejected dims from %s "
                "(inputs=%d hidden_layers=%d hidden=%d outputs=%d)\n",
                path, inputs, hidden_layers, hidden, outputs);
        exit(1);
    }

    if (fread(ann->weight, sizeof(double), (size_t)ann->total_weights, f) != (size_t)ann->total_weights) {
        fprintf(stderr, "short read: %s has fewer than %d weights for its own declared dims\n",
                path, ann->total_weights);
        exit(1);
    }

    fclose(f);
    return ann;
}

static int argmax10(double const *v) {
    int best = 0;
    for (int i = 1; i < N_OUTPUTS; ++i) {
        if (v[i] > v[best]) best = i;
    }
    return best;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.bin>\n", argv[0]);
        fprintf(stderr, "  (expects dataset_digits/test.bin in the current directory)\n");
        return 1;
    }

    genann *ann = load_model(argv[1]);
    if (ann->inputs != N_INPUTS || ann->outputs != N_OUTPUTS) {
        fprintf(stderr, "warning: %s has inputs=%d outputs=%d, expected %d/%d for the digits workload\n",
                argv[1], ann->inputs, ann->outputs, N_INPUTS, N_OUTPUTS);
    }

    FILE *f = fopen("dataset_digits/test.bin", "rb");
    if (!f) { perror("dataset_digits/test.bin"); return 1; }

    int32_t n_samples;
    if (fread(&n_samples, sizeof(int32_t), 1, f) != 1) {
        fprintf(stderr, "failed to read n_samples from dataset_digits/test.bin\n");
        return 1;
    }

    double pixels[N_INPUTS];
    int correct = 0;
    for (int32_t i = 0; i < n_samples; ++i) {
        if (fread(pixels, sizeof(double), N_INPUTS, f) != N_INPUTS) {
            fprintf(stderr, "short read (pixels) at sample %d\n", i);
            return 1;
        }
        int32_t label;
        if (fread(&label, sizeof(int32_t), 1, f) != 1) {
            fprintf(stderr, "short read (label) at sample %d\n", i);
            return 1;
        }
        double const *out = genann_run(ann, pixels);
        if (argmax10(out) == label) ++correct;
    }
    fclose(f);

    fprintf(stderr, "model:    %s\n", argv[1]);
    fprintf(stderr, "accuracy: %.2f%% (%d/%d)\n", 100.0 * correct / n_samples, correct, n_samples);
    printf("%.4f\n", 100.0 * correct / n_samples);

    genann_free(ann);
    return 0;
}
