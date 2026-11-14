/*
 * train_exact_host.c -- HOST ONLY. Exact, no AxPike, no AxRAM.
 *
 * AxCept-Bench -- MLP workload
 * Author: Guilherme Saides Serbai 
 * Year: 2026
 *
 * Trains a genann network on the sklearn `digits` dataset (exported by
 * dataset/export_digits.py into train.bin/test.bin) and dumps the trained
 * model in the raw binary format validate_kernel_axpike.cpp expects on
 * stdin, MINUS the 64-byte pk padding and MINUS any specific input vector
 * -- this is the MODEL file, reused across every AxPike run in the batch,
 * not a one-shot payload.
 *
 * Also runs an exact (host, no AxPike) accuracy check on the held-out test
 * set, printed to stderr, so you have a baseline accuracy number to compare
 * every AxRAM error rate against.
 *
 * Usage (from the project root, so dataset_digits/ is found):
 *   ./train_exact_host <hidden_units> <epochs> [hidden_layers] [init_weights.bin] [n_train_samples] > model.bin
 *
 * Example:
 *   ./train_exact_host 32 300 > model.bin
 *
 * Optional 4th arg: path to an init_weights.bin-format file (dims +
 * weights, from train_axram/gen_init_weights_host.c). When given, training
 * starts from THOSE weights instead of a fresh genann_init() random init --
 * use this to produce the exact-host baseline that starts from the SAME
 * initial weights as an AxRAM-trained run (train_axram/train_kernel_axpike.cpp
 * / train_axram/run_train_axram.py), so the two are comparable without
 * random-init variance confounding the result.
 *
 * Optional 5th arg: n_train_samples. When given, only the FIRST
 * n_train_samples of train.bin are used (same "first N in file order"
 * convention run_train_axram.py's load_train_subset() uses) instead of the
 * full 1437. Use this together with the 4th arg to reproduce EXACTLY the
 * same training run (same samples, same epochs, same starting weights)
 * that an AxRAM run used, so the two accuracy numbers are actually
 * comparable -- without it, an AxRAM run configured with a small
 * N_TRAIN_SAMPLES subset (as run_train_axram.py defaults to, for
 * wall-clock feasibility) would get compared against a baseline trained on
 * the full dataset, confounding "less data" with "the effect of
 * approximate memory" -- two very different things.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include "genann.h"

#define N_INPUTS 64
#define N_OUTPUTS 10

typedef struct {
    int n_samples;
    double *pixels;      /* n_samples * 64 */
    double *one_hot;      /* n_samples * 10, only used for train.bin */
    int *labels;          /* n_samples, only used for test.bin */
} dataset;

static dataset load_train(const char *path) {
    dataset d = {0, NULL, NULL, NULL};
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }

    if (fread(&d.n_samples, sizeof(int32_t), 1, f) != 1) {
        fprintf(stderr, "failed to read n_samples from %s\n", path);
        exit(1);
    }

    d.pixels = malloc(sizeof(double) * N_INPUTS * d.n_samples);
    d.one_hot = malloc(sizeof(double) * N_OUTPUTS * d.n_samples);

    for (int i = 0; i < d.n_samples; ++i) {
        if (fread(d.pixels + i * N_INPUTS, sizeof(double), N_INPUTS, f) != N_INPUTS) {
            fprintf(stderr, "short read (pixels) at sample %d in %s\n", i, path);
            exit(1);
        }
        if (fread(d.one_hot + i * N_OUTPUTS, sizeof(double), N_OUTPUTS, f) != N_OUTPUTS) {
            fprintf(stderr, "short read (one_hot) at sample %d in %s\n", i, path);
            exit(1);
        }
    }

    fclose(f);
    return d;
}

static dataset load_test(const char *path) {
    dataset d = {0, NULL, NULL, NULL};
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }

    if (fread(&d.n_samples, sizeof(int32_t), 1, f) != 1) {
        fprintf(stderr, "failed to read n_samples from %s\n", path);
        exit(1);
    }

    d.pixels = malloc(sizeof(double) * N_INPUTS * d.n_samples);
    d.labels = malloc(sizeof(int) * d.n_samples);

    for (int i = 0; i < d.n_samples; ++i) {
        if (fread(d.pixels + i * N_INPUTS, sizeof(double), N_INPUTS, f) != N_INPUTS) {
            fprintf(stderr, "short read (pixels) at sample %d in %s\n", i, path);
            exit(1);
        }
        int32_t label;
        if (fread(&label, sizeof(int32_t), 1, f) != 1) {
            fprintf(stderr, "short read (label) at sample %d in %s\n", i, path);
            exit(1);
        }
        d.labels[i] = label;
    }

    fclose(f);
    return d;
}

static int argmax10(double const *v) {
    int best = 0;
    for (int i = 1; i < N_OUTPUTS; ++i) {
        if (v[i] > v[best]) best = i;
    }
    return best;
}

/* Loads dims + weights from an init_weights.bin-format file and copies
 * them into ann->weight. Exits on any mismatch/short read rather than
 * silently training from a partially-overwritten buffer. */
static void load_init_weights(genann *ann, const char *path) {
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
    if (inputs != ann->inputs || hidden_layers != ann->hidden_layers ||
        hidden != ann->hidden || outputs != ann->outputs) {
        fprintf(stderr, "%s dims (%d,%d,%d,%d) don't match requested "
                "architecture (%d,%d,%d,%d)\n",
                path, inputs, hidden_layers, hidden, outputs,
                ann->inputs, ann->hidden_layers, ann->hidden, ann->outputs);
        exit(1);
    }
    if (fread(ann->weight, sizeof(double), (size_t)ann->total_weights, f) != (size_t)ann->total_weights) {
        fprintf(stderr, "short read: %s has fewer than %d weights\n", path, ann->total_weights);
        exit(1);
    }
    fclose(f);
    fprintf(stderr, "loaded initial weights from %s\n", path);
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <hidden_units> <epochs> [hidden_layers=1] > model.bin\n", argv[0]);
        fprintf(stderr, "  (expects dataset_digits/train.bin and dataset_digits/test.bin --\n");
        fprintf(stderr, "   run dataset/export_digits.py first, or `make dataset`)\n");
        return 1;
    }
    int hidden = atoi(argv[1]);
    int epochs = atoi(argv[2]);
    /* Optional 3rd arg -- defaults to 1 (single hidden layer) so old
     * invocations without it still behave exactly as before. genann uses
     * the SAME width ('hidden') for every hidden layer -- no funnel shape
     * like 128->64, all layers are 'hidden' neurons wide. */
    int hidden_layers = (argc >= 4) ? atoi(argv[3]) : 1;
    /* Optional 4th arg -- path to init_weights.bin. See the usage comment
     * at the top of this file. */
    const char *init_weights_path = (argc >= 5) ? argv[4] : NULL;
    /* Optional 5th arg -- limit training to the first N samples. See the
     * usage comment at the top of this file. */
    int n_train_samples_arg = (argc >= 6) ? atoi(argv[5]) : 0;

    srand((unsigned)time(0));

    dataset train = load_train("dataset_digits/train.bin");
    dataset test = load_test("dataset_digits/test.bin");
    fprintf(stderr, "loaded train.bin: %d samples, test.bin: %d samples\n",
            train.n_samples, test.n_samples);

    if (n_train_samples_arg > 0) {
        if (n_train_samples_arg > train.n_samples) {
            fprintf(stderr, "requested n_train_samples=%d but train.bin only has %d samples\n",
                    n_train_samples_arg, train.n_samples);
            return 1;
        }
        fprintf(stderr, "limiting training to the first %d samples (of %d) -- for a fair "
                "comparison against an AxRAM run configured with the same N_TRAIN_SAMPLES\n",
                n_train_samples_arg, train.n_samples);
        train.n_samples = n_train_samples_arg;
    }

    genann *ann = genann_init(N_INPUTS, hidden_layers, hidden, N_OUTPUTS);
    fprintf(stderr, "architecture: %d", N_INPUTS);
    for (int i = 0; i < hidden_layers; ++i) fprintf(stderr, " -> %d", hidden);
    fprintf(stderr, " -> %d  (%d weights, %d hidden layer%s)\n",
            N_OUTPUTS, ann->total_weights, hidden_layers, hidden_layers == 1 ? "" : "s");

    if (init_weights_path) {
        load_init_weights(ann, init_weights_path);
    }

    double learning_rate = 1.0;
    for (int e = 0; e < epochs; ++e) {
        for (int i = 0; i < train.n_samples; ++i) {
            genann_train(ann, train.pixels + i * N_INPUTS,
                         train.one_hot + i * N_OUTPUTS, learning_rate);
        }
        if ((e + 1) % 50 == 0 || e == epochs - 1) {
            int correct = 0;
            for (int i = 0; i < test.n_samples; ++i) {
                double const *out = genann_run(ann, test.pixels + i * N_INPUTS);
                if (argmax10(out) == test.labels[i]) ++correct;
            }
            fprintf(stderr, "epoch %4d/%d  test accuracy (exact, host): %.2f%% (%d/%d)\n",
                    e + 1, epochs, 100.0 * correct / test.n_samples, correct, test.n_samples);
        }
    }

    /* Final accuracy check, exact, host -- this is the number every AxRAM
     * error rate gets compared against. */
    int correct = 0;
    for (int i = 0; i < test.n_samples; ++i) {
        double const *out = genann_run(ann, test.pixels + i * N_INPUTS);
        if (argmax10(out) == test.labels[i]) ++correct;
    }
    fprintf(stderr, "\nFINAL exact (host) test accuracy: %.2f%% (%d/%d)\n",
            100.0 * correct / test.n_samples, correct, test.n_samples);

    /* Dump the model: dims + weights, no input vector, no padding --
     * run_validate_axram.py adds those per-sample. */
    int32_t inputs32 = ann->inputs, hidden_layers32 = ann->hidden_layers,
            hidden32 = ann->hidden, outputs32 = ann->outputs;
    fwrite(&inputs32, sizeof(inputs32), 1, stdout);
    fwrite(&hidden_layers32, sizeof(hidden_layers32), 1, stdout);
    fwrite(&hidden32, sizeof(hidden32), 1, stdout);
    fwrite(&outputs32, sizeof(outputs32), 1, stdout);
    fwrite(ann->weight, sizeof(double), (size_t)ann->total_weights, stdout);

    genann_free(ann);
    free(train.pixels); free(train.one_hot);
    free(test.pixels); free(test.labels);
    return 0;
}
