/*
 * gen_init_weights_host.c -- HOST ONLY. Exact, no AxPike.
 *
 * AxCept-Bench -- MLP workload
 * Author: Guilherme Saides Serbai 
 * Year: 2026
 *
 * Produces a reproducible random initial-weights blob (dims + weights,
 * same layout train_kernel_axpike.cpp / validate_kernel_axpike.cpp /
 * model.bin use) from a fixed seed.
 *
 * Why this needs to exist: to isolate "the effect of training under
 * approximate memory" from ordinary random-initialization variance, the
 * AxRAM-trained run (train_axram/run_train_axram.py) and the exact-host
 * baseline run (train_exact_and_validate/train_exact_host.c) must start
 * from the IDENTICAL initial weights. genann_init() calls
 * genann_randomize() internally with whatever rand() state it finds --
 * fine for a single run, useless if you need the SAME starting point
 * twice. This tool freezes that starting point to a file, once, from an
 * explicit seed.
 *
 * Usage (from the project root -- writes to stdout, redirect to a file):
 *   ./gen_init_weights_host <hidden_units> <hidden_layers> <seed> > init_weights.bin
 *
 * Dims are fixed to the digits workload (64 inputs, 10 outputs) to match
 * train_exact_and_validate/train_exact_host.c,
 * train_axram/train_kernel_axpike.cpp and dataset/export_digits.py.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "genann.h"

#define N_INPUTS 64
#define N_OUTPUTS 10

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <hidden_units> <hidden_layers> <seed> > init_weights.bin\n", argv[0]);
        return 1;
    }
    int hidden = atoi(argv[1]);
    int hidden_layers = atoi(argv[2]);
    unsigned seed = (unsigned)strtoul(argv[3], NULL, 10);

    srand(seed);

    genann *ann = genann_init(N_INPUTS, hidden_layers, hidden, N_OUTPUTS);
    if (!ann) {
        fprintf(stderr, "genann_init() failed for hidden=%d hidden_layers=%d\n", hidden, hidden_layers);
        return 1;
    }

    fprintf(stderr, "seed=%u  architecture: %d", seed, N_INPUTS);
    for (int i = 0; i < hidden_layers; ++i) fprintf(stderr, " -> %d", hidden);
    fprintf(stderr, " -> %d  (%d weights)\n", N_OUTPUTS, ann->total_weights);

    int32_t inputs32 = ann->inputs, hidden_layers32 = ann->hidden_layers,
            hidden32 = ann->hidden, outputs32 = ann->outputs;
    fwrite(&inputs32, sizeof(inputs32), 1, stdout);
    fwrite(&hidden_layers32, sizeof(hidden_layers32), 1, stdout);
    fwrite(&hidden32, sizeof(hidden32), 1, stdout);
    fwrite(&outputs32, sizeof(outputs32), 1, stdout);
    fwrite(ann->weight, sizeof(double), (size_t)ann->total_weights, stdout);

    genann_free(ann);
    return 0;
}
