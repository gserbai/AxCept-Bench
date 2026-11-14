/*
 * train_kernel_axpike.cpp -- AxCept-Bench TRAINING kernel wrapping GENANN
 * (codeplea/genann, Zlib license, unmodified lib/genann.c + lib/genann.h).
 *
 * AxCept-Bench / AxAccept-Bench -- MLP workload
 * Author: Guilherme Saides Serbai 
 * Year: 2026
 *
 * Unlike train_exact_and_validate/validate_kernel_axpike.cpp (one forward
 * pass per AxPike invocation, called once per test sample by its batch
 * script), THIS kernel runs the ENTIRE training loop -- every epoch, every
 * sample, every genann_train() call, every weight update -- inside a
 * SINGLE AxPike/AxRAM invocation. That is a deliberate, load-bearing design
 * choice, not a simplification:
 *
 *   - Training is sequential and stateful: weights from sample i feed into
 *     the gradient computed for sample i+1. Spawning a fresh pk process per
 *     sample (the way inference validation does it) would mean re-reading
 *     the weights from a host-side file between every single step, which
 *     defeats the point (the whole POINT is that the weights live in
 *     approximate memory continuously across the whole training run, the
 *     same way they would on real AxRAM hardware during an actual training
 *     job) and would also be catastrophically slow: thousands of pk/axpike
 *     process-boot overheads instead of one.
 *   - So: one process, one call into genann_init(), N epochs of
 *     genann_train() over every sample, all touching the SAME weight
 *     buffer inside the guest's (approximate) memory the whole time, then
 *     one final dump of the trained weights.
 *
 * PRACTICAL WARNING (read before running with real epoch/sample counts):
 * AxPike is a detailed simulator, not a native execution -- a single
 * genann_run() forward pass under it already costs meaningfully more wall
 * time than the same call would on host silicon. A full training run does
 * epochs * n_samples forward-AND-backward passes in a row, inside that same
 * slowdown, with NO parallelism (this is one guest process). Expect this to
 * be dramatically slower than train_exact_host's host training -- start
 * with a SMALL epoch count and a SMALL sample subset (run_train_axram.py's
 * N_TRAIN_SAMPLES / EPOCHS knobs) to get a wall-clock feel before
 * committing to the full 1437-sample / 300-epoch host-equivalent config.
 *
 * Stdin format (as seen by THIS program -- the 64-byte pk padding the batch
 * script prepends is consumed entirely by pk at the loader level, before
 * main() ever calls fread; same convention as validate_kernel_axpike.cpp,
 * do not re-discard it here):
 *
 *   int32 LE n_train_samples
 *   int32 LE inputs
 *   int32 LE hidden_layers
 *   int32 LE hidden
 *   int32 LE outputs
 *   int32 LE epochs
 *   float64 LE learning_rate
 *   total_weights * float64 LE     (INITIAL weights -- total_weights is
 *                                   genann_init()'s own formula from the
 *                                   four dims above. Sent explicitly,
 *                                   rather than randomized in-guest, so an
 *                                   exact-host baseline run can start from
 *                                   the IDENTICAL initial weights -- see
 *                                   gen_init_weights_host.c. Without this,
 *                                   any accuracy difference between the
 *                                   AxRAM-trained and host-trained networks
 *                                   would be confounded with ordinary
 *                                   random-init variance, not isolating the
 *                                   effect of training under approximate
 *                                   memory.)
 *   n_train_samples * (inputs * float64 LE pixels +
 *                       outputs * float64 LE desired_output)
 *                                   (training set, same per-sample layout
 *                                   as dataset_digits/train.bin's body --
 *                                   interleaved pixels-then-one_hot per
 *                                   sample, not two separate arrays)
 *
 * Stdout format (always written, even on error -- same "always emit a
 * record, classify on the host side" convention as validate_kernel_axpike.cpp):
 *
 *   8-byte magic "AXTRN001"
 *   int32 LE status   (0 = AX_OK, nonzero = see status codes below -- no
 *                       payload follows when status != 0)
 *   int32 LE inputs
 *   int32 LE hidden_layers
 *   int32 LE hidden
 *   int32 LE outputs
 *   total_weights * float64 LE   (final trained weights, same layout as
 *                                 model.bin -- directly usable as MODEL_BIN
 *                                 for train_exact_and_validate/run_validate_axram.py,
 *                                 or by eval_exact_host, only when status==0)
 */

#include <cstdio>
#include <cstdlib>
#include <cstdint>

extern "C" {
#include "genann.h"
}

enum {
    AX_OK = 0,
    AX_SHORT_INPUT = 1,   // stdin ran out before header/weights/dataset
                           // were fully read
    AX_INVALID_DIMS = 2,  // genann_init() rejected the dimensions
    AX_ALLOC_ERROR = 3,   // malloc failed (dims/sample count too large,
                           // or OOM)
};

static bool read_exact(void *buf, size_t n) {
    return std::fread(buf, 1, n, stdin) == n;
}

static void write_header_only(int32_t status) {
    std::fwrite("AXTRN001", 1, 8, stdout);
    std::fwrite(&status, sizeof(status), 1, stdout);
    std::fflush(stdout);
}

int main() {
    // NOTE: no in-program padding discard here -- same reasoning as
    // validate_kernel_axpike.cpp. Do not reintroduce one.

    int32_t n_train_samples = 0;
    int32_t inputs = 0, hidden_layers = 0, hidden = 0, outputs = 0;
    int32_t epochs = 0;
    double learning_rate = 0.0;

    if (!read_exact(&n_train_samples, 4) ||
        !read_exact(&inputs, 4) || !read_exact(&hidden_layers, 4) ||
        !read_exact(&hidden, 4) || !read_exact(&outputs, 4) ||
        !read_exact(&epochs, 4) || !read_exact(&learning_rate, 8)) {
        write_header_only(AX_SHORT_INPUT);
        return 1;
    }

    if (n_train_samples <= 0 || epochs <= 0) {
        write_header_only(AX_INVALID_DIMS);
        return 1;
    }

    genann *ann = genann_init(inputs, hidden_layers, hidden, outputs);
    if (!ann) {
        write_header_only(AX_INVALID_DIMS);
        return 1;
    }

    // Overwrite the freshly-randomized weights with the caller-supplied
    // initial weights -- see the stdin format comment above for why.
    if (!read_exact(ann->weight, sizeof(double) * (size_t)ann->total_weights)) {
        write_header_only(AX_SHORT_INPUT);
        genann_free(ann);
        return 1;
    }

    // Whole training set lives in guest (approximate) memory for the
    // duration of the run -- this is intentional: real AxRAM-backed
    // training would keep its dataset resident in the same approximate
    // memory too, not stream it in from some exact side-channel.
    double *pixels = (double *)std::malloc(sizeof(double) * (size_t)n_train_samples * (size_t)inputs);
    double *desired = (double *)std::malloc(sizeof(double) * (size_t)n_train_samples * (size_t)outputs);
    if (!pixels || !desired) {
        write_header_only(AX_ALLOC_ERROR);
        std::free(pixels);
        std::free(desired);
        genann_free(ann);
        return 1;
    }

    for (int32_t i = 0; i < n_train_samples; ++i) {
        if (!read_exact(pixels + (size_t)i * inputs, sizeof(double) * (size_t)inputs) ||
            !read_exact(desired + (size_t)i * outputs, sizeof(double) * (size_t)outputs)) {
            write_header_only(AX_SHORT_INPUT);
            std::free(pixels);
            std::free(desired);
            genann_free(ann);
            return 1;
        }
    }

    // The actual computation under test: every epoch, every sample, every
    // weight update, all touching ann->weight inside approximate memory.
    for (int32_t e = 0; e < epochs; ++e) {
        for (int32_t i = 0; i < n_train_samples; ++i) {
            genann_train(ann, pixels + (size_t)i * inputs,
                         desired + (size_t)i * outputs, learning_rate);
        }
        // Cheap, infrequent (once per epoch) progress marker -- useful to
        // see the run is alive during a long simulated training job.
        std::fprintf(stderr, "[train_kernel_axpike] epoch %d/%d done\n", e + 1, epochs);
        std::fflush(stderr);
    }

    std::fwrite("AXTRN001", 1, 8, stdout);
    int32_t status = AX_OK;
    std::fwrite(&status, sizeof(status), 1, stdout);
    std::fwrite(&inputs, sizeof(inputs), 1, stdout);
    std::fwrite(&hidden_layers, sizeof(hidden_layers), 1, stdout);
    std::fwrite(&hidden, sizeof(hidden), 1, stdout);
    std::fwrite(&outputs, sizeof(outputs), 1, stdout);
    std::fwrite(ann->weight, sizeof(double), (size_t)ann->total_weights, stdout);
    std::fflush(stdout);

    std::free(pixels);
    std::free(desired);
    genann_free(ann);
    return 0;
}
