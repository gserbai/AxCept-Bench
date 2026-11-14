/*
 * validate_kernel_axpike.cpp -- AxCept-Bench inference-only kernel wrapping
 * GENANN (codeplea/genann, Zlib license, unmodified lib/genann.c +
 * lib/genann.h).
 *
 * AxCept-Bench -- MLP workload
 * Author: Guilherme Saides Serbai 
 * Year: 2026
 *
 * This is the ONLY part of the "train exact, validate under AxRAM" path
 * that runs under AxPike/AxRAM. Training happens on the host with genann's
 * own genann_train(), exact, no AxPike involved -- see train_exact_host.c
 * for the trainer that produces the model.bin this program consumes.
 * (There is a second, separate path -- train_axram/ -- where the training
 * itself happens under AxPike instead; this file is not part of that one.)
 *
 * Deliberately does NOT use genann_read()/genann_write() (they use
 * fscanf/fprintf on a FILE*, which depends on whether this pk build proxies
 * file syscalls to the host -- untested). Instead it reads everything as
 * raw bytes from stdin, exactly like kernel/kmeans and the JPEG/FFT
 * kernels already do, so nothing here depends on that answer.
 *
 * Stdin format (as seen by THIS program -- the 64-byte pk padding that
 * run_validate_axram.py prepends is consumed entirely by pk at the loader
 * level, before main() ever calls fread; this program does not, and must
 * not, discard any bytes of its own on top of that -- see the note in
 * main() below):
 *
 *   int32 LE inputs
 *   int32 LE hidden_layers
 *   int32 LE hidden
 *   int32 LE outputs
 *   total_weights * float64 LE   (trained weights, raw doubles --
 *                                 total_weights = genann_init()'s own
 *                                 formula from the four ints above)
 *   inputs * float64 LE          (the input feature vector to classify)
 *
 * Stdout format (always written, even on error -- mirrors AXKMEAN1 /
 * AXDFREQ1's "always emit a record, classify it on the host side" pattern):
 *
 *   8-byte magic "AXGENN01"
 *   int32 LE status   (0 = AX_OK, nonzero = see the status codes below --
 *                       no payload follows when status != 0)
 *   int32 LE inputs
 *   int32 LE outputs
 *   outputs * float64 LE   (genann_run()'s output layer, only when status==0)
 */

#include <cstdio>
#include <cstdlib>
#include <cstdint>

extern "C" {
#include "genann.h"
}

// Status codes. 0 = AX_OK. Everything else means no output payload follows
// the header -- mirrors kmeans_axpike.cpp's AX_EMPTY_INPUT / AX_ALLOC_ERROR
// / AX_INVALID_DIMS / AX_SHORT_INPUT convention so the host-side classifier
// script can reuse the same "recoverable vs not" logic.
enum {
    AX_OK = 0,
    AX_SHORT_INPUT = 1,   // stdin ran out before the header/weights/input
                           // vector were fully read
    AX_INVALID_DIMS = 2,  // genann_init() rejected the dimensions
    AX_ALLOC_ERROR = 3,   // malloc failed (dims too large, or OOM)
};

static bool read_exact(void *buf, size_t n) {
    return std::fread(buf, 1, n, stdin) == n;
}

static void write_header_only(int32_t status) {
    std::fwrite("AXGENN01", 1, 8, stdout);
    std::fwrite(&status, sizeof(status), 1, stdout);
    std::fflush(stdout);
}

int main() {
    // NOTE: no in-program padding discard here. The pk stdin padding is
    // consumed entirely at the pk/loader level, before this program's own
    // fread calls ever run -- the batch script compensates for that by
    // prepending PK_STDIN_PADDING_BYTES dummy bytes to what it sends.
    // By the time main() starts reading, stdin already IS the real
    // payload (dims + weights + input vector), same as every other
    // AxCept-Bench kernel (kmeans_axpike.cpp, dominant_freq.cpp,
    // main.cpp/JPEG) -- none of them re-discard padding in-program either.
    // An earlier version of this file DID also discard
    // PK_STDIN_PADDING_BYTES here, which double-consumed the padding (once
    // via pk, once via this code) and ate straight into the real dims/
    // weights, corrupting every run regardless of the configured padding
    // value. Do not reintroduce that discard.

    int32_t inputs = 0, hidden_layers = 0, hidden = 0, outputs = 0;
    if (!read_exact(&inputs, 4) || !read_exact(&hidden_layers, 4) ||
        !read_exact(&hidden, 4) || !read_exact(&outputs, 4)) {
        write_header_only(AX_SHORT_INPUT);
        return 1;
    }

    genann *ann = genann_init(inputs, hidden_layers, hidden, outputs);
    if (!ann) {
        // genann_init() itself distinguishes "bad dimensions" from "malloc
        // failed" only via a single NULL return -- collapse both here too,
        // rather than guess which one happened.
        write_header_only(AX_INVALID_DIMS);
        return 1;
    }

    // THIS read is the interesting one for the experiment: it overwrites
    // the (currently randomly-initialized) weight buffer with the trained
    // weights. If AxRAM lives under this buffer, a corrupted read here
    // corrupts a value reused across every multiply-accumulate that touches
    // it -- unlike a corrupted read of the input vector below, which only
    // affects that one inference.
    if (!read_exact(ann->weight, sizeof(double) * (size_t)ann->total_weights)) {
        write_header_only(AX_SHORT_INPUT);
        genann_free(ann);
        return 1;
    }

    double *input_vec = (double *)std::malloc(sizeof(double) * (size_t)inputs);
    if (!input_vec) {
        write_header_only(AX_ALLOC_ERROR);
        genann_free(ann);
        return 1;
    }
    if (!read_exact(input_vec, sizeof(double) * (size_t)inputs)) {
        write_header_only(AX_SHORT_INPUT);
        std::free(input_vec);
        genann_free(ann);
        return 1;
    }

    // The only computation that happens under AxPike: one forward pass.
    double const *out = genann_run(ann, input_vec);

    std::fwrite("AXGENN01", 1, 8, stdout);
    int32_t status = AX_OK;
    std::fwrite(&status, sizeof(status), 1, stdout);
    std::fwrite(&inputs, sizeof(inputs), 1, stdout);
    std::fwrite(&outputs, sizeof(outputs), 1, stdout);
    std::fwrite(out, sizeof(double), (size_t)outputs, stdout);
    std::fflush(stdout);

    std::free(input_vec);
    genann_free(ann);
    return 0;
}
