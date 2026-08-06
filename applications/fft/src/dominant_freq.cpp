// ///////////////////////////////////////////////////////////////////
// dominant_freq_axpike_final_nosignal.cpp
// Dominant Frequency (Pitch Recognition) via STFT + HPS
// AxPike/AxRAM version with stdout-only binary result.
//
// Input  (stdin): raw float32 little-endian audio samples, mono, 44100 Hz.
// Output (stdout) when the application reaches the end or a controlled error:
//   [8 bytes  magic: AXDFREQ1]
//   [int32 LE status]
//   [int32 LE num_samples]
//   [int32 LE num_frames]
//   [float32 LE freq_hz]
//
// Output size for normal application records: 24 bytes.
//
// Status codes:
//   0 = AX_OK              valid frequency written in freq_hz
//   1 = AX_EMPTY_INPUT     no audio samples were received
//   2 = AX_ALLOC_ERROR     memory allocation failed
//   3 = AX_FFT_ALLOC_ERROR FFT configuration allocation failed
//   4 = AX_NUMERIC_ERROR   final frequency is NaN/Inf/out of physical range
//
// Low-level crash behavior:
//   No signal handler is installed. If AxPike detects a user segfault/trap,
//   its own register dump can be written to the same redirected stdout file,
//   exactly like in the JPEG workload. The host-side parser should classify
//   files starting with "z  " or containing "User ... segfault" as crashes.
// ///////////////////////////////////////////////////////////////////

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include "kiss_fft.h"

#define SAMPLE_RATE 44100
#define N_FFT       4096
#define HOP_LENGTH  512
#define PI 3.14159265358979323846

#define AX_OK              0
#define AX_EMPTY_INPUT     1
#define AX_ALLOC_ERROR     2
#define AX_FFT_ALLOC_ERROR 3
#define AX_NUMERIC_ERROR   4

static const unsigned char AX_MAGIC[8] = {'A','X','D','F','R','E','Q','1'};

static void write_byte(unsigned char b) {
    putchar((int)b);
}

static void write_u32_le(uint32_t v) {
    write_byte((unsigned char)( v        & 0xFF));
    write_byte((unsigned char)((v >>  8) & 0xFF));
    write_byte((unsigned char)((v >> 16) & 0xFF));
    write_byte((unsigned char)((v >> 24) & 0xFF));
}

static void write_i32_le(int32_t v) {
    write_u32_le((uint32_t)v);
}

static void write_f32_le(float f) {
    union {
        float f;
        uint32_t u;
    } conv;
    conv.f = f;
    write_u32_le(conv.u);
}

static void write_ax_record(int32_t status,
                            int32_t num_samples,
                            int32_t num_frames,
                            float freq_hz) {
    for (int i = 0; i < 8; i++) write_byte(AX_MAGIC[i]);
    write_i32_le(status);
    write_i32_le(num_samples);
    write_i32_le(num_frames);
    write_f32_le(freq_hz);
    fflush(stdout);
}

static int append_sample(float** data, int* size, int* capacity, float sample) {
    if (*size >= *capacity) {
        int new_capacity = (*capacity == 0) ? 4096 : (*capacity * 2);
        float* new_data = (float*)realloc(*data, (size_t)new_capacity * sizeof(float));
        if (!new_data) return 0;
        *data = new_data;
        *capacity = new_capacity;
    }

    (*data)[*size] = sample;
    (*size)++;
    return 1;
}

static float* read_audio_from_stdin(int* out_num_samples) {
    float* audio_data = NULL;
    int num_samples = 0;
    int capacity = 0;
    float sample;

    while (fread(&sample, sizeof(float), 1, stdin) == 1) {
        if (!append_sample(&audio_data, &num_samples, &capacity, sample)) {
            free(audio_data);
            *out_num_samples = -1;
            return NULL;
        }
    }

    *out_num_samples = num_samples;
    return audio_data;
}

static float compute_dominant_frequency(const float* audio_data,
                                        int num_samples,
                                        int* out_num_frames,
                                        int* out_status) {
    int num_frames = 1 + (num_samples - N_FFT) / HOP_LENGTH;
    if (num_frames < 1) num_frames = 1;
    *out_num_frames = num_frames;
    *out_status = AX_OK;

    kiss_fft_cfg cfg = kiss_fft_alloc(N_FFT, 0, NULL, NULL);
    if (!cfg) {
        *out_status = AX_FFT_ALLOC_ERROR;
        return -1.0f;
    }

    kiss_fft_cpx* cx_in  = (kiss_fft_cpx*)malloc((size_t)N_FFT * sizeof(kiss_fft_cpx));
    kiss_fft_cpx* cx_out = (kiss_fft_cpx*)malloc((size_t)N_FFT * sizeof(kiss_fft_cpx));

    int num_bins = N_FFT / 2 + 1;
    double* total_power = (double*)calloc((size_t)num_bins, sizeof(double));
    double* hps         = (double*)calloc((size_t)num_bins, sizeof(double));

    if (!cx_in || !cx_out || !total_power || !hps) {
        free(cfg);
        free(cx_in);
        free(cx_out);
        free(total_power);
        free(hps);
        *out_status = AX_ALLOC_ERROR;
        return -1.0f;
    }

    for (int f = 0; f < num_frames; f++) {
        int start_idx = f * HOP_LENGTH;

        for (int i = 0; i < N_FFT; i++) {
            float sample = (start_idx + i < num_samples) ? audio_data[start_idx + i] : 0.0f;
            float hann_window = 0.5f * (1.0f - cosf((float)((2.0 * PI * i) / (N_FFT - 1))));
            cx_in[i].r = sample * hann_window;
            cx_in[i].i = 0.0f;
        }

        kiss_fft(cfg, cx_in, cx_out);

        for (int k = 0; k < num_bins; k++) {
            double real = (double)cx_out[k].r;
            double imag = (double)cx_out[k].i;
            total_power[k] += real * real + imag * imag;
        }
    }

    // Harmonic Product Spectrum: reinforces the fundamental.
    for (int k = 1; k < num_bins; k++) {
        hps[k] = total_power[k];
        if (k * 2 < num_bins) hps[k] *= total_power[k * 2];
        if (k * 3 < num_bins) hps[k] *= total_power[k * 3];
    }

    int max_bin = 1;
    double max_power = 0.0;
    for (int k = 1; k < num_bins; k++) {
        if (hps[k] > max_power) {
            max_power = hps[k];
            max_bin = k;
        }
    }

    double refined_bin = (double)max_bin;
    if (max_bin > 1 && max_bin < num_bins - 1) {
        double alpha = hps[max_bin - 1];
        double beta  = hps[max_bin];
        double gamma = hps[max_bin + 1];
        double denom = alpha - 2.0 * beta + gamma;
        if (denom != 0.0) {
            refined_bin = (double)max_bin + 0.5 * (alpha - gamma) / denom;
        }
    }

    float freq_hz = (float)(refined_bin * (double)SAMPLE_RATE / (double)N_FFT);

    // This only marks numerically invalid final values. Wrong but finite
    // frequencies remain AX_OK and are judged later by MIDI/note comparison.
    if (!std::isfinite(freq_hz) || freq_hz <= 0.0f || freq_hz > ((float)SAMPLE_RATE / 2.0f)) {
        *out_status = AX_NUMERIC_ERROR;
        freq_hz = -1.0f;
    }

    free(cfg);
    free(cx_in);
    free(cx_out);
    free(total_power);
    free(hps);

    return freq_hz;
}

int main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;

    // Keep stdout as close as possible to the JPEG putchar-based behavior.
    setvbuf(stdout, NULL, _IONBF, 0);

    int num_samples = 0;
    float* audio_data = read_audio_from_stdin(&num_samples);

    if (num_samples < 0) {
        write_ax_record(AX_ALLOC_ERROR, 0, 0, -1.0f);
        return 1;
    }

    if (!audio_data || num_samples == 0) {
        write_ax_record(AX_EMPTY_INPUT, 0, 0, -1.0f);
        free(audio_data);
        return 1;
    }

    int num_frames = 0;
    int status = AX_OK;
    float freq_hz = compute_dominant_frequency(audio_data, num_samples, &num_frames, &status);

    write_ax_record(status, num_samples, num_frames, freq_hz);

    free(audio_data);
    return (status == AX_OK) ? 0 : 1;
}
