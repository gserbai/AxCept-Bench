// //////////////////////////////////////////////////////////
// sobel.cpp - Filtro de Sobel 
// Guilherme Saides Serbai 2026
// //////////////////////////////////////////////////////////
// Entrada (stdin):  PGM binário (P5), maxval 255
// Saída   (stdout): PGM binário (P5) com a magnitude do gradiente
// //////////////////////////////////////////////////////////

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

static const int MAX_DIM = 8192;  // sanity check (falhas podem corromper o header)

// ===============================
// Parser PGM a partir de buffer
// ===============================
static bool isSpace(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}

// Pula whitespace e comentários (#...\n) do header PGM
static void skipSpaceAndComments(const std::vector<unsigned char>& b, size_t& cur) {
    while (cur < b.size()) {
        if (isSpace(b[cur])) { cur++; continue; }
        if (b[cur] == '#') {
            while (cur < b.size() && b[cur] != '\n') cur++;
            continue;
        }
        break;
    }
}

static int readHeaderInt(const std::vector<unsigned char>& b, size_t& cur) {
    skipSpaceAndComments(b, cur);
    int v = 0;
    bool started = false;
    while (cur < b.size() && b[cur] >= '0' && b[cur] <= '9') {
        v = v * 10 + (b[cur] - '0');
        cur++;
        started = true;
        if (v > 1000000) return -1;
    }
    return started ? v : -1;
}

int main() {
    // ===============================
    // 1. Ler TODA a entrada
    // ===============================
    std::vector<unsigned char> in;
    unsigned char tmp[4096];
    size_t n;
    while ((n = fread(tmp, 1, sizeof(tmp), stdin)) > 0)
        in.insert(in.end(), tmp, tmp + n);

    if (in.empty()) {
        fprintf(stderr, "Erro: nenhuma entrada recebida.\n");
        return 1;
    }

    // ===============================
    // 2. Header PGM (P5)
    // ===============================
    size_t cur = 0;
    skipSpaceAndComments(in, cur);  // absorve o padding de espaços
    if (cur + 2 > in.size() || in[cur] != 'P' || in[cur + 1] != '5') {
        fprintf(stderr, "Erro: entrada nao e PGM P5.\n");
        return 1;
    }
    cur += 2;

    int w   = readHeaderInt(in, cur);
    int h   = readHeaderInt(in, cur);
    int max = readHeaderInt(in, cur);
    if (w < 3 || h < 3 || w > MAX_DIM || h > MAX_DIM || max != 255) {
        fprintf(stderr, "Erro: header invalido (w=%d h=%d max=%d).\n", w, h, max);
        return 1;
    }
    cur++;  // exatamente UM whitespace separa o header dos pixels

    const size_t npix = (size_t)w * h;
    if (in.size() - cur < npix) {
        fprintf(stderr, "Erro: faltam pixels (%zu de %zu).\n", in.size() - cur, npix);
        return 1;
    }

    // Copia os pixels para um buffer próprio e libera o buffer bruto
    std::vector<unsigned char> img(in.begin() + cur, in.begin() + cur + npix);
    std::vector<unsigned char>().swap(in);

    // ===============================
    // 3. Sobel 3x3
    // ===============================
    //      Gx = [-1 0 +1]      Gy = [-1 -2 -1]
    //           [-2 0 +2]           [ 0  0  0]
    //           [-1 0 +1]           [+1 +2 +1]
    std::vector<unsigned char> out(npix, 0);  // bordas ficam em 0

    // Saída em streaming (mesma regra do JPEG): header primeiro, depois
    // cada linha assim que é calculada, com fflush. Se houver segfault,
    // as linhas já escritas ficam no arquivo antes do dump do pk.
    // O buffer `out` continua com a imagem inteira, então a exposição
    // à memória aproximada é a mesma da versão que escrevia no final.
    printf("P5\n%d %d\n255\n", w, h);
    fwrite(&out[0], 1, (size_t)w, stdout);  // linha 0 (borda)
    fflush(stdout);

    for (int y = 1; y < h - 1; y++) {
        const unsigned char* r0 = &img[(size_t)(y - 1) * w];
        const unsigned char* r1 = &img[(size_t)y * w];
        const unsigned char* r2 = &img[(size_t)(y + 1) * w];
        unsigned char* o = &out[(size_t)y * w];

        for (int x = 1; x < w - 1; x++) {
            int gx = -r0[x - 1] + r0[x + 1]
                     - 2 * r1[x - 1] + 2 * r1[x + 1]
                     - r2[x - 1] + r2[x + 1];
            int gy = -r0[x - 1] - 2 * r0[x] - r0[x + 1]
                     + r2[x - 1] + 2 * r2[x] + r2[x + 1];

            int mag = (int)sqrtf((float)(gx * gx + gy * gy));
            o[x] = (unsigned char)(mag > 255 ? 255 : mag);
        }

        fwrite(o, 1, (size_t)w, stdout);  // linha y pronta -> stdout
        fflush(stdout);
    }

    fwrite(&out[(size_t)(h - 1) * w], 1, (size_t)w, stdout);  // última linha (borda)
    fflush(stdout);
    return 0;
}