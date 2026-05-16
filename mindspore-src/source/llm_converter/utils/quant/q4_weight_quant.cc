#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>
#include <algorithm>

// Quant group
constexpr uint32_t QKX_N_0_K = 128;

// The fb has 128 * 8b space(using db),and each 8b data correspond to each channel in l0c, so the max value of n is 128
constexpr uint32_t N_FACTOR_Q = 64;

// MMAD.s8/MMAD.s8s4: Fractal matrix in l0 is 16 * 32
constexpr uint32_t FRACTAL_SIZE_N_Q = 16;
constexpr uint32_t FRACTAL_SIZE_D_Q = 32;

extern "C" void quantize_row_q4_n_0_nz_reference(const float *__restrict__ x, int8_t *__restrict__ y, int n, int d) {
  const int qk = QKX_N_0_K;
  const int aligned_n = (n + FRACTAL_SIZE_N_Q - 1) / FRACTAL_SIZE_N_Q * FRACTAL_SIZE_N_Q;
  const int k = aligned_n * d;
  assert(k % qk == 0);
  const int nb = k / qk;
  const int nb_row = d / qk;
  float *y1 = (float *)(y + nb * qk / 2);

  for (uint32_t nr_n = 0; nr_n < n; nr_n += N_FACTOR_Q) {
    const uint32_t nr_loop = std::min(N_FACTOR_Q, n - nr_n);
    const uint32_t aligned_nr = (nr_loop + FRACTAL_SIZE_N_Q - 1) / FRACTAL_SIZE_N_Q * FRACTAL_SIZE_N_Q;
    for (uint32_t i = 0; i < aligned_nr; i++) {
      if (i >= n - nr_n) {
        for (uint32_t j = 0; j < d / 2 / FRACTAL_SIZE_D_Q; j++) {
          for (uint32_t k = 0; k < FRACTAL_SIZE_D_Q; k++) {
            y[nr_n * d / 2 + FRACTAL_SIZE_D_Q * (i + j * aligned_nr) + k] = 0;
          }
        }
        continue;
      };

      for (int j = 0; j < nb_row; j++) {
        float amax = 0.0f;
        float max = 0.0f;

        for (int k = 0; k < qk; k++) {
          const float v = x[(nr_n + i) * d + j * qk + k];
          if (amax < fabsf(v)) {
            amax = fabsf(v);
            max = v;
          }
        }

        const float scale = max / -8;
        const float id = scale ? 1.0f / scale : 0.0f;

        y1[nr_n + i + (j / nb_row) + (j % nb_row) * n] = (scale);

        for (int k = 0; k < qk / 2; ++k) {
          const float x0 = x[(nr_n + i) * d + j * qk + k * 2] * id;
          const float x1 = x[(nr_n + i) * d + j * qk + k * 2 + 1] * id;

          const int8_t xi0 = std::min((int8_t)15, (int8_t)(x0 + 8.5f)) - 8;
          const int8_t xi1 = std::min((int8_t)15, (int8_t)(x1 + 8.5f)) - 8;

          const uint32_t kj = (j * qk / 2 + k) / FRACTAL_SIZE_D_Q;
          const uint32_t kk = k % FRACTAL_SIZE_D_Q;
          y[(nr_n * d / 2 + (kj * aligned_nr + i) * FRACTAL_SIZE_D_Q) + kk] = xi0 & 0x0F;
          y[(nr_n * d / 2 + (kj * aligned_nr + i) * FRACTAL_SIZE_D_Q) + kk] |= xi1 << 4;
        }
      }
    }
  }
}