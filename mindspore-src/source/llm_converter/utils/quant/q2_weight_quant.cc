#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>
#include <algorithm>

#include <vector>
#include <string>
#if defined(__ARM_NEON)
using fp16_t = __fp16;
#else
using fp16_t = uint16_t;
#endif
static inline float fp32_from_bits(uint32_t w) {
  union {
    uint32_t as_bits;
    float as_value;
  } fp32;
  fp32.as_bits = w;
  return fp32.as_value;
}
static inline uint32_t fp32_to_bits(float f) {
  union {
    float as_value;
    uint32_t as_bits;
  } fp32;
  fp32.as_value = f;
  return fp32.as_bits;
}
static inline fp16_t ggml_compute_fp32_to_fp16(float f) {
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && \
  (!defined(__cplusplus) || __cplusplus >= 201703L)
  const float scale_to_inf = 0x1.0p+112f;
  const float scale_to_zero = 0x1.0p-110f;
#else
  const float scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
  const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
#endif
  float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

  const uint32_t w = fp32_to_bits(f);
  const uint32_t shl1_w = w + w;
  const uint32_t sign = w & UINT32_C(0x80000000);
  uint32_t bias = shl1_w & UINT32_C(0xFF000000);
  if (bias < UINT32_C(0x71000000)) {
    bias = UINT32_C(0x71000000);
  }

  base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
  const uint32_t bits = fp32_to_bits(base);
  const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
  const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
  const uint32_t nonsign = exp_bits + mantissa_bits;
  return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}
constexpr uint32_t QK2_N_0_K = 32;
constexpr uint32_t QKX_N_0_K = 32;
// The fb has 128 * 8b space(using db),and each 8b data correspond to each channel in l0c, so the max value of n is 128
constexpr uint32_t N_FACTOR_Q = 64;

// MMAD.s8/MMAD.s8s4: Fractal matrix in l0 is 16 * 32
constexpr uint32_t FRACTAL_SIZE_N_Q = 16;
constexpr uint32_t FRACTAL_SIZE_D_Q = 32;
constexpr uint32_t M_FACTOR_FP16 = 80;
constexpr uint32_t K_FACTOR_FP16 = 64;
constexpr uint32_t N_FACTOR_FP16 = 80;
constexpr uint32_t K_FACTOR_Q_2_0 = 32;
constexpr uint32_t FRACTAL_SIZE_N_F16 = 16;
constexpr uint32_t FRACTAL_SIZE_D_F16 = 16;
extern "C" void quantize_Q2_N_0_reference(const float *__restrict__ x, uint8_t *__restrict__ y, int n, int d) {
  static const int qk = QK2_N_0_K;
  assert(d % qk == 0);
  const int nb = n * d / qk;
  int idx = 0;
  const int nums_per_int = 4;
  fp16_t *yscales = (fp16_t *)(y + n * d / nums_per_int);
  for (int i = 0; i < nb; i++) {
    float amax = 0.0f;  // absolute max
    float max = 0.0f;

    for (int j = 0; j < qk; j++) {
      const float v = x[i * qk + j];
      if (amax < fabsf(v)) {
        amax = fabsf(v);
        max = v;
      }
    }

    const float scale = max / -2;
    const float id = scale ? 1.0f / scale : 0.0f;
    yscales[idx++] = ggml_compute_fp32_to_fp16(scale);
    // yscales[idx++] = (fp16_t)(scale);

    for (int j = 0; j < qk / nums_per_int; j++) {
      y[i * qk / nums_per_int + j] = 0;
      for (int z = 0; z < nums_per_int; z++) {
        const float xi = x[i * qk + z * (qk / nums_per_int) + j] * id;
        const uint8_t xii = std::min((uint8_t)3, (uint8_t)(xi + 2.5f));
        y[i * qk / nums_per_int + j] |= (xii << (2 * z));
      }
    }
  }
}