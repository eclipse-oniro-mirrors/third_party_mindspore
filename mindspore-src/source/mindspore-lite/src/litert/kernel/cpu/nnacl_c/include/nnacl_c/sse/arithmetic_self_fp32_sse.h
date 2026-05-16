/**
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MINDSPORE_NNACL_ARITHMETIC_SELF_SSE_H_
#define MINDSPORE_NNACL_ARITHMETIC_SELF_SSE_H_

#include "nnacl_c/intrinsics/ms_simd_instructions.h"
#include "nnacl_c/intrinsics/ms_simd_sse_instructions.h"

#ifdef __cplusplus
extern "C" {
#endif
#pragma GCC push_options
#pragma GCC target("sse4.1")
#define MS_SIMD_INSTRUCTION MS_SIMD_SSE_INSTRUCTION
#define BLOCK_NUM 4
#define MS_SIMD_SSE

#if defined(MS_SIMD_AVX512)
// only avx512 support abs fp32 instruction
static inline int ElementAbsSSE(int index, const float *input, float *output, const int element_size) {
  for (int block_max_size = element_size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_ST_F32(output + index, SIMD_ABS_F32(SIMD_LD_F32(input + index)));
  }
  return index;
}

static inline int ElementAbsIntSSE(int index, const int32_t *input, int32_t *output, const int element_size) {
  for (int block_max_size = element_size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_ST_EPI32(output + index, SIMD_ABS_EPI32(SIMD_LD_EPI32(input + index)));
  }
  return index;
}
#endif

#if !defined(MS_SIMD_NEON)
// not support neon
  static inline int ElementCosSSE(int index, const float *input, float *output, const int element_size) {
    for (int block_max_size = element_size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
      SIMD_F32 vin = SIMD_LD_F32(input + index);
      SIMD_ST_F32(output + index, SIMD_COS_F32(vin));
    }
    return index;
  }

  static inline int ElementLogSSE(int index, const float *input, float *output, const int element_size) {
    for (int block_max_size = element_size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
      SIMD_F32 vin = SIMD_LD_F32(input + index);
      SIMD_ST_F32(output + index, SIMD_LOG_F32(vin));
    }
    return index;
  }
#endif

static inline int ElementSquareSSE(int index, const float *input, float *output, const int element_size) {
  for (int block_max_size = element_size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_F32 vin = SIMD_LD_F32(input + index);
    SIMD_ST_F32(output + index, SIMD_MUL_F32(vin, vin));
  }
  return index;
}

static inline int ElementSqrtSSE(int index, const float *input, float *output, const int element_size) {
  for (int block_max_size = element_size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_ST_F32(output + index, SIMD_SQRT_F32(SIMD_LD_F32(input + index)));
  }
  return index;
}

static inline int ElementRsqrtSSE(int index, const float *input, float *output, const int element_size) {
  for (int block_max_size = element_size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_ST_F32(output + index, SIMD_RSQRT_F32(SIMD_LD_F32(input + index)));
  }
  return index;
}

static inline int ElementMishSSE(int index, const float *input, float *output, const int element_size) {
  SIMD_F32 one = SIMD_MOV_F32(1.0f);
  for (int block_max_size = element_size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_F32 exp_add_one = SIMD_ADD_F32(SIMD_EXP_F32(SIMD_LD_F32(input + index)), one);
    SIMD_F32 exp_pow = SIMD_MUL_F32(exp_add_one, exp_add_one);
    SIMD_ST_F32(output + index, SIMD_MUL_F32(SIMD_LD_F32(input + index),
                                             SIMD_DIV_F32(SIMD_SUB_F32(exp_pow, one), SIMD_ADD_F32(exp_pow, one))));
  }
  return index;
}

#if defined(MS_SIMD_AVX) || defined(MS_SIMD_SSE)
// avx512 dont support round fp32 instruction
static inline int ElementRoundSSE(int index, const float *input, float *output, const int element_size) {
  for (int block_max_size = element_size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_ST_F32(output + index, SIMD_ROUND_F32(SIMD_LD_F32(input + index)));
  }
  return index;
}
#endif

#ifndef MS_SIMD_NEON
// neon dont support floor fp32 instruction
static inline int ElementFloorSSE(int index, const float *input, float *output, const int element_size) {
  for (int block_max_size = element_size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_ST_F32(output + index, SIMD_FLOOR_F32(SIMD_LD_F32(input + index)));
  }
  return index;
}
#endif

#ifndef MS_SIMD_NEON
static inline int ElementCeilSSE(int index, const float *input, float *output, const int element_size) {
  for (int block_max_size = element_size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_ST_F32(output + index, SIMD_CEIL_F32(SIMD_LD_F32(input + index)));
  }
  return index;
}
#endif

static inline int ElementNegativeSSE(int index, const float *input, float *output, const int element_size) {
  for (int block_max_size = element_size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_ST_F32(output + index, SIMD_MUL_N_F32(SIMD_LD_F32(input + index), -1.0f));
  }
  return index;
}

static inline int ElementNegativeIntSSE(int index, const int32_t *input, int32_t *output, const int element_size) {
  for (int block_max_size = element_size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_ST_EPI32(output + index, SIMD_MUL_N_EPI32(SIMD_LD_EPI32(input + index), -1));
  }
  return index;
}

static inline int ElementReciprocalSSE(int index, const float *input, float *output, const int element_size) {
  SIMD_F32 num1 = SIMD_MOV_F32(1.0f);
  for (int block_max_size = element_size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_ST_F32(output + index, SIMD_DIV_F32(num1, SIMD_LD_F32(input + index)));
  }
  return index;
}

#undef MS_SIMD_INSTRUCTION
#undef BLOCK_NUM
#pragma GCC pop_options
#undef MS_SIMD_SSE
#ifdef __cplusplus
}
#endif
#endif
