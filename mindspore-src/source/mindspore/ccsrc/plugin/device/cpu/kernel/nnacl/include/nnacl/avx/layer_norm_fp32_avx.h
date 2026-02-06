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
#ifndef MINDSPORE_NNACL_FP32_LAYER_NORM_FP32_AVX_H_
#define MINDSPORE_NNACL_FP32_LAYER_NORM_FP32_AVX_H_

#include "nnacl/intrinsics/ms_simd_instructions.h"
#include "nnacl/intrinsics/ms_simd_avx_instructions.h"

#ifdef __cplusplus
extern "C" {
#endif
#pragma GCC push_options
#pragma GCC target("avx", "avx2")
#define MS_SIMD_INSTRUCTION MS_SIMD_AVX_INSTRUCTION
#define BLOCK_NUM 8
#define MS_SIMD_AVX

static inline int LayerNormMeanAndSquareAVX(int index, const float *src, int num, float *mean, float *square_mean) {
  if (num >= 4 * BLOCK_NUM) {
    SIMD_F32 sum_val = SIMD_SET0_F32;
    SIMD_F32 square_sum_val = SIMD_SET0_F32;
    for (int block_max_size = num - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
      SIMD_F32 value = SIMD_LD_F32(src + index);
      SIMD_F32 square_value = SIMD_MUL_F32(value, value);
      sum_val = SIMD_ADD_F32(sum_val, value);
      square_sum_val = SIMD_ADD_F32(square_sum_val, square_value);
    }
    *mean += SIMD_GET_SUM_F32(sum_val);
    *square_mean += SIMD_GET_SUM_F32(square_sum_val);
  }
  return index;
}

static inline int LayerNormGammaAndBetaAVX(int index, float *dst, const float *src, const float *gamma_data,
  const float *beta_data, int num, const float mean, const float deno) {
  SIMD_F32 mean_val = SIMD_MOV_F32(mean);
  SIMD_F32 deno_val = SIMD_MOV_F32(deno);
  for (int block_max_size = num - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_F32 value = SIMD_LD_F32(src + index);
    SIMD_F32 out_value = SIMD_SUB_F32(value, mean_val);
    out_value = SIMD_MUL_F32(out_value, deno_val);
    out_value = SIMD_FMADD_F32(out_value, SIMD_LD_F32(gamma_data + index), SIMD_LD_F32(beta_data + index));
    SIMD_ST_F32(dst + index, out_value);
  }
  return index;
}

#undef MS_SIMD_INSTRUCTION
#undef BLOCK_NUM
#pragma GCC pop_options
#undef MS_SIMD_AVX
#ifdef __cplusplus
}
#endif
#endif
