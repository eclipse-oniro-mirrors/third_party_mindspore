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

#ifndef MINDSPORE_NNACL_FP32_DIV_AVX512_H_
#define MINDSPORE_NNACL_FP32_DIV_AVX512_H_

#include "nnacl_c/intrinsics/ms_simd_instructions.h"
#include "nnacl_c/intrinsics/ms_simd_avx512_instructions.h"

#ifdef __cplusplus
extern "C" {
#endif
#pragma GCC push_options
#pragma GCC target("avx512f")
#define MS_SIMD_INSTRUCTION MS_SIMD_AVX512_INSTRUCTION
#define BLOCK_NUM 16
#define MS_SIMD_AVX512

static inline int64_t ExpFp32AVX512(int64_t index, const float *src, float *dst, int num) {
  for (int block_max_size = num - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_EXP_ST_F32(SIMD_LD_F32(src + index), dst + index);
  }
  return index;
}

static inline int64_t ExpFp32WithInScaleAVX512(int64_t index, const float *src, float *dst, int num, float in_scale) {
  SIMD_F32 scale_vec = SIMD_MOV_F32(in_scale);
  for (int block_max_size = num - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_EXP_ST_F32(SIMD_MUL_F32(SIMD_LD_F32(src + index), scale_vec), dst + index);
  }
  return index;
}

static inline int64_t ExpFp32WithOutScaleAVX512(int64_t index, const float *src, float *dst, int num, float out_scale) {
  SIMD_F32 scale_vec = SIMD_MOV_F32(out_scale);
  for (int block_max_size = num - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_EXP_ST_F32(SIMD_LD_F32(src + index), dst + index);
    SIMD_ST_F32(dst + index, SIMD_MUL_F32(SIMD_LD_F32(dst + index), scale_vec));
  }
  return index;
}

#undef MS_SIMD_INSTRUCTION
#undef BLOCK_NUM
#pragma GCC pop_options
#undef MS_SIMD_AVX512
#ifdef __cplusplus
};
#endif
#endif
