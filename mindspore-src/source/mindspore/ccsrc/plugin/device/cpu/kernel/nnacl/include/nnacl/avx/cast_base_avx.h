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
#ifndef NNACL_BASE_CAST_BASE_AVX_H_
#define NNACL_BASE_CAST_BASE_AVX_H_

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

static inline int Int32ToFloat32AVX(int index, const int32_t *input, float *output, int number) {
  for (int block_max_size = number - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_EPI32 value = SIMD_LD_EPI32(input + index);
    SIMD_ST_F32(output + index, SIMD_EPI32_TO_F32(value));
  }
  return index;
}

#ifndef MS_SIMD_NEON
static inline int Float32ToInt32AVX(int index, const float *input, int32_t *output, int number) {
  for (int block_max_size = number - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_F32 value = SIMD_LD_F32(input + index);
    SIMD_ST_EPI32(output + index, SIMD_F32_TO_EPI32(value));
  }
  return index;
}
#endif

#undef MS_SIMD_INSTRUCTION
#undef BLOCK_NUM
#pragma GCC pop_options
#undef MS_SIMD_AVX
#ifdef __cplusplus
}
#endif
#endif
