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
#ifndef NNACL_BASE_SCATTER_ND_BINARY_AVX512_H_
#define NNACL_BASE_SCATTER_ND_BINARY_AVX512_H_

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

  static inline int ScatterNDAddFp32AVX512(int index, const float *update, int size, float *output) {
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_ST_F32(output + index, SIMD_ADD_F32(SIMD_LD_F32(output + index), SIMD_LD_F32(update + index)));
  }
  return index;
}

static inline int ScatterNDAddInt32AVX512(int index, const int *update, int size, int *output) {
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_ST_EPI32(output + index, SIMD_ADD_EPI32(SIMD_LD_EPI32(output + index), SIMD_LD_EPI32(update + index)));
  }
  return index;
}

static inline int ScatterNDMaxFp32AVX512(int index, const float *update, int size, float *output) {
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_ST_F32(output + index, SIMD_MAX_F32(SIMD_LD_F32(output + index), SIMD_LD_F32(update + index)));
  }
  return index;
}

static inline int ScatterNDMaxInt32AVX512(int index, const int *update, int size, int *output) {
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_ST_EPI32(output + index, SIMD_MAX_EPI32(SIMD_LD_EPI32(output + index), SIMD_LD_EPI32(update + index)));
  }
  return index;
}

#undef MS_SIMD_INSTRUCTION
#undef BLOCK_NUM
#pragma GCC pop_options
#undef MS_SIMD_AVX512
#ifdef __cplusplus
}
#endif
#endif  // NNACL_BASE_SCATTER_ND_BINARY_AVX512_H_
