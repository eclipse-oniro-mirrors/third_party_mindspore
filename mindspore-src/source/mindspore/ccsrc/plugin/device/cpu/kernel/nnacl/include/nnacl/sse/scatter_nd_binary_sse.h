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
#ifndef NNACL_BASE_SCATTER_ND_BINARY_SSE_H_
#define NNACL_BASE_SCATTER_ND_BINARY_SSE_H_

#include "nnacl/intrinsics/ms_simd_instructions.h"
#include "nnacl/intrinsics/ms_simd_sse_instructions.h"

#ifdef __cplusplus
extern "C" {
#endif
#pragma GCC push_options
#pragma GCC target("sse4.1")
#define MS_SIMD_INSTRUCTION MS_SIMD_SSE_INSTRUCTION
#define BLOCK_NUM 4
#define MS_SIMD_SSE

  static inline int ScatterNDAddFp32SSE(int index, const float *update, int size, float *output) {
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_ST_F32(output + index, SIMD_ADD_F32(SIMD_LD_F32(output + index), SIMD_LD_F32(update + index)));
  }
  return index;
}

static inline int ScatterNDAddInt32SSE(int index, const int *update, int size, int *output) {
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_ST_EPI32(output + index, SIMD_ADD_EPI32(SIMD_LD_EPI32(output + index), SIMD_LD_EPI32(update + index)));
  }
  return index;
}

static inline int ScatterNDMaxFp32SSE(int index, const float *update, int size, float *output) {
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_ST_F32(output + index, SIMD_MAX_F32(SIMD_LD_F32(output + index), SIMD_LD_F32(update + index)));
  }
  return index;
}

static inline int ScatterNDMaxInt32SSE(int index, const int *update, int size, int *output) {
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_ST_EPI32(output + index, SIMD_MAX_EPI32(SIMD_LD_EPI32(output + index), SIMD_LD_EPI32(update + index)));
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
#endif  // NNACL_BASE_SCATTER_ND_BINARY_SSE_H_
