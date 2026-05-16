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
#ifndef MINDSPORE_NNACL_FP32_GROUP_NORM_FP32_SSE_H_
#define MINDSPORE_NNACL_FP32_GROUP_NORM_FP32_SSE_H_

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

static inline int64_t GroupNormFp32SSE(int64_t index, const float *unit_input, float scale, float offset, float mean,
  float var_sqrt, int unit, float *unit_output) {
  SIMD_F32 mean_val = SIMD_MOV_F32(mean);
  SIMD_F32 v_sqrt = SIMD_MOV_F32(var_sqrt);
  SIMD_F32 scale_val = SIMD_MOV_F32(scale);
  SIMD_F32 offset_val = SIMD_MOV_F32(offset);
  for (int block_max_size = unit - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_F32 input = SIMD_LD_F32(unit_input + index);
    SIMD_F32 norm_val = SIMD_DIV_F32(SIMD_SUB_F32(input, mean_val), v_sqrt);
    SIMD_F32 output = SIMD_ADD_F32(SIMD_MUL_F32(norm_val, scale_val), offset_val);
    SIMD_ST_F32(unit_output + index, output);
  }
  return index;
}

static inline int64_t GroupNormReduceSumSSE(int64_t index, const float *in, float *sum, int unit) {
  if (unit - index >= 4 * BLOCK_NUM) {
    SIMD_F32 tmp = SIMD_MOV_F32(0);
    for (int block_max_size = unit - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
      tmp = SIMD_ADD_F32(tmp, SIMD_LD_F32(in + index));
    }
    *sum += SIMD_GET_SUM_F32(tmp);
  }
  return index;
}

static inline int64_t GroupNormReduceVarSSE(int64_t index, const float *in, float mean, float *sum, int unit) {
  if (unit - index >= 4 * BLOCK_NUM) {
    SIMD_F32 mean_val = SIMD_MOV_F32(mean);
    SIMD_F32 tmp = SIMD_MOV_F32(0);
    for (int block_max_size = unit - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
      SIMD_F32 input = SIMD_SUB_F32(SIMD_LD_F32(in + index), mean_val);
      tmp = SIMD_ADD_F32(tmp, SIMD_MUL_F32(input, input));
    }
    *sum += SIMD_GET_SUM_F32(tmp);
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
