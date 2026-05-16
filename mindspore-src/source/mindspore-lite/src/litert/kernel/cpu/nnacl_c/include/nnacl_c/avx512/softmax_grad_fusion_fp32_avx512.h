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
#ifndef MINDSPORE_NNACL_FP32_SOFTMAX_GRAD_FUSION_AVX512_H_
#define MINDSPORE_NNACL_FP32_SOFTMAX_GRAD_FUSION_AVX512_H_

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

static inline int64_t SoftmaxGradFusionOptAVX512(int64_t index, const float *a, const float *b,
                                                                 float *out, int64_t size) {
  SIMD_F32 result_vec = SIMD_MOV_F32(0.0f);
  for (int64_t block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_F32 a_vec = SIMD_LD_F32(a + index);
    SIMD_F32 b_vec = SIMD_LD_F32(b + index);
    result_vec = SIMD_FMADD_F32(a_vec, b_vec, result_vec);
  }
  *out += SIMD_GET_SUM_F32(result_vec);

  return index;
}

static inline int64_t ElementOptSubMulAVX512(int index, const float *in0, const float *in1, float sum,
                                                           float *out, int size) {
  SIMD_F32 vin1_opt_ = SIMD_MOV_F32(sum);
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_F32 vin0 = SIMD_LD_F32(in0 + index);
    SIMD_F32 vin1 = SIMD_LD_F32(in1 + index);
    SIMD_ST_F32(out + index, SIMD_MUL_F32(vin0, SIMD_SUB_F32(vin1, vin1_opt_)));
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
#endif
