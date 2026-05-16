/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
// clang-format off
#ifndef MINDSPORE_NNACL_ARITHMETIC_SELF_AVX512_H_
#define MINDSPORE_NNACL_ARITHMETIC_SELF_AVX512_H_

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

static inline int Fp32CastGatherReduceInt64FusionAVX512(int index, float *output_data, const int64_t *input_indices, const float *input_data,
                                                          int32_t inner_size, int32_t input_data_inner_size, int32_t outer_start,
                                                          int32_t outer_end) {
  for (int block_max_size = input_data_inner_size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    for (int i = outer_start; i < outer_end; i++) {
      SIMD_F32 result = SIMD_SET0_F32;
      for (int j = 0; j < inner_size; j++) {
        int64_t indice = input_indices[i * inner_size + j];
        result = SIMD_ADD_F32(result, SIMD_LD_F32(input_data + indice * input_data_inner_size + index));
      }
      SIMD_ST_F32(output_data + i * input_data_inner_size + index, result);
    }
  }
  return index;
}


static inline int Fp32CastGatherReduceInt32FusionAVX512(int index, float *output_data, const int32_t *input_indices, const float *input_data,
                                                          int32_t inner_size, int32_t input_data_inner_size, int32_t outer_start,
                                                          int32_t outer_end) {
  for (int block_max_size = input_data_inner_size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    for (int i = outer_start; i < outer_end; i++) {
      SIMD_F32 result = SIMD_SET0_F32;
      for (int j = 0; j < inner_size; j++) {
        int32_t indice = input_indices[i * inner_size + j];
        result = SIMD_ADD_F32(result, SIMD_LD_F32(input_data + indice * input_data_inner_size + index));
      }
      SIMD_ST_F32(output_data + i * input_data_inner_size + index, result);
    }
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
