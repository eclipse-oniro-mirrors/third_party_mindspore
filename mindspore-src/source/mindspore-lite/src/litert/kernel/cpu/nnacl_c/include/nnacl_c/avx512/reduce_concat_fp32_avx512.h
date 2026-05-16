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
#ifndef MINDSPORE_NNACL_REDUCE_CONCAT_FP32_SIMD_AVX512_H_
#define MINDSPORE_NNACL_REDUCE_CONCAT_FP32_SIMD_AVX512_H_

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

#ifdef MS_SIMD_AVX512
static inline int Fp32ReduceSumConcatAxisSize16FusionAVX512(int index, float *output_data, const float *input_data, int64_t reduce_axis_size) {
  SIMD_F32 zmm00 = SIMD_LD_F32(input_data);
  for (int l = 1; l < reduce_axis_size; l++) {
    input_data += (1 * BLOCK_NUM);
    zmm00 = SIMD_ADD_F32(zmm00, SIMD_LD_F32(input_data));
  }
  SIMD_ST_F32(output_data, zmm00);
  return index;
}

static inline int Fp32ReduceSumConcatAxisSize32FusionAVX512(int index, float *output_data, const float *input_data, int64_t reduce_axis_size) {
  SIMD_F32 zmm00 = SIMD_LD_F32(input_data + 0 * BLOCK_NUM);
  SIMD_F32 zmm01 = SIMD_LD_F32(input_data + 1 * BLOCK_NUM);
  for (int l = 1; l < reduce_axis_size; l++) {
    input_data += (2 * BLOCK_NUM);
    zmm00 = SIMD_ADD_F32(zmm00, SIMD_LD_F32(input_data + 0 * BLOCK_NUM));
    zmm01 = SIMD_ADD_F32(zmm01, SIMD_LD_F32(input_data + 1 * BLOCK_NUM));
  }

  SIMD_ST_F32(output_data + 0 * BLOCK_NUM, zmm00);
  SIMD_ST_F32(output_data + 1 * BLOCK_NUM, zmm01);

  return index;
}

static inline int Fp32ReduceSumConcatAxisSize64FusionAVX512(int index, float *output_data, const float *input_data, int64_t reduce_axis_size) {
  SIMD_F32 zmm00 = SIMD_LD_F32(input_data + 0 * BLOCK_NUM);
  SIMD_F32 zmm01 = SIMD_LD_F32(input_data + 1 * BLOCK_NUM);
  SIMD_F32 zmm02 = SIMD_LD_F32(input_data + 2 * BLOCK_NUM);
  SIMD_F32 zmm03 = SIMD_LD_F32(input_data + 3 * BLOCK_NUM);
  for (int l = 1; l < reduce_axis_size; l++) {
    input_data += (4 * BLOCK_NUM);
    zmm00 = SIMD_ADD_F32(zmm00, SIMD_LD_F32(input_data + 0 * BLOCK_NUM));
    zmm01 = SIMD_ADD_F32(zmm01, SIMD_LD_F32(input_data + 1 * BLOCK_NUM));
    zmm02 = SIMD_ADD_F32(zmm02, SIMD_LD_F32(input_data + 2 * BLOCK_NUM));
    zmm03 = SIMD_ADD_F32(zmm03, SIMD_LD_F32(input_data + 3 * BLOCK_NUM));
  }

  SIMD_ST_F32(output_data + 0 * BLOCK_NUM, zmm00);
  SIMD_ST_F32(output_data + 1 * BLOCK_NUM, zmm01);
  SIMD_ST_F32(output_data + 2 * BLOCK_NUM, zmm02);
  SIMD_ST_F32(output_data + 3 * BLOCK_NUM, zmm03);

  return index;
}

static inline int Fp32ReduceSumConcatAxisSize128FusionAVX512(int index, float *output_data, const float *input_data, int64_t reduce_axis_size) {
  SIMD_F32 zmm00 = SIMD_LD_F32(input_data + 0 * BLOCK_NUM);
  SIMD_F32 zmm01 = SIMD_LD_F32(input_data + 1 * BLOCK_NUM);
  SIMD_F32 zmm02 = SIMD_LD_F32(input_data + 2 * BLOCK_NUM);
  SIMD_F32 zmm03 = SIMD_LD_F32(input_data + 3 * BLOCK_NUM);
  SIMD_F32 zmm04 = SIMD_LD_F32(input_data + 4 * BLOCK_NUM);
  SIMD_F32 zmm05 = SIMD_LD_F32(input_data + 5 * BLOCK_NUM);
  SIMD_F32 zmm06 = SIMD_LD_F32(input_data + 6 * BLOCK_NUM);
  SIMD_F32 zmm07 = SIMD_LD_F32(input_data + 7 * BLOCK_NUM);
  for (int l = 1; l < reduce_axis_size; l++) {
    input_data += (8 * BLOCK_NUM);
    zmm00 = SIMD_ADD_F32(zmm00, SIMD_LD_F32(input_data + 0 * BLOCK_NUM));
    zmm01 = SIMD_ADD_F32(zmm01, SIMD_LD_F32(input_data + 1 * BLOCK_NUM));
    zmm02 = SIMD_ADD_F32(zmm02, SIMD_LD_F32(input_data + 2 * BLOCK_NUM));
    zmm03 = SIMD_ADD_F32(zmm03, SIMD_LD_F32(input_data + 3 * BLOCK_NUM));
    zmm04 = SIMD_ADD_F32(zmm00, SIMD_LD_F32(input_data + 4 * BLOCK_NUM));
    zmm05 = SIMD_ADD_F32(zmm01, SIMD_LD_F32(input_data + 5 * BLOCK_NUM));
    zmm06 = SIMD_ADD_F32(zmm02, SIMD_LD_F32(input_data + 6 * BLOCK_NUM));
    zmm07 = SIMD_ADD_F32(zmm03, SIMD_LD_F32(input_data + 7 * BLOCK_NUM));
  }

  SIMD_ST_F32(output_data + 0 * BLOCK_NUM, zmm00);
  SIMD_ST_F32(output_data + 1 * BLOCK_NUM, zmm01);
  SIMD_ST_F32(output_data + 2 * BLOCK_NUM, zmm02);
  SIMD_ST_F32(output_data + 3 * BLOCK_NUM, zmm03);
  SIMD_ST_F32(output_data + 4 * BLOCK_NUM, zmm04);
  SIMD_ST_F32(output_data + 5 * BLOCK_NUM, zmm05);
  SIMD_ST_F32(output_data + 6 * BLOCK_NUM, zmm06);
  SIMD_ST_F32(output_data + 7 * BLOCK_NUM, zmm07);

  return index;
}

#endif


#undef MS_SIMD_INSTRUCTION
#undef BLOCK_NUM
#pragma GCC pop_options
#undef MS_SIMD_AVX512
#ifdef __cplusplus
}
#endif
#endif
