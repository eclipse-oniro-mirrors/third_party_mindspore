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
#ifndef MINDSPORE_NNACL_FP32_ACTIVATION_AVX_H_
#define MINDSPORE_NNACL_FP32_ACTIVATION_AVX_H_

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

static inline int ElementMulAVX(int index, const float *in0, const float *in1, float *out, int size) {
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_F32 vin0 = SIMD_LD_F32(in0 + index);
    SIMD_F32 vin1 = SIMD_LD_F32(in1 + index);
    SIMD_F32 vout = SIMD_MUL_F32(vin0, vin1);
    SIMD_ST_F32(out + index, vout);
  }
  return index;
}

static inline int ElementMulReluAVX(int index, const float *in0, const float *in1, float *out, int size) {
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_F32 vin0 = SIMD_LD_F32(in0 + index);
    SIMD_F32 vin1 = SIMD_LD_F32(in1 + index);
    SIMD_F32 vout = SIMD_MAX_N_F32(SIMD_MUL_F32(vin0, vin1), 0.0f);
    SIMD_ST_F32(out + index, vout);
  }
  return index;
}

static inline int ElementMulRelu6AVX(int index, const float *in0, const float *in1, float *out, int size) {
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_F32 vin0 = SIMD_LD_F32(in0 + index);
    SIMD_F32 vin1 = SIMD_LD_F32(in1 + index);
    SIMD_F32 vout = SIMD_MIN_N_F32(SIMD_MAX_N_F32(SIMD_MUL_F32(vin0, vin1), 0.0f), 6.0f);
    SIMD_ST_F32(out + index, vout);
  }
  return index;
}

static inline int ElementMulIntAVX(int index, const int32_t *in0, const int32_t *in1, int32_t *out, int size) {
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_EPI32 vin0 = SIMD_LD_EPI32(in0 + index);
    SIMD_EPI32 vin1 = SIMD_LD_EPI32(in1 + index);
    SIMD_EPI32 vout = SIMD_MUL_EPI32(vin0, vin1);
    SIMD_ST_EPI32(out + index, vout);
  }
  return index;
}

static inline int ElementMulReluIntAVX(int index, const int32_t *in0, const int32_t *in1, int32_t *out, int size) {
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_EPI32 vin0 = SIMD_LD_EPI32(in0 + index);
    SIMD_EPI32 vin1 = SIMD_LD_EPI32(in1 + index);
    SIMD_EPI32 vout = SIMD_MAX_N_EPI32(SIMD_MUL_EPI32(vin0, vin1), 0.0f);
    SIMD_ST_EPI32(out + index, vout);
  }
  return index;
}

static inline int ElementMulRelu6IntAVX(int index, const int32_t *in0, const int32_t *in1, int32_t *out, int size) {
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_EPI32 vin0 = SIMD_LD_EPI32(in0 + index);
    SIMD_EPI32 vin1 = SIMD_LD_EPI32(in1 + index);
    SIMD_EPI32 vout = SIMD_MIN_N_EPI32(SIMD_MAX_N_EPI32(SIMD_MUL_EPI32(vin0, vin1), 0.0f), 6.0f);
    SIMD_ST_EPI32(out + index, vout);
  }
  return index;
}

static inline int ElementOptMulNum0AVX(int index, const float *in0, const float *in1, float *out, int size) {
  SIMD_F32 vin0_opt_ = SIMD_MOV_F32(in0[0]);
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_F32 vin1 = SIMD_LD_F32(in1 + index);
    SIMD_F32 vout = SIMD_MUL_F32(vin0_opt_, vin1);
    SIMD_ST_F32(out + index, vout);
  }
  return index;
}

static inline int ElementOptMulNum1AVX(int index, const float *in0, const float *in1, float *out, int size) {
  SIMD_F32 vin1_opt_ = SIMD_MOV_F32(in1[0]);
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_F32 vin0 = SIMD_LD_F32(in0 + index);
    SIMD_F32 vout = SIMD_MUL_F32(vin0, vin1_opt_);
    SIMD_ST_F32(out + index, vout);
  }
  return index;
}

static inline int ElementOptMulReluNum0AVX(int index, const float *in0, const float *in1, float *out, int size) {
  SIMD_F32 vin0_opt_ = SIMD_MOV_F32(in0[0]);
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_F32 vin1 = SIMD_LD_F32(in1 + index);
    SIMD_F32 vout = SIMD_MAX_N_F32(SIMD_MUL_F32(vin0_opt_, vin1), 0.0f);
    SIMD_ST_F32(out + index, vout);
  }
  return index;
}

static inline int ElementOptMulReluNum1AVX(int index, const float *in0, const float *in1, float *out, int size) {
  SIMD_F32 vin1_opt_ = SIMD_MOV_F32(in1[0]);
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_F32 vin0 = SIMD_LD_F32(in0 + index);
    SIMD_F32 vout = SIMD_MAX_N_F32(SIMD_MUL_F32(vin0, vin1_opt_), 0.0f);
    SIMD_ST_F32(out + index, vout);
  }
  return index;
}

static inline int ElementOptMulRelu6Num0AVX(int index, const float *in0, const float *in1, float *out, int size) {
  SIMD_F32 vin0_opt_ = SIMD_MOV_F32(in0[0]);
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_F32 vin1 = SIMD_LD_F32(in1 + index);
    SIMD_F32 vout = SIMD_MIN_N_F32(SIMD_MAX_N_F32(SIMD_MUL_F32(vin0_opt_, vin1), 0.0f), 6.0f);
    SIMD_ST_F32(out + index, vout);
  }
  return index;
}

static inline int ElementOptMulRelu6Num1AVX(int index, const float *in0, const float *in1, float *out, int size) {
  SIMD_F32 vin1_opt_ = SIMD_MOV_F32(in1[0]);
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_F32 vin0 = SIMD_LD_F32(in0 + index);
    SIMD_F32 vout = SIMD_MIN_N_F32(SIMD_MAX_N_F32(SIMD_MUL_F32(vin0, vin1_opt_), 0.0f), 6.0f);
    SIMD_ST_F32(out + index, vout);
  }
  return index;
}

static inline int ElementOptMulIntNum0AVX(int index, const int32_t *in0, const int32_t *in1, int32_t *out, int size) {
  SIMD_EPI32 vin0_opt_ = SIMD_MOV_EPI32(in0[0]);
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_EPI32 vin1 = SIMD_LD_EPI32(in1 + index);
    SIMD_EPI32 vout = SIMD_MUL_EPI32(vin0_opt_, vin1);
    SIMD_ST_EPI32(out + index, vout);
  }
  return index;
}

static inline int ElementOptMulIntNum1AVX(int index, const int32_t *in0, const int32_t *in1, int32_t *out, int size) {
  SIMD_EPI32 vin1_opt_ = SIMD_MOV_EPI32(in1[0]);
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_EPI32 vin0 = SIMD_LD_EPI32(in0 + index);
    SIMD_EPI32 vout = SIMD_MUL_EPI32(vin0, vin1_opt_);
    SIMD_ST_EPI32(out + index, vout);
  }
  return index;
}

static inline int ElementOptMulReluIntNum0AVX(int index, const int32_t *in0, const int32_t *in1, int32_t *out, int size) {
  SIMD_EPI32 vin0_opt_ = SIMD_MOV_EPI32(in0[0]);
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_EPI32 vin1 = SIMD_LD_EPI32(in1 + index);
    SIMD_EPI32 vout = SIMD_MAX_N_EPI32(SIMD_MUL_EPI32(vin0_opt_, vin1), 0.0f);
    SIMD_ST_EPI32(out + index, vout);
  }
  return index;
}

static inline int ElementOptMulReluIntNum1AVX(int index, const int32_t *in0, const int32_t *in1, int32_t *out, int size) {
  SIMD_EPI32 vin1_opt_ = SIMD_MOV_EPI32(in1[0]);
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_EPI32 vin0 = SIMD_LD_EPI32(in0 + index);
    SIMD_EPI32 vout = SIMD_MAX_N_EPI32(SIMD_MUL_EPI32(vin0, vin1_opt_), 0.0f);
    SIMD_ST_EPI32(out + index, vout);
  }
  return index;
}

static inline int ElementOptMulRelu6IntNum0AVX(int index, const int32_t *in0, const int32_t *in1, int32_t *out, int size) {
  SIMD_EPI32 vin0_opt_ = SIMD_MOV_EPI32(in0[0]);
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_EPI32 vin1 = SIMD_LD_EPI32(in1 + index);
    SIMD_EPI32 vout = SIMD_MIN_N_EPI32(SIMD_MAX_N_EPI32(SIMD_MUL_EPI32(vin0_opt_, vin1), 0.0f), 6.0f);
    SIMD_ST_EPI32(out + index, vout);
  }
  return index;
}

static inline int ElementOptMulRelu6IntNum1AVX(int index, const int32_t *in0, const int32_t *in1, int32_t *out, int size) {
  SIMD_EPI32 vin1_opt_ = SIMD_MOV_EPI32(in1[0]);
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_EPI32 vin0 = SIMD_LD_EPI32(in0 + index);
    SIMD_EPI32 vout = SIMD_MIN_N_EPI32(SIMD_MAX_N_EPI32(SIMD_MUL_EPI32(vin0, vin1_opt_), 0.0f), 6.0f);
    SIMD_ST_EPI32(out + index, vout);
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
