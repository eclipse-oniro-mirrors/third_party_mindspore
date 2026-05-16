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

#ifndef MINDSPORE_NNACL_FP32_ADD_NEON_H_
#define MINDSPORE_NNACL_FP32_ADD_NEON_H_

#include "nnacl_c/intrinsics/ms_simd_instructions.h"
#include "nnacl_c/intrinsics/ms_simd_neon_instructions.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MS_SIMD_INSTRUCTION MS_SIMD_NEON_INSTRUCTION
#define BLOCK_NUM 4
#define MS_SIMD_NEON

static inline int ElementOptAddNEON(int index, const float *in0, const float *in1, float *out, int size) {
  SIMD_F32 vin0_ = SIMD_MOV_F32(in0[0]);
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_F32 vin1 = SIMD_LD_F32(in1 + index);
    SIMD_F32 vout = SIMD_ADD_F32(vin0_, vin1);
    SIMD_ST_F32(out + index, vout);
  }
  return index;
}

static inline int ElementOptAddExtNum0NEON(int index, const float *in0, const float *in1, const float alpha, float *out, int size) {
  SIMD_F32 vin0 = SIMD_MOV_F32(in0[0]);
  SIMD_F32 valpha = SIMD_MOV_F32(alpha);
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_F32 vin1 = SIMD_LD_F32(in1 + index);
    SIMD_F32 vin1_alpha = SIMD_MUL_F32(vin1, valpha);
    SIMD_F32 vout = SIMD_ADD_F32(vin0, vin1_alpha);
    SIMD_ST_F32(out + index, vout);
  }
  return index;
}

static inline int ElementOptAddExtNum1NEON(int index, const float *in0, const float *in1, const float alpha, float *out, int size) {
  SIMD_F32 vin1 = SIMD_MOV_F32(in1[0]);
  SIMD_F32 valpha = SIMD_MOV_F32(alpha);
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_F32 vin0 = SIMD_LD_F32(in0 + index);
    SIMD_F32 vin1_alpha = SIMD_MUL_F32(vin1, valpha);
    SIMD_F32 vout = SIMD_ADD_F32(vin0, vin1_alpha);
    SIMD_ST_F32(out + index, vout);
  }
  return index;
}

static inline int ElementOptAddIntNEON(int index, const int32_t *in0, const int32_t *in1, int32_t *out,
                                                     int size) {
  SIMD_EPI32 vin0_ = SIMD_MOV_EPI32(in0[0]);
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_EPI32 vin1 = SIMD_LD_EPI32(in1 + index);
    SIMD_EPI32 vout = SIMD_ADD_EPI32(vin0_, vin1);
    SIMD_ST_EPI32(out + index, vout);
  }
  return index;
}

static inline int ElementOptAddReluNEON(int index, const float *in0, const float *in1, float *out,
                                                      int size) {
  SIMD_F32 vin0_ = SIMD_MOV_F32(in0[0]);
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_F32 vin1 = SIMD_LD_F32(in1 + index);
    SIMD_F32 vout = SIMD_MAX_N_F32(SIMD_ADD_F32(vin0_, vin1), 0.0f);
    SIMD_ST_F32(out + index, vout);
  }
  return index;
}

static inline int ElementOptAddRelu6NEON(int index, const float *in0, const float *in1, float *out,
                                                       int size) {
  SIMD_F32 vin0_ = SIMD_MOV_F32(in0[0]);
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_F32 vin1 = SIMD_LD_F32(in1 + index);
    SIMD_F32 vout = SIMD_MIN_N_F32(SIMD_MAX_N_F32(SIMD_ADD_F32(vin0_, vin1), 0.0f), 6.0f);
    SIMD_ST_F32(out + index, vout);
  }
  return index;
}

static inline int ElementAddNEON(int index, const float *in0, const float *in1, float *out, int size) {
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_F32 vin0 = SIMD_LD_F32(in0 + index);
    SIMD_F32 vin1 = SIMD_LD_F32(in1 + index);
    SIMD_F32 vout = SIMD_ADD_F32(vin0, vin1);
    SIMD_ST_F32(out + index, vout);
  }
  return index;
}

static inline int ElementAddExtNEON(int index, const float *in0, const float *in1, const float alpha, float *out, int size) {
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_F32 vin0 = SIMD_LD_F32(in0 + index);
    SIMD_F32 vin1 = SIMD_LD_F32(in1 + index);
    SIMD_F32 valpha = SIMD_MOV_F32(alpha);
    SIMD_F32 vin1_alpha = SIMD_MUL_F32(vin1, valpha);
    SIMD_F32 vout = SIMD_ADD_F32(vin0, vin1_alpha);
    SIMD_ST_F32(out + index, vout);
  }
  return index;
}

static inline int ElementAddReluNEON(int index, const float *in0, const float *in1, float *out,
                                                   int size) {
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_F32 vin0 = SIMD_LD_F32(in0 + index);
    SIMD_F32 vin1 = SIMD_LD_F32(in1 + index);
    SIMD_F32 vout = SIMD_MAX_N_F32(SIMD_ADD_F32(vin0, vin1), 0.0f);
    SIMD_ST_F32(out + index, vout);
  }
  return index;
}

static inline int ElementAddRelu6NEON(int index, const float *in0, const float *in1, float *out,
                                                    int size) {
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_F32 vin0 = SIMD_LD_F32(in0 + index);
    SIMD_F32 vin1 = SIMD_LD_F32(in1 + index);
    SIMD_F32 vout = SIMD_MIN_N_F32(SIMD_MAX_N_F32(SIMD_ADD_F32(vin0, vin1), 0.0f), 6.0f);
    SIMD_ST_F32(out + index, vout);
  }
  return index;
}

static inline int ElementAddIntNEON(int index, const int32_t *in0, const int32_t *in1, int32_t *out, int size) {
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_EPI32 vin0 = SIMD_LD_EPI32(in0 + index);
    SIMD_EPI32 vin1 = SIMD_LD_EPI32(in1 + index);
    SIMD_EPI32 vout = SIMD_ADD_EPI32(vin0, vin1);
    SIMD_ST_EPI32(out + index, vout);
  }
  return index;
}

#undef MS_SIMD_INSTRUCTION
#undef BLOCK_NUM

#undef MS_SIMD_NEON
#ifdef __cplusplus
}
#endif
#endif
