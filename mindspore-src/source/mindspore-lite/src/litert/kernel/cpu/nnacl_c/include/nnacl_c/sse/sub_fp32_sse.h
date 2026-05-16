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

#ifndef MINDSPORE_NNACL_FP32_SUB_SSE_H_
#define MINDSPORE_NNACL_FP32_SUB_SSE_H_

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

static inline int ElementOptSubNum0SSE(int index, const float *in0, const float *in1, float *out,
                                                      int size) {
  SIMD_F32 vin0_opt = SIMD_MOV_F32(in0[0]);
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_F32 vin1 = SIMD_LD_F32(in1 + index);
    SIMD_F32 vout = SIMD_SUB_F32(vin0_opt, vin1);
    SIMD_ST_F32(out + index, vout);
  }
  return index;
}

static inline int ElementOptSubNum1SSE(int index, const float *in0, const float *in1, float *out,
                                                      int size) {
  SIMD_F32 vin1_opt_ = SIMD_MOV_F32(in1[0]);
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_F32 vin0 = SIMD_LD_F32(in0 + index);
    SIMD_F32 vout = SIMD_SUB_F32(vin0, vin1_opt_);
    SIMD_ST_F32(out + index, vout);
  }
  return index;
}


static inline int ElementOptSubExtNum0SSE(int index, const float *in0, const float *in1, const float alpha, float *out,
                                                      int size) {
  SIMD_F32 vin0_opt = SIMD_MOV_F32(in0[0]);
  SIMD_F32 valpha = SIMD_MOV_F32(alpha);
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
        SIMD_F32 vin1 = SIMD_LD_F32(in1 + index);
        SIMD_F32 vin1_alpha = SIMD_MUL_F32(vin1, valpha);
        SIMD_F32 vout = SIMD_SUB_F32(vin0_opt, vin1_alpha);
    SIMD_ST_F32(out + index, vout);
  }
  return index;
}

static inline int ElementOptSubExtNum1SSE(int index, const float *in0, const float *in1, const float alpha, float *out,
                                                      int size) {
  SIMD_F32 vin1_opt_ = SIMD_MOV_F32(in1[0]);
  SIMD_F32 valpha = SIMD_MOV_F32(alpha);
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
SIMD_F32 vin0 = SIMD_LD_F32(in0 + index);
SIMD_F32 vin1_alpha = SIMD_MUL_F32(vin1_opt_, valpha);
SIMD_F32 vout = SIMD_SUB_F32(vin0, vin1_alpha);
    SIMD_ST_F32(out + index, vout);
  }
  return index;
}

static inline int ElementOptSubIntNum0SSE(int index, const int32_t *in0, const int32_t *in1, int32_t *out, int size) {
  SIMD_EPI32 vin0_opt = SIMD_MOV_EPI32(in0[0]);
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_EPI32 vin1 = SIMD_LD_EPI32(in1 + index);
    SIMD_EPI32 vout = SIMD_SUB_EPI32(vin0_opt, vin1);
    SIMD_ST_EPI32(out + index, vout);
  }
  return index;
}

static inline int ElementOptSubIntNum1SSE(int index, const int32_t *in0, const int32_t *in1, int32_t *out, int size) {
  SIMD_EPI32 vin1_opt_ = SIMD_MOV_EPI32(in1[0]);
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_EPI32 vin0 = SIMD_LD_EPI32(in0 + index);
    SIMD_EPI32 vout = SIMD_SUB_EPI32(vin0, vin1_opt_);
    SIMD_ST_EPI32(out + index, vout);
  }
  return index;
}

static inline int ElementOptSubReluNum0SSE(int index, const float *in0, const float *in1, float *out,
                                                          int size) {
  SIMD_F32 vin0_opt = SIMD_MOV_F32(in0[0]);
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_F32 vin1 = SIMD_LD_F32(in1 + index);
    SIMD_F32 vout = SIMD_MAX_N_F32(SIMD_SUB_F32(vin0_opt, vin1), 0.0f);
    SIMD_ST_F32(out + index, vout);
  }
  return index;
}

static inline int ElementOptSubReluNum1SSE(int index, const float *in0, const float *in1, float *out,
                                                          int size) {
  SIMD_F32 vin1_opt_ = SIMD_MOV_F32(in1[0]);
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_F32 vin0 = SIMD_LD_F32(in0 + index);
    SIMD_F32 vout = SIMD_MAX_N_F32(SIMD_SUB_F32(vin0, vin1_opt_), 0.0f);
    SIMD_ST_F32(out + index, vout);
  }
  return index;
}

static inline int ElementOptSubRelu6Num0SSE(int index, const float *in0, const float *in1, float *out,
                                                           int size) {
  SIMD_F32 vin0_opt = SIMD_MOV_F32(in0[0]);
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_F32 vin1 = SIMD_LD_F32(in1 + index);
    SIMD_F32 vout = SIMD_MIN_N_F32(SIMD_MAX_N_F32(SIMD_SUB_F32(vin0_opt, vin1), 0.0f), 6.0f);
    SIMD_ST_F32(out + index, vout);
  }
  return index;
}

static inline int ElementOptSubRelu6Num1SSE(int index, const float *in0, const float *in1, float *out,
                                                           int size) {
  SIMD_F32 vin1_opt_ = SIMD_MOV_F32(in1[0]);
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_F32 vin0 = SIMD_LD_F32(in0 + index);
    SIMD_F32 vout = SIMD_MIN_N_F32(SIMD_MAX_N_F32(SIMD_SUB_F32(vin0, vin1_opt_), 0.0f), 6.0f);
    SIMD_ST_F32(out + index, vout);
  }
  return index;
}

static inline int ElementSubSSE(int index, const float *in0, const float *in1, float *out, int size) {
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_F32 vin0 = SIMD_LD_F32(in0 + index);
    SIMD_F32 vin1 = SIMD_LD_F32(in1 + index);
    SIMD_F32 vout = SIMD_SUB_F32(vin0, vin1);
    SIMD_ST_F32(out + index, vout);
  }
  return index;
}

static inline int ElementSubExtSSE(int index, const float *in0, const float *in1, const float alpha, float *out, int size) {
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_F32 vin0 = SIMD_LD_F32(in0 + index);
    SIMD_F32 vin1 = SIMD_LD_F32(in1 + index);
    SIMD_F32 valpha = SIMD_MOV_F32(alpha);
    SIMD_F32 vin1_alpha = SIMD_MUL_F32(vin1, valpha);
    SIMD_F32 vout = SIMD_SUB_F32(vin0, vin1_alpha);
    SIMD_ST_F32(out + index, vout);
  }
  return index;
}

static inline int ElementSubIntSSE(int index, const int32_t *in0, const int32_t *in1, int32_t *out, int size) {
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_EPI32 vin0 = SIMD_LD_EPI32(in0 + index);
    SIMD_EPI32 vin1 = SIMD_LD_EPI32(in1 + index);
    SIMD_EPI32 vout = SIMD_SUB_EPI32(vin0, vin1);
    SIMD_ST_EPI32(out + index, vout);
  }
  return index;
}

static inline int ElementSubReluSSE(int index, const float *in0, const float *in1, float *out,
                                                   int size) {
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_F32 vin0 = SIMD_LD_F32(in0 + index);
    SIMD_F32 vin1 = SIMD_LD_F32(in1 + index);
    SIMD_F32 vout = SIMD_MAX_N_F32(SIMD_SUB_F32(vin0, vin1), 0.0f);
    SIMD_ST_F32(out + index, vout);
  }
  return index;
}

static inline int ElementSubRelu6SSE(int index, const float *in0, const float *in1, float *out,
                                                    int size) {
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_F32 vin0 = SIMD_LD_F32(in0 + index);
    SIMD_F32 vin1 = SIMD_LD_F32(in1 + index);
    SIMD_F32 vout = SIMD_MIN_N_F32(SIMD_MAX_N_F32(SIMD_SUB_F32(vin0, vin1), 0.0f), 6.0f);
    SIMD_ST_F32(out + index, vout);
  }
  return index;
}

#undef MS_SIMD_INSTRUCTION
#undef BLOCK_NUM
#pragma GCC pop_options
#undef MS_SIMD_SSE
#ifdef __cplusplus
};
#endif
#endif
