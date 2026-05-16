/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_NNACL_ARITHMETIC_AVX512_H_
#define MINDSPORE_NNACL_ARITHMETIC_AVX512_H_

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

#ifndef MS_SIMD_NEON
static inline int ElementFloorModAVX512(int index, const float *in0, const float *in1, float *out, int size) {
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_F32 in0_tmp = SIMD_LD_F32(in0 + index);
    SIMD_F32 in1_tmp = SIMD_LD_F32(in1 + index);
    SIMD_F32 floor_tmp = SIMD_FLOOR_F32(SIMD_DIV_F32(in0_tmp, in1_tmp));
    SIMD_F32 out_tmp = SIMD_SUB_F32(in0_tmp, SIMD_MUL_F32(floor_tmp, in1_tmp));
    SIMD_ST_F32(out + index, out_tmp);
  }
  return index;
}

static inline int ElementOptFloorModNum0AVX512(int index, const float *in0, const float *in1, float *out, int size) {
  SIMD_F32 in0_tmp = SIMD_MOV_F32(in0[0]);
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_F32 in1_tmp = SIMD_LD_F32(in1 + index);
    SIMD_F32 floor_tmp = SIMD_FLOOR_F32(SIMD_DIV_F32(in0_tmp, in1_tmp));
    SIMD_F32 out_tmp = SIMD_SUB_F32(in0_tmp, SIMD_MUL_F32(floor_tmp, in1_tmp));
    SIMD_ST_F32(out + index, out_tmp);
  }
  return index;
}

static inline int ElementOptFloorModNum1AVX512(int index, const float *in0, const float *in1, float *out, int size) {
  SIMD_F32 in1_tmp = SIMD_MOV_F32(in1[0]);
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_F32 in0_tmp = SIMD_LD_F32(in0 + index);
    SIMD_F32 floor_tmp = SIMD_FLOOR_F32(SIMD_DIV_F32(in0_tmp, in1_tmp));
    SIMD_F32 out_tmp = SIMD_SUB_F32(in0_tmp, SIMD_MUL_F32(floor_tmp, in1_tmp));
    SIMD_ST_F32(out + index, out_tmp);
  }
  return index;
}

static inline int ElementFloorDivAVX512(int index, const float *in0, const float *in1, float *out, int size) {
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_F32 in0_tmp = SIMD_LD_F32(in0 + index);
    SIMD_F32 in1_tmp = SIMD_LD_F32(in1 + index);
    SIMD_F32 floor_tmp = SIMD_FLOOR_F32(SIMD_DIV_F32(in0_tmp, in1_tmp));
    SIMD_ST_F32(out + index, floor_tmp);
  }
  return index;
}

static inline int ElementOptFloorDivNum0AVX512(int index, const float *in0, const float *in1, float *out, int size) {
  SIMD_F32 in0_tmp = SIMD_MOV_F32(in0[0]);
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_F32 in1_tmp = SIMD_LD_F32(in1 + index);
    SIMD_F32 out_tmp = SIMD_FLOOR_F32(SIMD_DIV_F32(in0_tmp, in1_tmp));
    SIMD_ST_F32(out + index, out_tmp);
  }
  return index;
}

static inline int ElementOptFloorDivNum1AVX512(int index, const float *in0, const float *in1, float *out, int size) {
  SIMD_F32 in1_tmp = SIMD_MOV_F32(in1[0]);
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_F32 in0_tmp = SIMD_LD_F32(in0 + index);
    SIMD_F32 out_tmp = SIMD_FLOOR_F32(SIMD_DIV_F32(in0_tmp, in1_tmp));
    SIMD_ST_F32(out + index, out_tmp);
  }
  return index;
}
#endif

static inline int ElementFloorDivIntAVX512(int index, const int32_t *in0, const int32_t *in1, int32_t *out, int size) {
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_EPI32 in0_tmp = SIMD_LD_EPI32(in0 + index);
    SIMD_EPI32 in1_tmp = SIMD_LD_EPI32(in1 + index);
    SIMD_EPI32 out_tmp = SIMD_DIV_EPI32(in0_tmp, in1_tmp);
    SIMD_ST_EPI32(out + index, out_tmp);
  }
  return index;
}

static inline int ElementOptFloorDivIntNum0AVX512(int index, const int32_t *in0, const int32_t *in1, int32_t *out, int size) {
  SIMD_EPI32 in0_tmp = SIMD_MOV_EPI32(in0[0]);
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_EPI32 in1_tmp = SIMD_LD_EPI32(in1 + index);
    SIMD_EPI32 out_tmp = SIMD_DIV_EPI32(in0_tmp, in1_tmp);
    SIMD_ST_EPI32(out + index, out_tmp);
  }
  return index;
}

static inline int ElementOptFloorDivIntNum1AVX512(int index, const int32_t *in0, const int32_t *in1, int32_t *out, int size) {
  SIMD_EPI32 in1_tmp = SIMD_MOV_EPI32(in1[0]);
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_EPI32 in0_tmp = SIMD_LD_EPI32(in0 + index);
    SIMD_EPI32 out_tmp = SIMD_DIV_EPI32(in0_tmp, in1_tmp);
    SIMD_ST_EPI32(out + index, out_tmp);
  }
  return index;
}

static inline int ElementMaximumAVX512(int index, const float *in0, const float *in1, float *out, int size) {
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_F32 in0_tmp = SIMD_LD_F32(in0 + index);
    SIMD_F32 in1_tmp = SIMD_LD_F32(in1 + index);
    SIMD_F32 out_tmp = SIMD_MAX_F32(in0_tmp, in1_tmp);
    SIMD_ST_F32(out + index, out_tmp);
  }
  return index;
}

static inline int ElementOptMaximumNum0AVX512(int index, const float *in0, const float *in1, float *out, int size) {
  SIMD_F32 in0_tmp = SIMD_MOV_F32(in0[0]);
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_F32 in1_tmp = SIMD_LD_F32(in1 + index);
    SIMD_F32 out_tmp = SIMD_MAX_F32(in0_tmp, in1_tmp);
    SIMD_ST_F32(out + index, out_tmp);
  }
  return index;
}

static inline int ElementOptMaximumNum1AVX512(int index, const float *in0, const float *in1, float *out, int size) {
  SIMD_F32 in1_tmp = SIMD_MOV_F32(in1[0]);
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_F32 in0_tmp = SIMD_LD_F32(in0 + index);
    SIMD_F32 out_tmp = SIMD_MAX_F32(in0_tmp, in1_tmp);
    SIMD_ST_F32(out + index, out_tmp);
  }
  return index;
}

static inline int ElementMaximumIntAVX512(int index, const int32_t *in0, const int32_t *in1, int32_t *out, int size) {
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_EPI32 in0_tmp = SIMD_LD_EPI32(in0 + index);
    SIMD_EPI32 in1_tmp = SIMD_LD_EPI32(in1 + index);
    SIMD_EPI32 out_tmp = SIMD_MAX_EPI32(in0_tmp, in1_tmp);
    SIMD_ST_EPI32(out + index, out_tmp);
  }
  return index;
}

static inline int ElementOptMaximumIntNum0AVX512(int index, const int32_t *in0, const int32_t *in1, int32_t *out, int size) {
  SIMD_EPI32 in0_tmp = SIMD_MOV_EPI32(in0[0]);
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_EPI32 in1_tmp = SIMD_LD_EPI32(in1 + index);
    SIMD_EPI32 out_tmp = SIMD_MAX_EPI32(in0_tmp, in1_tmp);
    SIMD_ST_EPI32(out + index, out_tmp);
  }
  return index;
}

static inline int ElementOptMaximumIntNum1AVX512(int index, const int32_t *in0, const int32_t *in1, int32_t *out, int size) {
  SIMD_EPI32 in1_tmp = SIMD_MOV_EPI32(in1[0]);
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_EPI32 in0_tmp = SIMD_LD_EPI32(in0 + index);
    SIMD_EPI32 out_tmp = SIMD_MAX_EPI32(in0_tmp, in1_tmp);
    SIMD_ST_EPI32(out + index, out_tmp);
  }
  return index;
}

static inline int ElementMinimumIntAVX512(int index, const int32_t *in0, const int32_t *in1, int32_t *out, int size) {
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_EPI32 in0_tmp = SIMD_LD_EPI32(in0 + index);
    SIMD_EPI32 in1_tmp = SIMD_LD_EPI32(in1 + index);
    SIMD_EPI32 out_tmp = SIMD_MIN_EPI32(in0_tmp, in1_tmp);
    SIMD_ST_EPI32(out + index, out_tmp);
  }
  return index;
}

static inline int ElementOptMinimumIntNum0AVX512(int index, const int32_t *in0, const int32_t *in1, int32_t *out, int size) {
  SIMD_EPI32 in0_tmp = SIMD_MOV_EPI32(in0[0]);
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_EPI32 in1_tmp = SIMD_LD_EPI32(in1 + index);
    SIMD_EPI32 out_tmp = SIMD_MIN_EPI32(in0_tmp, in1_tmp);
    SIMD_ST_EPI32(out + index, out_tmp);
  }
  return index;
}

static inline int ElementOptMinimumIntNum1AVX512(int index, const int32_t *in0, const int32_t *in1, int32_t *out, int size) {
  SIMD_EPI32 in1_tmp = SIMD_MOV_EPI32(in1[0]);
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_EPI32 in0_tmp = SIMD_LD_EPI32(in0 + index);
    SIMD_EPI32 out_tmp = SIMD_MIN_EPI32(in0_tmp, in1_tmp);
    SIMD_ST_EPI32(out + index, out_tmp);
  }
  return index;
}

static inline int ElementMinimumAVX512(int index, const float *in0, const float *in1, float *out, int size) {
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_F32 in0_tmp = SIMD_LD_F32(in0 + index);
    SIMD_F32 in1_tmp = SIMD_LD_F32(in1 + index);
    SIMD_F32 out_tmp = SIMD_MIN_F32(in0_tmp, in1_tmp);
    SIMD_ST_F32(out + index, out_tmp);
  }
  return index;
}

static inline int ElementOptMinimumNum0AVX512(int index, const float *in0, const float *in1, float *out, int size) {
  SIMD_F32 in0_tmp = SIMD_MOV_F32(in0[0]);
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_F32 in1_tmp = SIMD_LD_F32(in1 + index);
    SIMD_F32 out_tmp = SIMD_MIN_F32(in0_tmp, in1_tmp);
    SIMD_ST_F32(out + index, out_tmp);
  }
  return index;
}

static inline int ElementOptMinimumNum1AVX512(int index, const float *in0, const float *in1, float *out, int size) {
  SIMD_F32 in1_tmp = SIMD_MOV_F32(in1[0]);
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_F32 in0_tmp = SIMD_LD_F32(in0 + index);
    SIMD_F32 out_tmp = SIMD_MIN_F32(in0_tmp, in1_tmp);
    SIMD_ST_F32(out + index, out_tmp);
  }
  return index;
}

static inline size_t AssignSubOptAVX512(int index, float *in0, const float *in1, size_t size) {
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_F32 in0_tmp = SIMD_LD_F32(in0 + index);
    SIMD_F32 in1_tmp = SIMD_LD_F32(in1 + index);
    SIMD_F32 out_tmp = SIMD_SUB_F32(in0_tmp, in1_tmp);
    SIMD_ST_F32(in0 + index, out_tmp);
  }
  return index;
}

int ElementLogicalAndAVX512(int index, const float *in0, const float *in1, float *out, int size) {
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_F32 in0_tmp = SIMD_LD_F32(in0 + index);
    SIMD_F32 in1_tmp = SIMD_LD_F32(in1 + index);
    SIMD_F32 out_tmp = SIMD_AND_F32(SIMD_GETSIGN_F32(in0_tmp), SIMD_GETSIGN_F32(in1_tmp));
    SIMD_ST_F32(out + index, out_tmp);
  }
  return index;
}

int ElementOptLogicalAndAVX512(int index, const float *in0, const float *in1, float *out, int size, bool first_scalar) {
  if (first_scalar) {
    SIMD_F32 in0_tmp = SIMD_MOV_F32(*in0);
    for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
      SIMD_F32 in1_tmp = SIMD_LD_F32(in1 + index);
      SIMD_F32 out_tmp = SIMD_AND_F32(SIMD_GETSIGN_F32(in0_tmp), SIMD_GETSIGN_F32(in1_tmp));
      SIMD_ST_F32(out + index, out_tmp);
    }
  } else {
    SIMD_F32 in1_tmp = SIMD_MOV_F32(*in1);
    for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
      SIMD_F32 in0_tmp = SIMD_LD_F32(in0 + index);
      SIMD_F32 out_tmp = SIMD_AND_F32(SIMD_GETSIGN_F32(in0_tmp), SIMD_GETSIGN_F32(in1_tmp));
      SIMD_ST_F32(out + index, out_tmp);
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
