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
// clang-format off
#ifndef MINDSPORE_NNACL_FP32_REDUCE_FP32_SSE_H_
#define MINDSPORE_NNACL_FP32_REDUCE_FP32_SSE_H_

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

static inline int64_t ReduceSumSSE(int64_t index, const float *outer_src, float *outer_dst, int inner_size,
  int axis_size) {
  for (int block_max_size = inner_size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    const float *inner_src = outer_src + index;
    SIMD_F32 tmp = SIMD_MOV_F32(0);
    for (int i = 0; i < axis_size; i++) {
      tmp = SIMD_ADD_F32(tmp, SIMD_LD_F32(inner_src + i * inner_size));
    }
    SIMD_ST_F32(outer_dst + index, tmp);
  }
  return index;
}

static inline int64_t ReduceSumByLastAxisSSE(int64_t index, const float *src, float* tmp_sum, int axis_size) {
  SIMD_F32 tmp = SIMD_MOV_F32(0);
  for (int block_max_size = axis_size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    tmp = SIMD_ADD_F32(tmp, SIMD_LD_F32(src + index));
  }
  *tmp_sum += SIMD_GET_SUM_F32(tmp);
  return index;
}

static inline int64_t ReduceMeanSSE(int64_t index, const float *outer_src, float *outer_dst, int inner_size,
  int axis_size) {
  for (int block_max_size = inner_size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    const float *inner_src = outer_src + index;
    SIMD_F32 tmp = SIMD_MOV_F32(0);
    for (int i = 0; i < axis_size; i++) {
      tmp = SIMD_ADD_F32(tmp, SIMD_LD_F32(inner_src + i * inner_size));
    }
    SIMD_ST_F32(outer_dst + index, SIMD_DIV_N_F32(tmp, axis_size));
  }
  return index;
}

static inline int64_t ReduceMinSSE(int64_t index, const float *outer_src, float *outer_dst, int inner_size,
  int axis_size) {
  for (int block_max_size = inner_size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    const float *inner_src = outer_src + index;
    SIMD_F32 tmp = SIMD_MOV_F32(FLT_MAX);
    for (int i = 0; i < axis_size; i++) {
      tmp = SIMD_MIN_F32(tmp, SIMD_LD_F32(inner_src + i * inner_size));
    }
    SIMD_ST_F32(outer_dst + index, tmp);
  }
  return index;
}

static inline int64_t ReduceMaxSSE(int64_t index, const float *outer_src, float *outer_dst, int inner_size,
  int axis_size) {
  for (int block_max_size = inner_size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    const float *inner_src = outer_src + index;
    SIMD_F32 tmp = SIMD_MOV_F32(FLT_MIN);
    for (int i = 0; i < axis_size; i++) {
      tmp = SIMD_MAX_F32(tmp, SIMD_LD_F32(inner_src + i * inner_size));
    }
    SIMD_ST_F32(outer_dst + index, tmp);
  }
  return index;
}

static inline int64_t ReduceMaxByLastAxisSSE(int64_t index, const float *src, float* tmp_max, int axis_size) {
  SIMD_F32 tmp = SIMD_MOV_F32(*tmp_max);
  for (int block_max_size = axis_size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    tmp = SIMD_MAX_F32(tmp, SIMD_LD_F32(src + index));
  }
  *tmp_max = SIMD_GET_MAX_F32(tmp);
  return index;
}

static inline int64_t ReduceProdSSE(int64_t index, const float *outer_src, float *outer_dst, int inner_size,
  int axis_size) {
  for (int block_max_size = inner_size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    const float *inner_src = outer_src + index;
    SIMD_F32 tmp = SIMD_MOV_F32(1.0f);
    for (int i = 0; i < axis_size; i++) {
      tmp = SIMD_MUL_F32(tmp, SIMD_LD_F32(inner_src + i * inner_size));
    }
    SIMD_ST_F32(outer_dst + index, tmp);
  }
  return index;
}

static inline int64_t ReduceSumSquareSSE(int64_t index, const float *outer_src, float *outer_dst, int inner_size,
  int axis_size) {
  for (int block_max_size = inner_size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    const float *inner_src = outer_src + index;
    SIMD_F32 tmp = SIMD_MOV_F32(0);
    for (int i = 0; i < axis_size; i++) {
      tmp = SIMD_ADD_F32(tmp, SIMD_MUL_SQUARE_F32(SIMD_LD_F32(inner_src + i * inner_size)));
    }
    SIMD_ST_F32(outer_dst + index, tmp);
  }
  return index;
}

static inline int64_t ReduceL2NormSSE(int64_t index, const float *outer_src, float *outer_dst, int inner_size,
  int axis_size) {
  for (int block_max_size = inner_size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    const float *inner_src = outer_src + index;
    SIMD_F32 tmp = SIMD_MOV_F32(0);
    for (int i = 0; i < axis_size; i++) {
      tmp = SIMD_ADD_F32(tmp, SIMD_MUL_SQUARE_F32(SIMD_LD_F32(inner_src + i * inner_size)));
    }
    SIMD_ST_F32(outer_dst + index, SIMD_SQRT_F32(tmp));
  }
  return index;
}

static inline int64_t IntReduceSumSSE(int64_t index, const int32_t *outer_src, int32_t *outer_dst, int inner_size,
  int axis_size) {
  for (int block_max_size = inner_size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    const int32_t *inner_src = outer_src + index;
    SIMD_EPI32 tmp = SIMD_MOV_EPI32(0);
    for (int i = 0; i < axis_size; i++) {
      tmp = SIMD_ADD_EPI32(tmp, SIMD_LD_EPI32(inner_src + i * inner_size));
    }
    SIMD_ST_EPI32(outer_dst + index, tmp);
  }
  return index;
}

static inline int64_t IntReduceMeanSSE(int64_t index, const int32_t *outer_src, int32_t *outer_dst, int inner_size,
  int axis_size) {
  for (int block_max_size = inner_size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    const int32_t *inner_src = outer_src + index;
    SIMD_EPI32 tmp = SIMD_MOV_EPI32(0);
    for (int i = 0; i < axis_size; i++) {
      tmp = SIMD_ADD_EPI32(tmp, SIMD_LD_EPI32(inner_src + i * inner_size));
    }
    SIMD_ST_EPI32(outer_dst + index, SIMD_DIV_N_EPI32(tmp, axis_size));
  }
  return index;
}

static inline int64_t IntReduceMinSSE(int64_t index, const int32_t *outer_src, int32_t *outer_dst, int inner_size,
  int axis_size) {
  for (int block_max_size = inner_size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    const int32_t *inner_src = outer_src + index;
    SIMD_EPI32 tmp = SIMD_MOV_EPI32(INT32_MAX);
    for (int i = 0; i < axis_size; i++) {
      tmp = SIMD_MIN_EPI32(tmp, SIMD_LD_EPI32(inner_src + i * inner_size));
    }
    SIMD_ST_EPI32(outer_dst + index, tmp);
  }
  return index;
}

static inline int64_t IntReduceMaxSSE(int64_t index, const int32_t *outer_src, int32_t *outer_dst, int inner_size,
  int axis_size) {
  for (int block_max_size = inner_size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    const int32_t *inner_src = outer_src + index;
    SIMD_EPI32 tmp = SIMD_MOV_EPI32(INT32_MIN);
    for (int i = 0; i < axis_size; i++) {
      tmp = SIMD_MAX_EPI32(tmp, SIMD_LD_EPI32(inner_src + i * inner_size));
    }
    SIMD_ST_EPI32(outer_dst + index, tmp);
  }
  return index;
}

static inline int64_t  ReduceSumDim2Axis0SSE(int64_t index, size_t col_size, size_t col_len, size_t row_len, const float *src_data, float *dst_data) {
  for (int block_max_size = col_size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_F32 tmp = SIMD_MOV_F32(0);
    const float *inner_src = src_data + index;
    float *inner_dst = dst_data + index;
    for (size_t i = 0; i < row_len; ++i) {
      tmp = SIMD_ADD_F32(tmp, SIMD_LD_F32(inner_src + i * col_len));
    }
    SIMD_ST_F32(inner_dst, tmp);
  }
  return index;
}

static inline int64_t FloatReduceDeviationSSE(int64_t index, const float *src_data, float mean, size_t size, float *deviation) {
  SIMD_F32 fs_deviation = SIMD_MOV_F32(0);
  SIMD_F32 fs_mean = SIMD_MOV_F32(mean);
  for (int block_max_size = size - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
    SIMD_F32 fs_sub = SIMD_LD_F32(src_data + index);

    fs_sub = SIMD_SUB_F32(fs_sub, fs_mean);
    SIMD_F32 fs_pow = SIMD_MUL_F32(fs_sub, fs_sub);
    fs_deviation = SIMD_ADD_F32(fs_deviation, fs_pow);
  }
  *deviation += SIMD_GET_SUM_F32(fs_deviation);
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
