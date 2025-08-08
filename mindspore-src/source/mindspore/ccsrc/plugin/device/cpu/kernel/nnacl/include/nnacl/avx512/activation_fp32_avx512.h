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
#ifndef MINDSPORE_NNACL_FP32_ACTIVATION_AVX512_H_
#define MINDSPORE_NNACL_FP32_ACTIVATION_AVX512_H_

#include "nnacl/intrinsics/ms_simd_instructions.h"
#include "nnacl/intrinsics/ms_simd_avx512_instructions.h"

#ifdef __cplusplus
extern "C" {
#endif
#pragma GCC push_options
#pragma GCC target("avx512f")
#define MS_SIMD_INSTRUCTION MS_SIMD_AVX512_INSTRUCTION
#define BLOCK_NUM 16
#define MS_SIMD_AVX512

static inline int Fp32ReluAVX512(int index, const float *src, int length, float *dst) {
    SIMD_F32 zero = SIMD_SET0_F32;
    for (int block_max_size = length - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
        SIMD_ST_F32(dst + index, SIMD_MAX_F32(SIMD_LD_F32(src + index), zero));
    }
    return index;
}

static inline int Int32ReluAVX512(int index, const int32_t *src, int length, int32_t *dst) {
    SIMD_EPI32 zero = SIMD_MOV_EPI32(0.0f);
    for (int block_max_size = length - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
        SIMD_ST_EPI32(dst + index, SIMD_MAX_EPI32(SIMD_LD_EPI32(src + index), zero));
    }
    return index;
}

static inline int Fp32Relu6AVX512(int index, const float *src, int length, float *dst) {
    SIMD_F32 zero = SIMD_SET0_F32;
    SIMD_F32 six = SIMD_MOV_F32(6.0f);
    for (int block_max_size = length - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
        SIMD_ST_F32(dst + index, SIMD_CLAMP_F32(SIMD_LD_F32(src + index), zero, six));
    }
    return index;
}

static inline int Fp32ClipAVX512(int index, const float *src, int length, float *dst, float min, float max) {
    SIMD_F32 min_val = SIMD_MOV_F32(min);
    SIMD_F32 max_val = SIMD_MOV_F32(max);
    for (int block_max_size = length - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
        SIMD_ST_F32(dst + index, SIMD_CLAMP_F32(SIMD_LD_F32(src + index), min_val, max_val));
    }
    return index;
}

static inline int Int32ClipAVX512(int index, const int32_t *src, int length, int32_t *dst, int min, int max) {
    SIMD_EPI32 min_val = SIMD_MOV_EPI32(min);
    SIMD_EPI32 max_val = SIMD_MOV_EPI32(max);
    for (int block_max_size = length - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
        SIMD_ST_EPI32(dst + index, SIMD_CLAMP_EPI32(SIMD_LD_EPI32(src + index), min_val, max_val));
    }
    return index;
}

static inline int LReluAVX512(int index, const float *src, int length, float *dst, float alpha) {
    SIMD_F32 alpha_data = SIMD_MOV_F32(alpha);
    for (int block_max_size = length - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
        SIMD_F32 src_tmp = SIMD_LD_F32(src + index);
        SIMD_MASK mask = SIMD_CMPGT_F32(SIMD_SET0_F32, src_tmp);
        SIMD_ST_F32(dst + index, SIMD_BLEND_F32(src_tmp, SIMD_MUL_F32(src_tmp, alpha_data), mask));
    }
    return index;
}

static inline int SigmoidAVX512(int index, const float *src, int length, float *dst) {
    for (int block_max_size = length - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
        SIMD_EXP_ST_F32(SIMD_SUB_F32(SIMD_SET0_F32, (SIMD_LD_F32(src + index))), dst + index);
        SIMD_ST_F32(dst + index,
                    SIMD_DIV_F32(SIMD_MOV_F32(1.0f), SIMD_ADD_F32(SIMD_MOV_F32(1.0f), SIMD_LD_F32(dst + index))));
    }
    return index;
}

static inline int SoftplusAVX512(int index, const float *src, int length, float *dst) {
    SIMD_F32 log_max = SIMD_MOV_F32(88.0);
    for (int block_max_size = length - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
        SIMD_F32 src_tmp = SIMD_LD_F32(src + index);
        SIMD_F32 dst_tmp = SIMD_EXP_F32(src_tmp);
        dst_tmp = SIMD_LOG_F32(SIMD_ADD_F32(SIMD_MOV_F32(1.0f), dst_tmp));
        SIMD_ST_F32(dst + index, SIMD_BLEND_F32(dst_tmp, src_tmp, SIMD_CMPGT_F32(src_tmp, log_max)));
    }
    return index;
}

static inline int TanhAVX512(int index, const float *src, int length, float *dst) {
    for (int block_max_size = length - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
        SIMD_F32 input = SIMD_LD_F32(src + index);
        SIMD_ST_F32(dst + index, SIMD_TANH_F32(input));
    }
    return index;
}

static inline int SwishAVX512(int index, const float *src, int length, float *dst) {
    for (int block_max_size = length - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
        SIMD_F32 src_value = SIMD_LD_F32(src + index);
        SIMD_EXP_ST_F32(SIMD_SUB_F32(SIMD_SET0_F32, src_value), dst + index);
        SIMD_ST_F32(dst + index,
                    SIMD_DIV_F32(src_value, SIMD_ADD_F32(SIMD_MOV_F32(1.0f), SIMD_LD_F32(dst + index))));
    }
    return index;
}

static inline int HSwishAVX512(int index, const float *src, int length, float *dst) {
    for (int block_max_size = length - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
        SIMD_F32 src_value = SIMD_LD_F32(src + index);
        SIMD_F32 relu6 = SIMD_CLAMP_N_F32(SIMD_ADD_N_F32(src_value, 3), 0, 6);
        SIMD_ST_F32(dst + index, SIMD_DIV_N_F32(SIMD_MUL_F32(src_value, relu6), 6));
    }
    return index;
}

static inline int HSigmoidAVX512(int index, const float *src, int length, float *dst) {
    for (int block_max_size = length - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
        SIMD_F32 src_value = SIMD_LD_F32(src + index);
        SIMD_F32 relu6 = SIMD_CLAMP_N_F32(SIMD_ADD_N_F32(src_value, 3), 0, 6);
        SIMD_ST_F32(dst + index, SIMD_DIV_N_F32(relu6, 6));
    }
    return index;
}

static inline int HardTanhNoLimitMinAVX512(int index, const float *src, int length, float *dst, float min_val,
                                            float max_val) {
    for (int block_max_size = length - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
        SIMD_ST_F32(dst + index, SIMD_MIN_N_F32(SIMD_LD_F32(src + index), max_val));
    }
    return index;
}

static inline int HardTanhNoLimitMaxAVX512(int index, const float *src, int length, float *dst, float min_val,
                                            float max_val) {
    for (int block_max_size = length - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
        SIMD_ST_F32(dst + index, SIMD_MAX_N_F32(SIMD_LD_F32(src + index), min_val));
    }
    return index;
}

static inline int HardTanhLimitMinMaxAVX512(int index, const float *src, int length, float *dst, float min_val,
                                             float max_val) {
    for (int block_max_size = length - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
        SIMD_ST_F32(dst + index, SIMD_CLAMP_N_F32(SIMD_LD_F32(src + index), min_val, max_val));
    }
    return index;
}

static inline int GeluTanhApproximateAVX512(int index, const float *src, int length, float *dst) {
    for (int block_max_size = length - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
        SIMD_F32 in = SIMD_LD_F32(src + index);
        SIMD_F32 tmp1 = SIMD_FMADD_F32(SIMD_MUL_N_F32(in, 0.035677408136f), in, SIMD_MOV_F32(0.79788456080287f));
        SIMD_F32 tmp2 = SIMD_MUL_F32(tmp1, in);
        SIMD_ST_F32(dst + index, SIMD_MUL_F32(SIMD_MUL_N_F32(in, 0.5f), SIMD_ADD_N_F32(SIMD_TANH_F32(tmp2), 1.0f)));
    }
    return index;
}

static inline int GeluAVX512(int index, const float *src, int length, float *dst) {
    SIMD_F32 para1 = SIMD_MOV_F32(1.4142135623730951f);
    SIMD_F32 para2 = SIMD_MOV_F32(1.0f);
    SIMD_F32 para3 = SIMD_MOV_F32(0.5f);
    for (int block_max_size = length - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
      SIMD_F32 in = SIMD_LD_F32(src + index);
      SIMD_F32 res = SIMD_MUL_F32(SIMD_MUL_F32(para3, in), SIMD_ADD_F32(para2, SIMD_ERF_F32(SIMD_DIV_F32(in, para1))));
      SIMD_ST_F32(dst + index, res);
    }
    return index;
}

static inline SIMD_F32 SIMD_ERFCCHEBAVX512(SIMD_F32 src) {
  static const int ncof = 7;
  const double cof[7] = {-1.3026537197817094,  6.4196979235649026e-1, 1.9476473204185836e-2, -9.561514786808631e-3,
                         -9.46595344482036e-4, 3.66839497852761e-4,   4.2523324806907e-5};
  SIMD_F32 dst;
  SIMD_F32 d = SIMD_SET0_F32;
  SIMD_F32 dd = SIMD_SET0_F32;
  SIMD_F32 t = SIMD_DIV_F32(SIMD_MOV_F32(2.0f), SIMD_ADD_F32(src, SIMD_MOV_F32(2.0f)));
  SIMD_F32 ty = SIMD_SUB_F32(SIMD_MUL_F32(SIMD_MOV_F32(4.0f), t), SIMD_MOV_F32(2.0f));

  for (int j = ncof - 1; j > 0; j--) {
    SIMD_F32 tmp = d;
    d = SIMD_SUB_F32(SIMD_FMADD_F32(ty, d, SIMD_MOV_F32(cof[j])), dd);
    dd = tmp;
  }

  dst =
    SIMD_FMADD_F32(src, src, MS_FSMUL_F32(dd, SIMD_FMADD_F32(ty, d, SIMD_MOV_F32(cof[0])), SIMD_MOV_F32(0.5f)));
  dst = SIMD_MUL_F32(t, SIMD_EXP_F32(SIMD_MUL_F32(SIMD_MOV_F32(-1.0f), dst)));
  return dst;
}

static inline SIMD_F32 SIMD_ERF_APPROXIMATEAVX512(SIMD_F32 src) {
  SIMD_F32 abs_src = SIMD_ABS_F32(src);
  SIMD_F32 sign = SIMD_GETSIGN_F32(src);
  SIMD_F32 dst = SIMD_ERFCCHEBAVX512(abs_src);
  return SIMD_MUL_F32(sign, SIMD_SUB_F32(SIMD_MOV_F32(1.0f), dst));
}

static inline int GeluErfAPPROXIMATEAVX512(int index, const float *src, int length, float *dst) {
    SIMD_F32 para1 = SIMD_MOV_F32(1.4142135623730951f);
    SIMD_F32 para2 = SIMD_MOV_F32(1.0f);
    SIMD_F32 para3 = SIMD_MOV_F32(0.5f);
    for (int block_max_size = length - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
      SIMD_F32 in = SIMD_LD_F32(src + index);
      SIMD_F32 res = SIMD_MUL_F32(SIMD_MUL_F32(para3, in), SIMD_ADD_F32(para2, SIMD_ERF_APPROXIMATEAVX512(SIMD_DIV_F32(in, para1))));
      SIMD_ST_F32(dst + index, res);
    }
    return index;
}

static inline int EluAVX512(int index, const float *src, int length, float *dst, float alpha) {
    for (int block_max_size = length - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
        SIMD_F32 src_tmp = SIMD_LD_F32(src + index);
        SIMD_F32 exp_tmp = SIMD_SUB_N_F32(SIMD_EXP_F32(src_tmp), 1.0f);
        SIMD_MASK mask = SIMD_CMPLE_F32(src_tmp, SIMD_SET0_F32);
        SIMD_ST_F32(dst + index, SIMD_BLEND_F32(src_tmp, SIMD_MUL_N_F32(exp_tmp, alpha), mask));
    }
    return index;
}

static inline int CeluAVX512(int index, const float *src, int length, float *dst, float alpha) {
    for (int block_max_size = length - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
        SIMD_F32 src_tmp = SIMD_LD_F32(src + index);
        SIMD_F32 exp_tmp = SIMD_SUB_N_F32(SIMD_EXP_F32(SIMD_DIV_N_F32(src_tmp, alpha)), 1.0f);
        SIMD_MASK mask = SIMD_CMPLE_F32(src_tmp, SIMD_SET0_F32);
        SIMD_ST_F32(dst + index, SIMD_BLEND_F32(src_tmp, SIMD_MUL_N_F32(exp_tmp, alpha), mask));
    }
    return index;
}

static inline int HardShrinkAVX512(int index, const float *src, int length, float *dst, float lambd) {
    SIMD_F32 pos_lamdb_v = SIMD_MOV_F32(lambd);
    SIMD_F32 neg_lamdb_v = SIMD_MOV_F32(-lambd);

    for (int block_max_size = length - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
        SIMD_F32 src_t = SIMD_LD_F32(src + index);
        /* v0 = (in > lamdb) & in */
        SIMD_F32 value0 = SIMD_AND_MASK_F32(SIMD_CMPGT_F32(src_t, pos_lamdb_v), src_t);
        /* v1 = (in < -lamdb) & in */
        SIMD_F32 value1 = SIMD_AND_MASK_F32(SIMD_CMPLT_F32(src_t, neg_lamdb_v), src_t);
        /* out = (v0 | v1) */
        SIMD_ST_F32(dst + index, SIMD_OR_F32(value0, value1));
    }
    return index;
}

static inline int SoftShrinkAVX512(int index, const float *src, int length, float *dst, float lambd) {
    SIMD_F32 pos_lamdb_v = SIMD_MOV_F32(lambd);
    SIMD_F32 neg_lamdb_v = SIMD_MOV_F32(-lambd);

    for (int block_max_size = length - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
        SIMD_F32 src_t = SIMD_LD_F32(src + index);
        /* v0 = (in > lamdb) & (in - lamdb) */
        SIMD_F32 value0 = SIMD_AND_MASK_F32(SIMD_CMPGT_F32(src_t, pos_lamdb_v), SIMD_SUB_F32(src_t, pos_lamdb_v));
        /* v1 = (in < -lamdb) & (in + lamdb) */
        SIMD_F32 value1 = SIMD_AND_MASK_F32(SIMD_CMPLT_F32(src_t, neg_lamdb_v), SIMD_ADD_F32(src_t, pos_lamdb_v));
        /* out = (v0 | v1) */
        SIMD_ST_F32(dst + index, SIMD_OR_F32(value0, value1));
    }
    return index;
}

static inline int SoftsignFp32OptAVX512(int index, const float *src, int length, float *dst) {
    for (int block_max_size = length - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
        SIMD_F32 src_tmp = SIMD_LD_F32(src + index);
        SIMD_F32 divisor_tmp = SIMD_ADD_F32(SIMD_MOV_F32(1.0f), SIMD_ABS_F32(src_tmp));
        SIMD_ST_F32(dst + index, SIMD_DIV_F32(src_tmp, divisor_tmp));
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
