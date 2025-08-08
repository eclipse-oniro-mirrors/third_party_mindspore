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
#ifndef MINDSPORE_NNACL_FP32_DROPOUTFP32_AVX_H_
#define MINDSPORE_NNACL_FP32_DROPOUTFP32_AVX_H_

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

static inline int DropoutFp32AVX(int index, const float *input, float scale,
    int length, float *output) {
    SIMD_F32 scale_value = SIMD_MOV_F32(scale);
    for (int block_max_size = length - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
        SIMD_ST_F32(output + index, SIMD_MUL_F32(SIMD_LD_F32(input + index), scale_value));
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
