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
#ifndef MINDSPORE_NNACL_FP32_DROPOUTFP32_NEON_H_
#define MINDSPORE_NNACL_FP32_DROPOUTFP32_NEON_H_

#include "nnacl_c/intrinsics/ms_simd_instructions.h"
#include "nnacl_c/intrinsics/ms_simd_neon_instructions.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MS_SIMD_INSTRUCTION MS_SIMD_NEON_INSTRUCTION
#define BLOCK_NUM 4
#define MS_SIMD_NEON

static inline int DropoutFp32NEON(int index, const float *input, float scale,
    int length, float *output) {
    SIMD_F32 scale_value = SIMD_MOV_F32(scale);
    for (int block_max_size = length - BLOCK_NUM + 1; index < block_max_size; index += BLOCK_NUM) {
        SIMD_ST_F32(output + index, SIMD_MUL_F32(SIMD_LD_F32(input + index), scale_value));
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
