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
#ifndef NNACL_DROPOUT_FP32_SIMD_H_
#define NNACL_DROPOUT_FP32_SIMD_H_

#include "nnacl/intrinsics/ms_simd_instructions.h"
#ifdef ENABLE_AVX512
#include "nnacl/avx512/dropout_fp32_avx512.h"
#endif

#ifdef ENABLE_AVX
#include "nnacl/avx/dropout_fp32_avx.h"
#endif

#ifdef ENABLE_SSE
#include "nnacl/sse/dropout_fp32_sse.h"
#endif

#ifdef ENABLE_ARM
#include "nnacl/neon/dropout_fp32_neon.h"
#endif

#endif
