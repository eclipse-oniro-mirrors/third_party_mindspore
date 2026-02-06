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
#ifndef MINDSPORE_LITE_SRC_LITERT_C_API_TYPE_C_PRIVATE_H_
#define MINDSPORE_LITE_SRC_LITERT_C_API_TYPE_C_PRIVATE_H_

#include <string>
#include <vector>
#include <memory>
#include <stddef.h>
#include "include/c_api/types_c.h"

#ifdef __cplusplus
extern "C" {
#endif

#define NNRT_DEVICE_NAME_MAX (128)

struct NNRTDeviceDesc {
  size_t device_id;
  OH_AI_NNRTDeviceType device_type;
  char device_name[NNRT_DEVICE_NAME_MAX];
};

#ifdef __cplusplus
}

void CleanAllocatorTable();

#endif
#endif  // MINDSPORE_LITE_SRC_LITERT_C_API_TYPE_C_PRIVATE_H_
