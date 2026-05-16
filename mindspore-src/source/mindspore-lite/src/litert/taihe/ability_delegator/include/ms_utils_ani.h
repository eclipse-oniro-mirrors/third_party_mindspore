/*
 * Copyright 2025 Huawei Technologies Co., Ltd
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef MS_UTILS_ANI
#define MS_UTILS_ANI
#include <sys/mman.h>
#include "taihe/runtime.hpp"
#include "include/api/model.h"
#include "include/api/data_type.h"
#include "include/c_api/context_c.h"
#include "include/c_api/types_c.h"
#include "include/c_api/model_c.h"
#include "include/api/serialization.h"
#include "src/common/log.h"
#include "ms_model_ani.h"
#include "ms_tensor_ani.h"
#include "ms_context_ani.h"
#include "ms_status_ani.h"

namespace mindspore_ani {

static std::mutex create_mutex_;

int32_t GetDeviceInfoContextANI(MSLiteContextInfoANI *context_ptr,
                                std::vector<std::shared_ptr<mindspore::DeviceInfoContext>> &device_infos);

int32_t TransTaiheContext(MSLiteModelInfoANI *model_info_ptr, MSLiteContextInfoANI *context_info_ptr,
                          ::ohos::ai::mindSporeLite::Context context);

void ConfigureDefaultCpuContext(std::unique_ptr<mindspore_ani::MSLiteContextInfoANI> &context_native);

std::shared_ptr<mindspore::Model> CreateModelANI(MSLiteModelInfoANI *model_info_ptr,
                                                 MSLiteContextInfoANI *context_info_ptr);

std::shared_ptr<mindspore::Model> CreateTrainModelANI(MSLiteModelInfoANI *model_info_ptr,
                                                      MSLiteContextInfoANI *context_info_ptr);

void ThrowBusinessError(MSLiteErrorCodeANI code_error);

}  // namespace mindspore_ani
#endif
