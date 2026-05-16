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
#ifndef MS_CONTEXT_ANI
#define MS_CONTEXT_ANI
#include <stdexcept>
#include <sys/mman.h>
#include "include/api/types.h"
#include "include/api/context.h"
#include "include/c_api/context_c.h"

namespace mindspore_ani {

const int DEFAULT_THREAD_NUM = 2;

const int DEFAULT_THREAD_AFFINITY = 0;
struct MSLiteCpuDeviceANI {
  int thread_num{0};
  int thread_affinity_mode{0};
  std::vector<int32_t> thread_affinity_cores{};
  std::string precision_mode{};

  MSLiteCpuDeviceANI() = default;

  MSLiteCpuDeviceANI(int thread_num, int affinity_mode, const std::vector<int32_t> &affinity_cores,
                     const std::string &precision)
      : thread_num(thread_num),
        thread_affinity_mode(affinity_mode),
        thread_affinity_cores(affinity_cores),
        precision_mode(precision) {}
};

struct MSLiteNNRTDeviceANI {
  size_t device_id{0};
  int performance_mode{-1};
  int priority{-1};

  MSLiteNNRTDeviceANI() = default;

  MSLiteNNRTDeviceANI(size_t device_id, int performance_mode, int priority)
      : device_id(device_id), performance_mode(performance_mode), priority(priority) {}
};

struct MSLiteTrainConfigANI {
  std::vector<std::string> loss_names;
  int optimization_level{mindspore::kO0};  // kAUTO

  MSLiteTrainConfigANI() = default;

  MSLiteTrainConfigANI(std::vector<std::string> loss_names, int optimization_level = mindspore::kO0)
      : loss_names(std::move(loss_names)), optimization_level(optimization_level) {}
};

struct MSLiteContextInfoANI {
  std::vector<std::string> target{};
  MSLiteCpuDeviceANI cpu_device{};
  MSLiteNNRTDeviceANI nnrt_device{};
  MSLiteTrainConfigANI train_cfg{};
};

const std::unordered_map<std::string, mindspore::DeviceType> kDeviceTypesANI{
  {"cpu", mindspore::kCPU},
  {"nnrt", mindspore::kNNRt},
  {"gpu", mindspore::kGPU},
};

enum ContextNnrtDeviceType : int32_t {
  NNRTDEVICE_OTHERS = 0,
  NNRTDEVICE_CPU = 1,
  NNRTDEVICE_GPU = 2,
  NNRTDEVICE_ACCELERATOR = 3,
};

struct NnrtDeviceDesc {
  std::string name{};
  ContextNnrtDeviceType type{ContextNnrtDeviceType::NNRTDEVICE_OTHERS};
  size_t id{0};
  NnrtDeviceDesc() = default;
  NnrtDeviceDesc(const std::string &name_, ContextNnrtDeviceType type_, size_t id_)
      : name(name_), type(type_), id(id_) {}
};

static const std::map<mindspore_ani::ContextNnrtDeviceType, ::ohos::ai::mindSporeLite::NNRTDeviceType>
  NNRTDeviceTypeMapANI = {
    {mindspore_ani::ContextNnrtDeviceType::NNRTDEVICE_OTHERS,
     ::ohos::ai::mindSporeLite::NNRTDeviceType::key_t::NNRTDEVICE_OTHERS},
    {mindspore_ani::ContextNnrtDeviceType::NNRTDEVICE_CPU,
     ::ohos::ai::mindSporeLite::NNRTDeviceType::key_t::NNRTDEVICE_CPU},
    {mindspore_ani::ContextNnrtDeviceType::NNRTDEVICE_GPU,
     ::ohos::ai::mindSporeLite::NNRTDeviceType::key_t::NNRTDEVICE_GPU},
    {mindspore_ani::ContextNnrtDeviceType::NNRTDEVICE_ACCELERATOR,
     ::ohos::ai::mindSporeLite::NNRTDeviceType::key_t::NNRTDEVICE_ACCELERATOR},
};

}  // namespace mindspore_ani
#endif
