/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "include/api/context.h"
#include "include/c_api/context_c.h"
#include <string.h>
#include "src/litert/c_api/type_c_private.h"
#include "src/litert/c_api/context_c.h"
#include "include/api/context.h"
#include "src/common/log_adapter.h"
#ifdef SUPPORT_NNRT_METAGRAPH
#include "src/litert/delegate/nnrt/hiai_foundation_wrapper.h"
#endif
#ifdef SUPPORT_NNRT
#include "neural_network_runtime/neural_network_runtime.h"
#endif

// ================ Context ================
OH_AI_ContextHandle OH_AI_ContextCreate() {
  auto impl = new (std::nothrow) mindspore::ContextC();
  if (impl == nullptr) {
    MS_LOG(ERROR) << "memory allocation failed.";
    return nullptr;
  }
  impl->context_ = new (std::nothrow) mindspore::Context();
  if (impl->context_ == nullptr) {
    MS_LOG(ERROR) << "memory allocation failed.";
    delete impl;
    return nullptr;
  }
  impl->owned_by_model_ = false;
  return static_cast<OH_AI_ContextHandle>(impl);
}

void OH_AI_ContextDestroy(OH_AI_ContextHandle *context) {
  if (context == nullptr || *context == nullptr) {
    MS_LOG(ERROR) << "context is nullptr.";
    return;
  }
  auto impl = static_cast<mindspore::ContextC *>(*context);
  if (impl->owned_by_model_) {
    impl->context_ = nullptr;
  }
  delete impl;
  *context = nullptr;
}

extern "C" {
#define TRUE 1

int32_t OH_AI_Inner_ContextNeedDestroy() {
    return TRUE;
}
}

void OH_AI_ContextSetThreadNum(OH_AI_ContextHandle context, int32_t thread_num) {
  if (context == nullptr) {
    MS_LOG(ERROR) << "context is nullptr.";
    return;
  }
  auto impl = static_cast<mindspore::ContextC *>(context);
  impl->context_->SetThreadNum(thread_num);
}

int32_t OH_AI_ContextGetThreadNum(const OH_AI_ContextHandle context) {
  if (context == nullptr) {
    MS_LOG(ERROR) << "context is nullptr.";
    return 0;
  }
  auto impl = static_cast<mindspore::ContextC *>(context);
  return impl->context_->GetThreadNum();
}

void OH_AI_ContextSetThreadAffinityMode(OH_AI_ContextHandle context, int mode) {
  if (context == nullptr) {
    MS_LOG(ERROR) << "context is nullptr.";
    return;
  }
  auto impl = static_cast<mindspore::ContextC *>(context);
  impl->context_->SetThreadAffinity(mode);
  return;
}

int OH_AI_ContextGetThreadAffinityMode(const OH_AI_ContextHandle context) {
  if (context == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return 0;
  }
  auto impl = static_cast<mindspore::ContextC *>(context);
  return impl->context_->GetThreadAffinityMode();
}

void OH_AI_ContextSetThreadAffinityCoreList(OH_AI_ContextHandle context, const int32_t *core_list, size_t core_num) {
  if (context == nullptr || core_list == nullptr) {
    MS_LOG(ERROR) << "context or core_list is nullptr.";
    return;
  }
  const std::vector<int32_t> vec_core_list(core_list, core_list + core_num);
  auto impl = static_cast<mindspore::ContextC *>(context);
  impl->context_->SetThreadAffinity(vec_core_list);
  return;
}

const int32_t *OH_AI_ContextGetThreadAffinityCoreList(const OH_AI_ContextHandle context, size_t *core_num) {
  if (context == nullptr || core_num == nullptr) {
    MS_LOG(ERROR) << "context or core_num is nullptr.";
    return nullptr;
  }
  auto impl = static_cast<mindspore::ContextC *>(context);
  auto affinity_core_list = impl->context_->GetThreadAffinityCoreList();
  *core_num = affinity_core_list.size();
  int32_t *core_list = static_cast<int32_t *>(malloc((*core_num) * sizeof(int32_t)));
  if (core_list == nullptr) {
    MS_LOG(ERROR) << "malloc core_list is null.";
    return nullptr;
  }
  for (size_t i = 0; i < affinity_core_list.size(); i++) {
    core_list[i] = affinity_core_list[i];
  }
  return core_list;
}

void OH_AI_ContextSetEnableParallel(OH_AI_ContextHandle context, bool is_parallel) {
  if (context == nullptr) {
    MS_LOG(ERROR) << "context is nullptr.";
    return;
  }
  auto impl = static_cast<mindspore::ContextC *>(context);
  impl->context_->SetEnableParallel(is_parallel);
}

bool OH_AI_ContextGetEnableParallel(const OH_AI_ContextHandle context) {
  if (context == nullptr) {
    MS_LOG(ERROR) << "context is nullptr.";
    return false;
  }
  auto impl = static_cast<mindspore::ContextC *>(context);
  return impl->context_->GetEnableParallel();
}

void OH_AI_ContextAddDeviceInfo(OH_AI_ContextHandle context, OH_AI_DeviceInfoHandle device_info) {
  if (context == nullptr || device_info == nullptr) {
    MS_LOG(ERROR) << "context or device_info is nullptr.";
    return;
  }
  auto impl = static_cast<mindspore::ContextC *>(context);
  std::shared_ptr<mindspore::DeviceInfoContext> device(static_cast<mindspore::DeviceInfoContext *>(device_info));
  impl->context_->MutableDeviceInfo().push_back(device);
}

// ================ DeviceInfo ================
OH_AI_DeviceInfoHandle OH_AI_DeviceInfoCreate(OH_AI_DeviceType device_type) {
  mindspore::DeviceInfoContext *impl;
  if (OH_AI_DEVICETYPE_CPU == device_type) {
    impl = new (std::nothrow) mindspore::CPUDeviceInfo();
  } else if (OH_AI_DEVICETYPE_GPU == device_type) {
    impl = new (std::nothrow) mindspore::GPUDeviceInfo();
  } else if (OH_AI_DEVICETYPE_KIRIN_NPU == device_type) {
    impl = new (std::nothrow) mindspore::KirinNPUDeviceInfo();
  } else if (OH_AI_DEVICETYPE_NNRT == device_type) {
    impl = new (std::nothrow) mindspore::NNRTDeviceInfo();
  } else {
    MS_LOG(ERROR) << "device_type is invalid.";
    impl = nullptr;
  }
  if (impl == nullptr) {
    MS_LOG(ERROR) << "memory allocation failed.";
    return nullptr;
  }
  return static_cast<OH_AI_DeviceInfoHandle>(impl);
}

void OH_AI_DeviceInfoDestroy(OH_AI_DeviceInfoHandle *device_info) {
  if (device_info == nullptr || *device_info == nullptr) {
    MS_LOG(ERROR) << "device_info is nullptr.";
    return;
  }
  auto impl = static_cast<mindspore::DeviceInfoContext *>(*device_info);
  delete impl;
  *device_info = nullptr;
}

void OH_AI_DeviceInfoSetProvider(OH_AI_DeviceInfoHandle device_info, const char *provider) {
  if (device_info == nullptr) {
    MS_LOG(ERROR) << "device_info is nullptr.";
    return;
  }
  if (provider == nullptr) {
    MS_LOG(ERROR) << "provider is nullptr.";
    return;
  }
  auto impl = static_cast<mindspore::DeviceInfoContext *>(device_info);
  impl->SetProvider(provider);
}

const char *OH_AI_DeviceInfoGetProvider(const OH_AI_DeviceInfoHandle device_info) {
  if (device_info == nullptr) {
    MS_LOG(ERROR) << "device_info is nullptr.";
    return nullptr;
  }
  auto impl = static_cast<mindspore::DeviceInfoContext *>(device_info);
  char *provider = static_cast<char *>(malloc(impl->GetProvider().size() + 1));
  if (provider == nullptr) {
    MS_LOG(ERROR) << "malloc provider is null.";
    return nullptr;
  }
  for (size_t i = 0; i < impl->GetProvider().size(); i++) {
    provider[i] = impl->GetProvider()[i];
  }
  provider[impl->GetProvider().size()] = '\0';
  return provider;
}

void OH_AI_DeviceInfoSetProviderDevice(OH_AI_DeviceInfoHandle device_info, const char *device) {
  if (device_info == nullptr) {
    MS_LOG(ERROR) << "device_info is nullptr.";
    return;
  }
  if (device == nullptr) {
    MS_LOG(ERROR) << "device is nullptr.";
    return;
  }
  auto impl = static_cast<mindspore::DeviceInfoContext *>(device_info);
  impl->SetProviderDevice(device);
}

const char *OH_AI_DeviceInfoGetProviderDevice(const OH_AI_DeviceInfoHandle device_info) {
  if (device_info == nullptr) {
    MS_LOG(ERROR) << "device_info is nullptr.";
    return nullptr;
  }
  auto impl = static_cast<mindspore::DeviceInfoContext *>(device_info);
  char *provider_device = static_cast<char *>(malloc(impl->GetProviderDevice().size() + 1));
  if (provider_device == nullptr) {
    MS_LOG(ERROR) << "malloc provider_device is null.";
    return nullptr;
  }
  for (size_t i = 0; i < impl->GetProviderDevice().size(); i++) {
    provider_device[i] = impl->GetProviderDevice()[i];
  }
  provider_device[impl->GetProviderDevice().size()] = '\0';
  return provider_device;
}

OH_AI_DeviceType OH_AI_DeviceInfoGetDeviceType(const OH_AI_DeviceInfoHandle device_info) {
  if (device_info == nullptr) {
    MS_LOG(ERROR) << "device_info is nullptr.";
    return OH_AI_DEVICETYPE_INVALID;
  }
  auto impl = static_cast<mindspore::DeviceInfoContext *>(device_info);
  return static_cast<OH_AI_DeviceType>(impl->GetDeviceType());
}

void OH_AI_DeviceInfoSetEnableFP16(OH_AI_DeviceInfoHandle device_info, bool is_fp16) {
  if (device_info == nullptr) {
    MS_LOG(ERROR) << "device_info is nullptr.";
    return;
  }
  auto impl_device = static_cast<mindspore::DeviceInfoContext *>(device_info);
  if (OH_AI_DEVICETYPE_CPU == static_cast<OH_AI_DeviceType>(impl_device->GetDeviceType())) {
    auto impl = static_cast<mindspore::CPUDeviceInfo *>(device_info);
    impl->SetEnableFP16(is_fp16);
  } else if (OH_AI_DEVICETYPE_GPU == static_cast<OH_AI_DeviceType>(impl_device->GetDeviceType())) {
    auto impl = static_cast<mindspore::GPUDeviceInfo *>(device_info);
    impl->SetEnableFP16(is_fp16);
  } else if (OH_AI_DEVICETYPE_NNRT == static_cast<OH_AI_DeviceType>(impl_device->GetDeviceType())) {
    auto impl = static_cast<mindspore::NNRTDeviceInfo *>(device_info);
    impl->SetEnableFP16(is_fp16);
  } else {
    MS_LOG(ERROR) << "Unsupported Feature.";
  }
}

bool OH_AI_DeviceInfoGetEnableFP16(const OH_AI_DeviceInfoHandle device_info) {
  if (device_info == nullptr) {
    MS_LOG(ERROR) << "device_info is nullptr.";
    return false;
  }
  auto impl_device = static_cast<mindspore::DeviceInfoContext *>(device_info);
  if (OH_AI_DEVICETYPE_CPU == static_cast<OH_AI_DeviceType>(impl_device->GetDeviceType())) {
    auto impl = static_cast<mindspore::CPUDeviceInfo *>(device_info);
    return impl->GetEnableFP16();
  } else if (OH_AI_DEVICETYPE_GPU == static_cast<OH_AI_DeviceType>(impl_device->GetDeviceType())) {
    auto impl = static_cast<mindspore::GPUDeviceInfo *>(device_info);
    return impl->GetEnableFP16();
  } else if (OH_AI_DEVICETYPE_NNRT == static_cast<OH_AI_DeviceType>(impl_device->GetDeviceType())) {
    auto impl = static_cast<mindspore::NNRTDeviceInfo *>(device_info);
    return impl->GetEnableFP16();
  } else {
    MS_LOG(ERROR) << "Unsupported Feature. device_type: " << impl_device->GetDeviceType();
    return false;
  }
}

void OH_AI_DeviceInfoSetFrequency(OH_AI_DeviceInfoHandle device_info, int frequency) {  // only for KirinNPU
  if (device_info == nullptr) {
    MS_LOG(ERROR) << "device_info is nullptr.";
    return;
  }
  auto impl_device = static_cast<mindspore::DeviceInfoContext *>(device_info);
  if (static_cast<OH_AI_DeviceType>(impl_device->GetDeviceType()) == OH_AI_DEVICETYPE_KIRIN_NPU) {
    auto impl = static_cast<mindspore::KirinNPUDeviceInfo *>(device_info);
    impl->SetFrequency(frequency);
  } else {
    MS_LOG(ERROR) << "Unsupported Feature.";
  }
}

int OH_AI_DeviceInfoGetFrequency(const OH_AI_DeviceInfoHandle device_info) {  // only for KirinNPU
  if (device_info == nullptr) {
    MS_LOG(ERROR) << "device_info is nullptr.";
    return -1;
  }
  auto impl_device = static_cast<mindspore::DeviceInfoContext *>(device_info);
  if (static_cast<OH_AI_DeviceType>(impl_device->GetDeviceType()) == OH_AI_DEVICETYPE_KIRIN_NPU) {
    auto impl = static_cast<mindspore::KirinNPUDeviceInfo *>(device_info);
    return impl->GetFrequency();
  } else {
    MS_LOG(ERROR) << "Unsupported Feature.";
    return -1;
  }
}

NNRTDeviceDesc *OH_AI_GetAllNNRTDeviceDescs(size_t *num) {
  if (num == nullptr) {
    MS_LOG(ERROR) << "Input num is null";
    return nullptr;
  }
#ifdef SUPPORT_NNRT
#ifdef SUPPORT_NNRT_METAGRAPH
  void *hiai_handle_{nullptr};
  auto ret_load = mindspore::lite::LoadHiaiFLibraryFromPath(&hiai_handle_);
  if (!ret_load || hiai_handle_ == nullptr) {
    MS_LOG(ERROR) << "Load HiAI_Foundation so failed.";
  }
#endif
  *num = 0;

  const size_t *all_device_ids;
  uint32_t device_count;
  auto ret = OH_NNDevice_GetAllDevicesID(&all_device_ids, &device_count);
  if ((ret != OH_NN_SUCCESS) || (device_count == 0)) {
    MS_LOG(ERROR) << "NNRT get all device id failed, ret: " << ret;
    return nullptr;
  }

  NNRTDeviceDesc *desc = (NNRTDeviceDesc *)malloc(sizeof(NNRTDeviceDesc) * device_count);
  if (desc == nullptr) {
    MS_LOG(ERROR) << "NNRT allocate desc failed";
    return nullptr;
  }

  for (uint32_t i = 0; i < device_count; i++) {
    desc[i].device_id = all_device_ids[i];
    OH_NN_DeviceType type;
    (void)OH_NNDevice_GetType(all_device_ids[i], &type);
    desc[i].device_type = static_cast<OH_AI_NNRTDeviceType>(type);

    const char *name = nullptr;
    (void)OH_NNDevice_GetName(all_device_ids[i], &name);
    if (name == nullptr) {
      MS_LOG(ERROR) << "OH_NNDevice_GetName error.";
      return nullptr;
    }
    desc[i].device_name[127] = '\0';
    strncpy(desc[i].device_name, name, 127);
  }
  *num = device_count;
  return desc;
#else
  return nullptr;
#endif
}

NNRTDeviceDesc *OH_AI_GetElementOfNNRTDeviceDescs(NNRTDeviceDesc *descs, size_t index) {
  if (descs == nullptr) {
    MS_LOG(ERROR) << "descs is null";
    return nullptr;
  }
  return descs + index;
}

void OH_AI_DestroyAllNNRTDeviceDescs(NNRTDeviceDesc **desc) {
  if (desc == nullptr) {
    MS_LOG(WARNING) << "desc is null";
    return;
  }
  free(*desc);
  *desc = nullptr;
}

size_t OH_AI_GetDeviceIdFromNNRTDeviceDesc(const NNRTDeviceDesc *desc) {
  if (desc == nullptr) {
    MS_LOG(ERROR) << "NNRT desc is null";
    return 0;
  }
  return desc->device_id;
}

const char *OH_AI_GetNameFromNNRTDeviceDesc(const NNRTDeviceDesc *desc) {
  if (desc == nullptr) {
    MS_LOG(ERROR) << "NNRT desc is null";
    return nullptr;
  }
  return desc->device_name;
}

OH_AI_NNRTDeviceType OH_AI_GetTypeFromNNRTDeviceDesc(const NNRTDeviceDesc *desc) {
  if (desc == nullptr) {
    MS_LOG(ERROR) << "NNRT desc is null";
    return OH_AI_NNRTDeviceType::OH_AI_NNRTDEVICE_OTHERS;
  }
  return desc->device_type;
}

OH_AI_DeviceInfoHandle OH_AI_CreateNNRTDeviceInfoByName(const char *name) {
  size_t num = 0;
  NNRTDeviceDesc *desc = OH_AI_GetAllNNRTDeviceDescs(&num);
  if (desc == nullptr) {
    MS_LOG(ERROR) << "Get all device desc failed";
    return nullptr;
  }
  if (name == nullptr) {
    MS_LOG(ERROR) << "NNRT device name is nullptr";
    return nullptr;
  }
  OH_AI_DeviceInfoHandle handle = nullptr;
  for (size_t i = 0; i < num; i++) {
    if (strncmp(desc[i].device_name, name, NNRT_DEVICE_NAME_MAX - 1) == 0) {
      handle = OH_AI_DeviceInfoCreate(OH_AI_DEVICETYPE_NNRT);
      OH_AI_DeviceInfoSetDeviceId(handle, desc[i].device_id);
      break;
    }
  }
  OH_AI_DestroyAllNNRTDeviceDescs(&desc);
  return handle;
}

OH_AI_DeviceInfoHandle OH_AI_CreateNNRTDeviceInfoByType(OH_AI_NNRTDeviceType type) {
  size_t num = 0;
  NNRTDeviceDesc *desc = OH_AI_GetAllNNRTDeviceDescs(&num);
  if (desc == nullptr) {
    MS_LOG(ERROR) << "Get all device desc failed";
    return nullptr;
  }

  OH_AI_DeviceInfoHandle handle = nullptr;
  for (size_t i = 0; i < num; i++) {
    if (desc[i].device_type == type) {
      handle = OH_AI_DeviceInfoCreate(OH_AI_DEVICETYPE_NNRT);
      OH_AI_DeviceInfoSetDeviceId(handle, desc[i].device_id);
      break;
    }
  }
  OH_AI_DestroyAllNNRTDeviceDescs(&desc);
  return handle;
}

void OH_AI_DeviceInfoSetDeviceId(OH_AI_DeviceInfoHandle device_info, size_t device_id) {
  if (device_info == nullptr) {
    MS_LOG(ERROR) << "device info is null";
    return;
  }
  if (OH_AI_DeviceInfoGetDeviceType(device_info) != OH_AI_DEVICETYPE_NNRT) {
    MS_LOG(ERROR) << "Set device_id of non-NNRT device is not allowable, ignored";
    return;
  }
  auto impl = reinterpret_cast<mindspore::NNRTDeviceInfo *>(device_info);
  impl->SetDeviceID(device_id);
}

size_t OH_AI_DeviceInfoGetDeviceId(const OH_AI_DeviceInfoHandle device_info) {
  if (device_info == nullptr) {
    MS_LOG(ERROR) << "device info is null";
    return 0;
  }
  if (OH_AI_DeviceInfoGetDeviceType(device_info) != OH_AI_DEVICETYPE_NNRT) {
    MS_LOG(ERROR) << "Get device_id of non-NNRT device is not allowable, ignored";
    return 0;
  }
  auto impl = reinterpret_cast<mindspore::NNRTDeviceInfo *>(device_info);
  return impl->GetDeviceID();
}

void OH_AI_DeviceInfoSetPerformanceMode(OH_AI_DeviceInfoHandle device_info, OH_AI_PerformanceMode mode) {
  if (device_info == nullptr) {
    MS_LOG(ERROR) << "device info is null";
    return;
  }
  if (OH_AI_DeviceInfoGetDeviceType(device_info) != OH_AI_DEVICETYPE_NNRT) {
    MS_LOG(ERROR) << "Set performance_mode of non-NNRT device is not allowable, ignored";
    return;
  }
  auto impl = reinterpret_cast<mindspore::NNRTDeviceInfo *>(device_info);
  impl->SetPerformanceMode(mode);
}

OH_AI_PerformanceMode OH_AI_DeviceInfoGetPerformanceMode(const OH_AI_DeviceInfoHandle device_info) {
  if (device_info == nullptr) {
    MS_LOG(ERROR) << "device info is null";
    return OH_AI_PERFORMANCE_NONE;
  }
  if (OH_AI_DeviceInfoGetDeviceType(device_info) != OH_AI_DEVICETYPE_NNRT) {
    MS_LOG(ERROR) << "Get performance_mode of non-NNRT device is not allowable, ignored";
    return OH_AI_PERFORMANCE_NONE;
  }
  auto impl = reinterpret_cast<mindspore::NNRTDeviceInfo *>(device_info);
  return static_cast<OH_AI_PerformanceMode>(impl->GetPerformanceMode());
}

void OH_AI_DeviceInfoSetPriority(OH_AI_DeviceInfoHandle device_info, OH_AI_Priority priority) {
  if (device_info == nullptr) {
    MS_LOG(ERROR) << "device info is null";
    return;
  }
  if (OH_AI_DeviceInfoGetDeviceType(device_info) != OH_AI_DEVICETYPE_NNRT) {
    MS_LOG(ERROR) << "Set priority of non-NNRT device is not allowable, ignored";
    return;
  }
  auto impl = reinterpret_cast<mindspore::NNRTDeviceInfo *>(device_info);
  impl->SetPriority(priority);
}

OH_AI_Priority OH_AI_DeviceInfoGetPriority(const OH_AI_DeviceInfoHandle device_info) {
  if (device_info == nullptr) {
    MS_LOG(ERROR) << "device info is null";
    return OH_AI_PRIORITY_NONE;
  }
  if (OH_AI_DeviceInfoGetDeviceType(device_info) != OH_AI_DEVICETYPE_NNRT) {
    MS_LOG(ERROR) << "Get priority of non-NNRT device is not allowable, ignored";
    return OH_AI_PRIORITY_NONE;
  }
  auto impl = reinterpret_cast<mindspore::NNRTDeviceInfo *>(device_info);
  return static_cast<OH_AI_Priority>(impl->GetPriority());
}

OH_AI_API OH_AI_Status OH_AI_DeviceInfoAddExtension(OH_AI_DeviceInfoHandle device_info,
                                                    const char *name, const char*value, size_t value_size) {
  if (device_info == nullptr) {
    MS_LOG(ERROR) << "device info is null";
    return OH_AI_STATUS_LITE_NULLPTR;
  }
  if (name == nullptr || value == nullptr) {
    MS_LOG(ERROR) << "name/value is not valid";
    return OH_AI_STATUS_LITE_NULLPTR;
  }
  if (OH_AI_DeviceInfoGetDeviceType(device_info) != OH_AI_DEVICETYPE_NNRT) {
    MS_LOG(ERROR) << "Add extension to non-NNRT device is not allowable, ignored";
    return OH_AI_STATUS_LITE_ERROR;
  }
  static std::vector<std::string> extension_keys = {"CachePath", "CacheVersion", "ModelName", "QuantBuffer",
                                                    "QuantConfigData", "isProfiling", "opLayout", "InputDims",
                                                    "DynamicDims", "BandMode", "NPU_FM_SHARED"};
  auto it = std::find(extension_keys.begin(), extension_keys.end(), std::string(name));
  if (it == extension_keys.end()) {
    MS_LOG(ERROR) << "The name of the extension is not allowable, only can be one of {CachePath, CacheVersion,"
                    << " ModelName, QuantBuffer, isProfiling, opLayout, InputDims, DynamicDims, BandMode,"
                    << "NPU_FM_SHARED}.";
    return OH_AI_STATUS_LITE_ERROR;
  }
  auto impl = reinterpret_cast<mindspore::NNRTDeviceInfo *>(device_info);
  mindspore::Extension extension;
  extension.name = std::string(name);
  extension.value = std::vector<uint8_t>(value, value + value_size);
  std::vector<mindspore::Extension> extension_list = impl->GetExtensions();
  extension_list.push_back(extension);
  impl->SetExtensions(extension_list);
  return OH_AI_STATUS_SUCCESS;
}
