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
#ifndef MINDSPORE_INCLUDE_C_API_CONTEXT_C_H
#define MINDSPORE_INCLUDE_C_API_CONTEXT_C_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "include/c_api/status_c.h"
#include "include/c_api/types_c.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void *OH_AI_ContextHandle;
typedef void *OH_AI_DeviceInfoHandle;

/// \brief Create a context object.
///
/// \return Context object handle.
OH_AI_API OH_AI_ContextHandle OH_AI_ContextCreate();

/// \brief Destroy the context object.
///
/// \param[in] context Context object handle address.
OH_AI_API void OH_AI_ContextDestroy(OH_AI_ContextHandle *context);

/// \brief Set the number of threads at runtime.
///
/// \param[in] context Context object handle.
/// \param[in] thread_num the number of threads at runtime.
OH_AI_API void OH_AI_ContextSetThreadNum(OH_AI_ContextHandle context, int32_t thread_num);

/// \brief Obtain the current thread number setting.
///
/// \param[in] context Context object handle.
///
/// \return The current thread number setting.
OH_AI_API int32_t OH_AI_ContextGetThreadNum(const OH_AI_ContextHandle context);

/// \brief Set the thread affinity to CPU cores.
///
/// \param[in] context Context object handle.
/// \param[in] mode: 0: no affinities, 1: big cores first, 2: little cores first
OH_AI_API void OH_AI_ContextSetThreadAffinityMode(OH_AI_ContextHandle context, int mode);

/// \brief Obtain the thread affinity of CPU cores.
///
/// \param[in] context Context object handle.
///
/// \return Thread affinity to CPU cores. 0: no affinities, 1: big cores first, 2: little cores first
OH_AI_API int OH_AI_ContextGetThreadAffinityMode(const OH_AI_ContextHandle context);

/// \brief Set the thread lists to CPU cores.
///
/// \note If core_list and mode are set by OH_AI_ContextSetThreadAffinityMode at the same time,
/// the core_list is effective, but the mode is not effective.
///
/// \param[in] context Context object handle.
/// \param[in] core_list: a array of thread core lists.
/// \param[in] core_num The number of core.
OH_AI_API void OH_AI_ContextSetThreadAffinityCoreList(OH_AI_ContextHandle context, const int32_t *core_list, size_t core_num);

/// \brief Obtain the thread lists of CPU cores.
///
/// \param[in] context Context object handle.
/// \param[out] core_num The number of core.
///
/// \return a array of thread core lists.
OH_AI_API const int32_t *OH_AI_ContextGetThreadAffinityCoreList(const OH_AI_ContextHandle context, size_t *core_num);

/// \brief Set the status whether to perform model inference or training in parallel.
///
/// \param[in] context Context object handle.
/// \param[in] is_parallel: true, parallel; false, not in parallel.
OH_AI_API void OH_AI_ContextSetEnableParallel(OH_AI_ContextHandle context, bool is_parallel);

/// \brief Obtain the status whether to perform model inference or training in parallel.
///
/// \param[in] context Context object handle.
///
/// \return Bool value that indicates whether in parallel.
OH_AI_API bool OH_AI_ContextGetEnableParallel(const OH_AI_ContextHandle context);

/// \brief Add device info to context object.
///
/// \param[in] context Context object handle.
/// \param[in] device_info Device info object handle.
OH_AI_API void OH_AI_ContextAddDeviceInfo(OH_AI_ContextHandle context, OH_AI_DeviceInfoHandle device_info);

/// \brief Create a device info object.
///
/// \param[in] device_info Device info object handle.
///
/// \return Device info object handle.
OH_AI_API OH_AI_DeviceInfoHandle OH_AI_DeviceInfoCreate(OH_AI_DeviceType device_type);

/// \brief Destroy the device info object.
///
/// \param[in] device_info Device info object handle address.
OH_AI_API void OH_AI_DeviceInfoDestroy(OH_AI_DeviceInfoHandle *device_info);

/// \brief Set provider's name.
///
/// \param[in] device_info Device info object handle.
/// \param[in] provider define the provider's name.
OH_AI_API void OH_AI_DeviceInfoSetProvider(OH_AI_DeviceInfoHandle device_info, const char *provider);

/// \brief Obtain provider's name
///
/// \param[in] device_info Device info object handle.
///
/// \return provider's name.
OH_AI_API const char *OH_AI_DeviceInfoGetProvider(const OH_AI_DeviceInfoHandle device_info);

/// \brief Set provider's device type.
///
/// \param[in] device_info Device info object handle.
/// \param[in] device define the provider's device type. EG: CPU.
OH_AI_API void OH_AI_DeviceInfoSetProviderDevice(OH_AI_DeviceInfoHandle device_info, const char *device);

/// \brief Obtain provider's device type.
///
/// \param[in] device_info Device info object handle.
///
/// \return provider's device type.
OH_AI_API const char *OH_AI_DeviceInfoGetProviderDevice(const OH_AI_DeviceInfoHandle device_info);

/// \brief Obtain the device type of the device info.
///
/// \param[in] device_info Device info object handle.
///
/// \return Device Type of the device info.
OH_AI_API OH_AI_DeviceType OH_AI_DeviceInfoGetDeviceType(const OH_AI_DeviceInfoHandle device_info);

/// \brief Set enables to perform the float16 inference, Only valid for CPU/GPU.
///
/// \param[in] device_info Device info object handle.
/// \param[in] is_fp16 Enable float16 inference or not.
OH_AI_API void OH_AI_DeviceInfoSetEnableFP16(OH_AI_DeviceInfoHandle device_info, bool is_fp16);

/// \brief Obtain enables to perform the float16 inference, Only valid for CPU/GPU.
///
/// \param[in] device_info Device info object handle.
///
/// \return Whether enable float16 inference.
OH_AI_API bool OH_AI_DeviceInfoGetEnableFP16(const OH_AI_DeviceInfoHandle device_info);

/// \brief Set the NPU frequency, Only valid for NPU.
///
/// \param[in] device_info Device info object handle.
/// \param[in] frequency Can be set to 1 (low power consumption), 2 (balanced), 3 (high performance), 4 (extreme
/// performance), default as 3.
OH_AI_API void OH_AI_DeviceInfoSetFrequency(OH_AI_DeviceInfoHandle device_info, int frequency);

/// \brief Obtain the NPU frequency, Only valid for NPU.
///
/// \param[in] device_info Device info object handle.
///
/// \return NPU frequency
OH_AI_API int OH_AI_DeviceInfoGetFrequency(const OH_AI_DeviceInfoHandle device_info);

/// \brief Obtain the all device descriptions in NNRT.
///
/// \param[out] num Number of NNRT device description.
///
/// \return NNRT device description array.
OH_AI_API NNRTDeviceDesc *OH_AI_GetAllNNRTDeviceDescs(size_t *num);

/// \brief Obtain the specified element in NNRt device description array.
///
/// \param[in] descs NNRT device description array.
/// \param[in] index Element index.
///
/// \return NNRT device description.
OH_AI_API NNRTDeviceDesc *OH_AI_GetElementOfNNRTDeviceDescs(NNRTDeviceDesc *descs, size_t index);

/// \brief Obtain the all device descriptions in NNRT.
///
/// \param[out] num Number of NNRT device description.
///
/// \return NNRT device description array.
OH_AI_API NNRTDeviceDesc *OH_AI_GetAllNNRTDeviceDescs(size_t *num);

/// \brief Destroy the NNRT device descriptions returned by OH_AI_GetAllNNRTDeviceDescs().
///
/// \param[in] desc NNRT device description array.
OH_AI_API void OH_AI_DestroyAllNNRTDeviceDescs(NNRTDeviceDesc **desc);

/// \brief Obtain the device id in NNRT device description.
///
/// \param[in] desc pointer to the NNRT device description instance.
///
/// \return NNRT device id.
OH_AI_API size_t OH_AI_GetDeviceIdFromNNRTDeviceDesc(const NNRTDeviceDesc *desc);

/// \brief Obtain the device name in NNRT device description.
///
/// \param[in] desc pointer to the NNRT device description instance.
///
/// \return NNRT device name.
OH_AI_API const char *OH_AI_GetNameFromNNRTDeviceDesc(const NNRTDeviceDesc *desc);

/// \brief Obtain the device type in NNRT device description.
///
/// \param[in] desc pointer to the NNRT device description instance.
///
/// \return NNRT device type.
OH_AI_API OH_AI_NNRTDeviceType OH_AI_GetTypeFromNNRTDeviceDesc(const NNRTDeviceDesc *desc);

/// \brief Create the NNRT device info by exactly matching the specific device name.
///
/// \param[in] name NNRt device name.
///
/// \return Device info object handle.
OH_AI_API OH_AI_DeviceInfoHandle OH_AI_CreateNNRTDeviceInfoByName(const char *name);

/// \brief Create the NNRT device info by finding the first device with the specific device type.
///
/// \param[in] name NNRt device type.
///
/// \return Device info object handle.
OH_AI_API OH_AI_DeviceInfoHandle OH_AI_CreateNNRTDeviceInfoByType(OH_AI_NNRTDeviceType type);

/// \brief Set the NNRT device id, Only valid for NNRT.
///
/// \param[in] device_info Device info object handle.
/// \param[in] device_id NNRT device id.
OH_AI_API void OH_AI_DeviceInfoSetDeviceId(OH_AI_DeviceInfoHandle device_info, size_t device_id);

/// \brief Obtain the NNRT device id, Only valid for NNRT.
///
/// \param[in] device_info Device info object handle.
///
/// \return NNRT device id.
OH_AI_API size_t OH_AI_DeviceInfoGetDeviceId(const OH_AI_DeviceInfoHandle device_info);

/// \brief Set the NNRT performance mode, Only valid for NNRT.
///
/// \param[in] device_info Device info object handle.
/// \param[in] device_id NNRT performance mode.
OH_AI_API void OH_AI_DeviceInfoSetPerformanceMode(OH_AI_DeviceInfoHandle device_info, OH_AI_PerformanceMode mode);

/// \brief Obtain the NNRT performance mode, Only valid for NNRT.
///
/// \param[in] device_info Device info object handle.
///
/// \return NNRT performance mode.
OH_AI_API OH_AI_PerformanceMode OH_AI_DeviceInfoGetPerformanceMode(const OH_AI_DeviceInfoHandle device_info);

/// \brief Set the NNRT priority, Only valid for NNRT.
///
/// \param[in] device_info Device info object handle.
/// \param[in] device_id NNRT priority.
OH_AI_API void OH_AI_DeviceInfoSetPriority(OH_AI_DeviceInfoHandle device_info, OH_AI_Priority priority);

/// \brief Obtain the NNRT priority, Only valid for NNRT.
///
/// \param[in] device_info Device info object handle.
///
/// \return NNRT priority.
OH_AI_API OH_AI_Priority OH_AI_DeviceInfoGetPriority(const OH_AI_DeviceInfoHandle device_info);

/// \brief Add extension of key/value format to device info, Only valid for NNRT.
///
/// \param[in] device_info Device info object handle.
/// \param[in] name The content of key as a C string.
/// \param[in] value The pointer to the value, which is a byte array.
/// \param[in] value_size The size of the value, which is a byte array.
///
/// \return OH_AI_STATUS_SUCCESS if success, or detail error code if failed.
OH_AI_API OH_AI_Status OH_AI_DeviceInfoAddExtension(OH_AI_DeviceInfoHandle device_info, const char *name, const char *value, size_t value_size);
#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_INCLUDE_C_API_CONTEXT_C_H
