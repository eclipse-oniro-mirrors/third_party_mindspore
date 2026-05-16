/*
 * Copyright (c) 2025 Huawei Device Co., Ltd.
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

#include "nnrt_wrapper.h"
#include <dlfcn.h>
#include <string>
#include "src/common/log_adapter.h"

template <class T>
bool LoadFunctionHelper(void *handle, const char *name, T *function) {
  if (name == nullptr) {
    MS_LOG(ERROR) << "load function failed, the " << name << " does not exist.";
    return false;
  }
  void *fn = dlsym(handle, name);
  if (fn == nullptr) {
    MS_LOG(ERROR) << "dlsym function " << name << " error: " << dlerror();
    return false;
  }
  *function = reinterpret_cast<T>(fn);
  return true;
}

namespace mindspore {
std::string kNNRTLibraryName = "libneural_network_runtime.so";
std::string kNNCoreLibraryName = "libneural_network_core.so";
bool NNRTWrapper::IsLoaded() { return is_loaded_; }
NNRTWrapper& NNRTWrapper::GetInstance() {
  static NNRTWrapper instance;
  if (!instance.IsLoaded()) {
    instance.LoadLibrary();
  }
  return instance;
}
void NNRTWrapper::LoadLibrary() {
  nnrt_library_handle_ = dlopen(kNNRTLibraryName.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (nnrt_library_handle_ == nullptr) {
    MS_LOG(ERROR) << "dlopen nnrt library " << kNNRTLibraryName << " error: " << dlerror();
    return;
  }
  MS_LOG(INFO) << "loaded nnrt library " << kNNRTLibraryName;
  nncore_library_handle_ = dlopen(kNNCoreLibraryName.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (nncore_library_handle_ == nullptr) {
    MS_LOG(ERROR) << "dlopen nncore library " << kNNCoreLibraryName << " error: " << dlerror();
    return;
  }
  MS_LOG(INFO) << "loaded nncore library " << kNNCoreLibraryName;

  bool nnrt_ok = LoadFunctionHelper(nnrt_library_handle_, "OH_NNModel_HasCache", &nnrt_hascache_func_) &&
            LoadFunctionHelper(nnrt_library_handle_, "OH_NNModel_Construct", &nnrt_construct_func_) &&
            LoadFunctionHelper(nnrt_library_handle_, "OH_NNModel_BuildFromLiteGraph", &nnrt_buildfromlitegraph_func_) &&
            LoadFunctionHelper(nnrt_library_handle_, "OH_NNModel_Destroy", &nnrt_destroy_func_) &&
            LoadFunctionHelper(nnrt_library_handle_, "OH_NNModel_GetAvailableOperations", &nnrt_getavailableoperations_func_) &&
            LoadFunctionHelper(nnrt_library_handle_, "OH_NNExecutor_SetInput", &nnrt_executorsetinput_func_) &&
            LoadFunctionHelper(nnrt_library_handle_, "OH_NNExecutor_SetOutput", &nnrt_executorsetoutput_func_) &&
            LoadFunctionHelper(nnrt_library_handle_, "OH_NN_IsSupportAIPP", &nnrt_issupportaipp_func_) &&
            LoadFunctionHelper(nnrt_library_handle_, "OH_NNExecutor_RunSyncWithAipp", &nnrt_executorrunsyncwithaipp_func_);

  if (!nnrt_ok) {
    MS_LOG(ERROR) << "failed to load necessary nnrt functions for mslite";
    int ret = dlclose(nnrt_library_handle_);
    if (ret != 0) {
      MS_LOG(WARNING) << "dlclose nnrt library " << kNNRTLibraryName << " error: " << dlerror();
    }
    nnrt_library_handle_ = nullptr;
    return;
  }

  bool nncore_ok = LoadFunctionHelper(nncore_library_handle_, "OH_NNExecutor_Destroy", &nncore_executordestroy_func_) &&
                   LoadFunctionHelper(nncore_library_handle_, "OH_NNDevice_GetName", &nncore_devicegetname_func_) &&
                   LoadFunctionHelper(nncore_library_handle_, "OH_NNCompilation_ConstructForCache", &nncore_compilationconstructforcache_func_) &&
                   LoadFunctionHelper(nncore_library_handle_, "OH_NNCompilation_Destroy", &nncore_compilationdestroy_func_) &&
                   LoadFunctionHelper(nncore_library_handle_, "OH_NNExecutor_Construct", &nncore_executorconstruct_func_) &&
                   LoadFunctionHelper(nncore_library_handle_, "OH_NNCompilation_SetDevice", &nncore_compilationsetdevice_func_) &&
                   LoadFunctionHelper(nncore_library_handle_, "OH_NNCompilation_SetPerformanceMode", &nncore_compilationsetperformancemode_func_) &&
                   LoadFunctionHelper(nncore_library_handle_, "OH_NNCompilation_EnableFloat16", &nncore_compilationenablefloat16_func_) &&
                   LoadFunctionHelper(nncore_library_handle_, "OH_NNCompilation_SetCache", &nncore_compilationsetcache_func_) &&
                   LoadFunctionHelper(nncore_library_handle_, "OH_NNCompilation_AddExtensionConfig", &nncore_compilationaddextensionconfig_func_) &&
                   LoadFunctionHelper(nncore_library_handle_, "OH_NNCompilation_Build", &nncore_compilationbuild_func_) &&
                   LoadFunctionHelper(nncore_library_handle_, "OH_NNCompilation_ConstructWithOfflineModelBuffer", &nncore_compilationconstructwithofflinemodelbuffer_func_) &&
                   LoadFunctionHelper(nncore_library_handle_, "OH_NNCompilation_Construct", &nncore_compilationconstruct_func_) &&
                   LoadFunctionHelper(nncore_library_handle_, "OH_NNExecutor_RunSync", &nncore_executorrunsync_func_) &&
                   LoadFunctionHelper(nncore_library_handle_, "OH_NNTensor_Destroy", &nncore_tensordestroy_func_) &&
                   LoadFunctionHelper(nncore_library_handle_, "OH_NNTensorDesc_Destroy", &nncore_tensordescdestroy_func_) &&
                   LoadFunctionHelper(nncore_library_handle_, "OH_NNExecutor_GetInputCount", &nncore_executorgetinputcount_func_) &&
                   LoadFunctionHelper(nncore_library_handle_, "OH_NNExecutor_CreateInputTensorDesc", &nncore_executorcreateinputtensordesc_func_) &&
                   LoadFunctionHelper(nncore_library_handle_, "OH_NNTensor_Create", &nncore_tensorcreate_func_) &&
                   LoadFunctionHelper(nncore_library_handle_, "OH_NNCompilation_SetPriority", &nncore_compilationsetpriority_func_) &&
                   LoadFunctionHelper(nncore_library_handle_, "OH_NNExecutor_GetOutputCount", &nncore_executorgetoutputcount_func_) &&
                   LoadFunctionHelper(nncore_library_handle_, "OH_NNExecutor_CreateOutputTensorDesc", &nncore_executorcreateoutputtensordesc_func_) &&
                   LoadFunctionHelper(nncore_library_handle_, "OH_NNTensor_GetDataBuffer", &nncore_tensorgetdatabuffer_func_) &&
                   LoadFunctionHelper(nncore_library_handle_, "OH_NNTensorDesc_SetShape", &nncore_tensordescsetshape_func_) &&
                   LoadFunctionHelper(nncore_library_handle_, "OH_NNTensorDesc_SetDataType", &nncore_tensordescsetdatatype_func_) &&
                   LoadFunctionHelper(nncore_library_handle_, "OH_NNTensorDesc_SetFormat", &nncore_tensordescsetformat_func_) &&
                   LoadFunctionHelper(nncore_library_handle_, "OH_NNTensorDesc_SetName", &nncore_tensordescsetname_func_) &&
                   LoadFunctionHelper(nncore_library_handle_, "OH_NNTensorDesc_Create", &nncore_tensordesccreate_func_) &&
                   LoadFunctionHelper(nncore_library_handle_, "OH_NNDevice_GetAllDevicesID", &nncore_devicegetalldevicesid_func_) &&
                   LoadFunctionHelper(nncore_library_handle_, "OH_NNDevice_GetType", &nncore_devicegettype_func_) &&
                   LoadFunctionHelper(nncore_library_handle_, "OH_NNTensorDesc_GetDataType", &nncore_tensordescgetdatatype_func_);
  
  if (!nncore_ok) {
    MS_LOG(ERROR) << "failed to load necessary nncore functions for mslite";
    int ret = dlclose(nncore_library_handle_);
    if (ret != 0) {
      MS_LOG(WARNING) << "dlclose nncore library " << kNNCoreLibraryName << " error: " << dlerror();
    }
    nncore_library_handle_ = nullptr;
    ret = dlclose(nnrt_library_handle_);
    if (ret != 0) {
      MS_LOG(WARNING) << "dlclose nnrt library " << kNNCoreLibraryName << " error: " << dlerror();
    }
    nnrt_library_handle_ = nullptr;
    return;
  }
  is_loaded_ = true;
}

NNRTWrapper::~NNRTWrapper() {
  if (!IsLoaded()) {
    return;
  }
  int ret = dlclose(nnrt_library_handle_);
  if (ret != 0) {
    MS_LOG(WARNING) << "dlclose nnrt library " << kNNRTLibraryName << " error: " << dlerror();
  }
  ret = dlclose(nncore_library_handle_);
  if (ret != 0) {
    MS_LOG(WARNING) << "dlclose nncore library " << kNNCoreLibraryName << " error: " << dlerror();
  }
  nnrt_library_handle_ = nullptr;
  MS_LOG(INFO) << "unloaded nnrt library " << kNNRTLibraryName;
  nncore_library_handle_ = nullptr;
  MS_LOG(INFO) << "unloaded nncore library " << kNNCoreLibraryName;
  return;
}

// nnrt function
bool NNRTWrapper::HasCache(const char *cacheDir, const char *modelName, uint32_t version) {
  if (!IsLoaded()) {
    MS_LOG(ERROR) << "nnrt wrapper is not created";
    return OH_AI_STATUS_LITE_ERROR;
  }
  if (cacheDir == nullptr) {
    MS_LOG(ERROR) << "cacheDir is nullptr";
    return OH_AI_STATUS_LITE_ERROR;
  }
  if (modelName == nullptr) {
    MS_LOG(ERROR) << "cacheDir is nullptr";
    return OH_AI_STATUS_LITE_ERROR;
  }
  return nnrt_hascache_func_(cacheDir, modelName, version);
}

OH_NNModel* NNRTWrapper::Construct(void) {
  if (!IsLoaded()) {
    MS_LOG(ERROR) << "nnrt wrapper is not created";
    return nullptr;
  }
  return nnrt_construct_func_();
}

OH_NN_ReturnCode NNRTWrapper::BuildFromLiteGraph(OH_NNModel *model, const void *liteGraph, const OH_NN_Extension *extensions, size_t extensionSize) {
  if (!IsLoaded()) {
    MS_LOG(ERROR) << "nnrt wrapper is not created";
    return OH_NN_FAILED;
  }
  return nnrt_buildfromlitegraph_func_(model, liteGraph, extensions, extensionSize);
}

void NNRTWrapper::Destroy(OH_NNModel **model) {
  if (!IsLoaded()) {
    MS_LOG(ERROR) << "nnrt wrapper is not created";
    return;
  }
  nnrt_destroy_func_(model);
}

OH_NN_ReturnCode NNRTWrapper::GetAvailableOperations(OH_NNModel *model, size_t deviceID, const bool **isSupported, uint32_t *opCount) {
  if (!IsLoaded()) {
    MS_LOG(ERROR) << "nnrt wrapper is not created";
    return OH_NN_FAILED;
  }
  return nnrt_getavailableoperations_func_(model, deviceID, isSupported, opCount);
}

OH_NN_ReturnCode NNRTWrapper::ExecutorSetInput(OH_NNExecutor *executor, uint32_t inputIndex, const OH_NN_Tensor *tensor, const void *dataBuffer, size_t length) {
  if (!IsLoaded()) {
    MS_LOG(ERROR) << "nnrt wrapper is not created";
    return OH_NN_FAILED;
  }
  return nnrt_executorsetinput_func_(executor, inputIndex, tensor, dataBuffer, length);
}

OH_NN_ReturnCode NNRTWrapper::ExecutorSetOutput(OH_NNExecutor *executor, uint32_t outputIndex, void *dataBuffer, size_t length) {
  if (!IsLoaded()) {
    MS_LOG(ERROR) << "nnrt wrapper is not created";
    return OH_NN_FAILED;
  }
  return nnrt_executorsetoutput_func_(executor, outputIndex, dataBuffer, length);
}

OH_NN_ReturnCode NNRTWrapper::IsSupportAIPP(bool& support) {
  if (!IsLoaded()) {
    MS_LOG(ERROR) << "nnrt wrapper is not created";
    return OH_NN_FAILED;
  }
  return nnrt_issupportaipp_func_(support);
}

OH_NN_ReturnCode NNRTWrapper::ExecutorRunSyncWithAipp(OH_NNExecutor *executor,
                                         NN_Tensor *inputTensor[],
                                         size_t inputCount,
                                         NN_Tensor *outputTensor[],
                                         size_t outputCount,
                                         const char* aippString) {
  if (!IsLoaded()) {
    MS_LOG(ERROR) << "nnrt wrapper is not created";
    return OH_NN_FAILED;
  }
  return nnrt_executorrunsyncwithaipp_func_(executor, inputTensor, inputCount, outputTensor, outputCount, aippString);
}

// nncore function
void NNRTWrapper::NNExecutorDestroy(OH_NNExecutor **executor) {
  if (!IsLoaded()) {
    MS_LOG(ERROR) << "nnrt wrapper is not created";
    return;
  }
  nncore_executordestroy_func_(executor);
}
OH_NN_ReturnCode NNRTWrapper::NNDeviceGetName(size_t deviceID, const char **name) {
  if (!IsLoaded()) {
    MS_LOG(ERROR) << "nnrt wrapper is not created";
    return OH_NN_FAILED;
  }
  return nncore_devicegetname_func_(deviceID, name);
}
OH_NNCompilation* NNRTWrapper::NNCompilationConstructForCache() {
  if (!IsLoaded()) {
    MS_LOG(ERROR) << "nnrt wrapper is not created";
    return nullptr;
  }
  return nncore_compilationconstructforcache_func_();
}
void NNRTWrapper::NNCompilationDestroy(OH_NNCompilation **compilation) {
  if (!IsLoaded()) {
    MS_LOG(ERROR) << "nnrt wrapper is not created";
    return;
  }
  nncore_compilationdestroy_func_(compilation);
}
OH_NNExecutor* NNRTWrapper::NNExecutorConstruct(OH_NNCompilation *compilation) {
  if (!IsLoaded()) {
    MS_LOG(ERROR) << "nnrt wrapper is not created";
    return nullptr;
  }
  return nncore_executorconstruct_func_(compilation);
}
OH_NN_ReturnCode NNRTWrapper::NNCompilationSetDevice(OH_NNCompilation *compilation, size_t deviceID) {
  if (!IsLoaded()) {
    MS_LOG(ERROR) << "nnrt wrapper is not created";
    return OH_NN_FAILED;
  }
  return nncore_compilationsetdevice_func_(compilation, deviceID);
}
OH_NN_ReturnCode NNRTWrapper::NNCompilationSetPerformanceMode(OH_NNCompilation *compilation, OH_NN_PerformanceMode performanceMode) {
  if (!IsLoaded()) {
    MS_LOG(ERROR) << "nnrt wrapper is not created";
    return OH_NN_FAILED;
  }
  return nncore_compilationsetperformancemode_func_(compilation, performanceMode);
}
OH_NN_ReturnCode NNRTWrapper::NNCompilationEnableFloat16(OH_NNCompilation *compilation, bool enableFloat16) {
  if (!IsLoaded()) {
    MS_LOG(ERROR) << "nnrt wrapper is not created";
    return OH_NN_FAILED;
  }
  return nncore_compilationenablefloat16_func_(compilation, enableFloat16);
}
OH_NN_ReturnCode NNRTWrapper::NNCompilationSetCache(OH_NNCompilation *compilation, const char *cachePath, uint32_t version) {
  if (!IsLoaded()) {
    MS_LOG(ERROR) << "nnrt wrapper is not created";
    return OH_NN_FAILED;
  }
  return nncore_compilationsetcache_func_(compilation, cachePath, version);
}
OH_NN_ReturnCode NNRTWrapper::NNCompilationAddExtensionConfig(OH_NNCompilation *compilation,
                                                    const char *configName,
                                                    const void *configValue,
                                                    const size_t configValueSize) {
  if (!IsLoaded()) {
    MS_LOG(ERROR) << "nnrt wrapper is not created";
    return OH_NN_FAILED;
  }
  return nncore_compilationaddextensionconfig_func_(compilation, configName, configValue, configValueSize);
}
OH_NN_ReturnCode NNRTWrapper::NNCompilationBuild(OH_NNCompilation *compilation) {
  if (!IsLoaded()) {
    MS_LOG(ERROR) << "nnrt wrapper is not created";
    return OH_NN_FAILED;
  }
  return nncore_compilationbuild_func_(compilation);
}
OH_NNCompilation* NNRTWrapper::NNCompilationConstructWithOfflineModelBuffer(const void *modelBuffer, size_t modelSize) {
  if (!IsLoaded()) {
    MS_LOG(ERROR) << "nnrt wrapper is not created";
    return nullptr;
  }
  return nncore_compilationconstructwithofflinemodelbuffer_func_(modelBuffer, modelSize);
}
OH_NNCompilation* NNRTWrapper::NNCompilationConstruct(const OH_NNModel *model) {
  if (!IsLoaded()) {
    MS_LOG(ERROR) << "nnrt wrapper is not created";
    return nullptr;
  }
  return nncore_compilationconstruct_func_(model);
}
OH_NN_ReturnCode NNRTWrapper::NNExecutorRunSync(OH_NNExecutor *executor,
                                      NN_Tensor *inputTensor[],
                                      size_t inputCount,
                                      NN_Tensor *outputTensor[],
                                      size_t outputCount) {
  if (!IsLoaded()) {
    MS_LOG(ERROR) << "nnrt wrapper is not created";
    return OH_NN_FAILED;
  }
  return nncore_executorrunsync_func_(executor, inputTensor, inputCount, outputTensor, outputCount);
}
OH_NN_ReturnCode NNRTWrapper::NNTensorDestroy(NN_Tensor **tensor) {
  if (!IsLoaded()) {
    MS_LOG(ERROR) << "nnrt wrapper is not created";
    return OH_NN_FAILED;
  }
  return nncore_tensordestroy_func_(tensor);
}
OH_NN_ReturnCode NNRTWrapper::NNTensorDescDestroy(NN_TensorDesc **tensorDesc) {
  if (!IsLoaded()) {
    MS_LOG(ERROR) << "nnrt wrapper is not created";
    return OH_NN_FAILED;
  }
  return nncore_tensordescdestroy_func_(tensorDesc);
}
OH_NN_ReturnCode NNRTWrapper::NNExecutorGetInputCount(const OH_NNExecutor *executor, size_t *inputCount) {
  if (!IsLoaded()) {
    MS_LOG(ERROR) << "nnrt wrapper is not created";
    return OH_NN_FAILED;
  }
  return nncore_executorgetinputcount_func_(executor, inputCount);
}
NN_TensorDesc* NNRTWrapper::NNExecutorCreateInputTensorDesc(const OH_NNExecutor *executor, size_t index) {
  if (!IsLoaded()) {
    MS_LOG(ERROR) << "nnrt wrapper is not created";
    return nullptr;
  }
  return nncore_executorcreateinputtensordesc_func_(executor, index);
}
NN_Tensor* NNRTWrapper::NNTensorCreate(size_t deviceID, NN_TensorDesc *tensorDesc) {
  if (!IsLoaded()) {
    MS_LOG(ERROR) << "nnrt wrapper is not created";
    return nullptr;
  }
  return nncore_tensorcreate_func_(deviceID, tensorDesc);
}
OH_NN_ReturnCode NNRTWrapper::NNCompilationSetPriority(OH_NNCompilation *compilation, OH_NN_Priority priority) {
  if (!IsLoaded()) {
    MS_LOG(ERROR) << "nnrt wrapper is not created";
    return OH_NN_FAILED;
  }
  return nncore_compilationsetpriority_func_(compilation, priority);
}

OH_NN_ReturnCode NNRTWrapper::NNExecutorGetOutputCount(const OH_NNExecutor *executor, size_t *outputCount) {
  if (!IsLoaded()) {
    MS_LOG(ERROR) << "nnrt wrapper is not created";
    return OH_NN_FAILED;
  }
  return nncore_executorgetoutputcount_func_(executor, outputCount);
}
NN_TensorDesc* NNRTWrapper::NNExecutorCreateOutputTensorDesc(const OH_NNExecutor *executor, size_t index) {
  if (!IsLoaded()) {
    MS_LOG(ERROR) << "nnrt wrapper is not created";
    return nullptr;
  }
  return nncore_executorcreateoutputtensordesc_func_(executor, index);
}
void* NNRTWrapper::NNTensorGetDataBuffer(const NN_Tensor *tensor) {
  if (!IsLoaded()) {
    MS_LOG(ERROR) << "nnrt wrapper is not created";
    return nullptr;
  }
  return nncore_tensorgetdatabuffer_func_(tensor);
}
OH_NN_ReturnCode NNRTWrapper::NNTensorDescSetShape(NN_TensorDesc *tensorDesc, const int32_t *shape, size_t shapeLength) {
  if (!IsLoaded()) {
    MS_LOG(ERROR) << "nnrt wrapper is not created";
    return OH_NN_FAILED;
  }
  return nncore_tensordescsetshape_func_(tensorDesc, shape, shapeLength);
}
OH_NN_ReturnCode NNRTWrapper::NNTensorDescSetDataType(NN_TensorDesc *tensorDesc, OH_NN_DataType dataType) {
  if (!IsLoaded()) {
    MS_LOG(ERROR) << "nnrt wrapper is not created";
    
  }
  return nncore_tensordescsetdatatype_func_(tensorDesc, dataType);
}
OH_NN_ReturnCode NNRTWrapper::NNTensorDescSetFormat(NN_TensorDesc *tensorDesc, OH_NN_Format format) {
  if (!IsLoaded()) {
    MS_LOG(ERROR) << "nnrt wrapper is not created";
    return OH_NN_FAILED;
  }
  return nncore_tensordescsetformat_func_(tensorDesc, format);
}
OH_NN_ReturnCode NNRTWrapper::NNTensorDescSetName(NN_TensorDesc *tensorDesc, const char *name) {
  if (!IsLoaded()) {
    MS_LOG(ERROR) << "nnrt wrapper is not created";
    return OH_NN_FAILED;
  }
  return nncore_tensordescsetname_func_(tensorDesc, name);
}
NN_TensorDesc* NNRTWrapper::NNTensorDescCreate() {
  if (!IsLoaded()) {
    MS_LOG(ERROR) << "nnrt wrapper is not created";
    return nullptr;
  }
  return nncore_tensordesccreate_func_();
}

OH_NN_ReturnCode NNRTWrapper::NNDeviceGetAllDevicesID(const size_t **allDevicesID, uint32_t *deviceCount) {
  if (!IsLoaded()) {
    MS_LOG(ERROR) << "nnrt wrapper is not created";
    return OH_NN_FAILED;
  }
  return nncore_devicegetalldevicesid_func_(allDevicesID, deviceCount);
}

OH_NN_ReturnCode NNRTWrapper::NNDeviceGetType(size_t deviceID, OH_NN_DeviceType *deviceType) {
  if (!IsLoaded()) {
    MS_LOG(ERROR) << "nnrt wrapper is not created";
    return OH_NN_FAILED;
  }
  return nncore_devicegettype_func_(deviceID, deviceType);
}

OH_NN_ReturnCode NNRTWrapper::OH_NNTensorDesc_GetDataType(const NN_TensorDesc *tensorDesc, OH_NN_DataType *dataType) {
   if (!IsLoaded()) {
      MS_LOG(ERROR) << "nnrt wrapper is not created";
      return OH_NN_FAILED;
   }
   return nncore_tensordescgetdatatype_func_(tensorDesc, dataType);
}
}  // namespace mindspore
