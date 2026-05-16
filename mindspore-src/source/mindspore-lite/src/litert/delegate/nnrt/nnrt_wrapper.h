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

#ifndef MINDSPORE_LITE_NNRT_NNRTWRAPPER_H
#define MINDSPORE_LITE_NNRT_NNRTWRAPPER_H


#include "src/litert/cxx_api/model/model_impl.h"
#include "src/litert/delegate/nnrt/neural_network_runtime.h"
#include "src/litert/delegate/nnrt/neural_network_runtime_inner.h"
#include "src/litert/delegate/nnrt/neural_network_runtime_type.h"

#include "include/c_api/types_c.h"
#include "include/c_api/status_c.h"
#include "src/common/utils.h"

// nnrt function
typedef bool (*ModelHasCacheFunc)(const char *cacheDir, const char *modelName, uint32_t version);

typedef OH_NNModel* (*ModelConstructFunc)(void);

typedef OH_NN_ReturnCode (*ModelBuildFromLiteGraphFunc)(OH_NNModel *model, const void *liteGraph,
    const OH_NN_Extension *extensions, size_t extensionSize);

typedef void (*ModelDestroyFunc)(OH_NNModel **model);

typedef OH_NN_ReturnCode (*ModelGetAvailableOperationsFunc)(OH_NNModel *model, size_t deviceID, const bool **isSupported, uint32_t *opCount);

typedef OH_NN_ReturnCode (*ModelExecutorSetInputFunc)(OH_NNExecutor *executor, uint32_t inputIndex, const OH_NN_Tensor *tensor, const void *dataBuffer, size_t length);

typedef OH_NN_ReturnCode (*ModelExecutorSetOutputFunc)(OH_NNExecutor *executor, uint32_t outputIndex, void *dataBuffer, size_t length);

typedef OH_NN_ReturnCode (*ModelIsSupportAIPPFunc)(bool& support);

typedef OH_NN_ReturnCode (*ModelExecutorRunSyncWithAippFunc)(OH_NNExecutor *executor,
                                                         NN_Tensor *inputTensor[],
                                                         size_t inputCount,
                                                         NN_Tensor *outputTensor[],
                                                         size_t outputCount,
                                                         const char* aippString);

// nncore function
typedef void (*NNExecutorDestroyFunc)(OH_NNExecutor **executor);

typedef OH_NN_ReturnCode (*NNDeviceGetNameFunc)(size_t deviceID, const char **name);

typedef OH_NNCompilation* (*NNCompilationConstructForCacheFunc)();

typedef void (*NNCompilationDestroyFunc)(OH_NNCompilation **compilation);

typedef OH_NNExecutor* (*NNExecutorConstructFunc)(OH_NNCompilation *compilation);

typedef OH_NN_ReturnCode (*NNCompilationSetDeviceFunc)(OH_NNCompilation *compilation, size_t deviceID);

typedef OH_NN_ReturnCode (*NNCompilationSetPerformanceModeFunc)(OH_NNCompilation *compilation, OH_NN_PerformanceMode performanceMode);

typedef OH_NN_ReturnCode (*NNCompilationEnableFloat16Func)(OH_NNCompilation *compilation, bool enableFloat16);

typedef OH_NN_ReturnCode (*NNCompilationSetCacheFunc)(OH_NNCompilation *compilation, const char *cachePath, uint32_t version);

typedef OH_NN_ReturnCode (*NNCompilationAddExtensionConfigFunc)(OH_NNCompilation *compilation,
                                                                const char *configName,
                                                                const void *configValue,
                                                                const size_t configValueSize);

typedef OH_NN_ReturnCode (*NNCompilationBuildFunc)(OH_NNCompilation *compilation);

typedef OH_NNCompilation* (*NNCompilationConstructWithOfflineModelBufferFunc)(const void *modelBuffer, size_t modelSize);

typedef OH_NNCompilation* (*NNCompilationConstructFunc)(const OH_NNModel *model);

typedef OH_NN_ReturnCode (*NNExecutorRunSyncFunc)(OH_NNExecutor *executor,
                                                 NN_Tensor *inputTensor[],
                                                 size_t inputCount,
                                                 NN_Tensor *outputTensor[],
                                                 size_t outputCount);

typedef OH_NN_ReturnCode (*NNTensorDestroyFunc)(NN_Tensor **tensor);

typedef OH_NN_ReturnCode (*NNTensorDescDestroyFunc)(NN_TensorDesc **tensorDesc);

typedef OH_NN_ReturnCode (*NNExecutorGetInputCountFunc)(const OH_NNExecutor *executor, size_t *inputCount);

typedef NN_TensorDesc* (*NNExecutorCreateInputTensorDescFunc)(const OH_NNExecutor *executor, size_t index);

typedef NN_Tensor* (*NNTensorCreateFunc)(size_t deviceID, NN_TensorDesc *tensorDesc);

typedef OH_NN_ReturnCode (*NNCompilationSetPriorityFunc)(OH_NNCompilation *compilation, OH_NN_Priority priority);

typedef OH_NN_ReturnCode (*NNExecutorGetOutputCountFunc)(const OH_NNExecutor *executor, size_t *outputCount);

typedef NN_TensorDesc* (*NNExecutorCreateOutputTensorDescFunc)(const OH_NNExecutor *executor, size_t index);

typedef void* (*NNTensorGetDataBufferFunc)(const NN_Tensor *tensor);

typedef OH_NN_ReturnCode (*NNTensorDescSetShapeFunc)(NN_TensorDesc *tensorDesc, const int32_t *shape, size_t shapeLength);

typedef OH_NN_ReturnCode (*NNTensorDescSetDataTypeFunc)(NN_TensorDesc *tensorDesc, OH_NN_DataType dataType);

typedef OH_NN_ReturnCode (*NNTensorDescSetFormatFunc)(NN_TensorDesc *tensorDesc, OH_NN_Format format);

typedef OH_NN_ReturnCode (*NNTensorDescSetNameFunc)(NN_TensorDesc *tensorDesc, const char *name);

typedef NN_TensorDesc* (*NNTensorDescCreateFunc)();

typedef OH_NN_ReturnCode (*NNDeviceGetAllDevicesIDFunc)(const size_t **allDevicesID, uint32_t *deviceCount);

typedef OH_NN_ReturnCode (*NNDeviceGetTypeFunc)(size_t deviceID, OH_NN_DeviceType *deviceType);

namespace mindspore {
  class NNRTWrapper {
    public:

      static NNRTWrapper& GetInstance();
      NNRTWrapper(const NNRTWrapper&) = delete;
      NNRTWrapper& operator=(const NNRTWrapper&) = delete;

      void LoadLibrary();

      bool IsLoaded();
      // nnrt function
      bool HasCache(const char *cacheDir, const char *modelName, uint32_t version);
      
      OH_NNModel* Construct(void);
      OH_NN_ReturnCode BuildFromLiteGraph(OH_NNModel *model, const void *liteGraph, const OH_NN_Extension *extensions, size_t extensionSize);
      void Destroy(OH_NNModel **model);
      OH_NN_ReturnCode GetAvailableOperations(OH_NNModel *model, size_t deviceID, const bool **isSupported, uint32_t *opCount);
      OH_NN_ReturnCode ExecutorSetInput(OH_NNExecutor *executor, uint32_t inputIndex, const OH_NN_Tensor *tensor, const void *dataBuffer, size_t length);
      OH_NN_ReturnCode ExecutorSetOutput(OH_NNExecutor *executor, uint32_t outputIndex, void *dataBuffer, size_t length);
      OH_NN_ReturnCode IsSupportAIPP(bool& support);
      OH_NN_ReturnCode ExecutorRunSyncWithAipp(OH_NNExecutor *executor,
                                               NN_Tensor *inputTensor[],
                                               size_t inputCount,
                                               NN_Tensor *outputTensor[],
                                               size_t outputCount,
                                               const char* aippString);
      // nncore function
      void NNExecutorDestroy(OH_NNExecutor **executor);
      OH_NN_ReturnCode NNDeviceGetName(size_t deviceID, const char **name);
      OH_NNCompilation* NNCompilationConstructForCache();
      void NNCompilationDestroy(OH_NNCompilation **compilation);
      OH_NNExecutor* NNExecutorConstruct(OH_NNCompilation *compilation);
      OH_NN_ReturnCode NNCompilationSetDevice(OH_NNCompilation *compilation, size_t deviceID);
      OH_NN_ReturnCode NNCompilationSetPerformanceMode(OH_NNCompilation *compilation, OH_NN_PerformanceMode performanceMode);
      OH_NN_ReturnCode NNCompilationEnableFloat16(OH_NNCompilation *compilation, bool enableFloat16);
      OH_NN_ReturnCode NNCompilationSetCache(OH_NNCompilation *compilation, const char *cachePath, uint32_t version);
      OH_NN_ReturnCode NNCompilationAddExtensionConfig(OH_NNCompilation *compilation,
                                                          const char *configName,
                                                          const void *configValue,
                                                          const size_t configValueSize);
      OH_NN_ReturnCode NNCompilationBuild(OH_NNCompilation *compilation);
      OH_NNCompilation *NNCompilationConstructWithOfflineModelBuffer(const void *modelBuffer, size_t modelSize);
      OH_NNCompilation *NNCompilationConstruct(const OH_NNModel *model);
      OH_NN_ReturnCode NNExecutorRunSync(OH_NNExecutor *executor,
                                            NN_Tensor *inputTensor[],
                                            size_t inputCount,
                                            NN_Tensor *outputTensor[],
                                            size_t outputCount);
      OH_NN_ReturnCode NNTensorDestroy(NN_Tensor **tensor);
      OH_NN_ReturnCode NNTensorDescDestroy(NN_TensorDesc **tensorDesc);
      OH_NN_ReturnCode NNExecutorGetInputCount(const OH_NNExecutor *executor, size_t *inputCount);
      NN_TensorDesc *NNExecutorCreateInputTensorDesc(const OH_NNExecutor *executor, size_t index);
      NN_Tensor *NNTensorCreate(size_t deviceID, NN_TensorDesc *tensorDesc);
      OH_NN_ReturnCode NNCompilationSetPriority(OH_NNCompilation *compilation, OH_NN_Priority priority);

      OH_NN_ReturnCode NNExecutorGetOutputCount(const OH_NNExecutor *executor, size_t *outputCount);
      NN_TensorDesc *NNExecutorCreateOutputTensorDesc(const OH_NNExecutor *executor, size_t index);
      void *NNTensorGetDataBuffer(const NN_Tensor *tensor);
      OH_NN_ReturnCode NNTensorDescSetShape(NN_TensorDesc *tensorDesc, const int32_t *shape, size_t shapeLength);
      OH_NN_ReturnCode NNTensorDescSetDataType(NN_TensorDesc *tensorDesc, OH_NN_DataType dataType);
      OH_NN_ReturnCode NNTensorDescSetFormat(NN_TensorDesc *tensorDesc, OH_NN_Format format);
      OH_NN_ReturnCode NNTensorDescSetName(NN_TensorDesc *tensorDesc, const char *name);
      NN_TensorDesc *NNTensorDescCreate();

      OH_NN_ReturnCode NNDeviceGetAllDevicesID(const size_t **allDevicesID, uint32_t *deviceCount);
      OH_NN_ReturnCode NNDeviceGetType(size_t deviceID, OH_NN_DeviceType *deviceType);

    private:
      NNRTWrapper() = default;
      ~NNRTWrapper();
      void *nnrt_library_handle_ = nullptr;
      void *nncore_library_handle_ = nullptr;
      bool is_loaded_ = false;
      // nnrt function pointer
      ModelHasCacheFunc nnrt_hascache_func_ = nullptr;
      ModelConstructFunc nnrt_construct_func_ = nullptr;
      ModelBuildFromLiteGraphFunc nnrt_buildfromlitegraph_func_ = nullptr;
      ModelDestroyFunc nnrt_destroy_func_ = nullptr;
      ModelGetAvailableOperationsFunc nnrt_getavailableoperations_func_ = nullptr;
      ModelExecutorSetInputFunc nnrt_executorsetinput_func_ = nullptr;
      ModelExecutorSetOutputFunc nnrt_executorsetoutput_func_ = nullptr;
      ModelIsSupportAIPPFunc nnrt_issupportaipp_func_ = nullptr;
      ModelExecutorRunSyncWithAippFunc nnrt_executorrunsyncwithaipp_func_ = nullptr;
      // nncore function pointer
      NNExecutorDestroyFunc nncore_executordestroy_func_ = nullptr;
      NNDeviceGetNameFunc nncore_devicegetname_func_ = nullptr;
      NNCompilationConstructForCacheFunc nncore_compilationconstructforcache_func_ = nullptr;
      NNCompilationDestroyFunc nncore_compilationdestroy_func_ = nullptr;
      NNExecutorConstructFunc nncore_executorconstruct_func_ = nullptr;
      NNCompilationSetDeviceFunc nncore_compilationsetdevice_func_ = nullptr;
      NNCompilationSetPerformanceModeFunc nncore_compilationsetperformancemode_func_ = nullptr;
      NNCompilationEnableFloat16Func nncore_compilationenablefloat16_func_ = nullptr;
      NNCompilationSetCacheFunc nncore_compilationsetcache_func_ = nullptr;
      NNCompilationAddExtensionConfigFunc nncore_compilationaddextensionconfig_func_ = nullptr;
      NNCompilationBuildFunc nncore_compilationbuild_func_ = nullptr;
      NNCompilationConstructWithOfflineModelBufferFunc nncore_compilationconstructwithofflinemodelbuffer_func_ = nullptr;
      NNCompilationConstructFunc nncore_compilationconstruct_func_ = nullptr;
      NNExecutorRunSyncFunc nncore_executorrunsync_func_ = nullptr;
      NNTensorDestroyFunc nncore_tensordestroy_func_ = nullptr;
      NNTensorDescDestroyFunc nncore_tensordescdestroy_func_ = nullptr;
      NNExecutorGetInputCountFunc nncore_executorgetinputcount_func_ = nullptr;
      NNExecutorCreateInputTensorDescFunc nncore_executorcreateinputtensordesc_func_ = nullptr;
      NNTensorCreateFunc nncore_tensorcreate_func_ = nullptr;
      NNCompilationSetPriorityFunc nncore_compilationsetpriority_func_ = nullptr;

      NNExecutorGetOutputCountFunc nncore_executorgetoutputcount_func_ = nullptr;
      NNExecutorCreateOutputTensorDescFunc nncore_executorcreateoutputtensordesc_func_ = nullptr;
      NNTensorGetDataBufferFunc nncore_tensorgetdatabuffer_func_ = nullptr;
      NNTensorDescSetShapeFunc nncore_tensordescsetshape_func_ = nullptr;
      NNTensorDescSetDataTypeFunc nncore_tensordescsetdatatype_func_ = nullptr;
      NNTensorDescSetFormatFunc nncore_tensordescsetformat_func_ = nullptr;
      NNTensorDescSetNameFunc nncore_tensordescsetname_func_ = nullptr;
      NNTensorDescCreateFunc nncore_tensordesccreate_func_ = nullptr;
      NNDeviceGetAllDevicesIDFunc nncore_devicegetalldevicesid_func_ = nullptr;
      NNDeviceGetTypeFunc nncore_devicegettype_func_ = nullptr;
  };
} // namespace mindspore

#endif //MINDSPORE_LITE_NNRT_NNRTWRAPPER_H
