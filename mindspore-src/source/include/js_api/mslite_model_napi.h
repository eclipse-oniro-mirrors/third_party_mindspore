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
#ifndef MINDSPORE_INCLUDE_JS_API_MSLITE_MODEL_NAPI_H
#define MINDSPORE_INCLUDE_JS_API_MSLITE_MODEL_NAPI_H

#include <memory>
#include <mutex>
#include "include/api/model.h"
#include "include/api/context.h"
#include "include/api/serialization.h"
#include "include/api/cell.h"
#include "common_napi.h"
#include "mslite_model_callback_napi.h"
#include "napi/native_api.h"
#include "napi/native_node_api.h"
#include "include/js_api/common_napi.h"

namespace mindspore {
static const std::map<std::string, TensorFormat> tensorFormatMap = {
    {"DEFAULT_FORMAT", TENSOR_DEFAULT_FORMAT},
    {"NCHW", TENSOR_NCHW},
    {"NHWC", TENSOR_NHWC},
    {"NHWC4", TENSOR_NHWC4},
    {"HWKC", TENSOR_HWKC},
    {"HWCK", TENSOR_HWCK},
    {"KCHW", TENSOR_KCHW}
};
static const std::map<std::string, TensorDataType> tensorDataTypeMap = {
    {"TYPE_UNKNOWN", TENSOR_UNKNOWN},
    {"NUMBER_TYPE_INT8", TENSOR_INT8},
    {"NUMBER_TYPE_INT16", TENSOR_INT16},
    {"NUMBER_TYPE_INT32", TENSOR_INT32},
    {"NUMBER_TYPE_INT64", TENSOR_INT64},
    {"NUMBER_TYPE_UINT8", TENSOR_UINT8},
    {"NUMBER_TYPE_UINT16", TENSOR_UINT16},
    {"NUMBER_TYPE_UINT32", TENSOR_UINT32},
    {"NUMBER_TYPE_UINT64", TENSOR_UINT64},
    {"NUMBER_TYPE_FLOAT16", TENSOR_FLOAT16},
    {"NUMBER_TYPE_FLOAT32", TENSOR_FLOAT32},
    {"NUMBER_TYPE_FLOAT64", TENSOR_FLOAT64}
};
static const std::map<std::string, ContextThreadAffinityMode> contextThreadAffinityModeMap = {
    {"NO_AFFINITIES", CONTEXT_AFFINITY_MODE},
    {"BIG_CORES_FIRST", CONTEXT_BIG_CORES_FIRST},
    {"LITTLE_CORES_FIRST", CONTEXT_LITTLE_CORES_FIRST},
};

static const std::map<std::string, ContextQuantizationType> contextQuantizationTypeMap = {
  {"NO_QUANT", NO_QUANT},
  {"WEIGHT_QUANT", WEIGHT_QUANT},
  {"FULL_QUANT", FULL_QUANT},
};

static const std::map<std::string, ContextOptimizationLevel> contextOptimizationLevelTypeMap = {
  {"O0", O0},
  {"O2", O2},
  {"O3", O3},
  {"AUTO", AUTO},
};

static const std::map<std::string, ContextPerformanceMode> contextPerformanceModeTypeMap = {
  {"PERFORMANCE_NONE", PERFORMANCE_NONE},
  {"PERFORMANCE_LOW", PERFORMANCE_LOW},
  {"PERFORMANCE_MEDIUM", PERFORMANCE_MEDIUM},
  {"PERFORMANCE_HIGH", PERFORMANCE_HIGH},
  {"PERFORMANCE_EXTREME", PERFORMANCE_EXTREME}
};

static const std::map<std::string, ContextPriority> contextPriorityTypeMap = {
  {"PRIORITY_NONE", PRIORITY_NONE},
  {"PRIORITY_LOW", PRIORITY_LOW},
  {"PRIORITY_MEDIUM", PRIORITY_MEDIUM},
  {"PRIORITY_HIGH", PRIORITY_HIGH},
};

static const std::map<std::string, ContextNnrtDeviceType> contextNnrtDeviceTypeTypeMap = {
  {"NNRTDEVICE_OTHERS", NNRTDEVICE_OTHERS},
  {"NNRTDEVICE_CPU", NNRTDEVICE_CPU},
  {"NNRTDEVICE_GPU", NNRTDEVICE_GPU},
  {"NNRTDEVICE_ACCELERATOR", NNRTDEVICE_ACCELERATOR},
};

class MSLiteModelNapi {
 public:
  MSLiteModelNapi();
  ~MSLiteModelNapi();

  static napi_value Init(napi_env env, napi_value exports);
  std::shared_ptr<mindspore::Model> native_model_ = nullptr;

 private:
  struct MSLiteModelAsyncContext {
    napi_async_work work;
    napi_deferred deferred = nullptr;
    napi_ref callbackRef = nullptr;
    int32_t status = SUCCESS;
    MSLiteModelNapi *lite_model = nullptr;
    ModelInfo model_info;
    ContextInfo context;

    MSLiteModelAsyncContext() {
      // setting context default value
      context.target.push_back("cpu");
      context.cpu_device.thread_num = 2;
      context.cpu_device.thread_affinity_mode = 0;
      context.cpu_device.precision_mode = "enforce_fp32";
    }
  };
  static napi_value Constructor(napi_env env, napi_callback_info info);
  static void Finalize(napi_env env, void *nativeObject, void *finalize);
  static napi_value LoadMSLiteModelFromFile(napi_env env, napi_callback_info info);
  static napi_value LoadMSLiteModelFromBuffer(napi_env env, napi_callback_info info);
  static napi_value LoadMSLiteModelFromFd(napi_env env, napi_callback_info info);
  static napi_value LoadMSLiteTrainModelFromFile(napi_env env, napi_callback_info info);
  static napi_value LoadMSLiteTrainModelFromBuffer(napi_env env, napi_callback_info info);
  static napi_value LoadMSLiteTrainModelFromFd(napi_env env, napi_callback_info info);
  static napi_value GetInputs(napi_env env, napi_callback_info info);
  static napi_value Resize(napi_env env, napi_callback_info info);
  static napi_value PredictAsync(napi_env env, napi_callback_info info);
  static napi_value RunStep(napi_env env, napi_callback_info info);
  static napi_value GetWeights(napi_env env, napi_callback_info info);
  static napi_value UpdateWeights(napi_env env, napi_callback_info info);
  static napi_value SetupVirtualBatch(napi_env env, napi_callback_info info);
  static napi_value ExportModel(napi_env env, napi_callback_info info);
  static napi_value ExportWeightsCollaborateWithMicro(napi_env env, napi_callback_info info);
  static napi_value GetTrainMode(napi_env env, napi_callback_info info);
  static napi_value SetTrainMode(napi_env env, napi_callback_info info);
  static napi_value GetLearningRate(napi_env env, napi_callback_info info);
  static napi_value SetLearningRate(napi_env env, napi_callback_info info);
  static int32_t ParseModelInfo(napi_env env, napi_value root, ModelInfo &model_info);
  static int32_t ParseContextInfo(napi_env env, napi_value root, ContextInfo &info);
  static int32_t ParseTrainCfgInfo(napi_env env, napi_value root, TrainConfig &cfg);
  static void GetMSLiteModelAsyncCallbackComplete(napi_env env, napi_status status, void *data);
  static void PredictAsyncCallbackComplete(napi_env env, napi_status status, void *data);
  static napi_value CreateMSLiteModelWrapper(napi_env env, MSLiteModelAsyncContext *async_context);
  static void CommonCallbackRoutine(napi_env env, MSLiteModelAsyncContext *&asyncContext, const napi_value &valueParam);
  static std::shared_ptr<mindspore::Model> CreateModel(ModelInfo *model_info_ptr, ContextInfo *contex_ptr);
  static std::shared_ptr<mindspore::Model> CreateTrainModel(ModelInfo *model_info_ptr, ContextInfo *contex_ptr);
  static int32_t GetCpuDeviceInfo(napi_env env, napi_value args, ContextInfo &context);
  static int32_t GetNNRTDeviceInfo(napi_env env, napi_value args, ContextInfo &context);
  static int32_t GetDeviceInfoContext(ContextInfo *context_info_ptr,
                                      std::vector<std::shared_ptr<DeviceInfoContext>> &device_infos);
  static int32_t SetTensorData(napi_env env, napi_value thisVar, napi_value argv,
                               MSLiteModelAsyncContext *async_context);
  static napi_status AddNamedProperty(napi_env env, napi_value object, const std::string name, int32_t enumValue);
  static napi_value GetAllNnrtDeviceDescs(napi_env env, napi_callback_info info);
  static napi_value CreateFormatObject(napi_env env);
  static napi_value CreateDataTypeObject(napi_env env);
  static napi_value CreateThreadAffinityModeObject(napi_env env);
  static napi_value CreateQuantizationTypeObject(napi_env env);
  static napi_value CreateOptimizationLevelObject(napi_env env);
  static napi_value CreatePerformanceModeObject(napi_env env);
  static napi_value CreatePriorityObject(napi_env env);
  static napi_value CreateNnrtDeviceTypeObject(napi_env env);


  static thread_local napi_ref constructor_;
  napi_env env_ = nullptr;
  static napi_ref tensorFormat_;
  static napi_ref tensorDataType_;
  static napi_ref contextThreadAffinityMode_;
  static napi_ref contextQuantizationType_;
  static napi_ref contextOptimizationLevel_;
  static napi_ref contextPerformanceMode_;
  static napi_ref contextPriority_;
  static napi_ref contextNnrtDeviceType_;

  static ModelInfo *model_info_;
  static ContextInfo *context_;
  static std::mutex create_mutex_;
};
}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_JS_API_MSLITE_MODEL_NAPI_H