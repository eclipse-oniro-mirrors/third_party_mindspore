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
#include "include/js_api/mslite_model_napi.h"
#include <climits>
#include <algorithm>
#include <random>
#include <cstring>
#include <memory>
#include <map>
#include <vector>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include "include/js_api/mstensor_napi.h"
#include "include/js_api/common_napi.h"
#include "include/js_api/ms_parameters_napi.h"
#include "include/js_api/ms_errors.h"
#include "include/js_api/mslite_model_callback_napi.h"
#include "src/common/log.h"
#include "mindspore-lite/src/common/log.h"
#include "include/c_api/model_c.h"
#include "include/c_api/context_c.h"
#include "include/c_api/types_c.h"
#include "include/js_api/nnrt_device_desc_napi.h"

namespace mindspore {
thread_local napi_ref MSLiteModelNapi::constructor_ = nullptr;
ModelInfo *MSLiteModelNapi::model_info_ = nullptr;
ContextInfo *MSLiteModelNapi::context_ = nullptr;
std::mutex MSLiteModelNapi::create_mutex_;
napi_ref MSLiteModelNapi::tensorFormat_ = nullptr;
napi_ref MSLiteModelNapi::tensorDataType_ = nullptr;
napi_ref MSLiteModelNapi::contextThreadAffinityMode_ = nullptr;
napi_ref MSLiteModelNapi::contextQuantizationType_ = nullptr;
napi_ref MSLiteModelNapi::contextOptimizationLevel_ = nullptr;
napi_ref MSLiteModelNapi::contextPerformanceMode_ = nullptr;
napi_ref MSLiteModelNapi::contextPriority_ = nullptr;
napi_ref MSLiteModelNapi::contextNnrtDeviceType_ = nullptr;

#define GET_PARAMS(env, info, num) \
  size_t argc = num;               \
  napi_value argv[num] = {0};      \
  napi_value thisVar = nullptr;    \
  void *data;                      \
  napi_get_cb_info(env, info, &argc, argv, &thisVar, &data)

namespace {
const int ARGS_ONE = 1;
const int ARGS_TWO = 2;
const int ARGS_THREE = 3;
const int ARGS_FOUR = 4;

const int PARAM0 = 0;
const int PARAM1 = 1;
const int PARAM2 = 2;
const int PARAM3 = 3;
const int PARAM4 = 4;
const int UNSET_VALUE = -1;

const int SIZE = 100;

const std::string CLASS_NAME = "Model";

const std::unordered_map<std::string, DeviceType> kDeviceTypes{
  {"cpu", kCPU},
  {"nnrt", kNNRt},
  {"gpu", kGPU},
};
}  // namespace

MSLiteModelNapi::MSLiteModelNapi() : native_model_(nullptr), env_(nullptr) {
  MS_LOG(INFO) << "MSLiteModelNapi Instances create.";
}

MSLiteModelNapi::~MSLiteModelNapi() {
  native_model_ = nullptr;
  env_ = nullptr;
  MS_LOG(INFO) << "MSLiteModelNapi Instances destroy.";
}

void MSLiteModelNapi::Finalize(napi_env env, void *nativeObject, void *finalize) {
  (void)env;
  (void)finalize;
  if (nativeObject != nullptr) {
    // delete nativeObject
    auto obj = static_cast<MSLiteModelNapi *>(nativeObject);
    delete obj;
    obj = nullptr;
  }
  MS_LOG(INFO) << "Finalize success";
}

napi_value MSLiteModelNapi::Init(napi_env env, napi_value exports) {
  napi_property_descriptor properties[] = {
    DECLARE_NAPI_FUNCTION("getInputs", GetInputs),
    DECLARE_NAPI_FUNCTION("resize", Resize),
    DECLARE_NAPI_FUNCTION("predict", PredictAsync),
    DECLARE_NAPI_FUNCTION("runStep", RunStep),
    DECLARE_NAPI_FUNCTION("getWeights", GetWeights),
    DECLARE_NAPI_FUNCTION("updateWeights", UpdateWeights),
    DECLARE_NAPI_FUNCTION("setupVirtualBatch", SetupVirtualBatch),
    DECLARE_NAPI_FUNCTION("exportModel", ExportModel),
    DECLARE_NAPI_FUNCTION("exportWeightsCollaborateWithMicro", ExportWeightsCollaborateWithMicro),
    DECLARE_NAPI_GETTER_SETTER("trainMode", GetTrainMode, SetTrainMode),
    DECLARE_NAPI_GETTER_SETTER("learningRate", GetLearningRate, SetLearningRate),
    };

  napi_property_descriptor staticProperty[] = {
    DECLARE_NAPI_STATIC_FUNCTION("loadModelFromFile", LoadMSLiteModelFromFile),
    DECLARE_NAPI_STATIC_FUNCTION("loadModelFromBuffer", LoadMSLiteModelFromBuffer),
    DECLARE_NAPI_STATIC_FUNCTION("loadModelFromFd", LoadMSLiteModelFromFd),
    DECLARE_NAPI_STATIC_FUNCTION("loadTrainModelFromFile", LoadMSLiteTrainModelFromFile),
    DECLARE_NAPI_STATIC_FUNCTION("loadTrainModelFromBuffer", LoadMSLiteTrainModelFromBuffer),
    DECLARE_NAPI_STATIC_FUNCTION("loadTrainModelFromFd", LoadMSLiteTrainModelFromFd),
    DECLARE_NAPI_STATIC_FUNCTION("getAllNNRTDeviceDescriptions", GetAllNnrtDeviceDescs),
    DECLARE_NAPI_PROPERTY("Format", CreateFormatObject(env)),
    DECLARE_NAPI_PROPERTY("DataType", CreateDataTypeObject(env)),
    DECLARE_NAPI_PROPERTY("ThreadAffinityMode", CreateThreadAffinityModeObject(env)),
    DECLARE_NAPI_PROPERTY("QuantizationType", CreateQuantizationTypeObject(env)),
    DECLARE_NAPI_PROPERTY("OptimizationLevel", CreateOptimizationLevelObject(env)),
    DECLARE_NAPI_PROPERTY("PerformanceMode", CreatePerformanceModeObject(env)),
    DECLARE_NAPI_PROPERTY("Priority", CreatePriorityObject(env)),
    DECLARE_NAPI_PROPERTY("NNRTDeviceType", CreateNnrtDeviceTypeObject(env)),
  };

  napi_value constructor = nullptr;
  napi_status status = napi_define_class(env, CLASS_NAME.c_str(), NAPI_AUTO_LENGTH, Constructor, nullptr,
                                         sizeof(properties) / sizeof(properties[0]), properties, &constructor);
  if (status != napi_ok) {
    MS_LOG(ERROR) << "Failed to define MSLiteModel class";
    return nullptr;
  }

  status = napi_create_reference(env, constructor, REFERENCE_CREATION_COUNT, &constructor_);
  if (status != napi_ok) {
    MS_LOG(ERROR) << "Failed to create reference of constructor";
    return nullptr;
  }

  status = napi_set_named_property(env, exports, CLASS_NAME.c_str(), constructor);
  if (status != napi_ok) {
    MS_LOG(ERROR) << "Failed to set constructor";
    return nullptr;
  }

  status = napi_define_properties(env, exports, sizeof(staticProperty) / sizeof(staticProperty[0]), staticProperty);
  if (status != napi_ok) {
    MS_LOG(ERROR) << "Failed to define static function";
    return nullptr;
  }

  MS_LOG(INFO) << "init success";
  return exports;
}

napi_value MSLiteModelNapi::CreateFormatObject(napi_env env)
{
  napi_value result = nullptr;
  napi_status status;
  std::string propName;
  int32_t refCount = 1;

  status = napi_create_object(env, &result);
  if (status == napi_ok) {
    for (auto &iter : tensorFormatMap) {
      propName = iter.first;
      status = AddNamedProperty(env, result, propName, iter.second);
      if (status != napi_ok) {
        MS_LOG(ERROR) << "Failed to add named prop in CreateFormatObject.";
        break;
      }
      propName.clear();
    }
    if (status == napi_ok) {
      status = napi_create_reference(env, result, refCount, &tensorFormat_);
      if (status == napi_ok) {
        return result;
      }
    }
  }
  MS_LOG(ERROR) << "CreateFormatObject is Failed!";
  napi_get_undefined(env, &result);
  return result;
}

napi_value MSLiteModelNapi::CreateDataTypeObject(napi_env env) {
  napi_value result = nullptr;
  napi_status status;
  std::string propName;
  int32_t refCount = 1;

  status = napi_create_object(env, &result);
  if (status == napi_ok) {
    for (auto &iter : tensorDataTypeMap) {
      propName = iter.first;
      status = AddNamedProperty(env, result, propName, iter.second);
      if (status != napi_ok) {
        MS_LOG(ERROR) << "Failed to add named prop in CreateDataTypeObject.";
        break;
      }
      propName.clear();
    }
    if (status == napi_ok) {
      status = napi_create_reference(env, result, refCount, &tensorDataType_);
      if (status == napi_ok) {
        return result;
      }
    }
  }
  MS_LOG(ERROR) << "CreateDataTypeObject is Failed!";
  napi_get_undefined(env, &result);
  return result;
}

napi_value MSLiteModelNapi::CreateThreadAffinityModeObject(napi_env env) {
  napi_value result = nullptr;
  napi_status status;
  std::string propName;
  int32_t refCount = 1;

  status = napi_create_object(env, &result);
  if (status == napi_ok) {
    for (auto &iter : contextThreadAffinityModeMap) {
      propName = iter.first;
      status = AddNamedProperty(env, result, propName, iter.second);
      if (status != napi_ok) {
        MS_LOG(ERROR) << "Failed to add named prop in CreateThreadAffinityModeObject.";
        break;
      }
      propName.clear();
    }
    if (status == napi_ok) {
      status = napi_create_reference(env, result, refCount, &contextThreadAffinityMode_);
      if (status == napi_ok) {
        return result;
      }
    }
  }
  MS_LOG(ERROR) << "CreateThreadAffinityModeObject is Failed!";
  napi_get_undefined(env, &result);
  return result;
}

napi_value MSLiteModelNapi::CreateQuantizationTypeObject(napi_env env) {
  napi_value result = nullptr;
  napi_status status;
  std::string propName;
  int32_t refCount = 1;

  status = napi_create_object(env, &result);
  if (status == napi_ok) {
    for (auto &iter : contextQuantizationTypeMap) {
      propName = iter.first;
      status = AddNamedProperty(env, result, propName, iter.second);
      if (status != napi_ok) {
        MS_LOG(ERROR) << "Failed to add named prop in CreateQuantizationTypeObject.";
        break;
      }
      propName.clear();
    }
    if (status == napi_ok) {
      status = napi_create_reference(env, result, refCount, &contextQuantizationType_);
      if (status == napi_ok) {
        return result;
      }
    }
  }
  MS_LOG(ERROR) << "CreateQuantizationTypeObject is Failed!";
  napi_get_undefined(env, &result);
  return result;
}

napi_value MSLiteModelNapi::CreateOptimizationLevelObject(napi_env env) {
  napi_value result = nullptr;
  napi_status status;
  std::string propName;
  int32_t refCount = 1;

  status = napi_create_object(env, &result);
  if (status == napi_ok) {
    for (auto &iter : contextOptimizationLevelTypeMap) {
      propName = iter.first;
      status = AddNamedProperty(env, result, propName, iter.second);
      if (status != napi_ok) {
        MS_LOG(ERROR) << "Failed to add named prop in CreateOptimizationLevelObject.";
        break;
      }
      propName.clear();
    }
    if (status == napi_ok) {
      status = napi_create_reference(env, result, refCount, &contextOptimizationLevel_);
      if (status == napi_ok) {
        return result;
      }
    }
  }
  MS_LOG(ERROR) << "CreateOptimizationLevelObject is Failed!";
  napi_get_undefined(env, &result);
  return result;
}

napi_value MSLiteModelNapi::CreatePerformanceModeObject(napi_env env) {
  napi_value result = nullptr;
  napi_status status;
  std::string propName;
  int32_t refCount = 1;

  status = napi_create_object(env, &result);
  if (status == napi_ok) {
    for (auto &iter : contextPerformanceModeTypeMap) {
      propName = iter.first;
      status = AddNamedProperty(env, result, propName, iter.second);
      if (status != napi_ok) {
        MS_LOG(ERROR) << "Failed to add named prop in CreatePerformanceModeObject.";
        break;
      }
      propName.clear();
    }
    if (status == napi_ok) {
      status = napi_create_reference(env, result, refCount, &contextPerformanceMode_);
      if (status == napi_ok) {
        return result;
      }
    }
  }
  MS_LOG(ERROR) << "CreatePerformanceModeObject is Failed!";
  napi_get_undefined(env, &result);
  return result;
}

napi_value MSLiteModelNapi::CreatePriorityObject(napi_env env) {
  napi_value result = nullptr;
  napi_status status;
  std::string propName;
  int32_t refCount = 1;

  status = napi_create_object(env, &result);
  if (status == napi_ok) {
    for (auto &iter : contextPriorityTypeMap) {
      propName = iter.first;
      status = AddNamedProperty(env, result, propName, iter.second);
      if (status != napi_ok) {
        MS_LOG(ERROR) << "Failed to add named prop in CreatePriorityObject.";
        break;
      }
      propName.clear();
    }
    if (status == napi_ok) {
      status = napi_create_reference(env, result, refCount, &contextPriority_);
      if (status == napi_ok) {
        return result;
      }
    }
  }
  MS_LOG(ERROR) << "CreatePriorityObject is Failed!";
  napi_get_undefined(env, &result);
  return result;
}

napi_value MSLiteModelNapi::CreateNnrtDeviceTypeObject(napi_env env) {
  napi_value result = nullptr;
  napi_status status;
  std::string propName;
  int32_t refCount = 1;

  status = napi_create_object(env, &result);
  if (status == napi_ok) {
    for (auto &iter : contextNnrtDeviceTypeTypeMap) {
      propName = iter.first;
      status = AddNamedProperty(env, result, propName, iter.second);
      if (status != napi_ok) {
        MS_LOG(ERROR) << "Failed to add named prop in CreateNnrtDeviceTypeObject.";
        break;
      }
      propName.clear();
    }
    if (status == napi_ok) {
      status = napi_create_reference(env, result, refCount, &contextNnrtDeviceType_);
      if (status == napi_ok) {
        return result;
      }
    }
  }
  MS_LOG(ERROR) << "CreateNnrtDeviceTypeObject is Failed!";
  napi_get_undefined(env, &result);
  return result;
}

napi_status MSLiteModelNapi::AddNamedProperty(napi_env env, napi_value object, const std::string name,
                                              int32_t enumValue) {
  napi_status status;
  napi_value enumNapiValue;

  status = napi_create_int32(env, enumValue, &enumNapiValue);
  if (status == napi_ok) {
    status = napi_set_named_property(env, object, name.c_str(), enumNapiValue);
  }
  return status;
}

napi_value MSLiteModelNapi::GetAllNnrtDeviceDescs(napi_env env, napi_callback_info info) {
  size_t num;
  napi_value jsResult = nullptr;
  NNRTDeviceDesc *devices = OH_AI_GetAllNNRTDeviceDescs(&num);
  if (devices == nullptr) {
    MS_LOG(ERROR) << "Get all nnrt devices error, may nnrt is not supported.";
    OH_AI_DestroyAllNNRTDeviceDescs(&devices);
    return jsResult;
  }

  MS_LOG(INFO) << "all nnrt devices size: " << num;
  napi_create_array_with_length(env, num, &jsResult);
  for (size_t i = 0; i < num; i++) {
    NnrtDeviceDesc nnrt_device;
    NNRTDeviceDesc *nnrt_device_desc = OH_AI_GetElementOfNNRTDeviceDescs(devices, i);
    nnrt_device.name.assign(OH_AI_GetNameFromNNRTDeviceDesc(nnrt_device_desc));
    size_t id = OH_AI_GetDeviceIdFromNNRTDeviceDesc(nnrt_device_desc);
    nnrt_device.id = id;
    nnrt_device.type = static_cast<ContextNnrtDeviceType>(OH_AI_GetTypeFromNNRTDeviceDesc(nnrt_device_desc));
    auto status = napi_set_element(env, jsResult, i, NnrtDeviceDescNapi::NewInstance(env, nnrt_device));
    if (status != napi_ok) {
      MS_LOG(ERROR) << "napi_set_element failed! code: " << status;
      OH_AI_DestroyAllNNRTDeviceDescs(&devices);
      return jsResult;
    }
  }
  MS_LOG(INFO) << "get All nnrt devices success!";
  OH_AI_DestroyAllNNRTDeviceDescs(&devices);
  return jsResult;
}

std::shared_ptr<mindspore::Model> MSLiteModelNapi::CreateModel(ModelInfo *model_info_ptr,
                                                               ContextInfo *context_info_ptr) {
  if (context_info_ptr == nullptr) {
    MS_LOG(ERROR) << "context_info_ptr is nullptr.";
    return nullptr;
  }
  // create and init context
  std::string s;
  for (const auto &device_name : context_info_ptr->target) {
    s += device_name + " ";
  }
  MS_LOG(DEBUG) << "target device: " << s.c_str();

  auto context = std::make_shared<mindspore::Context>();
  if (context == nullptr) {
    MS_LOG(ERROR) << "Failed to new context.";
    return nullptr;
  }

  auto &device_infos = context->MutableDeviceInfo();
  if (context_info_ptr->target.empty()) {
    MS_LOG(ERROR) << "context is empty.";
    return nullptr;
  }
  if (GetDeviceInfoContext(context_info_ptr, device_infos) != SUCCESS) {
    MS_LOG(ERROR) << "Create context failed.";
    return nullptr;
  }
  context->SetThreadNum(context_info_ptr->cpu_device.thread_num);
  MS_LOG(DEBUG) << "current thread num is : " << context->GetThreadNum();

  switch (model_info_ptr->mode) {
    case kBuffer: {
      MS_LOG(DEBUG) << "input model buffer, model_buffer_total: " << model_info_ptr->model_buffer_total;
      if (model_info_ptr->model_buffer_data == nullptr || model_info_ptr->model_buffer_total <= 0) {
        MS_LOG(ERROR) << "Failed to build model.";
        return nullptr;
      }
      std::shared_ptr<mindspore::Model> model_ptr = std::make_shared<mindspore::Model>();
      if (model_ptr == nullptr) {
        MS_LOG(ERROR) << "Failed to new mindspore::model.";
        return nullptr;
      }
      auto ret = model_ptr->Build(model_info_ptr->model_buffer_data, model_info_ptr->model_buffer_total,
                                  mindspore::kMindIR, context);
      if (ret == mindspore::kSuccess) {
        MS_LOG(INFO) << "Build model from buffer success.";
        return model_ptr;
      }
      break;
    }
    case kPath: {
      MS_LOG(DEBUG) << "input model path, model_buffer_total: " << model_info_ptr->model_path.c_str();
      std::shared_ptr<mindspore::Model> model_ptr = std::make_shared<mindspore::Model>();
      if (model_ptr == nullptr) {
        MS_LOG(ERROR) << "Failed to new mindspore::model.";
        return nullptr;
      }
      auto ret = model_ptr->Build(model_info_ptr->model_path, mindspore::kMindIR, context);
      if (ret == mindspore::kSuccess) {
        MS_LOG(INFO) << "Build model from path success.";
        return model_ptr;
      }
      return nullptr;
    }
    case kFD: {
      MS_LOG(DEBUG) << "input model fd:" << model_info_ptr->model_fd
                    << ", model_buffer_total: " << model_info_ptr->model_buffer_total;
      std::shared_ptr<mindspore::Model> model_ptr = std::make_shared<mindspore::Model>();
      if (model_ptr == nullptr) {
        MS_LOG(ERROR) << "Failed to new mindspore::model.";
        return nullptr;
      }
      auto ret = model_ptr->Build(model_info_ptr->model_buffer_data, model_info_ptr->model_buffer_total,
                                  mindspore::kMindIR, context);

      (void)munmap(model_info_ptr->model_buffer_data, model_info_ptr->model_buffer_total);
      if (ret == mindspore::kSuccess) {
        MS_LOG(INFO) << "Build model from fd success.";
        return model_ptr;
      }

      break;
    }
    default: {
      MS_LOG(ERROR) << "Invalid model mode.";
    }
  }
  MS_LOG(ERROR) << "Build model failed.";
  return nullptr;
}

std::shared_ptr<mindspore::Model> MSLiteModelNapi::CreateTrainModel(ModelInfo *model_info_ptr,
                                                                    ContextInfo *context_info_ptr) {
  // create and init context
  std::string s;
  for (const auto &device_name : context_info_ptr->target) {
    s += device_name + " ";
  }
  MS_LOG(DEBUG) << "target device: " << s.c_str();

  auto context = std::make_shared<mindspore::Context>();
  if (context == nullptr) {
    MS_LOG(ERROR) << "Failed to new context.";
    return nullptr;
  }

  auto &device_infos = context->MutableDeviceInfo();
  if (context_info_ptr->target.empty()) {
    MS_LOG(ERROR) << "context is empty.";
    return nullptr;
  }
  if (GetDeviceInfoContext(context_info_ptr, device_infos) != SUCCESS) {
    MS_LOG(ERROR) << "Create context failed.";
    return nullptr;
  }

  auto train_cfg = std::make_shared<TrainCfg>();
  std::vector<std::string> loss_names;
  for (const auto &name : train_cfg->GetLossName()) {
      loss_names.push_back(name);
  }
  for (const auto &name : context_info_ptr->train_cfg.loss_names) {
      loss_names.push_back(name);
  }
  train_cfg->SetLossName(loss_names);
  train_cfg->optimization_level_ = static_cast<OptimizationLevel>(context_info_ptr->train_cfg.optimization_level);

  switch (model_info_ptr->mode) {
    case kBuffer: {
      MS_LOG(DEBUG) << "input model buffer, model_buffer_total: " << model_info_ptr->model_buffer_total;
      if (model_info_ptr->model_buffer_data == nullptr || model_info_ptr->model_buffer_total <= 0) {
        MS_LOG(ERROR) << "Failed to build model.";
        return nullptr;
      }
      std::shared_ptr<mindspore::Model> model_ptr = std::make_shared<mindspore::Model>();
      if (model_ptr == nullptr) {
        MS_LOG(ERROR) << "Failed to new mindspore::model.";
        return nullptr;
      }
      mindspore::Graph graph;
      auto status = mindspore::Serialization::Load(model_info_ptr->model_buffer_data,
                                                   model_info_ptr->model_buffer_total, mindspore::kMindIR, &graph);
      if (status != mindspore::kSuccess) {
        MS_LOG(ERROR) << "load ms file failed.";
        return nullptr;
      }
      auto ret = model_ptr->Build(static_cast<mindspore::GraphCell>(graph), context, train_cfg);
      if (ret == mindspore::kSuccess) {
        MS_LOG(INFO) << "Build model from buffer success.";
        return model_ptr;
      }
      break;
    }
    case kPath: {
      MS_LOG(DEBUG) << "input model path, model_buffer_total: " << model_info_ptr->model_path.c_str();
      std::shared_ptr<mindspore::Model> model_ptr = std::make_shared<mindspore::Model>();
      if (model_ptr == nullptr) {
        MS_LOG(ERROR) << "Failed to new mindspore::model.";
        return nullptr;
      }

      mindspore::Graph graph;
      auto status = mindspore::Serialization::Load(model_info_ptr->model_path, mindspore::kMindIR, &graph);
      if (status != mindspore::kSuccess) {
        MS_LOG(ERROR) << "load ms file failed.";
        return nullptr;
      }
      auto ret = model_ptr->Build(static_cast<mindspore::GraphCell>(graph), context, train_cfg);
      if (ret == mindspore::kSuccess) {
        MS_LOG(INFO) << "Build model from path success.";
        return model_ptr;
      }
      return nullptr;
    }
    case kFD: {
      MS_LOG(DEBUG) << "input model fd:" << model_info_ptr->model_fd
                    << ", model_buffer_total: " << model_info_ptr->model_buffer_total;
      std::shared_ptr<mindspore::Model> model_ptr = std::make_shared<mindspore::Model>();
      if (model_ptr == nullptr) {
        MS_LOG(ERROR) << "Failed to new mindspore::model.";
        return nullptr;
      }

      mindspore::Graph graph;
      auto status = mindspore::Serialization::Load(model_info_ptr->model_buffer_data,
                                                   model_info_ptr->model_buffer_total, mindspore::kMindIR, &graph);
      if (status != mindspore::kSuccess) {
        MS_LOG(ERROR) << "load ms file failed.";
        return nullptr;
      }
      auto ret = model_ptr->Build(static_cast<mindspore::GraphCell>(graph), context, train_cfg);
      (void)munmap(model_info_ptr->model_buffer_data, model_info_ptr->model_buffer_total);
      if (ret == mindspore::kSuccess) {
        MS_LOG(INFO) << "Build model from fd success.";
        return model_ptr;
      }

      break;
    }
    default: {
      MS_LOG(ERROR) << "Invalid model mode.";
    }
  }
  MS_LOG(ERROR) << "Build model failed.";
  return nullptr;
}

int32_t MSLiteModelNapi::GetDeviceInfoContext(ContextInfo *context_ptr,
                                              std::vector<std::shared_ptr<DeviceInfoContext>> &device_infos) {
  for (auto device_name : context_ptr->target) {
    if (kDeviceTypes.find(device_name) == kDeviceTypes.end()) {
      MS_LOG(ERROR) << "Invalid device: " << device_name.c_str();
      return ERR_INVALID_OPERATION;
    }

    auto device_type = kDeviceTypes.at(device_name);
    switch (device_type) {
      case kCPU: {
        auto cpu_device = std::make_shared<mindspore::CPUDeviceInfo>();
        if (cpu_device == nullptr) {
          MS_LOG(ERROR) << "Failed to new CPU deviceInfo.";
          return ERR_INVALID_OPERATION;
        }
        bool is_fp16 = (context_ptr->cpu_device.precision_mode.compare("preferred_fp16") == 0) ? true : false;
        cpu_device->SetEnableFP16(is_fp16);
        device_infos.push_back(cpu_device);
        break;
      }
      case kNNRt: {
        auto nnrt_device = std::make_shared<mindspore::NNRTDeviceInfo>();
        if (nnrt_device == nullptr) {
          MS_LOG(ERROR) << "Failed to new NNRT deviceInfo.";
          return ERR_INVALID_OPERATION;
        }
        nnrt_device->SetDeviceID(context_ptr->nnrt_device.device_id);
        if (context_ptr->nnrt_device.performance_mode != UNSET_VALUE) {
          nnrt_device->SetPerformanceMode(context_ptr->nnrt_device.performance_mode);
        }
        if (context_ptr->nnrt_device.priority != UNSET_VALUE) {
          nnrt_device->SetPriority(context_ptr->nnrt_device.priority);
        }
        // ignore extensions
        device_infos.push_back(nnrt_device);
        break;
      }
      default: {
        MS_LOG(ERROR) << "invalid device.";
        return ERR_INVALID_OPERATION;
      }
    }
  }
  return SUCCESS;
}

napi_value MSLiteModelNapi::Constructor(napi_env env, napi_callback_info info) {
  napi_status status;
  napi_value result = nullptr;
  napi_get_undefined(env, &result);
  GET_PARAMS(env, info, ARGS_TWO);

  std::unique_ptr<MSLiteModelNapi> model_napi = std::make_unique<MSLiteModelNapi>();
  if (model_napi == nullptr) {
    MS_LOG(ERROR) << "No memory";
    return result;
  }

  model_napi->env_ = env;
  if (model_info_->train_model) {
    model_napi->native_model_ = CreateTrainModel(model_info_, context_);
  } else {
    model_napi->native_model_ = CreateModel(model_info_, context_);
  }
  if (model_napi->native_model_ == nullptr) {
    MS_LOG(ERROR) << "Failed to create model.";
    return result;
  }

  status =
    napi_wrap(env, thisVar, reinterpret_cast<void *>(model_napi.get()), MSLiteModelNapi::Finalize, nullptr, nullptr);
  if (status == napi_ok) {
    model_napi.release();
    return thisVar;
  }
  return result;
}

int32_t MSLiteModelNapi::ParseModelInfo(napi_env env, napi_value root, ModelInfo &model_info) {
  napi_valuetype valueType;
  napi_status status = napi_typeof(env, root, &valueType);
  if (status != napi_ok) {
    MS_LOG(ERROR) << "napi_typeof error.";
    return ERR_INVALID_PARAM;
  }
  if ((valueType != napi_object) && (valueType != napi_string) && (valueType != napi_number)) {
    MS_LOG(ERROR) << "model is invaild.";
    return ERR_INVALID_PARAM;
  }

  bool is_model_buffer = false;
  napi_is_arraybuffer(env, root, &is_model_buffer);
  if (is_model_buffer) {
    // copy buffer
    char *array_buffer_data;
    size_t array_buffer_total;
    status = napi_get_arraybuffer_info(env, root, reinterpret_cast<void **>(&array_buffer_data), &array_buffer_total);
    if ((status != napi_ok) || (array_buffer_total <= 0)) {
      MS_LOG(ERROR) << "Parse model buffer failed.";
      return ERR_INVALID_PARAM;
    }

    // shallow copy
    model_info.model_buffer_data = array_buffer_data;
    model_info.model_buffer_total = array_buffer_total;
    model_info.mode = kBuffer;
  } else if (valueType == napi_number) {
    int32_t fd;
    status = napi_get_value_int32(env, root, &fd);
    if ((status != napi_ok) || (fd <= 0)) {
      MS_LOG(ERROR) << "Parse model FD failed.";
      return ERR_INVALID_PARAM;
    }

    int size = lseek(fd, 0, SEEK_END);
    (void)lseek(fd, 0, SEEK_SET);
    auto mmap_buffers = mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0);
    if (mmap_buffers == NULL) {
      MS_LOG(ERROR) << "mmap_buffers is NULL.";
      return ERR_INVALID_PARAM;
    }
    model_info.model_fd = fd;
    model_info.model_buffer_data = static_cast<char *>(mmap_buffers);
    model_info.model_buffer_total = size;
    model_info.mode = kFD;
  } else {
    char char_buf[SIZE];
    size_t buf_length = 0;
    status = napi_get_value_string_utf8(env, root, char_buf, SIZE, &buf_length);
    if ((status != napi_ok) || (buf_length <= 0)) {
      MS_LOG(ERROR) << "Parse model file failed.";
      return ERR_INVALID_PARAM;
    }
    model_info.model_path.assign(char_buf, char_buf + buf_length);
    model_info.mode = kPath;
    MS_LOG(DEBUG) << "model_path: " << model_info.model_path.c_str();
  }
  return SUCCESS;
}

int32_t MSLiteModelNapi::ParseContextInfo(napi_env env, napi_value args, ContextInfo &context) {
  napi_valuetype valueType;
  napi_status status = napi_typeof(env, args, &valueType);
  if ((status != napi_ok) || (valueType != napi_object)) {
    MS_LOG(ERROR) << "context is invaild.";
    return ERR_NOT_EXISTED_PARAM;
  }

  std::vector<std::string> str_values;
  auto ret = CommonNapi::GetPropertyStringArray(env, args, "target", str_values);
  if (ret != SUCCESS) {
    MS_LOG(ERROR) << "Get context target failed.";
    return ret;
  }
  context.target.assign(str_values.begin(), str_values.end());

  ret = GetCpuDeviceInfo(env, args, context);
  if (ret != ERR_NOT_EXISTED_PARAM && ret != SUCCESS) {
    MS_LOG(ERROR) << "Get context CpuDeviceInfo failed.";
    return ret;
  }

  ret = GetNNRTDeviceInfo(env, args, context);
  if (ret != ERR_NOT_EXISTED_PARAM && ret != SUCCESS) {
    MS_LOG(ERROR) << "Get context NnrtDeviceInfo failed.";
    return ret;
  }
  return SUCCESS;
}

int32_t MSLiteModelNapi::ParseTrainCfgInfo(napi_env env, napi_value root, TrainConfig &cfg) {
  napi_valuetype valueType;
  napi_status status = napi_typeof(env, root, &valueType);
  if ((status != napi_ok) || (valueType != napi_object)) {
    MS_LOG(ERROR) << "TrainCfg is invaild.";
    return ERR_NOT_EXISTED_PARAM;
  }
  std::vector<std::string> str_values;
  auto ret = CommonNapi::GetPropertyStringArray(env, root, "lossName", str_values);
  if (ret != SUCCESS && ret != ERR_NOT_EXISTED_PARAM) {
    MS_LOG(ERROR) << "Get lossName failed.";
    return ret;
  }
  cfg.loss_names.assign(str_values.begin(), str_values.end());

  int32_t int_value = 0;
  ret = CommonNapi::GetPropertyInt32(env, root, "optimizationLevel", int_value);
  if (ret != SUCCESS && ret != ERR_NOT_EXISTED_PARAM) {
    MS_LOG(ERROR) << "Get optimization level failed";
    return ret;
  } else {
    cfg.optimization_level = int_value;
  }
  return SUCCESS;
}

napi_value MSLiteModelNapi::CreateMSLiteModelWrapper(napi_env env, MSLiteModelAsyncContext *async_context) {
  std::lock_guard<std::mutex> lock(create_mutex_);
  napi_status status;
  napi_value result = nullptr;
  napi_value constructor;
  napi_get_undefined(env, &result);

  status = napi_get_reference_value(env, constructor_, &constructor);
  if (status != napi_ok) {
    MS_LOG(ERROR) << "get reference failed.";
    return result;
  }
  model_info_ = &(async_context->model_info);
  context_ = &(async_context->context);
  status = napi_new_instance(env, constructor, 0, nullptr, &result);
  if (status == napi_ok) {
    return result;
  }

  return result;
}

void MSLiteModelNapi::GetMSLiteModelAsyncCallbackComplete(napi_env env, napi_status status, void *data) {
  napi_value valueParam = nullptr;
  auto async_context = static_cast<MSLiteModelAsyncContext *>(data);

  if (async_context != nullptr) {
    if (!async_context->status) {
      valueParam = CreateMSLiteModelWrapper(env, async_context);
    }
    CommonCallbackRoutine(env, async_context, valueParam);
  } else {
    MS_LOG(ERROR) << "GetMSLiteModelAsyncCallbackComplete asyncContext is Null!";
  }
}

void MSLiteModelNapi::CommonCallbackRoutine(napi_env env, MSLiteModelAsyncContext *&asyncContext,
                                            const napi_value &valueParam) {
  napi_value result[ARGS_ONE] = {0};
  napi_value retVal;
  napi_value error = nullptr;

  if (!asyncContext->status) {
    result[PARAM0] = valueParam;
  } else {
    napi_value message = nullptr;
    std::string messageValue = CommonNapi::getMessageByCode(asyncContext->status);
    napi_create_string_utf8(env, messageValue.c_str(), NAPI_AUTO_LENGTH, &message);

    napi_value code = nullptr;
    napi_create_string_utf8(env, (std::to_string(asyncContext->status)).c_str(), NAPI_AUTO_LENGTH, &code);

    napi_create_error(env, code, message, &error);
    napi_get_undefined(env, &result[PARAM0]);
  }

  if (asyncContext->deferred != nullptr) {
    if (!asyncContext->status) {
      napi_resolve_deferred(env, asyncContext->deferred, result[PARAM0]);
    } else {
      napi_reject_deferred(env, asyncContext->deferred, error);
    }
  } else {
    napi_value callback = nullptr;
    napi_get_reference_value(env, asyncContext->callbackRef, &callback);
    napi_call_function(env, nullptr, callback, ARGS_ONE, result, &retVal);
    napi_delete_reference(env, asyncContext->callbackRef);
  }
  napi_delete_async_work(env, asyncContext->work);

  delete asyncContext;
  asyncContext = nullptr;
}

napi_value MSLiteModelNapi::LoadMSLiteModelFromFile(napi_env env, napi_callback_info info) {
  napi_status status;
  napi_value result = nullptr;
  const int32_t refCount = 1;
  GET_PARAMS(env, info, ARGS_THREE);
  napi_valuetype valueType = napi_undefined;

  std::unique_ptr<MSLiteModelAsyncContext> asyncContext = std::make_unique<MSLiteModelAsyncContext>();

  int32_t ret;
  for (size_t i = PARAM0; i < argc; i++) {
    if (i == PARAM0) {
      ret = ParseModelInfo(env, argv[i], asyncContext->model_info);
      if (ret != SUCCESS) {
        MS_LOG(ERROR) << "Parsing model failed.";
        return result;
      }
    } else if (i == PARAM1) {
      napi_typeof(env, argv[i], &valueType);
      if (valueType == napi_function) {
        napi_create_reference(env, argv[i], refCount, &asyncContext->callbackRef);
      } else {
        ret = ParseContextInfo(env, argv[i], asyncContext->context);
        if (ret != SUCCESS) {
          MS_LOG(ERROR) << "Parsing context failed.";
          return result;
        }
      }
    } else if (i == PARAM2) {
      napi_typeof(env, argv[i], &valueType);
      if (valueType == napi_function) {
        napi_create_reference(env, argv[i], refCount, &asyncContext->callbackRef);
      }
      break;
    } else {
      MS_LOG(ERROR) << "Invalid input params.";
      return result;
    }
  }

  if (asyncContext->callbackRef == nullptr) {
    status = napi_create_promise(env, &asyncContext->deferred, &result);
    if (status != napi_ok) {
      MS_LOG(ERROR) << "create promise failed.";
      return result;
    }
  } else {
    status = napi_get_undefined(env, &result);
    if (status != napi_ok) {
      MS_LOG(ERROR) << "create callback failed.";
      return result;
    }
  }

  napi_value resource = nullptr;
  napi_create_string_utf8(env, "LoadMSLiteModelFromFile", NAPI_AUTO_LENGTH, &resource);
  status = napi_create_async_work(
    env, nullptr, resource,
    [](napi_env env, void *data) {
      auto context = static_cast<MSLiteModelAsyncContext *>(data);
      context->status = SUCCESS;
    },
    GetMSLiteModelAsyncCallbackComplete, static_cast<void *>(asyncContext.get()), &asyncContext->work);
  if (status != napi_ok) {
    result = nullptr;
  } else {
    status = napi_queue_async_work(env, asyncContext->work);
    if (status == napi_ok) {
      asyncContext.release();
    } else {
      result = nullptr;
    }
  }
  return result;
}

napi_value MSLiteModelNapi::LoadMSLiteTrainModelFromFile(napi_env env, napi_callback_info info) {
  napi_status status;
  napi_value result = nullptr;
  GET_PARAMS(env, info, ARGS_THREE);

  std::unique_ptr<MSLiteModelAsyncContext> asyncContext = std::make_unique<MSLiteModelAsyncContext>();

  asyncContext->model_info.train_model = true;
  int32_t ret;
  for (size_t i = PARAM0; i < argc; i++) {
    if (i == PARAM0) {
      ret = ParseModelInfo(env, argv[i], asyncContext->model_info);
      if (ret != SUCCESS) {
        MS_LOG(ERROR) << "Parsing model failed.";
        return result;
      }
    } else if (i == PARAM1) {
      ret = ParseTrainCfgInfo(env, argv[i], asyncContext->context.train_cfg);
      if (ret != SUCCESS) {
        MS_LOG(ERROR) << "Parsing TrainCfg failed.";
        return result;
      }
    } else if (i == PARAM2) {
      ret = ParseContextInfo(env, argv[i], asyncContext->context);
      if (ret != SUCCESS) {
        MS_LOG(ERROR) << "Parsing context failed.";
        return result;
      }
    } else {
      MS_LOG(ERROR) << "Invalid input params.";
      return result;
    }
  }

  if (asyncContext->callbackRef == nullptr) {
    status = napi_create_promise(env, &asyncContext->deferred, &result);
    if (status != napi_ok) {
      MS_LOG(ERROR) << "create promise failed.";
      return result;
    }
  } else {
    status = napi_get_undefined(env, &result);
    if (status != napi_ok) {
      MS_LOG(ERROR) << "create callback failed.";
      return result;
    }
  }

  napi_value resource = nullptr;
  napi_create_string_utf8(env, "LoadMSLiteTrainModelFromFile", NAPI_AUTO_LENGTH, &resource);
  status = napi_create_async_work(
    env, nullptr, resource,
    [](napi_env env, void *data) {
      auto context = static_cast<MSLiteModelAsyncContext *>(data);
      context->status = SUCCESS;
    },
    GetMSLiteModelAsyncCallbackComplete, static_cast<void *>(asyncContext.get()), &asyncContext->work);
  if (status != napi_ok) {
    result = nullptr;
  } else {
    status = napi_queue_async_work(env, asyncContext->work);
    if (status == napi_ok) {
      asyncContext.release();
    } else {
      result = nullptr;
    }
  }
  return result;
}

napi_value MSLiteModelNapi::LoadMSLiteTrainModelFromBuffer(napi_env env, napi_callback_info info) {
  napi_status status;
  napi_value result = nullptr;
  GET_PARAMS(env, info, ARGS_THREE);

  std::unique_ptr<MSLiteModelAsyncContext> asyncContext = std::make_unique<MSLiteModelAsyncContext>();

  asyncContext->model_info.train_model = true;
  int32_t ret;
  for (size_t i = PARAM0; i < argc; i++) {
    if (i == PARAM0) {
      ret = ParseModelInfo(env, argv[i], asyncContext->model_info);
      if (ret != SUCCESS) {
        MS_LOG(ERROR) << "Parsing model failed.";
        return result;
      }
    } else if (i == PARAM1) {
      ret = ParseTrainCfgInfo(env, argv[i], asyncContext->context.train_cfg);
      if (ret != SUCCESS) {
        MS_LOG(ERROR) << "Parsing TrainCfg failed.";
        return result;
      }
    } else if (i == PARAM2) {
      ret = ParseContextInfo(env, argv[i], asyncContext->context);
      if (ret != SUCCESS) {
        MS_LOG(ERROR) << "Parsing context failed.";
        return result;
      }
    } else {
      MS_LOG(ERROR) << "Invalid input params.";
      return result;
    }
  }

  if (asyncContext->callbackRef == nullptr) {
    status = napi_create_promise(env, &asyncContext->deferred, &result);
    if (status != napi_ok) {
      MS_LOG(ERROR) << "create promise failed.";
      return result;
    }
  } else {
    status = napi_get_undefined(env, &result);
    if (status != napi_ok) {
      MS_LOG(ERROR) << "create callback failed.";
      return result;
    }
  }

  napi_value resource = nullptr;
  napi_create_string_utf8(env, "LoadMSLiteTrainModelFromBuffer", NAPI_AUTO_LENGTH, &resource);
  status = napi_create_async_work(
    env, nullptr, resource,
    [](napi_env env, void *data) {
      auto context = static_cast<MSLiteModelAsyncContext *>(data);
      context->status = SUCCESS;
    },
    GetMSLiteModelAsyncCallbackComplete, static_cast<void *>(asyncContext.get()), &asyncContext->work);
  if (status != napi_ok) {
    result = nullptr;
  } else {
    status = napi_queue_async_work(env, asyncContext->work);
    if (status == napi_ok) {
      asyncContext.release();
    } else {
      result = nullptr;
    }
  }
  return result;
}

napi_value MSLiteModelNapi::LoadMSLiteTrainModelFromFd(napi_env env, napi_callback_info info) {
  napi_status status;
  napi_value result = nullptr;
  GET_PARAMS(env, info, ARGS_THREE);

  std::unique_ptr<MSLiteModelAsyncContext> asyncContext = std::make_unique<MSLiteModelAsyncContext>();

  asyncContext->model_info.train_model = true;
  int32_t ret;
  for (size_t i = PARAM0; i < argc; i++) {
    if (i == PARAM0) {
      ret = ParseModelInfo(env, argv[i], asyncContext->model_info);
      if (ret != SUCCESS) {
        MS_LOG(ERROR) << "Parsing model failed.";
        return result;
      }
    } else if (i == PARAM1) {
      ret = ParseTrainCfgInfo(env, argv[i], asyncContext->context.train_cfg);
      if (ret != SUCCESS) {
        MS_LOG(ERROR) << "Parsing TrainCfg failed.";
        return result;
      }
    } else if (i == PARAM2) {
      ret = ParseContextInfo(env, argv[i], asyncContext->context);
      if (ret != SUCCESS) {
        MS_LOG(ERROR) << "Parsing context failed.";
        return result;
      }
    } else {
      MS_LOG(ERROR) << "Invalid input params.";
      return result;
    }
  }

  if (asyncContext->callbackRef == nullptr) {
    status = napi_create_promise(env, &asyncContext->deferred, &result);
    if (status != napi_ok) {
      MS_LOG(ERROR) << "create promise failed.";
      return result;
    }
  } else {
    status = napi_get_undefined(env, &result);
    if (status != napi_ok) {
      MS_LOG(ERROR) << "create callback failed.";
      return result;
    }
  }

  napi_value resource = nullptr;
  napi_create_string_utf8(env, "LoadMSLiteTrainModelFromFd", NAPI_AUTO_LENGTH, &resource);
  status = napi_create_async_work(
    env, nullptr, resource,
    [](napi_env env, void *data) {
      auto context = static_cast<MSLiteModelAsyncContext *>(data);
      context->status = SUCCESS;
    },
    GetMSLiteModelAsyncCallbackComplete, static_cast<void *>(asyncContext.get()), &asyncContext->work);
  if (status != napi_ok) {
    result = nullptr;
  } else {
    status = napi_queue_async_work(env, asyncContext->work);
    if (status == napi_ok) {
      asyncContext.release();
    } else {
      result = nullptr;
    }
  }
  return result;
}

napi_value MSLiteModelNapi::LoadMSLiteModelFromBuffer(napi_env env, napi_callback_info info) {
  napi_status status;
  napi_value result = nullptr;
  const int32_t refCount = 1;
  GET_PARAMS(env, info, ARGS_THREE);
  napi_valuetype valueType = napi_undefined;

  std::unique_ptr<MSLiteModelAsyncContext> asyncContext = std::make_unique<MSLiteModelAsyncContext>();

  int32_t ret;
  for (size_t i = PARAM0; i < argc; i++) {
    if (i == PARAM0) {
      ret = ParseModelInfo(env, argv[i], asyncContext->model_info);
      if (ret != SUCCESS) {
        MS_LOG(ERROR) << "Parsing model failed.";
        return result;
      }
    } else if (i == PARAM1) {
      napi_typeof(env, argv[i], &valueType);
      if (valueType == napi_function) {
        napi_create_reference(env, argv[i], refCount, &asyncContext->callbackRef);
      } else {
        ret = ParseContextInfo(env, argv[i], asyncContext->context);
        if (ret != SUCCESS) {
          MS_LOG(ERROR) << "Parsing context failed.";
          return result;
        }
      }
    } else if (i == PARAM2) {
      napi_typeof(env, argv[i], &valueType);
      if (valueType == napi_function) {
        napi_create_reference(env, argv[i], refCount, &asyncContext->callbackRef);
      }
      break;
    } else {
      MS_LOG(ERROR) << "Invalid input params.";
      return result;
    }
  }

  if (asyncContext->callbackRef == nullptr) {
    status = napi_create_promise(env, &asyncContext->deferred, &result);
    if (status != napi_ok) {
      MS_LOG(ERROR) << "create promise failed.";
      return result;
    }
  } else {
    status = napi_get_undefined(env, &result);
    if (status != napi_ok) {
      MS_LOG(ERROR) << "create callback failed.";
      return result;
    }
  }

  napi_value resource = nullptr;
  napi_create_string_utf8(env, "LoadMSLiteModelFromBuffer", NAPI_AUTO_LENGTH, &resource);
  status = napi_create_async_work(
    env, nullptr, resource,
    [](napi_env env, void *data) {
      auto context = static_cast<MSLiteModelAsyncContext *>(data);
      context->status = SUCCESS;
    },
    GetMSLiteModelAsyncCallbackComplete, static_cast<void *>(asyncContext.get()), &asyncContext->work);
  if (status != napi_ok) {
    result = nullptr;
  } else {
    status = napi_queue_async_work(env, asyncContext->work);
    if (status == napi_ok) {
      asyncContext.release();
    } else {
      result = nullptr;
    }
  }
  return result;
}

napi_value MSLiteModelNapi::LoadMSLiteModelFromFd(napi_env env, napi_callback_info info) {
  napi_status status;
  napi_value result = nullptr;
  const int32_t refCount = 1;
  GET_PARAMS(env, info, ARGS_THREE);
  napi_valuetype valueType = napi_undefined;

  std::unique_ptr<MSLiteModelAsyncContext> asyncContext = std::make_unique<MSLiteModelAsyncContext>();

  int32_t ret;
  for (size_t i = PARAM0; i < argc; i++) {
    if (i == PARAM0) {
      ret = ParseModelInfo(env, argv[i], asyncContext->model_info);
      if (ret != SUCCESS) {
        MS_LOG(ERROR) << "Parsing model failed.";
        return result;
      }
    } else if (i == PARAM1) {
      napi_typeof(env, argv[i], &valueType);
      if (valueType == napi_function) {
        napi_create_reference(env, argv[i], refCount, &asyncContext->callbackRef);
      } else {
        ret = ParseContextInfo(env, argv[i], asyncContext->context);
        if (ret != SUCCESS) {
          MS_LOG(ERROR) << "Parsing context failed.";
          return result;
        }
      }
    } else if (i == PARAM2) {
      napi_typeof(env, argv[i], &valueType);
      if (valueType == napi_function) {
        napi_create_reference(env, argv[i], refCount, &asyncContext->callbackRef);
      }
      break;
    } else {
      MS_LOG(ERROR) << "Invalid input params.";
      return result;
    }
  }

  if (asyncContext->callbackRef == nullptr) {
    status = napi_create_promise(env, &asyncContext->deferred, &result);
    if (status != napi_ok) {
      MS_LOG(ERROR) << "create promise failed.";
      return result;
    }
  } else {
    status = napi_get_undefined(env, &result);
    if (status != napi_ok) {
      MS_LOG(ERROR) << "create callback failed.";
      return result;
    }
  }

  napi_value resource = nullptr;
  napi_create_string_utf8(env, "LoadMSLiteModelFromFd", NAPI_AUTO_LENGTH, &resource);
  status = napi_create_async_work(
    env, nullptr, resource,
    [](napi_env env, void *data) {
      auto context = static_cast<MSLiteModelAsyncContext *>(data);
      context->status = SUCCESS;
    },
    GetMSLiteModelAsyncCallbackComplete, static_cast<void *>(asyncContext.get()), &asyncContext->work);
  if (status != napi_ok) {
    result = nullptr;
  } else {
    status = napi_queue_async_work(env, asyncContext->work);
    if (status == napi_ok) {
      asyncContext.release();
    } else {
      result = nullptr;
    }
  }
  return result;
}

int32_t MSLiteModelNapi::GetCpuDeviceInfo(napi_env env, napi_value args, ContextInfo &context) {
  bool has_cpu_property = false;
  napi_status status = napi_has_named_property(env, args, "cpu", &has_cpu_property);
  if (status != napi_ok) {
    MS_LOG(ERROR) << "can not find cpu property";
    return ERR_INVALID_OPERATION;
  }
  if (!has_cpu_property) {
    return ERR_NOT_EXISTED_PARAM;
  }

  napi_value config_item = nullptr;
  status = napi_get_named_property(env, args, "cpu", &config_item);
  if (status != napi_ok) {
    MS_LOG(ERROR) << "can not get cpu property";
    return ERR_INVALID_OPERATION;
  }

  int32_t int_value = 0;
  std::string str_value = "";
  std::vector<int32_t> affinity_cores;

  if (CommonNapi::GetPropertyInt32(env, config_item, "threadNum", int_value) == SUCCESS) {
    MS_LOG(DEBUG) << "threadNum: " << int_value;
    context.cpu_device.thread_num = int_value;
  } else {
    context.cpu_device.thread_num = PARAM2;
  }

  if (CommonNapi::GetPropertyInt32(env, config_item, "threadAffinityMode", int_value) == SUCCESS) {
    MS_LOG(DEBUG) << "threadAffinityMode: " << int_value;
    if (int_value > PARAM2 || int_value < PARAM0) {
      MS_LOG(ERROR) << "threadAffinityMode value is set: " << int_value << ", is out of limition";
      return ERR_INVALID_OPERATION;
    }
    context.cpu_device.thread_affinity_mode = int_value;
  } else {
    context.cpu_device.thread_affinity_mode = PARAM0;
  }

  if (CommonNapi::GetPropertyInt32Array(env, config_item, "threadAffinityCoreList", affinity_cores) == SUCCESS) {
    MS_LOG(DEBUG) << "affinityCores size: " << affinity_cores.size();
    context.cpu_device.thread_affinity_cores.assign(affinity_cores.begin(), affinity_cores.end());
  } else {
    context.cpu_device.thread_affinity_cores = {};
  }

  if (CommonNapi::GetPropertyString(env, config_item, "precisionMode", str_value) == SUCCESS) {
    MS_LOG(DEBUG) << "precisionMode: " << str_value.c_str();
    context.cpu_device.precision_mode = str_value;
  } else {
    context.cpu_device.precision_mode = "enforce_fp32";
  }
  return SUCCESS;
}

int32_t MSLiteModelNapi::GetNNRTDeviceInfo(napi_env env, napi_value args, ContextInfo &context) {
  bool has_nnrt_property = false;
  napi_status status = napi_has_named_property(env, args, "nnrt", &has_nnrt_property);
  if (status != napi_ok) {
    MS_LOG(ERROR) << "can not find nnrt property";
    return ERR_ILLEGAL_STATE;
  }
  if (!has_nnrt_property) {
    return ERR_NOT_EXISTED_PARAM;
  }

  napi_value config_item = nullptr;
  status = napi_get_named_property(env, args, "nnrt", &config_item);
  if (status != napi_ok) {
    MS_LOG(ERROR) << "can not get nnrt property";
    return ERR_INVALID_PARAM;
  }

  int32_t int_value = 0;
  std::string str_value = "";
  std::vector<int32_t> affinity_cores;

  uint64_t device_id;
  auto ret = CommonNapi::GetPropertyBigIntUint64(env, config_item, "deviceID", device_id);
  if (ret == SUCCESS) {
    MS_LOG(DEBUG) << "deviceID: " << device_id;
    context.nnrt_device.device_id = static_cast<size_t>(device_id);
  } else if (ret == ERR_NOT_EXISTED_PARAM) {
    size_t num = 0;
    auto *desc = OH_AI_GetAllNNRTDeviceDescs(&num);
    if (desc == nullptr || num == 0) {
      MS_LOG(WARNING) << "Failed to get nnrt device id, skip adding nnrt device info.";
      return ERR_NOT_EXISTED_PARAM;
    }
    auto id = OH_AI_GetDeviceIdFromNNRTDeviceDesc(desc);
    OH_AI_DestroyAllNNRTDeviceDescs(&desc);
    MS_LOG(INFO) << "set nnrt device id to " << id;
    context.nnrt_device.device_id = id;
  } else {
    return ERR_INVALID_PARAM;
  }

  ret = CommonNapi::GetPropertyInt32(env, config_item, "performanceMode", int_value);
  if (ret == SUCCESS) {
    MS_LOG(DEBUG) << "performanceMode: " << int_value;
    if (int_value > PARAM4 || int_value < PARAM0) {
      MS_LOG(ERROR) << "performanceMode value is set to: " << int_value << ", which is out of range";
      return ERR_INVALID_PARAM;
    }
    context.nnrt_device.performance_mode = int_value;
  } else if (ret == ERR_NOT_EXISTED_PARAM) {
    context.nnrt_device.performance_mode = UNSET_VALUE;
  } else {
    return ERR_INVALID_PARAM;
  }

  ret = CommonNapi::GetPropertyInt32(env, config_item, "priority", int_value);
  if (ret == SUCCESS) {
    MS_LOG(DEBUG) << "priority: " << int_value;
    if (int_value > PARAM3 || int_value < PARAM0) {
      MS_LOG(ERROR) << "priority value is set to: " << int_value << ", which is out of range";
      return ERR_INVALID_PARAM;
    }
    context.nnrt_device.priority = int_value;
  } else if (ret == ERR_NOT_EXISTED_PARAM) {
    context.nnrt_device.priority = UNSET_VALUE;
  } else {
    return ERR_INVALID_PARAM;
  }

  // ignore extensions for now
  return SUCCESS;
}

napi_value MSLiteModelNapi::GetInputs(napi_env env, napi_callback_info info) {
  napi_value undefinedResult = nullptr;
  napi_get_undefined(env, &undefinedResult);

  size_t argCount = 0;
  napi_value jsThis = nullptr;
  napi_value jsResult = nullptr;
  MSLiteModelNapi *modelNapi = nullptr;

  napi_status status = napi_get_cb_info(env, info, &argCount, nullptr, &jsThis, nullptr);
  if (status != napi_ok || jsThis == nullptr) {
    MS_LOG(ERROR) << "failed to retrieve details about the callback";
    return undefinedResult;
  }

  status = napi_unwrap(env, jsThis, reinterpret_cast<void **>(&modelNapi));
  if (status != napi_ok || modelNapi == nullptr) {
    MS_LOG(ERROR) << "failed to get model";
    return undefinedResult;
  }

  if (modelNapi->native_model_ == nullptr) {
    MS_LOG(ERROR) << "model is released(null), please create model again";
    return undefinedResult;
  }
  std::vector<MSTensor> inputs = modelNapi->native_model_->GetInputs();
  std::vector<MSTensor> tensor_inputs;
  for (size_t i = 0; i < inputs.size(); i++) {
    auto tensor = mindspore::MSTensor::CreateTensor(inputs.at(i).Name(), inputs.at(i).DataType(), {}, nullptr, 0);
    if (tensor == nullptr) {
      MS_LOG(ERROR) << "create tensor failed.";
      return undefinedResult;
    }
    tensor->SetShape(inputs.at(i).Shape());
    tensor->SetFormat(inputs.at(i).format());
    tensor->SetDataType(inputs.at(i).DataType());
    tensor_inputs.push_back(*tensor);
    delete tensor;
  }

  size_t size = inputs.size();
  MS_LOG(DEBUG) << "inputs size: " << size;
  napi_create_array_with_length(env, size, &jsResult);
  for (size_t i = 0; i < size; i++) {
    status = napi_set_element(env, jsResult, i, MSTensorNapi::NewInstance(env, tensor_inputs[i]));
    if (status != napi_ok) {
      MS_LOG(ERROR) << "napi_set_element failed! code: " << status;
    } 
  }
  MS_LOG(DEBUG) << "get model inputs success: " << inputs[0].Name().c_str();
  return jsResult;
}

napi_value MSLiteModelNapi::Resize(napi_env env, napi_callback_info info) {
  napi_value undefinedResult = nullptr;
  bool result = false;
  napi_status status = napi_get_boolean(env, result, &undefinedResult);
  if (status != napi_ok) {
    MS_LOG(ERROR) << "get bool error";
    return undefinedResult;
  }

  napi_value jsThis = nullptr;
  napi_value jsResult = nullptr;
  MSLiteModelNapi *modelNapi = nullptr;
  napi_value argv[ARGS_TWO] = {0};
  size_t argCount = PARAM2;
  status = napi_get_cb_info(env, info, &argCount, argv, &jsThis, nullptr);
  if (status != napi_ok || jsThis == nullptr) {
    MS_LOG(ERROR) << "failed to retrieve details about the callback";
    return undefinedResult;
  }
  status = napi_unwrap(env, jsThis, reinterpret_cast<void **>(&modelNapi));
  if (status != napi_ok || modelNapi == nullptr) {
    MS_LOG(ERROR) << "get model napi error";
    return undefinedResult;
  }

  if (modelNapi->native_model_ == nullptr) {
    MS_LOG(ERROR) << "model is released(null), please create model again";
    return undefinedResult;
  }
  std::vector<MSTensor> inputs = modelNapi->native_model_->GetInputs();
  std::vector<MSTensor> tensor_inputs;
  std::vector<std::vector<int64_t>> dims;

  // set inputs data
  uint32_t array_length = 0;
  status = napi_get_array_length(env, argv[PARAM0], &array_length);
  if (status != napi_ok || array_length <= 0) {
    MS_LOG(ERROR) << "get inputs tensor length failed.";
    return undefinedResult;
  }
  if (inputs.size() != array_length) {
    MS_LOG(ERROR) << "array length not equal to model inputs size.";
    return undefinedResult;
  }
  for (size_t i = 0; i < array_length; i++) {
    napi_value element = nullptr;
    status = napi_get_element(env, argv[PARAM0], i, &element);
    if (status != napi_ok) {
      MS_LOG(ERROR) << "can not get element";
      return undefinedResult;
    }

    std::string property_name = "getData";
    bool exist = false;
    napi_value data_func = nullptr;

    status = napi_has_named_property(env, element, property_name.c_str(), &exist);
    if (status != napi_ok || !exist) {
      MS_LOG(ERROR) << "can not find target property";
      return undefinedResult;
    }

    if (status != napi_ok || !exist) {
      MS_LOG(ERROR) << "can not find " << property_name.c_str() << " property.";
      return undefinedResult;
    }

    if (napi_get_named_property(env, element, property_name.c_str(), &data_func) != napi_ok) {
      MS_LOG(ERROR) << "get " << property_name.c_str() << " property fail.";
      return undefinedResult;
    }
    void *js_data = nullptr;
    size_t length = 0;
    napi_value return_val;

    status = napi_call_function(env, element, data_func, 0, nullptr, &return_val);
    if (status != napi_ok || return_val == nullptr) {
      MS_LOG(ERROR) << "napi call function error.";
      return undefinedResult;
    }

    status = napi_get_arraybuffer_info(env, return_val, &js_data, &length);
    if (status != napi_ok || js_data == nullptr) {
      MS_LOG(ERROR) << "get js data error.";
      return undefinedResult;
    }
    if (inputs[i].DataSize() != length) {
      MS_LOG(ERROR) << "tensor size is: " << static_cast<int>(inputs[i].DataSize()) << ", but data length got "
                    << static_cast<int>(length);
      return undefinedResult;
    }

    auto tensor_data = inputs[i].MutableData();
    if (tensor_data == nullptr) {
      MS_LOG(ERROR) << "malloc data for tensor failed.";
      return undefinedResult;
    }
    memcpy(tensor_data, js_data, length);
  }

  napi_value dim_num = nullptr;
  int64_t dim_ele = 0;
  uint32_t dims_size = 0;
  uint32_t dim_size = 0;

  status = napi_is_array(env, argv[PARAM1], &result);
  if (status != napi_ok || result == false) {
    MS_LOG(ERROR) << "new dim is not a array";
    return undefinedResult;
  }

  status = napi_get_array_length(env, argv[PARAM1], &dims_size);
  if (status != napi_ok) {
    MS_LOG(ERROR) << "get new dims size error";
    return undefinedResult;
  }
  for (size_t i = 0; i < dims_size; i++) {
    napi_value dim_element = nullptr;
    status = napi_get_element(env, argv[PARAM1], i, &dim_element);
    if (status != napi_ok) {
      MS_LOG(ERROR) << "can not get element";
      return undefinedResult;
    }

    status = napi_is_array(env, dim_element, &result);
    if (status != napi_ok || result == false) {
      MS_LOG(ERROR) << "new dim's element is not a array";
      return undefinedResult;
    }

    status = napi_get_array_length(env, dim_element, &dim_size);
    if (status != napi_ok) {
      MS_LOG(ERROR) << "get new dim size error";
      return undefinedResult;
    }
    std::vector<int64_t> dim(dim_size);
    for (size_t j = 0; j < dim_size; j++) {
      status = napi_get_element(env, dim_element, j, &dim_num);
      if (status != napi_ok) {
        MS_LOG(ERROR) << "get dim num error";
        return undefinedResult;
      }
      status = napi_get_value_int64(env, dim_num, &dim_ele);
      if (status != napi_ok) {
        MS_LOG(ERROR) << "get dim element error";
        return undefinedResult;
      }
      dim[j] = dim_ele;
    }
    dims.push_back(dim);
  }
  if (modelNapi->native_model_->Resize(inputs, dims) != mindspore::kSuccess) {
    MS_LOG(ERROR) << "resize failed";
    return undefinedResult;
  }
  status = napi_get_boolean(env, result, &jsResult);
  if (status != napi_ok) {
    MS_LOG(ERROR) << "get bool error";
    return undefinedResult;
  }
  return jsResult;
}

template <typename T, typename Distribution>
void GenerateRandomData(int size, void *data, Distribution distribution) {
  std::mt19937 random_engine;
  int elements_num = size / sizeof(T);
  (void)std::generate_n(static_cast<T *>(data), elements_num,
                        [&distribution, &random_engine]() { return static_cast<T>(distribution(random_engine)); });
}

int GenerateInputDataWithRandom(std::vector<mindspore::MSTensor> inputs) {
  for (auto tensor : inputs) {
    auto input_data = tensor.MutableData();
    if (input_data == nullptr) {
      std::cerr << "mallocData for inTensor failed." << std::endl;
      return -1;
    }
    GenerateRandomData<float>(tensor.DataSize(), input_data, std::uniform_real_distribution<float>(0.1f, 1.0f));
  }
  return mindspore::kSuccess;
}

napi_value MSLiteModelNapi::PredictAsync(napi_env env, napi_callback_info info) {
  napi_status status = napi_ok;
  napi_value undefinedResult = nullptr;
  napi_value result = nullptr;
  const int32_t refCount = 1;
  napi_valuetype valueType;

  std::unique_ptr<MSLiteModelAsyncContext> asyncContext = std::make_unique<MSLiteModelAsyncContext>();
  if (asyncContext == nullptr) {
    MS_LOG(ERROR) << "MSLiteModelAsyncContext object create failed.";
    return undefinedResult;
  }

  GET_PARAMS(env, info, ARGS_TWO);
  for (size_t i = PARAM0; i < argc; i++) {
    if (i == PARAM1) {
      status = napi_typeof(env, argv[i], &valueType);
      if ((status != napi_ok) || (valueType != napi_function)) {
        MS_LOG(ERROR) << "napi_typeof check callback failed.";
        return result;
      }
      status = napi_create_reference(env, argv[i], refCount, &asyncContext->callbackRef);
      if (status != napi_ok) {
        MS_LOG(ERROR) << "failed to create reference of callback";
        return result;
      }
    }
  }

  if (SetTensorData(env, thisVar, argv[PARAM0], asyncContext.get()) != SUCCESS) {
    MS_LOG(ERROR) << "Set tensor data failed.";
    return undefinedResult;
  }

  if (asyncContext->callbackRef == nullptr) {
    status = napi_create_promise(env, &asyncContext->deferred, &result);
    if (status != napi_ok) {
      MS_LOG(ERROR) << "create promise failed.";
      return result;
    }
  } else {
    status = napi_get_undefined(env, &result);
    if (status != napi_ok) {
      MS_LOG(ERROR) << "create callback failed.";
      return result;
    }
  }

  napi_value resource = nullptr;
  napi_create_string_utf8(env, "Predict", NAPI_AUTO_LENGTH, &resource);
  status = napi_create_async_work(
    env, nullptr, resource,
    [](napi_env env, void *data) {
      auto context = static_cast<MSLiteModelAsyncContext *>(data);
      context->status = SUCCESS;
    },
    PredictAsyncCallbackComplete, static_cast<void *>(asyncContext.get()), &asyncContext->work);
  if (status != napi_ok) {
    result = nullptr;
  } else {
    status = napi_queue_async_work(env, asyncContext->work);
    if (status == napi_ok) {
      asyncContext.release();
    } else {
      result = nullptr;
    }
  }
  return result;
}

int32_t MSLiteModelNapi::SetTensorData(napi_env env, napi_value thisVar, napi_value argv,
                                       MSLiteModelAsyncContext *async_context) {
  uint32_t array_length = 0;
  napi_status status = napi_get_array_length(env, argv, &array_length);
  if (status != napi_ok || array_length <= 0) {
    MS_LOG(ERROR) << "get inputs tensor length failed.";
    return ERR_INVALID_PARAM;
  }

  status = napi_unwrap(env, thisVar, reinterpret_cast<void **>(&(async_context->lite_model)));
  if (status != napi_ok || async_context->lite_model == nullptr) {
    MS_LOG(ERROR) << "get model napi error";
    return ERROR;
  }
  auto modelNapi = async_context->lite_model;
  if (modelNapi->native_model_ == nullptr) {
    MS_LOG(ERROR) << "model is released(null), please create model again";
    return ERROR;
  }

  auto inputs = modelNapi->native_model_->GetInputs();
  if (inputs.size() != array_length) {
    MS_LOG(ERROR) << "array length not equal to model inputs size.";
    return ERR_INVALID_PARAM;
  }

  for (size_t i = 0; i < array_length; i++) {
    napi_value element = nullptr;
    status = napi_get_element(env, argv, i, &element);
    if (status != napi_ok) {
      MS_LOG(ERROR) << "can not get element";
      return ERROR;
    }

    std::string property_name = "getData";
    bool exist = false;
    napi_value data_func = nullptr;

    napi_status status = napi_has_named_property(env, element, property_name.c_str(), &exist);

    if (status != napi_ok || !exist) {
      MS_LOG(ERROR) << "can not find " << property_name.c_str() << " property.";
      return ERROR;
    }

    if (napi_get_named_property(env, element, property_name.c_str(), &data_func) != napi_ok) {
      MS_LOG(ERROR) << "get " << property_name.c_str() << " property fail.";
      return ERROR;
    }
    void *js_data = nullptr;
    size_t length = 0;
    napi_value return_val;

    status = napi_call_function(env, element, data_func, 0, nullptr, &return_val);
    if (status != napi_ok || return_val == nullptr) {
      MS_LOG(ERROR) << "napi call function error.";
      return ERROR;
    }
    status = napi_get_arraybuffer_info(env, return_val, &js_data, &length);
    if (status != napi_ok || js_data == nullptr) {
      MS_LOG(ERROR) << "Get js data error.";
      return ERROR;
    }
    if (inputs[i].DataSize() != length) {
      MS_LOG(ERROR) << "tensor size is: " << static_cast<int>(inputs[i].DataSize()) << ", but data length got "
                    << static_cast<int>(length);
      return ERROR;
    }

    auto tensor_data = inputs[i].MutableData();
    if (tensor_data == nullptr) {
      MS_LOG(ERROR) << "malloc data for tensor failed.";
      return ERROR;
    }
    memcpy(tensor_data, js_data, length);
  }
  return SUCCESS;
}

void MSLiteModelNapi::PredictAsyncCallbackComplete(napi_env env, napi_status status, void *data) {
  napi_value valueParam = nullptr;
  auto asyncContext = static_cast<MSLiteModelAsyncContext *>(data);

  if (asyncContext != nullptr) {
    if (!asyncContext->status) {
      auto modelNapi = asyncContext->lite_model;
      if (modelNapi->native_model_ == nullptr) {
        MS_LOG(ERROR) << "model is released(null), please create model again";
        return;
      }
      auto inputs = modelNapi->native_model_->GetInputs();
      std::vector<MSTensor> outputs;

      auto predict_ret = modelNapi->native_model_->Predict(inputs, &outputs);
      if (predict_ret != mindspore::kSuccess) {
        MS_LOG(ERROR) << "model predict failed.";
        return;
      }

      napi_create_array_with_length(env, outputs.size(), &valueParam);
      for (size_t i = 0; i < outputs.size(); i++) {
        status = napi_set_element(env, valueParam, i, MSTensorNapi::NewInstance(env, outputs[i]));
        if (status != napi_ok) {
          MS_LOG(ERROR) << "napi_set_element failed! code: " << status;
        }
      }
      MS_LOG(DEBUG) << "predict model success.";
    }
    CommonCallbackRoutine(env, asyncContext, valueParam);
  } else {
    MS_LOG(ERROR) << "ERROR: PredictAsyncCallbackComplete asyncContext is Null!";
  }
}

napi_value MSLiteModelNapi::GetWeights(napi_env env, napi_callback_info info) {
  napi_value undefinedResult = nullptr;
  napi_get_undefined(env, &undefinedResult);

  size_t argCount = 0;
  napi_value jsThis = nullptr;
  napi_value jsResult = nullptr;
  MSLiteModelNapi *modelNapi = nullptr;

  napi_status status = napi_get_cb_info(env, info, &argCount, nullptr, &jsThis, nullptr);
  if (status != napi_ok || jsThis == nullptr) {
    MS_LOG(ERROR) << "failed to retrieve details about the callback";
    return undefinedResult;
  }

  status = napi_unwrap(env, jsThis, reinterpret_cast<void **>(&modelNapi));
  if (status != napi_ok || modelNapi == nullptr) {
    MS_LOG(ERROR) << "failed to get model";
    return undefinedResult;
  }

  if (modelNapi->native_model_ == nullptr) {
    MS_LOG(ERROR) << "model is released(null), please create model again";
    return undefinedResult;
  }
  std::vector<MSTensor> weights = modelNapi->native_model_->GetFeatureMaps();
  std::vector<MSTensor> feature_maps;
  for (size_t i = 0; i < weights.size(); i++) {
    auto tensor = mindspore::MSTensor::CreateTensor(weights.at(i).Name(), weights.at(i).DataType(), {}, nullptr, 0);
    if (tensor == nullptr) {
      MS_LOG(ERROR) << "create tensor failed.";
      return undefinedResult;
    }
    tensor->SetShape(weights.at(i).Shape());
    tensor->SetFormat(weights.at(i).format());
    tensor->SetDataType(weights.at(i).DataType());
    tensor->SetData(weights.at(i).MutableData(), false);
    feature_maps.push_back(*tensor);
    delete tensor;
  }

  size_t size = weights.size();
  MS_LOG(INFO) << "weights size: " << size;
  napi_create_array_with_length(env, size, &jsResult);
  for (size_t i = 0; i < size; i++) {
    status = napi_set_element(env, jsResult, i, MSTensorNapi::NewInstance(env, feature_maps[i]));
    if (status != napi_ok) {
      MS_LOG(ERROR) << "napi_set_element failed! code: " << status;
    }
  }
  MS_LOG(INFO) << "get model weights success";
  return jsResult;
}

int32_t SetModelInputs(napi_env env, napi_value argv, std::shared_ptr<Model> model) {
  uint32_t array_length = 0;
  napi_status status = napi_get_array_length(env, argv, &array_length);
  if (status != napi_ok || array_length <= 0) {
    MS_LOG(ERROR) << "get inputs tensor length failed.";
    return ERR_INVALID_PARAM;
  }

  if (model == nullptr) {
    MS_LOG(ERROR) << "model is nullptr";
    return ERR_INVALID_PARAM;
  }

  auto inputs = model->GetInputs();
  if (inputs.size() != array_length) {
    MS_LOG(ERROR) << "array length not equal to model inputs size.";
    return ERR_INVALID_PARAM;
  }

  for (size_t i = 0; i < array_length; i++) {
    napi_value element = nullptr;
    status = napi_get_element(env, argv, i, &element);
    if (status != napi_ok) {
      MS_LOG(ERROR) << "can not get element";
      return ERROR;
    }

    std::string property_name = "getData";
    bool exist = false;
    napi_value data_func = nullptr;

    napi_status status = napi_has_named_property(env, element, property_name.c_str(), &exist);

    if (status != napi_ok || !exist) {
      MS_LOG(ERROR) << "can not find " << property_name.c_str() << " property.";
      return ERROR;
    }

    if (napi_get_named_property(env, element, property_name.c_str(), &data_func) != napi_ok) {
      MS_LOG(ERROR) << "get " << property_name.c_str() << " property fail.";
      return ERROR;
    }
    void *js_data = nullptr;
    size_t length = 0;
    napi_value return_val;

    status = napi_call_function(env, element, data_func, 0, nullptr, &return_val);
    if (status != napi_ok || return_val == nullptr) {
      MS_LOG(ERROR) << "napi call function error.";
      return ERROR;
    }
    status = napi_get_arraybuffer_info(env, return_val, &js_data, &length);
    if (status != napi_ok || js_data == nullptr) {
      MS_LOG(ERROR) << "Get js data error.";
      return ERROR;
    }
    if (inputs[i].DataSize() != length) {
      MS_LOG(ERROR) << "tensor size is: " << static_cast<int>(inputs[i].DataSize()) << ", but data length got "
                    << static_cast<int>(length);
      return ERROR;
    }

    auto tensor_data = inputs[i].MutableData();
    if (tensor_data == nullptr) {
      MS_LOG(ERROR) << "malloc data for tensor failed.";
      return ERROR;
    }
    memcpy(tensor_data, js_data, length);
  }
  return SUCCESS;
}

napi_value MSLiteModelNapi::RunStep(napi_env env, napi_callback_info info) {
  napi_value undefinedResult = nullptr;
  bool result = false;
  napi_status status = napi_get_boolean(env, result, &undefinedResult);
  if (status != napi_ok) {
    MS_LOG(ERROR) << "get bool error";
    return undefinedResult;
  }

  napi_value jsThis = nullptr;
  MSLiteModelNapi *modelNapi = nullptr;
  size_t argCount = PARAM1;
  napi_value argv[ARGS_ONE] = {0};

  status = napi_get_cb_info(env, info, &argCount, argv, &jsThis, nullptr);
  if (status != napi_ok || jsThis == nullptr) {
    MS_LOG(ERROR) << "failed to retrieve details about the callback";
    return undefinedResult;
  }

  if (argCount < ARGS_ONE) {
    MS_LOG(ERROR) << "argument num is less than one, please give input tensors";
    return undefinedResult;
  }

  status = napi_unwrap(env, jsThis, reinterpret_cast<void **>(&modelNapi));
  if (status != napi_ok || modelNapi == nullptr) {
    MS_LOG(ERROR) << "get model napi error";
    return undefinedResult;
  }

  if (SetModelInputs(env, argv[PARAM0], modelNapi->native_model_) != SUCCESS) {
    MS_LOG(ERROR) << "set tensor data failed";
    return undefinedResult;
  }

  if (modelNapi->native_model_ == nullptr) {
    MS_LOG(ERROR) << "model is released(null), please create model again";
    return undefinedResult;
  }

  auto ret = modelNapi->native_model_->RunStep();
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Model run step failed";
    return undefinedResult;
  }
  status = napi_get_boolean(env, true, &undefinedResult);
  if (status != napi_ok) {
    MS_LOG(ERROR) << "create bool true value failed";
    return undefinedResult;
  }
  return undefinedResult;
}

napi_value MSLiteModelNapi::UpdateWeights(napi_env env, napi_callback_info info) {
  napi_value undefinedResult = nullptr;
  bool result = false;
  napi_status status = napi_get_boolean(env, result, &undefinedResult);
  if (status != napi_ok) {
    MS_LOG(ERROR) << "get bool error";
    return undefinedResult;
  }

  napi_value jsThis = nullptr;
  napi_value jsResult = nullptr;
  MSLiteModelNapi *modelNapi = nullptr;
  napi_value argv[ARGS_ONE] = {0};
  size_t argCount = PARAM1;
  status = napi_get_cb_info(env, info, &argCount, argv, &jsThis, nullptr);
  if (status != napi_ok || jsThis == nullptr) {
    MS_LOG(ERROR) << "failed to retrieve details about the callback";
    return undefinedResult;
  }
  status = napi_unwrap(env, jsThis, reinterpret_cast<void **>(&modelNapi));
  if (status != napi_ok || modelNapi == nullptr) {
    MS_LOG(ERROR) << "get model napi error";
    return undefinedResult;
  }

  if (modelNapi->native_model_ == nullptr) {
    MS_LOG(ERROR) << "model is released(null), please create model again";
    return undefinedResult;
  }

  // set inputs data
  uint32_t array_length = 0;
  status = napi_get_array_length(env, argv[PARAM0], &array_length);
  if (status != napi_ok || array_length <= 0) {
    MS_LOG(ERROR) << "get inputs tensor length failed.";
    return undefinedResult;
  }

  std::vector<MSTensor> weights;
  for (size_t i = 0; i < array_length; i++) {
    napi_value element = nullptr;
    status = napi_get_element(env, argv[PARAM0], i, &element);
    if (status != napi_ok) {
      MS_LOG(ERROR) << "can not get element";
      return undefinedResult;
    }

    // get tensor name
    std::string tensor_name;
    auto ret = CommonNapi::GetPropertyString(env, element, "name", tensor_name);
    if (ret != SUCCESS) {
      MS_LOG(ERROR) << "get tensor name property failed";
      return undefinedResult;
    }

    // get tensor format
    int format;
    ret = CommonNapi::GetPropertyInt32(env, element, "format", format);
    if (ret != SUCCESS) {
      MS_LOG(ERROR) << "get format property failed";
      return undefinedResult;
    }

    // get dtype
    int dtype;
    ret = CommonNapi::GetPropertyInt32(env, element, "dtype", dtype);
    if (ret != SUCCESS) {
      MS_LOG(ERROR) << "get format property failed";
      return undefinedResult;
    }

    // get data size
    int data_size;
    ret = CommonNapi::GetPropertyInt32(env, element, "dataSize", data_size);
    if (ret != SUCCESS) {
      MS_LOG(ERROR) << "get dataSize property failed";
      return undefinedResult;
    }

    // get shape
    std::vector<int32_t> shape;
    ret = CommonNapi::GetPropertyInt32Array(env, element, "shape", shape);
    if (ret != SUCCESS) {
      MS_LOG(ERROR) << "get shape property failed";
      return undefinedResult;
    }

    // get data
    std::string property_name = "getData";
    bool exist = false;
    napi_value data_func = nullptr;

    status = napi_has_named_property(env, element, property_name.c_str(), &exist);
    if (status != napi_ok || !exist) {
      MS_LOG(ERROR) << "can not find target property";
      return undefinedResult;
    }

    if (napi_get_named_property(env, element, property_name.c_str(), &data_func) != napi_ok) {
      MS_LOG(ERROR) << "get " << property_name.c_str() << " property fail.";
      return undefinedResult;
    }
    void *js_data = nullptr;
    size_t length = 0;

    napi_value return_val;
    status = napi_call_function(env, element, data_func, 0, nullptr, &return_val);
    if (status != napi_ok || return_val == nullptr) {
      MS_LOG(ERROR) << "napi call function error.";
      return undefinedResult;
    }

    status = napi_get_arraybuffer_info(env, return_val, &js_data, &length);
    if (status != napi_ok || js_data == nullptr) {
      MS_LOG(ERROR) << "get js data error.";
      return undefinedResult;
    }

    std::vector<int64_t> int64_shape;
    int64_shape.reserve(shape.size());
    std::transform(shape.begin(), shape.end(), std::back_inserter(int64_shape), [](int32_t value) {
      return static_cast<int64_t>(value);
    });
    auto tensor = mindspore::MSTensor::CreateTensor(tensor_name, static_cast<mindspore::DataType>(dtype), int64_shape, nullptr, 0);
    if (tensor == nullptr) {
      MS_LOG(ERROR) << "create tensor failed.";
      return undefinedResult;
    }
    tensor->SetFormat(static_cast<mindspore::Format>(format));
    auto tensor_data = tensor->MutableData();
    if (tensor_data == nullptr) {
      MS_LOG(ERROR) << "mutable tensor data failed, get nullptr";
      return undefinedResult;
    }

    if (tensor->DataSize() != length) {
      MS_LOG(ERROR) << "tensor size is: " << static_cast<int>(tensor->DataSize()) << ", but data length got "
                    << static_cast<int>(length);
      return undefinedResult;
    }

    memcpy(tensor_data, js_data, length);
    weights.push_back(*tensor);
    delete tensor;
  }

  if (modelNapi->native_model_->UpdateFeatureMaps(weights) != mindspore::kSuccess) {
    MS_LOG(ERROR) << "UpdateFeatureMaps failed";
    return undefinedResult;
  }
  status = napi_get_boolean(env, true, &jsResult);
  if (status != napi_ok) {
    MS_LOG(ERROR) << "get bool error";
    return undefinedResult;
  }
  return jsResult;
}

napi_value MSLiteModelNapi::ExportModel(napi_env env, napi_callback_info info) {
  napi_value undefinedResult = nullptr;
  bool result = false;
  napi_status status = napi_get_boolean(env, result, &undefinedResult);
  if (status != napi_ok) {
    MS_LOG(ERROR) << "get bool error";
    return undefinedResult;
  }

  napi_value jsThis = nullptr;
  napi_value jsResult = nullptr;
  MSLiteModelNapi *modelNapi = nullptr;
  napi_value argv[ARGS_FOUR] = {0};
  size_t argCount = PARAM4;
  status = napi_get_cb_info(env, info, &argCount, argv, &jsThis, nullptr);
  if (status != napi_ok || jsThis == nullptr) {
    MS_LOG(ERROR) << "failed to retrieve details about the callback";
    return undefinedResult;
  }
  status = napi_unwrap(env, jsThis, reinterpret_cast<void **>(&modelNapi));
  if (status != napi_ok || modelNapi == nullptr) {
    MS_LOG(ERROR) << "get model napi error";
    return undefinedResult;
  }

  if (modelNapi->native_model_ == nullptr) {
    MS_LOG(ERROR) << "model is released(null), please create model again";
    return undefinedResult;
  }

  // get modelfile
  char char_buf[SIZE];
  size_t buf_length = 0;
  status = napi_get_value_string_utf8(env, argv[PARAM0], char_buf, SIZE, &buf_length);
  if ((status != napi_ok) || (buf_length <= 0)) {
    MS_LOG(ERROR) << "Parse model file failed.";
    return undefinedResult;
  }

  std::string model_path;
  model_path.assign(char_buf, char_buf + buf_length);
  MS_LOG(DEBUG) << "model_path: " << model_path.c_str();

  mindspore::QuantizationType quantization_type = kNoQuant;
  int32_t quantization_type_value;
  // get quantization
  if (argCount >= ARGS_TWO) {
    if (napi_get_value_int32(env, argv[PARAM1], &quantization_type_value) != napi_ok) {
      MS_LOG(WARNING) << "fail to get int32_t value from quantizationType";
      return undefinedResult;
    }
    quantization_type = static_cast<mindspore::QuantizationType>(quantization_type_value);
  }

  // get inference mode
  bool export_inference_only = true;
  if (argCount >= ARGS_THREE) {
    if (napi_get_value_bool(env, argv[PARAM2], &export_inference_only) != napi_ok) {
      MS_LOG(WARNING) << "fail to get bool value from exportInferenceOnly";
      return undefinedResult;
    }
  }

  // get output names
  std::vector<std::string> output_tensor_name;
  if (argCount >= ARGS_FOUR) {
    auto ret = CommonNapi::GetStringArray(env, argv[PARAM3], output_tensor_name);
    if (ret != SUCCESS) {
      MS_LOG(ERROR) << "Get context target failed.";
      return undefinedResult;
    }
  }

  auto ret = mindspore::Serialization::ExportModel(*(modelNapi->native_model_.get()), static_cast<mindspore::ModelType>(kMindIR),
                                        model_path, static_cast<mindspore::QuantizationType>(quantization_type),
                                        export_inference_only, output_tensor_name);
  if (ret != mindspore::kSuccess) {
    MS_LOG(ERROR) << "Export model failed";
    return undefinedResult;
  }

  status = napi_get_boolean(env, true, &jsResult);
  if (status != napi_ok) {
    MS_LOG(ERROR) << "get bool error";
    return undefinedResult;
  }
  MS_LOG(DEBUG) << "Export Model Success";
  return jsResult;
}

napi_value MSLiteModelNapi::ExportWeightsCollaborateWithMicro(napi_env env, napi_callback_info info) {
  napi_value undefinedResult = nullptr;
  bool result = false;
  napi_status status = napi_get_boolean(env, result, &undefinedResult);
  if (status != napi_ok) {
    MS_LOG(ERROR) << "get bool error";
    return undefinedResult;
  }

  napi_value jsThis = nullptr;
  napi_value jsResult = nullptr;
  MSLiteModelNapi *modelNapi = nullptr;
  napi_value argv[ARGS_FOUR] = {0};
  size_t argCount = PARAM4;
  status = napi_get_cb_info(env, info, &argCount, argv, &jsThis, nullptr);
  if (status != napi_ok || jsThis == nullptr) {
    MS_LOG(ERROR) << "failed to retrieve details about the callback";
    return undefinedResult;
  }
  status = napi_unwrap(env, jsThis, reinterpret_cast<void **>(&modelNapi));
  if (status != napi_ok || modelNapi == nullptr) {
    MS_LOG(ERROR) << "get model napi error";
    return undefinedResult;
  }

  if (modelNapi->native_model_ == nullptr) {
    MS_LOG(ERROR) << "model is released(null), please create model again";
    return undefinedResult;
  }

  // get weight file
  char char_buf[SIZE];
  size_t buf_length = 0;
  status = napi_get_value_string_utf8(env, argv[PARAM0], char_buf, SIZE, &buf_length);
  if ((status != napi_ok) || (buf_length <= 0)) {
    MS_LOG(ERROR) << "Parse model file failed.";
    return undefinedResult;
  }

  std::string weight_file;
  weight_file.assign(char_buf, char_buf + buf_length);
  MS_LOG(DEBUG) << "weight_file: " << weight_file.c_str();

  // get is inference
  bool is_inference = true;
  if (argCount >= ARGS_TWO) {
    if (napi_get_value_bool(env, argv[PARAM1], &is_inference) != napi_ok) {
      MS_LOG(WARNING) << "fail to get bool value from isInference";
      return undefinedResult;
    }
  }

  // get inference mode
  bool enable_fp16 = false;
  if (argCount >= ARGS_THREE) {
    if (napi_get_value_bool(env, argv[PARAM2], &enable_fp16) != napi_ok) {
      MS_LOG(WARNING) << "fail to get bool value from enableFp16";
      return undefinedResult;
    }
  }

  // get output names
  std::vector<std::string> changeable_weights_name;
  if (argCount >= ARGS_FOUR) {
    auto ret = CommonNapi::GetStringArray(env, argv[PARAM3], changeable_weights_name);
    if (ret != SUCCESS) {
      MS_LOG(ERROR) << "failed to get string array from changeableWeightsName";
      return undefinedResult;
    }
  }

  auto ret = mindspore::Serialization::ExportWeightsCollaborateWithMicro(*(modelNapi->native_model_.get()), static_cast<mindspore::ModelType>(kMindIR),
                                                              weight_file, is_inference, enable_fp16, changeable_weights_name);

  if (ret != mindspore::kSuccess) {
    MS_LOG(ERROR) << "ExportWeightsCollaborateWithMicro failed";
    return undefinedResult;
  }

  status = napi_get_boolean(env, true, &jsResult);
  if (status != napi_ok) {
    MS_LOG(ERROR) << "get bool error";
    return undefinedResult;
  }
  MS_LOG(DEBUG) << "ExportWeightsCollaborateWithMicro Success";
  return jsResult;
}

napi_value MSLiteModelNapi::SetupVirtualBatch(napi_env env, napi_callback_info info) {
  napi_value undefinedResult = nullptr;
  bool result = false;
  napi_status status = napi_get_boolean(env, result, &undefinedResult);
  if (status != napi_ok) {
    MS_LOG(ERROR) << "get bool error";
    return undefinedResult;
  }

  napi_value jsThis = nullptr;
  napi_value jsResult = nullptr;
  MSLiteModelNapi *modelNapi = nullptr;
  napi_value argv[ARGS_THREE] = {0};
  size_t argCount = ARGS_THREE;
  status = napi_get_cb_info(env, info, &argCount, argv, &jsThis, nullptr);
  if (status != napi_ok || jsThis == nullptr) {
    MS_LOG(ERROR) << "failed to retrieve details about the callback";
    return undefinedResult;
  }
  status = napi_unwrap(env, jsThis, reinterpret_cast<void **>(&modelNapi));
  if (status != napi_ok || modelNapi == nullptr) {
    MS_LOG(ERROR) << "get model napi error";
    return undefinedResult;
  }

  if (modelNapi->native_model_ == nullptr) {
    MS_LOG(ERROR) << "model is released(null), please create model again";
    return undefinedResult;
  }

  // get virtual batch
  int virtual_batch_multiplier;
  if (napi_get_value_int32(env, argv[PARAM0], &virtual_batch_multiplier) != napi_ok) {
    MS_LOG(WARNING) << "fail to get int32 value from virtualBatchMultiplier";
    return undefinedResult;
  }

  // get lr
  double lr = -1.0f;
  if (argCount >= ARGS_TWO) {
    if (napi_get_value_double(env, argv[PARAM1], &lr) != napi_ok) {
      MS_LOG(WARNING) << "fail to get double value from lr";
      return undefinedResult;
    }
  }

  // get lr
  double momentum = -1.0f;
  if (argCount >= ARGS_THREE) {
    if (napi_get_value_double(env, argv[PARAM2], &momentum) != napi_ok) {
      MS_LOG(WARNING) << "fail to get double value from momentum";
      return undefinedResult;
    }
  }


  auto ret = modelNapi->native_model_->SetupVirtualBatch(virtual_batch_multiplier, static_cast<float>(lr), static_cast<float>(momentum));

  if (ret != mindspore::kSuccess) {
    MS_LOG(ERROR) << "SetupVirtualBatch failed";
    return undefinedResult;
  }

  status = napi_get_boolean(env, true, &jsResult);
  if (status != napi_ok) {
    MS_LOG(ERROR) << "get bool error";
    return undefinedResult;
  }
  return jsResult;
}
napi_value MSLiteModelNapi::GetTrainMode(napi_env env, napi_callback_info info) {
  napi_value undefinedResult = nullptr;

  napi_value jsThis = nullptr;
  napi_value jsResult = nullptr;
  MSLiteModelNapi *modelNapi = nullptr;
  napi_value argv[ARGS_ONE] = {0};
  size_t argCount = ARGS_ONE;
  auto status = napi_get_cb_info(env, info, &argCount, argv, &jsThis, nullptr);
  if (status != napi_ok || jsThis == nullptr) {
    MS_LOG(ERROR) << "failed to retrieve details about the callback";
    return undefinedResult;
  }
  status = napi_unwrap(env, jsThis, reinterpret_cast<void **>(&modelNapi));
  if (status != napi_ok || modelNapi == nullptr) {
    MS_LOG(ERROR) << "get model napi error";
    return undefinedResult;
  }
  if (modelNapi->native_model_ == nullptr) {
    MS_LOG(ERROR) << "model is released(null), please create model again";
    return undefinedResult;
  }

  auto train_mode = modelNapi->native_model_->GetTrainMode();

  status = napi_get_boolean(env, train_mode, &jsResult);
  if (status != napi_ok) {
    MS_LOG(WARNING) << "create bool value error";
    return undefinedResult;
  }
  return jsResult;
}
napi_value MSLiteModelNapi::SetTrainMode(napi_env env, napi_callback_info info) {
  napi_value undefinedResult = nullptr;

  napi_value jsThis = nullptr;
  napi_value jsResult = nullptr;
  MSLiteModelNapi *modelNapi = nullptr;
  napi_value argv[ARGS_ONE] = {0};
  size_t argCount = ARGS_ONE;
  auto status = napi_get_cb_info(env, info, &argCount, argv, &jsThis, nullptr);
  if (status != napi_ok || jsThis == nullptr) {
    MS_LOG(ERROR) << "failed to retrieve details about the callback";
    return undefinedResult;
  }
  status = napi_unwrap(env, jsThis, reinterpret_cast<void **>(&modelNapi));
  if (status != napi_ok || modelNapi == nullptr) {
    MS_LOG(ERROR) << "get model napi error";
    return undefinedResult;
  }
  if (modelNapi->native_model_ == nullptr) {
    MS_LOG(ERROR) << "model is released(null), please create model again";
    return undefinedResult;
  }

  bool train_mode;
  if (napi_get_value_bool(env, argv[PARAM0], &train_mode) != napi_ok) {
    MS_LOG(WARNING) << "failed to get bool value from input train mode.";
    return undefinedResult;
  }
  if (!model_info_->train_model) {
    MS_LOG(WARNING) << "current model is not train model, unable to set train or eval mode";
    return undefinedResult;
  }
  if (modelNapi->native_model_->SetTrainMode(train_mode) != kSuccess) {
    MS_LOG(ERROR) << "set train mode failed";
    return undefinedResult;
  }

  status = napi_get_boolean(env, true, &jsResult);
  if (status != napi_ok) {
    MS_LOG(WARNING) << "create bool value error";
    return undefinedResult;
  }
  return jsResult;
}
napi_value MSLiteModelNapi::GetLearningRate(napi_env env, napi_callback_info info) {
  napi_value undefinedResult = nullptr;

  napi_value jsThis = nullptr;
  napi_value jsResult = nullptr;
  MSLiteModelNapi *modelNapi = nullptr;
  napi_value argv[ARGS_ONE] = {0};
  size_t argCount = ARGS_ONE;
  auto status = napi_get_cb_info(env, info, &argCount, argv, &jsThis, nullptr);
  if (status != napi_ok || jsThis == nullptr) {
    MS_LOG(ERROR) << "failed to retrieve details about the callback";
    return undefinedResult;
  }
  status = napi_unwrap(env, jsThis, reinterpret_cast<void **>(&modelNapi));
  if (status != napi_ok || modelNapi == nullptr) {
    MS_LOG(ERROR) << "get model napi error";
    return undefinedResult;
  }
  if (modelNapi->native_model_ == nullptr) {
    MS_LOG(ERROR) << "model is released(null), please create model again";
    return undefinedResult;
  }

  auto lr = modelNapi->native_model_->GetLearningRate();

  status = napi_create_double(env, lr, &jsResult);
  if (status != napi_ok) {
    MS_LOG(WARNING) << "create double value error";
    return undefinedResult;
  }
  return jsResult;
}
napi_value MSLiteModelNapi::SetLearningRate(napi_env env, napi_callback_info info) {
  napi_value undefinedResult = nullptr;

  napi_value jsThis = nullptr;
  napi_value jsResult = nullptr;
  MSLiteModelNapi *modelNapi = nullptr;
  napi_value argv[ARGS_ONE] = {0};
  size_t argCount = ARGS_ONE;
  auto status = napi_get_cb_info(env, info, &argCount, argv, &jsThis, nullptr);
  if (status != napi_ok || jsThis == nullptr) {
    MS_LOG(ERROR) << "failed to retrieve details about the callback";
    return undefinedResult;
  }
  status = napi_unwrap(env, jsThis, reinterpret_cast<void **>(&modelNapi));
  if (status != napi_ok || modelNapi == nullptr) {
    MS_LOG(ERROR) << "get model napi error";
    return undefinedResult;
  }
  if (modelNapi->native_model_ == nullptr) {
    MS_LOG(ERROR) << "model is released(null), please create model again";
    return undefinedResult;
  }

  if (!model_info_->train_model) {
    MS_LOG(WARNING) << "current model is not train model, unable to set learning rate";
    return undefinedResult;
  }

  double lr;
  if (napi_get_value_double(env, argv[PARAM0], &lr) != napi_ok) {
    MS_LOG(WARNING) << "failed to get double value.";
    return undefinedResult;
  }

  if (modelNapi->native_model_->SetLearningRate(static_cast<float>(lr)) != kSuccess) {
    MS_LOG(ERROR) << "set learning rate failed";
    return undefinedResult;
  }

  status = napi_get_boolean(env, true, &jsResult);
  if (status != napi_ok) {
    MS_LOG(WARNING) << "create bool value error";
    return undefinedResult;
  }
  return jsResult;
}
}  // namespace mindspore
