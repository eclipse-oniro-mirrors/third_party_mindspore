/**
 * Copyright (C) 2023 Huawei Device Co., Ltd.
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

#ifndef MINDSPORE_INCLUDE_JS_API_COMMON_NAPI_H
#define MINDSPORE_INCLUDE_JS_API_COMMON_NAPI_H

#include <string>
#include <fstream>
#include "napi/native_api.h"
#include "napi/native_node_api.h"
#include "ms_errors.h"
#include "include/api/types.h"

namespace mindspore {

class CommonNapi {
 public:
  CommonNapi() = delete;
  ~CommonNapi() = delete;

  static std::string getMessageByCode(int32_t &code);
  static int32_t GetPropertyInt32(napi_env env, napi_value config_obj, const std::string &type, int32_t &result);
  static int32_t GetPropertyString(napi_env env, napi_value config_obj, const std::string &type, std::string &result);
  static int32_t GetPropertyInt32Array(napi_env env, napi_value config_obj, const std::string &type,
                                       std::vector<int32_t> &result);
  static int32_t GetPropertyBigIntUint64(napi_env env, napi_value config_obj, const std::string &type,
                                         uint64_t &result);
  static int32_t GetPropertyStringArray(napi_env env, napi_value config_obj, const std::string &type,
                                        std::vector<std::string> &result);
  static int32_t GetStringArray(napi_env env, napi_value value, std::vector<std::string> &result);
  static void WriteTensorData(MSTensor tensor, std::string file_path);
  static void WriteOutputsData(const std::vector<MSTensor> outputs, std::string file_path);
};

struct MSLiteAsyncContext {
  explicit MSLiteAsyncContext(napi_env env);
  virtual ~MSLiteAsyncContext();
  int status = SUCCESS;
  std::string errMessage = "";
};

enum ContextThreadAffinityMode : int32_t {
  CONTEXT_AFFINITY_MODE = 0,
  CONTEXT_BIG_CORES_FIRST,
  CONTEXT_LITTLE_CORES_FIRST
};

enum TensorFormat : int32_t {
  TENSOR_DEFAULT_FORMAT = -1,
  TENSOR_NCHW,
  TENSOR_NHWC,
  TENSOR_NHWC4,
  TENSOR_HWKC,
  TENSOR_HWCK,
  TENSOR_KCHW
};

enum TensorDataType : int32_t {
  TENSOR_UNKNOWN = 0,
  TENSOR_INT8 = 32,
  TENSOR_INT16 = 33,
  TENSOR_INT32 = 34,
  TENSOR_INT64 = 35,
  TENSOR_UINT8 = 37,
  TENSOR_UINT16 = 38,
  TENSOR_UINT32 = 39,
  TENSOR_UINT64 = 40,
  TENSOR_FLOAT16 = 42,
  TENSOR_FLOAT32 = 43,
  TENSOR_FLOAT64 = 44
};

enum ModelMode : int32_t {
  kBuffer = 0,
  kPath,
  kFD,
  // add new type here
  kInvalidModelMode = 10,
};

enum ContextQuantizationType : int32_t {
  NO_QUANT = 0,
  WEIGHT_QUANT = 1,
  FULL_QUANT = 2,
};

enum ContextOptimizationLevel : int32_t {
  O0 = 0,
  O2 = 2,
  O3 = 3,
  AUTO = 4,
};

enum ContextPerformanceMode : int32_t {
  PERFORMANCE_NONE = 0,
  PERFORMANCE_LOW = 1,
  PERFORMANCE_MEDIUM = 2,
  PERFORMANCE_HIGH = 3,
  PERFORMANCE_EXTREME = 4,
};

enum ContextPriority : int32_t {
  PRIORITY_NONE = 0,
  PRIORITY_LOW = 1,
  PRIORITY_MEDIUM = 2,
  PRIORITY_HIGH = 3,
};

enum ContextNnrtDeviceType : int32_t {
  NNRTDEVICE_OTHERS = 0,
  NNRTDEVICE_CPU = 1,
  NNRTDEVICE_GPU = 2,
  NNRTDEVICE_ACCELERATOR = 3,
};

struct ModelInfo {
  std::string model_path = "";
  char *model_buffer_data = nullptr;
  size_t model_buffer_total = 0;
  int32_t model_fd = 0;
  ModelMode mode = kBuffer;
  bool train_model = false;
};

struct CpuDevice {
  int thread_num;
  int thread_affinity_mode;
  std::vector<int32_t> thread_affinity_cores;
  std::string precision_mode;
  CpuDevice(){};
  CpuDevice(int thread_num, int affinity_mode, std::vector<int32_t> affinity_cores, std::string precision)
      : thread_num(thread_num),
        thread_affinity_mode(affinity_mode),
        thread_affinity_cores(affinity_cores),
        precision_mode(precision){};
};

struct NnrtDeviceDesc {
  std::string name;
  ContextNnrtDeviceType type;
  size_t id;
};

struct NNRTDevice {
  size_t device_id;
  int performance_mode{-1};
  int priority{-1};
  NNRTDevice(){};
  NNRTDevice(int device_id, int performance_mode, int priority)
      : device_id(device_id), performance_mode(performance_mode), priority(priority){};
};

struct TrainConfig {
  std::vector<std::string> loss_names;
  int optimization_level = kO0; // kAUTO
};

struct ContextInfo {
  std::vector<std::string> target;
  CpuDevice cpu_device;
  NNRTDevice nnrt_device;
  TrainConfig train_cfg;
};

const int32_t NAPI_ERR_INPUT_INVALID = 401;
const int32_t NAPI_ERR_INVALID_PARAM = 1000101;
const int32_t NAPI_ERR_NO_MEMORY = 1000102;
const int32_t NAPI_ERR_ILLEGAL_STATE = 1000103;
const int32_t NAPI_ERR_UNSUPPORTED = 1000104;
const int32_t NAPI_ERR_TIMEOUT = 1000105;
const int32_t NAPI_ERR_STREAM_LIMIT = 1000201;
const int32_t NAPI_ERR_SYSTEM = 1000301;

const std::string NAPI_ERROR_INVALID_PARAM_INFO = "input parameter value error";
const std::string NAPI_ERR_INPUT_INVALID_INFO = "input parameter type or number mismatch";
const std::string NAPI_ERR_INVALID_PARAM_INFO = "invalid parameter";
const std::string NAPI_ERR_NO_MEMORY_INFO = "allocate memory failed";
const std::string NAPI_ERR_ILLEGAL_STATE_INFO = "Operation not permit at current state";
const std::string NAPI_ERR_UNSUPPORTED_INFO = "unsupported option";
const std::string NAPI_ERR_TIMEOUT_INFO = "time out";
const std::string NAPI_ERR_STREAM_LIMIT_INFO = "stream number limited";
const std::string NAPI_ERR_SYSTEM_INFO = "system error";
}  // namespace mindspore
#endif  // COMMON_NAPI_H