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

#include "include/js_api/mstensor_napi.h"
#include <climits>
#include <string.h>
#include <map>
#include "src/common/log.h"

namespace mindspore {
thread_local napi_ref MSTensorNapi::constructor_ = nullptr;
const std::string CLASS_NAME = "MSTensor";

#define GET_PARAMS(env, info, num) \
  size_t argc = num;               \
  napi_value argv[num] = {0};      \
  napi_value thisVar = nullptr;    \
  void *data;                      \
  napi_get_cb_info(env, info, &argc, argv, &thisVar, &data)

const std::unordered_map<std::string, napi_typedarray_type> kDTypeMap{
  {"int32", napi_int32_array},
  {"float32", napi_float32_array},
  {"int8", napi_int8_array},
  {"uint8", napi_uint8_array},
};

namespace {
const int ARGS_TWO = 2;
}

MSTensorNapi::MSTensorNapi() { MS_LOG(DEBUG) << "MSLITE MSTensorNapi Instances create."; }

MSTensorNapi::~MSTensorNapi() {
  if (nativeMSTensor_ != nullptr) {
    nativeMSTensor_ = nullptr;
  }
  MS_LOG(INFO) << "MSLITE MSTensorNapi Instances destroy.";
}

napi_value MSTensorNapi::Constructor(napi_env env, napi_callback_info info) {
  napi_value jsThis = nullptr;
  napi_status status = napi_get_cb_info(env, info, nullptr, nullptr, &jsThis, nullptr);
  if (status != napi_ok || jsThis == nullptr) {
    MS_LOG(ERROR) << "Failed to retrieve details about the callback";
    return nullptr;
  }

  std::unique_ptr<MSTensorNapi> tensorNapi = std::make_unique<MSTensorNapi>();
  if (tensorNapi == nullptr) {
    MS_LOG(ERROR) << "No memory";
    return nullptr;
  }

  tensorNapi->env_ = env;
  status = napi_wrap(env, jsThis, reinterpret_cast<void *>(tensorNapi.get()), MSTensorNapi::Finalize, nullptr, nullptr);
  if (status == napi_ok) {
    tensorNapi.release();
    return jsThis;
  }

  MS_LOG(ERROR) << "Constructor fail.";
  return nullptr;
}

void MSTensorNapi::Finalize(napi_env env, void *nativeObject, void *finalize) {
  (void)env;
  (void)finalize;
  if (nativeObject != nullptr) {
    delete reinterpret_cast<MSTensorNapi *>(nativeObject);
  }
  MS_LOG(INFO) << "Finalize success.";
}

napi_value MSTensorNapi::NewInstance(napi_env env, mindspore::MSTensor tensor) {
  napi_value cons = GetConstructor(env);
  if (cons == nullptr) {
    MS_LOG(ERROR) << "NewInstance GetConstructor is nullptr!";
    return nullptr;
  }
  napi_value instance;
  napi_status status = napi_new_instance(env, cons, 0, nullptr, &instance);
  if (status != napi_ok) {
    MS_LOG(ERROR) << "NewInstance napi_new_instance failed! code: " << status;
    return nullptr;
  }

  MSTensorNapi *proxy = nullptr;
  status = napi_unwrap(env, instance, reinterpret_cast<void **>(&proxy));
  if (proxy == nullptr) {
    MS_LOG(ERROR) << "NewInstance native instance is nullptr! code: " << status;
    return instance;
  }
  // MSTensor 不需要new 内存，直接获取Model.getInputs()
  proxy->nativeMSTensor_ = std::make_unique<mindspore::MSTensor>(tensor);
  if (proxy->nativeMSTensor_ == nullptr) {
    MS_LOG(ERROR) << "NewInstance native tensor unique ptr is nullptr!";
    return instance;
  }
  return instance;
}

napi_value MSTensorNapi::GetConstructor(napi_env env) {
  napi_value cons;
  if (constructor_ != nullptr) {
    napi_get_reference_value(env, constructor_, &cons);
    return cons;
  }

  MS_LOG(INFO) << "Get msTensorNapi constructor.";
  napi_property_descriptor properties[] = {
    DECLARE_NAPI_GETTER("name", GetName),
    DECLARE_NAPI_GETTER("shape", GetShape),
    DECLARE_NAPI_GETTER("elementNum", GetElementNum),
    DECLARE_NAPI_GETTER("dtype", GetDtype),
    DECLARE_NAPI_GETTER("format", GetFormat),
    DECLARE_NAPI_GETTER("dataSize", GetDataSize),

    DECLARE_NAPI_FUNCTION("getData", GetDataBuffer),
    DECLARE_NAPI_FUNCTION("setData", SetData),
  };

  napi_status status = napi_define_class(env, CLASS_NAME.c_str(), NAPI_AUTO_LENGTH, Constructor, nullptr,
                                         sizeof(properties) / sizeof(napi_property_descriptor), properties, &cons);
  if (status != napi_ok) {
    MS_LOG(ERROR) << "MSLITE Failed to define MSTensor class";
    return nullptr;
  }

  status = napi_create_reference(env, cons, 1, &constructor_);
  if (status != napi_ok) {
    MS_LOG(ERROR) << "MSLITE Failed to create reference of constructor";
    return nullptr;
  }

  return cons;
}

napi_value MSTensorNapi::GetName(napi_env env, napi_callback_info info) {
  napi_value undefinedResult = nullptr;
  napi_get_undefined(env, &undefinedResult);
  napi_value jsThis = nullptr;
  napi_value jsResult = nullptr;
  MSTensorNapi *tensor = nullptr;

  napi_status status = napi_get_cb_info(env, info, nullptr, nullptr, &jsThis, nullptr);
  if (status != napi_ok || jsThis == nullptr) {
    MS_LOG(ERROR) << "Failed to retrieve details about the callback";
    return undefinedResult;
  }

  status = napi_unwrap(env, jsThis, reinterpret_cast<void **>(&tensor));
  if (status != napi_ok || tensor == nullptr) {
    MS_LOG(ERROR) << "get tensor napi error";
    return undefinedResult;
  }

  status = napi_create_string_utf8(env, tensor->nativeMSTensor_->Name().c_str(), NAPI_AUTO_LENGTH, &jsResult);
  if (status != napi_ok) {
    MS_LOG(ERROR) << "napi_create_string_utf8 error";
    return undefinedResult;
  }

  MS_LOG(INFO) << "GetName success.";
  return jsResult;
}

napi_value MSTensorNapi::GetShape(napi_env env, napi_callback_info info) {
  napi_value undefinedResult = nullptr;
  napi_get_undefined(env, &undefinedResult);
  napi_value jsThis = nullptr;
  napi_value jsResult = nullptr;
  MSTensorNapi *tensor = nullptr;

  napi_status status = napi_get_cb_info(env, info, nullptr, nullptr, &jsThis, nullptr);
  if (status != napi_ok || jsThis == nullptr) {
    MS_LOG(ERROR) << "Failed to retrieve details about the callback";
    return undefinedResult;
  }

  status = napi_unwrap(env, jsThis, reinterpret_cast<void **>(&tensor));
  if (status != napi_ok || tensor == nullptr) {
    MS_LOG(ERROR) << "get tensor napi error";
    return undefinedResult;
  }

  // return array
  auto shape = tensor->nativeMSTensor_->Shape();
  size_t size = shape.size();
  napi_create_array_with_length(env, size, &jsResult);
  for (size_t i = 0; i < size; i++) {
    napi_value num;
    status = napi_create_int32(env, shape.at(i), &num);
    if (status != napi_ok) {
      MS_LOG(ERROR) << "napi_create_int32 error";
      return undefinedResult;
    }
    napi_set_element(env, jsResult, i, num);
  }

  MS_LOG(INFO) << "GetShape success.";
  return jsResult;
}

napi_value MSTensorNapi::GetElementNum(napi_env env, napi_callback_info info) {
  napi_value undefinedResult = nullptr;
  napi_get_undefined(env, &undefinedResult);
  napi_value jsThis = nullptr;
  napi_value jsResult = nullptr;
  MSTensorNapi *tensor = nullptr;

  napi_status status = napi_get_cb_info(env, info, nullptr, nullptr, &jsThis, nullptr);
  if (status != napi_ok || jsThis == nullptr) {
    MS_LOG(ERROR) << "Failed to retrieve details about the callback";
    return undefinedResult;
  }

  status = napi_unwrap(env, jsThis, reinterpret_cast<void **>(&tensor));
  if (status != napi_ok || tensor == nullptr) {
    MS_LOG(ERROR) << "get tensor napi error";
    return undefinedResult;
  }

  status = napi_create_int32(env, tensor->nativeMSTensor_->ElementNum(), &jsResult);
  if (status != napi_ok) {
    MS_LOG(ERROR) << "napi_create_int32 error";
    return undefinedResult;
  }

  MS_LOG(INFO) << "GetElementNum success.";
  return jsResult;
}

napi_value MSTensorNapi::GetDtype(napi_env env, napi_callback_info info) {
  napi_value undefinedResult = nullptr;
  napi_get_undefined(env, &undefinedResult);
  napi_value jsThis = nullptr;
  napi_value jsResult = nullptr;
  MSTensorNapi *tensor = nullptr;

  napi_status status = napi_get_cb_info(env, info, nullptr, nullptr, &jsThis, nullptr);
  if (status != napi_ok || jsThis == nullptr) {
    MS_LOG(ERROR) << "Failed to retrieve details about the callback";
    return undefinedResult;
  }

  status = napi_unwrap(env, jsThis, reinterpret_cast<void **>(&tensor));
  if (status != napi_ok || tensor == nullptr) {
    MS_LOG(ERROR) << "get tensor napi error";
    return undefinedResult;
  }

  status = napi_create_int32(env, static_cast<int32_t>(tensor->nativeMSTensor_->DataType()), &jsResult);
  if (status != napi_ok) {
    MS_LOG(ERROR) << "napi_create_int32 error";
    return undefinedResult;
  }

  MS_LOG(INFO) << "GetDtype success.";
  return jsResult;
}

napi_value MSTensorNapi::GetFormat(napi_env env, napi_callback_info info) {
  napi_value undefinedResult = nullptr;
  napi_get_undefined(env, &undefinedResult);
  napi_value jsThis = nullptr;
  napi_value jsResult = nullptr;
  MSTensorNapi *tensor = nullptr;

  napi_status status = napi_get_cb_info(env, info, nullptr, nullptr, &jsThis, nullptr);
  if (status != napi_ok || jsThis == nullptr) {
    MS_LOG(ERROR) << "Failed to retrieve details about the callback";
    return undefinedResult;
  }

  status = napi_unwrap(env, jsThis, reinterpret_cast<void **>(&tensor));
  if (status != napi_ok || tensor == nullptr) {
    MS_LOG(ERROR) << "get tensor napi error";
    return undefinedResult;
  }

  status = napi_create_int32(env, static_cast<int32_t>(tensor->nativeMSTensor_->format()), &jsResult);
  if (status != napi_ok) {
    MS_LOG(ERROR) << "napi_create_int32 error";
    return undefinedResult;
  }

  MS_LOG(INFO) << "GetFormat success.";
  return jsResult;
}

napi_value MSTensorNapi::GetDataSize(napi_env env, napi_callback_info info) {
  napi_value undefinedResult = nullptr;
  napi_get_undefined(env, &undefinedResult);
  napi_value jsThis = nullptr;
  napi_value jsResult = nullptr;
  MSTensorNapi *tensor = nullptr;

  napi_status status = napi_get_cb_info(env, info, nullptr, nullptr, &jsThis, nullptr);
  if (status != napi_ok || jsThis == nullptr) {
    MS_LOG(ERROR) << "Failed to retrieve details about the callback";
    return undefinedResult;
  }

  status = napi_unwrap(env, jsThis, reinterpret_cast<void **>(&tensor));
  if (status != napi_ok || tensor == nullptr) {
    MS_LOG(ERROR) << "get tensor napi error";
    return undefinedResult;
  }

  status = napi_create_int32(env, tensor->nativeMSTensor_->DataSize(), &jsResult);
  if (status != napi_ok) {
    MS_LOG(ERROR) << "napi_create_int32 error";
    return undefinedResult;
  }

  MS_LOG(INFO) << "GetDataSize success.";
  return jsResult;
}

napi_value MSTensorNapi::GetDataBuffer(napi_env env, napi_callback_info info) {
  napi_value undefinedResult = nullptr;
  napi_get_undefined(env, &undefinedResult);

  napi_value jsThis = nullptr;
  napi_value jsResult = nullptr;
  MSTensorNapi *tensor = nullptr;

  napi_status status = napi_get_cb_info(env, info, nullptr, nullptr, &jsThis, nullptr);
  if (status != napi_ok || jsThis == nullptr) {
    MS_LOG(ERROR) << "Failed to retrieve details about the callback";
    return undefinedResult;
  }

  status = napi_unwrap(env, jsThis, reinterpret_cast<void **>(&tensor));
  if (status != napi_ok || tensor == nullptr) {
    MS_LOG(ERROR) << "get tensor napi error";
    return undefinedResult;
  }

  size_t byte_length = tensor->nativeMSTensor_->DataSize();
  auto tensor_data = tensor->nativeMSTensor_->MutableData();
  if (tensor_data == nullptr) {
    MS_LOG(ERROR) << "tensor_data is null.";
    return undefinedResult;
  }

  void *data = nullptr;
  status = napi_create_arraybuffer(env, byte_length, &data, &jsResult);
  if (status != napi_ok) {
    MS_LOG(ERROR) << "napi_create_arraybuffer error";
    return undefinedResult;
  }
  if (data == nullptr || jsResult == nullptr) {
    MS_LOG(ERROR) << "napi_create_arraybuffer error";
    return undefinedResult;
  }

  memcpy(data, tensor_data, byte_length);
  MS_LOG(INFO) << "GetDataBuffer success.";
  return jsResult;
}

napi_value MSTensorNapi::SetData(napi_env env, napi_callback_info info) {
  napi_value undefinedResult = nullptr;
  napi_get_undefined(env, &undefinedResult);
  MSTensorNapi *tensor = nullptr;

  GET_PARAMS(env, info, ARGS_TWO);

  napi_status status = napi_unwrap(env, thisVar, reinterpret_cast<void **>(&tensor));
  if (status != napi_ok || tensor == nullptr) {
    MS_LOG(ERROR) << "get tensor napi error";
    return undefinedResult;
  }

  // convert napi_value to c++ type data
  void *js_data = nullptr;
  size_t length = 0;
  status = napi_get_arraybuffer_info(env, argv[0], &js_data, &length);
  if (status != napi_ok || js_data == nullptr) {
    MS_LOG(ERROR) << "Get js data error.";
    return undefinedResult;
  }

  if (tensor->nativeMSTensor_->DataSize() != length) {
    MS_LOG(ERROR) << "tensor size is: " << static_cast<int>(tensor->nativeMSTensor_->DataSize())
                  << "but data length got " << length;
    return undefinedResult;
  }

  // memcpy
  auto tensor_data = tensor->nativeMSTensor_->MutableData();
  if (tensor_data == nullptr) {
    MS_LOG(ERROR) << "malloc data for tensor failed.";
    return undefinedResult;
  }
  memcpy(tensor_data, js_data, length);

  MS_LOG(INFO) << "SetFloatData success.";
  return undefinedResult;
}
}  // namespace mindspore