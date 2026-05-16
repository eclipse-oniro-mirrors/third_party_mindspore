/*
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

#include "include/js_api/common_napi.h"
#include <climits>
#include "src/common/log.h"

namespace mindspore {

namespace {
const int SIZE = 100;
}

std::string CommonNapi::getMessageByCode(int32_t &code) {
  std::string err_message;
  switch (code) {
    case NAPI_ERR_INVALID_PARAM:
      err_message = NAPI_ERR_INVALID_PARAM_INFO;
      break;
    case NAPI_ERR_NO_MEMORY:
      err_message = NAPI_ERR_NO_MEMORY_INFO;
      break;
    case NAPI_ERR_ILLEGAL_STATE:
      err_message = NAPI_ERR_ILLEGAL_STATE_INFO;
      break;
    case NAPI_ERR_UNSUPPORTED:
      err_message = NAPI_ERR_UNSUPPORTED_INFO;
      break;
    case NAPI_ERR_TIMEOUT:
      err_message = NAPI_ERR_TIMEOUT_INFO;
      break;
    case NAPI_ERR_STREAM_LIMIT:
      err_message = NAPI_ERR_STREAM_LIMIT_INFO;
      break;
    case NAPI_ERR_SYSTEM:
      err_message = NAPI_ERR_SYSTEM_INFO;
      break;
    case NAPI_ERR_INPUT_INVALID:
      err_message = NAPI_ERR_INPUT_INVALID_INFO;
      break;
    default:
      err_message = NAPI_ERR_SYSTEM_INFO;
      code = NAPI_ERR_SYSTEM;
      break;
  }
  return err_message;
}

int32_t CommonNapi::GetPropertyInt32(napi_env env, napi_value config_obj, const std::string &type, int32_t &result) {
  napi_value item = nullptr;
  bool exist = false;
  napi_status status = napi_has_named_property(env, config_obj, type.c_str(), &exist);

  if (status != napi_ok || !exist) {
    MS_LOG(WARNING) << "can not find " << type.c_str() << " will set default value";
    return ERR_NOT_EXISTED_PARAM;
  }

  if (napi_get_named_property(env, config_obj, type.c_str(), &item) != napi_ok) {
    MS_LOG(WARNING) << "fail to get property: " << type.c_str();
    return ERR_INVALID_PARAM;
  }

  if (napi_get_value_int32(env, item, &result) != napi_ok) {
    MS_LOG(WARNING) << "fail to get property value " << type.c_str();
    return ERR_INVALID_PARAM;
  }
  return SUCCESS;
}

int32_t CommonNapi::GetPropertyString(napi_env env, napi_value config_obj, const std::string &type,
                                      std::string &result) {
  napi_value item = nullptr;
  bool exist = false;
  char buffer[SIZE];
  size_t length = 0;

  napi_status status = napi_has_named_property(env, config_obj, type.c_str(), &exist);

  if (status != napi_ok || !exist) {
    MS_LOG(WARNING) << "can not find " << type.c_str() << "will set default value";
    return ERR_NOT_EXISTED_PARAM;
  }

  if (napi_get_named_property(env, config_obj, type.c_str(), &item) != napi_ok) {
    MS_LOG(WARNING) << "fail to get property: " << type.c_str();
    return ERR_INVALID_PARAM;
  }

  if (napi_get_value_string_utf8(env, item, buffer, SIZE, &length) != napi_ok) {
    MS_LOG(WARNING) << "fail to get property value " << type.c_str();
    return ERR_INVALID_PARAM;
  }
  result = std::string(buffer);
  return SUCCESS;
}

int32_t CommonNapi::GetPropertyBigIntUint64(napi_env env, napi_value config_obj, const std::string &type,
                                uint64_t &result) {
  napi_value item = nullptr;
  bool exist = false;
  napi_status status = napi_has_named_property(env, config_obj, type.c_str(), &exist);

  if (status != napi_ok || !exist) {
    MS_LOG(WARNING) << "can not find " << type.c_str() << " will set default value";
    return ERR_NOT_EXISTED_PARAM;
  }

  if (napi_get_named_property(env, config_obj, type.c_str(), &item) != napi_ok) {
    MS_LOG(WARNING) << "fail to get property: " << type.c_str();
    return ERR_INVALID_PARAM;
  }

  bool lossless = false;
  if (napi_get_value_bigint_uint64(env, item, &result, &lossless) != napi_ok) {
    MS_LOG(WARNING) << "fail to get property value " << type.c_str();
    return ERR_INVALID_PARAM;
  }

  if (!lossless) {
    MS_LOG(WARNING) << "get uint64_t loss precision !";
    return ERR_INVALID_PARAM;
  }
  return SUCCESS;
}

int32_t CommonNapi::GetPropertyInt32Array(napi_env env, napi_value config_obj, const std::string &type,
                                          std::vector<int32_t> &result) {
  napi_value item = nullptr;
  bool exist = false;
  napi_status status = napi_has_named_property(env, config_obj, type.c_str(), &exist);
  if (status != napi_ok || !exist) {
    MS_LOG(WARNING) << "can not find " << type.c_str() << "will set default value";
    return ERR_NOT_EXISTED_PARAM;
  }

  if (napi_get_named_property(env, config_obj, type.c_str(), &item) != napi_ok) {
    MS_LOG(WARNING) << "fail to get property: " << type.c_str();
    return ERR_INVALID_PARAM;
  }

  uint32_t array_length = 0;
  status = napi_get_array_length(env, item, &array_length);
  if (status != napi_ok || array_length < 0) {
    MS_LOG(WARNING) << "can not get array length.";
    return ERR_INVALID_PARAM;
  }

  if (array_length == 0) {
    return SUCCESS;
  }

  for (size_t i = 0; i < array_length; i++) {
    int32_t int_value = {0};
    napi_value element = nullptr;
    status = napi_get_element(env, item, i, &element);
    if (status != napi_ok) {
      MS_LOG(WARNING) << "can not get element";
      return ERR_INVALID_PARAM;
    }

    if (napi_get_value_int32(env, element, &int_value) != napi_ok) {
      MS_LOG(WARNING) << "get " << type.c_str() << " property value fail";
      return ERR_INVALID_PARAM;
    }
    result.push_back(int_value);
  }

  return SUCCESS;
}

int32_t CommonNapi::GetPropertyStringArray(napi_env env, napi_value config_obj, const std::string &type,
                                           std::vector<std::string> &result) {
  napi_value item = nullptr;
  bool exist = false;
  napi_status status = napi_has_named_property(env, config_obj, type.c_str(), &exist);

  if (status != napi_ok || !exist) {
    MS_LOG(WARNING) << "can not find " << type.c_str() << "will set default value";
    return ERR_NOT_EXISTED_PARAM;
  }

  if (napi_get_named_property(env, config_obj, type.c_str(), &item) != napi_ok) {
    MS_LOG(WARNING) << "fail to get property: " << type.c_str();
    return ERR_INVALID_PARAM;
  }

  uint32_t array_length = 0;
  status = napi_get_array_length(env, item, &array_length);
  if (status != napi_ok || array_length <= 0) {
    MS_LOG(WARNING) << "can not get array length";
    return ERR_INVALID_PARAM;
  }

  for (size_t i = 0; i < array_length; i++) {
    char buffer[SIZE];
    size_t length = 0;

    napi_value element = nullptr;
    status = napi_get_element(env, item, i, &element);
    if (status != napi_ok) {
      MS_LOG(WARNING) << "can not get element";
      return ERR_INVALID_PARAM;
    }

    if (napi_get_value_string_utf8(env, element, buffer, SIZE, &length) != napi_ok) {
      MS_LOG(WARNING) << "fail to get property value " << type.c_str();
      return ERR_INVALID_PARAM;
    }
    result.push_back(std::string(buffer));
  }

  return SUCCESS;
}

int32_t CommonNapi::GetStringArray(napi_env env, napi_value value, std::vector<std::string> &result) {
  uint32_t array_length = 0;
  auto status = napi_get_array_length(env, value, &array_length);
  if (status != napi_ok || array_length <= 0) {
    MS_LOG(WARNING) << "can not get array length";
    return ERR_INVALID_PARAM;
  }

  for (size_t i = 0; i < array_length; i++) {
    char buffer[SIZE];
    size_t length = 0;

    napi_value element = nullptr;
    status = napi_get_element(env, value, i, &element);
    if (status != napi_ok) {
      MS_LOG(WARNING) << "can not get element";
      return ERR_INVALID_PARAM;
    }

    if (napi_get_value_string_utf8(env, element, buffer, SIZE, &length) != napi_ok) {
      MS_LOG(WARNING) << "fail to get string_utf8 value";
      return ERR_INVALID_PARAM;
    }
    result.push_back(std::string(buffer));
  }

  return SUCCESS;
}

void CommonNapi::WriteTensorData(MSTensor tensor, std::string file_path) {
  std::ofstream out_file;
  out_file.open(file_path, std::ios::out | std::ios::app);
  if (!out_file.is_open()) {
    MS_LOG(ERROR) << "output file open failed";
    return;
  }
  auto out_data = reinterpret_cast<const float *>(tensor.Data().get());
  out_file << tensor.Name() << " ";
  for (auto dim : tensor.Shape()) {
    out_file << dim << " ";
  }
  out_file << std::endl;
  for (int i = 0; i < tensor.ElementNum(); i++) {
    out_file << out_data[i] << " ";
  }
  out_file << std::endl;
  out_file.close();
}

void CommonNapi::WriteOutputsData(const std::vector<MSTensor> outputs, std::string file_path) {
  std::ofstream out_file;
  out_file.open(file_path, std::ios::out | std::ios::app);
  if (!out_file.is_open()) {
    MS_LOG(ERROR) << "output file open failed";
    return;
  }
  for (auto tensor : outputs) {
    MS_LOG(INFO) << "tensor name is: " << tensor.Name().c_str()
                 << "tensor size is: " << static_cast<int>(tensor.DataSize())
                 << "tensor elements num is: " << static_cast<int>(tensor.ElementNum());
    // dtype float
    auto out_data = reinterpret_cast<const float *>(tensor.Data().get());
    out_file << tensor.Name() << " ";
    for (auto dim : tensor.Shape()) {
      out_file << dim << " ";
    }
    out_file << std::endl;
    for (int i = 0; i < tensor.ElementNum(); i++) {
      out_file << out_data[i] << " ";
    }
    out_file << std::endl;
  }
  out_file.close();
}

}  // namespace mindspore