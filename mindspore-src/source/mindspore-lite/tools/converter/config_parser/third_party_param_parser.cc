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

#include "tools/converter/config_parser/third_party_param_parser.h"
#include <vector>
#include <string>
#include <map>
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "nnacl_c/op_base.h"
#include "tools/common/string_util.h"

namespace mindspore {
namespace lite {
namespace {
const std::map<std::string, TypeId> kDataTypeMap = {
  {"float64", TypeId::kNumberTypeFloat64}, {"float32", TypeId::kNumberTypeFloat32},
  {"float16", TypeId::kNumberTypeFloat16}, {"int64", TypeId::kNumberTypeInt64},
  {"int32", TypeId::kNumberTypeInt32},     {"int16", TypeId::kNumberTypeInt16},
  {"int8", TypeId::kNumberTypeInt8},       {"uint8", TypeId::kNumberTypeUInt8},
  {"bool", TypeId::kNumberTypeBool},
};

TypeId ConvertDataType(const std::string &type) {
  auto iter = kDataTypeMap.find(type);
  if (iter == kDataTypeMap.end()) {
    return TypeId::kTypeUnknown;
  }
  return iter->second;
}
}  // namespace

/**
 * Parse shapes like "1,256,256,3;3,96;96,96", and return like [[1,256,256,3], [3,96], [96,96]].
 */
int ThirdPartyParamParser::DoParseShape(const std::string &src, std::vector<std::vector<int64_t>> *dst_shapes) {
  MS_CHECK_TRUE_RET(dst_shapes != nullptr, RET_ERROR);
  dst_shapes->clear();

  auto tmp_shapes = SplitStringToVector(src, ";");
  for (auto tmp_shape : tmp_shapes) {
    auto tmp = SplitStringToVector(tmp_shape, ",");
    std::vector<int64_t> shape = {};
    for (auto t : tmp) {
      int value = 0;
      if (!ConvertIntNum(t, &value)) {
        MS_LOG(ERROR) << "Found error when convert shape string to integer";
        return RET_ERROR;
      }
      if (value <= 0) {  // Valid shape value should be greater than 0.
        MS_LOG(ERROR) << "Only support fixed shapes in third party param";
        return RET_ERROR;
      }
      shape.push_back(value);
    }
    dst_shapes->push_back(shape);
  }
  return RET_OK;
}

/**
 * Parse extended parameter like "key_1:value_1;key_2:value_2" and get {{"key_1", "value_1"}, {"key_2", "value_2"}}.
 */
int ThirdPartyParamParser::DoParseExtendedParameters(const std::string &src,
                                                     std::map<std::string, std::vector<uint8_t>> *dst_ext_param) {
  MS_CHECK_TRUE_RET(dst_ext_param != nullptr, RET_ERROR);
  constexpr size_t kKeyIndex = 0U;
  constexpr size_t kValueIndex = 1U;
  constexpr size_t kKeyValueSize = 2U;

  if (src == "") {  // Just return if 'extended_parameters' is configured.
    return RET_OK;
  }

  auto tmp_list = SplitStringToVector(src, ";");
  std::map<std::string, std::vector<uint8_t>> tmp_map = {};
  for (auto tmp : tmp_list) {
    auto key_and_value = SplitStringToVector(tmp, ":");
    if (key_and_value.size() != kKeyValueSize) {
      MS_LOG(ERROR) << "Parse extended parameters failed, should keep key:value format";
      return RET_ERROR;
    }
    auto key = key_and_value[kKeyIndex];
    auto value = key_and_value[kValueIndex];
    if (tmp_map.find(key) != tmp_map.end()) {
      MS_LOG(ERROR) << "Parse extended parameters failed, key should not be duplicated";
      return RET_ERROR;
    }
    tmp_map.emplace(key, std::vector<uint8_t>(value.begin(), value.end()));
  }

  *dst_ext_param = tmp_map;
  return RET_OK;
}

/**
 * Parse dtypes like "float32;float32;int32" and return [kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeInt32]
 */
int ThirdPartyParamParser::DoParseDtypes(const std::string &src, std::vector<TypeId> *dst_dtypes) {
  MS_CHECK_TRUE_RET(dst_dtypes != nullptr, RET_ERROR);
  dst_dtypes->clear();
  auto tmp_dtypes = SplitStringToVector(src, ";");
  for (auto tmp_dtype : tmp_dtypes) {
    TypeId type = ConvertDataType(tmp_dtype);
    if (type == kTypeUnknown) {
      MS_LOG(ERROR) << "Parse dtypes in third party model config failed";
      return RET_ERROR;
    }
    dst_dtypes->push_back(type);
  }
  return RET_OK;
}

/**
 * Parse names like "foo;bar;boo" and get ["foo", "bar", "boo"]
 * If input names are not provided in config, use the default prefix to generate like: "in_0;in_1;..;in_n"
 */
int ThirdPartyParamParser::DoParseNames(const std::string &src, size_t num, const std::string &default_prefix,
                                        std::vector<std::string> *dst_names) {
  MS_CHECK_TRUE_RET(dst_names != nullptr, RET_ERROR);
  std::string tmp_names = src;
  if (tmp_names.empty()) {
    std::string tmp = "";
    for (size_t i = 0; i < num; i++) {
      tmp += default_prefix + "_" + std::to_string(i);
      if (i + 1 < num) {
        tmp += ";";
      }
    }
    tmp_names = tmp;
  }

  *dst_names = SplitStringToVector(tmp_names, ";");
  if (dst_names->size() != num) {
    MS_LOG(ERROR) << "Name number " << dst_names->size() << " and input number: " << num << " are not equal";
    return RET_ERROR;
  }
  return RET_OK;
}

/**
 * Parse formats like "NCHW;NHWC" and get [NCHW, NHWC]
 */
namespace {
  int StringToFormat(const std::string &format_string, schema::Format *format) {
    static const std::unordered_map<std::string, schema::Format> kFormatTable = {
      {"NCHW", schema::Format::Format_NCHW},
      {"NHWC", schema::Format::Format_NHWC},
      {"NHWC4", schema::Format::Format_NHWC4},
      {"HWKC", schema::Format::Format_HWKC},
      {"HWCK", schema::Format::Format_HWCK},
      {"KCHW", schema::Format::Format_KCHW},
      {"CKHW", schema::Format::Format_CKHW},
      {"KHWC", schema::Format::Format_KHWC},
      {"CHWK", schema::Format::Format_CHWK},
      {"HW", schema::Format::Format_HW},
      {"HW4", schema::Format::Format_HW4},
      {"NC", schema::Format::Format_NC},
      {"NC4", schema::Format::Format_NC4},
      {"NC4HW4", schema::Format::Format_NC4HW4},
      {"NUM_OF_FORMAT", schema::Format::Format_NUM_OF_FORMAT},
      {"NCDHW", schema::Format::Format_NCDHW},
      {"NWC", schema::Format::Format_NWC},
      {"NCW", schema::Format::Format_NCW},
    };

    if (format == nullptr) {
      return RET_NULL_PTR;
    }

    auto iter = kFormatTable.find(format_string);
    if (iter == kFormatTable.end()) {
      return RET_PARAM_INVALID;
    }

    *format = iter->second;
    return RET_OK;
  }
}

int ThirdPartyParamParser::DoParseFormats(const std::string &src, size_t num,
                                          std::vector<schema::Format> *result_formats) {
  MS_CHECK_TRUE_RET(result_formats != nullptr, RET_ERROR);
  std::string tmp_names = src;
  if (tmp_names.empty()) {
    std::vector<schema::Format> default_formats(num, schema::Format::Format_NHWC);
    *result_formats = default_formats;
    return RET_OK;
  }

  auto format_strings = SplitStringToVector(tmp_names, ";");
  if (format_strings.size() != num) {
    MS_LOG(ERROR) << "Number of format: " << format_strings.size() << " and number of tensor: " << num << " are not equal";
    return RET_ERROR;
  }

  std::vector<schema::Format> result(num);
  for (size_t i = 0; i < num; i++) {
    if (StringToFormat(format_strings[i], &result[i]) != RET_OK) {
      MS_LOG(ERROR) << "Tensor format:" << format_strings[i] << " is invalid";
      return RET_PARAM_INVALID;
    }
  }
  *result_formats = result;
  return RET_OK;
}

int ThirdPartyParamParser::Parse(const ThirdPartyModelString &param_string, ThirdPartyModelParam *param) {
  MS_CHECK_TRUE_RET(param != nullptr, RET_ERROR);

  auto ret = DoParseShape(param_string.input_shapes, &(param->input_shapes));
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Parse input shapes of third party param failed";
    return RET_ERROR;
  }

  ret = DoParseDtypes(param_string.input_dtypes, &(param->input_dtypes));
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Parse input dtypes of third party param failed";
    return RET_ERROR;
  }

  auto input_shape_num = param->input_shapes.size();
  auto input_dtype_num = param->input_dtypes.size();
  if (input_shape_num != input_dtype_num) {
    MS_LOG(ERROR) << "Input shape number: " << input_shape_num << " and dtype number: " << input_dtype_num
                  << " are not equal";
    return RET_ERROR;
  }

  ret = DoParseFormats(param_string.input_formats, input_shape_num, &(param->input_formats));
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Parse input formats of third party param failed";
    return RET_ERROR;
  }

  const std::string kInputNamePrefix = "in";
  ret = DoParseNames(param_string.input_names, input_shape_num, kInputNamePrefix, &(param->input_names));
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Parse input names of third party param failed";
    return RET_ERROR;
  }

  ret = DoParseShape(param_string.output_shapes, &(param->output_shapes));
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Parse output shaped of third party param failed";
    return RET_ERROR;
  }

  ret = DoParseDtypes(param_string.output_dtypes, &(param->output_dtypes));
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Parse output dtypes of third party param failed";
    return RET_ERROR;
  }

  auto output_shape_num = param->output_shapes.size();
  auto output_dtype_num = param->output_dtypes.size();
  if (output_shape_num != output_dtype_num) {
    MS_LOG(ERROR) << "Output shape number: " << output_shape_num << " and dtype number: " << output_dtype_num
                  << " are not equal";
    return RET_ERROR;
  }

  ret = DoParseFormats(param_string.output_formats, output_shape_num, &(param->output_formats));
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Parse output formats of third party param failed";
    return RET_ERROR;
  }

  const std::string kOutputNamePrefix = "out";
  ret = DoParseNames(param_string.output_names, output_shape_num, kOutputNamePrefix, &(param->output_names));
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Parse output names of third party param failed";
    return RET_ERROR;
  }

  ret = DoParseExtendedParameters(param_string.extended_parameters, &(param->extended_parameters));
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Parse extended parameter of third party param failed";
    return RET_ERROR;
  }

  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
