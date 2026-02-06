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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_CONFIG_PARSER_THIRD_PARTY_PARAM_PARSER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_CONFIG_PARSER_THIRD_PARTY_PARAM_PARSER_H_
#include <string>
#include <vector>
#include <map>
#include "include/errorcode.h"
#include "tools/converter/cxx_api/converter_para.h"
#include "tools/converter/config_parser/config_file_parser.h"

namespace mindspore {
namespace lite {
class ThirdPartyParamParser {
 public:
  static int Parse(const lite::ThirdPartyModelString &param_string, ThirdPartyModelParam *param);

 private:
  static int DoParseShape(const std::string &src, std::vector<std::vector<int64_t>> *dst_shapes);
  static int DoParseExtendedParameters(const std::string &src,
                                       std::map<std::string, std::vector<uint8_t>> *dst_ext_param);
  static int DoParseDtypes(const std::string &src, std::vector<TypeId> *dst_dtypes);
  static int DoParseNames(const std::string &src, size_t num, const std::string &default_prefix,
                          std::vector<std::string> *dst_names);
  static int DoParseFormats(const std::string &src, size_t num, std::vector<schema::Format> *result_formats);
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_CONFIG_PARSER_THIRD_PARTY_PARAM_PARSER_H_
