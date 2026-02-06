/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_EXTENSION_OPTIONS_PARSER_H
#define MINDSPORE_LITE_EXTENSION_OPTIONS_PARSER_H

#include <vector>
#include "src/litert/inner_context.h"
#include "hiai_foundation_wrapper.h"

namespace mindspore::lite::nnrt {
struct ExtensionOptions {
  std::string cache_path_ = "";
  uint32_t cache_version_ = 0;
  mindspore::lite::HiAI_BandMode band_mode{HIAI_BANDMODE_UNSET};
  void *quant_config;
  size_t quant_config_size = 0;
  bool is_optional_quant_setted = false;
  std::string model_name = "";
};

class ExtensionOptionsParser {
public:
  static int Parse(const std::vector<Extension> &extensions, ExtensionOptions *param);

private:
  static void DoParseBondMode(const std::vector<Extension> &extensions, mindspore::lite::HiAI_BandMode *band_mode);
  static void DoParseQuantConfig(const std::vector<Extension> &extensions, void **quant_config, size_t *num,
                                 bool *quant_setted);
  static void DoParseCachePath(const std::vector<Extension> &extensions, std::string *cache_path);
  static void DoParseCacheVersion(const std::vector<Extension> &extensions, uint32_t *cache_version);
  static void DoParseModelName(const std::vector<Extension> &extensions, std::string *model_name);
};

}  // namespace mindspore::lite::nnrt

#endif  // MINDSPORE_LITE_EXTENSION_OPTIONS_PARSER_H
