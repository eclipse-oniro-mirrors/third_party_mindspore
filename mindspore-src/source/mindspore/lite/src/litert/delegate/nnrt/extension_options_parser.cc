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

#include "extension_options_parser.h"
#include "stdlib.h"
#include <map>

namespace mindspore::lite::nnrt {
namespace {
const std::map<std::string, mindspore::lite::HiAI_BandMode> kBandModeMap = {
    {"HIAI_BANDMODE_UNSET", mindspore::lite::HIAI_BANDMODE_UNSET},
    {"HIAI_BANDMODE_LOW", mindspore::lite::HIAI_BANDMODE_LOW},
    {"HIAI_BANDMODE_NORMAL", mindspore::lite::HIAI_BANDMODE_NORMAL},
    {"HIAI_BANDMODE_HIGH", mindspore::lite::HIAI_BANDMODE_HIGH},
};
const std::string kCachePath = "CachePath";
const std::string kCacheVersion = "CacheVersion";
const std::string kBandMode = "BandMode";
const std::string kQuantBuffer = "QuantBuffer";
const std::string kQuantConfigData = "QuantConfigData";
const std::string kModelName = "ModelName";
}  // namespace

int ExtensionOptionsParser::Parse(const std::vector<Extension> &extensions, ExtensionOptions *param) {
  MS_CHECK_TRUE_RET(param != nullptr, RET_ERROR);

  DoParseCachePath(extensions, &param->cache_path_);
  DoParseCacheVersion(extensions, &param->cache_version_);
  DoParseBondMode(extensions, &param->band_mode);
  DoParseQuantConfig(extensions, &param->quant_config, &param->quant_config_size, &param->is_optional_quant_setted);
  DoParseModelName(extensions, &param->model_name);
  return RET_OK;
}

void ExtensionOptionsParser::DoParseCachePath(const std::vector<Extension> &extensions, std::string *cache_path) {
  MS_CHECK_TRUE_RET_VOID(cache_path != nullptr);
  auto iter_config = std::find_if(extensions.begin(), extensions.end(), [](const Extension &extension) {
    return extension.name == kCachePath;
  });
  if (iter_config != extensions.end()) {
    *cache_path = std::string(iter_config->value.begin(), iter_config->value.end());
  }
}

void ExtensionOptionsParser::DoParseCacheVersion(const std::vector<Extension> &extensions, uint32_t *cache_version) {
  MS_CHECK_TRUE_RET_VOID(cache_version != nullptr);
  auto iter_config = std::find_if(extensions.begin(), extensions.end(), [](const Extension &extension) {
    return extension.name == kCacheVersion;
  });
  if (iter_config != extensions.end()) {
    std::string version_str = std::string(iter_config->value.begin(), iter_config->value.end());
    *cache_version = static_cast<uint32_t>(std::atol(version_str.c_str()));
  }
}

void ExtensionOptionsParser::DoParseBondMode(const std::vector<Extension> &extensions, mindspore::lite::HiAI_BandMode *band_mode) {
  MS_CHECK_TRUE_RET_VOID(band_mode != nullptr);
  auto iter_config = std::find_if(extensions.begin(), extensions.end(), [](const Extension &extension) {
    return extension.name == kBandMode;
  });
  if (iter_config != extensions.end()) {
    auto iter = kBandModeMap.find(std::string(iter_config->value.begin(), iter_config->value.end()));
    if (iter != kBandModeMap.end()) {
      *band_mode = iter->second;
    }
  }
}

void ExtensionOptionsParser::DoParseQuantConfig(const std::vector<Extension> &extensions,
                                                void **quant_config, size_t *num, bool *quant_setted) {
  MS_CHECK_TRUE_RET_VOID(quant_config != nullptr);
  MS_CHECK_TRUE_RET_VOID(num != nullptr);
  auto iter_config = std::find_if(extensions.begin(), extensions.end(), [](const Extension &extension) {
    return extension.name == kQuantBuffer || extension.name == kQuantConfigData;
  });
  if (iter_config != extensions.end()) {
    *quant_config = static_cast<void *>(const_cast<uint8_t *>(iter_config->value.data()));
    *num = iter_config->value.size();
    *quant_setted = true;
  }
}

void ExtensionOptionsParser::DoParseModelName(const std::vector<Extension> &extensions, std::string *model_name) {
  MS_CHECK_TRUE_RET_VOID(model_name != nullptr);
  auto iter_config = std::find_if(extensions.begin(), extensions.end(), [](const Extension &extension) {
    return extension.name == kModelName;
  });
  if (iter_config != extensions.end()) {
    *model_name = std::string(iter_config->value.begin(), iter_config->value.end());
  }
}
}  // mindspore::lite::nnrt