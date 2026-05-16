/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "src/common/config_file.h"

#ifdef _MSC_VER
#define PATH_MAX 1024
#endif
namespace {
constexpr size_t kLengthOfParentheses = 2;
constexpr size_t kMinSectionLineLength = 2;
constexpr size_t kMaxValidLineCount = 100000;
constexpr size_t kMaxLineCount = 100100;
const std::string kDataType = "data_type";
const std::string kDataTypeFloat16 = "float16";
const std::string kDataTypeFloat32 = "float32";
const std::string kBackend = "backend";
const std::string kBackendCPU = "CPU";
}  // namespace

namespace mindspore {
namespace lite {
namespace {
void ParseLine(const std::string &line, std::map<std::string, std::string> *section_config, std::string *section,
               size_t *valid_line_count, std::map<std::string, std::map<std::string, std::string>> *config) {
  // eg: [section]
  //     key=value
  if (line[0] == '[' && line[line.length() - 1] == ']') {
    if (!section->empty() && !section_config->empty()) {
      (void)config->insert(std::make_pair(*section, *section_config));
    }
    section_config->clear();
    *section = line.substr(1, line.length() - kLengthOfParentheses);
    *valid_line_count = *valid_line_count + 1;
  }

  if (!section->empty()) {
    auto index = line.find('=');
    if (index == std::string::npos) {
      return;
    }
    auto key = line.substr(0, index);
    if (index + 1 > line.size()) {
      return;
    }
    auto value = line.substr(index + 1);
    lite::Trim(&key);
    lite::Trim(&value);
    (void)section_config->insert(std::make_pair(key, value));
    *valid_line_count = *valid_line_count + 1;
  }
}
}  // namespace

int GetAllSectionInfoFromConfigFile(const std::string &file, ConfigInfos *config) {
  if (file.empty() || config == nullptr) {
    MS_LOG(ERROR) << "input Invalid!check file and config.";
    return RET_ERROR;
  }
  auto resolved_path = std::make_unique<char[]>(PATH_MAX);
  if (resolved_path == nullptr) {
    MS_LOG(ERROR) << "new resolved_path fail!";
    return RET_ERROR;
  }

#ifdef _WIN32
  char *real_path = _fullpath(resolved_path.get(), file.c_str(), MAX_CONFIG_FILE_LENGTH);
#else
  char *real_path = realpath(file.c_str(), resolved_path.get());
#endif
  if (real_path == nullptr || strlen(real_path) == 0) {
    MS_LOG(ERROR) << "file path is not valid.";
    return RET_ERROR;
  }
  std::ifstream ifs(resolved_path.get());
  if (!ifs.good()) {
    MS_LOG(ERROR) << "file is not exist";
    return RET_ERROR;
  }
  if (!ifs.is_open()) {
    MS_LOG(ERROR) << "file open failed";
    return RET_ERROR;
  }
  std::string line;
  std::string section;
  std::map<std::string, std::string> section_config;
  size_t line_count = 0;
  size_t valid_line_count = 0;
  while (std::getline(ifs, line)) {
    line_count++;
    if (line_count >= kMaxLineCount || valid_line_count >= kMaxValidLineCount) {
      MS_LOG(ERROR) << "config too many lines!";
      ifs.close();
      return RET_ERROR;
    }
    lite::Trim(&line);
    if (line.length() <= kMinSectionLineLength || line[0] == '#') {
      continue;
    }
    ParseLine(line, &section_config, &section, &valid_line_count, config);
  }
  if (!section.empty() && !section_config.empty()) {
    (void)config->insert(std::make_pair(section, section_config));
  }
  ifs.close();
  return RET_OK;
}

void ParserExecutionPlan(const std::map<std::string, std::string> *config_infos,
                         std::map<std::string, TypeId> *data_type_plan) {
  for (auto info : *config_infos) {
    std::string op_name = info.first;
    std::string value = info.second;
    if (value.empty()) {
      MS_LOG(WARNING) << "Empty info in execution_plan";
      continue;
    }
    if (value[0] == '"' && value[value.length() - 1] == '"') {
      value = value.substr(1, value.length() - kLengthOfParentheses);
    }
    auto index = value.find(':');
    if (index == std::string::npos) {
      MS_LOG(WARNING) << "Invalid info in execution_plan.";
      continue;
    }
    auto data_type_key = value.substr(0, index);
    if (index + 1 > value.size()) {
      return;
    }
    auto data_type_value = value.substr(index + 1);
    if (data_type_key != "data_type") {
      MS_LOG(WARNING) << "Invalid key in execution_plan.";
      continue;
    }
    TypeId type_id = kTypeUnknown;
    if (data_type_value == "float32") {
      type_id = kNumberTypeFloat32;
    } else if (data_type_value == "float16") {
      type_id = kNumberTypeFloat16;
    } else {
      MS_LOG(WARNING) << "Invalid value in execution_plan.";
      continue;
    }
    (void)data_type_plan->insert(std::make_pair(op_name, type_id));
  }
}

void ParseTargetTypeID(const std::string &key, const std::string &real_value,
                       bool &has_data_type, TypeId &type_id,
                       bool &has_op_backend, TypeId &backend_type_id) {
  if (key == kDataType) {
    has_data_type = true;
    if (real_value == kDataTypeFloat32) {
      type_id = kNumberTypeFloat32;
    } else if (real_value == kDataTypeFloat16) {
      type_id = kNumberTypeFloat16;
    } else {
      MS_LOG(WARNING) << "Invalid data_type value: " << real_value << ", will be ignored";
      type_id = kTypeUnknown;
    }
  } else if (key == kBackend) {
    has_op_backend = true;
    backend_type_id = kBackendTypeCPU;  // Only CPU reaches here after Phase 1 validation
  }
  MS_LOG(DEBUG) << "Success to parse current info in execution_plan: " << real_value
               << ", type_id:" << type_id << ", backend_type_id:" << backend_type_id;
}

int ParserMultiExecutionPlan(
    const std::map<std::string, std::string> *config_infos,
    std::map<std::string, TypeId> *data_type_plan,
    std::map<std::string, TypeId> *op_backend_plan) {
  MS_LOG(DEBUG) << "Parsing execution plan with " << config_infos->size() << " operators";

  // Phase 1: Validate backend configurations - only CPU allowed
  for (auto info : *config_infos) {
    std::string op_name = info.first;
    std::string value = info.second;

    if (value.empty()) {
      MS_LOG(ERROR) << "Empty configuration for operator: " << op_name;
      return RET_ERROR;
    }

    if (value[0] == '"' && value[value.length() - 1] == '"') {
      value = value.substr(1, value.length() - kLengthOfParentheses);
    }
    std::istringstream iss(value);
    std::string attr;
    while (std::getline(iss, attr, ';')) {
      size_t valid_index = attr.find(':');
      if (valid_index == std::string::npos) {
        MS_LOG(ERROR) << "Invalid format in config for operator '" << op_name << "': " << attr;
        return RET_ERROR;
      }
      std::string key = attr.substr(0, valid_index);
      std::string real_value = attr.substr(valid_index + 1);

      if (key == kBackend && real_value != kBackendCPU) {
        MS_LOG(ERROR) << "Invalid backend '" << real_value << "' for operator '" << op_name
                      << "'. Only CPU backend is supported.";
        return RET_ERROR;
      }
    }
  }

  // Phase 2: Insert configurations
  for (auto info : *config_infos) {
    std::string op_name = info.first;
    std::string value = info.second;
    if (value.empty()) {
      MS_LOG(ERROR) << "Empty configuration for operator: " << op_name;
      return RET_ERROR;
    }
    if (value[0] == '"' && value[value.length() - 1] == '"') {
      value = value.substr(1, value.length() - kLengthOfParentheses);
    }
    std::istringstream iss(value);
    std::string attr;
    bool has_data_type = false;
    bool has_op_backend = false;
    TypeId type_id = kTypeUnknown;
    TypeId backend_type_id = kTypeUnknown;
    while (std::getline(iss, attr, ';')) {
      size_t valid_index = attr.find(':');
      if (valid_index == std::string::npos) {
        MS_LOG(ERROR) << "Invalid format in config for operator '" << op_name << "'";
        return RET_ERROR;
      }
      std::string key = attr.substr(0, valid_index);
      std::string real_value = attr.substr(valid_index + 1);
      ParseTargetTypeID(key, real_value, has_data_type, type_id, has_op_backend, backend_type_id);
    }
    if (has_data_type) {
      (void)data_type_plan->insert(std::make_pair(op_name, type_id));
    }
    if (has_op_backend) {
      (void)op_backend_plan->insert(std::make_pair(op_name, backend_type_id));
    }
  }

  MS_LOG(DEBUG) << "Successfully loaded execution plan: " << op_backend_plan->size() << " operators configured";
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
