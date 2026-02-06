/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "tools/converter/legacy_optimizer/graph/node_name_pass.h"
#include <string>
#include <vector>
#include "tools/converter/converter_context.h"

namespace mindspore::lite {
std::string CutShortName(const std::string &fullname, const std::string &delimiter) {
  size_t end_pos = fullname.find_last_of(delimiter);
  std::string name = "";
  if (end_pos != std::string::npos) {
    name = fullname.substr(end_pos + 1);
  }
  if ((fullname.find("op") != std::string::npos) && (name.find("op") == std::string::npos) &&
      (end_pos != std::string::npos)) {
    size_t pos = fullname.rfind(delimiter, end_pos - 1);
    if (pos != std::string::npos) {
      name.insert(0, fullname.substr(pos + 1, end_pos - pos));
    } else {
      name.insert(0, fullname.substr(0, end_pos + 1));
    }
  }

  const std::vector<std::string> loss_names = {"loss_fct", "_loss_fn", "SigmoidCrossEntropy"};
  for (auto &s : loss_names) {
    if (fullname.find(s) != std::string::npos) {
      name.insert(0, s + "/");
      break;
    }
  }

  if (fullname.find("Gradients") != std::string::npos) {
    size_t pos = fullname.find(delimiter);
    if (pos != std::string::npos) {
      name.insert(0, fullname.substr(0, pos + 1));
    }
  }
  return name;
}

STATUS NodeNamePass::Run(schema::MetaGraphT *graph) {
  if (graph == nullptr) {
    MS_LOG(ERROR) << "graph is nullptr";
    return RET_NULL_PTR;
  }

  std::string delimiter = "/";
  for (auto &node : graph->nodes) {
    if (node == nullptr || node->primitive == nullptr) {
      MS_LOG(ERROR) << "node or node->primitive is nullptr";
      return RET_NULL_PTR;
    }
    std::string node_name = CutShortName(node->name, delimiter);
    node->name = node_name != "" ? node_name : node->name;

    for (int i = 0; i < static_cast<int>(node->inputIndex.size()); i++) {
      auto tensor_id = node->inputIndex.at(i);
      auto &tensor = graph->allTensors.at(tensor_id);
      if (tensor->name.empty()) {
        MS_LOG(DEBUG) << "input tensor (id = " << tensor_id << ") name is null";
        tensor->name = node->name + "/input-" + std::to_string(i);
      } else {
        std::string in_tensor_name = CutShortName(tensor->name, delimiter);
        tensor->name = in_tensor_name != "" ? in_tensor_name : tensor->name;
      }
    }

    for (int i = 0; i < static_cast<int>(node->outputIndex.size()); i++) {
      auto tensor_id = node->outputIndex.at(i);
      auto &tensor = graph->allTensors.at(tensor_id);
      if (tensor->name.empty()) {
        tensor->name = node->name + "/output-" + std::to_string(i);
      } else {
        std::string out_tensor_name = CutShortName(tensor->name, delimiter);
        tensor->name = out_tensor_name != "" ? out_tensor_name : tensor->name;
      }
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite