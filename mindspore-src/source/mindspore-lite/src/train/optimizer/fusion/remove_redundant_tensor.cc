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

#include "src/train/optimizer/fusion/remove_redundant_tensor.h"
#include <map>
#include "src/common/log_adapter.h"
#include "nnacl_c/op_base.h"

namespace mindspore {
namespace lite {
STATUS RemoveRedundantTensor::Run(schema::MetaGraphT *graph) {
  if (graph == nullptr) {
    MS_LOG(ERROR) << "The graph is a nullptr.";
    return RET_NULL_PTR;
  }
  std::map<uint32_t, uint32_t> index_map;
  uint32_t index = 0;
  auto graph_input_index = graph->inputIndex;
  graph->inputIndex.clear();
  for (auto input_index : graph_input_index) {
    if (index_map.find(input_index) == index_map.end()) {
      index_map[input_index] = index;
      ++index;
    }
    graph->inputIndex.push_back(index_map[input_index]);
  }
  for (auto &node : graph->nodes) {
    auto node_in_index = node->inputIndex;
    node->inputIndex.clear();
    for (auto in_index : node_in_index) {
      if (index_map.find(in_index) == index_map.end()) {
        index_map[in_index] = index;
        ++index;
      }
      node->inputIndex.push_back(index_map[in_index]);
    }
    auto node_out_index = node->outputIndex;
    node->outputIndex.clear();
    for (auto out_index : node_out_index) {
      if (index_map.find(out_index) == index_map.end()) {
        index_map[out_index] = index;
        ++index;
      }
      node->outputIndex.push_back(index_map[out_index]);
    }
  }
  auto graph_output_index = graph->outputIndex;
  graph->outputIndex.clear();
  for (auto output_index : graph_output_index) {
    if (index_map.find(output_index) == index_map.end()) {
      index_map[output_index] = index;
      ++index;
    }
    graph->outputIndex.push_back(index_map[output_index]);
  }
  std::vector<std::unique_ptr<mindspore::schema::TensorT>> old_tensors;
  old_tensors.swap(graph->allTensors);
  graph->allTensors.resize(index_map.size());
  for (size_t i = 0; i < old_tensors.size(); ++i) {
    if (index_map.find(i) == index_map.end()) {
      continue;
    }
    graph->allTensors[index_map[i]].swap(old_tensors[i]);
  }
  if (!graph->subGraph.empty()) {
    graph->subGraph[0]->inputIndices = graph->inputIndex;
    graph->subGraph[0]->outputIndices = graph->outputIndex;
    graph->subGraph[0]->tensorIndices = {};
    for (uint32_t i = 0; i < index; ++i) {
      graph->subGraph[0]->tensorIndices.push_back(i);
    }
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
