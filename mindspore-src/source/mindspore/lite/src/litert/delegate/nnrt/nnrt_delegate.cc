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

#include <unordered_set>
#include <numeric>
#include "nnrt_delegate.h"
#include "checker/primitive_check.h"
#include "src/common/log_adapter.h"
#include "neural_network_runtime/neural_network_runtime.h"
#include "neural_network_runtime_inner.h"
#include "nnrt_model_kernel.h"
#include "schema/model_generated.h"
#include "schema/ops_generated.h"
#include "flatbuffers/flatbuffers.h"
#include "litert/tensor_category.h"

namespace mindspore {
namespace lite {
Status NNRTDelegate::Init() {
#ifdef SUPPORT_NNRT_METAGRAPH
  auto ret = mindspore::lite::LoadHiaiFLibraryFromPath(&hiai_handle_);
  if (!ret || hiai_handle_ == nullptr) {
    MS_LOG(WARNING) << "Load HiAI_Foundation so failed.";
  }
#endif
  return kSuccess;
}

void NNRTDelegate::InitExtensionOptions() {
  const auto &extensions = nnrt_device_info_.extensions_;
  mindspore::lite::nnrt::ExtensionOptionsParser::Parse(extensions, &extension_options_);
}

Status NNRTDelegate::Build(DelegateModel<schema::Primitive> *model) {
  // dequant litegraph
  auto ret_dequant = DequantLiteGraph(lite_graph_);
  if (ret_dequant != kSuccess) {
    MS_LOG(ERROR) << "Dequant litegraph failed.";
    return kLiteError;
  }
#ifdef SUPPORT_NNRT_METAGRAPH
  InitExtensionOptions();
  if (IsKirinNPUWithOnlineInference()) {
    MS_LOG(DEBUG) << "Choose to build online inference model";
    return BuildKirinNPUModel(model);
  }
  if (IsKirinNPUWithOfflineInference()) {
    MS_LOG(DEBUG) << "Choose to build offline inference model";
    return BuildOfflineModel(model);
  }
#endif

  return BuildNormalModel(model);
}

bool NNRTDelegate::IsCustomModel() const {
  // check if there is only one Cutsom kernel in LiteModel.
  if (lite_graph_ == nullptr) {
    return false;
  }
  if (lite_graph_->all_nodes_.size() != 1) {
    return false;
  }
  auto node = lite_graph_->all_nodes_[0];
  if (node == nullptr) {
    return false;
  }
  if (node->node_type_ != mindspore::schema::PrimitiveType_Custom) {
    return false;
  }
  return true;
}

#ifdef SUPPORT_NNRT_METAGRAPH
bool NNRTDelegate::CheckNPUPrefix(const std::string prefix_name) const {
  const std::string kirin_npu_name_prefix = prefix_name;
  auto device_id = nnrt_device_info_.device_id_;
  const char *device_name;
  auto ret = OH_NNDevice_GetName(device_id, &device_name);
  if (ret != OH_NN_SUCCESS) {
    MS_LOG(WARNING) << "Get name of device: " << device_id << " failed, error: " << ret;
    return false;
  }

  if (strncmp(kirin_npu_name_prefix.c_str(), device_name, kirin_npu_name_prefix.size()) != 0) {
    MS_LOG(WARNING) << "strncmp: " << device_id << " failed, device_name: " << device_name;
    return false;
  }
  return true;
}

bool NNRTDelegate::IsKirinNPUWithOnlineInference() const {
  return CheckNPUPrefix("NPU_");
}

bool NNRTDelegate::IsKirinNPUWithOfflineInference() const {
  return CheckNPUPrefix("HIAI_F");
}

Status NNRTDelegate::BuildKirinNPUModel(DelegateModel<schema::Primitive> *model) {
  OH_NNModel *nn_model = OH_NNModel_Construct();
  if (nn_model == nullptr) {
    MS_LOG(ERROR) << "Create NNModel failed, result is nullptr";
    return kLiteNullptr;
  }

  size_t extension_size = nnrt_device_info_.extensions_.size();
  std::vector<OH_NN_Extension> extensions;
  MS_LOG_DEBUG << "set extensions, item number: " << extension_size;
  const size_t kExtensionNameMax = 128; // This is a length limitation in NNRT API.
  for (size_t i = 0; i < extension_size; i++) {
    auto &src_extension = nnrt_device_info_.extensions_[i];
    OH_NN_Extension dst_extension;
    dst_extension.name[kExtensionNameMax - 1] = '\0';
    strncpy(dst_extension.name, src_extension.name.c_str(), kExtensionNameMax - 1);
    dst_extension.value = (char *)((void *)src_extension.value.data());
    dst_extension.valueSize = src_extension.value.size();
    extensions.push_back(dst_extension);
    MS_LOG_DEBUG << "set extension, item name: " << dst_extension.name << ", value size: " << dst_extension.valueSize;
  }

  auto ret = OH_NNModel_BuildFromLiteGraph(nn_model, lite_graph_, extensions.data(), extensions.size());
  if (ret != OH_NN_SUCCESS) {
    MS_LOG(ERROR) << "Build NNModel failed, ret: " << ret;
    OH_NNModel_Destroy(&nn_model);
    return kLiteError;
  }

  auto ret2 =  CreateFullModelKernel(model, nn_model);
  if (ret2 != kSuccess) {
    MS_LOG(ERROR) << "Create full model kernel failed, ret: " << ret2;
    return kLiteError;
  }
  return kSuccess;
}

namespace {
constexpr int32_t kNum2 = 2;
}

Status NNRTDelegate::BuildOfflineModel(DelegateModel<schema::Primitive> *model) {
  if (!IsCustomModel()) {
    MS_LOG(ERROR) << "not third party model";
    return kLiteNullptr;
  }

  auto node = lite_graph_->all_nodes_[0];
  MS_CHECK_TRUE_RET(node != nullptr, kLiteError);
  auto input_num = node->input_indices_.size();
  // at least one input and one OM model buffer(as the last constant input)
  MS_CHECK_TRUE_RET(input_num >= kNum2, kLiteError);
  MS_CHECK_TRUE_RET(lite_graph_->all_tensors_.size() >= kNum2, kLiteError);
  auto tensor = lite_graph_->all_tensors_[node->input_indices_[input_num - 1]];
  MS_CHECK_TRUE_RET(tensor != nullptr, kLiteError);
  MS_CHECK_TRUE_RET(tensor->data() != nullptr, kLiteError);
  const uint8_t *model_buf = static_cast<const uint8_t *>(tensor->data()->data());
  size_t model_size = tensor->data()->size();

  OH_NNCompilation *nn_compilation = OH_NNCompilation_ConstructWithOfflineModelBuffer(model_buf, model_size);
  if (nn_compilation == nullptr) {
    MS_LOG(ERROR) << "Construct Offline NNCompilation failed";
    return kLiteError;
  }
  MS_LOG(DEBUG) << "NNRTDelegate creates NNCompilation success.";

  auto ret_code = InitNNCompilation(nn_compilation);
  if (ret_code != kSuccess) {
    MS_LOG(ERROR) << "Init NNCompilation failed";
    OH_NNCompilation_Destroy(&nn_compilation);
    return kLiteError;
  }
  MS_LOG(DEBUG) << "HiAI F InitNNCompilation success";

  OH_NNExecutor *nn_executor = nullptr;
  nn_executor = OH_NNExecutor_Construct(nn_compilation);
  if (nn_executor == nullptr) {
    MS_LOG(ERROR) << "Construct NNExecutor failed, ret: " << ret_code;
    OH_NNCompilation_Destroy(&nn_compilation);
    return kLiteError;
  }
  OH_NNCompilation_Destroy(&nn_compilation);

  auto nnrt_model_kernel = new (std::nothrow) NNRTModelKernel(nn_executor, nnrt_device_info_, model->inputs(), model->outputs());
  if (nnrt_model_kernel == nullptr) {
    OH_NNExecutor_Destroy(&nn_executor);
    MS_LOG(ERROR) << "new NNRTModelKernel failed";
    return kLiteError;
  }
  nn_executor_list_.push_back(nn_executor);

  (void)model->Replace(model->BeginKernelIterator(), model->EndKernelIterator(), nnrt_model_kernel);
  return kSuccess;
}

Status NNRTDelegate::CreateFullModelKernel(DelegateModel<schema::Primitive> *model, OH_NNModel *nn_model) {
  OH_NNCompilation *nn_compilation = OH_NNCompilation_Construct(nn_model);
  if (nn_compilation == nullptr) {
    MS_LOG(ERROR) << "Construct NNCompilation failed";
    OH_NNModel_Destroy(&nn_model);
    return kLiteError;
  }
  MS_LOG(DEBUG) << "NNRTDelegate creates NNCompilation success.";

  auto ret_code = InitNNCompilation(nn_compilation);
  if (ret_code != kSuccess) {
    MS_LOG(ERROR) << "Init NNCompilation failed";
    OH_NNModel_Destroy(&nn_model);
    OH_NNCompilation_Destroy(&nn_compilation);
    return kLiteError;
  }
  OH_NNModel_Destroy(&nn_model);

  OH_NNExecutor *nn_executor = nullptr;
  nn_executor = OH_NNExecutor_Construct(nn_compilation);
  if (nn_executor == nullptr) {
    MS_LOG(ERROR) << "Construct NNExecutor failed, ret: " << ret_code;
    OH_NNCompilation_Destroy(&nn_compilation);
    return kLiteError;
  }
  OH_NNCompilation_Destroy(&nn_compilation);

  auto nnrt_model_kernel = new (std::nothrow) NNRTModelKernel(nn_executor, nnrt_device_info_, model->inputs(), model->outputs());
  if (nnrt_model_kernel == nullptr) {
    OH_NNExecutor_Destroy(&nn_executor);
    MS_LOG(ERROR) << "new NNRTModelKernel failed";
    return kLiteError;
  }
  nn_executor_list_.push_back(nn_executor);

  model->Replace(model->BeginKernelIterator(), model->EndKernelIterator(), nnrt_model_kernel);
  return kSuccess;
}
#endif

Status NNRTDelegate::BuildNormalModel(DelegateModel<schema::Primitive> *model) {
  MS_LOG(DEBUG) << "Start to build NNRT model.";
  if ((lite_graph_ == nullptr) || (lite_graph_->sub_graphs_.size() > 1)) {
    MS_LOG(WARNING) << "LiteGraph contains more than one subgraph. NNRT does not support control-flow model yet, fallback to CPU";
    return kSuccess;
  }

  OH_NNModel *full_model = CreateFullNNModel();
  if (full_model == nullptr) {
    MS_LOG(WARNING) << "Build full NNModel failed, fallback to CPU";
    return kSuccess;
  }
  std::vector<bool> op_supports = QueryOpSupports(full_model);
  if (op_supports.empty()) {
    MS_LOG(WARNING) << "Query no op supports for full model, fallback to CPU";
    OH_NNModel_Destroy(&full_model);
    return kSuccess;
  }
  auto nnrt_subgraph_ranges = GetNNRTSubgraphRanges(model, op_supports);
  MS_LOG(INFO) << "Found NNRT subgraph count: " << nnrt_subgraph_ranges.size();

  std::vector<LiteGraph *> sub_lite_graphs;
  auto ret = CreateLiteGraphForNNRTSubgraph(nnrt_subgraph_ranges, &sub_lite_graphs);
  if (ret != kSuccess) {
    OH_NNModel_Destroy(&full_model);
    MS_LOG(WARNING) << "Create NNRT sub LiteGraph failed, fallback to CPU";
    return kSuccess;
  }

  std::vector<NNRTModelKernel *> nnrt_subgraph_kernels;
  ret = CreateNNRTSubgraphKernels(model, sub_lite_graphs, nnrt_subgraph_ranges, &nnrt_subgraph_kernels);
  if (ret != kSuccess) {
    OH_NNModel_Destroy(&full_model);
    MS_LOG(WARNING) << "Create NNRT subgraph kernel failed, fallback to CPU";
    return kSuccess;
  }

  ReplaceNNRTKernelsInDelegateModel(model, nnrt_subgraph_ranges, nnrt_subgraph_kernels);
  OH_NNModel_Destroy(&full_model);
  MS_LOG(INFO) << "NNRTDelegate build success.";
  return kSuccess;
}

OH_NNModel *NNRTDelegate::CreateFullNNModel() {
  if (lite_graph_ == nullptr) {
    MS_LOG(ERROR) << "Lite graph is null";
    return nullptr;
  }

  if (lite_graph_->sub_graphs_.empty()) {
    MS_LOG(ERROR) << "Lite graph must have at lease one subgraph";
    return nullptr;
  }

  OH_NNModel *nn_model = OH_NNModel_Construct();
  if (nn_model == nullptr) {
    MS_LOG(ERROR) << "Create NNModel failed, result is nullptr";
    return nullptr;
  }

  auto ret = OH_NNModel_BuildFromLiteGraph(nn_model, lite_graph_, nullptr, 0);
  if (ret != OH_NN_SUCCESS) {
    MS_LOG(ERROR) << "Build NNModel failed, ret: " << ret;
    OH_NNModel_Destroy(&nn_model);
    return nullptr;
  }
  return nn_model;
}

std::vector<bool> NNRTDelegate::QueryOpSupports(OH_NNModel *nn_model) {
  const bool *is_supported = nullptr; // Note: this memory is owned by nn_model, don't free alone.
  uint32_t op_count = 0;
  auto ret = OH_NNModel_GetAvailableOperations(nn_model, nnrt_device_info_.device_id_, &is_supported, &op_count);
  if (ret != OH_NN_SUCCESS) {
    MS_LOG(WARNING) << "NNModel GetAvailableOperations failed, ret: " << ret
                  << ", maybe caused by dataParcel data length limitation";
    return {};
  }
  std::vector<bool> op_supports(is_supported, is_supported + op_count);
  return op_supports;
}

/* Find continuous sub-sequence in op_supports. */
std::vector<NNRTOpRange> NNRTDelegate::GetNNRTSubgraphRanges(DelegateModel<schema::Primitive> *model,
                                                             const std::vector<bool> &op_supports) {
  std::vector<NNRTOpRange> nnrt_subgraph_ranges;
  NNRTOpRange op_range;
  bool start_count = false;
  for (size_t i = 0; i < op_supports.size(); i++) {
    if (op_supports[i]) {
      if (start_count == false) {
        start_count = true;
        op_range.begin_index_ = i;
        op_range.begin_iter_ = model->BeginKernelIterator() + i;
      }
    } else {
      if (start_count == true) {
        start_count = false;
        op_range.end_index_ = i;
        op_range.end_iter_ = model->BeginKernelIterator() + i;
        nnrt_subgraph_ranges.push_back(op_range);
      }
    }
  }
  // handle last true subsequence
  if (start_count == true) {
    op_range.end_index_ = op_supports.size();
    op_range.end_iter_ = model->EndKernelIterator();
    nnrt_subgraph_ranges.push_back(op_range);
    MS_LOG(INFO) << "Schedule NNRT subgraph range: [" << op_range.begin_index_ << ", " << op_range.end_index_ << ")";
  }
  return nnrt_subgraph_ranges;
}

/**
 * This method ONLY works when the follow pre-conditions are satisfied:
 * 1. The node order of lite_graph_->all_nodes should be consistent with DelegateModel sequence.
 *  This ensures the kernel replacement in DelegateModel based on the re-organizing info from lite_graph_ is correct.
 * 2. The node indices of lite_graph_->sub_graphs[0].node_indices should be monotonically increasing from 0 to size - 1.
 */
Status NNRTDelegate::CreateLiteGraphForNNRTSubgraph(
    const std::vector<NNRTOpRange> &nnrt_op_ranges,
    std::vector<LiteGraph *> *sub_lite_graphs) {
  MS_LOG(INFO) << "Start creating LiteGraph for NNRT subgraph";
  for (const auto &op_range: nnrt_op_ranges) {
    MS_LOG(INFO) << "Process op range: [" << op_range.begin_index_ << ", " << op_range.end_index_ << ")";
    LiteGraph *sub_lite_graph = new (std::nothrow)LiteGraph;
    if (sub_lite_graph == nullptr) {
      MS_LOG(ERROR) << "Allocate LiteGraph failed";
      return kLiteError;
    }
    sub_lite_graph->name_ = lite_graph_->name_;
    sub_lite_graph->version_ = lite_graph_->version_;

    auto sub_graph = new (std::nothrow)LiteGraph::SubGraph;
    if (sub_graph == nullptr) {
      MS_LOG(ERROR) << "Allocate SubGraph failed";
      return kLiteError;
    }
    sub_graph->name_ = lite_graph_->name_;
    sub_lite_graph->sub_graphs_.push_back(sub_graph);

    // deal with all_nodes
    MS_LOG(INFO) << "Assemble all_nodes...";
    int new_node_index = 0;
    std::map<uint32_t, schema::Tensor *> in_tensor_index_map;
    std::map<uint32_t, schema::Tensor *> out_tensor_index_map;
    for (size_t index = op_range.begin_index_; index < op_range.end_index_; index++) {
      LiteGraph::Node *node = new (std::nothrow)LiteGraph::Node;
      if (node == nullptr) {
        MS_LOG(ERROR) << "Allocate Node failed";
        return kLiteError;
      }
      *node = *lite_graph_->all_nodes_[index];
      sub_lite_graph->all_nodes_.push_back(node);
      sub_graph->node_indices_.push_back(new_node_index++);

      for (auto i: node->input_indices_) {
        in_tensor_index_map.emplace(i, lite_graph_->all_tensors_[i]);
      }
      for (auto i: node->output_indices_) {
        out_tensor_index_map.emplace(i, lite_graph_->all_tensors_[i]);
      }
    }

    // deal with all_tensors
    MS_LOG(INFO) << "Assemble all_tensors...";
    std::set<schema::Tensor *> tensors;
    for (auto iter: in_tensor_index_map) {
      tensors.emplace(iter.second);
    }
    for (auto iter: out_tensor_index_map) {
      tensors.emplace(iter.second);
    }

    uint32_t new_index = 0;
    std::map<schema::Tensor *, uint32_t> new_tensor_maps;
    for (auto tensor: tensors) {
      new_tensor_maps.emplace(tensor, new_index++);
    }

    sub_lite_graph->all_tensors_ = std::vector<schema::Tensor *>(tensors.begin(), tensors.end());

    // deal with every node's input/output indices
    MS_LOG(INFO) << "Set input/output indices of each node...";
    for (auto node: sub_lite_graph->all_nodes_) {
      for (auto &index : node->input_indices_) {
        index = new_tensor_maps.at(in_tensor_index_map.at(index));
      }
      for (auto &index : node->output_indices_) {
        index = new_tensor_maps.at(out_tensor_index_map.at(index));
      }
    }

    // deal with subgraph's input/output indices
    MS_LOG(INFO) << "Set input/output indices of each subgraph...";
    sub_graph->tensor_indices_ = std::vector<uint32_t>(tensors.size());
    std::iota(sub_graph->tensor_indices_.begin(), sub_graph->tensor_indices_.end(), 0U);

    for (auto iter: in_tensor_index_map) {
      auto new_tensor_index = new_tensor_maps[iter.second];
      MS_LOG(DEBUG) << "handle input: old: " << iter.first << ", new: " << new_tensor_index << std::endl;
      if (IsConstTensor(*iter.second)) {
        MS_LOG(DEBUG) << "- tensor: " << new_tensor_index << " is const." << std::endl;
        continue;
      }

      bool is_subgraph_input = true;
      for (auto node: sub_lite_graph->all_nodes_) {
        if (std::find(node->output_indices_.begin(), node->output_indices_.end(), new_tensor_index) !=
            node->output_indices_.end()) {
          is_subgraph_input = false;
          MS_LOG(DEBUG) << "- tensor: " << new_tensor_index << " is not subgraph input." << std::endl;
          break;
        }
      }
      if (is_subgraph_input) {
        sub_graph->input_indices_.push_back(new_tensor_index);
        MS_LOG(DEBUG) << "- select tensor: " << new_tensor_index << " as subgraph input." << std::endl;
      }
    }

    for (auto iter: out_tensor_index_map) {
      int new_tensor_index = new_tensor_maps.at(iter.second);
      MS_LOG(DEBUG) << "handle output: old: " << iter.first << ", new: " << new_tensor_index << std::endl;
      if (IsConstTensor(*iter.second)) {
        MS_LOG(DEBUG) << "- tensor: " << new_tensor_index << " is const." << std::endl;
        continue;
      }

      bool is_subgraph_output = false;
      for (size_t i = 0; i < lite_graph_->all_nodes_.size(); i++) {
        if ((i >= op_range.begin_index_) && (i < op_range.end_index_)) {
          continue;
        }
        auto node = lite_graph_->all_nodes_[i];
        if (std::find(node->input_indices_.begin(), node->input_indices_.end(), iter.first) !=
            node->input_indices_.end()) { // As the input of node which does not belong to the subgraph.
          is_subgraph_output = true;
          MS_LOG(DEBUG) << "- tensor: " << new_tensor_index << " is original subgraph output. node: " << node->primitive_ << std::endl;
          break;
        }
      }
      bool is_graph_output = (std::find(lite_graph_->output_indices_.begin(),lite_graph_->output_indices_.end(),
                                        iter.first) != lite_graph_->output_indices_.end());
      if (is_graph_output) {
        MS_LOG(DEBUG) << "- tensor: " << new_tensor_index << " is graph output." << std::endl;
      }
      if (is_subgraph_output || is_graph_output) {
        sub_graph->output_indices_.push_back(new_tensor_index);
        MS_LOG(DEBUG) << "- select tensor: " << new_tensor_index << " as subgraph output." << std::endl;
      }
    }

    // deal with full-graph's input/output indices
    sub_lite_graph->input_indices_ = sub_graph->input_indices_;
    sub_lite_graph->output_indices_ = sub_graph->output_indices_;
    sub_lite_graphs->push_back(sub_lite_graph);
  }
  MS_LOG(INFO) << "Finished creating LiteGraph for NNRT subgraph";
  return kSuccess;
}

struct TensorLocation {
  uint32_t node_index; // the index of node which the tensor belongs to.
  uint32_t tensor_index; // the index of node in/out tensors which the tensor is located at.
};

Status NNRTDelegate::InitNNCompilation(OH_NNCompilation *nn_compilation) const {
  auto ret_code = OH_NNCompilation_SetDevice(nn_compilation, nnrt_device_info_.device_id_);
  if (ret_code != OH_NN_SUCCESS) {
    MS_LOG(ERROR) << "NNCompilation set device id failed, ret: " << ret_code;
    return kLiteError;
  }
  ret_code = OH_NNCompilation_SetPerformanceMode(nn_compilation,
                                                 (OH_NN_PerformanceMode)(nnrt_device_info_.performance_mode_));
  if ((ret_code != OH_NN_SUCCESS) && (ret_code != OH_NN_OPERATION_FORBIDDEN)) {
    MS_LOG(ERROR) << "NNCompilation set performance mode failed, ret: " << ret_code;
    return kLiteError;
  }
  ret_code = OH_NNCompilation_SetPriority(nn_compilation, (OH_NN_Priority)(nnrt_device_info_.priority_));
  if ((ret_code != OH_NN_SUCCESS) && (ret_code != OH_NN_OPERATION_FORBIDDEN)) {
    MS_LOG(ERROR) << "NNCompilation set priority failed, ret: " << ret_code;
    return kLiteError;
  }
  ret_code = OH_NNCompilation_EnableFloat16(nn_compilation, nnrt_device_info_.enable_fp16_);
  if ((ret_code != OH_NN_SUCCESS) && (ret_code != OH_NN_OPERATION_FORBIDDEN)) {
    MS_LOG(ERROR) << "NNCompilation enable fp16 failed, ret: " << ret_code;
    return kLiteError;
  }

  if (!extension_options_.cache_path_.empty()) {  // Set cache path if user indeed set it.
    ret_code = OH_NNCompilation_SetCache(nn_compilation, extension_options_.cache_path_.c_str(),
                                         extension_options_.cache_version_);
    if ((ret_code != OH_NN_SUCCESS) && (ret_code != OH_NN_OPERATION_FORBIDDEN)) {
      MS_LOG(ERROR) << "NNCompilation set cache failed, ret: " << ret_code;
      return kLiteError;
    }
  }

#ifdef SUPPORT_NNRT_METAGRAPH
  if (hiai_handle_ != nullptr && IsKirinNPUWithOfflineInference()) {
    if (extension_options_.band_mode != mindspore::lite::HIAI_BANDMODE_UNSET) {
      ret_code = mindspore::lite::HMS_HiAIOptions_SetBandMode(nn_compilation, extension_options_.band_mode);
      if ((ret_code != OH_NN_SUCCESS) && (ret_code != OH_NN_OPERATION_FORBIDDEN)) {
        MS_LOG(ERROR) << "NNCompilation set BandMode failed, ret: " << ret_code;
        return kLiteError;
      }
    }

    if (extension_options_.is_optional_quant_setted) {
      if (extension_options_.quant_config == nullptr || extension_options_.quant_config_size <= 0) {
        MS_LOG(ERROR) << "NNCompilation set QuantConfig faild, input quant config is invalid, please make sure buffer "
                      << "is not null and size > 0.";
        return kLiteError;
      }
      ret_code = mindspore::lite::HMS_HiAIOptions_SetQuantConfig(nn_compilation, extension_options_.quant_config,
                                                                 extension_options_.quant_config_size);
      if ((ret_code != OH_NN_SUCCESS) && (ret_code != OH_NN_OPERATION_FORBIDDEN)) {
        MS_LOG(ERROR) << "NNCompilation set QuantConfig failed, ret: " << ret_code;
        return kLiteError;
      }
    }
  } else {
    MS_LOG(WARNING) << "hiai_foundation is nullptr.";
  }
#endif

  ret_code = OH_NNCompilation_Build(nn_compilation);
  if (ret_code != OH_NN_SUCCESS) {
    MS_LOG(ERROR) << "Build NNCompilation failed, ret: " << ret_code;
    return kLiteError;
  }
  return kSuccess;
}

Status NNRTDelegate::CreateNNRTSubgraphKernels(DelegateModel<schema::Primitive> *model,
                                               const std::vector<LiteGraph *> &sub_lite_graphs, const std::vector<NNRTOpRange> &nnrt_subgraph_ranges,
                                               std::vector<NNRTModelKernel *> *nnrt_subgraph_kernels) {
  for (size_t i = 0; i < sub_lite_graphs.size(); i++) {
    auto sub_lite_graph = sub_lite_graphs[i];

    OH_NNModel *nn_model = OH_NNModel_Construct();
    auto ret = OH_NNModel_BuildFromLiteGraph(nn_model, sub_lite_graph, nullptr, 0);
    if (ret != OH_NN_SUCCESS) {
      MS_LOG(ERROR) << "Build NNModel failed, ret: " << ret;
      OH_NNModel_Destroy(&nn_model);
      return kLiteError;
    }

    OH_NNCompilation *nn_compilation = OH_NNCompilation_Construct(nn_model);
    if (nn_compilation == nullptr) {
      MS_LOG(ERROR) << "Construct NNCompilation failed";
      OH_NNModel_Destroy(&nn_model);
      return kLiteError;
    }
    MS_LOG(DEBUG) << "NNRTDelegate creates NNCompilation success.";

    auto ret_code = InitNNCompilation(nn_compilation);
    if (ret_code != kSuccess) {
      MS_LOG(ERROR) << "Init NNCompilation failed";
      OH_NNCompilation_Destroy(&nn_compilation);
      OH_NNModel_Destroy(&nn_model);
      return kLiteError;
    }

    OH_NNExecutor *nn_executor = nullptr;
    nn_executor = OH_NNExecutor_Construct(nn_compilation);
    if (nn_executor == nullptr) {
      MS_LOG(ERROR) << "Construct NNExecutor failed, ret: " << ret_code;
      OH_NNCompilation_Destroy(&nn_compilation);
      OH_NNModel_Destroy(&nn_model);
      return kLiteError;
    }
    MS_LOG(DEBUG) << "NNRTDelegate creates NNExecutor success.";

    bool format_not_support = false;
    std::vector<MSTensor> in_tensors;
    for (auto index: sub_lite_graph->sub_graphs_[0]->input_indices_) {
      TensorLocation location;
      for (auto node_index: sub_lite_graph->sub_graphs_[0]->node_indices_) {
        auto node = sub_lite_graph->all_nodes_[node_index];
        auto iter = std::find(node->input_indices_.begin(), node->input_indices_.end(), index);
        if (iter != node->input_indices_.end()) {
          uint32_t tensor_index = iter - node->input_indices_.begin();
          location.node_index = node_index;
          location.tensor_index = tensor_index;
          MS_LOG(INFO) << "Found graph input index: " << index << " is the " << tensor_index << "th input of the node " << node->primitive_;
          break;
        }
      }
      KernelIter kernel_iter = nnrt_subgraph_ranges[i].begin_iter_ + location.node_index;
      in_tensors.push_back((*kernel_iter)->inputs()[location.tensor_index]);
      if (in_tensors.back().format() != Format::NHWC) {
        format_not_support = true;
        break ;
      }
    }

    std::vector<MSTensor> out_tensors;
    for (auto index: sub_lite_graph->sub_graphs_[0]->output_indices_) {
      TensorLocation location;
      for (auto node_index: sub_lite_graph->sub_graphs_[0]->node_indices_) {
        auto node = sub_lite_graph->all_nodes_[node_index];
        auto iter = std::find(node->output_indices_.begin(), node->output_indices_.end(), index);
        if (iter != node->output_indices_.end()) {
          uint32_t tensor_index = iter - node->output_indices_.begin();
          location.node_index = node_index;
          location.tensor_index = tensor_index;
          MS_LOG(INFO) << "Found graph output index: " << index << " is the " << tensor_index << "th output of the node " << node->primitive_;
          break;
        }
      }
      KernelIter kernel_iter = nnrt_subgraph_ranges[i].begin_iter_ + location.node_index;
      out_tensors.push_back((*kernel_iter)->outputs()[location.tensor_index]);
      if (out_tensors.back().format() != Format::NHWC) {
        format_not_support = true;
        break ;
      }
    }
    if (format_not_support) {
      MS_LOG(WARNING) << "Not support in/out tensor format, skip this subgraph";
      OH_NNCompilation_Destroy(&nn_compilation);
      OH_NNModel_Destroy(&nn_model);
      nnrt_subgraph_kernels->push_back(nullptr);
      continue ;
    }

    auto nnrt_model_kernel = new (std::nothrow) NNRTModelKernel(nn_executor, nnrt_device_info_, in_tensors, out_tensors);
    if (nnrt_model_kernel == nullptr) {
      MS_LOG(ERROR) << "new NNRTModelKernel failed";
      return kLiteError;
    }
    nn_executor_list_.push_back(nn_executor);
    OH_NNCompilation_Destroy(&nn_compilation);
    OH_NNModel_Destroy(&nn_model);
    nnrt_subgraph_kernels->push_back(nnrt_model_kernel);
  }
  return kSuccess;
}

void NNRTDelegate::ReplaceNNRTKernelsInDelegateModel(DelegateModel<schema::Primitive> *model,
                                       const std::vector<NNRTOpRange> &nnrt_subgraph_ranges,
                                       const std::vector<NNRTModelKernel *> &nnrt_subgraph_kernels) {
  // Here we perform the replacement from back to front intentionally! If replace from front to end, the kernel
  // sequence would shrink and the later begin_iter_/end_iter_ may be erased already.
  for (int i = nnrt_subgraph_ranges.size() - 1; i >= 0; i--) {
    if (nnrt_subgraph_kernels[i] == nullptr) {
      continue;
    }
    auto from = nnrt_subgraph_ranges[i].begin_iter_;
    auto end = nnrt_subgraph_ranges[i].end_iter_;
    (void)model->Replace(from, end, nnrt_subgraph_kernels[i]);
    MS_LOG(INFO) << "Replace nnrt subgraph kernel in range: [" << (from - model->BeginKernelIterator())
      << ", " << (end - model->BeginKernelIterator()) << ")";
  }
}

Status NNRTDelegate::PrepareInputs(DelegateModel<schema::Primitive> *model,
                                   OH_NNExecutor *oh_nn_executor) {
  auto input_tensors = model->inputs();
  for (size_t i = 0; i < input_tensors.size(); i++) {
    auto tensor = input_tensors[i];
    auto tensor_shape = tensor.Shape();
    auto tmp_quant_param = tensor.QuantParams();
    OH_NN_QuantParam *quant_param = nullptr;
    std::vector<uint32_t> bit_num;
    std::vector<double> scale;
    std::vector<int32_t> zero_point;
    if (!tmp_quant_param.empty()) {
      quant_param = new(std::nothrow) OH_NN_QuantParam;
      if (quant_param == nullptr) {
        MS_LOG(ERROR) << "new OH_NN_QuantParam failed.";
        return kLiteError;
      }
      for (auto qparam : tmp_quant_param) {
        bit_num.emplace_back(qparam.bit_num);
        scale.emplace_back(qparam.scale);
        zero_point.emplace_back(qparam.zero_point);
      }
      quant_param->quantCount = tmp_quant_param.size();
      quant_param->numBits = bit_num.data();
      quant_param->scale = scale.data();
      quant_param->zeroPoint = zero_point.data();
    }
    auto oprend = new(std::nothrow) OH_NN_Tensor;
    if (oprend == nullptr) {
      MS_LOG(ERROR) << "new OH_NN_Tensor Failed";
      return kLiteError;
    }
    oprend->dataType = CastToNNRTDataType(tensor.DataType());
    oprend->dimensionCount = tensor_shape.size();

    std::vector<int32_t> dimensions_list;
    for (auto shape : tensor_shape) {
      if (shape < INT32_MAX) {
        dimensions_list.emplace_back(static_cast<int32_t>(shape));
      } else {
        MS_LOG(ERROR) << "NNExecutor SetInput failed,tensor dimension is is too large, max dim = " << INT32_MAX
                      << ", but get dimension = " << shape;
        return kLiteError;
      }
    }
    oprend->dimensions = dimensions_list.data();
    oprend->quantParam = quant_param;
    oprend->type = OH_NN_TENSOR;
    OH_NN_ReturnCode ret_code =
        OH_NNExecutor_SetInput(oh_nn_executor, i, oprend, tensor.MutableData(), tensor.DataSize());
    delete (oprend);

    if (!tmp_quant_param.empty()) {
      delete (quant_param);
      quant_param = nullptr;
    }

    if (ret_code != OH_NN_SUCCESS) {
      MS_LOG(ERROR) << "NNExecutor SetInput failed, current input tensor is" << tensor.Name()
                    << "OH_NN_ReturnCode = " << ret_code;
      return kLiteError;
    }
  }
  return kSuccess;
}

OH_NN_DataType NNRTDelegate::CastToNNRTDataType(DataType data_type) {
  const std::unordered_map<DataType, OH_NN_DataType> kDataTypeMap = {
      {DataType::kNumberTypeBool, OH_NN_BOOL},
      {DataType::kNumberTypeInt8, OH_NN_INT8},
      {DataType::kNumberTypeInt16, OH_NN_INT16},
      {DataType::kNumberTypeInt32, OH_NN_INT32},
      {DataType::kNumberTypeInt64, OH_NN_INT64},
      {DataType::kNumberTypeUInt8, OH_NN_UINT8},
      {DataType::kNumberTypeUInt16, OH_NN_UINT16},
      {DataType::kNumberTypeUInt32, OH_NN_UINT32},
      {DataType::kNumberTypeUInt64, OH_NN_UINT64},
      {DataType::kNumberTypeFloat16, OH_NN_FLOAT16},
      {DataType::kNumberTypeFloat32, OH_NN_FLOAT32},
      {DataType::kNumberTypeFloat64, OH_NN_FLOAT64},
  };

  auto iter = kDataTypeMap.find(data_type);
  if (iter == kDataTypeMap.end()) {
    return OH_NN_UNKNOWN;
  }
  return iter->second;
}

Status NNRTDelegate::PrepareOutputs(DelegateModel<schema::Primitive> *model,
                                    OH_NNExecutor *oh_nn_executor) {
  auto output_tensors = model->outputs();
  for (size_t i = 0; i < output_tensors.size(); i++) {
    auto tensor = output_tensors[i];
    OH_NN_ReturnCode ret_code = OH_NNExecutor_SetOutput(oh_nn_executor, i, tensor.MutableData(), tensor.DataSize());
    if (ret_code != OH_NN_SUCCESS) {
      MS_LOG(ERROR) << "NNExecutor SetOutput failed, current out tensor is" << tensor.Name()
                    << ", OH_NN_ReturnCode = " << ret_code;
      return kLiteError;
    }
  }
  return kSuccess;
}

schema::Tensor *NNRTDelegate::TensorToSchemaTensor(Tensor *lite_tensor, schema::Tensor *schema_tensor) {
  flatbuffers::FlatBufferBuilder fbb(1024);
  auto shape = lite_tensor->shape();
  std::vector<int32_t> dim_vec(shape.begin(), shape.end());

  auto quant_params = lite_tensor->quant_params();
  std::vector<flatbuffers::Offset<mindspore::schema::QuantParam>> quant_vec;
  quant_vec.reserve(quant_params.size());
  for (auto q_param : quant_params) {
    quant_vec.emplace_back(schema::CreateQuantParam(fbb, q_param.scale, q_param.zeroPoint, 0, 0, true, q_param.bitNum));
  }
  auto quant_clusters = lite_tensor->quant_clusters();

  auto external_data = schema_tensor->externalData();
  std::vector<flatbuffers::Offset<mindspore::schema::ExternalData>> external_data_vec;
  if (external_data != nullptr) {
    for (auto ed : *external_data) {
      external_data_vec.emplace_back(schema::CreateExternalDataDirect(fbb, ed->checkSum()->c_str(), ed->location()->c_str(), 0, ed->length()));
    }
  }
  uint8_t *data_src = reinterpret_cast<uint8_t *>(lite_tensor->data());
  std::vector<uint8_t> data_vec(data_src, data_src + lite_tensor->Size());
  auto tensor_offset = schema::CreateTensorDirect(fbb, schema_tensor->nodeType(), lite_tensor->data_type(), &dim_vec,
                                                  schema_tensor->format(), 0, 0, &data_vec, &quant_vec,
                                                  &quant_clusters, schema_tensor->name()->c_str(),
                                                  schema_tensor->enableHuffmanCode(),
                                                  mindspore::schema::WeightQuantCompressType_NONE, &external_data_vec);
  fbb.Finish(tensor_offset);

  auto buf = fbb.GetBufferPointer();
  if (buf == nullptr) {
    MS_LOG(ERROR) << "GetBufferPointer return nullptr";
    fbb.Clear();
    return nullptr;
  }
  size_t byte_num = fbb.GetSize();
  auto tensor_buf = reinterpret_cast<char *>(malloc(byte_num));
  if (tensor_buf == nullptr) {
    MS_LOG(ERROR) << "malloc primitive_buf_ failed";
    fbb.Clear();
    return nullptr;
  }
  memcpy(tensor_buf, buf, fbb.GetSize());
  auto tensor = flatbuffers::GetRoot<schema::Tensor>(tensor_buf);
  fbb.Clear();
  if (tensor != nullptr) {
    // use to free tensor_buf
    auto iter = dequant_schema_tensors_buffer_map_.find(const_cast<schema::Tensor *>(tensor));
    if (iter != dequant_schema_tensors_buffer_map_.end()) {
      MS_LOG(ERROR) << "schema tensor is duplicated.";
      return nullptr;
    }
    dequant_schema_tensors_buffer_map_[const_cast<schema::Tensor *>(tensor)] = tensor_buf;
  }
  return const_cast<schema::Tensor *>(tensor);
}

int NNRTDelegate::DequantNodeInputs(LiteGraph::Node *node) {
  auto in_size = node->input_indices_.size();
  int ret = RET_OK;
  for (size_t i = 0; i < in_size; i++) {
    auto tensor_index = node->input_indices_[i];
    auto *src_tensor = lite_graph_->all_tensors_[tensor_index];
    auto input = dequant_src_tensors_->at(tensor_index);
    if (!input->IsConst() || !(src_tensor->dataType() == kNumberTypeInt8 ||
        src_tensor->dataType() == kNumberTypeInt16 || src_tensor->dataType() == kNumberTypeInt32)) {
      continue;
    }
    auto dst_tensor = TensorToSchemaTensor(input, src_tensor);
    if (dst_tensor != nullptr) {
      dequant_schema_tensors_.emplace(tensor_index, dst_tensor);
      replaced_schema_tensors_.emplace_back(src_tensor);
    } else {
      MS_LOG(ERROR) << "create dequant schema tensor failed, node: " << node->name_ << ", tensor_index: "
                    << tensor_index;
      ret = RET_ERROR;
      break;
    }
  }
  return ret;
}

Status NNRTDelegate::DequantLiteGraph(LiteGraph *lite_graph) {
  for (auto node_index : lite_graph->sub_graphs_[0]->node_indices_) {
    auto node = lite_graph->all_nodes_[node_index];

    if (node->quant_type_ != static_cast<int>(schema::QuantType_QUANT_WEIGHT)) {
      continue;
    }
    auto ret = DequantNodeInputs(node);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Dequant node failed: " << ret << ", node_name: " << node->name_;
      for (auto iter : dequant_schema_tensors_) {
        delete iter.second;
        iter.second = nullptr;
      }
      return kLiteNotSupport;
    }
    node->quant_type_ = schema::QuantType_QUANT_NONE;
  }
  for (auto iter : dequant_schema_tensors_) {
    lite_graph_->all_tensors_[iter.first] = iter.second;
  }
  return kSuccess;
}

void NNRTDelegate::ShallowCopyLiteGraph(const lite::LiteGraph &lite_graph) {
  std::vector<LiteGraph::Node *> node_list;
  node_list.reserve(lite_graph.all_nodes_.size());
  // copy node
  for (auto node : lite_graph.all_nodes_) {
    auto new_node = new(std::nothrow) LiteGraph::Node;
    if (new_node == nullptr) {
      MS_LOG(ERROR) << " new LiteGraph::Node failed.";
      return;
    }
    new_node->name_ = node->name_;
    new_node->op_type_ = node->op_type_;
    new_node->node_type_ = node->node_type_;
    new_node->primitive_ = node->primitive_;
    new_node->base_operator_ = node->base_operator_;
    new_node->input_indices_ = node->input_indices_;
    new_node->output_indices_ = node->output_indices_;
    new_node->quant_type_ = node->quant_type_;
    new_node->device_type_ = node->device_type_;
    node_list.emplace_back(new_node);
  }
  // copy subgraph
  std::vector<LiteGraph::SubGraph *> subgraph_list;
  for (auto subgraph : lite_graph.sub_graphs_) {
    auto new_subgraph = new(std::nothrow) LiteGraph::SubGraph;
    if (new_subgraph == nullptr) {
      MS_LOG(ERROR) << "new LiteGraph::Subgraph failed.";
      return;
    }
    new_subgraph->name_ = subgraph->name_;
    new_subgraph->input_indices_ = subgraph->input_indices_;
    new_subgraph->output_indices_ = subgraph->output_indices_;
    new_subgraph->node_indices_ = subgraph->node_indices_;
    subgraph_list.emplace_back(new_subgraph);
  }
  for (auto tensor : lite_graph.all_tensors_) {
    Status ret = lite::CheckTensorSupported(static_cast<const schema::Tensor *>(tensor));
    if (ret == kLiteError) {
      MS_LOG(ERROR) << "tensor supported check failed.";
      return;
    }
  }

  lite_graph_ = new(std::nothrow) lite::LiteGraph();
  if (lite_graph_ == nullptr) {
    MS_LOG(ERROR) << "new LiteGraph failed.";
    return;
  }

  lite_graph_->name_ = lite_graph.name_;
  lite_graph_->version_ = lite_graph.version_;
  lite_graph_->input_indices_ = lite_graph.input_indices_;
  lite_graph_->output_indices_ = lite_graph.output_indices_;
  lite_graph_->all_tensors_ = lite_graph.all_tensors_;
  lite_graph_->all_nodes_ = node_list;
  lite_graph_->sub_graphs_ = subgraph_list;
  MS_LOG(INFO) << "ShallowCopyLiteGraph success.";
}

void NNRTDelegate::FreeLiteGraph(lite::LiteGraph **liteGraph) {
  if (liteGraph != nullptr && *liteGraph != nullptr) {
    MS_LOG(INFO) << "start to free LiteGraph.";
    auto graph = *liteGraph;
    graph->name_.clear();
    graph->input_indices_.clear();
    graph->output_indices_.clear();
    MS_LOG(INFO) << "Destroying  nodes.";
    // node
    for (size_t idx = 0; idx < graph->all_nodes_.size(); idx++) {
      if (graph->all_nodes_[idx] != nullptr) {
        delete graph->all_nodes_[idx];
        graph->all_nodes_[idx] = nullptr;
      }
    }
    MS_LOG(INFO) << "Destroying  subgraphs.";
    // subgraph
    for (size_t idx = 0; idx < graph->sub_graphs_.size(); idx++) {
      if (graph->sub_graphs_[idx] != nullptr) {
        delete graph->sub_graphs_[idx];
        graph->sub_graphs_[idx] = nullptr;
      }
    }
    // graph
    delete graph;
    *liteGraph = nullptr;
  } else {
    MS_LOG(WARNING) << "nnrt_lite_graph is nullptr, no need to free.";
  }
}

NNRTDelegate::~NNRTDelegate() {
  for (size_t i = 0; i < nn_executor_list_.size(); i++) {
    if (nn_executor_list_[i] != nullptr) {
      MS_LOG(INFO) << "start NNExecutor Destroy.";
      OH_NNExecutor_Destroy(&(nn_executor_list_[i]));
      MS_LOG(INFO) << "Destroy NNExecutor Finish.";
    }
  }
  if (lite_graph_ != nullptr) {
    MS_LOG(ERROR) << "Delete NNRTDelegate.";
  }
  for (auto iter : dequant_schema_tensors_buffer_map_) {
    if (iter.second != nullptr) {
      free(iter.second);
      iter.second = nullptr;
    }
  }
  dequant_schema_tensors_buffer_map_.clear();
}
}  // namespace lite
}  // namespace mindspore
