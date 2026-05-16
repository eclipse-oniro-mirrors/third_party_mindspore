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
#include <set>
#include <numeric>
#include "nnrt_delegate.h"
#include "checker/primitive_check.h"
#include "src/common/log_adapter.h"
#include "nnrt_model_kernel.h"
#include "nnrt_allocator.h"
#include "schema/model_generated.h"
#include "schema/ops_generated.h"
#include "flatbuffers/flatbuffers.h"
#include "litert/tensor_category.h"
#include "src/litert/delegate/nnrt/nnrt_wrapper.h"
#include "src/common/utils.h"

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

  if (enable_hete_ == false) {
    MS_LOG(DEBUG) << "enable_hete_ is false, heterogeneous inference disabled";
  }

  bool is_kirin_online = IsKirinNPUWithOnlineInference();

  if (is_kirin_online) {
    MS_LOG(DEBUG) << "Path: Kirin NPU with online inference";
    if (xpu_backend_set_ == false) {
      MS_LOG(ERROR) << "Config should use 'backend:CPU' for heterogeneous execution.";
      return kLiteError;
    }
    if (enable_hete_ && hete_execution_plan_ != nullptr && !(*hete_execution_plan_).empty()) {
      MS_LOG(DEBUG) << "Path: Heterogeneous inference ENABLED, hete_execution_plan_ size = "
                   << hete_execution_plan_->size();
      if (!CheckXpuConfig()) {
        MS_LOG(ERROR) << "XPU backend config invalid, Build cannot proceed.";
        return kLiteError;
      }
      auto ret = BuildNormalModel(model);
      return ret;
    }
    return BuildKirinNPUModel(model);
  }

  bool is_kirin_offline = IsKirinNPUWithOfflineInference();

  if (is_kirin_offline) {
    MS_LOG(DEBUG) << "Path: Kirin NPU with offline inference, calling BuildOfflineModel()";
    build_offline_ = true;
    MS_LOG(DEBUG) << "========== NNRTDelegate::Build EXIT (BuildOfflineModel) ==========";
    return BuildOfflineModel(model);
  }

  MS_LOG(DEBUG) << "Path: No specific Kirin NPU type detected, returning kSuccess (no kernel built!)";
#else
  MS_LOG(INFO) << "SUPPORT_NNRT_METAGRAPH NOT defined, returning kSuccess (no kernel built!)";
#endif

  MS_LOG(DEBUG) << "========== NNRTDelegate::Build EXIT (kSuccess) ==========";
  return kSuccess;
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
bool NNRTDelegate::CheckXpuConfig() {
  MS_LOG(DEBUG) << "NNRTDelegate::Build - xpu_backend_set_=" << xpu_backend_set_ << ", enable_hete_=" << enable_hete_;
  std::set<std::string> model_op_names;
  if (lite_graph_ != nullptr && !lite_graph_->all_nodes_.empty()) {
    for (auto *node : lite_graph_->all_nodes_) {
      if (node != nullptr && !node->name_.empty()) {
        model_op_names.insert(node->name_);
      }
    }
  }
  for (const auto &[op_name, backend] : *original_hete_execution_plan_) {
    MS_LOG(DEBUG) << "Checking operator '" << op_name << "' from config";
    if (model_op_names.count(op_name) > 0) {
      MS_LOG(DEBUG) << "Operator '" << op_name << "' validated";
      continue;
    }

    MS_LOG(ERROR) << "Operator '" << op_name << "' not found in model (case-sensitive match required).";
    std::string op_list_str;
    for (const auto &name : model_op_names) {
      if (!op_list_str.empty()) {
        op_list_str += ", ";
      }
      op_list_str += name;
    }
    MS_LOG(ERROR) << "Model operators: " << op_list_str;
    MS_LOG(ERROR) << "Please check:";
    MS_LOG(ERROR) << "  1. Operator name spelling (case-sensitive)";
    MS_LOG(ERROR) << "  2. Config file matches the current model version";
    return false;
  }
  MS_LOG(INFO) << "All " << original_hete_execution_plan_->size() << " operators from config validated successfully";
  return true;
}

bool NNRTDelegate::CheckNPUPrefix(const std::string prefix_name) const {
  const std::string kirin_npu_name_prefix = prefix_name;
  auto device_id = nnrt_device_info_.device_id_;
  const char *device_name;

  auto& nnrtWrapper = mindspore::NNRTWrapper::GetInstance();
  auto ret = nnrtWrapper.NNDeviceGetName(device_id, &device_name);
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

bool NNRTDelegate::IsKirinNPUWithOnlineInference() const { return CheckNPUPrefix("NPU_"); }

bool NNRTDelegate::IsKirinNPUWithOfflineInference() const { return CheckNPUPrefix("HIAI_F"); }

Status NNRTDelegate::BuildKirinNPUModel(DelegateModel<schema::Primitive> *model) {
  uint64_t start_build_nnrt = mindspore::lite::GetTimeUs();

  OH_NNModel *nn_model = nullptr;
  auto& nnrtWrapper = mindspore::NNRTWrapper::GetInstance();

  MS_LOG(INFO) << "loaded nnrt library at OH_NNModel_Construct";
  nn_model = nnrtWrapper.Construct();
  
  if (nn_model == nullptr) {
    MS_LOG(ERROR) << "Create NNModel failed, result is nullptr";
    return kLiteNullptr;
  }

  size_t extension_size = nnrt_device_info_.extensions_.size();
  std::vector<OH_NN_Extension> extensions;
  MS_LOG_DEBUG << "set extensions, item number: " << extension_size;
  const size_t kExtensionNameMax = 128;  // This is a length limitation in NNRT API.
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

  auto ret = nnrtWrapper.BuildFromLiteGraph(nn_model, lite_graph_, extensions.data(), extensions.size());

  if (ret != OH_NN_SUCCESS) {
    MS_LOG(ERROR) << "Build NNModel failed, ret: " << ret;
    nnrtWrapper.Destroy(&nn_model);
    return kLiteError;
  }

  auto ret2 = CreateFullModelKernel(model, nn_model);
  if (ret2 != kSuccess) {
    MS_LOG(ERROR) << "Create full model kernel failed, ret: " << ret2;
    
    return kLiteError;
  }
  uint64_t build_nnrt_time = mindspore::lite::GetTimeUs() - start_build_nnrt;
  MS_LOG(DEBUG) << "The NNRT delegate online model build time is: " << build_nnrt_time << "us";
  
  return kSuccess;
}

namespace {
constexpr int32_t kNum2 = 2;
}

Status NNRTDelegate::BuildOfflineModel(DelegateModel<schema::Primitive> *model) {
  uint64_t start_build_offline_nnrt = mindspore::lite::GetTimeUs();
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
  auto& nnrtWrapper = mindspore::NNRTWrapper::GetInstance();
  OH_NNCompilation *nn_compilation = nnrtWrapper.NNCompilationConstructWithOfflineModelBuffer(model_buf, model_size);
  if (nn_compilation == nullptr) {
    MS_LOG(ERROR) << "Construct Offline NNCompilation failed";
    return kLiteError;
  }
  MS_LOG(DEBUG) << "NNRTDelegate creates NNCompilation success.";

  auto ret_code = InitNNCompilation(nn_compilation);
  if (ret_code != kSuccess) {
    MS_LOG(ERROR) << "Init NNCompilation failed";
    nnrtWrapper.NNCompilationDestroy(&nn_compilation);
    return kLiteError;
  }
  MS_LOG(DEBUG) << "HiAI F InitNNCompilation success";

  OH_NNExecutor *nn_executor = nullptr;

  nn_executor = nnrtWrapper.NNExecutorConstruct(nn_compilation);
  if (nn_executor == nullptr) {
    MS_LOG(ERROR) << "Construct NNExecutor failed, ret: " << ret_code;
    nnrtWrapper.NNCompilationDestroy(&nn_compilation);
    return kLiteError;
  }

  nnrtWrapper.NNCompilationDestroy(&nn_compilation);

  auto nnrt_model_kernel =
    new (std::nothrow) NNRTModelKernel(nn_executor, nnrt_device_info_, model->inputs(), model->outputs());
  if (nnrt_model_kernel == nullptr) {
    nnrtWrapper.NNExecutorDestroy(&nn_executor);
    MS_LOG(ERROR) << "new NNRTModelKernel failed";
    return kLiteError;
  }
  nn_executor_list_.push_back(nn_executor);

  (void)model->Replace(model->BeginKernelIterator(), model->EndKernelIterator(), nnrt_model_kernel);
  uint64_t build_offline_nnrt_time = mindspore::lite::GetTimeUs() - start_build_offline_nnrt;
  MS_LOG(DEBUG) << "The NNRT delegate offline model build time is: " << build_offline_nnrt_time << "us";
  
  return kSuccess;
}

Status NNRTDelegate::CreateFullModelKernel(DelegateModel<schema::Primitive> *model, OH_NNModel *nn_model) {
  auto& nnrtWrapper = mindspore::NNRTWrapper::GetInstance();
  MS_LOG(DEBUG) << "NNRTDelegate starts and creates NNCompilation";
  OH_NNCompilation *nn_compilation = nnrtWrapper.NNCompilationConstruct(nn_model);
  if (nn_compilation == nullptr) {
    MS_LOG(ERROR) << "Construct NNCompilation failed";
    nnrtWrapper.Destroy(&nn_model);
    return kLiteError;
  }
  MS_LOG(DEBUG) << "NNRTDelegate creates NNCompilation successfully";

  auto ret_code = InitNNCompilation(nn_compilation);
  if (ret_code != kSuccess) {
    MS_LOG(ERROR) << "Init NNCompilation failed";
    nnrtWrapper.Destroy(&nn_model);  
    nnrtWrapper.NNCompilationDestroy(&nn_compilation);
    return kLiteError;
  }
  nnrtWrapper.Destroy(&nn_model);
  MS_LOG(DEBUG) << "NNRTDelegate initialized NNCompilation successfully";
  OH_NNExecutor *nn_executor = nullptr;
  nn_executor = nnrtWrapper.NNExecutorConstruct(nn_compilation);
  if (nn_executor == nullptr) {
    MS_LOG(ERROR) << "Construct NNExecutor failed, ret: " << ret_code;
    nnrtWrapper.NNCompilationDestroy(&nn_compilation);
    return kLiteError;
  }
  MS_LOG(DEBUG) << "NNRTDelegate constructed NNExecutor successfully";
  nnrtWrapper.NNCompilationDestroy(&nn_compilation);

  auto nnrt_model_kernel =
    new (std::nothrow) NNRTModelKernel(nn_executor, nnrt_device_info_, model->inputs(), model->outputs());
  if (nnrt_model_kernel == nullptr) {
    nnrtWrapper.NNExecutorDestroy(&nn_executor);
    MS_LOG(ERROR) << "new NNRTModelKernel failed";
    return kLiteError;
  }
  MS_LOG(DEBUG) << "NNRTDelegate created NNRTModelKernel successfully";
  nn_executor_list_.push_back(nn_executor);

  model->Replace(model->BeginKernelIterator(), model->EndKernelIterator(), nnrt_model_kernel);
  MS_LOG(DEBUG) << "NNRTDelegate created FullModelKernel successfully";
  return kSuccess;
}
#endif

void NNRTDelegate::PrintNodeNameFormat()
{
  MS_LOG(DEBUG) << "Total node name " << lite_graph_->all_nodes_.size();
  for (size_t i = 0; i < lite_graph_->all_nodes_.size(); i++) {
    auto &node = lite_graph_->all_nodes_[i];
    if (node == nullptr) {
      MS_LOG(ERROR) << "Total node is null at index [" << i << "].";
      continue;
    }
    MS_LOG(DEBUG) << "Current node name " << node->name_ << ", node type "
                 << GetPrimitiveTypeName(node->primitive_, schema_version_);
    for (auto j : node->input_indices_) {
      MS_LOG(DEBUG) << "Input tensor format is " << EnumNameFormat(lite_graph_->all_tensors_[j]->format());
    }
    for (auto j : node->output_indices_) {
      MS_LOG(DEBUG) << "Output tensor format is " << EnumNameFormat(lite_graph_->all_tensors_[j]->format());
    }
  }
}

void NNRTDelegate::ApplyCPUOp(std::vector<bool> &op_supports)
{
  uint32_t op_count = 0;
  op_supports.resize(lite_graph_->all_nodes_.size());
  size_t index = 0;
  bool hete_op = true;
  if (!enable_hete_) {
    MS_LOG(WARNING) << "Disable heterogeneous online inference.";
  }
  if (hete_execution_plan_ == nullptr || (*hete_execution_plan_).empty()) {
    MS_LOG(ERROR) << "Heterogeneous OP config is invalid, please check config file. All OP will run on NPU by defalult.";
    hete_op = false;
  }
  MS_LOG(DEBUG) << "Current tatol op num " << op_supports.size() << " .";
  for (size_t i = 0; i < lite_graph_->all_nodes_.size(); i++) {
    auto &node = lite_graph_->all_nodes_[i];
    if (node == nullptr) {
      MS_LOG(ERROR) << "Current node is null at index [" << i << "].";
      continue;
    }
    // Use find() instead of operator[] to avoid auto-insertion
    bool is_cpu_op = false;
    if (hete_op) {
      auto it = hete_execution_plan_->find(node->name_);
      if (it != hete_execution_plan_->end() && it->second == kBackendTypeCPU) {
        is_cpu_op = true;
      }
    }
    if (is_cpu_op) {
      MS_LOG(INFO) << "Current node " << node->name_ << ", node type "
                   << GetPrimitiveTypeName(node->primitive_, schema_version_);
      op_supports[i] = false;
      index++;
      continue;
    }
    op_supports[i] = true;
    op_count++;
  }
  MS_LOG(DEBUG) << "Total OP num " << op_supports.size() << ", " << op_count << " OP will run on NPU, " << index
               << " OP will run CPU.";
}

Status NNRTDelegate::BuildNormalModel(DelegateModel<schema::Primitive> *model) {
  MS_LOG(INFO) << "========== BuildNormalModel ENTER ==========";
  MS_LOG(INFO) << "lite_graph_=" << (lite_graph_ ? "valid" : "NULL");

  if ((lite_graph_ == nullptr) || (lite_graph_->sub_graphs_.size() > 1)) {
    MS_LOG(WARNING)
      << "LiteGraph is nullptr or contains more than one subgraph (size="
      << (lite_graph_ ? lite_graph_->sub_graphs_.size() : 0)
      << "). NNRT does not support control-flow model yet, fallback to CPU";
    MS_LOG(INFO) << "========== BuildNormalModel EXIT (kSuccess - no build) ==========";
    return kSuccess;
  }

  auto& nnrtWrapper = mindspore::NNRTWrapper::GetInstance();
  OH_NNModel *full_model = CreateFullNNModel();
  if (full_model == nullptr) {
    MS_LOG(WARNING) << "Build full NNModel failed, fallback to CPU";
    MS_LOG(INFO) << "========== BuildNormalModel EXIT (kSuccess - create failed) ==========";
    return kSuccess;
  }

  PrintNodeNameFormat();

  std::vector<bool> op_supports = QueryOpSupports(full_model);

  ApplyCPUOp(op_supports);

  if (op_supports.empty()) {
    MS_LOG(WARNING) << "Query no op supports for full model, fallback to CPU";
    nnrtWrapper.Destroy(&full_model);
    MS_LOG(INFO) << "========== BuildNormalModel EXIT (kSuccess - no supports) ==========";
    return kSuccess;
  }

  auto nnrt_subgraph_ranges = GetNNRTSubgraphRanges(model, op_supports);

  std::vector<LiteGraph *> sub_lite_graphs;
  auto ret = CreateLiteGraphForNNRTSubgraph(nnrt_subgraph_ranges, &sub_lite_graphs);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Create NNRT sub LiteGraph failed, ret=" << ret;
    nnrtWrapper.Destroy(&full_model);
    for (size_t i = 0; i < sub_lite_graphs.size(); i++) {
      FreeLiteGraph(&sub_lite_graphs[i]);
    }
    MS_LOG(WARNING) << "Create NNRT sub LiteGraph failed, fallback to CPU";
    return kSuccess;
  }

  std::vector<NNRTModelKernel *> nnrt_subgraph_kernels;
  ret = CreateNNRTSubgraphKernels(model, sub_lite_graphs, nnrt_subgraph_ranges, &nnrt_subgraph_kernels);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Create NNRT subgraph kernels failed, ret=" << ret;
    nnrtWrapper.Destroy(&full_model);
    for (size_t i = 0; i < sub_lite_graphs.size(); i++) {
      FreeLiteGraph(&sub_lite_graphs[i]);
    }
    MS_LOG(WARNING) << "Create NNRT subgraph kernel failed, fallback to CPU";
    return kSuccess;
  }

  ReplaceNNRTKernelsInDelegateModel(model, nnrt_subgraph_ranges, nnrt_subgraph_kernels);
  nnrtWrapper.Destroy(&full_model);
  for (size_t i = 0; i < sub_lite_graphs.size(); i++) {
    // FreeLiteGraph(&sub_lite_graphs[i]);
  }
  return kSuccess;
}

OH_NNModel *NNRTDelegate::CreateFullNNModel() {
  MS_LOG(INFO) << "NNRTDelegate starts creating the Full NNRT Model";
  if (lite_graph_ == nullptr) {
    MS_LOG(ERROR) << "Lite graph is null";
    return nullptr;
  }

  if (lite_graph_->sub_graphs_.empty()) {
    MS_LOG(ERROR) << "Lite graph must have at lease one subgraph";
    return nullptr;
  }

  OH_NNModel *nn_model = nullptr;
  auto& nnrtWrapper = mindspore::NNRTWrapper::GetInstance();
  MS_LOG(INFO) << "loaded nnrt library at OH_NNModel_Construct";
  nn_model = nnrtWrapper.Construct();
  
  if (nn_model == nullptr) {
    MS_LOG(ERROR) << "Create NNModel failed, result is nullptr";
    return nullptr;
  }

  MS_LOG(INFO) << "NNRTDelegate starts building the NNRT Model from LiteGraph";

  auto ret = nnrtWrapper.BuildFromLiteGraph(nn_model, lite_graph_, nullptr, 0);
  
  MS_LOG(INFO) << "NNRTDelegate has successfully built the NNRT Model from LiteGraph";
  
  if (ret != OH_NN_SUCCESS) {
    MS_LOG(ERROR) << "Build NNModel failed, ret: " << ret;
    nnrtWrapper.Destroy(&nn_model);
    return nullptr;
  }
  
  return nn_model;
}

std::vector<bool> NNRTDelegate::QueryOpSupports(OH_NNModel *nn_model) {
  const bool *is_supported = nullptr;  // Note: this memory is owned by nn_model, don't free alone.
  uint32_t op_count = 0;

  auto ret = OH_NN_FAILED;
  auto& nnrtWrapper = mindspore::NNRTWrapper::GetInstance();
  MS_LOG(INFO) << "loaded nnrt library at OH_NNModel_GetAvailableOperations";
  ret = nnrtWrapper.GetAvailableOperations(nn_model, nnrt_device_info_.device_id_, &is_supported, &op_count);
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
Status NNRTDelegate::CreateLiteGraphForNNRTSubgraph(const std::vector<NNRTOpRange> &nnrt_op_ranges,
                                                    std::vector<LiteGraph *> *sub_lite_graphs) {
  MS_LOG(INFO) << "Start creating LiteGraph for NNRT subgraph";
  for (const auto &op_range : nnrt_op_ranges) {
    MS_LOG(INFO) << "Process op range: [" << op_range.begin_index_ << ", " << op_range.end_index_ << ")";
    LiteGraph *sub_lite_graph = new (std::nothrow) LiteGraph;
    if (sub_lite_graph == nullptr) {
      MS_LOG(ERROR) << "Allocate LiteGraph failed";
      return kLiteError;
    }
    sub_lite_graph->name_ = lite_graph_->name_;
    sub_lite_graph->version_ = lite_graph_->version_;

    auto sub_graph = new (std::nothrow) LiteGraph::SubGraph;
    if (sub_graph == nullptr) {
      MS_LOG(ERROR) << "Allocate SubGraph failed";
      return kLiteError;
    }
    sub_graph->name_ = lite_graph_->name_;
    sub_lite_graph->sub_graphs_.push_back(sub_graph);

    int new_node_index = 0;
    std::map<uint32_t, schema::Tensor *> in_tensor_index_map;
    std::map<uint32_t, schema::Tensor *> out_tensor_index_map;
    for (size_t index = op_range.begin_index_; index < op_range.end_index_; index++) {
      LiteGraph::Node *node = new (std::nothrow) LiteGraph::Node;
      if (node == nullptr) {
        MS_LOG(ERROR) << "Allocate Node failed";
        return kLiteError;
      }
      *node = *lite_graph_->all_nodes_[index];
      // Clear shared_ptr to prevent double-free (primitive_ is safe to keep)
      node->base_operator_ = nullptr;
      sub_lite_graph->all_nodes_.push_back(node);
      sub_graph->node_indices_.push_back(new_node_index++);

      for (auto i : node->input_indices_) {
        in_tensor_index_map.emplace(i, lite_graph_->all_tensors_[i]);
      }
      for (auto i : node->output_indices_) {
        out_tensor_index_map.emplace(i, lite_graph_->all_tensors_[i]);
      }
    }

    std::set<schema::Tensor *> tensors;
    for (auto iter : in_tensor_index_map) {
      tensors.emplace(iter.second);
    }
    for (auto iter : out_tensor_index_map) {
      tensors.emplace(iter.second);
    }

    uint32_t new_index = 0;
    std::map<schema::Tensor *, uint32_t> new_tensor_maps;
    for (auto tensor : tensors) {
      new_tensor_maps.emplace(tensor, new_index++);
    }

    sub_lite_graph->all_tensors_ = std::vector<schema::Tensor *>(tensors.begin(), tensors.end());

    MS_LOG(INFO) << "Set input/output indices of each node...";
    for (auto node : sub_lite_graph->all_nodes_) {
      for (auto &index : node->input_indices_) {
        index = new_tensor_maps.at(in_tensor_index_map.at(index));
      }
      for (auto &index : node->output_indices_) {
        index = new_tensor_maps.at(out_tensor_index_map.at(index));
      }
    }

    // deal with subgraph's input/output indices
    sub_graph->tensor_indices_ = std::vector<uint32_t>(tensors.size());
    std::iota(sub_graph->tensor_indices_.begin(), sub_graph->tensor_indices_.end(), 0U);

    for (auto iter : in_tensor_index_map) {
      auto new_tensor_index = new_tensor_maps[iter.second];
      MS_LOG(DEBUG) << "handle input: old: " << iter.first << ", new: " << new_tensor_index << std::endl;
      if (IsConstTensor(*iter.second)) {
        MS_LOG(DEBUG) << "- tensor: " << new_tensor_index << " is const." << std::endl;
        continue;
      }

      bool is_subgraph_input = true;
      for (auto node : sub_lite_graph->all_nodes_) {
        if (std::find(node->output_indices_.begin(), node->output_indices_.end(), new_tensor_index) !=
            node->output_indices_.end()) {
          is_subgraph_input = false;
          break;
        }
      }
      if (is_subgraph_input) {
        sub_graph->input_indices_.push_back(new_tensor_index);
      }
    }

    for (auto iter : out_tensor_index_map) {
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
            node->input_indices_.end()) {
          is_subgraph_output = true;
          break;
        }
      }
      bool is_graph_output = (std::find(lite_graph_->output_indices_.begin(), lite_graph_->output_indices_.end(),
                                        iter.first) != lite_graph_->output_indices_.end());
      if (is_subgraph_output || is_graph_output) {
        sub_graph->output_indices_.push_back(new_tensor_index);
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
  uint32_t node_index;    // the index of node which the tensor belongs to.
  uint32_t tensor_index;  // the index of node in/out tensors which the tensor is located at.
};

Status NNRTDelegate::InitNNCompilation(OH_NNCompilation *nn_compilation) const {
  auto& nnrtWrapper = mindspore::NNRTWrapper::GetInstance();
  auto ret_code = nnrtWrapper.NNCompilationSetDevice(nn_compilation, nnrt_device_info_.device_id_);
  if (ret_code != OH_NN_SUCCESS) {
    MS_LOG(ERROR) << "NNCompilation set device id failed, ret: " << ret_code;
    return kLiteError;
  }
  ret_code =
    nnrtWrapper.NNCompilationSetPerformanceMode(nn_compilation, (OH_NN_PerformanceMode)(nnrt_device_info_.performance_mode_));
  if ((ret_code != OH_NN_SUCCESS) && (ret_code != OH_NN_OPERATION_FORBIDDEN)) {
    MS_LOG(ERROR) << "NNCompilation set performance mode failed, ret: " << ret_code;
    return kLiteError;
  }

  ret_code = nnrtWrapper.NNCompilationSetPriority(nn_compilation, (OH_NN_Priority)(nnrt_device_info_.priority_));
  if ((ret_code != OH_NN_SUCCESS) && (ret_code != OH_NN_OPERATION_FORBIDDEN)) {
    MS_LOG(ERROR) << "NNCompilation set priority failed, ret: " << ret_code;
    return kLiteError;
  }

  ret_code = nnrtWrapper.NNCompilationEnableFloat16(nn_compilation, nnrt_device_info_.enable_fp16_);
  if ((ret_code != OH_NN_SUCCESS) && (ret_code != OH_NN_OPERATION_FORBIDDEN)) {
    MS_LOG(ERROR) << "NNCompilation enable fp16 failed, ret: " << ret_code;
    return kLiteError;
  }

  if (!extension_options_.cache_path_.empty()) {  // Set cache path if user indeed set it.
    
    ret_code = nnrtWrapper.NNCompilationSetCache(nn_compilation, extension_options_.cache_path_.c_str(),
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
    if (extension_options_.dynamic_dims.size() > 0) {
      ret_code = mindspore::lite::HMS_HiAIOptions_SetAsyncModeEnable(nn_compilation, false);
      if (ret_code != OH_NN_SUCCESS) {
        MS_LOG(ERROR) << "HMS_HiAIOptions_SetAsyncModeEnable failed, ret: " << ret_code;
        return kLiteError;
      }
    }
  } else {
    MS_LOG(WARNING) << "hiai_foundation is " << (hiai_handle_ == nullptr ? "" : "not")
                    << " nullptr, current device name prefix is not HIAI_F.";
  }
#endif

  ret_code = nnrtWrapper.NNCompilationBuild(nn_compilation);
  if (ret_code != OH_NN_SUCCESS) {
    MS_LOG(ERROR) << "Build NNCompilation failed, ret: " << ret_code;
    return kLiteError;
  }
  
  return kSuccess;
}

Status NNRTDelegate::CreateNNRTSubgraphKernels(DelegateModel<schema::Primitive> *model,
                                               const std::vector<LiteGraph *> &sub_lite_graphs,
                                               const std::vector<NNRTOpRange> &nnrt_subgraph_ranges,
                                               std::vector<NNRTModelKernel *> *nnrt_subgraph_kernels) {
  MS_LOG(INFO) << "========== CreateNNRTSubgraphKernels ENTER ==========";
  MS_LOG(INFO) << "sub_lite_graphs.size()=" << sub_lite_graphs.size();

  auto& nnrtWrapper = mindspore::NNRTWrapper::GetInstance();
  for (size_t i = 0; i < sub_lite_graphs.size(); i++) {
    MS_LOG(INFO) << "========== Processing subgraph " << i << " ==========";
    auto sub_lite_graph = sub_lite_graphs[i];
    MS_LOG(INFO) << "Subgraph " << i << ": ptr=" << sub_lite_graph
                 << ", all_nodes_.size()=" << (sub_lite_graph ? sub_lite_graph->all_nodes_.size() : 0)
                 << ", all_tensors_.size()=" << (sub_lite_graph ? sub_lite_graph->all_tensors_.size() : 0);

    OH_NNModel *nn_model = nullptr;
    nn_model = nnrtWrapper.Construct();

    MS_LOG(INFO) << "Subgraph " << i << ": BEFORE BuildFromLiteGraph - checking all subgraphs integrity";
    for (size_t j = 0; j < sub_lite_graphs.size(); j++) {
      if (sub_lite_graphs[j] != nullptr) {
        MS_LOG(INFO) << "  Subgraph " << j << ": nodes=" << sub_lite_graphs[j]->all_nodes_.size()
                     << ", tensors=" << sub_lite_graphs[j]->all_tensors_.size()
                     << ", sub_graphs=" << sub_lite_graphs[j]->sub_graphs_.size();
      }
    }

    auto ret = nnrtWrapper.BuildFromLiteGraph(nn_model, sub_lite_graph, nullptr, 0);

    MS_LOG(INFO) << "Subgraph " << i << ": AFTER BuildFromLiteGraph - checking all subgraphs integrity";
    for (size_t j = 0; j < sub_lite_graphs.size(); j++) {
      if (sub_lite_graphs[j] != nullptr) {
        MS_LOG(INFO) << "  Subgraph " << j << ": nodes=" << sub_lite_graphs[j]->all_nodes_.size()
                     << ", tensors=" << sub_lite_graphs[j]->all_tensors_.size()
                     << ", sub_graphs=" << sub_lite_graphs[j]->sub_graphs_.size();
        if (sub_lite_graphs[j]->sub_graphs_.size() > 1000) {
          MS_LOG(ERROR) << "  CORRUPTION DETECTED! Subgraph " << j << " has sub_graphs_.size()="
                        << sub_lite_graphs[j]->sub_graphs_.size();
        }
      }
    }

    if (ret != OH_NN_SUCCESS) {
      MS_LOG(ERROR) << "Subgraph " << i << ": Build NNModel failed, ret: " << ret;
      nnrtWrapper.Destroy(&nn_model);
      return kLiteError;
    }

    OH_NNCompilation *nn_compilation = nnrtWrapper.NNCompilationConstruct(nn_model);
    if (nn_compilation == nullptr) {
      MS_LOG(ERROR) << "Subgraph " << i << ": Construct NNCompilation failed";
      nnrtWrapper.Destroy(&nn_model);
      return kLiteError;
    }

    auto ret_code = InitNNCompilation(nn_compilation);
    if (ret_code != kSuccess) {
      MS_LOG(ERROR) << "Subgraph " << i << ": Init NNCompilation failed";
      nnrtWrapper.NNCompilationDestroy(&nn_compilation);
      nnrtWrapper.Destroy(&nn_model);
      return kLiteError;
    }

    OH_NNExecutor *nn_executor = nnrtWrapper.NNExecutorConstruct(nn_compilation);
    if (nn_executor == nullptr) {
      MS_LOG(ERROR) << "Subgraph " << i << ": Construct NNExecutor failed";
      nnrtWrapper.NNCompilationDestroy(&nn_compilation);
      nnrtWrapper.Destroy(&nn_model);
      return kLiteError;
    }

    bool format_not_support = false;
    std::vector<MSTensor> in_tensors;

    if (sub_lite_graph->sub_graphs_.empty() || sub_lite_graph->sub_graphs_[0] == nullptr) {
      MS_LOG(ERROR) << "Subgraph " << i << ": sub_lite_graph->sub_graphs_ is empty or NULL!";
      nnrtWrapper.NNCompilationDestroy(&nn_compilation);
      nnrtWrapper.Destroy(&nn_model);
      return kLiteError;
    }

    MS_LOG(INFO) << "Subgraph " << i << ": sub_graphs_[0]->input_indices_.size() = "
                 << sub_lite_graph->sub_graphs_[0]->input_indices_.size();

    for (auto index : sub_lite_graph->sub_graphs_[0]->input_indices_) {
      TensorLocation location;
      for (auto node_index : sub_lite_graph->sub_graphs_[0]->node_indices_) {
        if (node_index >= sub_lite_graph->all_nodes_.size()) {
          MS_LOG(ERROR) << "Subgraph " << i << ": node_index " << node_index
                        << " >= all_nodes_.size() " << sub_lite_graph->all_nodes_.size();
          continue;
        }
        auto node = sub_lite_graph->all_nodes_[node_index];
        if (node == nullptr) {
          MS_LOG(ERROR) << "Subgraph " << i << ": node at index " << node_index << " is NULL!";
          continue;
        }
        auto iter = std::find(node->input_indices_.begin(), node->input_indices_.end(), index);
        if (iter != node->input_indices_.end()) {
          uint32_t tensor_index = iter - node->input_indices_.begin();
          location.node_index = node_index;
          location.tensor_index = tensor_index;
          MS_LOG(INFO) << "Subgraph " << i << ": Found input index " << index << " at node " << node->name_
                       << ", tensor_index=" << tensor_index;
          break;
        }
      }
      KernelIter kernel_iter = nnrt_subgraph_ranges[i].begin_iter_ + location.node_index;
      in_tensors.push_back((*kernel_iter)->inputs()[location.tensor_index]);
      if (in_tensors.back().format() != Format::NHWC) {
        MS_LOG(WARNING) << "Subgraph " << i << ": Kernel " << (*kernel_iter)->name()
                        << ", in_tensor format " << mindspore::FormatEnumToString(in_tensors.back().format());
        format_not_support = true;
        break;
      }
    }

    std::vector<MSTensor> out_tensors;
    MS_LOG(INFO) << "Subgraph " << i << ": sub_graphs_[0]->output_indices_.size() = "
                 << sub_lite_graph->sub_graphs_[0]->output_indices_.size();

    for (auto index : sub_lite_graph->sub_graphs_[0]->output_indices_) {
      TensorLocation location;
      for (auto node_index : sub_lite_graph->sub_graphs_[0]->node_indices_) {
        if (node_index >= sub_lite_graph->all_nodes_.size()) {
          MS_LOG(ERROR) << "Subgraph " << i << ": node_index " << node_index
                        << " >= all_nodes_.size() " << sub_lite_graph->all_nodes_.size();
          continue;
        }
        auto node = sub_lite_graph->all_nodes_[node_index];
        if (node == nullptr) {
          MS_LOG(ERROR) << "Subgraph " << i << ": node at index " << node_index << " is NULL!";
          continue;
        }
        auto iter = std::find(node->output_indices_.begin(), node->output_indices_.end(), index);
        if (iter != node->output_indices_.end()) {
          uint32_t tensor_index = iter - node->output_indices_.begin();
          location.node_index = node_index;
          location.tensor_index = tensor_index;
          MS_LOG(INFO) << "Subgraph " << i << ": Found output index " << index << " at node " << node->name_
                       << ", tensor_index=" << tensor_index;
          break;
        }
      }
      KernelIter kernel_iter = nnrt_subgraph_ranges[i].begin_iter_ + location.node_index;
      out_tensors.push_back((*kernel_iter)->outputs()[location.tensor_index]);
      if (out_tensors.back().format() != Format::NHWC) {
        MS_LOG(WARNING) << "Subgraph " << i << ": Kernel " << (*kernel_iter)->name()
                        << ", out_tensor format " << mindspore::FormatEnumToString(out_tensors.back().format());
        format_not_support = true;
        break;
      }
    }

    if (format_not_support) {
      MS_LOG(WARNING) << "Subgraph " << i << ": Tensor format not supported, skipping";
      nnrtWrapper.NNCompilationDestroy(&nn_compilation);
      nnrtWrapper.Destroy(&nn_model);
      nnrt_subgraph_kernels->push_back(nullptr);
      continue;
    }

    auto nnrt_model_kernel =
      new (std::nothrow) NNRTModelKernel(nn_executor, nnrt_device_info_, in_tensors, out_tensors);
    if (nnrt_model_kernel == nullptr) {
      MS_LOG(ERROR) << "Subgraph " << i << ": new NNRTModelKernel failed";
      return kLiteError;
    }

    nn_executor_list_.push_back(nn_executor);
    nnrtWrapper.NNCompilationDestroy(&nn_compilation);
    nnrtWrapper.Destroy(&nn_model);
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
    MS_LOG(INFO) << "Replace nnrt subgraph kernel in range: [" << (from - model->BeginKernelIterator()) << ", "
                 << (end - model->BeginKernelIterator()) << ")";
  }
}

Status NNRTDelegate::PrepareInputs(DelegateModel<schema::Primitive> *model, OH_NNExecutor *oh_nn_executor) {
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
      quant_param = new (std::nothrow) OH_NN_QuantParam;
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
    auto oprend = new (std::nothrow) OH_NN_Tensor;
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

    auto ret_code = OH_NN_FAILED;
    auto& nnrtWrapper = mindspore::NNRTWrapper::GetInstance();
    MS_LOG(INFO) << "loaded nnrt library at OH_NNExecutor_SetInput";
    ret_code = nnrtWrapper.ExecutorSetInput(oh_nn_executor, i, oprend, tensor.MutableData(), tensor.DataSize());
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
    {DataType::kNumberTypeBool, OH_NN_BOOL},       {DataType::kNumberTypeInt8, OH_NN_INT8},
    {DataType::kNumberTypeInt16, OH_NN_INT16},     {DataType::kNumberTypeInt32, OH_NN_INT32},
    {DataType::kNumberTypeInt64, OH_NN_INT64},     {DataType::kNumberTypeUInt8, OH_NN_UINT8},
    {DataType::kNumberTypeUInt16, OH_NN_UINT16},   {DataType::kNumberTypeUInt32, OH_NN_UINT32},
    {DataType::kNumberTypeUInt64, OH_NN_UINT64},   {DataType::kNumberTypeFloat16, OH_NN_FLOAT16},
    {DataType::kNumberTypeFloat32, OH_NN_FLOAT32}, {DataType::kNumberTypeFloat64, OH_NN_FLOAT64},
  };

  auto iter = kDataTypeMap.find(data_type);
  if (iter == kDataTypeMap.end()) {
    return OH_NN_UNKNOWN;
  }
  return iter->second;
}

Status NNRTDelegate::PrepareOutputs(DelegateModel<schema::Primitive> *model, OH_NNExecutor *oh_nn_executor) {
  auto output_tensors = model->outputs();
  auto& nnrtWrapper = mindspore::NNRTWrapper::GetInstance();
  for (size_t i = 0; i < output_tensors.size(); i++) {
    auto tensor = output_tensors[i];
    auto ret_code = nnrtWrapper.ExecutorSetOutput(oh_nn_executor, i, tensor.MutableData(), tensor.DataSize());
    
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
      external_data_vec.emplace_back(
        schema::CreateExternalDataDirect(fbb, ed->checkSum()->c_str(), ed->location()->c_str(), 0, ed->length()));
    }
  }
  uint8_t *data_src = reinterpret_cast<uint8_t *>(lite_tensor->data());
  std::vector<uint8_t> data_vec(data_src, data_src + lite_tensor->Size());
  auto tensor_offset = schema::CreateTensorDirect(fbb, schema_tensor->nodeType(), lite_tensor->data_type(), &dim_vec,
                                                  schema_tensor->format(), 0, 0, &data_vec, &quant_vec, &quant_clusters,
                                                  schema_tensor->name()->c_str(), schema_tensor->enableHuffmanCode(),
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
    if (!input->IsConst() ||
        !(src_tensor->dataType() == kNumberTypeInt8 || src_tensor->dataType() == kNumberTypeInt16 ||
          src_tensor->dataType() == kNumberTypeInt32)) {
      continue;
    }
    auto dst_tensor = TensorToSchemaTensor(input, src_tensor);
    if (dst_tensor != nullptr) {
      dequant_schema_tensors_.emplace(tensor_index, dst_tensor);
      replaced_schema_tensors_.emplace_back(src_tensor);
    } else {
      MS_LOG(ERROR) << "create dequant schema tensor failed, node: " << node->name_
                    << ", tensor_index: " << tensor_index;
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
    auto new_node = new (std::nothrow) LiteGraph::Node;
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
    auto new_subgraph = new (std::nothrow) LiteGraph::SubGraph;
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

  lite_graph_ = new (std::nothrow) lite::LiteGraph();
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
  MS_LOG(DEBUG) << "ShallowCopyLiteGraph success.";
}

void NNRTDelegate::FreeLiteGraph(lite::LiteGraph **liteGraph) {
  MS_LOG(INFO) << "========== FreeLiteGraph ENTER ==========";
  MS_LOG(INFO) << "liteGraph ptr=" << liteGraph;

  if (liteGraph == nullptr) {
    MS_LOG(WARNING) << "liteGraph is NULL, no need to free";
    MS_LOG(INFO) << "========== FreeLiteGraph EXIT (NULL) ==========";
    return;
  }

  MS_LOG(INFO) << "*liteGraph ptr=" << *liteGraph;

  if (*liteGraph == nullptr) {
    MS_LOG(WARNING) << "*liteGraph is NULL, no need to free";
    MS_LOG(INFO) << "========== FreeLiteGraph EXIT (*NULL) ==========";
    return;
  }

  MS_LOG(INFO) << "FreeLiteGraph: ptr=" << *liteGraph;
  auto graph = *liteGraph;

  // Detect memory corruption: check if size values are abnormally large
  bool is_corrupted = (graph->all_nodes_.size() > 1000000 ||
                       graph->all_tensors_.size() > 1000000 ||
                       graph->sub_graphs_.size() > 1000000);

  if (is_corrupted) {
    MS_LOG(ERROR) << "MEMORY CORRUPTION DETECTED! Graph has abnormally large sizes:";
    MS_LOG(ERROR) << "  all_nodes_.size()=" << graph->all_nodes_.size();
    MS_LOG(ERROR) << "  all_tensors_.size()=" << graph->all_tensors_.size();
    MS_LOG(ERROR) << "  sub_graphs_.size()=" << graph->sub_graphs_.size();
    MS_LOG(ERROR) << "  Deleting graph object WITHOUT accessing members to prevent SIGSEGV";
    delete graph;
    *liteGraph = nullptr;
    return;
  }

  // Subgraph detection: subgraphs have exactly 1 sub_graphs_ element
  // Note: Can't check all_nodes_.empty() due to memory corruption
  bool is_subgraph = (graph->sub_graphs_.size() == 1);
  MS_LOG(INFO) << "is_subgraph=" << is_subgraph << " (based on sub_graphs_.size()=" << graph->sub_graphs_.size() << ")";

  if (is_subgraph) {
    MS_LOG(INFO) << "Path: Freeing SUBGRAPH - deleting nodes but NOT tensors";
    // Delete nodes (newly allocated), clear tensors vector (pointers to original graph)
    MS_LOG(INFO) << "Deleting " << graph->all_nodes_.size() << " nodes";
    for (size_t idx = 0; idx < graph->all_nodes_.size(); idx++) {
      if (graph->all_nodes_[idx] != nullptr) {
        MS_LOG(DEBUG) << "Deleting node " << idx;
        delete graph->all_nodes_[idx];
        graph->all_nodes_[idx] = nullptr;
      }
    }
    MS_LOG(INFO) << "All nodes deleted";

    MS_LOG(INFO) << "Clearing all_tensors_ vector (size=" << graph->all_tensors_.size() << ")";
    graph->all_tensors_.clear();

    MS_LOG(INFO) << "Deleting " << graph->sub_graphs_.size() << " sub_graphs";
    for (size_t idx = 0; idx < graph->sub_graphs_.size(); idx++) {
      if (graph->sub_graphs_[idx] != nullptr) {
        delete graph->sub_graphs_[idx];
        graph->sub_graphs_[idx] = nullptr;
      }
    }
    MS_LOG(INFO) << "All sub_graphs deleted";
  } else {
    MS_LOG(INFO) << "Path: Freeing FULL GRAPH - deleting nodes and tensors";
    MS_LOG(INFO) << "Deleting " << graph->all_nodes_.size() << " nodes";
    for (size_t idx = 0; idx < graph->all_nodes_.size(); idx++) {
      if (graph->all_nodes_[idx] != nullptr) {
        delete graph->all_nodes_[idx];
        graph->all_nodes_[idx] = nullptr;
      }
    }
    MS_LOG(INFO) << "All nodes deleted";
  }

  MS_LOG(INFO) << "Clearing graph name: " << graph->name_;
  graph->name_.clear();

  graph->input_indices_.clear();
  graph->output_indices_.clear();

  MS_LOG(INFO) << "Destroying " << graph->sub_graphs_.size() << " subgraphs";
  for (size_t idx = 0; idx < graph->sub_graphs_.size(); idx++) {
    if (graph->sub_graphs_[idx] != nullptr) {
      delete graph->sub_graphs_[idx];
      graph->sub_graphs_[idx] = nullptr;
    }
  }

  delete graph;
  *liteGraph = nullptr;
  MS_LOG(INFO) << "========== FreeLiteGraph EXIT (SUCCESS) ==========";
}

NNRTDelegate::~NNRTDelegate() {
#ifdef SUPPORT_NNRT_METAGRAPH
  if (hiai_handle_ != nullptr) {
    (void)mindspore::lite::UnLoadHiaiFLibrary(hiai_handle_);
    hiai_handle_ = nullptr;
  }
#endif
  auto& nnrtWrapper = mindspore::NNRTWrapper::GetInstance();
  for (size_t i = 0; i < nn_executor_list_.size(); i++) {
    if (nn_executor_list_[i] != nullptr) {
      MS_LOG(DEBUG) << "start NNExecutor Destroy.";
      nnrtWrapper.NNExecutorDestroy(&(nn_executor_list_[i]));
      MS_LOG(DEBUG) << "Destroy NNExecutor Finish.";
    }
  }
  
  if (lite_graph_ != nullptr) {
    #ifdef SUPPORT_NNRT_METAGRAPH
    if (IsKirinNPUWithOfflineInference()) {
      FreeLiteGraph(&lite_graph_);
      lite_graph_ = nullptr;
    }
    #endif
    MS_LOG(ERROR) << "Delete NNRTDelegate.";
  }
  for (auto iter : dequant_schema_tensors_buffer_map_) {
    if (iter.second != nullptr) {
      free(iter.second);
      iter.second = nullptr;
    }
  }
  dequant_schema_tensors_buffer_map_.clear();
  replaced_schema_tensors_.clear();
}
}  // namespace lite
}  // namespace mindspore
