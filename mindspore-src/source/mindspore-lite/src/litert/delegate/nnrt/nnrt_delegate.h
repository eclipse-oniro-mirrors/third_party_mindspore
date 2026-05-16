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
#ifndef MINDSPORE_NNR_DELEGATE_H
#define MINDSPORE_NNR_DELEGATE_H

#include <vector>
#include <map>
#include "include/api/delegate.h"
#include "include/model.h"
#include "src/litert/inner_context.h"
#include "nnrt_model_kernel.h"
#include "hiai_foundation_wrapper.h"
#include "extension_options_parser.h"
#include "schema/model_generated.h"
#include "src/litert/delegate/nnrt/nnrt_wrapper.h"

namespace mindspore {
namespace lite {
struct NNRTOpRange {
  /* NNRT kernel range in DelegateModel: [begin_iter_, end_iter_) */
  KernelIter begin_iter_;
  KernelIter end_iter_;
  /* NNRT node range in lite_graph_: [begin_index_, end_index_) */
  size_t begin_index_;
  size_t end_index_;
};

class NNRTDelegate : public Delegate {
 public:
  NNRTDelegate() = default;
  NNRTDelegate(const NNRtDeviceInfo &nnrt_device_info) : nnrt_device_info_(nnrt_device_info) {}
  ~NNRTDelegate() override;
  Status Init() override;
  Status Build(DelegateModel<schema::Primitive> *model) override;
  void ShallowCopyLiteGraph(const lite::LiteGraph &liteGraph);
  void FreeLiteGraph(lite::LiteGraph **liteGraph);
  void SetMetaGraph(const void *meta_graph) {
    meta_graph_ = meta_graph;
  }
  void SetDequantTensors(std::vector<Tensor *> *src_tensors) {
    dequant_src_tensors_ = src_tensors;
  }
  static std::vector<NNRTOpRange> GetNNRTSubgraphRanges(DelegateModel<schema::Primitive> *model,
                                                        const std::vector<bool> &op_supports);
  bool IsBuildOffline() const { return build_offline_; }
  bool enable_hete_ = false;
  std::map<std::string, TypeId> *hete_execution_plan_ = nullptr;
  std::map<std::string, TypeId> *original_hete_execution_plan_ = nullptr;  // Original config for validation
  bool xpu_backend_set_ = true;  // Config validation status

 private:
  void InitExtensionOptions();
  Status BuildNormalModel(DelegateModel<schema::Primitive> *model);
  OH_NNModel *CreateFullNNModel();
  std::vector<bool> QueryOpSupports(OH_NNModel *nn_model);
  Status CreateLiteGraphForNNRTSubgraph(
    const std::vector<NNRTOpRange> &nnrt_op_ranges,
    std::vector<LiteGraph *> *sub_lite_graphs);
  Status CreateNNRTSubgraphKernels(
    DelegateModel<schema::Primitive> *model,
    const std::vector<LiteGraph *> &sub_lite_graphs,
    const std::vector<NNRTOpRange> &nnrt_subgraph_ranges,
    std::vector<NNRTModelKernel *> *nnrt_subgraph_kernels);
  void ReplaceNNRTKernelsInDelegateModel(DelegateModel<schema::Primitive> *model,
                                         const std::vector<NNRTOpRange> &nnrt_subgraph_ranges,
                                         const std::vector<NNRTModelKernel *> &nnrt_subgraph_kernels);
  Status PrepareInputs(DelegateModel<schema::Primitive> *model, OH_NNExecutor *oh_nn_executor);
  Status PrepareOutputs(DelegateModel<schema::Primitive> *model, OH_NNExecutor *oh_nn_executor);
  Status InitNNCompilation(OH_NNCompilation *nn_compilation) const;
  static OH_NN_DataType CastToNNRTDataType(mindspore::DataType data_type);
  bool IsCustomModel() const;
  Status DequantLiteGraph(LiteGraph *lite_graph);
  int DequantNodeInputs(LiteGraph::Node *node);
  schema::Tensor *TensorToSchemaTensor(Tensor *lite_tensor, schema::Tensor *schema_tensor);
  void ApplyCPUOp(std::vector<bool> &op_supports);
  void PrintNodeNameFormat();

#ifdef SUPPORT_NNRT_METAGRAPH
  bool CheckNPUPrefix(const std::string prefix_name) const;
  bool IsKirinNPUWithOnlineInference() const;
  bool IsKirinNPUWithOfflineInference() const;
  bool CheckXpuConfig();
  Status BuildKirinNPUModel(DelegateModel<schema::Primitive> *model);
  Status BuildOfflineModel(DelegateModel<schema::Primitive> *model);
  Status CreateFullModelKernel(DelegateModel<schema::Primitive> *model, OH_NNModel *nn_model);
#endif

  NNRtDeviceInfo nnrt_device_info_;
  LiteGraph *lite_graph_ = nullptr;
  const void *meta_graph_ = nullptr;
  nnrt::ExtensionOptions extension_options_;
  std::vector<OH_NNExecutor *> nn_executor_list_;
  std::vector<Tensor *> *dequant_src_tensors_;
  std::map<uint32_t, schema::Tensor *> dequant_schema_tensors_;
  std::map<schema::Tensor *, void *> dequant_schema_tensors_buffer_map_;
  std::vector<schema::Tensor *> replaced_schema_tensors_;
  void *hiai_handle_{nullptr};
  bool build_offline_ = false;
  int schema_version_ = SCHEMA_VERSION::SCHEMA_CUR;
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_NNR_DELEGATE_H
