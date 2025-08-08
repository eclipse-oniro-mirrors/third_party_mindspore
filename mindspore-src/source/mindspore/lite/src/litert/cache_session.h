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

#ifndef MINDSPORE_LITE_SRC_LITERT_CACHE_SESSION_H_
#define MINDSPORE_LITE_SRC_LITERT_CACHE_SESSION_H_

#include "src/litert/lite_session.h"
#include "src/litert/inner_context.h"
#include "src/litert/lite_model.h"
#include "src/litert/delegate/nnrt/extension_options_parser.h"
#include "neural_network_runtime/neural_network_runtime_type.h"
#include "neural_network_runtime/neural_network_runtime.h"
#include "neural_network_runtime_inner.h"

namespace mindspore {
namespace lite {
class CacheSession : public LiteSession {
 public:
  CacheSession() = default;
  ~CacheSession() override;
  int Init(const std::shared_ptr<InnerContext> &context) override;
  int CompileGraph(Model *model) override;
  int LoadModelAndCompileByPath(const std::string &model_path, mindspore::ModelType model_type) override;
  static bool IsKirinNPUWithOnlineInference(size_t device_id);
  const char *LoadModelByPath(const std::string &file, mindspore::ModelType model_type, size_t *size,
                              bool use_mmap) override;
  Model* ImportInOutFromBuffer(const char *model_buf, size_t size, bool take_buf,
                               mindspore::ModelType model_type = mindspore::ModelType::kMindIR_Lite,
                               const std::string &path = "");

  template <typename T = schema::MetaGraph>
  bool ConvertInputOutputTensors(const T &meta_graph, LiteGraph &graph_) {
    if (meta_graph.allTensors() == nullptr) {
      MS_LOG(ERROR) << "meta_graph is invalid, please check your model file.";
      return false;
    }

    graph_.all_tensors_.resize(meta_graph.allTensors()->size());
    MS_LOG(INFO) << "convert input/output tensors";
    for (auto i: graph_.input_indices_) {
      auto *tensor = meta_graph.allTensors()->template GetAs<schema::Tensor>(i);
      if (tensor == nullptr) {
        MS_LOG(ERROR) << i << " the input tensor in metagraph is nullptr";
        return false;
      }
      MS_CHECK_TRUE_RET(tensor->format() >= schema::Format_MIN && tensor->format() <= schema::Format_MAX, false);
      graph_.all_tensors_[i] = (const_cast<mindspore::schema::Tensor *>(tensor));
    }

    for (auto i: graph_.output_indices_) {
      auto *tensor = meta_graph.allTensors()->template GetAs<schema::Tensor>(i);
      if (tensor == nullptr) {
        MS_LOG(ERROR) << i << " the output tensor in metagraph is nullptr";
      }
      MS_CHECK_TRUE_RET(tensor->format() >= schema::Format_MIN && tensor->format() <= schema::Format_MAX, false);
      graph_.all_tensors_[i] = (const_cast<mindspore::schema::Tensor *>(tensor));
    }
    return true;
  }

  template <typename T = schema::MetaGraph, typename U = schema::CNode>
  int GenerateModelInputOutput(const T &meta_graph, LiteGraph &graph_) {
    if (meta_graph.name() != nullptr) {
      graph_.name_ = meta_graph.name()->c_str();
    }
    if (meta_graph.version() != nullptr) {
      graph_.version_ = meta_graph.version()->c_str();
    }

    if (meta_graph.inputIndex() == nullptr || meta_graph.outputIndex() == nullptr ||
        meta_graph.allTensors() == nullptr) {
      MS_LOG(ERROR) << "meta_graph is invalid, please check your model file.";
      return RET_ERROR;
    }

    // converterInputOutput
    auto in_count = meta_graph.inputIndex()->size();
    for (uint32_t i = 0; i < in_count; ++i) {
      graph_.input_indices_.push_back(meta_graph.inputIndex()->Get(i));
    }
    auto out_count = meta_graph.outputIndex()->size();
    for (uint32_t i = 0; i < out_count; ++i) {
      graph_.output_indices_.push_back(meta_graph.outputIndex()->Get(i));
    }

    if (!ConvertInputOutputTensors<T>(meta_graph, graph_)) {
      MS_LOG(ERROR) << "convert tensor failed";
      return RET_ERROR;
    }
    return RET_OK;
  }

  int ParseInputOutputFromModelBuffer(const char *model_buf, LiteModel *model);
  int BindGLTexture2DMemory(const std::map<std::string, unsigned int> &inputGLTexture,
                            std::map<std::string, unsigned int> *outputGLTexture) override {
    return RET_ERROR;
  }

 protected:
  int ScheduleToNNRTKernel();
  Status CreateFullModelKernel();
  Status InitNNCompilation(OH_NNCompilation *nn_compilation) const;
  int ConvertInOutTensors(const lite::Model *model);
  int InitExecutor() override;
  std::vector<mindspore::MSTensor> ms_inputs_;
  std::vector<mindspore::MSTensor> ms_outputs_;

 private:
  NNRtDeviceInfo nnrt_device_info_;
  OH_NNExecutor *nn_executor_{nullptr};
  nnrt::ExtensionOptions extension_options_;
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_LITERT_CACHE_SESSION_H_
