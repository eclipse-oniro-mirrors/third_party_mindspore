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
#include "include/c_api/model_c.h"
#include "type_c_private.h"
#include "context_c.h"
#include <vector>
#include <cstdint>
#include "include/api/context.h"
#include "include/api/serialization.h"
#include "include/api/types.h"
#include "src/litert/cxx_api/tensor/tensor_impl.h"
#include "src/litert/cxx_api/model/model_impl.h"
#include "src/common/utils.h"
#ifdef ENABLE_HI_APP_EVENT
#include "src/common/hi_app_event/hi_app_event.h"
#endif

namespace mindspore {
class ModelC {
 public:
  ModelC() : model_(nullptr) {}
  ~ModelC() {
    for (auto in : inputs_) {
      if (in != nullptr) {
        delete in;
      }
    }
    for (auto out : outputs_) {
      if (out != nullptr) {
        delete out;
      }
    }
    for (auto out : outputs_train_) {
      if (out != nullptr) {
        delete out;
      }
    }

    // In zero copy scene where user will call set or get allocator function, but when model is destroyed, the allocator
    // table will not be freed, and its size continues to grow causing memory leak, so when ModelC is destroyed, clean
    // allocator table.
    CleanAllocatorTable();
  }

  MSTensor **GetInputs(size_t *input_num);
  MSTensor **GetOutputs(size_t *output_num);
  mindspore::MSKernelCallBack TransCallBack(const OH_AI_KernelCallBack &ms_callback);
  std::shared_ptr<Model> model_;
  std::shared_ptr<Context> context_;
  std::atomic_flag in_use_ = ATOMIC_FLAG_INIT;

 private:
  MSTensor **GetOutputsTensor(size_t *output_num, std::vector<MSTensor *> *vec_tensors);
  std::vector<MSTensor *> inputs_;
  std::vector<MSTensor *> outputs_;
  std::vector<MSTensor *> outputs_train_;
};

MSTensor **ModelC::GetInputs(size_t *input_num) {
  if (model_ == nullptr) {
    MS_LOG(ERROR) << "model_ is nullptr.";
    return nullptr;
  }
  if (!inputs_.empty()) {
    *input_num = inputs_.size();
    return inputs_.data();
  }
  auto inputs = model_->GetInputs();
  *input_num = inputs.size();
  inputs_.resize(inputs.size(), nullptr);
  for (size_t i = 0; i < inputs.size(); i++) {
    inputs_[i] = new (std::nothrow) MSTensor(inputs[i].impl());
    if (inputs_[i] == nullptr) {
      inputs_.clear();
      return nullptr;
    }
  }
  return inputs_.data();
}

MSTensor **ModelC::GetOutputs(size_t *output_num) {
  if (model_->GetTrainMode() == true) {
    return GetOutputsTensor(output_num, &outputs_train_);
  } else {
    return GetOutputsTensor(output_num, &outputs_);
  }
}

MSTensor **ModelC::GetOutputsTensor(size_t *output_num, std::vector<MSTensor *> *vec_tensors) {
  if (model_ == nullptr) {
    MS_LOG(ERROR) << "model_ is nullptr.";
    return nullptr;
  }
  if (!vec_tensors->empty()) {
    *output_num = vec_tensors->size();
    return vec_tensors->data();
  }

  auto outputs = model_->GetOutputs();
  *output_num = outputs.size();
  vec_tensors->resize(outputs.size(), nullptr);
  for (size_t i = 0; i < outputs.size(); i++) {
    (*vec_tensors)[i] = new (std::nothrow) MSTensor(outputs[i].impl());
    if ((*vec_tensors)[i] == nullptr) {
      vec_tensors->clear();
      return nullptr;
    }
  }
  return vec_tensors->data();
}

mindspore::MSKernelCallBack ModelC::TransCallBack(const OH_AI_KernelCallBack &ms_callback) {
  mindspore::MSKernelCallBack call_back = nullptr;
  if (ms_callback != nullptr) {
    call_back = [&](const std::vector<mindspore::MSTensor> &inputs, const std::vector<mindspore::MSTensor> &outputs,
                    const mindspore::MSCallBackParam &opInfo) {
      std::vector<OH_AI_TensorHandle> vec_inputs;
      std::vector<OH_AI_TensorHandle> vec_outputs;
      OH_AI_CallBackParam call_back = {const_cast<char *>(opInfo.node_name.c_str()),
                                    const_cast<char *>(opInfo.node_type.c_str())};
      size_t inputs_handle_num = inputs.size();
      for (size_t i = 0; i < inputs_handle_num; i++) {
        vec_inputs.push_back(static_cast<OH_AI_TensorHandle>(&(static_cast<std::vector<mindspore::MSTensor>>(inputs)[i])));
      }
      size_t outputs_handle_num = outputs.size();
      for (size_t i = 0; i < outputs_handle_num; i++) {
        vec_outputs.push_back(
          static_cast<OH_AI_TensorHandle>(&(static_cast<std::vector<mindspore::MSTensor>>(outputs)[i])));
      }
      OH_AI_TensorHandleArray handle_inputs = {inputs_handle_num, vec_inputs.data()};
      OH_AI_TensorHandleArray handle_outputs = {outputs_handle_num, vec_outputs.data()};
      return ms_callback(handle_inputs, handle_outputs, call_back);
    };
  }
  return call_back;
}
}  // namespace mindspore

OH_AI_ModelHandle OH_AI_ModelCreate() {
  MS_LOG(DEBUG) << "Start to create ms model";
  auto impl = new (std::nothrow) mindspore::ModelC();
  if (impl == nullptr) {
    MS_LOG(ERROR) << "Model implement is nullptr.";
    return nullptr;
  }
  impl->model_ = std::make_shared<mindspore::Model>();
  if (impl->model_ == nullptr) {
    MS_LOG(ERROR) << "inner model object is nullptr.";
    delete impl;
    return nullptr;
  }
  MS_LOG(DEBUG) << "Created ms model successfully";
  return static_cast<OH_AI_ModelHandle>(impl);
}

void OH_AI_ModelDestroy(OH_AI_ModelHandle *model) {
  MS_LOG(DEBUG) << "Start to destroy ms model";
  if (model == nullptr || *model == nullptr) {
    MS_LOG(ERROR) << "model is nullptr.";
    return;
  }
  auto impl = static_cast<mindspore::ModelC *>(*model);
  if (impl->in_use_.test_and_set()) {
    MS_LOG(ERROR) << "model in use by other interface, destroy fail. Need destroy again; otherwise, a memory leak may occur.";
    return;
  }
  delete impl;
  *model = nullptr;
  MS_LOG(DEBUG) << "Destroyed ms model successfully";
}

void OH_AI_ModelSetWorkspace(OH_AI_ModelHandle model, void *workspace, size_t workspace_size) {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return;
}

size_t OH_AI_ModelCalcWorkspaceSize(OH_AI_ModelHandle model) {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return 0;
}

OH_AI_Status OH_AI_ModelBuild(OH_AI_ModelHandle model, const void *model_data, size_t data_size, OH_AI_ModelType model_type,
                      const OH_AI_ContextHandle model_context) {
  MS_LOG(INFO) << "Start to build ms model";
  uint64_t start_build = mindspore::lite::GetTimeUs();
  if (model == nullptr || model_data == nullptr || model_context == nullptr) {
    MS_LOG(ERROR) << "model or model_data or model_context is nullptr.";
    return OH_AI_STATUS_LITE_NULLPTR;
  }
  if (model_type == OH_AI_MODELTYPE_INVALID) {
    MS_LOG(ERROR) << "model_type is invalid.";
    return OH_AI_STATUS_LITE_PARAM_INVALID;
  }
  mindspore::ContextC *context = static_cast<mindspore::ContextC *>(model_context);
  auto impl = static_cast<mindspore::ModelC *>(model);
  if (impl->context_.get() != context->context_ && context->owned_by_model_) {
    MS_LOG(ERROR) << "context is owned by other model.";
    return OH_AI_STATUS_LITE_PARAM_INVALID;
  }
  if (impl->context_.get() != context->context_) {
    impl->context_.reset(context->context_);
    context->owned_by_model_ = true;
  }
  auto ret = impl->model_->Build(model_data, data_size, static_cast<mindspore::ModelType>(model_type), impl->context_);
  if (ret.IsOk()) {
    MS_LOG(INFO) << "Built ms model successfully";
  } else {
    MS_LOG(ERROR) << "Built ms model failed, ret: " << ret;
  }
  uint64_t build_time = mindspore::lite::GetTimeUs() - start_build;
  MS_LOG(DEBUG) << "The build time of the Lite model is: " << build_time << "us";
  return static_cast<OH_AI_Status>(ret.StatusCode());
}

OH_AI_Status OH_AI_ModelBuildFromFile(OH_AI_ModelHandle model, const char *model_path, OH_AI_ModelType model_type,
                              const OH_AI_ContextHandle model_context) {
  uint64_t start_build = mindspore::lite::GetTimeUs();
  MS_LOG(DEBUG) << "Start to build ms model from file";
  if (model == nullptr || model_path == nullptr || model_context == nullptr) {
    MS_LOG(ERROR) << "model or model_path or model_context is nullptr.";
    return OH_AI_STATUS_LITE_NULLPTR;
  }
  if (model_type == OH_AI_MODELTYPE_INVALID) {
    MS_LOG(ERROR) << "model_type is invalid.";
    return OH_AI_STATUS_LITE_PARAM_INVALID;
  }
  mindspore::ContextC *context = static_cast<mindspore::ContextC *>(model_context);
  auto impl = static_cast<mindspore::ModelC *>(model);
  if (impl->in_use_.test_and_set()) {
    MS_LOG(ERROR) << "model in use by other interface, build model fail.";
    return OH_AI_STATUS_LITE_ERROR;
  }
  if (impl->context_.get() != context->context_ && context->owned_by_model_) {
    MS_LOG(ERROR) << "context is owned by other model.";
    impl->in_use_.clear();
    return OH_AI_STATUS_LITE_PARAM_INVALID;
  }
  if (impl->context_.get() != context->context_) {
    impl->context_.reset(context->context_);
    context->owned_by_model_ = true;
  }
  auto ret = impl->model_->Build(model_path, static_cast<mindspore::ModelType>(model_type), impl->context_);
  if (ret.IsOk()) {
    MS_LOG(DEBUG) << "Built ms model from file successfully";
  } else {
    MS_LOG(ERROR) << "Built ms model from file failed, ret: " << ret;
  }
  impl->in_use_.clear();
  uint64_t build_time = mindspore::lite::GetTimeUs() - start_build;
  MS_LOG(DEBUG) << "The build time of the Lite model is: " << build_time << "us";
  return static_cast<OH_AI_Status>(ret.StatusCode());
}

OH_AI_Status OH_AI_ModelResize(OH_AI_ModelHandle model, const OH_AI_TensorHandleArray inputs, OH_AI_ShapeInfo *shape_infos,
                       size_t shape_info_num) {
  MS_LOG(INFO) << "Start to resize ms model";
  if (model == nullptr || shape_infos == nullptr) {
    MS_LOG(ERROR) << "model or shape_infos is nullptr.";
    return OH_AI_STATUS_LITE_NULLPTR;
  }
  std::vector<mindspore::MSTensor> vec_inputs;
  for (size_t i = 0; i < inputs.handle_num; ++i) {
    vec_inputs.push_back(*static_cast<mindspore::MSTensor *>(inputs.handle_list[i]));
  }

  std::vector<std::vector<int64_t>> vec_dims;
  for (size_t i = 0; i < shape_info_num; i++) {
    std::vector<int64_t> shape(shape_infos[i].shape, shape_infos[i].shape + shape_infos[i].shape_num);
    if (std::any_of(shape.begin(), shape.end(), [](int64_t val) { return val < 0 || val > INT32_MAX; })) {
      MS_LOG(ERROR) << "Invalid shape: " << shape << ", each dimension must be in [0, INT32_MAX]";
      return OH_AI_STATUS_LITE_PARAM_INVALID;
    }
    vec_dims.push_back(shape);
  }
  auto impl = static_cast<mindspore::ModelC *>(model);
  if (impl->model_->GetUseAipp()) {
    MS_LOG(ERROR) << "aipp is true, not use model resize!!! ";
    return OH_AI_STATUS_LITE_ERROR;
  }
  auto ret = impl->model_->Resize(vec_inputs, vec_dims);
  if (ret.IsOk()) {
    MS_LOG(INFO) << "Resized ms model successfully";
  } else {
    MS_LOG(ERROR) << "Resized ms model failed, ret: " << ret;
  }
  return static_cast<OH_AI_Status>(ret.StatusCode());
}

OH_AI_Status OH_AI_ModelPredict(OH_AI_ModelHandle model, const OH_AI_TensorHandleArray inputs, OH_AI_TensorHandleArray *outputs,
                        const OH_AI_KernelCallBack before, const OH_AI_KernelCallBack after) {
  MS_LOG(DEBUG) << "Start to predict ms model";
  uint64_t start_predict = mindspore::lite::GetTimeUs();
  if (model == nullptr) {
    MS_LOG(ERROR) << "model is nullptr.";
    return OH_AI_STATUS_LITE_NULLPTR;
  }
  auto impl = static_cast<mindspore::ModelC *>(model);
  if (impl->in_use_.test_and_set()) {
    MS_LOG(ERROR) << "model in use by other interface, predict fail.";
    return OH_AI_STATUS_LITE_ERROR;
  }
  size_t input_num;
  (void)impl->GetInputs(&input_num);
  if (input_num != inputs.handle_num) {
    MS_LOG(ERROR) << "Wrong input size.";
    impl->in_use_.clear();
    return OH_AI_STATUS_LITE_ERROR;
  }

  std::vector<mindspore::MSTensor> ms_tensor_inputs;
  for (size_t i = 0; i < inputs.handle_num; i++) {
    if (inputs.handle_list[i] != nullptr) {
      auto user_input = static_cast<mindspore::MSTensor *>(inputs.handle_list[i]);
      ms_tensor_inputs.push_back(*user_input);
    } else {
      MS_LOG(ERROR) << "input handle is nullptr.";
      impl->in_use_.clear();
      return OH_AI_STATUS_LITE_NULLPTR;
    }
  }

  mindspore::MSKernelCallBack before_call_back = impl->TransCallBack(before);
  mindspore::MSKernelCallBack after_call_back = impl->TransCallBack(after);
  std::vector<mindspore::MSTensor> ms_tensor_outputs;

  size_t output_num;
  (void)impl->GetOutputs(&output_num);
  auto handle_num = outputs->handle_num;
  if (handle_num == output_num) {
    MS_LOG(DEBUG) << "use user provided output";
    for (size_t i = 0; i < output_num; i++) {
      if (outputs->handle_list[i] == nullptr) {
        MS_LOG(ERROR) << "user provided output array handle_list[" << i << "] is nullptr";
        impl->in_use_.clear();
        return OH_AI_STATUS_LITE_NULLPTR;
      }
      ms_tensor_outputs.push_back(*static_cast<mindspore::MSTensor *>(outputs->handle_list[i]));
    }
  }

  auto ret = impl->model_->Predict(ms_tensor_inputs, &ms_tensor_outputs, before_call_back, after_call_back);
  if (!ret.IsOk()) {
    MS_LOG(ERROR) << "Predict fail, ret :" << ret;
    impl->in_use_.clear();
    return static_cast<OH_AI_Status>(ret.StatusCode());
  }

  if (handle_num == output_num) {
    impl->in_use_.clear();
    return OH_AI_STATUS_SUCCESS;
  }

  outputs->handle_list = reinterpret_cast<OH_AI_TensorHandle *>(impl->GetOutputs(&(outputs->handle_num)));
  MS_LOG(DEBUG) << "Predicted ms model successfully";
  impl->in_use_.clear();
  uint64_t predict_time = mindspore::lite::GetTimeUs() - start_predict;
  MS_LOG(DEBUG) << "The inference time of the Lite model is: " << predict_time << "us";
  return static_cast<OH_AI_Status>(ret.StatusCode());
}

OH_AI_Status OH_AI_ModelPredictWithConfig(OH_AI_ModelHandle model, const OH_AI_TensorHandleArray inputs,
                                             OH_AI_TensorHandleArray *outputs, const char *predict_config,
                                             const OH_AI_KernelCallBack before, const OH_AI_KernelCallBack after) {
  if (predict_config == nullptr || predict_config[0] == '\0') {
    MS_LOG(ERROR) << "Unsupported predict_config is empty";
    return OH_AI_STATUS_LITE_PARAM_INVALID;
  }
  if (model == nullptr) {
    MS_LOG(ERROR) << "Input model cannot be nullptr";
    return OH_AI_STATUS_LITE_NULLPTR;
  }
  auto impl = static_cast<mindspore::ModelC *>(model);
  impl->model_->SetPredictModelPara(predict_config);
  OH_AI_Status ret = OH_AI_ModelPredict(model, inputs, outputs, before, after);
  return ret;
}

OH_AI_Status OH_AI_ModelRunStep(OH_AI_ModelHandle model, const OH_AI_KernelCallBack before, const OH_AI_KernelCallBack after) {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return OH_AI_STATUS_LITE_NOT_SUPPORT;
}

OH_AI_Status OH_AI_ModelExportWeight(const OH_AI_ModelHandle model, const char *export_path) {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return OH_AI_STATUS_LITE_NOT_SUPPORT;
}

OH_AI_TensorHandleArray OH_AI_ModelGetInputs(const OH_AI_ModelHandle model) {
  MS_LOG(DEBUG) << "Start to get ms model inputs";
  if (model == nullptr) {
    MS_LOG(ERROR) << "model is nullptr.";
    return {0, nullptr};
  }
  auto impl = static_cast<mindspore::ModelC *>(model);
  size_t input_num = 0;
  auto handles = reinterpret_cast<OH_AI_TensorHandle *>(impl->GetInputs(&input_num));
  MS_LOG(DEBUG) << "Got ms model " << input_num << " inputs successfully";
  return {input_num, handles};
}

OH_AI_TensorHandleArray OH_AI_ModelGetOutputs(const OH_AI_ModelHandle model) {
  MS_LOG(DEBUG) << "Start to get ms model outputs";
  if (model == nullptr) {
    MS_LOG(ERROR) << "model is nullptr.";
    return {0, nullptr};
  }
  auto impl = static_cast<mindspore::ModelC *>(model);
  size_t output_num;
  auto handles = reinterpret_cast<OH_AI_TensorHandle *>(impl->GetOutputs(&output_num));
  MS_LOG(DEBUG) << "Got ms model " << output_num << " outputs successfully";
  return {output_num, handles};
}

OH_AI_TensorHandle OH_AI_ModelGetInputByTensorName(const OH_AI_ModelHandle model, const char *tensor_name) {
  MS_LOG(INFO) << "Start to get ms model input by name";
  if (model == nullptr || tensor_name == nullptr) {
    MS_LOG(ERROR) << "model or tensor_name is nullptr.";
    return nullptr;
  }
  auto impl = static_cast<mindspore::ModelC *>(model);
  size_t input_num;
  auto inputs = impl->GetInputs(&input_num);
  for (size_t i = 0; i < input_num; i++) {
    if (inputs[i]->Name() == tensor_name) {
      MS_LOG(INFO) << "Got ms model input by name successfully";
      return static_cast<OH_AI_TensorHandle>(inputs[i]);
    }
  }
  MS_LOG(ERROR) << "Input tensor is not exist";
  return nullptr;
}

OH_AI_TensorHandle OH_AI_ModelGetOutputByTensorName(const OH_AI_ModelHandle model, const char *tensor_name) {
  MS_LOG(INFO) << "Start to get ms model output by name";
  if (model == nullptr || tensor_name == nullptr) {
    MS_LOG(ERROR) << "model or tensor_name is nullptr.";
    return nullptr;
  }
  auto impl = static_cast<mindspore::ModelC *>(model);
  size_t output_num;
  auto outputs = impl->GetOutputs(&output_num);
  for (size_t i = 0; i < output_num; i++) {
    if (outputs[i]->Name() == tensor_name) {
      MS_LOG(INFO) << "Got ms model output by name successfully";
      return static_cast<OH_AI_TensorHandle>(outputs[i]);
    }
  }
  MS_LOG(ERROR) << "Output tensor is not exist";
  return nullptr;
}

OH_AI_TrainCfgHandle OH_AI_TrainCfgCreate() {
  auto impl = new (std::nothrow) mindspore::TrainCfg();
  if (impl == nullptr) {
    MS_LOG(ERROR) << "TrainCfg implement is nullptr.";
    return nullptr;
  }
  return static_cast<OH_AI_TrainCfgHandle>(impl);
}

void OH_AI_TrainCfgDestroy(OH_AI_TrainCfgHandle *train_cfg) {
  if (train_cfg != nullptr && *train_cfg != nullptr) {
    auto impl = static_cast<mindspore::TrainCfg *>(*train_cfg);
    delete impl;
    *train_cfg = nullptr;
  }
}

char **OH_AI_TrainCfgGetLossName(OH_AI_TrainCfgHandle train_cfg, size_t *num) {
  if (train_cfg == nullptr || num == nullptr) {
    MS_LOG(ERROR) << "train_cfg or num is nullptr.";
    return nullptr;
  }
  auto impl = static_cast<mindspore::TrainCfg *>(train_cfg);
  auto loss_name = impl->GetLossName();
  *num = loss_name.size();
  char **name = static_cast<char **>(malloc(loss_name.size() * sizeof(char *)));
  if (name == nullptr) {
    MS_LOG(ERROR) << "Failed to malloc loss_name.";
    return nullptr;
  }
  for (size_t i = 0; i < loss_name.size(); i++) {
    name[i] = static_cast<char *>(malloc(loss_name[i].size() + 1));
    if (name[i] == nullptr) {
      for(size_t j = 0; j < i; j++){
        free(name[j]);
      }
      MS_LOG(ERROR) << "Failed to malloc name.";
      return nullptr;
    }
    memcpy(name[i], loss_name[i].c_str(), loss_name[i].size() + 1);
  }
  return name;
}

void OH_AI_TrainCfgSetLossName(OH_AI_TrainCfgHandle train_cfg, const char **loss_name, size_t num) {
  if (train_cfg == nullptr || loss_name == nullptr || *loss_name == nullptr) {
    MS_LOG(ERROR) << "train_cfg or loss_name is nullptr.";
    return;
  }
  auto impl = static_cast<mindspore::TrainCfg *>(train_cfg);
  std::vector<std::string> vec_name;
  for (size_t i = 0; i < num; i++) {
    vec_name.push_back(loss_name[i]);
  }
  impl->SetLossName(vec_name);
}

OH_AI_OptimizationLevel OH_AI_TrainCfgGetOptimizationLevel(OH_AI_TrainCfgHandle train_cfg) {
  if (train_cfg == nullptr) {
    MS_LOG(ERROR) << "train_cfg is nullptr, return OH_AI_KO0";
    return OH_AI_KO0;
  }
  auto impl = static_cast<mindspore::TrainCfg *>(train_cfg);
  return static_cast<OH_AI_OptimizationLevel>(impl->optimization_level_);
}

void OH_AI_TrainCfgSetOptimizationLevel(OH_AI_TrainCfgHandle train_cfg, OH_AI_OptimizationLevel level) {
  if (train_cfg == nullptr) {
    MS_LOG(ERROR) << "train_cfg is nullptr.";
    return;
  }
  auto impl = static_cast<mindspore::TrainCfg *>(train_cfg);
  impl->optimization_level_ = static_cast<mindspore::OptimizationLevel>(level);
}

OH_AI_Status OH_AI_TrainModelBuild(OH_AI_ModelHandle model, const void *model_data, size_t data_size, OH_AI_ModelType model_type,
                           const OH_AI_ContextHandle model_context, const OH_AI_TrainCfgHandle train_cfg) {
  if (model == nullptr || model_data == nullptr || model_context == nullptr) {
    MS_LOG(ERROR) << "model or model_data or model_context is nullptr.";
    return OH_AI_STATUS_LITE_NULLPTR;
  }
  if (model_type == OH_AI_MODELTYPE_INVALID) {
    MS_LOG(ERROR) << "model_type is invalid.";
    return OH_AI_STATUS_LITE_PARAM_INVALID;
  }
  auto impl = static_cast<mindspore::ModelC *>(model);

  mindspore::Graph graph;
  auto status =
    mindspore::Serialization::Load(model_data, data_size, static_cast<mindspore::ModelType>(model_type), &graph);
  if (status != mindspore::kSuccess) {
    MS_LOG(ERROR) << "load ms file failed.";
    return OH_AI_STATUS_LITE_ERROR;
  }
  auto context = static_cast<mindspore::ContextC *>(model_context);
  auto build_train_cfg = static_cast<mindspore::TrainCfg *>(train_cfg);
  if (impl->context_.get() != context->context_ && context->owned_by_model_) {
    MS_LOG(ERROR) << "context is owned by other model.";
    return OH_AI_STATUS_LITE_PARAM_INVALID;
  }
  if (impl->context_.get() != context->context_) {
    impl->context_.reset(context->context_);
    context->owned_by_model_ = true;
  }
  auto ret = impl->model_->Build(static_cast<mindspore::GraphCell>(graph), impl->context_,
                                 std::shared_ptr<mindspore::TrainCfg>(build_train_cfg));
  if (ret != mindspore::kSuccess) {
    MS_LOG(ERROR) << "Load and compile failed";
  }
  return static_cast<OH_AI_Status>(ret.StatusCode());
}

OH_AI_Status OH_AI_TrainModelBuildFromFile(OH_AI_ModelHandle model, const char *model_path, OH_AI_ModelType model_type,
                                   const OH_AI_ContextHandle model_context, const OH_AI_TrainCfgHandle train_cfg) {
  if (model == nullptr || model_path == nullptr || model_context == nullptr) {
    MS_LOG(ERROR) << "model or model_path or model_context is nullptr.";
    return OH_AI_STATUS_LITE_NULLPTR;
  }
  if (model_type == OH_AI_MODELTYPE_INVALID) {
    MS_LOG(ERROR) << "model_type is invalid.";
    return OH_AI_STATUS_LITE_PARAM_INVALID;
  }
  auto impl = static_cast<mindspore::ModelC *>(model);

  mindspore::Graph graph;
  auto status = mindspore::Serialization::Load(model_path, static_cast<mindspore::ModelType>(model_type), &graph);
  if (status != mindspore::kSuccess) {
    MS_LOG(ERROR) << "load ms file failed. " << model_path;
    return OH_AI_STATUS_LITE_ERROR;
  }
  auto context = static_cast<mindspore::ContextC *>(model_context);
  auto build_train_cfg = static_cast<mindspore::TrainCfg *>(train_cfg);
  if (impl->context_.get() != context->context_ && context->owned_by_model_) {
    MS_LOG(ERROR) << "context is owned by other model.";
    return OH_AI_STATUS_LITE_PARAM_INVALID;
  }
  if (impl->context_.get() != context->context_) {
    impl->context_.reset(context->context_);
    context->owned_by_model_ = true;
  }
  auto ret = impl->model_->Build(static_cast<mindspore::GraphCell>(graph), impl->context_,
                                 std::shared_ptr<mindspore::TrainCfg>(build_train_cfg));
  if (ret != mindspore::kSuccess) {
    MS_LOG(ERROR) << "Load and compile failed";
  }
  return static_cast<OH_AI_Status>(ret.StatusCode());
}

OH_AI_Status OH_AI_ModelSetLearningRate(OH_AI_ModelHandle model, float learning_rate) {
  if (model == nullptr) {
    MS_LOG(ERROR) << "model is nullptr.";
    return OH_AI_STATUS_LITE_PARAM_INVALID;
  }
  auto impl = static_cast<mindspore::ModelC *>(model);
  auto ret = impl->model_->SetLearningRate(learning_rate);
  return static_cast<OH_AI_Status>(ret.StatusCode());
}

float OH_AI_ModelGetLearningRate(OH_AI_ModelHandle model) {
  if (model == nullptr) {
    MS_LOG(ERROR) << "model is nullptr.";
    return OH_AI_STATUS_LITE_PARAM_INVALID;
  }
  auto impl = static_cast<mindspore::ModelC *>(model);
  return impl->model_->GetLearningRate();
}

OH_AI_Status OH_AI_RunStep(OH_AI_ModelHandle model, const OH_AI_KernelCallBack before, const OH_AI_KernelCallBack after) {
  if (model == nullptr) {
    MS_LOG(ERROR) << "model is nullptr.";
    return OH_AI_STATUS_LITE_PARAM_INVALID;
  }
  auto impl = static_cast<mindspore::ModelC *>(model);
  auto ret = impl->model_->RunStep(impl->TransCallBack(before), impl->TransCallBack(after));
  return static_cast<OH_AI_Status>(ret.StatusCode());
}

OH_AI_TensorHandleArray OH_AI_ModelGetWeights(OH_AI_ModelHandle model) {
  if (model == nullptr) {
    MS_LOG(ERROR) << "model is nullptr.";
    return {0, nullptr};
  }
  auto impl = static_cast<mindspore::ModelC *>(model);
  auto features = impl->model_->GetFeatureMaps();
  size_t handle_num = features.size();

  mindspore::MSTensor **handle_list =
    static_cast<mindspore::MSTensor **>(malloc(handle_num * sizeof(mindspore::MSTensor *)));
  if (handle_list == nullptr) {
    MS_LOG(ERROR) << "Failed to malloc handle_list.";
    return {0, nullptr};
  }
  for (size_t i = 0; i < handle_num; i++) {
    handle_list[i] = new (std::nothrow) mindspore::MSTensor(features[i].impl());
  }
  return {handle_num, reinterpret_cast<OH_AI_TensorHandle *>(handle_list)};
}

OH_AI_Status OH_AI_ModelUpdateWeights(OH_AI_ModelHandle model, const OH_AI_TensorHandleArray new_weights) {
  if (model == nullptr) {
    MS_LOG(ERROR) << "model is nullptr.";
    return OH_AI_STATUS_LITE_PARAM_INVALID;
  }
  auto impl = static_cast<mindspore::ModelC *>(model);
  std::vector<mindspore::MSTensor> weights;
  for (size_t i = 0; i < new_weights.handle_num; i++) {
    weights.push_back(*static_cast<mindspore::MSTensor *>(new_weights.handle_list[i]));
  }
  auto ret = impl->model_->UpdateWeights(weights);
  return static_cast<OH_AI_Status>(ret.StatusCode());
}

bool OH_AI_ModelGetTrainMode(OH_AI_ModelHandle model) {
  if (model == nullptr) {
    MS_LOG(ERROR) << "model is nullptr.";
    return false;
  }
  auto impl = static_cast<mindspore::ModelC *>(model);
  return impl->model_->GetTrainMode();
}

OH_AI_Status OH_AI_ModelSetTrainMode(OH_AI_ModelHandle model, bool train) {
  if (model == nullptr) {
    MS_LOG(ERROR) << "model is nullptr.";
    return OH_AI_STATUS_LITE_PARAM_INVALID;
  }
  auto impl = static_cast<mindspore::ModelC *>(model);
  auto ret = impl->model_->SetTrainMode(train);
  return static_cast<OH_AI_Status>(ret.StatusCode());
}

OH_AI_Status OH_AI_ModelSetupVirtualBatch(OH_AI_ModelHandle model, int virtual_batch_multiplier, float lr, float momentum) {
  if (model == nullptr) {
    MS_LOG(ERROR) << "model is nullptr.";
    return OH_AI_STATUS_LITE_PARAM_INVALID;
  }
  auto impl = static_cast<mindspore::ModelC *>(model);
  auto ret = impl->model_->SetupVirtualBatch(virtual_batch_multiplier, lr, momentum);
  return static_cast<OH_AI_Status>(ret.StatusCode());
}

OH_AI_Status OH_AI_ExportModel(OH_AI_ModelHandle model, OH_AI_ModelType model_type, const char *model_file,
                       OH_AI_QuantizationType quantization_type, bool export_inference_only, char **output_tensor_name,
                       size_t num) {
  if (model == nullptr) {
    MS_LOG(ERROR) << "model is nullptr.";
    return OH_AI_STATUS_LITE_PARAM_INVALID;
  }
  auto impl = static_cast<mindspore::ModelC *>(model);
  std::vector<std::string> tensor_name;
  for (size_t i = 0; i < num; i++) {
    tensor_name.push_back(output_tensor_name[i]);
  }
  auto ret = mindspore::Serialization::ExportModel(
    *(impl->model_.get()), static_cast<mindspore::ModelType>(model_type), model_file,
    static_cast<mindspore::QuantizationType>(quantization_type), export_inference_only, tensor_name);
  if (!ret.IsOk()) {
    MS_LOG(ERROR) << "export model fail, ret :" << ret;
  }
  return static_cast<OH_AI_Status>(ret.StatusCode());
}

OH_AI_Status OH_AI_ExportModelBuffer(OH_AI_ModelHandle model, OH_AI_ModelType model_type, char **model_data, size_t *data_size,
                             OH_AI_QuantizationType quantization_type, bool export_inference_only,
                             char **output_tensor_name, size_t num) {
  if (model == nullptr) {
    MS_LOG(ERROR) << "model is nullptr.";
    return OH_AI_STATUS_LITE_PARAM_INVALID;
  }
  auto impl = static_cast<mindspore::ModelC *>(model);
  std::vector<std::string> tensor_name;
  for (size_t i = 0; i < num; i++) {
    tensor_name.push_back(output_tensor_name[i]);
  }
  mindspore::Buffer buffer;
  auto ret = mindspore::Serialization::ExportModel(*(impl->model_.get()), static_cast<mindspore::ModelType>(model_type),
                                                   &buffer, static_cast<mindspore::QuantizationType>(quantization_type),
                                                   export_inference_only, tensor_name);
  auto data = reinterpret_cast<char *>(buffer.MutableData());
  *model_data = reinterpret_cast<char *>(malloc(buffer.DataSize()));
  if (*model_data == nullptr) {
    MS_LOG(ERROR) << "malloc model_data failed.";
    return OH_AI_STATUS_LITE_NULLPTR;
  }
  *data_size = buffer.DataSize();
  memcpy(*model_data, data, buffer.DataSize());
  if (!ret.IsOk()) {
    MS_LOG(ERROR) << "export model fail, ret :" << ret;
  }
  return static_cast<OH_AI_Status>(ret.StatusCode());
}

OH_AI_Status OH_AI_ExportWeightsCollaborateWithMicro(OH_AI_ModelHandle model, OH_AI_ModelType model_type, const char *weight_file,
                                             bool is_inference, bool enable_fp16, char **changeable_weights_name,
                                             size_t num) {
  if (model == nullptr) {
    MS_LOG(ERROR) << "model is nullptr.";
    return OH_AI_STATUS_LITE_PARAM_INVALID;
  }
  auto impl = static_cast<mindspore::ModelC *>(model);
  std::vector<std::string> weights_name;
  for (size_t i = 0; i < num; i++) {
    weights_name.push_back(changeable_weights_name[i]);
  }
  auto ret = mindspore::Serialization::ExportWeightsCollaborateWithMicro(
    *(impl->model_.get()), static_cast<mindspore::ModelType>(model_type), weight_file, is_inference, enable_fp16,
    weights_name);
  if (!ret.IsOk()) {
    MS_LOG(ERROR) << "export model fail, ret :" << ret;
  }
  return static_cast<OH_AI_Status>(ret.StatusCode());
}

OH_AI_Status OH_AI_ModelLoadConfig(OH_AI_ModelHandle model, const char *config_file_path) {
  MS_LOG(INFO) << "Start to load config file for ms model";
  if (model == nullptr || config_file_path == nullptr) {
    MS_LOG(ERROR) << "model or config_file_path is nullptr.";
    return OH_AI_STATUS_LITE_NULLPTR;
  }
  MS_LOG(INFO) << "config_file_path: " << config_file_path;

  auto impl = static_cast<mindspore::ModelC *>(model);
  auto ret = impl->model_->LoadConfig(config_file_path);

  if (ret.IsOk()) {
    MS_LOG(INFO) << "Loaded ms model config file successfully";
  } else {
    MS_LOG(ERROR) << "Loaded ms model config file failed, ret: " << ret;
  }
  return static_cast<OH_AI_Status>(ret.StatusCode());
}
