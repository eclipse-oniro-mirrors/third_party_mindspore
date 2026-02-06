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

#include "cache_session.h"
#include "src/common/context_util.h"
#include "src/common/tensor_util.h"
#include "src/common/mmap_utils.h"
#include "src/common/file_utils.h"
#include "src/litert/delegate/nnrt/nnrt_model_kernel.h"

namespace mindspore {
namespace lite {
CacheSession::~CacheSession() {
  if (nn_executor_ != nullptr) {
    OH_NNExecutor_Destroy(&nn_executor_);
    MS_LOG(INFO) << "Destroy NNExecutor Finish.";
  }
}

int CacheSession::CompileGraph(Model *model) {
  bool expected = false;
  if (!is_running_.compare_exchange_strong(expected, true)) {
    MS_LOG(ERROR) << "Not support multi-threading";
    return RET_ERROR;
  }
  // Convert to abstract base model interface
  auto ret = ConvertInOutTensors(model);
  context_->set_schema_version(reinterpret_cast<LiteModel *>(model)->GetSchemaVersion());
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvertTensors failed: " << ret;
    is_running_.store(false);
    return ret;
  }
  InitGraphInputTensors(model);
  InitGraphOutputTensors(model);

  // create NNRt kernel
  ret = ScheduleToNNRTKernel();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Schedule NNRt kernel failed: " << ret;
    is_running_.store(false);
    return ret;
  }

  InitGraphInOutTensorsMap(model);
  ret = PrepareKernels(model);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare kernels failed: " << ret;
    is_running_.store(false);
    return ret;
  }

  ret = InitExecutor();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "InitExecutor failed: " << ret;
    is_running_.store(false);
    return ret;
  }

  MarkSharedWeight(kernels_);
  FreePackOpWeight(kernels_);

  is_running_.store(false);
  return RET_OK;
}

int CacheSession::InitExecutor() {
  executor_ = new (std::nothrow) Executor();
  if (executor_ == nullptr) {
    MS_LOG(ERROR) << "New Executor failed";
    return RET_ERROR;
  }
  auto ret = executor_->Prepare(kernels_, inputs_, outputs_, context_.get());
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare executor failed: " << ret;
    return ret;
  }
  return RET_OK;
}

int CacheSession::ConvertInOutTensors(const lite::Model *model) {
  MS_ASSERT(model != nullptr);
  auto lite_model = reinterpret_cast<const lite::LiteModel *>(model);
  uint32_t tensor_count = model->graph_.all_tensors_.size();
  auto model_input_indices = model->graph_.input_indices_;
  auto model_output_indices = model->graph_.output_indices_;

  for (uint32_t i = 0; i < tensor_count; ++i) {
    auto *src_tensor = model->graph_.all_tensors_[i];
    if (!IsContain(model_input_indices, i) && !IsContain(model_output_indices, i)) {
      this->tensors_.emplace_back(nullptr);
      continue;
    }
    if (src_tensor == nullptr) {
      MS_LOG(ERROR) << i << "th tensor in model is nullptr";
      return RET_NULL_PTR;
    }
    auto *dst_tensor = ConvertTensor(*src_tensor);
    if (dst_tensor == nullptr) {
      MS_LOG(ERROR) << "Convert new " << i << "th tensor failed!";
      return RET_NULL_PTR;
    }
    auto ret = ConvertTensorsData(lite_model, i, dst_tensor);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Convert data of " << i << "th tensor failed";
      delete dst_tensor;
      return ret;
    }
    ConvertTensorsQuantParam(src_tensor, dst_tensor);
    if (IsContain(model_input_indices, i)) {
      dst_tensor->set_category(Category::GRAPH_INPUT);
    }
    if (IsContain(model_output_indices, i)) {
      // a tensor is as both input and output, would be treated as an input.
      if (!dst_tensor->IsGraphInput()) {
        dst_tensor->set_category(Category::GRAPH_OUTPUT);
      }
    }

    ret = CheckTensorValid(dst_tensor);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Check " << i << "th tensor failed";
      delete dst_tensor;
      return ret;
    }

    this->tensors_.emplace_back(dst_tensor);
  }
  return RET_OK;
}

int CacheSession::Init(const std::shared_ptr<InnerContext> &context) {
  if (context == nullptr) {
    MS_LOG(ERROR) << "context is nullptr";
    return RET_NULL_PTR;
  }
  bool expected = false;
  if (!is_running_.compare_exchange_strong(expected, true)) {
    MS_LOG(ERROR) << "Not support multi-threading";
    return RET_ERROR;
  }
  context_ = context;
  auto ret = context_->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init Context failed";
    return ret;
  }
  ms_context_ = MSContextFromContext(context);
  if (ms_context_ == nullptr) {
    MS_LOG(ERROR) << "transfer context to ms context failed.";
    return RET_NULL_PTR;
  }

  auto iter = std::find_if(context_->device_list_.begin(), context_->device_list_.end(),
                           [](DeviceContext &device) { return device.device_type_ == lite::DT_NNRT; });
  if(iter == context_->device_list_.end()) {
    MS_LOG(ERROR) << "Found non NNRT device info";
    return RET_ERROR;
  }
  nnrt_device_info_ = iter->device_info_.nnrt_device_info_;

  const auto &extensions = nnrt_device_info_.extensions_;
  mindspore::lite::nnrt::ExtensionOptionsParser::Parse(extensions, &extension_options_);

  is_running_.store(false);
  return RET_OK;
}

int CacheSession::ParseInputOutputFromModelBuffer(const char *model_buf, LiteModel *model) {
  const void *meta_graph = nullptr;
  meta_graph = reinterpret_cast<const void *>(schema::GetMetaGraph(model_buf));
  assert(meta_graph != nullptr);

  auto status = GenerateModelInputOutput<schema::MetaGraph, schema::CNode>(
    *reinterpret_cast<const schema::MetaGraph *>(meta_graph), model->graph_);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "fail to generate model";
    return status;
  }
  model->buf = const_cast<char *>(model_buf);
  return RET_OK;
}

int CacheSession::LoadModelAndCompileByPath(const std::string &model_path, mindspore::ModelType model_type) {
  size_t model_size;
  bool use_mmap = IsMmapEnable();
  auto model_buf = LoadModelByPath(model_path, model_type, &model_size, use_mmap);
  if (model_buf == nullptr) {
    MS_LOG(ERROR) << "Read model file failed";
    return RET_ERROR;
  }

  Model *model = nullptr;
  if (extension_options_.cache_path_.empty()) {
    MS_LOG(ERROR) << "cache path is empty";
    return RET_ERROR;
  } else {
    model = ImportInOutFromBuffer(model_buf, model_size, true, model_type, model_path);
    if (model == nullptr) {
      MS_LOG(ERROR) << "Import model failed";
      return RET_ERROR;
    }
    dynamic_cast<LiteModel *>(model)->PrepareInnerTensors();
  }
  if (model == nullptr) {
    MS_LOG(ERROR) << "Import model failed";
    return RET_ERROR;
  }

  if (use_mmap) {
    reinterpret_cast<lite::LiteModel *>(model)->model_buf_by_mmap_ = true;
  } else {
    MS_LOG(WARNING) << "Memory may exceed the limit of business demands.";
  }
  (reinterpret_cast<lite::LiteModel *>(model))->set_keep_model_buf(true);
  auto ret = CompileGraph(model);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Compile model failed";
    model->buf = nullptr;
    delete model;
    return RET_ERROR;
  }
  set_model(model);
  return RET_OK;
}

Model *CacheSession::ImportInOutFromBuffer(const char *model_buf, size_t size, bool take_buf, mindspore::ModelType model_type,
                               const std::string &path) {
  MS_LOG(INFO) << "import model from lite model";
  auto *model = new (std::nothrow) LiteModel(path);
  if (model == nullptr) {
    MS_LOG(ERROR) << "new model fail!";
    return nullptr;
  }

  auto status = ParseInputOutputFromModelBuffer(model_buf, model);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "construct model failed.";
    delete model;
    return nullptr;
  }
  model->buf = const_cast<char *>(model_buf);
  model->buf_size_ = size;
  return model;
}

int CacheSession::ScheduleToNNRTKernel() {
  if (!IsKirinNPUWithOnlineInference(nnrt_device_info_.device_id_)) {
    MS_LOG(ERROR) << "only support NPU_ device.";
    return RET_ERROR;
  }
  auto ret = CreateFullModelKernel();
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Build npu model failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

bool CacheSession::IsKirinNPUWithOnlineInference(size_t device_id) {
  const std::string kirin_npu_name_prefix = "NPU_";
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

Status CacheSession::CreateFullModelKernel() {
  OH_NNCompilation* nn_compilation = OH_NNCompilation_ConstructForCache();
  if (nn_compilation == nullptr) {
    MS_LOG(ERROR) << "Construct NNCompilation failed";
    return kLiteError;
  }
  MS_LOG(DEBUG) << "NNRTDelegate creates NNCompilation success.";

  auto ret_code = InitNNCompilation(nn_compilation);
  if (ret_code != kSuccess) {
    MS_LOG(ERROR) << "Init NNCompilation failed";
    OH_NNCompilation_Destroy(&nn_compilation);
    return kLiteError;
  }

  OH_NNExecutor *nn_executor = nullptr;
  nn_executor = OH_NNExecutor_Construct(nn_compilation);
  if (nn_executor == nullptr) {
    MS_LOG(ERROR) << "Construct NNExecutor failed, ret: " << ret_code;
    OH_NNCompilation_Destroy(&nn_compilation);
    return kLiteError;
  }
  OH_NNCompilation_Destroy(&nn_compilation);

  ms_inputs_ = LiteTensorsToMSTensors(inputs_);
  ms_outputs_ = LiteTensorsToMSTensors(outputs_);
  auto nnrt_model_kernel = new (std::nothrow) NNRTModelKernel(nn_executor, nnrt_device_info_, ms_inputs_, ms_outputs_);
  if (nnrt_model_kernel == nullptr) {
    OH_NNExecutor_Destroy(&nn_executor);
    MS_LOG(ERROR) << "new NNRTModelKernel failed";
    return kLiteError;
  }
  nn_executor_ = nn_executor;

  std::shared_ptr<kernel::Kernel> shared_kernel(nnrt_model_kernel);
  auto *kernel_exec = new (std::nothrow) kernel::KernelExec(shared_kernel);
  if (kernel_exec == nullptr) {
    MS_LOG(ERROR) << "nnrt kernel exec create failed.";
    return kLiteError;
  }
  auto delegate_type = kNumberTypeFloat32;
  for (auto &input : nnrt_model_kernel->inputs()) {
    if (static_cast<TypeId>(input.DataType()) == kNumberTypeFloat16) {
      delegate_type = kNumberTypeFloat16;
      break;
    }
  }
  kernel::KernelKey delegate_desc{kernel::kDelegate, delegate_type, NHWC, schema::PrimitiveType_NONE, "", ""};
  kernel_exec->set_desc(delegate_desc);
  kernel_exec->set_context(context_.get());
  kernels_.push_back(kernel_exec);

  return kSuccess;
}

Status CacheSession::InitNNCompilation(OH_NNCompilation *nn_compilation) const {
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

  if (!extension_options_.cache_path_.empty()) {
    ret_code = OH_NNCompilation_SetCache(nn_compilation, extension_options_.cache_path_.c_str(),
                                         extension_options_.cache_version_);
    if ((ret_code != OH_NN_SUCCESS) && (ret_code != OH_NN_OPERATION_FORBIDDEN)) {
      MS_LOG(ERROR) << "NNCompilation set cache failed, ret: " << ret_code;
      return kLiteError;
    }
  } else {
    MS_LOG(ERROR) << "NNCompilation must set Cache.";
    return kLiteError;
  }

  size_t extension_size = nnrt_device_info_.extensions_.size();
  for (size_t i = 0; i < extension_size; i++) {
    auto &src_extensoin = nnrt_device_info_.extensions_[i];
    ret_code = OH_NNCompilation_AddExtensionConfig(nn_compilation, src_extensoin.name.c_str(),
                                                   (char *)((void *)src_extensoin.value.data()),
                                                   src_extensoin.value.size());
    if (ret_code != OH_NN_SUCCESS) {
      MS_LOG(ERROR) << "OH_NNCompilation_AddExtensionConfig " << i << ": "<< src_extensoin.name << " failed, ret: "
                    << ret_code;
      return kLiteError;
    }
  }

  ret_code = OH_NNCompilation_Build(nn_compilation);
  if (ret_code != OH_NN_SUCCESS) {
    MS_LOG(ERROR) << "Build NNCompilation failed, ret: " << ret_code;
    return kLiteError;
  }
  return kSuccess;
}

const char *CacheSession::LoadModelByPath(const std::string &file, mindspore::ModelType model_type, size_t *size, bool use_mmap) {
  size_t buf_size;
  char *model_buf;
  if (use_mmap) {
    model_buf = reinterpret_cast<char *>(lite::ReadFileByMmap(file.c_str(), &buf_size, false));
  } else {
    MS_LOG(WARNING) << "Memory may exceed the limit of business demands.";
    model_buf = lite::ReadFile(file.c_str(), &buf_size);
  }
  if (model_buf == nullptr) {
    MS_LOG(ERROR) << "The model path is invalid";
    return model_buf;
  }

  char *lite_buf = nullptr;
  auto buf_model_type = LoadModelByBuff(model_buf, buf_size, &lite_buf, size, model_type);
  if (buf_model_type == mindspore::ModelType::kUnknownType || lite_buf == nullptr) {
    if (use_mmap) {
      lite::UnmapMmapBuffer(const_cast<void *>(static_cast<const void *>(model_buf)), buf_size);
    } else {
      delete[] model_buf;
    }
    model_buf = nullptr;
    return nullptr;
  }

  return lite_buf;
}
}  // namespace lite
}  // namespace mindspore
