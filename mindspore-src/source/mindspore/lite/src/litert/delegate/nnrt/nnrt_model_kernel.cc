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
#include <include/errorcode.h>
#include "nnrt_model_kernel.h"
#include "nnrt_allocator.h"
#include "litert/cxx_api/tensor/tensor_impl.h"

namespace mindspore{
namespace {
constexpr auto kDynamicDims = "DynamicDims";
}

int NNRTModelKernel::Prepare() {
  auto nnrt_allocator = lite::NNRTAllocator::GetInstance();
  if (nnrt_allocator == nullptr) {
    MS_LOG(ERROR) << "Get NNRTAllocator failed";
    return lite::RET_NULL_PTR;
  }
  nnrt_allocator->SetDeviceId(nnrt_device_info_.device_id_);
  for (size_t i = 0; i < inputs_.size(); i++) {
    inputs_[i].SetAllocator(nnrt_allocator);
  }
  for (size_t i = 0; i < outputs_.size(); i++) {
    outputs_[i].SetAllocator(nnrt_allocator);
  }
  return lite::RET_OK;
}

int NNRTModelKernel::ReSize() {
  const auto &extensions = nnrt_device_info_.extensions_;
  auto iter = std::find_if(extensions.begin(), extensions.end(), [](const lite::Extension &extension) {
    return extension.name == kDynamicDims;
  });
  if (iter != extensions.end() && !iter->value.empty()) {
    return lite::RET_OK;
  }
  MS_LOG(ERROR) << "NNRT only support the resize function when DynamicDims is enabled.";
  return lite::RET_ERROR;
}

int NNRTModelKernel::Execute() {
  MS_CHECK_TRUE_RET(this->outputs().empty() != true, lite::RET_ERROR);
  zero_copy_ = IS_NNRT_ALLOCATOR(this->outputs()[Index0].allocator());

  if (!zero_copy_) {
    FreeNNTensor();
  }
  nn_input_tensors_.clear();
  nn_output_tensors_.clear();
  nn_input_tensor_descs_.clear();
  nn_output_tensor_descs_.clear();

  lite::STATUS ret_val = SetInputs();
  if (ret_val != lite::RET_OK) {
    MS_LOG(ERROR) << "NNRTModelKernel SetInputs failed, STATUS is " << ret_val;
    return ret_val;
  }
  ret_val = SetOutputs();
  if (ret_val != lite::RET_OK) {
    MS_LOG(ERROR) << "NNRTModelKernel SetOutputs failed, STATUS is " << ret_val;
    return ret_val;
  }
  MS_LOG(INFO) << "Running NNRtModel Kernel...";
  OH_NN_ReturnCode ret_code;
  ret_code = OH_NNExecutor_RunSync(oh_nn_executor_, nn_input_tensors_.data(), nn_input_tensors_.size(),
                                   nn_output_tensors_.data(), nn_output_tensors_.size());

  if (ret_code != OH_NN_SUCCESS) {
    MS_LOG(ERROR) << "OH_NNExecutor_RunSync Run failed, OH_NN_ReturnCode = " << ret_code;
    return lite::RET_ERROR;
  }
  MS_LOG(INFO) << "Run NNRtModel Kernel success.";

  return lite::RET_OK;
}

int NNRTModelKernel::SetInputs() {
  if (!zero_copy_) {
    OH_NN_ReturnCode ret{OH_NN_FAILED};
    size_t nn_input_count = 0;
    ret = OH_NNExecutor_GetInputCount(oh_nn_executor_, &nn_input_count);
    if (ret != OH_NN_SUCCESS) {
      MS_LOG(ERROR) << "OH_NNExecutor_GetInputCount failed.";
      return lite::RET_ERROR;
    }
    if (nn_input_count != inputs_.size()) {
      MS_LOG(ERROR) << "input count is not equal between ms and nnrt.";
      return lite::RET_ERROR;
    }
    for (size_t i = 0; i < nn_input_count; i++) {
      NN_TensorDesc *tensor_desc_tmp = OH_NNExecutor_CreateInputTensorDesc(oh_nn_executor_, i);
      if (tensor_desc_tmp == nullptr) {
        MS_LOG(ERROR) << "OH_NNExecutor_CreateInputTensorDesc failed, i = " << i;
        return lite::RET_ERROR;
      }
      nn_input_tensor_descs_.emplace_back(tensor_desc_tmp);
      NN_Tensor *tensor_tmp = OH_NNTensor_Create(nnrt_device_info_.device_id_, tensor_desc_tmp);
      if (tensor_tmp == nullptr) {
        MS_LOG(ERROR) << "OH_NNTensor_Create input failed, i = " << i;
        return lite::RET_ERROR;
      }
      nn_input_tensors_.emplace_back(tensor_tmp);
      void *nn_data = OH_NNTensor_GetDataBuffer(nn_input_tensors_[i]);
      size_t tensor_size;
      ret = OH_NNTensorDesc_GetByteSize(tensor_desc_tmp, &tensor_size);
      if (ret != OH_NN_SUCCESS || tensor_size != inputs_[i].DataSize()) {
        MS_LOG(ERROR) << "NN_Tensor size is not equal to MSTensor, i = " << i;
        return lite::RET_ERROR;
      }
      memcpy(nn_data, inputs_[i].MutableData(), inputs_[i].DataSize());
    }
  } else {
    for (size_t i = 0; i < inputs_.size(); i++) {
      if (inputs_[i].allocator() == nullptr) {
        MS_LOG(ERROR) << "NNRTAllocator is nullptr, i = " << i;
        return lite::RET_ERROR;
      }
      void *data = inputs_[i].MutableData();
      NN_Tensor *tensor_tmp = reinterpret_cast<lite::NNRTAllocator *>(inputs_[i].allocator().get())->GetNNTensor(data);
      if (tensor_tmp == nullptr) {
        MS_LOG(ERROR) << "NNRTAllocator GetNNTensor failed, i = " << i;
        return lite::RET_ERROR;
      }
      nn_input_tensors_.emplace_back(tensor_tmp);
    }
  }
  return lite::RET_OK;
}

int NNRTModelKernel::SetOutputs() {
  if (!zero_copy_) {
    OH_NN_ReturnCode ret{OH_NN_FAILED};
    size_t nn_output_count = 0;
    ret = OH_NNExecutor_GetOutputCount(oh_nn_executor_, &nn_output_count);
    if (ret != OH_NN_SUCCESS) {
      MS_LOG(ERROR) << "OH_NNExecutor_GetOutputCount failed.";
      return lite::RET_ERROR;
    }
    if (nn_output_count != outputs_.size()) {
      MS_LOG(ERROR) << "output count is not equal between ms and nnrt.";
      return lite::RET_ERROR;
    }
    for (size_t i = 0; i < nn_output_count; i++) {
      NN_TensorDesc *tensor_desc_tmp = OH_NNExecutor_CreateOutputTensorDesc(oh_nn_executor_, i);
      if (tensor_desc_tmp == nullptr) {
        MS_LOG(ERROR) << "OH_NNExecutor_CreateOutputTensorDesc failed, i = " << i;
        return lite::RET_ERROR;
      }
      nn_output_tensor_descs_.emplace_back(tensor_desc_tmp);
      NN_Tensor *tensor_tmp = OH_NNTensor_Create(nnrt_device_info_.device_id_, tensor_desc_tmp);
      if (tensor_tmp == nullptr) {
        MS_LOG(ERROR) << "OH_NNTensor_Create output failed, i = " << i;
        return lite::RET_ERROR;
      }
      nn_output_tensors_.emplace_back(tensor_tmp);
      auto data = OH_NNTensor_GetDataBuffer(nn_output_tensors_[i]);
      reinterpret_cast<LiteTensorImpl *>(outputs_[i].impl().get())->lite_tensor()->FreeData();
      outputs_[i].SetData(data, false);
    }
  } else {
    for (size_t i = 0; i < outputs_.size(); i++) {
      if (outputs_[i].allocator() == nullptr) {
        MS_LOG(ERROR) << "NNRTAllocator is nullptr, i = " << i;
        return lite::RET_ERROR;
      }
      void *data = outputs_[i].MutableData();
      NN_Tensor *tensor_tmp = reinterpret_cast<lite::NNRTAllocator *>(outputs_[i].allocator().get())->GetNNTensor(data);
      if (tensor_tmp == nullptr) {
        MS_LOG(ERROR) << "NNRTAllocator GetNNTensor failed, i = " << i;
        return lite::RET_ERROR;
      }
      nn_output_tensors_.emplace_back(tensor_tmp);
    }
  }
  return lite::RET_OK;
}

void NNRTModelKernel::FreeNNTensor() {
  for (size_t i = 0; i < nn_input_tensors_.size(); i++) {
    OH_NNTensor_Destroy(&nn_input_tensors_[i]);
    OH_NNTensorDesc_Destroy(&nn_input_tensor_descs_[i]);
  }
  for (size_t i = 0; i < nn_output_tensors_.size(); i++) {
    OH_NNTensor_Destroy(&nn_output_tensors_[i]);
    OH_NNTensorDesc_Destroy(&nn_output_tensor_descs_[i]);
  }
}

}  // namespace mindspore
