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
#include "src/common/utils.h"
#include "src/litert/delegate/nnrt/nnrt_wrapper.h"
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
  uint64_t start_exec_nnrt = mindspore::lite::GetTimeUs();
  MS_LOG(DEBUG) << "Start to execute the NNRT model kernel";
  MS_CHECK_TRUE_RET(this->outputs().empty() != true, lite::RET_ERROR);
  zero_copy_ = IS_NNRT_ALLOCATOR(this->outputs()[Index0].allocator());

  if (!zero_copy_) {
    FreeNNTensor();
  }
  nn_input_tensors_.clear();
  nn_output_tensors_.clear();
  nn_input_tensor_descs_.clear();
  nn_output_tensor_descs_.clear();
  MS_LOG(DEBUG) << "Start to set inputs of the NNRT model";
  lite::STATUS ret_val = SetInputs();
  if (ret_val != lite::RET_OK) {
    MS_LOG(ERROR) << "NNRTModelKernel SetInputs failed, STATUS is " << ret_val;
    return ret_val;
  }
  MS_LOG(DEBUG) << "Start to set outputs of the NNRT model";
  ret_val = SetOutputs();
  if (ret_val != lite::RET_OK) {
    MS_LOG(ERROR) << "NNRTModelKernel SetOutputs failed, STATUS is " << ret_val;
    return ret_val;
  }
  MS_LOG(DEBUG) << "Running NNRtModel Kernel...";
  OH_NN_ReturnCode ret_code = OH_NN_SUCCESS;
  auto& nnrtWrapper = mindspore::NNRTWrapper::GetInstance();
  if (predict_config_ != nullptr) {
    bool support = false;
    auto ret_code = nnrtWrapper.IsSupportAIPP(support);
    if (ret_code != OH_NN_SUCCESS) {
      MS_LOG(ERROR) << "Aipp not support in current device";
      return lite::RET_NOT_SUPPORT_AIPP;
    }

    ret_code = nnrtWrapper.ExecutorRunSyncWithAipp(oh_nn_executor_, nn_input_tensors_.data(), nn_input_tensors_.size(),
                                                    nn_output_tensors_.data(), nn_output_tensors_.size(), predict_config_);
    if (ret_code == OH_NN_UNAVAILABLE_DEVICE) {
      MS_LOG(ERROR) << "OH_NNExcutor_RunSync run failed, OH_NN_ReturnCode = " << ret_code;
      return lite::RET_OP_EXECUTE_FAILURE;
    }
    if (ret_code != OH_NN_SUCCESS) {
      MS_LOG(ERROR) << "Inference failed when Aipp invoked the NNRT interface";
      return lite::RET_INFER_AIPP_FAIL;
    }
  } else {

    ret_code = nnrtWrapper.NNExecutorRunSync(oh_nn_executor_, nn_input_tensors_.data(), nn_input_tensors_.size(),
                                              nn_output_tensors_.data(), nn_output_tensors_.size());
  }

  if (ret_code == OH_NN_UNAVAILABLE_DEVICE) {
    MS_LOG(ERROR) << "OH_NNExcutor_RunSync run failed, OH_NN_ReturnCode = " << ret_code;
    return lite::RET_OP_EXECUTE_FAILURE;
  }

  if (ret_code != OH_NN_SUCCESS) {
    MS_LOG(ERROR) << "OH_NNExecutor_RunSync Run failed, OH_NN_ReturnCode = " << ret_code;
    return lite::RET_ERROR;
  }
  MS_LOG(DEBUG) << "Run NNRtModel Kernel success";
  uint64_t nnrt_exec_time = mindspore::lite::GetTimeUs() - start_exec_nnrt;
  MS_LOG(DEBUG) << "The NNRT inference time of the Lite model  is: " << nnrt_exec_time << "us";
  return lite::RET_OK;
}

int NNRTModelKernel::SetInputs() {
  if (!zero_copy_) {
    OH_NN_ReturnCode ret{OH_NN_FAILED};
    size_t nn_input_count = 0;
    auto& nnrtWrapper = mindspore::NNRTWrapper::GetInstance();
    nnrtWrapper.NNExecutorGetInputCount(oh_nn_executor_, &nn_input_count);
    if (ret != OH_NN_SUCCESS) {
      MS_LOG(ERROR) << "OH_NNExecutor_GetInputCount failed.";
      return lite::RET_ERROR;
    }
    if (nn_input_count != inputs_.size()) {
      MS_LOG(ERROR) << "input count is not equal between ms and nnrt.";
      return lite::RET_ERROR;
    }
    for (size_t i = 0; i < nn_input_count; i++) {

      NN_TensorDesc *tensor_desc_tmp = nnrtWrapper.NNExecutorCreateInputTensorDesc(oh_nn_executor_, i);
      if (tensor_desc_tmp == nullptr) {
        MS_LOG(ERROR) << "OH_NNExecutor_CreateInputTensorDesc failed, i = " << i;
        return lite::RET_ERROR;
      }
      nn_input_tensor_descs_.emplace_back(tensor_desc_tmp);

      NN_Tensor *tensor_tmp = nnrtWrapper.NNTensorCreate(nnrt_device_info_.device_id_, tensor_desc_tmp);
      if (tensor_tmp == nullptr) {
        MS_LOG(ERROR) << "OH_NNTensor_Create input failed, i = " << i;
        
        return lite::RET_ERROR;
      }
      nn_input_tensors_.emplace_back(tensor_tmp);

      void *nn_data = nnrtWrapper.NNTensorGetDataBuffer(nn_input_tensors_[i]);
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
    MS_LOG(DEBUG) << "Enable NNRTAllocator for input tensor success.";
  }
  return lite::RET_OK;
}

int NNRTModelKernel::SetOutputs() {
  if (!zero_copy_) {
    OH_NN_ReturnCode ret{OH_NN_FAILED};
    size_t nn_output_count = 0;
    auto& nnrtWrapper = mindspore::NNRTWrapper::GetInstance();
    ret = nnrtWrapper.NNExecutorGetOutputCount(oh_nn_executor_, &nn_output_count);
    if (ret != OH_NN_SUCCESS) {
      MS_LOG(ERROR) << "OH_NNExecutor_GetOutputCount failed.";
      return lite::RET_ERROR;
    }
    if (nn_output_count != outputs_.size()) {
      MS_LOG(ERROR) << "output count is not equal between ms and nnrt.";
      return lite::RET_ERROR;
    }
    
    for (size_t i = 0; i < nn_output_count; i++) {

      NN_TensorDesc *tensor_desc_tmp = nnrtWrapper.NNExecutorCreateOutputTensorDesc(oh_nn_executor_, i);
      if (tensor_desc_tmp == nullptr) {
        MS_LOG(ERROR) << "OH_NNExecutor_CreateOutputTensorDesc failed, i = " << i;
        return lite::RET_ERROR;
      }
      nn_output_tensor_descs_.emplace_back(tensor_desc_tmp);

      NN_Tensor *tensor_tmp = nnrtWrapper.NNTensorCreate(nnrt_device_info_.device_id_, tensor_desc_tmp);
      if (tensor_tmp == nullptr) {
        MS_LOG(ERROR) << "OH_NNTensor_Create output failed, i = " << i;
        return lite::RET_ERROR;
      }
      nn_output_tensors_.emplace_back(tensor_tmp);

      auto data = nnrtWrapper.NNTensorGetDataBuffer(nn_output_tensors_[i]);
      reinterpret_cast<LiteTensorImpl *>(outputs_[i].impl().get())->lite_tensor()->FreeData();
      outputs_[i].SetData(data, false);
    }
    
  } else {
    for (size_t i = 0; i < outputs_.size(); i++) {
      if (outputs_[i].allocator() == nullptr) {
        MS_LOG(ERROR) << "NNRTAllocator is nullptr, i = " << i;
        return lite::RET_ERROR;
      }

      OH_NN_DataType npuDataType;
      auto &nnrtWrapper = mindspore::NNRTWrapper::GetInstance();
      NN_TensorDesc *tensor_desc_tmp = nnrtWrapper.NNExecutorCreateOutputTensorDesc(oh_nn_executor_, i);
      if (tensor_desc_tmp == nullptr) {
        MS_LOG(ERROR) << "tensor_desc_tmp==nullptr,i=" << i;
        return lite::RET_ERROR;
      }

      OH_NN_ReturnCode ret = nnrtWrapper.OH_NNTensorDesc_GetDataType(tensor_desc_tmp, &npuDataType);
      if (ret != OH_NN_SUCCESS) {
        MS_LOG(ERROR) << "OH_NNTensorDesc_GetDataType failed, i=" << i;
        nnrtWrapper.NNTensorDescDestroy(&tensor_desc_tmp);
        return lite::RET_ERROR;
      }

      if (npuDataType == OH_NN_FLOAT32) {
        if (outputs_[i].DataType() == static_cast<mindspore::DataType>(mindspore::kNumberTypeFloat16)) {
          // need reset ms model output to fp32
          outputs_[i].SetDataType(static_cast<mindspore::DataType>(mindspore::kNumberTypeFloat32));
        }
      }

      nnrtWrapper.NNTensorDescDestroy(&tensor_desc_tmp);
      tensor_desc_tmp = nullptr;

      void *data = outputs_[i].MutableData();
      NN_Tensor *tensor_tmp = reinterpret_cast<lite::NNRTAllocator *>(outputs_[i].allocator().get())->GetNNTensor(data);
      if (tensor_tmp == nullptr) {
        MS_LOG(ERROR) << "NNRTAllocator GetNNTensor failed, i = " << i;
        return lite::RET_ERROR;
      }
      nn_output_tensors_.emplace_back(tensor_tmp);
    }
    MS_LOG(DEBUG) << "Enable NNRTAllocator for output tensor success.";
  }
  return lite::RET_OK;
}

void NNRTModelKernel::FreeNNTensor() {
  auto& nnrtWrapper = mindspore::NNRTWrapper::GetInstance();
  for (size_t i = 0; i < nn_input_tensors_.size(); i++) {
    nnrtWrapper.NNTensorDestroy(&nn_input_tensors_[i]);
    nnrtWrapper.NNTensorDescDestroy(&nn_input_tensor_descs_[i]);
  }
  for (size_t i = 0; i < nn_output_tensors_.size(); i++) {
    nnrtWrapper.NNTensorDestroy(&nn_output_tensors_[i]);
    nnrtWrapper.NNTensorDescDestroy(&nn_output_tensor_descs_[i]);
  }
  
}

}  // namespace mindspore
