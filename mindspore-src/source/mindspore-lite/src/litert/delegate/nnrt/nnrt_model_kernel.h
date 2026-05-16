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
#ifndef LITE_NNRT_MODEL_KERNEL_H
#define LITE_NNRT_MODEL_KERNEL_H
#include <vector>
#include <queue>
#include <map>
#include <utility>
#include "include/api/kernel.h"
#include "src/litert/delegate/nnrt/neural_network_runtime.h"
#include "src/common/log_adapter.h"
#include "src/litert/inner_context.h"
#include "include/errorcode.h"

namespace mindspore {

class NNRTModelKernel : public kernel::Kernel {
  /**
   * Because nnr can't run single op, but the whole model. So we decide to make the whole model into one kernel.
   * */
 public:
  NNRTModelKernel(OH_NNExecutor *oh_nn_executor, lite::NNRtDeviceInfo nnrt_device_info, const std::vector<mindspore::MSTensor> &inputs,
                  const std::vector<mindspore::MSTensor> &outputs)
      : kernel::Kernel(inputs, outputs, nullptr, nullptr), oh_nn_executor_(oh_nn_executor), nnrt_device_info_(nnrt_device_info) {}
  int Prepare() override;
  int Execute() override;
  int ReSize() override;
  int SetInputs();
  int SetOutputs();
  void FreeNNTensor();
  ~NNRTModelKernel() override {
    if (!zero_copy_) {
      FreeNNTensor();
    }
    MS_LOG(DEBUG) << "NNRTModelKernel Destroy.";
  }

 protected:
  OH_NNExecutor *oh_nn_executor_ = nullptr;
  lite::NNRtDeviceInfo nnrt_device_info_;
  std::vector<NN_Tensor *> nn_input_tensors_;
  std::vector<NN_TensorDesc *> nn_input_tensor_descs_;
  std::vector<NN_Tensor *> nn_output_tensors_;
  std::vector<NN_TensorDesc *> nn_output_tensor_descs_;

 private:
  bool zero_copy_{false};
};
}  // namespace mindspore

#endif  // LITE_NNRTT_MODEL_KERNEL_H
