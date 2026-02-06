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
#include "coder/opcoders/nnacl/fp32/activation_dynamic_fp32_coder.h"
#include <string>
#include "nnacl/fp32/activation_fp32.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/parallel.h"
#include "tools/common/string_util.h"
#include "coder/utils/coder_utils.h"

using mindspore::schema::PrimitiveType_Activation;

namespace mindspore::lite::micro::nnacl {
int ActivationDynamicFP32Coder::Preprocess() {
  // attribute
  auto in_shape = shape_info_container_->GetTemplateShape(input_tensor_);
  int64_t const_part = 1;
  std::string non_const_part;
  for (const auto &item : in_shape) {
    if (IsNumber(item)) {
      const_part *= std::atoi(item.c_str());
    } else {
      if (!non_const_part.empty()) {
        non_const_part += " * ";
      }
      non_const_part += item;
    }
  }
  count_ = std::to_string(const_part) + " * " + non_const_part;
  input_data_ = dynamic_mem_manager_->GetVarTensorAddr(input_tensor_);
  MS_CHECK_TRUE_MSG(!input_data_.empty(), RET_ERROR, "pointer is not allocated by the allocator");
  output_data_ = dynamic_mem_manager_->GetVarTensorAddr(output_tensor_);
  MS_CHECK_TRUE_MSG(!output_data_.empty(), RET_ERROR, "pointer is not allocated by the allocator");
  return RET_OK;
}

int ActivationDynamicFP32Coder::DoCode(CoderContext *const context) {
  Collect(context,
          {
            "wrapper/fp32/activation_fp32_wrapper.h",
            "nnacl/fp32/activation_fp32.h",
          },
          {
            "activation_fp32_wrapper.c",
            "activation_fp32.c",
          });
  NNaclFp32Serializer code;
  auto *activation_parameter = reinterpret_cast<ActivationParameter *>(parameter_);
  int ret = Preprocess();
  MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "Preprocess failed");

  switch (activation_parameter->type_) {
    case schema::ActivationType_RELU:
      code.CodeFunction("Fp32Relu", input_data_, count_, output_data_);
      break;
    case schema::ActivationType_RELU6:
      code.CodeFunction("Fp32Relu6", input_data_, count_, output_data_);
      break;
    case schema::ActivationType_LEAKY_RELU:
      code.CodeFunction("LRelu", input_data_, count_, output_data_, activation_parameter->alpha_);
      break;
    case schema::ActivationType_SIGMOID:
      if (!support_parallel_) {
        code.CodeFunction("Sigmoid", input_data_, count_, output_data_);
      } else {
        code.CodeStruct("activation_param", *activation_parameter);
        code.CodeBaseStruct("ActivationFp32Args", kRunArgs, input_data_, count_, output_data_, 0.0f,
                            "&activation_param");
        code.CodeFunction(kParallelLaunch, "DoSigmoid", kRunArgsAddr, "activation_param.op_parameter_.thread_num_");
      }
      break;
    case schema::ActivationType_TANH:
      code.CodeFunction("Tanh", input_data_, count_, output_data_);
      break;
    case schema::ActivationType_HSWISH:
      code.CodeFunction("HSwish", input_data_, count_, output_data_);
      break;
    case schema::ActivationType_SWISH:
      code.CodeFunction("Swish", input_data_, count_, output_data_);
      break;
    case schema::ActivationType_HSIGMOID:
      code.CodeFunction("HSigmoid", input_data_, count_, output_data_);
      break;
    case schema::ActivationType_ELU:
      code.CodeFunction("Elu", input_data_, count_, output_data_, activation_parameter->alpha_);
      break;
    default:
      MS_LOG(ERROR) << "Activation type error";
      return RET_ERROR;
  }
  MS_LOG(DEBUG) << "ActivationFP32Code has been called";
  context->AppendCode(code.str());
  return lite::RET_OK;
}

REG_DYNAMIC_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_Activation,
                           CPUOpCoderCreator<ActivationDynamicFP32Coder>)
}  // namespace mindspore::lite::micro::nnacl
