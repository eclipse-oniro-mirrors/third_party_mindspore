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

#include "coder/generator/component/const_blocks/msession.h"

namespace mindspore::lite::micro {
const char model_runtime_other_source[] = R"RAW(
OH_AI_TensorHandleArray OH_AI_ModelGetInputs(const OH_AI_ModelHandle model) {
  MicroModel *micro_model = (MicroModel *)model;
  if (micro_model == NULL) {
    OH_AI_TensorHandleArray tmp = {0, NULL};
    return tmp;
  }
  return micro_model->inputs;
}

OH_AI_TensorHandleArray OH_AI_ModelGetOutputs(const OH_AI_ModelHandle model) {
  MicroModel *micro_model = (MicroModel *)model;
  if (micro_model == NULL) {
    OH_AI_TensorHandleArray tmp = {0, NULL};
    return tmp;
  }
  return micro_model->outputs;
}

OH_AI_TensorHandle OH_AI_ModelGetInputByTensorName(const OH_AI_ModelHandle model, const char *tensor_name) {
  MicroModel *micro_model = (MicroModel *)model;
  if (micro_model == NULL || micro_model->inputs.handle_list == NULL) {
    return NULL;
  }
  for (size_t i = 0; i < micro_model->inputs.handle_num; i++) {
    MicroTensor *micro_tensor = (MicroTensor *)micro_model->inputs.handle_list[i];
    if (micro_tensor == NULL) {
      return NULL;
    }
    if (strcmp(micro_tensor->name, tensor_name) == 0) {
      return micro_tensor;
    }
  }
  return NULL;
}

OH_AI_TensorHandle OH_AI_ModelGetOutputByTensorName(const OH_AI_ModelHandle model, const char *tensor_name) {
  MicroModel *micro_model = (MicroModel *)model;
  if (micro_model == NULL || micro_model->outputs.handle_list == NULL) {
    return NULL;
  }
  for (size_t i = 0; i < micro_model->outputs.handle_num; i++) {
    MicroTensor *micro_tensor = (MicroTensor *)micro_model->outputs.handle_list[i];
    if (micro_tensor == NULL) {
      return NULL;
    }
    if (strcmp(micro_tensor->name, tensor_name) == 0) {
      return micro_tensor;
    }
  }
  return NULL;
}

OH_AI_Status OH_AI_ModelResize(OH_AI_ModelHandle model, const OH_AI_TensorHandleArray inputs, OH_AI_ShapeInfo *shape_infos,
                       size_t shape_info_num) {
  MicroModel *micro_model = (MicroModel *)model;
  if (micro_model == NULL) {
    return OH_AI_STATUS_LITE_NULLPTR;
  }
  if (micro_model->resize == NULL) {
    return OH_AI_STATUS_LITE_NULLPTR;
  }
  return micro_model->resize(model, inputs, shape_infos, shape_info_num);
}

)RAW";
}  // namespace mindspore::lite::micro
