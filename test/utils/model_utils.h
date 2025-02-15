/*
 * Copyright (c) 2023 Huawei Device Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef OHOS_MINDSPORE_TEST_MODEL_UTILS_H
#define OHOS_MINDSPORE_TEST_MODEL_UTILS_H

#include "gtest/gtest.h"
#include "include/c_api/context_c.h"
#include "include/c_api/model_c.h"
#include "include/c_api/types_c.h"
#include "include/c_api/status_c.h"
#include "include/c_api/data_type_c.h"
#include "include/c_api/tensor_c.h"
#include "include/c_api/format_c.h"

// function before callback
bool PrintBeforeCallback(const OH_AI_TensorHandleArray inputs, const OH_AI_TensorHandleArray outputs,
                         const OH_AI_CallBackParam kernelInfo);
// function after callback
bool PrintAfterCallback(const OH_AI_TensorHandleArray inputs, const OH_AI_TensorHandleArray outputs,
                        const OH_AI_CallBackParam kernelInfo);
// add cpu device info
void AddContextDeviceCPU(OH_AI_ContextHandle context);
bool IsNPU();
// add nnrt device info
void AddContextDeviceNNRT(OH_AI_ContextHandle context);
// fill data to inputs tensor
void FillInputsData(OH_AI_TensorHandleArray inputs, std::string modelName, bool isTranspose);
// compare result after predict
void CompareResult(OH_AI_TensorHandleArray outputs, std::string modelName, float atol = 0.01, float rtol = 0.01);
// model build and predict
void ModelPredict(OH_AI_ModelHandle model, OH_AI_ContextHandle context, std::string modelName,
                  OH_AI_ShapeInfo shapeInfos, bool buildByGraph, bool isTranspose, bool isCallback);
void ModelPredict_ModelBuild(OH_AI_ModelHandle model, OH_AI_ContextHandle context, std::string modelName,
                             bool buildByGraph);
void ModelTrain(OH_AI_ModelHandle model, OH_AI_ContextHandle context, std::string modelName,
                OH_AI_ShapeInfo shapeInfos, bool buildByGraph, bool isTranspose, bool isCallback);

#endif //OHOS_MINDSPORE_TEST_MODEL_UTILS_H
