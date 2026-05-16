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

#include "neural_network_runtime/neural_network_runtime.h"
#include "neural_network_runtime_inner.h"

OH_NNModel *OH_NNModel_Construct(void) {
  return NULL;
}

OH_NN_ReturnCode OH_NNExecutor_Run(OH_NNExecutor *executor) {
  return OH_NN_SUCCESS;
}

OH_NN_ReturnCode OH_NNCompilation_Build(OH_NNCompilation *compilation) {
  return OH_NN_SUCCESS;
}

void OH_NNCompilation_Destroy(OH_NNCompilation **compilation) {}

OH_NNExecutor *OH_NNExecutor_Construct(OH_NNCompilation *compilation) {
  return NULL;
}

void OH_NNExecutor_Destroy(OH_NNExecutor **executor) {}

OH_NNCompilation *OH_NNCompilation_Construct(const OH_NNModel *model) {
  return NULL;
}

OH_NN_ReturnCode OH_NNDevice_GetAllDevicesID(const size_t **allDevicesID, uint32_t *deviceCount) {
  return OH_NN_SUCCESS;
}

OH_NN_ReturnCode OH_NNExecutor_SetOutput(OH_NNExecutor *executor,
                                         uint32_t outputIndex,
                                         void *dataBuffer,
                                         size_t length) {
  return OH_NN_SUCCESS;
}

OH_NN_ReturnCode OH_NNCompilation_SetDevice(OH_NNCompilation *compilation, size_t deviceID) {
  return OH_NN_SUCCESS;
}

OH_NN_ReturnCode OH_NNExecutor_SetInput(OH_NNExecutor *executor,
                                        uint32_t inputIndex,
                                        const OH_NN_Tensor *tensor,
                                        const void *dataBuffer,
                                        size_t length) {
  return OH_NN_SUCCESS;
}

void OH_NNModel_Destroy(OH_NNModel **model) {}

OH_NN_ReturnCode OH_NNModel_GetAvailableOperations(OH_NNModel *model,
                                                   size_t deviceID,
                                                   const bool **isSupported,
                                                   uint32_t *opCount) {
  return OH_NN_SUCCESS;
}

OH_NN_ReturnCode OH_NNModel_BuildFromLiteGraph(OH_NNModel *model, const void *liteGraph) {
  return OH_NN_SUCCESS;
}

OH_NN_ReturnCode OH_NNDevice_GetName(size_t deviceID, const char **name) {
  return OH_NN_SUCCESS;
}

OH_NN_ReturnCode OH_NNDevice_GetType(size_t deviceID, OH_NN_DeviceType *deviceType) {
  return OH_NN_SUCCESS;
}

OH_NN_ReturnCode OH_NNCompilation_SetPriority(OH_NNCompilation *compilation, OH_NN_Priority priority) {
  return OH_NN_SUCCESS;
}

OH_NN_ReturnCode OH_NNCompilation_EnableFloat16(OH_NNCompilation *compilation, bool enableFloat16) {
  return OH_NN_SUCCESS;
}

OH_NN_ReturnCode OH_NNCompilation_SetPerformanceMode(OH_NNCompilation *compilation,
                                                     OH_NN_PerformanceMode performanceMode) {
  return OH_NN_SUCCESS;
}
