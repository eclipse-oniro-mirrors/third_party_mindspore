/*
 * Copyright (C) 2023 Huawei Device Co., Ltd.
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

#include <cstdio>
#include <stddef.h>
#include <stdint.h>
#include <iostream>
#include "mindspore_fuzzer.h"
#include "../data.h"
#include "include/c_api/context_c.h"
#include "include/c_api/model_c.h"
#include "../../utils/model_utils.h"
#include "context_c_fuzzer.h"

bool MSPreparedModelFuzzTest(const uint8_t* data, size_t size) {
    if (data == nullptr) {
        return false;
    }

    OH_AI_ContextHandle context = OH_AI_ContextCreate();
    if (context == nullptr) {
        printf("create context failed.\n");
        return false;
    }
    OH_AI_ContextSetThreadNum(context, 4);

    OH_AI_DeviceInfoHandle cpuDeviceInfo = OH_AI_DeviceInfoCreate(OH_AI_DEVICETYPE_CPU);
    if (cpuDeviceInfo == NULL) {
        printf("OH_AI_DeviceInfoCreate failed.\n");
        OH_AI_ContextDestroy(&context);
        return false;
    }
    OH_AI_ContextAddDeviceInfo(context, cpuDeviceInfo);

    OH_AI_ModelHandle model = OH_AI_ModelCreate();
    if (model == nullptr) {
        printf("create model failed.\n");
        return false;
    }

    OH_AI_ContextDestroy(&context);
    OH_AI_ModelDestroy(&model);
    return true;
}

bool MSContextFuzzTest_Null(const uint8_t* data, size_t size) {
    OH_AI_ContextHandle context = nullptr;

    OH_AI_ContextSetThreadNum(context, 4);
    auto retThreadNum = OH_AI_ContextGetThreadNum(context);
    if (retThreadNum != 0) {
        printf("OH_AI_ContextGetThreadNum failed.\n");
        return false;
    }

    OH_AI_ContextSetThreadAffinityMode(context, 1);
    auto ret = OH_AI_ContextGetThreadAffinityMode(context);
    if (ret != 0) {
        printf("OH_AI_ContextGetThreadAffinityMode failed.\n");
        return false;
    }

    OH_AI_ContextSetThreadAffinityCoreList(context, nullptr, 0);
    auto retCoreList = OH_AI_ContextGetThreadAffinityCoreList(context, nullptr);
    if (retCoreList != 0) {
        printf("OH_AI_ContextGetThreadAffinityCoreList failed.\n");
        return false;
    }

    OH_AI_ContextSetEnableParallel(context, true);
    auto retParallel = OH_AI_ContextGetEnableParallel(context);
    if (retParallel != false) {
        printf("OH_AI_ContextGetEnableParallel failed.\n");
        return false;
    }


    OH_AI_DeviceInfoHandle deviceInfo = OH_AI_DeviceInfoCreate(OH_AI_DEVICETYPE_CPU);
    if (deviceInfo == nullptr) {
        printf("OH_AI_DeviceInfoCreate cpu failed.\n");
        return false;
    }
    OH_AI_DeviceInfoDestroy(&deviceInfo);

    deviceInfo = OH_AI_DeviceInfoCreate(OH_AI_DEVICETYPE_INVALID);
    if (deviceInfo != nullptr) {
        printf("OH_AI_DeviceInfoCreate failed.\n");
        return false;
    }
    OH_AI_ContextAddDeviceInfo(context, deviceInfo);
    OH_AI_DeviceInfoSetProvider(deviceInfo, nullptr);
    auto retProvider = OH_AI_DeviceInfoGetProvider(deviceInfo);
    if (retProvider != nullptr) {
        printf("OH_AI_DeviceInfoGetProvider failed.\n");
        return false;
    }

    OH_AI_DeviceInfoSetProviderDevice(deviceInfo, nullptr);
    auto retProDevice = OH_AI_DeviceInfoGetProviderDevice(deviceInfo);
    if (retProDevice != nullptr) {
        printf("OH_AI_DeviceInfoGetProviderDevice failed.\n");
        return false;
    }
    auto deviceType = OH_AI_DeviceInfoGetDeviceType(deviceInfo);
    if (deviceType != OH_AI_DEVICETYPE_INVALID) {
        printf("OH_AI_DeviceInfoGetDeviceType failed.\n");
        return false;
    }

    OH_AI_DeviceInfoSetEnableFP16(deviceInfo, true);
    auto retEnableFp16 = OH_AI_DeviceInfoGetEnableFP16(deviceInfo);
    if (retEnableFp16 != false) {
        printf("OH_AI_DeviceInfoGetEnableFP16 failed.\n");
        return false;
    }
    OH_AI_DeviceInfoSetFrequency(deviceInfo, 1);
    auto retFrequency = OH_AI_DeviceInfoGetFrequency(deviceInfo);
    if (retFrequency != -1) {
        printf("OH_AI_DeviceInfoGetFrequency failed.\n");
        return false;
    }

    OH_AI_DeviceInfoSetDeviceId(deviceInfo, 1);
    auto retDeviceId = OH_AI_DeviceInfoGetDeviceId(deviceInfo);
    if (retDeviceId != 0) {
        printf("OH_AI_DeviceInfoGetDeviceId failed.\n");
        return false;
    }
    OH_AI_DeviceInfoSetPerformanceMode(deviceInfo, OH_AI_PERFORMANCE_HIGH);
    auto retPerMode = OH_AI_DeviceInfoGetPerformanceMode(deviceInfo);
    if (retPerMode != OH_AI_PERFORMANCE_NONE) {
        printf("OH_AI_DeviceInfoGetPerformanceMode failed.\n");
        return false;
    }
    OH_AI_DeviceInfoSetPriority(deviceInfo, OH_AI_PRIORITY_HIGH);
    auto retPriority = OH_AI_DeviceInfoGetPriority(deviceInfo);
    if (retPriority != OH_AI_PRIORITY_NONE) {
        printf("OH_AI_DeviceInfoGetPriority failed.\n");
        return false;
    }
    auto retExt = OH_AI_DeviceInfoAddExtension(deviceInfo, nullptr, nullptr, 0);
    if (retExt != OH_AI_STATUS_LITE_NULLPTR) {
        printf("OH_AI_DeviceInfoAddExtension failed.\n");
        return false;
    }
    return true;
}

bool MSTensorFuzzTest_Null(const uint8_t* data, size_t size) {
    OH_AI_TensorHandle tensor = OH_AI_TensorCreate(nullptr, OH_AI_DATATYPE_NUMBERTYPE_FLOAT32, nullptr, 0, nullptr, 0);
    OH_AI_TensorDestroy(&tensor);

    auto retClone = OH_AI_TensorClone(tensor);
    if (retClone != nullptr) {
        printf("OH_AI_TensorClone failed.\n");
        return false;
    }
    OH_AI_TensorSetName(tensor, nullptr);
    auto retGetName = OH_AI_TensorGetName(tensor);
    if (retGetName != nullptr) {
        printf("OH_AI_TensorGetName failed.\n");
        return false;
    }
    OH_AI_TensorSetDataType(tensor, OH_AI_DATATYPE_NUMBERTYPE_INT64);
    auto retGetDataType = OH_AI_TensorGetDataType(tensor);
    if (retGetDataType != OH_AI_DATATYPE_UNKNOWN) {
        printf("OH_AI_TensorGetDataType failed.\n");
        return false;
    }
    OH_AI_TensorSetShape(tensor, nullptr, 0);
    auto retGetShape = OH_AI_TensorGetShape(tensor, nullptr);
    if (retGetShape != nullptr) {
        printf("OH_AI_TensorGetShape failed.\n");
        return false;
    }
    OH_AI_TensorSetFormat(tensor, OH_AI_FORMAT_KHWC);
    auto retGetFormat = OH_AI_TensorGetFormat(tensor);
    if (retGetFormat != OH_AI_FORMAT_NHWC) {
        printf("OH_AI_TensorGetFormat failed.\n");
        return false;
    }
    OH_AI_TensorSetData(tensor, nullptr);
    auto retSetUserData = OH_AI_TensorSetUserData(tensor, nullptr, 0);
    if (retSetUserData != OH_AI_STATUS_LITE_NULLPTR) {
        printf("OH_AI_TensorSetUserData failed.\n");
        return false;
    }
    auto retGetData = OH_AI_TensorGetData(tensor);
    if (retGetData != nullptr) {
        printf("OH_AI_TensorGetData failed.\n");
        return false;
    }
    auto retMutaData = OH_AI_TensorGetMutableData(tensor);
    if (retMutaData != nullptr) {
        printf("OH_AI_TensorGetMutableData failed.\n");
        return false;
    }
    auto retEleNum = OH_AI_TensorGetElementNum(tensor);
    if (retEleNum != 0) {
        printf("OH_AI_TensorGetElementNum failed.\n");
        return false;
    }
    auto retDataSize = OH_AI_TensorGetDataSize(tensor);
    if (retDataSize != 0) {
        printf("OH_AI_TensorGetDataSize failed.\n");
        return false;
    }
    return true;
}

bool MSModelFuzzTest_Null(const uint8_t* data, size_t size) {
    OH_AI_ModelHandle model = OH_AI_ModelCreate();
    if (model == nullptr) {
        printf("create model failed.\n");
        return false;
    }
    OH_AI_ModelDestroy(&model);
    OH_AI_ModelSetWorkspace(model, nullptr, 0);
    OH_AI_ModelCalcWorkspaceSize(model);
    auto ret = OH_AI_ModelBuild(model, nullptr, 0, OH_AI_MODELTYPE_MINDIR, nullptr);
    if (ret != OH_AI_STATUS_LITE_NULLPTR) {
        printf("OH_AI_ModelBuild failed.\n");
        return false;
    }
    ret = OH_AI_ModelBuildFromFile(model, nullptr, OH_AI_MODELTYPE_MINDIR, nullptr);
    if (ret != OH_AI_STATUS_LITE_NULLPTR) {
        printf("OH_AI_ModelBuildFromFile failed.\n");
        return false;
    }
    ret = OH_AI_ModelResize(model, {0, nullptr}, nullptr, 0);
    if (ret != OH_AI_STATUS_LITE_NULLPTR) {
        printf("OH_AI_ModelResize failed.\n");
        return false;
    }
    OH_AI_TensorHandleArray outputs = {0, nullptr};
    ret = OH_AI_ModelPredict(model, {0, nullptr}, &outputs, nullptr, nullptr);
    if (ret != OH_AI_STATUS_LITE_NULLPTR) {
        printf("OH_AI_ModelPredict failed.\n");
        return false;
    }
    OH_AI_ModelRunStep(model, nullptr, nullptr);
    OH_AI_ModelExportWeight(model, nullptr);
    auto retInputs = OH_AI_ModelGetInputs(model);
    if (retInputs.handle_list != nullptr) {
        printf("OH_AI_ModelGetInputs failed.\n");
        return false;
    }
    auto retOutputs = OH_AI_ModelGetOutputs(model);
    if (retOutputs.handle_list != nullptr) {
        printf("OH_AI_ModelGetOutputs failed.\n");
        return false;
    }
    OH_AI_ModelGetInputByTensorName(model, nullptr);
    OH_AI_ModelGetOutputByTensorName(model, nullptr);
    auto trainCfg = OH_AI_TrainCfgCreate();
    if (trainCfg == nullptr) {
        printf("OH_AI_TrainCfgCreate failed.\n");
        return false;
    }
    OH_AI_TrainCfgDestroy(&trainCfg);
    auto retLossName = OH_AI_TrainCfgGetLossName(trainCfg, nullptr);
    if (retLossName != nullptr) {
        printf("OH_AI_TrainCfgGetLossName failed.\n");
        return false;
    }
    OH_AI_TrainCfgSetLossName(trainCfg, nullptr, 0);
    auto retOptLevel = OH_AI_TrainCfgGetOptimizationLevel(trainCfg);
    if (retOptLevel != OH_AI_KO0) {
        printf("OH_AI_TrainCfgGetOptimizationLevel failed.\n");
        return false;
    }
    OH_AI_TrainCfgSetOptimizationLevel(trainCfg, OH_AI_KAUTO);
    ret = OH_AI_TrainModelBuild(model, nullptr, 0, OH_AI_MODELTYPE_MINDIR, nullptr, trainCfg);
    if (ret != OH_AI_STATUS_LITE_NULLPTR) {
        printf("OH_AI_TrainModelBuild failed.\n");
        return false;
    }
    ret = OH_AI_TrainModelBuildFromFile(model, nullptr, OH_AI_MODELTYPE_MINDIR, nullptr, trainCfg);
    if (ret != OH_AI_STATUS_LITE_NULLPTR) {
        printf("OH_AI_TrainModelBuildFromFile failed.\n");
        return false;
    }
    ret = OH_AI_ModelSetLearningRate(model, 0.0f);
    if (ret != OH_AI_STATUS_LITE_PARAM_INVALID) {
        printf("OH_AI_ModelSetLearningRate failed.\n");
        return false;
    }
    OH_AI_ModelGetLearningRate(model);
    ret = OH_AI_RunStep(model, nullptr, nullptr);
    if (ret != OH_AI_STATUS_LITE_PARAM_INVALID) {
        printf("OH_AI_RunStep failed.\n");
        return false;
    }
    auto retGetWeights = OH_AI_ModelGetWeights(model);
    if (retGetWeights.handle_list != nullptr) {
        printf("OH_AI_ModelGetWeights failed.\n");
        return false;
    }
    ret = OH_AI_ModelUpdateWeights(model, {0, nullptr});
    if (ret != OH_AI_STATUS_LITE_PARAM_INVALID) {
        printf("OH_AI_ModelUpdateWeights failed.\n");
        return false;
    }
    auto retTrainMode = OH_AI_ModelGetTrainMode(model);
    if (retTrainMode != false) {
        printf("OH_AI_ModelGetTrainMode failed.\n");
        return false;
    }
    ret = OH_AI_ModelSetTrainMode(model, true);
    if (ret != OH_AI_STATUS_LITE_PARAM_INVALID) {
        printf("OH_AI_ModelSetTrainMode failed.\n");
        return false;
    }
    ret = OH_AI_ModelSetupVirtualBatch(model, 0, 0.0f, 0.0f);
    if (ret != OH_AI_STATUS_LITE_PARAM_INVALID) {
        printf("OH_AI_ModelSetupVirtualBatch failed.\n");
        return false;
    }
    ret = OH_AI_ExportModel(model, OH_AI_MODELTYPE_MINDIR, nullptr, OH_AI_UNKNOWN_QUANT_TYPE, true, nullptr, 0);
    if (ret != OH_AI_STATUS_LITE_PARAM_INVALID) {
        printf("OH_AI_ExportModel failed.\n");
        return false;
    }
    ret = OH_AI_ExportModelBuffer(model, OH_AI_MODELTYPE_MINDIR, nullptr, nullptr, OH_AI_FULL_QUANT, true, nullptr, 0);
    if (ret != OH_AI_STATUS_LITE_PARAM_INVALID) {
        printf("OH_AI_ExportModelBuffer failed.\n");
        return false;
    }
    ret = OH_AI_ExportWeightsCollaborateWithMicro(model, OH_AI_MODELTYPE_MINDIR, nullptr, true, true, nullptr, 0);
    if (ret != OH_AI_STATUS_LITE_PARAM_INVALID) {
        printf("OH_AI_ExportWeightsCollaborateWithMicro failed.\n");
        return false;
    }
    return true;
}

/* Fuzzer entry point */
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)
{
    if (data == nullptr) {
        LOGE("Pass data is nullptr.");
        return 0;
    }

    if (size < 4) {
        LOGE("Pass size is too small.");
        return 0;
    }

    MSPreparedModelFuzzTest(data, size);
    MSModelFuzzTest_Null(data, size);
    MSContextFuzzTest_Null(data, size);
    MSTensorFuzzTest_Null(data, size);
    MSContextFuzzTest(data, size);
    return 0;
}