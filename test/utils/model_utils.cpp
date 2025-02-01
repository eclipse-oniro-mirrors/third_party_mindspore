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

#include "model_utils.h"
#include <securec.h>
#include "gtest/gtest.h"
#include "include/c_api/context_c.h"
#include "include/c_api/model_c.h"
#include "include/c_api/types_c.h"
#include "include/c_api/status_c.h"
#include "include/c_api/data_type_c.h"
#include "include/c_api/tensor_c.h"
#include "include/c_api/format_c.h"
#include "common.h"

std::string g_testResourcesDir = "/data/test/resource/";

// function before callback
bool PrintBeforeCallback(const OH_AI_TensorHandleArray inputs, const OH_AI_TensorHandleArray outputs,
                         const OH_AI_CallBackParam kernelInfo) {
    std::cout << "Before forwarding " << kernelInfo.node_name << " " << kernelInfo.node_type << std::endl;
    return true;
}

// function after callback
bool PrintAfterCallback(const OH_AI_TensorHandleArray inputs, const OH_AI_TensorHandleArray outputs,
                        const OH_AI_CallBackParam kernelInfo) {
    std::cout << "After forwarding " << kernelInfo.node_name << " " << kernelInfo.node_type << std::endl;
    return true;
}

// add cpu device info
void AddContextDeviceCPU(OH_AI_ContextHandle context) {
    OH_AI_DeviceInfoHandle cpuDeviceInfo = OH_AI_DeviceInfoCreate(OH_AI_DEVICETYPE_CPU);
    ASSERT_NE(cpuDeviceInfo, nullptr);
    OH_AI_DeviceType deviceType = OH_AI_DeviceInfoGetDeviceType(cpuDeviceInfo);
    printf("==========deviceType:%d\n", deviceType);
    ASSERT_EQ(deviceType, OH_AI_DEVICETYPE_CPU);
    OH_AI_ContextAddDeviceInfo(context, cpuDeviceInfo);
}

bool IsNPU() {
    size_t num = 0;
    auto desc = OH_AI_GetAllNNRTDeviceDescs(&num);
    if (desc == nullptr) {
        return false;
    }
    auto name = OH_AI_GetNameFromNNRTDeviceDesc(desc);
    const std::string npuNamePrefix = "NPU_";
    if (strncmp(npuNamePrefix.c_str(), name, npuNamePrefix.size()) != 0) {
        return false;
    }
    return true;
}

// add nnrt device info
void AddContextDeviceNNRT(OH_AI_ContextHandle context) {
    size_t num = 0;
    auto desc = OH_AI_GetAllNNRTDeviceDescs(&num);
    if (desc == nullptr) {
        return;
    }

    std::cout << "found " << num << " nnrt devices" << std::endl;
    auto id = OH_AI_GetDeviceIdFromNNRTDeviceDesc(desc);
    auto name = OH_AI_GetNameFromNNRTDeviceDesc(desc);
    auto type = OH_AI_GetTypeFromNNRTDeviceDesc(desc);
    std::cout << "NNRT device: id = " << id << ", name: " << name << ", type:" << type << std::endl;

    OH_AI_DeviceInfoHandle nnrtDeviceInfo = OH_AI_DeviceInfoCreate(OH_AI_DEVICETYPE_NNRT);
    ASSERT_NE(nnrtDeviceInfo, nullptr);
    OH_AI_DeviceInfoSetDeviceId(nnrtDeviceInfo, id);
    OH_AI_DestroyAllNNRTDeviceDescs(&desc);

    OH_AI_DeviceType deviceType = OH_AI_DeviceInfoGetDeviceType(nnrtDeviceInfo);
    printf("==========deviceType:%d\n", deviceType);
    ASSERT_EQ(deviceType, OH_AI_DEVICETYPE_NNRT);

    OH_AI_DeviceInfoSetPerformanceMode(nnrtDeviceInfo, OH_AI_PERFORMANCE_MEDIUM);
    ASSERT_EQ(OH_AI_DeviceInfoGetPerformanceMode(nnrtDeviceInfo), OH_AI_PERFORMANCE_MEDIUM);
    OH_AI_DeviceInfoSetPriority(nnrtDeviceInfo, OH_AI_PRIORITY_MEDIUM);
    ASSERT_EQ(OH_AI_DeviceInfoGetPriority(nnrtDeviceInfo), OH_AI_PRIORITY_MEDIUM);

    OH_AI_ContextAddDeviceInfo(context, nnrtDeviceInfo);
}

// fill data to inputs tensor
void FillInputsData(OH_AI_TensorHandleArray inputs, std::string modelName, bool isTranspose) {
    for (size_t i = 0; i < inputs.handle_num; ++i) {
        printf("==========ReadFile==========\n");
        size_t size1;
        size_t *ptrSize1 = &size1;
        std::string inputDataPath = g_testResourcesDir + modelName + "_" + std::to_string(i) + ".input";
        const char *imagePath = inputDataPath.c_str();
        char *imageBuf = ReadFile(imagePath, ptrSize1);
        ASSERT_NE(imageBuf, nullptr);
        OH_AI_TensorHandle tensor = inputs.handle_list[i];
        int64_t elementNum = OH_AI_TensorGetElementNum(tensor);
        printf("Tensor name: %s. \n", OH_AI_TensorGetName(tensor));
        float *inputData = reinterpret_cast<float *>(OH_AI_TensorGetMutableData(inputs.handle_list[i]));
        ASSERT_NE(inputData, nullptr);
        if (isTranspose) {
            printf("==========Transpose==========\n");
            size_t shapeNum;
            const int64_t *shape = OH_AI_TensorGetShape(tensor, &shapeNum);
            auto imageBufNhwc = new char[size1];
            PackNCHWToNHWCFp32(imageBuf, imageBufNhwc, shape[0], shape[1] * shape[2], shape[3]);
            errno_t ret = memcpy_s(inputData, size1, imageBufNhwc, size1);
            if (ret != EOK) {
                printf("memcpy_s failed, ret: %d\n", ret);
            }
            delete[] imageBufNhwc;
        } else {
            errno_t ret = memcpy_s(inputData, size1, imageBuf, size1);
            if (ret != EOK) {
                printf("memcpy_s failed, ret: %d\n", ret);
            }
        }
        printf("input data after filling is: ");
        for (int j = 0; j < elementNum && j <= 20; ++j) {
            printf("%f ", inputData[j]);
        }
        printf("\n");
        delete[] imageBuf;
    }
}

// compare result after predict
void CompareResult(OH_AI_TensorHandleArray outputs, std::string modelName, float atol, float rtol) {
    printf("==========GetOutput==========\n");
    for (size_t i = 0; i < outputs.handle_num; ++i) {
        OH_AI_TensorHandle tensor = outputs.handle_list[i];
        int64_t elementNum = OH_AI_TensorGetElementNum(tensor);
        printf("Tensor name: %s .\n", OH_AI_TensorGetName(tensor));
        float *outputData = reinterpret_cast<float *>(OH_AI_TensorGetMutableData(tensor));
        printf("output data is:");
        for (int j = 0; j < elementNum && j <= 20; ++j) {
            printf("%f ", outputData[j]);
        }
        printf("\n");
        printf("==========compFp32WithTData==========\n");
        std::string outputFile = g_testResourcesDir + modelName + std::to_string(i) + ".output";
        bool result = compFp32WithTData(outputData, outputFile, atol, rtol, false);
        EXPECT_EQ(result, true);
    }
}

// model build and predict
void ModelPredict(OH_AI_ModelHandle model, OH_AI_ContextHandle context, std::string modelName,
                  OH_AI_ShapeInfo shapeInfos, bool buildByGraph, bool isTranspose, bool isCallback) {
    std::string modelPath = g_testResourcesDir + modelName + ".ms";
    const char *graphPath = modelPath.c_str();
    OH_AI_Status ret = OH_AI_STATUS_SUCCESS;
    if (buildByGraph) {
        printf("==========Build model by graphBuf==========\n");
        size_t size;
        size_t *ptrSize = &size;
        char *graphBuf = ReadFile(graphPath, ptrSize);
        ASSERT_NE(graphBuf, nullptr);
        ret = OH_AI_ModelBuild(model, graphBuf, size, OH_AI_MODELTYPE_MINDIR, context);
        delete[] graphBuf;
    } else {
        printf("==========Build model==========\n");
        ret = OH_AI_ModelBuildFromFile(model, graphPath, OH_AI_MODELTYPE_MINDIR, context);
    }
    printf("==========build model return code:%d\n", ret);
    ASSERT_EQ(ret, OH_AI_STATUS_SUCCESS);
    printf("==========GetInputs==========\n");
    OH_AI_TensorHandleArray inputs = OH_AI_ModelGetInputs(model);
    ASSERT_NE(inputs.handle_list, nullptr);
    if (shapeInfos.shape_num != 0) {
        printf("==========Resizes==========\n");
        OH_AI_Status resize_ret = OH_AI_ModelResize(model, inputs, &shapeInfos, inputs.handle_num);
        printf("==========Resizes return code:%d\n", resize_ret);
        ASSERT_EQ(resize_ret, OH_AI_STATUS_SUCCESS);
    }

    FillInputsData(inputs, modelName, isTranspose);
    OH_AI_TensorHandleArray outputs;
    OH_AI_Status predictRet = OH_AI_STATUS_SUCCESS;
    if (isCallback) {
        printf("==========Model Predict Callback==========\n");
        OH_AI_KernelCallBack beforeCallBack = PrintBeforeCallback;
        OH_AI_KernelCallBack afterCallBack = PrintAfterCallback;
        predictRet = OH_AI_ModelPredict(model, inputs, &outputs, beforeCallBack, afterCallBack);
    } else {
        printf("==========Model Predict==========\n");
        predictRet = OH_AI_ModelPredict(model, inputs, &outputs, nullptr, nullptr);
    }
    printf("==========Model Predict End==========\n");
    ASSERT_EQ(predictRet, OH_AI_STATUS_SUCCESS);
    printf("=========CompareResult===========\n");
    CompareResult(outputs, modelName);
    printf("=========OH_AI_ModelDestroy===========\n");
    OH_AI_ModelDestroy(&model);
    printf("=========OH_AI_ModelDestroy End===========\n");
}

// model train build and predict
void ModelTrain(OH_AI_ModelHandle model, OH_AI_ContextHandle context, std::string modelName,
                OH_AI_ShapeInfo shapeInfos, bool buildByGraph, bool isTranspose, bool isCallback) {
    std::string modelPath = g_testResourcesDir + modelName + ".ms";
    const char *graphPath = modelPath.c_str();
    OH_AI_TrainCfgHandle trainCfg = OH_AI_TrainCfgCreate();
    OH_AI_Status ret = OH_AI_STATUS_SUCCESS;
    if (buildByGraph) {
        printf("==========Build model by graphBuf==========\n");
        size_t size;
        size_t *ptrSize = &size;
        char *graphBuf = ReadFile(graphPath, ptrSize);
        ASSERT_NE(graphBuf, nullptr);
        ret = OH_AI_TrainModelBuild(model, graphBuf, size, OH_AI_MODELTYPE_MINDIR, context, trainCfg);
        delete[] graphBuf;
    } else {
        printf("==========Build model==========\n");
        ret = OH_AI_TrainModelBuildFromFile(model, graphPath, OH_AI_MODELTYPE_MINDIR, context, trainCfg);
    }
    printf("==========build model return code:%d\n", ret);
    ASSERT_EQ(ret, OH_AI_STATUS_SUCCESS);
    printf("==========GetInputs==========\n");
    OH_AI_TensorHandleArray inputs = OH_AI_ModelGetInputs(model);
    ASSERT_NE(inputs.handle_list, nullptr);
    if (shapeInfos.shape_num != 0) {
        printf("==========Resizes==========\n");
        OH_AI_Status resize_ret = OH_AI_ModelResize(model, inputs, &shapeInfos, inputs.handle_num);
        printf("==========Resizes return code:%d\n", resize_ret);
        ASSERT_EQ(resize_ret, OH_AI_STATUS_SUCCESS);
    }
    FillInputsData(inputs, modelName, isTranspose);
    ret = OH_AI_ModelSetTrainMode(model, true);
    ASSERT_EQ(ret, OH_AI_STATUS_SUCCESS);
    if (isCallback) {
        printf("==========Model RunStep Callback==========\n");
        OH_AI_KernelCallBack beforeCallBack = PrintBeforeCallback;
        OH_AI_KernelCallBack afterCallBack = PrintAfterCallback;
        ret = OH_AI_RunStep(model, beforeCallBack, afterCallBack);
    } else {
        printf("==========Model RunStep==========\n");
        ret = OH_AI_RunStep(model, nullptr, nullptr);
    }
    printf("==========Model RunStep End==========\n");
    ASSERT_EQ(ret, OH_AI_STATUS_SUCCESS);
}
