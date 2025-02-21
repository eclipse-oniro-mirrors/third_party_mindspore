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

#include "gtest/gtest.h"
#include <inttypes.h>
#include <random>
#include <securec.h>
#include "include/c_api/context_c.h"
#include "include/c_api/model_c.h"
#include "include/c_api/types_c.h"
#include "include/c_api/status_c.h"
#include "include/c_api/data_type_c.h"
#include "include/c_api/tensor_c.h"
#include "include/c_api/format_c.h"
#include "../utils/model_utils.h"
#include "../utils/common.h"

class MSLiteTest: public testing::Test {
  protected:
    static void SetUpTestCase(void) {}
    static void TearDownTestCase(void) {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

std::string g_testFilesDir = "/data/test/resource/";

/*
 * @tc.name: Context_Create
 * @tc.desc: Verify the return context of the OH_AI_ContextCreate function.
 * @tc.type: FUNC
 */
HWTEST(MSLiteTest, Context_Create, testing::ext::TestSize.Level0) {
    printf("==========Init Context==========\n");
    OH_AI_ContextHandle context = OH_AI_ContextCreate();
    ASSERT_NE(context, nullptr);
}

/*
 * @tc.name: Context_Destroy
 * @tc.desc: Verify the OH_AI_ContextDestroy function.
 * @tc.type: FUNC
 */
HWTEST(MSLiteTest, Context_Destroy, testing::ext::TestSize.Level0) {
    printf("==========Init Context==========\n");
    OH_AI_ContextHandle context = OH_AI_ContextCreate();
    ASSERT_NE(context, nullptr);
    OH_AI_ContextDestroy(&context);
    ASSERT_EQ(context, nullptr);
}

/*
 * @tc.name: Context_Thread_Num
 * @tc.desc: Verify the OH_AI_ContextSetThreadNum/OH_AI_ContextGetThreadNum function.
 * @tc.type: FUNC
 */
HWTEST(MSLiteTest, Context_Thread_Num, testing::ext::TestSize.Level0) {
    printf("==========Init Context==========\n");
    OH_AI_ContextHandle context = OH_AI_ContextCreate();
    ASSERT_NE(context, nullptr);
    OH_AI_ContextSetThreadNum(context, 2);
    auto thread_num = OH_AI_ContextGetThreadNum(context);
    printf("==========thread_num: %d\n", thread_num);
    ASSERT_EQ(thread_num, 2);
}

/*
 * @tc.name: Context_Thread_Affinity
 * @tc.desc: Verify the OH_AI_ContextSetThreadAffinityMode/OH_AI_ContextGetThreadAffinityMode function.
 * @tc.type: FUNC
 */
HWTEST(MSLiteTest, Context_Thread_Affinity, testing::ext::TestSize.Level0) {
    printf("==========Init Context==========\n");
    OH_AI_ContextHandle context = OH_AI_ContextCreate();
    ASSERT_NE(context, nullptr);
    OH_AI_ContextSetThreadNum(context, 4);
    auto thread_num = OH_AI_ContextGetThreadNum(context);
    printf("==========thread_num: %d\n", thread_num);
    ASSERT_EQ(thread_num, 4);

    OH_AI_ContextSetThreadAffinityMode(context, 2);
    int thread_affinity_mode = OH_AI_ContextGetThreadAffinityMode(context);
    printf("==========thread_affinity_mode:%d\n", thread_affinity_mode);
    ASSERT_EQ(thread_affinity_mode, 2);
}

/*
 * @tc.name: Context_Thread_Affinity_Corelist
 * @tc.desc: Verify the OH_AI_ContextSetThreadAffinityCoreList/OH_AI_ContextGetThreadAffinityCoreList function.
 * @tc.type: FUNC
 */
HWTEST(MSLiteTest, Context_Thread_Affinity_Corelist, testing::ext::TestSize.Level0) {
    printf("==========Init Context==========\n");
    OH_AI_ContextHandle context = OH_AI_ContextCreate();
    ASSERT_NE(context, nullptr);
    OH_AI_ContextSetThreadNum(context, 4);
    auto thread_num = OH_AI_ContextGetThreadNum(context);
    printf("==========thread_num: %d\n", thread_num);
    ASSERT_EQ(thread_num, 4);

    constexpr size_t coreNum = 4;
    int32_t coreList[coreNum] = {0, 1, 2, 3};
    OH_AI_ContextSetThreadAffinityCoreList(context, coreList, coreNum);
    size_t retCoreNum;
    const int32_t *retCoreList = nullptr;
    retCoreList = OH_AI_ContextGetThreadAffinityCoreList(context, &retCoreNum);
    ASSERT_EQ(retCoreNum, coreNum);
    for (size_t i = 0; i < retCoreNum; i++) {
        printf("==========retCoreList:%d\n", retCoreList[i]);
        ASSERT_EQ(retCoreList[i], coreList[i]);
    }
}

/*
 * @tc.name: Context_Enable_Parallel
 * @tc.desc: Verify the OH_AI_ContextSetEnableParallel/OH_AI_ContextGetEnableParallel function.
 * @tc.type: FUNC
 */
HWTEST(MSLiteTest, Context_Enable_Parallel, testing::ext::TestSize.Level0) {
    printf("==========Init Context==========\n");
    OH_AI_ContextHandle context = OH_AI_ContextCreate();
    ASSERT_NE(context, nullptr);
    OH_AI_ContextSetThreadNum(context, 4);
    auto thread_num = OH_AI_ContextGetThreadNum(context);
    printf("==========thread_num: %d\n", thread_num);
    ASSERT_EQ(thread_num, 4);

    OH_AI_ContextSetEnableParallel(context, true);
    bool isParallel = OH_AI_ContextGetEnableParallel(context);
    printf("==========isParallel:%d\n", isParallel);
    ASSERT_EQ(isParallel, true);

    AddContextDeviceCPU(context);
    printf("==========Create model==========\n");
    OH_AI_ModelHandle model = OH_AI_ModelCreate();
    ASSERT_NE(model, nullptr);
    ModelPredict(model, context, "ml_face_isface", {}, false, true, false);
}

/*
 * @tc.name: Context_Enable_FP16
 * @tc.desc: Verify the OH_AI_DeviceInfoSetEnableFP16/OH_AI_DeviceInfoGetEnableFP16 function.
 * @tc.type: FUNC
 */
HWTEST(MSLiteTest, Context_Enable_FP16, testing::ext::TestSize.Level0) {
    printf("==========Init Context==========\n");
    OH_AI_ContextHandle context = OH_AI_ContextCreate();
    ASSERT_NE(context, nullptr);

    OH_AI_DeviceInfoHandle cpu_device_info = OH_AI_DeviceInfoCreate(OH_AI_DEVICETYPE_CPU);
    ASSERT_NE(cpu_device_info, nullptr);
    OH_AI_DeviceInfoSetEnableFP16(cpu_device_info, true);
    bool isFp16 = OH_AI_DeviceInfoGetEnableFP16(cpu_device_info);
    printf("==========isFp16:%d\n", isFp16);
    ASSERT_EQ(isFp16, true);

    OH_AI_ContextAddDeviceInfo(context, cpu_device_info);

    printf("==========Create model==========\n");
    OH_AI_ModelHandle model = OH_AI_ModelCreate();
    ASSERT_NE(model, nullptr);
    ModelPredict(model, context, "ml_face_isface", {}, false, true, false);
}

/*
 * @tc.name: Context_Provider
 * @tc.desc: Verify the OH_AI_DeviceInfoSetProvider/OH_AI_DeviceInfoGetProvider function.
 * @tc.type: FUNC
 */
HWTEST(MSLiteTest, Context_Provider, testing::ext::TestSize.Level0) {
    printf("==========Init Context==========\n");
    OH_AI_ContextHandle context = OH_AI_ContextCreate();
    ASSERT_NE(context, nullptr);

    OH_AI_DeviceInfoHandle cpu_device_info = OH_AI_DeviceInfoCreate(OH_AI_DEVICETYPE_CPU);
    ASSERT_NE(cpu_device_info, nullptr);
    OH_AI_DeviceInfoSetProvider(cpu_device_info, "vendor_new");
    ASSERT_EQ(strcmp(OH_AI_DeviceInfoGetProvider(cpu_device_info), "vendor_new"), 0);

    OH_AI_ContextAddDeviceInfo(context, cpu_device_info);

    printf("==========Create model==========\n");
    OH_AI_ModelHandle model = OH_AI_ModelCreate();
    ASSERT_NE(model, nullptr);
    ModelPredict(model, context, "ml_face_isface", {}, false, true, false);
}

/*
 * @tc.name: Context_Provider_Device
 * @tc.desc: Verify the OH_AI_DeviceInfoSetProviderDevice/OH_AI_DeviceInfoGetProviderDevice function.
 * @tc.type: FUNC
 */
HWTEST(MSLiteTest, Context_Provider_Device, testing::ext::TestSize.Level0) {
    printf("==========Init Context==========\n");
    OH_AI_ContextHandle context = OH_AI_ContextCreate();
    ASSERT_NE(context, nullptr);

    OH_AI_DeviceInfoHandle cpu_device_info = OH_AI_DeviceInfoCreate(OH_AI_DEVICETYPE_CPU);
    ASSERT_NE(cpu_device_info, nullptr);
    OH_AI_DeviceInfoSetProviderDevice(cpu_device_info, "cpu_new");
    ASSERT_EQ(strcmp(OH_AI_DeviceInfoGetProviderDevice(cpu_device_info), "cpu_new"), 0);

    OH_AI_ContextAddDeviceInfo(context, cpu_device_info);

    printf("==========Create model==========\n");
    OH_AI_ModelHandle model = OH_AI_ModelCreate();
    ASSERT_NE(model, nullptr);
    ModelPredict(model, context, "ml_face_isface", {}, false, true, false);
}

/*
 * @tc.name: Context_Device
 * @tc.desc: Verify the OH_AI_DeviceInfoCreate/OH_AI_DeviceInfoDestroy function.
 * @tc.type: FUNC
 */
HWTEST(MSLiteTest, Context_Device, testing::ext::TestSize.Level0) {
    OH_AI_DeviceInfoHandle cpu_device_info = OH_AI_DeviceInfoCreate(OH_AI_DEVICETYPE_CPU);
    ASSERT_NE(cpu_device_info, nullptr);
    OH_AI_DeviceType device_type = OH_AI_DeviceInfoGetDeviceType(cpu_device_info);
    printf("==========device_type:%d\n", device_type);
    ASSERT_EQ(device_type, OH_AI_DEVICETYPE_CPU);

    OH_AI_DeviceInfoDestroy(&cpu_device_info);
    ASSERT_EQ(cpu_device_info, nullptr);
}

/*
 * @tc.name: Context_Kirin_Frequency
 * @tc.desc: Verify the OH_AI_DeviceInfoSetFrequency/OH_AI_DeviceInfoGetFrequency function.
 * @tc.type: FUNC
 */
HWTEST(MSLiteTest, Context_Kirin_Frequency, testing::ext::TestSize.Level0) {
    printf("==========Init Context==========\n");
    OH_AI_ContextHandle context = OH_AI_ContextCreate();
    ASSERT_NE(context, nullptr);

    OH_AI_DeviceInfoHandle npu_device_info = OH_AI_DeviceInfoCreate(OH_AI_DEVICETYPE_KIRIN_NPU);
    ASSERT_NE(npu_device_info, nullptr);
    OH_AI_DeviceInfoSetFrequency(npu_device_info, 1);
    int frequency = OH_AI_DeviceInfoGetFrequency(npu_device_info);
    printf("==========frequency:%d\n", frequency);
    ASSERT_EQ(frequency, 1);
    OH_AI_ContextAddDeviceInfo(context, npu_device_info);

    OH_AI_ContextDestroy(&context);
    ASSERT_EQ(context, nullptr);
}

/*
 * @tc.name: Model_BuildByBuffer
 * @tc.desc: Verify the OH_AI_ModelBuild function.
 * @tc.type: FUNC
 */
HWTEST(MSLiteTest, Model_BuildByBuffer, testing::ext::TestSize.Level0) {
    printf("==========Init Context==========\n");
    OH_AI_ContextHandle context = OH_AI_ContextCreate();
    ASSERT_NE(context, nullptr);

    AddContextDeviceCPU(context);
    printf("==========Create model==========\n");
    OH_AI_ModelHandle model = OH_AI_ModelCreate();
    ASSERT_NE(model, nullptr);
    ModelPredict(model, context, "ml_face_isface", {}, true, true, false);
}

/*
 * @tc.name: Model_GetOutputs
 * @tc.desc: Verify the OH_AI_ModelGetOutputs/OH_AI_ModelGetInputs function.
 * @tc.type: FUNC
 */
HWTEST(MSLiteTest, Model_GetOutputs, testing::ext::TestSize.Level0) {
    printf("==========Init Context==========\n");
    OH_AI_ContextHandle context = OH_AI_ContextCreate();
    ASSERT_NE(context, nullptr);

    AddContextDeviceCPU(context);
    printf("==========Create model==========\n");
    OH_AI_ModelHandle model = OH_AI_ModelCreate();
    ASSERT_NE(model, nullptr);
    printf("==========Model build==========\n");
    OH_AI_ModelBuildFromFile(model, "/data/test/resource/ml_face_isface.ms", OH_AI_MODELTYPE_MINDIR, context);

    printf("==========Model Predict==========\n");
    OH_AI_TensorHandleArray inputs = OH_AI_ModelGetInputs(model);
    OH_AI_TensorHandleArray output;
    FillInputsData(inputs, "ml_face_isface", false);
    OH_AI_Status ret = OH_AI_ModelPredict(model, inputs, &output, nullptr, nullptr);
    ASSERT_EQ(ret, OH_AI_STATUS_SUCCESS);

    printf("==========GetOutput==========\n");
    OH_AI_TensorHandleArray outputs = OH_AI_ModelGetOutputs(model);
    for (size_t i = 0; i < outputs.handle_num; ++i) {
        OH_AI_TensorHandle tensor = outputs.handle_list[i];
        int64_t elementNum = OH_AI_TensorGetElementNum(tensor);
        printf("Tensor name: %s, elements num: %" PRId64 ".\n", OH_AI_TensorGetName(tensor), elementNum);
        float *outputData = reinterpret_cast<float *>(OH_AI_TensorGetMutableData(tensor));
        printf("output data is:");
        constexpr int printNum = 20;
        for (int j = 0; j < elementNum && j <= printNum; ++j) {
            printf("%f ", outputData[j]);
        }
        printf("\n");
        printf("==========compFp32WithTData==========\n");
        std::string expectedDataFile = g_testFilesDir + "ml_face_isface" + std::to_string(i) + ".output";
        bool result = compFp32WithTData(outputData, expectedDataFile, 0.01, 0.01, false);
        EXPECT_EQ(result, true);
    }
}

/*
 * @tc.name: Model_Resize
 * @tc.desc: Verify the OH_AI_ModelResize function.
 * @tc.type: FUNC
 */
HWTEST(MSLiteTest, Model_Resize, testing::ext::TestSize.Level0) {
    printf("==========Init Context==========\n");
    OH_AI_ContextHandle context = OH_AI_ContextCreate();
    ASSERT_NE(context, nullptr);

    AddContextDeviceCPU(context);
    printf("==========Create model==========\n");
    OH_AI_ModelHandle model = OH_AI_ModelCreate();
    ASSERT_NE(model, nullptr);
    ModelPredict(model, context, "ml_ocr_cn", {4, {1, 32, 512, 1}}, false, true, false);
}

/*
 * @tc.name: Model_GetInputByTensorName
 * @tc.desc: Verify the OH_AI_ModelGetInputByTensorName function.
 * @tc.type: FUNC
 */
HWTEST(MSLiteTest, Model_GetInputByTensorName, testing::ext::TestSize.Level0) {
    printf("==========ReadFile==========\n");
    size_t size1;
    size_t *ptrSize1 = &size1;
    const char *imagePath = "/data/test/resource/ml_face_isface.input";
    char *imageBuf = ReadFile(imagePath, ptrSize1);
    ASSERT_NE(imageBuf, nullptr);

    printf("==========Init Context==========\n");
    OH_AI_ContextHandle context = OH_AI_ContextCreate();
    ASSERT_NE(context, nullptr);
    AddContextDeviceCPU(context);
    printf("==========Create model==========\n");
    OH_AI_ModelHandle model = OH_AI_ModelCreate();
    ASSERT_NE(model, nullptr);
    printf("==========Build model==========\n");
    OH_AI_Status ret = OH_AI_ModelBuildFromFile(model, "/data/test/resource/ml_face_isface.ms", OH_AI_MODELTYPE_MINDIR,
                                                context);
    printf("==========build model return code:%d\n", ret);
    ASSERT_EQ(ret, OH_AI_STATUS_SUCCESS);

    printf("==========GetInputs==========\n");
    OH_AI_TensorHandle tensor = OH_AI_ModelGetInputByTensorName(model, "data");
    ASSERT_NE(tensor, nullptr);
    int64_t elementNum = OH_AI_TensorGetElementNum(tensor);
    printf("Tensor name: %s, elements num: %" PRId64 ".\n", OH_AI_TensorGetName(tensor), elementNum);
    float *inputData = reinterpret_cast<float *>(OH_AI_TensorGetMutableData(tensor));
    ASSERT_NE(inputData, nullptr);
    printf("==========Transpose==========\n");
    size_t shapeNum;
    const int64_t *shape = OH_AI_TensorGetShape(tensor, &shapeNum);
    auto imageBufNhwc = new char[size1];
    PackNCHWToNHWCFp32(imageBuf, imageBufNhwc, shape[0], shape[1] * shape[2], shape[3]);
    errno_t mRet = memcpy_s(inputData, size1, imageBufNhwc, size1);
    if (mRet != EOK) {
        printf("memcpy_s failed, ret: %d\n", mRet);
    }
    printf("input data is:");
    constexpr int printNum = 20;
    for (int j = 0; j < elementNum && j <= printNum; ++j) {
        printf("%f ", inputData[j]);
    }
    printf("\n");

    printf("==========Model Predict==========\n");
    OH_AI_TensorHandleArray inputs = OH_AI_ModelGetInputs(model);
    ASSERT_NE(inputs.handle_list, nullptr);
    OH_AI_TensorHandleArray outputs;
    ret = OH_AI_ModelPredict(model, inputs, &outputs, nullptr, nullptr);
    ASSERT_EQ(ret, OH_AI_STATUS_SUCCESS);
    CompareResult(outputs, "ml_face_isface");
    delete[] imageBuf;
    OH_AI_ModelDestroy(&model);
}

/*
 * @tc.name: Model_GetOutputByTensorName
 * @tc.desc: Verify the OH_AI_ModelGetOutputByTensorName function.
 * @tc.type: FUNC
 */
HWTEST(MSLiteTest, Model_GetOutputByTensorName, testing::ext::TestSize.Level0) {
    printf("==========Init Context==========\n");
    OH_AI_ContextHandle context = OH_AI_ContextCreate();
    ASSERT_NE(context, nullptr);
    AddContextDeviceCPU(context);
    printf("==========Create model==========\n");
    OH_AI_ModelHandle model = OH_AI_ModelCreate();
    ASSERT_NE(model, nullptr);
    printf("==========Build model==========\n");
    OH_AI_Status ret = OH_AI_ModelBuildFromFile(model, "/data/test/resource/ml_face_isface.ms", OH_AI_MODELTYPE_MINDIR,
                                                context);
    printf("==========build model return code:%d\n", ret);
    ASSERT_EQ(ret, OH_AI_STATUS_SUCCESS);

    printf("==========GetInputs==========\n");
    OH_AI_TensorHandleArray inputs = OH_AI_ModelGetInputs(model);
    ASSERT_NE(inputs.handle_list, nullptr);
    FillInputsData(inputs, "ml_face_isface", true);
    printf("==========Model Predict==========\n");
    OH_AI_TensorHandleArray outputs;
    ret = OH_AI_ModelPredict(model, inputs, &outputs, nullptr, nullptr);
    ASSERT_EQ(ret, OH_AI_STATUS_SUCCESS);

    printf("==========GetOutput==========\n");
    OH_AI_TensorHandle tensor = OH_AI_ModelGetOutputByTensorName(model, "prob");
    ASSERT_NE(tensor, nullptr);
    int64_t elementNum = OH_AI_TensorGetElementNum(tensor);
    printf("Tensor name: %s, elements num: %" PRId64 ".\n", OH_AI_TensorGetName(tensor), elementNum);
    float *outputData = reinterpret_cast<float *>(OH_AI_TensorGetMutableData(tensor));
    printf("output data is:");
    constexpr int printNum = 20;
    for (int j = 0; j < elementNum && j <= printNum; ++j) {
        printf("%f ", outputData[j]);
    }
    printf("\n");
    printf("==========compFp32WithTData==========\n");
    bool result = compFp32WithTData(outputData, g_testFilesDir + "ml_face_isface0.output", 0.01, 0.01, false);
    EXPECT_EQ(result, true);
    OH_AI_ModelDestroy(&model);
}

/*
 * @tc.name: TrainCfg_CreateDestroy
 * @tc.desc: Verify the OH_AI_TrainCfgCreate/OH_AI_TrainCfgDestroy function.
 * @tc.type: FUNC
 */
HWTEST(MSLiteTest, TrainCfg_CreateDestroy, testing::ext::TestSize.Level0) {
    OH_AI_TrainCfgHandle trainCfg = OH_AI_TrainCfgCreate();
    ASSERT_NE(trainCfg, nullptr);

    OH_AI_TrainCfgDestroy(&trainCfg);
    ASSERT_EQ(trainCfg, nullptr);
}

/*
 * @tc.name: TrainCfg_LossName
 * @tc.desc: Verify the OH_AI_TrainCfgSetLossName/OH_AI_TrainCfgGetLossName function.
 * @tc.type: FUNC
 */
HWTEST(MSLiteTest, TrainCfg_LossName, testing::ext::TestSize.Level0) {
    OH_AI_TrainCfgHandle trainCfg = OH_AI_TrainCfgCreate();
    ASSERT_NE(trainCfg, nullptr);
    std::vector<std::string> set_train_cfg_loss_name = {"loss_fct"};
    char **setLossName = TransStrVectorToCharArrays(set_train_cfg_loss_name);
    OH_AI_TrainCfgSetLossName(trainCfg, const_cast<const char **>(setLossName), set_train_cfg_loss_name.size());

    size_t getNum = 0;
    char **getLossName = OH_AI_TrainCfgGetLossName(trainCfg, &getNum);
    printf("trainCfg loss name: ");
    for (size_t i = 0; i < getNum; i++) {
        printf("%s ", getLossName[i]);
    }
    printf("\n");
    ASSERT_EQ(strcmp(getLossName[0], "loss_fct"), 0);

    for (size_t i = 0; i < getNum; i++) {
        free(setLossName[i]);
        free(getLossName[i]);
    }
    free(setLossName);
    free(getLossName);
    OH_AI_TrainCfgDestroy(&trainCfg);
}

/*
 * @tc.name: TrainCfg_OptimizationLevel
 * @tc.desc: Verify the OH_AI_TrainCfgSetOptimizationLevel/OH_AI_TrainCfgGetOptimizationLevel function.
 * @tc.type: FUNC
 */
HWTEST(MSLiteTest, TrainCfg_OptimizationLevel, testing::ext::TestSize.Level0) {
    OH_AI_TrainCfgHandle trainCfg = OH_AI_TrainCfgCreate();
    ASSERT_NE(trainCfg, nullptr);

    OH_AI_OptimizationLevel optim_level = OH_AI_KO2;
    OH_AI_TrainCfgSetOptimizationLevel(trainCfg, optim_level);
    OH_AI_OptimizationLevel get_optim_level = OH_AI_TrainCfgGetOptimizationLevel(trainCfg);
    ASSERT_EQ(get_optim_level, OH_AI_KO2);

    OH_AI_TrainCfgDestroy(&trainCfg);
    ASSERT_EQ(trainCfg, nullptr);
}

/*
 * @tc.name: Model_TrainModelBuild
 * @tc.desc: Verify the OH_AI_TrainModelBuild function.
 * @tc.type: FUNC
 */
HWTEST(MSLiteTest, Model_TrainModelBuild, testing::ext::TestSize.Level0) {
    printf("==========OH_AI_ContextCreate==========\n");
    OH_AI_ContextHandle context = OH_AI_ContextCreate();
    ASSERT_NE(context, nullptr);
    AddContextDeviceCPU(context);
    printf("==========OH_AI_ModelCreate==========\n");
    OH_AI_ModelHandle model = OH_AI_ModelCreate();
    ASSERT_NE(model, nullptr);

    printf("==========OH_AI_RunStep==========\n");
    ModelTrain(model, context, "lenet_train", {}, true, false, false);
    printf("==========OH_AI_ExportModel==========\n");
    auto status = OH_AI_ExportModel(model, OH_AI_MODELTYPE_MINDIR, "/data/test/resource/lenet_train_infer.ms",
                                    OH_AI_NO_QUANT, true, nullptr, 0);
    ASSERT_EQ(status, OH_AI_STATUS_SUCCESS);
    OH_AI_ModelDestroy(&model);

    printf("==========OH_AI_ModelCreate2==========\n");
    context = OH_AI_ContextCreate();
    ASSERT_NE(context, nullptr);
    AddContextDeviceCPU(context);
    OH_AI_ModelHandle model2 = OH_AI_ModelCreate();
    ASSERT_NE(model2, nullptr);
    printf("==========ModelPredict==========\n");
    ModelPredict(model2, context, "lenet_train_infer", {}, true, false, true);
}

/*
 * @tc.name: Model_LearningRate
 * @tc.desc: Verify the OH_AI_ModelGetLearningRate/OH_AI_ModelSetLearningRate function.
 * @tc.type: FUNC
 */
HWTEST(MSLiteTest, Model_LearningRate, testing::ext::TestSize.Level0) {
    printf("==========OH_AI_ContextCreate==========\n");
    OH_AI_ContextHandle context = OH_AI_ContextCreate();
    ASSERT_NE(context, nullptr);
    AddContextDeviceCPU(context);
    printf("==========OH_AI_ModelCreate==========\n");
    OH_AI_ModelHandle model = OH_AI_ModelCreate();
    ASSERT_NE(model, nullptr);

    printf("==========OH_AI_TrainCfgCreate==========\n");
    OH_AI_TrainCfgHandle train_cfg = OH_AI_TrainCfgCreate();
    ASSERT_NE(train_cfg, nullptr);
    printf("==========OH_AI_TrainModelBuildFromFile==========\n");
    auto status = OH_AI_TrainModelBuildFromFile(model, "/data/test/resource/lenet_train.ms", OH_AI_MODELTYPE_MINDIR,
                                                context, train_cfg);
    ASSERT_EQ(status, OH_AI_STATUS_SUCCESS);

    auto learing_rate = OH_AI_ModelGetLearningRate(model);
    printf("learing_rate: %f\n", learing_rate);
    status = OH_AI_ModelSetLearningRate(model, 0.01f);
    ASSERT_EQ(status, OH_AI_STATUS_SUCCESS);
    learing_rate = OH_AI_ModelGetLearningRate(model);
    printf("get_learing_rate: %f", learing_rate);
    ASSERT_EQ(learing_rate, 0.01f);

    printf("==========GetInputs==========\n");
    OH_AI_TensorHandleArray inputs = OH_AI_ModelGetInputs(model);
    ASSERT_NE(inputs.handle_list, nullptr);
    FillInputsData(inputs, "lenet_train", false);
    status = OH_AI_ModelSetTrainMode(model, true);
    ASSERT_EQ(status, OH_AI_STATUS_SUCCESS);
    printf("==========Model RunStep==========\n");
    status = OH_AI_RunStep(model, nullptr, nullptr);
    ASSERT_EQ(status, OH_AI_STATUS_SUCCESS);

    OH_AI_ModelDestroy(&model);
}

/*
 * @tc.name: Model_UpdateWeights
 * @tc.desc: Verify the OH_AI_ModelUpdateWeights function.
 * @tc.type: FUNC
 */
HWTEST(MSLiteTest, Model_UpdateWeights, testing::ext::TestSize.Level0) {
    printf("==========OH_AI_ContextCreate==========\n");
    OH_AI_ContextHandle context = OH_AI_ContextCreate();
    ASSERT_NE(context, nullptr);
    AddContextDeviceCPU(context);
    printf("==========OH_AI_ModelCreate==========\n");
    OH_AI_ModelHandle model = OH_AI_ModelCreate();
    ASSERT_NE(model, nullptr);

    printf("==========OH_AI_TrainCfgCreate==========\n");
    OH_AI_TrainCfgHandle train_cfg = OH_AI_TrainCfgCreate();
    ASSERT_NE(train_cfg, nullptr);
    printf("==========OH_AI_TrainModelBuildFromFile==========\n");
    auto status = OH_AI_TrainModelBuildFromFile(model, "/data/test/resource/lenet_train.ms", OH_AI_MODELTYPE_MINDIR,
                                                context, train_cfg);
    ASSERT_EQ(status, OH_AI_STATUS_SUCCESS);

    auto genRandomData = [](size_t size, void *data) {
        auto generator = std::uniform_real_distribution<float>(0.0f, 1.0f);
        std::mt19937 randomEngine;
        size_t elementsNum = size / sizeof(float);
        (void)std::generate_n(static_cast<float *>(data), elementsNum,
                              [&]() { return static_cast<float>(generator(randomEngine)); });
    };
    std::vector<OH_AI_TensorHandle> vec_inputs;
    constexpr size_t createShapeNum = 1;
    int64_t createShape[createShapeNum] = {10};
    OH_AI_TensorHandle tensor = OH_AI_TensorCreate("fc3.bias", OH_AI_DATATYPE_NUMBERTYPE_FLOAT32, createShape,
                                                   createShapeNum, nullptr, 0);
    ASSERT_NE(tensor, nullptr);
    genRandomData(OH_AI_TensorGetDataSize(tensor), OH_AI_TensorGetMutableData(tensor));
    vec_inputs.push_back(tensor);

    OH_AI_TensorHandleArray update_weights = {1, vec_inputs.data()};
    status = OH_AI_ModelUpdateWeights(model, update_weights);
    ASSERT_EQ(status, OH_AI_STATUS_SUCCESS);
    printf("==========GetInputs==========\n");
    OH_AI_TensorHandleArray inputs = OH_AI_ModelGetInputs(model);
    ASSERT_NE(inputs.handle_list, nullptr);
    FillInputsData(inputs, "lenet_train", false);
    status = OH_AI_ModelSetTrainMode(model, true);
    ASSERT_EQ(status, OH_AI_STATUS_SUCCESS);
    printf("==========Model RunStep==========\n");
    status = OH_AI_RunStep(model, nullptr, nullptr);
    ASSERT_EQ(status, OH_AI_STATUS_SUCCESS);

    OH_AI_ModelDestroy(&model);
}

/*
 * @tc.name: Model_GetWeights
 * @tc.desc: Verify the OH_AI_ModelGetWeights function.
 * @tc.type: FUNC
 */
HWTEST(MSLiteTest, Model_GetWeights, testing::ext::TestSize.Level0) {
    printf("==========OH_AI_ContextCreate==========\n");
    OH_AI_ContextHandle context = OH_AI_ContextCreate();
    ASSERT_NE(context, nullptr);
    AddContextDeviceCPU(context);
    printf("==========OH_AI_ModelCreate==========\n");
    OH_AI_ModelHandle model = OH_AI_ModelCreate();
    ASSERT_NE(model, nullptr);
    printf("==========OH_AI_TrainCfgCreate==========\n");
    OH_AI_TrainCfgHandle train_cfg = OH_AI_TrainCfgCreate();
    ASSERT_NE(train_cfg, nullptr);
    printf("==========OH_AI_TrainModelBuildFromFile==========\n");
    auto status = OH_AI_TrainModelBuildFromFile(model, "/data/test/resource/lenet_train.ms", OH_AI_MODELTYPE_MINDIR,
                                                context, train_cfg);
    ASSERT_EQ(status, OH_AI_STATUS_SUCCESS);
    OH_AI_TensorHandleArray get_update_weights = OH_AI_ModelGetWeights(model);
    for (size_t i = 0; i < get_update_weights.handle_num; ++i) {
        OH_AI_TensorHandle weights_tensor = get_update_weights.handle_list[i];
        if (strcmp(OH_AI_TensorGetName(weights_tensor), "fc3.bias") == 0) {
            float *inputData = reinterpret_cast<float *>(OH_AI_TensorGetMutableData(weights_tensor));
            printf("fc3.bias: %f", inputData[0]);
        }
    }
    auto genRandomData = [](size_t size, void *data) {
        auto generator = std::uniform_real_distribution<float>(0.0f, 1.0f);
        std::mt19937 randomEngine;
        size_t elementsNum = size / sizeof(float);
        (void)std::generate_n(static_cast<float *>(data), elementsNum,
                              [&]() { return static_cast<float>(generator(randomEngine)); });
    };
    std::vector<OH_AI_TensorHandle> vec_inputs;
    constexpr size_t createShapeNum = 1;
    int64_t createShape[createShapeNum] = {10};
    OH_AI_TensorHandle tensor = OH_AI_TensorCreate("fc3.bias", OH_AI_DATATYPE_NUMBERTYPE_FLOAT32, createShape,
                                                   createShapeNum, nullptr, 0);
    ASSERT_NE(tensor, nullptr);
    genRandomData(OH_AI_TensorGetDataSize(tensor), OH_AI_TensorGetMutableData(tensor));
    vec_inputs.push_back(tensor);
    OH_AI_TensorHandleArray update_weights = {1, vec_inputs.data()};
    status = OH_AI_ModelUpdateWeights(model, update_weights);
    ASSERT_EQ(status, OH_AI_STATUS_SUCCESS);
    printf("==========GetInputs==========\n");
    OH_AI_TensorHandleArray inputs = OH_AI_ModelGetInputs(model);
    ASSERT_NE(inputs.handle_list, nullptr);
    FillInputsData(inputs, "lenet_train", false);
    status = OH_AI_ModelSetTrainMode(model, true);
    ASSERT_EQ(status, OH_AI_STATUS_SUCCESS);
    printf("==========Model RunStep==========\n");
    status = OH_AI_RunStep(model, nullptr, nullptr);
    ASSERT_EQ(status, OH_AI_STATUS_SUCCESS);
    printf("==========OH_AI_ExportModel==========\n");
    status = OH_AI_ExportModel(model, OH_AI_MODELTYPE_MINDIR, "/data/test/resource/lenet_train_infer.ms",
                               OH_AI_NO_QUANT, true, nullptr, 0);
    ASSERT_EQ(status, OH_AI_STATUS_SUCCESS);
    OH_AI_TensorHandleArray export_update_weights = OH_AI_ModelGetWeights(model);
    for (size_t i = 0; i < export_update_weights.handle_num; ++i) {
        OH_AI_TensorHandle weights_tensor = export_update_weights.handle_list[i];
        if (strcmp(OH_AI_TensorGetName(weights_tensor), "fc3.bias") == 0) {
            float *inputData = reinterpret_cast<float *>(OH_AI_TensorGetMutableData(weights_tensor));
            printf("fc3.bias: %f", inputData[0]);
        }
    }

    OH_AI_ModelDestroy(&model);
}

/*
 * @tc.name: Model_SetupVirtualBatch
 * @tc.desc: Verify the OH_AI_ModelSetupVirtualBatch/OH_AI_ExportModelBuffer function.
 * @tc.type: FUNC
 */
HWTEST(MSLiteTest, Model_SetupVirtualBatch, testing::ext::TestSize.Level0) {
    printf("==========OH_AI_ContextCreate==========\n");
    OH_AI_ContextHandle context = OH_AI_ContextCreate();
    ASSERT_NE(context, nullptr);
    AddContextDeviceCPU(context);
    printf("==========OH_AI_ModelCreate==========\n");
    OH_AI_ModelHandle model = OH_AI_ModelCreate();
    ASSERT_NE(model, nullptr);
    printf("==========OH_AI_TrainCfgCreate==========\n");
    OH_AI_TrainCfgHandle train_cfg = OH_AI_TrainCfgCreate();
    ASSERT_NE(train_cfg, nullptr);

    printf("==========OH_AI_TrainModelBuildFromFile==========\n");
    auto status = OH_AI_TrainModelBuildFromFile(model, "/data/test/resource/lenet_train.ms", OH_AI_MODELTYPE_MINDIR,
                                                context, train_cfg);
    ASSERT_EQ(status, OH_AI_STATUS_SUCCESS);
    status = OH_AI_ModelSetupVirtualBatch(model, 2, -1.0f, -1.0f);
    ASSERT_EQ(status, OH_AI_STATUS_SUCCESS);
    printf("==========GetInputs==========\n");
    OH_AI_TensorHandleArray inputs = OH_AI_ModelGetInputs(model);
    ASSERT_NE(inputs.handle_list, nullptr);
    FillInputsData(inputs, "lenet_train", false);
    status = OH_AI_ModelSetTrainMode(model, true);
    ASSERT_EQ(status, OH_AI_STATUS_SUCCESS);
    bool trainMode = OH_AI_ModelGetTrainMode(model);
    printf("get train mode: %d\n", trainMode);
    ASSERT_EQ(trainMode, true);

    printf("==========Model RunStep==========\n");
    status = OH_AI_RunStep(model, nullptr, nullptr);
    ASSERT_EQ(status, OH_AI_STATUS_SUCCESS);
    printf("==========OH_AI_ExportModelBuffer==========\n");
    char *modelBuffer;
    size_t modelSize = 0;
    status = OH_AI_ExportModelBuffer(model, OH_AI_MODELTYPE_MINDIR, &modelBuffer, &modelSize, OH_AI_NO_QUANT, true,
                                     nullptr, 0);
    printf("export model buffer size: %zu\n", modelSize);
    ASSERT_EQ(status, OH_AI_STATUS_SUCCESS);
    ASSERT_NE(modelBuffer, nullptr);

    OH_AI_ModelDestroy(&model);
    free(modelBuffer);
}

/*
 * @tc.name: Model_ExportWeights
 * @tc.desc: Verify the OH_AI_ExportWeightsCollaborateWithMicro function.
 * @tc.type: FUNC
 */
HWTEST(MSLiteTest, Model_ExportWeights, testing::ext::TestSize.Level0) {
    printf("==========OH_AI_ContextCreate==========\n");
    OH_AI_ContextHandle context = OH_AI_ContextCreate();
    ASSERT_NE(context, nullptr);
    AddContextDeviceCPU(context);
    printf("==========OH_AI_ModelCreate==========\n");
    OH_AI_ModelHandle model = OH_AI_ModelCreate();
    ASSERT_NE(model, nullptr);
    printf("==========OH_AI_TrainCfgCreate==========\n");
    OH_AI_TrainCfgHandle train_cfg = OH_AI_TrainCfgCreate();
    ASSERT_NE(train_cfg, nullptr);

    printf("==========OH_AI_TrainModelBuildFromFile==========\n");
    auto status = OH_AI_TrainModelBuildFromFile(model, "/data/test/resource/xiaoyi_train_codegen.ms",
                                                OH_AI_MODELTYPE_MINDIR, context, train_cfg);
    ASSERT_EQ(status, OH_AI_STATUS_SUCCESS);
    printf("==========OH_AI_ExportModel==========\n");
    status = OH_AI_ExportModel(model, OH_AI_MODELTYPE_MINDIR, "/data/test/resource/xiaoyi_train_codegen_gru_model1.ms",
                               OH_AI_NO_QUANT, true, nullptr, 0);
    ASSERT_EQ(status, OH_AI_STATUS_SUCCESS);
    status = OH_AI_ExportWeightsCollaborateWithMicro(model, OH_AI_MODELTYPE_MINDIR,
                                                     "/data/test/resource/xiaoyi_train_codegen_net1.bin", true, true,
                                                     nullptr, 0);
    ASSERT_EQ(status, OH_AI_STATUS_SUCCESS);
    status = OH_AI_ExportWeightsCollaborateWithMicro(model, OH_AI_MODELTYPE_MINDIR,
                                                     "/data/test/resource/xiaoyi_train_codegen_net1_fp32.bin", true,
                                                     false, nullptr, 0);
    ASSERT_EQ(status, OH_AI_STATUS_SUCCESS);

    OH_AI_ModelDestroy(&model);
}

/*
 * @tc.name: Tensor_Create
 * @tc.desc: Verify the OH_AI_TensorCreate function.
 * @tc.type: FUNC
 */
HWTEST(MSLiteTest, Tensor_Create, testing::ext::TestSize.Level0) {
    printf("==========Init Context==========\n");
    OH_AI_ContextHandle context = OH_AI_ContextCreate();
    ASSERT_NE(context, nullptr);
    AddContextDeviceCPU(context);
    printf("==========Create model==========\n");
    OH_AI_ModelHandle model = OH_AI_ModelCreate();
    ASSERT_NE(model, nullptr);
    printf("==========Build model==========\n");
    OH_AI_Status ret = OH_AI_ModelBuildFromFile(model, "/data/test/resource/ml_face_isface.ms", OH_AI_MODELTYPE_MINDIR,
                                                context);
    printf("==========build model return code:%d\n", ret);
    ASSERT_EQ(ret, OH_AI_STATUS_SUCCESS);
    printf("==========GetInputs==========\n");
    constexpr size_t createShapeNum = 4;
    int64_t createShape[createShapeNum] = {1, 48, 48, 3};
    OH_AI_TensorHandle tensor = OH_AI_TensorCreate("data", OH_AI_DATATYPE_NUMBERTYPE_FLOAT32, createShape,
                                                   createShapeNum, nullptr, 0);
    ASSERT_NE(tensor, nullptr);
    OH_AI_TensorHandleArray inputs = OH_AI_ModelGetInputs(model);
    inputs.handle_list[0] = tensor;
    FillInputsData(inputs, "ml_face_isface", true);
    printf("==========Model Predict==========\n");
    OH_AI_TensorHandleArray outputs;
    ret = OH_AI_ModelPredict(model, inputs, &outputs, nullptr, nullptr);
    ASSERT_EQ(ret, OH_AI_STATUS_SUCCESS);
    CompareResult(outputs, "ml_face_isface");
    OH_AI_ModelDestroy(&model);
}

/*
 * @tc.name: Tensor_Destroy
 * @tc.desc: Verify the OH_AI_TensorDestroy function.
 * @tc.type: FUNC
 */
HWTEST(MSLiteTest, Tensor_Destroy, testing::ext::TestSize.Level0) {
    printf("==========ReadFile==========\n");
    size_t size1;
    size_t *ptrSize1 = &size1;
    const char *imagePath = "/data/test/resource/ml_face_isface.input";
    char *imageBuf = ReadFile(imagePath, ptrSize1);
    ASSERT_NE(imageBuf, nullptr);
    printf("==========OH_AI_TensorCreate==========\n");
    constexpr size_t createShapeNum = 4;
    int64_t createShape[createShapeNum] = {1, 48, 48, 3};
    OH_AI_TensorHandle tensor = OH_AI_TensorCreate("data", OH_AI_DATATYPE_NUMBERTYPE_FLOAT32, createShape,
                                                   createShapeNum, imageBuf, size1);
    ASSERT_NE(tensor, nullptr);
    delete[] imageBuf;
    OH_AI_TensorDestroy(&tensor);
    ASSERT_EQ(tensor, nullptr);
}

/*
 * @tc.name: Tensor_Clone
 * @tc.desc: Verify the OH_AI_TensorClone function.
 * @tc.type: FUNC
 */
HWTEST(MSLiteTest, Tensor_Clone, testing::ext::TestSize.Level0) {
    printf("==========ReadFile==========\n");
    size_t size1;
    size_t *ptrSize1 = &size1;
    const char *imagePath = "/data/test/resource/ml_face_isface.input";
    char *imageBuf = ReadFile(imagePath, ptrSize1);
    ASSERT_NE(imageBuf, nullptr);
    printf("==========OH_AI_TensorCreate==========\n");
    constexpr size_t createShapeNum = 4;
    int64_t createShape[createShapeNum] = {1, 48, 48, 3};
    OH_AI_TensorHandle tensor = OH_AI_TensorCreate("data", OH_AI_DATATYPE_NUMBERTYPE_FLOAT32, createShape,
                                                   createShapeNum, imageBuf, size1);
    ASSERT_NE(tensor, nullptr);
    OH_AI_TensorHandle clone = OH_AI_TensorClone(tensor);
    ASSERT_NE(clone, nullptr);
    ASSERT_EQ(strcmp(OH_AI_TensorGetName(clone), "data_duplicate"), 0);
    delete[] imageBuf;
    OH_AI_TensorDestroy(&tensor);
    OH_AI_TensorDestroy(&clone);
}

/*
 * @tc.name: Tensor_GetName
 * @tc.desc: Verify the OH_AI_TensorGetName function.
 * @tc.type: FUNC
 */
HWTEST(MSLiteTest, Tensor_GetName, testing::ext::TestSize.Level0) {
    printf("==========ReadFile==========\n");
    size_t size1;
    size_t *ptrSize1 = &size1;
    const char *imagePath = "/data/test/resource/ml_face_isface.input";
    char *imageBuf = ReadFile(imagePath, ptrSize1);
    ASSERT_NE(imageBuf, nullptr);
    printf("==========OH_AI_TensorCreate==========\n");
    constexpr size_t createShapeNum = 4;
    int64_t createShape[createShapeNum] = {1, 48, 48, 3};
    OH_AI_TensorHandle tensor = OH_AI_TensorCreate("data", OH_AI_DATATYPE_NUMBERTYPE_FLOAT32, createShape,
                                                   createShapeNum, imageBuf, size1);
    ASSERT_NE(tensor, nullptr);
    const char *tensorName = OH_AI_TensorGetName(tensor);
    ASSERT_EQ(strcmp(tensorName, "data"), 0);
    delete[] imageBuf;
    OH_AI_TensorDestroy(&tensor);
}

/*
 * @tc.name: Tensor_SetName
 * @tc.desc: Verify the OH_AI_TensorSetName function.
 * @tc.type: FUNC
 */
HWTEST(MSLiteTest, Tensor_SetName, testing::ext::TestSize.Level0) {
    printf("==========ReadFile==========\n");
    size_t size1;
    size_t *ptrSize1 = &size1;
    const char *imagePath = "/data/test/resource/ml_face_isface.input";
    char *imageBuf = ReadFile(imagePath, ptrSize1);
    ASSERT_NE(imageBuf, nullptr);
    printf("==========OH_AI_TensorCreate==========\n");
    constexpr size_t createShapeNum = 4;
    int64_t createShape[createShapeNum] = {1, 48, 48, 3};
    OH_AI_TensorHandle tensor = OH_AI_TensorCreate("data", OH_AI_DATATYPE_NUMBERTYPE_FLOAT32, createShape,
                                                   createShapeNum, imageBuf, size1);
    ASSERT_NE(tensor, nullptr);
    OH_AI_TensorSetName(tensor, "new_data");
    const char *tensorName = OH_AI_TensorGetName(tensor);
    ASSERT_EQ(strcmp(tensorName, "new_data"), 0);
    delete[] imageBuf;
    OH_AI_TensorDestroy(&tensor);
}

/*
 * @tc.name: Tensor_GetDataType
 * @tc.desc: Verify the OH_AI_TensorGetDataType function.
 * @tc.type: FUNC
 */
HWTEST(MSLiteTest, Tensor_GetDataType, testing::ext::TestSize.Level0) {
    printf("==========ReadFile==========\n");
    size_t size1;
    size_t *ptrSize1 = &size1;
    const char *imagePath = "/data/test/resource/ml_face_isface.input";
    char *imageBuf = ReadFile(imagePath, ptrSize1);
    ASSERT_NE(imageBuf, nullptr);
    printf("==========OH_AI_TensorCreate==========\n");
    constexpr size_t createShapeNum = 4;
    int64_t createShape[createShapeNum] = {1, 48, 48, 3};
    OH_AI_TensorHandle tensor = OH_AI_TensorCreate("data", OH_AI_DATATYPE_NUMBERTYPE_FLOAT32, createShape,
                                                   createShapeNum, imageBuf, size1);
    ASSERT_NE(tensor, nullptr);
    OH_AI_DataType data_type = OH_AI_TensorGetDataType(tensor);
    ASSERT_EQ(data_type, OH_AI_DATATYPE_NUMBERTYPE_FLOAT32);
    delete[] imageBuf;
    OH_AI_TensorDestroy(&tensor);
}

/*
 * @tc.name: Tensor_SetDataType
 * @tc.desc: Verify the OH_AI_TensorSetDataType function.
 * @tc.type: FUNC
 */
HWTEST(MSLiteTest, Tensor_SetDataType, testing::ext::TestSize.Level0) {
    printf("==========ReadFile==========\n");
    size_t size1;
    size_t *ptrSize1 = &size1;
    const char *imagePath = "/data/test/resource/ml_face_isface.input";
    char *imageBuf = ReadFile(imagePath, ptrSize1);
    ASSERT_NE(imageBuf, nullptr);
    printf("==========OH_AI_TensorCreate==========\n");
    constexpr size_t createShapeNum = 4;
    int64_t createShape[createShapeNum] = {1, 48, 48, 3};
    OH_AI_TensorHandle tensor = OH_AI_TensorCreate("data", OH_AI_DATATYPE_NUMBERTYPE_FLOAT32, createShape,
                                                   createShapeNum, imageBuf, size1);
    ASSERT_NE(tensor, nullptr);
    OH_AI_TensorSetDataType(tensor, OH_AI_DATATYPE_NUMBERTYPE_FLOAT16);
    OH_AI_DataType data_type = OH_AI_TensorGetDataType(tensor);
    ASSERT_EQ(data_type, OH_AI_DATATYPE_NUMBERTYPE_FLOAT16);
    delete[] imageBuf;
    OH_AI_TensorDestroy(&tensor);
}

/*
 * @tc.name: Tensor_GetShape
 * @tc.desc: Verify the OH_AI_TensorGetShape function.
 * @tc.type: FUNC
 */
HWTEST(MSLiteTest, Tensor_GetShape, testing::ext::TestSize.Level0) {
    printf("==========ReadFile==========\n");
    size_t size1;
    size_t *ptrSize1 = &size1;
    const char *imagePath = "/data/test/resource/ml_face_isface.input";
    char *imageBuf = ReadFile(imagePath, ptrSize1);
    ASSERT_NE(imageBuf, nullptr);
    printf("==========OH_AI_TensorCreate==========\n");
    constexpr size_t createShapeNum = 4;
    int64_t createShape[createShapeNum] = {1, 48, 48, 3};
    OH_AI_TensorHandle tensor = OH_AI_TensorCreate("data", OH_AI_DATATYPE_NUMBERTYPE_FLOAT32, createShape,
                                                   createShapeNum, imageBuf, size1);
    ASSERT_NE(tensor, nullptr);
    size_t retShapeNum;
    const int64_t *retShape = OH_AI_TensorGetShape(tensor, &retShapeNum);
    ASSERT_EQ(retShapeNum, createShapeNum);
    for (size_t i = 0; i < retShapeNum; i++) {
        ASSERT_EQ(retShape[i], createShape[i]);
    }
    delete[] imageBuf;
    OH_AI_TensorDestroy(&tensor);
}

/*
 * @tc.name: Tensor_SetShape
 * @tc.desc: Verify the OH_AI_TensorSetShape function.
 * @tc.type: FUNC
 */
HWTEST(MSLiteTest, Tensor_SetShape, testing::ext::TestSize.Level0) {
    printf("==========ReadFile==========\n");
    size_t size1;
    size_t *ptrSize1 = &size1;
    const char *imagePath = "/data/test/resource/ml_face_isface.input";
    char *imageBuf = ReadFile(imagePath, ptrSize1);
    ASSERT_NE(imageBuf, nullptr);
    printf("==========OH_AI_TensorCreate==========\n");
    constexpr size_t createShapeNum = 4;
    int64_t createShape[createShapeNum] = {1, 48, 48, 3};
    OH_AI_TensorHandle tensor = OH_AI_TensorCreate("data", OH_AI_DATATYPE_NUMBERTYPE_FLOAT32, createShape,
                                                   createShapeNum, imageBuf, size1);
    ASSERT_NE(tensor, nullptr);
    size_t retShapeNum;
    const int64_t *retShape = OH_AI_TensorGetShape(tensor, &retShapeNum);
    ASSERT_EQ(retShapeNum, createShapeNum);
    for (size_t i = 0; i < retShapeNum; i++) {
        ASSERT_EQ(retShape[i], createShape[i]);
    }
    constexpr size_t newShapeNum = 4;
    int64_t newShape[newShapeNum] = {1, 32, 32, 1};
    OH_AI_TensorSetShape(tensor, newShape, newShapeNum);
    size_t newRetShapeNum;
    const int64_t *newRetShape = OH_AI_TensorGetShape(tensor, &newRetShapeNum);
    ASSERT_EQ(newRetShapeNum, newShapeNum);
    for (size_t i = 0; i < newRetShapeNum; i++) {
        ASSERT_EQ(newRetShape[i], newShape[i]);
    }
    delete[] imageBuf;
    OH_AI_TensorDestroy(&tensor);
}

/*
 * @tc.name: Tensor_GetFormat
 * @tc.desc: Verify the OH_AI_TensorGetFormat function.
 * @tc.type: FUNC
 */
HWTEST(MSLiteTest, Tensor_GetFormat, testing::ext::TestSize.Level0) {
    printf("==========ReadFile==========\n");
    size_t size1;
    size_t *ptrSize1 = &size1;
    const char *imagePath = "/data/test/resource/ml_face_isface.input";
    char *imageBuf = ReadFile(imagePath, ptrSize1);
    ASSERT_NE(imageBuf, nullptr);
    printf("==========OH_AI_TensorCreate==========\n");
    constexpr size_t createShapeNum = 4;
    int64_t createShape[createShapeNum] = {1, 48, 48, 3};
    OH_AI_TensorHandle tensor = OH_AI_TensorCreate("data", OH_AI_DATATYPE_NUMBERTYPE_FLOAT32, createShape,
                                                   createShapeNum, imageBuf, size1);
    ASSERT_NE(tensor, nullptr);
    OH_AI_Format data_format = OH_AI_TensorGetFormat(tensor);
    ASSERT_EQ(data_format, OH_AI_FORMAT_NHWC);
    delete[] imageBuf;
    OH_AI_TensorDestroy(&tensor);
}

/*
 * @tc.name: Tensor_SetFormat
 * @tc.desc: Verify the OH_AI_TensorSetFormat function.
 * @tc.type: FUNC
 */
HWTEST(MSLiteTest, Tensor_SetFormat, testing::ext::TestSize.Level0) {
    printf("==========ReadFile==========\n");
    size_t size1;
    size_t *ptrSize1 = &size1;
    const char *imagePath = "/data/test/resource/ml_face_isface.input";
    char *imageBuf = ReadFile(imagePath, ptrSize1);
    ASSERT_NE(imageBuf, nullptr);
    printf("==========OH_AI_TensorCreate==========\n");
    constexpr size_t createShapeNum = 4;
    int64_t createShape[createShapeNum] = {1, 48, 48, 3};
    OH_AI_TensorHandle tensor = OH_AI_TensorCreate("data", OH_AI_DATATYPE_NUMBERTYPE_FLOAT32, createShape,
                                                   createShapeNum, imageBuf, size1);
    ASSERT_NE(tensor, nullptr);
    OH_AI_TensorSetFormat(tensor, OH_AI_FORMAT_NCHW);
    OH_AI_Format data_format = OH_AI_TensorGetFormat(tensor);
    ASSERT_EQ(data_format, OH_AI_FORMAT_NCHW);
    delete[] imageBuf;
    OH_AI_TensorDestroy(&tensor);
}

/*
 * @tc.name: Tensor_GetData
 * @tc.desc: Verify the OH_AI_TensorGetData function.
 * @tc.type: FUNC
 */
HWTEST(MSLiteTest, Tensor_GetData, testing::ext::TestSize.Level0) {
    printf("==========ReadFile==========\n");
    size_t size1;
    size_t *ptrSize1 = &size1;
    const char *imagePath = "/data/test/resource/ml_face_isface.input";
    char *imageBuf = ReadFile(imagePath, ptrSize1);
    ASSERT_NE(imageBuf, nullptr);
    printf("==========OH_AI_TensorCreate==========\n");
    constexpr size_t createShapeNum = 4;
    int64_t createShape[createShapeNum] = {1, 48, 48, 3};
    OH_AI_TensorHandle tensor = OH_AI_TensorCreate("data", OH_AI_DATATYPE_NUMBERTYPE_FLOAT32, createShape,
                                                   createShapeNum, imageBuf, size1);
    ASSERT_NE(tensor, nullptr);
    const float *retData = static_cast<const float *>(OH_AI_TensorGetData(tensor));
    ASSERT_NE(retData, nullptr);
    printf("return data is: ");
    constexpr int printNum = 20;
    for (int i = 0; i < printNum; ++i) {
        printf("%f ", retData[i]);
    }
    printf("\n");
    delete[] imageBuf;
    OH_AI_TensorDestroy(&tensor);
}

/*
 * @tc.name: Tensor_SetData
 * @tc.desc: Verify the OH_AI_TensorSetData function.
 * @tc.type: FUNC
 */
HWTEST(MSLiteTest, Tensor_SetData, testing::ext::TestSize.Level0) {
    printf("==========ReadFile==========\n");
    size_t size1;
    size_t *ptrSize1 = &size1;
    const char *imagePath = "/data/test/resource/ml_face_isface.input";
    char *imageBuf = ReadFile(imagePath, ptrSize1);
    ASSERT_NE(imageBuf, nullptr);
    printf("==========OH_AI_TensorCreate==========\n");
    constexpr size_t createShapeNum = 4;
    int64_t createShape[createShapeNum] = {1, 48, 48, 3};
    OH_AI_TensorHandle tensor = OH_AI_TensorCreate("data", OH_AI_DATATYPE_NUMBERTYPE_FLOAT32, createShape,
                                                   createShapeNum, imageBuf, size1);
    ASSERT_NE(tensor, nullptr);
    constexpr size_t dataLen = 6;
    float data[dataLen] = {1, 2, 3, 4, 5, 6};
    OH_AI_TensorSetData(tensor, data);
    const float *retData = static_cast<const float *>(OH_AI_TensorGetData(tensor));
    ASSERT_NE(retData, nullptr);
    printf("return data is:");
    for (size_t i = 0; i < dataLen; i++) {
        ASSERT_EQ(retData[i], data[i]);
        printf("%f ", retData[i]);
    }
    printf("\n");
    delete[] imageBuf;
    OH_AI_TensorDestroy(&tensor);
}

/*
 * @tc.name: Tensor_SetUserData
 * @tc.desc: Verify the OH_AI_TensorSetData function.
 * @tc.type: FUNC
 */
HWTEST(MSLiteTest, Tensor_SetUserData, testing::ext::TestSize.Level0) {
    printf("==========ReadFile==========\n");
    size_t size1;
    size_t *ptrSize1 = &size1;
    const char *imagePath = "/data/test/resource/ml_face_isface.input";
    char *imageBuf = ReadFile(imagePath, ptrSize1);
    ASSERT_NE(imageBuf, nullptr);
    printf("==========OH_AI_TensorCreate==========\n");
    constexpr size_t createShapeNum = 4;
    int64_t createShape[createShapeNum] = {1, 48, 48, 3};
    OH_AI_TensorHandle tensor = OH_AI_TensorCreate("data", OH_AI_DATATYPE_NUMBERTYPE_FLOAT32, createShape,
                                                   createShapeNum, imageBuf, size1);
    ASSERT_NE(tensor, nullptr);
    float *inputData = reinterpret_cast<float *>(OH_AI_TensorGetMutableData(tensor));
    constexpr size_t dataLen = 6;
    float data[dataLen] = {1, 2, 3, 4, 5, 6};
    for (size_t i = 0; i < dataLen; i++) {
        inputData[i] = data[i];
    }

    OH_AI_TensorSetUserData(tensor, inputData, size1);
    const float *retData = static_cast<const float *>(OH_AI_TensorGetData(tensor));
    ASSERT_NE(retData, nullptr);
    printf("return data is:");
    for (size_t i = 0; i < dataLen; i++) {
        ASSERT_EQ(retData[i], data[i]);
        printf("%f ", retData[i]);
    }
    printf("\n");
    delete[] imageBuf;
    OH_AI_TensorDestroy(&tensor);
}

/*
 * @tc.name: Tensor_GetElementNum
 * @tc.desc: Verify the OH_AI_TensorGetElementNum function.
 * @tc.type: FUNC
 */
HWTEST(MSLiteTest, Tensor_GetElementNum, testing::ext::TestSize.Level0) {
    printf("==========ReadFile==========\n");
    size_t size1;
    size_t *ptrSize1 = &size1;
    const char *imagePath = "/data/test/resource/ml_face_isface.input";
    char *imageBuf = ReadFile(imagePath, ptrSize1);
    ASSERT_NE(imageBuf, nullptr);
    printf("==========OH_AI_TensorCreate==========\n");
    constexpr size_t createShapeNum = 4;
    int64_t createShape[createShapeNum] = {1, 48, 48, 3};
    OH_AI_TensorHandle tensor = OH_AI_TensorCreate("data", OH_AI_DATATYPE_NUMBERTYPE_FLOAT32, createShape,
                                                   createShapeNum, imageBuf, size1);
    ASSERT_NE(tensor, nullptr);
    int64_t elementNum = OH_AI_TensorGetElementNum(tensor);
    printf("Tensor name: %s, elements num: %" PRId64 ".\n", OH_AI_TensorGetName(tensor), elementNum);
    ASSERT_EQ(elementNum, 6912);
    delete[] imageBuf;
    OH_AI_TensorDestroy(&tensor);
}

/*
 * @tc.name: Tensor_GetDataSize
 * @tc.desc: Verify the OH_AI_TensorGetDataSize function.
 * @tc.type: FUNC
 */
HWTEST(MSLiteTest, Tensor_GetDataSize, testing::ext::TestSize.Level0) {
    printf("==========ReadFile==========\n");
    size_t size1;
    size_t *ptrSize1 = &size1;
    const char *imagePath = "/data/test/resource/ml_face_isface.input";
    char *imageBuf = ReadFile(imagePath, ptrSize1);
    ASSERT_NE(imageBuf, nullptr);
    printf("==========OH_AI_TensorCreate==========\n");
    constexpr size_t createShapeNum = 4;
    int64_t createShape[createShapeNum] = {1, 48, 48, 3};
    OH_AI_TensorHandle tensor = OH_AI_TensorCreate("data", OH_AI_DATATYPE_NUMBERTYPE_FLOAT32, createShape,
                                                   createShapeNum, imageBuf, size1);
    ASSERT_NE(tensor, nullptr);
    size_t dataSize = OH_AI_TensorGetDataSize(tensor);
    printf("Tensor data size: %zu.\n", dataSize);
    ASSERT_EQ(dataSize, 6912 * sizeof(float));
    delete[] imageBuf;
    OH_AI_TensorDestroy(&tensor);
}

/*
 * @tc.name: Tensor_GetMutableData
 * @tc.desc: Verify the OH_AI_TensorGetMutableData function.
 * @tc.type: FUNC
 */
HWTEST(MSLiteTest, Tensor_GetMutableData, testing::ext::TestSize.Level0) {
    printf("==========ReadFile==========\n");
    size_t size1;
    size_t *ptrSize1 = &size1;
    const char *imagePath = "/data/test/resource/ml_face_isface.input";
    char *imageBuf = ReadFile(imagePath, ptrSize1);
    ASSERT_NE(imageBuf, nullptr);
    printf("==========OH_AI_TensorCreate==========\n");
    constexpr size_t createShapeNum = 4;
    int64_t createShape[createShapeNum] = {1, 48, 48, 3};
    OH_AI_TensorHandle tensor = OH_AI_TensorCreate("data", OH_AI_DATATYPE_NUMBERTYPE_FLOAT32, createShape,
                                                   createShapeNum, imageBuf, size1);
    ASSERT_NE(tensor, nullptr);
    float *inputData = reinterpret_cast<float *>(OH_AI_TensorGetMutableData(tensor));
    ASSERT_NE(inputData, nullptr);
    delete[] imageBuf;
    OH_AI_TensorDestroy(&tensor);
}
