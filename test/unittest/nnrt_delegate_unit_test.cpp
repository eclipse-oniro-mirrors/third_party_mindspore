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
#include <random>
#include "../utils/model_utils.h"
#include "../utils/common.h"

class MSLiteNnrtTest: public testing::Test {
  protected:
    static void SetUpTestCase(void) {}
    static void TearDownTestCase(void) {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

/*
 * @tc.name: Nnrt_Test
 * @tc.desc: Verify the NNRT delegate.
 * @tc.type: FUNC
 */
HWTEST(MSLiteNnrtTest, Nnrt_ContextTest, testing::ext::TestSize.Level0) {
    std::cout << "==========Get All Nnrt Device Descs==========" << std::endl;
    size_t num = 0;
    auto descs = OH_AI_GetAllNNRTDeviceDescs(&num);
    if (descs == nullptr) {
        std::cout << "descs is nullptr , num: " << num << std::endl;
        ASSERT_EQ(num, 0);
        return;
    }

    std::cout << "found " << num << " nnrt devices" << std::endl;
    for (size_t i = 0; i < num; i++) {
        auto desc = OH_AI_GetElementOfNNRTDeviceDescs(descs, i);
        ASSERT_NE(desc, nullptr);
        auto id = OH_AI_GetDeviceIdFromNNRTDeviceDesc(desc);
        auto name = OH_AI_GetNameFromNNRTDeviceDesc(desc);
        auto type = OH_AI_GetTypeFromNNRTDeviceDesc(desc);
        std::cout << "NNRT device: id = " << id << ", name: " << name << ", type:" << type << std::endl;
    }

    OH_AI_DestroyAllNNRTDeviceDescs(&descs);
    ASSERT_EQ(descs, nullptr);
}

/*
 * @tc.name: Nnrt_CreateNnrtDevice
 * @tc.desc: Verify the NNRT device create function.
 * @tc.type: FUNC
 */
HWTEST(MSLiteNnrtTest, Nnrt_CreateNnrtDevice, testing::ext::TestSize.Level0) {
    std::cout << "==========Get All Nnrt Device Descs==========" << std::endl;
    size_t num = 0;
    auto desc = OH_AI_GetAllNNRTDeviceDescs(&num);
    if (desc == nullptr) {
        std::cout << "descs is nullptr , num: " << num << std::endl;
        ASSERT_EQ(num, 0);
        return;
    }

    std::cout << "found " << num << " nnrt devices" << std::endl;
    auto id = OH_AI_GetDeviceIdFromNNRTDeviceDesc(desc);
    auto name = OH_AI_GetNameFromNNRTDeviceDesc(desc);
    auto type = OH_AI_GetTypeFromNNRTDeviceDesc(desc);
    std::cout << "NNRT device: id = " << id << ", name = " << name << ", type = " << type << std::endl;

    // create by name
    auto nnrtDeviceInfo = OH_AI_CreateNNRTDeviceInfoByName(name);
    ASSERT_NE(nnrtDeviceInfo, nullptr);

    OH_AI_DeviceType deviceType = OH_AI_DeviceInfoGetDeviceType(nnrtDeviceInfo);
    printf("==========deviceType:%d\n", deviceType);
    ASSERT_EQ(OH_AI_DeviceInfoGetDeviceId(nnrtDeviceInfo), id);
    ASSERT_EQ(deviceType, OH_AI_DEVICETYPE_NNRT);
    OH_AI_DeviceInfoDestroy(&nnrtDeviceInfo);
    ASSERT_EQ(nnrtDeviceInfo, nullptr);

    // create by type
    nnrtDeviceInfo = OH_AI_CreateNNRTDeviceInfoByType(type);
    ASSERT_NE(nnrtDeviceInfo, nullptr);

    deviceType = OH_AI_DeviceInfoGetDeviceType(nnrtDeviceInfo);
    printf("==========deviceType:%d\n", deviceType);
    ASSERT_EQ(deviceType, OH_AI_DEVICETYPE_NNRT);
    ASSERT_EQ(OH_AI_DeviceInfoGetDeviceId(nnrtDeviceInfo), id);
    OH_AI_DeviceInfoDestroy(&nnrtDeviceInfo);
    ASSERT_EQ(nnrtDeviceInfo, nullptr);

    // create by id
    nnrtDeviceInfo = OH_AI_DeviceInfoCreate(OH_AI_DEVICETYPE_NNRT);
    ASSERT_NE(nnrtDeviceInfo, nullptr);
    OH_AI_DeviceInfoSetDeviceId(nnrtDeviceInfo, id);

    deviceType = OH_AI_DeviceInfoGetDeviceType(nnrtDeviceInfo);
    printf("==========deviceType:%d\n", deviceType);
    ASSERT_EQ(deviceType, OH_AI_DEVICETYPE_NNRT);

    OH_AI_DeviceInfoSetPerformanceMode(nnrtDeviceInfo, OH_AI_PERFORMANCE_MEDIUM);
    ASSERT_EQ(OH_AI_DeviceInfoGetPerformanceMode(nnrtDeviceInfo), OH_AI_PERFORMANCE_MEDIUM);
    OH_AI_DeviceInfoSetPriority(nnrtDeviceInfo, OH_AI_PRIORITY_MEDIUM);
    ASSERT_EQ(OH_AI_DeviceInfoGetPriority(nnrtDeviceInfo), OH_AI_PRIORITY_MEDIUM);
    std::string cachePath = "/data/local/tmp/";
    std::string cacheVersion = "1";
    OH_AI_DeviceInfoAddExtension(nnrtDeviceInfo, "CachePath", cachePath.c_str(), cachePath.size());
    OH_AI_DeviceInfoAddExtension(nnrtDeviceInfo, "CacheVersion", cacheVersion.c_str(), cacheVersion.size());
    OH_AI_DeviceInfoDestroy(&nnrtDeviceInfo);
    ASSERT_EQ(nnrtDeviceInfo, nullptr);

    OH_AI_DestroyAllNNRTDeviceDescs(&desc);
}

/*
 * @tc.name: Nnrt_NpuPredict
 * @tc.desc: Verify the NNRT predict.
 * @tc.type: FUNC
 */
HWTEST(MSLiteNnrtTest, Nnrt_NpuPredict, testing::ext::TestSize.Level0) {
    if (!IsNPU()) {
        printf("NNRt is not NPU, skip this test");
        return;
    }

    printf("==========Init Context==========\n");
    OH_AI_ContextHandle context = OH_AI_ContextCreate();
    ASSERT_NE(context, nullptr);
    AddContextDeviceNNRT(context);
    printf("==========Create model==========\n");
    OH_AI_ModelHandle model = OH_AI_ModelCreate();
    ASSERT_NE(model, nullptr);
    printf("==========Build model==========\n");
    OH_AI_Status ret = OH_AI_ModelBuildFromFile(model, "/data/test/resource/tinynet.om.ms",
                                                OH_AI_MODELTYPE_MINDIR, context);
    printf("==========build model return code:%d\n", ret);
    if (ret != OH_AI_STATUS_SUCCESS) {
        printf("==========build model failed, ret: %d\n", ret);
        OH_AI_ModelDestroy(&model);
        return;
    }

    printf("==========GetInputs==========\n");
    OH_AI_TensorHandleArray inputs = OH_AI_ModelGetInputs(model);
    ASSERT_NE(inputs.handle_list, nullptr);
    for (size_t i = 0; i < inputs.handle_num; ++i) {
        OH_AI_TensorHandle tensor = inputs.handle_list[i];
        float *inputData = reinterpret_cast<float *>(OH_AI_TensorGetMutableData(tensor));
        size_t elementNum = OH_AI_TensorGetElementNum(tensor);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f,1.0f);
        for (size_t z = 0; z < elementNum; z++) {
            inputData[z] = dis(gen);
        }
    }
    printf("==========Model Predict==========\n");
    OH_AI_TensorHandleArray outputs;
    ret = OH_AI_ModelPredict(model, inputs, &outputs, nullptr, nullptr);
    ASSERT_EQ(ret, OH_AI_STATUS_SUCCESS);
    OH_AI_ModelDestroy(&model);
}

/*
 * @tc.name: Nnrt_NpuCpuPredict
 * @tc.desc: Verify the NNRT npu/cpu predict.
 * @tc.type: FUNC
 */
HWTEST(MSLiteNnrtTest, Nnrt_NpuCpuPredict, testing::ext::TestSize.Level0) {
    printf("==========Init Context==========\n");
    OH_AI_ContextHandle context = OH_AI_ContextCreate();
    ASSERT_NE(context, nullptr);
    AddContextDeviceNNRT(context);
    AddContextDeviceCPU(context);
    printf("==========Create model==========\n");
    OH_AI_ModelHandle model = OH_AI_ModelCreate();
    ASSERT_NE(model, nullptr);
    printf("==========Build model==========\n");
    OH_AI_Status ret = OH_AI_ModelBuildFromFile(model, "/data/test/resource/ml_face_isface.ms",
                                                OH_AI_MODELTYPE_MINDIR, context);
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
    CompareResult(outputs, "ml_face_isface");
    OH_AI_ModelDestroy(&model);
}
