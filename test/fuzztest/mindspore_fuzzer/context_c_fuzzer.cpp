
#include "context_c_fuzzer.h"
#include "../data.h"
#include "../../utils/model_utils.h"

bool MSContextFuzzTest_ThreadNum(const uint8_t* data, size_t size) {
    OH_AI_ContextHandle context = OH_AI_ContextCreate();
    if (context == nullptr) {
        printf("create context failed.\n");
        return false;
    }
    Data dataFuzz(data, size);
    int32_t threadNum = dataFuzz.GetData<int32_t>();
    OH_AI_ContextSetThreadNum(context, threadNum);
    AddContextDeviceCPU(context);
    AddContextDeviceNNRT(context);

    OH_AI_ModelHandle model = OH_AI_ModelCreate();
    if (model == nullptr) {
        printf("create model failed.\n");
        return false;
    }
    ModelPredict(model, context, "ml_face_isface", {}, false, true, false);
    return true;
}

bool MSContextFuzzTest_ThreadAffinityMode(const uint8_t* data, size_t size) {
    OH_AI_ContextHandle context = OH_AI_ContextCreate();
    if (context == nullptr) {
        printf("create context failed.\n");
        return false;
    }
    Data dataFuzz(data, size);
    int32_t affinityMode = dataFuzz.GetData<int32_t>();
    OH_AI_ContextSetThreadAffinityMode(context, affinityMode);
    AddContextDeviceCPU(context);
    AddContextDeviceNNRT(context);

    OH_AI_ModelHandle model = OH_AI_ModelCreate();
    if (model == nullptr) {
        printf("create model failed.\n");
        return false;
    }
    ModelPredict(model, context, "ml_face_isface", {}, false, true, false);
    return true;
}

bool MSContextFuzzTest_Provider(const uint8_t* data, size_t size) {
    OH_AI_ContextHandle context = OH_AI_ContextCreate();
    if (context == nullptr) {
        printf("create context failed.\n");
        return false;
    }
    const char *infoProvider = reinterpret_cast<const char *>(data);
    OH_AI_DeviceInfoHandle cpuDeviceInfo = OH_AI_DeviceInfoCreate(OH_AI_DEVICETYPE_CPU);
    if (cpuDeviceInfo == nullptr) {
        printf("OH_AI_DeviceInfoCreate failed.\n");
        OH_AI_ContextDestroy(&context);
        return OH_AI_STATUS_LITE_ERROR;
    }
    OH_AI_DeviceInfoSetProvider(cpuDeviceInfo, infoProvider);
    OH_AI_DeviceInfoSetProviderDevice(cpuDeviceInfo, infoProvider);
    OH_AI_ContextAddDeviceInfo(context, cpuDeviceInfo);

    AddContextDeviceNNRT(context);

    OH_AI_ModelHandle model = OH_AI_ModelCreate();
    if (model == nullptr) {
        printf("create model failed.\n");
        return false;
    }
    ModelPredict(model, context, "ml_face_isface", {}, false, true, false);
    return true;
}

bool MSContextFuzzTest(const uint8_t* data, size_t size) {
    bool ret = MSContextFuzzTest_ThreadNum(data, size) && MSContextFuzzTest_ThreadAffinityMode(data, size) &&
               MSContextFuzzTest_Provider(data, size);
    return ret;
}