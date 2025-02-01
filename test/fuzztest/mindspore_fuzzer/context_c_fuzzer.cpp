
#include "context_c_fuzzer.h"
#include "../data.h"
#include "../../utils/model_utils.h"

bool MSContextFuzzTest_ThreadNum(const uint8_t *data, size_t size) {
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

bool MSContextFuzzTest_ThreadAffinityMode(const uint8_t *data, size_t size) {
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

bool MSContextFuzzTest_Provider(const uint8_t *data, size_t size) {
    OH_AI_ContextHandle context = OH_AI_ContextCreate();
    if (context == nullptr) {
        printf("create context failed.\n");
        return false;
    }
    char *infoProvider = static_cast<char *>(malloc(size + 1));
    if (infoProvider == nullptr) {
        printf("malloc failed.\n");
        return false;
    }
    if (memcpy_s(infoProvider, size + 1, data, size) != EOK) {
        printf("memcpy_s failed.");
        free(infoProvider);
        return false;
    }
    infoProvider[size] = '\0';
    OH_AI_DeviceInfoHandle cpuDeviceInfo = OH_AI_DeviceInfoCreate(OH_AI_DEVICETYPE_CPU);
    if (cpuDeviceInfo == nullptr) {
        printf("OH_AI_DeviceInfoCreate failed.\n");
        OH_AI_ContextDestroy(&context);
        free(infoProvider);
        return OH_AI_STATUS_LITE_ERROR;
    }
    OH_AI_DeviceInfoSetProvider(cpuDeviceInfo, infoProvider);
    OH_AI_DeviceInfoSetProviderDevice(cpuDeviceInfo, infoProvider);
    OH_AI_ContextAddDeviceInfo(context, cpuDeviceInfo);

    AddContextDeviceNNRT(context);

    OH_AI_ModelHandle model = OH_AI_ModelCreate();
    if (model == nullptr) {
        printf("create model failed.\n");
        free(infoProvider);
        return false;
    }
    ModelPredict(model, context, "ml_face_isface", {}, false, true, false);
    free(infoProvider);
    return true;
}


bool MSContextFuzzTest_ThreadNum_Add1(const uint8_t *data, size_t size) {
    OH_AI_ContextHandle context = OH_AI_ContextCreate();
    if (context == nullptr) {
        printf("create context failed.\n");
        return false;
    }
    Data dataFuzz(data, size);
    int32_t threadNum = dataFuzz.GetData<int32_t>();
    OH_AI_ContextSetThreadNum(context, threadNum);
    AddContextDeviceCPU(context);

    OH_AI_ModelHandle model = OH_AI_ModelCreate();
    if (model == nullptr) {
        printf("create model failed.\n");
        return false;
    }
    ModelPredict(model, context, "ml_face_isface", {}, true, true, false);
    return true;
}

bool MSContextFuzzTest_ThreadAffinityMode_Add1(const uint8_t *data, size_t size) {
    OH_AI_ContextHandle context = OH_AI_ContextCreate();
    if (context == nullptr) {
        printf("create context failed.\n");
        return false;
    }
    Data dataFuzz(data, size);
    int32_t affinityMode = dataFuzz.GetData<int32_t>();
    OH_AI_ContextSetThreadAffinityMode(context, affinityMode);
    AddContextDeviceCPU(context);

    OH_AI_ModelHandle model = OH_AI_ModelCreate();
    if (model == nullptr) {
        printf("create model failed.\n");
        return false;
    }
    ModelPredict(model, context, "ml_face_isface", {}, true, true, false);
    return true;
}

bool MSContextFuzzTest_Provider_Add1(const uint8_t *data, size_t size) {
    OH_AI_ContextHandle context = OH_AI_ContextCreate();
    if (context == nullptr) {
        printf("create context failed.\n");
        return false;
    }
    char *infoProvider = static_cast<char *>(malloc(size + 1));
    if (infoProvider == nullptr) {
        printf("malloc failed.\n");
        return false;
    }
    if (memcpy_s(infoProvider, size + 1, data, size) != EOK) {
        printf("memcpy_s failed.");
        free(infoProvider);
        return false;
    }
    infoProvider[size] = '\0';
    OH_AI_DeviceInfoHandle cpuDeviceInfo = OH_AI_DeviceInfoCreate(OH_AI_DEVICETYPE_CPU);
    if (cpuDeviceInfo == nullptr) {
        printf("OH_AI_DeviceInfoCreate failed.\n");
        OH_AI_ContextDestroy(&context);
        free(infoProvider);
        return OH_AI_STATUS_LITE_ERROR;
    }
    OH_AI_DeviceInfoSetProvider(cpuDeviceInfo, infoProvider);
    OH_AI_DeviceInfoSetProviderDevice(cpuDeviceInfo, infoProvider);
    OH_AI_ContextAddDeviceInfo(context, cpuDeviceInfo);

    OH_AI_ModelHandle model = OH_AI_ModelCreate();
    if (model == nullptr) {
        printf("create model failed.\n");
        free(infoProvider);
        return false;
    }
    ModelPredict(model, context, "ml_face_isface", {}, true, true, false);
    free(infoProvider);
    return true;
}

bool MSContextFuzzTest_ThreadNum_Add2(const uint8_t *data, size_t size) {
    OH_AI_ContextHandle context = OH_AI_ContextCreate();
    if (context == nullptr) {
        printf("create context failed.\n");
        return false;
    }
    Data dataFuzz(data, size);
    int32_t threadNum = dataFuzz.GetData<int32_t>();
    OH_AI_ContextSetThreadNum(context, threadNum);
    AddContextDeviceCPU(context);

    OH_AI_ModelHandle model = OH_AI_ModelCreate();
    if (model == nullptr) {
        printf("create model failed.\n");
        return false;
    }
    ModelPredict(model, context, "ml_face_isface", {}, false, true, true);
    return true;
}

bool MSContextFuzzTest_ThreadAffinityMode_Add2(const uint8_t *data, size_t size) {
    OH_AI_ContextHandle context = OH_AI_ContextCreate();
    if (context == nullptr) {
        printf("create context failed.\n");
        return false;
    }
    Data dataFuzz(data, size);
    int32_t affinityMode = dataFuzz.GetData<int32_t>();
    OH_AI_ContextSetThreadAffinityMode(context, affinityMode);
    AddContextDeviceCPU(context);

    OH_AI_ModelHandle model = OH_AI_ModelCreate();
    if (model == nullptr) {
        printf("create model failed.\n");
        return false;
    }
    ModelPredict(model, context, "ml_face_isface", {}, false, true, true);
    return true;
}

bool MSContextFuzzTest_Provider_Add2(const uint8_t *data, size_t size) {
    OH_AI_ContextHandle context = OH_AI_ContextCreate();
    if (context == nullptr) {
        printf("create context failed.\n");
        return false;
    }
    char *infoProvider = static_cast<char *>(malloc(size + 1));
    if (infoProvider == nullptr) {
        printf("malloc failed.\n");
        return false;
    }
    if (memcpy_s(infoProvider, size + 1, data, size) != EOK) {
        printf("memcpy_s failed.");
        free(infoProvider);
        return false;
    }
    infoProvider[size] = '\0';
    OH_AI_DeviceInfoHandle cpuDeviceInfo = OH_AI_DeviceInfoCreate(OH_AI_DEVICETYPE_CPU);
    if (cpuDeviceInfo == nullptr) {
        printf("OH_AI_DeviceInfoCreate failed.\n");
        OH_AI_ContextDestroy(&context);
        free(infoProvider);
        return OH_AI_STATUS_LITE_ERROR;
    }
    OH_AI_DeviceInfoSetProvider(cpuDeviceInfo, infoProvider);
    OH_AI_DeviceInfoSetProviderDevice(cpuDeviceInfo, infoProvider);
    OH_AI_ContextAddDeviceInfo(context, cpuDeviceInfo);

    OH_AI_ModelHandle model = OH_AI_ModelCreate();
    if (model == nullptr) {
        printf("create model failed.\n");
        free(infoProvider);
        return false;
    }
    ModelPredict(model, context, "ml_face_isface", {}, false, true, true);
    free(infoProvider);
    return true;
}

bool MSContextFuzzTest_ThreadNum_Add3(const uint8_t *data, size_t size) {
    OH_AI_ContextHandle context = OH_AI_ContextCreate();
    if (context == nullptr) {
        printf("create context failed.\n");
        return false;
    }
    Data dataFuzz(data, size);
    int32_t threadNum = dataFuzz.GetData<int32_t>();
    OH_AI_ContextSetThreadNum(context, threadNum);
    AddContextDeviceCPU(context);

    OH_AI_ModelHandle model = OH_AI_ModelCreate();
    if (model == nullptr) {
        printf("create model failed.\n");
        return false;
    }
    ModelPredict_ModelBuild(model, context, "ml_face_isface", true);
    return true;
}

bool MSContextFuzzTest_ThreadNum_Add4(const uint8_t *data, size_t size) {
    OH_AI_ContextHandle context = OH_AI_ContextCreate();
    if (context == nullptr) {
        printf("create context failed.\n");
        return false;
    }
    Data dataFuzz(data, size);
    int32_t threadNum = dataFuzz.GetData<int32_t>();
    OH_AI_ContextSetThreadNum(context, threadNum);
    AddContextDeviceCPU(context);

    OH_AI_ModelHandle model = OH_AI_ModelCreate();
    if (model == nullptr) {
        printf("create model failed.\n");
        return false;
    }
    ModelPredict_ModelBuild(model, context, "ml_face_isface", false);
    return true;
}

bool MSContextFuzzTest_ThreadAffinityMode_Add3(const uint8_t *data, size_t size) {
    OH_AI_ContextHandle context = OH_AI_ContextCreate();
    if (context == nullptr) {
        printf("create context failed.\n");
        return false;
    }
    Data dataFuzz(data, size);
    int32_t affinityMode = dataFuzz.GetData<int32_t>();
    OH_AI_ContextSetThreadAffinityMode(context, affinityMode);
    AddContextDeviceCPU(context);

    OH_AI_ModelHandle model = OH_AI_ModelCreate();
    if (model == nullptr) {
        printf("create model failed.\n");
        return false;
    }
    ModelPredict_ModelBuild(model, context, "ml_face_isface", true);
    return true;
}

bool MSContextFuzzTest_ThreadAffinityMode_Add4(const uint8_t *data, size_t size) {
    OH_AI_ContextHandle context = OH_AI_ContextCreate();
    if (context == nullptr) {
        printf("create context failed.\n");
        return false;
    }
    Data dataFuzz(data, size);
    int32_t affinityMode = dataFuzz.GetData<int32_t>();
    OH_AI_ContextSetThreadAffinityMode(context, affinityMode);
    AddContextDeviceCPU(context);

    OH_AI_ModelHandle model = OH_AI_ModelCreate();
    if (model == nullptr) {
        printf("create model failed.\n");
        return false;
    }
    ModelPredict_ModelBuild(model, context, "ml_face_isface", false);
    return true;
}

bool MSContextFuzzTest_Provider_Add3(const uint8_t* data, size_t size) {
    OH_AI_ContextHandle context = OH_AI_ContextCreate();
    if (context == nullptr) {
        printf("create context failed.\n");
        return false;
    }
    char *infoProvider = static_cast<char *>(malloc(size + 1));
    if (infoProvider == nullptr) {
        printf("malloc failed.\n");
        return false;
    }
    if (memcpy_s(infoProvider, size + 1, data, size) != EOK) {
        printf("memcpy_s failed.");
        free(infoProvider);
        return false;
    }
    infoProvider[size] = '\0';
    OH_AI_DeviceInfoHandle cpuDeviceInfo = OH_AI_DeviceInfoCreate(OH_AI_DEVICETYPE_CPU);
    if (cpuDeviceInfo == nullptr) {
        printf("OH_AI_DeviceInfoCreate failed.\n");
        OH_AI_ContextDestroy(&context);
        free(infoProvider);
        return OH_AI_STATUS_LITE_ERROR;
    }
    OH_AI_DeviceInfoSetProvider(cpuDeviceInfo, infoProvider);
    OH_AI_DeviceInfoSetProviderDevice(cpuDeviceInfo, infoProvider);
    OH_AI_ContextAddDeviceInfo(context, cpuDeviceInfo);

    OH_AI_ModelHandle model = OH_AI_ModelCreate();
    if (model == nullptr) {
        printf("create model failed.\n");
        free(infoProvider);
        return false;
    }
    ModelPredict_ModelBuild(model, context, "ml_face_isface", true);
    free(infoProvider);
    return true;
}

bool MSContextFuzzTest_Provider_Add4(const uint8_t *data, size_t size) {
    OH_AI_ContextHandle context = OH_AI_ContextCreate();
    if (context == nullptr) {
        printf("create context failed.\n");
        return false;
    }
    char *infoProvider = static_cast<char *>(malloc(size + 1));
    if (infoProvider == nullptr) {
        printf("malloc failed.\n");
        return false;
    }
    if (memcpy_s(infoProvider, size + 1, data, size) != EOK) {
        printf("memcpy_s failed.");
        free(infoProvider);
        return false;
    }
    infoProvider[size] = '\0';
    OH_AI_DeviceInfoHandle cpuDeviceInfo = OH_AI_DeviceInfoCreate(OH_AI_DEVICETYPE_CPU);
    if (cpuDeviceInfo == nullptr) {
        printf("OH_AI_DeviceInfoCreate failed.\n");
        OH_AI_ContextDestroy(&context);
        free(infoProvider);
        return OH_AI_STATUS_LITE_ERROR;
    }
    OH_AI_DeviceInfoSetProvider(cpuDeviceInfo, infoProvider);
    OH_AI_DeviceInfoSetProviderDevice(cpuDeviceInfo, infoProvider);
    OH_AI_ContextAddDeviceInfo(context, cpuDeviceInfo);

    OH_AI_ModelHandle model = OH_AI_ModelCreate();
    if (model == nullptr) {
        printf("create model failed.\n");
        free(infoProvider);
        return false;
    }
    ModelPredict_ModelBuild(model, context, "ml_face_isface", false);
    free(infoProvider);
    return true;
}


bool MSContextFuzzTest(const uint8_t *data, size_t size) {
    bool ret = MSContextFuzzTest_ThreadNum(data, size) && MSContextFuzzTest_ThreadNum_Add1(data, size) &&
               MSContextFuzzTest_ThreadNum_Add2(data, size) && MSContextFuzzTest_ThreadNum_Add3(data, size) &&
               MSContextFuzzTest_ThreadNum_Add4(data, size) && MSContextFuzzTest_ThreadAffinityMode(data, size) &&
               MSContextFuzzTest_ThreadAffinityMode_Add1(data, size) &&
               MSContextFuzzTest_ThreadAffinityMode_Add2(data, size) &&
               MSContextFuzzTest_ThreadAffinityMode_Add3(data, size) &&
               MSContextFuzzTest_ThreadAffinityMode_Add4(data, size) && MSContextFuzzTest_Provider(data, size) &&
               MSContextFuzzTest_Provider_Add1(data, size) && MSContextFuzzTest_Provider_Add2(data, size) &&
               MSContextFuzzTest_Provider_Add3(data, size) && MSContextFuzzTest_Provider_Add4(data, size);
    return ret;
}