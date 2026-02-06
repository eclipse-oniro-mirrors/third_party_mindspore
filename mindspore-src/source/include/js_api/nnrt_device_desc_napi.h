/**
* Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_INCLUDE_JS_API_NNRT_DEVICE_DESC_NAPI_H
#define MINDSPORE_INCLUDE_JS_API_NNRT_DEVICE_DESC_NAPI_H

#include "include/api/types.h"
#include "napi/native_api.h"
#include "napi/native_node_api.h"
#include "include/js_api/common_napi.h"

namespace mindspore {
class NnrtDeviceDescNapi {
public:
 static napi_value NewInstance(napi_env env, NnrtDeviceDesc decs);
 NnrtDeviceDescNapi();
 ~NnrtDeviceDescNapi();
private:
 static napi_value Constructor(napi_env env, napi_callback_info info);
 static void Finalize(napi_env env, void *nativeObject, void *finalize);
 static napi_value GetConstructor(napi_env env);

 static napi_value GetDeviceName(napi_env env, napi_callback_info info);
 static napi_value GetDeviceType(napi_env env, napi_callback_info info);
 static napi_value GetDeviceID(napi_env env, napi_callback_info info);

 static thread_local napi_ref constructor_;
 napi_env env_ = nullptr;

 std::unique_ptr<NnrtDeviceDesc> nativeNnrtDeviceDesc_ = nullptr;
};
}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_JS_API_NNRT_DEVICE_DESC_NAPI_H