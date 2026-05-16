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

#include <climits>
#include <string.h>
#include <map>
#include "src/common/log.h"
#include "include/js_api/nnrt_device_desc_napi.h"

namespace mindspore {
thread_local napi_ref NnrtDeviceDescNapi::constructor_ = nullptr;
const std::string CLASS_NAME = "NNRTDeviceDescription";

NnrtDeviceDescNapi::NnrtDeviceDescNapi() { MS_LOG(DEBUG) << "MSLITE NNRTDeviceDescNapi Instances create."; }

NnrtDeviceDescNapi::~NnrtDeviceDescNapi() {
 if (nativeNnrtDeviceDesc_ != nullptr) {
   nativeNnrtDeviceDesc_ = nullptr;
 }
 MS_LOG(INFO) << "MSLITE MSTensorNapi Instances destroy.";
}

napi_value NnrtDeviceDescNapi::Constructor(napi_env env, napi_callback_info info) {
 napi_value jsThis = nullptr;
 napi_status status = napi_get_cb_info(env, info, nullptr, nullptr, &jsThis, nullptr);
 if (status != napi_ok || jsThis == nullptr) {
   MS_LOG(ERROR) << "Failed to retrieve details about the callback";
   return nullptr;
 }

 std::unique_ptr<NnrtDeviceDescNapi> nnrt_device_desc = std::make_unique<NnrtDeviceDescNapi>();
 if (nnrt_device_desc == nullptr) {
   MS_LOG(ERROR) << "No memory";
   return nullptr;
 }

 nnrt_device_desc->env_ = env;
 status = napi_wrap(env, jsThis, reinterpret_cast<void *>(nnrt_device_desc.get()), NnrtDeviceDescNapi::Finalize, nullptr, nullptr);
 if (status == napi_ok) {
   nnrt_device_desc.release();
   return jsThis;
 }

 MS_LOG(ERROR) << "Constructor fail.";
 return nullptr;
}

void NnrtDeviceDescNapi::Finalize(napi_env env, void *nativeObject, void *finalize) {
 (void)env;
 (void)finalize;
 if (nativeObject != nullptr) {
   delete reinterpret_cast<NnrtDeviceDesc *>(nativeObject);
 }
 MS_LOG(INFO) << "Finalize success.";
}

napi_value NnrtDeviceDescNapi::NewInstance(napi_env env, NnrtDeviceDesc desc) {
 napi_value cons = GetConstructor(env);
 if (cons == nullptr) {
   MS_LOG(ERROR) << "NewInstance GetConstructor is nullptr!";
   return nullptr;
 }
 napi_value instance;
 napi_status status = napi_new_instance(env, cons, 0, nullptr, &instance);
 if (status != napi_ok) {
   MS_LOG(ERROR) << "NewInstance napi_new_instance failed! code: " << status;
   return nullptr;
 }

 NnrtDeviceDescNapi *proxy = nullptr;
 status = napi_unwrap(env, instance, reinterpret_cast<void **>(&proxy));
 if (proxy == nullptr) {
   MS_LOG(ERROR) << "NewInstance native instance is nullptr! code: " << status;
   return instance;
 }

 proxy->nativeNnrtDeviceDesc_ = std::make_unique<NnrtDeviceDesc>(desc);
 if (proxy->nativeNnrtDeviceDesc_ == nullptr) {
   MS_LOG(ERROR) << "NewInstance native nnrt deivce desc unique ptr is nullptr!";
   return instance;
 }
 return instance;
}

napi_value NnrtDeviceDescNapi::GetConstructor(napi_env env) {
 napi_value cons;
 if (constructor_ != nullptr) {
   napi_get_reference_value(env, constructor_, &cons);
   return cons;
 }

 MS_LOG(INFO) << "Get NnrtDeviceDesc constructor.";
 napi_property_descriptor properties[] = {
   DECLARE_NAPI_FUNCTION("deviceID", GetDeviceID),
   DECLARE_NAPI_FUNCTION("deviceType", GetDeviceType),
   DECLARE_NAPI_FUNCTION("deviceName", GetDeviceName),
 };

 napi_status status = napi_define_class(env, CLASS_NAME.c_str(), NAPI_AUTO_LENGTH, Constructor, nullptr,
                                        sizeof(properties) / sizeof(napi_property_descriptor), properties, &cons);
 if (status != napi_ok) {
   MS_LOG(ERROR) << "MSLITE Failed to define NnrtDeviceDesc class";
   return nullptr;
 }

 status = napi_create_reference(env, cons, 1, &constructor_);
 if (status != napi_ok) {
   MS_LOG(ERROR) << "MSLITE Failed to create reference of constructor";
   return nullptr;
 }

 return cons;
}

napi_value NnrtDeviceDescNapi::GetDeviceName(napi_env env, napi_callback_info info) {
 napi_value undefinedResult = nullptr;
 napi_get_undefined(env, &undefinedResult);
 napi_value jsThis = nullptr;
 napi_value jsResult = nullptr;
 NnrtDeviceDescNapi *desc = nullptr;

 napi_status status = napi_get_cb_info(env, info, nullptr, nullptr, &jsThis, nullptr);
 if (status != napi_ok || jsThis == nullptr) {
   MS_LOG(ERROR) << "Failed to retrieve details about the callback";
   return undefinedResult;
 }

 status = napi_unwrap(env, jsThis, reinterpret_cast<void **>(&desc));
 if (status != napi_ok || desc == nullptr) {
   MS_LOG(ERROR) << "get tensor napi error";
   return undefinedResult;
 }

 status = napi_create_string_utf8(env, desc->nativeNnrtDeviceDesc_->name.c_str(), NAPI_AUTO_LENGTH, &jsResult);
 if (status != napi_ok) {
   MS_LOG(ERROR) << "napi_create_string_utf8 error";
   return undefinedResult;
 }

 MS_LOG(INFO) << "GetName success.";
 return jsResult;
}

napi_value NnrtDeviceDescNapi::GetDeviceID(napi_env env, napi_callback_info info) {
 napi_value undefinedResult = nullptr;
 napi_get_undefined(env, &undefinedResult);
 napi_value jsThis = nullptr;
 napi_value jsResult = nullptr;
 NnrtDeviceDescNapi *desc = nullptr;

 napi_status status = napi_get_cb_info(env, info, nullptr, nullptr, &jsThis, nullptr);
 if (status != napi_ok || jsThis == nullptr) {
   MS_LOG(ERROR) << "Failed to retrieve details about the callback";
   return undefinedResult;
 }

 status = napi_unwrap(env, jsThis, reinterpret_cast<void **>(&desc));
 if (status != napi_ok || desc == nullptr) {
   MS_LOG(ERROR) << "get tensor napi error";
   return undefinedResult;
 }

 auto id = static_cast<uint64_t>(desc->nativeNnrtDeviceDesc_->id);
 status = napi_create_bigint_uint64(env, id, &jsResult);
 if (status != napi_ok) {
   MS_LOG(ERROR) << "napi_create_int32 error";
   return undefinedResult;
 }

 MS_LOG(INFO) << "GetShape success.";
 return jsResult;
}

napi_value NnrtDeviceDescNapi::GetDeviceType(napi_env env, napi_callback_info info) {
 napi_value undefinedResult = nullptr;
 napi_get_undefined(env, &undefinedResult);
 napi_value jsThis = nullptr;
 napi_value jsResult = nullptr;
 NnrtDeviceDescNapi *desc = nullptr;

 napi_status status = napi_get_cb_info(env, info, nullptr, nullptr, &jsThis, nullptr);
 if (status != napi_ok || jsThis == nullptr) {
   MS_LOG(ERROR) << "Failed to retrieve details about the callback";
   return undefinedResult;
 }

 status = napi_unwrap(env, jsThis, reinterpret_cast<void **>(&desc));
 if (status != napi_ok || desc == nullptr) {
   MS_LOG(ERROR) << "get nnrt device type napi error";
   return undefinedResult;
 }

 status = napi_create_int32(env, desc->nativeNnrtDeviceDesc_->type, &jsResult);

 if (status != napi_ok) {
   MS_LOG(ERROR) << "napi_create_int32 error";
   return undefinedResult;
 }

 MS_LOG(INFO) << "GetDeviceType success.";
 return jsResult;
}
}  // namespace mindspore