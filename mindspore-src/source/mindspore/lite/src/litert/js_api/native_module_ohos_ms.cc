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

#include "include/js_api/native_module_ohos_ms.h"
#include "src/common/log_adapter.h"

/*
 * Function registering all props and functions of ohos.ai.mslite module
 * which involves player and the recorder
 */
static napi_value Export(napi_env env, napi_value exports) {
  MS_LOG(INFO) << "Export() is called.";

  mindspore::MSLiteModelNapi::Init(env, exports);
  return exports;
}

/*
 * module define
 */
static napi_module g_module = {.nm_version = 1,
                               .nm_flags = 0,
                               .nm_filename = nullptr,
                               .nm_register_func = Export,
                               .nm_modname = "ai.mindSporeLite",
                               .nm_priv = ((void *)0),
                               .reserved = {0}};

/*
 * module register
 */
extern "C" __attribute__((constructor)) void RegisterModule(void) {
  MS_LOG(INFO) << "RegisterModule() is called";
  napi_module_register(&g_module);
}
