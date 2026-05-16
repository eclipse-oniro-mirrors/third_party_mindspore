/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "hiai_foundation_wrapper.h"
#include "dlfcn.h"
#include "src/common/log.h"

namespace mindspore::lite {
static const char *HIAI_F_LIB = "libhiai_foundation.so";

bool UnLoadHiaiFLibrary(void *handle) {
  if (handle != nullptr) {
    if (dlclose(handle) != 0) {
      MS_LOG(WARNING) << "dlclose failed, error: " << dlerror();
      return false;
    }
    return true;
  }
  return true;
}

bool LoadHiaiFLibraryFromPath(void **handle_ptr) {
  if (handle_ptr == nullptr) {
    return false;
  }

  *handle_ptr = dlopen(HIAI_F_LIB, RTLD_NOW | RTLD_LOCAL);
  if (*handle_ptr == nullptr) {
    MS_LOG(WARNING) << "dlopen failed, error: " << dlerror();
    return false;
  }

// load function ptr use dlopen and dlsym.
#define LOAD_HIAIF_FUNCTION_PTR(func_name)                                                    \
  func_name = reinterpret_cast<func_name##Func>(dlsym(*handle_ptr, #func_name));               \
  if (func_name == nullptr) {                                                                  \
    MS_LOG(ERROR) << "load func (" << #func_name << ") from (" << HIAI_F_LIB << ") failed!"; \
    UnLoadHiaiFLibrary(*handle_ptr);                                                          \
    return false;                                                                              \
  }

  LOAD_HIAIF_FUNCTION_PTR(HMS_HiAIOptions_SetQuantConfig);
  LOAD_HIAIF_FUNCTION_PTR(HMS_HiAIOptions_SetBandMode);
  LOAD_HIAIF_FUNCTION_PTR(HMS_HiAIOptions_SetAsyncModeEnable);
  return true;
}

#define HIAIF_DEFINE_FUNC_PTR(func) func##Func func = nullptr
HIAIF_DEFINE_FUNC_PTR(HMS_HiAIOptions_SetQuantConfig);
HIAIF_DEFINE_FUNC_PTR(HMS_HiAIOptions_SetBandMode);
HIAIF_DEFINE_FUNC_PTR(HMS_HiAIOptions_SetAsyncModeEnable);

#undef LOAD_HIAIF_FUNCTION_PTR
}  // mindspore::lite