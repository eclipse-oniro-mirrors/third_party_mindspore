/*
 * Copyright (c) 2025 Huawei Device Co., Ltd.
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

#include "llm_wrapper.h"
#include <dlfcn.h>
#include <string>
#include "src/common/log_adapter.h"

template <class T>
bool LoadFunctionHelper(void *handle, const char *name, T *function) {
  if (name == nullptr) {
    MS_LOG(ERROR) << "load function failed, the " << name << " does not exist.";
    return false;
  }
  void *fn = dlsym(handle, name);
  if (fn == nullptr) {
    MS_LOG(ERROR) << "dlsym function " << name << " error: " << dlerror();
    return false;
  }
  *function = reinterpret_cast<T>(fn);
  return true;
}

namespace mindspore {
std::string kLLMImplLibraryName = "libmindspore-lite-turbo.so";
void LLMWrapper::LoadLibrary() {
  library_handle_ = dlopen(kLLMImplLibraryName.c_str(), RTLD_LAZY);
  if (library_handle_ == nullptr) {
    MS_LOG(ERROR) << "dlopen llm library " << kLLMImplLibraryName << " error: " << dlerror();
    return;
  }
  MS_LOG(INFO) << "loaded llm library " << kLLMImplLibraryName;
  bool ok = LoadFunctionHelper(library_handle_, "OH_AI_LLMNativeCreate", &llm_create_func_) &&
            LoadFunctionHelper(library_handle_, "OH_AI_LLMNativeDestroy", &llm_destroy_func_) &&
            LoadFunctionHelper(library_handle_, "OH_AI_LLMNativeBuild", &llm_build_func_) &&
            LoadFunctionHelper(library_handle_, "OH_AI_LLMNativeGenerate", &llm_generate_func_) &&
            LoadFunctionHelper(library_handle_, "OH_AI_LLMNativeUpdateConfig", &llm_update_func_);
  if (llm_create_func_ == nullptr || llm_destroy_func_ == nullptr || llm_build_func_ == nullptr ||
      llm_generate_func_ == nullptr || llm_update_func_ == nullptr) {
    MS_LOG(ERROR) << "func is null";
  }
  if (!ok) {
    MS_LOG(ERROR) << "failed to load necessary functions for llm";
    int ret = dlclose(library_handle_);
    if (ret != 0) {
      MS_LOG(WARNING) << "dlclose llm library " << kLLMImplLibraryName << " error: " << dlerror();
    }
    library_handle_ = nullptr;
    return;
  }
  is_loaded_ = true;
}

LLMWrapper::~LLMWrapper() {
  if (!IsLoaded()) {
    return;
  }
  llm_destroy_func_(llm_impl_handle_);
  int ret = dlclose(library_handle_);
  if (ret != 0) {
    MS_LOG(WARNING) << "dlclose llm library " << kLLMImplLibraryName << " error: " << dlerror();
  }
  library_handle_ = nullptr;
  MS_LOG(INFO) << "unloaded llm library " << kLLMImplLibraryName;
  return;
}

bool LLMWrapper::IsLoaded() { return is_loaded_; }

bool LLMWrapper::IsCreated() { return llm_impl_handle_ != nullptr; }

OH_AI_Status LLMWrapper::Build(const char *model_path) {
  if (!IsCreated()) {
    MS_LOG(ERROR) << "llm wrapper is not created";
    return OH_AI_STATUS_LITE_ERROR;
  }
  if (model_path == nullptr) {
    MS_LOG(ERROR) << "model_path is nullptr";
    return OH_AI_STATUS_LITE_NULLPTR;
  }
  return llm_build_func_(llm_impl_handle_, model_path);
}

OH_AI_Status LLMWrapper::SetConfig(const char *generated_config) {
  if (!IsCreated()) {
    MS_LOG(ERROR) << "llm wrapper is not created";
    return OH_AI_STATUS_LITE_ERROR;
  }
  if (generated_config == nullptr) {
    MS_LOG(ERROR) << "generated_config is nullptr";
    return OH_AI_STATUS_LITE_NULLPTR;
  }
  return llm_update_func_(llm_impl_handle_, generated_config);
}

OH_AI_Status LLMWrapper::Create() {
  llm_impl_handle_ = llm_create_func_();
  if (llm_impl_handle_ == nullptr) {
    MS_LOG(ERROR) << "failed to create llm_impl_handle_!";
    return OH_AI_STATUS_LITE_ERROR;
  }
  return OH_AI_STATUS_SUCCESS;
}

OH_AI_Status LLMWrapper::Generate(const char *prompt, OH_AI_LLMGenerationMode generaion_mode, char *generated_buffer,
                                  int buffer_size, OH_AI_LLMFinishReason *finish_reason) {
  if (!IsCreated()) {
    MS_LOG(ERROR) << "llm wrapper is not created";
    return OH_AI_STATUS_LITE_ERROR;
  }
  if (prompt == nullptr) {
    MS_LOG(ERROR) << "prompt is nullptr";
    return OH_AI_STATUS_LITE_NULLPTR;
  }
  if (generated_buffer == nullptr) {
    MS_LOG(ERROR) << "generated_buffer is nullptr";
    return OH_AI_STATUS_LITE_NULLPTR;
  }
  if (finish_reason == nullptr) {
    MS_LOG(ERROR) << "finish_reason is nullptr";
    return OH_AI_STATUS_LITE_NULLPTR;
  }
  return llm_generate_func_(llm_impl_handle_, prompt, generaion_mode, generated_buffer, buffer_size, finish_reason);
}

}  // namespace mindspore
