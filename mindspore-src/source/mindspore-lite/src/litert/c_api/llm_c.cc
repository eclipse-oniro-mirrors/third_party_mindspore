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

#include "include/c_api/llm_c.h"
#include "include/c_api/types_c.h"
#include "llm_wrapper.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include "src/common/utils.h"
#include "src/common/log_adapter.h"
#include <algorithm>
#include <string>

/**
 * @brief Create llm inference model handle.
 * @since 23
 */
OH_AI_LLMHandle OH_AI_LLMCreateSession() {
  mindspore::LLMWrapper *llmWrapper = new (std::nothrow) mindspore::LLMWrapper();
  if (llmWrapper == nullptr) {
    MS_LOG(ERROR) << "llmWrapper is nullptr";
    return nullptr;
  }
  llmWrapper->LoadLibrary();
  if (!llmWrapper->IsLoaded()) {
    MS_LOG(ERROR) << "llmWrapper loadl ibrary failed";
    return nullptr;
  }
  llmWrapper->Create();
  return llmWrapper;
}

/**
 * @brief Build llm model by model_config
 * @param llm Model handle.
 * @param path of model.
 * @since 23
 */
OH_AI_Status OH_AI_LLMBuildModel(OH_AI_LLMHandle llm, const char *model_path) {
  mindspore::LLMWrapper *llmWrapper = static_cast<mindspore::LLMWrapper *>(llm);
  if (llmWrapper == nullptr) {
    MS_LOG(ERROR) << "llmWrapper is nullptr";
    return OH_AI_STATUS_LITE_NULLPTR;
  }
  if (!llmWrapper->IsLoaded()) {
    MS_LOG(ERROR) << "llmWrapper is not loaded";
    return OH_AI_STATUS_LITE_ERROR;
  }

  return llmWrapper->Build(model_path);
}

/**
 * @brief set llm config
 * @param llm Model handle.
 * @param generated config for model.
 * @since 23
 */
OH_AI_Status OH_AI_LLMSetSessionConfig(OH_AI_LLMHandle llm, const char *generated_config) {
  mindspore::LLMWrapper *llmWrapper = static_cast<mindspore::LLMWrapper *>(llm);
  if (llmWrapper == nullptr) {
    MS_LOG(ERROR) << "llmWrapper is nullptr";
    return OH_AI_STATUS_LITE_NULLPTR;
  }
  if (!llmWrapper->IsLoaded()) {
    MS_LOG(ERROR) << "llmWrapper is not loaded";
    return OH_AI_STATUS_LITE_ERROR;
  }

  return llmWrapper->SetConfig(generated_config);
}

/**
 * @brief Destroy model and free resource controled by model handle.
 * @param llm Model handle.
 * @since 23
 */
OH_AI_Status OH_AI_LLMDestroy(OH_AI_LLMHandle llm) {
  mindspore::LLMWrapper *llmWrapper = static_cast<mindspore::LLMWrapper *>(llm);
  if (llmWrapper == nullptr) {
    MS_LOG(ERROR) << "llmWrapper is nullptr";
    return OH_AI_STATUS_LITE_NULLPTR;
  }
  if (!llmWrapper->IsLoaded()) {
    MS_LOG(ERROR) << "llmWrapper is not loaded";
    return OH_AI_STATUS_LITE_ERROR;
  }
  delete llmWrapper;
  return OH_AI_STATUS_SUCCESS;
}

/**
 * @brief Set the number of threads at runtime.
 * @param llm Model handle.
 * @param prompt input prompt.
 * @param generaion_mode enum for selecting generatioh mode of whether stream of no-stream inference.
 * @param generated_buffer output buffer of model inference
 * @param buffer_size output buffer size
 * @param finish_reason flag for finish reason
 * @since 23
 */
OH_AI_Status OH_AI_LLMGenerate(OH_AI_LLMHandle llm, const char *prompt, OH_AI_LLMGenerationMode generaion_mode,
                               char *generated_buffer, int buffer_size, OH_AI_LLMFinishReason *finish_reason) {
  mindspore::LLMWrapper *llmWrapper = static_cast<mindspore::LLMWrapper *>(llm);
  if (llmWrapper == nullptr) {
    MS_LOG(ERROR) << "llmWrapper is nullptr";
    return OH_AI_STATUS_LITE_NULLPTR;
  }
  if (!llmWrapper->IsLoaded()) {
    MS_LOG(ERROR) << "llmWrapper is not loaded";
    return OH_AI_STATUS_LITE_ERROR;
  }
  return llmWrapper->Generate(prompt, generaion_mode, generated_buffer, buffer_size, finish_reason);
}
