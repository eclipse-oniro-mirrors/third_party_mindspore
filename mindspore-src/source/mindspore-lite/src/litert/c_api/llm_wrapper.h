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

#ifndef MINDSPORE_LITE_LLM_LLMWRAPPER_H
#define MINDSPORE_LITE_LLM_LLMWRAPPER_H

#include "include/c_api/llm_c.h"
#include "include/c_api/types_c.h"
#include "src/common/utils.h"

typedef OH_AI_LLMHandle OH_AI_LLMImplHandle;
typedef OH_AI_LLMImplHandle (*LLMCreateFunc)(void);
typedef OH_AI_Status (*LLMDestroyFunc)(OH_AI_LLMImplHandle llm);
typedef OH_AI_Status (*LLMBuildFunc)(OH_AI_LLMImplHandle llm, const char *model_config);
typedef OH_AI_Status (*LLMGenerateFunc)(OH_AI_LLMImplHandle llm, const char *prompt, OH_AI_LLMGenerationMode generaion_mode, char *generated_buffer, int buffer_size,OH_AI_LLMFinishReason *finish_reason);
typedef OH_AI_Status (*LLMUpdateFunc)(OH_AI_LLMImplHandle llm,const char *updated_config);
namespace mindspore {
  class LLMWrapper {
    public:
      LLMWrapper() = default;
      ~LLMWrapper();
      LLMWrapper(const LLMWrapper&) = delete;
      LLMWrapper& operator=(const LLMWrapper&) = delete;
      void LoadLibrary();
      bool IsLoaded();
      OH_AI_Status Create();
      OH_AI_Status Build(const char *model_path);
      OH_AI_Status Generate(const char *prompt,
                            OH_AI_LLMGenerationMode generaion_mode,
                            char *generated_buffer,
                            int buffer_size,
                            OH_AI_LLMFinishReason *finish_reason);
      OH_AI_Status SetConfig(const char *generated_config);
    private:
      bool IsCreated();
      void ParseLLMConfig(const char *config_str);
      void *library_handle_ = nullptr;
      LLMCreateFunc llm_create_func_ = nullptr;
      LLMDestroyFunc llm_destroy_func_ = nullptr;
      LLMBuildFunc llm_build_func_ = nullptr;
      LLMUpdateFunc llm_update_func_ = nullptr;
      LLMGenerateFunc llm_generate_func_ = nullptr;
      bool is_loaded_ = false;
      OH_AI_LLMImplHandle llm_impl_handle_ = nullptr;
  };
} // namespace mindspore

#endif //MINDSPORE_LITE_LLM_LLMWRAPPER_H
