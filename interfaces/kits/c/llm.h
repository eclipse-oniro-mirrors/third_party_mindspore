/**
* Copyright 2025 Huawei Technologies Co., Ltd
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

/**
* @addtogroup MindSpore
* @{
*
* @brief Provides APIs related to MindSpore Lite model inference.
*
* @Syscap SystemCapability.Ai.MindSpore
* @since 9
*/

/**
* @file llm.h
* @kit MindSporeLiteKit
* @brief provide apis for large language model inference.
* @library libmindspore_lite_ndk.so
* @since 23
*/

#ifndef MINDSPORE_INCLUDE_C_API_LLM_C_H
#define MINDSPORE_INCLUDE_C_API_LLM_C_H

#ifdef __cplusplus
extern "C" {
#endif
#include "status.h"
#include "types.h"

typedef void * OH_AI_LLMHandle;

/**
 * @brief Create llm inference model handle.
 * @since 23
 */
OH_AI_LLMHandle OH_AI_LLMCreateSession();
/**
 * @brief Build llm model by model_config
 * @param llm Model handle.
 * @param path of model.
 * @since 23
 */
OH_AI_Status OH_AI_LLMBuildModel(OH_AI_LLMHandle llm, const char *model_path);

/**
 * @brief set llm config
 * @param llm Model handle.
 * @param generated config for model.
 * @since 23
 */
OH_AI_Status OH_AI_LLMSetSessionConfig(OH_AI_LLMHandle llm, const char *generated_config);

/**
 * @brief Destroy model and free resource controled by model handle.
 * @param llm Model handle.
 * @since 23
 */
OH_AI_Status OH_AI_LLMDestroy(OH_AI_LLMHandle llm);
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
OH_AI_Status OH_AI_LLMGenerate(OH_AI_LLMHandle llm, const char *prompt, OH_AI_LLMGenerationMode generaion_mode, char *generated_buffer, int buffer_size,OH_AI_LLMFinishReason *finish_reason);
#ifdef __cplusplus
}
#endif
#endif // MINDSPORE_INCLUDE_C_API_LLM_C_H