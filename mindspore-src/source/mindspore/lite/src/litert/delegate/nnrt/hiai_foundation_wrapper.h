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

#ifndef LITE_HIAI_FOUNDATION_WRAPPER_H
#define LITE_HIAI_FOUNDATION_WRAPPER_H

#include <string>
#include "neural_network_runtime/neural_network_runtime.h"

namespace mindspore::lite {
bool LoadHiaiFLibraryFromPath(void **handle_ptr);
bool UnLoadHiaiFLibrary(void *handle);

typedef enum {
  /** Automatically adjusted by the system. */
  HIAI_BANDMODE_UNSET = 0,
  /** Low bandwidth mode. */
  HIAI_BANDMODE_LOW = 1,
  /** Medium bandwidth mode. */
  HIAI_BANDMODE_NORMAL = 2,
  /** High bandwidth mode. */
  HIAI_BANDMODE_HIGH = 3,
} HiAI_BandMode;

using HMS_HiAIOptions_SetQuantConfigFunc = OH_NN_ReturnCode (*)(OH_NNCompilation*, void*, size_t);
using HMS_HiAIOptions_SetBandModeFunc = OH_NN_ReturnCode (*)(OH_NNCompilation*, HiAI_BandMode);

#define HIAIF_DECLARE_FUNC_PTR(func) extern func##Func func
HIAIF_DECLARE_FUNC_PTR(HMS_HiAIOptions_SetQuantConfig);
HIAIF_DECLARE_FUNC_PTR(HMS_HiAIOptions_SetBandMode);
#undef HIAIF_DECLARE_FUNC_PTR
}  // mindspore::lite

#endif  // LITE_HIAI_FOUNDATION_WRAPPER_H
