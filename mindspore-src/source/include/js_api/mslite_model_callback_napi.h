/**
 * Copyright (C) 2023 Huawei Device Co., Ltd.
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
#ifndef MINDSPORE_INCLUDE_JS_API_MSLITE_MODEL_CALLBACK_NAPI_H
#define MINDSPORE_INCLUDE_JS_API_MSLITE_MODEL_CALLBACK_NAPI_H

#include <queue>
#include <uv.h>
#include "mslite_model_napi.h"
#include "ms_info.h"
#include "common_napi.h"

namespace mindspore {
enum class AsyncWorkType : int32_t {
  ASYNC_WORK_PREPARE = 0,
  ASYNC_WORK_PLAY,
  ASYNC_WORK_PAUSE,
  ASYNC_WORK_STOP,
  ASYNC_WORK_RESET,
  ASYNC_WORK_SEEK,
  ASYNC_WORK_SPEED,
  ASYNC_WORK_VOLUME,
  ASYNC_WORK_BITRATE,
  ASYNC_WORK_INVALID,
};
}  // namespace mindspore
#endif  // COMMON_NAPI_H