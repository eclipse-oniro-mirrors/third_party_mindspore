/*
 * Copyright (c) 2023 Huawei Device Co., Ltd.
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

#ifndef OHOS_MINDSPORE_TEST_LOG_H
#define OHOS_MINDSPORE_TEST_LOG_H

#include <cstdarg>
#include "hilog/log_c.h"

#ifdef __cplusplus
extern "C" {
#endif

constexpr unsigned int MSLITE_LOG_DOMAIN = 0xD002100;

#define LOGD(...) HiLogPrint(LOG_CORE, LOG_DEBUG, MSLITE_LOG_DOMAIN, "MS_LITE", __VA_ARGS__)
#define LOGI(...) HiLogPrint(LOG_CORE, LOG_INFO, MSLITE_LOG_DOMAIN, "MS_LITE", __VA_ARGS__)
#define LOGW(...) HiLogPrint(LOG_CORE, LOG_WARN, MSLITE_LOG_DOMAIN, "MS_LITE", __VA_ARGS__)
#define LOGE(...) HiLogPrint(LOG_CORE, LOG_ERROR, MSLITE_LOG_DOMAIN, "MS_LITE", __VA_ARGS__)
#define LOGF(...) HiLogPrint(LOG_CORE, LOG_FATAL, MSLITE_LOG_DOMAIN, "MS_LITE", __VA_ARGS__)

#ifdef __cplusplus
}
#endif

#endif //OHOS_MINDSPORE_TEST_LOG_H
