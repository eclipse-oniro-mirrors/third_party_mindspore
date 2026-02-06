/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_MINDRT_RUNTIME_THREADPOOL_LOG_H_
#define MINDSPORE_CORE_MINDRT_RUNTIME_THREADPOOL_LOG_H_
#ifdef MS_COMPILE_OHOS
#include "hilog/log.h"
#endif
namespace mindspore {
#ifdef THREAD_POOL_DEBUG
#include <stdio.h>
#define THREAD_DEBUG(content, args...) \
  { printf("[DEBUG] %s|%d: " #content "\r\n", __func__, __LINE__, ##args); }
#define THREAD_INFO(content, args...) \
  { printf("[INFO] %s|%d: " #content "\r\n", __func__, __LINE__, ##args); }
#define THREAD_ERROR(content, args...) \
  { printf("[ERROR] %s|%d: " #content "\r\n", __func__, __LINE__, ##args); }
#define THREAD_TEST_TRUE(flag)                                  \
  if (flag) {                                                   \
    printf("[ERROR] %s|%d: " #flag "\r\n", __func__, __LINE__); \
  }
#else
#define THREAD_DEBUG(content, ...)
#define THREAD_TEST_TRUE(flag)

#if defined(__ANDROID__)
#define THREAD_INFO(content, ...)
#include <android/log.h>
#define THREAD_ERROR(content, args...) \
  { __android_log_print(ANDROID_LOG_ERROR, "MS_LITE", "%s|%d: " #content "\r\n", __func__, __LINE__, ##args); }

#elif defined(MS_COMPILE_OHOS) // For OHOS, use hilog.

#define MINDRT_OHOS_LOG_DOMAIN 0x2102
#define MINDRT_OHOS_LOG_TAG "MS_LITE"

#ifdef MS_COMPILE_WITH_OHOS_NDK
// When build with OHOS NDK, use public api of hilog module.
#define THREAD_INFO(content, args...) \
  { OH_LOG_Print(LOG_APP, LOG_INFO, MINDRT_OHOS_LOG_DOMAIN, MINDRT_OHOS_LOG_TAG, "%s:%d " #content, __func__, __LINE__, ##args); }
#define THREAD_ERROR(content, args...) \
  { OH_LOG_Print(LOG_APP, LOG_ERROR, MINDRT_OHOS_LOG_DOMAIN, MINDRT_OHOS_LOG_TAG, "%s:%d " #content, __func__, __LINE__, ##args); }
#else
// When build in OHOS repo, use inner api of hilog module.
#define THREAD_INFO(content, args...) \
  { HiLogPrint(LOG_APP, LOG_INFO, MINDRT_OHOS_LOG_DOMAIN, MINDRT_OHOS_LOG_TAG, "%s:%d " #content, __func__, __LINE__, ##args); }
#define THREAD_ERROR(content, args...) \
  { HiLogPrint(LOG_APP, LOG_ERROR, MINDRT_OHOS_LOG_DOMAIN, MINDRT_OHOS_LOG_TAG, "%s:%d " #content, __func__, __LINE__, ##args); }
#endif

#else
#define THREAD_INFO(content, ...)
#define THREAD_ERROR(content, ...)
#endif
#endif

#define THREAD_ERROR_IF_NULL(ptr) \
  do {                            \
    if ((ptr) == nullptr) {       \
      return THREAD_ERROR;        \
    }                             \
  } while (0)

#define THREAD_RETURN_IF_NULL(ptr) \
  do {                             \
    if ((ptr) == nullptr) {        \
      return;                      \
    }                              \
  } while (0)

/* Thread return code */
constexpr int THREAD_OK = 0;
constexpr int THREAD_ERROR = 1;
}  // namespace mindspore
#endif  // MINDSPORE_CORE_MINDRT_RUNTIME_THREADPOOL_LOG_H_
