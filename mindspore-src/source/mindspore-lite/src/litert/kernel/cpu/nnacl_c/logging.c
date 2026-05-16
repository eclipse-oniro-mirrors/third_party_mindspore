/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "logging.h"
#if defined(__ANDROID__)
#include <android/log.h>
const char *ANDROID_LOG_TAG = "MS_LITE_NNACL";
static int GetAndroidLogLevel(NNACLLogLevel level) {
  switch (level) {
    case NNACL_LOG_DEBUG:
      return ANDROID_LOG_DEBUG;
    case NNACL_LOG_INFO:
      return ANDROID_LOG_INFO;
    case NNACL_LOG_WARNING:
      return ANDROID_LOG_WARN;
    case NNACL_LOG_ERROR:
    default:
      return ANDROID_LOG_ERROR;
  }
}
#endif

#ifdef MS_COMPILE_OHOS
#undef LOG_DOMAIN
#undef LOG_TAG
#define LOG_DOMAIN 0xD002102
#define LOG_TAG "MS_LITE"
#define NNACL_FORMAT "[%{public}s:%{public}d] %{public}s# %{public}s"
#include "hilog/log.h"
void NnaclPrintHiLog(NNACLLogLevel level, const char *file, int line, const char *func, const char *msg) {
#ifdef MS_COMPILE_WITH_OHOS_NDK
  // When build with OHOS NDK, use public api of hilog module.
  if (level == NNACL_LOG_DEBUG) {
    OH_LOG_Print(LOG_APP, NNACL_LOG_DEBUG, LOG_DOMAIN, LOG_TAG, NNACL_FORMAT, file, line, func, msg);
  } else if (level == NNACL_LOG_INFO) {
    OH_LOG_Print(LOG_APP, NNACL_LOG_INFO, LOG_DOMAIN, LOG_TAG, NNACL_FORMAT, file, line, func, msg);
  } else if (level == NNACL_LOG_WARNING) {
    OH_LOG_Print(LOG_APP, NNACL_LOG_WARNING, LOG_DOMAIN, LOG_TAG, NNACL_FORMAT, file, line, func, msg);
  } else if (level == NNACL_LOG_ERROR) {
    OH_LOG_Print(LOG_APP, NNACL_LOG_ERROR, LOG_DOMAIN, LOG_TAG, NNACL_FORMAT, file, line, func, msg);
  } else {
    OH_LOG_Print(LOG_APP, NNACL_LOG_ERROR, LOG_DOMAIN, LOG_TAG, NNACL_FORMAT, file, line, func, msg);
  }
#else
  // When build in OHOS repo, use inner api of hilog module.
  if (level == NNACL_LOG_DEBUG) {
    HILOG_DEBUG(LOG_APP, NNACL_FORMAT, file, line, func, msg);
  } else if (level == NNACL_LOG_INFO) {
    HILOG_INFO(LOG_APP, NNACL_FORMAT, file, line, func, msg);
  } else if (level == NNACL_LOG_WARNING) {
    HILOG_WARN(LOG_APP, NNACL_FORMAT, file, line, func, msg);
  } else if (level == NNACL_LOG_ERROR) {
    HILOG_ERROR(LOG_APP, NNACL_FORMAT, file, line, func, msg);
  } else {
    HILOG_ERROR(LOG_APP, NNACL_FORMAT, file, line, func, msg);
  }
#endif
}
#endif
int NnaclStrToInt(const char *env) {
  if (env == NULL) {
    return 2;
  }
  if (strcmp(env, "0") == 0) {
    return 0;
  }
  if (strcmp(env, "1") == 0) {
    return 1;
  }
  if (strcmp(env, "2") == 0) {
    return 2;
  }
  if (strcmp(env, "3") == 0) {
    return 3;
  }
  return 2;
}
int NnaclIsPrint(int level) {
  const char *env = getenv("GLOG_v");
  const int ms_level = NnaclStrToInt(env);
  if (level < 0) {
    level = 2;
  }
  if (level >= ms_level) {
    return 1;
  }
  return 0;
}
void NnaclLogOutput(NNACLLogLevel level, const char *file, int line, const char *func, const char *fmt, ...) {
  if (NnaclIsPrint(level) == 0) {
    return;
  }
  char buffer[256];
  va_list args;
  va_start(args, fmt);
  vsnprintf(buffer, sizeof(buffer), fmt, args);
  va_end(args);

#if defined(__ANDROID__)
  __android_log_print(GetAndroidLogLevel(level), ANDROID_LOG_TAG, "[%s:%d] (%s): %s", file, line, func, buffer);

#elif defined(MS_COMPILE_OHOS)
  NnaclPrintHiLog(level, file, line, func, buffer);

#else
  const char *level_str[] = {"DEBUG", "INFO", "WARNING", "ERROR"};
  printf("[%s] %s:%d (%s): %s", level_str[level], file, line, func, buffer);
  printf("\n");
#endif
}
