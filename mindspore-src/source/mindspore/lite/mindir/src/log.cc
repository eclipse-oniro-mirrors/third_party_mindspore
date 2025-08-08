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
#include "log.h"
#include <cstring>
#include <cstdio>

#if defined(__ANDROID__)
#include <android/log.h>
#endif

#ifdef MS_COMPILE_OHOS
#define LOG_DOMAIN 0xD002102
#define LOG_TAG "MS_LITE"
#define FORMAT "[%{public}s:%{public}d] %{public}s# %{public}s"
#include "hilog/log.h"
#endif

// namespace to support utils module definition namespace mindspore constexpr const char *ANDROID_LOG_TAG = "MS_LITE";
namespace mindspore::lite {
#if defined(__ANDROID__)
constexpr const char *ANDROID_LOG_TAG = "MS_LITE";
#endif

int StrToInt(const char *env) {
  if (env == nullptr) {
    return static_cast<int>(LiteLogLevel::WARNING);
  }
  if (strcmp(env, "0") == 0) {
    return static_cast<int>(LiteLogLevel::DEBUG);
  }
  if (strcmp(env, "1") == 0) {
    return static_cast<int>(LiteLogLevel::INFO);
  }
  if (strcmp(env, "2") == 0) {
    return static_cast<int>(LiteLogLevel::WARNING);
  }
  if (strcmp(env, "3") == 0) {
    return static_cast<int>(LiteLogLevel::ERROR);
  }
  return static_cast<int>(LiteLogLevel::WARNING);
}

bool IsPrint(int level) {
  static const char *const env = std::getenv("GLOG_v");
  static const int ms_level = StrToInt(env);
  if (level < 0) {
    level = 2;
  }
  return level >= ms_level;
}

#if defined(__ANDROID__)
static int GetAndroidLogLevel(LiteLogLevel level) {
  switch (level) {
    case LiteLogLevel::DEBUG:
      return ANDROID_LOG_DEBUG;
    case LiteLogLevel::INFO:
      return ANDROID_LOG_INFO;
    case LiteLogLevel::WARNING:
      return ANDROID_LOG_WARN;
    case LiteLogLevel::ERROR:
    default:
      return ANDROID_LOG_ERROR;
  }
}
#endif

#ifdef MS_COMPILE_OHOS
void PrintHiLog(LiteLogLevel level, const char *file, int line, const char *func, const char *msg) {
#ifdef MS_COMPILE_WITH_OHOS_NDK
  // When build with OHOS NDK, use public api of hilog module.
  if (level == LiteLogLevel::DEBUG) {
    OH_LOG_Print(LOG_APP, LOG_DEBUG, LOG_DOMAIN, LOG_TAG, FORMAT, file, line, func, msg);
  } else if (level == LiteLogLevel::INFO) {
    OH_LOG_Print(LOG_APP, LOG_INFO, LOG_DOMAIN, LOG_TAG, FORMAT, file, line, func, msg);
  } else if (level == LiteLogLevel::WARNING) {
    OH_LOG_Print(LOG_APP, LOG_WARN, LOG_DOMAIN, LOG_TAG, FORMAT, file, line, func, msg);
  } else if (level == LiteLogLevel::ERROR) {
    OH_LOG_Print(LOG_APP, LOG_ERROR, LOG_DOMAIN, LOG_TAG, FORMAT, file, line, func, msg);
  } else {
    OH_LOG_Print(LOG_APP, LOG_ERROR, LOG_DOMAIN, LOG_TAG, FORMAT, file, line, func, msg);
  }
#else
  // When build in OHOS repo, use inner api of hilog module.
  if (level == LiteLogLevel::DEBUG) {
    HILOG_DEBUG(LOG_APP, FORMAT, file, line, func, msg);
  } else if (level == LiteLogLevel::INFO) {
    HILOG_INFO(LOG_APP, FORMAT, file, line, func, msg);
  } else if (level == LiteLogLevel::WARNING) {
    HILOG_WARN(LOG_APP, FORMAT, file, line, func, msg);
  } else if (level == LiteLogLevel::ERROR) {
    HILOG_ERROR(LOG_APP, FORMAT, file, line, func, msg);
  } else {
    HILOG_ERROR(LOG_APP, FORMAT, file, line, func, msg);
  }
#endif
}
#endif

const char *EnumStrForMsLogLevel(LiteLogLevel level) {
  if (level == LiteLogLevel::DEBUG) {
    return "DEBUG";
  } else if (level == LiteLogLevel::INFO) {
    return "INFO";
  } else if (level == LiteLogLevel::WARNING) {
    return "WARNING";
  } else if (level == LiteLogLevel::ERROR) {
    return "ERROR";
  } else {
    return "NO_LEVEL";
  }
}

void LiteLogWriter::OutputLog(const std::ostringstream &msg) const {
  if (IsPrint(static_cast<int>(log_level_))) {
#if defined(__ANDROID__)
    __android_log_print(GetAndroidLogLevel(log_level_), ANDROID_LOG_TAG, "[%s:%d] %s] %s", location_.file_,
                        location_.line_, location_.func_, msg.str().c_str());
#elif defined(MS_COMPILE_OHOS)
    PrintHiLog(log_level_, location_.file_, location_.line_, location_.func_, msg.str().c_str());
#else
    printf("%s [%s:%d] %s] %s\n", EnumStrForMsLogLevel(log_level_), location_.file_, location_.line_, location_.func_,
           msg.str().c_str());
#endif
  }
}

void LiteLogWriter::operator<(const LiteLogStream &stream) const noexcept {
  std::ostringstream msg;
  msg << stream.sstream_->rdbuf();
  OutputLog(msg);
}
}  // namespace mindspore
