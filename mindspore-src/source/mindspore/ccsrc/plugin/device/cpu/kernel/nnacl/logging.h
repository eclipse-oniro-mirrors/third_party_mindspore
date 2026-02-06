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
#ifndef NNACL_LOG_H_
#define NNACL_LOG_H_
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef NNACL_LITE_LOG_HEAR_FILE_REL_PATH
#define NNACL_LITE_LOG_HEAR_FILE_REL_PATH "mindspore/ccsrc/plugin/device/cpu/kernel/nnacl/logging.h"
#endif

#define GET_NNACL_REAL_PATH_POS                                     \
  (sizeof(__FILE__) > sizeof(NNACL_LITE_LOG_HEAR_FILE_REL_PATH)     \
     ? sizeof(__FILE__) - sizeof(NNACL_LITE_LOG_HEAR_FILE_REL_PATH) \
     : 0)

#define NNACL_LITE_FILE_NAME (&__FILE__[GET_NNACL_REAL_PATH_POS])
typedef enum { NNACL_LOG_DEBUG = 0, NNACL_LOG_INFO, NNACL_LOG_WARNING, NNACL_LOG_ERROR } NNACLLogLevel;
void NnaclLogOutput(NNACLLogLevel level, const char *file, int line, const char *func, const char *fmt, ...);

#define NNACL_LOG_IF(level, msg, ...) \
  NnaclLogOutput(NNACL_LOG_##level, NNACL_LITE_FILE_NAME, __LINE__, __func__, msg, ##__VA_ARGS__)

#define NNACL_LOG_DEBUG(msg, ...) NNACL_LOG_IF(DEBUG, msg, ##__VA_ARGS__)
#define NNACL_LOG_INFO(msg, ...) NNACL_LOG_IF(INFO, msg, ##__VA_ARGS__)
#define NNACL_LOG_WARNING(msg, ...) NNACL_LOG_IF(WARNING, msg, ##__VA_ARGS__)
#define NNACL_LOG_ERROR(msg, ...) NNACL_LOG_IF(ERROR, msg, ##__VA_ARGS__)

#endif  // NNACL_LOG_H_
