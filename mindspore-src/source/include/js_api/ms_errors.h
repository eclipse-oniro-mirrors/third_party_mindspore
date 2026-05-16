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
#ifndef MINDSPORE_INCLUDE_JS_API_MS_ERRORS_H
#define MINDSPORE_INCLUDE_JS_API_MS_ERRORS_H

namespace mindspore {
const int32_t BASE_MSLITE_ERR_OFFSET = 1000199;

/** Success */
const int32_t SUCCESS = 0;

/** Fail */
const int32_t ERROR = BASE_MSLITE_ERR_OFFSET;

/** Status error */
const int32_t ERR_ILLEGAL_STATE = BASE_MSLITE_ERR_OFFSET - 1;

/** Invalid parameter */
const int32_t ERR_INVALID_PARAM = BASE_MSLITE_ERR_OFFSET - 2;

/** Not existed parameter */
const int32_t ERR_NOT_EXISTED_PARAM = BASE_MSLITE_ERR_OFFSET - 3;

/** Invalid operation */
const int32_t ERR_INVALID_OPERATION = BASE_MSLITE_ERR_OFFSET - 4;
}  // namespace mindspore
#endif  // MS_ERRORS_H