/*
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

#ifndef THIRD_PARTY_MINDSPORE_CONTEXT_C_FUZZER_H
#define THIRD_PARTY_MINDSPORE_CONTEXT_C_FUZZER_H

#include <stdint.h>
#include <stddef.h>

bool MSContextFuzzTest_ThreadNum(const uint8_t* data, size_t size);
bool MSContextFuzzTest_ThreadAffinityMode(const uint8_t* data, size_t size);
bool MSContextFuzzTest_Provider(const uint8_t* data, size_t size);
bool MSContextFuzzTest(const uint8_t* data, size_t size);

#endif //THIRD_PARTY_MINDSPORE_CONTEXT_C_FUZZER_H
