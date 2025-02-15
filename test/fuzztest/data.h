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

#ifndef OHOS_MINDSPORE_TEST_FUZZTEST_DATA_H
#define OHOS_MINDSPORE_TEST_FUZZTEST_DATA_H

#include "securec.h"
#include "../utils/log.h"

class Data {
public:
    Data(const uint8_t *data, size_t size) {
        dataFuzz = data;
        dataSize = size;
    }

    template<class T> T GetData() {
        T object {};
        size_t objectSize = sizeof(object);
        if (dataFuzz == nullptr || objectSize > dataSize - dataPos) {
            LOGE("Date is not enough");
            return {};
        }
        if (memcpy_s(&object, objectSize, dataFuzz + dataPos, objectSize) != EOK) {
            LOGE("memcpy_s failed.");
            return {};
        }
        dataPos = dataPos + objectSize;
        return object;
    }

    const uint8_t* GetNowData() const {
        return dataFuzz + dataPos;
    }

    size_t GetNowDataSize() const {
        return dataSize - dataPos;
    }

private:
    const uint8_t* dataFuzz {nullptr};
    size_t dataSize {0};
    size_t dataPos {0};
};

#endif //OHOS_MINDSPORE_TEST_FUZZTEST_DATA_H
