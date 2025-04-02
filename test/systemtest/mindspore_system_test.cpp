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

#include <inttypes.h>
#include <securec.h>
#include "gtest/gtest.h"
#include "include/c_api/context_c.h"
#include "include/c_api/model_c.h"
#include "include/c_api/types_c.h"
#include "include/c_api/status_c.h"
#include "include/c_api/data_type_c.h"
#include "include/c_api/tensor_c.h"
#include "include/c_api/format_c.h"
#include "../utils/common.h"
#include "../utils/model_utils.h"

class MSLiteTest: public testing::Test {
protected:
    static void SetUpTestCase(void) {}
    static void TearDownTestCase(void) {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

/*
 * @tc.name: end_to_end_0001
 * @tc.desc: Test MindSpore Runtime.
 * @tc.type: FUNC
 */
HWTEST(MSLiteTest, end_to_end_0001, testing::ext::TestSize.Level0) {
    printf("==========Init Context==========\n");
    OH_AI_ContextHandle context = OH_AI_ContextCreate();
    ASSERT_NE(context, nullptr);

    AddContextDeviceCPU(context);

    printf("==========Create model==========\n");
    OH_AI_ModelHandle model = OH_AI_ModelCreate();
    ASSERT_NE(model, nullptr);

    ModelPredict(model, context, "ml_face_isface", {}, false, true, false);
}