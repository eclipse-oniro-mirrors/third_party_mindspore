/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifdef ENABLE_HI_APP_EVENT
#include "src/common/hi_app_event/hi_app_event.h"
#include <time.h>
#include <cstdlib>
#include <map>
#include "app_event.h"
#include "app_event_processor_mgr.h"

namespace mindspore {
namespace lite {
HiAppEvent& HiAppEvent::GetInstance() {
    static HiAppEvent instance;
    return instance;
}

void HiAppEvent::Init() {
    bool ret = event_thread_.Init();
    if (!ret) {
        return;
    }
    is_ready_.store(true);
}

void HiAppEvent::Report(const int result, const int err_code, const std::string &api_name,
                              const uint64_t begin_time, const std::string &devices) {
    {
        std::lock_guard<std::mutex> init_guard_lock(init_mutex_);
        if (!is_ready_.load()) {
            Init();
        }
    }
    uint64_t end_time = GetTimeMs();
    event_thread_.Submit(result, err_code, api_name, begin_time, end_time, devices);
}

uint64_t HiAppEvent::GetTimeMs() {
    struct timespec ts = {0, 0};
    if (clock_gettime(CLOCK_REALTIME, &ts) != 0) {
        return 0;
    }
    uint64_t ret_val = static_cast<uint64_t>(ts.tv_sec * 1000LL + ts.tv_nsec / 1000000);
    return ret_val;
}

std::string HiAppEvent::GetApiType() const {
    return api_type_;
}

void HiAppEvent::SetApiType(const std::string &api_type){
    api_type_ = api_type;
}
}  // namespace lite
}  // namespace mindspore
#endif
