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

#ifndef MINDSPORE_LITE_HI_APP_EVENT_H_
#define MINDSPORE_LITE_HI_APP_EVENT_H_

#ifdef ENABLE_HI_APP_EVENT
#include "hi_app_event_thread.h"
#include <string>
#include <atomic>
#include <memory>
#include <mutex>

namespace mindspore {
namespace lite {
class HiAppEvent {
public:
    static HiAppEvent& GetInstance();

    HiAppEvent(const HiAppEvent &) = delete;
    HiAppEvent & operator=(const HiAppEvent &) = delete;

    void Report(const int result, const int err_code, const std::string &api_name,
                              const uint64_t begin_time, const std::string &devices = "None");

    static uint64_t GetTimeMs();

    std::string GetApiType() const;

    void SetApiType(const std::string &api_type);

private:
    void Init();
    HiAppEvent() = default;
    ~HiAppEvent() = default;

    HiAppEventThread event_thread_;
    std::string api_type_ = "ts_api";
    std::atomic_bool is_ready_ = false;
    std::mutex init_mutex_;
};
}  // namespace lite
}  // namespace mindspore
#endif
#endif  // MINDSPORE_LITE_HI_APP_EVENT_H_
