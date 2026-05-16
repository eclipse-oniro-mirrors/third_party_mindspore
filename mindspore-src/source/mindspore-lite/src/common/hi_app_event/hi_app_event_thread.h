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

#ifndef LITE_HANDLER_THREAD_H
#define LITE_HANDLER_THREAD_H

#include <list>
#include <thread>
#include <atomic>
#include <memory>
#include <functional>
#include <stdexcept>
#include <mutex>
#include <iostream>
#include <string>
#include <condition_variable>
#include <map>
#include <cstdint>

namespace mindspore {
namespace lite {
namespace {
    constexpr auto kName = "ha_app_event";
    constexpr auto kAppId = "com_huawei_hmos_sdk_ocg";
    constexpr int32_t kTimeOut = 90;
    constexpr int32_t kCondRow = 30;
    constexpr auto kDomain = "api_diagnostic";
    constexpr auto kEventName = "api_called_stat";
    constexpr auto kSdkName = "MindSporeLiteKit";
    constexpr int64_t kAppEventNotHapErrCode = -200;
    constexpr int32_t kWaitTime = 60;
    static int64_t kProcessID = -1;
    constexpr int32_t RET_OK = 0;

    struct EventInfo {
        int64_t begin_time;
        int64_t call_times;
        int64_t success_times;
        int64_t min_cost_time;
        int64_t max_cost_time;
        int64_t total_cost_time;
        EventInfo() : begin_time(0), call_times(0), success_times(0), min_cost_time(0), max_cost_time(0),
                      total_cost_time(0) {}
    };
}

class HiAppEventThread {
public:
    HiAppEventThread();
    ~HiAppEventThread();

    HiAppEventThread(const HiAppEventThread &) = delete;
    HiAppEventThread &operator()(const HiAppEventThread &) = delete;

    bool Init();

    void Submit(const int result, const int err_code, const std::string &api_name,
                const uint64_t begin_time, const uint64_t end_time, const std::string &devices);

    void WriteHiAppEvent();

    int64_t HiAppEventAddProcessor();

private:

    bool Running() const;

    void RunFunc();

    void AbortAndJoin();

private:
    std::thread thread_;
    std::mutex mutex_;
    std::condition_variable condition_;

    std::atomic_bool running_;
    std::map<std::string, std::shared_ptr<EventInfo>> event_map_;
};
}  // mindspore
}  // lite
#endif  // LITE_HANDLER_THREAD_H
