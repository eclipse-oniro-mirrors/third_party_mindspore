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

#include "hi_app_event_thread.h"
#include <time.h>
#include <cstdlib>
#include "src/common/log.h"
#include "app_event.h"
#include "app_event_processor_mgr.h"

namespace mindspore {
namespace lite {
HiAppEventThread::HiAppEventThread()
: running_(false), event_map_()
{ }

HiAppEventThread::~HiAppEventThread() {
    AbortAndJoin();
}

bool HiAppEventThread::Running() const {
    return running_.load();
}

bool HiAppEventThread::Init() {
    int ret = HiAppEventAddProcessor();
    if (ret == -1) {
        return false;
    }
    running_.store(true);
    thread_ = std::thread(&HiAppEventThread::RunFunc, this);
    return true;
}

void HiAppEventThread::WriteHiAppEvent() {
    // when there is no sandbox environment for hap, for example, xts, skip writing app events.
    if (kProcessID == kAppEventNotHapErrCode) {
        return;
    }
    for (auto &e : event_map_) {
        OHOS::HiviewDFX::HiAppEvent::Event event(kDomain, kEventName, OHOS::HiviewDFX::HiAppEvent::BEHAVIOR);
        std::string api_name = e.first;
        auto e_info = e.second;
        event.AddParam("api_name", api_name);
        event.AddParam("sdk_name", std::string(kSdkName));
        event.AddParam("begin_time", static_cast<int64_t>(e_info->begin_time));
        event.AddParam("success_times", e_info->success_times);
        event.AddParam("call_times", e_info->call_times);
        event.AddParam("min_cost_time", e_info->min_cost_time);
        event.AddParam("max_cost_time", e_info->max_cost_time);
        event.AddParam("total_cost_time", e_info->total_cost_time);
        OHOS::HiviewDFX::HiAppEvent::Write(event);
    }
    event_map_.clear();
}

int64_t HiAppEventThread::HiAppEventAddProcessor() {
    std::srand(std::time(NULL));
    OHOS::HiviewDFX::HiAppEvent::ReportConfig config;
    config.name = kName;
    config.appId = kAppId;
    config.routeInfo = "AUTO";
    config.triggerCond.timeout = kTimeOut;
    config.triggerCond.row = kCondRow;
    config.eventConfigs.clear();
    {// not allow to modify
        OHOS::HiviewDFX::HiAppEvent::EventConfig event1;
        event1.domain = "api_diagnostic";
        event1.name = "api_exec_end";
        event1.isRealTime = false;
        config.eventConfigs.push_back(event1);
    }
    { // not allow to modify
        OHOS::HiviewDFX::HiAppEvent::EventConfig event2;
        event2.domain = "api_diagnostic";
        event2.name = "api_called_stat";
        event2.isRealTime = true;
        config.eventConfigs.push_back(event2);
    }
    { // not allow to modify
        OHOS::HiviewDFX::HiAppEvent::EventConfig event3;
        event3.domain = "api_diagnostic";
        event3.name = "api_called_stat_cnt";
        event3.isRealTime = true;
        config.eventConfigs.push_back(event3);
    }
    if (kProcessID == -1) {
        kProcessID = OHOS::HiviewDFX::HiAppEvent::AppEventProcessorMgr::AddProcessor(config);
    }
    return kProcessID;
}

void HiAppEventThread::RunFunc() {
    (void) pthread_setname_np(pthread_self(), "OS_MSEvent");
    while (running_.load()) {
        std::unique_lock<std::mutex> lock(mutex_);
        auto status = condition_.wait_for(lock, std::chrono::seconds(kWaitTime));

        if (!running_.load() || status == std::cv_status::timeout) {
            // write all events and clear event_map
            WriteHiAppEvent();
        }
    }
    running_.store(false);
}

void HiAppEventThread::AbortAndJoin() {
    running_.store(false);
    condition_.notify_one();
    if (thread_.joinable()) {
        thread_.join();
    }
}

void HiAppEventThread::Submit(const int result, const int err_code, const std::string &api_name,
                              const uint64_t begin_time, const uint64_t end_time, const std::string &devices) {
    // add and merge the data by api_name
    std::lock_guard<std::mutex> lock(mutex_);
    int64_t cost_time = end_time - begin_time;
    if (cost_time < 0) {
        MS_LOG(WARNING) << api_name <<"'s cost_time is negative, don't report it.";
        return;
    }
    auto iter = event_map_.find(api_name);
    if (iter != event_map_.end()) {
        std::shared_ptr<EventInfo> event_info = iter->second;
        event_info->call_times++;
        event_info->success_times += (result == RET_OK);
        event_info->total_cost_time += cost_time;
        event_info->min_cost_time = std::min(event_info->min_cost_time, cost_time);
        event_info->max_cost_time = std::max(event_info->max_cost_time, cost_time);
        return;
    }
    std::shared_ptr<EventInfo> event_info = std::make_shared<EventInfo>();
    event_info->begin_time = begin_time;
    event_info->call_times = 1;
    event_info->success_times = (result == RET_OK);
    event_info->total_cost_time = cost_time;
    event_info->min_cost_time = cost_time;
    event_info->max_cost_time = cost_time;
    event_map_.emplace(api_name, event_info);
}
}  // mindspore
}  // lite