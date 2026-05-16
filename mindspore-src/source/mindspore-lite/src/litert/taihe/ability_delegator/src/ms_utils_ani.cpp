/*
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "ms_utils_ani.h"

namespace mindspore_ani {

struct TrainModelPrepareANI {
  std::shared_ptr<mindspore::Context> context;
  std::shared_ptr<mindspore::TrainCfg> train_cfg;
};

std::optional<TrainModelPrepareANI> CreateTrainModelPrepareANI(MSLiteContextInfoANI *context_info_ptr) {
  if (!context_info_ptr || context_info_ptr->target.empty()) {
    ThrowBusinessError(MS_LOAD_CONTEXT_ERROR);
  }

  auto context = std::make_shared<mindspore::Context>();
  if (!context) {
    ThrowBusinessError(MS_LOAD_CONTEXT_ERROR);
  }

  auto &device_infos = context->MutableDeviceInfo();
  if (GetDeviceInfoContextANI(context_info_ptr, device_infos) != MS_SUCCESS_ANI) {
    ThrowBusinessError(MS_LOAD_CONTEXT_ERROR);
  }

  auto train_cfg = std::make_shared<mindspore::TrainCfg>();
  std::vector<std::string> loss_names(train_cfg->GetLossName());
  loss_names.insert(loss_names.end(), context_info_ptr->train_cfg.loss_names.begin(),
                    context_info_ptr->train_cfg.loss_names.end());
  train_cfg->SetLossName(loss_names);
  train_cfg->optimization_level_ =
    static_cast<mindspore::OptimizationLevel>(context_info_ptr->train_cfg.optimization_level);

  return TrainModelPrepareANI{context, train_cfg};
}

std::shared_ptr<mindspore::CPUDeviceInfo> CreateCPUDevice(const MSLiteContextInfoANI &context_ptr) {
  auto cpu_device = std::make_shared<mindspore::CPUDeviceInfo>();
  if (!cpu_device) return nullptr;
  bool is_fp16 = context_ptr.cpu_device.precision_mode == "preferred_fp16";
  cpu_device->SetEnableFP16(is_fp16);
  return cpu_device;
}

std::shared_ptr<mindspore::NNRTDeviceInfo> CreateNNRTDevice(const MSLiteContextInfoANI &context_ptr) {
  auto nnrt_device = std::make_shared<mindspore::NNRTDeviceInfo>();
  if (!nnrt_device) return nullptr;
  nnrt_device->SetDeviceID(context_ptr.nnrt_device.device_id);
  if (context_ptr.nnrt_device.performance_mode != MS_UNSET_VALUE_ANI) {
    nnrt_device->SetPerformanceMode(context_ptr.nnrt_device.performance_mode);
  }
  if (context_ptr.nnrt_device.priority != MS_UNSET_VALUE_ANI) {
    nnrt_device->SetPriority(context_ptr.nnrt_device.priority);
  }
  return nnrt_device;
}

int32_t GetDeviceInfoContextANI(MSLiteContextInfoANI *context_ptr,
                                std::vector<std::shared_ptr<mindspore::DeviceInfoContext>> &device_infos) {
  if (!context_ptr) {
    MS_LOG(ERROR) << "ANI context_ptr is nullptr, cannot create native devices.";
    return MS_INVALID_OPERATION_ANI;
  }

  for (const auto &device_name : context_ptr->target) {
    if (kDeviceTypesANI.find(device_name) == kDeviceTypesANI.end()) {
      MS_LOG(ERROR) << "ANI Invalid device name: " << device_name.c_str();
      return MS_INVALID_OPERATION_ANI;
    }

    auto device_type = kDeviceTypesANI.at(device_name);
    MS_LOG(DEBUG) << "ANI current device name: " << device_name;

    std::shared_ptr<mindspore::DeviceInfoContext> device_ptr = nullptr;

    switch (device_type) {
      case mindspore::kCPU:
        device_ptr = CreateCPUDevice(*context_ptr);
        break;
      case mindspore::kNNRt:
        device_ptr = CreateNNRTDevice(*context_ptr);
        break;
      default:
        MS_LOG(ERROR) << "ANI invalid device type.";
        return MS_INVALID_OPERATION_ANI;
    }

    if (!device_ptr) {
      MS_LOG(ERROR) << "ANI failed to create device: " << device_name;
      return MS_INVALID_OPERATION_ANI;
    }

    device_infos.push_back(device_ptr);
  }

  return MS_SUCCESS_ANI;
}

std::shared_ptr<mindspore::Model> BuildModelFromBufferANI(void *buffer, size_t size,
                                                          const std::shared_ptr<mindspore::Context> &context,
                                                          bool is_fd) {
  if (buffer == nullptr || size <= 0) {
    if (is_fd) {
      ThrowBusinessError(MS_LOAD_FD_NATIVE_ERROR_PREDICT);
    } else {
      ThrowBusinessError(MS_LOAD_BUFFER_ERROR_PREDICT);
    }
    return nullptr;
  }
  auto model_ptr = std::make_shared<mindspore::Model>();
  auto ret = model_ptr->Build(buffer, size, mindspore::kMindIR, context);
  if (ret != mindspore::kSuccess) {
    if (is_fd) {
      ThrowBusinessError(MS_LOAD_FD_NATIVE_ERROR_PREDICT);
    } else {
      ThrowBusinessError(MS_LOAD_BUFFER_NATIVE_ERROR_PREDICT);
    }
    return nullptr;
  }
  MS_LOG(INFO) << "ANI build model from buffer success.";
  return model_ptr;
}

std::shared_ptr<mindspore::Model> BuildModelFromPathANI(const std::string &path,
                                                        const std::shared_ptr<mindspore::Context> &context) {
  auto model_ptr = std::make_shared<mindspore::Model>();
  if (model_ptr == nullptr) {
    MS_LOG(ERROR) << "ANI failed to new native mindspore::Model.";
    return nullptr;
  }
  if (path.empty()) {
    ThrowBusinessError(MS_LOAD_PATH_ERROR_PREDICT);
    return nullptr;
  }
  auto ret = model_ptr->Build(path, mindspore::kMindIR, context);
  if (ret != mindspore::kSuccess) {
    ThrowBusinessError(MS_LOAD_NATIVE_ERROR_PREDICT);
    return nullptr;
  }
  MS_LOG(INFO) << "ANI build model from path success.";
  return model_ptr;
}

std::shared_ptr<mindspore::Model> CreateModelANI(MSLiteModelInfoANI *model_info_ptr,
                                                 MSLiteContextInfoANI *context_info_ptr) {
  std::lock_guard<std::mutex> lock(create_mutex_);
  if (context_info_ptr == nullptr) {
    ThrowBusinessError(MS_LOAD_CONTEXT_ERROR);
    return nullptr;
  }

  // create and init context
  auto context = std::make_shared<mindspore::Context>();
  if (context == nullptr) {
    ThrowBusinessError(MS_LOAD_CONTEXT_ERROR);
    return nullptr;
  }

  if (context_info_ptr->target.empty()) {
    ThrowBusinessError(MS_LOAD_CONTEXT_ERROR);
    return nullptr;
  }

  auto &device_infos = context->MutableDeviceInfo();
  if (GetDeviceInfoContextANI(context_info_ptr, device_infos) != MS_SUCCESS_ANI) {
    ThrowBusinessError(MS_LOAD_CONTEXT_ERROR);
    return nullptr;
  }
  context->SetThreadNum(context_info_ptr->cpu_device.thread_num);

  // load model
  switch (model_info_ptr->mode) {
    case MSLiteLoadModelMode::kBuffer:
      return BuildModelFromBufferANI(model_info_ptr->model_buffer_data, model_info_ptr->model_buffer_total, context,
                                     false);

    case MSLiteLoadModelMode::kPath:
      return BuildModelFromPathANI(model_info_ptr->model_path, context);

    case MSLiteLoadModelMode::kFD: {
      auto model =
        BuildModelFromBufferANI(model_info_ptr->model_buffer_data, model_info_ptr->model_buffer_total, context, true);
      (void)munmap(model_info_ptr->model_buffer_data, model_info_ptr->model_buffer_total);
      model_info_ptr->model_buffer_data = nullptr;
      return model;
    }
    default:
      ThrowBusinessError(MS_LOAD_WAY_ERROR_PREDICT);
      return nullptr;
  }
}

std::optional<mindspore::Graph> LoadGraphFromBufferTrainANI(void *buffer, size_t size, bool is_fd) {
  if (buffer == nullptr || size <= 0) {
    if (is_fd) {
      ThrowBusinessError(MS_LOAD_FD_ERROR_TRAIN);
    } else {
      ThrowBusinessError(MS_LOAD_BUFFER_ERROR_TRAIN);
    }
    return std::nullopt;
  }
  mindspore::Graph graph;
  auto status = mindspore::Serialization::Load(buffer, size, mindspore::kMindIR, &graph);
  if (status != mindspore::kSuccess) {
    if (is_fd) {
      ThrowBusinessError(MS_LOAD_FD_ERROR_TRAIN);
    } else {
      ThrowBusinessError(MS_LOAD_BUFFER_NATIVE_ERROR_TRAIN);
    }
    return std::nullopt;
  }
  return graph;
}

std::optional<mindspore::Graph> LoadGraphFromPathTrainANI(const std::string &path) {
  mindspore::Graph graph;
  if (path.empty()) {
    ThrowBusinessError(MS_LOAD_PATH_ERROR_TRAIN);
    return std::nullopt;
  }
  auto status = mindspore::Serialization::Load(path, mindspore::kMindIR, &graph);
  if (status != mindspore::kSuccess) {
    ThrowBusinessError(MS_LOAD_PATH_NATIVE_ERROR_TRAIN);
    return std::nullopt;
  }
  return graph;
}

std::shared_ptr<mindspore::Model> BuildTrainModelANI(const mindspore::Graph &graph,
                                                     const std::shared_ptr<mindspore::Context> &context,
                                                     const std::shared_ptr<mindspore::TrainCfg> &train_cfg) {
  auto model_ptr = std::make_shared<mindspore::Model>();
  if (model_ptr == nullptr) {
    MS_LOG(ERROR) << "ANI failed to new mindspore::Model.";
    return nullptr;
  }
  auto ret = model_ptr->Build(static_cast<mindspore::GraphCell>(graph), context, train_cfg);
  if (ret != mindspore::kSuccess) {
    MS_LOG(ERROR) << "ANI build train model failed.";
    return nullptr;
  }
  MS_LOG(INFO) << "ANI build train model success.";
  return model_ptr;
}

std::shared_ptr<mindspore::Model> CreateTrainModelANI(MSLiteModelInfoANI *model_info_ptr,
                                                      MSLiteContextInfoANI *context_info_ptr) {
  std::lock_guard<std::mutex> lock(create_mutex_);

  auto prepare_opt = CreateTrainModelPrepareANI(context_info_ptr);
  if (!prepare_opt) return nullptr;

  auto &context = prepare_opt->context;
  auto &train_cfg = prepare_opt->train_cfg;

  switch (model_info_ptr->mode) {
    case MSLiteLoadModelMode::kBuffer: {
      auto graph_opt =
        LoadGraphFromBufferTrainANI(model_info_ptr->model_buffer_data, model_info_ptr->model_buffer_total, false);
      if (!graph_opt) return nullptr;
      return BuildTrainModelANI(*graph_opt, context, train_cfg);
    }
    case MSLiteLoadModelMode::kPath: {
      auto graph_opt = LoadGraphFromPathTrainANI(model_info_ptr->model_path);
      if (!graph_opt) return nullptr;
      return BuildTrainModelANI(*graph_opt, context, train_cfg);
    }
    case MSLiteLoadModelMode::kFD: {
      auto graph_opt =
        LoadGraphFromBufferTrainANI(model_info_ptr->model_buffer_data, model_info_ptr->model_buffer_total, true);
      if (!graph_opt) return nullptr;
      auto model = BuildTrainModelANI(*graph_opt, context, train_cfg);
      (void)munmap(model_info_ptr->model_buffer_data, model_info_ptr->model_buffer_total);
      model_info_ptr->model_buffer_data = nullptr;
      return model;
    }
    default:
      ThrowBusinessError(MS_LOAD_WAY_ERROR_PREDICT);
      return nullptr;
  }
}

void ParseTargetTaihe(const ::ohos::ai::mindSporeLite::Context &context, MSLiteContextInfoANI *context_info_ptr) {
  auto taihe_target = context.target;
  if (!bool(taihe_target)) {
    context_info_ptr->target.push_back("cpu");
    MS_LOG(WARNING) << "ANI target is {}, set default target to cpu.";
    return;
  }
  for (const auto &t : taihe_target.value()) {
    context_info_ptr->target.push_back(std::string(t));
  }
}

void ParseCpuTaihe(const ::ohos::ai::mindSporeLite::Context &context, MSLiteContextInfoANI *context_info_ptr) {
  auto taihe_cpu = context.cpu;
  if (!bool(taihe_cpu)) {
    return;
  }
  MS_LOG(DEBUG) << "ANI taihe_cpu is not null.";

  auto &cpu_device = context_info_ptr->cpu_device;

  cpu_device.thread_num =
    taihe_cpu.value().threadNum.has_value() ? static_cast<int>(taihe_cpu.value().threadNum.value()) : MS_PARAM2_ANI;

  cpu_device.thread_affinity_mode = taihe_cpu.value().threadAffinityMode.has_value()
                                      ? static_cast<int>(taihe_cpu.value().threadAffinityMode.value())
                                      : MS_PARAM0_ANI;

  cpu_device.thread_affinity_cores.clear();
  if (taihe_cpu.value().threadAffinityCoreList.has_value()) {
    for (auto core : taihe_cpu.value().threadAffinityCoreList.value()) {
      cpu_device.thread_affinity_cores.push_back(core);
    }
  }

  cpu_device.precision_mode =
    taihe_cpu.value().precisionMode.has_value() ? std::string(taihe_cpu.value().precisionMode.value()) : "enforce_fp32";
}

int ParseNnrtDeviceFromTaiheContext(const ::ohos::ai::mindSporeLite::Context &context,
                                    MSLiteContextInfoANI *context_info_ptr) {
  auto taihe_nnrt = context.nnrt;
  if (!bool(taihe_nnrt)) {
    return MS_SUCCESS_ANI;
  }
  auto &nnrt_device = context_info_ptr->nnrt_device;
  auto device_id = taihe_nnrt.value().deviceID;
  if (bool(device_id)) {
    size_t value = 0;
    const auto data = device_id.value();
    for (size_t i = 0; i < data.size(); i++) {
      value |= (static_cast<size_t>(data[i]) << (8 * i));
    }
    nnrt_device.device_id = value;
    MS_LOG(WARNING) << "ANI get nnrt device id from context, is:" << value;
  } else {
    size_t num = 0;
    auto *desc = OH_AI_GetAllNNRTDeviceDescs(&num);
    if (desc == nullptr || num == 0) {
      MS_LOG(WARNING) << "ANI failed to get nnrt device id, skip adding nnrt device info.";
    }
    auto id = OH_AI_GetDeviceIdFromNNRTDeviceDesc(desc);
    OH_AI_DestroyAllNNRTDeviceDescs(&desc);
    nnrt_device.device_id = id;
    MS_LOG(DEBUG) << "ANI set nnrt device id to " << id;
  }
  auto performance_mode = taihe_nnrt.value().performanceMode;
  nnrt_device.performance_mode = MS_UNSET_VALUE_ANI;
  if (bool(performance_mode)) {
    int performance_id = static_cast<int>(performance_mode.value());
    if (performance_id > MS_PARAM4_ANI || performance_id < MS_PARAM0_ANI) {
      MS_LOG(ERROR) << "ANI performanceMode out of range: " << performance_id;
      return MS_INVALID_PARAM_ANI;
    }
    nnrt_device.performance_mode = performance_id;
  }
  auto priority_id = taihe_nnrt.value().priority;
  nnrt_device.priority = MS_UNSET_VALUE_ANI;
  if (bool(priority_id)) {
    int priority_value = static_cast<int>(priority_id.value());
    if (priority_value > MS_PARAM3_ANI || priority_value < MS_PARAM0_ANI) {
      MS_LOG(ERROR) << "ANI priority out of range: " << priority_value;
      return MS_INVALID_PARAM_ANI;
    }
    nnrt_device.priority = priority_value;
  }
  return MS_SUCCESS_ANI;
}

int32_t TransTaiheContext(MSLiteModelInfoANI *model_info_ptr, MSLiteContextInfoANI *context_info_ptr,
                          ::ohos::ai::mindSporeLite::Context context) {
  ParseTargetTaihe(context, context_info_ptr);
  ParseCpuTaihe(context, context_info_ptr);
  return ParseNnrtDeviceFromTaiheContext(context, context_info_ptr);
}

void ConfigureDefaultCpuContext(std::unique_ptr<mindspore_ani::MSLiteContextInfoANI> &context_native) {
  if (!context_native) {
    MS_LOG(ERROR) << "context_native is nullptr, cant set ANI";
    return;
  }
  // set default target to cpu
  for (size_t i = 0; i < context_native->target.size(); i++) {
    MS_LOG(DEBUG) << i << "before context target is:" << context_native->target[i];
  }
  // set default cpu_device
  context_native->target.push_back("cpu");
  context_native->cpu_device.thread_num = mindspore_ani::DEFAULT_THREAD_NUM;
  context_native->cpu_device.thread_affinity_mode = mindspore_ani::DEFAULT_THREAD_AFFINITY;
  context_native->cpu_device.precision_mode = "enforce_fp32";
}

void ThrowBusinessError(MSLiteErrorCodeANI code_error) {
  taihe::set_business_error(code_error, MSLiteErrorCodeInfoANI.at(code_error));
}

}  // namespace mindspore_ani
