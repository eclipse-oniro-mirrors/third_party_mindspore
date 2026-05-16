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
#include "stdexcept"
#include <cstdint>
#include <algorithm>
#include <map>
#include <vector>
#include <unistd.h>
#include "ohos.ai.mindSporeLite.proj.hpp"
#include "ohos.ai.mindSporeLite.impl.hpp"
#include "taihe/runtime.hpp"
#include "ms_utils_ani.h"

namespace {

class MSTensorImpl {
 public:
  MSTensorImpl(mindspore::MSTensor native_tensor) {
    this->native_mstensor_ = std::make_shared<mindspore::MSTensor>(native_tensor);
    MS_LOG(DEBUG) << "MSLITE MSTensorANI Instances create by native_tensor.";
  }

  ::taihe::string getName() { return this->native_mstensor_->Name().c_str(); }

  void setName(::taihe::string_view uname) {
    MS_LOG(ERROR) << "MSLITE MSTensorANI setName not implemented.";
    TH_THROW(std::runtime_error, "setName not implemented");
  }

  ::taihe::array<int32_t> getShape() {
    auto tensor_shape = this->native_mstensor_->Shape();
    std::vector<int32_t> shape_native(tensor_shape.size());
    std::transform(tensor_shape.begin(), tensor_shape.end(), shape_native.begin(),
                   [](int64_t value) { return static_cast<int32_t>(value); });
    ::taihe::array<int32_t> res_shape((::taihe::array_view<int32_t>(shape_native)));
    return res_shape;
  }

  void setShape(::taihe::array_view<int32_t> ushape) {
    std::vector<int64_t> ushape_native(ushape.size());
    std::transform(ushape.begin(), ushape.end(), ushape_native.begin(),
                   [](int32_t value) { return static_cast<int64_t>(value); });
    this->native_mstensor_->SetShape(ushape_native);
  }

  int32_t getElementNum() {
    int64_t native_num = this->native_mstensor_->ElementNum();
    return static_cast<int32_t>(native_num);
  }

  void setElementNum(int32_t uelementNum) {
    MS_LOG(ERROR) << "MSLITE MSTensorANI setElementNum not implemented.";
    TH_THROW(std::runtime_error, "setElementNum not implemented");
  }

  int32_t getDataSize() {
    size_t native_size = this->native_mstensor_->DataSize();
    return static_cast<int32_t>(native_size);
  }

  void setDataSize(int32_t udataSize) {
    MS_LOG(ERROR) << "MSLITE MSTensorANI setDataSize not implemented.";
    TH_THROW(std::runtime_error, "setDataSize not implemented");
  }

  ::ohos::ai::mindSporeLite::DataType getDtype() {
    mindspore::DataType native_type = this->native_mstensor_->DataType();
    return mindspore_ani::tensorDataTypeMapANI.at(native_type);
  }

  void setDtype(::ohos::ai::mindSporeLite::DataType udtype) {
    MS_LOG(ERROR) << "MSLITE MSTensorANI setDtype not implemented.";
    TH_THROW(std::runtime_error, "setDtype not implemented");
  }

  ::ohos::ai::mindSporeLite::Format getFormat() {
    mindspore::Format native_format = this->native_mstensor_->format();
    return mindspore_ani::tensorFormatMapANI.at(native_format);
  }

  void setFormat(::ohos::ai::mindSporeLite::Format uformat) {
    MS_LOG(ERROR) << "MSLITE MSTensorANI setFormat not implemented.";
    TH_THROW(std::runtime_error, "setFormat not implemented");
  }

  ::taihe::array<uint8_t> getData() {
    size_t byte_length = this->native_mstensor_->DataSize();
    uint8_t *tensor_data = reinterpret_cast<uint8_t *>(this->native_mstensor_->MutableData());
    std::vector<uint8_t> buffer(tensor_data, tensor_data + byte_length);
    return ::taihe::array<uint8_t>(buffer);
  }

  void setData(::taihe::array_view<uint8_t> inputArray) {
    MS_LOG(DEBUG) << "Start setData MS_SUCCESS_ANI.";
    size_t native_length = this->native_mstensor_->DataSize();
    auto tensor_native = this->native_mstensor_->MutableData();
    if (inputArray.size() != native_length) {
      MS_LOG(ERROR) << "ANI size mismatch, setData fail, set data size:" << inputArray.size()
                    << ", tensor size=" << native_length;
      ThrowBusinessError(mindspore_ani::MS_SETDATA_SIZE_ERROR_MSTENSOR);
    }
    memcpy(tensor_native, inputArray.data(), native_length);
  }

  int64_t getInner() { return reinterpret_cast<int64_t>(this); }

  std::shared_ptr<mindspore::MSTensor> native_mstensor_ = nullptr;
};

class ModelImpl {
 public:
  ModelImpl(std::shared_ptr<mindspore::Model> native_model) { this->native_model_ = native_model; }

  ::taihe::optional<double> getLearningRate() {
    auto lr = this->native_model_->GetLearningRate();
    return ::taihe::optional<double>{std::in_place, lr};
  }

  void setLearningRate(::taihe::optional_view<double> ulearningRate) {
    if (bool(ulearningRate)) {
      this->native_model_->SetLearningRate(ulearningRate.value());
      MS_LOG(DEBUG) << "SetLearningRate MS_SUCCESS_ANI.";
    }
  }

  ::taihe::optional<bool> getTrainMode() {
    auto train_mode = this->native_model_->GetTrainMode();
    return ::taihe::optional<bool>{std::in_place, train_mode};
  }

  void setTrainMode(::taihe::optional_view<bool> utrainMode) {
    if (bool(utrainMode)) {
      this->native_model_->SetTrainMode(utrainMode.value());
      MS_LOG(DEBUG) << "SetTrainMode MS_SUCCESS_ANI.";
    }
  }

  ::taihe::array<::ohos::ai::mindSporeLite::MSTensor> getInputs() {
    std::vector<mindspore::MSTensor> inputs = this->native_model_->GetInputs();
    std::vector<mindspore::MSTensor> tensor_inputs = {};
    for (size_t i = 0; i < inputs.size(); i++) {
      auto tensor = mindspore::MSTensor::CreateTensor(inputs.at(i).Name(), inputs.at(i).DataType(), {}, nullptr, 0);
      if (tensor == nullptr) {
        MS_LOG(ERROR) << "MS_LITE ANI create tensor failed.";
      }
      tensor->SetShape(inputs.at(i).Shape());
      tensor->SetFormat(inputs.at(i).format());
      tensor->SetDataType(inputs.at(i).DataType());
      tensor_inputs.push_back(*tensor);
      delete tensor;
    }
    std::vector<::ohos::ai::mindSporeLite::MSTensor> taihe_inputs = {};
    for (size_t i = 0; i < tensor_inputs.size(); i++) {
      ::ohos::ai::mindSporeLite::MSTensor taihe_mstensor =
        taihe::make_holder<MSTensorImpl, ::ohos::ai::mindSporeLite::MSTensor>(tensor_inputs[i]);
      taihe_inputs.push_back(taihe_mstensor);
    }
    ::taihe::array<::ohos::ai::mindSporeLite::MSTensor> res_inputs(
      (::taihe::array_view<::ohos::ai::mindSporeLite::MSTensor>(taihe_inputs)));

    return res_inputs;
  }

  void predict(::taihe::array_view<::ohos::ai::mindSporeLite::MSTensor> inputs,
               ::taihe::callback_view<void(::taihe::array_view<::ohos::ai::mindSporeLite::MSTensor>)> callback) {
    std::vector<::ohos::ai::mindSporeLite::MSTensor> outputs_taihe = this->predict_native(inputs);
    ::taihe::array_view<::ohos::ai::mindSporeLite::MSTensor> outputs_view(outputs_taihe);

    callback(outputs_view);
  }

  ::taihe::array<::ohos::ai::mindSporeLite::MSTensor> predictSync(
    ::taihe::array_view<::ohos::ai::mindSporeLite::MSTensor> inputs) {
    std::vector<::ohos::ai::mindSporeLite::MSTensor> outputs_taihe = this->predict_native(inputs);
    ::taihe::array<::ohos::ai::mindSporeLite::MSTensor> outputs_res(outputs_taihe);
    return outputs_res;
  }

  std::vector<::ohos::ai::mindSporeLite::MSTensor> predict_native(
    ::taihe::array_view<::ohos::ai::mindSporeLite::MSTensor> inputs) {
    // convert taihe mstensor to native tensor
    std::vector<mindspore::MSTensor> inputs_vec = {};
    for (size_t i = 0; i < inputs.size(); i++) {
      inputs_vec.push_back(*reinterpret_cast<MSTensorImpl *>(inputs[i]->getInner())->native_mstensor_);
    }
    MS_LOG(DEBUG) << "ANI predict inputs size is: " << inputs_vec.size();
    std::vector<mindspore::MSTensor> outputs;
    auto predict_ret = this->native_model_->Predict(inputs_vec, &outputs);
    MS_LOG(DEBUG) << "ANI debug info: predict outputs size:" << outputs.size();
    // convert native output tensor to taihe MSTensor
    std::vector<::ohos::ai::mindSporeLite::MSTensor> outputs_taihe = {};
    if (predict_ret != mindspore::kSuccess) {
      MS_LOG(ERROR) << "ANI native model predict fail, output is null.";
      return outputs_taihe;
    } else {
      MS_LOG(DEBUG) << "ANI native model predict success";
    }
    for (size_t i = 0; i < outputs.size(); i++) {
      ::ohos::ai::mindSporeLite::MSTensor tensor_taihe =
        taihe::make_holder<MSTensorImpl, ::ohos::ai::mindSporeLite::MSTensor>(outputs[i]);
      outputs_taihe.push_back(tensor_taihe);
    }

    return outputs_taihe;
  }

  bool resize(::taihe::array_view<::ohos::ai::mindSporeLite::MSTensor> inputs,
              ::taihe::array_view<::taihe::array<int32_t>> dims) {
    std::vector<std::vector<int64_t>> dims_native = {};
    for (size_t i = 0; i < dims.size(); i++) {
      std::vector<int64_t> dim_native(dims[i].size());
      std::transform(dims[i].begin(), dims[i].end(), dim_native.begin(),
                     [](int32_t value) { return static_cast<int64_t>(value); });
      dims_native.push_back(dim_native);
    }
    std::vector<mindspore::MSTensor> inputs_model = this->native_model_->GetInputs();
    auto ret = this->native_model_->Resize(inputs_model, dims_native);
    if (ret != mindspore::kSuccess) {
      MS_LOG(DEBUG) << "ANI native model resize fail";
      return false;
    }

    return true;
  }

  bool runStep(::taihe::array_view<::ohos::ai::mindSporeLite::MSTensor> inputs) {
    MS_LOG(DEBUG) << "ANI start run function runStep.";
    std::vector<mindspore::MSTensor> inputs_vec = {};
    auto model_inputs = this->native_model_->GetInputs();
    if (model_inputs.size() != inputs.size()) {
      MS_LOG(ERROR) << "ANI wrong input numbers, runStep fail.";
      return false;
    }
    MS_LOG(DEBUG) << "ANI cur input tensor size is :" << inputs.size();
    for (size_t i = 0; i < inputs.size(); i++) {
      auto user_tensor = reinterpret_cast<MSTensorImpl *>(inputs[i]->getInner())->native_mstensor_;
      size_t user_length = user_tensor->DataSize();
      auto user_data = user_tensor->MutableData();
      if (model_inputs[i].DataSize() != user_length) {
        MS_LOG(ERROR) << "ANI size mismatch, runstep fail, model input size:" << model_inputs[i].DataSize()
                      << ", user size=" << user_length;
        return false;
      }
      memcpy(model_inputs[i].MutableData(), user_data, user_length);
    }
    auto ret = this->native_model_->RunStep();
    if (ret != mindspore::kSuccess) {
      MS_LOG(ERROR) << "ANI model run step failed";
      return false;
    }
    MS_LOG(DEBUG) << "ANI model run step success";

    return true;
  }

  ::taihe::array<::ohos::ai::mindSporeLite::MSTensor> getWeights() {
    std::vector<mindspore::MSTensor> weights = this->native_model_->GetFeatureMaps();
    std::vector<mindspore::MSTensor> feature_maps;
    for (size_t i = 0; i < weights.size(); i++) {
      auto tensor = mindspore::MSTensor::CreateTensor(weights.at(i).Name(), weights.at(i).DataType(), {}, nullptr, 0);
      if (tensor == nullptr) {
        MS_LOG(ERROR) << "ANI create tensor failed.";
        TH_THROW(std::runtime_error, "getWeights create tensor failed.");
      }
      tensor->SetShape(weights.at(i).Shape());
      tensor->SetFormat(weights.at(i).format());
      tensor->SetDataType(weights.at(i).DataType());
      tensor->SetData(weights.at(i).MutableData(), false);
      feature_maps.push_back(*tensor);
      delete tensor;
    }
    std::vector<::ohos::ai::mindSporeLite::MSTensor> taihe_weights = {};
    for (size_t i = 0; i < feature_maps.size(); i++) {
      ::ohos::ai::mindSporeLite::MSTensor taihe_featruremap =
        taihe::make_holder<MSTensorImpl, ::ohos::ai::mindSporeLite::MSTensor>(feature_maps[i]);
      taihe_weights.push_back(taihe_featruremap);
    }
    ::taihe::array<::ohos::ai::mindSporeLite::MSTensor> res_weights(
      (::taihe::array_view<::ohos::ai::mindSporeLite::MSTensor>(taihe_weights)));

    return res_weights;
  }

  bool updateWeights(::taihe::array_view<::ohos::ai::mindSporeLite::MSTensor> weights) {
    std::vector<mindspore::MSTensor> weight_taihe = {};
    for (size_t i = 0; i < weights.size(); i++) {
      weight_taihe.push_back(*reinterpret_cast<MSTensorImpl *>(weights[i]->getInner())->native_mstensor_);
    }
    auto ret = this->native_model_->UpdateFeatureMaps(weight_taihe);
    if (ret != mindspore::kSuccess) {
      MS_LOG(ERROR) << "ANI model updateWeights failed";
      return false;
    }

    return true;
  }

  bool setupVirtualBatch(int32_t virtualBatchMultiplier, double lr, double momentum) {
    auto ret = this->native_model_->SetupVirtualBatch(static_cast<int>(virtualBatchMultiplier), lr, momentum);
    if (ret != mindspore::kSuccess) {
      MS_LOG(ERROR) << "ANI model setupVirtualBatch failed";
      return false;
    }

    return true;
  }

  bool exportModel(::taihe::string_view modelFile,
                   ::taihe::optional_view<::ohos::ai::mindSporeLite::QuantizationType> quantizationType,
                   ::taihe::optional_view<bool> exportInferenceOnly,
                   ::taihe::optional_view<::taihe::array<::taihe::string>> outputTensorName) {
    std::string model_path(modelFile);
    int32_t quantization_type_value = 0;
    if (bool(quantizationType)) {
      quantization_type_value = static_cast<int32_t>(quantizationType.value().get_value());
    }
    bool export_inference_only = true;
    if (bool(exportInferenceOnly)) {
      export_inference_only = exportInferenceOnly.value();
    }
    std::vector<std::string> output_tensor_name;
    if (bool(outputTensorName)) {
      for (size_t i = 0; i < outputTensorName.value().size(); i++) {
        output_tensor_name.push_back(std::string(outputTensorName.value()[i]));
      }
    }
    auto ret = mindspore::Serialization::ExportModel(
      *(this->native_model_.get()), static_cast<mindspore::ModelType>(mindspore::kMindIR), model_path,
      static_cast<mindspore::QuantizationType>(quantization_type_value), export_inference_only, output_tensor_name);
    if (ret != mindspore::kSuccess) {
      MS_LOG(ERROR) << "ANI export model failed";
      return false;
    }

    return true;
  }

  bool exportWeightsCollaborateWithMicro(
    ::taihe::string_view weightFile, ::taihe::optional_view<bool> isInference, ::taihe::optional_view<bool> enableFp16,
    ::taihe::optional_view<::taihe::array<::taihe::string>> changeableWeightsName) {
    std::string weight_file(weightFile);
    bool is_inference = true;
    if (bool(isInference)) {
      is_inference = isInference.value();
    }
    bool enable_fp16 = false;
    if (bool(enableFp16)) {
      enable_fp16 = enableFp16.value();
    }
    std::vector<std::string> changeable_weights_name;
    if (bool(changeableWeightsName)) {
      for (size_t i = 0; i < changeableWeightsName.value().size(); i++) {
        changeable_weights_name.push_back(std::string(changeableWeightsName.value()[i]));
      }
    }
    auto ret = mindspore::Serialization::ExportWeightsCollaborateWithMicro(
      *(this->native_model_.get()), static_cast<mindspore::ModelType>(mindspore::kMindIR), weight_file, is_inference,
      enable_fp16, changeable_weights_name);
    if (ret != mindspore::kSuccess) {
      MS_LOG(ERROR) << "ANI exportWeightsCollaborateWithMicro failed";
      return false;
    }

    return true;
  }

  int64_t getInner() { return reinterpret_cast<int64_t>(this); }

  std::shared_ptr<mindspore::Model> native_model_ = nullptr;
};

class NNRTDeviceDescriptionImpl {
 public:
  NNRTDeviceDescriptionImpl(mindspore_ani::NnrtDeviceDesc nnrt_context) {
    auto nnrt_ptr = std::make_unique<mindspore_ani::NnrtDeviceDesc>(nnrt_context);
    this->nativeNnrtDeviceDesc_ = std::move(nnrt_ptr);
  }

  ::taihe::array<uint8_t> deviceID() {
    size_t nnrt_id = this->nativeNnrtDeviceDesc_->id;
    MS_LOG(DEBUG) << "success deviceID is: " << nnrt_id;
    std::vector<uint8_t> buf;
    while (nnrt_id > 0) {
      buf.push_back(static_cast<uint8_t>(nnrt_id & 0xFF));
      nnrt_id >>= 8;
    }
    if (buf.empty()) {
      buf.push_back(0);
    }
    ::taihe::array<uint8_t> res_nnrt((::taihe::array_view<uint8_t>(buf)));
    MS_LOG(DEBUG) << "ANI success converter deviceID to bigInt, buf size is:" << buf.size();
    return res_nnrt;
  }

  ::ohos::ai::mindSporeLite::NNRTDeviceType deviceType() {
    auto device_type = this->nativeNnrtDeviceDesc_->type;
    return mindspore_ani::NNRTDeviceTypeMapANI.at(device_type);
  }

  ::taihe::string deviceName() {
    auto device_name = this->nativeNnrtDeviceDesc_->name.c_str();
    MS_LOG(DEBUG) << "ANI success deviceName is: " << device_name;

    return device_name;
  }
  std::unique_ptr<mindspore_ani::NnrtDeviceDesc> nativeNnrtDeviceDesc_ = nullptr;
};

::ohos::ai::mindSporeLite::Model loadModelFromFileSync(
  ::taihe::string_view model, ::taihe::optional_view<::ohos::ai::mindSporeLite::Context> context) {
  // init build config
  std::unique_ptr<mindspore_ani::MSLiteModelInfoANI> model_info_native =
    std::make_unique<mindspore_ani::MSLiteModelInfoANI>();
  std::unique_ptr<mindspore_ani::MSLiteContextInfoANI> context_native =
    std::make_unique<mindspore_ani::MSLiteContextInfoANI>();
  // set model info
  MS_LOG(DEBUG) << "ANI start trans taihe string:" << std::string(model);
  model_info_native->model_path = std::string(model);
  model_info_native->mode = mindspore_ani::kPath;
  // parser context
  if (bool(context)) {
    int32_t context_status =
      mindspore_ani::TransTaiheContext(model_info_native.get(), context_native.get(), context.value());
    if (context_status != mindspore_ani::MS_SUCCESS_ANI) {
      MS_LOG(ERROR) << "ANI some context parameter set failed, maybe predict fail";
    }
  } else {
    ConfigureDefaultCpuContext(context_native);
  }
  // create native model
  std::shared_ptr<mindspore::Model> native_model =
    mindspore_ani::CreateModelANI(model_info_native.get(), context_native.get());
  if (!native_model) {
    MS_LOG(ERROR) << "ANI native_model is nullptr, CreateModelANI failed!";
    ThrowBusinessError(mindspore_ani::MS_LOAD_NATIVE_ERROR_PREDICT);
  } else {
    MS_LOG(INFO) << "ANI native_model created successfully!";
  }
  // create taihe model
  ::ohos::ai::mindSporeLite::Model taihe_model =
    taihe::make_holder<ModelImpl, ::ohos::ai::mindSporeLite::Model>(native_model);
  MS_LOG(DEBUG) << "ANI loadModelFromFileSync end make holder native model";

  return taihe_model;
}

void loadModelFromFileCallback(::taihe::string_view model,
                               ::taihe::callback_view<void(::ohos::ai::mindSporeLite::weak::Model)> callback) {
  // init build config
  std::unique_ptr<mindspore_ani::MSLiteModelInfoANI> model_info_native =
    std::make_unique<mindspore_ani::MSLiteModelInfoANI>();
  std::unique_ptr<mindspore_ani::MSLiteContextInfoANI> context_native =
    std::make_unique<mindspore_ani::MSLiteContextInfoANI>();
  // set model info
  MS_LOG(DEBUG) << "ANI loadModelFromFileCallback model_path: " << std::string(model);
  model_info_native->model_path = std::string(model);
  model_info_native->mode = mindspore_ani::kPath;
  // set default context
  ConfigureDefaultCpuContext(context_native);
  // create native model
  std::shared_ptr<mindspore::Model> native_model =
    mindspore_ani::CreateModelANI(model_info_native.get(), context_native.get());
  // create taihe model
  ::ohos::ai::mindSporeLite::Model taihe_model =
    taihe::make_holder<ModelImpl, ::ohos::ai::mindSporeLite::Model>(native_model);
  MS_LOG(DEBUG) << "ANI loadModelFromFileCallback end make holder native model";
  if (!native_model) {
    MS_LOG(ERROR) << "ANI native_model is nullptr, CreateModelANI failed!";
  } else {
    MS_LOG(INFO) << "ANI native_model created successfully!";
    callback(taihe_model);
  }
}

void loadModelFromFileContextCallback(::taihe::string_view model, ::ohos::ai::mindSporeLite::Context const &context,
                                      ::taihe::callback_view<void(::ohos::ai::mindSporeLite::weak::Model)> callback) {
  // init build config
  std::unique_ptr<mindspore_ani::MSLiteModelInfoANI> model_info_native =
    std::make_unique<mindspore_ani::MSLiteModelInfoANI>();
  std::unique_ptr<mindspore_ani::MSLiteContextInfoANI> context_native =
    std::make_unique<mindspore_ani::MSLiteContextInfoANI>();
  // set model info
  MS_LOG(DEBUG) << "Start loadModelFromFileContextCallback Path is: " << std::string(model);
  model_info_native->model_path = std::string(model);
  model_info_native->mode = mindspore_ani::kPath;
  // parser context
  int32_t context_status = mindspore_ani::TransTaiheContext(model_info_native.get(), context_native.get(), context);
  if (context_status != mindspore_ani::MS_SUCCESS_ANI) {
    MS_LOG(ERROR) << "ANI some context parameter set failed, maybe predict fail";
  } else {
    MS_LOG(INFO) << "ANI set context parameter success";
  }
  // create native model
  std::shared_ptr<mindspore::Model> native_model =
    mindspore_ani::CreateModelANI(model_info_native.get(), context_native.get());
  // create taihe model
  ::ohos::ai::mindSporeLite::Model taihe_model =
    taihe::make_holder<ModelImpl, ::ohos::ai::mindSporeLite::Model>(native_model);
  MS_LOG(DEBUG) << "ANI loadModelFromFileContextCallback end make holder native model";
  if (!native_model) {
    MS_LOG(ERROR) << "ANI native_model is nullptr, CreateModelANI failed!";
  } else {
    MS_LOG(INFO) << "ANI native_model created successfully!";
    callback(taihe_model);
  }
}

::ohos::ai::mindSporeLite::Model loadModelFromBufferSync(
  ::taihe::array_view<uint8_t> model, ::taihe::optional_view<::ohos::ai::mindSporeLite::Context> context) {
  // The parameters in the make_holder function should be of the same type
  // as the parameters in the constructor of the actual implementation class.
  // init build config
  std::unique_ptr<mindspore_ani::MSLiteModelInfoANI> model_info_native =
    std::make_unique<mindspore_ani::MSLiteModelInfoANI>();
  std::unique_ptr<mindspore_ani::MSLiteContextInfoANI> context_native =
    std::make_unique<mindspore_ani::MSLiteContextInfoANI>();
  // set model info
  model_info_native->model_buffer_data = reinterpret_cast<char *>(model.data());
  model_info_native->model_buffer_total = model.size();
  MS_LOG(DEBUG) << "ANI loadModelFromBufferSync size is: " << model.size();
  model_info_native->mode = mindspore_ani::kBuffer;
  // parser context
  if (bool(context)) {
    int32_t context_status =
      mindspore_ani::TransTaiheContext(model_info_native.get(), context_native.get(), context.value());
    if (context_status != mindspore_ani::MS_SUCCESS_ANI) {
      MS_LOG(ERROR) << "Some context parameter set failed, maybe predict fail";
    }
  } else {
    ConfigureDefaultCpuContext(context_native);
  }
  // create native model
  std::shared_ptr<mindspore::Model> native_model =
    mindspore_ani::CreateModelANI(model_info_native.get(), context_native.get());
  if (!native_model) {
    MS_LOG(ERROR) << "ANI native_model is nullptr, CreateModelANI failed!";
    ThrowBusinessError(mindspore_ani::MS_LOAD_BUFFER_NATIVE_ERROR_PREDICT);
  } else {
    MS_LOG(INFO) << "ANI native_model created successfully!";
  }
  // create taihe model
  ::ohos::ai::mindSporeLite::Model taihe_model =
    taihe::make_holder<ModelImpl, ::ohos::ai::mindSporeLite::Model>(native_model);
  MS_LOG(DEBUG) << "loadModelFromBufferSync end make holder native model";

  return taihe_model;
}

void loadModelFromBufferCallback(::taihe::array_view<uint8_t> model,
                                 ::taihe::callback_view<void(::ohos::ai::mindSporeLite::weak::Model)> callback) {
  MS_LOG(ERROR) << "Start loadModelFromBufferCallback func.";
  std::unique_ptr<mindspore_ani::MSLiteModelInfoANI> model_info_native =
    std::make_unique<mindspore_ani::MSLiteModelInfoANI>();
  std::unique_ptr<mindspore_ani::MSLiteContextInfoANI> context_native =
    std::make_unique<mindspore_ani::MSLiteContextInfoANI>();
  // set model info
  model_info_native->model_buffer_data = reinterpret_cast<char *>(model.data());
  model_info_native->model_buffer_total = model.size();
  MS_LOG(DEBUG) << "ANI loadModelFromBufferCallback size is: " << model.size();
  // set default context
  ConfigureDefaultCpuContext(context_native);
  model_info_native->mode = mindspore_ani::kBuffer;
  // create native model
  std::shared_ptr<mindspore::Model> native_model =
    mindspore_ani::CreateModelANI(model_info_native.get(), context_native.get());
  // create taihe model
  ::ohos::ai::mindSporeLite::Model taihe_model =
    taihe::make_holder<ModelImpl, ::ohos::ai::mindSporeLite::Model>(native_model);
  if (!native_model) {
    MS_LOG(ERROR) << "ANI native_model is nullptr, CreateModelANI failed!";
  } else {
    MS_LOG(INFO) << "ANI native_model created successfully!";
    callback(taihe_model);
  }
}

void loadModelFromBufferContextCallback(::taihe::array_view<uint8_t> model,
                                        ::ohos::ai::mindSporeLite::Context const &context,
                                        ::taihe::callback_view<void(::ohos::ai::mindSporeLite::weak::Model)> callback) {
  std::unique_ptr<mindspore_ani::MSLiteModelInfoANI> model_info_native =
    std::make_unique<mindspore_ani::MSLiteModelInfoANI>();
  std::unique_ptr<mindspore_ani::MSLiteContextInfoANI> context_native =
    std::make_unique<mindspore_ani::MSLiteContextInfoANI>();
  // set model info
  model_info_native->model_buffer_data = reinterpret_cast<char *>(model.data());
  model_info_native->model_buffer_total = model.size();
  MS_LOG(DEBUG) << "ANI loadModelFromBufferCallback size is: " << model.size();
  model_info_native->mode = mindspore_ani::kBuffer;
  // parser context
  int32_t context_status = mindspore_ani::TransTaiheContext(model_info_native.get(), context_native.get(), context);
  if (context_status != mindspore_ani::MS_SUCCESS_ANI) {
    MS_LOG(ERROR) << "ANIsome context parameter set failed, maybe predict fail";
  }
  // create native model
  std::shared_ptr<mindspore::Model> native_model =
    mindspore_ani::CreateModelANI(model_info_native.get(), context_native.get());
  // create taihe model
  ::ohos::ai::mindSporeLite::Model taihe_model =
    taihe::make_holder<ModelImpl, ::ohos::ai::mindSporeLite::Model>(native_model);
  if (!native_model) {
    MS_LOG(ERROR) << "ANI native_model is nullptr, CreateModelANI failed!";
  } else {
    MS_LOG(INFO) << "ANI native_model created successfully!";
    callback(taihe_model);
  }
}

::ohos::ai::mindSporeLite::Model loadModelFromFdSync(
  int32_t model, ::taihe::optional_view<::ohos::ai::mindSporeLite::Context> context) {
  // The parameters in the make_holder function should be of the same type
  // as the parameters in the constructor of the actual implementation class.
  // init build config
  std::unique_ptr<mindspore_ani::MSLiteModelInfoANI> model_info_native =
    std::make_unique<mindspore_ani::MSLiteModelInfoANI>();
  std::unique_ptr<mindspore_ani::MSLiteContextInfoANI> context_native =
    std::make_unique<mindspore_ani::MSLiteContextInfoANI>();
  //  set model info
  int32_t fd = model;
  int size = lseek(fd, 0, SEEK_END);
  (void)lseek(fd, 0, SEEK_SET);
  auto mmap_buffers = mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0);
  if (mmap_buffers == NULL) {
    MS_LOG(ERROR) << "mmap_buffers is NULL.";
  }
  model_info_native->model_fd = fd;
  model_info_native->model_buffer_data = static_cast<char *>(mmap_buffers);
  model_info_native->model_buffer_total = size;
  model_info_native->mode = mindspore_ani::kFD;
  MS_LOG(DEBUG) << "ANI mmap_buffers fd is:" << fd;
  // parser context
  if (bool(context)) {
    int32_t context_status =
      mindspore_ani::TransTaiheContext(model_info_native.get(), context_native.get(), context.value());
    if (context_status != mindspore_ani::MS_SUCCESS_ANI) {
      MS_LOG(ERROR) << "ANI some context parameter set failed, maybe predict fail";
    }
  } else {
    ConfigureDefaultCpuContext(context_native);
  }
  // create native model
  std::shared_ptr<mindspore::Model> native_model =
    mindspore_ani::CreateModelANI(model_info_native.get(), context_native.get());
  // create taihe model
  if (!native_model) {
    MS_LOG(ERROR) << "ANI native_model is nullptr, CreateModelANI failed!";
    ThrowBusinessError(mindspore_ani::MS_LOAD_FD_NATIVE_ERROR_PREDICT);
  } else {
    MS_LOG(INFO) << "ANI native_model created successfully!";
  }
  ::ohos::ai::mindSporeLite::Model taihe_model =
    taihe::make_holder<ModelImpl, ::ohos::ai::mindSporeLite::Model>(native_model);
  MS_LOG(DEBUG) << "ANI loadModelFromFdSync end make_holder native_model.";

  return taihe_model;
}

void loadModelFromFdCallback(int32_t model,
                             ::taihe::callback_view<void(::ohos::ai::mindSporeLite::weak::Model)> callback) {
  // The parameters in the make_holder function should be of the same type
  // as the parameters in the constructor of the actual implementation class.
  // init build config
  std::unique_ptr<mindspore_ani::MSLiteModelInfoANI> model_info_native =
    std::make_unique<mindspore_ani::MSLiteModelInfoANI>();
  std::unique_ptr<mindspore_ani::MSLiteContextInfoANI> context_native =
    std::make_unique<mindspore_ani::MSLiteContextInfoANI>();
  ConfigureDefaultCpuContext(context_native);
  //  set model info
  int32_t fd = model;
  int size = lseek(fd, 0, SEEK_END);
  (void)lseek(fd, 0, SEEK_SET);
  auto mmap_buffers = mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0);
  if (mmap_buffers == NULL) {
    MS_LOG(ERROR) << "mmap_buffers is NULL.";
  }
  model_info_native->model_fd = fd;
  MS_LOG(DEBUG) << "ANI mmap_buffers fd is:" << fd;
  model_info_native->model_buffer_data = static_cast<char *>(mmap_buffers);
  model_info_native->model_buffer_total = size;
  model_info_native->mode = mindspore_ani::kFD;
  // create native model
  std::shared_ptr<mindspore::Model> native_model =
    mindspore_ani::CreateModelANI(model_info_native.get(), context_native.get());
  // create taihe model
  ::ohos::ai::mindSporeLite::Model taihe_model =
    taihe::make_holder<ModelImpl, ::ohos::ai::mindSporeLite::Model>(native_model);
  MS_LOG(DEBUG) << "ANI loadModelFromFdCallback make_holder native_model .";
  if (!native_model) {
    MS_LOG(ERROR) << "ANI native_model is nullptr, CreateModelANI failed!";
  } else {
    MS_LOG(INFO) << "ANI native_model created successfully!";
    callback(taihe_model);
  }
}

void loadModelFromFdContextCallback(int32_t model, ::ohos::ai::mindSporeLite::Context const &context,
                                    ::taihe::callback_view<void(::ohos::ai::mindSporeLite::weak::Model)> callback) {
  // The parameters in the make_holder function should be of the same type
  // as the parameters in the constructor of the actual implementation class.
  // init build config
  std::unique_ptr<mindspore_ani::MSLiteModelInfoANI> model_info_native =
    std::make_unique<mindspore_ani::MSLiteModelInfoANI>();
  std::unique_ptr<mindspore_ani::MSLiteContextInfoANI> context_native =
    std::make_unique<mindspore_ani::MSLiteContextInfoANI>();
  //  set model info
  int32_t fd = model;
  MS_LOG(DEBUG) << "ANI mmap_buffers fd is:" << fd;
  int size = lseek(fd, 0, SEEK_END);
  (void)lseek(fd, 0, SEEK_SET);
  auto mmap_buffers = mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0);
  if (mmap_buffers == NULL) {
    MS_LOG(ERROR) << "ANI mmap_buffers is NULL.";
  }
  model_info_native->model_fd = fd;
  model_info_native->model_buffer_data = static_cast<char *>(mmap_buffers);
  model_info_native->model_buffer_total = size;
  model_info_native->mode = mindspore_ani::kFD;
  // parser context
  int32_t context_status = mindspore_ani::TransTaiheContext(model_info_native.get(), context_native.get(), context);
  if (context_status != mindspore_ani::MS_SUCCESS_ANI) {
    MS_LOG(ERROR) << "ANI some context parameter set failed, turn for default context predict";
    ConfigureDefaultCpuContext(context_native);
  }
  // create native model
  std::shared_ptr<mindspore::Model> native_model =
    mindspore_ani::CreateModelANI(model_info_native.get(), context_native.get());
  // create taihe model
  ::ohos::ai::mindSporeLite::Model taihe_model =
    taihe::make_holder<ModelImpl, ::ohos::ai::mindSporeLite::Model>(native_model);
  MS_LOG(DEBUG) << "ANI loadModelFromFdContextCallback end make_holder native_model.";
  if (!native_model) {
    MS_LOG(ERROR) << "ANI native_model is nullptr, CreateModelANI failed!";
  } else {
    MS_LOG(INFO) << "ANI native_model created successfully!";
    callback(taihe_model);
  }
}

::ohos::ai::mindSporeLite::Model loadTrainModelFromFile(
  ::taihe::string_view model, ::taihe::optional_view<::ohos::ai::mindSporeLite::TrainCfg> trainCfg,
  ::taihe::optional_view<::ohos::ai::mindSporeLite::Context> context) {
  // The parameters in the make_holder function should be of the same type
  // as the parameters in the constructor of the actual implementation class.
  // init build config
  std::unique_ptr<mindspore_ani::MSLiteModelInfoANI> model_info_native =
    std::make_unique<mindspore_ani::MSLiteModelInfoANI>();
  std::unique_ptr<mindspore_ani::MSLiteContextInfoANI> context_native =
    std::make_unique<mindspore_ani::MSLiteContextInfoANI>();
  // set model info
  model_info_native->model_path = std::string(model);
  MS_LOG(DEBUG) << "ANI model path is:" << std::string(model);
  model_info_native->mode = mindspore_ani::kPath;
  model_info_native->train_model = true;
  // parser train cfg
  if (bool(trainCfg)) {
    ::ohos::ai::mindSporeLite::TrainCfg train_cfg = trainCfg.value();
    if (bool(train_cfg.lossName)) {
      std::vector<std::string> str_lossname;
      for (size_t i = 0; i < train_cfg.lossName.value().size(); i++) {
        str_lossname.push_back(std::string(train_cfg.lossName.value()[i]));
      }
      context_native->train_cfg.loss_names.assign(str_lossname.begin(), str_lossname.end());
    }
    int32_t opt_value = 0;
    if (bool(train_cfg.optimizationLevel)) {
      opt_value = static_cast<int32_t>(train_cfg.optimizationLevel.value());
    }
    context_native->train_cfg.optimization_level = opt_value;
  }
  // parser context
  if (bool(context)) {
    int32_t context_status =
      mindspore_ani::TransTaiheContext(model_info_native.get(), context_native.get(), context.value());
    if (context_status != mindspore_ani::MS_SUCCESS_ANI) {
      MS_LOG(ERROR) << "ANI some context parameter set failed, maybe predict fail.";
    }
  } else {
    MS_LOG(DEBUG) << "ANI start set default ConfigureDefaultCpuContext.";
    ConfigureDefaultCpuContext(context_native);
  }
  // create native model
  std::shared_ptr<mindspore::Model> native_model =
    mindspore_ani::CreateTrainModelANI(model_info_native.get(), context_native.get());
  if (!native_model) {
    MS_LOG(ERROR) << "ANI native_model is nullptr, CreateModelANI failed!";
    ThrowBusinessError(mindspore_ani::MS_LOAD_PATH_NATIVE_ERROR_TRAIN);
  } else {
    MS_LOG(INFO) << "ANI native_model created successfully!";
  }
  // create taihe model
  ::ohos::ai::mindSporeLite::Model taihe_model =
    taihe::make_holder<ModelImpl, ::ohos::ai::mindSporeLite::Model>(native_model);

  return taihe_model;
}

::ohos::ai::mindSporeLite::Model loadTrainModelFromBuffer(
  ::taihe::array_view<uint8_t> model, ::taihe::optional_view<::ohos::ai::mindSporeLite::TrainCfg> trainCfg,
  ::taihe::optional_view<::ohos::ai::mindSporeLite::Context> context) {
  // The parameters in the make_holder function should be of the same type
  // as the parameters in the constructor of the actual implementation class.
  std::unique_ptr<mindspore_ani::MSLiteModelInfoANI> model_info_native =
    std::make_unique<mindspore_ani::MSLiteModelInfoANI>();
  std::unique_ptr<mindspore_ani::MSLiteContextInfoANI> context_native =
    std::make_unique<mindspore_ani::MSLiteContextInfoANI>();
  // set model info
  model_info_native->model_buffer_data = reinterpret_cast<char *>(model.data());
  model_info_native->model_buffer_total = model.size();
  model_info_native->mode = mindspore_ani::kBuffer;
  model_info_native->train_model = true;
  // parser train cfg
  if (bool(trainCfg)) {
    ::ohos::ai::mindSporeLite::TrainCfg train_cfg = trainCfg.value();
    if (bool(train_cfg.lossName)) {
      std::vector<std::string> str_lossname;
      for (size_t i = 0; i < train_cfg.lossName.value().size(); i++) {
        str_lossname.push_back(std::string(train_cfg.lossName.value()[i]));
      }
      context_native->train_cfg.loss_names.assign(str_lossname.begin(), str_lossname.end());
    }
    int32_t opt_value = 0;
    if (bool(train_cfg.optimizationLevel)) {
      opt_value = static_cast<int32_t>(train_cfg.optimizationLevel.value());
    }
    context_native->train_cfg.optimization_level = opt_value;
  }
  // parser context
  if (bool(context)) {
    int32_t context_status =
      mindspore_ani::TransTaiheContext(model_info_native.get(), context_native.get(), context.value());
    if (context_status != mindspore_ani::MS_SUCCESS_ANI) {
      MS_LOG(ERROR) << "Some context parameter set failed, maybe predict fail";
    }
  } else {
    ConfigureDefaultCpuContext(context_native);
  }
  // create native model
  std::shared_ptr<mindspore::Model> native_model =
    mindspore_ani::CreateTrainModelANI(model_info_native.get(), context_native.get());
  if (!native_model) {
    MS_LOG(ERROR) << "ANI native_model is nullptr, CreateModelANI failed!";
    ThrowBusinessError(mindspore_ani::MS_LOAD_BUFFER_NATIVE_ERROR_TRAIN);
  } else {
    MS_LOG(INFO) << "ANI native_model created successfully!";
  }
  // create taihe model
  ::ohos::ai::mindSporeLite::Model taihe_model =
    taihe::make_holder<ModelImpl, ::ohos::ai::mindSporeLite::Model>(native_model);

  return taihe_model;
}

::ohos::ai::mindSporeLite::Model loadTrainModelFromFd(
  int32_t model, ::taihe::optional_view<::ohos::ai::mindSporeLite::TrainCfg> trainCfg,
  ::taihe::optional_view<::ohos::ai::mindSporeLite::Context> context) {
  std::unique_ptr<mindspore_ani::MSLiteModelInfoANI> model_info_native =
    std::make_unique<mindspore_ani::MSLiteModelInfoANI>();
  std::unique_ptr<mindspore_ani::MSLiteContextInfoANI> context_native =
    std::make_unique<mindspore_ani::MSLiteContextInfoANI>();
  // set model info
  int32_t fd = model;
  int size = lseek(fd, 0, SEEK_END);
  (void)lseek(fd, 0, SEEK_SET);
  auto mmap_buffers = mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0);
  if (mmap_buffers == NULL) {
    MS_LOG(ERROR) << "ANI mmap_buffers is NULL.";
  }
  model_info_native->model_fd = fd;
  model_info_native->model_buffer_data = static_cast<char *>(mmap_buffers);
  model_info_native->model_buffer_total = size;
  model_info_native->mode = mindspore_ani::kFD;
  model_info_native->train_model = true;
  // parser train cfg
  if (bool(trainCfg)) {
    ::ohos::ai::mindSporeLite::TrainCfg train_cfg = trainCfg.value();
    if (bool(train_cfg.lossName)) {
      std::vector<std::string> str_lossname;
      for (size_t i = 0; i < train_cfg.lossName.value().size(); i++) {
        str_lossname.push_back(std::string(train_cfg.lossName.value()[i]));
      }
      context_native->train_cfg.loss_names.assign(str_lossname.begin(), str_lossname.end());
    }
    int32_t opt_value = 0;
    if (bool(train_cfg.optimizationLevel)) {
      opt_value = static_cast<int32_t>(train_cfg.optimizationLevel.value());
    }
    context_native->train_cfg.optimization_level = opt_value;
  }
  // parser context
  if (bool(context)) {
    int32_t context_status =
      mindspore_ani::TransTaiheContext(model_info_native.get(), context_native.get(), context.value());
    if (context_status != mindspore_ani::MS_SUCCESS_ANI) {
      MS_LOG(ERROR) << "ANI some context parameter set failed, maybe predict fail";
    }
  } else {
    ConfigureDefaultCpuContext(context_native);
  }
  // create native model
  std::shared_ptr<mindspore::Model> native_model =
    mindspore_ani::CreateTrainModelANI(model_info_native.get(), context_native.get());
  if (!native_model) {
    MS_LOG(ERROR) << "ANI native_model is nullptr, CreateModelANI failed!";
    ThrowBusinessError(mindspore_ani::MS_LOAD_FD_ERROR_TRAIN);
  } else {
    MS_LOG(INFO) << "ANI native_model created successfully!";
  }
  // create taihe model
  ::ohos::ai::mindSporeLite::Model taihe_model =
    taihe::make_holder<ModelImpl, ::ohos::ai::mindSporeLite::Model>(native_model);

  return taihe_model;
}

::ohos::ai::mindSporeLite::Context ConstructContext(
  ::taihe::optional_view<::taihe::array<::taihe::string>> target,
  ::taihe::optional_view<::ohos::ai::mindSporeLite::CpuDevice> cpu,
  ::taihe::optional_view<::ohos::ai::mindSporeLite::NNRTDevice> nnrt) {
  MS_LOG(ERROR) << "MSLITE MSTensorANI ConstructContext not implemented.";
  TH_THROW(std::runtime_error, "ConstructContext not implemented");
}

::ohos::ai::mindSporeLite::CpuDevice ConstructCpuDevice(
  ::taihe::optional_view<int32_t> threadNum,
  ::taihe::optional_view<::ohos::ai::mindSporeLite::ThreadAffinityMode> threadAffinityMode,
  ::taihe::optional_view<::taihe::array<int32_t>> threadAffinityCoreList,
  ::taihe::optional_view<::taihe::string> precisionMode) {
  MS_LOG(ERROR) << "MSLITE MSTensorANI ConstructCpuDevice not implemented.";
  TH_THROW(std::runtime_error, "ConstructCpuDevice not implemented");
}

::ohos::ai::mindSporeLite::Extension ConstructExtension(::taihe::string_view name, ::taihe::array_view<uint8_t> value) {
  MS_LOG(ERROR) << "MSLITE MSTensorANI ConstructExtension not implemented.";
  TH_THROW(std::runtime_error, "ConstructExtension not implemented");
}

::ohos::ai::mindSporeLite::TrainCfg ConstructTrainCfg(
  ::taihe::optional_view<::taihe::array<::taihe::string>> lossName,
  ::taihe::optional_view<::ohos::ai::mindSporeLite::OptimizationLevel> optimizationLevel) {
  MS_LOG(ERROR) << "MSLITE MSTensorANI ConstructTrainCfg not implemented.";
  TH_THROW(std::runtime_error, "ConstructTrainCfg not implemented");
}

::taihe::array<::ohos::ai::mindSporeLite::NNRTDeviceDescription> getAllNNRTDeviceDescriptions() {
  size_t num;
  NNRTDeviceDesc *devices = OH_AI_GetAllNNRTDeviceDescs(&num);
  MS_LOG(DEBUG) << "ANI all nnrt devices size: " << num;
  if (devices == nullptr) {
    MS_LOG(ERROR) << "Get all nnrt devices error, may nnrt is not supported.";
    OH_AI_DestroyAllNNRTDeviceDescs(&devices);
  }
  std::vector<::ohos::ai::mindSporeLite::NNRTDeviceDescription> taihe_nnrts = {};
  for (size_t i = 0; i < num; i++) {
    mindspore_ani::NnrtDeviceDesc nnrt_device;
    NNRTDeviceDesc *nnrt_device_desc = OH_AI_GetElementOfNNRTDeviceDescs(devices, i);
    nnrt_device.name.assign(OH_AI_GetNameFromNNRTDeviceDesc(nnrt_device_desc));
    size_t id = OH_AI_GetDeviceIdFromNNRTDeviceDesc(nnrt_device_desc);
    nnrt_device.id = id;
    nnrt_device.type =
      static_cast<mindspore_ani::ContextNnrtDeviceType>(OH_AI_GetTypeFromNNRTDeviceDesc(nnrt_device_desc));
    ::ohos::ai::mindSporeLite::NNRTDeviceDescription taihe_nnrt =
      taihe::make_holder<NNRTDeviceDescriptionImpl, ::ohos::ai::mindSporeLite::NNRTDeviceDescription>(nnrt_device);
    taihe_nnrts.push_back(taihe_nnrt);
    MS_LOG(DEBUG) << "ANI find nnrt device No." << i << "device id is:" << id;
  }
  OH_AI_DestroyAllNNRTDeviceDescs(&devices);
  ::taihe::array<::ohos::ai::mindSporeLite::NNRTDeviceDescription> taihe_nnrts_res(
    (::taihe::array_view<::ohos::ai::mindSporeLite::NNRTDeviceDescription>(taihe_nnrts)));

  return taihe_nnrts_res;
}

}  // namespace

// Since these macros are auto-generate, lint will cause false positive.
// NOLINTBEGIN
TH_EXPORT_CPP_API_loadModelFromFileSync(loadModelFromFileSync);
TH_EXPORT_CPP_API_loadModelFromFileCallback(loadModelFromFileCallback);
TH_EXPORT_CPP_API_loadModelFromFileContextCallback(loadModelFromFileContextCallback);
TH_EXPORT_CPP_API_loadModelFromBufferSync(loadModelFromBufferSync);
TH_EXPORT_CPP_API_loadModelFromBufferCallback(loadModelFromBufferCallback);
TH_EXPORT_CPP_API_loadModelFromBufferContextCallback(loadModelFromBufferContextCallback);
TH_EXPORT_CPP_API_loadModelFromFdSync(loadModelFromFdSync);
TH_EXPORT_CPP_API_loadModelFromFdCallback(loadModelFromFdCallback);
TH_EXPORT_CPP_API_loadModelFromFdContextCallback(loadModelFromFdContextCallback);
TH_EXPORT_CPP_API_loadTrainModelFromFile(loadTrainModelFromFile);
TH_EXPORT_CPP_API_loadTrainModelFromBuffer(loadTrainModelFromBuffer);
TH_EXPORT_CPP_API_loadTrainModelFromFd(loadTrainModelFromFd);
TH_EXPORT_CPP_API_ConstructContext(ConstructContext);
TH_EXPORT_CPP_API_ConstructCpuDevice(ConstructCpuDevice);
TH_EXPORT_CPP_API_ConstructExtension(ConstructExtension);
TH_EXPORT_CPP_API_ConstructTrainCfg(ConstructTrainCfg);
TH_EXPORT_CPP_API_getAllNNRTDeviceDescriptions(getAllNNRTDeviceDescriptions);
// NOLINTEND
