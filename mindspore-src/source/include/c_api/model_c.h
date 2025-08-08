/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_INCLUDE_C_API_MODEL_C_H
#define MINDSPORE_INCLUDE_C_API_MODEL_C_H

#include "include/c_api/tensor_c.h"
#include "include/c_api/context_c.h"
#include "include/c_api/status_c.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void *OH_AI_ModelHandle;

typedef void *OH_AI_TrainCfgHandle;

typedef struct OH_AI_TensorHandleArray {
  size_t handle_num;
  OH_AI_TensorHandle *handle_list;
} OH_AI_TensorHandleArray;

#define OH_AI_MAX_SHAPE_NUM 32
typedef struct OH_AI_ShapeInfo {
  size_t shape_num;
  int64_t shape[OH_AI_MAX_SHAPE_NUM];
} OH_AI_ShapeInfo;

typedef struct OH_AI_CallBackParam {
  char *node_name;
  char *node_type;
} OH_AI_CallBackParam;

typedef bool (*OH_AI_KernelCallBack)(const OH_AI_TensorHandleArray inputs, const OH_AI_TensorHandleArray outputs,
                                  const OH_AI_CallBackParam kernel_Info);

/// \brief Create a model object.
///
/// \return Model object handle.
OH_AI_API OH_AI_ModelHandle OH_AI_ModelCreate();

/// \brief Destroy the model object.
///
/// \param[in] model Model object handle address.
OH_AI_API void OH_AI_ModelDestroy(OH_AI_ModelHandle *model);

/// \brief Set workspace for the model object. Only valid for Iot.
///
/// \param[in] model Model object handle.
/// \param[in] workspace Define the workspace address.
/// \param[in] workspace_size Define the workspace size.
OH_AI_API void OH_AI_ModelSetWorkspace(OH_AI_ModelHandle model, void *workspace, size_t workspace_size);

/// \brief Calculate the workspace size required for model inference. Only valid for Iot.
///
/// \param[in] model Model object handle.
OH_AI_API size_t OH_AI_ModelCalcWorkspaceSize(OH_AI_ModelHandle model);

/// \brief Build the model from model file buffer so that it can run on a device.
///
/// \param[in] model Model object handle.
/// \param[in] model_data Define the buffer read from a model file.
/// \param[in] data_size Define bytes number of model file buffer.
/// \param[in] model_type Define The type of model file.
/// \param[in] model_context Define the context used to store options during execution.
///
/// \return OH_AI_Status.
OH_AI_API OH_AI_Status OH_AI_ModelBuild(OH_AI_ModelHandle model, const void *model_data, size_t data_size, OH_AI_ModelType model_type,
                                        const OH_AI_ContextHandle model_context);

/// \brief Load and build the model from model path so that it can run on a device.
///
/// \param[in] model Model object handle.
/// \param[in] model_path Define the model file path.
/// \param[in] model_type Define The type of model file.
/// \param[in] model_context Define the context used to store options during execution.
///
/// \return OH_AI_Status.
OH_AI_API OH_AI_Status OH_AI_ModelBuildFromFile(OH_AI_ModelHandle model, const char *model_path, OH_AI_ModelType model_type,
                                                const OH_AI_ContextHandle model_context);

/// \brief Resize the shapes of inputs.
///
/// \param[in] model Model object handle.
/// \param[in] inputs The array that includes all input tensor handles.
/// \param[in] shape_infos Defines the new shapes of inputs, should be consistent with inputs.
/// \param[in] shape_info_num The num of shape_infos.
///
/// \return OH_AI_Status.
OH_AI_API OH_AI_Status OH_AI_ModelResize(OH_AI_ModelHandle model, const OH_AI_TensorHandleArray inputs, OH_AI_ShapeInfo *shape_infos,
                              size_t shape_info_num);

/// \brief Inference model.
///
/// \param[in] model Model object handle.
/// \param[in] inputs The array that includes all input tensor handles.
/// \param[out] outputs The array that includes all output tensor handles.
/// \param[in] before CallBack before predict.
/// \param[in] after CallBack after predict.
///
/// \return OH_AI_Status.
OH_AI_API OH_AI_Status OH_AI_ModelPredict(OH_AI_ModelHandle model, const OH_AI_TensorHandleArray inputs, OH_AI_TensorHandleArray *outputs,
                                          const OH_AI_KernelCallBack before, const OH_AI_KernelCallBack after);

/// \brief Run model by step. Only valid for Iot.
///
/// \param[in] model Model object handle.
/// \param[in] before CallBack before RunStep.
/// \param[in] after CallBack after RunStep.
///
/// \return OH_AI_Status.
OH_AI_API OH_AI_Status OH_AI_ModelRunStep(OH_AI_ModelHandle model, const OH_AI_KernelCallBack before, const OH_AI_KernelCallBack after);

/// \brief Set the model running mode. Only valid for Iot.
///
/// \param[in] model Model object handle.
/// \param[in] train True means model runs in Train Mode, otherwise Eval Mode.
///
/// \return Status of operation.
OH_AI_API OH_AI_Status OH_AI_ModelSetTrainMode(const OH_AI_ModelHandle model, bool train);

/// \brief Export the weights of model to the binary file. Only valid for Iot.
///
/// \param[in] model Model object handle.
/// \param[in] export_path Define the export weight file path.
///
/// \return Status of operation.
OH_AI_API OH_AI_Status OH_AI_ModelExportWeight(const OH_AI_ModelHandle model, const char *export_path);

/// \brief Obtain all input tensor handles of the model.
///
/// \param[in] model Model object handle.
///
/// \return The array that includes all input tensor handles.
OH_AI_API OH_AI_TensorHandleArray OH_AI_ModelGetInputs(const OH_AI_ModelHandle model);

/// \brief Obtain all output tensor handles of the model.
///
/// \param[in] model Model object handle.
///
/// \return The array that includes all output tensor handles.
OH_AI_API OH_AI_TensorHandleArray OH_AI_ModelGetOutputs(const OH_AI_ModelHandle model);

/// \brief Obtain the input tensor handle of the model by name.
///
/// \param[in] model Model object handle.
/// \param[in] tensor_name The name of tensor.
///
/// \return The input tensor handle with the given name, if the name is not found, an NULL is returned.
OH_AI_API OH_AI_TensorHandle OH_AI_ModelGetInputByTensorName(const OH_AI_ModelHandle model, const char *tensor_name);

/// \brief Obtain the output tensor handle of the model by name.
///
/// \param[in] model Model object handle.
/// \param[in] tensor_name The name of tensor.
///
/// \return The output tensor handle with the given name, if the name is not found, an NULL is returned.
OH_AI_API OH_AI_TensorHandle OH_AI_ModelGetOutputByTensorName(const OH_AI_ModelHandle model, const char *tensor_name);

/// \brief Create a TrainCfg object. Only valid for Lite Train.
///
/// \return TrainCfg object handle.
OH_AI_API OH_AI_TrainCfgHandle OH_AI_TrainCfgCreate();

/// \brief Destroy the train_cfg object. Only valid for Lite Train.
///
/// \param[in] train_cfg TrainCfg object handle.
OH_AI_API void OH_AI_TrainCfgDestroy(OH_AI_TrainCfgHandle *train_cfg);

/// \brief Obtains part of the name that identify a loss kernel. Only valid for Lite Train.
///
/// \param[in] train_cfg TrainCfg object handle.
/// \param[in] num The num of loss_name.
///
/// \return loss_name.
OH_AI_API char **OH_AI_TrainCfgGetLossName(OH_AI_TrainCfgHandle train_cfg, size_t *num);

/// \brief Set part of the name that identify a loss kernel. Only valid for Lite Train.
///
/// \param[in] train_cfg TrainCfg object handle.
/// \param[in] loss_name define part of the name that identify a loss kernel.
/// \param[in] num The num of loss_name.
OH_AI_API void OH_AI_TrainCfgSetLossName(OH_AI_TrainCfgHandle train_cfg, const char **loss_name, size_t num);

/// \brief Obtains optimization level of the train_cfg. Only valid for Lite Train.
///
/// \param[in] train_cfg TrainCfg object handle.
///
/// \return OH_AI_OptimizationLevel.
OH_AI_API OH_AI_OptimizationLevel OH_AI_TrainCfgGetOptimizationLevel(OH_AI_TrainCfgHandle train_cfg);

/// \brief Set optimization level of the train_cfg. Only valid for Lite Train.
///
/// \param[in] train_cfg TrainCfg object handle.
/// \param[in] level The optimization level of train_cfg.
OH_AI_API void OH_AI_TrainCfgSetOptimizationLevel(OH_AI_TrainCfgHandle train_cfg, OH_AI_OptimizationLevel level);

/// \brief Build the train model from model buffer so that it can run on a device. Only valid for Lite Train.
///
/// \param[in] model Model object handle.
/// \param[in] model_data Define the buffer read from a model file.
/// \param[in] data_size Define bytes number of model file buffer.
/// \param[in] model_type Define The type of model file.
/// \param[in] model_context Define the context used to store options during execution.
/// \param[in] train_cfg Define the config used by training.
///
/// \return OH_AI_Status.
OH_AI_API OH_AI_Status OH_AI_TrainModelBuild(OH_AI_ModelHandle model, const void *model_data, size_t data_size, OH_AI_ModelType model_type,
                                  const OH_AI_ContextHandle model_context, const OH_AI_TrainCfgHandle train_cfg);

/// \brief Build the train model from model file buffer so that it can run on a device. Only valid for Lite Train.
///
/// \param[in] model Model object handle.
/// \param[in] model_path Define the model path.
/// \param[in] model_type Define The type of model file.
/// \param[in] model_context Define the context used to store options during execution.
/// \param[in] train_cfg Define the config used by training.
///
/// \return OH_AI_Status.
OH_AI_API OH_AI_Status OH_AI_TrainModelBuildFromFile(OH_AI_ModelHandle model, const char *model_path, OH_AI_ModelType model_type,
                                          const OH_AI_ContextHandle model_context, const OH_AI_TrainCfgHandle train_cfg);

/// \brief Train model by step. Only valid for Lite Train.
///
/// \param[in] model Model object handle.
/// \param[in] before CallBack before predict.
/// \param[in] after CallBack after predict.
///
/// \return OH_AI_Status.
OH_AI_API OH_AI_Status OH_AI_RunStep(OH_AI_ModelHandle model, const OH_AI_KernelCallBack before, const OH_AI_KernelCallBack after);

/// \brief Sets the Learning Rate of the training. Only valid for Lite Train.
///
/// \param[in] learning_rate to set.
///
/// \return OH_AI_Status of operation.
OH_AI_API OH_AI_Status OH_AI_ModelSetLearningRate(OH_AI_ModelHandle model, float learning_rate);

/// \brief Obtains the Learning Rate of the optimizer. Only valid for Lite Train.
///
/// \return Learning rate. 0.0 if no optimizer was found.
OH_AI_API float OH_AI_ModelGetLearningRate(OH_AI_ModelHandle model);

/// \brief Obtains all weights tensors of the model. Only valid for Lite Train.
///
/// \param[in] model Model object handle.
///
/// \return The vector that includes all gradient tensors.
OH_AI_API OH_AI_TensorHandleArray OH_AI_ModelGetWeights(OH_AI_ModelHandle model);

/// \brief update weights tensors of the model. Only valid for Lite Train.
///
/// \param[in] new_weights A vector new weights.
///
/// \return OH_AI_Status
OH_AI_API OH_AI_Status OH_AI_ModelUpdateWeights(OH_AI_ModelHandle model, const OH_AI_TensorHandleArray new_weights);

/// \brief Get the model running mode.
///
/// \param[in] model Model object handle.
///
/// \return Is Train Mode or not.
OH_AI_API bool OH_AI_ModelGetTrainMode(OH_AI_ModelHandle model);

/// \brief Set the model running mode. Only valid for Lite Train.
///
/// \param[in] model Model object handle.
/// \param[in] train True means model runs in Train Mode, otherwise Eval Mode.
///
/// \return OH_AI_Status.
OH_AI_API OH_AI_Status OH_AI_ModelSetTrainMode(OH_AI_ModelHandle model, bool train);

/// \brief Setup training with virtual batches. Only valid for Lite Train.
///
/// \param[in] model Model object handle.
/// \param[in] virtual_batch_multiplier - virtual batch multiplier, use any number < 1 to disable.
/// \param[in] lr - learning rate to use for virtual batch, -1 for internal configuration.
/// \param[in] momentum - batch norm momentum to use for virtual batch, -1 for internal configuration.
///
/// \return OH_AI_Status.
OH_AI_API OH_AI_Status OH_AI_ModelSetupVirtualBatch(OH_AI_ModelHandle model, int virtual_batch_multiplier, float lr, float momentum);

/// \brief Export training model from file. Only valid for Lite Train.
///
/// \param[in] model The model data.
/// \param[in] model_type The model file type.
/// \param[in] model_file The exported model file.
/// \param[in] quantization_type The quantification type.
/// \param[in] export_inference_only Whether to export a reasoning only model.
/// \param[in] output_tensor_name The set the name of the output tensor of the exported reasoning model, default as
/// empty, and export the complete reasoning model.
/// \param[in] num The number of output_tensor_name.
///
/// \return OH_AI_Status.
OH_AI_API OH_AI_Status OH_AI_ExportModel(OH_AI_ModelHandle model, OH_AI_ModelType model_type, const char *model_file,
                              OH_AI_QuantizationType quantization_type, bool export_inference_only,
                              char **output_tensor_name, size_t num);

/// \brief Export training model from buffer. Only valid for Lite Train.
///
/// \param[in] model The model data.
/// \param[in] model_type The model file type.
/// \param[in] model_data The exported model buffer.
/// \param[in] data_size The exported model buffer size.
/// \param[in] quantization_type The quantification type.
/// \param[in] export_inference_only Whether to export a reasoning only model.
/// \param[in] output_tensor_name The set the name of the output tensor of the exported reasoning model, default as
/// empty, and export the complete reasoning model.
/// \param[in] num The number of output_tensor_name.
///
/// \return OH_AI_Status.
OH_AI_API OH_AI_Status OH_AI_ExportModelBuffer(OH_AI_ModelHandle model, OH_AI_ModelType model_type, char **model_data, size_t *data_size,
                                    OH_AI_QuantizationType quantization_type, bool export_inference_only,
                                    char **output_tensor_name, size_t num);

/// \brief Export model's weights, which can be used in micro only. Only valid for Lite Train.
///
/// \param[in] model The model data.
/// \param[in] model_type The model file type.
/// \param[in] weight_file The path of exported weight file.
/// \param[in] is_inference Whether to export weights from a reasoning model. Currently, only support this is `true`.
/// \param[in] enable_fp16 Float-weight is whether to be saved in float16 format.
/// \param[in] changeable_weights_name The set the name of these weight tensors, whose shape is changeable.
/// \param[in] num The number of changeable_weights_name.
///
/// \return OH_AI_Status.
OH_AI_API OH_AI_Status OH_AI_ExportWeightsCollaborateWithMicro(OH_AI_ModelHandle model, OH_AI_ModelType model_type,
                                                    const char *weight_file, bool is_inference, bool enable_fp16,
                                                    char **changeable_weights_name, size_t num);

#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_INCLUDE_C_API_MODEL_C_H
