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
#ifndef MS_STATUS_ANI
#define MS_STATUS_ANI

namespace mindspore_ani {

enum MSLiteStatusANI : int32_t {
  MS_SUCCESS_ANI = 0,                  // Success
  MS_ERROR_ANI = 1000199,              // Fail
  MS_ILLEGAL_STATE_ANI = 1000198,      // Status error
  MS_INVALID_PARAM_ANI = 1000197,      // Invalid parameter
  MS_NOT_EXISTED_PARAM_ANI = 1000196,  // Not existed parameter
  MS_INVALID_OPERATION_ANI = 1000195   // Invalid operation
};

enum MSLiteContextParamANI : int {
  MS_UNSET_VALUE_ANI = -1,
  MS_PARAM0_ANI = 0,
  MS_PARAM1_ANI = 1,
  MS_PARAM2_ANI = 2,
  MS_PARAM3_ANI = 3,
  MS_PARAM4_ANI = 4
};

enum MSLiteErrorCodeANI : int32_t {
  MS_LOAD_PATH_ERROR_PREDICT = 1000000,
  MS_LOAD_CONTEXT_ERROR = 1000001,
  MS_LOAD_NATIVE_ERROR_PREDICT = 1000002,
  MS_LOAD_WAY_ERROR_PREDICT = 1000003,
  MS_LOAD_BUFFER_ERROR_PREDICT = 1000004,
  MS_LOAD_BUFFER_NATIVE_ERROR_PREDICT = 1000005,
  MS_LOAD_FD_NATIVE_ERROR_PREDICT = 1000007,
  MS_LOAD_PATH_ERROR_TRAIN = 1000008,
  MS_LOAD_PATH_NATIVE_ERROR_TRAIN = 1000009,
  MS_LOAD_BUFFER_ERROR_TRAIN = 1000010,
  MS_LOAD_BUFFER_NATIVE_ERROR_TRAIN = 1000011,
  MS_LOAD_FD_ERROR_TRAIN = 1000012,
  MS_SETDATA_SIZE_ERROR_MSTENSOR = 1000013,
};

static const std::unordered_map<MSLiteErrorCodeANI, std::string> MSLiteErrorCodeInfoANI{
  {MS_LOAD_PATH_ERROR_PREDICT,
   "Model path error. Possible causes: 1. The model path is null; 2. The model path does not exist."},
  {MS_LOAD_CONTEXT_ERROR,
   "Invalid context. Possible causes: 1. The context target is incorrect; 2. The device information is incorrect."},
  {MS_LOAD_NATIVE_ERROR_PREDICT,
   "Failed to create native model. Possible causes: 1. Insufficient permission to access the model path; 2. The model "
   "file is corrupted."},
  {MS_LOAD_WAY_ERROR_PREDICT,
   "Error in model loading method. Possible causes: 1. The loading method must be path, buffer, or fd."},
  {MS_LOAD_BUFFER_ERROR_PREDICT,
   "Model buffer error. Possible causes: 1. The buffer size is 0; 2. The buffer is a null pointer."},
  {MS_LOAD_BUFFER_NATIVE_ERROR_PREDICT,
   "Failed to create native model from buffer. Possible causes: 1. The buffer size is incorrect; 2. The buffer file is "
   "damaged."},
  {MS_LOAD_FD_NATIVE_ERROR_PREDICT,
   "Failed to create native model from file descriptor (fd). Possible causes: 1. The file descriptor (fd) is "
   "incorrect; 2. The model file is damaged."},
  {MS_LOAD_PATH_ERROR_TRAIN,
   "Invalid model path in training. Possible causes: 1. The model path is null; 2. The model path does not exist."},
  {MS_LOAD_PATH_NATIVE_ERROR_TRAIN,
   "Failed to create native training model from path. Possible causes: 1. The model file is incorrect; 2. The training "
   "configuration is incorrect."},
  {MS_LOAD_BUFFER_ERROR_TRAIN,
   "Invalid model buffer in training. Possible causes: 1. The model buffer size is incorrect; 2. The model buffer is "
   "null."},
  {MS_LOAD_BUFFER_NATIVE_ERROR_TRAIN,
   "Failed to create native training model from buffer. Possible causes: 1. The model buffer is incorrect; 2. The "
   "training configuration is incorrect."},
  {MS_LOAD_FD_ERROR_TRAIN,
   "Failed to create native training model from file descriptor (fd). Possible causes: 1. The model file or file "
   "descriptor (fd) is incorrect; 2. The training configuration is incorrect."},
  {MS_SETDATA_SIZE_ERROR_MSTENSOR,
   "Failed to set MSTensor data. Possible causes: 1. The input array buffer size is incorrect."}};

}  // namespace mindspore_ani
#endif
