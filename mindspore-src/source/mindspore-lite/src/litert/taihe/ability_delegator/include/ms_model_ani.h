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
#ifndef MS_MODEL_ANI
#define MS_MODEL_ANI

namespace mindspore_ani {

enum MSLiteLoadModelMode : int32_t { kBuffer = 0, kPath, kFD, Unknown };

struct MSLiteModelInfoANI {
  std::string model_path = "";
  char *model_buffer_data = nullptr;
  size_t model_buffer_total = 0;
  int32_t model_fd = 0;
  MSLiteLoadModelMode mode = MSLiteLoadModelMode::kBuffer;
  bool train_model = false;
};

}  // namespace mindspore_ani
#endif
