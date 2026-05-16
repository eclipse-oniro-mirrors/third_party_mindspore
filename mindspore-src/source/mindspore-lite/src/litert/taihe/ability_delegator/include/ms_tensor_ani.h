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
#ifndef MS_TENSOR_ANI
#define MS_TENSOR_ANI

#include "ohos.ai.mindSporeLite.proj.hpp"
#include "ohos.ai.mindSporeLite.impl.hpp"
#include "include/api/types.h"
#include <map>

namespace mindspore_ani {
static const std::map<mindspore::DataType, ::ohos::ai::mindSporeLite::DataType> tensorDataTypeMapANI = {
  {mindspore::DataType::kTypeUnknown, ::ohos::ai::mindSporeLite::DataType::key_t::TYPE_UNKNOWN},
  {mindspore::DataType::kObjectTypeString, ::ohos::ai::mindSporeLite::DataType::key_t::TYPE_UNKNOWN},
  {mindspore::DataType::kObjectTypeList, ::ohos::ai::mindSporeLite::DataType::key_t::TYPE_UNKNOWN},
  {mindspore::DataType::kObjectTypeTuple, ::ohos::ai::mindSporeLite::DataType::key_t::TYPE_UNKNOWN},
  {mindspore::DataType::kObjectTypeTensorType, ::ohos::ai::mindSporeLite::DataType::key_t::TYPE_UNKNOWN},
  {mindspore::DataType::kNumberTypeBegin, ::ohos::ai::mindSporeLite::DataType::key_t::TYPE_UNKNOWN},
  {mindspore::DataType::kNumberTypeBool, ::ohos::ai::mindSporeLite::DataType::key_t::TYPE_UNKNOWN},
  {mindspore::DataType::kNumberTypeInt8, ::ohos::ai::mindSporeLite::DataType::key_t::NUMBER_TYPE_INT8},
  {mindspore::DataType::kNumberTypeInt16, ::ohos::ai::mindSporeLite::DataType::key_t::NUMBER_TYPE_INT16},
  {mindspore::DataType::kNumberTypeInt32, ::ohos::ai::mindSporeLite::DataType::key_t::NUMBER_TYPE_INT32},
  {mindspore::DataType::kNumberTypeInt64, ::ohos::ai::mindSporeLite::DataType::key_t::NUMBER_TYPE_INT64},
  {mindspore::DataType::kNumberTypeUInt8, ::ohos::ai::mindSporeLite::DataType::key_t::NUMBER_TYPE_UINT8},
  {mindspore::DataType::kNumberTypeUInt16, ::ohos::ai::mindSporeLite::DataType::key_t::NUMBER_TYPE_UINT16},
  {mindspore::DataType::kNumberTypeInt32, ::ohos::ai::mindSporeLite::DataType::key_t::NUMBER_TYPE_UINT32},
  {mindspore::DataType::kNumberTypeUInt64, ::ohos::ai::mindSporeLite::DataType::key_t::NUMBER_TYPE_UINT64},
  {mindspore::DataType::kNumberTypeFloat16, ::ohos::ai::mindSporeLite::DataType::key_t::NUMBER_TYPE_FLOAT16},
  {mindspore::DataType::kNumberTypeFloat32, ::ohos::ai::mindSporeLite::DataType::key_t::NUMBER_TYPE_FLOAT32},
  {mindspore::DataType::kNumberTypeFloat64, ::ohos::ai::mindSporeLite::DataType::key_t::NUMBER_TYPE_FLOAT64},
  {mindspore::DataType::kNumberTypeBFloat16, ::ohos::ai::mindSporeLite::DataType::key_t::TYPE_UNKNOWN},
  {mindspore::DataType::kNumberTypeEnd, ::ohos::ai::mindSporeLite::DataType::key_t::TYPE_UNKNOWN},
  {mindspore::DataType::kInvalidType, ::ohos::ai::mindSporeLite::DataType::key_t::TYPE_UNKNOWN},
  {mindspore::DataType::kTypeUnknown, ::ohos::ai::mindSporeLite::DataType::key_t::TYPE_UNKNOWN},
};
static const std::map<mindspore::Format, ::ohos::ai::mindSporeLite::Format> tensorFormatMapANI = {
  {mindspore::Format::DEFAULT_FORMAT, ::ohos::ai::mindSporeLite::Format::key_t::DEFAULT_FORMAT},
  {mindspore::Format::NCHW, ::ohos::ai::mindSporeLite::Format::key_t::NCHW},
  {mindspore::Format::NHWC, ::ohos::ai::mindSporeLite::Format::key_t::NHWC},
  {mindspore::Format::NHWC4, ::ohos::ai::mindSporeLite::Format::key_t::NHWC4},
  {mindspore::Format::HWKC, ::ohos::ai::mindSporeLite::Format::key_t::HWKC},
  {mindspore::Format::HWCK, ::ohos::ai::mindSporeLite::Format::key_t::HWCK},
  {mindspore::Format::KCHW, ::ohos::ai::mindSporeLite::Format::key_t::KCHW},
  {mindspore::Format::CKHW, ::ohos::ai::mindSporeLite::Format::key_t::DEFAULT_FORMAT},
  {mindspore::Format::KHWC, ::ohos::ai::mindSporeLite::Format::key_t::DEFAULT_FORMAT},
  {mindspore::Format::CHWK, ::ohos::ai::mindSporeLite::Format::key_t::DEFAULT_FORMAT},
  {mindspore::Format::HW, ::ohos::ai::mindSporeLite::Format::key_t::DEFAULT_FORMAT},
  {mindspore::Format::HW4, ::ohos::ai::mindSporeLite::Format::key_t::DEFAULT_FORMAT},
  {mindspore::Format::NC, ::ohos::ai::mindSporeLite::Format::key_t::DEFAULT_FORMAT},
  {mindspore::Format::NC4, ::ohos::ai::mindSporeLite::Format::key_t::DEFAULT_FORMAT},
  {mindspore::Format::NC4HW4, ::ohos::ai::mindSporeLite::Format::key_t::DEFAULT_FORMAT},
  {mindspore::Format::NUM_OF_FORMAT, ::ohos::ai::mindSporeLite::Format::key_t::DEFAULT_FORMAT},
  {mindspore::Format::NCDHW, ::ohos::ai::mindSporeLite::Format::key_t::DEFAULT_FORMAT},
  {mindspore::Format::NWC, ::ohos::ai::mindSporeLite::Format::key_t::DEFAULT_FORMAT},
  {mindspore::Format::NCW, ::ohos::ai::mindSporeLite::Format::key_t::DEFAULT_FORMAT},
  {mindspore::Format::NDHWC, ::ohos::ai::mindSporeLite::Format::key_t::DEFAULT_FORMAT},
  {mindspore::Format::NC8HW8, ::ohos::ai::mindSporeLite::Format::key_t::DEFAULT_FORMAT},
};

}  // namespace mindspore_ani
#endif
