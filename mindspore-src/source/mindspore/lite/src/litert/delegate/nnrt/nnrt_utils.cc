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

#include "src/litert/delegate/nnrt/nnrt_utils.h"
#include <unordered_map>

namespace mindspore::lite {
OH_NN_Format CastToNNRtFormat(Format format) {
  const std::unordered_map<Format, OH_NN_Format> kFormatMap = {
    {Format::NCHW, OH_NN_FORMAT_NCHW},
    {Format::NHWC, OH_NN_FORMAT_NHWC},
  };
  auto iter = kFormatMap.find(format);
  if (iter == kFormatMap.end()) {
    return OH_NN_FORMAT_NONE;
  }
  return iter->second;
}

OH_NN_DataType CastToNNRtDataType(TypeId data_type) {
  OH_NN_DataType oh_data_type;
  switch (data_type) {
    case TypeId::kMetaTypeBegin:
    case TypeId::kMetaTypeType:
    case TypeId::kMetaTypeAny:
    case TypeId::kMetaTypeObject:
    case TypeId::kMetaTypeTypeType:
    case TypeId::kMetaTypeProblem:
    case TypeId::kMetaTypeExternal:
    case TypeId::kMetaTypeNone:
    case TypeId::kMetaTypeNull:
    case TypeId::kMetaTypeEllipsis:
    case TypeId::kMetaTypeEnd:
    case TypeId::kObjectTypeNumber:
    case TypeId::kObjectTypeString:
    case TypeId::kObjectTypeList:
    case TypeId::kObjectTypeTuple:
    case TypeId::kObjectTypeSlice:
    case TypeId::kObjectTypeKeyword:
    case TypeId::kObjectTypeTensorType:
    case TypeId::kObjectTypeRowTensorType:
    case TypeId::kObjectTypeCOOTensorType:
    case TypeId::kObjectTypeUndeterminedType:
    case TypeId::kObjectTypeClass:
    case TypeId::kObjectTypeDictionary:
    case TypeId::kObjectTypeFunction:
    case TypeId::kObjectTypeJTagged:
    case TypeId::kObjectTypeSymbolicKeyType:
    case TypeId::kObjectTypeEnvType:
    case TypeId::kObjectTypeRefKey:
    case TypeId::kObjectTypeRef:
    case TypeId::kObjectTypeEnd:
      oh_data_type = OH_NN_UNKNOWN;
      break;
    case TypeId::kNumberTypeBool:
      oh_data_type = OH_NN_BOOL;
      break;
    case TypeId::kNumberTypeInt8:
      oh_data_type = OH_NN_INT8;
      break;
    case TypeId::kNumberTypeInt16:
      oh_data_type = OH_NN_INT16;
      break;
    case TypeId::kNumberTypeInt32:
      oh_data_type = OH_NN_INT32;
      break;
    case TypeId::kNumberTypeInt64:
      oh_data_type = OH_NN_INT64;
      break;
    case TypeId::kNumberTypeUInt8:
      oh_data_type = OH_NN_UINT8;
      break;
    case TypeId::kNumberTypeUInt16:
      oh_data_type = OH_NN_UINT16;
      break;
    case TypeId::kNumberTypeUInt32:
      oh_data_type = OH_NN_UINT32;
      break;
    case TypeId::kNumberTypeUInt64:
      oh_data_type = OH_NN_UINT64;
      break;
    case TypeId::kNumberTypeFloat16:
      oh_data_type = OH_NN_FLOAT16;
      break;
    case TypeId::kNumberTypeFloat32:
      oh_data_type = OH_NN_FLOAT32;
      break;
    case TypeId::kNumberTypeFloat64:
      oh_data_type = OH_NN_FLOAT64;
      break;
    default: {
      oh_data_type = OH_NN_UNKNOWN;
    }
  }
  return oh_data_type;
}
}  // namespace mindspore::lite
