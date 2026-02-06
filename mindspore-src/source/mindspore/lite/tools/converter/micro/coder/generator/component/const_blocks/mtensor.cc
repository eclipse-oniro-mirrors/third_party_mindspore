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

#include "coder/generator/component/const_blocks/mtensor.h"

namespace mindspore::lite::micro {
const char tensor_header[] = R"RAW(
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

#ifndef MINDSPORE_LITE_MICRO_LIBRARY_SOURCE_TENSOR_H_
#define MINDSPORE_LITE_MICRO_LIBRARY_SOURCE_TENSOR_H_

#include "c_api/data_type_c.h"
#include "c_api/format_c.h"
#include "c_api/tensor_c.h"
#include <stdbool.h>
#ifdef ENABLE_FP16
#include <arm_neon.h>
#endif

typedef struct {
  enum OH_AI_DataType type;
  enum OH_AI_Format format;
  char *name;
  int ndim;
  int64_t *shape;
  void *data;
  int quant_nums;
  bool owned;
} MicroTensor; // if change MicroTensor parameter, need to update kMicroTensorSize

typedef struct {
  int num;
  MicroTensor **tensor;
} MicroTensorList;

typedef struct {
  int bit_num;
  double scale;
  int32_t zero_point;
  double min;
  double max;
} QuantParam;

enum TypeTransMode {
  TypeTransMode_FP32_TO_FP16 = 0,
  TypeTransMode_FP16_TO_FP32 = 1,
  TypeTransMode_UNSUPPORT = 2,
  TypeTransMode_MAX = TypeTransMode_UNSUPPORT
};

void *TransformInput(OH_AI_TensorHandle tensor, int expect_type, bool *type_changed);

#ifdef ENABLE_FP16
void Fp32CastToFp16(const float *input, float16_t *output, int number);
void Fp16CastToFp32(const float16_t *input, float *output, int number);
#endif

#endif  // MINDSPORE_LITE_MICRO_LIBRARY_SOURCE_TENSOR_H_

)RAW";

const char tensor_source[] = R"RAW(
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

#include "include/c_api/tensor_c.h"
#include "stdlib.h"
#include "string.h"
#include "tensor.h"

size_t DataTypeSize(const OH_AI_DataType type) {
  switch (type) {
    case OH_AI_DATATYPE_NUMBERTYPE_FLOAT64:
      return sizeof(double);
    case OH_AI_DATATYPE_NUMBERTYPE_FLOAT32:
      return sizeof(float);
    case OH_AI_DATATYPE_NUMBERTYPE_INT8:
      return sizeof(int8_t);
    case OH_AI_DATATYPE_NUMBERTYPE_UINT8:
      return sizeof(uint8_t);
    case OH_AI_DATATYPE_NUMBERTYPE_FLOAT16:
    case OH_AI_DATATYPE_NUMBERTYPE_INT16:
      return sizeof(int16_t);
    case OH_AI_DATATYPE_NUMBERTYPE_INT32:
      return sizeof(int32_t);
    case OH_AI_DATATYPE_NUMBERTYPE_INT64:
      return sizeof(int64_t);
    case OH_AI_DATATYPE_NUMBERTYPE_UINT16:
      return sizeof(uint16_t);
    case OH_AI_DATATYPE_NUMBERTYPE_UINT32:
      return sizeof(uint32_t);
    case OH_AI_DATATYPE_NUMBERTYPE_UINT64:
      return sizeof(uint64_t);
    case OH_AI_DATATYPE_OBJECTTYPE_STRING:
      return sizeof(char);
    default:
      return 0;
  }
}

OH_AI_TensorHandle OH_AI_TensorCreate(const char *name, OH_AI_DataType type, const int64_t *shape, size_t shape_num,
                              const void *data, size_t data_len) {
  size_t data_type_len = DataTypeSize(type);
  size_t acc_sum = 1;
  for (int i = 0; i < shape_num; i++) {
    acc_sum = acc_sum * shape[i];
  }
  if (acc_sum * data_type_len != data_len) {
    return NULL;
  }
  MicroTensor *micro_tensor = malloc(sizeof(MicroTensor));
  size_t len = strlen(name);
  micro_tensor->name = malloc(len + 1);
  memcpy(micro_tensor->name, name, len + 1);
  micro_tensor->type = type;
  micro_tensor->ndim = shape_num;
  micro_tensor->data = malloc(data_len);
  micro_tensor->owned = true;
  memcpy(micro_tensor->data, data, data_len);
  micro_tensor->shape = malloc(shape_num * sizeof(int64_t));
  memcpy(micro_tensor->shape, shape, shape_num * sizeof(int64_t));
  micro_tensor->format = OH_AI_FORMAT_NHWC;
  return micro_tensor;
}

void OH_AI_TensorDestroy(OH_AI_TensorHandle *tensor) {
  MicroTensor* micro_tensor = (MicroTensor*)(*tensor);
  if (micro_tensor->data != NULL && micro_tensor->owned) {
    free(micro_tensor->data);
  }
  free(micro_tensor);
  *tensor = NULL;
}

void OH_AI_TensorSetName(OH_AI_TensorHandle tensor, const char *name) {
  MicroTensor* micro_tensor = (MicroTensor*)(tensor);
  if(micro_tensor->name != NULL) {
    free(micro_tensor->name);
  }
  size_t len = strlen(name);
  micro_tensor->name = malloc(len + 1);
  memcpy(micro_tensor->name, name, len + 1);
}

OH_AI_TensorHandle OH_AI_TensorClone(OH_AI_TensorHandle tensor) {
  MicroTensor* micro_tensor = (MicroTensor*)(tensor);
  MicroTensor *clone_tensor = malloc( sizeof(MicroTensor));
  size_t tensor_data_size = OH_AI_TensorGetDataSize(micro_tensor);
  clone_tensor->data = malloc(tensor_data_size);
  clone_tensor->owned = true;
  memcpy(clone_tensor->data,micro_tensor->data,tensor_data_size);
  clone_tensor->name = micro_tensor->name;
  clone_tensor->type = micro_tensor->type;
  clone_tensor->ndim = micro_tensor->ndim;
  size_t shape_data_size = sizeof(int64_t) * micro_tensor->ndim;
  int64_t* clone_shape = malloc(shape_data_size);
  memcpy(clone_shape,micro_tensor->shape,shape_data_size);
  clone_tensor->shape = clone_shape;
  char* clone_name = malloc(strlen(micro_tensor->name));
  strcpy(clone_name,micro_tensor->name);
  clone_tensor->format = OH_AI_FORMAT_NHWC;
  return clone_tensor;
}

const char *OH_AI_TensorGetName(const OH_AI_TensorHandle tensor) {
  MicroTensor* micro_tensor = (MicroTensor*)(tensor);
  return micro_tensor->name;
}

void OH_AI_TensorSetDataType(OH_AI_TensorHandle tensor, OH_AI_DataType type) {
  MicroTensor* micro_tensor = (MicroTensor*)(tensor);
  micro_tensor->type = type;
}

OH_AI_DataType OH_AI_TensorGetDataType(const OH_AI_TensorHandle tensor) {
  MicroTensor* micro_tensor = (MicroTensor*)(tensor);
  return micro_tensor->type;
}

void OH_AI_TensorSetShape(OH_AI_TensorHandle tensor, const int64_t *shape, size_t shape_num) {
  MicroTensor* micro_tensor = (MicroTensor*)(tensor);
  if(micro_tensor->shape != NULL) {
    free(micro_tensor->shape);
  }
  micro_tensor->ndim = shape_num;
  micro_tensor->shape = malloc(shape_num * sizeof(int64_t));
  memcpy(micro_tensor->shape, shape, shape_num * sizeof(int64_t));
}

const int64_t *OH_AI_TensorGetShape(const OH_AI_TensorHandle tensor, size_t *shape_num) {
  MicroTensor* micro_tensor = (MicroTensor*)(tensor);
  *shape_num =  micro_tensor->ndim;
  return micro_tensor->shape;
}

void OH_AI_TensorSetFormat(OH_AI_TensorHandle tensor, OH_AI_Format format) {
  MicroTensor* micro_tensor = (MicroTensor*)(tensor);
  micro_tensor->format = format;
}

OH_AI_Format OH_AI_TensorGetFormat(const OH_AI_TensorHandle tensor) {
  MicroTensor* micro_tensor = (MicroTensor*)(tensor);
  return micro_tensor->format;
}

void OH_AI_TensorSetData(OH_AI_TensorHandle tensor, void *data) {
  MicroTensor* micro_tensor = (MicroTensor*)(tensor);
  if (micro_tensor->data == data) {
    return;
  }
  if(micro_tensor->data != NULL) {
    if (micro_tensor->owned) {
      free(micro_tensor->data);
      micro_tensor->owned = false;
    }
  }
  micro_tensor->data = data;
}

const void *OH_AI_TensorGetData(const OH_AI_TensorHandle tensor) {
  MicroTensor* micro_tensor = (MicroTensor*)(tensor);
  return micro_tensor->data;
}

void *OH_AI_TensorGetMutableData(const OH_AI_TensorHandle tensor) {
  MicroTensor* micro_tensor = (MicroTensor*)(tensor);
  if(micro_tensor->data) {
    return micro_tensor->data;
  }
  void* data = malloc(OH_AI_TensorGetDataSize(tensor));
  micro_tensor->owned = true;
  micro_tensor->data = data;
  return data;
}

int64_t OH_AI_TensorGetElementNum(const OH_AI_TensorHandle tensor) {
  MicroTensor* micro_tensor = (MicroTensor*)(tensor);
  int64_t acc_sum = 1;
  for(int i=0;i< micro_tensor->ndim;i++) {
    acc_sum = acc_sum * micro_tensor->shape[i];
  }
  return acc_sum;
}

size_t OH_AI_TensorGetDataSize(const OH_AI_TensorHandle tensor) {
  MicroTensor* micro_tensor = (MicroTensor*)(tensor);
  size_t data_type_size = DataTypeSize(micro_tensor->type);
  int64_t elements = OH_AI_TensorGetElementNum(tensor);
  return data_type_size * elements;
}

#ifdef ENABLE_FP16
void Fp32CastToFp16(const float *input, float16_t *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = (float16_t)(input[i]);
  }
}

void Fp16CastToFp32(const float16_t *input, float *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = (float)(input[i]);
  }
}
#endif

void *TransformInput(OH_AI_TensorHandle tensor, int expect_type, bool *type_changed) {
  MicroTensor* micro_tensor = (MicroTensor*)(tensor);
  int cur_type = micro_tensor->type;
  if (cur_type == expect_type) {
    return micro_tensor->data;
  }
  int type_trans_mode = TypeTransMode_MAX;
  if (expect_type == OH_AI_DATATYPE_NUMBERTYPE_FLOAT16 && cur_type == OH_AI_DATATYPE_NUMBERTYPE_FLOAT32) {
    type_trans_mode = TypeTransMode_FP32_TO_FP16;
  } else if (expect_type == OH_AI_DATATYPE_NUMBERTYPE_FLOAT32 && cur_type == OH_AI_DATATYPE_NUMBERTYPE_FLOAT16) {
    type_trans_mode = TypeTransMode_FP16_TO_FP32;
  }
  if (type_trans_mode == TypeTransMode_UNSUPPORT) {
    return NULL;
  }
#ifdef ENABLE_FP16
  int shape_size = micro_tensor->ndim;
  int num = 1;
  for (int i = 0; i < shape_size; ++i) {
    num *= micro_tensor->shape[i];
  }
  if (type_trans_mode == TypeTransMode_FP32_TO_FP16) {
    void *expect_input_fp16 = (void *)malloc(DataTypeSize(expect_type) * num);
    Fp32CastToFp16((float *)micro_tensor->data, (float16_t *)expect_input_fp16, num);
    *type_changed = true;
    return expect_input_fp16;
  } else if (type_trans_mode == TypeTransMode_FP16_TO_FP32) {
    void *expect_input_fp32 = (void *)malloc(DataTypeSize(expect_type) * num);
    Fp16CastToFp32((float16_t *)micro_tensor->data, (float *)expect_input_fp32, num);
    *type_changed = true;
    return expect_input_fp32;
  }
#endif
  return NULL;
}

)RAW";
}  // namespace mindspore::lite::micro
