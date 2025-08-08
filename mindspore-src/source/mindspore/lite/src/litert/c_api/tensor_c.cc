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
#include <unordered_map>
#include "include/c_api/tensor_c.h"
#include "include/api/status.h"
#include "src/tensor.h"
#include "src/litert/cxx_api/tensor/tensor_impl.h"
#ifdef SUPPORT_NNRT
#include "src/litert/delegate/nnrt/nnrt_allocator.h"
#endif

// allocator_table contains mapping of raw_ptr to weak_ptr of Allocator, allocator_table_mutex is used in multi-thread
// scene when user build multiple models, to avoid read/write unordered_map conflicts crash.
static std::unordered_map<void *, std::weak_ptr<mindspore::Allocator>> allocator_table;
static std::mutex allocator_table_mutex;

void CleanAllocatorTable() {
  std::lock_guard<std::mutex> lock(allocator_table_mutex);
#ifdef SUPPORT_NNRT
  auto nnrt_allocator = mindspore::lite::NNRTAllocator::GetInstance().get();
  for (auto it = allocator_table.begin(); it != allocator_table.end(); ) {
    if (it->first != nnrt_allocator) {
      it = allocator_table.erase(it);
    } else {
      ++it;
    }
  }
#else
  allocator_table.clear();
#endif
}

OH_AI_TensorHandle OH_AI_TensorCreate(const char *name, OH_AI_DataType type, const int64_t *shape, size_t shape_num,
                                      const void *data, size_t data_len) {
  if (name == nullptr || shape == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return nullptr;
  }
  if (*name == '\0') {
    MS_LOG(ERROR) << "tensor name is empty.";
    return nullptr;
  }
  std::vector<int32_t> vec_shape(shape_num);
  for (size_t i = 0; i < shape_num; i++) {
    vec_shape[i] = shape[i];
  }
  auto lite_tensor =
    mindspore::lite::Tensor::CreateTensor(name, static_cast<mindspore::TypeId>(type), vec_shape, data, data_len);
  auto lite_tensor_impl = std::make_shared<mindspore::LiteTensorImpl>(lite_tensor);
  if (lite_tensor_impl == nullptr || lite_tensor_impl->lite_tensor() == nullptr) {
    MS_LOG(ERROR) << "Failed to allocate tensor impl.";
    return nullptr;
  }
  lite_tensor_impl->set_from_session(false);
  lite_tensor_impl->set_own_data(lite_tensor_impl->lite_tensor()->own_data());
  auto impl = new (std::nothrow) mindspore::MSTensor(lite_tensor_impl);
  if (impl == nullptr) {
    MS_LOG(ERROR) << "Failed to allocate MSTensor.";
    return nullptr;
  }
  return impl;
}

void OH_AI_TensorDestroy(OH_AI_TensorHandle *tensor) {
  if (tensor == nullptr || *tensor == nullptr) {
    MS_LOG(ERROR) << "tensor is nullptr.";
    return;
  }
  auto impl = static_cast<mindspore::MSTensor *>(*tensor);
  delete impl;
  *tensor = nullptr;
}

OH_AI_TensorHandle OH_AI_TensorClone(OH_AI_TensorHandle tensor) {
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return nullptr;
  }
  auto impl = static_cast<mindspore::MSTensor *>(tensor);
  auto clone_impl = impl->Clone();
  if (clone_impl == nullptr) {
    MS_LOG(ERROR) << "Failed to allocate tensor impl.";
    return nullptr;
  }
  std::static_pointer_cast<mindspore::LiteTensorImpl>(clone_impl->impl())->set_own_data(false);
  clone_impl->SetTensorName(impl->Name() + "_duplicate");
  return clone_impl;
}

void OH_AI_TensorSetName(OH_AI_TensorHandle tensor, const char *name) {
  if (tensor == nullptr || name == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return;
  }
  if (*name == '\0') {
    MS_LOG(ERROR) << "tensor name is empty.";
    return;
  }
  auto impl = static_cast<mindspore::MSTensor *>(tensor);
  impl->SetTensorName(name);
}

const char *OH_AI_TensorGetName(const OH_AI_TensorHandle tensor) {
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return nullptr;
  }
  auto ms_tensor = static_cast<mindspore::MSTensor *>(tensor);
  return std::static_pointer_cast<mindspore::LiteTensorImpl>(ms_tensor->impl())->Name().c_str();
}

void OH_AI_TensorSetDataType(OH_AI_TensorHandle tensor, OH_AI_DataType type) {
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return;
  }
  auto impl = static_cast<mindspore::MSTensor *>(tensor);
  impl->SetDataType(static_cast<mindspore::DataType>(type));
}

OH_AI_DataType OH_AI_TensorGetDataType(const OH_AI_TensorHandle tensor) {
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return OH_AI_DATATYPE_UNKNOWN;
  }
  auto impl = static_cast<mindspore::MSTensor *>(tensor);
  auto dtype = impl->DataType();
  return static_cast<OH_AI_DataType>(dtype);
}

void OH_AI_TensorSetShape(OH_AI_TensorHandle tensor, const int64_t *shape, size_t shape_num) {
  if (tensor == nullptr || shape == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return;
  }
  auto impl = static_cast<mindspore::MSTensor *>(tensor);
  std::vector<int64_t> vec_shape(shape_num);
  for (size_t i = 0; i < shape_num; i++) {
    vec_shape[i] = shape[i];
  }
  impl->SetShape(vec_shape);
}

const int64_t *OH_AI_TensorGetShape(const OH_AI_TensorHandle tensor, size_t *shape_num) {
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return nullptr;
  }
  auto impl = static_cast<mindspore::MSTensor *>(tensor);
  *shape_num = impl->Shape().size();
  return impl->Shape().data();
}

void OH_AI_TensorSetFormat(OH_AI_TensorHandle tensor, OH_AI_Format format) {
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return;
  }
  auto impl = static_cast<mindspore::MSTensor *>(tensor);
  return impl->SetFormat(static_cast<mindspore::Format>(format));
}

OH_AI_Format OH_AI_TensorGetFormat(const OH_AI_TensorHandle tensor) {
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return OH_AI_FORMAT_NHWC;
  }
  auto impl = static_cast<mindspore::MSTensor *>(tensor);
  return static_cast<OH_AI_Format>(impl->format());
}

void OH_AI_TensorSetData(OH_AI_TensorHandle tensor, void *data) {
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return;
  }
  auto impl = static_cast<mindspore::MSTensor *>(tensor);
  return impl->SetData(data, true);
}

OH_AI_Status OH_AI_TensorSetUserData(OH_AI_TensorHandle tensor, void *data, size_t data_size) {
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return OH_AI_STATUS_LITE_NULLPTR;
  }

  auto impl = static_cast<mindspore::MSTensor *>(tensor);
  if ((impl->DataSize() > 0) && (data_size != impl->DataSize())) {
    MS_LOG(ERROR) << "input data size does not match inner data size";
    return OH_AI_STATUS_LITE_PARAM_INVALID;
  }

  // This is one tricky way to represent that the inner data is not owned by tensor itself.
  impl->SetAllocator(nullptr);
  impl->SetData(data, false);
  return OH_AI_STATUS_SUCCESS;
}

const void *OH_AI_TensorGetData(const OH_AI_TensorHandle tensor) {
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return nullptr;
  }
  auto impl = static_cast<mindspore::MSTensor *>(tensor);
  return impl->Data().get();
}

void *OH_AI_TensorGetMutableData(const OH_AI_TensorHandle tensor) {
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return nullptr;
  }
  auto impl = static_cast<mindspore::MSTensor *>(tensor);
  return impl->MutableData();
}

int64_t OH_AI_TensorGetElementNum(const OH_AI_TensorHandle tensor) {
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return 0;
  }
  auto impl = static_cast<mindspore::MSTensor *>(tensor);
  return impl->ElementNum();
}

size_t OH_AI_TensorGetDataSize(const OH_AI_TensorHandle tensor) {
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return 0;
  }
  auto impl = static_cast<mindspore::MSTensor *>(tensor);
  return impl->DataSize();
}

OH_AI_Status OH_AI_TensorSetAllocator(OH_AI_TensorHandle tensor, void *allocator) {
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return OH_AI_STATUS_LITE_NULLPTR;
  }
  auto impl = static_cast<mindspore::MSTensor *>(tensor);
  std::lock_guard<std::mutex> lock(allocator_table_mutex);
  if (allocator_table.count(allocator) == 0) {
    MS_LOG(ERROR) << "the input allocator does not belong to framework";
    return OH_AI_STATUS_LITE_PARAM_INVALID;
  }
  std::static_pointer_cast<mindspore::LiteTensorImpl>(impl->impl())->set_own_data(true);
  auto allocator_ptr = allocator_table[allocator].lock();
  if (allocator_ptr != nullptr) {
    impl->SetAllocator(allocator_ptr);
  } else {
    MS_LOG(ERROR) << "get allocator shared ptr failed.";
    return OH_AI_STATUS_LITE_NULLPTR;
  }
  return OH_AI_STATUS_SUCCESS;
}

void *OH_AI_TensorGetAllocator(OH_AI_TensorHandle tensor) {
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return nullptr;
  }
  auto impl = static_cast<mindspore::MSTensor *>(tensor);
  std::lock_guard<std::mutex> lock(allocator_table_mutex);
  allocator_table[impl->allocator().get()] = impl->allocator();
  return impl->allocator().get();
}
