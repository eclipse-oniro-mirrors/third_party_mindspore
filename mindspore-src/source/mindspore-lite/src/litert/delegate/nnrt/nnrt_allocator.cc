/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include <memory>
#include <atomic>
#include <unordered_map>
#include <map>
#include <mutex>
#include "src/litert/delegate/nnrt/nnrt_allocator.h"
#include "src/litert/delegate/nnrt/nnrt_utils.h"
#include "src/common/log.h"
#include "src/litert/delegate/nnrt/nnrt_wrapper.h"

namespace mindspore {
namespace lite {
std::shared_ptr<NNRTAllocator> NNRTAllocator::GetInstance() {
  static std::shared_ptr<NNRTAllocator> instance(new (std::nothrow) NNRTAllocator());
  return instance;
}

void NNRTAllocator::ClearFreeList() {
  std::lock_guard<std::mutex> locker(mutex_);
  for (auto &it : free_list_) {
    auto membuf = it.second;
    if (membuf == nullptr) {
      MS_LOG(ERROR) << "membuf in free_list_ is nullptr.";
      continue;
    }
    auto& nnrtWrapper = mindspore::NNRTWrapper::GetInstance();
    nnrtWrapper.NNTensorDestroy(&membuf->tensor_);
    nnrtWrapper.NNTensorDescDestroy(&membuf->tensor_desc_);
    delete membuf;
    membuf = nullptr;
  }
  free_list_.clear();
}

void NNRTAllocator::FreeAllocatedTensor(void *data, lite::Tensor *tensor) {
  std::lock_guard<std::mutex> locker(mutex_);
  auto iter = allocated_list_.find(data);
  if (iter == allocated_list_.end()) {
    return;
  }

  auto iter_tensor = allocated_lite_tensors_.find(data);
  if (iter_tensor == allocated_lite_tensors_.end() || iter_tensor->second != tensor) {
    MS_LOG(INFO) << "data: " << data << " not belong to lite tensor: " << tensor << ", skip free";
    return;
  }

  auto membuf = iter->second;
  if (membuf == nullptr) {
    MS_LOG(ERROR) << "membuf in allocated_list_ is nullptr, data: " << data;
    return;
  }
  membuf->ref_count_ = 0;
  (void)allocated_list_.erase(iter);
  (void)allocated_lite_tensors_.erase(data);
  auto& nnrtWrapper = mindspore::NNRTWrapper::GetInstance();
  nnrtWrapper.NNTensorDestroy(&membuf->tensor_);
  nnrtWrapper.NNTensorDescDestroy(&membuf->tensor_desc_);
  delete membuf;
  membuf = nullptr;
  data = nullptr;
}

NNRTAllocator::~NNRTAllocator() {
  std::lock_guard<std::mutex> locker(mutex_);
  for (auto &it : allocated_list_) {
    auto membuf = it.second;
    if (membuf != nullptr) {
      MS_LOG(ERROR) << "NN_Tensor is not released, may lead to memory leak, data ptr: " << membuf->data << ", size: "
                    << membuf->size;
    }
  }

  for (auto &it : free_list_) {
    auto membuf = it.second;
    if (membuf != nullptr) {
      MS_LOG(ERROR) << "NN_Tensor is not released, may lead to memory leak, data ptr: " << membuf->data << ", size: "
                    << membuf->size;
    }
  }
}

OH_NN_ReturnCode NNRTAllocator::SetTensorDesc(NN_TensorDesc *tensor_desc, const std::vector<int> &shape,
                                              const TypeId data_type, const Format format, const std::string &name) {
  auto& nnrtWrapper = mindspore::NNRTWrapper::GetInstance();
  OH_NN_ReturnCode ret = nnrtWrapper.NNTensorDescSetShape(tensor_desc, shape.data(), shape.size());
  if (ret != OH_NN_SUCCESS) {
    MS_LOG(ERROR) << "OH_NNTensorDesc_SetShape failed, shape: " << shape << ", name: " << name;
    return ret;
  }
  
  ret = nnrtWrapper.NNTensorDescSetDataType(tensor_desc, CastToNNRtDataType(data_type));
  if (ret != OH_NN_SUCCESS) {
    MS_LOG(ERROR) << "OH_NNTensorDesc_SetDataType failed, data_type: " << data_type << ", name: " << name;
    
    return ret;
  }

  ret = nnrtWrapper.NNTensorDescSetFormat(tensor_desc, CastToNNRtFormat(format));
  if (ret != OH_NN_SUCCESS) {
    MS_LOG(ERROR) << "OH_NNTensorDesc_SetFormat failed, format: " << format << ", name: " << name;
    
    return ret;
  }

  ret = nnrtWrapper.NNTensorDescSetName(tensor_desc, name.c_str());
  if (ret != OH_NN_SUCCESS) {
    MS_LOG(ERROR) << "OH_NNTensorDesc_SetName failed, name: " << name;
    
    return ret;
  }
  
  return ret;
}

NN_TensorDesc *NNRTAllocator::CreateNNRtTensorDesc(const std::vector<int> &shape, const TypeId data_type,
                                                   const Format format, const std::string &name) {
  auto& nnrtWrapper = mindspore::NNRTWrapper::GetInstance();
  auto tensor_desc = nnrtWrapper.NNTensorDescCreate();
  if (tensor_desc == nullptr) {
    MS_LOG(ERROR) << "OH_NNTensorDesc_Create failed, name: " << name;
    return nullptr;
  }
  OH_NN_ReturnCode ret = SetTensorDesc(tensor_desc, shape, data_type, format, name);
  if (ret != OH_NN_SUCCESS) {
    MS_LOG(ERROR) << "SetTensorDesc failed, name: " << name;
    nnrtWrapper.NNTensorDescDestroy(&tensor_desc);
    return nullptr;
  }
  return tensor_desc;
}

void *NNRTAllocator::MallocByDesc(size_t size, const std::vector<int> &shape, const TypeId data_type,
                                  const Format format, const std::string &name) {
  std::lock_guard<std::mutex> locker(mutex_);
  auto iter = free_list_.lower_bound(size);
  if (iter != free_list_.end() && (size == iter->second->size)) {
    auto membuf = iter->second;
    OH_NN_ReturnCode ret = SetTensorDesc(membuf->tensor_desc_, shape, data_type, format, name);
    if (ret != OH_NN_SUCCESS) {
      MS_LOG(ERROR) << "SetTensorDesc failed, name: " << name;
    } else {
      membuf->ref_count_ = 0;
      (void)free_list_.erase(iter);
      allocated_list_[membuf->data] = membuf;
      return membuf->data;
    }
  }

  auto membuf = new (std::nothrow) MemBuf();
  if (membuf == nullptr) {
    MS_LOG(ERROR) << "new Membuf failed.";
    return nullptr;
  }
  membuf->ref_count_ = 0;
  membuf->tensor_desc_ = CreateNNRtTensorDesc(shape, data_type, format, name);
  if (membuf->tensor_desc_ == nullptr) {
    MS_LOG(ERROR) << "create NN_TensorDesc failed.";
    delete membuf;
    return nullptr;
  }
  auto& nnrtWrapper = mindspore::NNRTWrapper::GetInstance();
  membuf->tensor_ = nnrtWrapper.NNTensorCreate(device_id_, membuf->tensor_desc_);
  if (membuf->tensor_ == nullptr) {
    MS_LOG(ERROR) << "OH_NNTensor_CreateWithSize failed, name: " << name;
    nnrtWrapper.NNTensorDescDestroy(&membuf->tensor_desc_);
    delete membuf;
    return nullptr;
  }

  membuf->data = nnrtWrapper.NNTensorGetDataBuffer(membuf->tensor_);
  if (membuf->data == nullptr) {
    MS_LOG(ERROR) << "OH_NNTensor_GetDataBuffer failed, name: " << name;
    nnrtWrapper.NNTensorDestroy(&membuf->tensor_);
    nnrtWrapper.NNTensorDescDestroy(&membuf->tensor_desc_);
    delete membuf;
    return nullptr;
  }
  
  membuf->size = size;
  allocated_list_[membuf->data] = membuf;
  return membuf->data;
}

void *NNRTAllocator::Malloc(size_t size) {
  MS_LOG(ERROR) << "NNRt Allocator is not support malloc by size.";
  return nullptr;
}

void NNRTAllocator::Free(void *ptr) {
  if (ptr == nullptr) {
    return;
  }

  std::lock_guard<std::mutex> locker(mutex_);
  auto iter = allocated_list_.find(ptr);
  if (iter == allocated_list_.end()) {
    return;
  }
  auto membuf = iter->second;
  membuf->ref_count_ = 0;
  (void)allocated_list_.erase(iter);
  (void)allocated_lite_tensors_.erase(ptr);
  (void)free_list_.insert(std::make_pair(membuf->size, membuf));
}

int NNRTAllocator::RefCount(void *ptr) {
  if (ptr == nullptr) {
    return NNRT_ALLOCATION;
  }
  std::lock_guard<std::mutex> locker(mutex_);
  auto iter = allocated_list_.find(ptr);
  if (iter != allocated_list_.end()) {
    auto membuf = iter->second;
    int ref_count = std::atomic_load(&membuf->ref_count_);
    return ref_count;
  }
  return -1;
}

int NNRTAllocator::SetRefCount(void *ptr, int ref_count) {
  if (ptr == nullptr) {
    return -1;
  }
  std::lock_guard<std::mutex> locker(mutex_);
  auto iter = allocated_list_.find(ptr);
  if (iter != allocated_list_.end()) {
    auto membuf = iter->second;
    std::atomic_store(&membuf->ref_count_, ref_count);
    return ref_count;
  }
  return -1;
}

int NNRTAllocator::DecRefCount(void *ptr, int ref_count) {
  if (ptr == nullptr) {
    return -1;
  }
  std::lock_guard<std::mutex> locker(mutex_);
  auto iter = allocated_list_.find(ptr);
  if (iter != allocated_list_.end()) {
    auto membuf = iter->second;
    std::atomic_fetch_sub(&membuf->ref_count_, ref_count);
    return membuf->ref_count_;
  }
  return -1;
}

int NNRTAllocator::IncRefCount(void *ptr, int ref_count) {
  if (ptr == nullptr) {
    return -1;
  }
  std::lock_guard<std::mutex> locker(mutex_);
  auto iter = allocated_list_.find(ptr);
  if (iter != allocated_list_.end()) {
    auto membuf = iter->second;
    std::atomic_fetch_add(&membuf->ref_count_, ref_count);
    return membuf->ref_count_;
  }
  return -1;
}
}  // namespace lite
}  // namespace mindspore