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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NNRT_NNRT_ALLOCATOR_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NNRT_NNRT_ALLOCATOR_H_

#include <vector>
#include <map>
#include <atomic>
#include <unordered_map>
#include <map>
#include <mutex>
#include "include/api/allocator.h"
#include "src/tensor.h"
#include "neural_network_runtime/neural_network_runtime.h"

struct OH_NN_Memory;
struct OH_NNExecutor;

namespace mindspore {
namespace lite {

class NNRTAllocator : public Allocator {
 public:
  NNRTAllocator() {}
  ~NNRTAllocator() override;
  static std::shared_ptr<NNRTAllocator> GetInstance();
  void *Malloc(size_t size) override;
  void *MallocByDesc(size_t size, const std::vector<int> &shape, const TypeId data_type, const Format format,
                     const std::string &name);
  NN_TensorDesc *CreateNNRtTensorDesc(const std::vector<int> &shape, const TypeId data_type, const Format format,
                                      const std::string &name);
  OH_NN_ReturnCode SetTensorDesc(NN_TensorDesc *tensor_desc, const std::vector<int> &shape, const TypeId data_type,
                                 const Format format, const std::string &name);
  void Free(void *ptr) override;
  int RefCount(void *ptr) override;
  int SetRefCount(void *ptr, int ref_count) override;
  int DecRefCount(void *ptr, int ref_count) override;
  int IncRefCount(void *ptr, int ref_count) override;
  NN_Tensor *GetNNTensor(void *ptr) {
    std::lock_guard<std::mutex> locker(mutex_);
    auto iter = allocated_list_.find(ptr);
    if (iter != allocated_list_.end()) {
      return iter->second->tensor_;
    }
    return nullptr;
  }
  void SetDeviceId(size_t id) { device_id_ = id; }
  void ClearFreeList();
  void FreeAllocatedTensor(void *data, lite::Tensor *tensor);
  void AddAllocatedLiteTensor(void *data, lite::Tensor *tensor) {
    if (data == nullptr) {
      return;
    }
    std::lock_guard<std::mutex> locker(mutex_);
    allocated_lite_tensors_[data] = tensor;
  }

 private:
  struct MemBuf {
    std::atomic_int ref_count_{0};
    NN_TensorDesc *tensor_desc_{nullptr};
    NN_Tensor *tensor_{nullptr};
    void *data{nullptr};
    size_t size{0};
  };

  size_t device_id_{0};
  OH_NNExecutor *executor_{nullptr};
  std::mutex mutex_;
  // <membuf->memory_->data, membuf>
  std::unordered_map<void *, MemBuf *> allocated_list_;
  std::multimap<size_t, MemBuf *> free_list_;
  std::unordered_map<void *, lite::Tensor *> allocated_lite_tensors_;
};

}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NNRT_NNRT_ALLOCATOR_H_
