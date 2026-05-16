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

#ifndef MINDSPORE_LITE_INCLUDE_REGISTRY_DEOBFPROCESSOR_H_
#define MINDSPORE_LITE_INCLUDE_REGISTRY_DEOBFPROCESSOR_H_

#include <vector>
#include <string>
#include <numeric>
#include "src/common/prim_util.h"
#include "src/common/log_adapter.h"
#include "include/model.h"
#include "schema/inner/model_generated.h"

namespace mindspore::lite {

  enum DeObfRet : uint32_t {
    kDeObfFailed = 0,        ///<Deobfuscator failed
    kNoObf = 1,               ///<The node has not been obfuscated
    kDeObfSuccess = 2,        ///<Deobfuscate success
  };

  class DeObfProcessor {
    public:
      DeObfProcessor() = default;

      bool GetModelDeObf(const void *meta_graph, Model *model);

      void DeObfuscate(Model *model);

      DeObfRet CreateDeObfNode(const schema::Primitive *&src_prim, int i, int schema__version);

      std::vector<uint32_t> all_prims_type_;
      std::vector<uint32_t> all_nodes_stat_;
      bool model_obfuscated_ = false;
      void *model_deobf = nullptr;
  };

  typedef void (*ObfCreateFunc)(Model &model);

  class MS_API DeObfRegister {
    public:
      static bool (DeObfProcessor::*GetModelDeObfReg)(const void *meta_graph, Model *model);
      static void (DeObfProcessor::*DeObfuscateReg)(Model *model);
      static DeObfRet (DeObfProcessor::*CreateDeObfNodeReg)(const schema::Primitive *&src_prim, int i, int schema__version);
      static void *deobf_handle;

      DeObfRegister() = default;
      ~DeObfRegister() = default;

      static ObfCreateFunc NewDeObfProcessor;

      static void Fail(Model &model){MS_LOG(INFO) << "DeObfuscator not registered!";}

      MS_API static void RegisterDeObfuscator(ObfCreateFunc func){
        if(func == nullptr){
          MS_LOG(WARNING) << "Register invalid deobfuscator";
          return;
        }
        NewDeObfProcessor = func;
      }
  };
}
#endif
