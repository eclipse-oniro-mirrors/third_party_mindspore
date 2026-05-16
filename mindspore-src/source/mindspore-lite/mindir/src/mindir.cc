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
#include "mindir.h"
#include "utils.h"
#include "schema/model_generated.h"
#include "mindir_memory_manager.h"
//----TODO---write an example to run MindIRMemoryManager
namespace mindspore {
namespace lite {

// ********** Activation **********
PrimitivePtr MindIR_Activation_CreatePrimitive(ActivationType activation_type, float alpha, float min_val,
                                               float max_val, bool approximate) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateActivation(fbb, static_cast<schema::ActivationType>(activation_type), alpha, min_val,
                                             max_val, approximate);
  auto prim_offset =
    schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_ACTIVATION), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}
ActivationType MindIR_Activation_GetActivationType(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    if (prim == nullptr) {
      ActivationType en = static_cast<ActivationType>(0);
      return en;
    }
    auto value = prim->value_as_Activation();
    if (value == nullptr) {
      ActivationType en = static_cast<ActivationType>(0);
      return en;
    }
    return static_cast<ActivationType>(value->activation_type());
  } else {
    ActivationType en = static_cast<ActivationType>(0);
    return en;
  }
}

void MindIR_Activation_SetActivationType(PrimitivePtr *primitive, ActivationType activation_type) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    if(prim != nullptr){
      auto value = prim->value_as_Activation();
      if (value != nullptr) {
        flatbuffers::FlatBufferBuilder fbb;
        auto ops_offset =
          schema::CreateActivation(fbb, static_cast<schema::ActivationType>(activation_type), value->alpha(),
                                  value->min_val(), value->max_val(), value->approximate());
        auto prim_offset =
          schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_ACTIVATION), ops_offset.o);
        fbb.Finish(prim_offset);
        auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
        auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
        *primitive = ret_value;
      }
    }
  }
}
float MindIR_Activation_GetAlpha(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    if(prim == nullptr){
      return .0;
    }
    auto value = prim->value_as_Activation();
    if(value == nullptr){
      return .0;
    }
    return value->alpha();
  } else {
    return .0;
  }
}

void MindIR_Activation_SetAlpha(PrimitivePtr *primitive, float alpha) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    if(prim != nullptr){
      auto value = prim->value_as_Activation();
      if (value != nullptr) {
        flatbuffers::FlatBufferBuilder fbb;
        auto ops_offset = schema::CreateActivation(fbb, static_cast<schema::ActivationType>(value->activation_type()),
                                                  alpha, value->min_val(), value->max_val(), value->approximate());
        auto prim_offset =
          schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_ACTIVATION), ops_offset.o);
        fbb.Finish(prim_offset);
        auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
        auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
        *primitive = ret_value;
      }
    }
  }
}
float MindIR_Activation_GetMinVal(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    if(prim == nullptr){
      return .0;
    }
    auto value = prim->value_as_Activation();
    if(value == nullptr){
      return .0;
    }
    return value->min_val();
  } else {
    return .0;
  }
}

void MindIR_Activation_SetMinVal(PrimitivePtr *primitive, float min_val) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    if(prim != nullptr){
      auto value = prim->value_as_Activation();
      if (value != nullptr) {
        flatbuffers::FlatBufferBuilder fbb;
        auto ops_offset = schema::CreateActivation(fbb, static_cast<schema::ActivationType>(value->activation_type()),
                                                  value->alpha(), min_val, value->max_val(), value->approximate());
        auto prim_offset =
          schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_ACTIVATION), ops_offset.o);
        fbb.Finish(prim_offset);
        auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
        auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
        *primitive = ret_value;
      }
    }
  }
}
float MindIR_Activation_GetMaxVal(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    if(prim == nullptr){
      return .0;
    }
    auto value = prim->value_as_Activation();
    if(value == nullptr){
      return .0;
    }
    return value->max_val();
  } else {
    return .0;
  }
}

void MindIR_Activation_SetMaxVal(PrimitivePtr *primitive, float max_val) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    if(prim != nullptr){
      auto value = prim->value_as_Activation();
      if (value != nullptr) {
        flatbuffers::FlatBufferBuilder fbb;
        auto ops_offset = schema::CreateActivation(fbb, static_cast<schema::ActivationType>(value->activation_type()),
                                                  value->alpha(), value->min_val(), max_val, value->approximate());
        auto prim_offset =
          schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_ACTIVATION), ops_offset.o);
        fbb.Finish(prim_offset);
        auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
        auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
        *primitive = ret_value;
      }
    }
  }
}
bool MindIR_Activation_GetApproximate(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    if(prim == nullptr){
      return false;
    }
    auto value = prim->value_as_Activation();
    if(value == nullptr){
      return false;
    }
    return value->approximate();
  } else {
    return false;
  }
}

void MindIR_Activation_SetApproximate(PrimitivePtr *primitive, bool approximate) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    if(prim != nullptr){
      auto value = prim->value_as_Activation();
      if (value != nullptr) {
        flatbuffers::FlatBufferBuilder fbb;
        auto ops_offset = schema::CreateActivation(fbb, static_cast<schema::ActivationType>(value->activation_type()),
                                                  value->alpha(), value->min_val(), value->max_val(), approximate);
        auto prim_offset =
          schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_ACTIVATION), ops_offset.o);
        fbb.Finish(prim_offset);
        auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
        auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
        *primitive = ret_value;
      }
    }
  }
}

// ********** AddFusion **********
PrimitivePtr MindIR_AddFusion_CreatePrimitive(ActivationType activation_type) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateAddFusion(fbb, static_cast<schema::ActivationType>(activation_type));
  auto prim_offset =
    schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_ADD_FUSION), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}
ActivationType MindIR_AddFusion_GetActivationType(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_AddFusion();
    if (prim != nullptr && value != nullptr) {
      return static_cast<ActivationType>(value->activation_type());
    } else {
      ActivationType en = static_cast<ActivationType>(0);
      return en;
    }
  } else {
    ActivationType en = static_cast<ActivationType>(0);
    return en;
  }
}

void MindIR_AddFusion_SetActivationType(PrimitivePtr *primitive, ActivationType activation_type) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_AddFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateAddFusion(fbb, static_cast<schema::ActivationType>(activation_type));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_ADD_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

// ********** ArgMaxFusion **********
PrimitivePtr MindIR_ArgMaxFusion_CreatePrimitive(int64_t axis, int64_t top_k, bool keep_dims, bool out_max_value) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateArgMaxFusion(fbb, axis, top_k, keep_dims, out_max_value);
  auto prim_offset =
    schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_ARGMAX_FUSION), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}
int64_t MindIR_ArgMaxFusion_GetAxis(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_ArgMaxFusion();
    if (prim != nullptr && value != nullptr) {
      return value->axis();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_ArgMaxFusion_SetAxis(PrimitivePtr *primitive, int64_t axis) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_ArgMaxFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset =
        schema::CreateArgMaxFusion(fbb, axis, value->top_k(), value->keep_dims(), value->out_max_value());
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_ARGMAX_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
int64_t MindIR_ArgMaxFusion_GetTopK(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_ArgMaxFusion();
    if (prim != nullptr && value != nullptr) {
      return value->top_k();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_ArgMaxFusion_SetTopK(PrimitivePtr *primitive, int64_t top_k) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_ArgMaxFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset =
        schema::CreateArgMaxFusion(fbb, value->axis(), top_k, value->keep_dims(), value->out_max_value());
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_ARGMAX_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
bool MindIR_ArgMaxFusion_GetKeepDims(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_ArgMaxFusion();
    if (prim != nullptr && value != nullptr) {
      return value->keep_dims();
    } else {
      return false;
    }
  } else {
    return false;
  }
}

void MindIR_ArgMaxFusion_SetKeepDims(PrimitivePtr *primitive, bool keep_dims) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_ArgMaxFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset =
        schema::CreateArgMaxFusion(fbb, value->axis(), value->top_k(), keep_dims, value->out_max_value());
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_ARGMAX_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
bool MindIR_ArgMaxFusion_GetOutMaxValue(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_ArgMaxFusion();
    if (prim != nullptr && value != nullptr) {
      return value->out_max_value();
    } else {
      return false;
    }
  } else {
    return false;
  }
}

void MindIR_ArgMaxFusion_SetOutMaxValue(PrimitivePtr *primitive, bool out_max_value) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_ArgMaxFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset =
        schema::CreateArgMaxFusion(fbb, value->axis(), value->top_k(), value->keep_dims(), out_max_value);
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_ARGMAX_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

// ********** AvgPoolFusion **********
PrimitivePtr MindIR_AvgPoolFusion_CreatePrimitive(const std::vector<int64_t> &kernel_size,
                                                  const std::vector<int64_t> &strides, const std::vector<int64_t> &pad,
                                                  PadMode pad_mode, RoundMode round_mode, Format format, bool global,
                                                  ActivationType activation_type) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateAvgPoolFusion(
    fbb, fbb.CreateVector(kernel_size.data(), kernel_size.size()), fbb.CreateVector(strides.data(), strides.size()),
    fbb.CreateVector(pad.data(), pad.size()), static_cast<schema::PadMode>(pad_mode),
    static_cast<schema::RoundMode>(round_mode), static_cast<schema::Format>(format), global,
    static_cast<schema::ActivationType>(activation_type));
  auto prim_offset =
    schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_AVG_POOL_FUSION), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}
std::vector<int64_t> MindIR_AvgPoolFusion_GetKernelSize(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_AvgPoolFusion();
    if (prim != nullptr && value != nullptr) {
      std::vector<int64_t> result;
      auto src = value->kernel_size();
      if (src == nullptr) {
        return {};
      }
      result.resize(src->size());
      std::transform(src->begin(), src->end(), result.begin(), [](int64_t item) { return item; });
      return result;
    } else {
      return {};
    }
  } else {
    return {};
  }
}

void MindIR_AvgPoolFusion_SetKernelSize(PrimitivePtr *primitive, const std::vector<int64_t> &kernel_size) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_AvgPoolFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateAvgPoolFusion(
        fbb, fbb.CreateVector(kernel_size.data(), kernel_size.size()),
        fbb.CreateVector(value->strides()->data(), value->strides()->size()),
        fbb.CreateVector(value->pad()->data(), value->pad()->size()), static_cast<schema::PadMode>(value->pad_mode()),
        static_cast<schema::RoundMode>(value->round_mode()), static_cast<schema::Format>(value->format()),
        value->global(), static_cast<schema::ActivationType>(value->activation_type()));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_AVG_POOL_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
std::vector<int64_t> MindIR_AvgPoolFusion_GetStrides(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_AvgPoolFusion();
    if (prim != nullptr && value != nullptr) {
      std::vector<int64_t> result;
      auto src = value->strides();
      if (src == nullptr) {
        return {};
      }
      result.resize(src->size());
      std::transform(src->begin(), src->end(), result.begin(), [](int64_t item) { return item; });
      return result;
    } else {
      return {};
    }
  } else {
    return {};
  }
}

void MindIR_AvgPoolFusion_SetStrides(PrimitivePtr *primitive, const std::vector<int64_t> &strides) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_AvgPoolFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateAvgPoolFusion(
        fbb, fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()),
        fbb.CreateVector(strides.data(), strides.size()), fbb.CreateVector(value->pad()->data(), value->pad()->size()),
        static_cast<schema::PadMode>(value->pad_mode()), static_cast<schema::RoundMode>(value->round_mode()),
        static_cast<schema::Format>(value->format()), value->global(),
        static_cast<schema::ActivationType>(value->activation_type()));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_AVG_POOL_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
std::vector<int64_t> MindIR_AvgPoolFusion_GetPad(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_AvgPoolFusion();
    if (prim != nullptr && value != nullptr) {
      std::vector<int64_t> result;
      auto src = value->pad();
      if (src == nullptr) {
        return {};
      }
      result.resize(src->size());
      std::transform(src->begin(), src->end(), result.begin(), [](int64_t item) { return item; });
      return result;
    } else {
      return {};
    }
  } else {
    return {};
  }
}

void MindIR_AvgPoolFusion_SetPad(PrimitivePtr *primitive, const std::vector<int64_t> &pad) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_AvgPoolFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateAvgPoolFusion(
        fbb, fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()),
        fbb.CreateVector(value->strides()->data(), value->strides()->size()), fbb.CreateVector(pad.data(), pad.size()),
        static_cast<schema::PadMode>(value->pad_mode()), static_cast<schema::RoundMode>(value->round_mode()),
        static_cast<schema::Format>(value->format()), value->global(),
        static_cast<schema::ActivationType>(value->activation_type()));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_AVG_POOL_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
PadMode MindIR_AvgPoolFusion_GetPadMode(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_AvgPoolFusion();
    if (prim != nullptr && value != nullptr) {
      return static_cast<PadMode>(value->pad_mode());
    } else {
      PadMode en = static_cast<PadMode>(0);
      return en;
    }
  } else {
    PadMode en = static_cast<PadMode>(0);
    return en;
  }
}

void MindIR_AvgPoolFusion_SetPadMode(PrimitivePtr *primitive, PadMode pad_mode) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_AvgPoolFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateAvgPoolFusion(
        fbb, fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()),
        fbb.CreateVector(value->strides()->data(), value->strides()->size()),
        fbb.CreateVector(value->pad()->data(), value->pad()->size()), static_cast<schema::PadMode>(pad_mode),
        static_cast<schema::RoundMode>(value->round_mode()), static_cast<schema::Format>(value->format()),
        value->global(), static_cast<schema::ActivationType>(value->activation_type()));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_AVG_POOL_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
RoundMode MindIR_AvgPoolFusion_GetRoundMode(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_AvgPoolFusion();
    if (prim != nullptr && value != nullptr) {
      return static_cast<RoundMode>(value->round_mode());
    } else {
      RoundMode en = static_cast<RoundMode>(0);
      return en;
    }
  } else {
    RoundMode en = static_cast<RoundMode>(0);
    return en;
  }
}

void MindIR_AvgPoolFusion_SetRoundMode(PrimitivePtr *primitive, RoundMode round_mode) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_AvgPoolFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateAvgPoolFusion(
        fbb, fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()),
        fbb.CreateVector(value->strides()->data(), value->strides()->size()),
        fbb.CreateVector(value->pad()->data(), value->pad()->size()), static_cast<schema::PadMode>(value->pad_mode()),
        static_cast<schema::RoundMode>(round_mode), static_cast<schema::Format>(value->format()), value->global(),
        static_cast<schema::ActivationType>(value->activation_type()));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_AVG_POOL_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
Format MindIR_AvgPoolFusion_GetFormat(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_AvgPoolFusion();
    if (prim != nullptr && value != nullptr) {
      return static_cast<Format>(value->format());
    } else {
      Format en = static_cast<Format>(0);
      return en;
    }
  } else {
    Format en = static_cast<Format>(0);
    return en;
  }
}

void MindIR_AvgPoolFusion_SetFormat(PrimitivePtr *primitive, Format format) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_AvgPoolFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateAvgPoolFusion(
        fbb, fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()),
        fbb.CreateVector(value->strides()->data(), value->strides()->size()),
        fbb.CreateVector(value->pad()->data(), value->pad()->size()), static_cast<schema::PadMode>(value->pad_mode()),
        static_cast<schema::RoundMode>(value->round_mode()), static_cast<schema::Format>(format), value->global(),
        static_cast<schema::ActivationType>(value->activation_type()));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_AVG_POOL_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
bool MindIR_AvgPoolFusion_GetGlobal(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_AvgPoolFusion();
    if (prim != nullptr && value != nullptr) {
      return value->global();
    } else {
      return false;
    }
  } else {
    return false;
  }
}

void MindIR_AvgPoolFusion_SetGlobal(PrimitivePtr *primitive, bool global) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_AvgPoolFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateAvgPoolFusion(
        fbb, fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()),
        fbb.CreateVector(value->strides()->data(), value->strides()->size()),
        fbb.CreateVector(value->pad()->data(), value->pad()->size()), static_cast<schema::PadMode>(value->pad_mode()),
        static_cast<schema::RoundMode>(value->round_mode()), static_cast<schema::Format>(value->format()), global,
        static_cast<schema::ActivationType>(value->activation_type()));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_AVG_POOL_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
ActivationType MindIR_AvgPoolFusion_GetActivationType(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_AvgPoolFusion();
    if (prim != nullptr && value != nullptr) {
      return static_cast<ActivationType>(value->activation_type());
    } else {
      ActivationType en = static_cast<ActivationType>(0);
      return en;
    }
  } else {
    ActivationType en = static_cast<ActivationType>(0);
    return en;
  }
}

void MindIR_AvgPoolFusion_SetActivationType(PrimitivePtr *primitive, ActivationType activation_type) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_AvgPoolFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateAvgPoolFusion(
        fbb, fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()),
        fbb.CreateVector(value->strides()->data(), value->strides()->size()),
        fbb.CreateVector(value->pad()->data(), value->pad()->size()), static_cast<schema::PadMode>(value->pad_mode()),
        static_cast<schema::RoundMode>(value->round_mode()), static_cast<schema::Format>(value->format()),
        value->global(), static_cast<schema::ActivationType>(activation_type));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_AVG_POOL_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

// ********** BatchToSpaceND **********
PrimitivePtr MindIR_BatchToSpaceND_CreatePrimitive(const std::vector<int64_t> &block_shape,
                                                   const std::vector<std::vector<int64_t>> &crops) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateBatchToSpaceND(fbb, fbb.CreateVector(block_shape.data(), block_shape.size()),
                                                 CreateVec2D(fbb, crops));
  auto prim_offset =
    schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_BATCH_TO_SPACE_ND), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}
std::vector<int64_t> MindIR_BatchToSpaceND_GetBlockShape(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_BatchToSpaceND();
    if (prim != nullptr && value != nullptr) {
      std::vector<int64_t> result;
      auto src = value->block_shape();
      if (src == nullptr) {
        return {};
      }
      result.resize(src->size());
      std::transform(src->begin(), src->end(), result.begin(), [](int64_t item) { return item; });
      return result;
    } else {
      return {};
    }
  } else {
    return {};
  }
}

void MindIR_BatchToSpaceND_SetBlockShape(PrimitivePtr *primitive, const std::vector<int64_t> &block_shape) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_BatchToSpaceND();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateBatchToSpaceND(fbb, fbb.CreateVector(block_shape.data(), block_shape.size()),
                                                     CreateVec2D(fbb, value->crops()));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_BATCH_TO_SPACE_ND), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
std::vector<std::vector<int64_t>> MindIR_BatchToSpaceND_GetCrops(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_BatchToSpaceND();
    if (prim != nullptr && value != nullptr) {
      std::vector<std::vector<int64_t>> out;
      auto src = value->crops();
      if (src == nullptr) {
        return {};
      }
      for (auto sub_list : *src->data()) {
        std::vector<int64_t> result_tmp;
        result_tmp.resize(sub_list->data()->size());
        std::transform(sub_list->data()->begin(), sub_list->data()->end(), result_tmp.begin(),
                       [](int64_t item) { return item; });
        out.emplace_back(result_tmp);
      }
      return out;
    } else {
      return {};
    }
  } else {
    return {};
  }
}

void MindIR_BatchToSpaceND_SetCrops(PrimitivePtr *primitive, const std::vector<std::vector<int64_t>> &crops) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_BatchToSpaceND();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateBatchToSpaceND(
        fbb, fbb.CreateVector(value->block_shape()->data(), value->block_shape()->size()), CreateVec2D(fbb, crops));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_BATCH_TO_SPACE_ND), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

// ********** BiasAdd **********
PrimitivePtr MindIR_BiasAdd_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateBiasAdd(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_BIAS_ADD), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** Cast **********
PrimitivePtr MindIR_Cast_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateCast(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_CAST), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** Concat **********
PrimitivePtr MindIR_Concat_CreatePrimitive(int64_t axis) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateConcat(fbb, axis);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_CONCAT), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}
int64_t MindIR_Concat_GetAxis(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Concat();
    if (prim != nullptr && value != nullptr) {
      return value->axis();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_Concat_SetAxis(PrimitivePtr *primitive, int64_t axis) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Concat();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateConcat(fbb, axis);
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_CONCAT), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

// ********** Conv2DFusion **********
PrimitivePtr MindIR_Conv2DFusion_CreatePrimitive(const std::vector<int64_t> &kernel_size,
                                                 const std::vector<int64_t> &stride,
                                                 const std::vector<int64_t> &dilation, PadMode pad_mode,
                                                 const std::vector<int64_t> &pad_list, int64_t group,
                                                 int64_t in_channel, int64_t out_channel,
                                                 ActivationType activation_type) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateConv2DFusion(
    fbb, mindspore::schema::Format_NCHW, fbb.CreateVector(kernel_size.data(), kernel_size.size()),
    fbb.CreateVector(stride.data(), stride.size()), fbb.CreateVector(dilation.data(), dilation.size()),
    static_cast<schema::PadMode>(pad_mode), fbb.CreateVector(pad_list.data(), pad_list.size()), 0, group, in_channel,
    out_channel, static_cast<schema::ActivationType>(activation_type));
  auto prim_offset =
    schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_CONV2D_FUSION), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}
std::vector<int64_t> MindIR_Conv2DFusion_GetKernelSize(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Conv2DFusion();
    if (prim != nullptr && value != nullptr) {
      std::vector<int64_t> result;
      auto src = value->kernel_size();
      if (src == nullptr) {
        return {};
      }
      result.resize(src->size());
      std::transform(src->begin(), src->end(), result.begin(), [](int64_t item) { return item; });
      return result;
    } else {
      return {};
    }
  } else {
    return {};
  }
}

void MindIR_Conv2DFusion_SetKernelSize(PrimitivePtr *primitive, const std::vector<int64_t> &kernel_size) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Conv2DFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateConv2DFusion(
        fbb, mindspore::schema::Format_NCHW, fbb.CreateVector(kernel_size.data(), kernel_size.size()),
        fbb.CreateVector(value->stride()->data(), value->stride()->size()),
        fbb.CreateVector(value->dilation()->data(), value->dilation()->size()),
        static_cast<schema::PadMode>(value->pad_mode()),
        fbb.CreateVector(value->pad_list()->data(), value->pad_list()->size()), 0, value->group(), value->in_channel(),
        value->out_channel(), static_cast<schema::ActivationType>(value->activation_type()));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_CONV2D_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
std::vector<int64_t> MindIR_Conv2DFusion_GetStride(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Conv2DFusion();
    if (prim != nullptr && value != nullptr) {
      std::vector<int64_t> result;
      auto src = value->stride();
      if (src == nullptr) {
        return {};
      }
      result.resize(src->size());
      std::transform(src->begin(), src->end(), result.begin(), [](int64_t item) { return item; });
      return result;
    } else {
      return {};
    }
  } else {
    return {};
  }
}

void MindIR_Conv2DFusion_SetStride(PrimitivePtr *primitive, const std::vector<int64_t> &stride) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Conv2DFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateConv2DFusion(
        fbb, mindspore::schema::Format_NCHW,
        fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()),
        fbb.CreateVector(stride.data(), stride.size()),
        fbb.CreateVector(value->dilation()->data(), value->dilation()->size()),
        static_cast<schema::PadMode>(value->pad_mode()),
        fbb.CreateVector(value->pad_list()->data(), value->pad_list()->size()), 0, value->group(), value->in_channel(),
        value->out_channel(), static_cast<schema::ActivationType>(value->activation_type()));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_CONV2D_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
std::vector<int64_t> MindIR_Conv2DFusion_GetDilation(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Conv2DFusion();
    if (prim != nullptr && value != nullptr) {
      std::vector<int64_t> result;
      auto src = value->dilation();
      if (src == nullptr) {
        return {};
      }
      result.resize(src->size());
      std::transform(src->begin(), src->end(), result.begin(), [](int64_t item) { return item; });
      return result;
    } else {
      return {};
    }
  } else {
    return {};
  }
}

void MindIR_Conv2DFusion_SetDilation(PrimitivePtr *primitive, const std::vector<int64_t> &dilation) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Conv2DFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateConv2DFusion(
        fbb, mindspore::schema::Format_NCHW,
        fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()),
        fbb.CreateVector(value->stride()->data(), value->stride()->size()),
        fbb.CreateVector(dilation.data(), dilation.size()), static_cast<schema::PadMode>(value->pad_mode()),
        fbb.CreateVector(value->pad_list()->data(), value->pad_list()->size()), 0, value->group(), value->in_channel(),
        value->out_channel(), static_cast<schema::ActivationType>(value->activation_type()));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_CONV2D_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
PadMode MindIR_Conv2DFusion_GetPadMode(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Conv2DFusion();
    if (prim != nullptr && value != nullptr) {
      return static_cast<PadMode>(value->pad_mode());
    } else {
      PadMode en = static_cast<PadMode>(0);
      return en;
    }
  } else {
    PadMode en = static_cast<PadMode>(0);
    return en;
  }
}

void MindIR_Conv2DFusion_SetPadMode(PrimitivePtr *primitive, PadMode pad_mode) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Conv2DFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateConv2DFusion(
        fbb, mindspore::schema::Format_NCHW,
        fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()),
        fbb.CreateVector(value->stride()->data(), value->stride()->size()),
        fbb.CreateVector(value->dilation()->data(), value->dilation()->size()), static_cast<schema::PadMode>(pad_mode),
        fbb.CreateVector(value->pad_list()->data(), value->pad_list()->size()), 0, value->group(), value->in_channel(),
        value->out_channel(), static_cast<schema::ActivationType>(value->activation_type()));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_CONV2D_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
std::vector<int64_t> MindIR_Conv2DFusion_GetPadList(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Conv2DFusion();
    if (prim != nullptr && value != nullptr) {
      std::vector<int64_t> result;
      auto src = value->pad_list();
      if (src == nullptr) {
        return {};
      }
      result.resize(src->size());
      std::transform(src->begin(), src->end(), result.begin(), [](int64_t item) { return item; });
      return result;
    } else {
      return {};
    }
  } else {
    return {};
  }
}

void MindIR_Conv2DFusion_SetPadList(PrimitivePtr *primitive, const std::vector<int64_t> &pad_list) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Conv2DFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateConv2DFusion(
        fbb, mindspore::schema::Format_NCHW,
        fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()),
        fbb.CreateVector(value->stride()->data(), value->stride()->size()),
        fbb.CreateVector(value->dilation()->data(), value->dilation()->size()),
        static_cast<schema::PadMode>(value->pad_mode()), fbb.CreateVector(pad_list.data(), pad_list.size()), 0,
        value->group(), value->in_channel(), value->out_channel(),
        static_cast<schema::ActivationType>(value->activation_type()));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_CONV2D_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
int64_t MindIR_Conv2DFusion_GetGroup(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Conv2DFusion();
    if (prim != nullptr && value != nullptr) {
      return value->group();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_Conv2DFusion_SetGroup(PrimitivePtr *primitive, int64_t group) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Conv2DFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateConv2DFusion(
        fbb, mindspore::schema::Format_NCHW,
        fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()),
        fbb.CreateVector(value->stride()->data(), value->stride()->size()),
        fbb.CreateVector(value->dilation()->data(), value->dilation()->size()),
        static_cast<schema::PadMode>(value->pad_mode()),
        fbb.CreateVector(value->pad_list()->data(), value->pad_list()->size()), 0, group, value->in_channel(),
        value->out_channel(), static_cast<schema::ActivationType>(value->activation_type()));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_CONV2D_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
int64_t MindIR_Conv2DFusion_GetInChannel(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Conv2DFusion();
    if (prim != nullptr && value != nullptr) {
      return value->in_channel();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_Conv2DFusion_SetInChannel(PrimitivePtr *primitive, int64_t in_channel) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Conv2DFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateConv2DFusion(
        fbb, mindspore::schema::Format_NCHW,
        fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()),
        fbb.CreateVector(value->stride()->data(), value->stride()->size()),
        fbb.CreateVector(value->dilation()->data(), value->dilation()->size()),
        static_cast<schema::PadMode>(value->pad_mode()),
        fbb.CreateVector(value->pad_list()->data(), value->pad_list()->size()), 0, value->group(), in_channel,
        value->out_channel(), static_cast<schema::ActivationType>(value->activation_type()));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_CONV2D_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
int64_t MindIR_Conv2DFusion_GetOutChannel(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Conv2DFusion();
    if (prim != nullptr && value != nullptr) {
      return value->out_channel();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_Conv2DFusion_SetOutChannel(PrimitivePtr *primitive, int64_t out_channel) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Conv2DFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateConv2DFusion(
        fbb, mindspore::schema::Format_NCHW,
        fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()),
        fbb.CreateVector(value->stride()->data(), value->stride()->size()),
        fbb.CreateVector(value->dilation()->data(), value->dilation()->size()),
        static_cast<schema::PadMode>(value->pad_mode()),
        fbb.CreateVector(value->pad_list()->data(), value->pad_list()->size()), 0, value->group(), value->in_channel(),
        out_channel, static_cast<schema::ActivationType>(value->activation_type()));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_CONV2D_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
ActivationType MindIR_Conv2DFusion_GetActivationType(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Conv2DFusion();
    if (prim != nullptr && value != nullptr) {
      return static_cast<ActivationType>(value->activation_type());
    } else {
      ActivationType en = static_cast<ActivationType>(0);
      return en;
    }
  } else {
    ActivationType en = static_cast<ActivationType>(0);
    return en;
  }
}

void MindIR_Conv2DFusion_SetActivationType(PrimitivePtr *primitive, ActivationType activation_type) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Conv2DFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateConv2DFusion(
        fbb, mindspore::schema::Format_NCHW,
        fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()),
        fbb.CreateVector(value->stride()->data(), value->stride()->size()),
        fbb.CreateVector(value->dilation()->data(), value->dilation()->size()),
        static_cast<schema::PadMode>(value->pad_mode()),
        fbb.CreateVector(value->pad_list()->data(), value->pad_list()->size()), 0, value->group(), value->in_channel(),
        value->out_channel(), static_cast<schema::ActivationType>(activation_type));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_CONV2D_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

Format MindIR_Conv2DFusion_GetFormat(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Conv2DFusion();
    if (prim != nullptr && value != nullptr) {
      return static_cast<Format>(value->format());
    } else {
      Format en = static_cast<Format>(0);
      return en;
    }
  } else {
    Format en = static_cast<Format>(0);
    return en;
  }
}

void MindIR_Conv2DFusion_SetFormat(PrimitivePtr *primitive, Format format) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Conv2DFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateConv2DFusion(
        fbb, static_cast<schema::Format>(format),
        fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()),
        fbb.CreateVector(value->stride()->data(), value->stride()->size()),
        fbb.CreateVector(value->dilation()->data(), value->dilation()->size()),
        static_cast<schema::PadMode>(value->pad_mode()),
        fbb.CreateVector(value->pad_list()->data(), value->pad_list()->size()), 0, value->group(), value->in_channel(),
        value->out_channel(), static_cast<schema::ActivationType>(value->activation_type()));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_CONV2D_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

// ********** Conv2dTransposeFusion **********
PrimitivePtr MindIR_Conv2dTransposeFusion_CreatePrimitive(
  const std::vector<int64_t> &kernel_size, const std::vector<int64_t> &stride, const std::vector<int64_t> &dilation,
  PadMode pad_mode, const std::vector<int64_t> &pad_list, int64_t group, int64_t in_channel, int64_t out_channel,
  ActivationType activation_type, const std::vector<int64_t> &output_paddings) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateConv2dTransposeFusion(
    fbb, mindspore::schema::Format_NCHW, fbb.CreateVector(kernel_size.data(), kernel_size.size()),
    fbb.CreateVector(stride.data(), stride.size()), fbb.CreateVector(dilation.data(), dilation.size()),
    static_cast<schema::PadMode>(pad_mode), 0, fbb.CreateVector(pad_list.data(), pad_list.size()), 0, group, in_channel,
    out_channel, static_cast<schema::ActivationType>(activation_type),
    fbb.CreateVector(output_paddings.data(), output_paddings.size()));
  auto prim_offset =
    schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_CONV2D_TRANSPOSE_FUSION), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}
std::vector<int64_t> MindIR_Conv2dTransposeFusion_GetKernelSize(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Conv2dTransposeFusion();
    if (prim != nullptr && value != nullptr) {
      std::vector<int64_t> result;
      auto src = value->kernel_size();
      if (src == nullptr) {
        return {};
      }
      result.resize(src->size());
      std::transform(src->begin(), src->end(), result.begin(), [](int64_t item) { return item; });
      return result;
    } else {
      return {};
    }
  } else {
    return {};
  }
}

void MindIR_Conv2dTransposeFusion_SetKernelSize(PrimitivePtr *primitive, const std::vector<int64_t> &kernel_size) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Conv2dTransposeFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateConv2dTransposeFusion(
        fbb, mindspore::schema::Format_NCHW, fbb.CreateVector(kernel_size.data(), kernel_size.size()),
        fbb.CreateVector(value->stride()->data(), value->stride()->size()),
        fbb.CreateVector(value->dilation()->data(), value->dilation()->size()),
        static_cast<schema::PadMode>(value->pad_mode()), 0,
        fbb.CreateVector(value->pad_list()->data(), value->pad_list()->size()), 0, value->group(), value->in_channel(),
        value->out_channel(), static_cast<schema::ActivationType>(value->activation_type()),
        fbb.CreateVector(value->output_paddings()->data(), value->output_paddings()->size()));
      auto prim_offset = schema::CreatePrimitive(
        fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_CONV2D_TRANSPOSE_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
std::vector<int64_t> MindIR_Conv2dTransposeFusion_GetStride(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Conv2dTransposeFusion();
    if (prim != nullptr && value != nullptr) {
      std::vector<int64_t> result;
      auto src = value->stride();
      if (src == nullptr) {
        return {};
      }
      result.resize(src->size());
      std::transform(src->begin(), src->end(), result.begin(), [](int64_t item) { return item; });
      return result;
    } else {
      return {};
    }
  } else {
    return {};
  }
}

void MindIR_Conv2dTransposeFusion_SetStride(PrimitivePtr *primitive, const std::vector<int64_t> &stride) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Conv2dTransposeFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateConv2dTransposeFusion(
        fbb, mindspore::schema::Format_NCHW,
        fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()),
        fbb.CreateVector(stride.data(), stride.size()),
        fbb.CreateVector(value->dilation()->data(), value->dilation()->size()),
        static_cast<schema::PadMode>(value->pad_mode()), 0,
        fbb.CreateVector(value->pad_list()->data(), value->pad_list()->size()), 0, value->group(), value->in_channel(),
        value->out_channel(), static_cast<schema::ActivationType>(value->activation_type()),
        fbb.CreateVector(value->output_paddings()->data(), value->output_paddings()->size()));
      auto prim_offset = schema::CreatePrimitive(
        fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_CONV2D_TRANSPOSE_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
std::vector<int64_t> MindIR_Conv2dTransposeFusion_GetDilation(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Conv2dTransposeFusion();
    if (prim != nullptr && value != nullptr) {
      std::vector<int64_t> result;
      auto src = value->dilation();
      if (src == nullptr) {
        return {};
      }
      result.resize(src->size());
      std::transform(src->begin(), src->end(), result.begin(), [](int64_t item) { return item; });
      return result;
    } else {
      return {};
    }
  } else {
    return {};
  }
}

void MindIR_Conv2dTransposeFusion_SetDilation(PrimitivePtr *primitive, const std::vector<int64_t> &dilation) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Conv2dTransposeFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateConv2dTransposeFusion(
        fbb, mindspore::schema::Format_NCHW,
        fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()),
        fbb.CreateVector(value->stride()->data(), value->stride()->size()),
        fbb.CreateVector(dilation.data(), dilation.size()), static_cast<schema::PadMode>(value->pad_mode()), 0,
        fbb.CreateVector(value->pad_list()->data(), value->pad_list()->size()), 0, value->group(), value->in_channel(),
        value->out_channel(), static_cast<schema::ActivationType>(value->activation_type()),
        fbb.CreateVector(value->output_paddings()->data(), value->output_paddings()->size()));
      auto prim_offset = schema::CreatePrimitive(
        fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_CONV2D_TRANSPOSE_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
PadMode MindIR_Conv2dTransposeFusion_GetPadMode(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Conv2dTransposeFusion();
    if (prim != nullptr && value != nullptr) {
      return static_cast<PadMode>(value->pad_mode());
    } else {
      PadMode en = static_cast<PadMode>(0);
      return en;
    }
  } else {
    PadMode en = static_cast<PadMode>(0);
    return en;
  }
}

void MindIR_Conv2dTransposeFusion_SetPadMode(PrimitivePtr *primitive, PadMode pad_mode) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Conv2dTransposeFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateConv2dTransposeFusion(
        fbb, mindspore::schema::Format_NCHW,
        fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()),
        fbb.CreateVector(value->stride()->data(), value->stride()->size()),
        fbb.CreateVector(value->dilation()->data(), value->dilation()->size()), static_cast<schema::PadMode>(pad_mode),
        0, fbb.CreateVector(value->pad_list()->data(), value->pad_list()->size()), 0, value->group(),
        value->in_channel(), value->out_channel(), static_cast<schema::ActivationType>(value->activation_type()),
        fbb.CreateVector(value->output_paddings()->data(), value->output_paddings()->size()));
      auto prim_offset = schema::CreatePrimitive(
        fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_CONV2D_TRANSPOSE_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
std::vector<int64_t> MindIR_Conv2dTransposeFusion_GetPadList(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Conv2dTransposeFusion();
    if (prim != nullptr && value != nullptr) {
      std::vector<int64_t> result;
      auto src = value->pad_list();
      if (src == nullptr) {
        return {};
      }
      result.resize(src->size());
      std::transform(src->begin(), src->end(), result.begin(), [](int64_t item) { return item; });
      return result;
    } else {
      return {};
    }
  } else {
    return {};
  }
}

void MindIR_Conv2dTransposeFusion_SetPadList(PrimitivePtr *primitive, const std::vector<int64_t> &pad_list) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Conv2dTransposeFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateConv2dTransposeFusion(
        fbb, mindspore::schema::Format_NCHW,
        fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()),
        fbb.CreateVector(value->stride()->data(), value->stride()->size()),
        fbb.CreateVector(value->dilation()->data(), value->dilation()->size()),
        static_cast<schema::PadMode>(value->pad_mode()), 0, fbb.CreateVector(pad_list.data(), pad_list.size()), 0,
        value->group(), value->in_channel(), value->out_channel(),
        static_cast<schema::ActivationType>(value->activation_type()),
        fbb.CreateVector(value->output_paddings()->data(), value->output_paddings()->size()));
      auto prim_offset = schema::CreatePrimitive(
        fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_CONV2D_TRANSPOSE_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
int64_t MindIR_Conv2dTransposeFusion_GetGroup(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Conv2dTransposeFusion();
    if (prim != nullptr && value != nullptr) {
      return value->group();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_Conv2dTransposeFusion_SetGroup(PrimitivePtr *primitive, int64_t group) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Conv2dTransposeFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateConv2dTransposeFusion(
        fbb, mindspore::schema::Format_NCHW,
        fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()),
        fbb.CreateVector(value->stride()->data(), value->stride()->size()),
        fbb.CreateVector(value->dilation()->data(), value->dilation()->size()),
        static_cast<schema::PadMode>(value->pad_mode()), 0,
        fbb.CreateVector(value->pad_list()->data(), value->pad_list()->size()), 0, group, value->in_channel(),
        value->out_channel(), static_cast<schema::ActivationType>(value->activation_type()),
        fbb.CreateVector(value->output_paddings()->data(), value->output_paddings()->size()));
      auto prim_offset = schema::CreatePrimitive(
        fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_CONV2D_TRANSPOSE_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
int64_t MindIR_Conv2dTransposeFusion_GetInChannel(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Conv2dTransposeFusion();
    if (prim != nullptr && value != nullptr) {
      return value->in_channel();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_Conv2dTransposeFusion_SetInChannel(PrimitivePtr *primitive, int64_t in_channel) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Conv2dTransposeFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateConv2dTransposeFusion(
        fbb, mindspore::schema::Format_NCHW,
        fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()),
        fbb.CreateVector(value->stride()->data(), value->stride()->size()),
        fbb.CreateVector(value->dilation()->data(), value->dilation()->size()),
        static_cast<schema::PadMode>(value->pad_mode()), 0,
        fbb.CreateVector(value->pad_list()->data(), value->pad_list()->size()), 0, value->group(), in_channel,
        value->out_channel(), static_cast<schema::ActivationType>(value->activation_type()),
        fbb.CreateVector(value->output_paddings()->data(), value->output_paddings()->size()));
      auto prim_offset = schema::CreatePrimitive(
        fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_CONV2D_TRANSPOSE_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
int64_t MindIR_Conv2dTransposeFusion_GetOutChannel(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Conv2dTransposeFusion();
    if (prim != nullptr && value != nullptr) {
      return value->out_channel();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_Conv2dTransposeFusion_SetOutChannel(PrimitivePtr *primitive, int64_t out_channel) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Conv2dTransposeFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateConv2dTransposeFusion(
        fbb, mindspore::schema::Format_NCHW,
        fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()),
        fbb.CreateVector(value->stride()->data(), value->stride()->size()),
        fbb.CreateVector(value->dilation()->data(), value->dilation()->size()),
        static_cast<schema::PadMode>(value->pad_mode()), 0,
        fbb.CreateVector(value->pad_list()->data(), value->pad_list()->size()), 0, value->group(), value->in_channel(),
        out_channel, static_cast<schema::ActivationType>(value->activation_type()),
        fbb.CreateVector(value->output_paddings()->data(), value->output_paddings()->size()));
      auto prim_offset = schema::CreatePrimitive(
        fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_CONV2D_TRANSPOSE_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
ActivationType MindIR_Conv2dTransposeFusion_GetActivationType(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Conv2dTransposeFusion();
    if (prim != nullptr && value != nullptr) {
      return static_cast<ActivationType>(value->activation_type());
    } else {
      ActivationType en = static_cast<ActivationType>(0);
      return en;
    }
  } else {
    ActivationType en = static_cast<ActivationType>(0);
    return en;
  }
}

void MindIR_Conv2dTransposeFusion_SetActivationType(PrimitivePtr *primitive, ActivationType activation_type) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Conv2dTransposeFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateConv2dTransposeFusion(
        fbb, mindspore::schema::Format_NCHW,
        fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()),
        fbb.CreateVector(value->stride()->data(), value->stride()->size()),
        fbb.CreateVector(value->dilation()->data(), value->dilation()->size()),
        static_cast<schema::PadMode>(value->pad_mode()), 0,
        fbb.CreateVector(value->pad_list()->data(), value->pad_list()->size()), 0, value->group(), value->in_channel(),
        value->out_channel(), static_cast<schema::ActivationType>(activation_type),
        fbb.CreateVector(value->output_paddings()->data(), value->output_paddings()->size()));
      auto prim_offset = schema::CreatePrimitive(
        fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_CONV2D_TRANSPOSE_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
std::vector<int64_t> MindIR_Conv2dTransposeFusion_GetOutputPaddings(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Conv2dTransposeFusion();
    if (prim != nullptr && value != nullptr) {
      std::vector<int64_t> result;
      auto src = value->output_paddings();
      if (src == nullptr) {
        return {};
      }
      result.resize(src->size());
      std::transform(src->begin(), src->end(), result.begin(), [](int64_t item) { return item; });
      return result;
    } else {
      return {};
    }
  } else {
    return {};
  }
}

void MindIR_Conv2dTransposeFusion_SetOutputPaddings(PrimitivePtr *primitive,
                                                    const std::vector<int64_t> &output_paddings) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Conv2dTransposeFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateConv2dTransposeFusion(
        fbb, mindspore::schema::Format_NCHW,
        fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()),
        fbb.CreateVector(value->stride()->data(), value->stride()->size()),
        fbb.CreateVector(value->dilation()->data(), value->dilation()->size()),
        static_cast<schema::PadMode>(value->pad_mode()), 0,
        fbb.CreateVector(value->pad_list()->data(), value->pad_list()->size()), 0, value->group(), value->in_channel(),
        value->out_channel(), static_cast<schema::ActivationType>(value->activation_type()),
        fbb.CreateVector(output_paddings.data(), output_paddings.size()));
      auto prim_offset = schema::CreatePrimitive(
        fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_CONV2D_TRANSPOSE_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

// ********** DivFusion **********
PrimitivePtr MindIR_DivFusion_CreatePrimitive(ActivationType activation_type) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateDivFusion(fbb, static_cast<schema::ActivationType>(activation_type));
  auto prim_offset =
    schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_DIV_FUSION), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}
ActivationType MindIR_DivFusion_GetActivationType(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_DivFusion();
    if (prim != nullptr && value != nullptr) {
      return static_cast<ActivationType>(value->activation_type());
    } else {
      ActivationType en = static_cast<ActivationType>(0);
      return en;
    }
  } else {
    ActivationType en = static_cast<ActivationType>(0);
    return en;
  }
}

void MindIR_DivFusion_SetActivationType(PrimitivePtr *primitive, ActivationType activation_type) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_DivFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateDivFusion(fbb, static_cast<schema::ActivationType>(activation_type));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_DIV_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

// ********** Eltwise **********
PrimitivePtr MindIR_Eltwise_CreatePrimitive(EltwiseMode mode) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateEltwise(fbb, static_cast<schema::EltwiseMode>(mode));
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_ELTWISE), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}
EltwiseMode MindIR_Eltwise_GetMode(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Eltwise();
    if (prim != nullptr && value != nullptr) {
      return static_cast<EltwiseMode>(value->mode());
    } else {
      EltwiseMode en = static_cast<EltwiseMode>(0);
      return en;
    }
  } else {
    EltwiseMode en = static_cast<EltwiseMode>(0);
    return en;
  }
}

void MindIR_Eltwise_SetMode(PrimitivePtr *primitive, EltwiseMode mode) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Eltwise();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateEltwise(fbb, static_cast<schema::EltwiseMode>(mode));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_ELTWISE), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

// ********** ExpandDims **********
PrimitivePtr MindIR_ExpandDims_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateExpandDims(fbb);
  auto prim_offset =
    schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_EXPAND_DIMS), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** Fill **********
PrimitivePtr MindIR_Fill_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateFill(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_FILL), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** FullConnection **********
PrimitivePtr MindIR_FullConnection_CreatePrimitive(bool has_bias, bool use_axis, int64_t axis,
                                                   ActivationType activation_type) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset =
    schema::CreateFullConnection(fbb, has_bias, use_axis, axis, static_cast<schema::ActivationType>(activation_type));
  auto prim_offset =
    schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_FULL_CONNECTION), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}
bool MindIR_FullConnection_GetHasBias(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_FullConnection();
    if (prim != nullptr && value != nullptr) {
      return value->has_bias();
    } else {
      return false;
    }
  } else {
    return false;
  }
}

void MindIR_FullConnection_SetHasBias(PrimitivePtr *primitive, bool has_bias) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_FullConnection();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateFullConnection(fbb, has_bias, value->use_axis(), value->axis(),
                                                     static_cast<schema::ActivationType>(value->activation_type()));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_FULL_CONNECTION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
bool MindIR_FullConnection_GetUseAxis(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_FullConnection();
    if (prim != nullptr && value != nullptr) {
      return value->use_axis();
    } else {
      return false;
    }
  } else {
    return false;
  }
}

void MindIR_FullConnection_SetUseAxis(PrimitivePtr *primitive, bool use_axis) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_FullConnection();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateFullConnection(fbb, value->has_bias(), use_axis, value->axis(),
                                                     static_cast<schema::ActivationType>(value->activation_type()));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_FULL_CONNECTION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
int64_t MindIR_FullConnection_GetAxis(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_FullConnection();
    if (prim != nullptr && value != nullptr) {
      return value->axis();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_FullConnection_SetAxis(PrimitivePtr *primitive, int64_t axis) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_FullConnection();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateFullConnection(fbb, value->has_bias(), value->use_axis(), axis,
                                                     static_cast<schema::ActivationType>(value->activation_type()));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_FULL_CONNECTION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
ActivationType MindIR_FullConnection_GetActivationType(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_FullConnection();
    if (prim != nullptr && value != nullptr) {
      return static_cast<ActivationType>(value->activation_type());
    } else {
      ActivationType en = static_cast<ActivationType>(0);
      return en;
    }
  } else {
    ActivationType en = static_cast<ActivationType>(0);
    return en;
  }
}

void MindIR_FullConnection_SetActivationType(PrimitivePtr *primitive, ActivationType activation_type) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_FullConnection();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateFullConnection(fbb, value->has_bias(), value->use_axis(), value->axis(),
                                                     static_cast<schema::ActivationType>(activation_type));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_FULL_CONNECTION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

// ********** FusedBatchNorm **********
PrimitivePtr MindIR_FusedBatchNorm_CreatePrimitive(float epsilon) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateFusedBatchNorm(fbb, 0.9, 0);
  auto prim_offset =
    schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_FUSED_BATCH_NORM), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}
float MindIR_FusedBatchNorm_GetEpsilon(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_FusedBatchNorm();
    if (prim != nullptr && value != nullptr) {
      return value->epsilon();
    } else {
      return .0;
    }
  } else {
    return .0;
  }
}

void MindIR_FusedBatchNorm_SetEpsilon(PrimitivePtr *primitive, float epsilon) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_FusedBatchNorm();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateFusedBatchNorm(fbb, epsilon, 0.9, 0);
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_FUSED_BATCH_NORM), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

// ********** Gather **********
PrimitivePtr MindIR_Gather_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateGather(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_GATHER), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** LayerNormFusion **********
PrimitivePtr MindIR_LayerNormFusion_CreatePrimitive(int64_t begin_norm_axis, float epsilon, bool elementwise_affine,
                                                    int64_t begin_params_axis) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateLayerNormFusion(fbb, begin_norm_axis, epsilon, elementwise_affine, begin_params_axis);
  auto prim_offset =
    schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_LAYER_NORM_FUSION), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}
int64_t MindIR_LayerNormFusion_GetBeginNormAxis(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_LayerNormFusion();
    if (prim != nullptr && value != nullptr) {
      return value->begin_norm_axis();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_LayerNormFusion_SetBeginNormAxis(PrimitivePtr *primitive, int64_t begin_norm_axis) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_LayerNormFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateLayerNormFusion(fbb, begin_norm_axis, value->epsilon(),
                                                      value->elementwise_affine(), value->begin_params_axis());
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_LAYER_NORM_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
float MindIR_LayerNormFusion_GetEpsilon(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_LayerNormFusion();
    if (prim != nullptr && value != nullptr) {
      return value->epsilon();
    } else {
      return .0;
    }
  } else {
    return .0;
  }
}

void MindIR_LayerNormFusion_SetEpsilon(PrimitivePtr *primitive, float epsilon) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_LayerNormFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateLayerNormFusion(fbb, value->begin_norm_axis(), epsilon,
                                                      value->elementwise_affine(), value->begin_params_axis());
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_LAYER_NORM_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
bool MindIR_LayerNormFusion_GetElementwiseAffine(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_LayerNormFusion();
    if (prim != nullptr && value != nullptr) {
      return value->elementwise_affine();
    } else {
      return false;
    }
  } else {
    return false;
  }
}

void MindIR_LayerNormFusion_SetElementwiseAffine(PrimitivePtr *primitive, bool elementwise_affine) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_LayerNormFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateLayerNormFusion(fbb, value->begin_norm_axis(), value->epsilon(),
                                                      elementwise_affine, value->begin_params_axis());
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_LAYER_NORM_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
int64_t MindIR_LayerNormFusion_GetBeginParamsAxis(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_LayerNormFusion();
    if (prim != nullptr && value != nullptr) {
      return value->begin_params_axis();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_LayerNormFusion_SetBeginParamsAxis(PrimitivePtr *primitive, int64_t begin_params_axis) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_LayerNormFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateLayerNormFusion(fbb, value->begin_norm_axis(), value->epsilon(),
                                                      value->elementwise_affine(), begin_params_axis);
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_LAYER_NORM_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

// ********** LessEqual **********
PrimitivePtr MindIR_LessEqual_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateLessEqual(fbb);
  auto prim_offset =
    schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_LESS_EQUAL), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** MatMulFusion **********
PrimitivePtr MindIR_MatMulFusion_CreatePrimitive(bool transpose_a, bool transpose_b, ActivationType activation_type) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset =
    schema::CreateMatMulFusion(fbb, transpose_a, transpose_b, static_cast<schema::ActivationType>(activation_type));
  auto prim_offset =
    schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_MATMUL_FUSION), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}
bool MindIR_MatMulFusion_GetTransposeA(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_MatMulFusion();
    if (prim != nullptr && value != nullptr) {
      return value->transpose_a();
    } else {
      return false;
    }
  } else {
    return false;
  }
}

void MindIR_MatMulFusion_SetTransposeA(PrimitivePtr *primitive, bool transpose_a) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_MatMulFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateMatMulFusion(fbb, transpose_a, value->transpose_b(),
                                                   static_cast<schema::ActivationType>(value->activation_type()));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_MATMUL_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
bool MindIR_MatMulFusion_GetTransposeB(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_MatMulFusion();
    if (prim != nullptr && value != nullptr) {
      return value->transpose_b();
    } else {
      return false;
    }
  } else {
    return false;
  }
}

void MindIR_MatMulFusion_SetTransposeB(PrimitivePtr *primitive, bool transpose_b) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_MatMulFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateMatMulFusion(fbb, value->transpose_a(), transpose_b,
                                                   static_cast<schema::ActivationType>(value->activation_type()));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_MATMUL_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
ActivationType MindIR_MatMulFusion_GetActivationType(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_MatMulFusion();
    if (prim != nullptr && value != nullptr) {
      return static_cast<ActivationType>(value->activation_type());
    } else {
      ActivationType en = static_cast<ActivationType>(0);
      return en;
    }
  } else {
    ActivationType en = static_cast<ActivationType>(0);
    return en;
  }
}

void MindIR_MatMulFusion_SetActivationType(PrimitivePtr *primitive, ActivationType activation_type) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_MatMulFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateMatMulFusion(fbb, value->transpose_a(), value->transpose_b(),
                                                   static_cast<schema::ActivationType>(activation_type));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_MATMUL_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

// ********** Maximum **********
PrimitivePtr MindIR_Maximum_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateMaximum(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_MAXIMUM), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** MaxPoolFusion **********
PrimitivePtr MindIR_MaxPoolFusion_CreatePrimitive(const std::vector<int64_t> &kernel_size,
                                                  const std::vector<int64_t> &strides, const std::vector<int64_t> &pad,
                                                  PadMode pad_mode, Format format, bool global,
                                                  ActivationType activation_type) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateMaxPoolFusion(
    fbb, fbb.CreateVector(kernel_size.data(), kernel_size.size()), fbb.CreateVector(strides.data(), strides.size()),
    fbb.CreateVector(pad.data(), pad.size()), static_cast<schema::PadMode>(pad_mode),
    mindspore::schema::RoundMode_FLOOR, static_cast<schema::Format>(format), global,
    static_cast<schema::ActivationType>(activation_type));
  auto prim_offset =
    schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_MAX_POOL_FUSION), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}
std::vector<int64_t> MindIR_MaxPoolFusion_GetKernelSize(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_MaxPoolFusion();
    if (prim != nullptr && value != nullptr) {
      std::vector<int64_t> result;
      auto src = value->kernel_size();
      if (src == nullptr) {
        return {};
      }
      result.resize(src->size());
      std::transform(src->begin(), src->end(), result.begin(), [](int64_t item) { return item; });
      return result;
    } else {
      return {};
    }
  } else {
    return {};
  }
}

void MindIR_MaxPoolFusion_SetKernelSize(PrimitivePtr *primitive, const std::vector<int64_t> &kernel_size) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_MaxPoolFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateMaxPoolFusion(
        fbb, fbb.CreateVector(kernel_size.data(), kernel_size.size()),
        fbb.CreateVector(value->strides()->data(), value->strides()->size()),
        fbb.CreateVector(value->pad()->data(), value->pad()->size()), static_cast<schema::PadMode>(value->pad_mode()),
        mindspore::schema::RoundMode_FLOOR, static_cast<schema::Format>(value->format()), value->global(),
        static_cast<schema::ActivationType>(value->activation_type()));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_MAX_POOL_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
std::vector<int64_t> MindIR_MaxPoolFusion_GetStrides(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_MaxPoolFusion();
    if (prim != nullptr && value != nullptr) {
      std::vector<int64_t> result;
      auto src = value->strides();
      if (src == nullptr) {
        return {};
      }
      result.resize(src->size());
      std::transform(src->begin(), src->end(), result.begin(), [](int64_t item) { return item; });
      return result;
    } else {
      return {};
    }
  } else {
    return {};
  }
}

void MindIR_MaxPoolFusion_SetStrides(PrimitivePtr *primitive, const std::vector<int64_t> &strides) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_MaxPoolFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateMaxPoolFusion(
        fbb, fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()),
        fbb.CreateVector(strides.data(), strides.size()), fbb.CreateVector(value->pad()->data(), value->pad()->size()),
        static_cast<schema::PadMode>(value->pad_mode()), mindspore::schema::RoundMode_FLOOR,
        static_cast<schema::Format>(value->format()), value->global(),
        static_cast<schema::ActivationType>(value->activation_type()));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_MAX_POOL_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
std::vector<int64_t> MindIR_MaxPoolFusion_GetPad(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_MaxPoolFusion();
    if (prim != nullptr && value != nullptr) {
      std::vector<int64_t> result;
      auto src = value->pad();
      if (src == nullptr) {
        return {};
      }
      result.resize(src->size());
      std::transform(src->begin(), src->end(), result.begin(), [](int64_t item) { return item; });
      return result;
    } else {
      return {};
    }
  } else {
    return {};
  }
}

void MindIR_MaxPoolFusion_SetPad(PrimitivePtr *primitive, const std::vector<int64_t> &pad) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_MaxPoolFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateMaxPoolFusion(
        fbb, fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()),
        fbb.CreateVector(value->strides()->data(), value->strides()->size()), fbb.CreateVector(pad.data(), pad.size()),
        static_cast<schema::PadMode>(value->pad_mode()), mindspore::schema::RoundMode_FLOOR,
        static_cast<schema::Format>(value->format()), value->global(),
        static_cast<schema::ActivationType>(value->activation_type()));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_MAX_POOL_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
PadMode MindIR_MaxPoolFusion_GetPadMode(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_MaxPoolFusion();
    if (prim != nullptr && value != nullptr) {
      return static_cast<PadMode>(value->pad_mode());
    } else {
      PadMode en = static_cast<PadMode>(0);
      return en;
    }
  } else {
    PadMode en = static_cast<PadMode>(0);
    return en;
  }
}

void MindIR_MaxPoolFusion_SetPadMode(PrimitivePtr *primitive, PadMode pad_mode) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_MaxPoolFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateMaxPoolFusion(
        fbb, fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()),
        fbb.CreateVector(value->strides()->data(), value->strides()->size()),
        fbb.CreateVector(value->pad()->data(), value->pad()->size()), static_cast<schema::PadMode>(pad_mode),
        mindspore::schema::RoundMode_FLOOR, static_cast<schema::Format>(value->format()), value->global(),
        static_cast<schema::ActivationType>(value->activation_type()));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_MAX_POOL_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
Format MindIR_MaxPoolFusion_GetFormat(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_MaxPoolFusion();
    if (prim != nullptr && value != nullptr) {
      return static_cast<Format>(value->format());
    } else {
      Format en = static_cast<Format>(0);
      return en;
    }
  } else {
    Format en = static_cast<Format>(0);
    return en;
  }
}

void MindIR_MaxPoolFusion_SetFormat(PrimitivePtr *primitive, Format format) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_MaxPoolFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateMaxPoolFusion(
        fbb, fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()),
        fbb.CreateVector(value->strides()->data(), value->strides()->size()),
        fbb.CreateVector(value->pad()->data(), value->pad()->size()), static_cast<schema::PadMode>(value->pad_mode()),
        mindspore::schema::RoundMode_FLOOR, static_cast<schema::Format>(format), value->global(),
        static_cast<schema::ActivationType>(value->activation_type()));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_MAX_POOL_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
bool MindIR_MaxPoolFusion_GetGlobal(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_MaxPoolFusion();
    if (prim != nullptr && value != nullptr) {
      return value->global();
    } else {
      return false;
    }
  } else {
    return false;
  }
}

void MindIR_MaxPoolFusion_SetGlobal(PrimitivePtr *primitive, bool global) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_MaxPoolFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateMaxPoolFusion(
        fbb, fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()),
        fbb.CreateVector(value->strides()->data(), value->strides()->size()),
        fbb.CreateVector(value->pad()->data(), value->pad()->size()), static_cast<schema::PadMode>(value->pad_mode()),
        mindspore::schema::RoundMode_FLOOR, static_cast<schema::Format>(value->format()), global,
        static_cast<schema::ActivationType>(value->activation_type()));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_MAX_POOL_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

RoundMode MindIR_MaxPoolFusion_GetRoundMode(ConstPrimitivePtr primitive) {
  RoundMode round_mode = static_cast<RoundMode>(0); // set default value: RoundMode_FLOOR

  if (primitive == nullptr) {
    return round_mode;
  }
  auto prim = static_cast<const schema::Primitive *>(primitive);

  auto value = prim->value_as_MaxPoolFusion();
  if (value == nullptr) {
    return round_mode;
  }
  round_mode = static_cast<RoundMode>(value->round_mode());
  return round_mode;
}

void MindIR_MaxPoolFusion_SetRoundMode(PrimitivePtr *primitive, RoundMode round_mode) {
  if (primitive == nullptr) {
    return;
  }
  auto prim = static_cast<schema::Primitive *>(*primitive);
  if (prim == nullptr) {
    return;
  }
  auto value = prim->value_as_MaxPoolFusion();
  if (value == nullptr) {
    return;
  }
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateMaxPoolFusion(
      fbb, fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()),
      fbb.CreateVector(value->strides()->data(), value->strides()->size()),
      fbb.CreateVector(value->pad()->data(), value->pad()->size()), static_cast<schema::PadMode>(value->pad_mode()),
      static_cast<schema::RoundMode>(round_mode), static_cast<schema::Format>(value->format()), value->global(),
      static_cast<schema::ActivationType>(value->activation_type()));
  auto prim_offset =
      schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_MAX_POOL_FUSION), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
  auto new_prim = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  *primitive = new_prim;
}

ActivationType MindIR_MaxPoolFusion_GetActivationType(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_MaxPoolFusion();
    if (prim != nullptr && value != nullptr) {
      return static_cast<ActivationType>(value->activation_type());
    } else {
      ActivationType en = static_cast<ActivationType>(0);
      return en;
    }
  } else {
    ActivationType en = static_cast<ActivationType>(0);
    return en;
  }
}

void MindIR_MaxPoolFusion_SetActivationType(PrimitivePtr *primitive, ActivationType activation_type) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_MaxPoolFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateMaxPoolFusion(
        fbb, fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()),
        fbb.CreateVector(value->strides()->data(), value->strides()->size()),
        fbb.CreateVector(value->pad()->data(), value->pad()->size()), static_cast<schema::PadMode>(value->pad_mode()),
        mindspore::schema::RoundMode_FLOOR, static_cast<schema::Format>(value->format()), value->global(),
        static_cast<schema::ActivationType>(activation_type));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_MAX_POOL_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

// ********** MulFusion **********
PrimitivePtr MindIR_MulFusion_CreatePrimitive(ActivationType activation_type) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateMulFusion(fbb, static_cast<schema::ActivationType>(activation_type));
  auto prim_offset =
    schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_MUL_FUSION), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

ActivationType MindIR_MulFusion_GetActivationType(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_MulFusion();
    if (prim != nullptr && value != nullptr) {
      return static_cast<ActivationType>(value->activation_type());
    } else {
      ActivationType en = static_cast<ActivationType>(0);
      return en;
    }
  } else {
    ActivationType en = static_cast<ActivationType>(0);
    return en;
  }
}

void MindIR_MulFusion_SetActivationType(PrimitivePtr *primitive, ActivationType activation_type) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_MulFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateMulFusion(fbb, static_cast<schema::ActivationType>(activation_type));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_MUL_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

// ********** OneHot **********
PrimitivePtr MindIR_OneHot_CreatePrimitive(int64_t axis) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateOneHot(fbb, axis);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_ONE_HOT), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}
int64_t MindIR_OneHot_GetAxis(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_OneHot();
    if (prim != nullptr && value != nullptr) {
      return value->axis();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_OneHot_SetAxis(PrimitivePtr *primitive, int64_t axis) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_OneHot();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateOneHot(fbb, axis);
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_ONE_HOT), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

// ********** PadFusion **********
PrimitivePtr MindIR_PadFusion_CreatePrimitive(const std::vector<std::vector<int64_t>> &paddings,
                                              PaddingMode padding_mode, float constant_value) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreatePadFusion(fbb, CreateVec2D(fbb, paddings),
                                            static_cast<schema::PaddingMode>(padding_mode), constant_value);
  auto prim_offset =
    schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_PAD_FUSION), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}
std::vector<std::vector<int64_t>> MindIR_PadFusion_GetPaddings(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_PadFusion();
    if (prim != nullptr && value != nullptr) {
      std::vector<std::vector<int64_t>> out;
      auto src = value->paddings();
      if (src == nullptr) {
        return {};
      }
      for (auto sub_list : *src->data()) {
        std::vector<int64_t> result_tmp;
        result_tmp.resize(sub_list->data()->size());
        std::transform(sub_list->data()->begin(), sub_list->data()->end(), result_tmp.begin(),
                       [](int64_t item) { return item; });
        out.emplace_back(result_tmp);
      }
      return out;
    } else {
      return {};
    }
  } else {
    return {};
  }
}

void MindIR_PadFusion_SetPaddings(PrimitivePtr *primitive, const std::vector<std::vector<int64_t>> &paddings) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_PadFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset =
        schema::CreatePadFusion(fbb, CreateVec2D(fbb, paddings),
                                static_cast<schema::PaddingMode>(value->padding_mode()), value->constant_value());
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_PAD_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
PaddingMode MindIR_PadFusion_GetPaddingMode(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_PadFusion();
    if (prim != nullptr && value != nullptr) {
      return static_cast<PaddingMode>(value->padding_mode());
    } else {
      PaddingMode en = static_cast<PaddingMode>(0);
      return en;
    }
  } else {
    PaddingMode en = static_cast<PaddingMode>(0);
    return en;
  }
}

void MindIR_PadFusion_SetPaddingMode(PrimitivePtr *primitive, PaddingMode padding_mode) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_PadFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset =
        schema::CreatePadFusion(fbb, CreateVec2D(fbb, value->paddings()),
                                static_cast<schema::PaddingMode>(padding_mode), value->constant_value());
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_PAD_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
float MindIR_PadFusion_GetConstantValue(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_PadFusion();
    if (prim != nullptr && value != nullptr) {
      return value->constant_value();
    } else {
      return .0;
    }
  } else {
    return .0;
  }
}

void MindIR_PadFusion_SetConstantValue(PrimitivePtr *primitive, float constant_value) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_PadFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset =
        schema::CreatePadFusion(fbb, CreateVec2D(fbb, value->paddings()),
                                static_cast<schema::PaddingMode>(value->padding_mode()), constant_value);
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_PAD_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

// ********** PowFusion **********
PrimitivePtr MindIR_PowFusion_CreatePrimitive(float scale, float shift) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreatePowFusion(fbb, scale, shift);
  auto prim_offset =
    schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_POW_FUSION), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}
float MindIR_PowFusion_GetScale(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_PowFusion();
    if (prim != nullptr && value != nullptr) {
      return value->scale();
    } else {
      return .0;
    }
  } else {
    return .0;
  }
}

void MindIR_PowFusion_SetScale(PrimitivePtr *primitive, float scale) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_PowFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreatePowFusion(fbb, scale, value->shift());
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_POW_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
float MindIR_PowFusion_GetShift(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_PowFusion();
    if (prim != nullptr && value != nullptr) {
      return value->shift();
    } else {
      return .0;
    }
  } else {
    return .0;
  }
}

void MindIR_PowFusion_SetShift(PrimitivePtr *primitive, float shift) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_PowFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreatePowFusion(fbb, value->scale(), shift);
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_POW_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

// ********** PReLUFusion **********
PrimitivePtr MindIR_PReLUFusion_CreatePrimitive(bool channel_shared) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreatePReLUFusion(fbb, channel_shared);
  auto prim_offset =
    schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_PRELU_FUSION), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}
bool MindIR_PReLUFusion_GetChannelShared(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_PReLUFusion();
    if (prim != nullptr && value != nullptr) {
      return value->channel_shared();
    } else {
      return false;
    }
  } else {
    return false;
  }
}

void MindIR_PReLUFusion_SetChannelShared(PrimitivePtr *primitive, bool channel_shared) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_PReLUFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreatePReLUFusion(fbb, channel_shared);
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_PRELU_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

// ********** QuantDTypeCast **********
PrimitivePtr MindIR_QuantDTypeCast_CreatePrimitive(int64_t src_t, int64_t dst_t, int64_t axis) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateQuantDTypeCast(fbb, src_t, dst_t, axis);
  auto prim_offset =
    schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_QUANT_DTYPE_CAST), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}
int64_t MindIR_QuantDTypeCast_GetSrcT(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_QuantDTypeCast();
    if (prim != nullptr && value != nullptr) {
      return value->src_t();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_QuantDTypeCast_SetSrcT(PrimitivePtr *primitive, int64_t src_t) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_QuantDTypeCast();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateQuantDTypeCast(fbb, src_t, value->dst_t(), value->axis());
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_QUANT_DTYPE_CAST), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
int64_t MindIR_QuantDTypeCast_GetDstT(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_QuantDTypeCast();
    if (prim != nullptr && value != nullptr) {
      return value->dst_t();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_QuantDTypeCast_SetDstT(PrimitivePtr *primitive, int64_t dst_t) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_QuantDTypeCast();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateQuantDTypeCast(fbb, value->src_t(), dst_t, value->axis());
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_QUANT_DTYPE_CAST), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

int64_t MindIR_QuantDTypeCast_GetAxis(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_QuantDTypeCast();
    if (prim != nullptr && value != nullptr) {
      return value->axis();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_QuantDTypeCast_SetAxis(PrimitivePtr *primitive, int64_t axis) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_QuantDTypeCast();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateQuantDTypeCast(fbb, value->src_t(), value->dst_t(), axis);
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_QUANT_DTYPE_CAST), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

// ********** ReduceFusion **********
PrimitivePtr MindIR_ReduceFusion_CreatePrimitive(bool keep_dims, ReduceMode mode, bool reduce_to_end, float coeff) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset =
    schema::CreateReduceFusion(fbb, keep_dims, static_cast<schema::ReduceMode>(mode), reduce_to_end, coeff);
  auto prim_offset =
    schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_REDUCE_FUSION), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}
bool MindIR_ReduceFusion_GetKeepDims(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_ReduceFusion();
    if (prim != nullptr && value != nullptr) {
      return value->keep_dims();
    } else {
      return false;
    }
  } else {
    return false;
  }
}

void MindIR_ReduceFusion_SetKeepDims(PrimitivePtr *primitive, bool keep_dims) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_ReduceFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateReduceFusion(fbb, keep_dims, static_cast<schema::ReduceMode>(value->mode()),
                                                   value->reduce_to_end(), value->coeff());
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_REDUCE_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
ReduceMode MindIR_ReduceFusion_GetMode(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_ReduceFusion();
    if (prim != nullptr && value != nullptr) {
      return static_cast<ReduceMode>(value->mode());
    } else {
      ReduceMode en = static_cast<ReduceMode>(0);
      return en;
    }
  } else {
    ReduceMode en = static_cast<ReduceMode>(0);
    return en;
  }
}

void MindIR_ReduceFusion_SetMode(PrimitivePtr *primitive, ReduceMode mode) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_ReduceFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateReduceFusion(fbb, value->keep_dims(), static_cast<schema::ReduceMode>(mode),
                                                   value->reduce_to_end(), value->coeff());
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_REDUCE_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
bool MindIR_ReduceFusion_GetReduceToEnd(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_ReduceFusion();
    if (prim != nullptr && value != nullptr) {
      return value->reduce_to_end();
    } else {
      return false;
    }
  } else {
    return false;
  }
}

void MindIR_ReduceFusion_SetReduceToEnd(PrimitivePtr *primitive, bool reduce_to_end) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_ReduceFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateReduceFusion(
        fbb, value->keep_dims(), static_cast<schema::ReduceMode>(value->mode()), reduce_to_end, value->coeff());
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_REDUCE_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
float MindIR_ReduceFusion_GetCoeff(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_ReduceFusion();
    if (prim != nullptr && value != nullptr) {
      return value->coeff();
    } else {
      return .0;
    }
  } else {
    return .0;
  }
}

void MindIR_ReduceFusion_SetCoeff(PrimitivePtr *primitive, float coeff) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_ReduceFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateReduceFusion(
        fbb, value->keep_dims(), static_cast<schema::ReduceMode>(value->mode()), value->reduce_to_end(), coeff);
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_REDUCE_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

// ********** Reshape **********
PrimitivePtr MindIR_Reshape_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateReshape(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_RESHAPE), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** Resize **********
PrimitivePtr MindIR_Resize_CreatePrimitive(ResizeMethod method, int64_t new_height, int64_t new_width,
                                           bool preserve_aspect_ratio,
                                           CoordinateTransformMode coordinate_transform_mode, float cubic_coeff,
                                           int64_t exclude_outside, float extrapolation_value,
                                           NearestMode nearest_mode) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateResize(
    fbb, mindspore::schema::Format_NCHW, static_cast<schema::ResizeMethod>(method), new_height, new_width,
    preserve_aspect_ratio, static_cast<schema::CoordinateTransformMode>(coordinate_transform_mode), cubic_coeff,
    exclude_outside, extrapolation_value, static_cast<schema::NearestMode>(nearest_mode));
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_RESIZE), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}
ResizeMethod MindIR_Resize_GetMethod(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Resize();
    if (prim != nullptr && value != nullptr) {
      return static_cast<ResizeMethod>(value->method());
    } else {
      ResizeMethod en = static_cast<ResizeMethod>(0);
      return en;
    }
  } else {
    ResizeMethod en = static_cast<ResizeMethod>(0);
    return en;
  }
}

void MindIR_Resize_SetMethod(PrimitivePtr *primitive, ResizeMethod method) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Resize();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset =
        schema::CreateResize(fbb, mindspore::schema::Format_NCHW, static_cast<schema::ResizeMethod>(method),
                             value->new_height(), value->new_width(), value->preserve_aspect_ratio(),
                             static_cast<schema::CoordinateTransformMode>(value->coordinate_transform_mode()),
                             value->cubic_coeff(), value->exclude_outside(), value->extrapolation_value(),
                             static_cast<schema::NearestMode>(value->nearest_mode()));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_RESIZE), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
int64_t MindIR_Resize_GetNewHeight(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Resize();
    if (prim != nullptr && value != nullptr) {
      return value->new_height();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_Resize_SetNewHeight(PrimitivePtr *primitive, int64_t new_height) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Resize();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset =
        schema::CreateResize(fbb, mindspore::schema::Format_NCHW, static_cast<schema::ResizeMethod>(value->method()),
                             new_height, value->new_width(), value->preserve_aspect_ratio(),
                             static_cast<schema::CoordinateTransformMode>(value->coordinate_transform_mode()),
                             value->cubic_coeff(), value->exclude_outside(), value->extrapolation_value(),
                             static_cast<schema::NearestMode>(value->nearest_mode()));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_RESIZE), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
int64_t MindIR_Resize_GetNewWidth(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Resize();
    if (prim != nullptr && value != nullptr) {
      return value->new_width();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_Resize_SetNewWidth(PrimitivePtr *primitive, int64_t new_width) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Resize();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset =
        schema::CreateResize(fbb, mindspore::schema::Format_NCHW, static_cast<schema::ResizeMethod>(value->method()),
                             value->new_height(), new_width, value->preserve_aspect_ratio(),
                             static_cast<schema::CoordinateTransformMode>(value->coordinate_transform_mode()),
                             value->cubic_coeff(), value->exclude_outside(), value->extrapolation_value(),
                             static_cast<schema::NearestMode>(value->nearest_mode()));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_RESIZE), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
bool MindIR_Resize_GetPreserveAspectRatio(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Resize();
    if (prim != nullptr && value != nullptr) {
      return value->preserve_aspect_ratio();
    } else {
      return false;
    }
  } else {
    return false;
  }
}

void MindIR_Resize_SetPreserveAspectRatio(PrimitivePtr *primitive, bool preserve_aspect_ratio) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Resize();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset =
        schema::CreateResize(fbb, mindspore::schema::Format_NCHW, static_cast<schema::ResizeMethod>(value->method()),
                             value->new_height(), value->new_width(), preserve_aspect_ratio,
                             static_cast<schema::CoordinateTransformMode>(value->coordinate_transform_mode()),
                             value->cubic_coeff(), value->exclude_outside(), value->extrapolation_value(),
                             static_cast<schema::NearestMode>(value->nearest_mode()));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_RESIZE), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
CoordinateTransformMode MindIR_Resize_GetCoordinateTransformMode(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Resize();
    if (prim != nullptr && value != nullptr) {
      return static_cast<CoordinateTransformMode>(value->coordinate_transform_mode());
    } else {
      CoordinateTransformMode en = static_cast<CoordinateTransformMode>(0);
      return en;
    }
  } else {
    CoordinateTransformMode en = static_cast<CoordinateTransformMode>(0);
    return en;
  }
}

void MindIR_Resize_SetCoordinateTransformMode(PrimitivePtr *primitive,
                                              CoordinateTransformMode coordinate_transform_mode) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Resize();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset =
        schema::CreateResize(fbb, mindspore::schema::Format_NCHW, static_cast<schema::ResizeMethod>(value->method()),
                             value->new_height(), value->new_width(), value->preserve_aspect_ratio(),
                             static_cast<schema::CoordinateTransformMode>(coordinate_transform_mode),
                             value->cubic_coeff(), value->exclude_outside(), value->extrapolation_value(),
                             static_cast<schema::NearestMode>(value->nearest_mode()));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_RESIZE), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
float MindIR_Resize_GetCubicCoeff(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Resize();
    if (prim != nullptr && value != nullptr) {
      return value->cubic_coeff();
    } else {
      return .0;
    }
  } else {
    return .0;
  }
}

void MindIR_Resize_SetCubicCoeff(PrimitivePtr *primitive, float cubic_coeff) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Resize();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset =
        schema::CreateResize(fbb, mindspore::schema::Format_NCHW, static_cast<schema::ResizeMethod>(value->method()),
                             value->new_height(), value->new_width(), value->preserve_aspect_ratio(),
                             static_cast<schema::CoordinateTransformMode>(value->coordinate_transform_mode()),
                             cubic_coeff, value->exclude_outside(), value->extrapolation_value(),
                             static_cast<schema::NearestMode>(value->nearest_mode()));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_RESIZE), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
int64_t MindIR_Resize_GetExcludeOutside(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Resize();
    if (prim != nullptr && value != nullptr) {
      return value->exclude_outside();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_Resize_SetExcludeOutside(PrimitivePtr *primitive, int64_t exclude_outside) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Resize();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateResize(
        fbb, mindspore::schema::Format_NCHW, static_cast<schema::ResizeMethod>(value->method()), value->new_height(),
        value->new_width(), value->preserve_aspect_ratio(),
        static_cast<schema::CoordinateTransformMode>(value->coordinate_transform_mode()), value->cubic_coeff(),
        exclude_outside, value->extrapolation_value(), static_cast<schema::NearestMode>(value->nearest_mode()));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_RESIZE), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
float MindIR_Resize_GetExtrapolationValue(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Resize();
    if (prim != nullptr && value != nullptr) {
      return value->extrapolation_value();
    } else {
      return .0;
    }
  } else {
    return .0;
  }
}

void MindIR_Resize_SetExtrapolationValue(PrimitivePtr *primitive, float extrapolation_value) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Resize();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateResize(
        fbb, mindspore::schema::Format_NCHW, static_cast<schema::ResizeMethod>(value->method()), value->new_height(),
        value->new_width(), value->preserve_aspect_ratio(),
        static_cast<schema::CoordinateTransformMode>(value->coordinate_transform_mode()), value->cubic_coeff(),
        value->exclude_outside(), extrapolation_value, static_cast<schema::NearestMode>(value->nearest_mode()));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_RESIZE), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
NearestMode MindIR_Resize_GetNearestMode(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Resize();
    if (prim != nullptr && value != nullptr) {
      return static_cast<NearestMode>(value->nearest_mode());
    } else {
      NearestMode en = static_cast<NearestMode>(0);
      return en;
    }
  } else {
    NearestMode en = static_cast<NearestMode>(0);
    return en;
  }
}

void MindIR_Resize_SetNearestMode(PrimitivePtr *primitive, NearestMode nearest_mode) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Resize();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateResize(
        fbb, mindspore::schema::Format_NCHW, static_cast<schema::ResizeMethod>(value->method()), value->new_height(),
        value->new_width(), value->preserve_aspect_ratio(),
        static_cast<schema::CoordinateTransformMode>(value->coordinate_transform_mode()), value->cubic_coeff(),
        value->exclude_outside(), value->extrapolation_value(), static_cast<schema::NearestMode>(nearest_mode));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_RESIZE), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

// ********** Rsqrt **********
PrimitivePtr MindIR_Rsqrt_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateRsqrt(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_RSQRT), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** ScaleFusion **********
PrimitivePtr MindIR_ScaleFusion_CreatePrimitive(int64_t axis, ActivationType activation_type) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateScaleFusion(fbb, axis, static_cast<schema::ActivationType>(activation_type));
  auto prim_offset =
    schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_SCALE_FUSION), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}
int64_t MindIR_ScaleFusion_GetAxis(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_ScaleFusion();
    if (prim != nullptr && value != nullptr) {
      return value->axis();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_ScaleFusion_SetAxis(PrimitivePtr *primitive, int64_t axis) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_ScaleFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset =
        schema::CreateScaleFusion(fbb, axis, static_cast<schema::ActivationType>(value->activation_type()));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_SCALE_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
ActivationType MindIR_ScaleFusion_GetActivationType(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_ScaleFusion();
    if (prim != nullptr && value != nullptr) {
      return static_cast<ActivationType>(value->activation_type());
    } else {
      ActivationType en = static_cast<ActivationType>(0);
      return en;
    }
  } else {
    ActivationType en = static_cast<ActivationType>(0);
    return en;
  }
}

void MindIR_ScaleFusion_SetActivationType(PrimitivePtr *primitive, ActivationType activation_type) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_ScaleFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset =
        schema::CreateScaleFusion(fbb, value->axis(), static_cast<schema::ActivationType>(activation_type));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_SCALE_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

// ********** Shape **********
PrimitivePtr MindIR_Shape_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateShape(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_SHAPE), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** SliceFusion **********
PrimitivePtr MindIR_SliceFusion_CreatePrimitive(const std::vector<int64_t> &axes) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateSliceFusion(fbb, fbb.CreateVector(axes.data(), axes.size()));
  auto prim_offset =
    schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_SLICE_FUSION), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}
std::vector<int64_t> MindIR_SliceFusion_GetAxes(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_SliceFusion();
    if (prim != nullptr && value != nullptr) {
      std::vector<int64_t> result;
      auto src = value->axes();
      if (src == nullptr) {
        return {};
      }
      result.resize(src->size());
      std::transform(src->begin(), src->end(), result.begin(), [](int64_t item) { return item; });
      return result;
    } else {
      return {};
    }
  } else {
    return {};
  }
}

void MindIR_SliceFusion_SetAxes(PrimitivePtr *primitive, const std::vector<int64_t> &axes) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_SliceFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateSliceFusion(fbb, fbb.CreateVector(axes.data(), axes.size()));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_SLICE_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

// ********** Softmax **********
PrimitivePtr MindIR_Softmax_CreatePrimitive(const std::vector<int64_t> &axis) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateSoftmax(fbb, fbb.CreateVector(axis.data(), axis.size()));
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_SOFTMAX), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}
std::vector<int64_t> MindIR_Softmax_GetAxis(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Softmax();
    if (prim != nullptr && value != nullptr) {
      std::vector<int64_t> result;
      auto src = value->axis();
      if (src == nullptr) {
        return {};
      }
      result.resize(src->size());
      std::transform(src->begin(), src->end(), result.begin(), [](int64_t item) { return item; });
      return result;
    } else {
      return {};
    }
  } else {
    return {};
  }
}

void MindIR_Softmax_SetAxis(PrimitivePtr *primitive, const std::vector<int64_t> &axis) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Softmax();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateSoftmax(fbb, fbb.CreateVector(axis.data(), axis.size()));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_SOFTMAX), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

// ********** SpaceToBatchND **********
PrimitivePtr MindIR_SpaceToBatchND_CreatePrimitive(const std::vector<int64_t> &block_shape,
                                                   const std::vector<std::vector<int64_t>> &paddings) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateSpaceToBatchND(fbb, fbb.CreateVector(block_shape.data(), block_shape.size()),
                                                 CreateVec2D(fbb, paddings));
  auto prim_offset =
    schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_SPACE_TO_BATCH_ND), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}
std::vector<int64_t> MindIR_SpaceToBatchND_GetBlockShape(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_SpaceToBatchND();
    if (prim != nullptr && value != nullptr) {
      std::vector<int64_t> result;
      auto src = value->block_shape();
      if (src == nullptr) {
        return {};
      }
      result.resize(src->size());
      std::transform(src->begin(), src->end(), result.begin(), [](int64_t item) { return item; });
      return result;
    } else {
      return {};
    }
  } else {
    return {};
  }
}

void MindIR_SpaceToBatchND_SetBlockShape(PrimitivePtr *primitive, const std::vector<int64_t> &block_shape) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_SpaceToBatchND();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateSpaceToBatchND(fbb, fbb.CreateVector(block_shape.data(), block_shape.size()),
                                                     CreateVec2D(fbb, value->paddings()));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_SPACE_TO_BATCH_ND), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
std::vector<std::vector<int64_t>> MindIR_SpaceToBatchND_GetPaddings(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_SpaceToBatchND();
    if (prim != nullptr && value != nullptr) {
      std::vector<std::vector<int64_t>> out;
      auto src = value->paddings();
      if (src == nullptr) {
        return {};
      }
      for (auto sub_list : *src->data()) {
        std::vector<int64_t> result_tmp;
        result_tmp.resize(sub_list->data()->size());
        std::transform(sub_list->data()->begin(), sub_list->data()->end(), result_tmp.begin(),
                       [](int64_t item) { return item; });
        out.emplace_back(result_tmp);
      }
      return out;
    } else {
      return {};
    }
  } else {
    return {};
  }
}

void MindIR_SpaceToBatchND_SetPaddings(PrimitivePtr *primitive, const std::vector<std::vector<int64_t>> &paddings) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_SpaceToBatchND();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateSpaceToBatchND(
        fbb, fbb.CreateVector(value->block_shape()->data(), value->block_shape()->size()), CreateVec2D(fbb, paddings));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_SPACE_TO_BATCH_ND), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

// ********** Split **********
PrimitivePtr MindIR_Split_CreatePrimitive(int64_t output_num, const std::vector<int64_t> &size_splits, int64_t axis) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset =
    schema::CreateSplit(fbb, output_num, fbb.CreateVector(size_splits.data(), size_splits.size()), axis);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_SPLIT), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}
int64_t MindIR_Split_GetOutputNum(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Split();
    if (prim != nullptr && value != nullptr) {
      return value->output_num();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_Split_SetOutputNum(PrimitivePtr *primitive, int64_t output_num) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Split();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateSplit(
        fbb, output_num, fbb.CreateVector(value->size_splits()->data(), value->size_splits()->size()), value->axis());
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_SPLIT), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
std::vector<int64_t> MindIR_Split_GetSizeSplits(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Split();
    if (prim != nullptr && value != nullptr) {
      std::vector<int64_t> result;
      auto src = value->size_splits();
      if (src == nullptr) {
        return {};
      }
      result.resize(src->size());
      std::transform(src->begin(), src->end(), result.begin(), [](int64_t item) { return item; });
      return result;
    } else {
      return {};
    }
  } else {
    return {};
  }
}

void MindIR_Split_SetSizeSplits(PrimitivePtr *primitive, const std::vector<int64_t> &size_splits) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Split();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateSplit(fbb, value->output_num(),
                                            fbb.CreateVector(size_splits.data(), size_splits.size()), value->axis());
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_SPLIT), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
int64_t MindIR_Split_GetAxis(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Split();
    if (prim != nullptr && value != nullptr) {
      return value->axis();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_Split_SetAxis(PrimitivePtr *primitive, int64_t axis) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Split();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateSplit(
        fbb, value->output_num(), fbb.CreateVector(value->size_splits()->data(), value->size_splits()->size()), axis);
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_SPLIT), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

// ********** Sqrt **********
PrimitivePtr MindIR_Sqrt_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateSqrt(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_SQRT), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** SquaredDifference **********
PrimitivePtr MindIR_SquaredDifference_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateSquaredDifference(fbb);
  auto prim_offset =
    schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_SQUARED_DIFFERENCE), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** Squeeze **********
PrimitivePtr MindIR_Squeeze_CreatePrimitive(const std::vector<int64_t> &axis) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateSqueeze(fbb, fbb.CreateVector(axis.data(), axis.size()));
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_SQUEEZE), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}
std::vector<int64_t> MindIR_Squeeze_GetAxis(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Squeeze();
    if (prim != nullptr && value != nullptr) {
      std::vector<int64_t> result;
      auto src = value->axis();
      if (src == nullptr) {
        return {};
      }
      result.resize(src->size());
      std::transform(src->begin(), src->end(), result.begin(), [](int64_t item) { return item; });
      return result;
    } else {
      return {};
    }
  } else {
    return {};
  }
}

void MindIR_Squeeze_SetAxis(PrimitivePtr *primitive, const std::vector<int64_t> &axis) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Squeeze();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateSqueeze(fbb, fbb.CreateVector(axis.data(), axis.size()));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_SQUEEZE), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

// ********** Stack **********
PrimitivePtr MindIR_Stack_CreatePrimitive(int64_t axis) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateStack(fbb, axis);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_STACK), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}
int64_t MindIR_Stack_GetAxis(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Stack();
    if (prim != nullptr && value != nullptr) {
      return value->axis();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_Stack_SetAxis(PrimitivePtr *primitive, int64_t axis) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Stack();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateStack(fbb, axis);
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_STACK), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

// ********** StridedSlice **********
PrimitivePtr MindIR_StridedSlice_CreatePrimitive(int64_t begin_mask, int64_t end_mask, int64_t ellipsis_mask,
                                                 int64_t new_axis_mask, int64_t shrink_axis_mask) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset =
    schema::CreateStridedSlice(fbb, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask);
  auto prim_offset =
    schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_STRIDED_SLICE), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}
int64_t MindIR_StridedSlice_GetBeginMask(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_StridedSlice();
    if (prim != nullptr && value != nullptr) {
      return value->begin_mask();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_StridedSlice_SetBeginMask(PrimitivePtr *primitive, int64_t begin_mask) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_StridedSlice();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateStridedSlice(fbb, begin_mask, value->end_mask(), value->ellipsis_mask(),
                                                   value->new_axis_mask(), value->shrink_axis_mask());
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_STRIDED_SLICE), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
int64_t MindIR_StridedSlice_GetEndMask(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_StridedSlice();
    if (prim != nullptr && value != nullptr) {
      return value->end_mask();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_StridedSlice_SetEndMask(PrimitivePtr *primitive, int64_t end_mask) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_StridedSlice();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateStridedSlice(fbb, value->begin_mask(), end_mask, value->ellipsis_mask(),
                                                   value->new_axis_mask(), value->shrink_axis_mask());
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_STRIDED_SLICE), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
int64_t MindIR_StridedSlice_GetEllipsisMask(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_StridedSlice();
    if (prim != nullptr && value != nullptr) {
      return value->ellipsis_mask();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_StridedSlice_SetEllipsisMask(PrimitivePtr *primitive, int64_t ellipsis_mask) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_StridedSlice();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateStridedSlice(fbb, value->begin_mask(), value->end_mask(), ellipsis_mask,
                                                   value->new_axis_mask(), value->shrink_axis_mask());
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_STRIDED_SLICE), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
int64_t MindIR_StridedSlice_GetNewAxisMask(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_StridedSlice();
    if (prim != nullptr && value != nullptr) {
      return value->new_axis_mask();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_StridedSlice_SetNewAxisMask(PrimitivePtr *primitive, int64_t new_axis_mask) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_StridedSlice();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateStridedSlice(fbb, value->begin_mask(), value->end_mask(), value->ellipsis_mask(),
                                                   new_axis_mask, value->shrink_axis_mask());
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_STRIDED_SLICE), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
int64_t MindIR_StridedSlice_GetShrinkAxisMask(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_StridedSlice();
    if (prim != nullptr && value != nullptr) {
      return value->shrink_axis_mask();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_StridedSlice_SetShrinkAxisMask(PrimitivePtr *primitive, int64_t shrink_axis_mask) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_StridedSlice();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateStridedSlice(fbb, value->begin_mask(), value->end_mask(), value->ellipsis_mask(),
                                                   value->new_axis_mask(), shrink_axis_mask);
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_STRIDED_SLICE), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

// ********** SubFusion **********
PrimitivePtr MindIR_SubFusion_CreatePrimitive(ActivationType activation_type) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateSubFusion(fbb, static_cast<schema::ActivationType>(activation_type));
  auto prim_offset =
    schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_SUB_FUSION), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}
ActivationType MindIR_SubFusion_GetActivationType(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_SubFusion();
    if (prim != nullptr && value != nullptr) {
      return static_cast<ActivationType>(value->activation_type());
    } else {
      ActivationType en = static_cast<ActivationType>(0);
      return en;
    }
  } else {
    ActivationType en = static_cast<ActivationType>(0);
    return en;
  }
}

void MindIR_SubFusion_SetActivationType(PrimitivePtr *primitive, ActivationType activation_type) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_SubFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateSubFusion(fbb, static_cast<schema::ActivationType>(activation_type));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_SUB_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

// ********** TileFusion **********
PrimitivePtr MindIR_TileFusion_CreatePrimitive(const std::vector<int64_t> &dims) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateTileFusion(fbb, fbb.CreateVector(dims.data(), dims.size()));
  auto prim_offset =
    schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_TILE_FUSION), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}
std::vector<int64_t> MindIR_TileFusion_GetDims(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_TileFusion();
    if (prim != nullptr && value != nullptr) {
      std::vector<int64_t> result;
      auto src = value->dims();
      if (src == nullptr) {
        return {};
      }
      result.resize(src->size());
      std::transform(src->begin(), src->end(), result.begin(), [](int64_t item) { return item; });
      return result;
    } else {
      return {};
    }
  } else {
    return {};
  }
}

void MindIR_TileFusion_SetDims(PrimitivePtr *primitive, const std::vector<int64_t> &dims) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_TileFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateTileFusion(fbb, fbb.CreateVector(dims.data(), dims.size()));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_TILE_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

// ********** TopKFusion **********
PrimitivePtr MindIR_TopKFusion_CreatePrimitive(bool sorted, int64_t axis) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateTopKFusion(fbb, sorted, axis);
  auto prim_offset =
    schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_TOPK_FUSION), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}
bool MindIR_TopKFusion_GetSorted(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_TopKFusion();
    if (prim != nullptr && value != nullptr) {
      return value->sorted();
    } else {
      return false;
    }
  } else {
    return false;
  }
}

void MindIR_TopKFusion_SetSorted(PrimitivePtr *primitive, bool sorted) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_TopKFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateTopKFusion(fbb, sorted, value->axis());
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_TOPK_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}
int64_t MindIR_TopKFusion_GetAxis(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_TopKFusion();
    if (prim != nullptr && value != nullptr) {
      return value->axis();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_TopKFusion_SetAxis(PrimitivePtr *primitive, int64_t axis) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_TopKFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateTopKFusion(fbb, value->sorted(), axis);
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_TOPK_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

// ********** Transpose **********
PrimitivePtr MindIR_Transpose_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateTranspose(fbb);
  auto prim_offset =
    schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_TRANSPOSE), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** Unsqueeze **********
PrimitivePtr MindIR_Unsqueeze_CreatePrimitive(const std::vector<int64_t> &axis) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateUnsqueeze(fbb, fbb.CreateVector(axis.data(), axis.size()));
  auto prim_offset =
    schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_UNSQUEEZE), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}
std::vector<int64_t> MindIR_Unsqueeze_GetAxis(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Unsqueeze();
    if (prim != nullptr && value != nullptr) {
      std::vector<int64_t> result;
      auto src = value->axis();
      if (src == nullptr) {
        return {};
      }
      result.resize(src->size());
      std::transform(src->begin(), src->end(), result.begin(), [](int64_t item) { return item; });
      return result;
    } else {
      return {};
    }
  } else {
    return {};
  }
}

void MindIR_Unsqueeze_SetAxis(PrimitivePtr *primitive, const std::vector<int64_t> &axis) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Unsqueeze();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateUnsqueeze(fbb, fbb.CreateVector(axis.data(), axis.size()));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_UNSQUEEZE), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

// ********** Abs **********
PrimitivePtr MindIR_Abs_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateAbs(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_ABS), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** BroadcastTo **********
PrimitivePtr MindIR_BroadcastTo_CreatePrimitive(const std::vector<int64_t> &shape) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateBroadcastTo(fbb, fbb.CreateVector(shape.data(), shape.size()));
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_BROADCAST_TO), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

std::vector<int64_t> MindIR_BroadcastTo_GetShape(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_BroadcastTo();
    if (prim != nullptr && value != nullptr) {
      std::vector<int64_t> result;
      auto src = value->shape();
      if (src == nullptr) {
        return {};
      }
      result.resize(src->size());
      std::transform(src->begin(), src->end(), result.begin(), [](int64_t item) { return item; });
      return result;
    } else {
      return {};
    }
  } else {
    return {};
  }
}

void MindIR_BroadcastTo_SetShape(PrimitivePtr *primitive, const std::vector<int64_t> &shape) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_BroadcastTo();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateBroadcastTo(
        fbb, fbb.CreateVector(shape.data(), shape.size()));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_BROADCAST_TO), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

// ********** ConstantOfShape **********
PrimitivePtr MindIR_ConstantOfShape_CreatePrimitive(int64_t data_type, const std::vector<float> &value) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateConstantOfShape(fbb, data_type, fbb.CreateVector(value.data(), value.size()));
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_CONSTANT_OF_SHAPE), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

int64_t MindIR_ConstantOfShape_GetDataType(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value_ = prim->value_as_ConstantOfShape();
    if (prim != nullptr && value_ != nullptr) {
      return value_->data_type();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_ConstantOfShape_SetDataType(PrimitivePtr *primitive, int64_t data_type) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value_ = prim->value_as_ConstantOfShape();
    if (prim != nullptr && value_ != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateConstantOfShape(fbb, data_type, fbb.CreateVector(value_->value()->data(), value_->value()->size()));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_CONSTANT_OF_SHAPE), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

std::vector<float> MindIR_ConstantOfShape_GetValue(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value_ = prim->value_as_ConstantOfShape();
    if (prim != nullptr && value_ != nullptr) {
      std::vector<float> result;
      auto src = value_->value();
      if (src == nullptr) {
        return {};
      }
      result.resize(src->size());
      std::transform(src->begin(), src->end(), result.begin(), [](float item) { return item; });
      return result;
    } else {
      return {};
    }
  } else {
    return {};
  }
}

void MindIR_ConstantOfShape_SetValue(PrimitivePtr *primitive, const std::vector<float> &value) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value_ = prim->value_as_ConstantOfShape();
    if (prim != nullptr && value_ != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateConstantOfShape(
        fbb, value_->data_type(), fbb.CreateVector(value.data(), value.size()));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_CONSTANT_OF_SHAPE), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

// ********** DepthToSpace **********
PrimitivePtr MindIR_DepthToSpace_CreatePrimitive(int64_t block_size, Format &format, std::string mode) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateDepthToSpace(fbb, block_size, static_cast<schema::Format>(format), fbb.CreateString(mode));
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_DEPTH_TO_SPACE), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

int64_t MindIR_DepthToSpace_GetBlockSize(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_DepthToSpace();
    if (prim != nullptr && value != nullptr) {
      return value->block_size();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_DepthToSpace_SetBlockSize(PrimitivePtr *primitive, int64_t block_size) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_DepthToSpace();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateDepthToSpace(fbb, block_size, static_cast<schema::Format>(value->format()), 
        fbb.CreateString(std::string(value->mode()->c_str(), value->mode()->size())));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_DEPTH_TO_SPACE), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

Format MindIR_DepthToSpace_GetFormat(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_DepthToSpace();
    if (prim != nullptr && value != nullptr) {
      return static_cast<Format>(value->format());
    } else {
      Format en = static_cast<Format>(0);
      return en;
    }
  } else {
    Format en = static_cast<Format>(0);
    return en;
  }
}

void MindIR_DepthToSpace_SetFormat(PrimitivePtr *primitive, Format &format) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_DepthToSpace();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateDepthToSpace(fbb, value->block_size(), static_cast<schema::Format>(format), 
        fbb.CreateString(std::string(value->mode()->c_str(), value->mode()->size())));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_DEPTH_TO_SPACE), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

std::string MindIR_DepthToSpace_GetMode(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_DepthToSpace();
    if (prim != nullptr && value != nullptr) {
      return std::string(value->mode()->c_str(),value->mode()->size());
    } else {
      return "";
    }
  } else {
    return "";
  }
}

void MindIR_DepthToSpace_SetMode(PrimitivePtr *primitive, std::string mode) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_DepthToSpace();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateDepthToSpace(fbb, value->block_size(), static_cast<schema::Format>(value->format()), fbb.CreateString(mode));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_DEPTH_TO_SPACE), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

// ********** ExpFusion **********
PrimitivePtr MindIR_ExpFusion_CreatePrimitive(float base, float scale, float shift) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateExpFusion(fbb, base, scale, shift);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_EXPFUSION), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

float MindIR_ExpFusion_GetBase(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_ExpFusion();
    if (prim != nullptr && value != nullptr) {
      return value->base();
    } else {
      return .0;
    }
  } else {
    return .0;
  }
}

void MindIR_ExpFusion_SetBase(PrimitivePtr *primitive, float base) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_ExpFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateExpFusion(fbb, base, value->scale(), value->shift());
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_EXPFUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

float MindIR_ExpFusion_GetScale(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_ExpFusion();
    if (prim != nullptr && value != nullptr) {
      return value->scale();
    } else {
      return .0;
    }
  } else {
    return .0;
  }
}

void MindIR_ExpFusion_SetScale(PrimitivePtr *primitive, float scale) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_ExpFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateExpFusion(fbb, value->base(), scale, value->shift());
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_EXPFUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

float MindIR_ExpFusion_GetShift(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_ExpFusion();
    if (prim != nullptr && value != nullptr) {
      return value->shift();
    } else {
      return .0;
    }
  } else {
    return .0;
  }
}

void MindIR_ExpFusion_SetShift(PrimitivePtr *primitive, float shift) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_ExpFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateExpFusion(fbb, value->base(), value->scale(), shift);
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_EXPFUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

// ********** Flatten **********
PrimitivePtr MindIR_Flatten_CreatePrimitive(int64_t axis) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateFlatten(fbb, axis);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_FLATTEN), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

int64_t MindIR_Flatten_GetAxis(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Flatten();
    if (prim != nullptr && value != nullptr) {
      return value->axis();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_Flatten_SetAxis(PrimitivePtr *primitive, int64_t axis) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Flatten();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateFlatten(fbb, axis);
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_FLATTEN), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

// ********** InstanceNorm **********
PrimitivePtr MindIR_InstanceNorm_CreatePrimitive(float epsilon) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateInstanceNorm(fbb, epsilon);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_INSTANCE_NORM), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

float MindIR_InstanceNorm_GetEpsilon(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_InstanceNorm();
    if (prim != nullptr && value != nullptr) {
      return value->epsilon();
    } else {
      return .0;
    }
  } else {
    return .0;
  }
}

void MindIR_InstanceNorm_SetEpsilon(PrimitivePtr *primitive, float epsilon) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_InstanceNorm();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateInstanceNorm(fbb, epsilon);
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_INSTANCE_NORM), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

// ********** Less **********
PrimitivePtr MindIR_Less_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateLess(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_LESS), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** Range **********
PrimitivePtr MindIR_Range_CreatePrimitive(int64_t d_type, int64_t start, int64_t limit, int64_t delta) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateRange(fbb, d_type, start, limit, delta);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_RANGE), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

int64_t MindIR_Range_GetDType(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Range();
    if (prim != nullptr && value != nullptr) {
      return value->d_type();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_Range_SetDType(PrimitivePtr *primitive, int64_t d_type) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Range();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateRange(fbb, d_type, value->start(), value->limit(), value->delta());
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_RANGE), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

int64_t MindIR_Range_GetStart(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Range();
    if (prim != nullptr && value != nullptr) {
      return value->start();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_Range_SetStart(PrimitivePtr *primitive, int64_t start) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Range();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateRange(fbb, value->d_type(), start, value->limit(), value->delta());
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_RANGE), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

int64_t MindIR_Range_GetLimit(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Range();
    if (prim != nullptr && value != nullptr) {
      return value->limit();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_Range_SetLimit(PrimitivePtr *primitive, int64_t limit) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Range();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateRange(fbb, value->d_type(), value->start(), limit, value->delta());
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_RANGE), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

int64_t MindIR_Range_GetDelta(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Range();
    if (prim != nullptr && value != nullptr) {
      return value->delta();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_Range_SetDelta(PrimitivePtr *primitive, int64_t delta) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Range();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateRange(fbb, value->d_type(), value->start(), value->limit(), delta);
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_RANGE), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

// ********** RealDiv **********
PrimitivePtr MindIR_RealDiv_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateRealDiv(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_REAL_DIV), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}


// ********** Square **********
PrimitivePtr MindIR_Square_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateSquare(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_SQUARE), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** Unstack **********
PrimitivePtr MindIR_Unstack_CreatePrimitive(int64_t axis) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateUnstack(fbb, axis);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_UNSTACK), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

int64_t MindIR_Unstack_GetAxis(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Unstack();
    if (prim != nullptr && value != nullptr) {
      return value->axis();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_Unstack_SetAxis(PrimitivePtr *primitive, int64_t axis) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Unstack();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateUnstack(fbb, axis);
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_UNSTACK), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

// ********** Select **********
PrimitivePtr MindIR_Select_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateSelect(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_SELECT), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** Erf **********
PrimitivePtr MindIR_Erf_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateErf(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_ERF), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** Equal **********
PrimitivePtr MindIR_Equal_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateEqual(fbb);
  auto prim_offset =
    schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_EQUAL), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** Greater **********
PrimitivePtr MindIR_Greater_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateGreater(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_GREATER), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** GreaterEqual **********
PrimitivePtr MindIR_GreaterEqual_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateGreaterEqual(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_GREATER_EQUAL), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** NotEqual **********
PrimitivePtr MindIR_NotEqual_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateNotEqual(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_NOT_EQUAL), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** LeakyRelu **********
PrimitivePtr MindIR_LeakyRelu_CreatePrimitive(float negative_slope) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateLeakyRelu(fbb, negative_slope);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_LEAKY_RELU), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

float MindIR_LeakyRelu_GetNegativeSlope(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_LeakyRelu();
    if (prim != nullptr && value != nullptr) {
      return value->negative_slope();
    } else {
      return .0;
    }
  } else {
    return .0;
  }
}

void MindIR_LeakyRelu_SetNegativeSlope(PrimitivePtr *primitive, float negative_slope) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_LeakyRelu();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateInstanceNorm(fbb, negative_slope);
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_LEAKY_RELU), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

// ********** LSTM **********
PrimitivePtr MindIR_LSTM_CreatePrimitive(
  bool bidirectional, bool has_bias, int64_t input_size, int64_t hidden_size, int64_t num_layers,
  int64_t num_directions, float dropout, float zoneout_cell, float zoneout_hidden, int64_t proj_size) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateLSTM(fbb, bidirectional, has_bias, input_size, hidden_size, num_layers,
                                       num_directions,dropout, zoneout_cell, zoneout_hidden, proj_size);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_LSTM), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

bool MindIR_LSTM_GetBidirectional(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_LSTM();
    if (prim != nullptr && value != nullptr) {
      return value->bidirectional();
    } else {
      return false;
    }
  } else {
    return false;
  }
}

void MindIR_LSTM_SetBidirectional(PrimitivePtr *primitive, bool bidirectional) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_LSTM();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = 
        schema::CreateLSTM(fbb, bidirectional, value->has_bias(), value->input_size(), value->hidden_size(),
                           value->num_layers(), value->num_directions(), value->dropout(), value->zoneout_cell(),
                           value->zoneout_hidden(), value->proj_size());
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_LSTM), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

bool MindIR_LSTM_GetHasBias(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_LSTM();
    if (prim != nullptr && value != nullptr) {
      return value->has_bias();
    } else {
      return false;
    }
  } else {
    return false;
  }
}

void MindIR_LSTM_SetHasBias(PrimitivePtr *primitive, bool has_bias) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_LSTM();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = 
        schema::CreateLSTM(fbb, value->bidirectional(), has_bias, value->input_size(), value->hidden_size(),
                           value->num_layers(), value->num_directions(), value->dropout(), value->zoneout_cell(),
                           value->zoneout_hidden(), value->proj_size());
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_LSTM), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

int64_t MindIR_LSTM_GetInputSize(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_LSTM();
    if (prim != nullptr && value != nullptr) {
      return value->input_size();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_LSTM_SetInputSize(PrimitivePtr *primitive, int64_t input_size) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_LSTM();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = 
        schema::CreateLSTM(fbb, value->bidirectional(), value->has_bias(), input_size, value->hidden_size(),
                           value->num_layers(), value->num_directions(), value->dropout(), value->zoneout_cell(),
                           value->zoneout_hidden(), value->proj_size());
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_LSTM), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

int64_t MindIR_LSTM_GetHiddenSize(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_LSTM();
    if (prim != nullptr && value != nullptr) {
      return value->hidden_size();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_LSTM_SetHiddenSize(PrimitivePtr *primitive, int64_t hidden_size) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_LSTM();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = 
        schema::CreateLSTM(fbb, value->bidirectional(), value->has_bias(), value->input_size(), hidden_size,
                           value->num_layers(), value->num_directions(), value->dropout(), value->zoneout_cell(),
                           value->zoneout_hidden(), value->proj_size());
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_LSTM), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

int64_t MindIR_LSTM_GetNumLayers(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_LSTM();
    if (prim != nullptr && value != nullptr) {
      return value->num_layers();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_LSTM_SetNumLayers(PrimitivePtr *primitive, int64_t num_layers) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_LSTM();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = 
        schema::CreateLSTM(fbb, value->bidirectional(), value->has_bias(), value->input_size(), value->hidden_size(),
                           num_layers, value->num_directions(), value->dropout(), value->zoneout_cell(),
                           value->zoneout_hidden(), value->proj_size());
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_LSTM), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

int64_t MindIR_LSTM_GetNumDirections(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_LSTM();
    if (prim != nullptr && value != nullptr) {
      return value->num_directions();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_LSTM_SetNumDirections(PrimitivePtr *primitive, int64_t num_directions) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_LSTM();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = 
        schema::CreateLSTM(fbb, value->bidirectional(), value->has_bias(), value->input_size(), value->hidden_size(),
                           value->num_layers(), num_directions, value->dropout(), value->zoneout_cell(),
                           value->zoneout_hidden(), value->proj_size());
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_LSTM), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

float MindIR_LSTM_GetDropout(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_LSTM();
    if (prim != nullptr && value != nullptr) {
      return value->dropout();
    } else {
      return .0;
    }
  } else {
    return .0;
  }
}

void MindIR_LSTM_SetDropout(PrimitivePtr *primitive, float dropout) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_LSTM();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = 
        schema::CreateLSTM(fbb, value->bidirectional(), value->has_bias(), value->input_size(), value->hidden_size(),
                           value->num_layers(), value->num_directions(), dropout, value->zoneout_cell(),
                           value->zoneout_hidden(), value->proj_size());
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_LSTM), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

float MindIR_LSTM_GetZoneoutCell(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_LSTM();
    if (prim != nullptr && value != nullptr) {
      return value->zoneout_cell();
    } else {
      return .0;
    }
  } else {
    return .0;
  }
}

void MindIR_LSTM_SetZoneoutCell(PrimitivePtr *primitive, float zoneout_cell) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_LSTM();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = 
        schema::CreateLSTM(fbb, value->bidirectional(), value->has_bias(), value->input_size(), value->hidden_size(),
                           value->num_layers(), value->num_directions(), value->dropout(), zoneout_cell,
                           value->zoneout_hidden(), value->proj_size());
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_LSTM), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

float MindIR_LSTM_GetZoneoutHidden(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_LSTM();
    if (prim != nullptr && value != nullptr) {
      return value->zoneout_hidden();
    } else {
      return .0;
    }
  } else {
    return .0;
  }
}

void MindIR_LSTM_SetZoneoutHidden(PrimitivePtr *primitive, float zoneout_hidden) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_LSTM();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = 
        schema::CreateLSTM(fbb, value->bidirectional(), value->has_bias(), value->input_size(), value->hidden_size(),
                           value->num_layers(), value->num_directions(), value->dropout(), value->zoneout_cell(),
                           zoneout_hidden, value->proj_size());
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_LSTM), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

int64_t MindIR_LSTM_GetProjSize(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_LSTM();
    if (prim != nullptr && value != nullptr) {
      return value->proj_size();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_LSTM_SetProjSize(PrimitivePtr *primitive, int64_t proj_size) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_LSTM();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = 
        schema::CreateLSTM(fbb, value->bidirectional(), value->has_bias(), value->input_size(), value->hidden_size(),
                           value->num_layers(), value->num_directions(), value->dropout(), value->zoneout_cell(),
                           value->zoneout_hidden(), proj_size);
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_LSTM), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

// ********** Clip **********
PrimitivePtr MindIR_Clip_CreatePrimitive(float max, float min) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateClip(fbb, max, min);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_CLIP), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

float MindIR_Clip_GetMax(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Clip();
    if (prim != nullptr && value != nullptr) {
      return value->max();
    } else {
      return .0;
    }
  } else {
    return .0;
  }
}

void MindIR_Clip_SetMax(PrimitivePtr *primitive, float max) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Clip();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateClip(fbb, max, value->min());
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_CLIP), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

float MindIR_Clip_GetMin(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Clip();
    if (prim != nullptr && value != nullptr) {
      return value->min();
    } else {
      return .0;
    }
  } else {
    return .0;
  }
}

void MindIR_Clip_SetMin(PrimitivePtr *primitive, float min) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Clip();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateClip(fbb, value->max(), min);
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_CLIP), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

// ********** All **********
PrimitivePtr MindIR_All_CreatePrimitive(int64_t keep_dims) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateAll(fbb, keep_dims);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_ALL), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

int64_t MindIR_All_GetKeepDims(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_All();
    if (prim != nullptr && value != nullptr) {
      return value->keep_dims();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_All_SetKeepDims(PrimitivePtr *primitive, int64_t keep_dims) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_All();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateAll(fbb, keep_dims);
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_ALL), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

// ********** Assert **********
PrimitivePtr MindIR_Assert_CreatePrimitive(int64_t summarize) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateAssert(fbb, summarize);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_ASSERT), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

int64_t MindIR_Assert_GetSummarize(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Assert();
    if (prim != nullptr && value != nullptr) {
      return value->summarize();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_Assert_SetSummarize(PrimitivePtr *primitive, int64_t summarize) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Assert();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateAssert(fbb, summarize);
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_ASSERT), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

// ********** LogicalAnd **********
PrimitivePtr MindIR_LogicalAnd_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateLogicalAnd(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_LOGICAL_AND), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** LogicalNot **********
PrimitivePtr MindIR_LogicalNot_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateLogicalNot(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_LOGICAL_NOT), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** Cos **********
PrimitivePtr MindIR_Cos_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateCos(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_COS), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** Mod **********
PrimitivePtr MindIR_Mod_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateMod(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_MOD), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** Neg **********
PrimitivePtr MindIR_Neg_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateNeg(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_NEG), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** Reciprocal **********
PrimitivePtr MindIR_Reciprocal_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateReciprocal(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_RECIPROCAL), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** Sin **********
PrimitivePtr MindIR_Sin_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateSin(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_SIN), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** Where **********
PrimitivePtr MindIR_Where_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateWhere(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_WHERE), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** Log **********
PrimitivePtr MindIR_Log_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateLog(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_LOG), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** LogicalOr **********
PrimitivePtr MindIR_LogicalOr_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateLogicalOr(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_LOGICAL_OR), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** SparseToDense **********
PrimitivePtr MindIR_SparseToDense_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateSparseToDense(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_SPARSE_TO_DENSE), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** Minimum **********
PrimitivePtr MindIR_Minimum_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateMinimum(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_MINIMUM), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** SpaceToDepth **********
PrimitivePtr MindIR_SpaceToDepth_CreatePrimitive(int64_t block_size, Format format) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateSpaceToDepth(fbb, block_size, static_cast<schema::Format>(format));
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_SPACE_TO_DEPTH), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

int64_t MindIR_SpaceToDepth_GetBlockSize(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_SpaceToDepth();
    if (prim != nullptr && value != nullptr) {
      return value->block_size();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_SpaceToDepth_SetBlockSize(PrimitivePtr *primitive, int64_t block_size) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_SpaceToDepth();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateSpaceToDepth(fbb, block_size, static_cast<schema::Format>(value->format()));
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_SPACE_TO_DEPTH), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

Format MindIR_SpaceToDepth_GetFormat(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_SpaceToDepth();
    if (prim != nullptr && value != nullptr) {
      return static_cast<Format>(value->format());
    } else {
      Format en = static_cast<Format>(0);
      return en;
    }
  } else {
    Format en = static_cast<Format>(0);
    return en;
  }
}

void MindIR_SpaceToDepth_SetFormat(PrimitivePtr *primitive, Format format) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_SpaceToDepth();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateSpaceToDepth(fbb, value->block_size(), static_cast<schema::Format>(format));
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_SPACE_TO_DEPTH), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

// ********** Round **********
PrimitivePtr MindIR_Round_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateRound(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_ROUND), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** Ceil **********
PrimitivePtr MindIR_Ceil_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateCeil(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_CEIL), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** Floor **********
PrimitivePtr MindIR_Floor_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateFloor(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_FLOOR), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** L2NormalizeFusion **********
PrimitivePtr MindIR_L2NormalizeFusion_CreatePrimitive(const std::vector<int64_t> &axis, float epsilon, ActivationType activation_type) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateL2NormalizeFusion(fbb, fbb.CreateVector(axis.data(), axis.size()), epsilon, static_cast<schema::ActivationType>(activation_type));
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_L2_NORMALIZE_FUSION), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

std::vector<int64_t> MindIR_L2NormalizeFusion_GetAxis(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_L2NormalizeFusion();
    if (prim != nullptr && value != nullptr) {
      std::vector<int64_t> result;
      auto src = value->axis();
      if (src == nullptr) {
        return {};
      }
      result.resize(src->size());
      std::transform(src->begin(), src->end(), result.begin(), [](int64_t item) { return item; });
      return result;
    } else {
      return {};
    }
  } else {
    return {};
  }
}

void MindIR_L2NormalizeFusion_SetAxis(PrimitivePtr *primitive, const std::vector<int64_t> &axis) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_L2NormalizeFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateL2NormalizeFusion(fbb, fbb.CreateVector(axis.data(), axis.size()), value->epsilon(), static_cast<schema::ActivationType>(value->activation_type()));
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_L2_NORMALIZE_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

float MindIR_L2NormalizeFusion_GetEpsilon(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_L2NormalizeFusion();
    if (prim != nullptr && value != nullptr) {
      return value->epsilon();
    } else {
      return .0;
    }
  } else {
    return .0;
  }
}

void MindIR_L2NormalizeFusion_SetEpsilon(PrimitivePtr *primitive, float epsilon) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_L2NormalizeFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateL2NormalizeFusion(fbb, fbb.CreateVector(value->axis()->data(), value->axis()->size()), epsilon, static_cast<schema::ActivationType>(value->activation_type()));
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_L2_NORMALIZE_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

ActivationType MindIR_L2NormalizeFusion_GetActivationType(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_L2NormalizeFusion();
    if (prim != nullptr && value != nullptr) {
      return static_cast<ActivationType>(value->activation_type());
    } else {
      ActivationType en = static_cast<ActivationType>(0);
      return en;
    }
  } else {
    ActivationType en = static_cast<ActivationType>(0);
    return en;
  }
}

void MindIR_L2NormalizeFusion_SetActivationType(PrimitivePtr *primitive, ActivationType activation_type) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_L2NormalizeFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateL2NormalizeFusion(fbb, fbb.CreateVector(value->axis()->data(), value->axis()->size()), value->epsilon(), static_cast<schema::ActivationType>(activation_type));
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_L2_NORMALIZE_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

// ********** LRN **********
PrimitivePtr MindIR_LRN_CreatePrimitive(int64_t depth_radius, float bias, float alpha, float beta, std::string norm_region) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateLRN(fbb, depth_radius, bias, alpha, beta, fbb.CreateString(norm_region));
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_L_R_N), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

int64_t MindIR_LRN_GetDepthRadius(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_LRN();
    if (prim != nullptr && value != nullptr) {
      return value->depth_radius();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_LRN_SetDepthRadius(PrimitivePtr *primitive, int64_t depth_radius) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_LRN();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateLRN(fbb, depth_radius, value->bias(), value->alpha(), value->beta(), fbb.CreateString(std::string(value->norm_region()->c_str(), value->norm_region()->size())));
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_L_R_N), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

float MindIR_LRN_GetBias(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_LRN();
    if (prim != nullptr && value != nullptr) {
      return value->bias();
    } else {
      return .0;
    }
  } else {
    return .0;
  }
}

void MindIR_LRN_SetBias(PrimitivePtr *primitive, float bias) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_LRN();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateLRN(fbb, value->depth_radius(), bias, value->alpha(), value->beta(), fbb.CreateString(std::string(value->norm_region()->c_str(), value->norm_region()->size())));
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_L_R_N), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

float MindIR_LRN_GetAlpha(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_LRN();
    if (prim != nullptr && value != nullptr) {
      return value->alpha();
    } else {
      return .0;
    }
  } else {
    return .0;
  }
}

void MindIR_LRN_SetAlpha(PrimitivePtr *primitive, float alpha) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_LRN();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateLRN(fbb, value->depth_radius(), value->bias(), alpha, value->beta(), fbb.CreateString(std::string(value->norm_region()->c_str(), value->norm_region()->size())));
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_L_R_N), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

float MindIR_LRN_GetBeta(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_LRN();
    if (prim != nullptr && value != nullptr) {
      return value->beta();
    } else {
      return .0;
    }
  } else {
    return .0;
  }
}

void MindIR_LRN_SetBeta(PrimitivePtr *primitive, float beta) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_LRN();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateLRN(fbb, value->depth_radius(), value->bias(), value->alpha(), beta, fbb.CreateString(std::string(value->norm_region()->c_str(), value->norm_region()->size())));
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_L_R_N), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

std::string MindIR_LRN_GetNormRegion(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_LRN();
    if (prim != nullptr && value != nullptr) {
      return std::string(value->norm_region()->c_str(),value->norm_region()->size());
    } else {
      return "";
    }
  } else {
    return "";
  }
}

void MindIR_LRN_SetNormRegion(PrimitivePtr *primitive, std::string norm_region) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_LRN();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateLRN(fbb, value->depth_radius(), value->bias(), value->alpha(), value->beta(), fbb.CreateString(norm_region));
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_L_R_N), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

// ********** LogSoftmax **********
PrimitivePtr MindIR_LogSoftmax_CreatePrimitive(int64_t axis) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateLogSoftmax(fbb, axis);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_LOG_SOFTMAX), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

int64_t MindIR_LogSoftmax_GetAxis(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_LogSoftmax();
    if (prim != nullptr && value != nullptr) {
      return value->axis();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_LogSoftmax_SetAxis(PrimitivePtr *primitive, int64_t axis) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_LogSoftmax();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateLogSoftmax(fbb, axis);
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_LOG_SOFTMAX), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

// ********** Crop **********
PrimitivePtr MindIR_Crop_CreatePrimitive(int64_t axis, const std::vector<int64_t> &offsets) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateCrop(fbb, axis, fbb.CreateVector(offsets.data(), offsets.size()));
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_CROP), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

int64_t MindIR_Crop_GetAxis(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Crop();
    if (prim != nullptr && value != nullptr) {
      return value->axis();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_Crop_SetAxis(PrimitivePtr *primitive, int64_t axis) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Crop();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateCrop(fbb, axis, fbb.CreateVector(value->offsets()->data(), value->offsets()->size()));
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_CROP), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

std::vector<int64_t> MindIR_Crop_GetOffsets(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Crop();
    if (prim != nullptr && value != nullptr) {
      std::vector<int64_t> result;
      auto src = value->offsets();
      if (src == nullptr) {
        return {};
      }
      result.resize(src->size());
      std::transform(src->begin(), src->end(), result.begin(), [](int64_t item) { return item; });
      return result;
    } else {
      return {};
    }
  } else {
    return {};
  }
}

void MindIR_Crop_SetOffsets(PrimitivePtr *primitive, const std::vector<int64_t> &offsets) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Crop();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateCrop(fbb, value->axis(), fbb.CreateVector(offsets.data(), offsets.size()));
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_CROP), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

// ********** DetectionPostProcess **********
PrimitivePtr MindIR_DetectionPostProcess_CreatePrimitive(Format format, int64_t input_size, const std::vector<float> &scale, float nms_iou_threshold, float nms_score_threshold, int64_t max_detections, int64_t detections_per_class, int64_t max_classes_per_detection, int64_t num_classes, bool use_regular_nms, bool out_quantized) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateDetectionPostProcess(fbb, static_cast<schema::Format>(format), input_size, fbb.CreateVector(scale.data(), scale.size()), nms_iou_threshold, nms_score_threshold, max_detections, detections_per_class, max_classes_per_detection, num_classes, use_regular_nms, out_quantized);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_DETECTION_POST_PROCESS), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

Format MindIR_DetectionPostProcess_GetFormat(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_DetectionPostProcess();
    if (prim != nullptr && value != nullptr) {
      return static_cast<Format>(value->format());
    } else {
      Format en = static_cast<Format>(0);
      return en;
    }
  } else {
    Format en = static_cast<Format>(0);
    return en;
  }
}

void MindIR_DetectionPostProcess_SetFormat(PrimitivePtr *primitive, Format format) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_DetectionPostProcess();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateDetectionPostProcess(fbb, static_cast<schema::Format>(format), value->input_size(), fbb.CreateVector(value->scale()->data(), value->scale()->size()), value->nms_iou_threshold(), value->nms_score_threshold(), value->max_detections(), value->detections_per_class(), value->max_classes_per_detection(), value->num_classes(), value->use_regular_nms(), value->out_quantized());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_DETECTION_POST_PROCESS), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

int64_t MindIR_DetectionPostProcess_GetInputSize(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_DetectionPostProcess();
    if (prim != nullptr && value != nullptr) {
      return value->input_size();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_DetectionPostProcess_SetInputSize(PrimitivePtr *primitive, int64_t input_size) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_DetectionPostProcess();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateDetectionPostProcess(fbb, static_cast<schema::Format>(value->format()), input_size, fbb.CreateVector(value->scale()->data(), value->scale()->size()), value->nms_iou_threshold(), value->nms_score_threshold(), value->max_detections(), value->detections_per_class(), value->max_classes_per_detection(), value->num_classes(), value->use_regular_nms(), value->out_quantized());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_DETECTION_POST_PROCESS), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

std::vector<float> MindIR_DetectionPostProcess_GetScale(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_DetectionPostProcess();
    if (prim != nullptr && value != nullptr) {
      std::vector<float> result;
      auto src = value->scale();
      if (src == nullptr) {
        return {};
      }
      result.resize(src->size());
      std::transform(src->begin(), src->end(), result.begin(), [](float item) { return item; });
      return result;
    } else {
      return {};
    }
  } else {
    return {};
  }
}

void MindIR_DetectionPostProcess_SetScale(PrimitivePtr *primitive, const std::vector<float> &scale) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_DetectionPostProcess();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateDetectionPostProcess(fbb, static_cast<schema::Format>(value->format()), value->input_size(), fbb.CreateVector(scale.data(), scale.size()), value->nms_iou_threshold(), value->nms_score_threshold(), value->max_detections(), value->detections_per_class(), value->max_classes_per_detection(), value->num_classes(), value->use_regular_nms(), value->out_quantized());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_DETECTION_POST_PROCESS), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

float MindIR_DetectionPostProcess_GetNmsIouThreshold(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_DetectionPostProcess();
    if (prim != nullptr && value != nullptr) {
      return value->nms_iou_threshold();
    } else {
      return .0;
    }
  } else {
    return .0;
  }
}

void MindIR_DetectionPostProcess_SetNmsIouThreshold(PrimitivePtr *primitive, float nms_iou_threshold) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_DetectionPostProcess();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateDetectionPostProcess(fbb, static_cast<schema::Format>(value->format()), value->input_size(), fbb.CreateVector(value->scale()->data(), value->scale()->size()), nms_iou_threshold, value->nms_score_threshold(), value->max_detections(), value->detections_per_class(), value->max_classes_per_detection(), value->num_classes(), value->use_regular_nms(), value->out_quantized());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_DETECTION_POST_PROCESS), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

float MindIR_DetectionPostProcess_GetNmsScoreThreshold(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_DetectionPostProcess();
    if (prim != nullptr && value != nullptr) {
      return value->nms_score_threshold();
    } else {
      return .0;
    }
  } else {
    return .0;
  }
}

void MindIR_DetectionPostProcess_SetNmsScoreThreshold(PrimitivePtr *primitive, float nms_score_threshold) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_DetectionPostProcess();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateDetectionPostProcess(fbb, static_cast<schema::Format>(value->format()), value->input_size(), fbb.CreateVector(value->scale()->data(), value->scale()->size()), value->nms_iou_threshold(), nms_score_threshold, value->max_detections(), value->detections_per_class(), value->max_classes_per_detection(), value->num_classes(), value->use_regular_nms(), value->out_quantized());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_DETECTION_POST_PROCESS), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

int64_t MindIR_DetectionPostProcess_GetMaxDetections(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_DetectionPostProcess();
    if (prim != nullptr && value != nullptr) {
      return value->max_detections();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_DetectionPostProcess_SetMaxDetections(PrimitivePtr *primitive, int64_t max_detections) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_DetectionPostProcess();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateDetectionPostProcess(fbb, static_cast<schema::Format>(value->format()), value->input_size(), fbb.CreateVector(value->scale()->data(), value->scale()->size()), value->nms_iou_threshold(), value->nms_score_threshold(), max_detections, value->detections_per_class(), value->max_classes_per_detection(), value->num_classes(), value->use_regular_nms(), value->out_quantized());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_DETECTION_POST_PROCESS), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

int64_t MindIR_DetectionPostProcess_GetDetectionsPerClass(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_DetectionPostProcess();
    if (prim != nullptr && value != nullptr) {
      return value->detections_per_class();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_DetectionPostProcess_SetDetectionsPerClass(PrimitivePtr *primitive, int64_t detections_per_class) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_DetectionPostProcess();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateDetectionPostProcess(fbb, static_cast<schema::Format>(value->format()), value->input_size(), fbb.CreateVector(value->scale()->data(), value->scale()->size()), value->nms_iou_threshold(), value->nms_score_threshold(), value->max_detections(), detections_per_class, value->max_classes_per_detection(), value->num_classes(), value->use_regular_nms(), value->out_quantized());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_DETECTION_POST_PROCESS), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

int64_t MindIR_DetectionPostProcess_GetMaxClassesPerDetection(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_DetectionPostProcess();
    if (prim != nullptr && value != nullptr) {
      return value->max_classes_per_detection();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_DetectionPostProcess_SetMaxClassesPerDetection(PrimitivePtr *primitive, int64_t max_classes_per_detection) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_DetectionPostProcess();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateDetectionPostProcess(fbb, static_cast<schema::Format>(value->format()), value->input_size(), fbb.CreateVector(value->scale()->data(), value->scale()->size()), value->nms_iou_threshold(), value->nms_score_threshold(), value->max_detections(), value->detections_per_class(), max_classes_per_detection, value->num_classes(), value->use_regular_nms(), value->out_quantized());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_DETECTION_POST_PROCESS), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

int64_t MindIR_DetectionPostProcess_GetNumClasses(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_DetectionPostProcess();
    if (prim != nullptr && value != nullptr) {
      return value->num_classes();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_DetectionPostProcess_SetNumClasses(PrimitivePtr *primitive, int64_t num_classes) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_DetectionPostProcess();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateDetectionPostProcess(fbb, static_cast<schema::Format>(value->format()), value->input_size(), fbb.CreateVector(value->scale()->data(), value->scale()->size()), value->nms_iou_threshold(), value->nms_score_threshold(), value->max_detections(), value->detections_per_class(), value->max_classes_per_detection(), num_classes, value->use_regular_nms(), value->out_quantized());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_DETECTION_POST_PROCESS), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

bool MindIR_DetectionPostProcess_GetUseRegularNms(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_DetectionPostProcess();
    if (prim != nullptr && value != nullptr) {
      return value->use_regular_nms();
    } else {
      return false;
    }
  } else {
    return false;
  }
}

void MindIR_DetectionPostProcess_SetUseRegularNms(PrimitivePtr *primitive, bool use_regular_nms) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_DetectionPostProcess();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateDetectionPostProcess(fbb, static_cast<schema::Format>(value->format()), value->input_size(), fbb.CreateVector(value->scale()->data(), value->scale()->size()), value->nms_iou_threshold(), value->nms_score_threshold(), value->max_detections(), value->detections_per_class(), value->max_classes_per_detection(), value->num_classes(), use_regular_nms, value->out_quantized());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_DETECTION_POST_PROCESS), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

bool MindIR_DetectionPostProcess_GetOutQuantized(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_DetectionPostProcess();
    if (prim != nullptr && value != nullptr) {
      return value->out_quantized();
    } else {
      return false;
    }
  } else {
    return false;
  }
}

void MindIR_DetectionPostProcess_SetOutQuantized(PrimitivePtr *primitive, bool out_quantized) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_DetectionPostProcess();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateDetectionPostProcess(fbb, static_cast<schema::Format>(value->format()), value->input_size(), fbb.CreateVector(value->scale()->data(), value->scale()->size()), value->nms_iou_threshold(), value->nms_score_threshold(), value->max_detections(), value->detections_per_class(), value->max_classes_per_detection(), value->num_classes(), value->use_regular_nms(), out_quantized);
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_DETECTION_POST_PROCESS), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

// ********** ScatterNd **********
PrimitivePtr MindIR_ScatterNd_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateScatterNd(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_SCATTER_ND), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** Rank **********
PrimitivePtr MindIR_Rank_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateRank(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_RANK), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** GatherNd **********
PrimitivePtr MindIR_GatherNd_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateGatherNd(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_GATHER_ND), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** BatchToSpace **********
PrimitivePtr MindIR_BatchToSpace_CreatePrimitive(const std::vector<int64_t> &block_size,
                                                 const std::vector<std::vector<int64_t>> &crops) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateBatchToSpace(fbb, fbb.CreateVector(block_size.data(), block_size.size()),
                                               CreateVec2D(fbb, crops));
  auto prim_offset =
    schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_BATCH_TO_SPACE), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

std::vector<int64_t> MindIR_BatchToSpace_GetBlockSize(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_BatchToSpace();
    if (prim != nullptr && value != nullptr) {
      std::vector<int64_t> result;
      auto src = value->block_size();
      if (src == nullptr) {
        return {};
      }
      result.resize(src->size());
      std::transform(src->begin(), src->end(), result.begin(), [](int64_t item) { return item; });
      return result;
    } else {
      return {};
    }
  } else {
    return {};
  }
}

void MindIR_BatchToSpace_SetBlockSize(PrimitivePtr *primitive, const std::vector<int64_t> &block_size) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_BatchToSpace();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateBatchToSpace(fbb, fbb.CreateVector(block_size.data(), block_size.size()),
                                                   CreateVec2D(fbb, value->crops()));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_BATCH_TO_SPACE), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

std::vector<std::vector<int64_t>> MindIR_BatchToSpace_GetCrops(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_BatchToSpace();
    if (prim != nullptr && value != nullptr) {
      std::vector<std::vector<int64_t>> out;
      auto src = value->crops();
      if (src == nullptr) {
        return {};
      }
      for (auto sub_list : *src->data()) {
        std::vector<int64_t> result_tmp;
        result_tmp.resize(sub_list->data()->size());
        std::transform(sub_list->data()->begin(), sub_list->data()->end(), result_tmp.begin(),
                       [](int64_t item) { return item; });
        out.emplace_back(result_tmp);
      }
      return out;
    } else {
      return {};
    }
  } else {
    return {};
  }
}

void MindIR_BatchToSpace_SetCrops(PrimitivePtr *primitive, const std::vector<std::vector<int64_t>> &crops) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_BatchToSpace();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateBatchToSpace(
        fbb, fbb.CreateVector(value->block_size()->data(), value->block_size()->size()), CreateVec2D(fbb, crops));
      auto prim_offset =
        schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_BATCH_TO_SPACE), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

// ********** Custom **********
std::vector<const mindspore::schema::Attribute *> MindIR_Custom_GetAttr(ConstPrimitivePtr primitive) {
  if (primitive == nullptr) {
    return {};
  }
  auto prim = static_cast<const schema::Primitive *>(primitive);
  auto value = prim->value_as_Custom();
  if (value == nullptr) {
    return {};
  }
  std::vector<const mindspore::schema::Attribute *> result;
  if (value->attr() == nullptr) {
    return {};
  }
  for (auto attr: *(value->attr())) {
    result.push_back(attr);
  }
  return result;
}

std::string MindIR_Attribute_GetName(const mindspore::schema::Attribute &attr) {
  if (attr.name() == nullptr) {
    return "";
  }
  return attr.name()->str();
}

std::vector<uint8_t> MindIR_Attribute_GetData(const mindspore::schema::Attribute &attr) {
  if (attr.data() == nullptr) {
    return {};
  }
  std::vector<uint8_t> tmp_data(attr.data()->size());
  std::transform(attr.data()->begin(), attr.data()->end(), tmp_data.begin(),
                  [](uint8_t item) { return item; });
  return tmp_data;
}
}  // namespace lite
}  // namespace mindspore
