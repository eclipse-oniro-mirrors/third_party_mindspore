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

#include "mindir.h"
#include "utils.h"
#include "schema/model_generated.h"
#include "mindir_memory_manager.h"

namespace mindspore {
namespace lite {
// ********** SplitWithOverlap **********
PrimitivePtr MindIR_SplitWithOverlap_CreatePrimitive(int64_t split_dim, int64_t number_split, const std::vector<int64_t> &ratio, const std::vector<int64_t> &extend_top, const std::vector<int64_t> &extend_bottom) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateSplitWithOverlap(fbb, split_dim, number_split, fbb.CreateVector(ratio.data(), ratio.size()), fbb.CreateVector(extend_top.data(), extend_top.size()), fbb.CreateVector(extend_bottom.data(), extend_bottom.size()));
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_SPLIT_WITH_OVERLAP), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

int64_t MindIR_SplitWithOverlap_GetSplitDim(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_SplitWithOverlap();
    if (prim != nullptr && value != nullptr) {
      return value->split_dim();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_SplitWithOverlap_SetSplitDim(PrimitivePtr *primitive, int64_t split_dim) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_SplitWithOverlap();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateSplitWithOverlap(fbb, split_dim, value->number_split(), fbb.CreateVector(value->ratio()->data(), value->ratio()->size()), fbb.CreateVector(value->extend_top()->data(), value->extend_top()->size()), fbb.CreateVector(value->extend_bottom()->data(), value->extend_bottom()->size()));
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_SPLIT_WITH_OVERLAP), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

int64_t MindIR_SplitWithOverlap_GetNumberSplit(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_SplitWithOverlap();
    if (prim != nullptr && value != nullptr) {
      return value->number_split();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_SplitWithOverlap_SetNumberSplit(PrimitivePtr *primitive, int64_t number_split) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_SplitWithOverlap();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateSplitWithOverlap(fbb, value->split_dim(), number_split, fbb.CreateVector(value->ratio()->data(), value->ratio()->size()), fbb.CreateVector(value->extend_top()->data(), value->extend_top()->size()), fbb.CreateVector(value->extend_bottom()->data(), value->extend_bottom()->size()));
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_SPLIT_WITH_OVERLAP), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

std::vector<int64_t> MindIR_SplitWithOverlap_GetRatio(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_SplitWithOverlap();
    if (prim != nullptr && value != nullptr) {
      std::vector<int64_t> result;
      auto src = value->ratio();
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

void MindIR_SplitWithOverlap_SetRatio(PrimitivePtr *primitive, const std::vector<int64_t> &ratio) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_SplitWithOverlap();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateSplitWithOverlap(fbb, value->split_dim(), value->number_split(), fbb.CreateVector(ratio.data(), ratio.size()), fbb.CreateVector(value->extend_top()->data(), value->extend_top()->size()), fbb.CreateVector(value->extend_bottom()->data(), value->extend_bottom()->size()));
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_SPLIT_WITH_OVERLAP), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

std::vector<int64_t> MindIR_SplitWithOverlap_GetExtendTop(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_SplitWithOverlap();
    if (prim != nullptr && value != nullptr) {
      std::vector<int64_t> result;
      auto src = value->extend_top();
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

void MindIR_SplitWithOverlap_SetExtendTop(PrimitivePtr *primitive, const std::vector<int64_t> &extend_top) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_SplitWithOverlap();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateSplitWithOverlap(fbb, value->split_dim(), value->number_split(), fbb.CreateVector(value->ratio()->data(), value->ratio()->size()), fbb.CreateVector(extend_top.data(), extend_top.size()), fbb.CreateVector(value->extend_bottom()->data(), value->extend_bottom()->size()));
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_SPLIT_WITH_OVERLAP), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

std::vector<int64_t> MindIR_SplitWithOverlap_GetExtendBottom(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_SplitWithOverlap();
    if (prim != nullptr && value != nullptr) {
      std::vector<int64_t> result;
      auto src = value->extend_bottom();
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

void MindIR_SplitWithOverlap_SetExtendBottom(PrimitivePtr *primitive, const std::vector<int64_t> &extend_bottom) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_SplitWithOverlap();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateSplitWithOverlap(fbb, value->split_dim(), value->number_split(), fbb.CreateVector(value->ratio()->data(), value->ratio()->size()), fbb.CreateVector(value->extend_top()->data(), value->extend_top()->size()), fbb.CreateVector(extend_bottom.data(), extend_bottom.size()));
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_SPLIT_WITH_OVERLAP), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

// ********** GenOP **********
PrimitivePtr MindIR_GenOP_CreatePrimitive(ActivationType activation_type, float alpha, float min_val, float max_val, bool is_training, Format format, const std::vector<int64_t> &kernel_size, const std::vector<int64_t> &stride, const std::vector<int64_t> &dilation, PadMode pad_mode, const std::vector<int64_t> &pad_list, int64_t mode, int64_t group, int64_t in_channel, int64_t out_channel, EltwiseMode eltwise_mode, bool has_bias, bool use_axis, int64_t axis, float epsilon, float momentum, bool transpose_a, bool transpose_b, const std::vector<int64_t> &pad, RoundMode round_mode, bool global, bool channel_shared, const std::vector<int64_t> &axes, bool keep_dims, ReduceMode reduce_mode, bool reduce_to_end, float coeff) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateGenOP(fbb, static_cast<schema::ActivationType>(activation_type), alpha, min_val, max_val, is_training, static_cast<schema::Format>(format), fbb.CreateVector(kernel_size.data(), kernel_size.size()), fbb.CreateVector(stride.data(), stride.size()), fbb.CreateVector(dilation.data(), dilation.size()), static_cast<schema::PadMode>(pad_mode), fbb.CreateVector(pad_list.data(), pad_list.size()), mode, group, in_channel, out_channel, static_cast<schema::EltwiseMode>(eltwise_mode), has_bias, use_axis, axis, epsilon, momentum, transpose_a, transpose_b, fbb.CreateVector(pad.data(), pad.size()), static_cast<schema::RoundMode>(round_mode), global, channel_shared, fbb.CreateVector(axes.data(), axes.size()), keep_dims, static_cast<schema::ReduceMode>(reduce_mode), reduce_to_end, coeff);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_GEN_O_P), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

ActivationType MindIR_GenOP_GetActivationType(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_GenOP();
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

void MindIR_GenOP_SetActivationType(PrimitivePtr *primitive, ActivationType activation_type) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateGenOP(fbb, static_cast<schema::ActivationType>(activation_type), value->alpha(), value->min_val(), value->max_val(), value->is_training(), static_cast<schema::Format>(value->format()), fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()), fbb.CreateVector(value->stride()->data(), value->stride()->size()), fbb.CreateVector(value->dilation()->data(), value->dilation()->size()), static_cast<schema::PadMode>(value->pad_mode()), fbb.CreateVector(value->pad_list()->data(), value->pad_list()->size()), value->mode(), value->group(), value->in_channel(), value->out_channel(), static_cast<schema::EltwiseMode>(value->eltwise_mode()), value->has_bias(), value->use_axis(), value->axis(), value->epsilon(), value->momentum(), value->transpose_a(), value->transpose_b(), fbb.CreateVector(value->pad()->data(), value->pad()->size()), static_cast<schema::RoundMode>(value->round_mode()), value->global(), value->channel_shared(), fbb.CreateVector(value->axes()->data(), value->axes()->size()), value->keep_dims(), static_cast<schema::ReduceMode>(value->reduce_mode()), value->reduce_to_end(), value->coeff());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_GEN_O_P), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

float MindIR_GenOP_GetAlpha(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      return value->alpha();
    } else {
      return .0;
    }
  } else {
    return .0;
  }
}

void MindIR_GenOP_SetAlpha(PrimitivePtr *primitive, float alpha) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateGenOP(fbb, static_cast<schema::ActivationType>(value->activation_type()), alpha, value->min_val(), value->max_val(), value->is_training(), static_cast<schema::Format>(value->format()), fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()), fbb.CreateVector(value->stride()->data(), value->stride()->size()), fbb.CreateVector(value->dilation()->data(), value->dilation()->size()), static_cast<schema::PadMode>(value->pad_mode()), fbb.CreateVector(value->pad_list()->data(), value->pad_list()->size()), value->mode(), value->group(), value->in_channel(), value->out_channel(), static_cast<schema::EltwiseMode>(value->eltwise_mode()), value->has_bias(), value->use_axis(), value->axis(), value->epsilon(), value->momentum(), value->transpose_a(), value->transpose_b(), fbb.CreateVector(value->pad()->data(), value->pad()->size()), static_cast<schema::RoundMode>(value->round_mode()), value->global(), value->channel_shared(), fbb.CreateVector(value->axes()->data(), value->axes()->size()), value->keep_dims(), static_cast<schema::ReduceMode>(value->reduce_mode()), value->reduce_to_end(), value->coeff());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_GEN_O_P), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

float MindIR_GenOP_GetMinVal(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      return value->min_val();
    } else {
      return .0;
    }
  } else {
    return .0;
  }
}

void MindIR_GenOP_SetMinVal(PrimitivePtr *primitive, float min_val) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateGenOP(fbb, static_cast<schema::ActivationType>(value->activation_type()), value->alpha(), min_val, value->max_val(), value->is_training(), static_cast<schema::Format>(value->format()), fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()), fbb.CreateVector(value->stride()->data(), value->stride()->size()), fbb.CreateVector(value->dilation()->data(), value->dilation()->size()), static_cast<schema::PadMode>(value->pad_mode()), fbb.CreateVector(value->pad_list()->data(), value->pad_list()->size()), value->mode(), value->group(), value->in_channel(), value->out_channel(), static_cast<schema::EltwiseMode>(value->eltwise_mode()), value->has_bias(), value->use_axis(), value->axis(), value->epsilon(), value->momentum(), value->transpose_a(), value->transpose_b(), fbb.CreateVector(value->pad()->data(), value->pad()->size()), static_cast<schema::RoundMode>(value->round_mode()), value->global(), value->channel_shared(), fbb.CreateVector(value->axes()->data(), value->axes()->size()), value->keep_dims(), static_cast<schema::ReduceMode>(value->reduce_mode()), value->reduce_to_end(), value->coeff());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_GEN_O_P), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

float MindIR_GenOP_GetMaxVal(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      return value->max_val();
    } else {
      return .0;
    }
  } else {
    return .0;
  }
}

void MindIR_GenOP_SetMaxVal(PrimitivePtr *primitive, float max_val) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateGenOP(fbb, static_cast<schema::ActivationType>(value->activation_type()), value->alpha(), value->min_val(), max_val, value->is_training(), static_cast<schema::Format>(value->format()), fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()), fbb.CreateVector(value->stride()->data(), value->stride()->size()), fbb.CreateVector(value->dilation()->data(), value->dilation()->size()), static_cast<schema::PadMode>(value->pad_mode()), fbb.CreateVector(value->pad_list()->data(), value->pad_list()->size()), value->mode(), value->group(), value->in_channel(), value->out_channel(), static_cast<schema::EltwiseMode>(value->eltwise_mode()), value->has_bias(), value->use_axis(), value->axis(), value->epsilon(), value->momentum(), value->transpose_a(), value->transpose_b(), fbb.CreateVector(value->pad()->data(), value->pad()->size()), static_cast<schema::RoundMode>(value->round_mode()), value->global(), value->channel_shared(), fbb.CreateVector(value->axes()->data(), value->axes()->size()), value->keep_dims(), static_cast<schema::ReduceMode>(value->reduce_mode()), value->reduce_to_end(), value->coeff());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_GEN_O_P), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

bool MindIR_GenOP_GetIsTraining(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      return value->is_training();
    } else {
      return false;
    }
  } else {
    return false;
  }
}

void MindIR_GenOP_SetIsTraining(PrimitivePtr *primitive, bool is_training) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateGenOP(fbb, static_cast<schema::ActivationType>(value->activation_type()), value->alpha(), value->min_val(), value->max_val(), is_training, static_cast<schema::Format>(value->format()), fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()), fbb.CreateVector(value->stride()->data(), value->stride()->size()), fbb.CreateVector(value->dilation()->data(), value->dilation()->size()), static_cast<schema::PadMode>(value->pad_mode()), fbb.CreateVector(value->pad_list()->data(), value->pad_list()->size()), value->mode(), value->group(), value->in_channel(), value->out_channel(), static_cast<schema::EltwiseMode>(value->eltwise_mode()), value->has_bias(), value->use_axis(), value->axis(), value->epsilon(), value->momentum(), value->transpose_a(), value->transpose_b(), fbb.CreateVector(value->pad()->data(), value->pad()->size()), static_cast<schema::RoundMode>(value->round_mode()), value->global(), value->channel_shared(), fbb.CreateVector(value->axes()->data(), value->axes()->size()), value->keep_dims(), static_cast<schema::ReduceMode>(value->reduce_mode()), value->reduce_to_end(), value->coeff());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_GEN_O_P), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

Format MindIR_GenOP_GetFormat(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_GenOP();
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

void MindIR_GenOP_SetFormat(PrimitivePtr *primitive, Format format) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateGenOP(fbb, static_cast<schema::ActivationType>(value->activation_type()), value->alpha(), value->min_val(), value->max_val(), value->is_training(), static_cast<schema::Format>(format), fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()), fbb.CreateVector(value->stride()->data(), value->stride()->size()), fbb.CreateVector(value->dilation()->data(), value->dilation()->size()), static_cast<schema::PadMode>(value->pad_mode()), fbb.CreateVector(value->pad_list()->data(), value->pad_list()->size()), value->mode(), value->group(), value->in_channel(), value->out_channel(), static_cast<schema::EltwiseMode>(value->eltwise_mode()), value->has_bias(), value->use_axis(), value->axis(), value->epsilon(), value->momentum(), value->transpose_a(), value->transpose_b(), fbb.CreateVector(value->pad()->data(), value->pad()->size()), static_cast<schema::RoundMode>(value->round_mode()), value->global(), value->channel_shared(), fbb.CreateVector(value->axes()->data(), value->axes()->size()), value->keep_dims(), static_cast<schema::ReduceMode>(value->reduce_mode()), value->reduce_to_end(), value->coeff());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_GEN_O_P), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

std::vector<int64_t> MindIR_GenOP_GetKernelSize(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_GenOP();
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

void MindIR_GenOP_SetKernelSize(PrimitivePtr *primitive, const std::vector<int64_t> &kernel_size) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateGenOP(fbb, static_cast<schema::ActivationType>(value->activation_type()), value->alpha(), value->min_val(), value->max_val(), value->is_training(), static_cast<schema::Format>(value->format()), fbb.CreateVector(kernel_size.data(), kernel_size.size()), fbb.CreateVector(value->stride()->data(), value->stride()->size()), fbb.CreateVector(value->dilation()->data(), value->dilation()->size()), static_cast<schema::PadMode>(value->pad_mode()), fbb.CreateVector(value->pad_list()->data(), value->pad_list()->size()), value->mode(), value->group(), value->in_channel(), value->out_channel(), static_cast<schema::EltwiseMode>(value->eltwise_mode()), value->has_bias(), value->use_axis(), value->axis(), value->epsilon(), value->momentum(), value->transpose_a(), value->transpose_b(), fbb.CreateVector(value->pad()->data(), value->pad()->size()), static_cast<schema::RoundMode>(value->round_mode()), value->global(), value->channel_shared(), fbb.CreateVector(value->axes()->data(), value->axes()->size()), value->keep_dims(), static_cast<schema::ReduceMode>(value->reduce_mode()), value->reduce_to_end(), value->coeff());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_GEN_O_P), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

std::vector<int64_t> MindIR_GenOP_GetStride(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_GenOP();
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

void MindIR_GenOP_SetStride(PrimitivePtr *primitive, const std::vector<int64_t> &stride) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateGenOP(fbb, static_cast<schema::ActivationType>(value->activation_type()), value->alpha(), value->min_val(), value->max_val(), value->is_training(), static_cast<schema::Format>(value->format()), fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()), fbb.CreateVector(stride.data(), stride.size()), fbb.CreateVector(value->dilation()->data(), value->dilation()->size()), static_cast<schema::PadMode>(value->pad_mode()), fbb.CreateVector(value->pad_list()->data(), value->pad_list()->size()), value->mode(), value->group(), value->in_channel(), value->out_channel(), static_cast<schema::EltwiseMode>(value->eltwise_mode()), value->has_bias(), value->use_axis(), value->axis(), value->epsilon(), value->momentum(), value->transpose_a(), value->transpose_b(), fbb.CreateVector(value->pad()->data(), value->pad()->size()), static_cast<schema::RoundMode>(value->round_mode()), value->global(), value->channel_shared(), fbb.CreateVector(value->axes()->data(), value->axes()->size()), value->keep_dims(), static_cast<schema::ReduceMode>(value->reduce_mode()), value->reduce_to_end(), value->coeff());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_GEN_O_P), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

std::vector<int64_t> MindIR_GenOP_GetDilation(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_GenOP();
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

void MindIR_GenOP_SetDilation(PrimitivePtr *primitive, const std::vector<int64_t> &dilation) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateGenOP(fbb, static_cast<schema::ActivationType>(value->activation_type()), value->alpha(), value->min_val(), value->max_val(), value->is_training(), static_cast<schema::Format>(value->format()), fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()), fbb.CreateVector(value->stride()->data(), value->stride()->size()), fbb.CreateVector(dilation.data(), dilation.size()), static_cast<schema::PadMode>(value->pad_mode()), fbb.CreateVector(value->pad_list()->data(), value->pad_list()->size()), value->mode(), value->group(), value->in_channel(), value->out_channel(), static_cast<schema::EltwiseMode>(value->eltwise_mode()), value->has_bias(), value->use_axis(), value->axis(), value->epsilon(), value->momentum(), value->transpose_a(), value->transpose_b(), fbb.CreateVector(value->pad()->data(), value->pad()->size()), static_cast<schema::RoundMode>(value->round_mode()), value->global(), value->channel_shared(), fbb.CreateVector(value->axes()->data(), value->axes()->size()), value->keep_dims(), static_cast<schema::ReduceMode>(value->reduce_mode()), value->reduce_to_end(), value->coeff());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_GEN_O_P), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

PadMode MindIR_GenOP_GetPadMode(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_GenOP();
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

void MindIR_GenOP_SetPadMode(PrimitivePtr *primitive, PadMode pad_mode) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateGenOP(fbb, static_cast<schema::ActivationType>(value->activation_type()), value->alpha(), value->min_val(), value->max_val(), value->is_training(), static_cast<schema::Format>(value->format()), fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()), fbb.CreateVector(value->stride()->data(), value->stride()->size()), fbb.CreateVector(value->dilation()->data(), value->dilation()->size()), static_cast<schema::PadMode>(pad_mode), fbb.CreateVector(value->pad_list()->data(), value->pad_list()->size()), value->mode(), value->group(), value->in_channel(), value->out_channel(), static_cast<schema::EltwiseMode>(value->eltwise_mode()), value->has_bias(), value->use_axis(), value->axis(), value->epsilon(), value->momentum(), value->transpose_a(), value->transpose_b(), fbb.CreateVector(value->pad()->data(), value->pad()->size()), static_cast<schema::RoundMode>(value->round_mode()), value->global(), value->channel_shared(), fbb.CreateVector(value->axes()->data(), value->axes()->size()), value->keep_dims(), static_cast<schema::ReduceMode>(value->reduce_mode()), value->reduce_to_end(), value->coeff());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_GEN_O_P), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

std::vector<int64_t> MindIR_GenOP_GetPadList(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_GenOP();
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

void MindIR_GenOP_SetPadList(PrimitivePtr *primitive, const std::vector<int64_t> &pad_list) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateGenOP(fbb, static_cast<schema::ActivationType>(value->activation_type()), value->alpha(), value->min_val(), value->max_val(), value->is_training(), static_cast<schema::Format>(value->format()), fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()), fbb.CreateVector(value->stride()->data(), value->stride()->size()), fbb.CreateVector(value->dilation()->data(), value->dilation()->size()), static_cast<schema::PadMode>(value->pad_mode()), fbb.CreateVector(pad_list.data(), pad_list.size()), value->mode(), value->group(), value->in_channel(), value->out_channel(), static_cast<schema::EltwiseMode>(value->eltwise_mode()), value->has_bias(), value->use_axis(), value->axis(), value->epsilon(), value->momentum(), value->transpose_a(), value->transpose_b(), fbb.CreateVector(value->pad()->data(), value->pad()->size()), static_cast<schema::RoundMode>(value->round_mode()), value->global(), value->channel_shared(), fbb.CreateVector(value->axes()->data(), value->axes()->size()), value->keep_dims(), static_cast<schema::ReduceMode>(value->reduce_mode()), value->reduce_to_end(), value->coeff());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_GEN_O_P), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

int64_t MindIR_GenOP_GetMode(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      return value->mode();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_GenOP_SetMode(PrimitivePtr *primitive, int64_t mode) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateGenOP(fbb, static_cast<schema::ActivationType>(value->activation_type()), value->alpha(), value->min_val(), value->max_val(), value->is_training(), static_cast<schema::Format>(value->format()), fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()), fbb.CreateVector(value->stride()->data(), value->stride()->size()), fbb.CreateVector(value->dilation()->data(), value->dilation()->size()), static_cast<schema::PadMode>(value->pad_mode()), fbb.CreateVector(value->pad_list()->data(), value->pad_list()->size()), mode, value->group(), value->in_channel(), value->out_channel(), static_cast<schema::EltwiseMode>(value->eltwise_mode()), value->has_bias(), value->use_axis(), value->axis(), value->epsilon(), value->momentum(), value->transpose_a(), value->transpose_b(), fbb.CreateVector(value->pad()->data(), value->pad()->size()), static_cast<schema::RoundMode>(value->round_mode()), value->global(), value->channel_shared(), fbb.CreateVector(value->axes()->data(), value->axes()->size()), value->keep_dims(), static_cast<schema::ReduceMode>(value->reduce_mode()), value->reduce_to_end(), value->coeff());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_GEN_O_P), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

int64_t MindIR_GenOP_GetGroup(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      return value->group();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_GenOP_SetGroup(PrimitivePtr *primitive, int64_t group) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateGenOP(fbb, static_cast<schema::ActivationType>(value->activation_type()), value->alpha(), value->min_val(), value->max_val(), value->is_training(), static_cast<schema::Format>(value->format()), fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()), fbb.CreateVector(value->stride()->data(), value->stride()->size()), fbb.CreateVector(value->dilation()->data(), value->dilation()->size()), static_cast<schema::PadMode>(value->pad_mode()), fbb.CreateVector(value->pad_list()->data(), value->pad_list()->size()), value->mode(), group, value->in_channel(), value->out_channel(), static_cast<schema::EltwiseMode>(value->eltwise_mode()), value->has_bias(), value->use_axis(), value->axis(), value->epsilon(), value->momentum(), value->transpose_a(), value->transpose_b(), fbb.CreateVector(value->pad()->data(), value->pad()->size()), static_cast<schema::RoundMode>(value->round_mode()), value->global(), value->channel_shared(), fbb.CreateVector(value->axes()->data(), value->axes()->size()), value->keep_dims(), static_cast<schema::ReduceMode>(value->reduce_mode()), value->reduce_to_end(), value->coeff());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_GEN_O_P), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

int64_t MindIR_GenOP_GetInChannel(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      return value->in_channel();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_GenOP_SetInChannel(PrimitivePtr *primitive, int64_t in_channel) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateGenOP(fbb, static_cast<schema::ActivationType>(value->activation_type()), value->alpha(), value->min_val(), value->max_val(), value->is_training(), static_cast<schema::Format>(value->format()), fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()), fbb.CreateVector(value->stride()->data(), value->stride()->size()), fbb.CreateVector(value->dilation()->data(), value->dilation()->size()), static_cast<schema::PadMode>(value->pad_mode()), fbb.CreateVector(value->pad_list()->data(), value->pad_list()->size()), value->mode(), value->group(), in_channel, value->out_channel(), static_cast<schema::EltwiseMode>(value->eltwise_mode()), value->has_bias(), value->use_axis(), value->axis(), value->epsilon(), value->momentum(), value->transpose_a(), value->transpose_b(), fbb.CreateVector(value->pad()->data(), value->pad()->size()), static_cast<schema::RoundMode>(value->round_mode()), value->global(), value->channel_shared(), fbb.CreateVector(value->axes()->data(), value->axes()->size()), value->keep_dims(), static_cast<schema::ReduceMode>(value->reduce_mode()), value->reduce_to_end(), value->coeff());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_GEN_O_P), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

int64_t MindIR_GenOP_GetOutChannel(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      return value->out_channel();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_GenOP_SetOutChannel(PrimitivePtr *primitive, int64_t out_channel) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateGenOP(fbb, static_cast<schema::ActivationType>(value->activation_type()), value->alpha(), value->min_val(), value->max_val(), value->is_training(), static_cast<schema::Format>(value->format()), fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()), fbb.CreateVector(value->stride()->data(), value->stride()->size()), fbb.CreateVector(value->dilation()->data(), value->dilation()->size()), static_cast<schema::PadMode>(value->pad_mode()), fbb.CreateVector(value->pad_list()->data(), value->pad_list()->size()), value->mode(), value->group(), value->in_channel(), out_channel, static_cast<schema::EltwiseMode>(value->eltwise_mode()), value->has_bias(), value->use_axis(), value->axis(), value->epsilon(), value->momentum(), value->transpose_a(), value->transpose_b(), fbb.CreateVector(value->pad()->data(), value->pad()->size()), static_cast<schema::RoundMode>(value->round_mode()), value->global(), value->channel_shared(), fbb.CreateVector(value->axes()->data(), value->axes()->size()), value->keep_dims(), static_cast<schema::ReduceMode>(value->reduce_mode()), value->reduce_to_end(), value->coeff());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_GEN_O_P), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

EltwiseMode MindIR_GenOP_GetEltwiseMode(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      return static_cast<EltwiseMode>(value->eltwise_mode());
    } else {
      EltwiseMode en = static_cast<EltwiseMode>(0);
      return en;
    }
  } else {
    EltwiseMode en = static_cast<EltwiseMode>(0);
    return en;
  }
}

void MindIR_GenOP_SetEltwiseMode(PrimitivePtr *primitive, EltwiseMode eltwise_mode) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateGenOP(fbb, static_cast<schema::ActivationType>(value->activation_type()), value->alpha(), value->min_val(), value->max_val(), value->is_training(), static_cast<schema::Format>(value->format()), fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()), fbb.CreateVector(value->stride()->data(), value->stride()->size()), fbb.CreateVector(value->dilation()->data(), value->dilation()->size()), static_cast<schema::PadMode>(value->pad_mode()), fbb.CreateVector(value->pad_list()->data(), value->pad_list()->size()), value->mode(), value->group(), value->in_channel(), value->out_channel(), static_cast<schema::EltwiseMode>(eltwise_mode), value->has_bias(), value->use_axis(), value->axis(), value->epsilon(), value->momentum(), value->transpose_a(), value->transpose_b(), fbb.CreateVector(value->pad()->data(), value->pad()->size()), static_cast<schema::RoundMode>(value->round_mode()), value->global(), value->channel_shared(), fbb.CreateVector(value->axes()->data(), value->axes()->size()), value->keep_dims(), static_cast<schema::ReduceMode>(value->reduce_mode()), value->reduce_to_end(), value->coeff());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_GEN_O_P), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

bool MindIR_GenOP_GetHasBias(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      return value->has_bias();
    } else {
      return false;
    }
  } else {
    return false;
  }
}

void MindIR_GenOP_SetHasBias(PrimitivePtr *primitive, bool has_bias) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateGenOP(fbb, static_cast<schema::ActivationType>(value->activation_type()), value->alpha(), value->min_val(), value->max_val(), value->is_training(), static_cast<schema::Format>(value->format()), fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()), fbb.CreateVector(value->stride()->data(), value->stride()->size()), fbb.CreateVector(value->dilation()->data(), value->dilation()->size()), static_cast<schema::PadMode>(value->pad_mode()), fbb.CreateVector(value->pad_list()->data(), value->pad_list()->size()), value->mode(), value->group(), value->in_channel(), value->out_channel(), static_cast<schema::EltwiseMode>(value->eltwise_mode()), has_bias, value->use_axis(), value->axis(), value->epsilon(), value->momentum(), value->transpose_a(), value->transpose_b(), fbb.CreateVector(value->pad()->data(), value->pad()->size()), static_cast<schema::RoundMode>(value->round_mode()), value->global(), value->channel_shared(), fbb.CreateVector(value->axes()->data(), value->axes()->size()), value->keep_dims(), static_cast<schema::ReduceMode>(value->reduce_mode()), value->reduce_to_end(), value->coeff());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_GEN_O_P), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

bool MindIR_GenOP_GetUseAxis(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      return value->use_axis();
    } else {
      return false;
    }
  } else {
    return false;
  }
}

void MindIR_GenOP_SetUseAxis(PrimitivePtr *primitive, bool use_axis) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateGenOP(fbb, static_cast<schema::ActivationType>(value->activation_type()), value->alpha(), value->min_val(), value->max_val(), value->is_training(), static_cast<schema::Format>(value->format()), fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()), fbb.CreateVector(value->stride()->data(), value->stride()->size()), fbb.CreateVector(value->dilation()->data(), value->dilation()->size()), static_cast<schema::PadMode>(value->pad_mode()), fbb.CreateVector(value->pad_list()->data(), value->pad_list()->size()), value->mode(), value->group(), value->in_channel(), value->out_channel(), static_cast<schema::EltwiseMode>(value->eltwise_mode()), value->has_bias(), use_axis, value->axis(), value->epsilon(), value->momentum(), value->transpose_a(), value->transpose_b(), fbb.CreateVector(value->pad()->data(), value->pad()->size()), static_cast<schema::RoundMode>(value->round_mode()), value->global(), value->channel_shared(), fbb.CreateVector(value->axes()->data(), value->axes()->size()), value->keep_dims(), static_cast<schema::ReduceMode>(value->reduce_mode()), value->reduce_to_end(), value->coeff());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_GEN_O_P), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

int64_t MindIR_GenOP_GetAxis(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      return value->axis();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_GenOP_SetAxis(PrimitivePtr *primitive, int64_t axis) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateGenOP(fbb, static_cast<schema::ActivationType>(value->activation_type()), value->alpha(), value->min_val(), value->max_val(), value->is_training(), static_cast<schema::Format>(value->format()), fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()), fbb.CreateVector(value->stride()->data(), value->stride()->size()), fbb.CreateVector(value->dilation()->data(), value->dilation()->size()), static_cast<schema::PadMode>(value->pad_mode()), fbb.CreateVector(value->pad_list()->data(), value->pad_list()->size()), value->mode(), value->group(), value->in_channel(), value->out_channel(), static_cast<schema::EltwiseMode>(value->eltwise_mode()), value->has_bias(), value->use_axis(), axis, value->epsilon(), value->momentum(), value->transpose_a(), value->transpose_b(), fbb.CreateVector(value->pad()->data(), value->pad()->size()), static_cast<schema::RoundMode>(value->round_mode()), value->global(), value->channel_shared(), fbb.CreateVector(value->axes()->data(), value->axes()->size()), value->keep_dims(), static_cast<schema::ReduceMode>(value->reduce_mode()), value->reduce_to_end(), value->coeff());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_GEN_O_P), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

float MindIR_GenOP_GetEpsilon(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      return value->epsilon();
    } else {
      return .0;
    }
  } else {
    return .0;
  }
}

void MindIR_GenOP_SetEpsilon(PrimitivePtr *primitive, float epsilon) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateGenOP(fbb, static_cast<schema::ActivationType>(value->activation_type()), value->alpha(), value->min_val(), value->max_val(), value->is_training(), static_cast<schema::Format>(value->format()), fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()), fbb.CreateVector(value->stride()->data(), value->stride()->size()), fbb.CreateVector(value->dilation()->data(), value->dilation()->size()), static_cast<schema::PadMode>(value->pad_mode()), fbb.CreateVector(value->pad_list()->data(), value->pad_list()->size()), value->mode(), value->group(), value->in_channel(), value->out_channel(), static_cast<schema::EltwiseMode>(value->eltwise_mode()), value->has_bias(), value->use_axis(), value->axis(), epsilon, value->momentum(), value->transpose_a(), value->transpose_b(), fbb.CreateVector(value->pad()->data(), value->pad()->size()), static_cast<schema::RoundMode>(value->round_mode()), value->global(), value->channel_shared(), fbb.CreateVector(value->axes()->data(), value->axes()->size()), value->keep_dims(), static_cast<schema::ReduceMode>(value->reduce_mode()), value->reduce_to_end(), value->coeff());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_GEN_O_P), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

float MindIR_GenOP_GetMomentum(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      return value->momentum();
    } else {
      return .0;
    }
  } else {
    return .0;
  }
}

void MindIR_GenOP_SetMomentum(PrimitivePtr *primitive, float momentum) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateGenOP(fbb, static_cast<schema::ActivationType>(value->activation_type()), value->alpha(), value->min_val(), value->max_val(), value->is_training(), static_cast<schema::Format>(value->format()), fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()), fbb.CreateVector(value->stride()->data(), value->stride()->size()), fbb.CreateVector(value->dilation()->data(), value->dilation()->size()), static_cast<schema::PadMode>(value->pad_mode()), fbb.CreateVector(value->pad_list()->data(), value->pad_list()->size()), value->mode(), value->group(), value->in_channel(), value->out_channel(), static_cast<schema::EltwiseMode>(value->eltwise_mode()), value->has_bias(), value->use_axis(), value->axis(), value->epsilon(), momentum, value->transpose_a(), value->transpose_b(), fbb.CreateVector(value->pad()->data(), value->pad()->size()), static_cast<schema::RoundMode>(value->round_mode()), value->global(), value->channel_shared(), fbb.CreateVector(value->axes()->data(), value->axes()->size()), value->keep_dims(), static_cast<schema::ReduceMode>(value->reduce_mode()), value->reduce_to_end(), value->coeff());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_GEN_O_P), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

bool MindIR_GenOP_GetTransposeA(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      return value->transpose_a();
    } else {
      return false;
    }
  } else {
    return false;
  }
}

void MindIR_GenOP_SetTransposeA(PrimitivePtr *primitive, bool transpose_a) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateGenOP(fbb, static_cast<schema::ActivationType>(value->activation_type()), value->alpha(), value->min_val(), value->max_val(), value->is_training(), static_cast<schema::Format>(value->format()), fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()), fbb.CreateVector(value->stride()->data(), value->stride()->size()), fbb.CreateVector(value->dilation()->data(), value->dilation()->size()), static_cast<schema::PadMode>(value->pad_mode()), fbb.CreateVector(value->pad_list()->data(), value->pad_list()->size()), value->mode(), value->group(), value->in_channel(), value->out_channel(), static_cast<schema::EltwiseMode>(value->eltwise_mode()), value->has_bias(), value->use_axis(), value->axis(), value->epsilon(), value->momentum(), transpose_a, value->transpose_b(), fbb.CreateVector(value->pad()->data(), value->pad()->size()), static_cast<schema::RoundMode>(value->round_mode()), value->global(), value->channel_shared(), fbb.CreateVector(value->axes()->data(), value->axes()->size()), value->keep_dims(), static_cast<schema::ReduceMode>(value->reduce_mode()), value->reduce_to_end(), value->coeff());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_GEN_O_P), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

bool MindIR_GenOP_GetTransposeB(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      return value->transpose_b();
    } else {
      return false;
    }
  } else {
    return false;
  }
}

void MindIR_GenOP_SetTransposeB(PrimitivePtr *primitive, bool transpose_b) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateGenOP(fbb, static_cast<schema::ActivationType>(value->activation_type()), value->alpha(), value->min_val(), value->max_val(), value->is_training(), static_cast<schema::Format>(value->format()), fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()), fbb.CreateVector(value->stride()->data(), value->stride()->size()), fbb.CreateVector(value->dilation()->data(), value->dilation()->size()), static_cast<schema::PadMode>(value->pad_mode()), fbb.CreateVector(value->pad_list()->data(), value->pad_list()->size()), value->mode(), value->group(), value->in_channel(), value->out_channel(), static_cast<schema::EltwiseMode>(value->eltwise_mode()), value->has_bias(), value->use_axis(), value->axis(), value->epsilon(), value->momentum(), value->transpose_a(), transpose_b, fbb.CreateVector(value->pad()->data(), value->pad()->size()), static_cast<schema::RoundMode>(value->round_mode()), value->global(), value->channel_shared(), fbb.CreateVector(value->axes()->data(), value->axes()->size()), value->keep_dims(), static_cast<schema::ReduceMode>(value->reduce_mode()), value->reduce_to_end(), value->coeff());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_GEN_O_P), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

std::vector<int64_t> MindIR_GenOP_GetPad(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_GenOP();
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

void MindIR_GenOP_SetPad(PrimitivePtr *primitive, const std::vector<int64_t> &pad) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateGenOP(fbb, static_cast<schema::ActivationType>(value->activation_type()), value->alpha(), value->min_val(), value->max_val(), value->is_training(), static_cast<schema::Format>(value->format()), fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()), fbb.CreateVector(value->stride()->data(), value->stride()->size()), fbb.CreateVector(value->dilation()->data(), value->dilation()->size()), static_cast<schema::PadMode>(value->pad_mode()), fbb.CreateVector(value->pad_list()->data(), value->pad_list()->size()), value->mode(), value->group(), value->in_channel(), value->out_channel(), static_cast<schema::EltwiseMode>(value->eltwise_mode()), value->has_bias(), value->use_axis(), value->axis(), value->epsilon(), value->momentum(), value->transpose_a(), value->transpose_b(), fbb.CreateVector(pad.data(), pad.size()), static_cast<schema::RoundMode>(value->round_mode()), value->global(), value->channel_shared(), fbb.CreateVector(value->axes()->data(), value->axes()->size()), value->keep_dims(), static_cast<schema::ReduceMode>(value->reduce_mode()), value->reduce_to_end(), value->coeff());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_GEN_O_P), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

RoundMode MindIR_GenOP_GetRoundMode(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_GenOP();
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

void MindIR_GenOP_SetRoundMode(PrimitivePtr *primitive, RoundMode round_mode) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateGenOP(fbb, static_cast<schema::ActivationType>(value->activation_type()), value->alpha(), value->min_val(), value->max_val(), value->is_training(), static_cast<schema::Format>(value->format()), fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()), fbb.CreateVector(value->stride()->data(), value->stride()->size()), fbb.CreateVector(value->dilation()->data(), value->dilation()->size()), static_cast<schema::PadMode>(value->pad_mode()), fbb.CreateVector(value->pad_list()->data(), value->pad_list()->size()), value->mode(), value->group(), value->in_channel(), value->out_channel(), static_cast<schema::EltwiseMode>(value->eltwise_mode()), value->has_bias(), value->use_axis(), value->axis(), value->epsilon(), value->momentum(), value->transpose_a(), value->transpose_b(), fbb.CreateVector(value->pad()->data(), value->pad()->size()), static_cast<schema::RoundMode>(round_mode), value->global(), value->channel_shared(), fbb.CreateVector(value->axes()->data(), value->axes()->size()), value->keep_dims(), static_cast<schema::ReduceMode>(value->reduce_mode()), value->reduce_to_end(), value->coeff());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_GEN_O_P), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

bool MindIR_GenOP_GetGlobal(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      return value->global();
    } else {
      return false;
    }
  } else {
    return false;
  }
}

void MindIR_GenOP_SetGlobal(PrimitivePtr *primitive, bool global) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateGenOP(fbb, static_cast<schema::ActivationType>(value->activation_type()), value->alpha(), value->min_val(), value->max_val(), value->is_training(), static_cast<schema::Format>(value->format()), fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()), fbb.CreateVector(value->stride()->data(), value->stride()->size()), fbb.CreateVector(value->dilation()->data(), value->dilation()->size()), static_cast<schema::PadMode>(value->pad_mode()), fbb.CreateVector(value->pad_list()->data(), value->pad_list()->size()), value->mode(), value->group(), value->in_channel(), value->out_channel(), static_cast<schema::EltwiseMode>(value->eltwise_mode()), value->has_bias(), value->use_axis(), value->axis(), value->epsilon(), value->momentum(), value->transpose_a(), value->transpose_b(), fbb.CreateVector(value->pad()->data(), value->pad()->size()), static_cast<schema::RoundMode>(value->round_mode()), global, value->channel_shared(), fbb.CreateVector(value->axes()->data(), value->axes()->size()), value->keep_dims(), static_cast<schema::ReduceMode>(value->reduce_mode()), value->reduce_to_end(), value->coeff());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_GEN_O_P), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

bool MindIR_GenOP_GetChannelShared(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      return value->channel_shared();
    } else {
      return false;
    }
  } else {
    return false;
  }
}

void MindIR_GenOP_SetChannelShared(PrimitivePtr *primitive, bool channel_shared) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateGenOP(fbb, static_cast<schema::ActivationType>(value->activation_type()), value->alpha(), value->min_val(), value->max_val(), value->is_training(), static_cast<schema::Format>(value->format()), fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()), fbb.CreateVector(value->stride()->data(), value->stride()->size()), fbb.CreateVector(value->dilation()->data(), value->dilation()->size()), static_cast<schema::PadMode>(value->pad_mode()), fbb.CreateVector(value->pad_list()->data(), value->pad_list()->size()), value->mode(), value->group(), value->in_channel(), value->out_channel(), static_cast<schema::EltwiseMode>(value->eltwise_mode()), value->has_bias(), value->use_axis(), value->axis(), value->epsilon(), value->momentum(), value->transpose_a(), value->transpose_b(), fbb.CreateVector(value->pad()->data(), value->pad()->size()), static_cast<schema::RoundMode>(value->round_mode()), value->global(), channel_shared, fbb.CreateVector(value->axes()->data(), value->axes()->size()), value->keep_dims(), static_cast<schema::ReduceMode>(value->reduce_mode()), value->reduce_to_end(), value->coeff());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_GEN_O_P), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

std::vector<int64_t> MindIR_GenOP_GetAxes(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_GenOP();
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

void MindIR_GenOP_SetAxes(PrimitivePtr *primitive, const std::vector<int64_t> &axes) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateGenOP(fbb, static_cast<schema::ActivationType>(value->activation_type()), value->alpha(), value->min_val(), value->max_val(), value->is_training(), static_cast<schema::Format>(value->format()), fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()), fbb.CreateVector(value->stride()->data(), value->stride()->size()), fbb.CreateVector(value->dilation()->data(), value->dilation()->size()), static_cast<schema::PadMode>(value->pad_mode()), fbb.CreateVector(value->pad_list()->data(), value->pad_list()->size()), value->mode(), value->group(), value->in_channel(), value->out_channel(), static_cast<schema::EltwiseMode>(value->eltwise_mode()), value->has_bias(), value->use_axis(), value->axis(), value->epsilon(), value->momentum(), value->transpose_a(), value->transpose_b(), fbb.CreateVector(value->pad()->data(), value->pad()->size()), static_cast<schema::RoundMode>(value->round_mode()), value->global(), value->channel_shared(), fbb.CreateVector(axes.data(), axes.size()), value->keep_dims(), static_cast<schema::ReduceMode>(value->reduce_mode()), value->reduce_to_end(), value->coeff());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_GEN_O_P), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

bool MindIR_GenOP_GetKeepDims(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      return value->keep_dims();
    } else {
      return false;
    }
  } else {
    return false;
  }
}

void MindIR_GenOP_SetKeepDims(PrimitivePtr *primitive, bool keep_dims) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateGenOP(fbb, static_cast<schema::ActivationType>(value->activation_type()), value->alpha(), value->min_val(), value->max_val(), value->is_training(), static_cast<schema::Format>(value->format()), fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()), fbb.CreateVector(value->stride()->data(), value->stride()->size()), fbb.CreateVector(value->dilation()->data(), value->dilation()->size()), static_cast<schema::PadMode>(value->pad_mode()), fbb.CreateVector(value->pad_list()->data(), value->pad_list()->size()), value->mode(), value->group(), value->in_channel(), value->out_channel(), static_cast<schema::EltwiseMode>(value->eltwise_mode()), value->has_bias(), value->use_axis(), value->axis(), value->epsilon(), value->momentum(), value->transpose_a(), value->transpose_b(), fbb.CreateVector(value->pad()->data(), value->pad()->size()), static_cast<schema::RoundMode>(value->round_mode()), value->global(), value->channel_shared(), fbb.CreateVector(value->axes()->data(), value->axes()->size()), keep_dims, static_cast<schema::ReduceMode>(value->reduce_mode()), value->reduce_to_end(), value->coeff());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_GEN_O_P), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

ReduceMode MindIR_GenOP_GetReduceMode(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      return static_cast<ReduceMode>(value->reduce_mode());
    } else {
      ReduceMode en = static_cast<ReduceMode>(0);
      return en;
    }
  } else {
    ReduceMode en = static_cast<ReduceMode>(0);
    return en;
  }
}

void MindIR_GenOP_SetReduceMode(PrimitivePtr *primitive, ReduceMode reduce_mode) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateGenOP(fbb, static_cast<schema::ActivationType>(value->activation_type()), value->alpha(), value->min_val(), value->max_val(), value->is_training(), static_cast<schema::Format>(value->format()), fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()), fbb.CreateVector(value->stride()->data(), value->stride()->size()), fbb.CreateVector(value->dilation()->data(), value->dilation()->size()), static_cast<schema::PadMode>(value->pad_mode()), fbb.CreateVector(value->pad_list()->data(), value->pad_list()->size()), value->mode(), value->group(), value->in_channel(), value->out_channel(), static_cast<schema::EltwiseMode>(value->eltwise_mode()), value->has_bias(), value->use_axis(), value->axis(), value->epsilon(), value->momentum(), value->transpose_a(), value->transpose_b(), fbb.CreateVector(value->pad()->data(), value->pad()->size()), static_cast<schema::RoundMode>(value->round_mode()), value->global(), value->channel_shared(), fbb.CreateVector(value->axes()->data(), value->axes()->size()), value->keep_dims(), static_cast<schema::ReduceMode>(reduce_mode), value->reduce_to_end(), value->coeff());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_GEN_O_P), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

bool MindIR_GenOP_GetReduceToEnd(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      return value->reduce_to_end();
    } else {
      return false;
    }
  } else {
    return false;
  }
}

void MindIR_GenOP_SetReduceToEnd(PrimitivePtr *primitive, bool reduce_to_end) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateGenOP(fbb, static_cast<schema::ActivationType>(value->activation_type()), value->alpha(), value->min_val(), value->max_val(), value->is_training(), static_cast<schema::Format>(value->format()), fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()), fbb.CreateVector(value->stride()->data(), value->stride()->size()), fbb.CreateVector(value->dilation()->data(), value->dilation()->size()), static_cast<schema::PadMode>(value->pad_mode()), fbb.CreateVector(value->pad_list()->data(), value->pad_list()->size()), value->mode(), value->group(), value->in_channel(), value->out_channel(), static_cast<schema::EltwiseMode>(value->eltwise_mode()), value->has_bias(), value->use_axis(), value->axis(), value->epsilon(), value->momentum(), value->transpose_a(), value->transpose_b(), fbb.CreateVector(value->pad()->data(), value->pad()->size()), static_cast<schema::RoundMode>(value->round_mode()), value->global(), value->channel_shared(), fbb.CreateVector(value->axes()->data(), value->axes()->size()), value->keep_dims(), static_cast<schema::ReduceMode>(value->reduce_mode()), reduce_to_end, value->coeff());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_GEN_O_P), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

float MindIR_GenOP_GetCoeff(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      return value->coeff();
    } else {
      return .0;
    }
  } else {
    return .0;
  }
}

void MindIR_GenOP_SetCoeff(PrimitivePtr *primitive, float coeff) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_GenOP();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateGenOP(fbb, static_cast<schema::ActivationType>(value->activation_type()), value->alpha(), value->min_val(), value->max_val(), value->is_training(), static_cast<schema::Format>(value->format()), fbb.CreateVector(value->kernel_size()->data(), value->kernel_size()->size()), fbb.CreateVector(value->stride()->data(), value->stride()->size()), fbb.CreateVector(value->dilation()->data(), value->dilation()->size()), static_cast<schema::PadMode>(value->pad_mode()), fbb.CreateVector(value->pad_list()->data(), value->pad_list()->size()), value->mode(), value->group(), value->in_channel(), value->out_channel(), static_cast<schema::EltwiseMode>(value->eltwise_mode()), value->has_bias(), value->use_axis(), value->axis(), value->epsilon(), value->momentum(), value->transpose_a(), value->transpose_b(), fbb.CreateVector(value->pad()->data(), value->pad()->size()), static_cast<schema::RoundMode>(value->round_mode()), value->global(), value->channel_shared(), fbb.CreateVector(value->axes()->data(), value->axes()->size()), value->keep_dims(), static_cast<schema::ReduceMode>(value->reduce_mode()), value->reduce_to_end(), coeff);
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_GEN_O_P), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

// ********** RaggedRange **********
PrimitivePtr MindIR_RaggedRange_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateRaggedRange(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_RAGGED_RANGE), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** GLU **********
PrimitivePtr MindIR_GLU_CreatePrimitive(int64_t axis) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateGLU(fbb, axis);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_G_L_U), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

int64_t MindIR_GLU_GetAxis(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_GLU();
    if (prim != nullptr && value != nullptr) {
      return value->axis();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_GLU_SetAxis(PrimitivePtr *primitive, int64_t axis) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_GLU();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateGLU(fbb, axis);
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_G_L_U), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

// ********** Affine **********
PrimitivePtr MindIR_Affine_CreatePrimitive(const std::vector<int64_t> &context, int64_t output_dim, ActivationType activation_type, bool transpose_a, bool transpose_b) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateAffine(fbb, fbb.CreateVector(context.data(), context.size()), output_dim, static_cast<schema::ActivationType>(activation_type), transpose_a, transpose_b);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_AFFINE), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

std::vector<int64_t> MindIR_Affine_GetContext(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Affine();
    if (prim != nullptr && value != nullptr) {
      std::vector<int64_t> result;
      auto src = value->context();
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

void MindIR_Affine_SetContext(PrimitivePtr *primitive, const std::vector<int64_t> &context) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Affine();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateAffine(fbb, fbb.CreateVector(context.data(), context.size()), value->output_dim(), static_cast<schema::ActivationType>(value->activation_type()), value->transpose_a(), value->transpose_b());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_AFFINE), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

int64_t MindIR_Affine_GetOutputDim(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Affine();
    if (prim != nullptr && value != nullptr) {
      return value->output_dim();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_Affine_SetOutputDim(PrimitivePtr *primitive, int64_t output_dim) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Affine();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateAffine(fbb, fbb.CreateVector(value->context()->data(), value->context()->size()), output_dim, static_cast<schema::ActivationType>(value->activation_type()), value->transpose_a(), value->transpose_b());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_AFFINE), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

ActivationType MindIR_Affine_GetActivationType(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Affine();
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

void MindIR_Affine_SetActivationType(PrimitivePtr *primitive, ActivationType activation_type) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Affine();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateAffine(fbb, fbb.CreateVector(value->context()->data(), value->context()->size()), value->output_dim(), static_cast<schema::ActivationType>(activation_type), value->transpose_a(), value->transpose_b());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_AFFINE), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

bool MindIR_Affine_GetTransposeA(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Affine();
    if (prim != nullptr && value != nullptr) {
      return value->transpose_a();
    } else {
      return false;
    }
  } else {
    return false;
  }
}

void MindIR_Affine_SetTransposeA(PrimitivePtr *primitive, bool transpose_a) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Affine();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateAffine(fbb, fbb.CreateVector(value->context()->data(), value->context()->size()), value->output_dim(), static_cast<schema::ActivationType>(value->activation_type()), transpose_a, value->transpose_b());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_AFFINE), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

bool MindIR_Affine_GetTransposeB(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Affine();
    if (prim != nullptr && value != nullptr) {
      return value->transpose_b();
    } else {
      return false;
    }
  } else {
    return false;
  }
}

void MindIR_Affine_SetTransposeB(PrimitivePtr *primitive, bool transpose_b) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Affine();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateAffine(fbb, fbb.CreateVector(value->context()->data(), value->context()->size()), value->output_dim(), static_cast<schema::ActivationType>(value->activation_type()), value->transpose_a(), transpose_b);
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_AFFINE), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

// ********** AllGather **********
PrimitivePtr MindIR_AllGather_CreatePrimitive(const std::string &group, int32_t rank_size) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateAllGather(fbb, fbb.CreateString(group), rank_size);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_ALL_GATHER), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

std::string MindIR_AllGather_GetGroup(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_AllGather();
    if (prim != nullptr && value != nullptr) {
      return std::string(value->group()->c_str(),value->group()->size());
    } else {
      return "";
    }
  } else {
    return "";
  }
}

void MindIR_AllGather_SetGroup(PrimitivePtr *primitive, const std::string &group) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_AllGather();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateAllGather(fbb, fbb.CreateString(group), value->rank_size());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_ALL_GATHER), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

int32_t MindIR_AllGather_GetRankSize(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_AllGather();
    if (prim != nullptr && value != nullptr) {
      return value->rank_size();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_AllGather_SetRankSize(PrimitivePtr *primitive, int32_t rank_size) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_AllGather();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateAllGather(fbb, fbb.CreateString(std::string(value->group()->c_str(), value->group()->size())), rank_size);
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_ALL_GATHER), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

// ********** ReduceScatter **********
PrimitivePtr MindIR_ReduceScatter_CreatePrimitive(const std::string &group, ReduceMode mode, int32_t rank_size) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateReduceScatter(fbb, fbb.CreateString(group), static_cast<schema::ReduceMode>(mode), rank_size);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_REDUCE_SCATTER), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

std::string MindIR_ReduceScatter_GetGroup(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_ReduceScatter();
    if (prim != nullptr && value != nullptr) {
      return std::string(value->group()->c_str(),value->group()->size());
    } else {
      return "";
    }
  } else {
    return "";
  }
}

void MindIR_ReduceScatter_SetGroup(PrimitivePtr *primitive, const std::string &group) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_ReduceScatter();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateReduceScatter(fbb, fbb.CreateString(group), static_cast<schema::ReduceMode>(value->mode()), value->rank_size());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_REDUCE_SCATTER), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

ReduceMode MindIR_ReduceScatter_GetMode(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_ReduceScatter();
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

void MindIR_ReduceScatter_SetMode(PrimitivePtr *primitive, ReduceMode mode) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_ReduceScatter();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateReduceScatter(fbb, fbb.CreateString(std::string(value->group()->c_str(), value->group()->size())), static_cast<schema::ReduceMode>(mode), value->rank_size());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_REDUCE_SCATTER), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

int32_t MindIR_ReduceScatter_GetRankSize(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_ReduceScatter();
    if (prim != nullptr && value != nullptr) {
      return value->rank_size();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_ReduceScatter_SetRankSize(PrimitivePtr *primitive, int32_t rank_size) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_ReduceScatter();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateReduceScatter(fbb, fbb.CreateString(std::string(value->group()->c_str(), value->group()->size())), static_cast<schema::ReduceMode>(value->mode()), rank_size);
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_REDUCE_SCATTER), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

// ********** DynamicQuant **********
PrimitivePtr MindIR_DynamicQuant_CreatePrimitive(bool symmetric, int64_t dst_type) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateDynamicQuant(fbb, symmetric, dst_type);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_DYNAMIC_QUANT), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

bool MindIR_DynamicQuant_GetSymmetric(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_DynamicQuant();
    if (prim != nullptr && value != nullptr) {
      return value->symmetric();
    } else {
      return false;
    }
  } else {
    return false;
  }
}

void MindIR_DynamicQuant_SetSymmetric(PrimitivePtr *primitive, bool symmetric) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_DynamicQuant();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateDynamicQuant(fbb, symmetric, value->dst_type());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_DYNAMIC_QUANT), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

int64_t MindIR_DynamicQuant_GetDstType(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_DynamicQuant();
    if (prim != nullptr && value != nullptr) {
      return value->dst_type();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_DynamicQuant_SetDstType(PrimitivePtr *primitive, int64_t dst_type) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_DynamicQuant();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateDynamicQuant(fbb, value->symmetric(), dst_type);
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_DYNAMIC_QUANT), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

// ********** RandomNormal **********
PrimitivePtr MindIR_RandomNormal_CreatePrimitive(float seed, float mean, float scale) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateRandomNormal(fbb, seed, mean, scale);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_RANDOM_NORMAL), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

float MindIR_RandomNormal_GetSeed(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_RandomNormal();
    if (prim != nullptr && value != nullptr) {
      return value->seed();
    } else {
      return .0;
    }
  } else {
    return .0;
  }
}

void MindIR_RandomNormal_SetSeed(PrimitivePtr *primitive, float seed) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_RandomNormal();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateRandomNormal(fbb, seed, value->mean(), value->scale());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_RANDOM_NORMAL), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

float MindIR_RandomNormal_GetMean(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_RandomNormal();
    if (prim != nullptr && value != nullptr) {
      return value->mean();
    } else {
      return .0;
    }
  } else {
    return .0;
  }
}

void MindIR_RandomNormal_SetMean(PrimitivePtr *primitive, float mean) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_RandomNormal();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateRandomNormal(fbb, value->seed(), mean, value->scale());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_RANDOM_NORMAL), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

float MindIR_RandomNormal_GetScale(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_RandomNormal();
    if (prim != nullptr && value != nullptr) {
      return value->scale();
    } else {
      return .0;
    }
  } else {
    return .0;
  }
}

void MindIR_RandomNormal_SetScale(PrimitivePtr *primitive, float scale) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_RandomNormal();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateRandomNormal(fbb, value->seed(), value->mean(), scale);
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_RANDOM_NORMAL), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

// ********** FormatTranspose **********
PrimitivePtr MindIR_FormatTranspose_CreatePrimitive(Format src_format, Format dst_format) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateFormatTranspose(fbb, static_cast<schema::Format>(src_format), static_cast<schema::Format>(dst_format));
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_FORMAT_TRANSPOSE), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

Format MindIR_FormatTranspose_GetSrcFormat(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_FormatTranspose();
    if (prim != nullptr && value != nullptr) {
      return static_cast<Format>(value->src_format());
    } else {
      Format en = static_cast<Format>(0);
      return en;
    }
  } else {
    Format en = static_cast<Format>(0);
    return en;
  }
}

void MindIR_FormatTranspose_SetSrcFormat(PrimitivePtr *primitive, Format src_format) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_FormatTranspose();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateFormatTranspose(fbb, static_cast<schema::Format>(src_format), static_cast<schema::Format>(value->dst_format()));
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_FORMAT_TRANSPOSE), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

Format MindIR_FormatTranspose_GetDstFormat(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_FormatTranspose();
    if (prim != nullptr && value != nullptr) {
      return static_cast<Format>(value->dst_format());
    } else {
      Format en = static_cast<Format>(0);
      return en;
    }
  } else {
    Format en = static_cast<Format>(0);
    return en;
  }
}

void MindIR_FormatTranspose_SetDstFormat(PrimitivePtr *primitive, Format dst_format) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_FormatTranspose();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateFormatTranspose(fbb, static_cast<schema::Format>(value->src_format()), static_cast<schema::Format>(dst_format));
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_FORMAT_TRANSPOSE), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

// ********** GatherD **********
PrimitivePtr MindIR_GatherD_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateGatherD(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_GATHER_D), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** GroupNormFusion **********
PrimitivePtr MindIR_GroupNormFusion_CreatePrimitive(int64_t num_groups, float epsilon, bool affine) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateGroupNormFusion(fbb, num_groups, epsilon, affine);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_GROUP_NORM_FUSION), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

int64_t MindIR_GroupNormFusion_GetNumGroups(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_GroupNormFusion();
    if (prim != nullptr && value != nullptr) {
      return value->num_groups();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_GroupNormFusion_SetNumGroups(PrimitivePtr *primitive, int64_t num_groups) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_GroupNormFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateGroupNormFusion(fbb, num_groups, value->epsilon(), value->affine());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_GROUP_NORM_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

float MindIR_GroupNormFusion_GetEpsilon(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_GroupNormFusion();
    if (prim != nullptr && value != nullptr) {
      return value->epsilon();
    } else {
      return .0;
    }
  } else {
    return .0;
  }
}

void MindIR_GroupNormFusion_SetEpsilon(PrimitivePtr *primitive, float epsilon) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_GroupNormFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateGroupNormFusion(fbb, value->num_groups(), epsilon, value->affine());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_GROUP_NORM_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

bool MindIR_GroupNormFusion_GetAffine(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_GroupNormFusion();
    if (prim != nullptr && value != nullptr) {
      return value->affine();
    } else {
      return false;
    }
  } else {
    return false;
  }
}

void MindIR_GroupNormFusion_SetAffine(PrimitivePtr *primitive, bool affine) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_GroupNormFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateGroupNormFusion(fbb, value->num_groups(), value->epsilon(), affine);
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_GROUP_NORM_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

// ********** Log1p **********
PrimitivePtr MindIR_Log1p_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateLog1p(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_LOG1P), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** SparseFillEmptyRows **********
PrimitivePtr MindIR_SparseFillEmptyRows_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateSparseFillEmptyRows(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_SPARSE_FILL_EMPTY_ROWS), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** SparseReshape **********
PrimitivePtr MindIR_SparseReshape_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateSparseReshape(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_SPARSE_RESHAPE), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** SparseSegmentSum **********
PrimitivePtr MindIR_SparseSegmentSum_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateSparseSegmentSum(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_SPARSE_SEGMENT_SUM), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** ScatterElements **********
PrimitivePtr MindIR_ScatterElements_CreatePrimitive(int64_t axis) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateScatterElements(fbb, axis);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_SCATTER_ELEMENTS), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

int64_t MindIR_ScatterElements_GetAxis(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_ScatterElements();
    if (prim != nullptr && value != nullptr) {
      return value->axis();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_ScatterElements_SetAxis(PrimitivePtr *primitive, int64_t axis) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_ScatterElements();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateScatterElements(fbb, axis);
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_SCATTER_ELEMENTS), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

// ********** Triu **********
PrimitivePtr MindIR_Triu_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateTriu(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_TRIU), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** Tril **********
PrimitivePtr MindIR_Tril_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateTril(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_TRIL), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}
}  // namespace lite
}  // namespace mindspore
