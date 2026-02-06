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
// ********** Depend **********
PrimitivePtr MindIR_Depend_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateDepend(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_DEPEND), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** Dropout **********
PrimitivePtr MindIR_Dropout_CreatePrimitive(float keep_prob) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateDropout(fbb, keep_prob);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_DROPOUT), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

float MindIR_Dropout_GetKeepProb(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Dropout();
    if (prim != nullptr && value != nullptr) {
      return value->keep_prob();
    } else {
      return .0;
    }
  } else {
    return .0;
  }
}

void MindIR_Dropout_SetKeepProb(PrimitivePtr *primitive, float keep_prob) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Dropout();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateDropout(fbb, keep_prob);
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_DROPOUT), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

// ********** Elu **********
PrimitivePtr MindIR_Elu_CreatePrimitive(float alpha) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateElu(fbb, alpha);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_ELU), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

float MindIR_Elu_GetAlpha(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Elu();
    if (prim != nullptr && value != nullptr) {
      return value->alpha();
    } else {
      return .0;
    }
  } else {
    return .0;
  }
}

void MindIR_Elu_SetAlpha(PrimitivePtr *primitive, float alpha) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Elu();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateElu(fbb, alpha);
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_ELU), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

// ********** EmbeddingLookupFusion **********
PrimitivePtr MindIR_EmbeddingLookupFusion_CreatePrimitive(float max_norm) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateEmbeddingLookupFusion(fbb, max_norm);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_EMBEDDING_LOOKUP_FUSION), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

float MindIR_EmbeddingLookupFusion_GetMaxNorm(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_EmbeddingLookupFusion();
    if (prim != nullptr && value != nullptr) {
      return value->max_norm();
    } else {
      return .0;
    }
  } else {
    return .0;
  }
}

void MindIR_EmbeddingLookupFusion_SetMaxNorm(PrimitivePtr *primitive, float max_norm) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_EmbeddingLookupFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateEmbeddingLookupFusion(fbb, max_norm);
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_EMBEDDING_LOOKUP_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

// ********** FakeQuantWithMinMaxVars **********
PrimitivePtr MindIR_FakeQuantWithMinMaxVars_CreatePrimitive(int64_t num_bits, bool narrow_range) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateFakeQuantWithMinMaxVars(fbb, num_bits, narrow_range);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_FAKE_QUANT_WITH_MIN_MAX_VARS), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

int64_t MindIR_FakeQuantWithMinMaxVars_GetNumBits(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_FakeQuantWithMinMaxVars();
    if (prim != nullptr && value != nullptr) {
      return value->num_bits();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_FakeQuantWithMinMaxVars_SetNumBits(PrimitivePtr *primitive, int64_t num_bits) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_FakeQuantWithMinMaxVars();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateFakeQuantWithMinMaxVars(fbb, num_bits, value->narrow_range());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_FAKE_QUANT_WITH_MIN_MAX_VARS), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

bool MindIR_FakeQuantWithMinMaxVars_GetNarrowRange(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_FakeQuantWithMinMaxVars();
    if (prim != nullptr && value != nullptr) {
      return value->narrow_range();
    } else {
      return false;
    }
  } else {
    return false;
  }
}

void MindIR_FakeQuantWithMinMaxVars_SetNarrowRange(PrimitivePtr *primitive, bool narrow_range) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_FakeQuantWithMinMaxVars();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateFakeQuantWithMinMaxVars(fbb, value->num_bits(), narrow_range);
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_FAKE_QUANT_WITH_MIN_MAX_VARS), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

// ********** FakeQuantWithMinMaxVarsPerChannel **********
PrimitivePtr MindIR_FakeQuantWithMinMaxVarsPerChannel_CreatePrimitive(int64_t num_bits, bool narrow_range) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateFakeQuantWithMinMaxVarsPerChannel(fbb, num_bits, narrow_range);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_FAKE_QUANT_WITH_MIN_MAX_VARS_PER_CHANNEL), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

int64_t MindIR_FakeQuantWithMinMaxVarsPerChannel_GetNumBits(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_FakeQuantWithMinMaxVarsPerChannel();
    if (prim != nullptr && value != nullptr) {
      return value->num_bits();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_FakeQuantWithMinMaxVarsPerChannel_SetNumBits(PrimitivePtr *primitive, int64_t num_bits) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_FakeQuantWithMinMaxVarsPerChannel();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateFakeQuantWithMinMaxVarsPerChannel(fbb, num_bits, value->narrow_range());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_FAKE_QUANT_WITH_MIN_MAX_VARS_PER_CHANNEL), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

bool MindIR_FakeQuantWithMinMaxVarsPerChannel_GetNarrowRange(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_FakeQuantWithMinMaxVarsPerChannel();
    if (prim != nullptr && value != nullptr) {
      return value->narrow_range();
    } else {
      return false;
    }
  } else {
    return false;
  }
}

void MindIR_FakeQuantWithMinMaxVarsPerChannel_SetNarrowRange(PrimitivePtr *primitive, bool narrow_range) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_FakeQuantWithMinMaxVarsPerChannel();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateFakeQuantWithMinMaxVarsPerChannel(fbb, value->num_bits(), narrow_range);
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_FAKE_QUANT_WITH_MIN_MAX_VARS_PER_CHANNEL), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

// ********** FftReal **********
PrimitivePtr MindIR_FftReal_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateFftReal(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_FFT_REAL), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** FftImag **********
PrimitivePtr MindIR_FftImag_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateFftImag(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_FFT_IMAG), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** FloorDiv **********
PrimitivePtr MindIR_FloorDiv_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateFloorDiv(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_FLOOR_DIV), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** FloorMod **********
PrimitivePtr MindIR_FloorMod_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateFloorMod(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_FLOOR_MOD), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** HashtableLookup **********
PrimitivePtr MindIR_HashtableLookup_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateHashtableLookup(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_HASHTABLE_LOOKUP), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** LpNormalization **********
PrimitivePtr MindIR_LpNormalization_CreatePrimitive(int64_t axis, int64_t p) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateLpNormalization(fbb, axis, p);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_LP_NORMALIZATION), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

int64_t MindIR_LpNormalization_GetAxis(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_LpNormalization();
    if (prim != nullptr && value != nullptr) {
      return value->axis();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_LpNormalization_SetAxis(PrimitivePtr *primitive, int64_t axis) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_LpNormalization();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateLpNormalization(fbb, axis, value->p());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_LP_NORMALIZATION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

int64_t MindIR_LpNormalization_GetP(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_LpNormalization();
    if (prim != nullptr && value != nullptr) {
      return value->p();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_LpNormalization_SetP(PrimitivePtr *primitive, int64_t p) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_LpNormalization();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateLpNormalization(fbb, value->axis(), p);
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_LP_NORMALIZATION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, prim);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      *primitive = ret_value;
    }
  }
}

// ********** LshProjection **********
PrimitivePtr MindIR_LshProjection_CreatePrimitive(LshProjectionType type) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateLshProjection(fbb, static_cast<schema::LshProjectionType>(type));
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_LSH_PROJECTION), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

LshProjectionType MindIR_LshProjection_GetType(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_LshProjection();
    if (prim != nullptr && value != nullptr) {
      return static_cast<LshProjectionType>(value->type());
    } else {
      LshProjectionType en = static_cast<LshProjectionType>(0);
      return en;
    }
  } else {
    LshProjectionType en = static_cast<LshProjectionType>(0);
    return en;
  }
}

void MindIR_LshProjection_SetType(PrimitivePtr *primitive, LshProjectionType type) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_LshProjection();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateLshProjection(fbb, static_cast<schema::LshProjectionType>(type));
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_LSH_PROJECTION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

// ********** SwitchLayer **********
PrimitivePtr MindIR_SwitchLayer_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateSwitchLayer(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_SWITCH_LAYER), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** Mfcc **********
PrimitivePtr MindIR_Mfcc_CreatePrimitive(float freq_upper_limit, float freq_lower_limit, int64_t filter_bank_channel_num, int64_t dct_coeff_num) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateMfcc(fbb, freq_upper_limit, freq_lower_limit, filter_bank_channel_num, dct_coeff_num);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_MFCC), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

float MindIR_Mfcc_GetFreqUpperLimit(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Mfcc();
    if (prim != nullptr && value != nullptr) {
      return value->freq_upper_limit();
    } else {
      return .0;
    }
  } else {
    return .0;
  }
}

void MindIR_Mfcc_SetFreqUpperLimit(PrimitivePtr *primitive, float freq_upper_limit) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Mfcc();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateMfcc(fbb, freq_upper_limit, value->freq_lower_limit(), value->filter_bank_channel_num(), value->dct_coeff_num());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_MFCC), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

float MindIR_Mfcc_GetFreqLowerLimit(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Mfcc();
    if (prim != nullptr && value != nullptr) {
      return value->freq_lower_limit();
    } else {
      return .0;
    }
  } else {
    return .0;
  }
}

void MindIR_Mfcc_SetFreqLowerLimit(PrimitivePtr *primitive, float freq_lower_limit) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Mfcc();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateMfcc(fbb, value->freq_upper_limit(), freq_lower_limit, value->filter_bank_channel_num(), value->dct_coeff_num());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_MFCC), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

int64_t MindIR_Mfcc_GetFilterBankChannelNum(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Mfcc();
    if (prim != nullptr && value != nullptr) {
      return value->filter_bank_channel_num();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_Mfcc_SetFilterBankChannelNum(PrimitivePtr *primitive, int64_t filter_bank_channel_num) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Mfcc();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateMfcc(fbb, value->freq_upper_limit(), value->freq_lower_limit(), filter_bank_channel_num, value->dct_coeff_num());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_MFCC), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

int64_t MindIR_Mfcc_GetDctCoeffNum(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Mfcc();
    if (prim != nullptr && value != nullptr) {
      return value->dct_coeff_num();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_Mfcc_SetDctCoeffNum(PrimitivePtr *primitive, int64_t dct_coeff_num) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Mfcc();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateMfcc(fbb, value->freq_upper_limit(), value->freq_lower_limit(), value->filter_bank_channel_num(), dct_coeff_num);
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_MFCC), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

// ********** NonMaxSuppression **********
PrimitivePtr MindIR_NonMaxSuppression_CreatePrimitive(int64_t center_point_box) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateNonMaxSuppression(fbb, center_point_box);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_NON_MAX_SUPPRESSION), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

int64_t MindIR_NonMaxSuppression_GetCenterPointBox(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_NonMaxSuppression();
    if (prim != nullptr && value != nullptr) {
      return value->center_point_box();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_NonMaxSuppression_SetCenterPointBox(PrimitivePtr *primitive, int64_t center_point_box) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_NonMaxSuppression();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateNonMaxSuppression(fbb, center_point_box);
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_NON_MAX_SUPPRESSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

// ********** OnesLike **********
PrimitivePtr MindIR_OnesLike_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateOnesLike(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_ONES_LIKE), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** PartialFusion **********
PrimitivePtr MindIR_PartialFusion_CreatePrimitive(int64_t sub_graph_index) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreatePartialFusion(fbb, sub_graph_index);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_PARTIAL_FUSION), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

int64_t MindIR_PartialFusion_GetSubGraphIndex(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_PartialFusion();
    if (prim != nullptr && value != nullptr) {
      return value->sub_graph_index();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_PartialFusion_SetSubGraphIndex(PrimitivePtr *primitive, int64_t sub_graph_index) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_PartialFusion();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreatePartialFusion(fbb, sub_graph_index);
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_PARTIAL_FUSION), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

// ********** PriorBox **********
PrimitivePtr MindIR_PriorBox_CreatePrimitive(const std::vector<int64_t> &min_sizes, const std::vector<int64_t> &max_sizes, const std::vector<float> &aspect_ratios, const std::vector<float> &variances, int64_t image_size_w, int64_t image_size_h, float step_w, float step_h, bool clip, bool flip, float offset) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreatePriorBox(fbb, fbb.CreateVector(min_sizes.data(), min_sizes.size()), fbb.CreateVector(max_sizes.data(), max_sizes.size()), fbb.CreateVector(aspect_ratios.data(), aspect_ratios.size()), fbb.CreateVector(variances.data(), variances.size()), image_size_w, image_size_h, step_w, step_h, clip, flip, offset);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_PRIOR_BOX), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

std::vector<int64_t> MindIR_PriorBox_GetMinSizes(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_PriorBox();
    if (prim != nullptr && value != nullptr) {
      std::vector<int64_t> result;
      auto src = value->min_sizes();
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

void MindIR_PriorBox_SetMinSizes(PrimitivePtr *primitive, const std::vector<int64_t> &min_sizes) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_PriorBox();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreatePriorBox(fbb, fbb.CreateVector(min_sizes.data(), min_sizes.size()), fbb.CreateVector(value->max_sizes()->data(), value->max_sizes()->size()), fbb.CreateVector(value->aspect_ratios()->data(), value->aspect_ratios()->size()), fbb.CreateVector(value->variances()->data(), value->variances()->size()), value->image_size_w(), value->image_size_h(), value->step_w(), value->step_h(), value->clip(), value->flip(), value->offset());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_PRIOR_BOX), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

std::vector<int64_t> MindIR_PriorBox_GetMaxSizes(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_PriorBox();
    if (prim != nullptr && value != nullptr) {
      std::vector<int64_t> result;
      auto src = value->max_sizes();
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

void MindIR_PriorBox_SetMaxSizes(PrimitivePtr *primitive, const std::vector<int64_t> &max_sizes) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_PriorBox();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreatePriorBox(fbb, fbb.CreateVector(value->min_sizes()->data(), value->min_sizes()->size()), fbb.CreateVector(max_sizes.data(), max_sizes.size()), fbb.CreateVector(value->aspect_ratios()->data(), value->aspect_ratios()->size()), fbb.CreateVector(value->variances()->data(), value->variances()->size()), value->image_size_w(), value->image_size_h(), value->step_w(), value->step_h(), value->clip(), value->flip(), value->offset());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_PRIOR_BOX), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

std::vector<float> MindIR_PriorBox_GetAspectRatios(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_PriorBox();
    if (prim != nullptr && value != nullptr) {
      std::vector<float> result;
      auto src = value->aspect_ratios();
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

void MindIR_PriorBox_SetAspectRatios(PrimitivePtr *primitive, const std::vector<float> &aspect_ratios) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_PriorBox();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreatePriorBox(fbb, fbb.CreateVector(value->min_sizes()->data(), value->min_sizes()->size()), fbb.CreateVector(value->max_sizes()->data(), value->max_sizes()->size()), fbb.CreateVector(aspect_ratios.data(), aspect_ratios.size()), fbb.CreateVector(value->variances()->data(), value->variances()->size()), value->image_size_w(), value->image_size_h(), value->step_w(), value->step_h(), value->clip(), value->flip(), value->offset());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_PRIOR_BOX), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

std::vector<float> MindIR_PriorBox_GetVariances(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_PriorBox();
    if (prim != nullptr && value != nullptr) {
      std::vector<float> result;
      auto src = value->variances();
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

void MindIR_PriorBox_SetVariances(PrimitivePtr *primitive, const std::vector<float> &variances) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_PriorBox();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreatePriorBox(fbb, fbb.CreateVector(value->min_sizes()->data(), value->min_sizes()->size()), fbb.CreateVector(value->max_sizes()->data(), value->max_sizes()->size()), fbb.CreateVector(value->aspect_ratios()->data(), value->aspect_ratios()->size()), fbb.CreateVector(variances.data(), variances.size()), value->image_size_w(), value->image_size_h(), value->step_w(), value->step_h(), value->clip(), value->flip(), value->offset());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_PRIOR_BOX), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

int64_t MindIR_PriorBox_GetImageSizeW(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_PriorBox();
    if (prim != nullptr && value != nullptr) {
      return value->image_size_w();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_PriorBox_SetImageSizeW(PrimitivePtr *primitive, int64_t image_size_w) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_PriorBox();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreatePriorBox(fbb, fbb.CreateVector(value->min_sizes()->data(), value->min_sizes()->size()), fbb.CreateVector(value->max_sizes()->data(), value->max_sizes()->size()), fbb.CreateVector(value->aspect_ratios()->data(), value->aspect_ratios()->size()), fbb.CreateVector(value->variances()->data(), value->variances()->size()), image_size_w, value->image_size_h(), value->step_w(), value->step_h(), value->clip(), value->flip(), value->offset());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_PRIOR_BOX), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

int64_t MindIR_PriorBox_GetImageSizeH(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_PriorBox();
    if (prim != nullptr && value != nullptr) {
      return value->image_size_h();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_PriorBox_SetImageSizeH(PrimitivePtr *primitive, int64_t image_size_h) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_PriorBox();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreatePriorBox(fbb, fbb.CreateVector(value->min_sizes()->data(), value->min_sizes()->size()), fbb.CreateVector(value->max_sizes()->data(), value->max_sizes()->size()), fbb.CreateVector(value->aspect_ratios()->data(), value->aspect_ratios()->size()), fbb.CreateVector(value->variances()->data(), value->variances()->size()), value->image_size_w(), image_size_h, value->step_w(), value->step_h(), value->clip(), value->flip(), value->offset());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_PRIOR_BOX), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

float MindIR_PriorBox_GetStepW(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_PriorBox();
    if (prim != nullptr && value != nullptr) {
      return value->step_w();
    } else {
      return .0;
    }
  } else {
    return .0;
  }
}

void MindIR_PriorBox_SetStepW(PrimitivePtr *primitive, float step_w) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_PriorBox();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreatePriorBox(fbb, fbb.CreateVector(value->min_sizes()->data(), value->min_sizes()->size()), fbb.CreateVector(value->max_sizes()->data(), value->max_sizes()->size()), fbb.CreateVector(value->aspect_ratios()->data(), value->aspect_ratios()->size()), fbb.CreateVector(value->variances()->data(), value->variances()->size()), value->image_size_w(), value->image_size_h(), step_w, value->step_h(), value->clip(), value->flip(), value->offset());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_PRIOR_BOX), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

float MindIR_PriorBox_GetStepH(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_PriorBox();
    if (prim != nullptr && value != nullptr) {
      return value->step_h();
    } else {
      return .0;
    }
  } else {
    return .0;
  }
}

void MindIR_PriorBox_SetStepH(PrimitivePtr *primitive, float step_h) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_PriorBox();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreatePriorBox(fbb, fbb.CreateVector(value->min_sizes()->data(), value->min_sizes()->size()), fbb.CreateVector(value->max_sizes()->data(), value->max_sizes()->size()), fbb.CreateVector(value->aspect_ratios()->data(), value->aspect_ratios()->size()), fbb.CreateVector(value->variances()->data(), value->variances()->size()), value->image_size_w(), value->image_size_h(), value->step_w(), step_h, value->clip(), value->flip(), value->offset());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_PRIOR_BOX), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

bool MindIR_PriorBox_GetClip(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_PriorBox();
    if (prim != nullptr && value != nullptr) {
      return value->clip();
    } else {
      return false;
    }
  } else {
    return false;
  }
}

void MindIR_PriorBox_SetClip(PrimitivePtr *primitive, bool clip) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_PriorBox();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreatePriorBox(fbb, fbb.CreateVector(value->min_sizes()->data(), value->min_sizes()->size()), fbb.CreateVector(value->max_sizes()->data(), value->max_sizes()->size()), fbb.CreateVector(value->aspect_ratios()->data(), value->aspect_ratios()->size()), fbb.CreateVector(value->variances()->data(), value->variances()->size()), value->image_size_w(), value->image_size_h(), value->step_w(), value->step_h(), clip, value->flip(), value->offset());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_PRIOR_BOX), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

bool MindIR_PriorBox_GetFlip(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_PriorBox();
    if (prim != nullptr && value != nullptr) {
      return value->flip();
    } else {
      return false;
    }
  } else {
    return false;
  }
}

void MindIR_PriorBox_SetFlip(PrimitivePtr *primitive, bool flip) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_PriorBox();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreatePriorBox(fbb, fbb.CreateVector(value->min_sizes()->data(), value->min_sizes()->size()), fbb.CreateVector(value->max_sizes()->data(), value->max_sizes()->size()), fbb.CreateVector(value->aspect_ratios()->data(), value->aspect_ratios()->size()), fbb.CreateVector(value->variances()->data(), value->variances()->size()), value->image_size_w(), value->image_size_h(), value->step_w(), value->step_h(), value->clip(), flip, value->offset());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_PRIOR_BOX), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

float MindIR_PriorBox_GetOffset(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_PriorBox();
    if (prim != nullptr && value != nullptr) {
      return value->offset();
    } else {
      return .0;
    }
  } else {
    return .0;
  }
}

void MindIR_PriorBox_SetOffset(PrimitivePtr *primitive, float offset) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_PriorBox();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreatePriorBox(fbb, fbb.CreateVector(value->min_sizes()->data(), value->min_sizes()->size()), fbb.CreateVector(value->max_sizes()->data(), value->max_sizes()->size()), fbb.CreateVector(value->aspect_ratios()->data(), value->aspect_ratios()->size()), fbb.CreateVector(value->variances()->data(), value->variances()->size()), value->image_size_w(), value->image_size_h(), value->step_w(), value->step_h(), value->clip(), value->flip(), offset);
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_PRIOR_BOX), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

// ********** ReverseSequence **********
PrimitivePtr MindIR_ReverseSequence_CreatePrimitive(int64_t seq_dim, int64_t batch_dim) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateReverseSequence(fbb, seq_dim, batch_dim);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_REVERSE_SEQUENCE), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

int64_t MindIR_ReverseSequence_GetSeqDim(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_ReverseSequence();
    if (prim != nullptr && value != nullptr) {
      return value->seq_dim();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_ReverseSequence_SetSeqDim(PrimitivePtr *primitive, int64_t seq_dim) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_ReverseSequence();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateReverseSequence(fbb, seq_dim, value->batch_dim());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_REVERSE_SEQUENCE), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

int64_t MindIR_ReverseSequence_GetBatchDim(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_ReverseSequence();
    if (prim != nullptr && value != nullptr) {
      return value->batch_dim();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_ReverseSequence_SetBatchDim(PrimitivePtr *primitive, int64_t batch_dim) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_ReverseSequence();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateReverseSequence(fbb, value->seq_dim(), batch_dim);
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_REVERSE_SEQUENCE), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

// ********** ReverseV2 **********
PrimitivePtr MindIR_ReverseV2_CreatePrimitive(const std::vector<int64_t> &axis) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateReverseV2(fbb, fbb.CreateVector(axis.data(), axis.size()));
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_REVERSE_V2), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

std::vector<int64_t> MindIR_ReverseV2_GetAxis(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_ReverseV2();
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

void MindIR_ReverseV2_SetAxis(PrimitivePtr *primitive, const std::vector<int64_t> &axis) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_ReverseV2();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateReverseV2(fbb, fbb.CreateVector(axis.data(), axis.size()));
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_REVERSE_V2), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

// ********** Rfft **********
PrimitivePtr MindIR_Rfft_CreatePrimitive(int64_t fft_length) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateRfft(fbb, fft_length);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_RFFT), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

int64_t MindIR_Rfft_GetFftLength(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Rfft();
    if (prim != nullptr && value != nullptr) {
      return value->fft_length();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_Rfft_SetFftLength(PrimitivePtr *primitive, int64_t fft_length) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Rfft();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateRfft(fbb, fft_length);
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_RFFT), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

// ********** ROIPooling **********
PrimitivePtr MindIR_ROIPooling_CreatePrimitive(int64_t pooled_h, int64_t pooled_w, float scale) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateROIPooling(fbb, pooled_h, pooled_w, scale);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_R_O_I_POOLING), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

int64_t MindIR_ROIPooling_GetPooledH(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_ROIPooling();
    if (prim != nullptr && value != nullptr) {
      return value->pooled_h();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_ROIPooling_SetPooledH(PrimitivePtr *primitive, int64_t pooled_h) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_ROIPooling();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateROIPooling(fbb, pooled_h, value->pooled_w(), value->scale());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_R_O_I_POOLING), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

int64_t MindIR_ROIPooling_GetPooledW(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_ROIPooling();
    if (prim != nullptr && value != nullptr) {
      return value->pooled_w();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_ROIPooling_SetPooledW(PrimitivePtr *primitive, int64_t pooled_w) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_ROIPooling();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateROIPooling(fbb, value->pooled_h(), pooled_w, value->scale());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_R_O_I_POOLING), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

float MindIR_ROIPooling_GetScale(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_ROIPooling();
    if (prim != nullptr && value != nullptr) {
      return value->scale();
    } else {
      return .0;
    }
  } else {
    return .0;
  }
}

void MindIR_ROIPooling_SetScale(PrimitivePtr *primitive, float scale) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_ROIPooling();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateROIPooling(fbb, value->pooled_h(), value->pooled_w(), scale);
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_R_O_I_POOLING), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

// ********** SkipGram **********
PrimitivePtr MindIR_SkipGram_CreatePrimitive(bool include_all_grams, int64_t max_skip_size, int64_t ngram_size) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateSkipGram(fbb, include_all_grams, max_skip_size, ngram_size);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_SKIP_GRAM), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

bool MindIR_SkipGram_GetIncludeAllGrams(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_SkipGram();
    if (prim != nullptr && value != nullptr) {
      return value->include_all_grams();
    } else {
      return false;
    }
  } else {
    return false;
  }
}

void MindIR_SkipGram_SetIncludeAllGrams(PrimitivePtr *primitive, bool include_all_grams) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_SkipGram();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateSkipGram(fbb, include_all_grams, value->max_skip_size(), value->ngram_size());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_SKIP_GRAM), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

int64_t MindIR_SkipGram_GetMaxSkipSize(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_SkipGram();
    if (prim != nullptr && value != nullptr) {
      return value->max_skip_size();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_SkipGram_SetMaxSkipSize(PrimitivePtr *primitive, int64_t max_skip_size) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_SkipGram();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateSkipGram(fbb, value->include_all_grams(), max_skip_size, value->ngram_size());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_SKIP_GRAM), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

int64_t MindIR_SkipGram_GetNgramSize(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_SkipGram();
    if (prim != nullptr && value != nullptr) {
      return value->ngram_size();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_SkipGram_SetNgramSize(PrimitivePtr *primitive, int64_t ngram_size) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_SkipGram();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateSkipGram(fbb, value->include_all_grams(), value->max_skip_size(), ngram_size);
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_SKIP_GRAM), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

// ********** Switch **********
PrimitivePtr MindIR_Switch_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateSwitch(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_SWITCH), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** Unique **********
PrimitivePtr MindIR_Unique_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateUnique(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_UNIQUE), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** UnsortedSegmentSum **********
PrimitivePtr MindIR_UnsortedSegmentSum_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateUnsortedSegmentSum(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_UNSORTED_SEGMENT_SUM), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** ZerosLike **********
PrimitivePtr MindIR_ZerosLike_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateZerosLike(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_ZEROS_LIKE), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** GRU **********
PrimitivePtr MindIR_GRU_CreatePrimitive(bool bidirectional) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateGRU(fbb, bidirectional);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_G_R_U), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

bool MindIR_GRU_GetBidirectional(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_GRU();
    if (prim != nullptr && value != nullptr) {
      return value->bidirectional();
    } else {
      return false;
    }
  } else {
    return false;
  }
}

void MindIR_GRU_SetBidirectional(PrimitivePtr *primitive, bool bidirectional) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_GRU();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateGRU(fbb, bidirectional);
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_G_R_U), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

// ********** NonZero **********
PrimitivePtr MindIR_NonZero_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateNonZero(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_NON_ZERO), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** InvertPermutation **********
PrimitivePtr MindIR_InvertPermutation_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateInvertPermutation(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_INVERT_PERMUTATION), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** Size **********
PrimitivePtr MindIR_Size_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateSize(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_SIZE), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** RandomStandardNormal **********
PrimitivePtr MindIR_RandomStandardNormal_CreatePrimitive(int64_t seed, int64_t seed2) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateRandomStandardNormal(fbb, seed, seed2);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_RANDOM_STANDARD_NORMAL), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

int64_t MindIR_RandomStandardNormal_GetSeed(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_RandomStandardNormal();
    if (prim != nullptr && value != nullptr) {
      return value->seed();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_RandomStandardNormal_SetSeed(PrimitivePtr *primitive, int64_t seed) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_RandomStandardNormal();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateRandomStandardNormal(fbb, seed, value->seed2());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_RANDOM_STANDARD_NORMAL), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}
int64_t MindIR_RandomStandardNormal_GetSeed2(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_RandomStandardNormal();
    if (prim != nullptr && value != nullptr) {
      return value->seed2();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_RandomStandardNormal_SetSeed2(PrimitivePtr *primitive, int64_t seed2) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_RandomStandardNormal();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateRandomStandardNormal(fbb, value->seed(), seed2);
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_RANDOM_STANDARD_NORMAL), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

// ********** CropAndResize **********
PrimitivePtr MindIR_CropAndResize_CreatePrimitive(ResizeMethod method, float extrapolation_value) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateCropAndResize(fbb, static_cast<schema::ResizeMethod>(method), extrapolation_value);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_CROP_AND_RESIZE), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

ResizeMethod MindIR_CropAndResize_GetMethod(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_CropAndResize();
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

void MindIR_CropAndResize_SetMethod(PrimitivePtr *primitive, ResizeMethod method) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_CropAndResize();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateCropAndResize(fbb, static_cast<schema::ResizeMethod>(method), value->extrapolation_value());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_CROP_AND_RESIZE), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

float MindIR_CropAndResize_GetExtrapolationValue(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_CropAndResize();
    if (prim != nullptr && value != nullptr) {
      return value->extrapolation_value();
    } else {
      return .0;
    }
  } else {
    return .0;
  }
}

void MindIR_CropAndResize_SetExtrapolationValue(PrimitivePtr *primitive, float extrapolation_value) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_CropAndResize();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateCropAndResize(fbb, static_cast<schema::ResizeMethod>(value->method()), extrapolation_value);
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_CROP_AND_RESIZE), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

// ********** IsFinite **********
PrimitivePtr MindIR_IsFinite_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateIsFinite(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_IS_FINITE), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** LinSpace **********
PrimitivePtr MindIR_LinSpace_CreatePrimitive() {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateLinSpace(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_LIN_SPACE), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

// ********** UniformReal **********
PrimitivePtr MindIR_UniformReal_CreatePrimitive(int64_t seed, int64_t seed2) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateUniformReal(fbb, seed, seed2);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_UNIFORM_REAL), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

int64_t MindIR_UniformReal_GetSeed(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_UniformReal();
    if (prim != nullptr && value != nullptr) {
      return value->seed();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_UniformReal_SetSeed(PrimitivePtr *primitive, int64_t seed) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_UniformReal();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateUniformReal(fbb, seed, value->seed2());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_UNIFORM_REAL), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

int64_t MindIR_UniformReal_GetSeed2(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_UniformReal();
    if (prim != nullptr && value != nullptr) {
      return value->seed2();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_UniformReal_SetSeed2(PrimitivePtr *primitive, int64_t seed2) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_UniformReal();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateUniformReal(fbb, value->seed(), seed2);
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_UNIFORM_REAL), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

// ********** Splice **********
PrimitivePtr MindIR_Splice_CreatePrimitive(const std::vector<int64_t> &context, const std::vector<int64_t> &forward_indexes, int64_t output_dim) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateSplice(fbb, fbb.CreateVector(context.data(), context.size()), fbb.CreateVector(forward_indexes.data(), forward_indexes.size()), output_dim);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_SPLICE), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

std::vector<int64_t> MindIR_Splice_GetContext(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Splice();
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

void MindIR_Splice_SetContext(PrimitivePtr *primitive, const std::vector<int64_t> &context) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Splice();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateSplice(fbb, fbb.CreateVector(context.data(), context.size()), fbb.CreateVector(value->forward_indexes()->data(), value->forward_indexes()->size()), value->output_dim());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_SPLICE), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

std::vector<int64_t> MindIR_Splice_GetForwardIndexes(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Splice();
    if (prim != nullptr && value != nullptr) {
      std::vector<int64_t> result;
      auto src = value->forward_indexes();
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

void MindIR_Splice_SetForwardIndexes(PrimitivePtr *primitive, const std::vector<int64_t> &forward_indexes) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Splice();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateSplice(fbb, fbb.CreateVector(value->context()->data(), value->context()->size()), fbb.CreateVector(forward_indexes.data(), forward_indexes.size()), value->output_dim());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_SPLICE), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

int64_t MindIR_Splice_GetOutputDim(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Splice();
    if (prim != nullptr && value != nullptr) {
      return value->output_dim();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void MindIR_Splice_SetOutputDim(PrimitivePtr *primitive, int64_t output_dim) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Splice();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateSplice(fbb, fbb.CreateVector(value->context()->data(), value->context()->size()), fbb.CreateVector(value->forward_indexes()->data(), value->forward_indexes()->size()), output_dim);
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_SPLICE), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

// ********** Call **********
PrimitivePtr MindIR_Call_CreatePrimitive(bool is_tail_call) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateCall(fbb, is_tail_call);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_CALL), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

bool MindIR_Call_GetIsTailCall(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_Call();
    if (prim != nullptr && value != nullptr) {
      return value->is_tail_call();
    } else {
      return false;
    }
  } else {
    return false;
  }
}

void MindIR_Call_SetIsTailCall(PrimitivePtr *primitive, bool is_tail_call) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_Call();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateCall(fbb, is_tail_call);
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_CALL), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

// ********** CumSum **********
PrimitivePtr MindIR_CumSum_CreatePrimitive(bool exclusive, bool reverse) {
  flatbuffers::FlatBufferBuilder fbb;
  auto ops_offset = schema::CreateCumSum(fbb, exclusive, reverse);
  auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_CUM_SUM), ops_offset.o);
  fbb.Finish(prim_offset);
  auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
  auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
  return ret_value;
}

bool MindIR_CumSum_GetExclusive(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_CumSum();
    if (prim != nullptr && value != nullptr) {
      return value->exclusive();
    } else {
      return false;
    }
  } else {
    return false;
  }
}

void MindIR_CumSum_SetExclusive(PrimitivePtr *primitive, bool exclusive) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_CumSum();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateCumSum(fbb, exclusive, value->reverse());
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_CUM_SUM), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}

bool MindIR_CumSum_GetReverse(ConstPrimitivePtr primitive) {
  if (primitive != nullptr) {
    auto prim = static_cast<const schema::Primitive *>(primitive);
    auto value = prim->value_as_CumSum();
    if (prim != nullptr && value != nullptr) {
      return value->reverse();
    } else {
      return false;
    }
  } else {
    return false;
  }
}

void MindIR_CumSum_SetReverse(PrimitivePtr *primitive, bool reverse) {
  if (primitive != nullptr && *primitive != nullptr) {
    auto prim = static_cast<schema::Primitive *>(*primitive);
    auto value = prim->value_as_CumSum();
    if (prim != nullptr && value != nullptr) {
      flatbuffers::FlatBufferBuilder fbb;
      auto ops_offset = schema::CreateCumSum(fbb, value->exclusive(), reverse);
      auto prim_offset = schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(NODE_TYPE_CUM_SUM), ops_offset.o);
      fbb.Finish(prim_offset);
      auto new_addr = MindIRMemoryManager::GetInstance()->CreatePrimitiveFromBuilder(fbb, nullptr);
      auto ret_value = flatbuffers::GetMutableRoot<schema::Primitive>(new_addr);
      free(*primitive);
      *primitive = ret_value;
    }
  }
}
}  // namespace lite
}  // namespace mindspore
