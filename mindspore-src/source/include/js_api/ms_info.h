/**
 * Copyright (C) 2023 Huawei Device Co., Ltd.
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
#ifndef MINDSPORE_INCLUDE_JS_API_MS_INFO_H
#define MINDSPORE_INCLUDE_JS_API_MS_INFO_H

namespace mindspore {
enum InterruptType {
  INTERRUPT_TYPE_BEGIN = 1,
  INTERRUPT_TYPE_END = 2,
};

enum InterruptHint {
  INTERRUPT_HINT_NONE = 0,
  INTERRUPT_HINT_RESUME,
  INTERRUPT_HINT_PAUSE,
  INTERRUPT_HINT_STOP,
  INTERRUPT_HINT_DUCK,
  INTERRUPT_HINT_UNDUCK
};

enum InterruptForceType {
  /**
   * Force type, system change audio state.
   */
  INTERRUPT_FORCE = 0,
  /**
   * Share type, application change audio state.
   */
  INTERRUPT_SHARE
};

struct InterruptEvent {
  /**
   * Interrupt event type, begin or end
   */
  InterruptType eventType;
  /**
   * Interrupt force type, force or share
   */
  InterruptForceType forceType;
  /**
   * Interrupt hint type. In force type, the audio state already changed,
   * but in share mode, only provide a hint for application to decide.
   */
  InterruptHint hintType;
};

// Used internally only by AudioFramework
struct InterruptEventInternal {
  InterruptType eventType;
  InterruptForceType forceType;
  InterruptHint hintType;
  float duckVolume;
};

}  // namespace mindspore
#endif  // MS_INFO_H