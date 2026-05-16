/**
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.objrg/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "coder/generator/component/cmake_component.h"
#include <set>
#include <memory>

namespace mindspore::lite::micro {
void CodeCMakeNetLibrary(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx, const Configurator *config) {
  ofs << "include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include/)\n";
  ofs << "include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../)\n";
  if (config->target() == kCortex_M) {
    // cmsis is not supported and currently disabled in the current version.
  }
  ofs << "set(OP_SRC\n";
  for (const std::string &c_file : ctx->c_files()) {
    ofs << "    " << c_file << ".obj\n";
  }
  for (int i = 0; i <= ctx->GetCurModelIndex(); ++i) {
    ofs << "    weight" << i << ".c.obj\n"
        << "    net" << i << ".c.obj\n"
        << "    model" << i << ".c.obj\n";
  }
  ofs << "    model.c.obj\n"
      << "    context.c.obj\n"
      << "    tensor.c.obj\n";
  if (config->target() != kCortex_M && !config->dynamic_shape()) {
    ofs << "    allocator.c.obj\n";
  }
  if (config->debug_mode()) {
    ofs << "    debug_utils.c.obj\n";
  }
  if (config->support_parallel()) {
    ofs << "    micro_core_affinity.c.obj\n"
           "    micro_thread_pool.c.obj\n";
  }
  ofs << ")\n";
  std::set<std::string> kernel_cmake_asm_set_files = ctx->asm_files();
  if (!kernel_cmake_asm_set_files.empty() && (config->target() == kARM32 || config->target() == kARM64)) {
    ofs << "set(ASSEMBLY_SRC\n";
    for (const std::string &asm_file : kernel_cmake_asm_set_files) {
      ofs << "    " << asm_file << ".obj\n";
    }
    ofs << ")\n"
        << "set_property(SOURCE ${ASSEMBLY_SRC} PROPERTY LANGUAGE C)\n"
        << "list(APPEND OP_SRC ${ASSEMBLY_SRC})\n";
  }
  ofs << "file(GLOB_RECURSE NET_SRC\n"
         "     ${CMAKE_CURRENT_SOURCE_DIR}/*.cc\n"
         "     ${CMAKE_CURRENT_SOURCE_DIR}/*.c\n"
         "     )\n"
         "add_library(net STATIC ${NET_SRC})\n";
}
}  // namespace mindspore::lite::micro
