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
#include "tools/converter/parser/third_party/third_party_model_parser.h"
#include <string>
#include <vector>
#include <memory>
#include "ir/value.h"
#include "mindapi/base/type_id.h"
#include "src/common/log_util.h"
#include "src/common/file_utils.h"
#include "nnacl_c/op_base.h"
#include "ops/primitive_c.h"
#include "mindspore/ops/infer/custom.h"
#include "mindspore/ops/infer/tuple_get_item.h"
#include "mindspore/ops/infer/make_tuple.h"
#include "mindspore/ops/infer/return.h"
#include "tools/converter/config_parser/config_file_parser.h"
#include "include/registry/model_parser_registry.h"
#include "tools/common/graph_util.h"
#include "tools/common/tensor_util.h"
#include "tools/converter/converter_context.h"
#include "tools/converter/parser/lite_model_parser_creator.h"

using mindspore::converter::kFmkTypeThirdParty;

namespace mindspore {
namespace lite {
api::FuncGraphPtr ThirdPartyModelParser::Parse(const converter::ConverterParameters &flag) {
  model_file_ = flag.model_file;
  auto &attrs = flag.attrs;
  auto iter = attrs.find("config_file");
  if (iter == attrs.end()) {
    return nullptr;
  }
  auto config_file = iter->second;

  auto ret = InitConfig(config_file);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init config for third party model parsing failed";
    return nullptr;
  }

  return CreateFuncGraph();
}

STATUS ThirdPartyModelParser::InitConfig(const std::string &config_file) {
  lite::ConfigFileParser config_parser;
  if (config_file.empty()) {
    MS_LOG(ERROR) << "Missing config file in converting third party model";
    return RET_ERROR;
  }
  auto ret = config_parser.ParseConfigFile(config_file, nullptr);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Get third party model section from config file failed";
    return RET_ERROR;
  }

  ret = ThirdPartyParamParser::Parse(config_parser.GetThirdPartyModelString(), &param_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Parse third party model param failed.";
    return ret;
  }
  return RET_OK;
}

api::FuncGraphPtr ThirdPartyModelParser::CreateFuncGraph() {
  auto func_graph = std::make_shared<FuncGraph>();
  MS_CHECK_TRUE_RET(func_graph != nullptr, nullptr);
  auto type_value = MakeValue(static_cast<int>(converter::kFmkTypeThirdParty));
  MS_CHECK_TRUE_RET(type_value != nullptr, nullptr);
  func_graph->set_attr("fmk", type_value);
  auto attr_value = MakeValue("third_party");
  MS_CHECK_TRUE_RET(attr_value != nullptr, nullptr);
  func_graph->set_attr("graph_name", attr_value);

  std::vector<AnfNodePtr> input_nodes = {};
  auto ret = BuildGraphInputs(func_graph, &input_nodes);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Create func graph input nodes failed";
    return nullptr;
  }

  CNodePtr custom_node = nullptr;
  ret = BuildCustomOp(func_graph, input_nodes, &custom_node);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Create func graph custom op node failed";
    return nullptr;
  }

  ret = BuildGraphOutputs(func_graph, custom_node);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Create func graph output nodes failed";
    return nullptr;
  }

  static auto manager = Manage(func_graph);
  func_graph->set_manager(manager);

  auto result_graph = api::MakeShared<api::FuncGraph>(func_graph);
  return result_graph;
}

STATUS ThirdPartyModelParser::BuildGraphInputs(const FuncGraphPtr &func_graph, std::vector<AnfNodePtr> *op_inputs) {
  MS_ASSERT(anf_node_map != nullptr && func_graph != nullptr);
  auto &dtypes = param_.input_dtypes;
  auto &shapes = param_.input_shapes;
  auto &names = param_.input_names;

  auto input_size = dtypes.size();

  // Create parameter nodes for graph inputs
  for (size_t i = 0; i < input_size; i++) {
    auto parameter = func_graph->add_parameter();
    MSLITE_CHECK_PTR(parameter);
    auto abstract_tensor = CreateTensorAbstract(shapes[i], dtypes[i]);
    if (abstract_tensor == nullptr) {
      MS_LOG(ERROR) << "Create tensor abstract failed";
      return RET_ERROR;
    }
    parameter->set_abstract(abstract_tensor);
    parameter->set_name(names[i]);
    op_inputs->push_back(parameter);
  }

  // Create parameter nodes for const tensor which wrapped third model buffer.
  size_t model_size = 0U;
  auto model_data = ReadFile(model_file_.c_str(), &model_size);
  std::vector<int64_t> model_shape = {static_cast<int64_t>(model_size)};
  auto tensor_info = CreateTensorInfo(nullptr, 0, model_shape, kNumberTypeUInt8);
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "init tensor info failed";
    delete model_data;
    return RET_NULL_PTR;
  }
  auto tensor_data = reinterpret_cast<uint8_t *>(tensor_info->data_c());
  if (memcpy_s(tensor_data, tensor_info->Size(), model_data, model_size) != EOK) {
    MS_LOG(ERROR) << "memcpy failed.";
    delete model_data;
    return RET_ERROR;
  }
  delete model_data;
  auto parameter = func_graph->add_parameter();
  MSLITE_CHECK_PTR(parameter);
  auto status = InitParameterFromTensorInfo(parameter, tensor_info);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "init parameter from tensor info failed.";
    return RET_ERROR;
  }
  parameter->set_name("ThirdPartyModel");
  op_inputs->push_back(parameter);
  return RET_OK;
}

STATUS ThirdPartyModelParser::BuildCustomOp(const FuncGraphPtr &func_graph, const std::vector<AnfNodePtr> &op_inputs,
                                            CNodePtr *operator_node) {
  MS_ASSERT(anf_node_map != nullptr && func_graph != nullptr);
  NotSupportOp::GetInstance()->set_fmk_type("THIRDPARTY");
  STATUS status = RET_OK;

  // create primitive and build CNode of CUSTOM operator
  ops::PrimitiveCPtr primitive_c;
  auto prim = std::make_unique<ops::Custom>();
  MS_CHECK_TRUE_RET(prim != nullptr, RET_ERROR);
  prim->set_type("ThirdPartyModel");

  const auto &attr = param_.extended_parameters;
  prim->set_attr(attr);
  primitive_c = prim->GetPrim();
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "failed to create primitive: custom";
    return RET_ERROR;
  }

  auto operator_cnode = func_graph->NewCNode(primitive_c, op_inputs);
  MSLITE_CHECK_PTR(operator_cnode);
  operator_cnode->set_fullname_with_scope("Custom");
  *operator_node = operator_cnode;
  return status;
}

STATUS ThirdPartyModelParser::BuildGraphOutputs(const FuncGraphPtr &func_graph, const CNodePtr &operator_node) {
  MS_ASSERT(anf_node_map != nullptr && func_graph != nullptr);

  auto dtypes = param_.output_dtypes;
  auto shapes = param_.output_shapes;
  auto names = param_.output_names;

  auto output_size = dtypes.size();
  std::vector<AnfNodePtr> output_nodes = {};

  // Use TupleGetItem to wrap op outputs.
  AbstractBasePtrList abstract_list;
  for (size_t i = 0; i < output_size; i++) {
    auto abstract_tensor = CreateTensorAbstract(shapes[i], dtypes[i]);
    if (abstract_tensor == nullptr) {
      MS_LOG(ERROR) << "Create tensor abstract failed";
      return RET_ERROR;
    }
    abstract_list.emplace_back(abstract_tensor);
    auto tuple_get_item_prim_ptr = std::make_shared<ops::TupleGetItem>();
    if (tuple_get_item_prim_ptr == nullptr) {
      MS_LOG(ERROR) << "new TupleGetItem failed";
      return RET_NULL_PTR;
    }
    auto tuple_get_item_prim_c = tuple_get_item_prim_ptr->GetPrim();
    MSLITE_CHECK_PTR(tuple_get_item_prim_c);
    auto tuple_get_item_prim = NewValueNode(tuple_get_item_prim_c);
    MSLITE_CHECK_PTR(tuple_get_item_prim);
    auto get_item_value = NewValueNode(MakeValue<int>(i));
    MSLITE_CHECK_PTR(get_item_value);
    std::vector<AnfNodePtr> inputs = {tuple_get_item_prim, operator_node, get_item_value};
    CNodePtr get_item_cnode = func_graph->NewCNode(inputs);
    MSLITE_CHECK_PTR(get_item_cnode);
    std::string output_item_name = operator_node->fullname_with_scope() + "_getitem_" + std::to_string(i);
    auto get_item_abstract = CreateTensorAbstract({}, kNumberTypeFloat32);
    if (get_item_abstract == nullptr) {
      MS_LOG(ERROR) << "Create tensor abstarct failed";
      return RET_ERROR;
    }
    get_item_cnode->set_fullname_with_scope(output_item_name);
    get_item_cnode->set_abstract(get_item_abstract);
    output_nodes.push_back(get_item_cnode);
  }
  auto abstract_tuple = std::make_shared<abstract::AbstractTuple>(abstract_list);
  MSLITE_CHECK_PTR(abstract_tuple);
  operator_node->set_abstract(abstract_tuple);

  // Use MakeTuple node to wrap all outputs as single input of Return node.
  auto make_tuple_prim_ptr = std::make_shared<ops::MakeTuple>();
  if (make_tuple_prim_ptr == nullptr) {
    MS_LOG(ERROR) << "new MakeTuple failed";
    return RET_NULL_PTR;
  }
  auto make_tuple_prim_c = make_tuple_prim_ptr->GetPrim();
  MSLITE_CHECK_PTR(make_tuple_prim_c);
  auto make_tuple_prim = NewValueNode(make_tuple_prim_c);
  MSLITE_CHECK_PTR(make_tuple_prim);
  std::vector<AnfNodePtr> make_tuple_inputs = output_nodes;
  make_tuple_inputs.insert(make_tuple_inputs.begin(), make_tuple_prim);
  auto make_tuple_cnode = func_graph->NewCNode(make_tuple_inputs);
  MSLITE_CHECK_PTR(make_tuple_cnode);
  make_tuple_cnode->set_fullname_with_scope("return_tuple");

  auto return_prim_ptr = std::make_shared<ops::Return>();
  if (return_prim_ptr == nullptr) {
    MS_LOG(ERROR) << "new Return failed";
    return RET_NULL_PTR;
  }
  auto return_prim_c = return_prim_ptr->GetPrim();
  MSLITE_CHECK_PTR(return_prim_c);
  std::vector<AnfNodePtr> op_inputs{make_tuple_cnode};
  auto cnode = func_graph->NewCNode(return_prim_c, op_inputs);
  MSLITE_CHECK_PTR(cnode);
  cnode->set_fullname_with_scope("Return");
  func_graph->set_return(cnode);

  // Save original output tensor names.
  ConverterInnerContext::GetInstance()->SetGraphOutputTensorNames(names);
  return RET_OK;
}

REG_MODEL_PARSER(kFmkTypeThirdParty, LiteModelParserCreator<ThirdPartyModelParser>)
}  // namespace lite
}  // namespace mindspore
