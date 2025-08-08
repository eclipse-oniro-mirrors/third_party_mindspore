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

#include "gtest/gtest.h"
#include "tools/converter/config_parser/third_party_param_parser.h"

using mindspore::ThirdPartyModelParam;
using mindspore::TypeId;
using mindspore::lite::RET_OK;
using mindspore::lite::ThirdPartyModelString;
using mindspore::lite::ThirdPartyParamParser;

const ThirdPartyModelString kDemoSISOParam = {
  // SISO is short for single-input-single-output.
  .input_dtypes = "float32",
  .input_shapes = "1,2,3,4",
  .input_names = "siso_input",
  .output_dtypes = "int32",
  .output_shapes = "2",
  .output_names = "siso_output",
  .extended_parameters = "siso_foo:siso_foo_value;siso_bar:siso_bar_value",
};

const ThirdPartyModelString kDemoMIMOParam = {
  // MIMO is short for multiple-input-multiple-output.
  .input_dtypes = "float32;int8;float16",
  .input_shapes = "1,2,3,4;5,6;7,8,9",
  .input_names = "mimo_in_0;mimo_in_1;mimo_in_2",
  .output_dtypes = "int32;float32",
  .output_shapes = "2,4;10,20,30",
  .output_names = "mimo_out_0;mimo_out_1",
  .extended_parameters = "mimo_foo:mimo_foo_value;mimo_bar:mimo_bar_value",
};

TEST(TestThirdPartyParamParser, ParseSISOParam) {
  ThirdPartyModelString param_string = kDemoSISOParam;
  ThirdPartyModelParam result;
  ASSERT_EQ(ThirdPartyParamParser::Parse(param_string, &result), RET_OK);

  ASSERT_EQ(result.input_names, std::vector<std::string>{"siso_input"});
  ASSERT_EQ(result.input_shapes.size(), 1U);
  std::vector<int64_t> expect_in_shape = {1, 2, 3, 4};
  ASSERT_EQ(result.input_shapes[0], expect_in_shape);
  ASSERT_EQ(result.input_dtypes, std::vector<TypeId>{TypeId::kNumberTypeFloat32});

  ASSERT_EQ(result.output_names, std::vector<std::string>{"siso_output"});
  ASSERT_EQ(result.output_shapes.size(), 1U);
  std::vector<int64_t> expect_out_shape = {2};
  ASSERT_EQ(result.output_shapes[0], expect_out_shape);
  ASSERT_EQ(result.output_dtypes, std::vector<TypeId>{TypeId::kNumberTypeInt32});

  const auto &ext_param = result.extended_parameters;
  ASSERT_EQ(ext_param.size(), 2U);
  ASSERT_TRUE(ext_param.find("siso_foo") != ext_param.end());
  auto expect_foo_value = ext_param.at("siso_foo");
  ASSERT_EQ(std::string(expect_foo_value.begin(), expect_foo_value.end()), "siso_foo_value");
  ASSERT_TRUE(ext_param.find("siso_bar") != ext_param.end());
  auto expect_bar_value = ext_param.at("siso_bar");
  ASSERT_EQ(std::string(expect_bar_value.begin(), expect_bar_value.end()), "siso_bar_value");
}

TEST(TestThirdPartyParamParser, ParseValidDtype) {
  ThirdPartyModelString param_string = kDemoSISOParam;
  const std::vector<std::string> kValidDtypeStrings = {
    "float64", "float32", "float16", "int64", "int32", "int16", "int8", "uint8", "bool",
  };

  const std::vector<TypeId> kExpects = {
    TypeId::kNumberTypeFloat64, TypeId::kNumberTypeFloat32, TypeId::kNumberTypeFloat16,
    TypeId::kNumberTypeInt64,   TypeId::kNumberTypeInt32,   TypeId::kNumberTypeInt16,
    TypeId::kNumberTypeInt8,    TypeId::kNumberTypeUInt8,   TypeId::kNumberTypeBool};

  for (size_t i = 0; i < kValidDtypeStrings.size(); i++) {
    param_string.input_dtypes = kValidDtypeStrings[i];
    ThirdPartyModelParam result;
    ASSERT_EQ(ThirdPartyParamParser::Parse(param_string, &result), RET_OK);
    ASSERT_EQ(result.input_dtypes[0], kExpects[i]);
  }
}

TEST(TestThirdPartyParamParser, ParseInvalidDtype) {
  ThirdPartyModelParam result;
  ThirdPartyModelString param_string = kDemoSISOParam;
  ASSERT_EQ(ThirdPartyParamParser::Parse(param_string, &result), RET_OK);
  param_string.input_dtypes = "bad_dtype";
  ASSERT_NE(ThirdPartyParamParser::Parse(param_string, &result), RET_OK);
}

TEST(TestThirdPartyParamParser, ParseValidShape) {
  ThirdPartyModelString param_string = kDemoSISOParam;
  param_string.input_shapes = "256,256,1024,96";  // Only support fixed shape.
  ThirdPartyModelParam result;
  ASSERT_EQ(ThirdPartyParamParser::Parse(param_string, &result), RET_OK);
  std::vector<int64_t> expect = {256, 256, 1024, 96};
  ASSERT_EQ(result.input_shapes[0], expect);
}

TEST(TestThirdPartyParamParser, ParseInvalidShape) {
  ThirdPartyModelParam result;
  ThirdPartyModelString param_string = kDemoSISOParam;
  ASSERT_EQ(ThirdPartyParamParser::Parse(param_string, &result), RET_OK);

  param_string.input_shapes = "256,256,1024,-1";
  ASSERT_NE(ThirdPartyParamParser::Parse(param_string, &result), RET_OK);

  param_string.input_shapes = "256,256,0,96";
  ASSERT_NE(ThirdPartyParamParser::Parse(param_string, &result), RET_OK);

  param_string.input_shapes = "256,-256,1024,96";
  ASSERT_NE(ThirdPartyParamParser::Parse(param_string, &result), RET_OK);

  param_string.input_shapes = "256,foo,1024,96";
  ASSERT_NE(ThirdPartyParamParser::Parse(param_string, &result), RET_OK);
}

TEST(TestThirdPartyParamParser, ParseDefaultName) {
  ThirdPartyModelParam result;
  ThirdPartyModelString param_string = kDemoSISOParam;
  param_string.input_names = "";
  param_string.output_names = "";
  ASSERT_EQ(ThirdPartyParamParser::Parse(param_string, &result), RET_OK);
  ASSERT_EQ(result.input_names[0], "in_0");
  ASSERT_EQ(result.output_names[0], "out_0");
}

TEST(TestThirdPartyParamParser, ParseMIMOParam) {
  ThirdPartyModelString param_string = kDemoMIMOParam;
  ThirdPartyModelParam result;
  ASSERT_EQ(ThirdPartyParamParser::Parse(param_string, &result), RET_OK);

  std::vector<std::string> expect_input_names = {"mimo_in_0", "mimo_in_1", "mimo_in_2"};
  ASSERT_EQ(result.input_names, expect_input_names);
  std::vector<std::vector<int64_t>> expect_input_shapes = {{1, 2, 3, 4}, {5, 6}, {7, 8, 9}};
  ASSERT_EQ(result.input_shapes, expect_input_shapes);
  std::vector<TypeId> expect_input_dtypes = {TypeId::kNumberTypeFloat32, TypeId::kNumberTypeInt8,
                                             TypeId::kNumberTypeFloat16};
  ASSERT_EQ(result.input_dtypes, expect_input_dtypes);

  std::vector<std::string> expect_output_names = {"mimo_out_0", "mimo_out_1"};
  ASSERT_EQ(result.output_names, expect_output_names);
  std::vector<std::vector<int64_t>> expect_output_shapes = {{2, 4}, {10, 20, 30}};
  ASSERT_EQ(result.output_shapes, expect_output_shapes);
  std::vector<TypeId> expect_output_dtypes = {TypeId::kNumberTypeInt32, TypeId::kNumberTypeFloat32};
  ASSERT_EQ(result.output_dtypes, expect_output_dtypes);
}

TEST(TestThirdPartyParamParser, ParseMismatchedShapeAndDtypeSize) {
  ThirdPartyModelString param_string = kDemoMIMOParam;
  ThirdPartyModelParam result;
  ASSERT_EQ(ThirdPartyParamParser::Parse(param_string, &result), RET_OK);

  param_string.input_shapes = "1,2,3,4;5,6";  // shape size is 2 while dtype size is 3.
  ASSERT_NE(ThirdPartyParamParser::Parse(param_string, &result), RET_OK);
}

TEST(TestThirdPartyParamParser, ParseMismatchedNameAndDtypeSize) {
  ThirdPartyModelString param_string = kDemoMIMOParam;
  ThirdPartyModelParam result;
  ASSERT_EQ(ThirdPartyParamParser::Parse(param_string, &result), RET_OK);

  param_string.input_names = "mimo_in_0;mimo_in_1";  // name size is 2 while dtype size is 3.
  ASSERT_NE(ThirdPartyParamParser::Parse(param_string, &result), RET_OK);
}
