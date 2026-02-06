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
#ifndef MINDSPORE_LITE_MINDIR_H
#define MINDSPORE_LITE_MINDIR_H
#include "mindir_types.h"
#include "mindir_lite_graph.h"
#include "mindir_tensor.h"
#include "mindir_primitive.h"

namespace mindspore {
namespace schema {
struct Attribute;
}

namespace lite {
// ********** Activation **********
PrimitivePtr MindIR_Activation_CreatePrimitive(ActivationType activation_type, float alpha, float min_val,
                                               float max_val, bool approximate);
ActivationType MindIR_Activation_GetActivationType(ConstPrimitivePtr primitive);
void MindIR_Activation_SetActivationType(PrimitivePtr *primitive, ActivationType activation_type);
float MindIR_Activation_GetAlpha(ConstPrimitivePtr primitive);
void MindIR_Activation_SetAlpha(PrimitivePtr *primitive, float alpha);
float MindIR_Activation_GetMinVal(ConstPrimitivePtr primitive);
void MindIR_Activation_SetMinVal(PrimitivePtr *primitive, float min_val);
float MindIR_Activation_GetMaxVal(ConstPrimitivePtr primitive);
void MindIR_Activation_SetMaxVal(PrimitivePtr *primitive, float max_val);
bool MindIR_Activation_GetApproximate(ConstPrimitivePtr primitive);
void MindIR_Activation_SetApproximate(PrimitivePtr *primitive, bool approximate);

// ********** AddFusion **********
PrimitivePtr MindIR_AddFusion_CreatePrimitive(ActivationType activation_type);
ActivationType MindIR_AddFusion_GetActivationType(ConstPrimitivePtr primitive);
void MindIR_AddFusion_SetActivationType(PrimitivePtr *primitive, ActivationType activation_type);

// ********** ArgMaxFusion **********
PrimitivePtr MindIR_ArgMaxFusion_CreatePrimitive(int64_t axis, int64_t top_k, bool keep_dims, bool out_max_value);
int64_t MindIR_ArgMaxFusion_GetAxis(ConstPrimitivePtr primitive);
void MindIR_ArgMaxFusion_SetAxis(PrimitivePtr *primitive, int64_t axis);
int64_t MindIR_ArgMaxFusion_GetTopK(ConstPrimitivePtr primitive);
void MindIR_ArgMaxFusion_SetTopK(PrimitivePtr *primitive, int64_t top_k);
bool MindIR_ArgMaxFusion_GetKeepDims(ConstPrimitivePtr primitive);
void MindIR_ArgMaxFusion_SetKeepDims(PrimitivePtr *primitive, bool keep_dims);
bool MindIR_ArgMaxFusion_GetOutMaxValue(ConstPrimitivePtr primitive);
void MindIR_ArgMaxFusion_SetOutMaxValue(PrimitivePtr *primitive, bool out_max_value);

// ********** AvgPoolFusion **********
PrimitivePtr MindIR_AvgPoolFusion_CreatePrimitive(const std::vector<int64_t> &kernel_size,
                                                  const std::vector<int64_t> &strides, const std::vector<int64_t> &pad,
                                                  PadMode pad_mode, RoundMode round_mode, Format format, bool global,
                                                  ActivationType activation_type);
std::vector<int64_t> MindIR_AvgPoolFusion_GetKernelSize(ConstPrimitivePtr primitive);
void MindIR_AvgPoolFusion_SetKernelSize(PrimitivePtr *primitive, const std::vector<int64_t> &kernel_size);
std::vector<int64_t> MindIR_AvgPoolFusion_GetStrides(ConstPrimitivePtr primitive);
void MindIR_AvgPoolFusion_SetStrides(PrimitivePtr *primitive, const std::vector<int64_t> &strides);
std::vector<int64_t> MindIR_AvgPoolFusion_GetPad(ConstPrimitivePtr primitive);
void MindIR_AvgPoolFusion_SetPad(PrimitivePtr *primitive, const std::vector<int64_t> &pad);
PadMode MindIR_AvgPoolFusion_GetPadMode(ConstPrimitivePtr primitive);
void MindIR_AvgPoolFusion_SetPadMode(PrimitivePtr *primitive, PadMode pad_mode);
RoundMode MindIR_AvgPoolFusion_GetRoundMode(ConstPrimitivePtr primitive);
void MindIR_AvgPoolFusion_SetRoundMode(PrimitivePtr *primitive, RoundMode round_mode);
Format MindIR_AvgPoolFusion_GetFormat(ConstPrimitivePtr primitive);
void MindIR_AvgPoolFusion_SetFormat(PrimitivePtr *primitive, Format format);
bool MindIR_AvgPoolFusion_GetGlobal(ConstPrimitivePtr primitive);
void MindIR_AvgPoolFusion_SetGlobal(PrimitivePtr *primitive, bool global);
ActivationType MindIR_AvgPoolFusion_GetActivationType(ConstPrimitivePtr primitive);
void MindIR_AvgPoolFusion_SetActivationType(PrimitivePtr *primitive, ActivationType activation_type);

// ********** BatchToSpaceND **********
PrimitivePtr MindIR_BatchToSpaceND_CreatePrimitive(const std::vector<int64_t> &block_shape,
                                                   const std::vector<std::vector<int64_t>> &crops);
std::vector<int64_t> MindIR_BatchToSpaceND_GetBlockShape(ConstPrimitivePtr primitive);
void MindIR_BatchToSpaceND_SetBlockShape(PrimitivePtr *primitive, const std::vector<int64_t> &block_shape);
std::vector<std::vector<int64_t>> MindIR_BatchToSpaceND_GetCrops(ConstPrimitivePtr primitive);
void MindIR_BatchToSpaceND_SetCrops(PrimitivePtr *primitive, const std::vector<std::vector<int64_t>> &crops);

// ********** BiasAdd **********
PrimitivePtr MindIR_BiasAdd_CreatePrimitive();

// ********** Cast **********
PrimitivePtr MindIR_Cast_CreatePrimitive();

// ********** Concat **********
PrimitivePtr MindIR_Concat_CreatePrimitive(int64_t axis);
int64_t MindIR_Concat_GetAxis(ConstPrimitivePtr primitive);
void MindIR_Concat_SetAxis(PrimitivePtr *primitive, int64_t axis);

// ********** Conv2DFusion **********
PrimitivePtr MindIR_Conv2DFusion_CreatePrimitive(const std::vector<int64_t> &kernel_size,
                                                 const std::vector<int64_t> &stride,
                                                 const std::vector<int64_t> &dilation, PadMode pad_mode,
                                                 const std::vector<int64_t> &pad_list, int64_t group,
                                                 int64_t in_channel, int64_t out_channel,
                                                 ActivationType activation_type);
std::vector<int64_t> MindIR_Conv2DFusion_GetKernelSize(ConstPrimitivePtr primitive);
void MindIR_Conv2DFusion_SetKernelSize(PrimitivePtr *primitive, const std::vector<int64_t> &kernel_size);
std::vector<int64_t> MindIR_Conv2DFusion_GetStride(ConstPrimitivePtr primitive);
void MindIR_Conv2DFusion_SetStride(PrimitivePtr *primitive, const std::vector<int64_t> &stride);
std::vector<int64_t> MindIR_Conv2DFusion_GetDilation(ConstPrimitivePtr primitive);
void MindIR_Conv2DFusion_SetDilation(PrimitivePtr *primitive, const std::vector<int64_t> &dilation);
PadMode MindIR_Conv2DFusion_GetPadMode(ConstPrimitivePtr primitive);
void MindIR_Conv2DFusion_SetPadMode(PrimitivePtr *primitive, PadMode pad_mode);
std::vector<int64_t> MindIR_Conv2DFusion_GetPadList(ConstPrimitivePtr primitive);
void MindIR_Conv2DFusion_SetPadList(PrimitivePtr *primitive, const std::vector<int64_t> &pad_list);
int64_t MindIR_Conv2DFusion_GetGroup(ConstPrimitivePtr primitive);
void MindIR_Conv2DFusion_SetGroup(PrimitivePtr *primitive, int64_t group);
int64_t MindIR_Conv2DFusion_GetInChannel(ConstPrimitivePtr primitive);
void MindIR_Conv2DFusion_SetInChannel(PrimitivePtr *primitive, int64_t in_channel);
int64_t MindIR_Conv2DFusion_GetOutChannel(ConstPrimitivePtr primitive);
void MindIR_Conv2DFusion_SetOutChannel(PrimitivePtr *primitive, int64_t out_channel);
ActivationType MindIR_Conv2DFusion_GetActivationType(ConstPrimitivePtr primitive);
void MindIR_Conv2DFusion_SetActivationType(PrimitivePtr *primitive, ActivationType activation_type);
Format MindIR_Conv2DFusion_GetFormat(ConstPrimitivePtr primitive);
void MindIR_Conv2DFusion_SetFormat(PrimitivePtr *primitive, Format format);

// ********** Conv2dTransposeFusion **********
PrimitivePtr MindIR_Conv2dTransposeFusion_CreatePrimitive(
  const std::vector<int64_t> &kernel_size, const std::vector<int64_t> &stride, const std::vector<int64_t> &dilation,
  PadMode pad_mode, const std::vector<int64_t> &pad_list, int64_t group, int64_t in_channel, int64_t out_channel,
  ActivationType activation_type, const std::vector<int64_t> &output_paddings);
std::vector<int64_t> MindIR_Conv2dTransposeFusion_GetKernelSize(ConstPrimitivePtr primitive);
void MindIR_Conv2dTransposeFusion_SetKernelSize(PrimitivePtr *primitive, const std::vector<int64_t> &kernel_size);
std::vector<int64_t> MindIR_Conv2dTransposeFusion_GetStride(ConstPrimitivePtr primitive);
void MindIR_Conv2dTransposeFusion_SetStride(PrimitivePtr *primitive, const std::vector<int64_t> &stride);
std::vector<int64_t> MindIR_Conv2dTransposeFusion_GetDilation(ConstPrimitivePtr primitive);
void MindIR_Conv2dTransposeFusion_SetDilation(PrimitivePtr *primitive, const std::vector<int64_t> &dilation);
PadMode MindIR_Conv2dTransposeFusion_GetPadMode(ConstPrimitivePtr primitive);
void MindIR_Conv2dTransposeFusion_SetPadMode(PrimitivePtr *primitive, PadMode pad_mode);
std::vector<int64_t> MindIR_Conv2dTransposeFusion_GetPadList(ConstPrimitivePtr primitive);
void MindIR_Conv2dTransposeFusion_SetPadList(PrimitivePtr *primitive, const std::vector<int64_t> &pad_list);
int64_t MindIR_Conv2dTransposeFusion_GetGroup(ConstPrimitivePtr primitive);
void MindIR_Conv2dTransposeFusion_SetGroup(PrimitivePtr *primitive, int64_t group);
int64_t MindIR_Conv2dTransposeFusion_GetInChannel(ConstPrimitivePtr primitive);
void MindIR_Conv2dTransposeFusion_SetInChannel(PrimitivePtr *primitive, int64_t in_channel);
int64_t MindIR_Conv2dTransposeFusion_GetOutChannel(ConstPrimitivePtr primitive);
void MindIR_Conv2dTransposeFusion_SetOutChannel(PrimitivePtr *primitive, int64_t out_channel);
ActivationType MindIR_Conv2dTransposeFusion_GetActivationType(ConstPrimitivePtr primitive);
void MindIR_Conv2dTransposeFusion_SetActivationType(PrimitivePtr *primitive, ActivationType activation_type);
std::vector<int64_t> MindIR_Conv2dTransposeFusion_GetOutputPaddings(ConstPrimitivePtr primitive);
void MindIR_Conv2dTransposeFusion_SetOutputPaddings(PrimitivePtr *primitive,
                                                    const std::vector<int64_t> &output_paddings);

// ********** DivFusion **********
PrimitivePtr MindIR_DivFusion_CreatePrimitive(ActivationType activation_type);
ActivationType MindIR_DivFusion_GetActivationType(ConstPrimitivePtr primitive);
void MindIR_DivFusion_SetActivationType(PrimitivePtr *primitive, ActivationType activation_type);

// ********** Eltwise **********
PrimitivePtr MindIR_Eltwise_CreatePrimitive(EltwiseMode mode);
EltwiseMode MindIR_Eltwise_GetMode(ConstPrimitivePtr primitive);
void MindIR_Eltwise_SetMode(PrimitivePtr *primitive, EltwiseMode mode);

// ********** ExpandDims **********
PrimitivePtr MindIR_ExpandDims_CreatePrimitive();

// ********** Fill **********
PrimitivePtr MindIR_Fill_CreatePrimitive();

// ********** FullConnection **********
PrimitivePtr MindIR_FullConnection_CreatePrimitive(bool has_bias, bool use_axis, int64_t axis,
                                                   ActivationType activation_type);
bool MindIR_FullConnection_GetHasBias(ConstPrimitivePtr primitive);
void MindIR_FullConnection_SetHasBias(PrimitivePtr *primitive, bool has_bias);
bool MindIR_FullConnection_GetUseAxis(ConstPrimitivePtr primitive);
void MindIR_FullConnection_SetUseAxis(PrimitivePtr *primitive, bool use_axis);
int64_t MindIR_FullConnection_GetAxis(ConstPrimitivePtr primitive);
void MindIR_FullConnection_SetAxis(PrimitivePtr *primitive, int64_t axis);
ActivationType MindIR_FullConnection_GetActivationType(ConstPrimitivePtr primitive);
void MindIR_FullConnection_SetActivationType(PrimitivePtr *primitive, ActivationType activation_type);

// ********** FusedBatchNorm **********
PrimitivePtr MindIR_FusedBatchNorm_CreatePrimitive(float epsilon);
float MindIR_FusedBatchNorm_GetEpsilon(ConstPrimitivePtr primitive);
void MindIR_FusedBatchNorm_SetEpsilon(PrimitivePtr *primitive, float epsilon);

// ********** Gather **********
PrimitivePtr MindIR_Gather_CreatePrimitive();

// ********** LayerNormFusion **********
PrimitivePtr MindIR_LayerNormFusion_CreatePrimitive(int64_t begin_norm_axis, float epsilon, bool elementwise_affine,
                                                    int64_t begin_params_axis);
int64_t MindIR_LayerNormFusion_GetBeginNormAxis(ConstPrimitivePtr primitive);
void MindIR_LayerNormFusion_SetBeginNormAxis(PrimitivePtr *primitive, int64_t begin_norm_axis);
float MindIR_LayerNormFusion_GetEpsilon(ConstPrimitivePtr primitive);
void MindIR_LayerNormFusion_SetEpsilon(PrimitivePtr *primitive, float epsilon);
bool MindIR_LayerNormFusion_GetElementwiseAffine(ConstPrimitivePtr primitive);
void MindIR_LayerNormFusion_SetElementwiseAffine(PrimitivePtr *primitive, bool elementwise_affine);
int64_t MindIR_LayerNormFusion_GetBeginParamsAxis(ConstPrimitivePtr primitive);
void MindIR_LayerNormFusion_SetBeginParamsAxis(PrimitivePtr *primitive, int64_t begin_params_axis);

// ********** LessEqual **********
PrimitivePtr MindIR_LessEqual_CreatePrimitive();

// ********** MatMulFusion **********
PrimitivePtr MindIR_MatMulFusion_CreatePrimitive(bool transpose_a, bool transpose_b, ActivationType activation_type);
bool MindIR_MatMulFusion_GetTransposeA(ConstPrimitivePtr primitive);
void MindIR_MatMulFusion_SetTransposeA(PrimitivePtr *primitive, bool transpose_a);
bool MindIR_MatMulFusion_GetTransposeB(ConstPrimitivePtr primitive);
void MindIR_MatMulFusion_SetTransposeB(PrimitivePtr *primitive, bool transpose_b);
ActivationType MindIR_MatMulFusion_GetActivationType(ConstPrimitivePtr primitive);
void MindIR_MatMulFusion_SetActivationType(PrimitivePtr *primitive, ActivationType activation_type);

// ********** Maximum **********
PrimitivePtr MindIR_Maximum_CreatePrimitive();

// ********** MaxPoolFusion **********
PrimitivePtr MindIR_MaxPoolFusion_CreatePrimitive(const std::vector<int64_t> &kernel_size,
                                                  const std::vector<int64_t> &strides, const std::vector<int64_t> &pad,
                                                  PadMode pad_mode, Format format, bool global,
                                                  ActivationType activation_type);
std::vector<int64_t> MindIR_MaxPoolFusion_GetKernelSize(ConstPrimitivePtr primitive);
void MindIR_MaxPoolFusion_SetKernelSize(PrimitivePtr *primitive, const std::vector<int64_t> &kernel_size);
std::vector<int64_t> MindIR_MaxPoolFusion_GetStrides(ConstPrimitivePtr primitive);
void MindIR_MaxPoolFusion_SetStrides(PrimitivePtr *primitive, const std::vector<int64_t> &strides);
std::vector<int64_t> MindIR_MaxPoolFusion_GetPad(ConstPrimitivePtr primitive);
void MindIR_MaxPoolFusion_SetPad(PrimitivePtr *primitive, const std::vector<int64_t> &pad);
PadMode MindIR_MaxPoolFusion_GetPadMode(ConstPrimitivePtr primitive);
void MindIR_MaxPoolFusion_SetPadMode(PrimitivePtr *primitive, PadMode pad_mode);
Format MindIR_MaxPoolFusion_GetFormat(ConstPrimitivePtr primitive);
void MindIR_MaxPoolFusion_SetFormat(PrimitivePtr *primitive, Format format);
bool MindIR_MaxPoolFusion_GetGlobal(ConstPrimitivePtr primitive);
void MindIR_MaxPoolFusion_SetGlobal(PrimitivePtr *primitive, bool global);
RoundMode MindIR_MaxPoolFusion_GetRoundMode(ConstPrimitivePtr primitive);
void MindIR_MaxPoolFusion_SetRoundMode(PrimitivePtr *primitive, RoundMode round_mode);
ActivationType MindIR_MaxPoolFusion_GetActivationType(ConstPrimitivePtr primitive);
void MindIR_MaxPoolFusion_SetActivationType(PrimitivePtr *primitive, ActivationType activation_type);

// ********** MulFusion **********
PrimitivePtr MindIR_MulFusion_CreatePrimitive(ActivationType activation_type);
ActivationType MindIR_MulFusion_GetActivationType(ConstPrimitivePtr primitive);
void MindIR_MulFusion_SetActivationType(PrimitivePtr *primitive, ActivationType activation_type);

// ********** OneHot **********
PrimitivePtr MindIR_OneHot_CreatePrimitive(int64_t axis);
int64_t MindIR_OneHot_GetAxis(ConstPrimitivePtr primitive);
void MindIR_OneHot_SetAxis(PrimitivePtr *primitive, int64_t axis);

// ********** PadFusion **********
PrimitivePtr MindIR_PadFusion_CreatePrimitive(const std::vector<std::vector<int64_t>> &paddings,
                                              PaddingMode padding_mode, float constant_value);
std::vector<std::vector<int64_t>> MindIR_PadFusion_GetPaddings(ConstPrimitivePtr primitive);
void MindIR_PadFusion_SetPaddings(PrimitivePtr *primitive, const std::vector<std::vector<int64_t>> &paddings);
PaddingMode MindIR_PadFusion_GetPaddingMode(ConstPrimitivePtr primitive);
void MindIR_PadFusion_SetPaddingMode(PrimitivePtr *primitive, PaddingMode padding_mode);
float MindIR_PadFusion_GetConstantValue(ConstPrimitivePtr primitive);
void MindIR_PadFusion_SetConstantValue(PrimitivePtr *primitive, float constant_value);

// ********** PowFusion **********
PrimitivePtr MindIR_PowFusion_CreatePrimitive(float scale, float shift);
float MindIR_PowFusion_GetScale(ConstPrimitivePtr primitive);
void MindIR_PowFusion_SetScale(PrimitivePtr *primitive, float scale);
float MindIR_PowFusion_GetShift(ConstPrimitivePtr primitive);
void MindIR_PowFusion_SetShift(PrimitivePtr *primitive, float shift);

// ********** PReLUFusion **********
PrimitivePtr MindIR_PReLUFusion_CreatePrimitive(bool channel_shared);
bool MindIR_PReLUFusion_GetChannelShared(ConstPrimitivePtr primitive);
void MindIR_PReLUFusion_SetChannelShared(PrimitivePtr *primitive, bool channel_shared);

// ********** QuantDTypeCast **********
PrimitivePtr MindIR_QuantDTypeCast_CreatePrimitive(int64_t src_t, int64_t dst_t, int64_t axis);
int64_t MindIR_QuantDTypeCast_GetSrcT(ConstPrimitivePtr primitive);
void MindIR_QuantDTypeCast_SetSrcT(PrimitivePtr *primitive, int64_t src_t);
int64_t MindIR_QuantDTypeCast_GetDstT(ConstPrimitivePtr primitive);
void MindIR_QuantDTypeCast_SetDstT(PrimitivePtr *primitive, int64_t dst_t);
int64_t MindIR_QuantDTypeCast_GetAxis(ConstPrimitivePtr primitive);
void MindIR_QuantDTypeCast_SetAxis(PrimitivePtr *primitive, int64_t axis);

// ********** ReduceFusion **********
PrimitivePtr MindIR_ReduceFusion_CreatePrimitive(bool keep_dims, ReduceMode mode, bool reduce_to_end, float coeff);
bool MindIR_ReduceFusion_GetKeepDims(ConstPrimitivePtr primitive);
void MindIR_ReduceFusion_SetKeepDims(PrimitivePtr *primitive, bool keep_dims);
ReduceMode MindIR_ReduceFusion_GetMode(ConstPrimitivePtr primitive);
void MindIR_ReduceFusion_SetMode(PrimitivePtr *primitive, ReduceMode mode);
bool MindIR_ReduceFusion_GetReduceToEnd(ConstPrimitivePtr primitive);
void MindIR_ReduceFusion_SetReduceToEnd(PrimitivePtr *primitive, bool reduce_to_end);
float MindIR_ReduceFusion_GetCoeff(ConstPrimitivePtr primitive);
void MindIR_ReduceFusion_SetCoeff(PrimitivePtr *primitive, float coeff);

// ********** Reshape **********
PrimitivePtr MindIR_Reshape_CreatePrimitive();

// ********** Resize **********
PrimitivePtr MindIR_Resize_CreatePrimitive(ResizeMethod method, int64_t new_height, int64_t new_width,
                                           bool preserve_aspect_ratio,
                                           CoordinateTransformMode coordinate_transform_mode, float cubic_coeff,
                                           int64_t exclude_outside, float extrapolation_value,
                                           NearestMode nearest_mode);
ResizeMethod MindIR_Resize_GetMethod(ConstPrimitivePtr primitive);
void MindIR_Resize_SetMethod(PrimitivePtr *primitive, ResizeMethod method);
int64_t MindIR_Resize_GetNewHeight(ConstPrimitivePtr primitive);
void MindIR_Resize_SetNewHeight(PrimitivePtr *primitive, int64_t new_height);
int64_t MindIR_Resize_GetNewWidth(ConstPrimitivePtr primitive);
void MindIR_Resize_SetNewWidth(PrimitivePtr *primitive, int64_t new_width);
bool MindIR_Resize_GetPreserveAspectRatio(ConstPrimitivePtr primitive);
void MindIR_Resize_SetPreserveAspectRatio(PrimitivePtr *primitive, bool preserve_aspect_ratio);
CoordinateTransformMode MindIR_Resize_GetCoordinateTransformMode(ConstPrimitivePtr primitive);
void MindIR_Resize_SetCoordinateTransformMode(PrimitivePtr *primitive,
                                              CoordinateTransformMode coordinate_transform_mode);
float MindIR_Resize_GetCubicCoeff(ConstPrimitivePtr primitive);
void MindIR_Resize_SetCubicCoeff(PrimitivePtr *primitive, float cubic_coeff);
int64_t MindIR_Resize_GetExcludeOutside(ConstPrimitivePtr primitive);
void MindIR_Resize_SetExcludeOutside(PrimitivePtr *primitive, int64_t exclude_outside);
float MindIR_Resize_GetExtrapolationValue(ConstPrimitivePtr primitive);
void MindIR_Resize_SetExtrapolationValue(PrimitivePtr *primitive, float extrapolation_value);
NearestMode MindIR_Resize_GetNearestMode(ConstPrimitivePtr primitive);
void MindIR_Resize_SetNearestMode(PrimitivePtr *primitive, NearestMode nearest_mode);

// ********** Rsqrt **********
PrimitivePtr MindIR_Rsqrt_CreatePrimitive();

// ********** ScaleFusion **********
PrimitivePtr MindIR_ScaleFusion_CreatePrimitive(int64_t axis, ActivationType activation_type);
int64_t MindIR_ScaleFusion_GetAxis(ConstPrimitivePtr primitive);
void MindIR_ScaleFusion_SetAxis(PrimitivePtr *primitive, int64_t axis);
ActivationType MindIR_ScaleFusion_GetActivationType(ConstPrimitivePtr primitive);
void MindIR_ScaleFusion_SetActivationType(PrimitivePtr *primitive, ActivationType activation_type);

// ********** Shape **********
PrimitivePtr MindIR_Shape_CreatePrimitive();

// ********** SliceFusion **********
PrimitivePtr MindIR_SliceFusion_CreatePrimitive(const std::vector<int64_t> &axes);
std::vector<int64_t> MindIR_SliceFusion_GetAxes(ConstPrimitivePtr primitive);
void MindIR_SliceFusion_SetAxes(PrimitivePtr *primitive, const std::vector<int64_t> &axes);

// ********** Softmax **********
PrimitivePtr MindIR_Softmax_CreatePrimitive(const std::vector<int64_t> &axis);
std::vector<int64_t> MindIR_Softmax_GetAxis(ConstPrimitivePtr primitive);
void MindIR_Softmax_SetAxis(PrimitivePtr *primitive, const std::vector<int64_t> &axis);

// ********** SpaceToBatchND **********
PrimitivePtr MindIR_SpaceToBatchND_CreatePrimitive(const std::vector<int64_t> &block_shape,
                                                   const std::vector<std::vector<int64_t>> &paddings);
std::vector<int64_t> MindIR_SpaceToBatchND_GetBlockShape(ConstPrimitivePtr primitive);
void MindIR_SpaceToBatchND_SetBlockShape(PrimitivePtr *primitive, const std::vector<int64_t> &block_shape);
std::vector<std::vector<int64_t>> MindIR_SpaceToBatchND_GetPaddings(ConstPrimitivePtr primitive);
void MindIR_SpaceToBatchND_SetPaddings(PrimitivePtr *primitive, const std::vector<std::vector<int64_t>> &paddings);

// ********** Split **********
PrimitivePtr MindIR_Split_CreatePrimitive(int64_t output_num, const std::vector<int64_t> &size_splits, int64_t axis);
int64_t MindIR_Split_GetOutputNum(ConstPrimitivePtr primitive);
void MindIR_Split_SetOutputNum(PrimitivePtr *primitive, int64_t output_num);
std::vector<int64_t> MindIR_Split_GetSizeSplits(ConstPrimitivePtr primitive);
void MindIR_Split_SetSizeSplits(PrimitivePtr *primitive, const std::vector<int64_t> &size_splits);
int64_t MindIR_Split_GetAxis(ConstPrimitivePtr primitive);
void MindIR_Split_SetAxis(PrimitivePtr *primitive, int64_t axis);

// ********** Sqrt **********
PrimitivePtr MindIR_Sqrt_CreatePrimitive();

// ********** SquaredDifference **********
PrimitivePtr MindIR_SquaredDifference_CreatePrimitive();

// ********** Squeeze **********
PrimitivePtr MindIR_Squeeze_CreatePrimitive(const std::vector<int64_t> &axis);
std::vector<int64_t> MindIR_Squeeze_GetAxis(ConstPrimitivePtr primitive);
void MindIR_Squeeze_SetAxis(PrimitivePtr *primitive, const std::vector<int64_t> &axis);

// ********** Stack **********
PrimitivePtr MindIR_Stack_CreatePrimitive(int64_t axis);
int64_t MindIR_Stack_GetAxis(ConstPrimitivePtr primitive);
void MindIR_Stack_SetAxis(PrimitivePtr *primitive, int64_t axis);

// ********** StridedSlice **********
PrimitivePtr MindIR_StridedSlice_CreatePrimitive(int64_t begin_mask, int64_t end_mask, int64_t ellipsis_mask,
                                                 int64_t new_axis_mask, int64_t shrink_axis_mask);
int64_t MindIR_StridedSlice_GetBeginMask(ConstPrimitivePtr primitive);
void MindIR_StridedSlice_SetBeginMask(PrimitivePtr *primitive, int64_t begin_mask);
int64_t MindIR_StridedSlice_GetEndMask(ConstPrimitivePtr primitive);
void MindIR_StridedSlice_SetEndMask(PrimitivePtr *primitive, int64_t end_mask);
int64_t MindIR_StridedSlice_GetEllipsisMask(ConstPrimitivePtr primitive);
void MindIR_StridedSlice_SetEllipsisMask(PrimitivePtr *primitive, int64_t ellipsis_mask);
int64_t MindIR_StridedSlice_GetNewAxisMask(ConstPrimitivePtr primitive);
void MindIR_StridedSlice_SetNewAxisMask(PrimitivePtr *primitive, int64_t new_axis_mask);
int64_t MindIR_StridedSlice_GetShrinkAxisMask(ConstPrimitivePtr primitive);
void MindIR_StridedSlice_SetShrinkAxisMask(PrimitivePtr *primitive, int64_t shrink_axis_mask);

// ********** SubFusion **********
PrimitivePtr MindIR_SubFusion_CreatePrimitive(ActivationType activation_type);
ActivationType MindIR_SubFusion_GetActivationType(ConstPrimitivePtr primitive);
void MindIR_SubFusion_SetActivationType(PrimitivePtr *primitive, ActivationType activation_type);

// ********** TileFusion **********
PrimitivePtr MindIR_TileFusion_CreatePrimitive(const std::vector<int64_t> &dims);
std::vector<int64_t> MindIR_TileFusion_GetDims(ConstPrimitivePtr primitive);
void MindIR_TileFusion_SetDims(PrimitivePtr *primitive, const std::vector<int64_t> &dims);

// ********** TopKFusion **********
PrimitivePtr MindIR_TopKFusion_CreatePrimitive(bool sorted, int64_t axis);
bool MindIR_TopKFusion_GetSorted(ConstPrimitivePtr primitive);
void MindIR_TopKFusion_SetSorted(PrimitivePtr *primitive, bool sorted);
int64_t MindIR_TopKFusion_GetAxis(ConstPrimitivePtr primitive);
void MindIR_TopKFusion_SetAxis(PrimitivePtr *primitive, int64_t axis);

// ********** Transpose **********
PrimitivePtr MindIR_Transpose_CreatePrimitive();

// ********** Unsqueeze **********
PrimitivePtr MindIR_Unsqueeze_CreatePrimitive(const std::vector<int64_t> &axis);
std::vector<int64_t> MindIR_Unsqueeze_GetAxis(ConstPrimitivePtr primitive);
void MindIR_Unsqueeze_SetAxis(PrimitivePtr *primitive, const std::vector<int64_t> &axis);

// ********** Abs **********
PrimitivePtr MindIR_Abs_CreatePrimitive();

// ********** BroadcastTo **********
PrimitivePtr MindIR_BroadcastTo_CreatePrimitive(const std::vector<int64_t> &shape);
std::vector<int64_t> MindIR_BroadcastTo_GetShape(ConstPrimitivePtr primitive);
void MindIR_BroadcastTo_SetShape(PrimitivePtr *primitive, const std::vector<int64_t> &shape);

// ********** ConstantOfShape **********
PrimitivePtr MindIR_ConstantOfShape_CreatePrimitive(int64_t data_type, const std::vector<float> &value);
int64_t MindIR_ConstantOfShape_GetDataType(ConstPrimitivePtr primitive);
void MindIR_ConstantOfShape_SetDataType(PrimitivePtr *primitive, int64_t data_type);
std::vector<float> MindIR_ConstantOfShape_GetValue(ConstPrimitivePtr primitive);
void MindIR_ConstantOfShape_SetValue(PrimitivePtr *primitive, const std::vector<float> &value);

// ********** DepthToSpace **********
PrimitivePtr MindIR_DepthToSpace_CreatePrimitive(int64_t block_size, Format &format, std::string mode);
int64_t MindIR_DepthToSpace_GetBlockSize(ConstPrimitivePtr primitive);
void MindIR_DepthToSpace_SetBlockSize(PrimitivePtr *primitive, int64_t block_size);
Format MindIR_DepthToSpace_GetFormat(ConstPrimitivePtr primitive);
void MindIR_DepthToSpace_SetFormat(PrimitivePtr *primitive, Format &format);
std::string MindIR_DepthToSpace_GetMode(ConstPrimitivePtr primitive);
void MindIR_DepthToSpace_SetMode(PrimitivePtr *primitive, std::string mode);

// ********** ExpFusion **********
PrimitivePtr MindIR_ExpFusion_CreatePrimitive(float base, float scale, float shift);
float MindIR_ExpFusion_GetBase(ConstPrimitivePtr primitive);
void MindIR_ExpFusion_SetBase(PrimitivePtr *primitive, float base);
float MindIR_ExpFusion_GetScale(ConstPrimitivePtr primitive);
void MindIR_ExpFusion_SetScale(PrimitivePtr *primitive, float scale);
float MindIR_ExpFusion_GetShift(ConstPrimitivePtr primitive);
void MindIR_ExpFusion_SetShift(PrimitivePtr *primitive, float shift);

// ********** Flatten **********
PrimitivePtr MindIR_Flatten_CreatePrimitive(int64_t axis);
int64_t MindIR_Flatten_GetAxis(ConstPrimitivePtr primitive);
void MindIR_Flatten_SetAxis(PrimitivePtr *primitive, int64_t axis);

// ********** InstanceNorm **********
PrimitivePtr MindIR_InstanceNorm_CreatePrimitive(float epsilon);
float MindIR_InstanceNorm_GetEpsilon(ConstPrimitivePtr primitive);
void MindIR_InstanceNorm_SetEpsilon(PrimitivePtr *primitive, float epsilon);

// ********** Less **********
PrimitivePtr MindIR_Less_CreatePrimitive();

// ********** Range **********
PrimitivePtr MindIR_Range_CreatePrimitive(int64_t d_type, int64_t start, int64_t limit, int64_t delta);
int64_t MindIR_Range_GetDType(ConstPrimitivePtr primitive);
void MindIR_Range_SetDType(PrimitivePtr *primitive, int64_t d_type);
int64_t MindIR_Range_GetStart(ConstPrimitivePtr primitive);
void MindIR_Range_SetStart(PrimitivePtr *primitive, int64_t start);
int64_t MindIR_Range_GetLimit(ConstPrimitivePtr primitive);
void MindIR_Range_SetLimit(PrimitivePtr *primitive, int64_t limit);
int64_t MindIR_Range_GetDelta(ConstPrimitivePtr primitive);
void MindIR_Range_SetDelta(PrimitivePtr *primitive, int64_t delta);

// ********** RealDiv **********
PrimitivePtr MindIR_RealDiv_CreatePrimitive();

// ********** Square **********
PrimitivePtr MindIR_Square_CreatePrimitive();

// ********** Unstack **********
PrimitivePtr MindIR_Unstack_CreatePrimitive(int64_t axis);
int64_t MindIR_Unstack_GetAxis(ConstPrimitivePtr primitive);
void MindIR_Unstack_SetAxis(PrimitivePtr *primitive, int64_t axis);

// ********** Select **********
PrimitivePtr MindIR_Select_CreatePrimitive();

// ********** Erf **********
PrimitivePtr MindIR_Erf_CreatePrimitive();

// ********** Equal **********
PrimitivePtr MindIR_Equal_CreatePrimitive();

// ********** Greater **********
PrimitivePtr MindIR_Greater_CreatePrimitive();

// ********** GreaterEqual **********
PrimitivePtr MindIR_GreaterEqual_CreatePrimitive();

// ********** NotEqual **********
PrimitivePtr MindIR_NotEqual_CreatePrimitive();

// ********** LeakyRelu **********
PrimitivePtr MindIR_LeakyRelu_CreatePrimitive(float negative_slope);
float MindIR_LeakyRelu_GetNegativeSlope(ConstPrimitivePtr primitive);
void MindIR_LeakyRelu_SetNegativeSlope(PrimitivePtr *primitive, float negative_slope);

// ********** LSTM **********
PrimitivePtr MindIR_LSTM_CreatePrimitive(
  bool bidirectional, bool has_bias, int64_t input_size, int64_t hidden_size, int64_t num_layers,
  int64_t num_directions, float dropout, float zoneout_cell, float zoneout_hidden, int64_t proj_size);
bool MindIR_LSTM_GetBidirectional(ConstPrimitivePtr primitive);
void MindIR_LSTM_SetBidirectional(PrimitivePtr *primitive, bool bidirectional);
bool MindIR_LSTM_GetHasBias(ConstPrimitivePtr primitive);
void MindIR_LSTM_SetHasBias(PrimitivePtr *primitive, bool has_bias);
int64_t MindIR_LSTM_GetInputSize(ConstPrimitivePtr primitive);
void MindIR_LSTM_SetInputSize(PrimitivePtr *primitive, int64_t input_size);
int64_t MindIR_LSTM_GetHiddenSize(ConstPrimitivePtr primitive);
void MindIR_LSTM_SetHiddenSize(PrimitivePtr *primitive, int64_t hidden_size);
int64_t MindIR_LSTM_GetNumLayers(ConstPrimitivePtr primitive);
void MindIR_LSTM_SetNumLayers(PrimitivePtr *primitive, int64_t num_layers);
int64_t MindIR_LSTM_GetNumDirections(ConstPrimitivePtr primitive);
void MindIR_LSTM_SetNumDirections(PrimitivePtr *primitive, int64_t num_directions);
float MindIR_LSTM_GetDropout(ConstPrimitivePtr primitive);
void MindIR_LSTM_SetDropout(PrimitivePtr *primitive, float dropout);
float MindIR_LSTM_GetZoneoutCell(ConstPrimitivePtr primitive);
void MindIR_LSTM_SetZoneoutCell(PrimitivePtr *primitive, float zoneout_cell);
float MindIR_LSTM_GetZoneoutHidden(ConstPrimitivePtr primitive);
void MindIR_LSTM_SetZoneoutHidden(PrimitivePtr *primitive, float zoneout_hidden);
int64_t MindIR_LSTM_GetProjSize(ConstPrimitivePtr primitive);
void MindIR_LSTM_SetProjSize(PrimitivePtr *primitive, int64_t proj_size);

// ********** Clip **********
PrimitivePtr MindIR_Clip_CreatePrimitive(float max, float min);
float MindIR_Clip_GetMax(ConstPrimitivePtr primitive);
void MindIR_Clip_SetMax(PrimitivePtr *primitive, float max);
float MindIR_Clip_GetMin(ConstPrimitivePtr primitive);
void MindIR_Clip_SetMin(PrimitivePtr *primitive, float min);

// ********** All **********
PrimitivePtr MindIR_All_CreatePrimitive(int64_t keep_dims);
int64_t MindIR_All_GetKeepDims(ConstPrimitivePtr primitive);
void MindIR_All_SetKeepDims(PrimitivePtr *primitive, int64_t keep_dims);

// ********** Assert **********
PrimitivePtr MindIR_Assert_CreatePrimitive(int64_t summarize);
int64_t MindIR_Assert_GetSummarize(ConstPrimitivePtr primitive);
void MindIR_Assert_SetSummarize(PrimitivePtr *primitive, int64_t summarize);

// ********** LogicalAnd **********
PrimitivePtr MindIR_LogicalAnd_CreatePrimitive();

// ********** LogicalNot **********
PrimitivePtr MindIR_LogicalNot_CreatePrimitive();

// ********** Cos **********
PrimitivePtr MindIR_Cos_CreatePrimitive();

// ********** Mod **********
PrimitivePtr MindIR_Mod_CreatePrimitive();

// ********** Neg **********
PrimitivePtr MindIR_Neg_CreatePrimitive();

// ********** Reciprocal **********
PrimitivePtr MindIR_Reciprocal_CreatePrimitive();

// ********** Sin **********
PrimitivePtr MindIR_Sin_CreatePrimitive();

// ********** Where **********
PrimitivePtr MindIR_Where_CreatePrimitive();

// ********** Log **********
PrimitivePtr MindIR_Log_CreatePrimitive();

// ********** LogicalOr **********
PrimitivePtr MindIR_LogicalOr_CreatePrimitive();

// ********** SparseToDense **********
PrimitivePtr MindIR_SparseToDense_CreatePrimitive();

// ********** Minimum **********
PrimitivePtr MindIR_Minimum_CreatePrimitive();

// ********** SpaceToDepth **********
PrimitivePtr MindIR_SpaceToDepth_CreatePrimitive(int64_t block_size, Format format);
int64_t MindIR_SpaceToDepth_GetBlockSize(ConstPrimitivePtr primitive);
void MindIR_SpaceToDepth_SetBlockSize(PrimitivePtr *primitive, int64_t block_size);
Format MindIR_SpaceToDepth_GetFormat(ConstPrimitivePtr primitive);
void MindIR_SpaceToDepth_SetFormat(PrimitivePtr *primitive, Format format);

// ********** Round **********
PrimitivePtr MindIR_Round_CreatePrimitive();

// ********** Ceil **********
PrimitivePtr MindIR_Ceil_CreatePrimitive();

// ********** Floor **********
PrimitivePtr MindIR_Floor_CreatePrimitive();

// ********** L2NormalizeFusion **********
PrimitivePtr MindIR_L2NormalizeFusion_CreatePrimitive(const std::vector<int64_t>& axis, float epsilon, ActivationType activation_type);
std::vector<int64_t> MindIR_L2NormalizeFusion_GetAxis(ConstPrimitivePtr primitive);
void MindIR_L2NormalizeFusion_SetAxis(PrimitivePtr *primitive, const std::vector<int64_t>& axis);
float MindIR_L2NormalizeFusion_GetEpsilon(ConstPrimitivePtr primitive);
void MindIR_L2NormalizeFusion_SetEpsilon(PrimitivePtr *primitive, float epsilon);
ActivationType MindIR_L2NormalizeFusion_GetActivationType(ConstPrimitivePtr primitive);
void MindIR_L2NormalizeFusion_SetActivationType(PrimitivePtr *primitive, ActivationType activation_type);

// ********** LRN **********
PrimitivePtr MindIR_LRN_CreatePrimitive(int64_t depth_radius, float bias, float alpha, float beta, std::string norm_region);
int64_t MindIR_LRN_GetDepthRadius(ConstPrimitivePtr primitive);
void MindIR_LRN_SetDepthRadius(PrimitivePtr *primitive, int64_t depth_radius);
float MindIR_LRN_GetBias(ConstPrimitivePtr primitive);
void MindIR_LRN_SetBias(PrimitivePtr *primitive, float bias);
float MindIR_LRN_GetAlpha(ConstPrimitivePtr primitive);
void MindIR_LRN_SetAlpha(PrimitivePtr *primitive, float alpha);
float MindIR_LRN_GetBeta(ConstPrimitivePtr primitive);
void MindIR_LRN_SetBeta(PrimitivePtr *primitive, float beta);
std::string MindIR_LRN_GetNormRegion(ConstPrimitivePtr primitive);
void MindIR_LRN_SetNormRegion(PrimitivePtr *primitive, std::string norm_region);

// ********** LogSoftmax **********
PrimitivePtr MindIR_LogSoftmax_CreatePrimitive(int64_t axis);
int64_t MindIR_LogSoftmax_GetAxis(ConstPrimitivePtr primitive);
void MindIR_LogSoftmax_SetAxis(PrimitivePtr *primitive, int64_t axis);

// ********** Crop **********
PrimitivePtr MindIR_Crop_CreatePrimitive(int64_t axis, const std::vector<int64_t>& offsets);
int64_t MindIR_Crop_GetAxis(ConstPrimitivePtr primitive);
void MindIR_Crop_SetAxis(PrimitivePtr *primitive, int64_t axis);
std::vector<int64_t> MindIR_Crop_GetOffsets(ConstPrimitivePtr primitive);
void MindIR_Crop_SetOffsets(PrimitivePtr *primitive, const std::vector<int64_t>& offsets);

// ********** DetectionPostProcess **********
PrimitivePtr MindIR_DetectionPostProcess_CreatePrimitive(Format format, int64_t input_size, const std::vector<float>& scale, float nms_iou_threshold, float nms_score_threshold, int64_t max_detections, int64_t detections_per_class, int64_t max_classes_per_detection, int64_t num_classes, bool use_regular_nms, bool out_quantized);
Format MindIR_DetectionPostProcess_GetFormat(ConstPrimitivePtr primitive);
void MindIR_DetectionPostProcess_SetFormat(PrimitivePtr *primitive, Format format);
int64_t MindIR_DetectionPostProcess_GetInputSize(ConstPrimitivePtr primitive);
void MindIR_DetectionPostProcess_SetInputSize(PrimitivePtr *primitive, int64_t input_size);
std::vector<float> MindIR_DetectionPostProcess_GetScale(ConstPrimitivePtr primitive);
void MindIR_DetectionPostProcess_SetScale(PrimitivePtr *primitive, const std::vector<float>& scale);
float MindIR_DetectionPostProcess_GetNmsIouThreshold(ConstPrimitivePtr primitive);
void MindIR_DetectionPostProcess_SetNmsIouThreshold(PrimitivePtr *primitive, float nms_iou_threshold);
float MindIR_DetectionPostProcess_GetNmsScoreThreshold(ConstPrimitivePtr primitive);
void MindIR_DetectionPostProcess_SetNmsScoreThreshold(PrimitivePtr *primitive, float nms_score_threshold);
int64_t MindIR_DetectionPostProcess_GetMaxDetections(ConstPrimitivePtr primitive);
void MindIR_DetectionPostProcess_SetMaxDetections(PrimitivePtr *primitive, int64_t max_detections);
int64_t MindIR_DetectionPostProcess_GetDetectionsPerClass(ConstPrimitivePtr primitive);
void MindIR_DetectionPostProcess_SetDetectionsPerClass(PrimitivePtr *primitive, int64_t detections_per_class);
int64_t MindIR_DetectionPostProcess_GetMaxClassesPerDetection(ConstPrimitivePtr primitive);
void MindIR_DetectionPostProcess_SetMaxClassesPerDetection(PrimitivePtr *primitive, int64_t max_classes_per_detection);
int64_t MindIR_DetectionPostProcess_GetNumClasses(ConstPrimitivePtr primitive);
void MindIR_DetectionPostProcess_SetNumClasses(PrimitivePtr *primitive, int64_t num_classes);
bool MindIR_DetectionPostProcess_GetUseRegularNms(ConstPrimitivePtr primitive);
void MindIR_DetectionPostProcess_SetUseRegularNms(PrimitivePtr *primitive, bool use_regular_nms);
bool MindIR_DetectionPostProcess_GetOutQuantized(ConstPrimitivePtr primitive);
void MindIR_DetectionPostProcess_SetOutQuantized(PrimitivePtr *primitive, bool out_quantized);

// ********** ScatterNd **********
PrimitivePtr MindIR_ScatterNd_CreatePrimitive();

// ********** Rank **********
PrimitivePtr MindIR_Rank_CreatePrimitive();

// ********** GatherNd **********
PrimitivePtr MindIR_GatherNd_CreatePrimitive();

// ********** BatchToSpace **********
PrimitivePtr MindIR_BatchToSpace_CreatePrimitive(const std::vector<int64_t>& block_size, const std::vector<std::vector<int64_t>>& crops);
std::vector<int64_t> MindIR_BatchToSpace_GetBlockSize(ConstPrimitivePtr primitive);
void MindIR_BatchToSpace_SetBlockSize(PrimitivePtr *primitive, const std::vector<int64_t>& block_size);
std::vector<std::vector<int64_t>> MindIR_BatchToSpace_GetCrops(ConstPrimitivePtr primitive);
void MindIR_BatchToSpace_SetCrops(PrimitivePtr *primitive, const std::vector<std::vector<int64_t>>& crops);

// ********** Depend **********
PrimitivePtr MindIR_Depend_CreatePrimitive();

// ********** Dropout **********
PrimitivePtr MindIR_Dropout_CreatePrimitive(float keep_prob);
float MindIR_Dropout_GetKeepProb(ConstPrimitivePtr primitive);
void MindIR_Dropout_SetKeepProb(PrimitivePtr *primitive, float keep_prob);

// ********** Elu **********
PrimitivePtr MindIR_Elu_CreatePrimitive(float alpha);
float MindIR_Elu_GetAlpha(ConstPrimitivePtr primitive);
void MindIR_Elu_SetAlpha(PrimitivePtr *primitive, float alpha);

// ********** EmbeddingLookupFusion **********
PrimitivePtr MindIR_EmbeddingLookupFusion_CreatePrimitive(float max_norm);
float MindIR_EmbeddingLookupFusion_GetMaxNorm(ConstPrimitivePtr primitive);
void MindIR_EmbeddingLookupFusion_SetMaxNorm(PrimitivePtr *primitive, float max_norm);

// ********** FakeQuantWithMinMaxVars **********
PrimitivePtr MindIR_FakeQuantWithMinMaxVars_CreatePrimitive(int64_t num_bits, bool narrow_range);
int64_t MindIR_FakeQuantWithMinMaxVars_GetNumBits(ConstPrimitivePtr primitive);
void MindIR_FakeQuantWithMinMaxVars_SetNumBits(PrimitivePtr *primitive, int64_t num_bits);
bool MindIR_FakeQuantWithMinMaxVars_GetNarrowRange(ConstPrimitivePtr primitive);
void MindIR_FakeQuantWithMinMaxVars_SetNarrowRange(PrimitivePtr *primitive, bool narrow_range);

// ********** FakeQuantWithMinMaxVarsPerChannel **********
PrimitivePtr MindIR_FakeQuantWithMinMaxVarsPerChannel_CreatePrimitive(int64_t num_bits, bool narrow_range);
int64_t MindIR_FakeQuantWithMinMaxVarsPerChannel_GetNumBits(ConstPrimitivePtr primitive);
void MindIR_FakeQuantWithMinMaxVarsPerChannel_SetNumBits(PrimitivePtr *primitive, int64_t num_bits);
bool MindIR_FakeQuantWithMinMaxVarsPerChannel_GetNarrowRange(ConstPrimitivePtr primitive);
void MindIR_FakeQuantWithMinMaxVarsPerChannel_SetNarrowRange(PrimitivePtr *primitive, bool narrow_range);

// ********** FftReal **********
PrimitivePtr MindIR_FftReal_CreatePrimitive();

// ********** FftImag **********
PrimitivePtr MindIR_FftImag_CreatePrimitive();

// ********** FloorDiv **********
PrimitivePtr MindIR_FloorDiv_CreatePrimitive();

// ********** FloorMod **********
PrimitivePtr MindIR_FloorMod_CreatePrimitive();

// ********** HashtableLookup **********
PrimitivePtr MindIR_HashtableLookup_CreatePrimitive();

// ********** LpNormalization **********
PrimitivePtr MindIR_LpNormalization_CreatePrimitive(int64_t axis, int64_t p);
int64_t MindIR_LpNormalization_GetAxis(ConstPrimitivePtr primitive);
void MindIR_LpNormalization_SetAxis(PrimitivePtr *primitive, int64_t axis);
int64_t MindIR_LpNormalization_GetP(ConstPrimitivePtr primitive);
void MindIR_LpNormalization_SetP(PrimitivePtr *primitive, int64_t p);

// ********** LshProjection **********
PrimitivePtr MindIR_LshProjection_CreatePrimitive(LshProjectionType type);
LshProjectionType MindIR_LshProjection_GetType(ConstPrimitivePtr primitive);
void MindIR_LshProjection_SetType(PrimitivePtr *primitive, LshProjectionType type);

// ********** SwitchLayer **********
PrimitivePtr MindIR_SwitchLayer_CreatePrimitive();

// ********** Mfcc **********
PrimitivePtr MindIR_Mfcc_CreatePrimitive(float freq_upper_limit, float freq_lower_limit, int64_t filter_bank_channel_num, int64_t dct_coeff_num);
float MindIR_Mfcc_GetFreqUpperLimit(ConstPrimitivePtr primitive);
void MindIR_Mfcc_SetFreqUpperLimit(PrimitivePtr *primitive, float freq_upper_limit);
float MindIR_Mfcc_GetFreqLowerLimit(ConstPrimitivePtr primitive);
void MindIR_Mfcc_SetFreqLowerLimit(PrimitivePtr *primitive, float freq_lower_limit);
int64_t MindIR_Mfcc_GetFilterBankChannelNum(ConstPrimitivePtr primitive);
void MindIR_Mfcc_SetFilterBankChannelNum(PrimitivePtr *primitive, int64_t filter_bank_channel_num);
int64_t MindIR_Mfcc_GetDctCoeffNum(ConstPrimitivePtr primitive);
void MindIR_Mfcc_SetDctCoeffNum(PrimitivePtr *primitive, int64_t dct_coeff_num);

// ********** NonMaxSuppression **********
PrimitivePtr MindIR_NonMaxSuppression_CreatePrimitive(int64_t center_point_box);
int64_t MindIR_NonMaxSuppression_GetCenterPointBox(ConstPrimitivePtr primitive);
void MindIR_NonMaxSuppression_SetCenterPointBox(PrimitivePtr *primitive, int64_t center_point_box);

// ********** OnesLike **********
PrimitivePtr MindIR_OnesLike_CreatePrimitive();

// ********** PartialFusion **********
PrimitivePtr MindIR_PartialFusion_CreatePrimitive(int64_t sub_graph_index);
int64_t MindIR_PartialFusion_GetSubGraphIndex(ConstPrimitivePtr primitive);
void MindIR_PartialFusion_SetSubGraphIndex(PrimitivePtr *primitive, int64_t sub_graph_index);

// ********** PriorBox **********
PrimitivePtr MindIR_PriorBox_CreatePrimitive(const std::vector<int64_t>& min_sizes, const std::vector<int64_t>& max_sizes, const std::vector<float>& aspect_ratios, const std::vector<float>& variances, int64_t image_size_w, int64_t image_size_h, float step_w, float step_h, bool clip, bool flip, float offset);
std::vector<int64_t> MindIR_PriorBox_GetMinSizes(ConstPrimitivePtr primitive);
void MindIR_PriorBox_SetMinSizes(PrimitivePtr *primitive, const std::vector<int64_t>& min_sizes);
std::vector<int64_t> MindIR_PriorBox_GetMaxSizes(ConstPrimitivePtr primitive);
void MindIR_PriorBox_SetMaxSizes(PrimitivePtr *primitive, const std::vector<int64_t>& max_sizes);
std::vector<float> MindIR_PriorBox_GetAspectRatios(ConstPrimitivePtr primitive);
void MindIR_PriorBox_SetAspectRatios(PrimitivePtr *primitive, const std::vector<float>& aspect_ratios);
std::vector<float> MindIR_PriorBox_GetVariances(ConstPrimitivePtr primitive);
void MindIR_PriorBox_SetVariances(PrimitivePtr *primitive, const std::vector<float>& variances);
int64_t MindIR_PriorBox_GetImageSizeW(ConstPrimitivePtr primitive);
void MindIR_PriorBox_SetImageSizeW(PrimitivePtr *primitive, int64_t image_size_w);
int64_t MindIR_PriorBox_GetImageSizeH(ConstPrimitivePtr primitive);
void MindIR_PriorBox_SetImageSizeH(PrimitivePtr *primitive, int64_t image_size_h);
float MindIR_PriorBox_GetStepW(ConstPrimitivePtr primitive);
void MindIR_PriorBox_SetStepW(PrimitivePtr *primitive, float step_w);
float MindIR_PriorBox_GetStepH(ConstPrimitivePtr primitive);
void MindIR_PriorBox_SetStepH(PrimitivePtr *primitive, float step_h);
bool MindIR_PriorBox_GetClip(ConstPrimitivePtr primitive);
void MindIR_PriorBox_SetClip(PrimitivePtr *primitive, bool clip);
bool MindIR_PriorBox_GetFlip(ConstPrimitivePtr primitive);
void MindIR_PriorBox_SetFlip(PrimitivePtr *primitive, bool flip);
float MindIR_PriorBox_GetOffset(ConstPrimitivePtr primitive);
void MindIR_PriorBox_SetOffset(PrimitivePtr *primitive, float offset);

// ********** ReverseSequence **********
PrimitivePtr MindIR_ReverseSequence_CreatePrimitive(int64_t seq_dim, int64_t batch_dim);
int64_t MindIR_ReverseSequence_GetSeqDim(ConstPrimitivePtr primitive);
void MindIR_ReverseSequence_SetSeqDim(PrimitivePtr *primitive, int64_t seq_dim);
int64_t MindIR_ReverseSequence_GetBatchDim(ConstPrimitivePtr primitive);
void MindIR_ReverseSequence_SetBatchDim(PrimitivePtr *primitive, int64_t batch_dim);

// ********** ReverseV2 **********
PrimitivePtr MindIR_ReverseV2_CreatePrimitive(const std::vector<int64_t>& axis);
std::vector<int64_t> MindIR_ReverseV2_GetAxis(ConstPrimitivePtr primitive);
void MindIR_ReverseV2_SetAxis(PrimitivePtr *primitive, const std::vector<int64_t>& axis);

// ********** Rfft **********
PrimitivePtr MindIR_Rfft_CreatePrimitive(int64_t fft_length);
int64_t MindIR_Rfft_GetFftLength(ConstPrimitivePtr primitive);
void MindIR_Rfft_SetFftLength(PrimitivePtr *primitive, int64_t fft_length);

// ********** ROIPooling **********
PrimitivePtr MindIR_ROIPooling_CreatePrimitive(int64_t pooled_h, int64_t pooled_w, float scale);
int64_t MindIR_ROIPooling_GetPooledH(ConstPrimitivePtr primitive);
void MindIR_ROIPooling_SetPooledH(PrimitivePtr *primitive, int64_t pooled_h);
int64_t MindIR_ROIPooling_GetPooledW(ConstPrimitivePtr primitive);
void MindIR_ROIPooling_SetPooledW(PrimitivePtr *primitive, int64_t pooled_w);
float MindIR_ROIPooling_GetScale(ConstPrimitivePtr primitive);
void MindIR_ROIPooling_SetScale(PrimitivePtr *primitive, float scale);

// ********** SkipGram **********
PrimitivePtr MindIR_SkipGram_CreatePrimitive(bool include_all_grams, int64_t max_skip_size, int64_t ngram_size);
bool MindIR_SkipGram_GetIncludeAllGrams(ConstPrimitivePtr primitive);
void MindIR_SkipGram_SetIncludeAllGrams(PrimitivePtr *primitive, bool include_all_grams);
int64_t MindIR_SkipGram_GetMaxSkipSize(ConstPrimitivePtr primitive);
void MindIR_SkipGram_SetMaxSkipSize(PrimitivePtr *primitive, int64_t max_skip_size);
int64_t MindIR_SkipGram_GetNgramSize(ConstPrimitivePtr primitive);
void MindIR_SkipGram_SetNgramSize(PrimitivePtr *primitive, int64_t ngram_size);

// ********** Switch **********
PrimitivePtr MindIR_Switch_CreatePrimitive();

// ********** Unique **********
PrimitivePtr MindIR_Unique_CreatePrimitive();

// ********** UnsortedSegmentSum **********
PrimitivePtr MindIR_UnsortedSegmentSum_CreatePrimitive();

// ********** ZerosLike **********
PrimitivePtr MindIR_ZerosLike_CreatePrimitive();

// ********** GRU **********
PrimitivePtr MindIR_GRU_CreatePrimitive( bool bidirectional);
bool MindIR_GRU_GetBidirectional(ConstPrimitivePtr primitive);
void MindIR_GRU_SetBidirectional(PrimitivePtr *primitive, bool bidirectional);

// ********** NonZero **********
PrimitivePtr MindIR_NonZero_CreatePrimitive();

// ********** InvertPermutation **********
PrimitivePtr MindIR_InvertPermutation_CreatePrimitive();

// ********** Size **********
PrimitivePtr MindIR_Size_CreatePrimitive();

// ********** RandomStandardNormal **********
PrimitivePtr MindIR_RandomStandardNormal_CreatePrimitive(int64_t seed, int64_t seed2);
int64_t MindIR_RandomStandardNormal_GetSeed(ConstPrimitivePtr primitive);
void MindIR_RandomStandardNormal_SetSeed(PrimitivePtr *primitive, int64_t seed);
int64_t MindIR_RandomStandardNormal_GetSeed2(ConstPrimitivePtr primitive);
void MindIR_RandomStandardNormal_SetSeed2(PrimitivePtr *primitive, int64_t seed2);

// ********** CropAndResize **********
PrimitivePtr MindIR_CropAndResize_CreatePrimitive(ResizeMethod method, float extrapolation_value);
ResizeMethod MindIR_CropAndResize_GetMethod(ConstPrimitivePtr primitive);
void MindIR_CropAndResize_SetMethod(PrimitivePtr *primitive, ResizeMethod method);
float MindIR_CropAndResize_GetExtrapolationValue(ConstPrimitivePtr primitive);
void MindIR_CropAndResize_SetExtrapolationValue(PrimitivePtr *primitive, float extrapolation_value);

// ********** IsFinite **********
PrimitivePtr MindIR_IsFinite_CreatePrimitive();

// ********** LinSpace **********
PrimitivePtr MindIR_LinSpace_CreatePrimitive();

// ********** UniformReal **********
PrimitivePtr MindIR_UniformReal_CreatePrimitive(int64_t seed, int64_t seed2);
int64_t MindIR_UniformReal_GetSeed(ConstPrimitivePtr primitive);
void MindIR_UniformReal_SetSeed(PrimitivePtr *primitive, int64_t seed);
int64_t MindIR_UniformReal_GetSeed2(ConstPrimitivePtr primitive);
void MindIR_UniformReal_SetSeed2(PrimitivePtr *primitive, int64_t seed2);

// ********** Splice **********
PrimitivePtr MindIR_Splice_CreatePrimitive(const std::vector<int64_t>& context, const std::vector<int64_t>& forward_indexes, int64_t output_dim);
std::vector<int64_t> MindIR_Splice_GetContext(ConstPrimitivePtr primitive);
void MindIR_Splice_SetContext(PrimitivePtr *primitive, const std::vector<int64_t>& context);
std::vector<int64_t> MindIR_Splice_GetForwardIndexes(ConstPrimitivePtr primitive);
void MindIR_Splice_SetForwardIndexes(PrimitivePtr *primitive, const std::vector<int64_t>& forward_indexes);
int64_t MindIR_Splice_GetOutputDim(ConstPrimitivePtr primitive);
void MindIR_Splice_SetOutputDim(PrimitivePtr *primitive, int64_t output_dim);

// ********** Call **********
PrimitivePtr MindIR_Call_CreatePrimitive(bool is_tail_call);
bool MindIR_Call_GetIsTailCall(ConstPrimitivePtr primitive);
void MindIR_Call_SetIsTailCall(PrimitivePtr *primitive, bool is_tail_call);

// ********** CumSum **********
PrimitivePtr MindIR_CumSum_CreatePrimitive(bool exclusive, bool reverse);
bool MindIR_CumSum_GetExclusive(ConstPrimitivePtr primitive);
void MindIR_CumSum_SetExclusive(PrimitivePtr *primitive, bool exclusive);
bool MindIR_CumSum_GetReverse(ConstPrimitivePtr primitive);
void MindIR_CumSum_SetReverse(PrimitivePtr *primitive, bool reverse);

// ********** SplitWithOverlap **********
PrimitivePtr MindIR_SplitWithOverlap_CreatePrimitive(int64_t split_dim, int64_t number_split, const std::vector<int64_t>& ratio, const std::vector<int64_t>& extend_top, const std::vector<int64_t>& extend_bottom);
int64_t MindIR_SplitWithOverlap_GetSplitDim(ConstPrimitivePtr primitive);
void MindIR_SplitWithOverlap_SetSplitDim(PrimitivePtr *primitive, int64_t split_dim);
int64_t MindIR_SplitWithOverlap_GetNumberSplit(ConstPrimitivePtr primitive);
void MindIR_SplitWithOverlap_SetNumberSplit(PrimitivePtr *primitive, int64_t number_split);
std::vector<int64_t> MindIR_SplitWithOverlap_GetRatio(ConstPrimitivePtr primitive);
void MindIR_SplitWithOverlap_SetRatio(PrimitivePtr *primitive, const std::vector<int64_t>& ratio);
std::vector<int64_t> MindIR_SplitWithOverlap_GetExtendTop(ConstPrimitivePtr primitive);
void MindIR_SplitWithOverlap_SetExtendTop(PrimitivePtr *primitive, const std::vector<int64_t>& extend_top);
std::vector<int64_t> MindIR_SplitWithOverlap_GetExtendBottom(ConstPrimitivePtr primitive);
void MindIR_SplitWithOverlap_SetExtendBottom(PrimitivePtr *primitive, const std::vector<int64_t>& extend_bottom);

// ********** GenOP **********
PrimitivePtr MindIR_GenOP_CreatePrimitive(ActivationType activation_type, float alpha, float min_val, float max_val, bool is_training, Format format, const std::vector<int64_t>& kernel_size, const std::vector<int64_t>& stride, const std::vector<int64_t>& dilation, PadMode pad_mode, const std::vector<int64_t>& pad_list, int64_t mode, int64_t group, int64_t in_channel, int64_t out_channel, EltwiseMode eltwise_mode, bool has_bias, bool use_axis, int64_t axis, float epsilon, float momentum, bool transpose_a, bool transpose_b, const std::vector<int64_t>& pad, RoundMode round_mode, bool global, bool channel_shared, const std::vector<int64_t>& axes, bool keep_dims, ReduceMode reduce_mode, bool reduce_to_end, float coeff);
ActivationType MindIR_GenOP_GetActivationType(ConstPrimitivePtr primitive);
void MindIR_GenOP_SetActivationType(PrimitivePtr *primitive, ActivationType activation_type);
float MindIR_GenOP_GetAlpha(ConstPrimitivePtr primitive);
void MindIR_GenOP_SetAlpha(PrimitivePtr *primitive, float alpha);
float MindIR_GenOP_GetMinVal(ConstPrimitivePtr primitive);
void MindIR_GenOP_SetMinVal(PrimitivePtr *primitive, float min_val);
float MindIR_GenOP_GetMaxVal(ConstPrimitivePtr primitive);
void MindIR_GenOP_SetMaxVal(PrimitivePtr *primitive, float max_val);
bool MindIR_GenOP_GetIsTraining(ConstPrimitivePtr primitive);
void MindIR_GenOP_SetIsTraining(PrimitivePtr *primitive, bool is_training);
Format MindIR_GenOP_GetFormat(ConstPrimitivePtr primitive);
void MindIR_GenOP_SetFormat(PrimitivePtr *primitive, Format format);
std::vector<int64_t> MindIR_GenOP_GetKernelSize(ConstPrimitivePtr primitive);
void MindIR_GenOP_SetKernelSize(PrimitivePtr *primitive, const std::vector<int64_t>& kernel_size);
std::vector<int64_t> MindIR_GenOP_GetStride(ConstPrimitivePtr primitive);
void MindIR_GenOP_SetStride(PrimitivePtr *primitive, const std::vector<int64_t>& stride);
std::vector<int64_t> MindIR_GenOP_GetDilation(ConstPrimitivePtr primitive);
void MindIR_GenOP_SetDilation(PrimitivePtr *primitive, const std::vector<int64_t>& dilation);
PadMode MindIR_GenOP_GetPadMode(ConstPrimitivePtr primitive);
void MindIR_GenOP_SetPadMode(PrimitivePtr *primitive, PadMode pad_mode);
std::vector<int64_t> MindIR_GenOP_GetPadList(ConstPrimitivePtr primitive);
void MindIR_GenOP_SetPadList(PrimitivePtr *primitive, const std::vector<int64_t>& pad_list);
int64_t MindIR_GenOP_GetMode(ConstPrimitivePtr primitive);
void MindIR_GenOP_SetMode(PrimitivePtr *primitive, int64_t mode);
int64_t MindIR_GenOP_GetGroup(ConstPrimitivePtr primitive);
void MindIR_GenOP_SetGroup(PrimitivePtr *primitive, int64_t group);
int64_t MindIR_GenOP_GetInChannel(ConstPrimitivePtr primitive);
void MindIR_GenOP_SetInChannel(PrimitivePtr *primitive, int64_t in_channel);
int64_t MindIR_GenOP_GetOutChannel(ConstPrimitivePtr primitive);
void MindIR_GenOP_SetOutChannel(PrimitivePtr *primitive, int64_t out_channel);
EltwiseMode MindIR_GenOP_GetEltwiseMode(ConstPrimitivePtr primitive);
void MindIR_GenOP_SetEltwiseMode(PrimitivePtr *primitive, EltwiseMode eltwise_mode);
bool MindIR_GenOP_GetHasBias(ConstPrimitivePtr primitive);
void MindIR_GenOP_SetHasBias(PrimitivePtr *primitive, bool has_bias);
bool MindIR_GenOP_GetUseAxis(ConstPrimitivePtr primitive);
void MindIR_GenOP_SetUseAxis(PrimitivePtr *primitive, bool use_axis);
int64_t MindIR_GenOP_GetAxis(ConstPrimitivePtr primitive);
void MindIR_GenOP_SetAxis(PrimitivePtr *primitive, int64_t axis);
float MindIR_GenOP_GetEpsilon(ConstPrimitivePtr primitive);
void MindIR_GenOP_SetEpsilon(PrimitivePtr *primitive, float epsilon);
float MindIR_GenOP_GetMomentum(ConstPrimitivePtr primitive);
void MindIR_GenOP_SetMomentum(PrimitivePtr *primitive, float momentum);
bool MindIR_GenOP_GetTransposeA(ConstPrimitivePtr primitive);
void MindIR_GenOP_SetTransposeA(PrimitivePtr *primitive, bool transpose_a);
bool MindIR_GenOP_GetTransposeB(ConstPrimitivePtr primitive);
void MindIR_GenOP_SetTransposeB(PrimitivePtr *primitive, bool transpose_b);
std::vector<int64_t> MindIR_GenOP_GetPad(ConstPrimitivePtr primitive);
void MindIR_GenOP_SetPad(PrimitivePtr *primitive, const std::vector<int64_t>& pad);
RoundMode MindIR_GenOP_GetRoundMode(ConstPrimitivePtr primitive);
void MindIR_GenOP_SetRoundMode(PrimitivePtr *primitive, RoundMode round_mode);
bool MindIR_GenOP_GetGlobal(ConstPrimitivePtr primitive);
void MindIR_GenOP_SetGlobal(PrimitivePtr *primitive, bool global);
bool MindIR_GenOP_GetChannelShared(ConstPrimitivePtr primitive);
void MindIR_GenOP_SetChannelShared(PrimitivePtr *primitive, bool channel_shared);
std::vector<int64_t> MindIR_GenOP_GetAxes(ConstPrimitivePtr primitive);
void MindIR_GenOP_SetAxes(PrimitivePtr *primitive, const std::vector<int64_t>& axes);
bool MindIR_GenOP_GetKeepDims(ConstPrimitivePtr primitive);
void MindIR_GenOP_SetKeepDims(PrimitivePtr *primitive, bool keep_dims);
ReduceMode MindIR_GenOP_GetReduceMode(ConstPrimitivePtr primitive);
void MindIR_GenOP_SetReduceMode(PrimitivePtr *primitive, ReduceMode reduce_mode);
bool MindIR_GenOP_GetReduceToEnd(ConstPrimitivePtr primitive);
void MindIR_GenOP_SetReduceToEnd(PrimitivePtr *primitive, bool reduce_to_end);
float MindIR_GenOP_GetCoeff(ConstPrimitivePtr primitive);
void MindIR_GenOP_SetCoeff(PrimitivePtr *primitive, float coeff);

// ********** RaggedRange **********
PrimitivePtr MindIR_RaggedRange_CreatePrimitive();

// ********** GLU **********
PrimitivePtr MindIR_GLU_CreatePrimitive(int64_t axis);
int64_t MindIR_GLU_GetAxis(ConstPrimitivePtr primitive);
void MindIR_GLU_SetAxis(PrimitivePtr *primitive, int64_t axis);

// ********** Affine **********
PrimitivePtr MindIR_Affine_CreatePrimitive(const std::vector<int64_t>& context, int64_t output_dim, ActivationType activation_type, bool transpose_a, bool transpose_b);
std::vector<int64_t> MindIR_Affine_GetContext(ConstPrimitivePtr primitive);
void MindIR_Affine_SetContext(PrimitivePtr *primitive, const std::vector<int64_t>& context);
int64_t MindIR_Affine_GetOutputDim(ConstPrimitivePtr primitive);
void MindIR_Affine_SetOutputDim(PrimitivePtr *primitive, int64_t output_dim);
ActivationType MindIR_Affine_GetActivationType(ConstPrimitivePtr primitive);
void MindIR_Affine_SetActivationType(PrimitivePtr *primitive, ActivationType activation_type);
bool MindIR_Affine_GetTransposeA(ConstPrimitivePtr primitive);
void MindIR_Affine_SetTransposeA(PrimitivePtr *primitive, bool transpose_a);
bool MindIR_Affine_GetTransposeB(ConstPrimitivePtr primitive);
void MindIR_Affine_SetTransposeB(PrimitivePtr *primitive, bool transpose_b);

// ********** AllGather **********
PrimitivePtr MindIR_AllGather_CreatePrimitive(const std::string& group, int32_t rank_size);
std::string MindIR_AllGather_GetGroup(ConstPrimitivePtr primitive);
void MindIR_AllGather_SetGroup(PrimitivePtr *primitive, const std::string& group);
int32_t MindIR_AllGather_GetRankSize(ConstPrimitivePtr primitive);
void MindIR_AllGather_SetRankSize(PrimitivePtr *primitive, int32_t rank_size);



// ********** ReduceScatter **********
PrimitivePtr MindIR_ReduceScatter_CreatePrimitive(const std::string& group, ReduceMode mode, int32_t rank_size);
std::string MindIR_ReduceScatter_GetGroup(ConstPrimitivePtr primitive);
void MindIR_ReduceScatter_SetGroup(PrimitivePtr *primitive, const std::string& group);
ReduceMode MindIR_ReduceScatter_GetMode(ConstPrimitivePtr primitive);
void MindIR_ReduceScatter_SetMode(PrimitivePtr *primitive, ReduceMode mode);
int32_t MindIR_ReduceScatter_GetRankSize(ConstPrimitivePtr primitive);
void MindIR_ReduceScatter_SetRankSize(PrimitivePtr *primitive, int32_t rank_size);



// ********** DynamicQuant **********
PrimitivePtr MindIR_DynamicQuant_CreatePrimitive(bool symmetric, int64_t dst_type);
bool MindIR_DynamicQuant_GetSymmetric(ConstPrimitivePtr primitive);
void MindIR_DynamicQuant_SetSymmetric(PrimitivePtr *primitive, bool symmetric);
int64_t MindIR_DynamicQuant_GetDstType(ConstPrimitivePtr primitive);
void MindIR_DynamicQuant_SetDstType(PrimitivePtr *primitive, int64_t dst_type);

// ********** RandomNormal **********
PrimitivePtr MindIR_RandomNormal_CreatePrimitive(float seed, float mean, float scale);
float MindIR_RandomNormal_GetSeed(ConstPrimitivePtr primitive);
void MindIR_RandomNormal_SetSeed(PrimitivePtr *primitive, float seed);
float MindIR_RandomNormal_GetMean(ConstPrimitivePtr primitive);
void MindIR_RandomNormal_SetMean(PrimitivePtr *primitive, float mean);
float MindIR_RandomNormal_GetScale(ConstPrimitivePtr primitive);
void MindIR_RandomNormal_SetScale(PrimitivePtr *primitive, float scale);

// ********** FormatTranspose **********
PrimitivePtr MindIR_FormatTranspose_CreatePrimitive(Format src_format, Format dst_format);
Format MindIR_FormatTranspose_GetSrcFormat(ConstPrimitivePtr primitive);
void MindIR_FormatTranspose_SetSrcFormat(PrimitivePtr *primitive, Format src_format);
Format MindIR_FormatTranspose_GetDstFormat(ConstPrimitivePtr primitive);
void MindIR_FormatTranspose_SetDstFormat(PrimitivePtr *primitive, Format dst_format);

// ********** GatherD **********
PrimitivePtr MindIR_GatherD_CreatePrimitive();

// ********** GroupNormFusion **********
PrimitivePtr MindIR_GroupNormFusion_CreatePrimitive(int64_t num_groups, float epsilon, bool affine);
int64_t MindIR_GroupNormFusion_GetNumGroups(ConstPrimitivePtr primitive);
void MindIR_GroupNormFusion_SetNumGroups(PrimitivePtr *primitive, int64_t num_groups);
float MindIR_GroupNormFusion_GetEpsilon(ConstPrimitivePtr primitive);
void MindIR_GroupNormFusion_SetEpsilon(PrimitivePtr *primitive, float epsilon);
bool MindIR_GroupNormFusion_GetAffine(ConstPrimitivePtr primitive);
void MindIR_GroupNormFusion_SetAffine(PrimitivePtr *primitive, bool affine);

// ********** Log1p **********
PrimitivePtr MindIR_Log1p_CreatePrimitive();

// ********** SparseFillEmptyRows **********
PrimitivePtr MindIR_SparseFillEmptyRows_CreatePrimitive();

// ********** SparseReshape **********
PrimitivePtr MindIR_SparseReshape_CreatePrimitive();

// ********** SparseSegmentSum **********
PrimitivePtr MindIR_SparseSegmentSum_CreatePrimitive();

// ********** ScatterElements **********
PrimitivePtr MindIR_ScatterElements_CreatePrimitive(int64_t axis);
int64_t MindIR_ScatterElements_GetAxis(ConstPrimitivePtr primitive);
void MindIR_ScatterElements_SetAxis(PrimitivePtr *primitive, int64_t axis);

// ********** Triu **********
PrimitivePtr MindIR_Triu_CreatePrimitive();

// ********** Tril **********
PrimitivePtr MindIR_Tril_CreatePrimitive();

// ********** Custom **********
std::vector<const mindspore::schema::Attribute *> MindIR_Custom_GetAttr(ConstPrimitivePtr primitive);
std::string MindIR_Attribute_GetName(const mindspore::schema::Attribute &attr);
std::vector<uint8_t> MindIR_Attribute_GetData(const mindspore::schema::Attribute &attr);
}  // namespace lite
}  // namespace mindspore
#endif
