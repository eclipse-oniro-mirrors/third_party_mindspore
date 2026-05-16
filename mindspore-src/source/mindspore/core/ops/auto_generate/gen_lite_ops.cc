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
#include "ops/auto_generate/gen_lite_ops.h"
#include "mindapi/src/helper.h"
#include "ops/primitive_c.h"
#include "ops/base_operator.h"
#include "abstract/abstract_value.h"

namespace mindspore::ops {
REGISTER_PRIMITIVE_C(kNameACosGrad, ACosGrad);
MIND_API_OPERATOR_IMPL(ACosGrad, BaseOperator);

REGISTER_PRIMITIVE_C(kNameAbsGrad, AbsGrad);
MIND_API_OPERATOR_IMPL(AbsGrad, BaseOperator);

REGISTER_PRIMITIVE_C(kNameAbs, Abs);
MIND_API_OPERATOR_IMPL(Abs, BaseOperator);

REGISTER_PRIMITIVE_C(kNameACos, ACos);
MIND_API_OPERATOR_IMPL(ACos, BaseOperator);

REGISTER_PRIMITIVE_C(kNameAcoshGrad, AcoshGrad);
MIND_API_OPERATOR_IMPL(AcoshGrad, BaseOperator);

REGISTER_PRIMITIVE_C(kNameAcosh, Acosh);
MIND_API_OPERATOR_IMPL(Acosh, BaseOperator);

void AdamWeightDecay::set_use_locking(const bool &use_locking)             { (void)this->AddAttr("use_locking", api::MakeValue(use_locking)); }

bool AdamWeightDecay::get_use_locking() const             { return GetValue<bool>(GetAttr("use_locking")); }

REGISTER_PRIMITIVE_C(kNameAdamWeightDecay, AdamWeightDecay);
MIND_API_OPERATOR_IMPL(AdamWeightDecay, BaseOperator);

REGISTER_PRIMITIVE_C(kNameAdamW, AdamW);
MIND_API_OPERATOR_IMPL(AdamW, BaseOperator);

REGISTER_PRIMITIVE_C(kNameAddExt, AddExt);
MIND_API_OPERATOR_IMPL(AddExt, BaseOperator);

REGISTER_PRIMITIVE_C(kNameAddLayerNormV2, AddLayerNormV2);
MIND_API_OPERATOR_IMPL(AddLayerNormV2, BaseOperator);

REGISTER_PRIMITIVE_C(kNameAdd, Add);
MIND_API_OPERATOR_IMPL(Add, BaseOperator);

REGISTER_PRIMITIVE_C(kNameAddcdiv, Addcdiv);
MIND_API_OPERATOR_IMPL(Addcdiv, BaseOperator);

REGISTER_PRIMITIVE_C(kNameAddcmul, Addcmul);
MIND_API_OPERATOR_IMPL(Addcmul, BaseOperator);

REGISTER_PRIMITIVE_C(kNameAddmm, Addmm);
MIND_API_OPERATOR_IMPL(Addmm, BaseOperator);

REGISTER_PRIMITIVE_C(kNameAddN, AddN);
MIND_API_OPERATOR_IMPL(AddN, BaseOperator);

REGISTER_PRIMITIVE_C(kNameAngle, Angle);
MIND_API_OPERATOR_IMPL(Angle, BaseOperator);

REGISTER_PRIMITIVE_C(kNameApplyCamePart1, ApplyCamePart1);
MIND_API_OPERATOR_IMPL(ApplyCamePart1, BaseOperator);

REGISTER_PRIMITIVE_C(kNameApplyCamePart2, ApplyCamePart2);
MIND_API_OPERATOR_IMPL(ApplyCamePart2, BaseOperator);

REGISTER_PRIMITIVE_C(kNameApplyCamePart3, ApplyCamePart3);
MIND_API_OPERATOR_IMPL(ApplyCamePart3, BaseOperator);

REGISTER_PRIMITIVE_C(kNameApplyCamePart4, ApplyCamePart4);
MIND_API_OPERATOR_IMPL(ApplyCamePart4, BaseOperator);

void ApplyRotaryPosEmb::set_cos_format(const int64_t &cos_format)             { (void)this->AddAttr("cos_format", api::MakeValue(cos_format)); }

int64_t ApplyRotaryPosEmb::get_cos_format() const             { return GetValue<int64_t>(GetAttr("cos_format")); }

REGISTER_PRIMITIVE_C(kNameApplyRotaryPosEmb, ApplyRotaryPosEmb);
MIND_API_OPERATOR_IMPL(ApplyRotaryPosEmb, BaseOperator);

REGISTER_PRIMITIVE_C(kNameArange, Arange);
MIND_API_OPERATOR_IMPL(Arange, BaseOperator);

REGISTER_PRIMITIVE_C(kNameArgMaxExt, ArgMaxExt);
MIND_API_OPERATOR_IMPL(ArgMaxExt, BaseOperator);

void Argmax::set_axis(const int64_t &axis)             { (void)this->AddAttr("axis", api::MakeValue(axis)); }

int64_t Argmax::get_axis() const             { return GetValue<int64_t>(GetAttr("axis")); }

void Argmax::set_output_type(const int64_t &output_type)             { (void)this->AddAttr("output_type", api::MakeValue(output_type)); }

int64_t Argmax::get_output_type() const             { return GetValue<int64_t>(GetAttr("output_type")); }

REGISTER_PRIMITIVE_C(kNameArgmax, Argmax);
MIND_API_OPERATOR_IMPL(Argmax, BaseOperator);

void ArgMaxWithValue::set_axis(const int64_t &axis)             { (void)this->AddAttr("axis", api::MakeValue(axis)); }

int64_t ArgMaxWithValue::get_axis() const             { return GetValue<int64_t>(GetAttr("axis")); }

void ArgMaxWithValue::set_keep_dims(const bool &keep_dims)             { (void)this->AddAttr("keep_dims", api::MakeValue(keep_dims)); }

bool ArgMaxWithValue::get_keep_dims() const             { return GetValue<bool>(GetAttr("keep_dims")); }

REGISTER_PRIMITIVE_C(kNameArgMaxWithValue, ArgMaxWithValue);
MIND_API_OPERATOR_IMPL(ArgMaxWithValue, BaseOperator);

void Argmin::set_axis(const int64_t &axis)             { (void)this->AddAttr("axis", api::MakeValue(axis)); }

int64_t Argmin::get_axis() const             { return GetValue<int64_t>(GetAttr("axis")); }

void Argmin::set_output_type(const int64_t &output_type)             { (void)this->AddAttr("output_type", api::MakeValue(output_type)); }

int64_t Argmin::get_output_type() const             { return GetValue<int64_t>(GetAttr("output_type")); }

REGISTER_PRIMITIVE_C(kNameArgmin, Argmin);
MIND_API_OPERATOR_IMPL(Argmin, BaseOperator);

void ArgMinWithValue::set_axis(const int64_t &axis)             { (void)this->AddAttr("axis", api::MakeValue(axis)); }

int64_t ArgMinWithValue::get_axis() const             { return GetValue<int64_t>(GetAttr("axis")); }

void ArgMinWithValue::set_keep_dims(const bool &keep_dims)             { (void)this->AddAttr("keep_dims", api::MakeValue(keep_dims)); }

bool ArgMinWithValue::get_keep_dims() const             { return GetValue<bool>(GetAttr("keep_dims")); }

REGISTER_PRIMITIVE_C(kNameArgMinWithValue, ArgMinWithValue);
MIND_API_OPERATOR_IMPL(ArgMinWithValue, BaseOperator);

REGISTER_PRIMITIVE_C(kNameAsinGrad, AsinGrad);
MIND_API_OPERATOR_IMPL(AsinGrad, BaseOperator);

REGISTER_PRIMITIVE_C(kNameAsin, Asin);
MIND_API_OPERATOR_IMPL(Asin, BaseOperator);

REGISTER_PRIMITIVE_C(kNameAsinhGrad, AsinhGrad);
MIND_API_OPERATOR_IMPL(AsinhGrad, BaseOperator);

REGISTER_PRIMITIVE_C(kNameAsinh, Asinh);
MIND_API_OPERATOR_IMPL(Asinh, BaseOperator);

REGISTER_PRIMITIVE_C(kNameAssignAdd, AssignAdd);
MIND_API_OPERATOR_IMPL(AssignAdd, BaseOperator);

REGISTER_PRIMITIVE_C(kNameAssign, Assign);
MIND_API_OPERATOR_IMPL(Assign, BaseOperator);

REGISTER_PRIMITIVE_C(kNameAtan2Ext, Atan2Ext);
MIND_API_OPERATOR_IMPL(Atan2Ext, BaseOperator);

REGISTER_PRIMITIVE_C(kNameAtan2, Atan2);
MIND_API_OPERATOR_IMPL(Atan2, BaseOperator);

REGISTER_PRIMITIVE_C(kNameAtanGrad, AtanGrad);
MIND_API_OPERATOR_IMPL(AtanGrad, BaseOperator);

REGISTER_PRIMITIVE_C(kNameAtan, Atan);
MIND_API_OPERATOR_IMPL(Atan, BaseOperator);

REGISTER_PRIMITIVE_C(kNameAtanh, Atanh);
MIND_API_OPERATOR_IMPL(Atanh, BaseOperator);

REGISTER_PRIMITIVE_C(kNameAvgPool2DGrad, AvgPool2DGrad);
MIND_API_OPERATOR_IMPL(AvgPool2DGrad, BaseOperator);

REGISTER_PRIMITIVE_C(kNameAvgPool2D, AvgPool2D);
MIND_API_OPERATOR_IMPL(AvgPool2D, BaseOperator);

void AvgPoolGrad::set_kernel_size(const std::vector<int64_t> &kernel_size)             { (void)this->AddAttr("kernel_size", api::MakeValue(kernel_size)); }

std::vector<int64_t> AvgPoolGrad::get_kernel_size() const             { return GetValue<std::vector<int64_t>>(GetAttr("kernel_size")); }

void AvgPoolGrad::set_strides(const std::vector<int64_t> &strides)             { (void)this->AddAttr("strides", api::MakeValue(strides)); }

std::vector<int64_t> AvgPoolGrad::get_strides() const             { return GetValue<std::vector<int64_t>>(GetAttr("strides")); }

void AvgPoolGrad::set_pad_mode(const int64_t &pad_mode)             { (void)this->AddAttr("pad_mode", api::MakeValue(pad_mode)); }

int64_t AvgPoolGrad::get_pad_mode() const             { return GetValue<int64_t>(GetAttr("pad_mode")); }

void AvgPoolGrad::set_data_format(const int64_t &data_format)             { (void)this->AddAttr("data_format", api::MakeValue(data_format)); }

int64_t AvgPoolGrad::get_data_format() const             { return GetValue<int64_t>(GetAttr("data_format")); }

REGISTER_PRIMITIVE_C(kNameAvgPoolGrad, AvgPoolGrad);
MIND_API_OPERATOR_IMPL(AvgPoolGrad, BaseOperator);

void AvgPool::set_kernel_size(const std::vector<int64_t> &kernel_size)             { (void)this->AddAttr("kernel_size", api::MakeValue(kernel_size)); }

std::vector<int64_t> AvgPool::get_kernel_size() const             { return GetValue<std::vector<int64_t>>(GetAttr("kernel_size")); }

void AvgPool::set_strides(const std::vector<int64_t> &strides)             { (void)this->AddAttr("strides", api::MakeValue(strides)); }

std::vector<int64_t> AvgPool::get_strides() const             { return GetValue<std::vector<int64_t>>(GetAttr("strides")); }

void AvgPool::set_pad_mode(const int64_t &pad_mode)             { (void)this->AddAttr("pad_mode", api::MakeValue(pad_mode)); }

int64_t AvgPool::get_pad_mode() const             { return GetValue<int64_t>(GetAttr("pad_mode")); }

void AvgPool::set_data_format(const int64_t &data_format)             { (void)this->AddAttr("data_format", api::MakeValue(data_format)); }

int64_t AvgPool::get_data_format() const             { return GetValue<int64_t>(GetAttr("data_format")); }

REGISTER_PRIMITIVE_C(kNameAvgPool, AvgPool);
MIND_API_OPERATOR_IMPL(AvgPool, BaseOperator);

void BatchMatMul::set_transpose_a(const bool &transpose_a)             { (void)this->AddAttr("transpose_a", api::MakeValue(transpose_a)); }

bool BatchMatMul::get_transpose_a() const             { return GetValue<bool>(GetAttr("transpose_a")); }

void BatchMatMul::set_transpose_b(const bool &transpose_b)             { (void)this->AddAttr("transpose_b", api::MakeValue(transpose_b)); }

bool BatchMatMul::get_transpose_b() const             { return GetValue<bool>(GetAttr("transpose_b")); }

REGISTER_PRIMITIVE_C(kNameBatchMatMul, BatchMatMul);
MIND_API_OPERATOR_IMPL(BatchMatMul, BaseOperator);

REGISTER_PRIMITIVE_C(kNameBatchNormExt, BatchNormExt);
MIND_API_OPERATOR_IMPL(BatchNormExt, BaseOperator);

void BatchNormGradExt::set_training(const bool &training)             { (void)this->AddAttr("training", api::MakeValue(training)); }

bool BatchNormGradExt::get_training() const             { return GetValue<bool>(GetAttr("training")); }

void BatchNormGradExt::set_eps(const float &eps)             { (void)this->AddAttr("eps", api::MakeValue(eps)); }

float BatchNormGradExt::get_eps() const             { return GetValue<float>(GetAttr("eps")); }

REGISTER_PRIMITIVE_C(kNameBatchNormGradExt, BatchNormGradExt);
MIND_API_OPERATOR_IMPL(BatchNormGradExt, BaseOperator);

void BatchNormGradGrad::set_is_training(const bool &is_training)             { (void)this->AddAttr("is_training", api::MakeValue(is_training)); }

bool BatchNormGradGrad::get_is_training() const             { return GetValue<bool>(GetAttr("is_training")); }

void BatchNormGradGrad::set_epsilon(const float &epsilon)             { (void)this->AddAttr("epsilon", api::MakeValue(epsilon)); }

float BatchNormGradGrad::get_epsilon() const             { return GetValue<float>(GetAttr("epsilon")); }

void BatchNormGradGrad::set_data_format(const int64_t &data_format)             { (void)this->AddAttr("data_format", api::MakeValue(data_format)); }

int64_t BatchNormGradGrad::get_data_format() const             { return GetValue<int64_t>(GetAttr("data_format")); }

REGISTER_PRIMITIVE_C(kNameBatchNormGradGrad, BatchNormGradGrad);
MIND_API_OPERATOR_IMPL(BatchNormGradGrad, BaseOperator);

void BatchNormGrad::set_is_training(const bool &is_training)             { (void)this->AddAttr("is_training", api::MakeValue(is_training)); }

bool BatchNormGrad::get_is_training() const             { return GetValue<bool>(GetAttr("is_training")); }

void BatchNormGrad::set_epsilon(const float &epsilon)             { (void)this->AddAttr("epsilon", api::MakeValue(epsilon)); }

float BatchNormGrad::get_epsilon() const             { return GetValue<float>(GetAttr("epsilon")); }

void BatchNormGrad::set_data_format(const int64_t &data_format)             { (void)this->AddAttr("data_format", api::MakeValue(data_format)); }

int64_t BatchNormGrad::get_data_format() const             { return GetValue<int64_t>(GetAttr("data_format")); }

REGISTER_PRIMITIVE_C(kNameBatchNormGrad, BatchNormGrad);
MIND_API_OPERATOR_IMPL(BatchNormGrad, BaseOperator);

void BatchNormGradWithActivation::set_is_training(const bool &is_training)             { (void)this->AddAttr("is_training", api::MakeValue(is_training)); }

bool BatchNormGradWithActivation::get_is_training() const             { return GetValue<bool>(GetAttr("is_training")); }

void BatchNormGradWithActivation::set_epsilon(const float &epsilon)             { (void)this->AddAttr("epsilon", api::MakeValue(epsilon)); }

float BatchNormGradWithActivation::get_epsilon() const             { return GetValue<float>(GetAttr("epsilon")); }

void BatchNormGradWithActivation::set_data_format(const int64_t &data_format)             { (void)this->AddAttr("data_format", api::MakeValue(data_format)); }

int64_t BatchNormGradWithActivation::get_data_format() const             { return GetValue<int64_t>(GetAttr("data_format")); }

REGISTER_PRIMITIVE_C(kNameBatchNormGradWithActivation, BatchNormGradWithActivation);
MIND_API_OPERATOR_IMPL(BatchNormGradWithActivation, BaseOperator);

void BatchNorm::set_is_training(const bool &is_training)             { (void)this->AddAttr("is_training", api::MakeValue(is_training)); }

bool BatchNorm::get_is_training() const             { return GetValue<bool>(GetAttr("is_training")); }

void BatchNorm::set_epsilon(const float &epsilon)             { (void)this->AddAttr("epsilon", api::MakeValue(epsilon)); }

float BatchNorm::get_epsilon() const             { return GetValue<float>(GetAttr("epsilon")); }

void BatchNorm::set_momentum(const float &momentum)             { (void)this->AddAttr("momentum", api::MakeValue(momentum)); }

float BatchNorm::get_momentum() const             { return GetValue<float>(GetAttr("momentum")); }

void BatchNorm::set_data_format(const int64_t &data_format)             { (void)this->AddAttr("data_format", api::MakeValue(data_format)); }

int64_t BatchNorm::get_data_format() const             { return GetValue<int64_t>(GetAttr("data_format")); }

REGISTER_PRIMITIVE_C(kNameBatchNorm, BatchNorm);
MIND_API_OPERATOR_IMPL(BatchNorm, BaseOperator);

void BatchNormWithActivation::set_is_training(const bool &is_training)             { (void)this->AddAttr("is_training", api::MakeValue(is_training)); }

bool BatchNormWithActivation::get_is_training() const             { return GetValue<bool>(GetAttr("is_training")); }

void BatchNormWithActivation::set_epsilon(const float &epsilon)             { (void)this->AddAttr("epsilon", api::MakeValue(epsilon)); }

float BatchNormWithActivation::get_epsilon() const             { return GetValue<float>(GetAttr("epsilon")); }

void BatchNormWithActivation::set_momentum(const float &momentum)             { (void)this->AddAttr("momentum", api::MakeValue(momentum)); }

float BatchNormWithActivation::get_momentum() const             { return GetValue<float>(GetAttr("momentum")); }

void BatchNormWithActivation::set_data_format(const int64_t &data_format)             { (void)this->AddAttr("data_format", api::MakeValue(data_format)); }

int64_t BatchNormWithActivation::get_data_format() const             { return GetValue<int64_t>(GetAttr("data_format")); }

REGISTER_PRIMITIVE_C(kNameBatchNormWithActivation, BatchNormWithActivation);
MIND_API_OPERATOR_IMPL(BatchNormWithActivation, BaseOperator);

void BatchNormWithAddAndActivation::set_is_training(const bool &is_training)             { (void)this->AddAttr("is_training", api::MakeValue(is_training)); }

bool BatchNormWithAddAndActivation::get_is_training() const             { return GetValue<bool>(GetAttr("is_training")); }

void BatchNormWithAddAndActivation::set_epsilon(const float &epsilon)             { (void)this->AddAttr("epsilon", api::MakeValue(epsilon)); }

float BatchNormWithAddAndActivation::get_epsilon() const             { return GetValue<float>(GetAttr("epsilon")); }

void BatchNormWithAddAndActivation::set_momentum(const float &momentum)             { (void)this->AddAttr("momentum", api::MakeValue(momentum)); }

float BatchNormWithAddAndActivation::get_momentum() const             { return GetValue<float>(GetAttr("momentum")); }

void BatchNormWithAddAndActivation::set_data_format(const int64_t &data_format)             { (void)this->AddAttr("data_format", api::MakeValue(data_format)); }

int64_t BatchNormWithAddAndActivation::get_data_format() const             { return GetValue<int64_t>(GetAttr("data_format")); }

REGISTER_PRIMITIVE_C(kNameBatchNormWithAddAndActivation, BatchNormWithAddAndActivation);
MIND_API_OPERATOR_IMPL(BatchNormWithAddAndActivation, BaseOperator);

REGISTER_PRIMITIVE_C(kNameBetainc, Betainc);
MIND_API_OPERATOR_IMPL(Betainc, BaseOperator);

void BiasAddGrad::set_data_format(const int64_t &data_format)             { (void)this->AddAttr("data_format", api::MakeValue(data_format)); }

int64_t BiasAddGrad::get_data_format() const             { return GetValue<int64_t>(GetAttr("data_format")); }

REGISTER_PRIMITIVE_C(kNameBiasAddGrad, BiasAddGrad);
MIND_API_OPERATOR_IMPL(BiasAddGrad, BaseOperator);

void BiasAdd::set_data_format(const int64_t &data_format)             { (void)this->AddAttr("data_format", api::MakeValue(data_format)); }

int64_t BiasAdd::get_data_format() const             { return GetValue<int64_t>(GetAttr("data_format")); }

REGISTER_PRIMITIVE_C(kNameBiasAdd, BiasAdd);
MIND_API_OPERATOR_IMPL(BiasAdd, BaseOperator);

void BinaryCrossEntropyGrad::set_reduction(const int64_t &reduction)             { (void)this->AddAttr("reduction", api::MakeValue(reduction)); }

int64_t BinaryCrossEntropyGrad::get_reduction() const             { return GetValue<int64_t>(GetAttr("reduction")); }

REGISTER_PRIMITIVE_C(kNameBinaryCrossEntropyGrad, BinaryCrossEntropyGrad);
MIND_API_OPERATOR_IMPL(BinaryCrossEntropyGrad, BaseOperator);

void BinaryCrossEntropy::set_reduction(const int64_t &reduction)             { (void)this->AddAttr("reduction", api::MakeValue(reduction)); }

int64_t BinaryCrossEntropy::get_reduction() const             { return GetValue<int64_t>(GetAttr("reduction")); }

REGISTER_PRIMITIVE_C(kNameBinaryCrossEntropy, BinaryCrossEntropy);
MIND_API_OPERATOR_IMPL(BinaryCrossEntropy, BaseOperator);

REGISTER_PRIMITIVE_C(kNameBinaryCrossEntropyWithLogitsBackward, BinaryCrossEntropyWithLogitsBackward);
MIND_API_OPERATOR_IMPL(BinaryCrossEntropyWithLogitsBackward, BaseOperator);

void BCEWithLogitsLoss::set_reduction(const int64_t &reduction)             { (void)this->AddAttr("reduction", api::MakeValue(reduction)); }

int64_t BCEWithLogitsLoss::get_reduction() const             { return GetValue<int64_t>(GetAttr("reduction")); }

REGISTER_PRIMITIVE_C(kNameBCEWithLogitsLoss, BCEWithLogitsLoss);
MIND_API_OPERATOR_IMPL(BCEWithLogitsLoss, BaseOperator);

REGISTER_PRIMITIVE_C(kNameBatchMatMulExt, BatchMatMulExt);
MIND_API_OPERATOR_IMPL(BatchMatMulExt, BaseOperator);

REGISTER_PRIMITIVE_C(kNameBoolNot, BoolNot);
MIND_API_OPERATOR_IMPL(BoolNot, BaseOperator);

void BroadcastTo::set_shape(const std::vector<int64_t> &shape)             { (void)this->AddAttr("shape", api::MakeValue(shape)); }

std::vector<int64_t> BroadcastTo::get_shape() const             { return GetValue<std::vector<int64_t>>(GetAttr("shape")); }

REGISTER_PRIMITIVE_C(kNameBroadcastTo, BroadcastTo);
MIND_API_OPERATOR_IMPL(BroadcastTo, BaseOperator);

REGISTER_PRIMITIVE_C(kNameCast, Cast);
MIND_API_OPERATOR_IMPL(Cast, BaseOperator);

REGISTER_PRIMITIVE_C(kNameCeil, Ceil);
MIND_API_OPERATOR_IMPL(Ceil, BaseOperator);

void CeLU::set_alpha(const float &alpha)             { (void)this->AddAttr("alpha", api::MakeValue(alpha)); }

float CeLU::get_alpha() const             { return GetValue<float>(GetAttr("alpha")); }

REGISTER_PRIMITIVE_C(kNameCeLU, CeLU);
MIND_API_OPERATOR_IMPL(CeLU, BaseOperator);

REGISTER_PRIMITIVE_C(kNameCholeskyGrad, CholeskyGrad);
MIND_API_OPERATOR_IMPL(CholeskyGrad, BaseOperator);

void CholeskyInverse::set_upper(const bool &upper)             { (void)this->AddAttr("upper", api::MakeValue(upper)); }

bool CholeskyInverse::get_upper() const             { return GetValue<bool>(GetAttr("upper")); }

REGISTER_PRIMITIVE_C(kNameCholeskyInverse, CholeskyInverse);
MIND_API_OPERATOR_IMPL(CholeskyInverse, BaseOperator);

void Cholesky::set_upper(const bool &upper)             { (void)this->AddAttr("upper", api::MakeValue(upper)); }

bool Cholesky::get_upper() const             { return GetValue<bool>(GetAttr("upper")); }

REGISTER_PRIMITIVE_C(kNameCholesky, Cholesky);
MIND_API_OPERATOR_IMPL(Cholesky, BaseOperator);

REGISTER_PRIMITIVE_C(kNameChunk, Chunk);
MIND_API_OPERATOR_IMPL(Chunk, BaseOperator);

REGISTER_PRIMITIVE_C(kNameClampScalar, ClampScalar);
MIND_API_OPERATOR_IMPL(ClampScalar, BaseOperator);

REGISTER_PRIMITIVE_C(kNameClampTensor, ClampTensor);
MIND_API_OPERATOR_IMPL(ClampTensor, BaseOperator);

REGISTER_PRIMITIVE_C(kNameCol2ImExt, Col2ImExt);
MIND_API_OPERATOR_IMPL(Col2ImExt, BaseOperator);

REGISTER_PRIMITIVE_C(kNameCol2ImGrad, Col2ImGrad);
MIND_API_OPERATOR_IMPL(Col2ImGrad, BaseOperator);

REGISTER_PRIMITIVE_C(kNameComplex, Complex);
MIND_API_OPERATOR_IMPL(Complex, BaseOperator);

void Concat::set_axis(const int64_t &axis)             { (void)this->AddAttr("axis", api::MakeValue(axis)); }

int64_t Concat::get_axis() const             { return GetValue<int64_t>(GetAttr("axis")); }

REGISTER_PRIMITIVE_C(kNameConcat, Concat);
MIND_API_OPERATOR_IMPL(Concat, BaseOperator);

REGISTER_PRIMITIVE_C(kNameConj, Conj);
MIND_API_OPERATOR_IMPL(Conj, BaseOperator);

REGISTER_PRIMITIVE_C(kNameConstantPadND, ConstantPadND);
MIND_API_OPERATOR_IMPL(ConstantPadND, BaseOperator);

REGISTER_PRIMITIVE_C(kNameContiguous, Contiguous);
MIND_API_OPERATOR_IMPL(Contiguous, BaseOperator);

void ConvolutionGrad::set_stride(const std::vector<int64_t> &stride)             { (void)this->AddAttr("stride", api::MakeValue(stride)); }

std::vector<int64_t> ConvolutionGrad::get_stride() const             { return GetValue<std::vector<int64_t>>(GetAttr("stride")); }

void ConvolutionGrad::set_padding(const std::vector<int64_t> &padding)             { (void)this->AddAttr("padding", api::MakeValue(padding)); }

std::vector<int64_t> ConvolutionGrad::get_padding() const             { return GetValue<std::vector<int64_t>>(GetAttr("padding")); }

void ConvolutionGrad::set_dilation(const std::vector<int64_t> &dilation)             { (void)this->AddAttr("dilation", api::MakeValue(dilation)); }

std::vector<int64_t> ConvolutionGrad::get_dilation() const             { return GetValue<std::vector<int64_t>>(GetAttr("dilation")); }

void ConvolutionGrad::set_transposed(const bool &transposed)             { (void)this->AddAttr("transposed", api::MakeValue(transposed)); }

bool ConvolutionGrad::get_transposed() const             { return GetValue<bool>(GetAttr("transposed")); }

void ConvolutionGrad::set_output_padding(const std::vector<int64_t> &output_padding)             { (void)this->AddAttr("output_padding", api::MakeValue(output_padding)); }

std::vector<int64_t> ConvolutionGrad::get_output_padding() const             { return GetValue<std::vector<int64_t>>(GetAttr("output_padding")); }

void ConvolutionGrad::set_groups(const int64_t &groups)             { (void)this->AddAttr("groups", api::MakeValue(groups)); }

int64_t ConvolutionGrad::get_groups() const             { return GetValue<int64_t>(GetAttr("groups")); }

void ConvolutionGrad::set_output_mask(const std::vector<int64_t> &output_mask)             { (void)this->AddAttr("output_mask", api::MakeValue(output_mask)); }

std::vector<int64_t> ConvolutionGrad::get_output_mask() const             { return GetValue<std::vector<int64_t>>(GetAttr("output_mask")); }

REGISTER_PRIMITIVE_C(kNameConvolutionGrad, ConvolutionGrad);
MIND_API_OPERATOR_IMPL(ConvolutionGrad, BaseOperator);

void Convolution::set_stride(const std::vector<int64_t> &stride)             { (void)this->AddAttr("stride", api::MakeValue(stride)); }

std::vector<int64_t> Convolution::get_stride() const             { return GetValue<std::vector<int64_t>>(GetAttr("stride")); }

void Convolution::set_padding(const std::vector<int64_t> &padding)             { (void)this->AddAttr("padding", api::MakeValue(padding)); }

std::vector<int64_t> Convolution::get_padding() const             { return GetValue<std::vector<int64_t>>(GetAttr("padding")); }

void Convolution::set_dilation(const std::vector<int64_t> &dilation)             { (void)this->AddAttr("dilation", api::MakeValue(dilation)); }

std::vector<int64_t> Convolution::get_dilation() const             { return GetValue<std::vector<int64_t>>(GetAttr("dilation")); }

void Convolution::set_transposed(const bool &transposed)             { (void)this->AddAttr("transposed", api::MakeValue(transposed)); }

bool Convolution::get_transposed() const             { return GetValue<bool>(GetAttr("transposed")); }

void Convolution::set_output_padding(const std::vector<int64_t> &output_padding)             { (void)this->AddAttr("output_padding", api::MakeValue(output_padding)); }

std::vector<int64_t> Convolution::get_output_padding() const             { return GetValue<std::vector<int64_t>>(GetAttr("output_padding")); }

void Convolution::set_groups(const int64_t &groups)             { (void)this->AddAttr("groups", api::MakeValue(groups)); }

int64_t Convolution::get_groups() const             { return GetValue<int64_t>(GetAttr("groups")); }

REGISTER_PRIMITIVE_C(kNameConvolution, Convolution);
MIND_API_OPERATOR_IMPL(Convolution, BaseOperator);

REGISTER_PRIMITIVE_C(kNameCopy, Copy);
MIND_API_OPERATOR_IMPL(Copy, BaseOperator);

void Correlate::set_mode(const int64_t &mode)             { (void)this->AddAttr("mode", api::MakeValue(mode)); }

int64_t Correlate::get_mode() const             { return GetValue<int64_t>(GetAttr("mode")); }

REGISTER_PRIMITIVE_C(kNameCorrelate, Correlate);
MIND_API_OPERATOR_IMPL(Correlate, BaseOperator);

REGISTER_PRIMITIVE_C(kNameCos, Cos);
MIND_API_OPERATOR_IMPL(Cos, BaseOperator);

REGISTER_PRIMITIVE_C(kNameCosh, Cosh);
MIND_API_OPERATOR_IMPL(Cosh, BaseOperator);

void CumProd::set_exclusive(const bool &exclusive)             { (void)this->AddAttr("exclusive", api::MakeValue(exclusive)); }

bool CumProd::get_exclusive() const             { return GetValue<bool>(GetAttr("exclusive")); }

void CumProd::set_reverse(const bool &reverse)             { (void)this->AddAttr("reverse", api::MakeValue(reverse)); }

bool CumProd::get_reverse() const             { return GetValue<bool>(GetAttr("reverse")); }

REGISTER_PRIMITIVE_C(kNameCumProd, CumProd);
MIND_API_OPERATOR_IMPL(CumProd, BaseOperator);

void CumSum::set_exclusive(const bool &exclusive)             { (void)this->AddAttr("exclusive", api::MakeValue(exclusive)); }

bool CumSum::get_exclusive() const             { return GetValue<bool>(GetAttr("exclusive")); }

void CumSum::set_reverse(const bool &reverse)             { (void)this->AddAttr("reverse", api::MakeValue(reverse)); }

bool CumSum::get_reverse() const             { return GetValue<bool>(GetAttr("reverse")); }

REGISTER_PRIMITIVE_C(kNameCumSum, CumSum);
MIND_API_OPERATOR_IMPL(CumSum, BaseOperator);

void Cummax::set_axis(const int64_t &axis)             { (void)this->AddAttr("axis", api::MakeValue(axis)); }

int64_t Cummax::get_axis() const             { return GetValue<int64_t>(GetAttr("axis")); }

REGISTER_PRIMITIVE_C(kNameCummax, Cummax);
MIND_API_OPERATOR_IMPL(Cummax, BaseOperator);

void Cummin::set_axis(const int64_t &axis)             { (void)this->AddAttr("axis", api::MakeValue(axis)); }

int64_t Cummin::get_axis() const             { return GetValue<int64_t>(GetAttr("axis")); }

REGISTER_PRIMITIVE_C(kNameCummin, Cummin);
MIND_API_OPERATOR_IMPL(Cummin, BaseOperator);

REGISTER_PRIMITIVE_C(kNameCumsumExt, CumsumExt);
MIND_API_OPERATOR_IMPL(CumsumExt, BaseOperator);

REGISTER_PRIMITIVE_C(kNameDCT, DCT);
MIND_API_OPERATOR_IMPL(DCT, BaseOperator);

REGISTER_PRIMITIVE_C(kNameDecoderKVCache, DecoderKVCache);
MIND_API_OPERATOR_IMPL(DecoderKVCache, BaseOperator);

REGISTER_PRIMITIVE_C(kNameDense, Dense);
MIND_API_OPERATOR_IMPL(Dense, BaseOperator);

REGISTER_PRIMITIVE_C(kNameDiag, Diag);
MIND_API_OPERATOR_IMPL(Diag, BaseOperator);

void Diagonal::set_offset(const int64_t &offset)             { (void)this->AddAttr("offset", api::MakeValue(offset)); }

int64_t Diagonal::get_offset() const             { return GetValue<int64_t>(GetAttr("offset")); }

void Diagonal::set_dim1(const int64_t &dim1)             { (void)this->AddAttr("dim1", api::MakeValue(dim1)); }

int64_t Diagonal::get_dim1() const             { return GetValue<int64_t>(GetAttr("dim1")); }

void Diagonal::set_dim2(const int64_t &dim2)             { (void)this->AddAttr("dim2", api::MakeValue(dim2)); }

int64_t Diagonal::get_dim2() const             { return GetValue<int64_t>(GetAttr("dim2")); }

REGISTER_PRIMITIVE_C(kNameDiagonal, Diagonal);
MIND_API_OPERATOR_IMPL(Diagonal, BaseOperator);

REGISTER_PRIMITIVE_C(kNameDiv, Div);
MIND_API_OPERATOR_IMPL(Div, BaseOperator);

REGISTER_PRIMITIVE_C(kNameDivMod, DivMod);
MIND_API_OPERATOR_IMPL(DivMod, BaseOperator);

REGISTER_PRIMITIVE_C(kNameDot, Dot);
MIND_API_OPERATOR_IMPL(Dot, BaseOperator);

REGISTER_PRIMITIVE_C(kNameDropoutDoMaskExt, DropoutDoMaskExt);
MIND_API_OPERATOR_IMPL(DropoutDoMaskExt, BaseOperator);

REGISTER_PRIMITIVE_C(kNameDropoutExt, DropoutExt);
MIND_API_OPERATOR_IMPL(DropoutExt, BaseOperator);

REGISTER_PRIMITIVE_C(kNameDropoutGenMaskExt, DropoutGenMaskExt);
MIND_API_OPERATOR_IMPL(DropoutGenMaskExt, BaseOperator);

REGISTER_PRIMITIVE_C(kNameDropoutGradExt, DropoutGradExt);
MIND_API_OPERATOR_IMPL(DropoutGradExt, BaseOperator);

void Dropout::set_keep_prob(const float &keep_prob)             { (void)this->AddAttr("keep_prob", api::MakeValue(keep_prob)); }

float Dropout::get_keep_prob() const             { return GetValue<float>(GetAttr("keep_prob")); }

void Dropout::set_Seed0(const int64_t &Seed0)             { (void)this->AddAttr("Seed0", api::MakeValue(Seed0)); }

int64_t Dropout::get_Seed0() const             { return GetValue<int64_t>(GetAttr("Seed0")); }

void Dropout::set_Seed1(const int64_t &Seed1)             { (void)this->AddAttr("Seed1", api::MakeValue(Seed1)); }

int64_t Dropout::get_Seed1() const             { return GetValue<int64_t>(GetAttr("Seed1")); }

REGISTER_PRIMITIVE_C(kNameDropout, Dropout);
MIND_API_OPERATOR_IMPL(Dropout, BaseOperator);

void Eig::set_compute_v(const bool &compute_v)             { (void)this->AddAttr("compute_v", api::MakeValue(compute_v)); }

bool Eig::get_compute_v() const             { return GetValue<bool>(GetAttr("compute_v")); }

REGISTER_PRIMITIVE_C(kNameEig, Eig);
MIND_API_OPERATOR_IMPL(Eig, BaseOperator);

void EluExt::set_alpha(const float &alpha)             { (void)this->AddAttr("alpha", api::MakeValue(alpha)); }

float EluExt::get_alpha() const             { return GetValue<float>(GetAttr("alpha")); }

REGISTER_PRIMITIVE_C(kNameEluExt, EluExt);
MIND_API_OPERATOR_IMPL(EluExt, BaseOperator);

REGISTER_PRIMITIVE_C(kNameEluGradExt, EluGradExt);
MIND_API_OPERATOR_IMPL(EluGradExt, BaseOperator);

REGISTER_PRIMITIVE_C(kNameEluGrad, EluGrad);
MIND_API_OPERATOR_IMPL(EluGrad, BaseOperator);

void Elu::set_alpha(const float &alpha)             { (void)this->AddAttr("alpha", api::MakeValue(alpha)); }

float Elu::get_alpha() const             { return GetValue<float>(GetAttr("alpha")); }

REGISTER_PRIMITIVE_C(kNameElu, Elu);
MIND_API_OPERATOR_IMPL(Elu, BaseOperator);

REGISTER_PRIMITIVE_C(kNameEmbeddingDenseBackward, EmbeddingDenseBackward);
MIND_API_OPERATOR_IMPL(EmbeddingDenseBackward, BaseOperator);

REGISTER_PRIMITIVE_C(kNameEmbedding, Embedding);
MIND_API_OPERATOR_IMPL(Embedding, BaseOperator);

REGISTER_PRIMITIVE_C(kNameEqual, Equal);
MIND_API_OPERATOR_IMPL(Equal, BaseOperator);

REGISTER_PRIMITIVE_C(kNameErf, Erf);
MIND_API_OPERATOR_IMPL(Erf, BaseOperator);

REGISTER_PRIMITIVE_C(kNameErfc, Erfc);
MIND_API_OPERATOR_IMPL(Erfc, BaseOperator);

REGISTER_PRIMITIVE_C(kNameErfinv, Erfinv);
MIND_API_OPERATOR_IMPL(Erfinv, BaseOperator);

REGISTER_PRIMITIVE_C(kNameExp, Exp);
MIND_API_OPERATOR_IMPL(Exp, BaseOperator);

REGISTER_PRIMITIVE_C(kNameExpandDims, ExpandDims);
MIND_API_OPERATOR_IMPL(ExpandDims, BaseOperator);

REGISTER_PRIMITIVE_C(kNameExpm1, Expm1);
MIND_API_OPERATOR_IMPL(Expm1, BaseOperator);

void ExtractImagePatches::set_ksizes(const std::vector<int64_t> &ksizes)             { (void)this->AddAttr("ksizes", api::MakeValue(ksizes)); }

std::vector<int64_t> ExtractImagePatches::get_ksizes() const             { return GetValue<std::vector<int64_t>>(GetAttr("ksizes")); }

void ExtractImagePatches::set_strides(const std::vector<int64_t> &strides)             { (void)this->AddAttr("strides", api::MakeValue(strides)); }

std::vector<int64_t> ExtractImagePatches::get_strides() const             { return GetValue<std::vector<int64_t>>(GetAttr("strides")); }

void ExtractImagePatches::set_rates(const std::vector<int64_t> &rates)             { (void)this->AddAttr("rates", api::MakeValue(rates)); }

std::vector<int64_t> ExtractImagePatches::get_rates() const             { return GetValue<std::vector<int64_t>>(GetAttr("rates")); }

void ExtractImagePatches::set_padding(const int64_t &padding)             { (void)this->AddAttr("padding", api::MakeValue(padding)); }

int64_t ExtractImagePatches::get_padding() const             { return GetValue<int64_t>(GetAttr("padding")); }

REGISTER_PRIMITIVE_C(kNameExtractImagePatches, ExtractImagePatches);
MIND_API_OPERATOR_IMPL(ExtractImagePatches, BaseOperator);

REGISTER_PRIMITIVE_C(kNameEye, Eye);
MIND_API_OPERATOR_IMPL(Eye, BaseOperator);

REGISTER_PRIMITIVE_C(kNameFastGeLUGrad, FastGeLUGrad);
MIND_API_OPERATOR_IMPL(FastGeLUGrad, BaseOperator);

REGISTER_PRIMITIVE_C(kNameFastGeLU, FastGeLU);
MIND_API_OPERATOR_IMPL(FastGeLU, BaseOperator);

void FFNExt::set_activation(const int64_t &activation)             { (void)this->AddAttr("activation", api::MakeValue(activation)); }

int64_t FFNExt::get_activation() const             { return GetValue<int64_t>(GetAttr("activation")); }

void FFNExt::set_inner_precise(const int64_t &inner_precise)             { (void)this->AddAttr("inner_precise", api::MakeValue(inner_precise)); }

int64_t FFNExt::get_inner_precise() const             { return GetValue<int64_t>(GetAttr("inner_precise")); }

REGISTER_PRIMITIVE_C(kNameFFNExt, FFNExt);
MIND_API_OPERATOR_IMPL(FFNExt, BaseOperator);

REGISTER_PRIMITIVE_C(kNameFFT2, FFT2);
MIND_API_OPERATOR_IMPL(FFT2, BaseOperator);

REGISTER_PRIMITIVE_C(kNameFFT, FFT);
MIND_API_OPERATOR_IMPL(FFT, BaseOperator);

REGISTER_PRIMITIVE_C(kNameFFTShapeCopy, FFTShapeCopy);
MIND_API_OPERATOR_IMPL(FFTShapeCopy, BaseOperator);

void FFTWithSize::set_signal_ndim(const int64_t &signal_ndim)             { (void)this->AddAttr("signal_ndim", api::MakeValue(signal_ndim)); }

int64_t FFTWithSize::get_signal_ndim() const             { return GetValue<int64_t>(GetAttr("signal_ndim")); }

void FFTWithSize::set_inverse(const bool &inverse)             { (void)this->AddAttr("inverse", api::MakeValue(inverse)); }

bool FFTWithSize::get_inverse() const             { return GetValue<bool>(GetAttr("inverse")); }

void FFTWithSize::set_real(const bool &real)             { (void)this->AddAttr("real", api::MakeValue(real)); }

bool FFTWithSize::get_real() const             { return GetValue<bool>(GetAttr("real")); }

void FFTWithSize::set_norm(const int64_t &norm)             { (void)this->AddAttr("norm", api::MakeValue(norm)); }

int64_t FFTWithSize::get_norm() const             { return GetValue<int64_t>(GetAttr("norm")); }

void FFTWithSize::set_onesided(const bool &onesided)             { (void)this->AddAttr("onesided", api::MakeValue(onesided)); }

bool FFTWithSize::get_onesided() const             { return GetValue<bool>(GetAttr("onesided")); }

void FFTWithSize::set_signal_sizes(const std::vector<int64_t> &signal_sizes)             { (void)this->AddAttr("signal_sizes", api::MakeValue(signal_sizes)); }

std::vector<int64_t> FFTWithSize::get_signal_sizes() const             { return GetValue<std::vector<int64_t>>(GetAttr("signal_sizes")); }

REGISTER_PRIMITIVE_C(kNameFFTWithSize, FFTWithSize);
MIND_API_OPERATOR_IMPL(FFTWithSize, BaseOperator);

REGISTER_PRIMITIVE_C(kNameFFTN, FFTN);
MIND_API_OPERATOR_IMPL(FFTN, BaseOperator);

REGISTER_PRIMITIVE_C(kNameFFTShift, FFTShift);
MIND_API_OPERATOR_IMPL(FFTShift, BaseOperator);

REGISTER_PRIMITIVE_C(kNameFillScalar, FillScalar);
MIND_API_OPERATOR_IMPL(FillScalar, BaseOperator);

REGISTER_PRIMITIVE_C(kNameFillTensor, FillTensor);
MIND_API_OPERATOR_IMPL(FillTensor, BaseOperator);

void FlashAttentionScoreGrad::set_head_num(const int64_t &head_num)             { (void)this->AddAttr("head_num", api::MakeValue(head_num)); }

int64_t FlashAttentionScoreGrad::get_head_num() const             { return GetValue<int64_t>(GetAttr("head_num")); }

void FlashAttentionScoreGrad::set_keep_prob(const float &keep_prob)             { (void)this->AddAttr("keep_prob", api::MakeValue(keep_prob)); }

float FlashAttentionScoreGrad::get_keep_prob() const             { return GetValue<float>(GetAttr("keep_prob")); }

void FlashAttentionScoreGrad::set_scale_value(const float &scale_value)             { (void)this->AddAttr("scale_value", api::MakeValue(scale_value)); }

float FlashAttentionScoreGrad::get_scale_value() const             { return GetValue<float>(GetAttr("scale_value")); }

void FlashAttentionScoreGrad::set_pre_tokens(const int64_t &pre_tokens)             { (void)this->AddAttr("pre_tokens", api::MakeValue(pre_tokens)); }

int64_t FlashAttentionScoreGrad::get_pre_tokens() const             { return GetValue<int64_t>(GetAttr("pre_tokens")); }

void FlashAttentionScoreGrad::set_next_tokens(const int64_t &next_tokens)             { (void)this->AddAttr("next_tokens", api::MakeValue(next_tokens)); }

int64_t FlashAttentionScoreGrad::get_next_tokens() const             { return GetValue<int64_t>(GetAttr("next_tokens")); }

void FlashAttentionScoreGrad::set_inner_precise(const int64_t &inner_precise)             { (void)this->AddAttr("inner_precise", api::MakeValue(inner_precise)); }

int64_t FlashAttentionScoreGrad::get_inner_precise() const             { return GetValue<int64_t>(GetAttr("inner_precise")); }

void FlashAttentionScoreGrad::set_input_layout(const int64_t &input_layout)             { (void)this->AddAttr("input_layout", api::MakeValue(input_layout)); }

int64_t FlashAttentionScoreGrad::get_input_layout() const             { return GetValue<int64_t>(GetAttr("input_layout")); }

void FlashAttentionScoreGrad::set_sparse_mode(const int64_t &sparse_mode)             { (void)this->AddAttr("sparse_mode", api::MakeValue(sparse_mode)); }

int64_t FlashAttentionScoreGrad::get_sparse_mode() const             { return GetValue<int64_t>(GetAttr("sparse_mode")); }

REGISTER_PRIMITIVE_C(kNameFlashAttentionScoreGrad, FlashAttentionScoreGrad);
MIND_API_OPERATOR_IMPL(FlashAttentionScoreGrad, BaseOperator);

void FlashAttentionScore::set_head_num(const int64_t &head_num)             { (void)this->AddAttr("head_num", api::MakeValue(head_num)); }

int64_t FlashAttentionScore::get_head_num() const             { return GetValue<int64_t>(GetAttr("head_num")); }

void FlashAttentionScore::set_keep_prob(const float &keep_prob)             { (void)this->AddAttr("keep_prob", api::MakeValue(keep_prob)); }

float FlashAttentionScore::get_keep_prob() const             { return GetValue<float>(GetAttr("keep_prob")); }

void FlashAttentionScore::set_scale_value(const float &scale_value)             { (void)this->AddAttr("scale_value", api::MakeValue(scale_value)); }

float FlashAttentionScore::get_scale_value() const             { return GetValue<float>(GetAttr("scale_value")); }

void FlashAttentionScore::set_pre_tokens(const int64_t &pre_tokens)             { (void)this->AddAttr("pre_tokens", api::MakeValue(pre_tokens)); }

int64_t FlashAttentionScore::get_pre_tokens() const             { return GetValue<int64_t>(GetAttr("pre_tokens")); }

void FlashAttentionScore::set_next_tokens(const int64_t &next_tokens)             { (void)this->AddAttr("next_tokens", api::MakeValue(next_tokens)); }

int64_t FlashAttentionScore::get_next_tokens() const             { return GetValue<int64_t>(GetAttr("next_tokens")); }

void FlashAttentionScore::set_inner_precise(const int64_t &inner_precise)             { (void)this->AddAttr("inner_precise", api::MakeValue(inner_precise)); }

int64_t FlashAttentionScore::get_inner_precise() const             { return GetValue<int64_t>(GetAttr("inner_precise")); }

void FlashAttentionScore::set_input_layout(const int64_t &input_layout)             { (void)this->AddAttr("input_layout", api::MakeValue(input_layout)); }

int64_t FlashAttentionScore::get_input_layout() const             { return GetValue<int64_t>(GetAttr("input_layout")); }

void FlashAttentionScore::set_sparse_mode(const int64_t &sparse_mode)             { (void)this->AddAttr("sparse_mode", api::MakeValue(sparse_mode)); }

int64_t FlashAttentionScore::get_sparse_mode() const             { return GetValue<int64_t>(GetAttr("sparse_mode")); }

REGISTER_PRIMITIVE_C(kNameFlashAttentionScore, FlashAttentionScore);
MIND_API_OPERATOR_IMPL(FlashAttentionScore, BaseOperator);

REGISTER_PRIMITIVE_C(kNameFlattenExt, FlattenExt);
MIND_API_OPERATOR_IMPL(FlattenExt, BaseOperator);

REGISTER_PRIMITIVE_C(kNameFlatten, Flatten);
MIND_API_OPERATOR_IMPL(Flatten, BaseOperator);

REGISTER_PRIMITIVE_C(kNameFloorDiv, FloorDiv);
MIND_API_OPERATOR_IMPL(FloorDiv, BaseOperator);

REGISTER_PRIMITIVE_C(kNameFloorMod, FloorMod);
MIND_API_OPERATOR_IMPL(FloorMod, BaseOperator);

REGISTER_PRIMITIVE_C(kNameFloor, Floor);
MIND_API_OPERATOR_IMPL(Floor, BaseOperator);

REGISTER_PRIMITIVE_C(kNameGatherDGradV2, GatherDGradV2);
MIND_API_OPERATOR_IMPL(GatherDGradV2, BaseOperator);

REGISTER_PRIMITIVE_C(kNameGatherD, GatherD);
MIND_API_OPERATOR_IMPL(GatherD, BaseOperator);

REGISTER_PRIMITIVE_C(kNameGatherNd, GatherNd);
MIND_API_OPERATOR_IMPL(GatherNd, BaseOperator);

void Gather::set_batch_dims(const int64_t &batch_dims)             { (void)this->AddAttr("batch_dims", api::MakeValue(batch_dims)); }

int64_t Gather::get_batch_dims() const             { return GetValue<int64_t>(GetAttr("batch_dims")); }

REGISTER_PRIMITIVE_C(kNameGather, Gather);
MIND_API_OPERATOR_IMPL(Gather, BaseOperator);

REGISTER_PRIMITIVE_C(kNameGcd, Gcd);
MIND_API_OPERATOR_IMPL(Gcd, BaseOperator);

REGISTER_PRIMITIVE_C(kNameGeLUGrad, GeLUGrad);
MIND_API_OPERATOR_IMPL(GeLUGrad, BaseOperator);

REGISTER_PRIMITIVE_C(kNameGeLU, GeLU);
MIND_API_OPERATOR_IMPL(GeLU, BaseOperator);

REGISTER_PRIMITIVE_C(kNameGenerator, Generator);
MIND_API_OPERATOR_IMPL(Generator, BaseOperator);

REGISTER_PRIMITIVE_C(kNameGeqrf, Geqrf);
MIND_API_OPERATOR_IMPL(Geqrf, BaseOperator);

REGISTER_PRIMITIVE_C(kNameGreaterEqual, GreaterEqual);
MIND_API_OPERATOR_IMPL(GreaterEqual, BaseOperator);

REGISTER_PRIMITIVE_C(kNameGreater, Greater);
MIND_API_OPERATOR_IMPL(Greater, BaseOperator);

void GridSampler2DGrad::set_interpolation_mode(const int64_t &interpolation_mode)             { (void)this->AddAttr("interpolation_mode", api::MakeValue(interpolation_mode)); }

int64_t GridSampler2DGrad::get_interpolation_mode() const             { return GetValue<int64_t>(GetAttr("interpolation_mode")); }

void GridSampler2DGrad::set_padding_mode(const int64_t &padding_mode)             { (void)this->AddAttr("padding_mode", api::MakeValue(padding_mode)); }

int64_t GridSampler2DGrad::get_padding_mode() const             { return GetValue<int64_t>(GetAttr("padding_mode")); }

void GridSampler2DGrad::set_align_corners(const bool &align_corners)             { (void)this->AddAttr("align_corners", api::MakeValue(align_corners)); }

bool GridSampler2DGrad::get_align_corners() const             { return GetValue<bool>(GetAttr("align_corners")); }

REGISTER_PRIMITIVE_C(kNameGridSampler2DGrad, GridSampler2DGrad);
MIND_API_OPERATOR_IMPL(GridSampler2DGrad, BaseOperator);

void GridSampler2D::set_interpolation_mode(const int64_t &interpolation_mode)             { (void)this->AddAttr("interpolation_mode", api::MakeValue(interpolation_mode)); }

int64_t GridSampler2D::get_interpolation_mode() const             { return GetValue<int64_t>(GetAttr("interpolation_mode")); }

void GridSampler2D::set_padding_mode(const int64_t &padding_mode)             { (void)this->AddAttr("padding_mode", api::MakeValue(padding_mode)); }

int64_t GridSampler2D::get_padding_mode() const             { return GetValue<int64_t>(GetAttr("padding_mode")); }

void GridSampler2D::set_align_corners(const bool &align_corners)             { (void)this->AddAttr("align_corners", api::MakeValue(align_corners)); }

bool GridSampler2D::get_align_corners() const             { return GetValue<bool>(GetAttr("align_corners")); }

REGISTER_PRIMITIVE_C(kNameGridSampler2D, GridSampler2D);
MIND_API_OPERATOR_IMPL(GridSampler2D, BaseOperator);

void GridSampler3DGrad::set_interpolation_mode(const int64_t &interpolation_mode)             { (void)this->AddAttr("interpolation_mode", api::MakeValue(interpolation_mode)); }

int64_t GridSampler3DGrad::get_interpolation_mode() const             { return GetValue<int64_t>(GetAttr("interpolation_mode")); }

void GridSampler3DGrad::set_padding_mode(const int64_t &padding_mode)             { (void)this->AddAttr("padding_mode", api::MakeValue(padding_mode)); }

int64_t GridSampler3DGrad::get_padding_mode() const             { return GetValue<int64_t>(GetAttr("padding_mode")); }

void GridSampler3DGrad::set_align_corners(const bool &align_corners)             { (void)this->AddAttr("align_corners", api::MakeValue(align_corners)); }

bool GridSampler3DGrad::get_align_corners() const             { return GetValue<bool>(GetAttr("align_corners")); }

REGISTER_PRIMITIVE_C(kNameGridSampler3DGrad, GridSampler3DGrad);
MIND_API_OPERATOR_IMPL(GridSampler3DGrad, BaseOperator);

void GridSampler3D::set_interpolation_mode(const int64_t &interpolation_mode)             { (void)this->AddAttr("interpolation_mode", api::MakeValue(interpolation_mode)); }

int64_t GridSampler3D::get_interpolation_mode() const             { return GetValue<int64_t>(GetAttr("interpolation_mode")); }

void GridSampler3D::set_padding_mode(const int64_t &padding_mode)             { (void)this->AddAttr("padding_mode", api::MakeValue(padding_mode)); }

int64_t GridSampler3D::get_padding_mode() const             { return GetValue<int64_t>(GetAttr("padding_mode")); }

void GridSampler3D::set_align_corners(const bool &align_corners)             { (void)this->AddAttr("align_corners", api::MakeValue(align_corners)); }

bool GridSampler3D::get_align_corners() const             { return GetValue<bool>(GetAttr("align_corners")); }

REGISTER_PRIMITIVE_C(kNameGridSampler3D, GridSampler3D);
MIND_API_OPERATOR_IMPL(GridSampler3D, BaseOperator);

REGISTER_PRIMITIVE_C(kNameGroupNormGrad, GroupNormGrad);
MIND_API_OPERATOR_IMPL(GroupNormGrad, BaseOperator);

REGISTER_PRIMITIVE_C(kNameGroupNorm, GroupNorm);
MIND_API_OPERATOR_IMPL(GroupNorm, BaseOperator);

void HShrinkGrad::set_lambd(const float &lambd)             { (void)this->AddAttr("lambd", api::MakeValue(lambd)); }

float HShrinkGrad::get_lambd() const             { return GetValue<float>(GetAttr("lambd")); }

REGISTER_PRIMITIVE_C(kNameHShrinkGrad, HShrinkGrad);
MIND_API_OPERATOR_IMPL(HShrinkGrad, BaseOperator);

void HShrink::set_lambd(const float &lambd)             { (void)this->AddAttr("lambd", api::MakeValue(lambd)); }

float HShrink::get_lambd() const             { return GetValue<float>(GetAttr("lambd")); }

REGISTER_PRIMITIVE_C(kNameHShrink, HShrink);
MIND_API_OPERATOR_IMPL(HShrink, BaseOperator);

REGISTER_PRIMITIVE_C(kNameHSigmoidGrad, HSigmoidGrad);
MIND_API_OPERATOR_IMPL(HSigmoidGrad, BaseOperator);

REGISTER_PRIMITIVE_C(kNameHSigmoid, HSigmoid);
MIND_API_OPERATOR_IMPL(HSigmoid, BaseOperator);

REGISTER_PRIMITIVE_C(kNameHSwishGrad, HSwishGrad);
MIND_API_OPERATOR_IMPL(HSwishGrad, BaseOperator);

REGISTER_PRIMITIVE_C(kNameHSwish, HSwish);
MIND_API_OPERATOR_IMPL(HSwish, BaseOperator);

REGISTER_PRIMITIVE_C(kNameIdentity, Identity);
MIND_API_OPERATOR_IMPL(Identity, BaseOperator);

REGISTER_PRIMITIVE_C(kNameIFFT2, IFFT2);
MIND_API_OPERATOR_IMPL(IFFT2, BaseOperator);

REGISTER_PRIMITIVE_C(kNameIFFT, IFFT);
MIND_API_OPERATOR_IMPL(IFFT, BaseOperator);

REGISTER_PRIMITIVE_C(kNameIFFTN, IFFTN);
MIND_API_OPERATOR_IMPL(IFFTN, BaseOperator);

REGISTER_PRIMITIVE_C(kNameIFFTShift, IFFTShift);
MIND_API_OPERATOR_IMPL(IFFTShift, BaseOperator);

REGISTER_PRIMITIVE_C(kNameIm2ColExt, Im2ColExt);
MIND_API_OPERATOR_IMPL(Im2ColExt, BaseOperator);

REGISTER_PRIMITIVE_C(kNameIndexAddExt, IndexAddExt);
MIND_API_OPERATOR_IMPL(IndexAddExt, BaseOperator);

REGISTER_PRIMITIVE_C(kNameIndexSelect, IndexSelect);
MIND_API_OPERATOR_IMPL(IndexSelect, BaseOperator);

REGISTER_PRIMITIVE_C(kNameIRFFTGrad, IRFFTGrad);
MIND_API_OPERATOR_IMPL(IRFFTGrad, BaseOperator);

REGISTER_PRIMITIVE_C(kNameIRFFT, IRFFT);
MIND_API_OPERATOR_IMPL(IRFFT, BaseOperator);

void IsClose::set_rtol(const float &rtol)             { (void)this->AddAttr("rtol", api::MakeValue(rtol)); }

float IsClose::get_rtol() const             { return GetValue<float>(GetAttr("rtol")); }

void IsClose::set_atol(const float &atol)             { (void)this->AddAttr("atol", api::MakeValue(atol)); }

float IsClose::get_atol() const             { return GetValue<float>(GetAttr("atol")); }

void IsClose::set_equal_nan(const bool &equal_nan)             { (void)this->AddAttr("equal_nan", api::MakeValue(equal_nan)); }

bool IsClose::get_equal_nan() const             { return GetValue<bool>(GetAttr("equal_nan")); }

REGISTER_PRIMITIVE_C(kNameIsClose, IsClose);
MIND_API_OPERATOR_IMPL(IsClose, BaseOperator);

REGISTER_PRIMITIVE_C(kNameIsFinite, IsFinite);
MIND_API_OPERATOR_IMPL(IsFinite, BaseOperator);

REGISTER_PRIMITIVE_C(kNameLayerNormExt, LayerNormExt);
MIND_API_OPERATOR_IMPL(LayerNormExt, BaseOperator);

REGISTER_PRIMITIVE_C(kNameLayerNormGradExt, LayerNormGradExt);
MIND_API_OPERATOR_IMPL(LayerNormGradExt, BaseOperator);

void LayerNormGradGrad::set_begin_norm_axis(const int64_t &begin_norm_axis)             { (void)this->AddAttr("begin_norm_axis", api::MakeValue(begin_norm_axis)); }

int64_t LayerNormGradGrad::get_begin_norm_axis() const             { return GetValue<int64_t>(GetAttr("begin_norm_axis")); }

void LayerNormGradGrad::set_begin_params_axis(const int64_t &begin_params_axis)             { (void)this->AddAttr("begin_params_axis", api::MakeValue(begin_params_axis)); }

int64_t LayerNormGradGrad::get_begin_params_axis() const             { return GetValue<int64_t>(GetAttr("begin_params_axis")); }

REGISTER_PRIMITIVE_C(kNameLayerNormGradGrad, LayerNormGradGrad);
MIND_API_OPERATOR_IMPL(LayerNormGradGrad, BaseOperator);

void LayerNormGrad::set_begin_norm_axis(const int64_t &begin_norm_axis)             { (void)this->AddAttr("begin_norm_axis", api::MakeValue(begin_norm_axis)); }

int64_t LayerNormGrad::get_begin_norm_axis() const             { return GetValue<int64_t>(GetAttr("begin_norm_axis")); }

void LayerNormGrad::set_begin_params_axis(const int64_t &begin_params_axis)             { (void)this->AddAttr("begin_params_axis", api::MakeValue(begin_params_axis)); }

int64_t LayerNormGrad::get_begin_params_axis() const             { return GetValue<int64_t>(GetAttr("begin_params_axis")); }

REGISTER_PRIMITIVE_C(kNameLayerNormGrad, LayerNormGrad);
MIND_API_OPERATOR_IMPL(LayerNormGrad, BaseOperator);

void LayerNormGradV3::set_begin_norm_axis(const int64_t &begin_norm_axis)             { (void)this->AddAttr("begin_norm_axis", api::MakeValue(begin_norm_axis)); }

int64_t LayerNormGradV3::get_begin_norm_axis() const             { return GetValue<int64_t>(GetAttr("begin_norm_axis")); }

void LayerNormGradV3::set_begin_params_axis(const int64_t &begin_params_axis)             { (void)this->AddAttr("begin_params_axis", api::MakeValue(begin_params_axis)); }

int64_t LayerNormGradV3::get_begin_params_axis() const             { return GetValue<int64_t>(GetAttr("begin_params_axis")); }

REGISTER_PRIMITIVE_C(kNameLayerNormGradV3, LayerNormGradV3);
MIND_API_OPERATOR_IMPL(LayerNormGradV3, BaseOperator);

void LayerNorm::set_begin_norm_axis(const int64_t &begin_norm_axis)             { (void)this->AddAttr("begin_norm_axis", api::MakeValue(begin_norm_axis)); }

int64_t LayerNorm::get_begin_norm_axis() const             { return GetValue<int64_t>(GetAttr("begin_norm_axis")); }

void LayerNorm::set_begin_params_axis(const int64_t &begin_params_axis)             { (void)this->AddAttr("begin_params_axis", api::MakeValue(begin_params_axis)); }

int64_t LayerNorm::get_begin_params_axis() const             { return GetValue<int64_t>(GetAttr("begin_params_axis")); }

void LayerNorm::set_epsilon(const float &epsilon)             { (void)this->AddAttr("epsilon", api::MakeValue(epsilon)); }

float LayerNorm::get_epsilon() const             { return GetValue<float>(GetAttr("epsilon")); }

REGISTER_PRIMITIVE_C(kNameLayerNorm, LayerNorm);
MIND_API_OPERATOR_IMPL(LayerNorm, BaseOperator);

void LayerNormV3::set_begin_norm_axis(const int64_t &begin_norm_axis)             { (void)this->AddAttr("begin_norm_axis", api::MakeValue(begin_norm_axis)); }

int64_t LayerNormV3::get_begin_norm_axis() const             { return GetValue<int64_t>(GetAttr("begin_norm_axis")); }

void LayerNormV3::set_begin_params_axis(const int64_t &begin_params_axis)             { (void)this->AddAttr("begin_params_axis", api::MakeValue(begin_params_axis)); }

int64_t LayerNormV3::get_begin_params_axis() const             { return GetValue<int64_t>(GetAttr("begin_params_axis")); }

void LayerNormV3::set_epsilon(const float &epsilon)             { (void)this->AddAttr("epsilon", api::MakeValue(epsilon)); }

float LayerNormV3::get_epsilon() const             { return GetValue<float>(GetAttr("epsilon")); }

REGISTER_PRIMITIVE_C(kNameLayerNormV3, LayerNormV3);
MIND_API_OPERATOR_IMPL(LayerNormV3, BaseOperator);

REGISTER_PRIMITIVE_C(kNameLeakyReLUExt, LeakyReLUExt);
MIND_API_OPERATOR_IMPL(LeakyReLUExt, BaseOperator);

REGISTER_PRIMITIVE_C(kNameLeakyReLUGradExt, LeakyReLUGradExt);
MIND_API_OPERATOR_IMPL(LeakyReLUGradExt, BaseOperator);

REGISTER_PRIMITIVE_C(kNameLessEqual, LessEqual);
MIND_API_OPERATOR_IMPL(LessEqual, BaseOperator);

REGISTER_PRIMITIVE_C(kNameLess, Less);
MIND_API_OPERATOR_IMPL(Less, BaseOperator);

REGISTER_PRIMITIVE_C(kNameLinSpaceExt, LinSpaceExt);
MIND_API_OPERATOR_IMPL(LinSpaceExt, BaseOperator);

REGISTER_PRIMITIVE_C(kNameLinSpace, LinSpace);
MIND_API_OPERATOR_IMPL(LinSpace, BaseOperator);

REGISTER_PRIMITIVE_C(kNameListToTuple, ListToTuple);
MIND_API_OPERATOR_IMPL(ListToTuple, BaseOperator);

REGISTER_PRIMITIVE_C(kNameLog1p, Log1p);
MIND_API_OPERATOR_IMPL(Log1p, BaseOperator);

REGISTER_PRIMITIVE_C(kNameLogMatrixDeterminant, LogMatrixDeterminant);
MIND_API_OPERATOR_IMPL(LogMatrixDeterminant, BaseOperator);

REGISTER_PRIMITIVE_C(kNameLog, Log);
MIND_API_OPERATOR_IMPL(Log, BaseOperator);

void LogSoftmaxGrad::set_axis(const int64_t &axis)             { (void)this->AddAttr("axis", api::MakeValue(axis)); }

int64_t LogSoftmaxGrad::get_axis() const             { return GetValue<int64_t>(GetAttr("axis")); }

REGISTER_PRIMITIVE_C(kNameLogSoftmaxGrad, LogSoftmaxGrad);
MIND_API_OPERATOR_IMPL(LogSoftmaxGrad, BaseOperator);

void LogSoftmax::set_axis(const int64_t &axis)             { (void)this->AddAttr("axis", api::MakeValue(axis)); }

int64_t LogSoftmax::get_axis() const             { return GetValue<int64_t>(GetAttr("axis")); }

REGISTER_PRIMITIVE_C(kNameLogSoftmax, LogSoftmax);
MIND_API_OPERATOR_IMPL(LogSoftmax, BaseOperator);

REGISTER_PRIMITIVE_C(kNameLogicalAnd, LogicalAnd);
MIND_API_OPERATOR_IMPL(LogicalAnd, BaseOperator);

REGISTER_PRIMITIVE_C(kNameLogicalNot, LogicalNot);
MIND_API_OPERATOR_IMPL(LogicalNot, BaseOperator);

REGISTER_PRIMITIVE_C(kNameLogicalOr, LogicalOr);
MIND_API_OPERATOR_IMPL(LogicalOr, BaseOperator);

REGISTER_PRIMITIVE_C(kNameLogicalXor, LogicalXor);
MIND_API_OPERATOR_IMPL(LogicalXor, BaseOperator);

void LogitGrad::set_eps(const float &eps)             { (void)this->AddAttr("eps", api::MakeValue(eps)); }

float LogitGrad::get_eps() const             { return GetValue<float>(GetAttr("eps")); }

REGISTER_PRIMITIVE_C(kNameLogitGrad, LogitGrad);
MIND_API_OPERATOR_IMPL(LogitGrad, BaseOperator);

void Logit::set_eps(const float &eps)             { (void)this->AddAttr("eps", api::MakeValue(eps)); }

float Logit::get_eps() const             { return GetValue<float>(GetAttr("eps")); }

REGISTER_PRIMITIVE_C(kNameLogit, Logit);
MIND_API_OPERATOR_IMPL(Logit, BaseOperator);

REGISTER_PRIMITIVE_C(kNameMaskedFill, MaskedFill);
MIND_API_OPERATOR_IMPL(MaskedFill, BaseOperator);

REGISTER_PRIMITIVE_C(kNameMatMulExt, MatMulExt);
MIND_API_OPERATOR_IMPL(MatMulExt, BaseOperator);

void MatMul::set_transpose_a(const bool &transpose_a)             { (void)this->AddAttr("transpose_a", api::MakeValue(transpose_a)); }

bool MatMul::get_transpose_a() const             { return GetValue<bool>(GetAttr("transpose_a")); }

void MatMul::set_transpose_b(const bool &transpose_b)             { (void)this->AddAttr("transpose_b", api::MakeValue(transpose_b)); }

bool MatMul::get_transpose_b() const             { return GetValue<bool>(GetAttr("transpose_b")); }

REGISTER_PRIMITIVE_C(kNameMatMul, MatMul);
MIND_API_OPERATOR_IMPL(MatMul, BaseOperator);

REGISTER_PRIMITIVE_C(kNameMatrixDeterminant, MatrixDeterminant);
MIND_API_OPERATOR_IMPL(MatrixDeterminant, BaseOperator);

REGISTER_PRIMITIVE_C(kNameMatrixExp, MatrixExp);
MIND_API_OPERATOR_IMPL(MatrixExp, BaseOperator);

REGISTER_PRIMITIVE_C(kNameMatrixInverseExt, MatrixInverseExt);
MIND_API_OPERATOR_IMPL(MatrixInverseExt, BaseOperator);

REGISTER_PRIMITIVE_C(kNameMax, Max);
MIND_API_OPERATOR_IMPL(Max, BaseOperator);

void MaxPoolGradWithIndices::set_kernel_size(const std::vector<int64_t> &kernel_size)             { (void)this->AddAttr("kernel_size", api::MakeValue(kernel_size)); }

std::vector<int64_t> MaxPoolGradWithIndices::get_kernel_size() const             { return GetValue<std::vector<int64_t>>(GetAttr("kernel_size")); }

void MaxPoolGradWithIndices::set_strides(const std::vector<int64_t> &strides)             { (void)this->AddAttr("strides", api::MakeValue(strides)); }

std::vector<int64_t> MaxPoolGradWithIndices::get_strides() const             { return GetValue<std::vector<int64_t>>(GetAttr("strides")); }

void MaxPoolGradWithIndices::set_pads(const std::vector<int64_t> &pads)             { (void)this->AddAttr("pads", api::MakeValue(pads)); }

std::vector<int64_t> MaxPoolGradWithIndices::get_pads() const             { return GetValue<std::vector<int64_t>>(GetAttr("pads")); }

void MaxPoolGradWithIndices::set_dilation(const std::vector<int64_t> &dilation)             { (void)this->AddAttr("dilation", api::MakeValue(dilation)); }

std::vector<int64_t> MaxPoolGradWithIndices::get_dilation() const             { return GetValue<std::vector<int64_t>>(GetAttr("dilation")); }

void MaxPoolGradWithIndices::set_ceil_mode(const bool &ceil_mode)             { (void)this->AddAttr("ceil_mode", api::MakeValue(ceil_mode)); }

bool MaxPoolGradWithIndices::get_ceil_mode() const             { return GetValue<bool>(GetAttr("ceil_mode")); }

void MaxPoolGradWithIndices::set_argmax_type(const int64_t &argmax_type)             { (void)this->AddAttr("argmax_type", api::MakeValue(argmax_type)); }

int64_t MaxPoolGradWithIndices::get_argmax_type() const             { return GetValue<int64_t>(GetAttr("argmax_type")); }

REGISTER_PRIMITIVE_C(kNameMaxPoolGradWithIndices, MaxPoolGradWithIndices);
MIND_API_OPERATOR_IMPL(MaxPoolGradWithIndices, BaseOperator);

void MaxPoolGradWithMask::set_kernel_size(const std::vector<int64_t> &kernel_size)             { (void)this->AddAttr("kernel_size", api::MakeValue(kernel_size)); }

std::vector<int64_t> MaxPoolGradWithMask::get_kernel_size() const             { return GetValue<std::vector<int64_t>>(GetAttr("kernel_size")); }

void MaxPoolGradWithMask::set_strides(const std::vector<int64_t> &strides)             { (void)this->AddAttr("strides", api::MakeValue(strides)); }

std::vector<int64_t> MaxPoolGradWithMask::get_strides() const             { return GetValue<std::vector<int64_t>>(GetAttr("strides")); }

void MaxPoolGradWithMask::set_pads(const std::vector<int64_t> &pads)             { (void)this->AddAttr("pads", api::MakeValue(pads)); }

std::vector<int64_t> MaxPoolGradWithMask::get_pads() const             { return GetValue<std::vector<int64_t>>(GetAttr("pads")); }

void MaxPoolGradWithMask::set_dilation(const std::vector<int64_t> &dilation)             { (void)this->AddAttr("dilation", api::MakeValue(dilation)); }

std::vector<int64_t> MaxPoolGradWithMask::get_dilation() const             { return GetValue<std::vector<int64_t>>(GetAttr("dilation")); }

void MaxPoolGradWithMask::set_ceil_mode(const bool &ceil_mode)             { (void)this->AddAttr("ceil_mode", api::MakeValue(ceil_mode)); }

bool MaxPoolGradWithMask::get_ceil_mode() const             { return GetValue<bool>(GetAttr("ceil_mode")); }

void MaxPoolGradWithMask::set_argmax_type(const int64_t &argmax_type)             { (void)this->AddAttr("argmax_type", api::MakeValue(argmax_type)); }

int64_t MaxPoolGradWithMask::get_argmax_type() const             { return GetValue<int64_t>(GetAttr("argmax_type")); }

REGISTER_PRIMITIVE_C(kNameMaxPoolGradWithMask, MaxPoolGradWithMask);
MIND_API_OPERATOR_IMPL(MaxPoolGradWithMask, BaseOperator);

void MaxPoolWithIndices::set_kernel_size(const std::vector<int64_t> &kernel_size)             { (void)this->AddAttr("kernel_size", api::MakeValue(kernel_size)); }

std::vector<int64_t> MaxPoolWithIndices::get_kernel_size() const             { return GetValue<std::vector<int64_t>>(GetAttr("kernel_size")); }

void MaxPoolWithIndices::set_strides(const std::vector<int64_t> &strides)             { (void)this->AddAttr("strides", api::MakeValue(strides)); }

std::vector<int64_t> MaxPoolWithIndices::get_strides() const             { return GetValue<std::vector<int64_t>>(GetAttr("strides")); }

void MaxPoolWithIndices::set_pads(const std::vector<int64_t> &pads)             { (void)this->AddAttr("pads", api::MakeValue(pads)); }

std::vector<int64_t> MaxPoolWithIndices::get_pads() const             { return GetValue<std::vector<int64_t>>(GetAttr("pads")); }

void MaxPoolWithIndices::set_dilation(const std::vector<int64_t> &dilation)             { (void)this->AddAttr("dilation", api::MakeValue(dilation)); }

std::vector<int64_t> MaxPoolWithIndices::get_dilation() const             { return GetValue<std::vector<int64_t>>(GetAttr("dilation")); }

void MaxPoolWithIndices::set_ceil_mode(const bool &ceil_mode)             { (void)this->AddAttr("ceil_mode", api::MakeValue(ceil_mode)); }

bool MaxPoolWithIndices::get_ceil_mode() const             { return GetValue<bool>(GetAttr("ceil_mode")); }

void MaxPoolWithIndices::set_argmax_type(const int64_t &argmax_type)             { (void)this->AddAttr("argmax_type", api::MakeValue(argmax_type)); }

int64_t MaxPoolWithIndices::get_argmax_type() const             { return GetValue<int64_t>(GetAttr("argmax_type")); }

REGISTER_PRIMITIVE_C(kNameMaxPoolWithIndices, MaxPoolWithIndices);
MIND_API_OPERATOR_IMPL(MaxPoolWithIndices, BaseOperator);

void MaxPoolWithMask::set_kernel_size(const std::vector<int64_t> &kernel_size)             { (void)this->AddAttr("kernel_size", api::MakeValue(kernel_size)); }

std::vector<int64_t> MaxPoolWithMask::get_kernel_size() const             { return GetValue<std::vector<int64_t>>(GetAttr("kernel_size")); }

void MaxPoolWithMask::set_strides(const std::vector<int64_t> &strides)             { (void)this->AddAttr("strides", api::MakeValue(strides)); }

std::vector<int64_t> MaxPoolWithMask::get_strides() const             { return GetValue<std::vector<int64_t>>(GetAttr("strides")); }

void MaxPoolWithMask::set_pads(const std::vector<int64_t> &pads)             { (void)this->AddAttr("pads", api::MakeValue(pads)); }

std::vector<int64_t> MaxPoolWithMask::get_pads() const             { return GetValue<std::vector<int64_t>>(GetAttr("pads")); }

void MaxPoolWithMask::set_dilation(const std::vector<int64_t> &dilation)             { (void)this->AddAttr("dilation", api::MakeValue(dilation)); }

std::vector<int64_t> MaxPoolWithMask::get_dilation() const             { return GetValue<std::vector<int64_t>>(GetAttr("dilation")); }

void MaxPoolWithMask::set_ceil_mode(const bool &ceil_mode)             { (void)this->AddAttr("ceil_mode", api::MakeValue(ceil_mode)); }

bool MaxPoolWithMask::get_ceil_mode() const             { return GetValue<bool>(GetAttr("ceil_mode")); }

void MaxPoolWithMask::set_argmax_type(const int64_t &argmax_type)             { (void)this->AddAttr("argmax_type", api::MakeValue(argmax_type)); }

int64_t MaxPoolWithMask::get_argmax_type() const             { return GetValue<int64_t>(GetAttr("argmax_type")); }

REGISTER_PRIMITIVE_C(kNameMaxPoolWithMask, MaxPoolWithMask);
MIND_API_OPERATOR_IMPL(MaxPoolWithMask, BaseOperator);

void MaximumGradGrad::set_grad_x(const bool &grad_x)             { (void)this->AddAttr("grad_x", api::MakeValue(grad_x)); }

bool MaximumGradGrad::get_grad_x() const             { return GetValue<bool>(GetAttr("grad_x")); }

void MaximumGradGrad::set_grad_y(const bool &grad_y)             { (void)this->AddAttr("grad_y", api::MakeValue(grad_y)); }

bool MaximumGradGrad::get_grad_y() const             { return GetValue<bool>(GetAttr("grad_y")); }

REGISTER_PRIMITIVE_C(kNameMaximumGradGrad, MaximumGradGrad);
MIND_API_OPERATOR_IMPL(MaximumGradGrad, BaseOperator);

void MaximumGrad::set_grad_x(const bool &grad_x)             { (void)this->AddAttr("grad_x", api::MakeValue(grad_x)); }

bool MaximumGrad::get_grad_x() const             { return GetValue<bool>(GetAttr("grad_x")); }

void MaximumGrad::set_grad_y(const bool &grad_y)             { (void)this->AddAttr("grad_y", api::MakeValue(grad_y)); }

bool MaximumGrad::get_grad_y() const             { return GetValue<bool>(GetAttr("grad_y")); }

REGISTER_PRIMITIVE_C(kNameMaximumGrad, MaximumGrad);
MIND_API_OPERATOR_IMPL(MaximumGrad, BaseOperator);

REGISTER_PRIMITIVE_C(kNameMaximum, Maximum);
MIND_API_OPERATOR_IMPL(Maximum, BaseOperator);

REGISTER_PRIMITIVE_C(kNameMeanExt, MeanExt);
MIND_API_OPERATOR_IMPL(MeanExt, BaseOperator);

REGISTER_PRIMITIVE_C(kNameMin, Min);
MIND_API_OPERATOR_IMPL(Min, BaseOperator);

void MinimumGrad::set_grad_x(const bool &grad_x)             { (void)this->AddAttr("grad_x", api::MakeValue(grad_x)); }

bool MinimumGrad::get_grad_x() const             { return GetValue<bool>(GetAttr("grad_x")); }

void MinimumGrad::set_grad_y(const bool &grad_y)             { (void)this->AddAttr("grad_y", api::MakeValue(grad_y)); }

bool MinimumGrad::get_grad_y() const             { return GetValue<bool>(GetAttr("grad_y")); }

REGISTER_PRIMITIVE_C(kNameMinimumGrad, MinimumGrad);
MIND_API_OPERATOR_IMPL(MinimumGrad, BaseOperator);

REGISTER_PRIMITIVE_C(kNameMinimum, Minimum);
MIND_API_OPERATOR_IMPL(Minimum, BaseOperator);

REGISTER_PRIMITIVE_C(kNameMul, Mul);
MIND_API_OPERATOR_IMPL(Mul, BaseOperator);

REGISTER_PRIMITIVE_C(kNameMv, Mv);
MIND_API_OPERATOR_IMPL(Mv, BaseOperator);

void NanToNum::set_nan(const float &nan)             { (void)this->AddAttr("nan", api::MakeValue(nan)); }

float NanToNum::get_nan() const             { return GetValue<float>(GetAttr("nan")); }

void NanToNum::set_posinf(const float &posinf)             { (void)this->AddAttr("posinf", api::MakeValue(posinf)); }

float NanToNum::get_posinf() const             { return GetValue<float>(GetAttr("posinf")); }

void NanToNum::set_neginf(const float &neginf)             { (void)this->AddAttr("neginf", api::MakeValue(neginf)); }

float NanToNum::get_neginf() const             { return GetValue<float>(GetAttr("neginf")); }

REGISTER_PRIMITIVE_C(kNameNanToNum, NanToNum);
MIND_API_OPERATOR_IMPL(NanToNum, BaseOperator);

REGISTER_PRIMITIVE_C(kNameNeg, Neg);
MIND_API_OPERATOR_IMPL(Neg, BaseOperator);

REGISTER_PRIMITIVE_C(kNameNextAfter, NextAfter);
MIND_API_OPERATOR_IMPL(NextAfter, BaseOperator);

void NLLLossGrad::set_reduction(const int64_t &reduction)             { (void)this->AddAttr("reduction", api::MakeValue(reduction)); }

int64_t NLLLossGrad::get_reduction() const             { return GetValue<int64_t>(GetAttr("reduction")); }

void NLLLossGrad::set_ignore_index(const int64_t &ignore_index)             { (void)this->AddAttr("ignore_index", api::MakeValue(ignore_index)); }

int64_t NLLLossGrad::get_ignore_index() const             { return GetValue<int64_t>(GetAttr("ignore_index")); }

REGISTER_PRIMITIVE_C(kNameNLLLossGrad, NLLLossGrad);
MIND_API_OPERATOR_IMPL(NLLLossGrad, BaseOperator);

void NLLLoss::set_reduction(const int64_t &reduction)             { (void)this->AddAttr("reduction", api::MakeValue(reduction)); }

int64_t NLLLoss::get_reduction() const             { return GetValue<int64_t>(GetAttr("reduction")); }

void NLLLoss::set_ignore_index(const int64_t &ignore_index)             { (void)this->AddAttr("ignore_index", api::MakeValue(ignore_index)); }

int64_t NLLLoss::get_ignore_index() const             { return GetValue<int64_t>(GetAttr("ignore_index")); }

REGISTER_PRIMITIVE_C(kNameNLLLoss, NLLLoss);
MIND_API_OPERATOR_IMPL(NLLLoss, BaseOperator);

REGISTER_PRIMITIVE_C(kNameNonZeroExt, NonZeroExt);
MIND_API_OPERATOR_IMPL(NonZeroExt, BaseOperator);

REGISTER_PRIMITIVE_C(kNameNonZero, NonZero);
MIND_API_OPERATOR_IMPL(NonZero, BaseOperator);

REGISTER_PRIMITIVE_C(kNameNorm, Norm);
MIND_API_OPERATOR_IMPL(Norm, BaseOperator);

REGISTER_PRIMITIVE_C(kNameNormalFloatFloat, NormalFloatFloat);
MIND_API_OPERATOR_IMPL(NormalFloatFloat, BaseOperator);

REGISTER_PRIMITIVE_C(kNameNormalFloatTensor, NormalFloatTensor);
MIND_API_OPERATOR_IMPL(NormalFloatTensor, BaseOperator);

REGISTER_PRIMITIVE_C(kNameNormalTensorFloat, NormalTensorFloat);
MIND_API_OPERATOR_IMPL(NormalTensorFloat, BaseOperator);

REGISTER_PRIMITIVE_C(kNameNormalTensorTensor, NormalTensorTensor);
MIND_API_OPERATOR_IMPL(NormalTensorTensor, BaseOperator);

REGISTER_PRIMITIVE_C(kNameNotEqual, NotEqual);
MIND_API_OPERATOR_IMPL(NotEqual, BaseOperator);

REGISTER_PRIMITIVE_C(kNameNPUClearFloatStatusV2, NPUClearFloatStatusV2);
MIND_API_OPERATOR_IMPL(NPUClearFloatStatusV2, BaseOperator);

REGISTER_PRIMITIVE_C(kNameNPUGetFloatStatusV2, NPUGetFloatStatusV2);
MIND_API_OPERATOR_IMPL(NPUGetFloatStatusV2, BaseOperator);

void OneHotExt::set_axis(const int64_t &axis)             { (void)this->AddAttr("axis", api::MakeValue(axis)); }

int64_t OneHotExt::get_axis() const             { return GetValue<int64_t>(GetAttr("axis")); }

REGISTER_PRIMITIVE_C(kNameOneHotExt, OneHotExt);
MIND_API_OPERATOR_IMPL(OneHotExt, BaseOperator);

void OneHot::set_axis(const int64_t &axis)             { (void)this->AddAttr("axis", api::MakeValue(axis)); }

int64_t OneHot::get_axis() const             { return GetValue<int64_t>(GetAttr("axis")); }

REGISTER_PRIMITIVE_C(kNameOneHot, OneHot);
MIND_API_OPERATOR_IMPL(OneHot, BaseOperator);

REGISTER_PRIMITIVE_C(kNameOnesLikeExt, OnesLikeExt);
MIND_API_OPERATOR_IMPL(OnesLikeExt, BaseOperator);

REGISTER_PRIMITIVE_C(kNameOnesLike, OnesLike);
MIND_API_OPERATOR_IMPL(OnesLike, BaseOperator);

REGISTER_PRIMITIVE_C(kNameOnes, Ones);
MIND_API_OPERATOR_IMPL(Ones, BaseOperator);

void PagedAttentionMask::set_head_num(const int64_t &head_num)             { (void)this->AddAttr("head_num", api::MakeValue(head_num)); }

int64_t PagedAttentionMask::get_head_num() const             { return GetValue<int64_t>(GetAttr("head_num")); }

void PagedAttentionMask::set_scale_value(const float &scale_value)             { (void)this->AddAttr("scale_value", api::MakeValue(scale_value)); }

float PagedAttentionMask::get_scale_value() const             { return GetValue<float>(GetAttr("scale_value")); }

void PagedAttentionMask::set_kv_head_num(const int64_t &kv_head_num)             { (void)this->AddAttr("kv_head_num", api::MakeValue(kv_head_num)); }

int64_t PagedAttentionMask::get_kv_head_num() const             { return GetValue<int64_t>(GetAttr("kv_head_num")); }

REGISTER_PRIMITIVE_C(kNamePagedAttentionMask, PagedAttentionMask);
MIND_API_OPERATOR_IMPL(PagedAttentionMask, BaseOperator);

void PagedAttention::set_head_num(const int64_t &head_num)             { (void)this->AddAttr("head_num", api::MakeValue(head_num)); }

int64_t PagedAttention::get_head_num() const             { return GetValue<int64_t>(GetAttr("head_num")); }

void PagedAttention::set_scale_value(const float &scale_value)             { (void)this->AddAttr("scale_value", api::MakeValue(scale_value)); }

float PagedAttention::get_scale_value() const             { return GetValue<float>(GetAttr("scale_value")); }

void PagedAttention::set_kv_head_num(const int64_t &kv_head_num)             { (void)this->AddAttr("kv_head_num", api::MakeValue(kv_head_num)); }

int64_t PagedAttention::get_kv_head_num() const             { return GetValue<int64_t>(GetAttr("kv_head_num")); }

REGISTER_PRIMITIVE_C(kNamePagedAttention, PagedAttention);
MIND_API_OPERATOR_IMPL(PagedAttention, BaseOperator);

REGISTER_PRIMITIVE_C(kNamePow, Pow);
MIND_API_OPERATOR_IMPL(Pow, BaseOperator);

REGISTER_PRIMITIVE_C(kNamePReLUGrad, PReLUGrad);
MIND_API_OPERATOR_IMPL(PReLUGrad, BaseOperator);

REGISTER_PRIMITIVE_C(kNamePReLU, PReLU);
MIND_API_OPERATOR_IMPL(PReLU, BaseOperator);

REGISTER_PRIMITIVE_C(kNameProdExt, ProdExt);
MIND_API_OPERATOR_IMPL(ProdExt, BaseOperator);

void PromptKVCache::set_align_mode(const int64_t &align_mode)             { (void)this->AddAttr("align_mode", api::MakeValue(align_mode)); }

int64_t PromptKVCache::get_align_mode() const             { return GetValue<int64_t>(GetAttr("align_mode")); }

REGISTER_PRIMITIVE_C(kNamePromptKVCache, PromptKVCache);
MIND_API_OPERATOR_IMPL(PromptKVCache, BaseOperator);

void Qr::set_full_matrices(const bool &full_matrices)             { (void)this->AddAttr("full_matrices", api::MakeValue(full_matrices)); }

bool Qr::get_full_matrices() const             { return GetValue<bool>(GetAttr("full_matrices")); }

REGISTER_PRIMITIVE_C(kNameQr, Qr);
MIND_API_OPERATOR_IMPL(Qr, BaseOperator);

REGISTER_PRIMITIVE_C(kNameRandExt, RandExt);
MIND_API_OPERATOR_IMPL(RandExt, BaseOperator);

REGISTER_PRIMITIVE_C(kNameRandLikeExt, RandLikeExt);
MIND_API_OPERATOR_IMPL(RandLikeExt, BaseOperator);

void RandpermV2::set_seed(const int64_t &seed)             { (void)this->AddAttr("seed", api::MakeValue(seed)); }

int64_t RandpermV2::get_seed() const             { return GetValue<int64_t>(GetAttr("seed")); }

void RandpermV2::set_offset(const int64_t &offset)             { (void)this->AddAttr("offset", api::MakeValue(offset)); }

int64_t RandpermV2::get_offset() const             { return GetValue<int64_t>(GetAttr("offset")); }

void RandpermV2::set_dtype(const int64_t &dtype)             { (void)this->AddAttr("dtype", api::MakeValue(dtype)); }

int64_t RandpermV2::get_dtype() const             { return GetValue<int64_t>(GetAttr("dtype")); }

REGISTER_PRIMITIVE_C(kNameRandpermV2, RandpermV2);
MIND_API_OPERATOR_IMPL(RandpermV2, BaseOperator);

void Range::set_maxlen(const int64_t &maxlen)             { (void)this->AddAttr("maxlen", api::MakeValue(maxlen)); }

int64_t Range::get_maxlen() const             { return GetValue<int64_t>(GetAttr("maxlen")); }

REGISTER_PRIMITIVE_C(kNameRange, Range);
MIND_API_OPERATOR_IMPL(Range, BaseOperator);

REGISTER_PRIMITIVE_C(kNameRank, Rank);
MIND_API_OPERATOR_IMPL(Rank, BaseOperator);

REGISTER_PRIMITIVE_C(kNameRealDiv, RealDiv);
MIND_API_OPERATOR_IMPL(RealDiv, BaseOperator);

REGISTER_PRIMITIVE_C(kNameReal, Real);
MIND_API_OPERATOR_IMPL(Real, BaseOperator);

REGISTER_PRIMITIVE_C(kNameReciprocalGrad, ReciprocalGrad);
MIND_API_OPERATOR_IMPL(ReciprocalGrad, BaseOperator);

REGISTER_PRIMITIVE_C(kNameReciprocal, Reciprocal);
MIND_API_OPERATOR_IMPL(Reciprocal, BaseOperator);

void ReduceAll::set_keep_dims(const bool &keep_dims)             { (void)this->AddAttr("keep_dims", api::MakeValue(keep_dims)); }

bool ReduceAll::get_keep_dims() const             { return GetValue<bool>(GetAttr("keep_dims")); }

REGISTER_PRIMITIVE_C(kNameReduceAll, ReduceAll);
MIND_API_OPERATOR_IMPL(ReduceAll, BaseOperator);

void ReduceAny::set_keep_dims(const bool &keep_dims)             { (void)this->AddAttr("keep_dims", api::MakeValue(keep_dims)); }

bool ReduceAny::get_keep_dims() const             { return GetValue<bool>(GetAttr("keep_dims")); }

REGISTER_PRIMITIVE_C(kNameReduceAny, ReduceAny);
MIND_API_OPERATOR_IMPL(ReduceAny, BaseOperator);

void ReduceMax::set_keep_dims(const bool &keep_dims)             { (void)this->AddAttr("keep_dims", api::MakeValue(keep_dims)); }

bool ReduceMax::get_keep_dims() const             { return GetValue<bool>(GetAttr("keep_dims")); }

REGISTER_PRIMITIVE_C(kNameReduceMax, ReduceMax);
MIND_API_OPERATOR_IMPL(ReduceMax, BaseOperator);

void ReduceMean::set_keep_dims(const bool &keep_dims)             { (void)this->AddAttr("keep_dims", api::MakeValue(keep_dims)); }

bool ReduceMean::get_keep_dims() const             { return GetValue<bool>(GetAttr("keep_dims")); }

REGISTER_PRIMITIVE_C(kNameReduceMean, ReduceMean);
MIND_API_OPERATOR_IMPL(ReduceMean, BaseOperator);

void ReduceMin::set_keep_dims(const bool &keep_dims)             { (void)this->AddAttr("keep_dims", api::MakeValue(keep_dims)); }

bool ReduceMin::get_keep_dims() const             { return GetValue<bool>(GetAttr("keep_dims")); }

REGISTER_PRIMITIVE_C(kNameReduceMin, ReduceMin);
MIND_API_OPERATOR_IMPL(ReduceMin, BaseOperator);

void ReduceProd::set_keep_dims(const bool &keep_dims)             { (void)this->AddAttr("keep_dims", api::MakeValue(keep_dims)); }

bool ReduceProd::get_keep_dims() const             { return GetValue<bool>(GetAttr("keep_dims")); }

REGISTER_PRIMITIVE_C(kNameReduceProd, ReduceProd);
MIND_API_OPERATOR_IMPL(ReduceProd, BaseOperator);

void ReduceStd::set_axis(const std::vector<int64_t> &axis)             { (void)this->AddAttr("axis", api::MakeValue(axis)); }

std::vector<int64_t> ReduceStd::get_axis() const             { return GetValue<std::vector<int64_t>>(GetAttr("axis")); }

void ReduceStd::set_unbiased(const bool &unbiased)             { (void)this->AddAttr("unbiased", api::MakeValue(unbiased)); }

bool ReduceStd::get_unbiased() const             { return GetValue<bool>(GetAttr("unbiased")); }

void ReduceStd::set_keep_dims(const bool &keep_dims)             { (void)this->AddAttr("keep_dims", api::MakeValue(keep_dims)); }

bool ReduceStd::get_keep_dims() const             { return GetValue<bool>(GetAttr("keep_dims")); }

REGISTER_PRIMITIVE_C(kNameReduceStd, ReduceStd);
MIND_API_OPERATOR_IMPL(ReduceStd, BaseOperator);

void ReduceSum::set_keep_dims(const bool &keep_dims)             { (void)this->AddAttr("keep_dims", api::MakeValue(keep_dims)); }

bool ReduceSum::get_keep_dims() const             { return GetValue<bool>(GetAttr("keep_dims")); }

void ReduceSum::set_skip_mode(const bool &skip_mode)             { (void)this->AddAttr("skip_mode", api::MakeValue(skip_mode)); }

bool ReduceSum::get_skip_mode() const             { return GetValue<bool>(GetAttr("skip_mode")); }

REGISTER_PRIMITIVE_C(kNameReduceSum, ReduceSum);
MIND_API_OPERATOR_IMPL(ReduceSum, BaseOperator);

REGISTER_PRIMITIVE_C(kNameReflectionPad1DGrad, ReflectionPad1DGrad);
MIND_API_OPERATOR_IMPL(ReflectionPad1DGrad, BaseOperator);

REGISTER_PRIMITIVE_C(kNameReflectionPad1D, ReflectionPad1D);
MIND_API_OPERATOR_IMPL(ReflectionPad1D, BaseOperator);

REGISTER_PRIMITIVE_C(kNameReflectionPad2DGrad, ReflectionPad2DGrad);
MIND_API_OPERATOR_IMPL(ReflectionPad2DGrad, BaseOperator);

REGISTER_PRIMITIVE_C(kNameReflectionPad2D, ReflectionPad2D);
MIND_API_OPERATOR_IMPL(ReflectionPad2D, BaseOperator);

REGISTER_PRIMITIVE_C(kNameReflectionPad3DGrad, ReflectionPad3DGrad);
MIND_API_OPERATOR_IMPL(ReflectionPad3DGrad, BaseOperator);

REGISTER_PRIMITIVE_C(kNameReflectionPad3D, ReflectionPad3D);
MIND_API_OPERATOR_IMPL(ReflectionPad3D, BaseOperator);

REGISTER_PRIMITIVE_C(kNameReLU6Grad, ReLU6Grad);
MIND_API_OPERATOR_IMPL(ReLU6Grad, BaseOperator);

REGISTER_PRIMITIVE_C(kNameReLU6, ReLU6);
MIND_API_OPERATOR_IMPL(ReLU6, BaseOperator);

REGISTER_PRIMITIVE_C(kNameReluGrad, ReluGrad);
MIND_API_OPERATOR_IMPL(ReluGrad, BaseOperator);

REGISTER_PRIMITIVE_C(kNameReLU, ReLU);
MIND_API_OPERATOR_IMPL(ReLU, BaseOperator);

REGISTER_PRIMITIVE_C(kNameRepeatInterleaveGrad, RepeatInterleaveGrad);
MIND_API_OPERATOR_IMPL(RepeatInterleaveGrad, BaseOperator);

REGISTER_PRIMITIVE_C(kNameRepeatInterleaveInt, RepeatInterleaveInt);
MIND_API_OPERATOR_IMPL(RepeatInterleaveInt, BaseOperator);

REGISTER_PRIMITIVE_C(kNameRepeatInterleaveTensor, RepeatInterleaveTensor);
MIND_API_OPERATOR_IMPL(RepeatInterleaveTensor, BaseOperator);

REGISTER_PRIMITIVE_C(kNameReplicationPad1DGrad, ReplicationPad1DGrad);
MIND_API_OPERATOR_IMPL(ReplicationPad1DGrad, BaseOperator);

REGISTER_PRIMITIVE_C(kNameReplicationPad1D, ReplicationPad1D);
MIND_API_OPERATOR_IMPL(ReplicationPad1D, BaseOperator);

REGISTER_PRIMITIVE_C(kNameReplicationPad2DGrad, ReplicationPad2DGrad);
MIND_API_OPERATOR_IMPL(ReplicationPad2DGrad, BaseOperator);

REGISTER_PRIMITIVE_C(kNameReplicationPad2D, ReplicationPad2D);
MIND_API_OPERATOR_IMPL(ReplicationPad2D, BaseOperator);

REGISTER_PRIMITIVE_C(kNameReplicationPad3DGrad, ReplicationPad3DGrad);
MIND_API_OPERATOR_IMPL(ReplicationPad3DGrad, BaseOperator);

REGISTER_PRIMITIVE_C(kNameReplicationPad3D, ReplicationPad3D);
MIND_API_OPERATOR_IMPL(ReplicationPad3D, BaseOperator);

REGISTER_PRIMITIVE_C(kNameReshapeAndCache, ReshapeAndCache);
MIND_API_OPERATOR_IMPL(ReshapeAndCache, BaseOperator);

REGISTER_PRIMITIVE_C(kNameReshape, Reshape);
MIND_API_OPERATOR_IMPL(Reshape, BaseOperator);

void ResizeBicubicGrad::set_align_corners(const bool &align_corners)             { (void)this->AddAttr("align_corners", api::MakeValue(align_corners)); }

bool ResizeBicubicGrad::get_align_corners() const             { return GetValue<bool>(GetAttr("align_corners")); }

void ResizeBicubicGrad::set_half_pixel_centers(const bool &half_pixel_centers)             { (void)this->AddAttr("half_pixel_centers", api::MakeValue(half_pixel_centers)); }

bool ResizeBicubicGrad::get_half_pixel_centers() const             { return GetValue<bool>(GetAttr("half_pixel_centers")); }

REGISTER_PRIMITIVE_C(kNameResizeBicubicGrad, ResizeBicubicGrad);
MIND_API_OPERATOR_IMPL(ResizeBicubicGrad, BaseOperator);

void ResizeBicubic::set_align_corners(const bool &align_corners)             { (void)this->AddAttr("align_corners", api::MakeValue(align_corners)); }

bool ResizeBicubic::get_align_corners() const             { return GetValue<bool>(GetAttr("align_corners")); }

void ResizeBicubic::set_half_pixel_centers(const bool &half_pixel_centers)             { (void)this->AddAttr("half_pixel_centers", api::MakeValue(half_pixel_centers)); }

bool ResizeBicubic::get_half_pixel_centers() const             { return GetValue<bool>(GetAttr("half_pixel_centers")); }

REGISTER_PRIMITIVE_C(kNameResizeBicubic, ResizeBicubic);
MIND_API_OPERATOR_IMPL(ResizeBicubic, BaseOperator);

void ResizeBilinearGrad::set_align_corners(const bool &align_corners)             { (void)this->AddAttr("align_corners", api::MakeValue(align_corners)); }

bool ResizeBilinearGrad::get_align_corners() const             { return GetValue<bool>(GetAttr("align_corners")); }

void ResizeBilinearGrad::set_half_pixel_centers(const bool &half_pixel_centers)             { (void)this->AddAttr("half_pixel_centers", api::MakeValue(half_pixel_centers)); }

bool ResizeBilinearGrad::get_half_pixel_centers() const             { return GetValue<bool>(GetAttr("half_pixel_centers")); }

REGISTER_PRIMITIVE_C(kNameResizeBilinearGrad, ResizeBilinearGrad);
MIND_API_OPERATOR_IMPL(ResizeBilinearGrad, BaseOperator);

void ResizeBilinearV2::set_align_corners(const bool &align_corners)             { (void)this->AddAttr("align_corners", api::MakeValue(align_corners)); }

bool ResizeBilinearV2::get_align_corners() const             { return GetValue<bool>(GetAttr("align_corners")); }

void ResizeBilinearV2::set_half_pixel_centers(const bool &half_pixel_centers)             { (void)this->AddAttr("half_pixel_centers", api::MakeValue(half_pixel_centers)); }

bool ResizeBilinearV2::get_half_pixel_centers() const             { return GetValue<bool>(GetAttr("half_pixel_centers")); }

REGISTER_PRIMITIVE_C(kNameResizeBilinearV2, ResizeBilinearV2);
MIND_API_OPERATOR_IMPL(ResizeBilinearV2, BaseOperator);

REGISTER_PRIMITIVE_C(kNameResizeD, ResizeD);
MIND_API_OPERATOR_IMPL(ResizeD, BaseOperator);

void ResizeLinear1DGrad::set_coordinate_transformation_mode(const int64_t &coordinate_transformation_mode)             { (void)this->AddAttr("coordinate_transformation_mode", api::MakeValue(coordinate_transformation_mode)); }

int64_t ResizeLinear1DGrad::get_coordinate_transformation_mode() const             { return GetValue<int64_t>(GetAttr("coordinate_transformation_mode")); }

REGISTER_PRIMITIVE_C(kNameResizeLinear1DGrad, ResizeLinear1DGrad);
MIND_API_OPERATOR_IMPL(ResizeLinear1DGrad, BaseOperator);

void ResizeLinear1D::set_coordinate_transformation_mode(const int64_t &coordinate_transformation_mode)             { (void)this->AddAttr("coordinate_transformation_mode", api::MakeValue(coordinate_transformation_mode)); }

int64_t ResizeLinear1D::get_coordinate_transformation_mode() const             { return GetValue<int64_t>(GetAttr("coordinate_transformation_mode")); }

REGISTER_PRIMITIVE_C(kNameResizeLinear1D, ResizeLinear1D);
MIND_API_OPERATOR_IMPL(ResizeLinear1D, BaseOperator);

void ResizeNearestNeighborGrad::set_align_corners(const bool &align_corners)             { (void)this->AddAttr("align_corners", api::MakeValue(align_corners)); }

bool ResizeNearestNeighborGrad::get_align_corners() const             { return GetValue<bool>(GetAttr("align_corners")); }

void ResizeNearestNeighborGrad::set_half_pixel_centers(const bool &half_pixel_centers)             { (void)this->AddAttr("half_pixel_centers", api::MakeValue(half_pixel_centers)); }

bool ResizeNearestNeighborGrad::get_half_pixel_centers() const             { return GetValue<bool>(GetAttr("half_pixel_centers")); }

REGISTER_PRIMITIVE_C(kNameResizeNearestNeighborGrad, ResizeNearestNeighborGrad);
MIND_API_OPERATOR_IMPL(ResizeNearestNeighborGrad, BaseOperator);

void ResizeNearestNeighbor::set_size(const std::vector<int64_t> &size)             { (void)this->AddAttr("size", api::MakeValue(size)); }

std::vector<int64_t> ResizeNearestNeighbor::get_size() const             { return GetValue<std::vector<int64_t>>(GetAttr("size")); }

void ResizeNearestNeighbor::set_align_corners(const bool &align_corners)             { (void)this->AddAttr("align_corners", api::MakeValue(align_corners)); }

bool ResizeNearestNeighbor::get_align_corners() const             { return GetValue<bool>(GetAttr("align_corners")); }

void ResizeNearestNeighbor::set_half_pixel_centers(const bool &half_pixel_centers)             { (void)this->AddAttr("half_pixel_centers", api::MakeValue(half_pixel_centers)); }

bool ResizeNearestNeighbor::get_half_pixel_centers() const             { return GetValue<bool>(GetAttr("half_pixel_centers")); }

REGISTER_PRIMITIVE_C(kNameResizeNearestNeighbor, ResizeNearestNeighbor);
MIND_API_OPERATOR_IMPL(ResizeNearestNeighbor, BaseOperator);

void ResizeNearestNeighborV2Grad::set_align_corners(const bool &align_corners)             { (void)this->AddAttr("align_corners", api::MakeValue(align_corners)); }

bool ResizeNearestNeighborV2Grad::get_align_corners() const             { return GetValue<bool>(GetAttr("align_corners")); }

void ResizeNearestNeighborV2Grad::set_half_pixel_centers(const bool &half_pixel_centers)             { (void)this->AddAttr("half_pixel_centers", api::MakeValue(half_pixel_centers)); }

bool ResizeNearestNeighborV2Grad::get_half_pixel_centers() const             { return GetValue<bool>(GetAttr("half_pixel_centers")); }

REGISTER_PRIMITIVE_C(kNameResizeNearestNeighborV2Grad, ResizeNearestNeighborV2Grad);
MIND_API_OPERATOR_IMPL(ResizeNearestNeighborV2Grad, BaseOperator);

void ResizeNearestNeighborV2::set_align_corners(const bool &align_corners)             { (void)this->AddAttr("align_corners", api::MakeValue(align_corners)); }

bool ResizeNearestNeighborV2::get_align_corners() const             { return GetValue<bool>(GetAttr("align_corners")); }

void ResizeNearestNeighborV2::set_half_pixel_centers(const bool &half_pixel_centers)             { (void)this->AddAttr("half_pixel_centers", api::MakeValue(half_pixel_centers)); }

bool ResizeNearestNeighborV2::get_half_pixel_centers() const             { return GetValue<bool>(GetAttr("half_pixel_centers")); }

REGISTER_PRIMITIVE_C(kNameResizeNearestNeighborV2, ResizeNearestNeighborV2);
MIND_API_OPERATOR_IMPL(ResizeNearestNeighborV2, BaseOperator);

void ReverseV2::set_axis(const std::vector<int64_t> &axis)             { (void)this->AddAttr("axis", api::MakeValue(axis)); }

std::vector<int64_t> ReverseV2::get_axis() const             { return GetValue<std::vector<int64_t>>(GetAttr("axis")); }

REGISTER_PRIMITIVE_C(kNameReverseV2, ReverseV2);
MIND_API_OPERATOR_IMPL(ReverseV2, BaseOperator);

REGISTER_PRIMITIVE_C(kNameRFFTGrad, RFFTGrad);
MIND_API_OPERATOR_IMPL(RFFTGrad, BaseOperator);

REGISTER_PRIMITIVE_C(kNameRFFT, RFFT);
MIND_API_OPERATOR_IMPL(RFFT, BaseOperator);

REGISTER_PRIMITIVE_C(kNameRightShift, RightShift);
MIND_API_OPERATOR_IMPL(RightShift, BaseOperator);

REGISTER_PRIMITIVE_C(kNameRmsNormGrad, RmsNormGrad);
MIND_API_OPERATOR_IMPL(RmsNormGrad, BaseOperator);

void RmsNorm::set_epsilon(const float &epsilon)             { (void)this->AddAttr("epsilon", api::MakeValue(epsilon)); }

float RmsNorm::get_epsilon() const             { return GetValue<float>(GetAttr("epsilon")); }

REGISTER_PRIMITIVE_C(kNameRmsNorm, RmsNorm);
MIND_API_OPERATOR_IMPL(RmsNorm, BaseOperator);

void Roll::set_shift(const std::vector<int64_t> &shift)             { (void)this->AddAttr("shift", api::MakeValue(shift)); }

std::vector<int64_t> Roll::get_shift() const             { return GetValue<std::vector<int64_t>>(GetAttr("shift")); }

void Roll::set_axis(const std::vector<int64_t> &axis)             { (void)this->AddAttr("axis", api::MakeValue(axis)); }

std::vector<int64_t> Roll::get_axis() const             { return GetValue<std::vector<int64_t>>(GetAttr("axis")); }

REGISTER_PRIMITIVE_C(kNameRoll, Roll);
MIND_API_OPERATOR_IMPL(Roll, BaseOperator);

REGISTER_PRIMITIVE_C(kNameRound, Round);
MIND_API_OPERATOR_IMPL(Round, BaseOperator);

REGISTER_PRIMITIVE_C(kNameRsqrtGrad, RsqrtGrad);
MIND_API_OPERATOR_IMPL(RsqrtGrad, BaseOperator);

REGISTER_PRIMITIVE_C(kNameRsqrt, Rsqrt);
MIND_API_OPERATOR_IMPL(Rsqrt, BaseOperator);

REGISTER_PRIMITIVE_C(kNameScalarAdd, ScalarAdd);
MIND_API_OPERATOR_IMPL(ScalarAdd, BaseOperator);

REGISTER_PRIMITIVE_C(kNameScalarBool, ScalarBool);
MIND_API_OPERATOR_IMPL(ScalarBool, BaseOperator);

REGISTER_PRIMITIVE_C(kNameScalarCast, ScalarCast);
MIND_API_OPERATOR_IMPL(ScalarCast, BaseOperator);

REGISTER_PRIMITIVE_C(kNameScalarDiv, ScalarDiv);
MIND_API_OPERATOR_IMPL(ScalarDiv, BaseOperator);

REGISTER_PRIMITIVE_C(kNameScalarEq, ScalarEq);
MIND_API_OPERATOR_IMPL(ScalarEq, BaseOperator);

REGISTER_PRIMITIVE_C(kNameScalarFloorDiv, ScalarFloorDiv);
MIND_API_OPERATOR_IMPL(ScalarFloorDiv, BaseOperator);

REGISTER_PRIMITIVE_C(kNameScalarGe, ScalarGe);
MIND_API_OPERATOR_IMPL(ScalarGe, BaseOperator);

REGISTER_PRIMITIVE_C(kNameScalarGt, ScalarGt);
MIND_API_OPERATOR_IMPL(ScalarGt, BaseOperator);

REGISTER_PRIMITIVE_C(kNameScalarLe, ScalarLe);
MIND_API_OPERATOR_IMPL(ScalarLe, BaseOperator);

REGISTER_PRIMITIVE_C(kNameScalarLog, ScalarLog);
MIND_API_OPERATOR_IMPL(ScalarLog, BaseOperator);

REGISTER_PRIMITIVE_C(kNameScalarLt, ScalarLt);
MIND_API_OPERATOR_IMPL(ScalarLt, BaseOperator);

REGISTER_PRIMITIVE_C(kNameScalarMod, ScalarMod);
MIND_API_OPERATOR_IMPL(ScalarMod, BaseOperator);

REGISTER_PRIMITIVE_C(kNameScalarMul, ScalarMul);
MIND_API_OPERATOR_IMPL(ScalarMul, BaseOperator);

REGISTER_PRIMITIVE_C(kNameScalarPow, ScalarPow);
MIND_API_OPERATOR_IMPL(ScalarPow, BaseOperator);

REGISTER_PRIMITIVE_C(kNameScalarSub, ScalarSub);
MIND_API_OPERATOR_IMPL(ScalarSub, BaseOperator);

REGISTER_PRIMITIVE_C(kNameScalarToTensor, ScalarToTensor);
MIND_API_OPERATOR_IMPL(ScalarToTensor, BaseOperator);

REGISTER_PRIMITIVE_C(kNameScalarUadd, ScalarUadd);
MIND_API_OPERATOR_IMPL(ScalarUadd, BaseOperator);

REGISTER_PRIMITIVE_C(kNameScalarUsub, ScalarUsub);
MIND_API_OPERATOR_IMPL(ScalarUsub, BaseOperator);

REGISTER_PRIMITIVE_C(kNameScatterAddExt, ScatterAddExt);
MIND_API_OPERATOR_IMPL(ScatterAddExt, BaseOperator);

REGISTER_PRIMITIVE_C(kNameScatterNd, ScatterNd);
MIND_API_OPERATOR_IMPL(ScatterNd, BaseOperator);

REGISTER_PRIMITIVE_C(kNameScatter, Scatter);
MIND_API_OPERATOR_IMPL(Scatter, BaseOperator);

void SearchSorted::set_dtype(const int64_t &dtype)             { (void)this->AddAttr("dtype", api::MakeValue(dtype)); }

int64_t SearchSorted::get_dtype() const             { return GetValue<int64_t>(GetAttr("dtype")); }

void SearchSorted::set_right(const bool &right)             { (void)this->AddAttr("right", api::MakeValue(right)); }

bool SearchSorted::get_right() const             { return GetValue<bool>(GetAttr("right")); }

REGISTER_PRIMITIVE_C(kNameSearchSorted, SearchSorted);
MIND_API_OPERATOR_IMPL(SearchSorted, BaseOperator);

REGISTER_PRIMITIVE_C(kNameSelect, Select);
MIND_API_OPERATOR_IMPL(Select, BaseOperator);

void SequenceConcat::set_axis(const int64_t &axis)             { (void)this->AddAttr("axis", api::MakeValue(axis)); }

int64_t SequenceConcat::get_axis() const             { return GetValue<int64_t>(GetAttr("axis")); }

REGISTER_PRIMITIVE_C(kNameSequenceConcat, SequenceConcat);
MIND_API_OPERATOR_IMPL(SequenceConcat, BaseOperator);

REGISTER_PRIMITIVE_C(kNameShape, Shape);
MIND_API_OPERATOR_IMPL(Shape, BaseOperator);

REGISTER_PRIMITIVE_C(kNameSigmoidGrad, SigmoidGrad);
MIND_API_OPERATOR_IMPL(SigmoidGrad, BaseOperator);

REGISTER_PRIMITIVE_C(kNameSigmoid, Sigmoid);
MIND_API_OPERATOR_IMPL(Sigmoid, BaseOperator);

REGISTER_PRIMITIVE_C(kNameSign, Sign);
MIND_API_OPERATOR_IMPL(Sign, BaseOperator);

REGISTER_PRIMITIVE_C(kNameSiLUGrad, SiLUGrad);
MIND_API_OPERATOR_IMPL(SiLUGrad, BaseOperator);

REGISTER_PRIMITIVE_C(kNameSiLU, SiLU);
MIND_API_OPERATOR_IMPL(SiLU, BaseOperator);

REGISTER_PRIMITIVE_C(kNameSin, Sin);
MIND_API_OPERATOR_IMPL(Sin, BaseOperator);

REGISTER_PRIMITIVE_C(kNameSinc, Sinc);
MIND_API_OPERATOR_IMPL(Sinc, BaseOperator);

REGISTER_PRIMITIVE_C(kNameSinh, Sinh);
MIND_API_OPERATOR_IMPL(Sinh, BaseOperator);

REGISTER_PRIMITIVE_C(kNameSliceExt, SliceExt);
MIND_API_OPERATOR_IMPL(SliceExt, BaseOperator);

REGISTER_PRIMITIVE_C(kNameSoftmaxBackward, SoftmaxBackward);
MIND_API_OPERATOR_IMPL(SoftmaxBackward, BaseOperator);

void Softmax::set_axis(const std::vector<int64_t> &axis)             { (void)this->AddAttr("axis", api::MakeValue(axis)); }

std::vector<int64_t> Softmax::get_axis() const             { return GetValue<std::vector<int64_t>>(GetAttr("axis")); }

REGISTER_PRIMITIVE_C(kNameSoftmax, Softmax);
MIND_API_OPERATOR_IMPL(Softmax, BaseOperator);

REGISTER_PRIMITIVE_C(kNameSoftplusExt, SoftplusExt);
MIND_API_OPERATOR_IMPL(SoftplusExt, BaseOperator);

REGISTER_PRIMITIVE_C(kNameSoftplusGradExt, SoftplusGradExt);
MIND_API_OPERATOR_IMPL(SoftplusGradExt, BaseOperator);

REGISTER_PRIMITIVE_C(kNameSolveTriangular, SolveTriangular);
MIND_API_OPERATOR_IMPL(SolveTriangular, BaseOperator);

REGISTER_PRIMITIVE_C(kNameSortExt, SortExt);
MIND_API_OPERATOR_IMPL(SortExt, BaseOperator);

void Split::set_axis(const int64_t &axis)             { (void)this->AddAttr("axis", api::MakeValue(axis)); }

int64_t Split::get_axis() const             { return GetValue<int64_t>(GetAttr("axis")); }

void Split::set_output_num(const int64_t &output_num)             { (void)this->AddAttr("output_num", api::MakeValue(output_num)); }

int64_t Split::get_output_num() const             { return GetValue<int64_t>(GetAttr("output_num")); }

REGISTER_PRIMITIVE_C(kNameSplit, Split);
MIND_API_OPERATOR_IMPL(Split, BaseOperator);

REGISTER_PRIMITIVE_C(kNameSplitTensor, SplitTensor);
MIND_API_OPERATOR_IMPL(SplitTensor, BaseOperator);

REGISTER_PRIMITIVE_C(kNameSplitWithSize, SplitWithSize);
MIND_API_OPERATOR_IMPL(SplitWithSize, BaseOperator);

REGISTER_PRIMITIVE_C(kNameSqrtGrad, SqrtGrad);
MIND_API_OPERATOR_IMPL(SqrtGrad, BaseOperator);

REGISTER_PRIMITIVE_C(kNameSqrt, Sqrt);
MIND_API_OPERATOR_IMPL(Sqrt, BaseOperator);

REGISTER_PRIMITIVE_C(kNameSquare, Square);
MIND_API_OPERATOR_IMPL(Square, BaseOperator);

void StackExt::set_dim(const int64_t &dim)             { (void)this->AddAttr("dim", api::MakeValue(dim)); }

int64_t StackExt::get_dim() const             { return GetValue<int64_t>(GetAttr("dim")); }

REGISTER_PRIMITIVE_C(kNameStackExt, StackExt);
MIND_API_OPERATOR_IMPL(StackExt, BaseOperator);

void StridedSlice::set_begin_mask(const int64_t &begin_mask)             { (void)this->AddAttr("begin_mask", api::MakeValue(begin_mask)); }

int64_t StridedSlice::get_begin_mask() const             { return GetValue<int64_t>(GetAttr("begin_mask")); }

void StridedSlice::set_end_mask(const int64_t &end_mask)             { (void)this->AddAttr("end_mask", api::MakeValue(end_mask)); }

int64_t StridedSlice::get_end_mask() const             { return GetValue<int64_t>(GetAttr("end_mask")); }

void StridedSlice::set_ellipsis_mask(const int64_t &ellipsis_mask)             { (void)this->AddAttr("ellipsis_mask", api::MakeValue(ellipsis_mask)); }

int64_t StridedSlice::get_ellipsis_mask() const             { return GetValue<int64_t>(GetAttr("ellipsis_mask")); }

void StridedSlice::set_new_axis_mask(const int64_t &new_axis_mask)             { (void)this->AddAttr("new_axis_mask", api::MakeValue(new_axis_mask)); }

int64_t StridedSlice::get_new_axis_mask() const             { return GetValue<int64_t>(GetAttr("new_axis_mask")); }

void StridedSlice::set_shrink_axis_mask(const int64_t &shrink_axis_mask)             { (void)this->AddAttr("shrink_axis_mask", api::MakeValue(shrink_axis_mask)); }

int64_t StridedSlice::get_shrink_axis_mask() const             { return GetValue<int64_t>(GetAttr("shrink_axis_mask")); }

REGISTER_PRIMITIVE_C(kNameStridedSlice, StridedSlice);
MIND_API_OPERATOR_IMPL(StridedSlice, BaseOperator);

REGISTER_PRIMITIVE_C(kNameSubExt, SubExt);
MIND_API_OPERATOR_IMPL(SubExt, BaseOperator);

REGISTER_PRIMITIVE_C(kNameSub, Sub);
MIND_API_OPERATOR_IMPL(Sub, BaseOperator);

REGISTER_PRIMITIVE_C(kNameSumExt, SumExt);
MIND_API_OPERATOR_IMPL(SumExt, BaseOperator);

REGISTER_PRIMITIVE_C(kNameTanhGrad, TanhGrad);
MIND_API_OPERATOR_IMPL(TanhGrad, BaseOperator);

REGISTER_PRIMITIVE_C(kNameTanh, Tanh);
MIND_API_OPERATOR_IMPL(Tanh, BaseOperator);

REGISTER_PRIMITIVE_C(kNameTensorCopySlices, TensorCopySlices);
MIND_API_OPERATOR_IMPL(TensorCopySlices, BaseOperator);

REGISTER_PRIMITIVE_C(kNameTensorShape, TensorShape);
MIND_API_OPERATOR_IMPL(TensorShape, BaseOperator);

REGISTER_PRIMITIVE_C(kNameTile, Tile);
MIND_API_OPERATOR_IMPL(Tile, BaseOperator);

REGISTER_PRIMITIVE_C(kNameTopkExt, TopkExt);
MIND_API_OPERATOR_IMPL(TopkExt, BaseOperator);

REGISTER_PRIMITIVE_C(kNameTopKRouter, TopKRouter);
MIND_API_OPERATOR_IMPL(TopKRouter, BaseOperator);

REGISTER_PRIMITIVE_C(kNameTrace, Trace);
MIND_API_OPERATOR_IMPL(Trace, BaseOperator);

REGISTER_PRIMITIVE_C(kNameTranspose, Transpose);
MIND_API_OPERATOR_IMPL(Transpose, BaseOperator);

void Triu::set_diagonal(const int64_t &diagonal)             { (void)this->AddAttr("diagonal", api::MakeValue(diagonal)); }

int64_t Triu::get_diagonal() const             { return GetValue<int64_t>(GetAttr("diagonal")); }

REGISTER_PRIMITIVE_C(kNameTriu, Triu);
MIND_API_OPERATOR_IMPL(Triu, BaseOperator);

REGISTER_PRIMITIVE_C(kNameTupleToList, TupleToList);
MIND_API_OPERATOR_IMPL(TupleToList, BaseOperator);

REGISTER_PRIMITIVE_C(kNameTupleToTensor, TupleToTensor);
MIND_API_OPERATOR_IMPL(TupleToTensor, BaseOperator);

REGISTER_PRIMITIVE_C(kNameUniformExt, UniformExt);
MIND_API_OPERATOR_IMPL(UniformExt, BaseOperator);

REGISTER_PRIMITIVE_C(kNameUnique2, Unique2);
MIND_API_OPERATOR_IMPL(Unique2, BaseOperator);

REGISTER_PRIMITIVE_C(kNameUniqueDim, UniqueDim);
MIND_API_OPERATOR_IMPL(UniqueDim, BaseOperator);

REGISTER_PRIMITIVE_C(kNameUnsortedSegmentSum, UnsortedSegmentSum);
MIND_API_OPERATOR_IMPL(UnsortedSegmentSum, BaseOperator);

void UnstackExt::set_axis(const int64_t &axis)             { (void)this->AddAttr("axis", api::MakeValue(axis)); }

int64_t UnstackExt::get_axis() const             { return GetValue<int64_t>(GetAttr("axis")); }

REGISTER_PRIMITIVE_C(kNameUnstackExt, UnstackExt);
MIND_API_OPERATOR_IMPL(UnstackExt, BaseOperator);

REGISTER_PRIMITIVE_C(kNameUpsampleBilinear2DGrad, UpsampleBilinear2DGrad);
MIND_API_OPERATOR_IMPL(UpsampleBilinear2DGrad, BaseOperator);

REGISTER_PRIMITIVE_C(kNameUpsampleBilinear2D, UpsampleBilinear2D);
MIND_API_OPERATOR_IMPL(UpsampleBilinear2D, BaseOperator);

REGISTER_PRIMITIVE_C(kNameUpsampleLinear1DGrad, UpsampleLinear1DGrad);
MIND_API_OPERATOR_IMPL(UpsampleLinear1DGrad, BaseOperator);

REGISTER_PRIMITIVE_C(kNameUpsampleLinear1D, UpsampleLinear1D);
MIND_API_OPERATOR_IMPL(UpsampleLinear1D, BaseOperator);

REGISTER_PRIMITIVE_C(kNameUpsampleNearest1DGrad, UpsampleNearest1DGrad);
MIND_API_OPERATOR_IMPL(UpsampleNearest1DGrad, BaseOperator);

REGISTER_PRIMITIVE_C(kNameUpsampleNearest1D, UpsampleNearest1D);
MIND_API_OPERATOR_IMPL(UpsampleNearest1D, BaseOperator);

REGISTER_PRIMITIVE_C(kNameUpsampleNearest2DGrad, UpsampleNearest2DGrad);
MIND_API_OPERATOR_IMPL(UpsampleNearest2DGrad, BaseOperator);

REGISTER_PRIMITIVE_C(kNameUpsampleNearest2D, UpsampleNearest2D);
MIND_API_OPERATOR_IMPL(UpsampleNearest2D, BaseOperator);

REGISTER_PRIMITIVE_C(kNameUpsampleNearest3DGrad, UpsampleNearest3DGrad);
MIND_API_OPERATOR_IMPL(UpsampleNearest3DGrad, BaseOperator);

REGISTER_PRIMITIVE_C(kNameUpsampleNearest3D, UpsampleNearest3D);
MIND_API_OPERATOR_IMPL(UpsampleNearest3D, BaseOperator);

void UpsampleTrilinear3DGrad::set_align_corners(const bool &align_corners)             { (void)this->AddAttr("align_corners", api::MakeValue(align_corners)); }

bool UpsampleTrilinear3DGrad::get_align_corners() const             { return GetValue<bool>(GetAttr("align_corners")); }

REGISTER_PRIMITIVE_C(kNameUpsampleTrilinear3DGrad, UpsampleTrilinear3DGrad);
MIND_API_OPERATOR_IMPL(UpsampleTrilinear3DGrad, BaseOperator);

void UpsampleTrilinear3D::set_align_corners(const bool &align_corners)             { (void)this->AddAttr("align_corners", api::MakeValue(align_corners)); }

bool UpsampleTrilinear3D::get_align_corners() const             { return GetValue<bool>(GetAttr("align_corners")); }

REGISTER_PRIMITIVE_C(kNameUpsampleTrilinear3D, UpsampleTrilinear3D);
MIND_API_OPERATOR_IMPL(UpsampleTrilinear3D, BaseOperator);

REGISTER_PRIMITIVE_C(kNameView, View);
MIND_API_OPERATOR_IMPL(View, BaseOperator);

REGISTER_PRIMITIVE_C(kNameZerosLikeExt, ZerosLikeExt);
MIND_API_OPERATOR_IMPL(ZerosLikeExt, BaseOperator);

REGISTER_PRIMITIVE_C(kNameZerosLike, ZerosLike);
MIND_API_OPERATOR_IMPL(ZerosLike, BaseOperator);

REGISTER_PRIMITIVE_C(kNameZeros, Zeros);
MIND_API_OPERATOR_IMPL(Zeros, BaseOperator);

REGISTER_PRIMITIVE_C(kNameDynamicQuantExt, DynamicQuantExt);
MIND_API_OPERATOR_IMPL(DynamicQuantExt, BaseOperator);

void FusedInferAttentionScore::set_num_heads(const int64_t &num_heads)             { (void)this->AddAttr("num_heads", api::MakeValue(num_heads)); }

int64_t FusedInferAttentionScore::get_num_heads() const             { return GetValue<int64_t>(GetAttr("num_heads")); }

void FusedInferAttentionScore::set_scale_value(const float &scale_value)             { (void)this->AddAttr("scale_value", api::MakeValue(scale_value)); }

float FusedInferAttentionScore::get_scale_value() const             { return GetValue<float>(GetAttr("scale_value")); }

void FusedInferAttentionScore::set_pre_tokens(const int64_t &pre_tokens)             { (void)this->AddAttr("pre_tokens", api::MakeValue(pre_tokens)); }

int64_t FusedInferAttentionScore::get_pre_tokens() const             { return GetValue<int64_t>(GetAttr("pre_tokens")); }

void FusedInferAttentionScore::set_next_tokens(const int64_t &next_tokens)             { (void)this->AddAttr("next_tokens", api::MakeValue(next_tokens)); }

int64_t FusedInferAttentionScore::get_next_tokens() const             { return GetValue<int64_t>(GetAttr("next_tokens")); }

void FusedInferAttentionScore::set_input_layout(const int64_t &input_layout)             { (void)this->AddAttr("input_layout", api::MakeValue(input_layout)); }

int64_t FusedInferAttentionScore::get_input_layout() const             { return GetValue<int64_t>(GetAttr("input_layout")); }

void FusedInferAttentionScore::set_num_key_value_heads(const int64_t &num_key_value_heads)             { (void)this->AddAttr("num_key_value_heads", api::MakeValue(num_key_value_heads)); }

int64_t FusedInferAttentionScore::get_num_key_value_heads() const             { return GetValue<int64_t>(GetAttr("num_key_value_heads")); }

void FusedInferAttentionScore::set_sparse_mode(const int64_t &sparse_mode)             { (void)this->AddAttr("sparse_mode", api::MakeValue(sparse_mode)); }

int64_t FusedInferAttentionScore::get_sparse_mode() const             { return GetValue<int64_t>(GetAttr("sparse_mode")); }

void FusedInferAttentionScore::set_inner_precise(const int64_t &inner_precise)             { (void)this->AddAttr("inner_precise", api::MakeValue(inner_precise)); }

int64_t FusedInferAttentionScore::get_inner_precise() const             { return GetValue<int64_t>(GetAttr("inner_precise")); }

void FusedInferAttentionScore::set_block_size(const int64_t &block_size)             { (void)this->AddAttr("block_size", api::MakeValue(block_size)); }

int64_t FusedInferAttentionScore::get_block_size() const             { return GetValue<int64_t>(GetAttr("block_size")); }

void FusedInferAttentionScore::set_antiquant_mode(const int64_t &antiquant_mode)             { (void)this->AddAttr("antiquant_mode", api::MakeValue(antiquant_mode)); }

int64_t FusedInferAttentionScore::get_antiquant_mode() const             { return GetValue<int64_t>(GetAttr("antiquant_mode")); }

void FusedInferAttentionScore::set_softmax_lse_flag(const bool &softmax_lse_flag)             { (void)this->AddAttr("softmax_lse_flag", api::MakeValue(softmax_lse_flag)); }

bool FusedInferAttentionScore::get_softmax_lse_flag() const             { return GetValue<bool>(GetAttr("softmax_lse_flag")); }

REGISTER_PRIMITIVE_C(kNameFusedInferAttentionScore, FusedInferAttentionScore);
MIND_API_OPERATOR_IMPL(FusedInferAttentionScore, BaseOperator);

void GroupedMatmul::set_split_item(const int64_t &split_item)             { (void)this->AddAttr("split_item", api::MakeValue(split_item)); }

int64_t GroupedMatmul::get_split_item() const             { return GetValue<int64_t>(GetAttr("split_item")); }

void GroupedMatmul::set_group_type(const int64_t &group_type)             { (void)this->AddAttr("group_type", api::MakeValue(group_type)); }

int64_t GroupedMatmul::get_group_type() const             { return GetValue<int64_t>(GetAttr("group_type")); }

REGISTER_PRIMITIVE_C(kNameGroupedMatmul, GroupedMatmul);
MIND_API_OPERATOR_IMPL(GroupedMatmul, BaseOperator);

REGISTER_PRIMITIVE_C(kNameKVCacheScatterUpdate, KVCacheScatterUpdate);
MIND_API_OPERATOR_IMPL(KVCacheScatterUpdate, BaseOperator);

REGISTER_PRIMITIVE_C(kNameMatmulBiasSplitOut2, MatmulBiasSplitOut2);
MIND_API_OPERATOR_IMPL(MatmulBiasSplitOut2, BaseOperator);

REGISTER_PRIMITIVE_C(kNameMatmulBiasSplitOut3, MatmulBiasSplitOut3);
MIND_API_OPERATOR_IMPL(MatmulBiasSplitOut3, BaseOperator);

REGISTER_PRIMITIVE_C(kNameMatmulBiasSplitSiluOut2, MatmulBiasSplitSiluOut2);
MIND_API_OPERATOR_IMPL(MatmulBiasSplitSiluOut2, BaseOperator);

REGISTER_PRIMITIVE_C(kNameMatmulSplitOut2, MatmulSplitOut2);
MIND_API_OPERATOR_IMPL(MatmulSplitOut2, BaseOperator);

REGISTER_PRIMITIVE_C(kNameMatmulSplitOut3, MatmulSplitOut3);
MIND_API_OPERATOR_IMPL(MatmulSplitOut3, BaseOperator);

REGISTER_PRIMITIVE_C(kNameMatmulSplitSiluOut2, MatmulSplitSiluOut2);
MIND_API_OPERATOR_IMPL(MatmulSplitSiluOut2, BaseOperator);

REGISTER_PRIMITIVE_C(kNameMoeFinalizeRouting, MoeFinalizeRouting);
MIND_API_OPERATOR_IMPL(MoeFinalizeRouting, BaseOperator);

void QuantBatchMatmul::set_transpose_x1(const bool &transpose_x1)             { (void)this->AddAttr("transpose_x1", api::MakeValue(transpose_x1)); }

bool QuantBatchMatmul::get_transpose_x1() const             { return GetValue<bool>(GetAttr("transpose_x1")); }

void QuantBatchMatmul::set_transpose_x2(const bool &transpose_x2)             { (void)this->AddAttr("transpose_x2", api::MakeValue(transpose_x2)); }

bool QuantBatchMatmul::get_transpose_x2() const             { return GetValue<bool>(GetAttr("transpose_x2")); }

void QuantBatchMatmul::set_dtype(const int64_t &dtype)             { (void)this->AddAttr("dtype", api::MakeValue(dtype)); }

int64_t QuantBatchMatmul::get_dtype() const             { return GetValue<int64_t>(GetAttr("dtype")); }

REGISTER_PRIMITIVE_C(kNameQuantBatchMatmul, QuantBatchMatmul);
MIND_API_OPERATOR_IMPL(QuantBatchMatmul, BaseOperator);

REGISTER_PRIMITIVE_C(kNameQuantV2, QuantV2);
MIND_API_OPERATOR_IMPL(QuantV2, BaseOperator);

REGISTER_PRIMITIVE_C(kNameQuantbatchmatmulSplitOut2, QuantbatchmatmulSplitOut2);
MIND_API_OPERATOR_IMPL(QuantbatchmatmulSplitOut2, BaseOperator);

REGISTER_PRIMITIVE_C(kNameQuantbatchmatmulSplitOut3, QuantbatchmatmulSplitOut3);
MIND_API_OPERATOR_IMPL(QuantbatchmatmulSplitOut3, BaseOperator);

REGISTER_PRIMITIVE_C(kNameQuantbatchmatmulSplitSiluOut2, QuantbatchmatmulSplitSiluOut2);
MIND_API_OPERATOR_IMPL(QuantbatchmatmulSplitSiluOut2, BaseOperator);

void WeightQuantBatchMatmul::set_transpose_x(const bool &transpose_x)             { (void)this->AddAttr("transpose_x", api::MakeValue(transpose_x)); }

bool WeightQuantBatchMatmul::get_transpose_x() const             { return GetValue<bool>(GetAttr("transpose_x")); }

void WeightQuantBatchMatmul::set_transpose_weight(const bool &transpose_weight)             { (void)this->AddAttr("transpose_weight", api::MakeValue(transpose_weight)); }

bool WeightQuantBatchMatmul::get_transpose_weight() const             { return GetValue<bool>(GetAttr("transpose_weight")); }

void WeightQuantBatchMatmul::set_antiquant_group_size(const int64_t &antiquant_group_size)             { (void)this->AddAttr("antiquant_group_size", api::MakeValue(antiquant_group_size)); }

int64_t WeightQuantBatchMatmul::get_antiquant_group_size() const             { return GetValue<int64_t>(GetAttr("antiquant_group_size")); }

REGISTER_PRIMITIVE_C(kNameWeightQuantBatchMatmul, WeightQuantBatchMatmul);
MIND_API_OPERATOR_IMPL(WeightQuantBatchMatmul, BaseOperator);

}  // namespace mindspore::ops
    