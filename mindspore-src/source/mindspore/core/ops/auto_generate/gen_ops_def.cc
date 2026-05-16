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

#include "ops/auto_generate/gen_ops_def.h"
#include "mindspore/core/ir/signature.h"
#include "ops/ops_func_impl/a_cos_grad.h"
#include "ops/ops_func_impl/abs_grad.h"
#include "ops/ops_func_impl/abs.h"
#include "ops/ops_func_impl/acos.h"
#include "ops/ops_func_impl/acosh_grad.h"
#include "ops/ops_func_impl/acosh.h"
#include "ops/ops_func_impl/adam_weight_decay.h"
#include "ops/ops_func_impl/adamw.h"
#include "ops/ops_func_impl/add_ext.h"
#include "ops/ops_func_impl/add_layernorm_v2.h"
#include "ops/ops_func_impl/add.h"
#include "ops/ops_func_impl/addcdiv.h"
#include "ops/ops_func_impl/addcmul.h"
#include "ops/ops_func_impl/addmm.h"
#include "ops/ops_func_impl/addn.h"
#include "ops/ops_func_impl/angle.h"
#include "ops/ops_func_impl/apply_came_part1.h"
#include "ops/ops_func_impl/apply_came_part2.h"
#include "ops/ops_func_impl/apply_came_part3.h"
#include "ops/ops_func_impl/apply_came_part4.h"
#include "ops/ops_func_impl/apply_rotary_pos_emb.h"
#include "ops/ops_func_impl/arange.h"
#include "ops/ops_func_impl/argmax_ext.h"
#include "ops/ops_func_impl/argmax.h"
#include "ops/ops_func_impl/argmax_with_value.h"
#include "ops/ops_func_impl/argmin.h"
#include "ops/ops_func_impl/argmin_with_value.h"
#include "ops/ops_func_impl/asin_grad.h"
#include "ops/ops_func_impl/asin.h"
#include "ops/ops_func_impl/asinh_grad.h"
#include "ops/ops_func_impl/asinh.h"
#include "ops/ops_func_impl/assign_add.h"
#include "ops/ops_func_impl/assign.h"
#include "ops/ops_func_impl/atan2_ext.h"
#include "ops/ops_func_impl/atan2.h"
#include "ops/ops_func_impl/atan_grad.h"
#include "ops/ops_func_impl/atan.h"
#include "ops/ops_func_impl/atanh.h"
#include "ops/ops_func_impl/avg_pool2d_grad.h"
#include "ops/ops_func_impl/avg_pool2d.h"
#include "ops/ops_func_impl/avg_pool_grad.h"
#include "ops/ops_func_impl/avg_pool.h"
#include "ops/ops_func_impl/batch_mat_mul.h"
#include "ops/ops_func_impl/batch_norm_ext.h"
#include "ops/ops_func_impl/batch_norm_grad_ext.h"
#include "ops/ops_func_impl/batch_norm_grad_grad.h"
#include "ops/ops_func_impl/batch_norm_grad.h"
#include "ops/ops_func_impl/batch_norm_grad_with_activation.h"
#include "ops/ops_func_impl/batch_norm.h"
#include "ops/ops_func_impl/batch_norm_with_activation.h"
#include "ops/ops_func_impl/batch_norm_with_add_and_activation.h"
#include "ops/ops_func_impl/betainc.h"
#include "ops/ops_func_impl/bias_add_grad.h"
#include "ops/ops_func_impl/bias_add.h"
#include "ops/ops_func_impl/binary_cross_entropy_grad.h"
#include "ops/ops_func_impl/binary_cross_entropy.h"
#include "ops/ops_func_impl/binary_cross_entropy_with_logits_backward.h"
#include "ops/ops_func_impl/binary_cross_entropy_with_logits.h"
#include "ops/ops_func_impl/bmm_ext.h"
#include "ops/ops_func_impl/bool_not.h"
#include "ops/ops_func_impl/broadcast_to.h"
#include "ops/ops_func_impl/cast.h"
#include "ops/ops_func_impl/ceil.h"
#include "ops/ops_func_impl/celu.h"
#include "ops/ops_func_impl/cholesky_grad.h"
#include "ops/ops_func_impl/cholesky_inverse.h"
#include "ops/ops_func_impl/cholesky.h"
#include "ops/ops_func_impl/chunk.h"
#include "ops/ops_func_impl/clamp_scalar.h"
#include "ops/ops_func_impl/clamp_tensor.h"
#include "ops/ops_func_impl/col2im_ext.h"
#include "ops/ops_func_impl/col2im_grad.h"
#include "ops/ops_func_impl/complex.h"
#include "ops/ops_func_impl/concat.h"
#include "ops/ops_func_impl/conj.h"
#include "ops/ops_func_impl/constant_pad_nd.h"
#include "ops/ops_func_impl/contiguous.h"
#include "ops/ops_func_impl/convolution_grad.h"
#include "ops/ops_func_impl/convolution.h"
#include "ops/ops_func_impl/copy.h"
#include "ops/ops_func_impl/correlate.h"
#include "ops/ops_func_impl/cos.h"
#include "ops/ops_func_impl/cosh.h"
#include "ops/ops_func_impl/cum_prod.h"
#include "ops/ops_func_impl/cum_sum.h"
#include "ops/ops_func_impl/cummax.h"
#include "ops/ops_func_impl/cummin.h"
#include "ops/ops_func_impl/cumsum_ext.h"
#include "ops/ops_func_impl/dct.h"
#include "ops/ops_func_impl/decoder_k_v_cache.h"
#include "ops/ops_func_impl/dense.h"
#include "ops/ops_func_impl/diag.h"
#include "ops/ops_func_impl/diagonal.h"
#include "ops/ops_func_impl/div.h"
#include "ops/ops_func_impl/divmod.h"
#include "ops/ops_func_impl/dot.h"
#include "ops/ops_func_impl/dropout_do_mask_ext.h"
#include "ops/ops_func_impl/dropout_ext.h"
#include "ops/ops_func_impl/dropout_gen_mask_ext.h"
#include "ops/ops_func_impl/dropout_grad_ext.h"
#include "ops/ops_func_impl/dropout.h"
#include "ops/ops_func_impl/eig.h"
#include "ops/ops_func_impl/elu_ext.h"
#include "ops/ops_func_impl/elu_grad_ext.h"
#include "ops/ops_func_impl/elu_grad.h"
#include "ops/ops_func_impl/elu.h"
#include "ops/ops_func_impl/embedding_dense_backward.h"
#include "ops/ops_func_impl/embedding.h"
#include "ops/ops_func_impl/equal.h"
#include "ops/ops_func_impl/erf.h"
#include "ops/ops_func_impl/erfc.h"
#include "ops/ops_func_impl/erfinv.h"
#include "ops/ops_func_impl/exp.h"
#include "ops/ops_func_impl/expand_dims.h"
#include "ops/ops_func_impl/expm1.h"
#include "ops/ops_func_impl/extract_image_patches.h"
#include "ops/ops_func_impl/eye.h"
#include "ops/ops_func_impl/fast_gelu_grad.h"
#include "ops/ops_func_impl/fast_gelu.h"
#include "ops/ops_func_impl/ffn_ext.h"
#include "ops/ops_func_impl/fft2.h"
#include "ops/ops_func_impl/fft.h"
#include "ops/ops_func_impl/fft_shapecopy.h"
#include "ops/ops_func_impl/fft_with_size.h"
#include "ops/ops_func_impl/fftn.h"
#include "ops/ops_func_impl/fftshift.h"
#include "ops/ops_func_impl/fill_scalar.h"
#include "ops/ops_func_impl/fill_tensor.h"
#include "ops/ops_func_impl/flash_attention_score_grad.h"
#include "ops/ops_func_impl/flash_attention_score.h"
#include "ops/ops_func_impl/flatten_ext.h"
#include "ops/ops_func_impl/flatten.h"
#include "ops/ops_func_impl/floor_div.h"
#include "ops/ops_func_impl/floor_mod.h"
#include "ops/ops_func_impl/floor.h"
#include "ops/ops_func_impl/gather_d_grad_v2.h"
#include "ops/ops_func_impl/gather_d.h"
#include "ops/ops_func_impl/gather_nd.h"
#include "ops/ops_func_impl/gather.h"
#include "ops/ops_func_impl/gcd.h"
#include "ops/ops_func_impl/gelu_grad.h"
#include "ops/ops_func_impl/gelu.h"
#include "ops/ops_func_impl/generator.h"
#include "ops/ops_func_impl/geqrf.h"
#include "ops/ops_func_impl/greater_equal.h"
#include "ops/ops_func_impl/greater.h"
#include "ops/ops_func_impl/grid_sampler_2d_grad.h"
#include "ops/ops_func_impl/grid_sampler_2d.h"
#include "ops/ops_func_impl/grid_sampler_3d_grad.h"
#include "ops/ops_func_impl/grid_sampler_3d.h"
#include "ops/ops_func_impl/group_norm_grad.h"
#include "ops/ops_func_impl/group_norm.h"
#include "ops/ops_func_impl/hshrink_grad.h"
#include "ops/ops_func_impl/hshrink.h"
#include "ops/ops_func_impl/hsigmoid_grad.h"
#include "ops/ops_func_impl/hsigmoid.h"
#include "ops/ops_func_impl/hswish_grad.h"
#include "ops/ops_func_impl/hswish.h"
#include "ops/ops_func_impl/identity.h"
#include "ops/ops_func_impl/ifft2.h"
#include "ops/ops_func_impl/ifft.h"
#include "ops/ops_func_impl/ifftn.h"
#include "ops/ops_func_impl/ifftshift.h"
#include "ops/ops_func_impl/im2col_ext.h"
#include "ops/ops_func_impl/index_add_ext.h"
#include "ops/ops_func_impl/index_select.h"
#include "ops/ops_func_impl/irfft_grad.h"
#include "ops/ops_func_impl/irfft.h"
#include "ops/ops_func_impl/isclose.h"
#include "ops/ops_func_impl/isfinite.h"
#include "ops/ops_func_impl/layer_norm_ext.h"
#include "ops/ops_func_impl/layer_norm_grad_ext.h"
#include "ops/ops_func_impl/layer_norm_grad_grad.h"
#include "ops/ops_func_impl/layer_norm_grad.h"
#include "ops/ops_func_impl/layer_norm_grad_v3.h"
#include "ops/ops_func_impl/layer_norm.h"
#include "ops/ops_func_impl/layer_norm_v3.h"
#include "ops/ops_func_impl/leaky_relu_ext.h"
#include "ops/ops_func_impl/leaky_relu_grad_ext.h"
#include "ops/ops_func_impl/less_equal.h"
#include "ops/ops_func_impl/less.h"
#include "ops/ops_func_impl/lin_space_ext.h"
#include "ops/ops_func_impl/lin_space.h"
#include "ops/ops_func_impl/list_to_tuple.h"
#include "ops/ops_func_impl/log1p.h"
#include "ops/ops_func_impl/log_matrix_determinant.h"
#include "ops/ops_func_impl/log.h"
#include "ops/ops_func_impl/log_softmax_grad.h"
#include "ops/ops_func_impl/log_softmax.h"
#include "ops/ops_func_impl/logical_and.h"
#include "ops/ops_func_impl/logical_not.h"
#include "ops/ops_func_impl/logical_or.h"
#include "ops/ops_func_impl/logical_xor.h"
#include "ops/ops_func_impl/logit_grad.h"
#include "ops/ops_func_impl/logit.h"
#include "ops/ops_func_impl/masked_fill.h"
#include "ops/ops_func_impl/matmul_ext.h"
#include "ops/ops_func_impl/matmul.h"
#include "ops/ops_func_impl/matrix_determinant.h"
#include "ops/ops_func_impl/matrix_exp.h"
#include "ops/ops_func_impl/matrix_inverse_ext.h"
#include "ops/ops_func_impl/max.h"
#include "ops/ops_func_impl/max_pool_grad_with_indices.h"
#include "ops/ops_func_impl/max_pool_grad_with_mask.h"
#include "ops/ops_func_impl/max_pool_with_indices.h"
#include "ops/ops_func_impl/max_pool_with_mask.h"
#include "ops/ops_func_impl/maximum_grad_grad.h"
#include "ops/ops_func_impl/maximum_grad.h"
#include "ops/ops_func_impl/maximum.h"
#include "ops/ops_func_impl/mean_ext.h"
#include "ops/ops_func_impl/min.h"
#include "ops/ops_func_impl/minimum_grad.h"
#include "ops/ops_func_impl/minimum.h"
#include "ops/ops_func_impl/mul.h"
#include "ops/ops_func_impl/mv.h"
#include "ops/ops_func_impl/nan_to_num.h"
#include "ops/ops_func_impl/neg.h"
#include "ops/ops_func_impl/next_after.h"
#include "ops/ops_func_impl/nllloss_grad.h"
#include "ops/ops_func_impl/nllloss.h"
#include "ops/ops_func_impl/non_zero_ext.h"
#include "ops/ops_func_impl/non_zero.h"
#include "ops/ops_func_impl/norm.h"
#include "ops/ops_func_impl/normal_float_float.h"
#include "ops/ops_func_impl/normal_float_tensor.h"
#include "ops/ops_func_impl/normal_tensor_float.h"
#include "ops/ops_func_impl/normal_tensor_tensor.h"
#include "ops/ops_func_impl/not_equal.h"
#include "ops/ops_func_impl/npu_clear_float_status_v2.h"
#include "ops/ops_func_impl/npu_get_float_status_v2.h"
#include "ops/ops_func_impl/one_hot_ext.h"
#include "ops/ops_func_impl/one_hot.h"
#include "ops/ops_func_impl/ones_like_ext.h"
#include "ops/ops_func_impl/ones_like.h"
#include "ops/ops_func_impl/ones.h"
#include "ops/ops_func_impl/paged_attention_mask.h"
#include "ops/ops_func_impl/paged_attention.h"
#include "ops/ops_func_impl/pow.h"
#include "ops/ops_func_impl/prelu_grad.h"
#include "ops/ops_func_impl/prelu.h"
#include "ops/ops_func_impl/prod_ext.h"
#include "ops/ops_func_impl/prompt_k_v_cache.h"
#include "ops/ops_func_impl/qr.h"
#include "ops/ops_func_impl/rand_ext.h"
#include "ops/ops_func_impl/rand_like_ext.h"
#include "ops/ops_func_impl/randperm_v2.h"
#include "ops/ops_func_impl/range.h"
#include "ops/ops_func_impl/rank.h"
#include "ops/ops_func_impl/real_div.h"
#include "ops/ops_func_impl/real.h"
#include "ops/ops_func_impl/reciprocal_grad.h"
#include "ops/ops_func_impl/reciprocal.h"
#include "ops/ops_func_impl/reduce_all.h"
#include "ops/ops_func_impl/reduce_any.h"
#include "ops/ops_func_impl/reduce_max.h"
#include "ops/ops_func_impl/reduce_mean.h"
#include "ops/ops_func_impl/reduce_min.h"
#include "ops/ops_func_impl/reduce_prod.h"
#include "ops/ops_func_impl/reduce_std.h"
#include "ops/ops_func_impl/reduce_sum.h"
#include "ops/ops_func_impl/reflection_pad_1d_grad.h"
#include "ops/ops_func_impl/reflection_pad_1d.h"
#include "ops/ops_func_impl/reflection_pad_2d_grad.h"
#include "ops/ops_func_impl/reflection_pad_2d.h"
#include "ops/ops_func_impl/reflection_pad_3d_grad.h"
#include "ops/ops_func_impl/reflection_pad_3d.h"
#include "ops/ops_func_impl/relu6_grad.h"
#include "ops/ops_func_impl/relu6.h"
#include "ops/ops_func_impl/relu_grad.h"
#include "ops/ops_func_impl/relu.h"
#include "ops/ops_func_impl/repeat_interleave_grad.h"
#include "ops/ops_func_impl/repeat_interleave_int.h"
#include "ops/ops_func_impl/repeat_interleave_tensor.h"
#include "ops/ops_func_impl/replication_pad_1d_grad.h"
#include "ops/ops_func_impl/replication_pad_1d.h"
#include "ops/ops_func_impl/replication_pad_2d_grad.h"
#include "ops/ops_func_impl/replication_pad_2d.h"
#include "ops/ops_func_impl/replication_pad_3d_grad.h"
#include "ops/ops_func_impl/replication_pad_3d.h"
#include "ops/ops_func_impl/reshape_and_cache.h"
#include "ops/ops_func_impl/reshape.h"
#include "ops/ops_func_impl/resize_bicubic_grad.h"
#include "ops/ops_func_impl/resize_bicubic.h"
#include "ops/ops_func_impl/resize_bilinear_grad.h"
#include "ops/ops_func_impl/resize_bilinear_v2.h"
#include "ops/ops_func_impl/resize_d.h"
#include "ops/ops_func_impl/resize_linear_1d_grad.h"
#include "ops/ops_func_impl/resize_linear_1d.h"
#include "ops/ops_func_impl/resize_nearest_neighbor_grad.h"
#include "ops/ops_func_impl/resize_nearest_neighbor.h"
#include "ops/ops_func_impl/resize_nearest_neighbor_v2_grad.h"
#include "ops/ops_func_impl/resize_nearest_neighbor_v2.h"
#include "ops/ops_func_impl/reverse_v2.h"
#include "ops/ops_func_impl/rfft_grad.h"
#include "ops/ops_func_impl/rfft.h"
#include "ops/ops_func_impl/right_shift.h"
#include "ops/ops_func_impl/rms_norm_grad.h"
#include "ops/ops_func_impl/rms_norm.h"
#include "ops/ops_func_impl/roll.h"
#include "ops/ops_func_impl/round.h"
#include "ops/ops_func_impl/rsqrt_grad.h"
#include "ops/ops_func_impl/rsqrt.h"
#include "ops/ops_func_impl/scalar_add.h"
#include "ops/ops_func_impl/scalar_bool.h"
#include "ops/ops_func_impl/scalar_cast.h"
#include "ops/ops_func_impl/scalar_div.h"
#include "ops/ops_func_impl/scalar_eq.h"
#include "ops/ops_func_impl/scalar_floor_div.h"
#include "ops/ops_func_impl/scalar_ge.h"
#include "ops/ops_func_impl/scalar_gt.h"
#include "ops/ops_func_impl/scalar_le.h"
#include "ops/ops_func_impl/scalar_log.h"
#include "ops/ops_func_impl/scalar_lt.h"
#include "ops/ops_func_impl/scalar_mod.h"
#include "ops/ops_func_impl/scalar_mul.h"
#include "ops/ops_func_impl/scalar_pow.h"
#include "ops/ops_func_impl/scalar_sub.h"
#include "ops/ops_func_impl/scalar_to_tensor.h"
#include "ops/ops_func_impl/scalar_uadd.h"
#include "ops/ops_func_impl/scalar_usub.h"
#include "ops/ops_func_impl/scatter_add_ext.h"
#include "ops/ops_func_impl/scatter_nd.h"
#include "ops/ops_func_impl/scatter.h"
#include "ops/ops_func_impl/searchsorted.h"
#include "ops/ops_func_impl/select.h"
#include "ops/ops_func_impl/sequence_concat.h"
#include "ops/ops_func_impl/shape.h"
#include "ops/ops_func_impl/sigmoid_grad.h"
#include "ops/ops_func_impl/sigmoid.h"
#include "ops/ops_func_impl/sign.h"
#include "ops/ops_func_impl/silu_grad.h"
#include "ops/ops_func_impl/silu.h"
#include "ops/ops_func_impl/sin.h"
#include "ops/ops_func_impl/sinc.h"
#include "ops/ops_func_impl/sinh.h"
#include "ops/ops_func_impl/slice_ext.h"
#include "ops/ops_func_impl/softmax_backward.h"
#include "ops/ops_func_impl/softmax.h"
#include "ops/ops_func_impl/softplus_ext.h"
#include "ops/ops_func_impl/softplus_grad_ext.h"
#include "ops/ops_func_impl/solve_triangular.h"
#include "ops/ops_func_impl/sort_ext.h"
#include "ops/ops_func_impl/split.h"
#include "ops/ops_func_impl/split_tensor.h"
#include "ops/ops_func_impl/split_with_size.h"
#include "ops/ops_func_impl/sqrt_grad.h"
#include "ops/ops_func_impl/sqrt.h"
#include "ops/ops_func_impl/square.h"
#include "ops/ops_func_impl/stack_ext.h"
#include "ops/ops_func_impl/strided_slice.h"
#include "ops/ops_func_impl/sub_ext.h"
#include "ops/ops_func_impl/sub.h"
#include "ops/ops_func_impl/sum_ext.h"
#include "ops/ops_func_impl/tanh_grad.h"
#include "ops/ops_func_impl/tanh.h"
#include "ops/ops_func_impl/tensor_copy_slices.h"
#include "ops/ops_func_impl/tensor_shape.h"
#include "ops/ops_func_impl/tile.h"
#include "ops/ops_func_impl/topk_ext.h"
#include "ops/ops_func_impl/topkrouter.h"
#include "ops/ops_func_impl/trace.h"
#include "ops/ops_func_impl/transpose.h"
#include "ops/ops_func_impl/triu.h"
#include "ops/ops_func_impl/tuple_to_list.h"
#include "ops/ops_func_impl/tuple_to_tensor.h"
#include "ops/ops_func_impl/uniform_ext.h"
#include "ops/ops_func_impl/unique2.h"
#include "ops/ops_func_impl/unique_dim.h"
#include "ops/ops_func_impl/unsorted_segment_sum.h"
#include "ops/ops_func_impl/unstack_ext.h"
#include "ops/ops_func_impl/upsample_bilinear2d_grad.h"
#include "ops/ops_func_impl/upsample_bilinear2d.h"
#include "ops/ops_func_impl/upsample_linear1d_grad.h"
#include "ops/ops_func_impl/upsample_linear1d.h"
#include "ops/ops_func_impl/upsample_nearest1d_grad.h"
#include "ops/ops_func_impl/upsample_nearest1d.h"
#include "ops/ops_func_impl/upsample_nearest2d_grad.h"
#include "ops/ops_func_impl/upsample_nearest2d.h"
#include "ops/ops_func_impl/upsample_nearest3d_grad.h"
#include "ops/ops_func_impl/upsample_nearest3d.h"
#include "ops/ops_func_impl/upsample_trilinear3d_grad.h"
#include "ops/ops_func_impl/upsample_trilinear3d.h"
#include "ops/ops_func_impl/view.h"
#include "ops/ops_func_impl/zeros_like_ext.h"
#include "ops/ops_func_impl/zeros_like.h"
#include "ops/ops_func_impl/zeros.h"
#include "ops/ops_func_impl/dynamic_quant_ext.h"
#include "ops/ops_func_impl/fused_infer_attention_score.h"
#include "ops/ops_func_impl/grouped_matmul.h"
#include "ops/ops_func_impl/kv_cache_scatter_update.h"
#include "ops/ops_func_impl/matmul_bias_split_out2.h"
#include "ops/ops_func_impl/matmul_bias_split_out3.h"
#include "ops/ops_func_impl/matmul_bias_split_silu_out2.h"
#include "ops/ops_func_impl/matmul_split_out2.h"
#include "ops/ops_func_impl/matmul_split_out3.h"
#include "ops/ops_func_impl/matmul_split_silu_out2.h"
#include "ops/ops_func_impl/moe_finalize_routing.h"
#include "ops/ops_func_impl/quant_batch_matmul.h"
#include "ops/ops_func_impl/quant_v2.h"
#include "ops/ops_func_impl/quantbatchmatmul_split_out2.h"
#include "ops/ops_func_impl/quantbatchmatmul_split_out3.h"
#include "ops/ops_func_impl/quantbatchmatmul_split_silu_out2.h"
#include "ops/ops_func_impl/weight_quant_batch_matmul.h"

namespace mindspore::ops {
ACosGradFuncImpl gACosGradFuncImpl;
OpDef gACosGrad = {
  /*.name_=*/"ACosGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dout", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"dx", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"dout", 1},
  },
  /*.func_impl_=*/gACosGradFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

AbsGradFuncImpl gAbsGradFuncImpl;
OpDef gAbsGrad = {
  /*.name_=*/"AbsGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dout", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"dx", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"dout", 1},
  },
  /*.func_impl_=*/gAbsGradFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

AbsFuncImpl gAbsFuncImpl;
OpDef gAbs = {
  /*.name_=*/"Abs",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
  },
  /*.func_impl_=*/gAbsFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

ACosFuncImpl gACosFuncImpl;
OpDef gACos = {
  /*.name_=*/"ACos",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
  },
  /*.func_impl_=*/gACosFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

AcoshGradFuncImpl gAcoshGradFuncImpl;
OpDef gAcoshGrad = {
  /*.name_=*/"AcoshGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"out", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dout", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"dx", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"out", 0},
    {"dout", 1},
  },
  /*.func_impl_=*/gAcoshGradFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

AcoshFuncImpl gAcoshFuncImpl;
OpDef gAcosh = {
  /*.name_=*/"Acosh",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
  },
  /*.func_impl_=*/gAcoshFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

AdamWeightDecayFuncImpl gAdamWeightDecayFuncImpl;
OpDef gAdamWeightDecay = {
  /*.name_=*/"AdamWeightDecay",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"var", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"m", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"v", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"lr", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_FLOAT}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"beta1", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_FLOAT}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"beta2", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_FLOAT}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"epsilon", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_FLOAT}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"decay", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_FLOAT}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"gradient", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"use_locking", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"var", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"m", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"v", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {
    Signature("var", SignatureEnumRW::kRWWrite,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
     Signature("m", SignatureEnumRW::kRWWrite,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType1),
     Signature("v", SignatureEnumRW::kRWWrite,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType1),
     Signature("lr", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType2),
     Signature("beta1", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType2),
     Signature("beta2", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType2),
     Signature("epsilon", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType2),
     Signature("decay", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType2),
     Signature("gradient", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
  },
  /*.indexes_ =*/ {
    {"var", 0},
    {"m", 1},
    {"v", 2},
    {"lr", 3},
    {"beta1", 4},
    {"beta2", 5},
    {"epsilon", 6},
    {"decay", 7},
    {"gradient", 8},
    {"use_locking", 9},
  },
  /*.func_impl_=*/gAdamWeightDecayFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

AdamWFuncImpl gAdamWFuncImpl;
OpDef gAdamW = {
  /*.name_=*/"AdamW",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"var", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"m", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"v", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"max_v", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"gradient", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"step", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"lr", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_TENSOR}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"beta1", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_TENSOR}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"beta2", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_TENSOR}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"decay", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_TENSOR}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"eps", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_TENSOR}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"amsgrad", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"maximize", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"var", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"m", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"v", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {
    Signature("var", SignatureEnumRW::kRWWrite,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
     Signature("m", SignatureEnumRW::kRWWrite,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType1),
     Signature("v", SignatureEnumRW::kRWWrite,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType1),
     Signature("max_v", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType1),
     Signature("gradient", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
     Signature("step", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType2),
     Signature("lr", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType3),
     Signature("beta1", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType3),
     Signature("beta2", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType3),
     Signature("decay", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType3),
     Signature("eps", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType3),
     Signature("amsgrad", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType4),
     Signature("maximize", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType5),
  },
  /*.indexes_ =*/ {
    {"var", 0},
    {"m", 1},
    {"v", 2},
    {"max_v", 3},
    {"gradient", 4},
    {"step", 5},
    {"lr", 6},
    {"beta1", 7},
    {"beta2", 8},
    {"decay", 9},
    {"eps", 10},
    {"amsgrad", 11},
    {"maximize", 12},
  },
  /*.func_impl_=*/gAdamWFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

AddExtFuncImpl gAddExtFuncImpl;
OpDef gAddExt = {
  /*.name_=*/"AddExt",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"other", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"alpha", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {
    Signature("input", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
     Signature("other", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
     Signature("alpha", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType1),
  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"other", 1},
    {"alpha", 2},
  },
  /*.func_impl_=*/gAddExtFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

AddLayerNormV2FuncImpl gAddLayerNormV2FuncImpl;
OpDef gAddLayerNormV2 = {
  /*.name_=*/"AddLayerNormV2",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x1", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"x2", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"gamma", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"beta", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"epsilon", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"additionalOut", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"mean", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"rstd", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x1", 0},
    {"x2", 1},
    {"gamma", 2},
    {"beta", 3},
    {"epsilon", 4},
    {"additionalOut", 5},
  },
  /*.func_impl_=*/gAddLayerNormV2FuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

AddFuncImpl gAddFuncImpl;
OpDef gAdd = {
  /*.name_=*/"Add",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"other", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {
    Signature("input", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
     Signature("other", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"other", 1},
  },
  /*.func_impl_=*/gAddFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

AddcdivFuncImpl gAddcdivFuncImpl;
OpDef gAddcdiv = {
  /*.name_=*/"Addcdiv",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"tensor1", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"tensor2", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"value", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"tensor1", 1},
    {"tensor2", 2},
    {"value", 3},
  },
  /*.func_impl_=*/gAddcdivFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

AddcmulFuncImpl gAddcmulFuncImpl;
OpDef gAddcmul = {
  /*.name_=*/"Addcmul",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"tensor1", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"tensor2", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"value", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"tensor1", 1},
    {"tensor2", 2},
    {"value", 3},
  },
  /*.func_impl_=*/gAddcmulFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

AddmmFuncImpl gAddmmFuncImpl;
OpDef gAddmm = {
  /*.name_=*/"Addmm",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"mat1", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"mat2", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"beta", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"alpha", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"mat1", 1},
    {"mat2", 2},
    {"beta", 3},
    {"alpha", 4},
  },
  /*.func_impl_=*/gAddmmFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

AddNFuncImpl gAddNFuncImpl;
OpDef gAddN = {
  /*.name_=*/"AddN",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TUPLE_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_TENSOR}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
  },
  /*.func_impl_=*/gAddNFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

AngleFuncImpl gAngleFuncImpl;
OpDef gAngle = {
  /*.name_=*/"Angle",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
  },
  /*.func_impl_=*/gAngleFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ApplyCamePart1FuncImpl gApplyCamePart1FuncImpl;
OpDef gApplyCamePart1 = {
  /*.name_=*/"ApplyCamePart1",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"grad", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"eps", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"sum_grad_r", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"sum_grad_c", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"sum_grad_rc", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"grad", 0},
    {"eps", 1},
  },
  /*.func_impl_=*/gApplyCamePart1FuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ApplyCamePart2FuncImpl gApplyCamePart2FuncImpl;
OpDef gApplyCamePart2 = {
  /*.name_=*/"ApplyCamePart2",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"grad", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"sum_grad_r", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"sum_grad_c", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"sum_grad_rc", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"r", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"c", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"beta2", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"sum_r", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"global_shape", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"r", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"c", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"u", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"sum_square_u", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"grad", 0},
    {"sum_grad_r", 1},
    {"sum_grad_c", 2},
    {"sum_grad_rc", 3},
    {"r", 4},
    {"c", 5},
    {"beta2", 6},
    {"sum_r", 7},
    {"global_shape", 8},
  },
  /*.func_impl_=*/gApplyCamePart2FuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ApplyCamePart3FuncImpl gApplyCamePart3FuncImpl;
OpDef gApplyCamePart3 = {
  /*.name_=*/"ApplyCamePart3",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"u", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"m", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"eps", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"beta1", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"clip_threshold", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"sum_square_u", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"global_shape", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"use_first_moment", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"m", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"sum_u_r", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"sum_u_c", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"sum_u_rc", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"u", 0},
    {"m", 1},
    {"eps", 2},
    {"beta1", 3},
    {"clip_threshold", 4},
    {"sum_square_u", 5},
    {"global_shape", 6},
    {"use_first_moment", 7},
  },
  /*.func_impl_=*/gApplyCamePart3FuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ApplyCamePart4FuncImpl gApplyCamePart4FuncImpl;
OpDef gApplyCamePart4 = {
  /*.name_=*/"ApplyCamePart4",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"param", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"m", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"r", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"c", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"weight_decay", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"lr", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"beta3", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"sum_r", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"sum_u_r", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"sum_u_c", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"sum_u_rc", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"global_shape", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"param", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"r", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"c", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"param", 0},
    {"m", 1},
    {"r", 2},
    {"c", 3},
    {"weight_decay", 4},
    {"lr", 5},
    {"beta3", 6},
    {"sum_r", 7},
    {"sum_u_r", 8},
    {"sum_u_c", 9},
    {"sum_u_rc", 10},
    {"global_shape", 11},
  },
  /*.func_impl_=*/gApplyCamePart4FuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ApplyRotaryPosEmbFuncImpl gApplyRotaryPosEmbFuncImpl;
OpDef gApplyRotaryPosEmb = {
  /*.name_=*/"ApplyRotaryPosEmb",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"query", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"key", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"cos", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"sin", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"position_ids", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"cos_format", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"query_embed", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"key_embed", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"query", 0},
    {"key", 1},
    {"cos", 2},
    {"sin", 3},
    {"position_ids", 4},
    {"cos_format", 5},
  },
  /*.func_impl_=*/gApplyRotaryPosEmbFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ArangeFuncImpl gArangeFuncImpl;
OpDef gArange = {
  /*.name_=*/"Arange",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"start", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"end", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"step", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dtype", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"dtype_to_type_id", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {
    Signature("start", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
     Signature("end", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
     Signature("step", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
     Signature("dtype", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType1),
  },
  /*.indexes_ =*/ {
    {"start", 0},
    {"end", 1},
    {"step", 2},
    {"dtype", 3},
  },
  /*.func_impl_=*/gArangeFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

ArgMaxExtFuncImpl gArgMaxExtFuncImpl;
OpDef gArgMaxExt = {
  /*.name_=*/"ArgMaxExt",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dim", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"keepdim", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"dim", 1},
    {"keepdim", 2},
  },
  /*.func_impl_=*/gArgMaxExtFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

ArgmaxFuncImpl gArgmaxFuncImpl;
OpDef gArgmax = {
  /*.name_=*/"Argmax",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input_x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"axis", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"output_type", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"dtype_to_type_id", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input_x", 0},
    {"axis", 1},
    {"output_type", 2},
  },
  /*.func_impl_=*/gArgmaxFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ArgMaxWithValueFuncImpl gArgMaxWithValueFuncImpl;
OpDef gArgMaxWithValue = {
  /*.name_=*/"ArgMaxWithValue",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"axis", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"keep_dims", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"index", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"values", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"axis", 1},
    {"keep_dims", 2},
  },
  /*.func_impl_=*/gArgMaxWithValueFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

ArgminFuncImpl gArgminFuncImpl;
OpDef gArgmin = {
  /*.name_=*/"Argmin",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"axis", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"output_type", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"dtype_to_type_id", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"axis", 1},
    {"output_type", 2},
  },
  /*.func_impl_=*/gArgminFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ArgMinWithValueFuncImpl gArgMinWithValueFuncImpl;
OpDef gArgMinWithValue = {
  /*.name_=*/"ArgMinWithValue",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"axis", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"keep_dims", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"index", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"values", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"axis", 1},
    {"keep_dims", 2},
  },
  /*.func_impl_=*/gArgMinWithValueFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

AsinGradFuncImpl gAsinGradFuncImpl;
OpDef gAsinGrad = {
  /*.name_=*/"AsinGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dout", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"dx", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"dout", 1},
  },
  /*.func_impl_=*/gAsinGradFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

AsinFuncImpl gAsinFuncImpl;
OpDef gAsin = {
  /*.name_=*/"Asin",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
  },
  /*.func_impl_=*/gAsinFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

AsinhGradFuncImpl gAsinhGradFuncImpl;
OpDef gAsinhGrad = {
  /*.name_=*/"AsinhGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"out", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dout", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"dx", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"out", 0},
    {"dout", 1},
  },
  /*.func_impl_=*/gAsinhGradFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

AsinhFuncImpl gAsinhFuncImpl;
OpDef gAsinh = {
  /*.name_=*/"Asinh",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
  },
  /*.func_impl_=*/gAsinhFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

AssignAddFuncImpl gAssignAddFuncImpl;
OpDef gAssignAdd = {
  /*.name_=*/"AssignAdd",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"variable", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"value", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {
    Signature("variable", SignatureEnumRW::kRWWrite,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
     Signature("value", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
  },
  /*.indexes_ =*/ {
    {"variable", 0},
    {"value", 1},
  },
  /*.func_impl_=*/gAssignAddFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

AssignFuncImpl gAssignFuncImpl;
OpDef gAssign = {
  /*.name_=*/"Assign",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"variable", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"value", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {
    Signature("variable", SignatureEnumRW::kRWWrite,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
     Signature("value", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
  },
  /*.indexes_ =*/ {
    {"variable", 0},
    {"value", 1},
  },
  /*.func_impl_=*/gAssignFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

Atan2ExtFuncImpl gAtan2ExtFuncImpl;
OpDef gAtan2Ext = {
  /*.name_=*/"Atan2Ext",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"other", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {
    Signature("input", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
     Signature("other", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"other", 1},
  },
  /*.func_impl_=*/gAtan2ExtFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

Atan2FuncImpl gAtan2FuncImpl;
OpDef gAtan2 = {
  /*.name_=*/"Atan2",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"other", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {
    Signature("input", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
     Signature("other", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"other", 1},
  },
  /*.func_impl_=*/gAtan2FuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

AtanGradFuncImpl gAtanGradFuncImpl;
OpDef gAtanGrad = {
  /*.name_=*/"AtanGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dout", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"dx", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"dout", 1},
  },
  /*.func_impl_=*/gAtanGradFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

AtanFuncImpl gAtanFuncImpl;
OpDef gAtan = {
  /*.name_=*/"Atan",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
  },
  /*.func_impl_=*/gAtanFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

AtanhFuncImpl gAtanhFuncImpl;
OpDef gAtanh = {
  /*.name_=*/"Atanh",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
  },
  /*.func_impl_=*/gAtanhFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

AvgPool2DGradFuncImpl gAvgPool2DGradFuncImpl;
OpDef gAvgPool2DGrad = {
  /*.name_=*/"AvgPool2DGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"grad", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"image", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"kernel_size", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_INT, DT_LIST_INT}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"stride", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_INT, DT_LIST_INT}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"padding", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_INT, DT_LIST_INT}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"ceil_mode", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"count_include_pad", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"divisor_override", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"grad", 0},
    {"image", 1},
    {"kernel_size", 2},
    {"stride", 3},
    {"padding", 4},
    {"ceil_mode", 5},
    {"count_include_pad", 6},
    {"divisor_override", 7},
  },
  /*.func_impl_=*/gAvgPool2DGradFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

AvgPool2DFuncImpl gAvgPool2DFuncImpl;
OpDef gAvgPool2D = {
  /*.name_=*/"AvgPool2D",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"kernel_size", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_INT, DT_LIST_INT}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"stride", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_INT, DT_LIST_INT}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"padding", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_INT, DT_LIST_INT}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"ceil_mode", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"count_include_pad", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"divisor_override", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"kernel_size", 1},
    {"stride", 2},
    {"padding", 3},
    {"ceil_mode", 4},
    {"count_include_pad", 5},
    {"divisor_override", 6},
  },
  /*.func_impl_=*/gAvgPool2DFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

AvgPoolGradFuncImpl gAvgPoolGradFuncImpl;
OpDef gAvgPoolGrad = {
  /*.name_=*/"AvgPoolGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"out", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dout", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"kernel_size", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"to_kernel_size", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"strides", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"to_strides", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"pad_mode", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"str_to_enum", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"data_format", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"str_to_enum", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"out", 1},
    {"dout", 2},
    {"kernel_size", 3},
    {"strides", 4},
    {"pad_mode", 5},
    {"data_format", 6},
  },
  /*.func_impl_=*/gAvgPoolGradFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

AvgPoolFuncImpl gAvgPoolFuncImpl;
OpDef gAvgPool = {
  /*.name_=*/"AvgPool",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"kernel_size", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"to_kernel_size", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"strides", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"to_strides", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"pad_mode", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"str_to_enum", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"data_format", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"str_to_enum", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"kernel_size", 1},
    {"strides", 2},
    {"pad_mode", 3},
    {"data_format", 4},
  },
  /*.func_impl_=*/gAvgPoolFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

BatchMatMulFuncImpl gBatchMatMulFuncImpl;
OpDef gBatchMatMul = {
  /*.name_=*/"BatchMatMul",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"transpose_a", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"transpose_b", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"y", 1},
    {"transpose_a", 2},
    {"transpose_b", 3},
  },
  /*.func_impl_=*/gBatchMatMulFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

BatchNormExtFuncImpl gBatchNormExtFuncImpl;
OpDef gBatchNormExt = {
  /*.name_=*/"BatchNormExt",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"weight", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"bias", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"running_mean", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"runnning_var", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"training", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"momentum", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"epsilon", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"saved_mean", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"saved_variance", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"weight", 1},
    {"bias", 2},
    {"running_mean", 3},
    {"runnning_var", 4},
    {"training", 5},
    {"momentum", 6},
    {"epsilon", 7},
  },
  /*.func_impl_=*/gBatchNormExtFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

BatchNormGradExtFuncImpl gBatchNormGradExtFuncImpl;
OpDef gBatchNormGradExt = {
  /*.name_=*/"BatchNormGradExt",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"dout", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"weight", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"running_mean", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"running_var", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"saved_mean", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"saved_rstd", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"training", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"eps", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"dx", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"dweight", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"dbias", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"dout", 0},
    {"input", 1},
    {"weight", 2},
    {"running_mean", 3},
    {"running_var", 4},
    {"saved_mean", 5},
    {"saved_rstd", 6},
    {"training", 7},
    {"eps", 8},
  },
  /*.func_impl_=*/gBatchNormGradExtFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

BatchNormGradGradFuncImpl gBatchNormGradGradFuncImpl;
OpDef gBatchNormGradGrad = {
  /*.name_=*/"BatchNormGradGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dy", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"scale", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"saved_mean", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"saved_variance", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dout_dx", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dout_dscale", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dout_dbias", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"is_training", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"epsilon", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"data_format", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"str_to_enum", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"dx", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"ddy", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"dscale", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"dy", 1},
    {"scale", 2},
    {"saved_mean", 3},
    {"saved_variance", 4},
    {"dout_dx", 5},
    {"dout_dscale", 6},
    {"dout_dbias", 7},
    {"is_training", 8},
    {"epsilon", 9},
    {"data_format", 10},
  },
  /*.func_impl_=*/gBatchNormGradGradFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

BatchNormGradFuncImpl gBatchNormGradFuncImpl;
OpDef gBatchNormGrad = {
  /*.name_=*/"BatchNormGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"dout", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"scale", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"saved_mean", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"saved_variance", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"reserve", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"is_training", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"epsilon", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"data_format", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"str_to_enum", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"dx", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"dscale", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"dbias", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"dout", 0},
    {"x", 1},
    {"scale", 2},
    {"saved_mean", 3},
    {"saved_variance", 4},
    {"reserve", 5},
    {"is_training", 6},
    {"epsilon", 7},
    {"data_format", 8},
  },
  /*.func_impl_=*/gBatchNormGradFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

BatchNormGradWithActivationFuncImpl gBatchNormGradWithActivationFuncImpl;
OpDef gBatchNormGradWithActivation = {
  /*.name_=*/"BatchNormGradWithActivation",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"dy", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"scale", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"saved_mean", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"saved_variance", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"reserve", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"bias", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"is_training", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"epsilon", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"data_format", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"str_to_enum", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"dx", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"dscale", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"dbias", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"dy", 0},
    {"x", 1},
    {"scale", 2},
    {"saved_mean", 3},
    {"saved_variance", 4},
    {"reserve", 5},
    {"bias", 6},
    {"y", 7},
    {"is_training", 8},
    {"epsilon", 9},
    {"data_format", 10},
  },
  /*.func_impl_=*/gBatchNormGradWithActivationFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

BatchNormFuncImpl gBatchNormFuncImpl;
OpDef gBatchNorm = {
  /*.name_=*/"BatchNorm",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input_x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"scale", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"bias", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"mean", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"variance", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"is_training", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"epsilon", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"momentum", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"data_format", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"str_to_enum", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output_x", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"batch_mean", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"batch_variance", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"reserve_space_1", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"reserve_space_2", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input_x", 0},
    {"scale", 1},
    {"bias", 2},
    {"mean", 3},
    {"variance", 4},
    {"is_training", 5},
    {"epsilon", 6},
    {"momentum", 7},
    {"data_format", 8},
  },
  /*.func_impl_=*/gBatchNormFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

BatchNormWithActivationFuncImpl gBatchNormWithActivationFuncImpl;
OpDef gBatchNormWithActivation = {
  /*.name_=*/"BatchNormWithActivation",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"scale", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"bias", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"mean", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"var", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"is_training", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"epsilon", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"momentum", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"data_format", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"str_to_enum", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output_x", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"batch_mean", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"batch_variance", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"reserve_space_1", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"reserve_space_2", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"scale", 1},
    {"bias", 2},
    {"mean", 3},
    {"var", 4},
    {"is_training", 5},
    {"epsilon", 6},
    {"momentum", 7},
    {"data_format", 8},
  },
  /*.func_impl_=*/gBatchNormWithActivationFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

BatchNormWithAddAndActivationFuncImpl gBatchNormWithAddAndActivationFuncImpl;
OpDef gBatchNormWithAddAndActivation = {
  /*.name_=*/"BatchNormWithAddAndActivation",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"scale", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"bias", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"mean", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"var", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"z", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"is_training", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"epsilon", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"momentum", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"data_format", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"str_to_enum", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output_x", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"batch_mean", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"batch_variance", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"reserve_space_1", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"reserve_space_2", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"scale", 1},
    {"bias", 2},
    {"mean", 3},
    {"var", 4},
    {"z", 5},
    {"is_training", 6},
    {"epsilon", 7},
    {"momentum", 8},
    {"data_format", 9},
  },
  /*.func_impl_=*/gBatchNormWithAddAndActivationFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

BetaincFuncImpl gBetaincFuncImpl;
OpDef gBetainc = {
  /*.name_=*/"Betainc",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"a", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"b", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"a", 0},
    {"b", 1},
    {"x", 2},
  },
  /*.func_impl_=*/gBetaincFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

BiasAddGradFuncImpl gBiasAddGradFuncImpl;
OpDef gBiasAddGrad = {
  /*.name_=*/"BiasAddGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"dout", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"data_format", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"str_to_enum", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"dout", 0},
    {"data_format", 1},
  },
  /*.func_impl_=*/gBiasAddGradFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

BiasAddFuncImpl gBiasAddFuncImpl;
OpDef gBiasAdd = {
  /*.name_=*/"BiasAdd",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input_x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"bias", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"data_format", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"str_to_enum", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input_x", 0},
    {"bias", 1},
    {"data_format", 2},
  },
  /*.func_impl_=*/gBiasAddFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

BinaryCrossEntropyGradFuncImpl gBinaryCrossEntropyGradFuncImpl;
OpDef gBinaryCrossEntropyGrad = {
  /*.name_=*/"BinaryCrossEntropyGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"target", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"grad_output", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"weight", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"reduction", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"str_to_enum", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"out", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"target", 1},
    {"grad_output", 2},
    {"weight", 3},
    {"reduction", 4},
  },
  /*.func_impl_=*/gBinaryCrossEntropyGradFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

BinaryCrossEntropyFuncImpl gBinaryCrossEntropyFuncImpl;
OpDef gBinaryCrossEntropy = {
  /*.name_=*/"BinaryCrossEntropy",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"target", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"weight", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"reduction", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"str_to_enum", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"out", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"target", 1},
    {"weight", 2},
    {"reduction", 3},
  },
  /*.func_impl_=*/gBinaryCrossEntropyFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

BinaryCrossEntropyWithLogitsBackwardFuncImpl gBinaryCrossEntropyWithLogitsBackwardFuncImpl;
OpDef gBinaryCrossEntropyWithLogitsBackward = {
  /*.name_=*/"BinaryCrossEntropyWithLogitsBackward",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"grad_output", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"target", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"weight", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"posWeight", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"reduction", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"str_to_enum", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"out", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"grad_output", 0},
    {"input", 1},
    {"target", 2},
    {"weight", 3},
    {"posWeight", 4},
    {"reduction", 5},
  },
  /*.func_impl_=*/gBinaryCrossEntropyWithLogitsBackwardFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

BCEWithLogitsLossFuncImpl gBCEWithLogitsLossFuncImpl;
OpDef gBCEWithLogitsLoss = {
  /*.name_=*/"BCEWithLogitsLoss",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"target", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"weight", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"posWeight", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"reduction", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"str_to_enum", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"out", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"target", 1},
    {"weight", 2},
    {"posWeight", 3},
    {"reduction", 4},
  },
  /*.func_impl_=*/gBCEWithLogitsLossFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

BatchMatMulExtFuncImpl gBatchMatMulExtFuncImpl;
OpDef gBatchMatMulExt = {
  /*.name_=*/"BatchMatMulExt",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"mat2", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"mat2", 1},
  },
  /*.func_impl_=*/gBatchMatMulExtFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

BoolNotFuncImpl gBoolNotFuncImpl;
OpDef gBoolNot = {
  /*.name_=*/"BoolNot",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_NUMBER,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
  },
  /*.func_impl_=*/gBoolNotFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

BroadcastToFuncImpl gBroadcastToFuncImpl;
OpDef gBroadcastTo = {
  /*.name_=*/"BroadcastTo",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"shape", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT, DT_TENSOR}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"shape", 1},
  },
  /*.func_impl_=*/gBroadcastToFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/true,
};

CastFuncImpl gCastFuncImpl;
OpDef gCast = {
  /*.name_=*/"Cast",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input_x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dtype", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"dtype_to_type_id", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input_x", 0},
    {"dtype", 1},
  },
  /*.func_impl_=*/gCastFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

CeilFuncImpl gCeilFuncImpl;
OpDef gCeil = {
  /*.name_=*/"Ceil",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
  },
  /*.func_impl_=*/gCeilFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

CeLUFuncImpl gCeLUFuncImpl;
OpDef gCeLU = {
  /*.name_=*/"CeLU",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"alpha", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"alpha", 1},
  },
  /*.func_impl_=*/gCeLUFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

CholeskyGradFuncImpl gCholeskyGradFuncImpl;
OpDef gCholeskyGrad = {
  /*.name_=*/"CholeskyGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"grad", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"dx", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"grad", 1},
  },
  /*.func_impl_=*/gCholeskyGradFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

CholeskyInverseFuncImpl gCholeskyInverseFuncImpl;
OpDef gCholeskyInverse = {
  /*.name_=*/"CholeskyInverse",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input_x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"upper", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input_x", 0},
    {"upper", 1},
  },
  /*.func_impl_=*/gCholeskyInverseFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

CholeskyFuncImpl gCholeskyFuncImpl;
OpDef gCholesky = {
  /*.name_=*/"Cholesky",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input_x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"upper", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input_x", 0},
    {"upper", 1},
  },
  /*.func_impl_=*/gCholeskyFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ChunkFuncImpl gChunkFuncImpl;
OpDef gChunk = {
  /*.name_=*/"Chunk",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"chunks", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dim", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_TUPLE_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"chunks", 1},
    {"dim", 2},
  },
  /*.func_impl_=*/gChunkFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

ClampScalarFuncImpl gClampScalarFuncImpl;
OpDef gClampScalar = {
  /*.name_=*/"ClampScalar",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"min", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"max", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"min", 1},
    {"max", 2},
  },
  /*.func_impl_=*/gClampScalarFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

ClampTensorFuncImpl gClampTensorFuncImpl;
OpDef gClampTensor = {
  /*.name_=*/"ClampTensor",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"min", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"max", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"min", 1},
    {"max", 2},
  },
  /*.func_impl_=*/gClampTensorFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

Col2ImExtFuncImpl gCol2ImExtFuncImpl;
OpDef gCol2ImExt = {
  /*.name_=*/"Col2ImExt",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"output_size", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"to_pair", /*.cast_dtype_ =*/{DT_LIST_INT}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"kernel_size", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"to_pair", /*.cast_dtype_ =*/{DT_LIST_INT}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dilation", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"to_pair", /*.cast_dtype_ =*/{DT_LIST_INT}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"padding", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"to_pair", /*.cast_dtype_ =*/{DT_LIST_INT}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"stride", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"to_pair", /*.cast_dtype_ =*/{DT_LIST_INT}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"output_size", 1},
    {"kernel_size", 2},
    {"dilation", 3},
    {"padding", 4},
    {"stride", 5},
  },
  /*.func_impl_=*/gCol2ImExtFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

Col2ImGradFuncImpl gCol2ImGradFuncImpl;
OpDef gCol2ImGrad = {
  /*.name_=*/"Col2ImGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"kernel_size", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"to_pair", /*.cast_dtype_ =*/{DT_LIST_INT}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dilation", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"to_pair", /*.cast_dtype_ =*/{DT_LIST_INT}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"padding", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"to_pair", /*.cast_dtype_ =*/{DT_LIST_INT}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"stride", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"to_pair", /*.cast_dtype_ =*/{DT_LIST_INT}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"kernel_size", 1},
    {"dilation", 2},
    {"padding", 3},
    {"stride", 4},
  },
  /*.func_impl_=*/gCol2ImGradFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

ComplexFuncImpl gComplexFuncImpl;
OpDef gComplex = {
  /*.name_=*/"Complex",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"real", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"imag", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"real", 0},
    {"imag", 1},
  },
  /*.func_impl_=*/gComplexFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ConcatFuncImpl gConcatFuncImpl;
OpDef gConcat = {
  /*.name_=*/"Concat",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"tensors", /*.arg_dtype_=*/DT_TUPLE_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_TENSOR}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"axis", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"out", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"tensors", 0},
    {"axis", 1},
  },
  /*.func_impl_=*/gConcatFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

ConjFuncImpl gConjFuncImpl;
OpDef gConj = {
  /*.name_=*/"Conj",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
  },
  /*.func_impl_=*/gConjFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ConstantPadNDFuncImpl gConstantPadNDFuncImpl;
OpDef gConstantPadND = {
  /*.name_=*/"ConstantPadND",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"padding", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT, DT_TENSOR}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"value", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"padding", 1},
    {"value", 2},
  },
  /*.func_impl_=*/gConstantPadNDFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

ContiguousFuncImpl gContiguousFuncImpl;
OpDef gContiguous = {
  /*.name_=*/"Contiguous",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
  },
  /*.func_impl_=*/gContiguousFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/true,
};

ConvolutionGradFuncImpl gConvolutionGradFuncImpl;
OpDef gConvolutionGrad = {
  /*.name_=*/"ConvolutionGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"dout", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"weight", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"bias", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"stride", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"to_strides", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"padding", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"to_2d_paddings", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dilation", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"to_dilations", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"transposed", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"output_padding", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"to_output_padding", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"groups", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"output_mask", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"dx", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"dw", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"dbias", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"dout", 0},
    {"input", 1},
    {"weight", 2},
    {"bias", 3},
    {"stride", 4},
    {"padding", 5},
    {"dilation", 6},
    {"transposed", 7},
    {"output_padding", 8},
    {"groups", 9},
    {"output_mask", 10},
  },
  /*.func_impl_=*/gConvolutionGradFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

ConvolutionFuncImpl gConvolutionFuncImpl;
OpDef gConvolution = {
  /*.name_=*/"Convolution",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"weight", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"bias", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"stride", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"to_strides", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"padding", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"to_2d_paddings", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dilation", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"to_dilations", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"transposed", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"output_padding", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"to_output_padding", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"groups", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"weight", 1},
    {"bias", 2},
    {"stride", 3},
    {"padding", 4},
    {"dilation", 5},
    {"transposed", 6},
    {"output_padding", 7},
    {"groups", 8},
  },
  /*.func_impl_=*/gConvolutionFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

CopyFuncImpl gCopyFuncImpl;
OpDef gCopy = {
  /*.name_=*/"Copy",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
  },
  /*.func_impl_=*/gCopyFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/true,
};

CorrelateFuncImpl gCorrelateFuncImpl;
OpDef gCorrelate = {
  /*.name_=*/"Correlate",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"a", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"v", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"mode", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"str_to_enum", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"a", 0},
    {"v", 1},
    {"mode", 2},
  },
  /*.func_impl_=*/gCorrelateFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

CosFuncImpl gCosFuncImpl;
OpDef gCos = {
  /*.name_=*/"Cos",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
  },
  /*.func_impl_=*/gCosFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

CoshFuncImpl gCoshFuncImpl;
OpDef gCosh = {
  /*.name_=*/"Cosh",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
  },
  /*.func_impl_=*/gCoshFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

CumProdFuncImpl gCumProdFuncImpl;
OpDef gCumProd = {
  /*.name_=*/"CumProd",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"axis", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"exclusive", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"reverse", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"axis", 1},
    {"exclusive", 2},
    {"reverse", 3},
  },
  /*.func_impl_=*/gCumProdFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

CumSumFuncImpl gCumSumFuncImpl;
OpDef gCumSum = {
  /*.name_=*/"CumSum",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"axis", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"exclusive", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"reverse", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"axis", 1},
    {"exclusive", 2},
    {"reverse", 3},
  },
  /*.func_impl_=*/gCumSumFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

CummaxFuncImpl gCummaxFuncImpl;
OpDef gCummax = {
  /*.name_=*/"Cummax",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"axis", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"values", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"indices", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"axis", 1},
  },
  /*.func_impl_=*/gCummaxFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

CumminFuncImpl gCumminFuncImpl;
OpDef gCummin = {
  /*.name_=*/"Cummin",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"axis", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"values", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"indices", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"axis", 1},
  },
  /*.func_impl_=*/gCumminFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

CumsumExtFuncImpl gCumsumExtFuncImpl;
OpDef gCumsumExt = {
  /*.name_=*/"CumsumExt",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dim", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dtype", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"dtype_to_type_id", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"dim", 1},
    {"dtype", 2},
  },
  /*.func_impl_=*/gCumsumExtFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

DCTFuncImpl gDCTFuncImpl;
OpDef gDCT = {
  /*.name_=*/"DCT",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"type", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"n", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"axis", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"norm", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"str_to_enum", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"forward", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"grad", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"type", 1},
    {"n", 2},
    {"axis", 3},
    {"norm", 4},
    {"forward", 5},
    {"grad", 6},
  },
  /*.func_impl_=*/gDCTFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

DecoderKVCacheFuncImpl gDecoderKVCacheFuncImpl;
OpDef gDecoderKVCache = {
  /*.name_=*/"DecoderKVCache",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"cache", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"update", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"valid_seq_len", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"batch_index", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"seq_len_axis", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"new_max_seq_len", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"cur_max_seq_len", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"out", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"cache", 0},
    {"update", 1},
    {"valid_seq_len", 2},
    {"batch_index", 3},
    {"seq_len_axis", 4},
    {"new_max_seq_len", 5},
    {"cur_max_seq_len", 6},
  },
  /*.func_impl_=*/gDecoderKVCacheFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

DenseFuncImpl gDenseFuncImpl;
OpDef gDense = {
  /*.name_=*/"Dense",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"weight", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"bias", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"weight", 1},
    {"bias", 2},
  },
  /*.func_impl_=*/gDenseFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

DiagFuncImpl gDiagFuncImpl;
OpDef gDiag = {
  /*.name_=*/"Diag",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
  },
  /*.func_impl_=*/gDiagFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

DiagonalFuncImpl gDiagonalFuncImpl;
OpDef gDiagonal = {
  /*.name_=*/"Diagonal",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"offset", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dim1", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dim2", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"offset", 1},
    {"dim1", 2},
    {"dim2", 3},
  },
  /*.func_impl_=*/gDiagonalFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

DivFuncImpl gDivFuncImpl;
OpDef gDiv = {
  /*.name_=*/"Div",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {
    Signature("x", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
     Signature("y", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"y", 1},
  },
  /*.func_impl_=*/gDivFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

DivModFuncImpl gDivModFuncImpl;
OpDef gDivMod = {
  /*.name_=*/"DivMod",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"rounding_mode", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"str_to_enum", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {
    Signature("x", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
     Signature("y", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
     Signature("rounding_mode", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType1),
  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"y", 1},
    {"rounding_mode", 2},
  },
  /*.func_impl_=*/gDivModFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

DotFuncImpl gDotFuncImpl;
OpDef gDot = {
  /*.name_=*/"Dot",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"other", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"other", 1},
  },
  /*.func_impl_=*/gDotFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

DropoutDoMaskExtFuncImpl gDropoutDoMaskExtFuncImpl;
OpDef gDropoutDoMaskExt = {
  /*.name_=*/"DropoutDoMaskExt",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"mask", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"p", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"mask", 1},
    {"p", 2},
  },
  /*.func_impl_=*/gDropoutDoMaskExtFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

DropoutExtFuncImpl gDropoutExtFuncImpl;
OpDef gDropoutExt = {
  /*.name_=*/"DropoutExt",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"p", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"seed", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"offset", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"mask", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"p", 1},
    {"seed", 2},
    {"offset", 3},
  },
  /*.func_impl_=*/gDropoutExtFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

DropoutGenMaskExtFuncImpl gDropoutGenMaskExtFuncImpl;
OpDef gDropoutGenMaskExt = {
  /*.name_=*/"DropoutGenMaskExt",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"shape", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"p", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"seed", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_TENSOR}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"offset", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_TENSOR}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dtype", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"dtype_to_type_id", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"shape", 0},
    {"p", 1},
    {"seed", 2},
    {"offset", 3},
    {"dtype", 4},
  },
  /*.func_impl_=*/gDropoutGenMaskExtFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

DropoutGradExtFuncImpl gDropoutGradExtFuncImpl;
OpDef gDropoutGradExt = {
  /*.name_=*/"DropoutGradExt",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"mask", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"p", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"mask", 1},
    {"p", 2},
  },
  /*.func_impl_=*/gDropoutGradExtFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

DropoutFuncImpl gDropoutFuncImpl;
OpDef gDropout = {
  /*.name_=*/"Dropout",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"keep_prob", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"Seed0", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"Seed1", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"mask", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"keep_prob", 1},
    {"Seed0", 2},
    {"Seed1", 3},
  },
  /*.func_impl_=*/gDropoutFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

EigFuncImpl gEigFuncImpl;
OpDef gEig = {
  /*.name_=*/"Eig",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"compute_v", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"eigen_values", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"eigen_vectors", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"compute_v", 1},
  },
  /*.func_impl_=*/gEigFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

EluExtFuncImpl gEluExtFuncImpl;
OpDef gEluExt = {
  /*.name_=*/"EluExt",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"alpha", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"alpha", 1},
  },
  /*.func_impl_=*/gEluExtFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

EluGradExtFuncImpl gEluGradExtFuncImpl;
OpDef gEluGradExt = {
  /*.name_=*/"EluGradExt",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"dout", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"alpha", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"dx", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"dout", 0},
    {"x", 1},
    {"alpha", 2},
  },
  /*.func_impl_=*/gEluGradExtFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

EluGradFuncImpl gEluGradFuncImpl;
OpDef gEluGrad = {
  /*.name_=*/"EluGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"dout", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"out", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"dx", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"dout", 0},
    {"out", 1},
  },
  /*.func_impl_=*/gEluGradFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

EluFuncImpl gEluFuncImpl;
OpDef gElu = {
  /*.name_=*/"Elu",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input_x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"alpha", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input_x", 0},
    {"alpha", 1},
  },
  /*.func_impl_=*/gEluFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

EmbeddingDenseBackwardFuncImpl gEmbeddingDenseBackwardFuncImpl;
OpDef gEmbeddingDenseBackward = {
  /*.name_=*/"EmbeddingDenseBackward",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"grad", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"indices", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"num_weights", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"padding_idx", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"scale_grad_by_freq", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"grad", 0},
    {"indices", 1},
    {"num_weights", 2},
    {"padding_idx", 3},
    {"scale_grad_by_freq", 4},
  },
  /*.func_impl_=*/gEmbeddingDenseBackwardFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

EmbeddingFuncImpl gEmbeddingFuncImpl;
OpDef gEmbedding = {
  /*.name_=*/"Embedding",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"weight", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"padding_idx", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"max_norm", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"norm_type", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"scale_grad_by_freq", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output1", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {
    Signature("input", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDTypeEmptyDefaultValue),
     Signature("weight", SignatureEnumRW::kRWWrite,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDTypeEmptyDefaultValue),
     Signature("padding_idx", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDTypeEmptyDefaultValue),
     Signature("max_norm", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDTypeEmptyDefaultValue),
     Signature("norm_type", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDTypeEmptyDefaultValue),
     Signature("scale_grad_by_freq", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDTypeEmptyDefaultValue),
  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"weight", 1},
    {"padding_idx", 2},
    {"max_norm", 3},
    {"norm_type", 4},
    {"scale_grad_by_freq", 5},
  },
  /*.func_impl_=*/gEmbeddingFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

EqualFuncImpl gEqualFuncImpl;
OpDef gEqual = {
  /*.name_=*/"Equal",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"other", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {
    Signature("input", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
     Signature("other", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"other", 1},
  },
  /*.func_impl_=*/gEqualFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

ErfFuncImpl gErfFuncImpl;
OpDef gErf = {
  /*.name_=*/"Erf",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
  },
  /*.func_impl_=*/gErfFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

ErfcFuncImpl gErfcFuncImpl;
OpDef gErfc = {
  /*.name_=*/"Erfc",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
  },
  /*.func_impl_=*/gErfcFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ErfinvFuncImpl gErfinvFuncImpl;
OpDef gErfinv = {
  /*.name_=*/"Erfinv",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
  },
  /*.func_impl_=*/gErfinvFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

ExpFuncImpl gExpFuncImpl;
OpDef gExp = {
  /*.name_=*/"Exp",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
  },
  /*.func_impl_=*/gExpFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

ExpandDimsFuncImpl gExpandDimsFuncImpl;
OpDef gExpandDims = {
  /*.name_=*/"ExpandDims",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input_x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"axis", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_TENSOR}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input_x", 0},
    {"axis", 1},
  },
  /*.func_impl_=*/gExpandDimsFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

Expm1FuncImpl gExpm1FuncImpl;
OpDef gExpm1 = {
  /*.name_=*/"Expm1",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
  },
  /*.func_impl_=*/gExpm1FuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ExtractImagePatchesFuncImpl gExtractImagePatchesFuncImpl;
OpDef gExtractImagePatches = {
  /*.name_=*/"ExtractImagePatches",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input_x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"ksizes", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"to_kernel_size", /*.cast_dtype_ =*/{DT_LIST_INT}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"strides", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"to_strides", /*.cast_dtype_ =*/{DT_LIST_INT}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"rates", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"to_rates", /*.cast_dtype_ =*/{DT_LIST_INT}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"padding", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"str_to_enum", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input_x", 0},
    {"ksizes", 1},
    {"strides", 2},
    {"rates", 3},
    {"padding", 4},
  },
  /*.func_impl_=*/gExtractImagePatchesFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

EyeFuncImpl gEyeFuncImpl;
OpDef gEye = {
  /*.name_=*/"Eye",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"n", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"m", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dtype", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"dtype_to_type_id", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"n", 0},
    {"m", 1},
    {"dtype", 2},
  },
  /*.func_impl_=*/gEyeFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

FastGeLUGradFuncImpl gFastGeLUGradFuncImpl;
OpDef gFastGeLUGrad = {
  /*.name_=*/"FastGeLUGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"dy", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"dy", 0},
    {"x", 1},
  },
  /*.func_impl_=*/gFastGeLUGradFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

FastGeLUFuncImpl gFastGeLUFuncImpl;
OpDef gFastGeLU = {
  /*.name_=*/"FastGeLU",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
  },
  /*.func_impl_=*/gFastGeLUFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

FFNExtFuncImpl gFFNExtFuncImpl;
OpDef gFFNExt = {
  /*.name_=*/"FFNExt",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"weight1", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"weight2", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"expertTokens", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_TENSOR}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"bias1", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"bias2", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"scale", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"offset", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"deqScale1", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"deqScale2", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"antiquant_scale1", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"antiquant_scale2", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"antiquant_offset1", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"antiquant_offset2", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"activation", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"str_to_enum", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"inner_precise", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"weight1", 1},
    {"weight2", 2},
    {"expertTokens", 3},
    {"bias1", 4},
    {"bias2", 5},
    {"scale", 6},
    {"offset", 7},
    {"deqScale1", 8},
    {"deqScale2", 9},
    {"antiquant_scale1", 10},
    {"antiquant_scale2", 11},
    {"antiquant_offset1", 12},
    {"antiquant_offset2", 13},
    {"activation", 14},
    {"inner_precise", 15},
  },
  /*.func_impl_=*/gFFNExtFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

FFT2FuncImpl gFFT2FuncImpl;
OpDef gFFT2 = {
  /*.name_=*/"FFT2",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"s", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"dim", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"norm", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"str_to_enum", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"s", 1},
    {"dim", 2},
    {"norm", 3},
  },
  /*.func_impl_=*/gFFT2FuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

FFTFuncImpl gFFTFuncImpl;
OpDef gFFT = {
  /*.name_=*/"FFT",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"n", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"dim", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"norm", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"str_to_enum", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"n", 1},
    {"dim", 2},
    {"norm", 3},
  },
  /*.func_impl_=*/gFFTFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

FFTShapeCopyFuncImpl gFFTShapeCopyFuncImpl;
OpDef gFFTShapeCopy = {
  /*.name_=*/"FFTShapeCopy",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"dout", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"shape", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"dout", 0},
    {"shape", 1},
  },
  /*.func_impl_=*/gFFTShapeCopyFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

FFTWithSizeFuncImpl gFFTWithSizeFuncImpl;
OpDef gFFTWithSize = {
  /*.name_=*/"FFTWithSize",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"signal_ndim", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"inverse", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"real", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"norm", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"str_to_enum", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"onesided", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"signal_sizes", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"signal_ndim", 1},
    {"inverse", 2},
    {"real", 3},
    {"norm", 4},
    {"onesided", 5},
    {"signal_sizes", 6},
  },
  /*.func_impl_=*/gFFTWithSizeFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

FFTNFuncImpl gFFTNFuncImpl;
OpDef gFFTN = {
  /*.name_=*/"FFTN",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"s", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"dim", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"norm", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"str_to_enum", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"s", 1},
    {"dim", 2},
    {"norm", 3},
  },
  /*.func_impl_=*/gFFTNFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

FFTShiftFuncImpl gFFTShiftFuncImpl;
OpDef gFFTShift = {
  /*.name_=*/"FFTShift",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dim", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_INT, DT_LIST_INT}, /*.is_optional_=*/true},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"dim", 1},
  },
  /*.func_impl_=*/gFFTShiftFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

FillScalarFuncImpl gFillScalarFuncImpl;
OpDef gFillScalar = {
  /*.name_=*/"FillScalar",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"size", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT, DT_TENSOR}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"fill_value", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dtype", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"dtype_to_type_id", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"size", 0},
    {"fill_value", 1},
    {"dtype", 2},
  },
  /*.func_impl_=*/gFillScalarFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

FillTensorFuncImpl gFillTensorFuncImpl;
OpDef gFillTensor = {
  /*.name_=*/"FillTensor",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"size", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT, DT_TENSOR}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"fill_value", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dtype", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"dtype_to_type_id", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"size", 0},
    {"fill_value", 1},
    {"dtype", 2},
  },
  /*.func_impl_=*/gFillTensorFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

FlashAttentionScoreGradFuncImpl gFlashAttentionScoreGradFuncImpl;
OpDef gFlashAttentionScoreGrad = {
  /*.name_=*/"FlashAttentionScoreGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"query", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"key", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"value", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dy", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"pse_shift", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"drop_mask", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"padding_mask", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"atten_mask", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"softmax_max", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"softmax_sum", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"softmax_in", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"attention_in", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"prefix", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT, DT_TENSOR}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"actual_seq_qlen", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT, DT_TENSOR}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"actual_seq_kvlen", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT, DT_TENSOR}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"head_num", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"keep_prob", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"scale_value", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"pre_tokens", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"next_tokens", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"inner_precise", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"input_layout", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"str_to_enum", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"sparse_mode", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"dq", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"dk", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"dv", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"dpse", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"query", 0},
    {"key", 1},
    {"value", 2},
    {"dy", 3},
    {"pse_shift", 4},
    {"drop_mask", 5},
    {"padding_mask", 6},
    {"atten_mask", 7},
    {"softmax_max", 8},
    {"softmax_sum", 9},
    {"softmax_in", 10},
    {"attention_in", 11},
    {"prefix", 12},
    {"actual_seq_qlen", 13},
    {"actual_seq_kvlen", 14},
    {"head_num", 15},
    {"keep_prob", 16},
    {"scale_value", 17},
    {"pre_tokens", 18},
    {"next_tokens", 19},
    {"inner_precise", 20},
    {"input_layout", 21},
    {"sparse_mode", 22},
  },
  /*.func_impl_=*/gFlashAttentionScoreGradFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

FlashAttentionScoreFuncImpl gFlashAttentionScoreFuncImpl;
OpDef gFlashAttentionScore = {
  /*.name_=*/"FlashAttentionScore",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"query", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"key", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"value", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"real_shift", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"drop_mask", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"padding_mask", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"attn_mask", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"prefix", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT, DT_TENSOR}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"actual_seq_qlen", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT, DT_TENSOR}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"actual_seq_kvlen", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT, DT_TENSOR}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"head_num", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"keep_prob", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"scale_value", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"pre_tokens", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"next_tokens", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"inner_precise", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"input_layout", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"str_to_enum", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"sparse_mode", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"softmax_max", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"softmax_sum", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"softmax_out", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"attention_out", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"query", 0},
    {"key", 1},
    {"value", 2},
    {"real_shift", 3},
    {"drop_mask", 4},
    {"padding_mask", 5},
    {"attn_mask", 6},
    {"prefix", 7},
    {"actual_seq_qlen", 8},
    {"actual_seq_kvlen", 9},
    {"head_num", 10},
    {"keep_prob", 11},
    {"scale_value", 12},
    {"pre_tokens", 13},
    {"next_tokens", 14},
    {"inner_precise", 15},
    {"input_layout", 16},
    {"sparse_mode", 17},
  },
  /*.func_impl_=*/gFlashAttentionScoreFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

FlattenExtFuncImpl gFlattenExtFuncImpl;
OpDef gFlattenExt = {
  /*.name_=*/"FlattenExt",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"start_dim", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"end_dim", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"start_dim", 1},
    {"end_dim", 2},
  },
  /*.func_impl_=*/gFlattenExtFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

FlattenFuncImpl gFlattenFuncImpl;
OpDef gFlatten = {
  /*.name_=*/"Flatten",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input_x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input_x", 0},
  },
  /*.func_impl_=*/gFlattenFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

FloorDivFuncImpl gFloorDivFuncImpl;
OpDef gFloorDiv = {
  /*.name_=*/"FloorDiv",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"other", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {
    Signature("input", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
     Signature("other", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"other", 1},
  },
  /*.func_impl_=*/gFloorDivFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

FloorModFuncImpl gFloorModFuncImpl;
OpDef gFloorMod = {
  /*.name_=*/"FloorMod",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {
    Signature("x", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
     Signature("y", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"y", 1},
  },
  /*.func_impl_=*/gFloorModFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

FloorFuncImpl gFloorFuncImpl;
OpDef gFloor = {
  /*.name_=*/"Floor",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
  },
  /*.func_impl_=*/gFloorFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

GatherDGradV2FuncImpl gGatherDGradV2FuncImpl;
OpDef gGatherDGradV2 = {
  /*.name_=*/"GatherDGradV2",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dim", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"index", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dout", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"dx", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"dim", 1},
    {"index", 2},
    {"dout", 3},
  },
  /*.func_impl_=*/gGatherDGradV2FuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

GatherDFuncImpl gGatherDFuncImpl;
OpDef gGatherD = {
  /*.name_=*/"GatherD",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dim", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_TENSOR}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"index", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"dim", 1},
    {"index", 2},
  },
  /*.func_impl_=*/gGatherDFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

GatherNdFuncImpl gGatherNdFuncImpl;
OpDef gGatherNd = {
  /*.name_=*/"GatherNd",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input_x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"indices", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input_x", 0},
    {"indices", 1},
  },
  /*.func_impl_=*/gGatherNdFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

GatherFuncImpl gGatherFuncImpl;
OpDef gGather = {
  /*.name_=*/"Gather",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input_params", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"input_indices", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"axis", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_TENSOR}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"batch_dims", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input_params", 0},
    {"input_indices", 1},
    {"axis", 2},
    {"batch_dims", 3},
  },
  /*.func_impl_=*/gGatherFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

GcdFuncImpl gGcdFuncImpl;
OpDef gGcd = {
  /*.name_=*/"Gcd",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"other", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {
    Signature("input", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
     Signature("other", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"other", 1},
  },
  /*.func_impl_=*/gGcdFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

GeLUGradFuncImpl gGeLUGradFuncImpl;
OpDef gGeLUGrad = {
  /*.name_=*/"GeLUGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"dy", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"dy", 0},
    {"x", 1},
    {"y", 2},
  },
  /*.func_impl_=*/gGeLUGradFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

GeLUFuncImpl gGeLUFuncImpl;
OpDef gGeLU = {
  /*.name_=*/"GeLU",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
  },
  /*.func_impl_=*/gGeLUFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

GeneratorFuncImpl gGeneratorFuncImpl;
OpDef gGenerator = {
  /*.name_=*/"Generator",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"cmd", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"inputs", /*.arg_dtype_=*/DT_TUPLE_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"seed", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"offset", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"state", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"cmd", 0},
    {"inputs", 1},
  },
  /*.func_impl_=*/gGeneratorFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

GeqrfFuncImpl gGeqrfFuncImpl;
OpDef gGeqrf = {
  /*.name_=*/"Geqrf",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"tau", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
  },
  /*.func_impl_=*/gGeqrfFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

GreaterEqualFuncImpl gGreaterEqualFuncImpl;
OpDef gGreaterEqual = {
  /*.name_=*/"GreaterEqual",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"other", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {
    Signature("input", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
     Signature("other", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"other", 1},
  },
  /*.func_impl_=*/gGreaterEqualFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

GreaterFuncImpl gGreaterFuncImpl;
OpDef gGreater = {
  /*.name_=*/"Greater",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"other", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {
    Signature("input", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
     Signature("other", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"other", 1},
  },
  /*.func_impl_=*/gGreaterFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

GridSampler2DGradFuncImpl gGridSampler2DGradFuncImpl;
OpDef gGridSampler2DGrad = {
  /*.name_=*/"GridSampler2DGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"grad", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"input_x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"grid", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"interpolation_mode", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"str_to_enum", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"padding_mode", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"str_to_enum", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"align_corners", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"dx", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"dgrid", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"grad", 0},
    {"input_x", 1},
    {"grid", 2},
    {"interpolation_mode", 3},
    {"padding_mode", 4},
    {"align_corners", 5},
  },
  /*.func_impl_=*/gGridSampler2DGradFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

GridSampler2DFuncImpl gGridSampler2DFuncImpl;
OpDef gGridSampler2D = {
  /*.name_=*/"GridSampler2D",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input_x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"grid", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"interpolation_mode", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"str_to_enum", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"padding_mode", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"str_to_enum", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"align_corners", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input_x", 0},
    {"grid", 1},
    {"interpolation_mode", 2},
    {"padding_mode", 3},
    {"align_corners", 4},
  },
  /*.func_impl_=*/gGridSampler2DFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

GridSampler3DGradFuncImpl gGridSampler3DGradFuncImpl;
OpDef gGridSampler3DGrad = {
  /*.name_=*/"GridSampler3DGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"grad", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"input_x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"grid", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"interpolation_mode", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"str_to_enum", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"padding_mode", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"str_to_enum", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"align_corners", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"dx", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"dgrid", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"grad", 0},
    {"input_x", 1},
    {"grid", 2},
    {"interpolation_mode", 3},
    {"padding_mode", 4},
    {"align_corners", 5},
  },
  /*.func_impl_=*/gGridSampler3DGradFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

GridSampler3DFuncImpl gGridSampler3DFuncImpl;
OpDef gGridSampler3D = {
  /*.name_=*/"GridSampler3D",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input_x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"grid", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"interpolation_mode", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"str_to_enum", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"padding_mode", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"str_to_enum", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"align_corners", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input_x", 0},
    {"grid", 1},
    {"interpolation_mode", 2},
    {"padding_mode", 3},
    {"align_corners", 4},
  },
  /*.func_impl_=*/gGridSampler3DFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

GroupNormGradFuncImpl gGroupNormGradFuncImpl;
OpDef gGroupNormGrad = {
  /*.name_=*/"GroupNormGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"dy", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"mean", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"rstd", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"gamma_opt", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"num_groups", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dx_is_require", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dgamma_is_require", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dbeta_is_require", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"pd_x", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"pd_gamma", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"pd_beta", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"dy", 0},
    {"x", 1},
    {"mean", 2},
    {"rstd", 3},
    {"gamma_opt", 4},
    {"num_groups", 5},
    {"dx_is_require", 6},
    {"dgamma_is_require", 7},
    {"dbeta_is_require", 8},
  },
  /*.func_impl_=*/gGroupNormGradFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

GroupNormFuncImpl gGroupNormFuncImpl;
OpDef gGroupNorm = {
  /*.name_=*/"GroupNorm",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"num_groups", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"weight", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"bias", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"eps", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"out", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"meanOut", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"rstdOut", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"num_groups", 1},
    {"weight", 2},
    {"bias", 3},
    {"eps", 4},
  },
  /*.func_impl_=*/gGroupNormFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

HShrinkGradFuncImpl gHShrinkGradFuncImpl;
OpDef gHShrinkGrad = {
  /*.name_=*/"HShrinkGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"gradients", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"features", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"lambd", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"gradients", 0},
    {"features", 1},
    {"lambd", 2},
  },
  /*.func_impl_=*/gHShrinkGradFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

HShrinkFuncImpl gHShrinkFuncImpl;
OpDef gHShrink = {
  /*.name_=*/"HShrink",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"lambd", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"lambd", 1},
  },
  /*.func_impl_=*/gHShrinkFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

HSigmoidGradFuncImpl gHSigmoidGradFuncImpl;
OpDef gHSigmoidGrad = {
  /*.name_=*/"HSigmoidGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"grads", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"input_x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"grads", 0},
    {"input_x", 1},
  },
  /*.func_impl_=*/gHSigmoidGradFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

HSigmoidFuncImpl gHSigmoidFuncImpl;
OpDef gHSigmoid = {
  /*.name_=*/"HSigmoid",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input_x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input_x", 0},
  },
  /*.func_impl_=*/gHSigmoidFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

HSwishGradFuncImpl gHSwishGradFuncImpl;
OpDef gHSwishGrad = {
  /*.name_=*/"HSwishGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"y_grad", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"y_grad", 0},
    {"x", 1},
  },
  /*.func_impl_=*/gHSwishGradFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

HSwishFuncImpl gHSwishFuncImpl;
OpDef gHSwish = {
  /*.name_=*/"HSwish",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
  },
  /*.func_impl_=*/gHSwishFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

IdentityFuncImpl gIdentityFuncImpl;
OpDef gIdentity = {
  /*.name_=*/"Identity",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input_x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input_x", 0},
  },
  /*.func_impl_=*/gIdentityFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

IFFT2FuncImpl gIFFT2FuncImpl;
OpDef gIFFT2 = {
  /*.name_=*/"IFFT2",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"s", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"dim", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"norm", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"str_to_enum", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"s", 1},
    {"dim", 2},
    {"norm", 3},
  },
  /*.func_impl_=*/gIFFT2FuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

IFFTFuncImpl gIFFTFuncImpl;
OpDef gIFFT = {
  /*.name_=*/"IFFT",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"n", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"dim", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"norm", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"str_to_enum", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"n", 1},
    {"dim", 2},
    {"norm", 3},
  },
  /*.func_impl_=*/gIFFTFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

IFFTNFuncImpl gIFFTNFuncImpl;
OpDef gIFFTN = {
  /*.name_=*/"IFFTN",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"s", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"dim", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"norm", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"str_to_enum", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"s", 1},
    {"dim", 2},
    {"norm", 3},
  },
  /*.func_impl_=*/gIFFTNFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

IFFTShiftFuncImpl gIFFTShiftFuncImpl;
OpDef gIFFTShift = {
  /*.name_=*/"IFFTShift",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dim", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_INT, DT_LIST_INT}, /*.is_optional_=*/true},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"dim", 1},
  },
  /*.func_impl_=*/gIFFTShiftFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

Im2ColExtFuncImpl gIm2ColExtFuncImpl;
OpDef gIm2ColExt = {
  /*.name_=*/"Im2ColExt",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"kernel_size", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"to_pair", /*.cast_dtype_ =*/{DT_LIST_INT}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dilation", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"to_pair", /*.cast_dtype_ =*/{DT_LIST_INT}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"padding", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"to_pair", /*.cast_dtype_ =*/{DT_LIST_INT}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"stride", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"to_pair", /*.cast_dtype_ =*/{DT_LIST_INT}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"kernel_size", 1},
    {"dilation", 2},
    {"padding", 3},
    {"stride", 4},
  },
  /*.func_impl_=*/gIm2ColExtFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

IndexAddExtFuncImpl gIndexAddExtFuncImpl;
OpDef gIndexAddExt = {
  /*.name_=*/"IndexAddExt",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"index", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"source", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"axis", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"alpha", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {
    Signature("input", SignatureEnumRW::kRWWrite,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
     Signature("index", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType1),
     Signature("source", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
     Signature("axis", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType2),
     Signature("alpha", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType3),
  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"index", 1},
    {"source", 2},
    {"axis", 3},
    {"alpha", 4},
  },
  /*.func_impl_=*/gIndexAddExtFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

IndexSelectFuncImpl gIndexSelectFuncImpl;
OpDef gIndexSelect = {
  /*.name_=*/"IndexSelect",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dim", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"index", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"dim", 1},
    {"index", 2},
  },
  /*.func_impl_=*/gIndexSelectFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

IRFFTGradFuncImpl gIRFFTGradFuncImpl;
OpDef gIRFFTGrad = {
  /*.name_=*/"IRFFTGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input1", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"input2", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"n", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"dim", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"norm", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"str_to_enum", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input1", 0},
    {"input2", 1},
    {"n", 2},
    {"dim", 3},
    {"norm", 4},
  },
  /*.func_impl_=*/gIRFFTGradFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

IRFFTFuncImpl gIRFFTFuncImpl;
OpDef gIRFFT = {
  /*.name_=*/"IRFFT",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"n", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"dim", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"norm", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"str_to_enum", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"n", 1},
    {"dim", 2},
    {"norm", 3},
  },
  /*.func_impl_=*/gIRFFTFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

IsCloseFuncImpl gIsCloseFuncImpl;
OpDef gIsClose = {
  /*.name_=*/"IsClose",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"other", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"rtol", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_BOOL, DT_INT}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"atol", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_BOOL, DT_INT}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"equal_nan", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"other", 1},
    {"rtol", 2},
    {"atol", 3},
    {"equal_nan", 4},
  },
  /*.func_impl_=*/gIsCloseFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

IsFiniteFuncImpl gIsFiniteFuncImpl;
OpDef gIsFinite = {
  /*.name_=*/"IsFinite",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
  },
  /*.func_impl_=*/gIsFiniteFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

LayerNormExtFuncImpl gLayerNormExtFuncImpl;
OpDef gLayerNormExt = {
  /*.name_=*/"LayerNormExt",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"normalized_shape", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_INT, DT_LIST_INT}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"weight", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"bias", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"eps", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output_x", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"mean", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"rstd", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"normalized_shape", 1},
    {"weight", 2},
    {"bias", 3},
    {"eps", 4},
  },
  /*.func_impl_=*/gLayerNormExtFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

LayerNormGradExtFuncImpl gLayerNormGradExtFuncImpl;
OpDef gLayerNormGradExt = {
  /*.name_=*/"LayerNormGradExt",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"dy", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"normalized_shape", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_INT, DT_LIST_INT}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"mean", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"variance", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"gamma", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"beta", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"pd_x", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"pd_gamma", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"pd_beta", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"dy", 0},
    {"x", 1},
    {"normalized_shape", 2},
    {"mean", 3},
    {"variance", 4},
    {"gamma", 5},
    {"beta", 6},
  },
  /*.func_impl_=*/gLayerNormGradExtFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

LayerNormGradGradFuncImpl gLayerNormGradGradFuncImpl;
OpDef gLayerNormGradGrad = {
  /*.name_=*/"LayerNormGradGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dy", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"variance", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"mean", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"gamma", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"d_dx", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"d_dg", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"d_db", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"begin_norm_axis", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"begin_params_axis", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"sopd_x", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"sopd_dy", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"sopd_gamma", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"dy", 1},
    {"variance", 2},
    {"mean", 3},
    {"gamma", 4},
    {"d_dx", 5},
    {"d_dg", 6},
    {"d_db", 7},
    {"begin_norm_axis", 8},
    {"begin_params_axis", 9},
  },
  /*.func_impl_=*/gLayerNormGradGradFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

LayerNormGradFuncImpl gLayerNormGradFuncImpl;
OpDef gLayerNormGrad = {
  /*.name_=*/"LayerNormGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dy", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"variance", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"mean", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"gamma", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"begin_norm_axis", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"begin_params_axis", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"pd_x", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"pd_gamma", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"pd_beta", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"dy", 1},
    {"variance", 2},
    {"mean", 3},
    {"gamma", 4},
    {"begin_norm_axis", 5},
    {"begin_params_axis", 6},
  },
  /*.func_impl_=*/gLayerNormGradFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

LayerNormGradV3FuncImpl gLayerNormGradV3FuncImpl;
OpDef gLayerNormGradV3 = {
  /*.name_=*/"LayerNormGradV3",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dy", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"variance", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"mean", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"gamma", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"begin_norm_axis", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"begin_params_axis", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"pd_x", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"pd_gamma", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"pd_beta", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"dy", 1},
    {"variance", 2},
    {"mean", 3},
    {"gamma", 4},
    {"begin_norm_axis", 5},
    {"begin_params_axis", 6},
  },
  /*.func_impl_=*/gLayerNormGradV3FuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

LayerNormFuncImpl gLayerNormFuncImpl;
OpDef gLayerNorm = {
  /*.name_=*/"LayerNorm",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input_x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"gamma", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"beta", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"begin_norm_axis", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"begin_params_axis", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"epsilon", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output_x", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"mean", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"variance", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input_x", 0},
    {"gamma", 1},
    {"beta", 2},
    {"begin_norm_axis", 3},
    {"begin_params_axis", 4},
    {"epsilon", 5},
  },
  /*.func_impl_=*/gLayerNormFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

LayerNormV3FuncImpl gLayerNormV3FuncImpl;
OpDef gLayerNormV3 = {
  /*.name_=*/"LayerNormV3",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input_x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"gamma", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"beta", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"begin_norm_axis", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"begin_params_axis", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"epsilon", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output_x", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"mean", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"variance", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input_x", 0},
    {"gamma", 1},
    {"beta", 2},
    {"begin_norm_axis", 3},
    {"begin_params_axis", 4},
    {"epsilon", 5},
  },
  /*.func_impl_=*/gLayerNormV3FuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

LeakyReLUExtFuncImpl gLeakyReLUExtFuncImpl;
OpDef gLeakyReLUExt = {
  /*.name_=*/"LeakyReLUExt",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"negative_slope", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"negative_slope", 1},
  },
  /*.func_impl_=*/gLeakyReLUExtFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

LeakyReLUGradExtFuncImpl gLeakyReLUGradExtFuncImpl;
OpDef gLeakyReLUGradExt = {
  /*.name_=*/"LeakyReLUGradExt",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"dy", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"negative_slope", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"is_result", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"dx", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"dy", 0},
    {"input", 1},
    {"negative_slope", 2},
    {"is_result", 3},
  },
  /*.func_impl_=*/gLeakyReLUGradExtFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

LessEqualFuncImpl gLessEqualFuncImpl;
OpDef gLessEqual = {
  /*.name_=*/"LessEqual",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER, DT_BOOL}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"other", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER, DT_BOOL}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {
    Signature("input", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
     Signature("other", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"other", 1},
  },
  /*.func_impl_=*/gLessEqualFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

LessFuncImpl gLessFuncImpl;
OpDef gLess = {
  /*.name_=*/"Less",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER, DT_BOOL}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"other", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER, DT_BOOL}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {
    Signature("input", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
     Signature("other", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"other", 1},
  },
  /*.func_impl_=*/gLessFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

LinSpaceExtFuncImpl gLinSpaceExtFuncImpl;
OpDef gLinSpaceExt = {
  /*.name_=*/"LinSpaceExt",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"start", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_TENSOR}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"end", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_TENSOR}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"steps", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_TENSOR}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dtype", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"dtype_to_type_id", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"start", 0},
    {"end", 1},
    {"steps", 2},
    {"dtype", 3},
  },
  /*.func_impl_=*/gLinSpaceExtFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

LinSpaceFuncImpl gLinSpaceFuncImpl;
OpDef gLinSpace = {
  /*.name_=*/"LinSpace",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"start", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"stop", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"num", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_TENSOR}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"start", 0},
    {"stop", 1},
    {"num", 2},
  },
  /*.func_impl_=*/gLinSpaceFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ListToTupleFuncImpl gListToTupleFuncImpl;
OpDef gListToTuple = {
  /*.name_=*/"ListToTuple",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_LIST_ANY, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TUPLE_ANY,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
  },
  /*.func_impl_=*/gListToTupleFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

Log1pFuncImpl gLog1pFuncImpl;
OpDef gLog1p = {
  /*.name_=*/"Log1p",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
  },
  /*.func_impl_=*/gLog1pFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

LogMatrixDeterminantFuncImpl gLogMatrixDeterminantFuncImpl;
OpDef gLogMatrixDeterminant = {
  /*.name_=*/"LogMatrixDeterminant",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"sign", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
  },
  /*.func_impl_=*/gLogMatrixDeterminantFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

LogFuncImpl gLogFuncImpl;
OpDef gLog = {
  /*.name_=*/"Log",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
  },
  /*.func_impl_=*/gLogFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

LogSoftmaxGradFuncImpl gLogSoftmaxGradFuncImpl;
OpDef gLogSoftmaxGrad = {
  /*.name_=*/"LogSoftmaxGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"logits", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"grad", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"axis", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"logits_grad", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"logits", 0},
    {"grad", 1},
    {"axis", 2},
  },
  /*.func_impl_=*/gLogSoftmaxGradFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

LogSoftmaxFuncImpl gLogSoftmaxFuncImpl;
OpDef gLogSoftmax = {
  /*.name_=*/"LogSoftmax",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"logits", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"axis", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"logits", 0},
    {"axis", 1},
  },
  /*.func_impl_=*/gLogSoftmaxFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

LogicalAndFuncImpl gLogicalAndFuncImpl;
OpDef gLogicalAnd = {
  /*.name_=*/"LogicalAnd",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_BOOL}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_BOOL}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {
    Signature("x", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
     Signature("y", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"y", 1},
  },
  /*.func_impl_=*/gLogicalAndFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

LogicalNotFuncImpl gLogicalNotFuncImpl;
OpDef gLogicalNot = {
  /*.name_=*/"LogicalNot",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
  },
  /*.func_impl_=*/gLogicalNotFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

LogicalOrFuncImpl gLogicalOrFuncImpl;
OpDef gLogicalOr = {
  /*.name_=*/"LogicalOr",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_BOOL}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_BOOL}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {
    Signature("x", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
     Signature("y", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"y", 1},
  },
  /*.func_impl_=*/gLogicalOrFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

LogicalXorFuncImpl gLogicalXorFuncImpl;
OpDef gLogicalXor = {
  /*.name_=*/"LogicalXor",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_BOOL}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"other", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_BOOL}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {
    Signature("input", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
     Signature("other", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"other", 1},
  },
  /*.func_impl_=*/gLogicalXorFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

LogitGradFuncImpl gLogitGradFuncImpl;
OpDef gLogitGrad = {
  /*.name_=*/"LogitGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"grad", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"eps", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"input_grad", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"grad", 0},
    {"input", 1},
    {"eps", 2},
  },
  /*.func_impl_=*/gLogitGradFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

LogitFuncImpl gLogitFuncImpl;
OpDef gLogit = {
  /*.name_=*/"Logit",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"eps", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"eps", 1},
  },
  /*.func_impl_=*/gLogitFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

MaskedFillFuncImpl gMaskedFillFuncImpl;
OpDef gMaskedFill = {
  /*.name_=*/"MaskedFill",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input_x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"mask", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"value", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input_x", 0},
    {"mask", 1},
    {"value", 2},
  },
  /*.func_impl_=*/gMaskedFillFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

MatMulExtFuncImpl gMatMulExtFuncImpl;
OpDef gMatMulExt = {
  /*.name_=*/"MatMulExt",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"mat2", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"mat2", 1},
  },
  /*.func_impl_=*/gMatMulExtFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

MatMulFuncImpl gMatMulFuncImpl;
OpDef gMatMul = {
  /*.name_=*/"MatMul",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"mat2", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"transpose_a", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"transpose_b", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"mat2", 1},
    {"transpose_a", 2},
    {"transpose_b", 3},
  },
  /*.func_impl_=*/gMatMulFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

MatrixDeterminantFuncImpl gMatrixDeterminantFuncImpl;
OpDef gMatrixDeterminant = {
  /*.name_=*/"MatrixDeterminant",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
  },
  /*.func_impl_=*/gMatrixDeterminantFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

MatrixExpFuncImpl gMatrixExpFuncImpl;
OpDef gMatrixExp = {
  /*.name_=*/"MatrixExp",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
  },
  /*.func_impl_=*/gMatrixExpFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

MatrixInverseExtFuncImpl gMatrixInverseExtFuncImpl;
OpDef gMatrixInverseExt = {
  /*.name_=*/"MatrixInverseExt",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
  },
  /*.func_impl_=*/gMatrixInverseExtFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

MaxFuncImpl gMaxFuncImpl;
OpDef gMax = {
  /*.name_=*/"Max",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"out", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
  },
  /*.func_impl_=*/gMaxFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

MaxPoolGradWithIndicesFuncImpl gMaxPoolGradWithIndicesFuncImpl;
OpDef gMaxPoolGradWithIndices = {
  /*.name_=*/"MaxPoolGradWithIndices",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"grad", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"argmax", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"kernel_size", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"to_kernel_size", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"strides", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"to_strides", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"pads", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"to_output_padding", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dilation", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"to_dilations", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"ceil_mode", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"argmax_type", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"dtype_to_type_id", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"grad", 1},
    {"argmax", 2},
    {"kernel_size", 3},
    {"strides", 4},
    {"pads", 5},
    {"dilation", 6},
    {"ceil_mode", 7},
    {"argmax_type", 8},
  },
  /*.func_impl_=*/gMaxPoolGradWithIndicesFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

MaxPoolGradWithMaskFuncImpl gMaxPoolGradWithMaskFuncImpl;
OpDef gMaxPoolGradWithMask = {
  /*.name_=*/"MaxPoolGradWithMask",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"grad", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"mask", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"kernel_size", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"to_kernel_size", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"strides", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"to_strides", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"pads", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"to_output_padding", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dilation", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"to_dilations", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"ceil_mode", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"argmax_type", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"dtype_to_type_id", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"grad", 1},
    {"mask", 2},
    {"kernel_size", 3},
    {"strides", 4},
    {"pads", 5},
    {"dilation", 6},
    {"ceil_mode", 7},
    {"argmax_type", 8},
  },
  /*.func_impl_=*/gMaxPoolGradWithMaskFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

MaxPoolWithIndicesFuncImpl gMaxPoolWithIndicesFuncImpl;
OpDef gMaxPoolWithIndices = {
  /*.name_=*/"MaxPoolWithIndices",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"kernel_size", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"to_kernel_size", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"strides", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"to_strides", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"pads", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"to_output_padding", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dilation", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"to_dilations", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"ceil_mode", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"argmax_type", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"dtype_to_type_id", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"argmax", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"kernel_size", 1},
    {"strides", 2},
    {"pads", 3},
    {"dilation", 4},
    {"ceil_mode", 5},
    {"argmax_type", 6},
  },
  /*.func_impl_=*/gMaxPoolWithIndicesFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

MaxPoolWithMaskFuncImpl gMaxPoolWithMaskFuncImpl;
OpDef gMaxPoolWithMask = {
  /*.name_=*/"MaxPoolWithMask",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"kernel_size", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"to_kernel_size", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"strides", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"to_strides", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"pads", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"to_output_padding", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dilation", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"to_dilations", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"ceil_mode", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"argmax_type", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"dtype_to_type_id", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"mask", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"kernel_size", 1},
    {"strides", 2},
    {"pads", 3},
    {"dilation", 4},
    {"ceil_mode", 5},
    {"argmax_type", 6},
  },
  /*.func_impl_=*/gMaxPoolWithMaskFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

MaximumGradGradFuncImpl gMaximumGradGradFuncImpl;
OpDef gMaximumGradGrad = {
  /*.name_=*/"MaximumGradGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dx", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dy", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"grad_x", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"grad_y", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"sopd_x", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"sopd_y", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"sopd_grad", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"y", 1},
    {"dx", 2},
    {"dy", 3},
    {"grad_x", 4},
    {"grad_y", 5},
  },
  /*.func_impl_=*/gMaximumGradGradFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

MaximumGradFuncImpl gMaximumGradFuncImpl;
OpDef gMaximumGrad = {
  /*.name_=*/"MaximumGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"grads", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"grad_x", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"grad_y", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"dx", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"dy", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"y", 1},
    {"grads", 2},
    {"grad_x", 3},
    {"grad_y", 4},
  },
  /*.func_impl_=*/gMaximumGradFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

MaximumFuncImpl gMaximumFuncImpl;
OpDef gMaximum = {
  /*.name_=*/"Maximum",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"other", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {
    Signature("input", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
     Signature("other", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"other", 1},
  },
  /*.func_impl_=*/gMaximumFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

MeanExtFuncImpl gMeanExtFuncImpl;
OpDef gMeanExt = {
  /*.name_=*/"MeanExt",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"axis", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_INT, DT_LIST_INT, DT_TENSOR}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"keep_dims", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dtype", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"dtype_to_type_id", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"axis", 1},
    {"keep_dims", 2},
    {"dtype", 3},
  },
  /*.func_impl_=*/gMeanExtFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

MinFuncImpl gMinFuncImpl;
OpDef gMin = {
  /*.name_=*/"Min",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"out", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
  },
  /*.func_impl_=*/gMinFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

MinimumGradFuncImpl gMinimumGradFuncImpl;
OpDef gMinimumGrad = {
  /*.name_=*/"MinimumGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x1", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"x2", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"grads", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"grad_x", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"grad_y", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"y1", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"y2", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x1", 0},
    {"x2", 1},
    {"grads", 2},
    {"grad_x", 3},
    {"grad_y", 4},
  },
  /*.func_impl_=*/gMinimumGradFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

MinimumFuncImpl gMinimumFuncImpl;
OpDef gMinimum = {
  /*.name_=*/"Minimum",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"other", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {
    Signature("input", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
     Signature("other", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"other", 1},
  },
  /*.func_impl_=*/gMinimumFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

MulFuncImpl gMulFuncImpl;
OpDef gMul = {
  /*.name_=*/"Mul",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"other", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {
    Signature("input", SignatureEnumRW::kRWRead,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
     Signature("other", SignatureEnumRW::kRWRead,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"other", 1},
  },
  /*.func_impl_=*/gMulFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

MvFuncImpl gMvFuncImpl;
OpDef gMv = {
  /*.name_=*/"Mv",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"vec", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"vec", 1},
  },
  /*.func_impl_=*/gMvFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

NanToNumFuncImpl gNanToNumFuncImpl;
OpDef gNanToNum = {
  /*.name_=*/"NanToNum",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"nan", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"posinf", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"neginf", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"nan", 1},
    {"posinf", 2},
    {"neginf", 3},
  },
  /*.func_impl_=*/gNanToNumFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

NegFuncImpl gNegFuncImpl;
OpDef gNeg = {
  /*.name_=*/"Neg",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
  },
  /*.func_impl_=*/gNegFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

NextAfterFuncImpl gNextAfterFuncImpl;
OpDef gNextAfter = {
  /*.name_=*/"NextAfter",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"other", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"other", 1},
  },
  /*.func_impl_=*/gNextAfterFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

NLLLossGradFuncImpl gNLLLossGradFuncImpl;
OpDef gNLLLossGrad = {
  /*.name_=*/"NLLLossGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"logits", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"loss_grad", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"labels", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"weight", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"total_weight", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"reduction", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"str_to_enum", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"ignore_index", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"logits_grad", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"logits", 0},
    {"loss_grad", 1},
    {"labels", 2},
    {"weight", 3},
    {"total_weight", 4},
    {"reduction", 5},
    {"ignore_index", 6},
  },
  /*.func_impl_=*/gNLLLossGradFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

NLLLossFuncImpl gNLLLossFuncImpl;
OpDef gNLLLoss = {
  /*.name_=*/"NLLLoss",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"logits", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"labels", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"weight", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"reduction", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"str_to_enum", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"ignore_index", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"loss", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"total_weight", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"logits", 0},
    {"labels", 1},
    {"weight", 2},
    {"reduction", 3},
    {"ignore_index", 4},
  },
  /*.func_impl_=*/gNLLLossFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

NonZeroExtFuncImpl gNonZeroExtFuncImpl;
OpDef gNonZeroExt = {
  /*.name_=*/"NonZeroExt",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TUPLE_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
  },
  /*.func_impl_=*/gNonZeroExtFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

NonZeroFuncImpl gNonZeroFuncImpl;
OpDef gNonZero = {
  /*.name_=*/"NonZero",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
  },
  /*.func_impl_=*/gNonZeroFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

NormFuncImpl gNormFuncImpl;
OpDef gNorm = {
  /*.name_=*/"Norm",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input_x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"ord", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_TENSOR}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"dim", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_INT, DT_LIST_INT}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"keepdim", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dtype", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"dtype_to_type_id", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input_x", 0},
    {"ord", 1},
    {"dim", 2},
    {"keepdim", 3},
    {"dtype", 4},
  },
  /*.func_impl_=*/gNormFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

NormalFloatFloatFuncImpl gNormalFloatFloatFuncImpl;
OpDef gNormalFloatFloat = {
  /*.name_=*/"NormalFloatFloat",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"mean", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"std", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"size", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"seed", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"offset", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"mean", 0},
    {"std", 1},
    {"size", 2},
    {"seed", 3},
    {"offset", 4},
  },
  /*.func_impl_=*/gNormalFloatFloatFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

NormalFloatTensorFuncImpl gNormalFloatTensorFuncImpl;
OpDef gNormalFloatTensor = {
  /*.name_=*/"NormalFloatTensor",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"mean", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"std", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"seed", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"offset", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"mean", 0},
    {"std", 1},
    {"seed", 2},
    {"offset", 3},
  },
  /*.func_impl_=*/gNormalFloatTensorFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

NormalTensorFloatFuncImpl gNormalTensorFloatFuncImpl;
OpDef gNormalTensorFloat = {
  /*.name_=*/"NormalTensorFloat",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"mean", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"std", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"seed", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"offset", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"mean", 0},
    {"std", 1},
    {"seed", 2},
    {"offset", 3},
  },
  /*.func_impl_=*/gNormalTensorFloatFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

NormalTensorTensorFuncImpl gNormalTensorTensorFuncImpl;
OpDef gNormalTensorTensor = {
  /*.name_=*/"NormalTensorTensor",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"mean", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"std", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"seed", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"offset", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"mean", 0},
    {"std", 1},
    {"seed", 2},
    {"offset", 3},
  },
  /*.func_impl_=*/gNormalTensorTensorFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

NotEqualFuncImpl gNotEqualFuncImpl;
OpDef gNotEqual = {
  /*.name_=*/"NotEqual",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"other", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {
    Signature("input", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
     Signature("other", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"other", 1},
  },
  /*.func_impl_=*/gNotEqualFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

NPUClearFloatStatusV2FuncImpl gNPUClearFloatStatusV2FuncImpl;
OpDef gNPUClearFloatStatusV2 = {
  /*.name_=*/"NPUClearFloatStatusV2",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
  },
  /*.func_impl_=*/gNPUClearFloatStatusV2FuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

NPUGetFloatStatusV2FuncImpl gNPUGetFloatStatusV2FuncImpl;
OpDef gNPUGetFloatStatusV2 = {
  /*.name_=*/"NPUGetFloatStatusV2",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
  },
  /*.func_impl_=*/gNPUGetFloatStatusV2FuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

OneHotExtFuncImpl gOneHotExtFuncImpl;
OpDef gOneHotExt = {
  /*.name_=*/"OneHotExt",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"tensor", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"num_classes", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"on_value", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"off_value", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"axis", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"tensor", 0},
    {"num_classes", 1},
    {"on_value", 2},
    {"off_value", 3},
    {"axis", 4},
  },
  /*.func_impl_=*/gOneHotExtFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

OneHotFuncImpl gOneHotFuncImpl;
OpDef gOneHot = {
  /*.name_=*/"OneHot",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"indices", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"depth", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_TENSOR}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"on_value", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"off_value", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"axis", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"indices", 0},
    {"depth", 1},
    {"on_value", 2},
    {"off_value", 3},
    {"axis", 4},
  },
  /*.func_impl_=*/gOneHotFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

OnesLikeExtFuncImpl gOnesLikeExtFuncImpl;
OpDef gOnesLikeExt = {
  /*.name_=*/"OnesLikeExt",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dtype", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"dtype_to_type_id", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"dtype", 1},
  },
  /*.func_impl_=*/gOnesLikeExtFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

OnesLikeFuncImpl gOnesLikeFuncImpl;
OpDef gOnesLike = {
  /*.name_=*/"OnesLike",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
  },
  /*.func_impl_=*/gOnesLikeFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

OnesFuncImpl gOnesFuncImpl;
OpDef gOnes = {
  /*.name_=*/"Ones",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"shape", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_INT, DT_LIST_INT, DT_TENSOR}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dtype", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"dtype_to_type_id", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"shape", 0},
    {"dtype", 1},
  },
  /*.func_impl_=*/gOnesFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

PagedAttentionMaskFuncImpl gPagedAttentionMaskFuncImpl;
OpDef gPagedAttentionMask = {
  /*.name_=*/"PagedAttentionMask",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"query", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"key_cache", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"value_cache", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"block_tables", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"context_lens", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"alibi_mask", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"head_num", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"scale_value", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"kv_head_num", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"attention_out", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"query", 0},
    {"key_cache", 1},
    {"value_cache", 2},
    {"block_tables", 3},
    {"context_lens", 4},
    {"alibi_mask", 5},
    {"head_num", 6},
    {"scale_value", 7},
    {"kv_head_num", 8},
  },
  /*.func_impl_=*/gPagedAttentionMaskFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

PagedAttentionFuncImpl gPagedAttentionFuncImpl;
OpDef gPagedAttention = {
  /*.name_=*/"PagedAttention",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"query", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"key_cache", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"value_cache", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"block_tables", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"context_lens", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"head_num", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"scale_value", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"kv_head_num", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"attention_out", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"query", 0},
    {"key_cache", 1},
    {"value_cache", 2},
    {"block_tables", 3},
    {"context_lens", 4},
    {"head_num", 5},
    {"scale_value", 6},
    {"kv_head_num", 7},
  },
  /*.func_impl_=*/gPagedAttentionFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

PowFuncImpl gPowFuncImpl;
OpDef gPow = {
  /*.name_=*/"Pow",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"exponent", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {
    Signature("input", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
     Signature("exponent", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"exponent", 1},
  },
  /*.func_impl_=*/gPowFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

PReLUGradFuncImpl gPReLUGradFuncImpl;
OpDef gPReLUGrad = {
  /*.name_=*/"PReLUGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"dy", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"weight", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"dx", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"dw", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"dy", 0},
    {"x", 1},
    {"weight", 2},
  },
  /*.func_impl_=*/gPReLUGradFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

PReLUFuncImpl gPReLUFuncImpl;
OpDef gPReLU = {
  /*.name_=*/"PReLU",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"weight", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"weight", 1},
  },
  /*.func_impl_=*/gPReLUFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ProdExtFuncImpl gProdExtFuncImpl;
OpDef gProdExt = {
  /*.name_=*/"ProdExt",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"axis", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"keep_dims", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dtype", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"dtype_to_type_id", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"axis", 1},
    {"keep_dims", 2},
    {"dtype", 3},
  },
  /*.func_impl_=*/gProdExtFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

PromptKVCacheFuncImpl gPromptKVCacheFuncImpl;
OpDef gPromptKVCache = {
  /*.name_=*/"PromptKVCache",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"cache", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"update", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"valid_seq_len", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"batch_index", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"seq_len_axis", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"new_max_seq_len", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"cur_max_seq_len", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"align_mode", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"str_to_enum", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"out", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"cache", 0},
    {"update", 1},
    {"valid_seq_len", 2},
    {"batch_index", 3},
    {"seq_len_axis", 4},
    {"new_max_seq_len", 5},
    {"cur_max_seq_len", 6},
    {"align_mode", 7},
  },
  /*.func_impl_=*/gPromptKVCacheFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

QrFuncImpl gQrFuncImpl;
OpDef gQr = {
  /*.name_=*/"Qr",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"full_matrices", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"q", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"r", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"full_matrices", 1},
  },
  /*.func_impl_=*/gQrFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

RandExtFuncImpl gRandExtFuncImpl;
OpDef gRandExt = {
  /*.name_=*/"RandExt",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"shape", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"seed", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"offset", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dtype", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"dtype_to_type_id", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"shape", 0},
    {"seed", 1},
    {"offset", 2},
    {"dtype", 3},
  },
  /*.func_impl_=*/gRandExtFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

RandLikeExtFuncImpl gRandLikeExtFuncImpl;
OpDef gRandLikeExt = {
  /*.name_=*/"RandLikeExt",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"tensor", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"seed", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"offset", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dtype", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"dtype_to_type_id", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"tensor", 0},
    {"seed", 1},
    {"offset", 2},
    {"dtype", 3},
  },
  /*.func_impl_=*/gRandLikeExtFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

RandpermV2FuncImpl gRandpermV2FuncImpl;
OpDef gRandpermV2 = {
  /*.name_=*/"RandpermV2",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"n", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_TENSOR}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"seed", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_TENSOR}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"offset", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_TENSOR}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dtype", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"dtype_to_type_id", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"n", 0},
    {"seed", 1},
    {"offset", 2},
    {"dtype", 3},
  },
  /*.func_impl_=*/gRandpermV2FuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

RangeFuncImpl gRangeFuncImpl;
OpDef gRange = {
  /*.name_=*/"Range",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"start", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_TENSOR}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"end", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_TENSOR}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"step", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_TENSOR}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"maxlen", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"start", 0},
    {"end", 1},
    {"step", 2},
    {"maxlen", 3},
  },
  /*.func_impl_=*/gRangeFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

RankFuncImpl gRankFuncImpl;
OpDef gRank = {
  /*.name_=*/"Rank",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input_x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_INT,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input_x", 0},
  },
  /*.func_impl_=*/gRankFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

RealDivFuncImpl gRealDivFuncImpl;
OpDef gRealDiv = {
  /*.name_=*/"RealDiv",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {
    Signature("x", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
     Signature("y", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"y", 1},
  },
  /*.func_impl_=*/gRealDivFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

RealFuncImpl gRealFuncImpl;
OpDef gReal = {
  /*.name_=*/"Real",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
  },
  /*.func_impl_=*/gRealFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ReciprocalGradFuncImpl gReciprocalGradFuncImpl;
OpDef gReciprocalGrad = {
  /*.name_=*/"ReciprocalGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dy", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"z", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"y", 0},
    {"dy", 1},
  },
  /*.func_impl_=*/gReciprocalGradFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ReciprocalFuncImpl gReciprocalFuncImpl;
OpDef gReciprocal = {
  /*.name_=*/"Reciprocal",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
  },
  /*.func_impl_=*/gReciprocalFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

ReduceAllFuncImpl gReduceAllFuncImpl;
OpDef gReduceAll = {
  /*.name_=*/"ReduceAll",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"axis", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_INT, DT_LIST_INT, DT_TENSOR}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"keep_dims", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"axis", 1},
    {"keep_dims", 2},
  },
  /*.func_impl_=*/gReduceAllFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

ReduceAnyFuncImpl gReduceAnyFuncImpl;
OpDef gReduceAny = {
  /*.name_=*/"ReduceAny",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"axis", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_INT, DT_LIST_INT, DT_TENSOR}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"keep_dims", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"axis", 1},
    {"keep_dims", 2},
  },
  /*.func_impl_=*/gReduceAnyFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

ReduceMaxFuncImpl gReduceMaxFuncImpl;
OpDef gReduceMax = {
  /*.name_=*/"ReduceMax",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"axis", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_INT, DT_LIST_INT, DT_TENSOR}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"keep_dims", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"axis", 1},
    {"keep_dims", 2},
  },
  /*.func_impl_=*/gReduceMaxFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ReduceMeanFuncImpl gReduceMeanFuncImpl;
OpDef gReduceMean = {
  /*.name_=*/"ReduceMean",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"axis", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_INT, DT_LIST_INT, DT_TENSOR}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"keep_dims", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"axis", 1},
    {"keep_dims", 2},
  },
  /*.func_impl_=*/gReduceMeanFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ReduceMinFuncImpl gReduceMinFuncImpl;
OpDef gReduceMin = {
  /*.name_=*/"ReduceMin",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"axis", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_INT, DT_LIST_INT, DT_TENSOR}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"keep_dims", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"axis", 1},
    {"keep_dims", 2},
  },
  /*.func_impl_=*/gReduceMinFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ReduceProdFuncImpl gReduceProdFuncImpl;
OpDef gReduceProd = {
  /*.name_=*/"ReduceProd",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"axis", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_INT, DT_LIST_INT, DT_TENSOR}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"keep_dims", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"axis", 1},
    {"keep_dims", 2},
  },
  /*.func_impl_=*/gReduceProdFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ReduceStdFuncImpl gReduceStdFuncImpl;
OpDef gReduceStd = {
  /*.name_=*/"ReduceStd",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"axis", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_INT, DT_LIST_INT, DT_TENSOR}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"unbiased", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"keep_dims", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output_std", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"output_mean", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"axis", 1},
    {"unbiased", 2},
    {"keep_dims", 3},
  },
  /*.func_impl_=*/gReduceStdFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ReduceSumFuncImpl gReduceSumFuncImpl;
OpDef gReduceSum = {
  /*.name_=*/"ReduceSum",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"axis", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_INT, DT_LIST_INT, DT_TENSOR}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"keep_dims", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"skip_mode", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"axis", 1},
    {"keep_dims", 2},
    {"skip_mode", 3},
  },
  /*.func_impl_=*/gReduceSumFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ReflectionPad1DGradFuncImpl gReflectionPad1DGradFuncImpl;
OpDef gReflectionPad1DGrad = {
  /*.name_=*/"ReflectionPad1DGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"grad_output", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"padding", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT, DT_TENSOR}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"grad_output", 0},
    {"input", 1},
    {"padding", 2},
  },
  /*.func_impl_=*/gReflectionPad1DGradFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

ReflectionPad1DFuncImpl gReflectionPad1DFuncImpl;
OpDef gReflectionPad1D = {
  /*.name_=*/"ReflectionPad1D",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"padding", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT, DT_TENSOR}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"padding", 1},
  },
  /*.func_impl_=*/gReflectionPad1DFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

ReflectionPad2DGradFuncImpl gReflectionPad2DGradFuncImpl;
OpDef gReflectionPad2DGrad = {
  /*.name_=*/"ReflectionPad2DGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"grad_output", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"padding", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT, DT_TENSOR}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"grad_output", 0},
    {"input", 1},
    {"padding", 2},
  },
  /*.func_impl_=*/gReflectionPad2DGradFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

ReflectionPad2DFuncImpl gReflectionPad2DFuncImpl;
OpDef gReflectionPad2D = {
  /*.name_=*/"ReflectionPad2D",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"padding", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT, DT_TENSOR}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"padding", 1},
  },
  /*.func_impl_=*/gReflectionPad2DFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

ReflectionPad3DGradFuncImpl gReflectionPad3DGradFuncImpl;
OpDef gReflectionPad3DGrad = {
  /*.name_=*/"ReflectionPad3DGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"grad_output", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"padding", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT, DT_TENSOR}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"grad_output", 0},
    {"input", 1},
    {"padding", 2},
  },
  /*.func_impl_=*/gReflectionPad3DGradFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

ReflectionPad3DFuncImpl gReflectionPad3DFuncImpl;
OpDef gReflectionPad3D = {
  /*.name_=*/"ReflectionPad3D",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"padding", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT, DT_TENSOR}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"padding", 1},
  },
  /*.func_impl_=*/gReflectionPad3DFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

ReLU6GradFuncImpl gReLU6GradFuncImpl;
OpDef gReLU6Grad = {
  /*.name_=*/"ReLU6Grad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"y_backprop", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"y_backprop", 0},
    {"x", 1},
  },
  /*.func_impl_=*/gReLU6GradFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ReLU6FuncImpl gReLU6FuncImpl;
OpDef gReLU6 = {
  /*.name_=*/"ReLU6",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
  },
  /*.func_impl_=*/gReLU6FuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ReluGradFuncImpl gReluGradFuncImpl;
OpDef gReluGrad = {
  /*.name_=*/"ReluGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"y_backprop", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"y_backprop", 0},
    {"x", 1},
  },
  /*.func_impl_=*/gReluGradFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

ReLUFuncImpl gReLUFuncImpl;
OpDef gReLU = {
  /*.name_=*/"ReLU",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
  },
  /*.func_impl_=*/gReLUFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

RepeatInterleaveGradFuncImpl gRepeatInterleaveGradFuncImpl;
OpDef gRepeatInterleaveGrad = {
  /*.name_=*/"RepeatInterleaveGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"repeats", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dim", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"repeats", 1},
    {"dim", 2},
  },
  /*.func_impl_=*/gRepeatInterleaveGradFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

RepeatInterleaveIntFuncImpl gRepeatInterleaveIntFuncImpl;
OpDef gRepeatInterleaveInt = {
  /*.name_=*/"RepeatInterleaveInt",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"repeats", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dim", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"output_size", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"repeats", 1},
    {"dim", 2},
    {"output_size", 3},
  },
  /*.func_impl_=*/gRepeatInterleaveIntFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

RepeatInterleaveTensorFuncImpl gRepeatInterleaveTensorFuncImpl;
OpDef gRepeatInterleaveTensor = {
  /*.name_=*/"RepeatInterleaveTensor",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"repeats", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT, DT_TUPLE_INT}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dim", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"output_size", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"repeats", 1},
    {"dim", 2},
    {"output_size", 3},
  },
  /*.func_impl_=*/gRepeatInterleaveTensorFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

ReplicationPad1DGradFuncImpl gReplicationPad1DGradFuncImpl;
OpDef gReplicationPad1DGrad = {
  /*.name_=*/"ReplicationPad1DGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"grad_output", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"padding", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT, DT_TENSOR}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"grad_output", 0},
    {"input", 1},
    {"padding", 2},
  },
  /*.func_impl_=*/gReplicationPad1DGradFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

ReplicationPad1DFuncImpl gReplicationPad1DFuncImpl;
OpDef gReplicationPad1D = {
  /*.name_=*/"ReplicationPad1D",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"padding", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT, DT_TENSOR}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"padding", 1},
  },
  /*.func_impl_=*/gReplicationPad1DFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

ReplicationPad2DGradFuncImpl gReplicationPad2DGradFuncImpl;
OpDef gReplicationPad2DGrad = {
  /*.name_=*/"ReplicationPad2DGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"grad_output", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"padding", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT, DT_TENSOR}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"grad_output", 0},
    {"input", 1},
    {"padding", 2},
  },
  /*.func_impl_=*/gReplicationPad2DGradFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

ReplicationPad2DFuncImpl gReplicationPad2DFuncImpl;
OpDef gReplicationPad2D = {
  /*.name_=*/"ReplicationPad2D",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"padding", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT, DT_TENSOR}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"padding", 1},
  },
  /*.func_impl_=*/gReplicationPad2DFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

ReplicationPad3DGradFuncImpl gReplicationPad3DGradFuncImpl;
OpDef gReplicationPad3DGrad = {
  /*.name_=*/"ReplicationPad3DGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"grad_output", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"padding", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT, DT_TENSOR}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"grad_output", 0},
    {"input", 1},
    {"padding", 2},
  },
  /*.func_impl_=*/gReplicationPad3DGradFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

ReplicationPad3DFuncImpl gReplicationPad3DFuncImpl;
OpDef gReplicationPad3D = {
  /*.name_=*/"ReplicationPad3D",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"padding", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT, DT_TENSOR}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"padding", 1},
  },
  /*.func_impl_=*/gReplicationPad3DFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

ReshapeAndCacheFuncImpl gReshapeAndCacheFuncImpl;
OpDef gReshapeAndCache = {
  /*.name_=*/"ReshapeAndCache",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"key", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"value", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"key_cache", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"value_cache", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"slot_mapping", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"out", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {
    Signature("key", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
     Signature("value", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
     Signature("key_cache", SignatureEnumRW::kRWWrite,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
     Signature("value_cache", SignatureEnumRW::kRWWrite,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
     Signature("slot_mapping", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType1),
  },
  /*.indexes_ =*/ {
    {"key", 0},
    {"value", 1},
    {"key_cache", 2},
    {"value_cache", 3},
    {"slot_mapping", 4},
  },
  /*.func_impl_=*/gReshapeAndCacheFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ReshapeFuncImpl gReshapeFuncImpl;
OpDef gReshape = {
  /*.name_=*/"Reshape",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"shape", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_TENSOR, DT_LIST_INT}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"shape", 1},
  },
  /*.func_impl_=*/gReshapeFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/true,
};

ResizeBicubicGradFuncImpl gResizeBicubicGradFuncImpl;
OpDef gResizeBicubicGrad = {
  /*.name_=*/"ResizeBicubicGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"grads", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"image", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"align_corners", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"half_pixel_centers", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"grads", 0},
    {"image", 1},
    {"align_corners", 2},
    {"half_pixel_centers", 3},
  },
  /*.func_impl_=*/gResizeBicubicGradFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ResizeBicubicFuncImpl gResizeBicubicFuncImpl;
OpDef gResizeBicubic = {
  /*.name_=*/"ResizeBicubic",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"image", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"size", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_TENSOR, DT_LIST_INT}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"align_corners", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"half_pixel_centers", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"image", 0},
    {"size", 1},
    {"align_corners", 2},
    {"half_pixel_centers", 3},
  },
  /*.func_impl_=*/gResizeBicubicFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ResizeBilinearGradFuncImpl gResizeBilinearGradFuncImpl;
OpDef gResizeBilinearGrad = {
  /*.name_=*/"ResizeBilinearGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"grads", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"image", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"align_corners", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"half_pixel_centers", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"grads", 0},
    {"image", 1},
    {"align_corners", 2},
    {"half_pixel_centers", 3},
  },
  /*.func_impl_=*/gResizeBilinearGradFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ResizeBilinearV2FuncImpl gResizeBilinearV2FuncImpl;
OpDef gResizeBilinearV2 = {
  /*.name_=*/"ResizeBilinearV2",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"image", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"size", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT, DT_TENSOR}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"align_corners", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"half_pixel_centers", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"image", 0},
    {"size", 1},
    {"align_corners", 2},
    {"half_pixel_centers", 3},
  },
  /*.func_impl_=*/gResizeBilinearV2FuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ResizeDFuncImpl gResizeDFuncImpl;
OpDef gResizeD = {
  /*.name_=*/"ResizeD",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"sizes", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"scales", /*.arg_dtype_=*/DT_TUPLE_FLOAT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_FLOAT}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"coordinate_transformation_mode", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"str_to_enum", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"sizes", 1},
    {"scales", 2},
    {"coordinate_transformation_mode", 3},
  },
  /*.func_impl_=*/gResizeDFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ResizeLinear1DGradFuncImpl gResizeLinear1DGradFuncImpl;
OpDef gResizeLinear1DGrad = {
  /*.name_=*/"ResizeLinear1DGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"grads", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"coordinate_transformation_mode", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"str_to_enum", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"grads", 0},
    {"x", 1},
    {"coordinate_transformation_mode", 2},
  },
  /*.func_impl_=*/gResizeLinear1DGradFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ResizeLinear1DFuncImpl gResizeLinear1DFuncImpl;
OpDef gResizeLinear1D = {
  /*.name_=*/"ResizeLinear1D",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"size", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT, DT_TENSOR}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"coordinate_transformation_mode", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"str_to_enum", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"size", 1},
    {"coordinate_transformation_mode", 2},
  },
  /*.func_impl_=*/gResizeLinear1DFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ResizeNearestNeighborGradFuncImpl gResizeNearestNeighborGradFuncImpl;
OpDef gResizeNearestNeighborGrad = {
  /*.name_=*/"ResizeNearestNeighborGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"grads", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"size", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"align_corners", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"half_pixel_centers", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"grads", 0},
    {"size", 1},
    {"align_corners", 2},
    {"half_pixel_centers", 3},
  },
  /*.func_impl_=*/gResizeNearestNeighborGradFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ResizeNearestNeighborFuncImpl gResizeNearestNeighborFuncImpl;
OpDef gResizeNearestNeighbor = {
  /*.name_=*/"ResizeNearestNeighbor",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input_x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"size", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"align_corners", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"half_pixel_centers", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"image_out", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input_x", 0},
    {"size", 1},
    {"align_corners", 2},
    {"half_pixel_centers", 3},
  },
  /*.func_impl_=*/gResizeNearestNeighborFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ResizeNearestNeighborV2GradFuncImpl gResizeNearestNeighborV2GradFuncImpl;
OpDef gResizeNearestNeighborV2Grad = {
  /*.name_=*/"ResizeNearestNeighborV2Grad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"grads", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"size", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_TENSOR}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"align_corners", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"half_pixel_centers", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"grads", 0},
    {"size", 1},
    {"align_corners", 2},
    {"half_pixel_centers", 3},
  },
  /*.func_impl_=*/gResizeNearestNeighborV2GradFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ResizeNearestNeighborV2FuncImpl gResizeNearestNeighborV2FuncImpl;
OpDef gResizeNearestNeighborV2 = {
  /*.name_=*/"ResizeNearestNeighborV2",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"image", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"size", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_TENSOR, DT_LIST_INT}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"align_corners", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"half_pixel_centers", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"image", 0},
    {"size", 1},
    {"align_corners", 2},
    {"half_pixel_centers", 3},
  },
  /*.func_impl_=*/gResizeNearestNeighborV2FuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ReverseV2FuncImpl gReverseV2FuncImpl;
OpDef gReverseV2 = {
  /*.name_=*/"ReverseV2",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"axis", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"axis", 1},
  },
  /*.func_impl_=*/gReverseV2FuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

RFFTGradFuncImpl gRFFTGradFuncImpl;
OpDef gRFFTGrad = {
  /*.name_=*/"RFFTGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input1", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"input2", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"n", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"dim", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"norm", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"str_to_enum", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input1", 0},
    {"input2", 1},
    {"n", 2},
    {"dim", 3},
    {"norm", 4},
  },
  /*.func_impl_=*/gRFFTGradFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

RFFTFuncImpl gRFFTFuncImpl;
OpDef gRFFT = {
  /*.name_=*/"RFFT",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"n", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"dim", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"norm", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"str_to_enum", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"n", 1},
    {"dim", 2},
    {"norm", 3},
  },
  /*.func_impl_=*/gRFFTFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

RightShiftFuncImpl gRightShiftFuncImpl;
OpDef gRightShift = {
  /*.name_=*/"RightShift",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input_x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"input_y", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input_x", 0},
    {"input_y", 1},
  },
  /*.func_impl_=*/gRightShiftFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

RmsNormGradFuncImpl gRmsNormGradFuncImpl;
OpDef gRmsNormGrad = {
  /*.name_=*/"RmsNormGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"dy", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"rstd", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"gamma", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"dx", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"dgamma", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"dy", 0},
    {"x", 1},
    {"rstd", 2},
    {"gamma", 3},
  },
  /*.func_impl_=*/gRmsNormGradFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

RmsNormFuncImpl gRmsNormFuncImpl;
OpDef gRmsNorm = {
  /*.name_=*/"RmsNorm",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"gamma", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"epsilon", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"rstd", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"gamma", 1},
    {"epsilon", 2},
  },
  /*.func_impl_=*/gRmsNormFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

RollFuncImpl gRollFuncImpl;
OpDef gRoll = {
  /*.name_=*/"Roll",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"shift", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_INT, DT_LIST_INT}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"axis", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_INT, DT_LIST_INT}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"shift", 1},
    {"axis", 2},
  },
  /*.func_impl_=*/gRollFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

RoundFuncImpl gRoundFuncImpl;
OpDef gRound = {
  /*.name_=*/"Round",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
  },
  /*.func_impl_=*/gRoundFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

RsqrtGradFuncImpl gRsqrtGradFuncImpl;
OpDef gRsqrtGrad = {
  /*.name_=*/"RsqrtGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"y_backprop", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"y_backprop", 0},
    {"x", 1},
  },
  /*.func_impl_=*/gRsqrtGradFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

RsqrtFuncImpl gRsqrtFuncImpl;
OpDef gRsqrt = {
  /*.name_=*/"Rsqrt",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
  },
  /*.func_impl_=*/gRsqrtFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

ScalarAddFuncImpl gScalarAddFuncImpl;
OpDef gScalarAdd = {
  /*.name_=*/"ScalarAdd",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_NUMBER,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"y", 1},
  },
  /*.func_impl_=*/gScalarAddFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ScalarBoolFuncImpl gScalarBoolFuncImpl;
OpDef gScalarBool = {
  /*.name_=*/"ScalarBool",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_NUMBER,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
  },
  /*.func_impl_=*/gScalarBoolFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ScalarCastFuncImpl gScalarCastFuncImpl;
OpDef gScalarCast = {
  /*.name_=*/"ScalarCast",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input_x", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"input_y", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"dtype_to_type_id", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output_data", /*.arg_dtype_=*/DT_NUMBER,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input_x", 0},
    {"input_y", 1},
  },
  /*.func_impl_=*/gScalarCastFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ScalarDivFuncImpl gScalarDivFuncImpl;
OpDef gScalarDiv = {
  /*.name_=*/"ScalarDiv",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_NUMBER,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"y", 1},
  },
  /*.func_impl_=*/gScalarDivFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ScalarEqFuncImpl gScalarEqFuncImpl;
OpDef gScalarEq = {
  /*.name_=*/"ScalarEq",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_NUMBER,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"y", 1},
  },
  /*.func_impl_=*/gScalarEqFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ScalarFloorDivFuncImpl gScalarFloorDivFuncImpl;
OpDef gScalarFloorDiv = {
  /*.name_=*/"ScalarFloorDiv",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_NUMBER,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"y", 1},
  },
  /*.func_impl_=*/gScalarFloorDivFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ScalarGeFuncImpl gScalarGeFuncImpl;
OpDef gScalarGe = {
  /*.name_=*/"ScalarGe",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_NUMBER,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"y", 1},
  },
  /*.func_impl_=*/gScalarGeFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ScalarGtFuncImpl gScalarGtFuncImpl;
OpDef gScalarGt = {
  /*.name_=*/"ScalarGt",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_NUMBER,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"y", 1},
  },
  /*.func_impl_=*/gScalarGtFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ScalarLeFuncImpl gScalarLeFuncImpl;
OpDef gScalarLe = {
  /*.name_=*/"ScalarLe",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_NUMBER,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"y", 1},
  },
  /*.func_impl_=*/gScalarLeFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ScalarLogFuncImpl gScalarLogFuncImpl;
OpDef gScalarLog = {
  /*.name_=*/"ScalarLog",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_NUMBER,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
  },
  /*.func_impl_=*/gScalarLogFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ScalarLtFuncImpl gScalarLtFuncImpl;
OpDef gScalarLt = {
  /*.name_=*/"ScalarLt",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_NUMBER,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"y", 1},
  },
  /*.func_impl_=*/gScalarLtFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ScalarModFuncImpl gScalarModFuncImpl;
OpDef gScalarMod = {
  /*.name_=*/"ScalarMod",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_NUMBER,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"y", 1},
  },
  /*.func_impl_=*/gScalarModFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ScalarMulFuncImpl gScalarMulFuncImpl;
OpDef gScalarMul = {
  /*.name_=*/"ScalarMul",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_NUMBER,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"y", 1},
  },
  /*.func_impl_=*/gScalarMulFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ScalarPowFuncImpl gScalarPowFuncImpl;
OpDef gScalarPow = {
  /*.name_=*/"ScalarPow",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_NUMBER,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"y", 1},
  },
  /*.func_impl_=*/gScalarPowFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ScalarSubFuncImpl gScalarSubFuncImpl;
OpDef gScalarSub = {
  /*.name_=*/"ScalarSub",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_NUMBER,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"y", 1},
  },
  /*.func_impl_=*/gScalarSubFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ScalarToTensorFuncImpl gScalarToTensorFuncImpl;
OpDef gScalarToTensor = {
  /*.name_=*/"ScalarToTensor",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input_x", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dtype", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"dtype_to_type_id", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input_x", 0},
    {"dtype", 1},
  },
  /*.func_impl_=*/gScalarToTensorFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ScalarUaddFuncImpl gScalarUaddFuncImpl;
OpDef gScalarUadd = {
  /*.name_=*/"ScalarUadd",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_NUMBER,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
  },
  /*.func_impl_=*/gScalarUaddFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ScalarUsubFuncImpl gScalarUsubFuncImpl;
OpDef gScalarUsub = {
  /*.name_=*/"ScalarUsub",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_NUMBER,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
  },
  /*.func_impl_=*/gScalarUsubFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ScatterAddExtFuncImpl gScatterAddExtFuncImpl;
OpDef gScatterAddExt = {
  /*.name_=*/"ScatterAddExt",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dim", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"index", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"src", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"dim", 1},
    {"index", 2},
    {"src", 3},
  },
  /*.func_impl_=*/gScatterAddExtFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

ScatterNdFuncImpl gScatterNdFuncImpl;
OpDef gScatterNd = {
  /*.name_=*/"ScatterNd",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"indices", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"updates", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"shape", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"indices", 0},
    {"updates", 1},
    {"shape", 2},
  },
  /*.func_impl_=*/gScatterNdFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ScatterFuncImpl gScatterFuncImpl;
OpDef gScatter = {
  /*.name_=*/"Scatter",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dim", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"index", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"src", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"reduce", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"out", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"dim", 1},
    {"index", 2},
    {"src", 3},
    {"reduce", 4},
  },
  /*.func_impl_=*/gScatterFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

SearchSortedFuncImpl gSearchSortedFuncImpl;
OpDef gSearchSorted = {
  /*.name_=*/"SearchSorted",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"sorted_sequence", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"values", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"sorter", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"dtype", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"dtype_to_type_id", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"right", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"sorted_sequence", 0},
    {"values", 1},
    {"sorter", 2},
    {"dtype", 3},
    {"right", 4},
  },
  /*.func_impl_=*/gSearchSortedFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

SelectFuncImpl gSelectFuncImpl;
OpDef gSelect = {
  /*.name_=*/"Select",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"condition", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"other", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"condition", 0},
    {"input", 1},
    {"other", 2},
  },
  /*.func_impl_=*/gSelectFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

SequenceConcatFuncImpl gSequenceConcatFuncImpl;
OpDef gSequenceConcat = {
  /*.name_=*/"SequenceConcat",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TUPLE_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_TENSOR}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"axis", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"axis", 1},
  },
  /*.func_impl_=*/gSequenceConcatFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ShapeFuncImpl gShapeFuncImpl;
OpDef gShape = {
  /*.name_=*/"Shape",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input_x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TUPLE_INT,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input_x", 0},
  },
  /*.func_impl_=*/gShapeFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

SigmoidGradFuncImpl gSigmoidGradFuncImpl;
OpDef gSigmoidGrad = {
  /*.name_=*/"SigmoidGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dy", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"y", 0},
    {"dy", 1},
  },
  /*.func_impl_=*/gSigmoidGradFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

SigmoidFuncImpl gSigmoidFuncImpl;
OpDef gSigmoid = {
  /*.name_=*/"Sigmoid",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
  },
  /*.func_impl_=*/gSigmoidFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

SignFuncImpl gSignFuncImpl;
OpDef gSign = {
  /*.name_=*/"Sign",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
  },
  /*.func_impl_=*/gSignFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

SiLUGradFuncImpl gSiLUGradFuncImpl;
OpDef gSiLUGrad = {
  /*.name_=*/"SiLUGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"dout", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"dout", 0},
    {"x", 1},
  },
  /*.func_impl_=*/gSiLUGradFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

SiLUFuncImpl gSiLUFuncImpl;
OpDef gSiLU = {
  /*.name_=*/"SiLU",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
  },
  /*.func_impl_=*/gSiLUFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

SinFuncImpl gSinFuncImpl;
OpDef gSin = {
  /*.name_=*/"Sin",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
  },
  /*.func_impl_=*/gSinFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

SincFuncImpl gSincFuncImpl;
OpDef gSinc = {
  /*.name_=*/"Sinc",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
  },
  /*.func_impl_=*/gSincFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

SinhFuncImpl gSinhFuncImpl;
OpDef gSinh = {
  /*.name_=*/"Sinh",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
  },
  /*.func_impl_=*/gSinhFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

SliceExtFuncImpl gSliceExtFuncImpl;
OpDef gSliceExt = {
  /*.name_=*/"SliceExt",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dim", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"start", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"end", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"step", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"dim", 1},
    {"start", 2},
    {"end", 3},
    {"step", 4},
  },
  /*.func_impl_=*/gSliceExtFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/true,
};

SoftmaxBackwardFuncImpl gSoftmaxBackwardFuncImpl;
OpDef gSoftmaxBackward = {
  /*.name_=*/"SoftmaxBackward",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"dout", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"out", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dim", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"dout", 0},
    {"out", 1},
    {"dim", 2},
  },
  /*.func_impl_=*/gSoftmaxBackwardFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

SoftmaxFuncImpl gSoftmaxFuncImpl;
OpDef gSoftmax = {
  /*.name_=*/"Softmax",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"axis", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_INT}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"axis", 1},
  },
  /*.func_impl_=*/gSoftmaxFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

SoftplusExtFuncImpl gSoftplusExtFuncImpl;
OpDef gSoftplusExt = {
  /*.name_=*/"SoftplusExt",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"beta", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"threshold", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"beta", 1},
    {"threshold", 2},
  },
  /*.func_impl_=*/gSoftplusExtFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

SoftplusGradExtFuncImpl gSoftplusGradExtFuncImpl;
OpDef gSoftplusGradExt = {
  /*.name_=*/"SoftplusGradExt",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"dout", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"beta", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"threshold", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"dout", 0},
    {"x", 1},
    {"beta", 2},
    {"threshold", 3},
  },
  /*.func_impl_=*/gSoftplusGradExtFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

SolveTriangularFuncImpl gSolveTriangularFuncImpl;
OpDef gSolveTriangular = {
  /*.name_=*/"SolveTriangular",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"a", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"b", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"trans", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"lower", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"unit_diagonal", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"a", 0},
    {"b", 1},
    {"trans", 2},
    {"lower", 3},
    {"unit_diagonal", 4},
  },
  /*.func_impl_=*/gSolveTriangularFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

SortExtFuncImpl gSortExtFuncImpl;
OpDef gSortExt = {
  /*.name_=*/"SortExt",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dim", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"descending", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"stable", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"indices", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"dim", 1},
    {"descending", 2},
    {"stable", 3},
  },
  /*.func_impl_=*/gSortExtFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

SplitFuncImpl gSplitFuncImpl;
OpDef gSplit = {
  /*.name_=*/"Split",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input_x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"axis", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"output_num", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TUPLE_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input_x", 0},
    {"axis", 1},
    {"output_num", 2},
  },
  /*.func_impl_=*/gSplitFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

SplitTensorFuncImpl gSplitTensorFuncImpl;
OpDef gSplitTensor = {
  /*.name_=*/"SplitTensor",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input_x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"split_int", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"axis", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TUPLE_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input_x", 0},
    {"split_int", 1},
    {"axis", 2},
  },
  /*.func_impl_=*/gSplitTensorFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/true,
};

SplitWithSizeFuncImpl gSplitWithSizeFuncImpl;
OpDef gSplitWithSize = {
  /*.name_=*/"SplitWithSize",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input_x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"split_sections", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"axis", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TUPLE_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input_x", 0},
    {"split_sections", 1},
    {"axis", 2},
  },
  /*.func_impl_=*/gSplitWithSizeFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/true,
};

SqrtGradFuncImpl gSqrtGradFuncImpl;
OpDef gSqrtGrad = {
  /*.name_=*/"SqrtGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"dy", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"z", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"dy", 0},
    {"y", 1},
  },
  /*.func_impl_=*/gSqrtGradFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

SqrtFuncImpl gSqrtFuncImpl;
OpDef gSqrt = {
  /*.name_=*/"Sqrt",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
  },
  /*.func_impl_=*/gSqrtFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

SquareFuncImpl gSquareFuncImpl;
OpDef gSquare = {
  /*.name_=*/"Square",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
  },
  /*.func_impl_=*/gSquareFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

StackExtFuncImpl gStackExtFuncImpl;
OpDef gStackExt = {
  /*.name_=*/"StackExt",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"tensors", /*.arg_dtype_=*/DT_TUPLE_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_TENSOR}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dim", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"tensors", 0},
    {"dim", 1},
  },
  /*.func_impl_=*/gStackExtFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

StridedSliceFuncImpl gStridedSliceFuncImpl;
OpDef gStridedSlice = {
  /*.name_=*/"StridedSlice",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input_x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"begin", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT, DT_TENSOR}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"end", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT, DT_TENSOR}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"strides", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT, DT_TENSOR}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"begin_mask", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"end_mask", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"ellipsis_mask", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"new_axis_mask", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"shrink_axis_mask", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input_x", 0},
    {"begin", 1},
    {"end", 2},
    {"strides", 3},
    {"begin_mask", 4},
    {"end_mask", 5},
    {"ellipsis_mask", 6},
    {"new_axis_mask", 7},
    {"shrink_axis_mask", 8},
  },
  /*.func_impl_=*/gStridedSliceFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

SubExtFuncImpl gSubExtFuncImpl;
OpDef gSubExt = {
  /*.name_=*/"SubExt",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"other", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"alpha", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {
    Signature("input", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
     Signature("other", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
     Signature("alpha", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType1),
  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"other", 1},
    {"alpha", 2},
  },
  /*.func_impl_=*/gSubExtFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

SubFuncImpl gSubFuncImpl;
OpDef gSub = {
  /*.name_=*/"Sub",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"other", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_NUMBER}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {
    Signature("input", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
     Signature("other", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"other", 1},
  },
  /*.func_impl_=*/gSubFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

SumExtFuncImpl gSumExtFuncImpl;
OpDef gSumExt = {
  /*.name_=*/"SumExt",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dim", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_INT, DT_LIST_INT, DT_TENSOR}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"keepdim", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dtype", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"dtype_to_type_id", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"dim", 1},
    {"keepdim", 2},
    {"dtype", 3},
  },
  /*.func_impl_=*/gSumExtFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

TanhGradFuncImpl gTanhGradFuncImpl;
OpDef gTanhGrad = {
  /*.name_=*/"TanhGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dy", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"dx", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"y", 0},
    {"dy", 1},
  },
  /*.func_impl_=*/gTanhGradFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

TanhFuncImpl gTanhFuncImpl;
OpDef gTanh = {
  /*.name_=*/"Tanh",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
  },
  /*.func_impl_=*/gTanhFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

TensorCopySlicesFuncImpl gTensorCopySlicesFuncImpl;
OpDef gTensorCopySlices = {
  /*.name_=*/"TensorCopySlices",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"value", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"begin", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT, DT_TENSOR}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"end", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT, DT_TENSOR}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"strides", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT, DT_TENSOR}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"value", 1},
    {"begin", 2},
    {"end", 3},
    {"strides", 4},
  },
  /*.func_impl_=*/gTensorCopySlicesFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

TensorShapeFuncImpl gTensorShapeFuncImpl;
OpDef gTensorShape = {
  /*.name_=*/"TensorShape",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input_x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input_x", 0},
  },
  /*.func_impl_=*/gTensorShapeFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

TileFuncImpl gTileFuncImpl;
OpDef gTile = {
  /*.name_=*/"Tile",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dims", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_TENSOR}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"dims", 1},
  },
  /*.func_impl_=*/gTileFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

TopkExtFuncImpl gTopkExtFuncImpl;
OpDef gTopkExt = {
  /*.name_=*/"TopkExt",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"k", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dim", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"largest", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"sorted", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"values", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"indices", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"k", 1},
    {"dim", 2},
    {"largest", 3},
    {"sorted", 4},
  },
  /*.func_impl_=*/gTopkExtFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

TopKRouterFuncImpl gTopKRouterFuncImpl;
OpDef gTopKRouter = {
  /*.name_=*/"TopKRouter",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"capacity", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"expert_num", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"dispatch_index", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"combine_index", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"capacity", 1},
    {"expert_num", 2},
  },
  /*.func_impl_=*/gTopKRouterFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

TraceFuncImpl gTraceFuncImpl;
OpDef gTrace = {
  /*.name_=*/"Trace",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
  },
  /*.func_impl_=*/gTraceFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

TransposeFuncImpl gTransposeFuncImpl;
OpDef gTranspose = {
  /*.name_=*/"Transpose",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"input_perm", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"input_perm", 1},
  },
  /*.func_impl_=*/gTransposeFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/true,
};

TriuFuncImpl gTriuFuncImpl;
OpDef gTriu = {
  /*.name_=*/"Triu",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"diagonal", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"diagonal", 1},
  },
  /*.func_impl_=*/gTriuFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

TupleToListFuncImpl gTupleToListFuncImpl;
OpDef gTupleToList = {
  /*.name_=*/"TupleToList",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TUPLE_ANY, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_LIST_ANY,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
  },
  /*.func_impl_=*/gTupleToListFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

TupleToTensorFuncImpl gTupleToTensorFuncImpl;
OpDef gTupleToTensor = {
  /*.name_=*/"TupleToTensor",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input_tuple", /*.arg_dtype_=*/DT_TUPLE_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dtype", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"dtype_to_type_id", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input_tuple", 0},
    {"dtype", 1},
  },
  /*.func_impl_=*/gTupleToTensorFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

UniformExtFuncImpl gUniformExtFuncImpl;
OpDef gUniformExt = {
  /*.name_=*/"UniformExt",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"tensor", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"a", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"b", /*.arg_dtype_=*/DT_NUMBER, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"seed", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"offset", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"tensor", 0},
    {"a", 1},
    {"b", 2},
    {"seed", 3},
    {"offset", 4},
  },
  /*.func_impl_=*/gUniformExtFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

Unique2FuncImpl gUnique2FuncImpl;
OpDef gUnique2 = {
  /*.name_=*/"Unique2",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"sorted", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"return_inverse", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"return_counts", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"inverse_indices", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"counts", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"sorted", 1},
    {"return_inverse", 2},
    {"return_counts", 3},
  },
  /*.func_impl_=*/gUnique2FuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

UniqueDimFuncImpl gUniqueDimFuncImpl;
OpDef gUniqueDim = {
  /*.name_=*/"UniqueDim",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"sorted", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"return_inverse", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dim", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"inverse_indices", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"counts", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"sorted", 1},
    {"return_inverse", 2},
    {"dim", 3},
  },
  /*.func_impl_=*/gUniqueDimFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

UnsortedSegmentSumFuncImpl gUnsortedSegmentSumFuncImpl;
OpDef gUnsortedSegmentSum = {
  /*.name_=*/"UnsortedSegmentSum",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input_x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"segment_ids", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"num_segments", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_TENSOR}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input_x", 0},
    {"segment_ids", 1},
    {"num_segments", 2},
  },
  /*.func_impl_=*/gUnsortedSegmentSumFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

UnstackExtFuncImpl gUnstackExtFuncImpl;
OpDef gUnstackExt = {
  /*.name_=*/"UnstackExt",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input_x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"axis", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TUPLE_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input_x", 0},
    {"axis", 1},
  },
  /*.func_impl_=*/gUnstackExtFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/true,
};

UpsampleBilinear2DGradFuncImpl gUpsampleBilinear2DGradFuncImpl;
OpDef gUpsampleBilinear2DGrad = {
  /*.name_=*/"UpsampleBilinear2DGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"dy", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"input_size", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"output_size", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"scales", /*.arg_dtype_=*/DT_TUPLE_FLOAT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_FLOAT}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"align_corners", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"dx", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"dy", 0},
    {"input_size", 1},
    {"output_size", 2},
    {"scales", 3},
    {"align_corners", 4},
  },
  /*.func_impl_=*/gUpsampleBilinear2DGradFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

UpsampleBilinear2DFuncImpl gUpsampleBilinear2DFuncImpl;
OpDef gUpsampleBilinear2D = {
  /*.name_=*/"UpsampleBilinear2D",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"output_size", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"scales", /*.arg_dtype_=*/DT_TUPLE_FLOAT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_FLOAT}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"align_corners", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"output_size", 1},
    {"scales", 2},
    {"align_corners", 3},
  },
  /*.func_impl_=*/gUpsampleBilinear2DFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

UpsampleLinear1DGradFuncImpl gUpsampleLinear1DGradFuncImpl;
OpDef gUpsampleLinear1DGrad = {
  /*.name_=*/"UpsampleLinear1DGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"dy", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"input_size", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"output_size", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"scales", /*.arg_dtype_=*/DT_TUPLE_FLOAT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_FLOAT}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"align_corners", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"dx", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"dy", 0},
    {"input_size", 1},
    {"output_size", 2},
    {"scales", 3},
    {"align_corners", 4},
  },
  /*.func_impl_=*/gUpsampleLinear1DGradFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

UpsampleLinear1DFuncImpl gUpsampleLinear1DFuncImpl;
OpDef gUpsampleLinear1D = {
  /*.name_=*/"UpsampleLinear1D",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"output_size", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"scales", /*.arg_dtype_=*/DT_TUPLE_FLOAT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_FLOAT}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"align_corners", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"output_size", 1},
    {"scales", 2},
    {"align_corners", 3},
  },
  /*.func_impl_=*/gUpsampleLinear1DFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

UpsampleNearest1DGradFuncImpl gUpsampleNearest1DGradFuncImpl;
OpDef gUpsampleNearest1DGrad = {
  /*.name_=*/"UpsampleNearest1DGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"dy", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"input_size", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"output_size", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"scales", /*.arg_dtype_=*/DT_TUPLE_FLOAT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_FLOAT}, /*.is_optional_=*/true},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"dx", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"dy", 0},
    {"input_size", 1},
    {"output_size", 2},
    {"scales", 3},
  },
  /*.func_impl_=*/gUpsampleNearest1DGradFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

UpsampleNearest1DFuncImpl gUpsampleNearest1DFuncImpl;
OpDef gUpsampleNearest1D = {
  /*.name_=*/"UpsampleNearest1D",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"output_size", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"scales", /*.arg_dtype_=*/DT_TUPLE_FLOAT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_FLOAT}, /*.is_optional_=*/true},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"output_size", 1},
    {"scales", 2},
  },
  /*.func_impl_=*/gUpsampleNearest1DFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

UpsampleNearest2DGradFuncImpl gUpsampleNearest2DGradFuncImpl;
OpDef gUpsampleNearest2DGrad = {
  /*.name_=*/"UpsampleNearest2DGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"dy", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"input_size", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"output_size", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"scales", /*.arg_dtype_=*/DT_TUPLE_FLOAT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_FLOAT}, /*.is_optional_=*/true},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"dx", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"dy", 0},
    {"input_size", 1},
    {"output_size", 2},
    {"scales", 3},
  },
  /*.func_impl_=*/gUpsampleNearest2DGradFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

UpsampleNearest2DFuncImpl gUpsampleNearest2DFuncImpl;
OpDef gUpsampleNearest2D = {
  /*.name_=*/"UpsampleNearest2D",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"output_size", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"scales", /*.arg_dtype_=*/DT_TUPLE_FLOAT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_FLOAT}, /*.is_optional_=*/true},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"output_size", 1},
    {"scales", 2},
  },
  /*.func_impl_=*/gUpsampleNearest2DFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

UpsampleNearest3DGradFuncImpl gUpsampleNearest3DGradFuncImpl;
OpDef gUpsampleNearest3DGrad = {
  /*.name_=*/"UpsampleNearest3DGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"dy", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"input_size", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"output_size", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"scales", /*.arg_dtype_=*/DT_TUPLE_FLOAT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_FLOAT}, /*.is_optional_=*/true},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"dx", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"dy", 0},
    {"input_size", 1},
    {"output_size", 2},
    {"scales", 3},
  },
  /*.func_impl_=*/gUpsampleNearest3DGradFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

UpsampleNearest3DFuncImpl gUpsampleNearest3DFuncImpl;
OpDef gUpsampleNearest3D = {
  /*.name_=*/"UpsampleNearest3D",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"output_size", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"scales", /*.arg_dtype_=*/DT_TUPLE_FLOAT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_FLOAT}, /*.is_optional_=*/true},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"output_size", 1},
    {"scales", 2},
  },
  /*.func_impl_=*/gUpsampleNearest3DFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

UpsampleTrilinear3DGradFuncImpl gUpsampleTrilinear3DGradFuncImpl;
OpDef gUpsampleTrilinear3DGrad = {
  /*.name_=*/"UpsampleTrilinear3DGrad",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"dy", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"input_size", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"output_size", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"scales", /*.arg_dtype_=*/DT_TUPLE_FLOAT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_FLOAT}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"align_corners", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"dx", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"dy", 0},
    {"input_size", 1},
    {"output_size", 2},
    {"scales", 3},
    {"align_corners", 4},
  },
  /*.func_impl_=*/gUpsampleTrilinear3DGradFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

UpsampleTrilinear3DFuncImpl gUpsampleTrilinear3DFuncImpl;
OpDef gUpsampleTrilinear3D = {
  /*.name_=*/"UpsampleTrilinear3D",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"output_size", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"scales", /*.arg_dtype_=*/DT_TUPLE_FLOAT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_FLOAT}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"align_corners", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"output_size", 1},
    {"scales", 2},
    {"align_corners", 3},
  },
  /*.func_impl_=*/gUpsampleTrilinear3DFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

ViewFuncImpl gViewFuncImpl;
OpDef gView = {
  /*.name_=*/"View",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"shape", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"shape", 1},
  },
  /*.func_impl_=*/gViewFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/true,
};

ZerosLikeExtFuncImpl gZerosLikeExtFuncImpl;
OpDef gZerosLikeExt = {
  /*.name_=*/"ZerosLikeExt",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dtype", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"dtype_to_type_id", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"dtype", 1},
  },
  /*.func_impl_=*/gZerosLikeExtFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

ZerosLikeFuncImpl gZerosLikeFuncImpl;
OpDef gZerosLike = {
  /*.name_=*/"ZerosLike",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
  },
  /*.func_impl_=*/gZerosLikeFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

ZerosFuncImpl gZerosFuncImpl;
OpDef gZeros = {
  /*.name_=*/"Zeros",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"size", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_INT, DT_LIST_INT, DT_TENSOR}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dtype", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"dtype_to_type_id", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"size", 0},
    {"dtype", 1},
  },
  /*.func_impl_=*/gZerosFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

DynamicQuantExtFuncImpl gDynamicQuantExtFuncImpl;
OpDef gDynamicQuantExt = {
  /*.name_=*/"DynamicQuantExt",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"smooth_scales", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"scale", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"smooth_scales", 1},
  },
  /*.func_impl_=*/gDynamicQuantExtFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

FusedInferAttentionScoreFuncImpl gFusedInferAttentionScoreFuncImpl;
OpDef gFusedInferAttentionScore = {
  /*.name_=*/"FusedInferAttentionScore",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"query", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"key", /*.arg_dtype_=*/DT_TUPLE_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_TENSOR}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"value", /*.arg_dtype_=*/DT_TUPLE_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_TENSOR}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"pse_shift", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"attn_mask", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"actual_seq_lengths", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT, DT_TUPLE_INT}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"actual_seq_lengths_kv", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_INT, DT_TUPLE_INT}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"dequant_scale1", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"quant_scale1", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"dequant_scale2", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"quant_scale2", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"quant_offset2", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"antiquant_scale", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"antiquant_offset", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"block_table", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"query_padding_size", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"kv_padding_size", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"num_heads", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"scale_value", /*.arg_dtype_=*/DT_FLOAT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"pre_tokens", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"next_tokens", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"input_layout", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"str_to_enum", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"num_key_value_heads", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"sparse_mode", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"inner_precise", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"block_size", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"antiquant_mode", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"softmax_lse_flag", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"attention_out", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"softmax_lse", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"query", 0},
    {"key", 1},
    {"value", 2},
    {"pse_shift", 3},
    {"attn_mask", 4},
    {"actual_seq_lengths", 5},
    {"actual_seq_lengths_kv", 6},
    {"dequant_scale1", 7},
    {"quant_scale1", 8},
    {"dequant_scale2", 9},
    {"quant_scale2", 10},
    {"quant_offset2", 11},
    {"antiquant_scale", 12},
    {"antiquant_offset", 13},
    {"block_table", 14},
    {"query_padding_size", 15},
    {"kv_padding_size", 16},
    {"num_heads", 17},
    {"scale_value", 18},
    {"pre_tokens", 19},
    {"next_tokens", 20},
    {"input_layout", 21},
    {"num_key_value_heads", 22},
    {"sparse_mode", 23},
    {"inner_precise", 24},
    {"block_size", 25},
    {"antiquant_mode", 26},
    {"softmax_lse_flag", 27},
  },
  /*.func_impl_=*/gFusedInferAttentionScoreFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

GroupedMatmulFuncImpl gGroupedMatmulFuncImpl;
OpDef gGroupedMatmul = {
  /*.name_=*/"GroupedMatmul",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TUPLE_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_TENSOR}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"weight", /*.arg_dtype_=*/DT_TUPLE_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_TENSOR}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"bias", /*.arg_dtype_=*/DT_TUPLE_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_TENSOR}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"scale", /*.arg_dtype_=*/DT_TUPLE_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_TENSOR}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"offset", /*.arg_dtype_=*/DT_TUPLE_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_TENSOR}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"antiquant_scale", /*.arg_dtype_=*/DT_TUPLE_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_TENSOR}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"antiquant_offset", /*.arg_dtype_=*/DT_TUPLE_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{DT_LIST_TENSOR}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"group_list", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"split_item", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"group_type", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_TUPLE_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"weight", 1},
    {"bias", 2},
    {"scale", 3},
    {"offset", 4},
    {"antiquant_scale", 5},
    {"antiquant_offset", 6},
    {"group_list", 7},
    {"split_item", 8},
    {"group_type", 9},
  },
  /*.func_impl_=*/gGroupedMatmulFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

KVCacheScatterUpdateFuncImpl gKVCacheScatterUpdateFuncImpl;
OpDef gKVCacheScatterUpdate = {
  /*.name_=*/"KVCacheScatterUpdate",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"var", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"indices", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"updates", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"axis", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"reduce", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"str_to_enum", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"out", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"var", 0},
    {"indices", 1},
    {"updates", 2},
    {"axis", 3},
    {"reduce", 4},
  },
  /*.func_impl_=*/gKVCacheScatterUpdateFuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

MatmulBiasSplitOut2FuncImpl gMatmulBiasSplitOut2FuncImpl;
OpDef gMatmulBiasSplitOut2 = {
  /*.name_=*/"MatmulBiasSplitOut2",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"weight", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"reshape", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"bias", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output0", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"output1", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"weight", 1},
    {"reshape", 2},
    {"bias", 3},
  },
  /*.func_impl_=*/gMatmulBiasSplitOut2FuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

MatmulBiasSplitOut3FuncImpl gMatmulBiasSplitOut3FuncImpl;
OpDef gMatmulBiasSplitOut3 = {
  /*.name_=*/"MatmulBiasSplitOut3",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"weight", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"reshape", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"bias", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output0", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"output1", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"output2", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"weight", 1},
    {"reshape", 2},
    {"bias", 3},
  },
  /*.func_impl_=*/gMatmulBiasSplitOut3FuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

MatmulBiasSplitSiluOut2FuncImpl gMatmulBiasSplitSiluOut2FuncImpl;
OpDef gMatmulBiasSplitSiluOut2 = {
  /*.name_=*/"MatmulBiasSplitSiluOut2",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"weight", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"reshape", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"bias", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output0", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"output1", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"weight", 1},
    {"reshape", 2},
    {"bias", 3},
  },
  /*.func_impl_=*/gMatmulBiasSplitSiluOut2FuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

MatmulSplitOut2FuncImpl gMatmulSplitOut2FuncImpl;
OpDef gMatmulSplitOut2 = {
  /*.name_=*/"MatmulSplitOut2",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"weight", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"reshape", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output0", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"output1", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"weight", 1},
    {"reshape", 2},
  },
  /*.func_impl_=*/gMatmulSplitOut2FuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

MatmulSplitOut3FuncImpl gMatmulSplitOut3FuncImpl;
OpDef gMatmulSplitOut3 = {
  /*.name_=*/"MatmulSplitOut3",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"weight", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"reshape", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output0", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"output1", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"output2", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"weight", 1},
    {"reshape", 2},
  },
  /*.func_impl_=*/gMatmulSplitOut3FuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

MatmulSplitSiluOut2FuncImpl gMatmulSplitSiluOut2FuncImpl;
OpDef gMatmulSplitSiluOut2 = {
  /*.name_=*/"MatmulSplitSiluOut2",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"weight", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"reshape", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output0", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"output1", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"weight", 1},
    {"reshape", 2},
  },
  /*.func_impl_=*/gMatmulSplitSiluOut2FuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

MoeFinalizeRoutingFuncImpl gMoeFinalizeRoutingFuncImpl;
OpDef gMoeFinalizeRouting = {
  /*.name_=*/"MoeFinalizeRouting",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"expanded_x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"x1", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"x2", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"bias", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"scales", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"expanded_row_idx", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"expanded_expert_idx", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"expanded_x", 0},
    {"x1", 1},
    {"x2", 2},
    {"bias", 3},
    {"scales", 4},
    {"expanded_row_idx", 5},
    {"expanded_expert_idx", 6},
  },
  /*.func_impl_=*/gMoeFinalizeRoutingFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

QuantBatchMatmulFuncImpl gQuantBatchMatmulFuncImpl;
OpDef gQuantBatchMatmul = {
  /*.name_=*/"QuantBatchMatmul",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x1", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"x2", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"scale", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"offset", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"bias", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"transpose_x1", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"transpose_x2", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dtype", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"dtype_to_type_id", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {
    Signature("x1", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
     Signature("x2", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType),
     Signature("scale", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType1),
     Signature("offset", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType2),
     Signature("bias", SignatureEnumRW::kRWDefault,          SignatureEnumKind::kKindPositionalKeyword, nullptr, SignatureEnumDType::kDType3),
  },
  /*.indexes_ =*/ {
    {"x1", 0},
    {"x2", 1},
    {"scale", 2},
    {"offset", 3},
    {"bias", 4},
    {"transpose_x1", 5},
    {"transpose_x2", 6},
    {"dtype", 7},
  },
  /*.func_impl_=*/gQuantBatchMatmulFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

QuantV2FuncImpl gQuantV2FuncImpl;
OpDef gQuantV2 = {
  /*.name_=*/"QuantV2",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"scale", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"offset", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"sqrt_mode", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"rounding_mode", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"str_to_enum", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"dst_type", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"dtype_to_type_id", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"scale", 1},
    {"offset", 2},
    {"sqrt_mode", 3},
    {"rounding_mode", 4},
    {"dst_type", 5},
  },
  /*.func_impl_=*/gQuantV2FuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

QuantbatchmatmulSplitOut2FuncImpl gQuantbatchmatmulSplitOut2FuncImpl;
OpDef gQuantbatchmatmulSplitOut2 = {
  /*.name_=*/"QuantbatchmatmulSplitOut2",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"weight", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"reshape", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"bias", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"scale", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output0", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"output1", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"weight", 1},
    {"reshape", 2},
    {"bias", 3},
    {"scale", 4},
  },
  /*.func_impl_=*/gQuantbatchmatmulSplitOut2FuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

QuantbatchmatmulSplitOut3FuncImpl gQuantbatchmatmulSplitOut3FuncImpl;
OpDef gQuantbatchmatmulSplitOut3 = {
  /*.name_=*/"QuantbatchmatmulSplitOut3",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"weight", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"reshape", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"bias", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"scale", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output0", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"output1", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"output2", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"weight", 1},
    {"reshape", 2},
    {"bias", 3},
    {"scale", 4},
  },
  /*.func_impl_=*/gQuantbatchmatmulSplitOut3FuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

QuantbatchmatmulSplitSiluOut2FuncImpl gQuantbatchmatmulSplitSiluOut2FuncImpl;
OpDef gQuantbatchmatmulSplitSiluOut2 = {
  /*.name_=*/"QuantbatchmatmulSplitSiluOut2",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"input", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"weight", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"reshape", /*.arg_dtype_=*/DT_TUPLE_INT, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"bias", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"scale", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"output0", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1},
    {/*.arg_name_=*/"output1", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"input", 0},
    {"weight", 1},
    {"reshape", 2},
    {"bias", 3},
    {"scale", 4},
  },
  /*.func_impl_=*/gQuantbatchmatmulSplitSiluOut2FuncImpl,
  /*.enable_dispatch_ =*/false,
  /*.is_view_ =*/false,
};

WeightQuantBatchMatmulFuncImpl gWeightQuantBatchMatmulFuncImpl;
OpDef gWeightQuantBatchMatmul = {
  /*.name_=*/"WeightQuantBatchMatmul",
  /*.args_=*/ {
    
        {/*.arg_name_=*/"x", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"weight", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"antiquant_scale", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"antiquant_offset", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"quant_scale", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"quant_offset", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"bias", /*.arg_dtype_=*/DT_TENSOR, /*.as_init_arg_=*/0, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/true},
        {/*.arg_name_=*/"transpose_x", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"transpose_weight", /*.arg_dtype_=*/DT_BOOL, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
        {/*.arg_name_=*/"antiquant_group_size", /*.arg_dtype_=*/DT_INT, /*.as_init_arg_=*/1, /*.arg_handler_=*/"", /*.cast_dtype_ =*/{}, /*.is_optional_=*/false},
  },
  /* .returns_ = */ {
    {/*.arg_name_=*/"y", /*.arg_dtype_=*/DT_TENSOR,
                /*.inplace_input_index_=*/-1}, 
  },
  /*.signatures_ =*/ {

  },
  /*.indexes_ =*/ {
    {"x", 0},
    {"weight", 1},
    {"antiquant_scale", 2},
    {"antiquant_offset", 3},
    {"quant_scale", 4},
    {"quant_offset", 5},
    {"bias", 6},
    {"transpose_x", 7},
    {"transpose_weight", 8},
    {"antiquant_group_size", 9},
  },
  /*.func_impl_=*/gWeightQuantBatchMatmulFuncImpl,
  /*.enable_dispatch_ =*/true,
  /*.is_view_ =*/false,
};

std::unordered_map<std::string, OpDefPtr> gOpDefTable = {
  {"ACosGrad", &gACosGrad},
  {"AbsGrad", &gAbsGrad},
  {"Abs", &gAbs},
  {"ACos", &gACos},
  {"AcoshGrad", &gAcoshGrad},
  {"Acosh", &gAcosh},
  {"AdamWeightDecay", &gAdamWeightDecay},
  {"AdamW", &gAdamW},
  {"AddExt", &gAddExt},
  {"AddLayerNormV2", &gAddLayerNormV2},
  {"Add", &gAdd},
  {"Addcdiv", &gAddcdiv},
  {"Addcmul", &gAddcmul},
  {"Addmm", &gAddmm},
  {"AddN", &gAddN},
  {"Angle", &gAngle},
  {"ApplyCamePart1", &gApplyCamePart1},
  {"ApplyCamePart2", &gApplyCamePart2},
  {"ApplyCamePart3", &gApplyCamePart3},
  {"ApplyCamePart4", &gApplyCamePart4},
  {"ApplyRotaryPosEmb", &gApplyRotaryPosEmb},
  {"Arange", &gArange},
  {"ArgMaxExt", &gArgMaxExt},
  {"Argmax", &gArgmax},
  {"ArgMaxWithValue", &gArgMaxWithValue},
  {"Argmin", &gArgmin},
  {"ArgMinWithValue", &gArgMinWithValue},
  {"AsinGrad", &gAsinGrad},
  {"Asin", &gAsin},
  {"AsinhGrad", &gAsinhGrad},
  {"Asinh", &gAsinh},
  {"AssignAdd", &gAssignAdd},
  {"Assign", &gAssign},
  {"Atan2Ext", &gAtan2Ext},
  {"Atan2", &gAtan2},
  {"AtanGrad", &gAtanGrad},
  {"Atan", &gAtan},
  {"Atanh", &gAtanh},
  {"AvgPool2DGrad", &gAvgPool2DGrad},
  {"AvgPool2D", &gAvgPool2D},
  {"AvgPoolGrad", &gAvgPoolGrad},
  {"AvgPool", &gAvgPool},
  {"BatchMatMul", &gBatchMatMul},
  {"BatchNormExt", &gBatchNormExt},
  {"BatchNormGradExt", &gBatchNormGradExt},
  {"BatchNormGradGrad", &gBatchNormGradGrad},
  {"BatchNormGrad", &gBatchNormGrad},
  {"BatchNormGradWithActivation", &gBatchNormGradWithActivation},
  {"BatchNorm", &gBatchNorm},
  {"BatchNormWithActivation", &gBatchNormWithActivation},
  {"BatchNormWithAddAndActivation", &gBatchNormWithAddAndActivation},
  {"Betainc", &gBetainc},
  {"BiasAddGrad", &gBiasAddGrad},
  {"BiasAdd", &gBiasAdd},
  {"BinaryCrossEntropyGrad", &gBinaryCrossEntropyGrad},
  {"BinaryCrossEntropy", &gBinaryCrossEntropy},
  {"BinaryCrossEntropyWithLogitsBackward", &gBinaryCrossEntropyWithLogitsBackward},
  {"BCEWithLogitsLoss", &gBCEWithLogitsLoss},
  {"BatchMatMulExt", &gBatchMatMulExt},
  {"BoolNot", &gBoolNot},
  {"BroadcastTo", &gBroadcastTo},
  {"Cast", &gCast},
  {"Ceil", &gCeil},
  {"CeLU", &gCeLU},
  {"CholeskyGrad", &gCholeskyGrad},
  {"CholeskyInverse", &gCholeskyInverse},
  {"Cholesky", &gCholesky},
  {"Chunk", &gChunk},
  {"ClampScalar", &gClampScalar},
  {"ClampTensor", &gClampTensor},
  {"Col2ImExt", &gCol2ImExt},
  {"Col2ImGrad", &gCol2ImGrad},
  {"Complex", &gComplex},
  {"Concat", &gConcat},
  {"Conj", &gConj},
  {"ConstantPadND", &gConstantPadND},
  {"Contiguous", &gContiguous},
  {"ConvolutionGrad", &gConvolutionGrad},
  {"Convolution", &gConvolution},
  {"Copy", &gCopy},
  {"Correlate", &gCorrelate},
  {"Cos", &gCos},
  {"Cosh", &gCosh},
  {"CumProd", &gCumProd},
  {"CumSum", &gCumSum},
  {"Cummax", &gCummax},
  {"Cummin", &gCummin},
  {"CumsumExt", &gCumsumExt},
  {"DCT", &gDCT},
  {"DecoderKVCache", &gDecoderKVCache},
  {"Dense", &gDense},
  {"Diag", &gDiag},
  {"Diagonal", &gDiagonal},
  {"Div", &gDiv},
  {"DivMod", &gDivMod},
  {"Dot", &gDot},
  {"DropoutDoMaskExt", &gDropoutDoMaskExt},
  {"DropoutExt", &gDropoutExt},
  {"DropoutGenMaskExt", &gDropoutGenMaskExt},
  {"DropoutGradExt", &gDropoutGradExt},
  {"Dropout", &gDropout},
  {"Eig", &gEig},
  {"EluExt", &gEluExt},
  {"EluGradExt", &gEluGradExt},
  {"EluGrad", &gEluGrad},
  {"Elu", &gElu},
  {"EmbeddingDenseBackward", &gEmbeddingDenseBackward},
  {"Embedding", &gEmbedding},
  {"Equal", &gEqual},
  {"Erf", &gErf},
  {"Erfc", &gErfc},
  {"Erfinv", &gErfinv},
  {"Exp", &gExp},
  {"ExpandDims", &gExpandDims},
  {"Expm1", &gExpm1},
  {"ExtractImagePatches", &gExtractImagePatches},
  {"Eye", &gEye},
  {"FastGeLUGrad", &gFastGeLUGrad},
  {"FastGeLU", &gFastGeLU},
  {"FFNExt", &gFFNExt},
  {"FFT2", &gFFT2},
  {"FFT", &gFFT},
  {"FFTShapeCopy", &gFFTShapeCopy},
  {"FFTWithSize", &gFFTWithSize},
  {"FFTN", &gFFTN},
  {"FFTShift", &gFFTShift},
  {"FillScalar", &gFillScalar},
  {"FillTensor", &gFillTensor},
  {"FlashAttentionScoreGrad", &gFlashAttentionScoreGrad},
  {"FlashAttentionScore", &gFlashAttentionScore},
  {"FlattenExt", &gFlattenExt},
  {"Flatten", &gFlatten},
  {"FloorDiv", &gFloorDiv},
  {"FloorMod", &gFloorMod},
  {"Floor", &gFloor},
  {"GatherDGradV2", &gGatherDGradV2},
  {"GatherD", &gGatherD},
  {"GatherNd", &gGatherNd},
  {"Gather", &gGather},
  {"Gcd", &gGcd},
  {"GeLUGrad", &gGeLUGrad},
  {"GeLU", &gGeLU},
  {"Generator", &gGenerator},
  {"Geqrf", &gGeqrf},
  {"GreaterEqual", &gGreaterEqual},
  {"Greater", &gGreater},
  {"GridSampler2DGrad", &gGridSampler2DGrad},
  {"GridSampler2D", &gGridSampler2D},
  {"GridSampler3DGrad", &gGridSampler3DGrad},
  {"GridSampler3D", &gGridSampler3D},
  {"GroupNormGrad", &gGroupNormGrad},
  {"GroupNorm", &gGroupNorm},
  {"HShrinkGrad", &gHShrinkGrad},
  {"HShrink", &gHShrink},
  {"HSigmoidGrad", &gHSigmoidGrad},
  {"HSigmoid", &gHSigmoid},
  {"HSwishGrad", &gHSwishGrad},
  {"HSwish", &gHSwish},
  {"Identity", &gIdentity},
  {"IFFT2", &gIFFT2},
  {"IFFT", &gIFFT},
  {"IFFTN", &gIFFTN},
  {"IFFTShift", &gIFFTShift},
  {"Im2ColExt", &gIm2ColExt},
  {"IndexAddExt", &gIndexAddExt},
  {"IndexSelect", &gIndexSelect},
  {"IRFFTGrad", &gIRFFTGrad},
  {"IRFFT", &gIRFFT},
  {"IsClose", &gIsClose},
  {"IsFinite", &gIsFinite},
  {"LayerNormExt", &gLayerNormExt},
  {"LayerNormGradExt", &gLayerNormGradExt},
  {"LayerNormGradGrad", &gLayerNormGradGrad},
  {"LayerNormGrad", &gLayerNormGrad},
  {"LayerNormGradV3", &gLayerNormGradV3},
  {"LayerNorm", &gLayerNorm},
  {"LayerNormV3", &gLayerNormV3},
  {"LeakyReLUExt", &gLeakyReLUExt},
  {"LeakyReLUGradExt", &gLeakyReLUGradExt},
  {"LessEqual", &gLessEqual},
  {"Less", &gLess},
  {"LinSpaceExt", &gLinSpaceExt},
  {"LinSpace", &gLinSpace},
  {"ListToTuple", &gListToTuple},
  {"Log1p", &gLog1p},
  {"LogMatrixDeterminant", &gLogMatrixDeterminant},
  {"Log", &gLog},
  {"LogSoftmaxGrad", &gLogSoftmaxGrad},
  {"LogSoftmax", &gLogSoftmax},
  {"LogicalAnd", &gLogicalAnd},
  {"LogicalNot", &gLogicalNot},
  {"LogicalOr", &gLogicalOr},
  {"LogicalXor", &gLogicalXor},
  {"LogitGrad", &gLogitGrad},
  {"Logit", &gLogit},
  {"MaskedFill", &gMaskedFill},
  {"MatMulExt", &gMatMulExt},
  {"MatMul", &gMatMul},
  {"MatrixDeterminant", &gMatrixDeterminant},
  {"MatrixExp", &gMatrixExp},
  {"MatrixInverseExt", &gMatrixInverseExt},
  {"Max", &gMax},
  {"MaxPoolGradWithIndices", &gMaxPoolGradWithIndices},
  {"MaxPoolGradWithMask", &gMaxPoolGradWithMask},
  {"MaxPoolWithIndices", &gMaxPoolWithIndices},
  {"MaxPoolWithMask", &gMaxPoolWithMask},
  {"MaximumGradGrad", &gMaximumGradGrad},
  {"MaximumGrad", &gMaximumGrad},
  {"Maximum", &gMaximum},
  {"MeanExt", &gMeanExt},
  {"Min", &gMin},
  {"MinimumGrad", &gMinimumGrad},
  {"Minimum", &gMinimum},
  {"Mul", &gMul},
  {"Mv", &gMv},
  {"NanToNum", &gNanToNum},
  {"Neg", &gNeg},
  {"NextAfter", &gNextAfter},
  {"NLLLossGrad", &gNLLLossGrad},
  {"NLLLoss", &gNLLLoss},
  {"NonZeroExt", &gNonZeroExt},
  {"NonZero", &gNonZero},
  {"Norm", &gNorm},
  {"NormalFloatFloat", &gNormalFloatFloat},
  {"NormalFloatTensor", &gNormalFloatTensor},
  {"NormalTensorFloat", &gNormalTensorFloat},
  {"NormalTensorTensor", &gNormalTensorTensor},
  {"NotEqual", &gNotEqual},
  {"NPUClearFloatStatusV2", &gNPUClearFloatStatusV2},
  {"NPUGetFloatStatusV2", &gNPUGetFloatStatusV2},
  {"OneHotExt", &gOneHotExt},
  {"OneHot", &gOneHot},
  {"OnesLikeExt", &gOnesLikeExt},
  {"OnesLike", &gOnesLike},
  {"Ones", &gOnes},
  {"PagedAttentionMask", &gPagedAttentionMask},
  {"PagedAttention", &gPagedAttention},
  {"Pow", &gPow},
  {"PReLUGrad", &gPReLUGrad},
  {"PReLU", &gPReLU},
  {"ProdExt", &gProdExt},
  {"PromptKVCache", &gPromptKVCache},
  {"Qr", &gQr},
  {"RandExt", &gRandExt},
  {"RandLikeExt", &gRandLikeExt},
  {"RandpermV2", &gRandpermV2},
  {"Range", &gRange},
  {"Rank", &gRank},
  {"RealDiv", &gRealDiv},
  {"Real", &gReal},
  {"ReciprocalGrad", &gReciprocalGrad},
  {"Reciprocal", &gReciprocal},
  {"ReduceAll", &gReduceAll},
  {"ReduceAny", &gReduceAny},
  {"ReduceMax", &gReduceMax},
  {"ReduceMean", &gReduceMean},
  {"ReduceMin", &gReduceMin},
  {"ReduceProd", &gReduceProd},
  {"ReduceStd", &gReduceStd},
  {"ReduceSum", &gReduceSum},
  {"ReflectionPad1DGrad", &gReflectionPad1DGrad},
  {"ReflectionPad1D", &gReflectionPad1D},
  {"ReflectionPad2DGrad", &gReflectionPad2DGrad},
  {"ReflectionPad2D", &gReflectionPad2D},
  {"ReflectionPad3DGrad", &gReflectionPad3DGrad},
  {"ReflectionPad3D", &gReflectionPad3D},
  {"ReLU6Grad", &gReLU6Grad},
  {"ReLU6", &gReLU6},
  {"ReluGrad", &gReluGrad},
  {"ReLU", &gReLU},
  {"RepeatInterleaveGrad", &gRepeatInterleaveGrad},
  {"RepeatInterleaveInt", &gRepeatInterleaveInt},
  {"RepeatInterleaveTensor", &gRepeatInterleaveTensor},
  {"ReplicationPad1DGrad", &gReplicationPad1DGrad},
  {"ReplicationPad1D", &gReplicationPad1D},
  {"ReplicationPad2DGrad", &gReplicationPad2DGrad},
  {"ReplicationPad2D", &gReplicationPad2D},
  {"ReplicationPad3DGrad", &gReplicationPad3DGrad},
  {"ReplicationPad3D", &gReplicationPad3D},
  {"ReshapeAndCache", &gReshapeAndCache},
  {"Reshape", &gReshape},
  {"ResizeBicubicGrad", &gResizeBicubicGrad},
  {"ResizeBicubic", &gResizeBicubic},
  {"ResizeBilinearGrad", &gResizeBilinearGrad},
  {"ResizeBilinearV2", &gResizeBilinearV2},
  {"ResizeD", &gResizeD},
  {"ResizeLinear1DGrad", &gResizeLinear1DGrad},
  {"ResizeLinear1D", &gResizeLinear1D},
  {"ResizeNearestNeighborGrad", &gResizeNearestNeighborGrad},
  {"ResizeNearestNeighbor", &gResizeNearestNeighbor},
  {"ResizeNearestNeighborV2Grad", &gResizeNearestNeighborV2Grad},
  {"ResizeNearestNeighborV2", &gResizeNearestNeighborV2},
  {"ReverseV2", &gReverseV2},
  {"RFFTGrad", &gRFFTGrad},
  {"RFFT", &gRFFT},
  {"RightShift", &gRightShift},
  {"RmsNormGrad", &gRmsNormGrad},
  {"RmsNorm", &gRmsNorm},
  {"Roll", &gRoll},
  {"Round", &gRound},
  {"RsqrtGrad", &gRsqrtGrad},
  {"Rsqrt", &gRsqrt},
  {"ScalarAdd", &gScalarAdd},
  {"ScalarBool", &gScalarBool},
  {"ScalarCast", &gScalarCast},
  {"ScalarDiv", &gScalarDiv},
  {"ScalarEq", &gScalarEq},
  {"ScalarFloorDiv", &gScalarFloorDiv},
  {"ScalarGe", &gScalarGe},
  {"ScalarGt", &gScalarGt},
  {"ScalarLe", &gScalarLe},
  {"ScalarLog", &gScalarLog},
  {"ScalarLt", &gScalarLt},
  {"ScalarMod", &gScalarMod},
  {"ScalarMul", &gScalarMul},
  {"ScalarPow", &gScalarPow},
  {"ScalarSub", &gScalarSub},
  {"ScalarToTensor", &gScalarToTensor},
  {"ScalarUadd", &gScalarUadd},
  {"ScalarUsub", &gScalarUsub},
  {"ScatterAddExt", &gScatterAddExt},
  {"ScatterNd", &gScatterNd},
  {"Scatter", &gScatter},
  {"SearchSorted", &gSearchSorted},
  {"Select", &gSelect},
  {"SequenceConcat", &gSequenceConcat},
  {"Shape", &gShape},
  {"SigmoidGrad", &gSigmoidGrad},
  {"Sigmoid", &gSigmoid},
  {"Sign", &gSign},
  {"SiLUGrad", &gSiLUGrad},
  {"SiLU", &gSiLU},
  {"Sin", &gSin},
  {"Sinc", &gSinc},
  {"Sinh", &gSinh},
  {"SliceExt", &gSliceExt},
  {"SoftmaxBackward", &gSoftmaxBackward},
  {"Softmax", &gSoftmax},
  {"SoftplusExt", &gSoftplusExt},
  {"SoftplusGradExt", &gSoftplusGradExt},
  {"SolveTriangular", &gSolveTriangular},
  {"SortExt", &gSortExt},
  {"Split", &gSplit},
  {"SplitTensor", &gSplitTensor},
  {"SplitWithSize", &gSplitWithSize},
  {"SqrtGrad", &gSqrtGrad},
  {"Sqrt", &gSqrt},
  {"Square", &gSquare},
  {"StackExt", &gStackExt},
  {"StridedSlice", &gStridedSlice},
  {"SubExt", &gSubExt},
  {"Sub", &gSub},
  {"SumExt", &gSumExt},
  {"TanhGrad", &gTanhGrad},
  {"Tanh", &gTanh},
  {"TensorCopySlices", &gTensorCopySlices},
  {"TensorShape", &gTensorShape},
  {"Tile", &gTile},
  {"TopkExt", &gTopkExt},
  {"TopKRouter", &gTopKRouter},
  {"Trace", &gTrace},
  {"Transpose", &gTranspose},
  {"Triu", &gTriu},
  {"TupleToList", &gTupleToList},
  {"TupleToTensor", &gTupleToTensor},
  {"UniformExt", &gUniformExt},
  {"Unique2", &gUnique2},
  {"UniqueDim", &gUniqueDim},
  {"UnsortedSegmentSum", &gUnsortedSegmentSum},
  {"UnstackExt", &gUnstackExt},
  {"UpsampleBilinear2DGrad", &gUpsampleBilinear2DGrad},
  {"UpsampleBilinear2D", &gUpsampleBilinear2D},
  {"UpsampleLinear1DGrad", &gUpsampleLinear1DGrad},
  {"UpsampleLinear1D", &gUpsampleLinear1D},
  {"UpsampleNearest1DGrad", &gUpsampleNearest1DGrad},
  {"UpsampleNearest1D", &gUpsampleNearest1D},
  {"UpsampleNearest2DGrad", &gUpsampleNearest2DGrad},
  {"UpsampleNearest2D", &gUpsampleNearest2D},
  {"UpsampleNearest3DGrad", &gUpsampleNearest3DGrad},
  {"UpsampleNearest3D", &gUpsampleNearest3D},
  {"UpsampleTrilinear3DGrad", &gUpsampleTrilinear3DGrad},
  {"UpsampleTrilinear3D", &gUpsampleTrilinear3D},
  {"View", &gView},
  {"ZerosLikeExt", &gZerosLikeExt},
  {"ZerosLike", &gZerosLike},
  {"Zeros", &gZeros},
  {"DynamicQuantExt", &gDynamicQuantExt},
  {"FusedInferAttentionScore", &gFusedInferAttentionScore},
  {"GroupedMatmul", &gGroupedMatmul},
  {"KVCacheScatterUpdate", &gKVCacheScatterUpdate},
  {"MatmulBiasSplitOut2", &gMatmulBiasSplitOut2},
  {"MatmulBiasSplitOut3", &gMatmulBiasSplitOut3},
  {"MatmulBiasSplitSiluOut2", &gMatmulBiasSplitSiluOut2},
  {"MatmulSplitOut2", &gMatmulSplitOut2},
  {"MatmulSplitOut3", &gMatmulSplitOut3},
  {"MatmulSplitSiluOut2", &gMatmulSplitSiluOut2},
  {"MoeFinalizeRouting", &gMoeFinalizeRouting},
  {"QuantBatchMatmul", &gQuantBatchMatmul},
  {"QuantV2", &gQuantV2},
  {"QuantbatchmatmulSplitOut2", &gQuantbatchmatmulSplitOut2},
  {"QuantbatchmatmulSplitOut3", &gQuantbatchmatmulSplitOut3},
  {"QuantbatchmatmulSplitSiluOut2", &gQuantbatchmatmulSplitSiluOut2},
  {"WeightQuantBatchMatmul", &gWeightQuantBatchMatmul},
};
}  // namespace mindspore::ops
