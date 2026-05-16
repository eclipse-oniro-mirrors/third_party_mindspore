/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "nnacl_c/base/depth_to_space_base.h"
#include "nnacl_c/op_base.h"
#include "nnacl_c/errorcode.h"

void DepthToSpaceForNHWC(const void *input, void *output, const int *in_shape, const DepthToSpaceArgs *param) {
  int32_t block_size = param->block_size_;
  int32_t in_shape_dim2 = in_shape[2];
  int32_t in_shape_dim1 = in_shape[1];
  size_t copy_size = (size_t)block_size * param->out_stride_dim2_ * param->data_type_size_;
  for (int i = 0; i < in_shape[0]; ++i) {
    int64_t in_offset_n = i * param->in_stride_dim0_;
    int64_t out_offset_n = i * param->out_stride_dim0_;
    for (int j = 0; j < in_shape_dim1; ++j) {
      int64_t in_offset_h = in_offset_n + j * param->in_stride_dim1_;
      int64_t out_offset_h = out_offset_n + j * block_size * param->out_stride_dim1_;
      for (int k = 0; k < in_shape_dim2; ++k) {
        int64_t in_offset_w = in_offset_h + k * param->in_stride_dim2_;
        int64_t out_offset_w = out_offset_h + k * block_size * param->out_stride_dim2_;
        for (int l = 0; l < block_size; ++l) {
          int64_t out_offset = (out_offset_w + l * param->out_stride_dim1_) * param->data_type_size_;
          int64_t in_offset = (in_offset_w + l * block_size * param->out_stride_dim2_) * param->data_type_size_;
          memcpy((int8_t *)output + out_offset, (int8_t *)input + in_offset, copy_size);
        }
      }
    }
  }
}

void DepthToSpaceCRDForNHWC(const void *input, void *output, const int *in_shape, const DepthToSpaceArgs *param) {
  int32_t block_size = param->block_size_;
  int32_t in_shape_dim3 = in_shape[3];
  int32_t in_shape_dim2 = in_shape[2];
  int32_t in_shape_dim1 = in_shape[1];
  size_t copy_size = param->data_type_size_;
  for (int i = 0; i < in_shape[0]; ++i) {
    int64_t in_offset_n = i * param->in_stride_dim0_;
    int64_t out_offset_n = i * param->out_stride_dim0_;
    for (int j = 0; j < in_shape_dim1; ++j) {
      int64_t in_offset_h = in_offset_n + j * param->in_stride_dim1_;
      int64_t out_offset_h = out_offset_n + j * block_size * param->out_stride_dim1_;
      for (int k = 0; k < in_shape_dim2; ++k) {
        int64_t in_offset_w = in_offset_h + k * param->in_stride_dim2_;
        int64_t out_offset_w = out_offset_h + k * block_size * param->out_stride_dim2_;
        for (int l = 0; l < in_shape_dim3; ++l) {
          int64_t offset = l % (block_size * block_size);
          int64_t out_offset_c =
            out_offset_w +
            offset / block_size * block_size * in_shape_dim2 * in_shape_dim3 / (block_size * block_size) +
            offset % block_size * in_shape_dim3 / (block_size * block_size);
          int64_t out_offset = (out_offset_c + l / (block_size * block_size)) * param->data_type_size_;
          int64_t in_offset = (in_offset_w + l) * param->data_type_size_;
          memcpy((int8_t *)output + out_offset, (int8_t *)input + in_offset, copy_size);
        }
      }
    }
  }
}

inline int getInputIndex(int b, int c, int h, int w, const DepthToSpaceArgs *param) {
  return b * param->in_stride_dim0_ + c * param->in_stride_dim1_ + h * param->in_stride_dim2_ + w;
}

inline int getOuputIndex(int b, int c, int h, int w, const DepthToSpaceArgs *param) {
  return b * param->out_stride_dim0_ + c * param->out_stride_dim1_ + h * param->out_stride_dim2_ + w;
}

void DepthToSpaceCRDForNCHWFP32(const float *input, float *output, const int *in_shape, int start_h, int end_h,
                                const DepthToSpaceArgs *param) {
  int b = in_shape[kNCHW_N];
  int c = in_shape[kNCHW_C];
  int w = in_shape[kNCHW_W];
  int32_t blocksize = param->block_size_;
  const int blocksize_sq = blocksize * blocksize;
  // output channel
  const int c_out = c / blocksize_sq;
  for (int batch = 0; batch < b; ++batch) {
    for (int c_idx = 0; c_idx < c_out; ++c_idx) {
      int c_stride = c_idx * blocksize_sq;
      for (int height = start_h; height < end_h; ++height) {
        int out_h_stride = height * blocksize;
        for (int bs1 = 0; bs1 < blocksize; ++bs1) {
          int bs1_stride = bs1 * blocksize;
          for (int width = 0; width < w; ++width) {
            int out_width_stride = width * blocksize;
            for (int bs2 = 0; bs2 < blocksize; ++bs2) {
              int input_c = c_stride + bs1_stride + bs2;
              int input_idx = getInputIndex(batch, input_c, height, width, param);

              int output_h = out_h_stride + bs1;
              int output_w = out_width_stride + bs2;
              int output_idx = getOuputIndex(batch, c_idx, output_h, output_w, param);

              *(output + output_idx) = *(input + input_idx);
            }
          }
        }
      }
    }
  }
}

#ifdef ENABLE_FP16
void DepthToSpaceCRDForNCHWFP16(const float16_t *input, float16_t *output, const int *in_shape, int start_h, int end_h,
                                const DepthToSpaceArgs *param) {
  int b = in_shape[kNCHW_N];
  int c = in_shape[kNCHW_C];
  int w = in_shape[kNCHW_W];
  int32_t blocksize = param->block_size_;
  const int blocksize_sq = blocksize * blocksize;
  // output channel
  const int c_out = c / blocksize_sq;
  for (int batch = 0; batch < b; ++batch) {
    for (int c_idx = 0; c_idx < c_out; ++c_idx) {
      int c_stride = c_idx * blocksize_sq;
      for (int height = start_h; height < end_h; ++height) {
        int out_h_stride = height * blocksize;
        for (int bs1 = 0; bs1 < blocksize; ++bs1) {
          int bs1_stride = bs1 * blocksize;
          for (int width = 0; width < w; ++width) {
            int out_width_stride = width * blocksize;
            for (int bs2 = 0; bs2 < blocksize; ++bs2) {
              int input_c = c_stride + bs1_stride + bs2;
              int input_idx = getInputIndex(batch, input_c, height, width, param);

              int output_h = out_h_stride + bs1;
              int output_w = out_width_stride + bs2;
              int output_idx = getOuputIndex(batch, c_idx, output_h, output_w, param);

              *(output + output_idx) = *(input + input_idx);
            }
          }
        }
      }
    }
  }
}
#endif

void DepthToSpaceCRDForNCHW(const void *input, void *output, const int *in_shape, int start_h, int end_h,
                            const DepthToSpaceArgs *param) {
  if (param->data_type_size_ == Num4) {
    DepthToSpaceCRDForNCHWFP32((const float *)input, (float *)output, in_shape, start_h, end_h, param);
  } else if (param->data_type_size_ == Num2) {
#ifdef ENABLE_FP16
    DepthToSpaceCRDForNCHWFP16((const float16_t *)input, (float16_t *)output, in_shape, start_h, end_h, param);
#endif
  }
}