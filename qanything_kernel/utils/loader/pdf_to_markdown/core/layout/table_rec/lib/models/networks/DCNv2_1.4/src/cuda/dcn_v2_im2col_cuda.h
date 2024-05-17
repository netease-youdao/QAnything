
/*!
 ******************* BEGIN Caffe Copyright Notice and Disclaimer ****************
 *
 * COPYRIGHT
 *
 * All contributions by the University of California:
 * Copyright (c) 2014-2017 The Regents of the University of California (Regents)
 * All rights reserved.
 *
 * All other contributions:
 * Copyright (c) 2014-2017, the respective contributors
 * All rights reserved.
 *
 * Caffe uses a shared copyright model: each contributor holds copyright over
 * their contributions to Caffe. The project versioning records all such
 * contribution and copyright details. If a contributor wants to further mark
 * their specific copyright on a particular contribution, they should indicate
 * their copyright solely in the commit message of the change when it is
 * committed.
 *
 * LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * CONTRIBUTION AGREEMENT
 *
 * By contributing to the BVLC/caffe repository through pull-request, comment,
 * or otherwise, the contributor releases their content to the
 * license and copyright terms herein.
 *
 ***************** END Caffe Copyright Notice and Disclaimer ********************
 *
 * Copyright (c) 2018 Microsoft
 * Licensed under The MIT License [see LICENSE for details]
 * \file modulated_deformable_im2col.h
 * \brief Function definitions of converting an image to
 * column matrix based on kernel, padding, dilation, and offset.
 * These functions are mainly used in deformable convolution operators.
 * \ref: https://arxiv.org/abs/1811.11168
 * \author Yuwen Xiong, Haozhi Qi, Jifeng Dai, Xizhou Zhu, Han Hu
 */

/***************** Adapted by Charles Shang *********************/

#ifndef DCN_V2_IM2COL_CUDA
#define DCN_V2_IM2COL_CUDA

#ifdef __cplusplus
extern "C"
{
#endif

  void modulated_deformable_im2col_cuda(cudaStream_t stream,
                                        const float *data_im, const float *data_offset, const float *data_mask,
                                        const int batch_size, const int channels, const int height_im, const int width_im,
                                        const int height_col, const int width_col, const int kernel_h, const int kenerl_w,
                                        const int pad_h, const int pad_w, const int stride_h, const int stride_w,
                                        const int dilation_h, const int dilation_w,
                                        const int deformable_group, float *data_col);

  void modulated_deformable_col2im_cuda(cudaStream_t stream,
                                        const float *data_col, const float *data_offset, const float *data_mask,
                                        const int batch_size, const int channels, const int height_im, const int width_im,
                                        const int height_col, const int width_col, const int kernel_h, const int kenerl_w,
                                        const int pad_h, const int pad_w, const int stride_h, const int stride_w,
                                        const int dilation_h, const int dilation_w,
                                        const int deformable_group, float *grad_im);

  void modulated_deformable_col2im_coord_cuda(cudaStream_t stream,
                                         const float *data_col, const float *data_im, const float *data_offset, const float *data_mask,
                                         const int batch_size, const int channels, const int height_im, const int width_im,
                                         const int height_col, const int width_col, const int kernel_h, const int kenerl_w,
                                         const int pad_h, const int pad_w, const int stride_h, const int stride_w,
                                         const int dilation_h, const int dilation_w,
                                         const int deformable_group,
                                         float *grad_offset, float *grad_mask);

#ifdef __cplusplus
}
#endif

#endif