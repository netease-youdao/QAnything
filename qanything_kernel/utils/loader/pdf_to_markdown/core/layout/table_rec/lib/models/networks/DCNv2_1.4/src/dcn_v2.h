#pragma once

#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif

at::Tensor
dcn_v2_forward(const at::Tensor &input,
               const at::Tensor &weight,
               const at::Tensor &bias,
               const at::Tensor &offset,
               const at::Tensor &mask,
               const int kernel_h,
               const int kernel_w,
               const int stride_h,
               const int stride_w,
               const int pad_h,
               const int pad_w,
               const int dilation_h,
               const int dilation_w,
               const int deformable_group)
{
    if (input.type().is_cuda())
    {
#ifdef WITH_CUDA
        return dcn_v2_cuda_forward(input, weight, bias, offset, mask,
                                   kernel_h, kernel_w,
                                   stride_h, stride_w,
                                   pad_h, pad_w,
                                   dilation_h, dilation_w,
                                   deformable_group);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    else{
        return dcn_v2_cpu_forward(input, weight, bias, offset, mask,
                                   kernel_h, kernel_w,
                                   stride_h, stride_w,
                                   pad_h, pad_w,
                                   dilation_h, dilation_w,
                                   deformable_group);
    }
}

std::vector<at::Tensor>
dcn_v2_backward(const at::Tensor &input,
                const at::Tensor &weight,
                const at::Tensor &bias,
                const at::Tensor &offset,
                const at::Tensor &mask,
                const at::Tensor &grad_output,
                int kernel_h, int kernel_w,
                int stride_h, int stride_w,
                int pad_h, int pad_w,
                int dilation_h, int dilation_w,
                int deformable_group)
{
    if (input.type().is_cuda())
    {
#ifdef WITH_CUDA
        return dcn_v2_cuda_backward(input,
                                    weight,
                                    bias,
                                    offset,
                                    mask,
                                    grad_output,
                                    kernel_h, kernel_w,
                                    stride_h, stride_w,
                                    pad_h, pad_w,
                                    dilation_h, dilation_w,
                                    deformable_group);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    else{
        return dcn_v2_cpu_backward(input,
                                    weight,
                                    bias,
                                    offset,
                                    mask,
                                    grad_output,
                                    kernel_h, kernel_w,
                                    stride_h, stride_w,
                                    pad_h, pad_w,
                                    dilation_h, dilation_w,
                                    deformable_group);
    }
}

std::tuple<at::Tensor, at::Tensor>
dcn_v2_psroi_pooling_forward(const at::Tensor &input,
                             const at::Tensor &bbox,
                             const at::Tensor &trans,
                             const int no_trans,
                             const float spatial_scale,
                             const int output_dim,
                             const int group_size,
                             const int pooled_size,
                             const int part_size,
                             const int sample_per_part,
                             const float trans_std)
{
    if (input.type().is_cuda())
    {
#ifdef WITH_CUDA
        return dcn_v2_psroi_pooling_cuda_forward(input,
                                                 bbox,
                                                 trans,
                                                 no_trans,
                                                 spatial_scale,
                                                 output_dim,
                                                 group_size,
                                                 pooled_size,
                                                 part_size,
                                                 sample_per_part,
                                                 trans_std);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    else{
        return dcn_v2_psroi_pooling_cpu_forward(input,
                                                 bbox,
                                                 trans,
                                                 no_trans,
                                                 spatial_scale,
                                                 output_dim,
                                                 group_size,
                                                 pooled_size,
                                                 part_size,
                                                 sample_per_part,
                                                 trans_std);
    }
}

std::tuple<at::Tensor, at::Tensor>
dcn_v2_psroi_pooling_backward(const at::Tensor &out_grad,
                              const at::Tensor &input,
                              const at::Tensor &bbox,
                              const at::Tensor &trans,
                              const at::Tensor &top_count,
                              const int no_trans,
                              const float spatial_scale,
                              const int output_dim,
                              const int group_size,
                              const int pooled_size,
                              const int part_size,
                              const int sample_per_part,
                              const float trans_std)
{
    if (input.type().is_cuda())
    {
#ifdef WITH_CUDA
        return dcn_v2_psroi_pooling_cuda_backward(out_grad,
                                                  input,
                                                  bbox,
                                                  trans,
                                                  top_count,
                                                  no_trans,
                                                  spatial_scale,
                                                  output_dim,
                                                  group_size,
                                                  pooled_size,
                                                  part_size,
                                                  sample_per_part,
                                                  trans_std);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    else{
        return dcn_v2_psroi_pooling_cpu_backward(out_grad,
                                                  input,
                                                  bbox,
                                                  trans,
                                                  top_count,
                                                  no_trans,
                                                  spatial_scale,
                                                  output_dim,
                                                  group_size,
                                                  pooled_size,
                                                  part_size,
                                                  sample_per_part,
                                                  trans_std);
    }
}