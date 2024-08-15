#include <vector>
#include "cuda/dcn_v2_im2col_cuda.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/Dispatch.h>
#include <ATen/div_rtn.h>
#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/Exceptions.h>

THCState *state = at::globalContext().lazyInitCUDA();

static cublasOperation_t _cublasOpFromChar(char op) {
    switch (op) {
      case 'n':
      case 'N':
        return CUBLAS_OP_N;
      case 't':
      case 'T':
        return CUBLAS_OP_T;
      case 'c':
      case 'C':
        return CUBLAS_OP_C;
    }
    AT_ERROR(
        "_cublasOpFromChar input should be 't', 'n' or 'c' but got `", op, "`");
  }

  static void _cublasAdjustLdLevel2(int64_t m, int64_t n, int64_t* lda) {
    // Note: leading dimensions generally are checked that they are > 0
    // and at least as big the result requires (even if the value won't
    // be used).
  
    // Q: Why does Level3 check trans but this doesn't?
    // A: In level 2, the sizes (m, n) specify the size of A
    // (independent of trans value). In level 3. the sizes (m, n, k)
    // specify the sizes of op(A), op(B) where op depend on trans
    // values.
    if (n <= 1)
      *lda = std::max<int64_t>(m, 1);
  }



// author: Charles Shang
// https://github.com/torch/cunn/blob/master/lib/THCUNN/generic/SpatialConvolutionMM.cu

// [batch gemm]
// https://github.com/pytorch/pytorch/blob/master/aten/src/THC/generic/THCTensorMathBlas.cu

__global__ void createBatchGemmBuffer(const float **input_b, float **output_b,
                                      float **columns_b, const float **ones_b,
                                      const float **weight_b, const float **bias_b,
                                      float *input, float *output,
                                      float *columns, float *ones,
                                      float *weight, float *bias,
                                      const int input_stride, const int output_stride,
                                      const int columns_stride, const int ones_stride,
                                      const int num_batches)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_batches)
    {
        input_b[idx] = input + idx * input_stride;
        output_b[idx] = output + idx * output_stride;
        columns_b[idx] = columns + idx * columns_stride;
        ones_b[idx] = ones + idx * ones_stride;
        // share weights and bias within a Mini-Batch
        weight_b[idx] = weight;
        bias_b[idx] = bias;
    }
}

at::Tensor
dcn_v2_cuda_forward(const at::Tensor &input,
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
    using scalar_t = float;
    // THCAssertSameGPU(THCudaTensor_checkGPU(state, 5, input, weight, bias, offset, mask));
    AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(weight.type().is_cuda(), "weight must be a CUDA tensor");
    AT_ASSERTM(bias.type().is_cuda(), "bias must be a CUDA tensor");
    AT_ASSERTM(offset.type().is_cuda(), "offset must be a CUDA tensor");
    AT_ASSERTM(mask.type().is_cuda(), "mask must be a CUDA tensor");

    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    const int channels_out = weight.size(0);
    const int channels_kernel = weight.size(1);
    const int kernel_h_ = weight.size(2);
    const int kernel_w_ = weight.size(3);

    // printf("Kernels: %d %d %d %d\n", kernel_h_, kernel_w_, kernel_w, kernel_h);
    // printf("Channels: %d %d\n", channels, channels_kernel);
    // printf("Channels: %d %d\n", channels_out, channels_kernel);

    AT_ASSERTM(kernel_h_ == kernel_h && kernel_w_ == kernel_w,
               "Input shape and kernel shape wont match: (%d x %d vs %d x %d).", kernel_h_, kernel_w, kernel_h_, kernel_w_);

    AT_ASSERTM(channels == channels_kernel,
               "Input shape and kernel channels wont match: (%d vs %d).", channels, channels_kernel);

    const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    auto ones = at::ones({batch, height_out, width_out}, input.options());
    auto columns = at::empty({batch, channels * kernel_h * kernel_w, 1 * height_out * width_out}, input.options());
    auto output = at::empty({batch, channels_out, height_out, width_out}, input.options());

    // prepare for batch-wise computing, which is significantly faster than instance-wise computing
    // when batch size is large.
    // launch batch threads
    int matrices_size = batch * sizeof(float *);
    auto input_b = static_cast<const float **>(THCudaMalloc(state, matrices_size));
    auto output_b = static_cast<float **>(THCudaMalloc(state, matrices_size));
    auto columns_b = static_cast<float **>(THCudaMalloc(state, matrices_size));
    auto ones_b = static_cast<const float **>(THCudaMalloc(state, matrices_size));
    auto weight_b = static_cast<const float **>(THCudaMalloc(state, matrices_size));
    auto bias_b = static_cast<const float **>(THCudaMalloc(state, matrices_size));

    const int block = 128;
    const int grid = (batch + block - 1) / block;

    createBatchGemmBuffer<<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
        input_b, output_b,
        columns_b, ones_b,
        weight_b, bias_b,
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        columns.data_ptr<scalar_t>(),
        ones.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        bias.data_ptr<scalar_t>(),
        channels * width * height,
        channels_out * width_out * height_out,
        channels * kernel_h * kernel_w * height_out * width_out,
        height_out * width_out,
        batch);

    long m_ = channels_out;
    long n_ = height_out * width_out;
    long k_ = 1;
    THCudaBlas_SgemmBatched(state,
                            't',
                            'n',
                            n_,
                            m_,
                            k_,
                            1.0f,
                            ones_b, k_,
                            bias_b, k_,
                            0.0f,
                            output_b, n_,
                            batch);

    modulated_deformable_im2col_cuda(c10::cuda::getCurrentCUDAStream(),
                                     input.data_ptr<scalar_t>(),
                                     offset.data_ptr<scalar_t>(),
                                     mask.data_ptr<scalar_t>(),
                                     batch, channels, height, width,
                                     height_out, width_out, kernel_h, kernel_w,
                                     pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
                                     deformable_group,
                                     columns.data_ptr<scalar_t>());

    long m = channels_out;
    long n = height_out * width_out;
    long k = channels * kernel_h * kernel_w;
    THCudaBlas_SgemmBatched(state,
                            'n',
                            'n',
                            n,
                            m,
                            k,
                            1.0f,
                            (const float **)columns_b, n,
                            weight_b, k,
                            1.0f,
                            output_b, n,
                            batch);

    THCudaFree(state, input_b);
    THCudaFree(state, output_b);
    THCudaFree(state, columns_b);
    THCudaFree(state, ones_b);
    THCudaFree(state, weight_b);
    THCudaFree(state, bias_b);
    return output;
}

__global__ void createBatchGemmBufferBackward(
    float **grad_output_b,
    float **columns_b,
    float **ones_b,
    float **weight_b,
    float **grad_weight_b,
    float **grad_bias_b,
    float *grad_output,
    float *columns,
    float *ones,
    float *weight,
    float *grad_weight,
    float *grad_bias,
    const int grad_output_stride,
    const int columns_stride,
    const int ones_stride,
    const int num_batches)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_batches)
    {
        grad_output_b[idx] = grad_output + idx * grad_output_stride;
        columns_b[idx] = columns + idx * columns_stride;
        ones_b[idx] = ones + idx * ones_stride;

        // share weights and bias within a Mini-Batch
        weight_b[idx] = weight;
        grad_weight_b[idx] = grad_weight;
        grad_bias_b[idx] = grad_bias;
    }
}

std::vector<at::Tensor> dcn_v2_cuda_backward(const at::Tensor &input,
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

    THArgCheck(input.is_contiguous(), 1, "input tensor has to be contiguous");
    THArgCheck(weight.is_contiguous(), 2, "weight tensor has to be contiguous");

    AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(weight.type().is_cuda(), "weight must be a CUDA tensor");
    AT_ASSERTM(bias.type().is_cuda(), "bias must be a CUDA tensor");
    AT_ASSERTM(offset.type().is_cuda(), "offset must be a CUDA tensor");
    AT_ASSERTM(mask.type().is_cuda(), "mask must be a CUDA tensor");

    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    const int channels_out = weight.size(0);
    const int channels_kernel = weight.size(1);
    const int kernel_h_ = weight.size(2);
    const int kernel_w_ = weight.size(3);

    AT_ASSERTM(kernel_h_ == kernel_h && kernel_w_ == kernel_w,
               "Input shape and kernel shape wont match: (%d x %d vs %d x %d).", kernel_h_, kernel_w, kernel_h_, kernel_w_);

    AT_ASSERTM(channels == channels_kernel,
               "Input shape and kernel channels wont match: (%d vs %d).", channels, channels_kernel);

    const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    auto ones = at::ones({height_out, width_out}, input.options());
    auto columns = at::empty({channels * kernel_h * kernel_w, 1 * height_out * width_out}, input.options());
    auto output = at::empty({batch, channels_out, height_out, width_out}, input.options());

    auto grad_input = at::zeros_like(input);
    auto grad_weight = at::zeros_like(weight);
    auto grad_bias = at::zeros_like(bias);
    auto grad_offset = at::zeros_like(offset);
    auto grad_mask = at::zeros_like(mask);

    using scalar_t = float;

    for (int b = 0; b < batch; b++)
    {
        auto input_n = input.select(0, b);
        auto offset_n = offset.select(0, b);
        auto mask_n = mask.select(0, b);
        auto grad_output_n = grad_output.select(0, b);
        auto grad_input_n = grad_input.select(0, b);
        auto grad_offset_n = grad_offset.select(0, b);
        auto grad_mask_n = grad_mask.select(0, b);

        long m = channels * kernel_h * kernel_w;
        long n = height_out * width_out;
        long k = channels_out;

        THCudaBlas_Sgemm(state, 'n', 't', n, m, k, 1.0f,
                         grad_output_n.data_ptr<scalar_t>(), n,
                         weight.data_ptr<scalar_t>(), m, 0.0f,
                         columns.data_ptr<scalar_t>(), n);

        // gradient w.r.t. input coordinate data
        modulated_deformable_col2im_coord_cuda(c10::cuda::getCurrentCUDAStream(),
                                               columns.data_ptr<scalar_t>(),
                                               input_n.data_ptr<scalar_t>(),
                                               offset_n.data_ptr<scalar_t>(),
                                               mask_n.data_ptr<scalar_t>(),
                                               1, channels, height, width,
                                               height_out, width_out, kernel_h, kernel_w,
                                               pad_h, pad_w, stride_h, stride_w,
                                               dilation_h, dilation_w, deformable_group,
                                               grad_offset_n.data_ptr<scalar_t>(),
                                               grad_mask_n.data_ptr<scalar_t>());
        // gradient w.r.t. input data
        modulated_deformable_col2im_cuda(c10::cuda::getCurrentCUDAStream(),
                                         columns.data_ptr<scalar_t>(),
                                         offset_n.data_ptr<scalar_t>(),
                                         mask_n.data_ptr<scalar_t>(),
                                         1, channels, height, width,
                                         height_out, width_out, kernel_h, kernel_w,
                                         pad_h, pad_w, stride_h, stride_w,
                                         dilation_h, dilation_w, deformable_group,
                                         grad_input_n.data_ptr<scalar_t>());

        // gradient w.r.t. weight, dWeight should accumulate across the batch and group
        modulated_deformable_im2col_cuda(c10::cuda::getCurrentCUDAStream(),
                                         input_n.data_ptr<scalar_t>(),
                                         offset_n.data_ptr<scalar_t>(),
                                         mask_n.data_ptr<scalar_t>(),
                                         1, channels, height, width,
                                         height_out, width_out, kernel_h, kernel_w,
                                         pad_h, pad_w, stride_h, stride_w,
                                         dilation_h, dilation_w, deformable_group,
                                         columns.data_ptr<scalar_t>());

        long m_ = channels_out;
        long n_ = channels * kernel_h * kernel_w;
        long k_ = height_out * width_out;

        THCudaBlas_Sgemm(state, 't', 'n', n_, m_, k_, 1.0f,
                         columns.data_ptr<scalar_t>(), k_,
                         grad_output_n.data_ptr<scalar_t>(), k_, 1.0f,
                         grad_weight.data_ptr<scalar_t>(), n_);

        cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
        cublasOperation_t op = _cublasOpFromChar('t');
        _cublasAdjustLdLevel2(k_, m_, &k_);
        scalar_t* grad_output_n_float = grad_output_n.data_ptr<scalar_t>();
        scalar_t* one_float = ones.data_ptr<scalar_t>();
        scalar_t alpha = 1.0;
        scalar_t beta = 1.0;
        cublasSgemv(handle, op, k_, m_, &alpha, grad_output_n_float,k_, one_float,1, &beta, grad_bias.data_ptr<scalar_t>(), 1);

    }
    

    return {
        grad_input, grad_offset, grad_mask, grad_weight, grad_bias
    };
}
