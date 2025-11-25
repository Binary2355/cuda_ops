#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x);

constexpr int BLOCK_SIZE = 128;

__global__ void rotary_forward_kernel(
    const __half* t,
    const float* freqs_cos,
    const float* freqs_sin,
    float* output,
    int bs0, int bs1, int m, int n,
    int t_stride_0, int t_stride_1, int t_stride_2, int freqs_cos_stride_0, int freqs_sin_stride_0)
{
    int row_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_id >= bs0*bs1*m) return;

    int mid = row_id % m;
    row_id /= m;
    int bs1id = row_id % bs1;
    row_id /= bs1;
    int bs0id = row_id % bs0;

    int output_stride_0 = bs1 * m * n;
    int output_stride_1 = m * n;
    int output_stride_2 = n;

    const __half* t_ptr = t + (t_stride_0 * bs0id + t_stride_1 * bs1id + t_stride_2 * mid);
    const float* cos_ptr = freqs_cos + (freqs_cos_stride_0 * mid);
    const float* sin_ptr = freqs_sin + (freqs_sin_stride_0 * mid);
    float* output_ptr = output + (output_stride_0 * bs0id + output_stride_1 * bs1id + output_stride_2 * mid);
    #pragma unroll
    for (int col_id=0; col_id < n; col_id += 4) {
        float4 cos_val = *reinterpret_cast<const float4*>(cos_ptr+col_id);
        float4 sin_val = *reinterpret_cast<const float4*>(sin_ptr+col_id);

        float tx = __half2float(t_ptr[col_id]);
        float ty = __half2float(t_ptr[col_id+1]); 
        float tz = __half2float(t_ptr[col_id+2]);
        float tw = __half2float(t_ptr[col_id+3]);

        float4 output_val = {
            tx * cos_val.x - ty * sin_val.x,
            ty * cos_val.y + tx * sin_val.y,
            tz * cos_val.z - tw * sin_val.z,
            tw * cos_val.w + tz * sin_val.w
        };

        *reinterpret_cast<float4*>(output_ptr+col_id) = output_val;
    }
}

__global__ void rotary_backward_kernel(
    const float* grad_output,
    const float* freqs_cos,
    const float* freqs_sin,
    __half* output,
    int bs0, int bs1, int m, int n,
    int grad_output_stride_0, int grad_output_stride_1, int grad_output_stride_2, int freqs_cos_stride_0, int freqs_sin_stride_0)
{
    int row_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_id >= bs0*bs1*m) return;

    int mid = row_id % m;
    row_id /= m;
    int bs1id = row_id % bs1;
    row_id /= bs1;
    int bs0id = row_id % bs0;

    int output_stride_0 = bs1 * m * n;
    int output_stride_1 = m * n;
    int output_stride_2 = n;

    const float* grad_output_ptr = grad_output + \
        (grad_output_stride_0 * bs0id + grad_output_stride_1 * bs1id + grad_output_stride_2 * mid);
    const float* cos_ptr = freqs_cos + (freqs_cos_stride_0 * mid);
    const float* sin_ptr = freqs_sin + (freqs_sin_stride_0 * mid);
    __half* output_ptr = output + (output_stride_0 * bs0id + output_stride_1 * bs1id + output_stride_2 * mid);

    #pragma unroll
    for (int col_id=0; col_id < n; col_id += 4) {
        float4 grad_output_val = *reinterpret_cast<const float4*>(grad_output_ptr+col_id);
        float4 cos_val = *reinterpret_cast<const float4*>(cos_ptr+col_id);
        float4 sin_val = *reinterpret_cast<const float4*>(sin_ptr+col_id);

        float4 output_val = {
            grad_output_val.x * cos_val.x + grad_output_val.y * sin_val.y,
            grad_output_val.y * cos_val.y - grad_output_val.x * sin_val.x,
            grad_output_val.z * cos_val.z + grad_output_val.w * sin_val.w,
            grad_output_val.w * cos_val.w - grad_output_val.z * sin_val.z,
        };

        output_ptr[col_id] = __float2half(output_val.x);
        output_ptr[col_id+1] = __float2half(output_val.y);
        output_ptr[col_id+2] = __float2half(output_val.z);
        output_ptr[col_id+3] = __float2half(output_val.w);
    }
}

torch::Tensor rotary_forward(
    torch::Tensor t,
    torch::Tensor freqs_cos,
    torch::Tensor freqs_sin,
    int t_stride_0, int t_stride_1, int t_stride_2, int t_stride_3,
    int freqs_cos_stride_0, int freqs_cos_stride_1, int freqs_sin_stride_0, int freqs_sin_stride_1)
{
    CHECK_INPUT(t);
    CHECK_INPUT(freqs_cos);
    CHECK_INPUT(freqs_sin);

    int64_t bs0 = t.size(0);
    int64_t bs1 = t.size(1);
    int64_t m = t.size(2);
    int64_t n = t.size(3);

    TORCH_CHECK(n % 2 == 0, "t.size(3) must be even");
    TORCH_CHECK(t_stride_3 == 1, "t_stride_3 should be 1");
    TORCH_CHECK(freqs_cos_stride_1 == 1, "freqs_cos_stride_1 should be 1");
    TORCH_CHECK(freqs_sin_stride_1 == 1, "freqs_sin_stride_1 should be 1");
    TORCH_CHECK(freqs_cos.size(0) == m && freqs_cos.size(1) == n, 
                "last two dim of freqs_cos must have same shape as t");
    TORCH_CHECK(freqs_sin.size(0) == m && freqs_sin.size(1) == n, 
                "last two dim of freqs_sin must have same shape as t");

    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(t.device());
    torch::Tensor output = torch::empty({bs0, bs1, m, n}, options);

    dim3 block(BLOCK_SIZE);
    dim3 grid((bs0*bs1*m + BLOCK_SIZE - 1) / BLOCK_SIZE);
    rotary_forward_kernel<<<grid, block>>>(
        reinterpret_cast<const __half*>(t.data_ptr<torch::Half>()),
        freqs_cos.data_ptr<float>(),
        freqs_sin.data_ptr<float>(),
        output.data_ptr<float>(),
        bs0, bs1, m, n,
        t_stride_0, t_stride_1, t_stride_2,
        freqs_cos_stride_0, freqs_sin_stride_0);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA error: ", cudaGetErrorString(err));
    }

    return output;
}

torch::Tensor rotary_backward(
    torch::Tensor grad_output,
    torch::Tensor freqs_cos,
    torch::Tensor freqs_sin,
    int grad_output_stride_0, int grad_output_stride_1, int grad_output_stride_2, int grad_output_stride_3,
    int freqs_cos_stride_0, int freqs_cos_stride_1, int freqs_sin_stride_0, int freqs_sin_stride_1)
{

    CHECK_INPUT(grad_output);
    CHECK_INPUT(freqs_cos);
    CHECK_INPUT(freqs_sin);

    int64_t bs0 = grad_output.size(0);
    int64_t bs1 = grad_output.size(1);
    int64_t m = grad_output.size(2);
    int64_t n = grad_output.size(3);

    TORCH_CHECK(n % 2 == 0, "grad_output.size(3) must be even");
    TORCH_CHECK(grad_output_stride_3 == 1, "grad_output_stride_3 should be 1");
    TORCH_CHECK(freqs_cos_stride_1 == 1, "freqs_cos_stride_1 should be 1");
    TORCH_CHECK(freqs_sin_stride_1 == 1, "freqs_sin_stride_1 should be 1");
    TORCH_CHECK(freqs_cos.size(0) == m && freqs_cos.size(1) == n, 
                "last two dim of freqs_cos must have same shape as grad_output");
    TORCH_CHECK(freqs_sin.size(0) == m && freqs_sin.size(1) == n, 
                "last two dim of freqs_sin must have same shape as grad_output");

    auto options = torch::TensorOptions()
        .dtype(torch::kFloat16)
        .device(grad_output.device());
    torch::Tensor output = torch::empty({bs0, bs1, m, n}, options);

    dim3 block(BLOCK_SIZE);
    dim3 grid((bs0*bs1*m + BLOCK_SIZE - 1) / BLOCK_SIZE);
    rotary_backward_kernel<<<grid, block>>>(
        grad_output.data_ptr<float>(),
        freqs_cos.data_ptr<float>(),
        freqs_sin.data_ptr<float>(),
        reinterpret_cast<__half*>(output.data_ptr<torch::Half>()),
        bs0, bs1, m, n,
        grad_output_stride_0, grad_output_stride_1, grad_output_stride_2,
        freqs_cos_stride_0, freqs_sin_stride_0);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA error: ", cudaGetErrorString(err));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rotary_forward", &rotary_forward, "Rotary forward");
    m.def("rotary_backward", &rotary_backward, "Rotary backward");
}