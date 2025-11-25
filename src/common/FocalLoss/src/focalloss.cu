#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>

template<typename T>
__device__ __forceinline__ T sigmoid(T x) {
    return T(1.0) / (T(1.0) + __expf(-x));
}

template<>
__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

template<typename T>
__device__ __forceinline__ T fast_pow(T base, T exponent) {
    return __powf(base, exponent);
}

template<>
__device__ __forceinline__ half fast_pow(half base, half exponent) {
    return __float2half(__powf(__half2float(base), __half2float(exponent)));
}

template<typename T>
__device__ __forceinline__ T stable_log(T x) {
    return __logf(fmaxf(x, 1e-12f));
}

template<>
__device__ __forceinline__ half stable_log(half x) {
    return __float2half(__logf(fmaxf(__half2float(x), 1e-12f)));
}

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

template<typename T>
__global__ void focal_loss_backward_kernel_optimized(
    const T* __restrict__ d_loss,
    const T* __restrict__ pred,
    const long* __restrict__ target,
    T* __restrict__ grad_input,
    const float alpha, const float gamma, 
    const int num_classes, const float eps,
    const int batch_size, const int spatial_size) {

    int batch_idx = blockIdx.x;
    int spatial_idx = blockIdx.y;
    int class_idx = threadIdx.x;

    if (batch_idx >= batch_size || spatial_idx >= spatial_size || class_idx >= num_classes - 1)
        return;

    long target_val = target[batch_idx * spatial_size + spatial_idx];
    if (target_val < 0) target_val = num_classes - 1;
    
    int data_idx = batch_idx * spatial_size * (num_classes - 1) + 
                   spatial_idx * (num_classes - 1) + class_idx;
    
    T dL = d_loss[data_idx];
    T x = pred[data_idx];
    
    T p = sigmoid(x);

    bool is_positive = (class_idx == target_val);

    T pt;
    if (is_positive) {
        pt = p;
    } else {
        pt = T(1.0) - p;
    }
    
    T at;
    if (is_positive) {
        at = T(alpha);
    } else {
        at = T(1.0 - alpha);
    }
    
    T pt_clamped = fmaxf(pt, T(eps));
    pt_clamped = fminf(pt_clamped, T(1.0));
    
    T dL_dpt;
    if (pt_clamped + eps <= T(1.0)) {
        // ∂L/∂pt = at * [γ * (1-pt)^(γ-1) * log(pt+ε) - (1-pt)^γ / (pt+ε)]
        T one_minus_pt = T(1.0) - pt;
        T log_term = stable_log(pt_clamped);
        T pow_term1 = fast_pow(one_minus_pt, T(gamma - 1.0));
        T pow_term2 = fast_pow(one_minus_pt, T(gamma));
        
        T term1 = T(gamma) * pow_term1 * log_term;
        T term2 = pow_term2 / pt_clamped;
        dL_dpt = at * (term1 - term2);
    } else {
        dL_dpt = T(0.0);
    }
    
    T dpt_dpred;
    if (is_positive) {
        dpt_dpred = T(1.0);
    } else {
        dpt_dpred = T(-1.0);
    }

    T dpred_dx = p * (T(1.0) - p);

    T grad = dL * dL_dpt * dpt_dpred * dpred_dx;

    grad_input[data_idx] = grad;
}

torch::Tensor focalloss_backward(
    const torch::Tensor d_loss,
    const torch::Tensor pred,
    const torch::Tensor target,
    int num_classes, float alpha, float gamma, float eps)
{
    CHECK_INPUT(d_loss);
    CHECK_INPUT(pred);
    CHECK_INPUT(target);

    int64_t spatial_size = pred.size(0);
    int64_t channel_size = pred.size(1);

    TORCH_CHECK(d_loss.size(0) == spatial_size, "grad_output must have same length as pred");
    TORCH_CHECK(d_loss.size(1) == channel_size, "grad_output must have same samples as pred");
    TORCH_CHECK(target.size(0) == spatial_size, "target must have same length as pred");
    TORCH_CHECK(channel_size == (num_classes-1), "num_classes must be one more than num channels");

    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(pred.device());
    torch::Tensor grad_input = torch::empty({spatial_size, channel_size}, options);

    dim3 blocks(1, spatial_size);
    dim3 threads(num_classes - 1);

    focal_loss_backward_kernel_optimized<float>
        <<<blocks, threads>>>(
            d_loss.data_ptr<float>(),
            pred.data_ptr<float>(),
            target.data_ptr<long>(),
            grad_input.data_ptr<float>(),
            alpha, gamma, num_classes, eps,
            1, spatial_size);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in focal_loss_backward_kernel: %s\n", cudaGetErrorString(err));
    }

    return grad_input;
}

__global__ void focal_loss_kernel_per_sample(
    const float* pred, const long* target, float* loss,
    float alpha, float gamma, int num_classes, float eps,
    int batch_size, int spatial_size) {
    
    int batch_idx = blockIdx.x;
    int spatial_idx = blockIdx.y;
    int class_idx = threadIdx.x;

    if (batch_idx >= batch_size || spatial_idx >= spatial_size || class_idx >= num_classes - 1) 
        return;

    int global_idx = batch_idx * spatial_size * (num_classes - 1) + 
                    spatial_idx * (num_classes - 1) + class_idx;

    long target_val = target[batch_idx * spatial_size + spatial_idx];
    if (target_val < 0) target_val = num_classes - 1;

    float pred_idx = pred[global_idx];
    float p = 1/(1+expf(-pred_idx));
    bool is_positive = (class_idx == target_val);
    
    float pt = is_positive ? p : (1.0f - p);
    float at = is_positive ? alpha : (1.0f - alpha);
    
    pt = fminf(pt + eps, 1.0f);
    loss[global_idx] = -at * powf(1.0f - pt, gamma) * logf(pt);
}

torch::Tensor focalloss_forward(
    const torch::Tensor pred,
    const torch::Tensor target,
    int num_classes,
    float alpha,
    float gamma,
    float eps) {

    CHECK_INPUT(pred);
    CHECK_INPUT(target);

    int64_t spatial_size = pred.size(0);
    int64_t channel_size = pred.size(1);

    TORCH_CHECK(target.size(0) == spatial_size, "target must have same length as pred");
    TORCH_CHECK(channel_size == (num_classes-1), "num_classes must be one more than num channels");

    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(pred.device());
    torch::Tensor output = torch::empty({spatial_size, channel_size}, options);

    dim3 blocks(1, spatial_size);
    dim3 threads(num_classes - 1);
    focal_loss_kernel_per_sample<<<blocks, threads>>>(
        pred.data_ptr<float>(),
        target.data_ptr<long>(),
        output.data_ptr<float>(),
        alpha, gamma, num_classes, eps,
        1, spatial_size);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA error: ", cudaGetErrorString(err));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("focalloss_forward", &focalloss_forward, "FocalLoss forward");
    m.def("focalloss_backward", &focalloss_backward, "FocalLoss forward");
}