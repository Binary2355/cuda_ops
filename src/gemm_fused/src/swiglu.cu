#define TORCH_EXTENSION_NAME swiglu_ext
#define TORCH_LIBRARY_NAME swiglu_ext

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <tuple>
#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cutlass/half.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "custom_combination.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

cudaError_t CutlassHGemmNN(
    int M,
    int N,
    int K,
    float alpha,
    cutlass::half_t const *A,
    int lda,
    cutlass::half_t const *B,
    int ldb,
    float beta,
    cutlass::half_t *C,
    int ldc)
{
    using ElementA = cutlass::half_t;
    using ElementB = cutlass::half_t;
    using ElementC = cutlass::half_t;
    using ElementAccumulator = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;

    using OpClass = cutlass::arch::OpClassTensorOp;
    using ArchTag = cutlass::arch::Sm80;

    using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementC,
        16,
        ElementAccumulator,
        ElementAccumulator>;

    using Gemm = cutlass::gemm::device::Gemm<
        ElementA,
        LayoutA,
        ElementB,
        LayoutB,
        ElementC,
        LayoutC,
        ElementAccumulator,
        OpClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        EpilogueOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        3  // stages
    >;

    typename Gemm::Arguments args(
        {M, N, K},
        {A, lda},
        {B, ldb},
        {C, ldc},
        {C, ldc},
        {alpha, beta}
    );

    Gemm gemm;
    cutlass::Status status = gemm.initialize(args);

    if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
    }

    status = gemm();

    if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
    }

    return cudaSuccess;
}

cudaError_t CutlassHGemmSwiGlu(
    int M,
    int N,
    int K,
    cutlass::half_t const *A,
    int lda,
    cutlass::half_t const *B,
    int ldb,
    cutlass::half_t const *C,
    int ldc,
    float beta,
    cutlass::half_t *D)
{
    using ElementA = cutlass::half_t;
    using ElementB = cutlass::half_t;
    using ElementC = cutlass::half_t;
    using ElementAccumulator = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;

    using OpClass = cutlass::arch::OpClassTensorOp;
    using ArchTag = cutlass::arch::Sm80;

    using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

    using EpilogueOp = cutlass::epilogue::thread::CustomSwiGluCombination<
        ElementC,
        16,
        ElementAccumulator,
        ElementAccumulator>;

    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::half_t,                           // ElementA
        cutlass::layout::RowMajor,                 // LayoutA
        cutlass::half_t,                           // ElementB
        cutlass::layout::RowMajor,                 // LayoutB
        cutlass::half_t,                           // ElementC
        cutlass::layout::RowMajor,                 // LayoutC
        float,                                     // ElementAccumulator
        cutlass::arch::OpClassTensorOp,            // OperatorClass
        cutlass::arch::Sm80,                       // Architecture
        cutlass::gemm::GemmShape<128, 128, 32>,   // ThreadblockShape
        cutlass::gemm::GemmShape<64, 64, 32>,     // WarpShape
        cutlass::gemm::GemmShape<16, 8, 16>,      // InstructionShape
        EpilogueOp,                                // EpilogueOutputOp
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        3,                                         // Stages
        8,                                         // AlignmentA
        8                                          // AlignmentB
    >;

    cutlass::gemm::GemmCoord problem_size(M, N, K);
    cutlass::TensorRef<ElementA const, LayoutA> ref_A(A, LayoutA(lda));
    cutlass::TensorRef<ElementB const, LayoutB> ref_B(B, LayoutB(ldb));
    cutlass::TensorRef<ElementC const, LayoutC> ref_C(C, LayoutC(ldc));
    cutlass::TensorRef<ElementC, LayoutC> ref_D(D, LayoutC(ldc));
    typename EpilogueOp::Params epilogue_params(beta);

    typename Gemm::Arguments args(
        problem_size, ref_A, ref_B, ref_C, ref_D, epilogue_params
    );

    Gemm gemm;
    cutlass::Status status = gemm.initialize(args);

    if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
    }

    status = gemm();

    if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
    }

    return cudaSuccess;
}

torch::Tensor pure_gemm(
    const torch::Tensor& A,
    const torch::Tensor& B,
    float alpha = 1.0f,
    float beta = 0.0f)
{
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16 tensor");
    TORCH_CHECK(B.dtype() == torch::kFloat16, "B must be float16 tensor");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    int lda = K;
    int ldb = N;
    int ldc = N;
    
    TORCH_CHECK(B.size(0) == K, "the first dim of B should be equal to the second dim of A");

    auto options = torch::TensorOptions()
        .dtype(torch::kFloat16)
        .device(A.device());

    auto C_cutlass = torch::empty({M, N}, options);

    cutlass::half_t* A_ptr = reinterpret_cast<cutlass::half_t*>(A.data_ptr<at::Half>());
    cutlass::half_t* B_ptr = reinterpret_cast<cutlass::half_t*>(B.data_ptr<at::Half>());
    cutlass::half_t* C_ptr = reinterpret_cast<cutlass::half_t*>(C_cutlass.data_ptr<at::Half>());

    cudaError_t result = CutlassHGemmNN(M, N, K, alpha, A_ptr, lda, B_ptr, ldb, beta, C_ptr, ldc);

    if (result != cudaSuccess) {
        std::cerr << "CUTLASS GEMM kernel failed: " << cudaGetErrorString(result) << std::endl;
        TORCH_CHECK(false, "CUTLASS GEMM kernel failed");
    }

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA error: ", cudaGetErrorString(err));
    }

    return C_cutlass;
};

torch::Tensor gemm_swiglu(
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& C,
    float beta
)
{
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);

    TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16 tensor");
    TORCH_CHECK(B.dtype() == torch::kFloat16, "B must be float16 tensor");
    TORCH_CHECK(C.dtype() == torch::kFloat16, "B must be float16 tensor");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    int lda = K;
    int ldb = N;
    int ldc = N;
    
    TORCH_CHECK(B.size(0) == K, "the first dim of B should be equal to the second dim of A");
    TORCH_CHECK(C.size(0) == M, "the first dim of C should be equal to the first dim of A");
    TORCH_CHECK(C.size(1) == N, "the second dim of C should be equal to the second dim of A");

    auto options = torch::TensorOptions()
        .dtype(torch::kFloat16)
        .device(A.device());

    auto D_cutlass = torch::empty({M, N}, options);

    cutlass::half_t* A_ptr = reinterpret_cast<cutlass::half_t*>(A.data_ptr<at::Half>());
    cutlass::half_t* B_ptr = reinterpret_cast<cutlass::half_t*>(B.data_ptr<at::Half>());
    cutlass::half_t* C_ptr = reinterpret_cast<cutlass::half_t*>(C.data_ptr<at::Half>());
    cutlass::half_t* D_ptr = reinterpret_cast<cutlass::half_t*>(D_cutlass.data_ptr<at::Half>());

    cudaError_t result = CutlassHGemmSwiGlu(M, N, K, A_ptr, lda, B_ptr, ldb, C_ptr, ldc, beta, D_ptr);

    if (result != cudaSuccess) {
        std::cerr << "CUTLASS GEMM kernel failed: " << cudaGetErrorString(result) << std::endl;
        TORCH_CHECK(false, "CUTLASS GEMM kernel failed");
    }

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA error: ", cudaGetErrorString(err));
    }

    return D_cutlass;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pure_gemm", &pure_gemm, "Test CUTLASS GEMM");
    m.def("gemm_swiglu", &gemm_swiglu, "Epilogue: SwiGlu");
}