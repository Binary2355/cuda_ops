import time
import torch

from swiglu_ext import pure_gemm

if __name__ == "__main__":
    device = 'cuda'
    print(f"Using device: {device}")

    M, N, K, alpha, beta = 2048, 4096, 1000, 1.0, 0.0

    A_cutlass = torch.randn((K, M), device=device, dtype=torch.float16, requires_grad=True).permute(1,0)
    B_cutlass = torch.randn((K, N), device=device, dtype=torch.float16, requires_grad=True)

    A_pytorch = A_cutlass.detach().clone().requires_grad_(True)
    B_pytorch = B_cutlass.detach().clone().requires_grad_(True)


    cutlass_output = pure_gemm(A_cutlass, B_cutlass, alpha, beta).permute(1,0)
    pytorch_output = torch.matmul(A_pytorch, B_pytorch)

    print(f"====================================== 精度对比 ======================================")
    diff0 = torch.max(torch.abs(cutlass_output - pytorch_output))
    print(f"{cutlass_output=}")
    print(f"{pytorch_output=}")

    print(f"Max output difference: {diff0.item()}")
    print(f"Outputs match: {torch.allclose(cutlass_output, pytorch_output, atol=1e-3, rtol=1e-3)}")

    print(f"====================================== 性能对比 ======================================")
    if torch.cuda.is_available():
        import time
        # warmup
        for _ in range(10):
            _ = pure_gemm(A_cutlass, B_cutlass, alpha, beta)
            _ = torch.matmul(A_pytorch, B_pytorch)
        start_time = time.time()
        LOOPS=100
        for _ in range(LOOPS):
            _ = pure_gemm(A_cutlass, B_cutlass, alpha, beta)
        vfx_time = (time.time() - start_time) * 1000 / LOOPS
        start_time = time.time()
        for _ in range(LOOPS):
            _ = pytorch_output = torch.matmul(A_pytorch, B_pytorch)
        pytorch_time = (time.time() - start_time) * 1000 / LOOPS
        print(f"\nPerformance comparison:")
        print(f"CUDA time: {vfx_time:.4f}ms")
        print(f"PyTorch time: {pytorch_time:.4f}ms")
        if vfx_time > 0:
            print(f"Speedup: {pytorch_time / vfx_time:.2f}x")


