import torch
import numpy as np
from swiglu_ext import gemm_swiglu

global_silu = torch.nn.SiLU()

def benchmark(A, B, C):
    return global_silu(torch.matmul(A, B)) * C

if __name__ == "__main__":
    M, N, K = 1024, 1024, 8
    left = torch.randn((M, K), dtype=torch.float16, device=torch.cuda.current_device()).contiguous()
    right = torch.randn((K, N), dtype=torch.float16, device=torch.cuda.current_device()).contiguous()
    source = torch.randn((M, N), dtype=torch.float16, device=torch.cuda.current_device()).contiguous()


    custom_res = gemm_swiglu(left, right, source, 1.0)
    torch_res = benchmark(left, right, source)

    diff = torch.max(torch.abs(custom_res - torch_res))
    print(f"Max output difference: {diff.item()}")
    print(f"Outputs match: {torch.allclose(custom_res, torch_res, atol=1e-3, rtol=1e-3)}")

    elemwise_diff = torch.abs(custom_res - torch_res)
    threshold = 1e-3
    large_diff_mask = torch.abs(elemwise_diff) > threshold
    large_diff_count = large_diff_mask.sum().item()
    total_elements = elemwise_diff.numel()
    print(f"Elements with elemwise_diff > {threshold}: {large_diff_count}/{total_elements} ({large_diff_count/total_elements*100:.2f}%)")
    if large_diff_count > 0:
        print(f"\n=== 最大差异位置 ===")
        flat_diff = elemwise_diff.abs().flatten()
        topk_values, topk_indices = torch.topk(flat_diff, min(5, large_diff_count))
        for i, (val, idx) in enumerate(zip(topk_values, topk_indices)):
            orig_idx = np.unravel_index(idx.cpu().numpy(), elemwise_diff.shape)
            print(f"Position {orig_idx}: custom={custom_res[orig_idx].item():.6f}, PyTorch={torch_res[orig_idx].item():.6f}, Diff={val.item():.6f}")

    if torch.cuda.is_available():
        import time
        # warmup
        for _ in range(10):
            _ = gemm_swiglu(left, right, source, 1.0)
            _ = benchmark(left, right, source)
        start_time = time.time()
        LOOPS=100
        for _ in range(LOOPS):
            _ = gemm_swiglu(left, right, source, 1.0)
        vfx_time = (time.time() - start_time) * 1000 / LOOPS
        start_time = time.time()
        for _ in range(LOOPS):
            _ = benchmark(left, right, source)
        pytorch_time = (time.time() - start_time) * 1000 / LOOPS
        print(f"\nPerformance comparison:")
        print(f"CUDA time: {vfx_time:.4f}ms")
        print(f"PyTorch time: {pytorch_time:.4f}ms")
        if vfx_time > 0:
            print(f"Speedup: {pytorch_time / vfx_time:.2f}x")