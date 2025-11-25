import torch
import numpy as np
from einops import rearrange
from rotary_ext import rotary_forward
from rotary_ext import rotary_backward

class RotateHalfFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t, freqs_cos, freqs_sin):
        ctx.save_for_backward(freqs_cos, freqs_sin)
        t_stride = t.stride()
        freqs_cos_stride = freqs_cos.stride()
        freqs_sin_stride = freqs_sin.stride()

        return rotary_forward(t, freqs_cos, freqs_sin,
            t_stride[0], t_stride[1], t_stride[2], t_stride[3],
            freqs_cos_stride[0], freqs_cos_stride[1], freqs_sin_stride[0], freqs_sin_stride[1])

    @staticmethod
    def backward(ctx, grad_output):
        freqs_cos, freqs_sin = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_output_stride = grad_output.stride()
        freqs_cos_stride = freqs_cos.stride()
        freqs_sin_stride = freqs_sin.stride()
        return rotary_backward(grad_output, freqs_cos, freqs_sin,
            grad_output_stride[0], grad_output_stride[1], grad_output_stride[2], grad_output_stride[3],
            freqs_cos_stride[0], freqs_cos_stride[1], freqs_sin_stride[0], freqs_sin_stride[1]), None, None

class FusedRotateHalf(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t, freqs_cos, freqs_sin):
        res = RotateHalfFunction.apply(t, freqs_cos, freqs_sin)
        return res

class OrigRotateHalf(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def rotate_half_pytorch(self, x):
        x = rearrange(x, '... (d r) -> ... d r', r = 2)
        x1, x2 = x.unbind(dim = -1)
        x = torch.stack((-x2, x1), dim = -1)
        return rearrange(x, '... d r -> ... (d r)')

    def forward(self, t, freqs_cos, freqs_sin):
        return t * freqs_cos + self.rotate_half_pytorch(t) * freqs_sin

if __name__ == "__main__":
    ########################### 单测 ###########################
    rotate_half_fused = FusedRotateHalf().cuda()
    rotate_half_orig = OrigRotateHalf().cuda()

    rotate_half_fused.train()
    rotate_half_orig.train()

    device = 'cuda'
    print(f"Using device: {device}")

    bs0, bs1, seq_len, hidden_dim = 192, 16, 256, 64
    # bs0, bs1, seq_len, hidden_dim = 48, 16, 1024, 64

    t = torch.randn(bs0, bs1, seq_len, hidden_dim, device=device, dtype=torch.float16, requires_grad=True)
    t_pytorch = t.detach().clone().requires_grad_(True)
    freqs_cos = torch.randn(seq_len, hidden_dim, device=device, dtype=torch.float32)
    freqs_sin = torch.randn(seq_len, hidden_dim, device=device, dtype=torch.float32)

    print("Testing forward pass...")
    try:
        output = rotate_half_fused(t, freqs_cos, freqs_sin)
        print(f"✓ Forward pass successful!")
        print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        exit(1)

    output_pytorch = rotate_half_orig(t_pytorch, freqs_cos, freqs_sin)

    print("\n=== 正向传播验证 ===")
    diff = torch.max(torch.abs(output - output_pytorch))
    print(f"Max output difference: {diff.item()}")
    print(f"Outputs match: {torch.allclose(output, output_pytorch, atol=1e-3, rtol=1e-3)}")

    print("\n=== 反向传播验证 ===")
    loss_fused = output.sum()
    print(f"Loss fused: {loss_fused.item()}")
    loss_fused.backward()
    print(f"Fused implementation - t.grad shape: {t.grad.shape if t.grad is not None else 'None'}")

    loss_pytorch = output_pytorch.sum()
    print(f"Loss PyTorch: {loss_pytorch.item()}")
    loss_pytorch.backward()
    print(f"PyTorch implementation - t.grad shape: {t_pytorch.grad.shape if t_pytorch.grad is not None else 'None'}")

    if t.grad is not None and t_pytorch.grad is not None:
        grad_fused = t.grad.float()
        grad_pytorch = t_pytorch.grad.float()
        grad_diff_abs = torch.max(torch.abs(grad_fused - grad_pytorch))
        grad_diff_rel = torch.max(torch.abs(grad_fused - grad_pytorch) / (torch.abs(grad_pytorch) + 1e-8))
        # print(f"{grad_pytorch=} {grad_fused=}")
        print(f"\n=== 梯度比较结果 ===")
        print(f"Max absolute gradient difference: {grad_diff_abs.item()}")
        print(f"Max relative gradient difference: {grad_diff_rel.item()}")
        atol_grad = 1e-2
        rtol_grad = 1e-1
        grad_close = torch.allclose(grad_fused, grad_pytorch, atol=atol_grad, rtol=rtol_grad)
        print(f"Gradients match (atol={atol_grad}, rtol={rtol_grad}): {grad_close}")
        print(f"\n=== 梯度统计分析 ===")
        print(f"Fused grad - min: {grad_fused.min().item():.6f}, max: {grad_fused.max().item():.6f}, mean: {grad_fused.mean().item():.6f}")
        print(f"PyTorch grad - min: {grad_pytorch.min().item():.6f}, max: {grad_pytorch.max().item():.6f}, mean: {grad_pytorch.mean().item():.6f}")
        grad_diff = grad_fused - grad_pytorch
        print(f"Gradient diff - min: {grad_diff.min().item():.6f}, max: {grad_diff.max().item():.6f}, mean: {grad_diff.mean().item():.6f}, std: {grad_diff.std().item():.6f}")
        threshold = 1e-3
        large_diff_mask = torch.abs(grad_diff) > threshold
        large_diff_count = large_diff_mask.sum().item()
        total_elements = grad_diff.numel()
        print(f"Elements with gradient diff > {threshold}: {large_diff_count}/{total_elements} ({large_diff_count/total_elements*100:.2f}%)")
        if large_diff_count > 0:
            print(f"\n=== 最大梯度差异位置 ===")
            flat_diff = grad_diff.abs().flatten()
            topk_values, topk_indices = torch.topk(flat_diff, min(5, large_diff_count))
            for i, (val, idx) in enumerate(zip(topk_values, topk_indices)):
                orig_idx = np.unravel_index(idx.cpu().numpy(), grad_diff.shape)
                print(f"Position {orig_idx}: Fused={grad_fused[orig_idx].item():.6f}, PyTorch={grad_pytorch[orig_idx].item():.6f}, Diff={val.item():.6f}")
    else:
        print(f"✗ 梯度计算失败：t.grad is None ==> {t.grad is None} OR t_pytorch.grad is None ==>{t_pytorch.grad is None}")

    if torch.cuda.is_available():
        import time
        # warmup
        for _ in range(10):
            _ = rotate_half_fused(t, freqs_cos, freqs_sin)
            _ = rotate_half_orig(t, freqs_cos, freqs_sin)
        start_time = time.time()
        LOOPS=100
        for _ in range(LOOPS):
            _ = rotate_half_fused(t, freqs_cos, freqs_sin)
        vfx_time = (time.time() - start_time) * 1000 / LOOPS
        start_time = time.time()
        for _ in range(LOOPS):
            _ = rotate_half_orig(t, freqs_cos, freqs_sin)
        pytorch_time = (time.time() - start_time) * 1000 / LOOPS
        print(f"\nPerformance comparison:")
        print(f"CUDA time: {vfx_time:.4f}ms")
        print(f"PyTorch time: {pytorch_time:.4f}ms")
        if vfx_time > 0:
            print(f"Speedup: {pytorch_time / vfx_time:.2f}x")