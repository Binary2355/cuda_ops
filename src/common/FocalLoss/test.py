import torch
import numpy as np
from einops import rearrange
from focalloss_ext import focalloss_forward, focalloss_backward

class FusedFocalLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pred, target, num_classes, alpha, gamma, eps):
        ctx.save_for_backward(pred, target)
        ctx.num_classes = num_classes
        ctx.alpha = alpha
        ctx.gamma = gamma
        ctx.eps = eps

        return focalloss_forward(pred, target, num_classes, alpha, gamma, eps)

    @staticmethod
    def backward(ctx, grad_output):
        pred, target = ctx.saved_tensors
        num_classes = ctx.num_classes
        alpha = ctx.alpha
        gamma = ctx.gamma
        eps = ctx.eps

        return focalloss_backward(grad_output.contiguous(), pred, target, num_classes, alpha, gamma, eps), None, None, None, None, None

class FusedFocalLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, num_classes, alpha, gamma, eps):
        res = FusedFocalLossFunction.apply(pred, target, num_classes, alpha, gamma, eps)
        return res

class OrigFocalLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,  pred, target, num_classes, alpha, gamma, eps):
        pred = pred.sigmoid()
        target[target < 0] = num_classes - 1
        one_hot = torch.nn.functional.one_hot(target, num_classes)  # N x C+1
        one_hot = one_hot[..., : num_classes - 1]  # N x C
        pt = torch.where(torch.eq(one_hot, 1), pred, 1 - pred)
        t = torch.ones_like(one_hot)
        at = torch.where(
            torch.eq(one_hot, 1), alpha * t, (1 - alpha) * t
        )
        loss = (
            -at
            * torch.pow((1 - pt), gamma)
            * torch.log(torch.minimum(pt + eps, t))
        )  # noqa
        return loss

if __name__ == "__main__":
    ########################### 单测 ###########################
    focal_loss_fused = FusedFocalLoss().cuda()
    focal_loss_orig = OrigFocalLoss().cuda()

    focal_loss_fused.train()
    focal_loss_orig.train()

    device = 'cuda'
    print(f"Using device: {device}")

    spatial_size, num_classes, alpha, gamma, eps = 900, 2, 0.25, 2.0, 1e-12

    pred_fused = torch.randn(spatial_size, num_classes-1, device=device, dtype=torch.float32, requires_grad=True)
    pred_orig = pred_fused.detach().clone().requires_grad_(True)
    target_fused = torch.randint(0, num_classes, (spatial_size,), device=device, dtype=torch.int64)
    target_orig = target_fused.detach().clone()

    print("Testing forward pass...")
    try:
        output = focal_loss_fused(pred_fused, target_fused, num_classes, alpha, gamma, eps)
        print(f"✓ Forward pass successful!")
        print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        exit(1)

    output_pytorch = focal_loss_orig(pred_orig, target_orig, num_classes, alpha, gamma, eps)

    print("\n=== 正向传播验证 ===")
    diff = torch.max(torch.abs(output - output_pytorch))
    print(f"Max output difference: {diff.item()}")
    print(f"Outputs match: {torch.allclose(output, output_pytorch, atol=1e-3, rtol=1e-3)}")

    print("\n=== 反向传播验证 ===")
    loss_fused = output.sum()
    print(f"Loss fused: {loss_fused.item()}")
    loss_fused.backward()
    print(f"Fused implementation - pred_fused.grad shape: {pred_fused.grad.shape if pred_fused.grad is not None else 'None'}")

    loss_pytorch = output_pytorch.sum()
    print(f"Loss PyTorch: {loss_pytorch.item()}")
    loss_pytorch.backward()
    print(f"PyTorch implementation - pred_orig.grad shape: {pred_orig.grad.shape if pred_orig.grad is not None else 'None'}")

    if pred_fused.grad is not None and pred_orig.grad is not None:
        grad_fused = pred_fused.grad.float()
        grad_pytorch = pred_orig.grad.float()
        grad_diff_abs = torch.max(torch.abs(grad_fused - grad_pytorch))
        grad_diff_rel = torch.max(torch.abs(grad_fused - grad_pytorch) / (torch.abs(grad_pytorch) + 1e-8))
        print(f"{grad_pytorch=} {grad_fused=}")
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
        print(f"✗ 梯度计算失败：pred_fused.grad is None ==> {pred_fused.grad is None} OR pred_orig.grad is None ==>{pred_orig.grad is None}")

    if torch.cuda.is_available():
        import time
        # warmup
        for _ in range(10):
            _ = focal_loss_fused(pred_fused, target_fused, num_classes, alpha, gamma, eps)
            _ = focal_loss_orig(pred_orig, target_orig, num_classes, alpha, gamma, eps)
        start_time = time.time()
        LOOPS=100
        for _ in range(LOOPS):
            _ = focal_loss_fused(pred_fused, target_fused, num_classes, alpha, gamma, eps)
        vfx_time = (time.time() - start_time) * 1000 / LOOPS
        start_time = time.time()
        for _ in range(LOOPS):
            _ = focal_loss_orig(pred_orig, target_orig, num_classes, alpha, gamma, eps)
        pytorch_time = (time.time() - start_time) * 1000 / LOOPS
        print(f"\nPerformance comparison:")
        print(f"CUDA time: {vfx_time:.4f}ms")
        print(f"PyTorch time: {pytorch_time:.4f}ms")
        if vfx_time > 0:
            print(f"Speedup: {pytorch_time / vfx_time:.2f}x")