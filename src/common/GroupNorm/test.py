import torch
import operator
from functools import reduce
import numpy as np

from groupnorm_ext import groupnorm_forward
from groupnorm_ext import groupnorm_backward

class CustomGroupNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, num_groups, weight, bias, eps):
        N = input.shape[0]
        C = input.shape[1]
        HxW = reduce(operator.mul, input.shape[2:], 1)

        # torch.native_group_norm,
        output, mean, rstd = groupnorm_forward(input, weight, bias, N, C, HxW, num_groups, eps)
        ctx.input, ctx.num_groups = input, num_groups
        ctx.weight, ctx.bias = weight, bias
        ctx.mean, ctx.rstd = mean, rstd
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, num_groups, weight, bias, mean, rstd = ctx.input, ctx.num_groups, ctx.weight, ctx.bias, ctx.mean, ctx.rstd

        grad_input = None
        grad_weight = None
        grad_bias = None
        if input.requires_grad:
            weight_c = weight.contiguous()
            input_c = input.contiguous()
            grad_output_c = (
                grad_output.contiguous() if grad_output is not None else None
            )
            N = input.shape[0]
            C = input.shape[1]
            HxW = 1
            for s in input.shape[2:]:
                HxW *= s
            # torch.ops.aten.native_group_norm_backward
            grad_input, grad_weight, grad_bias = groupnorm_backward(
                    grad_output_c,
                    input_c,
                    mean,
                    rstd,
                    weight_c,
                    N,
                    C,
                    HxW,
                    num_groups,
                    (True, True, True),
                )

        return grad_input, None, grad_weight, grad_bias, None

class CustomGroupNorm(torch.nn.Module):
    __constants__ = ["num_groups", "num_channels", "eps", "affine"]
    num_groups: int
    num_channels: int
    eps: float
    affine: bool

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError("num_channels must be divisible by num_groups")

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.empty(num_channels, **factory_kwargs))
        self.bias = torch.nn.Parameter(torch.empty(num_channels, **factory_kwargs))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return CustomGroupNormFunction.apply(input, self.num_groups, self.weight, self.bias, self.eps)

if __name__ == "__main__":
    ########################### 单测 ###########################
    num_channels = 8
    custom_model = CustomGroupNorm(num_groups=8, num_channels=num_channels).cuda()
    ori_model = torch.nn.GroupNorm(num_groups=8, num_channels=num_channels).cuda()

    custom_model.train()
    ori_model.train()

    device = 'cuda'
    print(f"Using device: {device}")

    # input_custom = torch.randn(1, 192, 72, 240, 384).cuda().requires_grad_(True)
    input_custom = torch.randn(1, num_channels, 67108864).cuda().requires_grad_(True)
    input_pytorch = input_custom.detach().clone().requires_grad_(True)
    custom_model.weight.data = ori_model.weight.data.clone()
    custom_model.bias.data = ori_model.bias.data.clone()

    print("Testing forward pass...")
    try:
        output_custom = custom_model(input_custom)
        print(f"✓ Forward pass successful!")
        print(f"Output shape: {output_custom.shape}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        exit(1)

    output_pytorch = ori_model(input_pytorch)

    print("\n=== 正向传播验证 ===")
    print(f"{output_custom=}")
    print(f"{output_pytorch=}")
    diff = torch.max(torch.abs(output_custom - output_pytorch))
    print(f"Max output difference: {diff.item()}")
    print(f"Outputs match: {torch.allclose(output_custom, output_pytorch, atol=1e-3, rtol=1e-3)}")

    print("\n=== 反向传播验证 ===")
    loss_fused = output_custom.sum()
    print(f"Loss fused: {loss_fused.item()}")
    loss_fused.backward()
    print(f"Custom implementation - input.grad shape: {input_custom.grad.shape if input_custom.grad is not None else 'None'}")
    print(f"Custom implementation - weight.grad shape: {custom_model.weight.grad.shape if custom_model.weight.grad is not None else 'None'}")
    print(f"Custom implementation - bias.grad shape: {custom_model.bias.grad.shape if custom_model.bias.grad is not None else 'None'}")

    loss_pytorch = output_pytorch.sum()
    print(f"Loss PyTorch: {loss_pytorch.item()}")
    loss_pytorch.backward()
    print(f"Pytorch implementation - input.grad shape: {input_pytorch.grad.shape if input_pytorch.grad is not None else 'None'}")
    print(f"Pytorch implementation - weight.grad shape: {ori_model.weight.grad.shape if ori_model.weight.grad is not None else 'None'}")
    print(f"Pytorch implementation - bias.grad shape: {ori_model.bias.grad.shape if ori_model.bias.grad is not None else 'None'}")

    ################################ input grad ################################
    print(f"\n==================================== input 梯度比较结果 ====================================")
    if input_custom.grad is not None and input_pytorch.grad is not None:
        grad_fused = input_custom.grad.float()
        grad_pytorch = input_pytorch.grad.float()
        grad_diff_abs = torch.max(torch.abs(grad_fused - grad_pytorch))
        grad_diff_rel = torch.max(torch.abs(grad_fused - grad_pytorch) / (torch.abs(grad_pytorch) + 1e-8))
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
        print(f"✗ 梯度计算失败：input_custom.grad is None ==> {input_custom.grad is None} OR input_pytorch.grad is None ==>{input_pytorch.grad is None}")

    ################################ weight grad ################################
    print(f"\n==================================== weight 梯度比较结果 ====================================")
    if custom_model.weight.grad is not None and ori_model.weight.grad is not None:
        grad_fused = custom_model.weight.grad.float()
        grad_pytorch = ori_model.weight.grad.float()
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
        print(f"✗ 梯度计算失败：custom_model.weight.grad is None ==> {custom_model.weight.grad is None} OR ori_model.weight.grad is None ==>{ori_model.weight.grad is None}")

    ################################ bias grad ################################
    print(f"\n==================================== bias 梯度比较结果 ====================================")
    if custom_model.bias.grad is not None and ori_model.bias.grad is not None:
        grad_fused = custom_model.bias.grad.float()
        grad_pytorch = ori_model.bias.grad.float()
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
        print(f"✗ 梯度计算失败：custom_model.bias.grad is None ==> {custom_model.bias.grad is None} OR ori_model.bias.grad is None ==>{ori_model.weight.grad is None}")

    if torch.cuda.is_available():
        import time
        # warmup
        for _ in range(10):
            _ = custom_model(input_custom)
            _ = ori_model(input_pytorch)
        start_time = time.time()
        LOOPS=1000
        for _ in range(LOOPS):
            _ = custom_model(input_custom)
        vfx_time = (time.time() - start_time) * 1000 / LOOPS
        start_time = time.time()
        for _ in range(LOOPS):
            _ = ori_model(input_pytorch)
        pytorch_time = (time.time() - start_time) * 1000 / LOOPS
        print(f"\nPerformance comparison:")
        print(f"CUDA time: {vfx_time:.4f}ms")
        print(f"PyTorch time: {pytorch_time:.4f}ms")
        if vfx_time > 0:
            print(f"Speedup: {pytorch_time / vfx_time:.2f}x")