"""Warm a small, model-agnostic set of common CUDA operators."""

import gc
import time

import triton
import triton.language as tl


@triton.jit
def _add_one_copy_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(y_ptr + offsets, values + 1.0, mask=mask)


def warmup(device):
    """Exercise fixed-size framework and Triton kernels without model or PG."""
    import torch
    import torch._dynamo
    import torch.distributed.run
    import torch.optim
    import torch.nn.functional as F

    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("common GPU warmup requires a CUDA device")
    if torch.distributed.is_initialized():
        raise RuntimeError("common warmup must run before training process-group initialization")

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    bf16_supported = torch.cuda.is_bf16_supported()

    a = torch.randn((64, 64), device=device, dtype=torch.float32)
    b = torch.randn((64, 64), device=device, dtype=torch.float32)
    mm32 = torch.mm(a, b)

    mm_bf16 = None
    if bf16_supported:
        mm_bf16 = torch.mm(a.to(torch.bfloat16), b.to(torch.bfloat16))

    x = torch.randn((64, 128), device=device, dtype=torch.float32, requires_grad=True)
    activated = F.silu(x)
    normalized = F.layer_norm(activated, (128,))
    probabilities = F.softmax(normalized, dim=-1)
    top_values, top_indices = torch.topk(probabilities, 16, dim=-1)
    (top_values.sum() + mm32.sum() * 1e-6).backward()

    triton_input = torch.zeros((1024,), device=device, dtype=torch.float32)
    triton_output = torch.empty_like(triton_input)
    _add_one_copy_kernel[(triton.cdiv(1024, 128),)](
        triton_input, triton_output, 1024, 128, num_warps=4
    )

    parameter = torch.nn.Parameter(torch.randn((128,), device=device, dtype=torch.float32))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)

    del optimizer, parameter, triton_output, triton_input
    del top_indices, top_values, probabilities, normalized
    del activated, x, mm_bf16, mm32, a, b
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    cuda_initialized = torch.cuda.is_initialized()
    if not cuda_initialized:
        raise RuntimeError("common warmup did not initialize the CUDA context")
    if torch.distributed.is_initialized():
        raise RuntimeError("common warmup unexpectedly initialized training process groups")

    return {
        "common_warmup_seconds": time.perf_counter() - started,
        "cuda_initialized": cuda_initialized,
        "cuda_allocated_bytes": torch.cuda.memory_allocated(device),
        "cuda_reserved_bytes": torch.cuda.memory_reserved(device),
        "cuda_peak_allocated_bytes": peak_allocated,
        "cuda_peak_reserved_bytes": peak_reserved,
        "bf16_supported": bf16_supported,
        "triton_runtime_warmed": True,
    }
