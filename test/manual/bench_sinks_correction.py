"""Microbench: cost of `_apply_sinks_correction` per layer, decode-step shapes.

gpt-oss-20b: 24 layers, 64 Q heads, head_dim=64. Decode batch ranges 1..N.
We bench three variants on shapes that match the production decode path:
    1. Current production helper (fp32 round-trip on output, fp32 sigmoid)
    2. dtype-preserving variant (sigmoid in fp32 on small [N,H], multiply in bf16)
    3. dtype-preserving in-place variant
And we report ms/call and est. ms/decode-step (= 24 * ms/call) for context.

Run:
    python test/manual/bench_sinks_correction.py
"""

import math
import time

import torch

LN2 = math.log(2.0)


def cur(output, lse, sinks):
    """Current production code in flashinfer_backend._apply_sinks_correction."""
    lse_ln = lse.float() * LN2
    correction = torch.sigmoid(lse_ln - sinks.float().unsqueeze(0))
    return (output.float() * correction.unsqueeze(-1)).to(output.dtype)


def dtype_preserving(output, lse, sinks):
    """Compute correction in fp32 on small [N, H] tensor; broadcast-multiply in original dtype."""
    correction = torch.sigmoid(lse.float() * LN2 - sinks.float().unsqueeze(0)).to(
        output.dtype
    )
    return output * correction.unsqueeze(-1)


def dtype_preserving_inplace(output, lse, sinks):
    """As dtype_preserving but multiplies in-place."""
    correction = torch.sigmoid(lse.float() * LN2 - sinks.float().unsqueeze(0)).to(
        output.dtype
    )
    output.mul_(correction.unsqueeze(-1))
    return output


from sglang.srt.layers.attention.flashinfer_backend import (
    _apply_sinks_correction as fused_triton,
)


def time_fn(fn, args, iters=2000, warmup=200):
    """Time fn(*args) repeatedly with the same pre-allocated tensors.

    Some variants (in-place) mutate args, so we re-clone the output tensor
    each iter to keep timing on a clean state. lse and sinks are read-only,
    so they are reused as-is.
    """
    out_template, lse, sinks = args
    torch.cuda.synchronize()
    for _ in range(warmup):
        out = out_template.clone()
        fn(out, lse, sinks)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = out_template.clone()
        fn(out, lse, sinks)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    # Subtract clone-only baseline so we report op-only time.
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    for _ in range(iters):
        out = out_template.clone()
    torch.cuda.synchronize()
    t3 = time.perf_counter()
    op_time = (t1 - t0) - (t3 - t2)
    return op_time / iters * 1e3  # ms/call


def make_args(N, H, D, dtype, device):
    out = torch.randn(N, H, D, dtype=dtype, device=device)
    lse = torch.randn(N, H, dtype=torch.float32, device=device)
    sinks = torch.randn(H, dtype=dtype, device=device)
    return out, lse, sinks


def main():
    device = "cuda"
    dtype = torch.bfloat16
    H = 64
    D = 64
    NUM_LAYERS = 24

    print(
        f"{'shape':<22} {'cur (us)':<12} {'dtype-pres (us)':<18} {'inplace (us)':<14} | "
        f"{'cur*L (ms)':<12} {'dtype-pres*L (ms)':<18} {'savings@L=24 (ms)':<14}"
    )
    print("-" * 120)

    print(
        f"{'shape':<14} {'cur(us)':<10} {'inplace(us)':<14} {'fused(us)':<12} | {'cur*L24(ms)':<14} {'fused*L24(ms)':<16} {'savings(ms)':<14} {'speedup vs cur':<16}"
    )
    print("-" * 130)
    for N in (1, 4, 16, 32, 64, 128, 256, 1024):
        args = make_args(N, H, D, dtype, device)
        cur_t = time_fn(cur, args)
        ip_t = time_fn(dtype_preserving_inplace, args)
        ft_t = time_fn(fused_triton, args)
        cur_layer = cur_t * NUM_LAYERS
        ft_layer = ft_t * NUM_LAYERS
        savings = (cur_t - ft_t) * NUM_LAYERS
        speedup = cur_t / ft_t if ft_t > 0 else float("inf")
        print(
            f"N={N:<10} {cur_t * 1e3:<10.2f} {ip_t * 1e3:<14.2f} {ft_t * 1e3:<12.2f} | "
            f"{cur_layer:<14.3f} {ft_layer:<16.3f} {savings:<14.3f} {speedup:<16.2f}x"
        )


if __name__ == "__main__":
    main()
