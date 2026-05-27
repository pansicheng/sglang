"""Microbench: cost of `forward` vs `forward_return_lse` in FlashInfer.

Background
----------
gpt-oss requires sinks correction. On A10/Hopper the fa2/fa3 decode kernel
silently drops the `sinks` argument, so the SGLang FlashInfer backend
calls ``decode_wrapper.forward_return_lse(...)`` and applies a post-hoc
correction. This microbench isolates the marginal cost of asking the
FlashInfer decode kernel to write LSE alongside the output.

For each (batch, kv_len) we:
  1. Build a ``BatchDecodeWithPagedKVCacheWrapper`` with the same plan
     SGLang uses (NHD layout, sm_scale, etc.).
  2. Time ``forward`` (no LSE) and ``forward_return_lse`` (with LSE),
     using CUDA events; report median ms/call.

Run:
    python test/manual/bench_forward_return_lse.py
"""

import argparse
import math
import statistics

import torch
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper


def time_call(fn, iters: int, warmup: int) -> float:
    """Return median ms/call using CUDA events."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def build_wrapper(
    batch: int,
    kv_len: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    dtype: torch.dtype,
    device: str,
):
    """Build a planned BatchDecodeWithPagedKVCacheWrapper + matching paged KV."""
    pages_per_seq = math.ceil(kv_len / page_size)
    total_pages = pages_per_seq * batch
    last_page_len = kv_len - (pages_per_seq - 1) * page_size

    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace, kv_layout="NHD")

    # NHD: (num_pages, 2, page_size, num_kv_heads, head_dim)
    paged_kv = torch.randn(
        total_pages,
        2,
        page_size,
        num_kv_heads,
        head_dim,
        dtype=dtype,
        device=device,
    )

    indptr = (
        torch.arange(0, batch + 1, dtype=torch.int32, device=device) * pages_per_seq
    )
    indices = torch.arange(total_pages, dtype=torch.int32, device=device)
    last_page_lens = torch.full(
        (batch,), last_page_len, dtype=torch.int32, device=device
    )

    wrapper.plan(
        indptr,
        indices,
        last_page_lens,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        "NONE",
        q_data_type=dtype,
        kv_data_type=dtype,
    )

    q = torch.randn(batch, num_qo_heads, head_dim, dtype=dtype, device=device)
    return wrapper, q, paged_kv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-qo-heads", type=int, default=64)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    args = parser.parse_args()

    device = "cuda"
    dtype = getattr(torch, args.dtype)

    print(
        f"GPU: {torch.cuda.get_device_name(0)} "
        f"(SM {'.'.join(map(str, torch.cuda.get_device_capability(0)))})"
    )
    print(
        f"Q heads={args.num_qo_heads} KV heads={args.num_kv_heads} "
        f"head_dim={args.head_dim} page_size={args.page_size} dtype={args.dtype}"
    )
    print()

    header = (
        f"{'batch':>6} {'kv_len':>8} | "
        f"{'forward (us)':>14} {'forward_return_lse (us)':>24} | "
        f"{'delta (us)':>12} {'overhead %':>11} {'ratio':>7}"
    )
    print(header)
    print("-" * len(header))

    workloads = [
        (1, 256),
        (1, 1024),
        (1, 4096),
        (4, 256),
        (4, 1024),
        (4, 4096),
        (16, 256),
        (16, 1024),
        (16, 4096),
        (32, 1024),
        (32, 4096),
        (64, 1024),
        (64, 4096),
        (128, 1024),
    ]

    for batch, kv_len in workloads:
        wrapper, q, paged_kv = build_wrapper(
            batch,
            kv_len,
            args.num_qo_heads,
            args.num_kv_heads,
            args.head_dim,
            args.page_size,
            dtype,
            device,
        )

        # forward (no LSE)
        def run_no_lse(w=wrapper, q_=q, kv=paged_kv):
            w.run(q_, kv)

        # forward_return_lse
        def run_with_lse(w=wrapper, q_=q, kv=paged_kv):
            w.run(q_, kv, return_lse=True)

        t_no = time_call(run_no_lse, args.iters, args.warmup)
        t_lse = time_call(run_with_lse, args.iters, args.warmup)
        delta = t_lse - t_no
        pct = delta / t_no * 100.0 if t_no > 0 else 0.0
        ratio = t_lse / t_no if t_no > 0 else 0.0

        print(
            f"{batch:>6} {kv_len:>8} | "
            f"{t_no * 1e3:>14.2f} {t_lse * 1e3:>24.2f} | "
            f"{delta * 1e3:>12.2f} {pct:>10.2f}% {ratio:>6.2f}x"
        )

        # Free GPU mem between rows
        del wrapper, q, paged_kv
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
