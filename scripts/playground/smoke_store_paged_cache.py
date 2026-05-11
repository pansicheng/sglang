"""Smoke test for the paged JIT store_paged_cache kernel.

Builds a fake [P, 2, L, S, H, D] buffer, writes random K/V at random loc via
store_paged_cache, and compares against a reference scatter.
"""

import sys

import torch

from sglang.jit_kernel.kvcache import can_use_store_paged_cache, store_paged_cache


def _run_case(P, L, S, H, D, B, dtype, device="cuda"):
    row_dim = H * D
    kv_storage = torch.zeros((P, 2, L, S, H, D), dtype=dtype, device=device)
    kv_ref = kv_storage.clone()

    # loc in [S, P*S) so page_idx >= 1 (matches runtime, which reserves page 0).
    # Use randperm to guarantee unique slots -- otherwise duplicate (page,slot)
    # pairs yield non-deterministic scatter ordering between ref and kernel.
    assert B <= (P - 1) * S, "need unique loc across [S, P*S)"
    loc = (S + torch.randperm((P - 1) * S, device=device)[:B]).to(torch.int64)

    k = torch.randn((B, H, D), dtype=dtype, device=device).contiguous()
    v = torch.randn((B, H, D), dtype=dtype, device=device).contiguous()

    page_idx = loc // S
    slot = loc % S
    for layer_idx in range(L):
        kv_ref[page_idx, 0, layer_idx, slot] = k
        kv_ref[page_idx, 1, layer_idx, slot] = v

    flat = kv_storage.view(-1, row_dim)
    row_bytes = row_dim * kv_storage.element_size()
    assert can_use_store_paged_cache(row_bytes), f"row_bytes={row_bytes} not supported"

    k_flat = k.reshape(B, row_dim)
    v_flat = v.reshape(B, row_dim)
    for layer_idx in range(L):
        store_paged_cache(
            k_flat,
            v_flat,
            flat,
            loc,
            layer_idx,
            S,
            L,
            row_bytes=row_bytes,
        )

    if not torch.equal(kv_storage, kv_ref):
        mismatch = (kv_storage != kv_ref).nonzero()
        n = mismatch.shape[0]
        first = mismatch[0].tolist() if n else None
        raise AssertionError(
            f"mismatch in {n} elements; first at {first} "
            f"(kv_storage={kv_storage[tuple(first)]}, ref={kv_ref[tuple(first)]})"
        )


def main():
    torch.manual_seed(0)
    cases = [
        (8, 4, 16, 8, 64, 32, torch.bfloat16),
        (16, 36, 64, 8, 128, 128, torch.bfloat16),  # Qwen3-8B-like geometry
        (8, 4, 16, 8, 64, 1, torch.bfloat16),
        (4, 2, 8, 4, 64, 17, torch.float16),
        (4, 2, 8, 4, 64, 17, torch.float32),
    ]
    for i, args in enumerate(cases):
        print(
            f"[case {i}] P={args[0]} L={args[1]} S={args[2]} H={args[3]} "
            f"D={args[4]} B={args[5]} dtype={args[6]} ... ",
            end="",
            flush=True,
        )
        _run_case(*args)
        print("OK")
    print("all cases passed")


if __name__ == "__main__":
    sys.exit(main())
