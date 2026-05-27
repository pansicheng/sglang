"""End-to-end decode-step profiler for `forward_return_lse` overhead.

Runs `sglang.bench_one_batch` twice on the same workload:
  arm A (default): production path uses `decode_wrapper.forward_return_lse(...)`
  arm B (no-LSE):  monkey-patches `forward_return_lse` to call plain `forward`
                    and return a zero LSE tensor. This makes the sink-correction
                    output numerically wrong (fine — we are timing only) while
                    keeping the *shape* of the decode step (kernels launched,
                    sinks-correction kernel still fires) identical.

The wall-clock delta between arm A and arm B is the cost of asking the FlashInfer
fa2 decode kernel to also write LSE, integrated across all 24 layers of a full
decode step.

Environment toggle:
    BENCH_DISABLE_LSE=1  -> activate arm B (monkey-patch)

Usage (driven by the companion shell script):
    BENCH_DISABLE_LSE=0 python test/manual/profile_forward_return_lse_decode_step.py \\
        --model-path <path> --batch-size 1 --input-len 1024 --output-len 256
    BENCH_DISABLE_LSE=1 python test/manual/profile_forward_return_lse_decode_step.py \\
        --model-path <path> --batch-size 1 --input-len 1024 --output-len 256
"""

import os

import torch


def maybe_patch_forward_return_lse() -> None:
    """If BENCH_DISABLE_LSE=1, replace forward_return_lse with a no-LSE variant.

    The replacement still calls the FlashInfer decode kernel (so all attention
    work is performed and timed), but uses plain `forward` (no LSE write) and
    fabricates a zero LSE tensor so the downstream sinks-correction call still
    runs without error. This isolates the cost of the LSE write itself from
    everything else in the decode step.
    """
    if os.environ.get("BENCH_DISABLE_LSE", "0") != "1":
        return

    from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

    original_forward_return_lse = BatchDecodeWithPagedKVCacheWrapper.forward_return_lse

    def patched_forward_return_lse(self, q, paged_kv_cache, *args, **kwargs):
        # Drop LSE: call run(...) without return_lse, fabricate zero LSE.
        # We use self.run (not self.forward, which silently drops sm_scale /
        # logits_soft_cap and would change the kernel selection).
        kwargs.pop("return_lse", None)
        out = self.run(q, paged_kv_cache, *args, **kwargs)
        # LSE shape is [num_qo_tokens, num_qo_heads], fp32 on same device.
        lse = torch.zeros(
            (q.shape[0], q.shape[1]),
            dtype=torch.float32,
            device=q.device,
        )
        return out, lse

    BatchDecodeWithPagedKVCacheWrapper.forward_return_lse = patched_forward_return_lse
    print(
        "[profile_forward_return_lse] BENCH_DISABLE_LSE=1: patched "
        "BatchDecodeWithPagedKVCacheWrapper.forward_return_lse to skip LSE write",
        flush=True,
    )


def main():
    maybe_patch_forward_return_lse()

    # Hand off to the standard bench_one_batch entrypoint.
    import argparse

    from sglang.bench_one_batch import BenchArgs
    from sglang.bench_one_batch import main as bench_main
    from sglang.srt.server_args import ServerArgs

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()

    server_args = ServerArgs.from_cli_args(args)
    bench_args = BenchArgs.from_cli_args(args)
    bench_main(server_args, bench_args)


if __name__ == "__main__":
    main()
