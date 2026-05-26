"""Validate FlashInfer LSE-correction path matches Triton native-sink path.

Tests the mathematical equivalence:
    output_with_sink = output_no_sink * sigmoid(lse - sink)

Usage:
    python -m pytest test/manual/test_sinks_correction_parity.py -xvs
"""

import math
import unittest

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.flashinfer_backend import _apply_sinks_correction
from sglang.srt.layers.attention.triton_ops.decode_attention import decode_attention_fwd
from sglang.srt.layers.attention.triton_ops.extend_attention import extend_attention_fwd

# Default test dimensions (GQA: 64 query heads, 8 KV heads, dim=64)
H_Q, H_KV, D = 64, 8, 64
DTYPE = torch.bfloat16


def _sm_scale():
    return 1.0 / (D**0.5)


def _has_flashinfer():
    try:
        from flashinfer import BatchPrefillWithRaggedKVCacheWrapper  # noqa: F401

        return True
    except ImportError:
        return False


def reference_attention_with_sinks(q, k, v, sinks, sm_scale, causal=True):
    """fp32 reference: returns (output_with_sink, output_no_sink, lse_no_sink)."""
    num_tokens, n_q, d = q.shape
    _, n_kv, d_v = v.shape
    g = n_q // n_kv

    q_f, k_f, v_f = q.float(), k.float(), v.float()
    if g > 1:
        k_f = k_f.repeat_interleave(g, dim=1)
        v_f = v_f.repeat_interleave(g, dim=1)

    scores = torch.einsum("thd,shd->ths", q_f, k_f) * sm_scale
    if causal:
        mask = torch.triu(
            torch.ones(num_tokens, num_tokens, device=q.device, dtype=torch.bool),
            diagonal=1,
        )
        scores.masked_fill_(mask.unsqueeze(1), float("-inf"))

    lse = torch.logsumexp(scores, dim=-1)
    out_no_sink = torch.einsum("ths,shd->thd", F.softmax(scores, dim=-1), v_f)

    exp_s = scores.exp().masked_fill(scores == float("-inf"), 0)
    numer = torch.einsum("ths,shd->thd", exp_s, v_f)
    denom = exp_s.sum(-1) + sinks.float().unsqueeze(0).exp()
    out_with_sink = numer / denom.unsqueeze(-1)

    return out_with_sink, out_no_sink, lse


class TestSinksCorrectionMath(unittest.TestCase):
    """Pure math validation of _apply_sinks_correction."""

    def setUp(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

    def test_fp32_exact(self):
        """Correction is exact in fp32.

        ``_apply_sinks_correction`` expects log2-based LSE (FlashInfer
        contract), so we convert the natural-log reference LSE to log2
        before passing it in.
        """
        q = torch.randn(16, 8, D, device="cuda")
        k = torch.randn(16, 8, D, device="cuda")
        v = torch.randn(16, 8, D, device="cuda")
        sinks = torch.randn(8, device="cuda")

        ref, no_sink, lse_ln = reference_attention_with_sinks(
            q, k, v, sinks, _sm_scale()
        )
        lse_log2 = lse_ln / math.log(2.0)
        corrected = _apply_sinks_correction(no_sink, lse_log2, sinks)

        diff = (ref - corrected).abs().max().item()
        print(f"\n[fp32] max_diff={diff:.2e}")
        self.assertLess(diff, 1e-5)

    def test_boundary_conditions(self):
        """sink=-inf => identity, sink=+inf => zeros.

        Boundary behavior is independent of LSE base (sigmoid saturates),
        so any finite ``lse`` works regardless of base.
        """
        out = torch.randn(16, 8, D, dtype=DTYPE, device="cuda")
        lse = torch.randn(16, 8, dtype=DTYPE, device="cuda")

        # sink=-inf: correction factor = sigmoid(lse - (-inf)) = 1
        c = _apply_sinks_correction(out, lse, torch.full((8,), -100.0, device="cuda"))
        self.assertLess((c.float() - out.float()).abs().max().item(), 1e-3)

        # sink=+inf: correction factor = sigmoid(lse - (+inf)) = 0
        c = _apply_sinks_correction(out, lse, torch.full((8,), 100.0, device="cuda"))
        self.assertLess(c.float().abs().max().item(), 1e-3)


class TestTritonNativeSinks(unittest.TestCase):
    """Triton kernels: native sink output vs fp32 reference."""

    def setUp(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

    def test_extend(self):
        """extend_attention_fwd with sinks produces valid, changed output."""
        B, prefix_len, extend_len = 4, 32, 16
        total_ext = B * extend_len
        total_pre = B * prefix_len
        device = "cuda"
        sinks = torch.randn(H_Q, device=device, dtype=torch.float32)

        q = torch.randn(total_ext, H_Q, D, dtype=DTYPE, device=device)
        k = torch.randn(total_ext, H_KV, D, dtype=DTYPE, device=device)
        v = torch.randn(total_ext, H_KV, D, dtype=DTYPE, device=device)
        k_buf = torch.randn(total_pre, H_KV, D, dtype=DTYPE, device=device)
        v_buf = torch.randn(total_pre, H_KV, D, dtype=DTYPE, device=device)

        qo_indptr = torch.arange(
            0, (B + 1) * extend_len, extend_len, device=device, dtype=torch.int32
        )
        kv_indptr = torch.arange(
            0, (B + 1) * prefix_len, prefix_len, device=device, dtype=torch.int32
        )
        kv_indices = torch.arange(total_pre, device=device, dtype=torch.int32)

        def run(s):
            o = torch.empty(total_ext, H_Q, D, dtype=DTYPE, device=device)
            extend_attention_fwd(
                q,
                k,
                v,
                o,
                k_buf,
                v_buf,
                qo_indptr,
                kv_indptr,
                kv_indices,
                custom_mask=None,
                is_causal=True,
                mask_indptr=None,
                max_len_extend=extend_len,
                k_scale=1.0,
                v_scale=1.0,
                sm_scale=_sm_scale(),
                sliding_window_size=-1,
                sinks=s,
            )
            return o

        o_sink = run(sinks)
        o_none = run(None)
        self.assertFalse(torch.isnan(o_sink).any())
        diff = (o_sink.float() - o_none.float()).abs().mean().item()
        print(f"\n[Triton extend] mean diff with/without sink: {diff:.4e}")
        self.assertGreater(diff, 1e-4)

    def test_decode(self):
        """decode_attention_fwd with sinks matches fp32 reference."""
        B, seq_len = 8, 128
        total = B * seq_len
        device = "cuda"
        sinks = torch.randn(H_Q, device=device, dtype=torch.float32)
        max_kv_splits = 8

        q = torch.randn(B, H_Q, D, dtype=DTYPE, device=device)
        k_buf = torch.randn(total, H_KV, D, dtype=DTYPE, device=device)
        v_buf = torch.randn(total, H_KV, D, dtype=DTYPE, device=device)
        kv_indptr = torch.arange(
            0, (B + 1) * seq_len, seq_len, device=device, dtype=torch.int32
        )
        kv_indices = torch.arange(total, device=device, dtype=torch.int32)

        o = torch.zeros(B, H_Q, D, dtype=DTYPE, device=device)
        decode_attention_fwd(
            q,
            k_buf,
            v_buf,
            o,
            kv_indptr,
            kv_indices,
            torch.empty(B, H_Q, max_kv_splits, D, dtype=torch.float32, device=device),
            torch.empty(B, H_Q, max_kv_splits, D, dtype=torch.float32, device=device),
            torch.full((B,), 4, dtype=torch.int32, device=device),
            max_kv_splits,
            _sm_scale(),
            1.0,
            1.0,
            logit_cap=0.0,
            sinks=sinks,
        )
        self.assertFalse(torch.isnan(o).any())

        # Compare against fp32 ref
        g = H_Q // H_KV
        max_d = 0.0
        for b in range(B):
            ks = (
                k_buf[b * seq_len : (b + 1) * seq_len]
                .float()
                .repeat_interleave(g, dim=1)
            )
            vs = (
                v_buf[b * seq_len : (b + 1) * seq_len]
                .float()
                .repeat_interleave(g, dim=1)
            )
            logits = torch.einsum("hd,lhd->hl", q[b].float(), ks) * _sm_scale()
            e = logits.exp()
            ref = torch.einsum("hl,lhd->hd", e, vs) / (
                e.sum(-1) + sinks.float().exp()
            ).unsqueeze(-1)
            max_d = max(max_d, (ref - o[b].float()).abs().max().item())

        print(f"\n[Triton decode] max diff vs fp32 ref: {max_d:.4e}")
        self.assertLess(max_d, 0.15)


@unittest.skipUnless(_has_flashinfer(), "FlashInfer not installed")
class TestFlashInferVsTriton(unittest.TestCase):
    """FlashInfer (no sinks + LSE correction) vs Triton (native sinks)."""

    def setUp(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

    def test_prefill(self):
        """FlashInfer ragged prefill + correction ≈ Triton extend with native sinks."""
        from flashinfer import BatchPrefillWithRaggedKVCacheWrapper

        B, seq_len = 4, 32
        total = B * seq_len
        device = "cuda"
        sinks = torch.randn(H_Q, device=device, dtype=torch.float32)

        q = torch.randn(total, H_Q, D, dtype=DTYPE, device=device)
        k = torch.randn(total, H_KV, D, dtype=DTYPE, device=device)
        v = torch.randn(total, H_KV, D, dtype=DTYPE, device=device)
        qo_indptr = torch.arange(
            0, (B + 1) * seq_len, seq_len, device=device, dtype=torch.int32
        )

        # FlashInfer: no sinks + LSE correction
        ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        wrapper = BatchPrefillWithRaggedKVCacheWrapper(ws, "NHD")
        wrapper.begin_forward(qo_indptr, qo_indptr, H_Q, H_KV, D, q_data_type=DTYPE)
        o_fi, lse = wrapper.forward_return_lse(
            q, k, v, causal=True, sm_scale=_sm_scale()
        )
        o_fi = _apply_sinks_correction(o_fi, lse, sinks)

        # Triton: native sinks (no prefix)
        o_tr = torch.empty_like(q).view(total, H_Q, D)
        extend_attention_fwd(
            q,
            k,
            v,
            o_tr,
            torch.empty(0, H_KV, D, dtype=DTYPE, device=device),
            torch.empty(0, H_KV, D, dtype=DTYPE, device=device),
            qo_indptr,
            torch.zeros(B + 1, device=device, dtype=torch.int32),
            torch.empty(0, device=device, dtype=torch.int32),
            custom_mask=None,
            is_causal=True,
            mask_indptr=None,
            max_len_extend=seq_len,
            k_scale=1.0,
            v_scale=1.0,
            sm_scale=_sm_scale(),
            sliding_window_size=-1,
            sinks=sinks,
        )

        diff = (o_fi.float() - o_tr.float()).abs()
        print(
            f"\n[FI vs TR prefill] max={diff.max().item():.4e}, mean={diff.mean().item():.4e}"
        )
        self.assertLess(diff.mean().item(), 0.03)

    def test_decode(self):
        """FlashInfer decode + correction ≈ Triton decode with native sinks."""
        from flashinfer import BatchDecodeWithPagedKVCacheWrapper

        from sglang.srt.layers.attention.flashinfer_backend import (
            should_use_tensor_core,
        )

        B, seq_len = 8, 128
        total = B * seq_len
        device = "cuda"
        sinks = torch.randn(H_Q, device=device, dtype=torch.float32)

        q = torch.randn(B, H_Q, D, dtype=DTYPE, device=device)
        k_buf = torch.randn(total, H_KV, D, dtype=DTYPE, device=device)
        v_buf = torch.randn(total, H_KV, D, dtype=DTYPE, device=device)
        kv_indptr = torch.arange(
            0, (B + 1) * seq_len, seq_len, device=device, dtype=torch.int32
        )
        kv_indices = torch.arange(total, device=device, dtype=torch.int32)

        # FlashInfer decode
        ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        wrapper = BatchDecodeWithPagedKVCacheWrapper(
            ws,
            "NHD",
            use_tensor_cores=should_use_tensor_core(
                kv_cache_dtype=DTYPE, num_attention_heads=H_Q, num_kv_heads=H_KV
            ),
        )
        wrapper.begin_forward(
            kv_indptr,
            kv_indices,
            torch.ones(B, dtype=torch.int32, device=device),
            H_Q,
            H_KV,
            D,
            1,
            pos_encoding_mode="NONE",
            q_data_type=DTYPE,
            data_type=DTYPE,
        )
        o_fi, lse = wrapper.forward_return_lse(
            q.view(-1, H_Q, D), (k_buf, v_buf), sm_scale=_sm_scale()
        )
        o_fi = _apply_sinks_correction(o_fi, lse, sinks)

        # Triton decode
        max_kv_splits = 8
        o_tr = torch.zeros(B, H_Q, D, dtype=DTYPE, device=device)
        decode_attention_fwd(
            q,
            k_buf,
            v_buf,
            o_tr,
            kv_indptr,
            kv_indices,
            torch.empty(B, H_Q, max_kv_splits, D, dtype=torch.float32, device=device),
            torch.empty(B, H_Q, max_kv_splits, D, dtype=torch.float32, device=device),
            torch.full((B,), 4, dtype=torch.int32, device=device),
            max_kv_splits,
            _sm_scale(),
            1.0,
            1.0,
            logit_cap=0.0,
            sinks=sinks,
        )

        diff = (o_fi.view(B, H_Q, D).float() - o_tr.float()).abs()
        print(
            f"\n[FI vs TR decode] max={diff.max().item():.4e}, mean={diff.mean().item():.4e}"
        )
        self.assertLess(diff.mean().item(), 0.03)


if __name__ == "__main__":
    unittest.main()
