"""
Test paged KV cache layout (SGLANG_PAGED_KV_LAYOUT=1) with Triton attention backend.

This verifies that the paged KV cache layout produces correct results
(MMLU accuracy) with different page sizes.

Usage:
SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 python3 -m pytest test/registered/attention/test_paged_kv_layout.py -xvs
"""

import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# In case of some machine lack internet connection, we can set OFFLINE_MODE to True.
OFFLINE_MODE = True

# Change the path below when OFFLINE_MODE is True.
OFFLINE_PATH_DICT = {
    DEFAULT_MODEL_NAME_FOR_TEST: "/data-mnt/Qwen3-8B/",
}

if OFFLINE_MODE:
    MODEL_PATH = OFFLINE_PATH_DICT[DEFAULT_MODEL_NAME_FOR_TEST]
else:
    MODEL_PATH = DEFAULT_MODEL_NAME_FOR_TEST

# Environment with paged KV layout enabled
PAGED_KV_ENV = {**os.environ, "SGLANG_PAGED_KV_LAYOUT": "1"}


class TestPagedKVLayout(CustomTestCase):
    """Integration tests for paged KV cache layout."""

    def test_mmlu(self):
        """Verify MMLU accuracy with paged KV layout (default page_size=1)."""
        model = MODEL_PATH
        base_url = DEFAULT_URL_FOR_TEST
        process = popen_launch_server(
            model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--attention-backend", "triton"],
            env=PAGED_KV_ENV,
        )

        try:
            args = SimpleNamespace(
                base_url=base_url,
                model=model,
                eval_name="mmlu",
                num_examples=64,
                num_threads=32,
            )

            metrics = run_eval(args)
            print(f"MMLU score with paged KV layout: {metrics['score']}")
            self.assertGreaterEqual(metrics["score"], 0.65)
        finally:
            kill_process_tree(process.pid)

    def test_mmlu_page_size_16(self):
        """Verify MMLU accuracy with paged KV layout and page_size=16."""
        model = MODEL_PATH
        base_url = DEFAULT_URL_FOR_TEST
        process = popen_launch_server(
            model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "triton",
                "--page-size",
                "16",
            ],
            env=PAGED_KV_ENV,
        )

        try:
            args = SimpleNamespace(
                base_url=base_url,
                model=model,
                eval_name="mmlu",
                num_examples=64,
                num_threads=32,
            )

            metrics = run_eval(args)
            print(f"MMLU score with paged KV layout (page_size=16): {metrics['score']}")
            self.assertGreaterEqual(metrics["score"], 0.65)
        finally:
            kill_process_tree(process.pid)


    def test_mmlu_torch_native(self):
        """Verify MMLU accuracy with paged KV layout and torch_native backend."""
        model = MODEL_PATH
        base_url = DEFAULT_URL_FOR_TEST
        process = popen_launch_server(
            model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--attention-backend", "torch_native"],
            env=PAGED_KV_ENV,
        )

        try:
            args = SimpleNamespace(
                base_url=base_url,
                model=model,
                eval_name="mmlu",
                num_examples=64,
                num_threads=32,
            )

            metrics = run_eval(args)
            print(f"MMLU score with paged KV layout (torch_native): {metrics['score']}")
            self.assertGreaterEqual(metrics["score"], 0.65)
        finally:
            kill_process_tree(process.pid)

    def test_mmlu_torch_native_page_size_16(self):
        """Verify MMLU accuracy with paged KV layout, torch_native backend, page_size=16."""
        model = MODEL_PATH
        base_url = DEFAULT_URL_FOR_TEST
        process = popen_launch_server(
            model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "torch_native",
                "--page-size",
                "16",
            ],
            env=PAGED_KV_ENV,
        )

        try:
            args = SimpleNamespace(
                base_url=base_url,
                model=model,
                eval_name="mmlu",
                num_examples=64,
                num_threads=32,
            )

            metrics = run_eval(args)
            print(f"MMLU score with paged KV layout (torch_native, page_size=16): {metrics['score']}")
            self.assertGreaterEqual(metrics["score"], 0.65)
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
