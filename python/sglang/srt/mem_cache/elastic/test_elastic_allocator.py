"""
Test elastic_allocator.py

cd /sgl-workspace/sglang/python/sglang/srt/mem_cache/elastic
LD_PRELOAD=/usr/local/lib/python3.12/dist-packages/torch_memory_saver_hook_mode_preload.abi3.so \
SGLANG_ELASTIC_MEM_POOL=true \
SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
pytest test_elastic_allocator.py -s
"""

import pytest
import torch

from sglang.srt.mem_cache.elastic.elastic_allocator import (
    ElasticTokenToKVPoolAllocator,
    get_tail_consecutive_start,
)
from sglang.srt.mem_cache.elastic.elasticmem_orchestrator import USE_ELASTICMEM


@pytest.fixture
def skip_if_no_cuda():
    """Skip tests if CUDA is not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")


def test_get_tail_consecutive_start():
    """Test get_tail_consecutive_start function."""
    # Empty tensor
    empty = torch.tensor([], dtype=torch.bool)
    assert get_tail_consecutive_start(empty) == 0

    # All True
    all_true = torch.tensor([True, True, True], dtype=torch.bool)
    assert get_tail_consecutive_start(all_true) == 0

    # All False
    all_false = torch.tensor([False, False, False], dtype=torch.bool)
    assert get_tail_consecutive_start(all_false) == 3

    # Mixed: [False, True, True]
    mixed1 = torch.tensor([False, True, True], dtype=torch.bool)
    assert get_tail_consecutive_start(mixed1) == 1

    # Mixed: [False, False, True, True]
    mixed2 = torch.tensor([False, False, True, True], dtype=torch.bool)
    assert get_tail_consecutive_start(mixed2) == 2

    # Mixed: [True, False, True]
    mixed3 = torch.tensor([True, False, True], dtype=torch.bool)
    assert get_tail_consecutive_start(mixed3) == 2


def test_elastic_allocator_thresholds(skip_if_no_cuda):
    """Test ElasticAllocator threshold constants are defined."""
    from sglang.srt.mem_cache.elastic.elasticmem_orchestrator import (
        CAN_MAP_THRESHOLD,
        CAN_UNMAP_THRESHOLD,
        RESIZE_TRIGGER_DIFF_RATIO,
    )

    # Verify thresholds are valid floats
    assert isinstance(CAN_UNMAP_THRESHOLD, float)
    assert isinstance(CAN_MAP_THRESHOLD, float)
    assert isinstance(RESIZE_TRIGGER_DIFF_RATIO, float)

    # Verify logical ordering
    assert 0.0 <= CAN_UNMAP_THRESHOLD <= 1.0
    assert 0.0 <= CAN_MAP_THRESHOLD <= 1.0
    assert CAN_UNMAP_THRESHOLD < CAN_MAP_THRESHOLD


def test_elastic_allocator_class_interface(skip_if_no_cuda):
    """Test ElasticAllocator abstract class interface."""
    from sglang.srt.mem_cache.elastic.elasticmem_orchestrator import ElasticAllocator

    # Verify abstract methods are defined
    abstract_methods = [
        "can_unmap",
        "can_map",
        "reduce",
        "expand",
        "cu_page_to_token",
        "register_evict_func",
        "token_usage",
        "evictable_size",
        "evict",
        "update_size",
    ]

    for method_name in abstract_methods:
        assert hasattr(ElasticAllocator, method_name)


def test_elastic_token_allocator_mock(skip_if_no_cuda):
    """Test ElasticTokenToKVPoolAllocator with mock KV cache."""
    if not USE_ELASTICMEM:
        pytest.skip(f"Elastic memory not enabled: {USE_ELASTICMEM=}")

    # This test requires ElasticMHATokenToKVPool which needs CUDA VMM
    # We just verify the class can be imported and has expected methods
    expected_methods = [
        "clear",
        "alloc",
        "free",
        "available_size",
        "can_unmap",
        "can_map",
        "can_do_unmap",
        "reduce",
        "expand",
        "token_usage",
        "evictable_size",
        "evict",
        "update_size",
        "mark_unmap_candidate",
        "register_evict_func",
        "register_scheduler",
    ]

    for method_name in expected_methods:
        assert hasattr(ElasticTokenToKVPoolAllocator, method_name)
