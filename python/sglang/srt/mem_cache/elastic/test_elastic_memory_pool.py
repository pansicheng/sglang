"""
cd /sgl-workspace/sglang/python/sglang/srt/mem_cache/elastic
LD_PRELOAD=/usr/local/lib/python3.12/dist-packages/torch_memory_saver_hook_mode_preload.abi3.so \
SGLANG_ELASTIC_MEM_POOL=true \
SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
pytest test_elastic_memory_pool.py -s
"""

import pytest
import torch

from sglang.srt.mem_cache.elastic.elastic_memory_pool import ElasticSWAKVPool
from sglang.srt.mem_cache.elastic.elasticmem_orchestrator import USE_ELASTICMEM
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool


@pytest.fixture
def skip_if_no_cuda_elasticmem():
    """Skip tests if CUDA or elastic memory is not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if not USE_ELASTICMEM:
        pytest.skip(f"Elastic memory not enabled: {USE_ELASTICMEM=}")


def test_elastic_swa_kvpool(skip_if_no_cuda_elasticmem):
    page_size = 1
    dtype = torch.bfloat16
    head_num = 8
    head_dim = 64
    swa_attention_layer_ids = [i for i in range(24) if i % 2 == 0]
    full_attention_layer_ids = [i for i in range(24) if i % 2 == 1]
    enable_kvcache_transpose = False
    device = "cuda"
    token_to_kv_pool_class = MHATokenToKVPool
    enable_memory_saver = False

    def get_size():
        free_memory, _ = torch.cuda.mem_get_info()
        memory_per_token_per_layer = head_num * head_dim * dtype.itemsize * 2
        swa_memory_per_token = memory_per_token_per_layer * len(swa_attention_layer_ids)
        full_memory_per_token = memory_per_token_per_layer * len(
            full_attention_layer_ids
        )

        memory_for_full = free_memory * 1 / 3
        memory_for_swa = memory_for_full
        size = (
            (int(memory_for_full / full_memory_per_token) + page_size - 1)
            // page_size
            * page_size
        )
        size_swa = (
            (int(memory_for_swa / swa_memory_per_token) + page_size - 1)
            // page_size
            * page_size
        )
        return size, size_swa, memory_for_full, memory_for_swa

    size, size_swa, memory_for_full, memory_for_swa = get_size()

    free_memory_before, _ = torch.cuda.mem_get_info()

    elastic_swa_kvpool = ElasticSWAKVPool(
        size=size,
        size_swa=size_swa,
        page_size=page_size,
        dtype=dtype,
        head_num=head_num,
        head_dim=head_dim,
        swa_attention_layer_ids=swa_attention_layer_ids,
        full_attention_layer_ids=full_attention_layer_ids,
        enable_kvcache_transpose=enable_kvcache_transpose,
        device=device,
        token_to_kv_pool_class=token_to_kv_pool_class,
        enable_memory_saver=enable_memory_saver,
    )

    free_memory_after, _ = torch.cuda.mem_get_info()
    memory_used = free_memory_before - free_memory_after
    expected_memory = memory_for_full + memory_for_swa
    memory_fluctuation = abs(memory_used - expected_memory)
    assert memory_fluctuation < (128 << 20)

    swa_k_size, swa_v_size = elastic_swa_kvpool.swa_kv_pool.get_kv_size_bytes()
    full_k_size, full_v_size = elastic_swa_kvpool.full_kv_pool.get_kv_size_bytes()
    total_kv_size = swa_k_size + swa_v_size + full_k_size + full_v_size
    memory_fluctuation = abs(memory_used - total_kv_size)
    assert abs(memory_used - total_kv_size) < (128 << 20)

    assert elastic_swa_kvpool is not None
    assert hasattr(elastic_swa_kvpool, "swa_kv_pool")
    assert hasattr(elastic_swa_kvpool, "full_kv_pool")
    assert elastic_swa_kvpool.swa_layer_nums == len(swa_attention_layer_ids)
    assert elastic_swa_kvpool.full_layer_nums == len(full_attention_layer_ids)

    # Shrink SWA by 1/4 tokens and expand full by 1/4 tokens
    current_swa_size = elastic_swa_kvpool.swa_kv_pool.size
    shrink_tokens = (current_swa_size // 4) // page_size * page_size
    new_swa_size = current_swa_size - shrink_tokens
    new_full_size = elastic_swa_kvpool.full_kv_pool.size + shrink_tokens
    new_swa_size = new_swa_size // page_size * page_size
    new_full_size = new_full_size // page_size * page_size

    # Test data integrity
    layer = 0
    state_shape = (head_num, head_dim)
    swa = elastic_swa_kvpool.swa_kv_pool
    full = elastic_swa_kvpool.full_kv_pool

    # Initialize test data
    test_data = _init_test_data(swa, full, layer, state_shape, dtype, device)

    # Perform shrink/expand
    free_before, _ = torch.cuda.mem_get_info()
    swa.shrink(new_swa_size)
    full.expand(new_full_size)
    free_after, _ = torch.cuda.mem_get_info()
    memory_fluctuation = abs(free_before - free_after)
    assert memory_fluctuation < (128 << 20)

    # Verify data integrity
    _verify_test_data(
        swa,
        full,
        layer,
        new_swa_size,
        new_full_size,
        state_shape,
        dtype,
        device,
        test_data,
    )


def _init_test_data(swa, full, layer, state_shape, dtype, device):
    """Initialize test data for integrity verification."""
    test_data = {
        "swa_k0": torch.randn(state_shape, dtype=dtype, device=device),
        "swa_v0": torch.randn(state_shape, dtype=dtype, device=device),
        "full_k0": torch.randn(state_shape, dtype=dtype, device=device),
        "full_v0": torch.randn(state_shape, dtype=dtype, device=device),
    }

    swa.k_buffer[layer][0, :, :] = test_data["swa_k0"]
    swa.v_buffer[layer][0, :, :] = test_data["swa_v0"]
    full.k_buffer[layer][0, :, :] = test_data["full_k0"]
    full.v_buffer[layer][0, :, :] = test_data["full_v0"]

    return test_data


def _verify_test_data(
    swa, full, layer, new_swa_size, new_full_size, state_shape, dtype, device, test_data
):
    """Verify data integrity after shrink/expand."""
    # Verify token 0 unchanged
    assert torch.allclose(swa.k_buffer[layer][0, :, :], test_data["swa_k0"])
    assert torch.allclose(swa.v_buffer[layer][0, :, :], test_data["swa_v0"])
    assert torch.allclose(full.k_buffer[layer][0, :, :], test_data["full_k0"])
    assert torch.allclose(
        full.v_buffer[layer][0, :, :], test_data["full_v0"]
    ), "Full v0 corrupted"

    # Verify new last token writable (full expanded)
    new_full_last = torch.randn(state_shape, dtype=dtype, device=device)
    full.k_buffer[layer][new_full_size - 1, :, :] = new_full_last
    assert torch.allclose(
        full.k_buffer[layer][new_full_size - 1, :, :], new_full_last
    ), "Full last not writable"

    # Verify new last token writable (swa shrunk)
    new_swa_last = torch.randn(state_shape, dtype=dtype, device=device)
    swa.k_buffer[layer][new_swa_size - 1, :, :] = new_swa_last
    assert torch.allclose(
        swa.k_buffer[layer][new_swa_size - 1, :, :], new_swa_last
    ), "SWA last not writable"
