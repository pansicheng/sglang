"""
Test ElasticSWATokenToKVPoolAllocator and ElasticSWAKVPool.

cd /sgl-workspace/sglang/python/sglang/srt/mem_cache/elastic
LD_PRELOAD=/usr/local/lib/python3.12/dist-packages/torch_memory_saver_hook_mode_preload.abi3.so \
SGLANG_ELASTIC_MEM_POOL=true \
SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
pytest test_elastic_swa_allocator.py -s
"""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.mem_cache.elastic.elasticmem_orchestrator import (
    USE_ELASTICMEM,
    ElasticMempoolOrchestrator,
)


@pytest.fixture
def skip_if_no_cuda_elasticmem():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if not USE_ELASTICMEM:
        pytest.skip(f"Elastic memory not enabled: {USE_ELASTICMEM=}")


def _make_pool_and_allocator(
    size=1024, size_swa=512, head_num=4, head_dim=32, device="cuda"
):
    from sglang.srt.mem_cache.elastic.elastic_allocator import (
        ElasticSWATokenToKVPoolAllocator,
    )
    from sglang.srt.mem_cache.elastic.elastic_memory_pool import ElasticSWAKVPool
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    swa_layer_ids = [i for i in range(8) if i % 2 == 0]
    full_layer_ids = [i for i in range(8) if i % 2 == 1]

    pool = ElasticSWAKVPool(
        size=size,
        size_swa=size_swa,
        page_size=1,
        dtype=torch.bfloat16,
        head_num=head_num,
        head_dim=head_dim,
        swa_attention_layer_ids=swa_layer_ids,
        full_attention_layer_ids=full_layer_ids,
        enable_kvcache_transpose=False,
        device=device,
        token_to_kv_pool_class=MHATokenToKVPool,
        enable_memory_saver=False,
    )
    orch = ElasticMempoolOrchestrator()
    allocator = ElasticSWATokenToKVPoolAllocator(
        size,
        size_swa,
        page_size=1,
        dtype=torch.bfloat16,
        device=device,
        kvcache=pool,
        need_sort=False,
        emem_orch=orch,
    )
    return pool, allocator, orch


def test_elastic_swa_init_and_properties(skip_if_no_cuda_elasticmem):
    """Init, orchestrator registration, flags, mapping oversubscription, sentinel."""
    from sglang.srt.mem_cache.elastic.elastic_allocator import (
        ElasticTokenToKVPoolAllocator,
    )

    size, size_swa = 1024, 512
    pool, allocator, orch = _make_pool_and_allocator(size=size, size_swa=size_swa)

    # ElasticSWAKVPool: shrink/expand/cu_page_to_token must not be called
    for fn, args in [
        (pool.shrink, (256,)),
        (pool.expand, (2048,)),
        (pool.cu_page_to_token, (1,)),
    ]:
        with pytest.raises(RuntimeError):
            fn(*args)

    # Wrapper allocator is not an orchestrator candidate
    assert allocator.can_unmap() is False
    assert allocator.can_map() is False

    # All three allocators registered
    assert allocator in orch.allocators
    assert allocator.full_attn_allocator in orch.allocators
    assert allocator.swa_attn_allocator in orch.allocators
    assert isinstance(allocator.full_attn_allocator, ElasticTokenToKVPoolAllocator)
    assert isinstance(allocator.swa_attn_allocator, ElasticTokenToKVPoolAllocator)

    # Mapping oversubscribed with -1 sentinel
    assert allocator.full_to_swa_index_mapping.numel() > size + 1
    assert allocator.full_to_swa_index_mapping[-1].item() == -1


def test_elastic_swa_shrink_expand(skip_if_no_cuda_elasticmem):
    """Real GPU shrink swa + expand full, update_size, data integrity."""
    size, size_swa, head_num, head_dim = 1024, 512, 4, 32
    pool, allocator, _ = _make_pool_and_allocator(size=size, size_swa=size_swa)
    allocator.scheduler = SimpleNamespace(
        swa_tokens_per_layer=size_swa, full_tokens_per_layer=size
    )

    swa, full = pool.swa_kv_pool, pool.full_kv_pool

    # Write test data at token 0
    swa_k0 = torch.randn(head_num, head_dim, dtype=torch.bfloat16, device="cuda")
    full_k0 = torch.randn(head_num, head_dim, dtype=torch.bfloat16, device="cuda")
    swa.k_buffer[0][0, :, :] = swa_k0
    full.k_buffer[0][0, :, :] = full_k0

    # Shrink swa by 128, expand full by 128
    new_swa, new_full = size_swa - 128, size + 128
    swa.shrink(new_swa)
    full.expand(new_full)

    # Propagate via update_size
    allocator.full_attn_allocator.size = new_full
    allocator.swa_attn_allocator.size = new_swa
    allocator.update_size()

    # All sizes consistent
    assert (allocator._size_full, allocator._size_swa) == (new_full, new_swa)
    assert (pool.size, pool.size_swa) == (new_full, new_swa)
    assert allocator.scheduler.full_tokens_per_layer == new_full
    assert allocator.scheduler.swa_tokens_per_layer == new_swa

    # Sentinel preserved
    assert allocator.full_to_swa_index_mapping[-1].item() == -1

    # Data at token 0 survived
    assert torch.allclose(swa.k_buffer[0][0, :, :], swa_k0)
    assert torch.allclose(full.k_buffer[0][0, :, :], full_k0)

    # New last tokens writable
    for buf, last_idx in [
        (full.k_buffer[0], new_full - 1),
        (swa.k_buffer[0], new_swa - 1),
    ]:
        v = torch.randn(head_num, head_dim, dtype=torch.bfloat16, device="cuda")
        buf[last_idx, :, :] = v
        assert torch.allclose(buf[last_idx, :, :], v)
