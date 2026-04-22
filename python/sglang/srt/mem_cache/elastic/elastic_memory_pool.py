"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import logging
from typing import Tuple, override

import torch

from sglang.srt.mem_cache.elastic.elasticmem_orchestrator import (
    CU_PAGE_SIZE,
    USE_ELASTICMEM,
    ElasticMempool,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

if USE_ELASTICMEM:
    from sglang.srt.mem_cache.elastic.elastic_tensor import ElasticTensor


logger = logging.getLogger(__name__)


class ElasticMHATokenToKVPool(MHATokenToKVPool, ElasticMempool):
    """Elastic memory pool for MHA token-to-KV mapping."""

    def __init__(self, *args, **kwargs):
        assert (
            USE_ELASTICMEM
        ), "ElasticMHATokenToKVPool requires SGLANG_ELASTIC_MEM_POOL=true"
        super().__init__(*args, **kwargs)
        logger.info("ElasticMHATokenToKVPool initialized")

    def _create_buffers(self):
        self.create_elastic_buffers()

    @override
    def create_elastic_buffers(self):
        free_memory, _ = torch.cuda.mem_get_info()
        free_memory_per_layer = free_memory // (2 * self.layer_num)

        state_shape = (self.head_num, self.head_dim)
        state_elements = self.head_num * self.head_dim
        dtype = self.store_dtype

        max_num = free_memory_per_layer // (dtype.itemsize * state_elements)
        max_num = (max_num + self.page_size - 1) // self.page_size * self.page_size

        # cur_num includes page_size offset for allocator metadata
        self.cur_num = self.size + self.page_size
        self.max_num = max_num

        assert self.cur_num % self.page_size == 0
        assert self.max_num % self.page_size == 0

        device = torch.device(torch.cuda.current_device())
        self.ek_buffer = [
            ElasticTensor(self.cur_num, max_num, state_shape, dtype, device)
            for _ in range(self.layer_num)
        ]
        self.ev_buffer = [
            ElasticTensor(self.cur_num, max_num, state_shape, dtype, device)
            for _ in range(self.layer_num)
        ]
        self.k_buffer = [et.tensor for et in self.ek_buffer]
        self.v_buffer = [et.tensor for et in self.ev_buffer]
        self.state_memsize = self.ek_buffer[0].state_memsize

        memory_used = torch.cuda.mem_get_info()[0] - free_memory
        logger.debug(
            f"create_elastic_buffers: cur_num={self.cur_num}, memory_used={memory_used}"
        )

    @override
    def get_kv_size_bytes(self):
        k_size = sum(et.psize for et in self.ek_buffer)
        v_size = sum(et.psize for et in self.ev_buffer)
        return k_size, v_size

    @override
    def shrink(self, new_size: int) -> Tuple[int, int]:
        new_num = new_size + self.page_size
        total_pages = 0
        for ek, ev in zip(self.ek_buffer, self.ev_buffer):
            total_pages += ek.shrink(new_num) + ev.shrink(new_num)
        self.size = new_size
        self.cur_num = new_num
        return total_pages, self.size

    @override
    def expand(self, new_size: int) -> Tuple[int, int]:
        new_num = new_size + self.page_size
        total_pages = 0
        for ek, ev in zip(self.ek_buffer, self.ev_buffer):
            total_pages += ek.expand(new_num) + ev.expand(new_num)
        self.size = new_size
        self.cur_num = new_num
        return total_pages, self.size

    @override
    def cu_page_to_token(self, cu_page_num: int) -> int:
        pages_per_layer = cu_page_num // (2 * self.layer_num)
        mem_per_layer = pages_per_layer * CU_PAGE_SIZE
        tokens = mem_per_layer // self.state_memsize
        return tokens // self.page_size * self.page_size


class ElasticSWAKVPool(SWAKVPool, ElasticMempool):
    """Elastic memory pool with separate SWA and full attention KV pools."""

    def __init__(self, *args, **kwargs):
        assert USE_ELASTICMEM, "ElasticSWAKVPool requires SGLANG_ELASTIC_MEM_POOL=true"
        super().__init__(*args, **kwargs)
        logger.info("ElasticSWAKVPool initialized")

    def _create_buffers(self, token_to_kv_pool_class, **kwargs):
        self.create_elastic_buffers(token_to_kv_pool_class, **kwargs)

    @override
    def create_elastic_buffers(self, token_to_kv_pool_class, **kwargs):
        assert token_to_kv_pool_class == MHATokenToKVPool
        pool_class = ElasticMHATokenToKVPool

        self.swa_kv_pool = pool_class(
            size=self.size_swa,
            dtype=self.dtype,
            layer_num=self.swa_layer_nums,
            **kwargs,
        )
        kwargs.pop("swa_head_num", None)
        kwargs.pop("swa_head_dim", None)
        kwargs.pop("swa_v_head_dim", None)
        self.full_kv_pool = pool_class(
            size=self.size, dtype=self.dtype, layer_num=self.full_layer_nums, **kwargs
        )

    @override
    def shrink(self, new_size: int) -> Tuple[int, int]:
        raise RuntimeError("ElasticSWAKVPool.shrink() should not be called directly.")

    @override
    def expand(self, new_size: int) -> Tuple[int, int]:
        raise RuntimeError("ElasticSWAKVPool.expand() should not be called directly.")

    @override
    def cu_page_to_token(self, cu_page_num: int) -> int:
        raise RuntimeError(
            "ElasticSWAKVPool.cu_page_to_token() should not be called directly."
        )
