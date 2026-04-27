from __future__ import annotations

"""
Copyright 2025 SGLang Team
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

import logging
import time
from typing import override

import torch

from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.elastic.elasticmem_orchestrator import (
    CAN_MAP_THRESHOLD,
    CAN_UNMAP_THRESHOLD,
    CU_PAGE_SIZE,
    ENABLE_SANITY_CHECK,
    ElasticAllocator,
)
from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator

logger = logging.getLogger(__name__)


def get_tail_consecutive_start(unused_pages: torch.Tensor) -> int:
    """Find the start index of consecutive True values at the tail."""
    if unused_pages.numel() == 0:
        return 0
    # Find the last False index using vectorized operations
    false_mask = ~unused_pages
    if not false_mask.any():
        return 0  # All True, consecutive from start
    last_false_idx = false_mask.nonzero()[-1].item()
    return last_false_idx + 1


class ElasticTokenToKVPoolAllocator(TokenToKVPoolAllocator, ElasticAllocator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cu_page_token_num = CU_PAGE_SIZE // self._kvcache.state_memsize
        logger.debug(
            f"ElasticTokenToKVPoolAllocator initialized, " f"{self.cu_page_token_num=}"
        )

    @override
    def clear(self):
        super().clear()
        self.unused_pages = torch.ones(
            (self.size + 1,), dtype=torch.bool, device=self.device
        )
        self.candidate_size = self.size
        self.candidate_unmap_pages = torch.empty(
            (0,), dtype=torch.int64, device=self.device
        )

    def available_size(self):
        return (
            len(self.free_pages)
            + len(self.release_pages)
            + len(self.candidate_unmap_pages)
        )

    def _check_unused_pages(self):
        if not ENABLE_SANITY_CHECK:
            return
        self_unused_pages = self.unused_pages[1:]
        self_unused_pages = self_unused_pages[self_unused_pages == True]
        assert (self.available_size() + self.evictable_size()) == len(
            self_unused_pages
        ), (
            f"{len(self.free_pages)=}, {len(self.release_pages)=}, {len(self.candidate_unmap_pages)=}, "
            f"{self.available_size()=} + {self.evictable_size()=} = {self.available_size() + self.evictable_size()}; "
            f"{len(self_unused_pages)=}, {len(self.unused_pages)=}"
        )

    def _unmap_candidate_pre_alloc(self, need_size: int):
        if self.candidate_size == self.size:
            return

        available_size = len(self.free_pages) + len(self.release_pages)
        if need_size <= available_size:
            return

        self._evict_tail()
        evict_size = min(self.evictable_size(), need_size)
        self.evict(evict_size)

        available_size = len(self.free_pages) + len(self.release_pages)
        if need_size <= available_size:
            return

        extra_need_size = need_size - available_size
        logger.debug(
            f"{extra_need_size=} {available_size=} {self.evictable_size()=} {len(self.candidate_unmap_pages)=}"
        )

        if self.need_sort:
            self.release_pages = torch.cat(
                (
                    self.release_pages,
                    self.candidate_unmap_pages[:extra_need_size],
                )
            )
        else:
            self.free_pages = torch.cat(
                (
                    self.free_pages,
                    self.candidate_unmap_pages[:extra_need_size],
                )
            )

        self.candidate_unmap_pages = self.candidate_unmap_pages[extra_need_size:]

    @override
    def alloc(self, need_size: int):
        self._check_unused_pages()
        self._unmap_candidate_pre_alloc(need_size)
        self._check_unused_pages()

        select_index = super().alloc(need_size)
        self.unused_pages[select_index] = False

        self._check_unused_pages()

        return select_index

    def _unmap_candidate_post_free(self, free_index: torch.Tensor):
        if self.candidate_size == self.size:
            return

        if not (free_index > self.candidate_size).any():
            return

        prev_count = len(self.candidate_unmap_pages)

        if self.need_sort:
            unmap_candidate_mask = self.release_pages > self.candidate_size
            self.candidate_unmap_pages = torch.cat(
                (self.candidate_unmap_pages, self.release_pages[unmap_candidate_mask])
            )
            self.release_pages = self.release_pages[~unmap_candidate_mask]
        else:
            unmap_candidate_mask = self.free_pages > self.candidate_size
            self.candidate_unmap_pages = torch.cat(
                (self.candidate_unmap_pages, self.free_pages[unmap_candidate_mask])
            )
            self.free_pages = self.free_pages[~unmap_candidate_mask]

        cur_count = len(self.candidate_unmap_pages)
        target = self.size - self.candidate_size
        if cur_count != prev_count:
            logger.debug(
                f"post_free: candidate_unmap_pages {prev_count}->{cur_count}/{target}"
            )

    @override
    def free(self, free_index: torch.Tensor):
        super().free(free_index)

        if not self.is_not_in_free_group:
            return

        # Free first, then delete_leaf updates evictable_size - may temporarily exceed len(self_unused_pages), skip self._check_unused_pages().
        self.unused_pages[free_index] = True

        self._unmap_candidate_post_free(free_index)

    @override
    def can_be_candidate(self) -> bool:
        return True

    def _evict_tail(self):
        if self.candidate_size == self.size:
            return

        start_time = time.perf_counter()
        evict_indices = self.unused_pages.nonzero(as_tuple=True)[0]
        evict_indices = evict_indices[evict_indices > self.candidate_size]

        if self.candidate_unmap_pages.numel() == evict_indices.numel():
            return

        if self.candidate_unmap_pages.numel() > 0:
            evict_indices = evict_indices[
                ~torch.isin(evict_indices, self.candidate_unmap_pages)
            ]

        if len(evict_indices) == 0:
            return

        logger.info(
            f"_evict_tail: to_evict={len(evict_indices)}, "
            f"evictable={self.evictable_size()}, "
            f"candidate_unmap_pages={len(self.candidate_unmap_pages)}, "
            f"{self.candidate_size=}, {self.size=}"
        )

        while (self.evictable_size() >= len(evict_indices)) and (
            len(evict_indices) > 0
        ):
            self.evict(len(evict_indices))
            evict_indices = evict_indices[
                ~torch.isin(evict_indices, self.candidate_unmap_pages)
            ]

        logger.info(
            f"_evict_tail done: took {(time.perf_counter() - start_time) * 1000:.1f} ms, "
            f"candidate_unmap_pages={len(self.candidate_unmap_pages)}, "
            f"expected={self.size - self.candidate_size}"
        )

    @override
    def mark_unmap_candidate(self, is_candidate: bool) -> ElasticAllocator:
        # is_candidate == False
        if not is_candidate:
            assert self.candidate_size <= self.size
            if self.candidate_size < self.size:
                self.candidate_size = self.size
                if self.need_sort:
                    self.release_pages = torch.cat(
                        (self.release_pages, self.candidate_unmap_pages)
                    )
                else:
                    self.free_pages = torch.cat(
                        (self.free_pages, self.candidate_unmap_pages)
                    )

                self.candidate_unmap_pages = torch.empty(
                    (0,), dtype=torch.int64, device=self.device
                )
            assert len(self.candidate_unmap_pages) == 0
            logger.info(
                f"mark_unmap_candidate: {is_candidate=}, {self.size=}, {self.candidate_size=}"
            )
            return None

        # is_candidate == True
        assert (
            self.candidate_size == self.size and len(self.candidate_unmap_pages) == 0
        ), f"{self.candidate_size=}, {self.size=}, {len(self.candidate_unmap_pages)=}"
        token_usage = self.token_usage()
        new_size = int((token_usage + 1) / 2 * self.size)

        if self.size - new_size < 2 * self.cu_page_token_num:
            logger.info(
                f"mark_unmap_candidate: {is_candidate=}, {self.size=}, {self.candidate_size=}"
            )
            return None

        self.candidate_size = new_size

        unmap_candidate_mask = self.release_pages > self.candidate_size
        self.candidate_unmap_pages = torch.cat(
            (self.candidate_unmap_pages, self.release_pages[unmap_candidate_mask])
        )
        self.release_pages = self.release_pages[~unmap_candidate_mask]
        unmap_candidate_mask = self.free_pages > self.candidate_size
        self.candidate_unmap_pages = torch.cat(
            (self.candidate_unmap_pages, self.free_pages[unmap_candidate_mask])
        )
        self.free_pages = self.free_pages[~unmap_candidate_mask]

        logger.info(
            f"mark_unmap_candidate: set candidate, {self.size=}, {self.candidate_size=}, "
            f"candidate_unmap_pages={len(self.candidate_unmap_pages)}, "
            f"target_unmap={self.size - self.candidate_size}"
        )
        return self

    @override
    def can_unmap(self) -> bool:
        return self.token_usage() < CAN_UNMAP_THRESHOLD

    @override
    def can_do_unmap(self) -> bool:
        # Proactively evict cached entries from the tail region
        # to accelerate page collection (instead of waiting until reduce()).
        self._evict_tail()

        tail_consecutive_start = get_tail_consecutive_start(self.unused_pages)

        # Allow partial shrink: if the full candidate_size isn't reachable yet,
        # check if the consecutive free tail is large enough to be worth shrinking
        # (at least 2 * cu_page_token_num).
        if tail_consecutive_start <= self.candidate_size + 1:
            return True

        # Partial shrink: ceiling-align to cu_page boundary so we never
        # unmap pages below tail_consecutive_start (which may be in use).
        partial_size = (
            (tail_consecutive_start - 1 + self.cu_page_token_num - 1)
            // self.cu_page_token_num
        ) * self.cu_page_token_num
        freeable = self.size - partial_size
        if freeable >= 2 * self.cu_page_token_num and partial_size < self.size:
            # Adjust candidate_size upward to what's actually achievable now
            old_candidate = self.candidate_size
            self.candidate_size = partial_size
            # Return pages between old candidate and new candidate to free pool
            if old_candidate < partial_size:
                reclaim_mask = (self.candidate_unmap_pages <= partial_size) & (
                    self.candidate_unmap_pages > old_candidate
                )
                if reclaim_mask.any():
                    if self.need_sort:
                        self.release_pages = torch.cat(
                            (
                                self.release_pages,
                                self.candidate_unmap_pages[reclaim_mask],
                            )
                        )
                    else:
                        self.free_pages = torch.cat(
                            (self.free_pages, self.candidate_unmap_pages[reclaim_mask])
                        )
                    self.candidate_unmap_pages = self.candidate_unmap_pages[
                        ~reclaim_mask
                    ]
            logger.info(
                f"can_do_unmap: partial shrink, adjusted candidate_size "
                f"{old_candidate}->{partial_size}, freeable={freeable}"
            )
            return True

        logger.debug(
            f"can_do_unmap=False: {tail_consecutive_start=}, {self.candidate_size=}, "
            f"{self.size=}, gap={tail_consecutive_start - self.candidate_size - 1}"
        )
        return False

    @override
    def can_map(self) -> bool:
        return self.token_usage() > CAN_MAP_THRESHOLD

    @override
    def reduce(self) -> int:
        start_time = time.perf_counter()

        self._evict_tail()

        # With partial shrink, candidate_unmap_pages may not cover the full
        # original target.  Recalculate the expected count.
        expected = self.size - self.candidate_size
        actual = len(self.candidate_unmap_pages)
        if actual != expected:
            logger.info(
                f"reduce: adjusting for partial shrink, "
                f"candidate_unmap_pages={actual}, expected={expected}"
            )

        new_size = self.candidate_size
        self.candidate_unmap_pages = torch.empty(
            (0,), dtype=torch.int64, device=self.device
        )
        self.unused_pages = self.unused_pages[: new_size + 1]
        unmap_num, cur_size = self._kvcache.shrink(self.candidate_size)
        logger.debug(f"{(self.size, self.candidate_size, unmap_num, cur_size)=}")
        self.size = cur_size

        logger.info(f"reduce took {(time.perf_counter() - start_time) * 1000:.1f} ms")
        return unmap_num

    @override
    def expand(self, expand_size: int) -> int:
        start_time = time.perf_counter()

        assert self.candidate_size == self.size and len(self.candidate_unmap_pages) == 0

        if expand_size <= 0:
            return 0

        assert expand_size % self.page_size == 0

        new_size = self.size + expand_size
        map_num, cur_size = self._kvcache.expand(new_size)
        logger.debug(f"{expand_size=}, {(map_num, cur_size)=}")

        self.free_pages = torch.cat(
            (
                self.free_pages,
                torch.arange(
                    self.size + 1,
                    cur_size + 1,
                    dtype=torch.int64,
                    device=self.device,
                ),
            )
        )
        self.unused_pages = torch.cat(
            (
                self.unused_pages,
                torch.ones(
                    (cur_size - self.size,), dtype=torch.bool, device=self.device
                ),
            )
        )

        self.size = cur_size
        self.candidate_size = self.size

        logger.info(f"expand took {(time.perf_counter() - start_time) * 1000} ms")
        return map_num

    @override
    def cu_page_to_token(self, cu_page_num: int) -> int:
        return self._kvcache.cu_page_to_token(cu_page_num)

    @override
    def register_evict_func(self, func_evictable_size, func_evict) -> None:
        self.func_evictable_size = func_evictable_size
        self.func_evict = func_evict

    @override
    def token_usage(self) -> float:
        num_used = self.size - (self.available_size() + self.evictable_size())
        return num_used / self.size

    @override
    def evictable_size(self) -> int:
        return self.func_evictable_size()

    @override
    def evict(self, evictable_size: int) -> None:
        self.func_evict(evictable_size)

    @override
    def update_size(self):
        # size is updated in enable()/disable()
        assert self.size == self._kvcache.size


class ElasticSWATokenToKVPoolAllocator(SWATokenToKVPoolAllocator, ElasticAllocator):
    def __init__(self, *args, emem_orch, **kwargs):
        self.emem_orch = emem_orch

        super().__init__(*args, **kwargs)
        logger.debug("ElasticSWATokenToKVPoolAllocator initialized")

        self.emem_orch.register_allocator(self)
        self.emem_orch.register_allocator(self.full_attn_allocator)
        self.emem_orch.register_allocator(self.swa_attn_allocator)
        logger.debug(f"ElasticSWATokenToKVPoolAllocator register_allocator")

    def _create_allocator(self):
        self.full_attn_allocator = ElasticTokenToKVPoolAllocator(
            self._size_full,
            self.dtype,
            self.device,
            self._kvcache.full_kv_pool,
            self.need_sort,
        )
        self.swa_attn_allocator = ElasticTokenToKVPoolAllocator(
            self._size_swa,
            self.dtype,
            self.device,
            self._kvcache.swa_kv_pool,
            self.need_sort,
        )
        # oversubscribe as full pool may expand
        total_tokens = (
            self._size_full * self._kvcache.full_kv_pool.layer_num
            + self._size_swa * self._kvcache.swa_kv_pool.layer_num
        )
        oversubscribe_tokens = (
            max(
                total_tokens // self._kvcache.full_kv_pool.layer_num,
                total_tokens // self._kvcache.swa_kv_pool.layer_num,
            )
            * 2
        )
        self.full_to_swa_index_mapping = torch.cat(
            [
                torch.zeros(
                    oversubscribe_tokens,
                    dtype=torch.int64,
                    device=self.device,
                ),
                torch.tensor([-1], dtype=torch.int64, device=self.device),
            ]
        )
        logger.debug(
            f"{(self.full_to_swa_index_mapping.numel() * self.full_to_swa_index_mapping.dtype.itemsize // (1<<20))=}"
        )

    @override
    def register_scheduler(self, scheduler) -> None:
        self.scheduler = scheduler
        self.full_attn_allocator.register_evict_func(
            func_evictable_size=self.scheduler.tree_cache.full_evictable_size,
            func_evict=lambda evictable_size: self.scheduler.tree_cache.evict(
                EvictParams(num_tokens=evictable_size, swa_num_tokens=0)
            ),
        )
        self.swa_attn_allocator.register_evict_func(
            func_evictable_size=self.scheduler.tree_cache.swa_evictable_size,
            func_evict=lambda evictable_size: self.scheduler.tree_cache.evict(
                EvictParams(num_tokens=0, swa_num_tokens=evictable_size)
            ),
        )

    @override
    def can_unmap(self) -> bool:
        return False

    @override
    def can_map(self) -> bool:
        return False

    @override
    def token_usage(self) -> float:
        return 1

    @override
    def update_size(self):
        self._kvcache.size = self.full_attn_allocator.size
        self._kvcache.size_swa = self.swa_attn_allocator.size
        self._size_full = self.full_attn_allocator.size
        self._size_swa = self.swa_attn_allocator.size
        self.scheduler.swa_tokens_per_layer = self._size_swa
        self.scheduler.full_tokens_per_layer = self._size_full
        self.full_to_swa_index_mapping[self._size_full + 1 : -1] = 0
        logger.info(
            "ElasticSWATokenToKVPoolAllocator update_size: "
            f"{(self._size_swa, self._size_full)=}, "
            f"{(self.scheduler.swa_tokens_per_layer, self.scheduler.full_tokens_per_layer)=}"
        )
