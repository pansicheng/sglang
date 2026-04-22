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

import logging
import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.cpp_extension import load

logger = logging.getLogger(__name__)

# Load CUDA extension
_root = Path(__file__).parent.resolve()
_start_time = time.time()
elastic_utils = load(
    name="elastic_utils",
    sources=[f"{_root}/elastic_utils.cu"],
    extra_ldflags=["-lcuda"],
    verbose=True,
)
logger.info(
    f"Time taken to load elastic_utils: {time.time() - _start_time:.4f} seconds"
)

# Cached physical page size
_physical_page_size: Optional[int] = None

# TMS integration state
_memory_saver = None
_pause_callbacks = set()
_resume_callbacks = set()
_orig_pause = None
_orig_resume = None


def _get_physical_page_size() -> int:
    global _physical_page_size
    if _physical_page_size is not None:
        return _physical_page_size
    _physical_page_size = elastic_utils.get_physical_page_size()
    return _physical_page_size


def initialize_tms_integration(memory_saver):
    """Initialize torch_memory_saver integration with pause/resume callbacks."""
    global _memory_saver, _orig_pause, _orig_resume, _pause_callbacks, _resume_callbacks

    if _memory_saver is not None:
        return

    _memory_saver = memory_saver
    _orig_pause = memory_saver.pause
    _orig_resume = memory_saver.resume

    def _run_pause_callbacks():
        for cb in _pause_callbacks:
            cb()

    def _run_resume_callbacks():
        for cb in _resume_callbacks:
            cb()

    def _pause_hook(*args, **kwargs):
        _run_pause_callbacks()
        return _orig_pause(*args, **kwargs)

    def _resume_hook(*args, **kwargs):
        _run_resume_callbacks()
        return _orig_resume(*args, **kwargs)

    _memory_saver.pause = _pause_hook
    _memory_saver.resume = _resume_hook


def register_tms_pause_callback(callback):
    """Register a callback to be called before memory saver pauses."""
    if _memory_saver is not None:
        _pause_callbacks.add(callback)


def register_tms_resume_callback(callback):
    """Register a callback to be called before memory saver resumes."""
    if _memory_saver is not None:
        _resume_callbacks.add(callback)


class ElasticTensor:
    """Tensor with elastic memory mapping supporting dynamic expand/shrink."""

    def __init__(
        self,
        num: int,
        max_num: int,
        state_shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.state_shape = state_shape
        self.dtype = dtype
        self.device = device

        elastic_utils.initialize_elastic_utils(self.device.index)
        self._page_size = _get_physical_page_size()

        self._num = num
        self._max_num = max_num

        # Calculate state elements and memory size
        self._state_elements = 1
        for dim in self.state_shape:
            self._state_elements *= dim
        self._state_memsize = self._state_elements * self.dtype.itemsize

        # Calculate initial and max physical sizes
        self._current_psize = self._round_up_to_page_size(num * self._state_memsize)
        max_vsize = self._round_up_to_page_size(max_num * self._state_memsize)

        # Create elastic tensor
        self._full_tensor = elastic_utils.create_etensor(
            max_vsize, self._current_psize, self.dtype
        )
        self.tensor = self._full_tensor[: max_num * self._state_elements].view(
            (max_num,) + self.state_shape
        )

        # For TMS integration: saves num before pause, used to restore on resume
        # -1 means not in pause state
        self.tms_saved_num = -1

    def __del__(self):
        try:
            elastic_utils.cleanup_etensor(self._full_tensor)
        except Exception:
            pass

    @property
    def state_memsize(self) -> int:
        return self._state_memsize

    @property
    def shape(self) -> tuple[int, ...]:
        return (self._num,) + self.state_shape

    @property
    def num(self) -> int:
        return self._num

    @property
    def psize(self) -> int:
        return self._current_psize

    def _round_up_to_page_size(self, size: int) -> int:
        remainder = size % self._page_size
        return size + (self._page_size - remainder) if remainder else size

    def expand(self, new_num: int) -> int:
        """Expand tensor to new_num elements, returning pages mapped."""
        if new_num <= self._num:
            raise ValueError(f"new_num ({new_num}) must be > current num ({self._num})")

        new_psize = self._round_up_to_page_size(new_num * self._state_memsize)
        additional_size = new_psize - self._current_psize
        pages_mapped = 0

        if additional_size > 0:
            elastic_utils.map_physical_page(
                self._full_tensor, self._current_psize, additional_size
            )
            pages_mapped = additional_size // self._page_size

        self._num = new_num
        self._current_psize = new_psize
        return pages_mapped

    def shrink(self, new_num: int) -> int:
        """Shrink tensor to new_num elements, returning pages unmapped."""
        if new_num >= self._num:
            raise ValueError(f"new_num ({new_num}) must be < current num ({self._num})")

        new_psize = self._round_up_to_page_size(new_num * self._state_memsize)
        unmap_size = self._current_psize - new_psize
        pages_unmapped = 0

        if unmap_size > 0:
            elastic_utils.unmap_physical_page(self._full_tensor, new_psize, unmap_size)
            pages_unmapped = unmap_size // self._page_size

        self._num = new_num
        self._current_psize = new_psize
        return pages_unmapped


def create_test_tensor(
    num: int,
    state_shape: tuple[int, ...],
    dtype: torch.dtype,
    device: Optional[torch.device] = None,
) -> ElasticTensor:
    """Create ElasticTensor for testing with max_num derived from GPU memory."""
    if device is None:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")

    total_memory = torch.cuda.get_device_properties(device).total_memory
    state_elements = 1
    for dim in state_shape:
        state_elements *= dim
    max_num = total_memory // dtype.itemsize // state_elements

    return ElasticTensor(num, max_num, state_shape, dtype, device)
