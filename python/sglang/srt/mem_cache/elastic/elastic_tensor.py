import time
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

root = Path(__file__).parent.resolve()
start_time = time.time()
elastic_utils = load(
    name="elastic_utils",
    sources=[f"{root}/elastic_utils.cu"],
    extra_ldflags=["-lcuda"],
    verbose=True,
)
load_time = time.time() - start_time
print(f"Time taken to load elastic_utils: {load_time:.4f} seconds")

PHYSICAL_PAGE_SIZE = None


def _get_physical_page_size() -> int:
    global PHYSICAL_PAGE_SIZE
    if PHYSICAL_PAGE_SIZE is not None:
        return PHYSICAL_PAGE_SIZE
    PHYSICAL_PAGE_SIZE = elastic_utils.get_physical_page_size()
    return PHYSICAL_PAGE_SIZE


def initialize_tms_integration(memory_saver):
    global g_memory_saver, g_orig_pause, g_orig_resume, g_pause_cbs, g_resume_cbs

    if g_memory_saver is not None:
        return

    g_memory_saver = memory_saver
    g_orig_pause = memory_saver.pause
    g_orig_resume = memory_saver.resume

    def run_pause_callbacks():
        for cb in g_pause_cbs:
            cb()

    def run_resume_callbacks():
        for cb in g_resume_cbs:
            cb()

    def pause_hook(*args, **kwargs):
        run_pause_callbacks()
        return g_orig_pause(*args, **kwargs)

    def resume_hook(*args, **kwargs):
        run_resume_callbacks()
        return g_orig_resume(*args, **kwargs)

    g_memory_saver.pause = pause_hook
    g_memory_saver.resume = resume_hook


g_memory_saver = None
g_pause_cbs = set()
g_resume_cbs = set()
g_orig_pause = None
g_orig_resume = None


def register_tms_pause_callback(cb):
    global g_memory_saver, g_pause_cbs
    if g_memory_saver is None:
        return
    g_pause_cbs.add(cb)


def register_tms_resume_callback(cb):
    global g_memory_saver, g_resume_cbs
    if g_memory_saver is None:
        return
    g_resume_cbs.add(cb)


class ElasticTensor:

    def __init__(
        self,
        num: int,
        max_num: int,
        state_shape: tuple[int],
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.state_shape = state_shape
        self.dtype = dtype
        self.device = device

        elastic_utils.initialize_elastic_utils(self.device.index)
        self.physical_page_size = _get_physical_page_size()

        self._num = num
        self._max_num = max_num

        # Calculate initial physical size
        self._current_psize = self._round_up_to_page_size(
            self._num * self._state_elements * self.dtype.itemsize
        )

        # Calculate max virtual size
        max_vsize = self._round_up_to_page_size(
            self._max_num * self._state_elements * self.dtype.itemsize
        )

        # Create elastic tensor
        self._full_tensor = elastic_utils.create_etensor(
            max_vsize, self._current_psize, self.dtype
        )
        self.tensor = self._full_tensor[: self._max_num * self._state_elements].view(
            (self._max_num,) + self.state_shape
        )

        self.pause_num = -1

    def __del__(self):
        elastic_utils.cleanup_etensor(self._full_tensor)

    @property
    def _state_elements(self):
        elements = 1
        for dim in self.state_shape:
            elements *= dim
        return elements

    def _round_up_to_page_size(self, size: int) -> int:
        if size % self.physical_page_size != 0:
            return ((size // self.physical_page_size) + 1) * self.physical_page_size
        return size

    @property
    def shape(self):
        return (self._num,) + self.state_shape

    @property
    def num(self):
        return self._num

    def expand(self, new_num: int):
        if new_num <= self._num:
            raise ValueError(
                f"New num ({new_num}) must be greater than current num ({self._num})"
            )

        # Calculate new physical size
        new_psize = self._round_up_to_page_size(
            new_num * self._state_elements * self.dtype.itemsize
        )

        # Map additional pages if needed
        additional_size = new_psize - self._current_psize
        pages_mapped = 0

        if additional_size > 0:
            elastic_utils.map_physical_page(
                self._full_tensor, self._current_psize, additional_size
            )
            pages_mapped = additional_size // self.physical_page_size

        self._num = new_num
        self._current_psize = new_psize

        return pages_mapped

    def shrink(self, new_num: int):
        if new_num >= self._num:
            raise ValueError(
                f"New num ({new_num}) must be less than current num ({self._num})"
            )

        # Calculate new physical size
        new_psize = self._round_up_to_page_size(
            new_num * self._state_elements * self.dtype.itemsize
        )

        # Unmap pages if needed
        unmap_size = self._current_psize - new_psize
        pages_unmapped = 0

        if unmap_size > 0:
            elastic_utils.unmap_physical_page(self._full_tensor, new_psize, unmap_size)
            pages_unmapped = unmap_size // self.physical_page_size

        self._num = new_num
        self._current_psize = new_psize

        return pages_unmapped


def create_test_tensor(num, state_shape, dtype, device=None):
    """Helper function to create a test tensor with common initialization logic"""
    if device is None:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")

    # Calculate max_num from GPU memory
    total_gpu_memory = torch.cuda.get_device_properties(device).total_memory
    element_size = dtype.itemsize
    state_elements = 1
    for dim in state_shape:
        state_elements *= dim
    max_num = total_gpu_memory // element_size // state_elements

    etensor = ElasticTensor(
        num=num, max_num=max_num, state_shape=state_shape, dtype=dtype, device=device
    )
    return etensor
