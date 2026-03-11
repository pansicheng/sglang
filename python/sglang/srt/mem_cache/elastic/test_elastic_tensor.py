# LD_PRELOAD=/usr/local/lib/python3.12/dist-packages/torch_memory_saver_hook_mode_preload.abi3.so \
# pytest test_elastic_tensor.py -s

import pytest
import torch
from elastic_tensor import (
    create_test_tensor,
    elastic_utils,
)


@pytest.fixture
def cuda_device():
    """Fixture to provide a CUDA device for testing."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return torch.device(f"cuda:{torch.cuda.current_device()}")


def test_elastic_tensor_initialization(cuda_device):
    """Test ElasticTensor initialization with correct properties."""
    num = 8
    state_shape = (1024, 512)
    dtype = torch.bfloat16

    etensor = create_test_tensor(num, state_shape, dtype, cuda_device)

    # Verify initial state
    assert etensor.num == num
    assert etensor.shape == (num,) + state_shape
    assert etensor.tensor.shape == (etensor._max_num,) + state_shape
    assert etensor.tensor.dtype == dtype
    assert etensor.tensor.device.type == "cuda"

    del etensor
    elastic_utils.shutdown_elastic_utils()


def test_elastic_tensor_expansion_and_shrinkage(cuda_device):
    """Test ElasticTensor expansion and shrinkage operations."""
    num = 8
    state_shape = (1024, 512)
    dtype = torch.bfloat16

    etensor = create_test_tensor(num, state_shape, dtype, cuda_device)

    # Assign values to mapped portion
    initial_values = torch.arange(
        num * torch.prod(torch.tensor(state_shape)).item(),
        dtype=dtype,
        device=cuda_device,
    ).view((num,) + state_shape)
    etensor.tensor[:num].copy_(initial_values)

    original_mapped_values = etensor.tensor[:num].clone()

    # Expansion test
    new_num = num * 2
    pages_mapped = etensor.expand(new_num)
    assert etensor.num == new_num

    # Verify values preserved after expansion
    assert torch.equal(etensor.tensor[:num], original_mapped_values)

    # Test expanded portion accessibility
    new_values = (
        torch.ones((new_num - num,) + state_shape, dtype=dtype, device=cuda_device)
        * 100
    )
    etensor.tensor[num:new_num] = new_values
    assert torch.equal(etensor.tensor[:num], original_mapped_values)
    assert torch.equal(etensor.tensor[num:new_num], new_values)

    # Second expansion
    new_num2 = new_num + 5
    pages_mapped2 = etensor.expand(new_num2)
    assert etensor.num == new_num2

    # Shrinking test
    shrink_num = new_num
    pages_unmapped = etensor.shrink(shrink_num)
    assert etensor.num == shrink_num

    # Verify values after shrink
    assert torch.equal(etensor.tensor[:num], original_mapped_values)

    # Final shrink
    shrink_num2 = shrink_num // 2
    pages_unmapped2 = etensor.shrink(shrink_num2)
    assert etensor.num == shrink_num2

    # Cleanup
    del etensor
    elastic_utils.shutdown_elastic_utils()


def test_unmap_accessibility(cuda_device):
    """Test accessing out-of-bounds indices raises appropriate error."""
    num = 4
    state_shape = (1024, 512)
    dtype = torch.float32

    etensor = create_test_tensor(num, state_shape, dtype, cuda_device)

    # Access index num (which should be out of bounds since valid range is [0, num))
    with pytest.raises(RuntimeError):
        _ = etensor.tensor[num : num + 1].clone()

    # Cleanup
    del etensor
    elastic_utils.shutdown_elastic_utils()


def test_tms(cuda_device):
    """Test ElasticTensor integration with torch_memory_saver."""
    num = 512
    state_shape = (1024, 1024)
    dtype = torch.bfloat16

    import weakref

    import torch_memory_saver
    from elastic_tensor import (
        initialize_tms_integration,
        register_tms_pause_callback,
        register_tms_resume_callback,
    )

    _memory_saver = torch_memory_saver.torch_memory_saver

    GPU_MEMORY_TYPE_KV_CACHE = "kv_cache"
    pause_flag = False
    resume_flag = False
    cuda_free_memory_before_region = torch.cuda.mem_get_info()[0]

    etensor_num = 2
    etensor_list = [None] * etensor_num
    et_ref_list = [None] * etensor_num
    pause_flags = [False] * etensor_num
    resume_flags = [False] * etensor_num
    for i in range(etensor_num):
        with _memory_saver.region(GPU_MEMORY_TYPE_KV_CACHE):
            initialize_tms_integration(_memory_saver)
            etensor_list[i] = create_test_tensor(num, state_shape, dtype, cuda_device)
            et_ref_list[i] = weakref.ref(etensor_list[i])

            def make_on_pause(index):
                def on_pause():
                    t = et_ref_list[index]()
                    if t is None:
                        return
                    assert t.num >= 0 and t.pause_num == -1
                    t.pause_num = t.num
                    t.shrink(0)
                    assert t.num == 0 and t.pause_num != -1
                    pause_flags[index] = True

                return on_pause

            def make_on_resume(index):
                def on_resume():
                    t = et_ref_list[index]()
                    if t is None:
                        return
                    assert t.num == 0 and t.pause_num != -1
                    t.expand(t.pause_num)
                    t.pause_num = -1
                    assert t.num >= 0 and t.pause_num == -1
                    resume_flags[index] = True

                return on_resume

            register_tms_pause_callback(make_on_pause(i))
            register_tms_resume_callback(make_on_resume(i))

    cuda_free_memory_after_region = torch.cuda.mem_get_info()[0]
    cuda_used = cuda_free_memory_before_region - cuda_free_memory_after_region

    GB = 1 << 30
    MB = 1 << 20
    assert abs(cuda_used - 2 * GB) <= 5 * MB

    _memory_saver.pause()
    assert all(pause_flags)

    cuda_free_memory_after_pause = torch.cuda.mem_get_info()[0]
    cuda_free = cuda_free_memory_after_pause - cuda_free_memory_after_region
    assert abs(cuda_free - 2 * GB) <= 5 * MB

    _memory_saver.resume()
    assert all(resume_flags)

    cuda_free_memory_after_resume = torch.cuda.mem_get_info()[0]
    cuda_used = cuda_free_memory_after_pause - cuda_free_memory_after_resume
    assert abs(cuda_used - 2 * GB) <= 5 * MB
