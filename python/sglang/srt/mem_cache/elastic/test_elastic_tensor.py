"""
cd /sgl-workspace/sglang/python/sglang/srt/mem_cache/elastic
LD_PRELOAD=/usr/local/lib/python3.12/dist-packages/torch_memory_saver_hook_mode_preload.abi3.so \
pytest test_elastic_tensor.py -s
"""

import gc
import time

import pytest
import torch

from sglang.srt.mem_cache.elastic.elastic_tensor import (
    create_test_tensor,
    elastic_utils,
)


@pytest.fixture
def skip_if_no_cuda():
    """Skip tests if CUDA is not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")


def test_elastic_tensor_initialization(skip_if_no_cuda):
    """Test ElasticTensor initialization with correct properties."""
    num = 8
    state_shape = (1024, 512)
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    etensor = create_test_tensor(num, state_shape, dtype, device)

    # Verify initial state
    assert etensor.num == num
    assert etensor.shape == (num,) + state_shape
    assert etensor.tensor.shape == (etensor._max_num,) + state_shape
    assert etensor.tensor.dtype == dtype
    assert etensor.tensor.device.type == "cuda"

    del etensor
    gc.collect()
    elastic_utils.shutdown_elastic_utils()


def test_elastic_tensor_expansion_and_shrinkage(skip_if_no_cuda):
    """Test ElasticTensor expansion and shrinkage operations."""
    num = 8
    state_shape = (1024, 512)
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    etensor = create_test_tensor(num, state_shape, dtype, device)

    # Assign values to mapped portion
    initial_values = torch.arange(
        num * torch.prod(torch.tensor(state_shape)).item(),
        dtype=dtype,
        device=device,
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
        torch.ones(
            (new_num - num,) + state_shape,
            dtype=dtype,
            device=device,
        )
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
    gc.collect()
    elastic_utils.shutdown_elastic_utils()


def test_unmap_accessibility(skip_if_no_cuda):
    """Test accessing out-of-bounds indices raises appropriate error."""
    num = 4
    state_shape = (1024, 512)
    dtype = torch.float32
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    etensor = create_test_tensor(num, state_shape, dtype, device)

    # Access index num (which should be out of bounds since valid range is [0, num))
    with pytest.raises(RuntimeError):
        _ = etensor.tensor[num : num + 1].clone()

    # Cleanup
    del etensor
    gc.collect()
    elastic_utils.shutdown_elastic_utils()


def test_tms(skip_if_no_cuda):
    """Test ElasticTensor integration with torch_memory_saver."""
    num = 512
    state_shape = (1024, 1024)
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    import weakref

    import torch_memory_saver

    from sglang.srt.mem_cache.elastic.elastic_tensor import (
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
            etensor_list[i] = create_test_tensor(
                num,
                state_shape,
                dtype,
                device,
            )
            et_ref_list[i] = weakref.ref(etensor_list[i])

            def make_on_pause(index):
                def on_pause():
                    t = et_ref_list[index]()
                    if t is None:
                        return
                    assert t.num >= 0 and t.tms_saved_num == -1
                    t.tms_saved_num = t.num
                    t.shrink(0)
                    assert t.num == 0 and t.tms_saved_num != -1
                    pause_flags[index] = True

                return on_pause

            def make_on_resume(index):
                def on_resume():
                    t = et_ref_list[index]()
                    if t is None:
                        return
                    assert t.num == 0 and t.tms_saved_num != -1
                    t.expand(t.tms_saved_num)
                    t.tms_saved_num = -1
                    assert t.num >= 0 and t.tms_saved_num == -1
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


def _calc_stats(times: list) -> dict:
    """Calculate statistics for timing data."""
    s = sorted(times)
    n = len(s)
    pct = lambda p: s[int((n - 1) * p / 100)]
    return {
        "min": s[0],
        "max": s[-1],
        "avg": sum(s) / n,
        "p50": pct(50),
        "p90": pct(90),
        "p99": pct(99),
    }


def test_benchmark_map_unmap_10gb(skip_if_no_cuda):
    """Benchmark map/unmap latency for ~10GB (or max available) physical pages."""
    GB = 1 << 30
    state_shape, dtype = (1024, 512), torch.bfloat16
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    element_size = 1024 * 512 * dtype.itemsize  # 1 MB per element

    # Use min(10 GB, 80% of free GPU memory) to avoid OOM
    bench_bytes = min(10 * GB, int(torch.cuda.mem_get_info(device)[0] * 0.8))
    target_num = bench_bytes // element_size
    if target_num < 2:
        pytest.skip("Not enough free GPU memory for benchmark")

    etensor = create_test_tensor(1, state_shape, dtype, device)
    torch.cuda.synchronize()

    # Warm-up
    etensor.expand(2)
    etensor.shrink(1)
    torch.cuda.synchronize()

    # Run benchmark iterations
    N, map_times, unmap_times = 10, [], []
    for _ in range(N):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        pages = etensor.expand(target_num)
        torch.cuda.synchronize()
        map_times.append(time.perf_counter() - t0)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        etensor.shrink(1)
        torch.cuda.synchronize()
        unmap_times.append(time.perf_counter() - t0)

    data_gb = pages * etensor._page_size / GB
    m, u = _calc_stats(map_times), _calc_stats(unmap_times)

    # Print results
    fmt = (
        lambda s, gb: f"{s['min']*1e3:>9.2f}ms {s['avg']*1e3:>9.2f}ms {s['p50']*1e3:>9.2f}ms {s['p90']*1e3:>9.2f}ms {s['p99']*1e3:>9.2f}ms {s['max']*1e3:>9.2f}ms {gb/s['avg']:>10.2f} GB/s"
    )
    print(
        f"\n{'='*92}\n  Benchmark: map/unmap physical pages ({N} iterations)\n{'='*92}"
    )
    print(
        f"  Page size : {etensor._page_size / 1024:.0f} KB | Data size : {data_gb:.2f} GB ({pages} pages)"
    )
    print(
        f"{'-'*92}\n  {'Operation':<8} {'Min':>10} {'Avg':>10} {'P50':>10} {'P90':>10} {'P99':>10} {'Max':>10} {'Throughput':>12}\n{'-'*92}"
    )
    print(f"  {'Map':<8} {fmt(m, data_gb)}\n  {'Unmap':<8} {fmt(u, data_gb)}\n{'='*92}")

    del etensor
    gc.collect()
    elastic_utils.shutdown_elastic_utils()
