#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <torch/extension.h>
#include <unordered_map>

// Error checking utility
static inline void check_cu_result(CUresult cu_result, const char *call,
                                   const char *file, unsigned line) {
  if (cu_result != CUDA_SUCCESS) {
    const char *error_string = nullptr;
    (void)cuGetErrorString(cu_result, &error_string);
    std::string error_msg = std::string(file) + ':' + std::to_string(line) +
                            ' ' + call + " error (" +
                            std::to_string(static_cast<unsigned>(cu_result)) +
                            "): " + error_string;
    throw std::invalid_argument(error_msg);
  }
}

#define CHECK_CU_RESULT(call) check_cu_result(call, #call, __FILE__, __LINE__)

// Global variables
static bool g_initialized = false;
static int g_device_id = 0;
static size_t k_granularity_size = 0;
static size_t k_physical_page_size = 0;
static CUmemGenericAllocationHandle k_zero_page = 0;

static constexpr size_t k_start_addr = 0x1f0'000'000'000ULL;
static std::atomic<size_t> g_vaddr_allocated_offset = 0;

static std::unordered_map<void *, CUmemGenericAllocationHandle> g_mapped_pages;
static std::mutex g_mapped_pages_mutex;

// Forward declarations for helper functions
static inline void check_cu_result(CUresult cu_result, const char *call,
                                   const char *file, unsigned line);
static inline CUmemAllocationProp create_mem_allocation_prop();
static void initialize_elastic_utils(int device_id);
static inline void init_with_zero(void *vaddr, size_t vsize);
static inline void validate_and_calculate_address(torch::Tensor tensor,
                                                  size_t offset, size_t size,
                                                  void *&vaddr,
                                                  size_t &num_pages);

// Forward declarations for API Functions
size_t get_physical_page_size();
void map_physical_page(torch::Tensor tensor, size_t offset, size_t size);
torch::Tensor create_etensor(size_t vsize, size_t psize,
                             torch::ScalarType dtype);
void unmap_physical_page(torch::Tensor tensor, size_t offset, size_t size);

// Helper function implementations
static inline CUmemAllocationProp create_mem_allocation_prop() {
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = g_device_id;
  return prop;
}

static void initialize_elastic_utils(int device_id) {
  if (g_initialized) {
    TORCH_CHECK(device_id == g_device_id);
    return;
  }

  g_device_id = device_id;

  CUmemAllocationProp prop = create_mem_allocation_prop();

  CHECK_CU_RESULT(cuMemGetAllocationGranularity(
      &k_granularity_size, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  CHECK_CU_RESULT(cuMemCreate(&k_zero_page, k_granularity_size, &prop, 0));

  const char *page_size_mb_str = std::getenv("PHYSICAL_PAGE_SIZE_MB");
  k_physical_page_size = k_granularity_size;
  if (page_size_mb_str != nullptr) {
    int page_size_mb = std::stoi(page_size_mb_str);
    size_t page_size = static_cast<size_t>(page_size_mb) << 20;
    TORCH_CHECK(page_size % k_granularity_size == 0);
    k_physical_page_size = page_size;
  }

  g_initialized = true;
}

static inline void init_with_zero(void *vaddr, size_t vsize) {
  for (size_t offset = 0; offset < vsize; offset += k_granularity_size) {
    void *chunk_addr =
        reinterpret_cast<void *>(reinterpret_cast<size_t>(vaddr) + offset);
    CHECK_CU_RESULT(cuMemMap(reinterpret_cast<CUdeviceptr>(chunk_addr),
                             k_granularity_size, 0, k_zero_page, 0));
  }
}

static inline void validate_and_calculate_address(torch::Tensor tensor,
                                                  size_t offset, size_t size,
                                                  void *&vaddr,
                                                  size_t &num_pages) {
  TORCH_CHECK(g_initialized);
  TORCH_CHECK(size % k_granularity_size == 0);
  TORCH_CHECK(offset % k_granularity_size == 0);
  void *base_vaddr = reinterpret_cast<void *>(tensor.data_ptr());
  vaddr = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(base_vaddr) +
                                   offset);
  num_pages = size / k_granularity_size;
}

// API Functions - These are the externally visible functions
size_t get_physical_page_size() {
  TORCH_CHECK(g_initialized);
  return k_physical_page_size;
}

torch::Tensor create_etensor(size_t vsize, size_t psize,
                             torch::ScalarType dtype) {
  TORCH_CHECK(vsize % k_granularity_size == 0);
  TORCH_CHECK(psize % k_granularity_size == 0);
  TORCH_CHECK(psize <= vsize);
  size_t element_size = torch::elementSize(dtype);
  auto device = torch::Device(torch::kCUDA, g_device_id);
  void *vaddr;
  size_t offset = g_vaddr_allocated_offset.fetch_add(vsize);
  CHECK_CU_RESULT(cuMemAddressReserve(reinterpret_cast<CUdeviceptr *>(&vaddr),
                                      vsize, k_granularity_size,
                                      k_start_addr + offset, 0ULL));
  init_with_zero(vaddr, vsize);
  int64_t numel = vsize / element_size;
  auto options =
      torch::TensorOptions().dtype(dtype).device(device).requires_grad(false);
  auto tensor = torch::from_blob(vaddr, {numel}, options);

  map_physical_page(tensor, 0, psize);

  return tensor;
}

void map_physical_page(torch::Tensor tensor, size_t offset, size_t size) {
  void *vaddr;
  size_t num_pages;
  validate_and_calculate_address(tensor, offset, size, vaddr, num_pages);

  std::lock_guard<std::mutex> lock(g_mapped_pages_mutex);
  for (size_t i = 0; i < num_pages; ++i) {
    void *chunk_vaddr = reinterpret_cast<void *>(
        reinterpret_cast<uintptr_t>(vaddr) + i * k_granularity_size);
    TORCH_CHECK(g_mapped_pages.find(chunk_vaddr) == g_mapped_pages.end());
  }

  CUmemAllocationProp prop = create_mem_allocation_prop();

  CUmemAccessDesc desc = {};
  desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  desc.location.id = g_device_id;
  desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  for (size_t i = 0; i < num_pages; ++i) {
    void *chunk_vaddr = reinterpret_cast<void *>(
        reinterpret_cast<uintptr_t>(vaddr) + i * k_granularity_size);
    CHECK_CU_RESULT(cuMemUnmap(reinterpret_cast<CUdeviceptr>(chunk_vaddr),
                               k_granularity_size));
    CUmemGenericAllocationHandle handle;
    CHECK_CU_RESULT(cuMemCreate(&handle, k_granularity_size, &prop, 0));
    CHECK_CU_RESULT(cuMemMap(reinterpret_cast<CUdeviceptr>(chunk_vaddr),
                             k_granularity_size, 0, handle, 0));
    CHECK_CU_RESULT(cuMemSetAccess(reinterpret_cast<CUdeviceptr>(chunk_vaddr),
                                   k_granularity_size, &desc, 1));
    g_mapped_pages[chunk_vaddr] = handle;
  }
}

void unmap_physical_page(torch::Tensor tensor, size_t offset, size_t size) {
  void *vaddr;
  size_t num_pages;
  validate_and_calculate_address(tensor, offset, size, vaddr, num_pages);

  std::lock_guard<std::mutex> lock(g_mapped_pages_mutex);
  for (size_t i = 0; i < num_pages; ++i) {
    void *chunk_vaddr = reinterpret_cast<void *>(
        reinterpret_cast<uintptr_t>(vaddr) + i * k_granularity_size);
    TORCH_CHECK(g_mapped_pages.find(chunk_vaddr) != g_mapped_pages.end());
  }

  for (size_t i = 0; i < num_pages; ++i) {
    void *chunk_vaddr = reinterpret_cast<void *>(
        reinterpret_cast<uintptr_t>(vaddr) + i * k_granularity_size);
    auto it = g_mapped_pages.find(chunk_vaddr);
    TORCH_CHECK(it != g_mapped_pages.end());
    CUmemGenericAllocationHandle handle = it->second;
    CHECK_CU_RESULT(cuMemUnmap(reinterpret_cast<CUdeviceptr>(chunk_vaddr),
                               k_granularity_size));
    CHECK_CU_RESULT(cuMemRelease(handle));
    g_mapped_pages.erase(it);

    CHECK_CU_RESULT(cuMemMap(reinterpret_cast<CUdeviceptr>(chunk_vaddr),
                             k_granularity_size, 0, k_zero_page, 0));
  }
}

void cleanup_etensor(torch::Tensor tensor) {
  size_t element_size =
      torch::elementSize(c10::typeMetaToScalarType(tensor.dtype()));
  size_t vsize = tensor.numel() * element_size;
  TORCH_CHECK(vsize % k_granularity_size == 0);
  size_t num_pages = vsize / k_granularity_size;
  void *base_vaddr = reinterpret_cast<void *>(tensor.data_ptr());

  std::lock_guard<std::mutex> lock(g_mapped_pages_mutex);
  for (size_t i = 0; i < num_pages; ++i) {
    void *chunk_vaddr = reinterpret_cast<void *>(
        reinterpret_cast<uintptr_t>(base_vaddr) + i * k_granularity_size);

    auto it = g_mapped_pages.find(chunk_vaddr);
    if (it == g_mapped_pages.end()) {
      CHECK_CU_RESULT(cuMemUnmap(reinterpret_cast<CUdeviceptr>(chunk_vaddr),
                                 k_granularity_size));
      continue;
    }

    CUmemGenericAllocationHandle handle = it->second;
    CHECK_CU_RESULT(cuMemUnmap(reinterpret_cast<CUdeviceptr>(chunk_vaddr),
                               k_granularity_size));
    CHECK_CU_RESULT(cuMemRelease(handle));
    g_mapped_pages.erase(it);
  }

  CHECK_CU_RESULT(
      cuMemAddressFree(reinterpret_cast<CUdeviceptr>(base_vaddr), vsize));
}

void shutdown_elastic_utils() {
  std::lock_guard<std::mutex> lock(g_mapped_pages_mutex);
  TORCH_CHECK(g_mapped_pages.empty());
  if (k_zero_page != 0) {
    cuMemRelease(k_zero_page);
    k_zero_page = 0;
  }
  g_initialized = false;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("initialize_elastic_utils", &initialize_elastic_utils,
        "Initialize elastic utils");
  m.def("get_physical_page_size", &get_physical_page_size,
        "Get physical page size");
  m.def("create_etensor", &create_etensor, "Create elastic tensor");
  m.def("map_physical_page", &map_physical_page,
        "Map physical page to tensor address + offset");
  m.def("unmap_physical_page", &unmap_physical_page,
        "Unmap physical page from tensor address + offset");
  m.def("cleanup_etensor", &cleanup_etensor,
        "Cleanup elastic tensor resources");
  m.def("shutdown_elastic_utils", &shutdown_elastic_utils,
        "Shutdown elastic utils and release all resources");
}
