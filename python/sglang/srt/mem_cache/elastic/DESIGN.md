# ElasticMem Integration — Design Document

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                               Scheduler                                 │
│       (calls emem_orch.try_resize() after each scheduling round)        │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      ElasticMempoolOrchestrator                         │
│                                                                         │
│  - Maintains allocators of various pool types                           │
│  - Configurable algorithm to decide when to trigger resize              │
│    and which pool to shrink (unmap) / grow (map)                        │
└────────┬───────────────────────────┬───────────────────────────┬────────┘
         │                           │                           │
         ▼                           ▼                           ▼
┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐
│ Elastic Allocator  │  │ Elastic Allocator  │  │ Elastic Allocator  │
│ (full)             │  │ (swa)              │  │ (mamba)            │
│                    │  │                    │  │                    │
│ - page-level       │  │ - page-level       │  │ - page-level       │
│   alloc / free     │  │   alloc / free     │  │   alloc / free     │
│ - candidate unmap  │  │ - candidate unmap  │  │ - candidate unmap  │
│   tracking         │  │   tracking         │  │   tracking         │
└─────────┬──────────┘  └─────────┬──────────┘  └─────────┬──────────┘
          │                       │                       │
          ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           Elastic Pool                                  │
│                                                                         │
│  Implementation A: Separate Pools + CUDA VMM (ETensor)                  │
│                                                                         │
│  - shrink() / expand()                                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

The system is layered into three tiers:

| Layer            | Class                                         | Responsibility                                                            |
|------------------|-----------------------------------------------|---------------------------------------------------------------------------|
| **Orchestrator** | `ElasticMempoolOrchestrator`                  | Maintains allocators of various pool types; configurable resize algorithm |
| **Allocator**    | `ElasticAllocator` (full / swa / mamba / ...) | Page-level alloc/free (page_size=1 → token-level), candidate tracking     |
| **Pool**         | `ElasticMempool`                              | KV buffer storage; shrink()/expand(); see Section 3 for implementations   |

---

## 2. Resize Trigger & Memory Defragmentation

The core question: **when to resize**, and once a donor pool is chosen, **how to reclaim its occupied pages** so that memory can be released or reused.

### 2.1 When to Trigger Resize

The orchestrator periodically checks whether memory should be rebalanced. The key inputs are the `token_usage()` of each registered allocator. A resize is triggered when:

- One pool is **overloaded** (usage exceeds a high-water mark) — it needs memory.
- Another pool is **underutilized** (usage below a low-water mark) — it can donate memory.
- The **usage gap** between the two exceeds a configurable threshold.

**Current implementation**: checked once per scheduling round in `Scheduler.get_next_batch_to_run()`, using threshold-based env vars (`SGLANG_CAN_MAP_THRESHOLD`, `SGLANG_CAN_UNMAP_THRESHOLD`, `SGLANG_RESIZE_TRIGGER_DIFF_RATIO`).

**Alternative trigger strategies** (for future exploration):

| Strategy                  | Description                                        | Trade-off                                                                    |
|---------------------------|----------------------------------------------------|------------------------------------------------------------------------------|
| Threshold-based (current) | Fire when usage gap exceeds static thresholds      | Simple; gives donor pool time to defragment before shrink is needed          |

### 2.2 Memory Defragmentation

Before a pool can shrink, the pages to be reclaimed must be free of live data. The defragmentation strategy depends on the pool implementation (see Section 3), but the general approaches are:

#### Approach A: Lazy Eviction

1. **Mark candidate region**: compute a boundary; pages beyond it become unmap candidates.
2. **Track unused pages**: maintain a bitmap, updated by the radix cache on every state transition (`_add_unused` / `_rm_unused`).
3. **Evict lazily**: proactively evict cached (but unlocked) entries from the candidate region.
4. **Check readiness**: verify all candidate pages are free.
5. **Defer if not ready**: if active requests still pin pages, defer to next round — no data is moved.

**Pros**: No data copying. Zero overhead when candidate region is naturally cleared.
**Cons**: Shrink may be delayed indefinitely if long-running requests pin pages.

---

## 3. Pool Implementation: Shrink / Grow Mechanics

The `ElasticMempool` interface exposes `shrink()` / `expand()`. Below are two implementations with different trade-offs.

### 3.1 Implementation A: Separate Pools + CUDA VMM (ETensor)

Each pool type (full / swa / mamba) has its own **separate** GPU buffer backed by `ETensor`, which uses CUDA Virtual Memory Management to map/unmap physical pages at the tail.

```
  Pool A (full)                            Pool B (swa / mamba)
┌──────────────────────────────┐         ┌──────────────────────────────┐
│ ██████████░░░░░░░░░░░░░░░░░░ │         │ ██████████████████████░░░░░░ │
│ ← used →  ← free / unmap →   │         │ ← used →           ← free →  │
└──────────────────────────────┘         └──────────────────────────────┘
             │ cuMemUnmap tail                            ▲ cuMemMap tail
             └──────────── physical pages ────────────────┘
```

**Shrink** (donor pool):

```
ElasticAllocator.reduce()
  → ElasticMempool.shrink(new_size)
      → ETensor.shrink() for each K/V buffer layer
          → cuMemUnmap tail pages
      → returns (released_cu_pages, new_pool_size)
  → update allocator bookkeeping (size, unused_pages, free_pages)
```

**Grow** (receiver pool):

```
ElasticAllocator.expand(expand_token_count)
  → ElasticMempool.expand(new_size)
      → ETensor.expand() for each K/V buffer layer
          → cuMemMap new pages at tail
      → returns (mapped_cu_pages, new_pool_size)
  → append new token slots to free_pages
  → extend unused_pages bitmap
```

**Page budget tracking**: because cu_page granularity and per-pool token-to-byte ratio differ, unmapping N cu_pages from pool A may not produce exactly N cu_pages for pool B. The orchestrator carries a `remaining_page` budget across cycles:

```
unmap_pages    = donor.reduce()
budget         = unmap_pages + remaining_page
map_tokens     = receiver.cu_page_to_token(budget)
map_pages      = receiver.expand(map_tokens)
remaining_page = budget - map_pages
```

**Defragmentation**: requires a contiguous tail region to be free before `cuMemUnmap`. Uses Approach A (lazy eviction of tail) or Approach B/C if urgent.

| Aspect              | Detail                                                        |
|---------------------|---------------------------------------------------------------|
| Memory isolation    | Each pool has its own virtual address range                   |
| Resize granularity  | cu_page aligned (e.g. 2 MB)                                   |
| Defrag requirement  | Contiguous tail must be cleared before unmap                  |
| External dependency | `kvcached` library (`ETensor`, `vmm_ops`)                     |
| Fragmentation       | No internal fragmentation within each pool (page-level alloc) |

**CUDA VMM Operation Latency** (2 MiB cu_page, full GPU memory, 100 iterations):

| GPU        | Memory     | Pages | MAP mean | UNMAP mean | REMAP mean | MAP /page | UNMAP /page |
|------------|------------|-------|----------|------------|------------|-----------|-------------|
| H20 140 GB | 111.78 GiB | 57233 | 52.6 ms  | 1680.3 ms  | 1743.9 ms  | 0.92 µs   | 29.4 µs     |
| A10 24 GB  | 17.65 GiB  | 9039  | 5.8 ms   | 156.7 ms   | 162.5 ms   | 0.64 µs   | 17.3 µs     |

Key observations:

- **MAP is ~30x faster than UNMAP**: `cuMemMap` ~1 µs/page; `cuMemUnmap` ~17–30 µs/page.
- **REMAP ≈ MAP + UNMAP**: no hidden overhead (<1% difference).
- **UNMAP dominates resize latency**: ~1000 pages (~2 GB) → UNMAP ~17–30 ms, MAP ~0.6–0.9 ms.
- **Implication**: UNMAP cost motivates early/predictive triggers (Section 2.1) so defrag + unmap can overlap with useful compute.

Raw benchmark data: see [Appendix A](#appendix-a-cuda-vmm-benchmark-data).

### 3.2 Post-Resize Sync

After shrink/grow (either implementation), all allocators call `update_size()` to propagate new pool sizes back to the scheduler (e.g., `scheduler.swa_tokens_per_layer`, `scheduler.full_tokens_per_layer`).

---

## Appendix A: CUDA VMM Benchmark Data

Benchmark configuration: 2 MiB cu_page, mapping full GPU memory, 100 iterations.

<details>
<summary>H20 140 GB</summary>

```
========== Benchmark Results ==========
Pages            : 57233
Page size        : 2 MiB
Total phys mapped: 111.78 GiB
Iterations       : 100

--- MAP (page-by-page cuMemMap) ---
  Metric       Total (ms)  Per-page (us)
  ------       ----------  -------------
  Mean             52.637         0.920
  Min              46.398         0.811
  P50              50.458         0.882
  P90              59.698         1.043
  Max              90.443         1.580

--- UNMAP (bulk cuMemUnmap) ---
  Metric       Total (ms)  Per-page (us)
  ------       ----------  -------------
  Mean           1680.326        29.359
  Min            1636.732        28.598
  P50            1678.775        29.332
  P90            1701.488        29.729
  Max            1761.224        30.773

--- REMAP-TO-DUMMY (unmap + page-by-page map to dummy) ---
  Metric       Total (ms)  Per-page (us)
  ------       ----------  -------------
  Mean           1743.867        30.470
  Min            1682.244        29.393
  P50            1744.303        30.477
  P90            1782.333        31.142
  Max            1815.750        31.726

--- Verification: REMAP ≈ MAP + UNMAP ---
  MAP mean:              52.637 ms
  UNMAP mean:          1680.326 ms
  MAP + UNMAP:         1732.963 ms (theoretical)
  REMAP-TO-DUMMY mean: 1743.867 ms (actual)
  Difference:            10.904 ms (0.6%)
=======================================
```

</details>

<details>
<summary>A10 24 GB</summary>

```
========== Benchmark Results ==========
Pages            : 9039
Page size        : 2 MiB
Total phys mapped: 17.65 GiB
Iterations       : 100

--- MAP (page-by-page cuMemMap) ---
  Metric       Total (ms)  Per-page (us)
  ------       ----------  -------------
  Mean              5.751         0.636
  Min               5.525         0.611
  P50               5.740         0.635
  P90               5.948         0.658
  Max               6.134         0.679

--- UNMAP (bulk cuMemUnmap) ---
  Metric       Total (ms)  Per-page (us)
  ------       ----------  -------------
  Mean            156.726        17.339
  Min             154.555        17.099
  P50             156.615        17.327
  P90             157.476        17.422
  Max             176.928        19.574

--- REMAP-TO-DUMMY (unmap + page-by-page map to dummy) ---
  Metric       Total (ms)  Per-page (us)
  ------       ----------  -------------
  Mean            162.515        17.979
  Min             159.453        17.641
  P50             162.116        17.935
  P90             163.800        18.121
  Max             186.182        20.598

--- Verification: REMAP ≈ MAP + UNMAP ---
  MAP mean:               5.751 ms
  UNMAP mean:           156.726 ms
  MAP + UNMAP:          162.477 ms (theoretical)
  REMAP-TO-DUMMY mean:  162.515 ms (actual)
  Difference:             0.038 ms (0.0%)
=======================================
```

</details>
