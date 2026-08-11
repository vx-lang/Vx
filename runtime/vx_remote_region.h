//===- vx_remote_region.h - Naming memory on another machine ----*- C++ -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// A buffer that lives on a remote worker, named by something a program can do
// pointer arithmetic on.
//
// See docs/discussions/implementation_plans/remote_dispatch_marshalling.md for
// why it is shaped this way. The short version, because it is not obvious:
//
//   * It cannot be an opaque token. `llama2.vx` stages each projection as one
//     blob spanning every layer and slices it with `vx_advance_ptr(w.wq, off)`,
//     seven times per layer. A token has no meaningful `+ offset`.
//   * It cannot be the worker's own pointer. `is_device_ptr` would answer "no"
//     for a remote address, and the plugin would then *stage* it -- reading
//     host memory at an address that is really remote. Two workers also mint
//     the same numeric value independently, so identity is (worker, address).
//
// So it is a worker-assigned address in a synthetic space, resolved here to a
// region and an offset. Arithmetic is plain addition and every existing pointer
// expression keeps working.
//
// Vendor-free on purpose, exactly as runtime/vx_dispatch_plan.h is: interval
// lookup, bounds checks and generation counters are pure logic whose failure
// mode is a silent misresolution rather than a build error, so they are tested
// on any machine (tests/runtime/remote_region_test.cpp). It is one shared
// header rather than a copy per backend because vx_host_call.h already records
// what happens otherwise -- copies "kept in sync" by a comment stop being so
// quietly.
//
//===----------------------------------------------------------------------===//

#ifndef VX_REMOTE_REGION_H
#define VX_REMOTE_REGION_H

#include <stddef.h>
#include <stdint.h>

/// Address layout.
///
///  63      56 55      48 47  46                                0
/// +----------+----------+---+---------------------------------+
/// |   0x00   | worker   | 0 |   offset -- 47 bits, 128 TiB    |
/// | (marker) | 1..255   |   |                                 |
/// +----------+----------+---+---------------------------------+
///
/// Bits 63:56 zero, 55:48 non-zero and bit 47 zero makes bits 63:47 neither
/// all-zeros nor all-ones, so every such value is **non-canonical** on x86-64
/// and AArch64. It can therefore never equal a user pointer (top bits zero) or
/// a kernel one (top bits set), and dereferencing it faults at the instruction
/// instead of quietly reading whatever is there.
///
/// That fault is the entire protection against a host touching remote memory,
/// and it is worth having: the `matmul_ane` crash during the GPU campaign was a
/// host loop dereferencing a staged device pointer, and the only reason it was
/// diagnosable is that it faulted rather than returning numbers.
#define VX_REMOTE_WORKER_SHIFT 48
#define VX_REMOTE_WORKER_MAX 255
#define VX_REMOTE_OFFSET_BITS 47
#define VX_REMOTE_OFFSET_MASK ((UINT64_C(1) << VX_REMOTE_OFFSET_BITS) - 1)
#define VX_REMOTE_WORKER_MASK (UINT64_C(0xFF) << VX_REMOTE_WORKER_SHIFT)

/// The smallest gap left after a region. Regions are also never followed
/// immediately by another regardless of size -- see `vx_remote_table_alloc`.
///
/// This was 4 GiB. Since the space is never reused -- deliberately, so that a
/// stale handle stays dead -- that set a ceiling on how many allocations a
/// worker could perform in its entire life:
///
///     128 TiB / 4 GiB = 32768
///
/// A dispatch stages two operands and llama2 issues 43 dispatches per token, so
/// that is roughly 380 tokens. Not 380 per request: 380 for as long as the
/// process runs. A worker on a two-A100 pod reached it after 16145 dispatches,
/// having served three shorter runs correctly first, which is exactly why it
/// read as "128 tokens is too many" rather than as a lifetime limit.
///
/// At 1 MiB the same arithmetic gives 134 million, and what binds instead is
/// the regions that are genuinely large: llama2 stages about 121 MB of stride
/// per token, so a worker manages on the order of a million tokens. Still
/// finite. Removing it properly means putting the generation in the handle so
/// freed space can be recycled without resurrecting anything.
///
/// Shrinking the floor does not weaken overrun detection, which is what the gap
/// is for: the gap is `max(size, VX_REMOTE_MIN_GAP)`, so every region already
/// gets a gap at least as large as itself. This floor governs only regions
/// smaller than the floor, and an overrun from a small region big enough to
/// clear 1 MiB is not the off-by-one being guarded against.
#define VX_REMOTE_MIN_GAP (UINT64_C(1) << 20)

/// Compose a synthetic address. `worker` is 1..255; 0 is reserved so that a
/// zeroed word is never a valid handle.
static inline uint64_t vx_remote_addr(uint32_t worker, uint64_t offset) {
  return ((uint64_t)worker << VX_REMOTE_WORKER_SHIFT) |
         (offset & VX_REMOTE_OFFSET_MASK);
}

static inline uint32_t vx_remote_addr_worker(uint64_t addr) {
  return (uint32_t)((addr & VX_REMOTE_WORKER_MASK) >> VX_REMOTE_WORKER_SHIFT);
}

static inline uint64_t vx_remote_addr_offset(uint64_t addr) {
  return addr & VX_REMOTE_OFFSET_MASK;
}

/// Whether a value is one of ours, decided by shape alone and without a table.
///
/// This is what has to be asked *before* `is_device_ptr`, which would answer
/// "not a device pointer" for a remote address and send the plugin off to stage
/// host memory that does not exist. Cheap enough to ask on every argument.
static inline int vx_remote_addr_is_handle(uint64_t addr) {
  /* bits 63:56 clear, worker non-zero, bit 47 clear */
  return (addr >> 56) == 0 && (addr & VX_REMOTE_WORKER_MASK) != 0 &&
         ((addr >> (VX_REMOTE_OFFSET_BITS)) & 1) == 0;
}

/// One resident buffer. Grows freely: nothing outside a plugin sees it, which
/// is the same arrangement `vx_gemm_plan` has.
typedef struct {
  uint64_t base;       /* synthetic address of the first byte */
  uint64_t size;       /* bytes; bounds which offsets resolve here */
  uint32_t worker;     /* identity is (worker, base), never base alone */
  int32_t dtype;       /* VX_DTYPE_*, for descriptor reconstruction */
  void *remote;        /* the worker's own pointer; never leaves the plugin */
  uint64_t generation; /* bumped on free, so a stale handle is detectable */
  int32_t live;        /* 0 once freed; the slot is not reused */
} vx_remote_region;

/// Where a resolved handle points.
typedef struct {
  const vx_remote_region *region;
  uint64_t offset; /* bytes from the region's base */
} vx_remote_ref;

/// The table. Deliberately a flat array kept sorted by base: a dispatch is at
/// minimum a PCIe round trip, so a binary search over a few thousand entries is
/// not on any critical path, and an array is trivially inspectable in a
/// debugger when a resolution goes wrong.
typedef struct {
  vx_remote_region *regions;
  size_t count;
  size_t capacity;
  /* Next free offset per worker; index 0 unused so worker ids match. */
  uint64_t next_offset[VX_REMOTE_WORKER_MAX + 1];
  uint64_t generation;
} vx_remote_table;

static inline void vx_remote_table_init(vx_remote_table *t,
                                        vx_remote_region *storage,
                                        size_t capacity) {
  size_t i;
  t->regions = storage;
  t->count = 0;
  t->capacity = capacity;
  t->generation = 1;
  for (i = 0; i <= VX_REMOTE_WORKER_MAX; ++i) {
    /* Start past zero so a region never has base with offset 0, which keeps a
       zeroed or one-off-by-one word from resolving to the first buffer. */
    t->next_offset[i] = VX_REMOTE_MIN_GAP;
  }
}

/// Reserve `size` bytes of address space for `worker` and record what backs it.
///
/// The gap after each region is `max(size, VX_REMOTE_MIN_GAP)`. A gap as large
/// as the region it follows catches any overrun shorter than the region itself,
/// and address space is the one resource here with no other use: 128 TiB
/// against plausibly a terabyte resident.
///
/// The gap is *not* what makes an out-of-bounds handle detectable -- that is
/// the bounds check in `vx_remote_resolve`. What it buys is the one case the
/// bounds check cannot see by itself: an overrun that reaches the *next*
/// region's base passes the check against that region and is indistinguishable
/// from a correct handle. That is not a contrived case, it is the off-by-one
/// this program is shaped to produce, `vx_advance_ptr(w.wq, l * dim * dim)`
/// with `l` one too large, which overshoots by exactly one region.
///
/// Returns the region's base address, or 0 on failure (bad worker, no room).
static inline uint64_t vx_remote_table_alloc(vx_remote_table *t,
                                             uint32_t worker, uint64_t size,
                                             int32_t dtype, void *remote) {
  vx_remote_region *r;
  uint64_t base, gap;

  if (worker == 0 || worker > VX_REMOTE_WORKER_MAX || size == 0) {
    return 0;
  }
  if (t->count == t->capacity) {
    return 0;
  }

  base = t->next_offset[worker];
  gap = size > VX_REMOTE_MIN_GAP ? size : VX_REMOTE_MIN_GAP;

  /* Overflow of a worker's 128 TiB, or of the gap arithmetic, means no room
     rather than a wrapped address that would alias the start of the space. */
  if (size > VX_REMOTE_OFFSET_MASK || base > VX_REMOTE_OFFSET_MASK - size ||
      base + size > VX_REMOTE_OFFSET_MASK - gap) {
    return 0;
  }

  r = &t->regions[t->count++];
  r->base = vx_remote_addr(worker, base);
  r->size = size;
  r->worker = worker;
  r->dtype = dtype;
  r->remote = remote;
  r->generation = t->generation++;
  r->live = 1;

  t->next_offset[worker] = base + size + gap;
  return r->base;
}

/// Resolve an address to the region containing it and the offset within.
///
/// Returns 0 and leaves `out` untouched when the address is not a handle, names
/// no live region, or lies outside the region it would fall in. That last check
/// is the mechanism: the search alone finds the region with the greatest base
/// not exceeding the address, and would happily return a neighbour for an
/// address past the end of its predecessor.
static inline int vx_remote_resolve(const vx_remote_table *t, uint64_t addr,
                                    vx_remote_ref *out) {
  size_t lo = 0, hi = t->count, best = (size_t)-1;

  if (!vx_remote_addr_is_handle(addr)) {
    return 0;
  }

  /* Greatest base <= addr. Bases are ascending because allocation only
     appends and each worker's offsets only increase; the worker field is in
     the high bits, so ordering by address orders by (worker, offset). */
  while (lo < hi) {
    size_t mid = lo + (hi - lo) / 2;
    if (t->regions[mid].base <= addr) {
      best = mid;
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }

  if (best == (size_t)-1) {
    return 0;
  }

  {
    const vx_remote_region *r = &t->regions[best];
    uint64_t off = addr - r->base;
    if (!r->live || off >= r->size) {
      return 0;
    }
    /* A handle for one worker must never resolve into another's space. The
       address ordering makes this true already; check it rather than trust it,
       because the failure is silent cross-machine aliasing. */
    if (vx_remote_addr_worker(addr) != r->worker) {
      return 0;
    }
    out->region = r;
    out->offset = off;
    return 1;
  }
}

/// Mark a region dead. Its address space is not reused, so a stale handle
/// resolves to a dead region and is refused rather than landing on whatever was
/// allocated next.
static inline int vx_remote_table_free(vx_remote_table *t, uint64_t base) {
  size_t i;
  for (i = 0; i < t->count; ++i) {
    if (t->regions[i].base == base && t->regions[i].live) {
      t->regions[i].live = 0;
      t->regions[i].generation = t->generation++;
      return 1;
    }
  }
  return 0;
}

#endif /* VX_REMOTE_REGION_H */
