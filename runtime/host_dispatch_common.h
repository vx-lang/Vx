//===- host_dispatch_common.h - Shared CPU backend body ---------*- C++ -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The whole vx_plugin_* ABI for a CPU target, parameterized by the two things
// that actually differ between them: what the backend calls itself, and the
// alignment its vector unit wants.
//
// Every target has a backend. That is the point of the arrangement rather than
// an accident of it: the compiler emits calls to the plugin ABI -- allocate,
// transfer, free, dispatch -- and never names a vendor. `transfer` becomes
// cudaMalloc plus an H2D copy where runtime/cuda_dispatch.cpp is linked, and an
// aligned host allocation here, and the program is correct either way. A
// compiler that special-cased libc for "the CPU case" would have to know which
// case it was in, which is exactly the knowledge the plugin boundary exists to
// avoid.
//
// A backend including this file defines:
//   VX_BACKEND_NAME    what it calls itself in diagnostics
//   VX_BACKEND_ALIGN   allocation alignment in bytes
//
//===----------------------------------------------------------------------===//

#ifndef VX_HOST_DISPATCH_COMMON_H
#define VX_HOST_DISPATCH_COMMON_H

#include "vx_dispatch_plan.h"
#include "vx_host_call.h"
#include "vx_remote_routing.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifndef VX_BACKEND_NAME
#error "a backend including host_dispatch_common.h must define VX_BACKEND_NAME"
#endif
#ifndef VX_BACKEND_ALIGN
#error "a backend including host_dispatch_common.h must define VX_BACKEND_ALIGN"
#endif

namespace {

// Diagnostics are opt-in: the backend test harness compares a program's stdout
// against its `// EXPECT:` lines, so an unconditional banner would fail every
// placed test.
inline bool vx_verbose() {
  static const bool on = [] {
    const char *v = getenv("VX_DISPATCH_VERBOSE");
    return v && v[0] != '\0' && strcmp(v, "0") != 0;
  }();
  return on;
}

/// C = A * B, row-major, for the element types this backend can carry.
///
/// The reference implementation, deliberately: it exists so a CPU backend gives
/// the same *answer to the routing question* an accelerator one does, not so it
/// goes faster. See the call site in vx_plugin_dispatch_async.
template <typename T>
inline void vx_host_gemm_typed(const vx_gemm_plan &plan, T *out,
                               int64_t out_row_stride) {
  const T *a = static_cast<const T *>(plan.a_data);
  const T *b = static_cast<const T *>(plan.b_data);
  for (int64_t i = 0; i < plan.m; ++i) {
    for (int64_t j = 0; j < plan.n; ++j) {
      T acc = static_cast<T>(0);
      for (int64_t k = 0; k < plan.k; ++k) {
        acc += a[i * plan.a_row_stride + k] * b[k * plan.b_row_stride + j];
      }
      out[i * out_row_stride + j] = acc;
    }
  }
}

/// Run a decoded plan. Returns false for anything it cannot carry, which sends
/// the caller back to the outlined kernel -- a refusal costs performance and
/// never correctness, exactly as it does in the CUDA backend.
inline bool vx_host_run_gemm(const vx_gemm_plan &plan) {
  size_t esz = vx_dtype_bytes(plan.dtype);
  if (esz == 0 || (plan.dtype != VX_DTYPE_F32 && plan.dtype != VX_DTYPE_F64)) {
    return false;
  }

  void *out = plan.out_data;
  int64_t stride = plan.out_row_stride;

  if (plan.out_kind == VX_GEMM_OUT_SLOT) {
    /* Standing in for the kernel means doing what it would have: allocate the
       result and publish a descriptor naming it. The allocation outlives this
       call because the caller loads that descriptor out of the slot. */
    out = malloc((size_t)plan.m * (size_t)plan.n * esz);
    if (!out) {
      return false;
    }
    stride = plan.n;
  } else if (!out) {
    return false;
  }

  if (plan.dtype == VX_DTYPE_F32) {
    vx_host_gemm_typed<float>(plan, static_cast<float *>(out), stride);
  } else {
    vx_host_gemm_typed<double>(plan, static_cast<double *>(out), stride);
  }

  if (plan.out_kind == VX_GEMM_OUT_SLOT) {
    vx_gemm_publish_slot(&plan, out);
  }
  if (vx_verbose()) {
    fprintf(stderr, "[Vx " VX_BACKEND_NAME "] GEMM %lldx%lldx%lld %s -> %s\n",
            (long long)plan.m, (long long)plan.n, (long long)plan.k,
            vx_dtype_name(plan.dtype),
            plan.out_kind == VX_GEMM_OUT_SLOT ? "slot" : "buffer");
  }
  return true;
}

/// Describe one argument, for VX_DISPATCH_VERBOSE. The tag says what each
/// pointer is; for anything ranked the descriptor says how big it is.
inline void vx_describe_arg(int64_t i, int32_t tag, void *arg) {
  if (VX_ABI_KIND(tag) != VX_ABI_KIND_MEMREF) {
    fprintf(stderr, "  arg %lld: scalar kind=%d\n", (long long)i,
            VX_ABI_KIND(tag));
    return;
  }

  int32_t rank = VX_ABI_RANK(tag);
  int32_t elem = VX_ABI_ELEM(tag);
  if (rank == 0 && elem == VX_DTYPE_UNKNOWN) {
    // Kind 0 covers both a ranked memref and the pointer fallback in
    // abiTagForType; only the latter carries no element type or rank.
    fprintf(stderr, "  arg %lld: opaque ptr\n", (long long)i);
    return;
  }

  const void *desc = *(const void **)arg;
  bool is_slot = VX_ABI_IS_SLOT(tag);
  if (is_slot && desc) {
    desc = vx_memref_aligned(desc);
  }

  fprintf(stderr, "  arg %lld: %s %s rank=%d shape=[", (long long)i,
          is_slot ? "slot ->" : "memref", vx_dtype_name(elem), rank);
  if (desc) {
    const int64_t *sizes = vx_memref_sizes(desc);
    for (int32_t d = 0; d < rank; ++d) {
      fprintf(stderr, "%s%lld", d ? "x" : "", (long long)sizes[d]);
    }
  }
  fprintf(stderr, "] elem_bytes=%zu\n", vx_dtype_bytes(elem));
}

} // namespace

/// Binding an allocation to the NUMA node the machine model named.
///
/// A memory space that declares `node: N` is dispatched with a banded id rather than the hash of
/// its name (`arch::numa_dispatch_id`), because a hash cannot be turned back into a node number
/// and `mbind` needs one. THIS RANGE MIRRORS src/arch.rs AND THE TWO MUST CHANGE TOGETHER.
///
/// The syscall is used directly rather than libnuma, so the runtime gains no build dependency and
/// no link flag: `mbind` is a kernel interface, and libnuma is a convenience wrapper over it.
/// Everything here is Linux-only and compiles out elsewhere -- macOS has one memory and nothing to
/// choose between.
#define VX_NUMA_DISPATCH_BASE 1000u
#define VX_NUMA_MAX_NODE 999u

/// The node an id names, or -1 if the id is not a NUMA one.
static int vx_numa_node_of(uint32_t topology_id) {
  if (topology_id >= VX_NUMA_DISPATCH_BASE &&
      topology_id <= VX_NUMA_DISPATCH_BASE + VX_NUMA_MAX_NODE) {
    return (int)(topology_id - VX_NUMA_DISPATCH_BASE);
  }
  return -1;
}

#if defined(__linux__)
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#ifndef MPOL_BIND
#define MPOL_BIND 2
#endif
#ifndef MPOL_MF_MOVE
#define MPOL_MF_MOVE (1 << 1)
#endif

/// One page in front of every NUMA allocation, holding the mapping's length.
///
/// `munmap` needs the size and `vx_plugin_free` is handed only a pointer. A side table would work
/// and would need a lock on every allocation and free; a header costs one page and no
/// synchronization. The page also keeps the returned pointer page-aligned, which is what `mbind`
/// wants anyway.
#define VX_NUMA_HEADER 4096

static void vx_numa_registry_record(void *ptr, size_t bytes, int node);
static void vx_numa_registry_forget(void *ptr);

/// Map `bytes` and bind the mapping to `node`. `*bound` says whether the BINDING took; the
/// mapping either happened or the call returns null. Keeping those two outcomes apart is what
/// lets the free path stay unambiguous: every allocation for a NUMA-banded id is an mmap with a
/// header, whether or not the bind succeeded, so the free always unmaps and never has to guess
/// which allocator produced a pointer.
static void *vx_numa_alloc(size_t bytes, int node, int *bound) {
  *bound = 0;
  size_t total = bytes + VX_NUMA_HEADER;
  void *base = mmap(nullptr, total, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (base == MAP_FAILED) {
    return nullptr;
  }

  // A bitmask over nodes, wide enough for the largest node the band can carry.
  unsigned long mask[VX_NUMA_MAX_NODE / (8 * sizeof(unsigned long)) + 1] = {0};
  mask[node / (int)(8 * sizeof(unsigned long))] |=
      1UL << (node % (int)(8 * sizeof(unsigned long)));

  // MPOL_BIND rather than MPOL_PREFERRED: the model said this memory is on that node, and a
  // policy that silently falls back to another one would make the declaration advisory. If the
  // node cannot satisfy it the allocation should fail loudly rather than land somewhere else and
  // be measured as though it had not.
  long rc = syscall(__NR_mbind, base, total, MPOL_BIND, mask,
                    (unsigned long)(sizeof(mask) * 8), MPOL_MF_MOVE);
  // A failed bind keeps the mapping: the memory is valid and merely unplaced, and unmapping it
  // here would turn a performance outcome into an allocation failure.
  *bound = (rc == 0);

  *(size_t *)base = total;
  return (char *)base + VX_NUMA_HEADER;
}

static void vx_numa_free(void *ptr) {
  char *base = (char *)ptr - VX_NUMA_HEADER;
  size_t total = *(size_t *)base;
  vx_numa_registry_forget(ptr);
  munmap(base, total);
}

/// Which node each live placed allocation is on.
///
/// The allocations already know -- the node is implied by the id that made them -- but a
/// DISPATCH does not: `vx_plugin_dispatch_async` is handed argument pointers and a payload, and
/// the payload's `topo=` names the topology the kernel spawned on, which on a host is
/// `Topology::CPU` and says nothing about a socket. So the node has to be recovered from the
/// arguments, and that means remembering where each one went.
///
/// Small and linear on purpose: a program has a handful of placed tiles, and both the insert and
/// the lookup happen once per transfer and once per dispatch. Neither is on a path where a hash
/// map would repay its own complexity.
struct vx_numa_entry {
  void *ptr;
  size_t bytes;
  int node;
};
static vx_numa_entry vx_numa_table[256];
static size_t vx_numa_table_len = 0;
static pthread_mutex_t vx_numa_table_lock = PTHREAD_MUTEX_INITIALIZER;

static void vx_numa_registry_record(void *ptr, size_t bytes, int node) {
  pthread_mutex_lock(&vx_numa_table_lock);
  if (vx_numa_table_len < sizeof(vx_numa_table) / sizeof(vx_numa_table[0])) {
    vx_numa_table[vx_numa_table_len++] = {ptr, bytes, node};
  }
  // Overflowing is not an error and must not be fatal: the allocation is placed either way, and
  // the only thing lost is the dispatch's ability to follow it. A program with more than 256 live
  // placed tiles gets correct results and unpinned threads.
  pthread_mutex_unlock(&vx_numa_table_lock);
}

static void vx_numa_registry_forget(void *ptr) {
  pthread_mutex_lock(&vx_numa_table_lock);
  for (size_t i = 0; i < vx_numa_table_len; i++) {
    if (vx_numa_table[i].ptr == ptr) {
      vx_numa_table[i] = vx_numa_table[vx_numa_table_len - 1];
      vx_numa_table_len--;
      break;
    }
  }
  pthread_mutex_unlock(&vx_numa_table_lock);
}

/// The node a pointer was placed on, or -1 if it was not one of ours. Exact rather than a guess:
/// only pointers this file placed are in the table, so an ordinary heap pointer answers -1 and
/// nothing reads a header it does not have.
static int vx_numa_registry_node_of(void *ptr) {
  int node = -1;
  pthread_mutex_lock(&vx_numa_table_lock);
  for (size_t i = 0; i < vx_numa_table_len; i++) {
    if (vx_numa_table[i].ptr == ptr) {
      node = vx_numa_table[i].node;
      break;
    }
  }
  pthread_mutex_unlock(&vx_numa_table_lock);
  return node;
}

/// Pin the calling thread to the CPUs of `node`, read from sysfs rather than libnuma.
///
/// Best effort throughout: a machine without the sysfs entry, or a thread whose affinity is
/// already constrained by something outside this process, keeps whatever it had. Being unable to
/// pin is not an error -- the kernel still runs and reads the same bytes, more slowly.
static bool vx_numa_pin_to_node(int node) {
  char path[128];
  snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpulist", node);
  FILE *f = fopen(path, "r");
  if (!f) {
    return false;
  }
  char list[1024];
  if (!fgets(list, sizeof(list), f)) {
    fclose(f);
    return false;
  }
  fclose(f);

  cpu_set_t set;
  CPU_ZERO(&set);
  // "0-23,48-71" -- ranges and singletons, comma separated.
  const char *p = list;
  while (*p) {
    int lo = 0, hi = 0, consumed = 0;
    if (sscanf(p, "%d-%d%n", &lo, &hi, &consumed) == 2) {
    } else if (sscanf(p, "%d%n", &lo, &consumed) == 1) {
      hi = lo;
    } else {
      break;
    }
    for (int c = lo; c <= hi && c < CPU_SETSIZE; c++) {
      CPU_SET(c, &set);
    }
    p += consumed;
    if (*p == ',') {
      p++;
    } else {
      break;
    }
  }
  if (CPU_COUNT(&set) == 0) {
    return false;
  }
  return sched_setaffinity(0, sizeof(set), &set) == 0;
}
#else
static void *vx_numa_alloc(size_t bytes, int node, int *bound) {
  (void)bytes;
  (void)node;
  *bound = 0;
  return nullptr;
}
static void vx_numa_free(void *ptr) { (void)ptr; }
static void vx_numa_registry_record(void *ptr, size_t bytes, int node) {
  (void)ptr;
  (void)bytes;
  (void)node;
}
static int vx_numa_registry_node_of(void *ptr) {
  (void)ptr;
  return -1;
}
static bool vx_numa_pin_to_node(int node) {
  (void)node;
  return false;
}
#endif

extern "C" {

/// Allocate in this backend's memory and copy into it.
///
/// A CPU backend's "device memory" is host memory, so this is an aligned
/// allocation -- but it goes through the same entry point a GPU backend
/// implements with cudaMalloc, which is what lets `transfer` mean the same
/// thing in a program compiled for either.
///
/// `space_access` is accepted and ignored, and that is the correct behaviour
/// rather than an omission: what it reports is whether the program believes the
/// host can read the result, and here the result is host memory, so the belief
/// is true however the space was declared. Only a backend that hands out memory
/// the host cannot read has anything to check.
void *vx_plugin_alloc_and_transfer(size_t bytes, void *host_ptr,
                                   uint32_t topology_id,
                                   uint32_t space_access) {
  (void)space_access;
  void *remote = nullptr;
  if (vx_routing_try_alloc(bytes, host_ptr, topology_id, &remote)) {
    return remote;
  }
  if (bytes == 0) {
    return nullptr;
  }

  // A space that named a NUMA node gets its bytes on that node, rather than wherever the first
  // touch happens to fall. This is the whole difference between the model pricing a placement and
  // the placement actually happening.
  //
  // Two failures, kept apart because they mean different things. Failing to MAP is an ordinary
  // out-of-memory and aborts, like the fallback below. Failing to BIND is not an error at all --
  // the node may be offline, the kernel may have no NUMA support, or this may not be Linux --
  // and the program is correct in every one of those cases, only unplaced. It warns rather than
  // aborting, because a machine with one memory has nothing this could have been faster than.
  int numa_node = vx_numa_node_of(topology_id);
  if (numa_node >= 0) {
    int bound = 0;
    void *placed = vx_numa_alloc(bytes, numa_node, &bound);
    if (!placed) {
      fprintf(stderr,
              "[Vx " VX_BACKEND_NAME "] FATAL: could not map %zu bytes for NUMA node %d\n",
              bytes, numa_node);
      abort();
    }
    if (!bound) {
      // The mapping is valid and unplaced. Said once, because a run expected to be placed must
      // not quietly read as one that was -- the measurement would be of the wrong thing, and
      // nothing else about the program would look different.
      static bool warned = false;
      if (!warned) {
        warned = true;
        fprintf(stderr,
                "[Vx " VX_BACKEND_NAME "] WARNING: could not bind an allocation to NUMA node %d; "
                "the memory is unplaced. The program is correct and the placement the machine "
                "model declared is not in effect.\n",
                numa_node);
      }
    }
    if (bound) {
      vx_numa_registry_record(placed, bytes, numa_node);
    }
    if (host_ptr) {
      memcpy(placed, host_ptr, bytes);
    }
    if (vx_verbose()) {
      fprintf(stderr, "[Vx " VX_BACKEND_NAME "] staged %zu bytes on NUMA node %d%s\n", bytes,
              numa_node, bound ? "" : " (bind failed; unplaced)");
    }
    return placed;
  }

  // Rounded up because aligned_alloc requires a size that is a multiple of the
  // alignment; the caller only ever addresses the bytes it asked for.
  size_t rounded =
      (bytes + VX_BACKEND_ALIGN - 1) & ~(size_t)(VX_BACKEND_ALIGN - 1);
  void *ptr = aligned_alloc(VX_BACKEND_ALIGN, rounded);
  if (!ptr) {
    fprintf(stderr,
            "[Vx " VX_BACKEND_NAME "] FATAL: out of memory for %zu bytes\n",
            bytes);
    abort();
  }
  if (host_ptr) {
    memcpy(ptr, host_ptr, bytes);
  }
  if (vx_verbose()) {
    fprintf(stderr, "[Vx " VX_BACKEND_NAME "] staged %zu bytes\n", bytes);
  }
  return ptr;
}

/// Release what the entry point above returned.
///
/// Paired with it deliberately: an allocation made by a backend has to be freed
/// by the same backend. Handing this pointer to libc would be correct here and
/// heap corruption under the CUDA backend, which is why the compiler emits this
/// call rather than a `free`.
void vx_plugin_free(void *device_ptr, uint32_t topology_id) {
  if (vx_routing_try_free(device_ptr, topology_id)) {
    return;
  }
  vx_routing_refuse_handle("a free", device_ptr, topology_id);
  // Paired with the allocator above by the same id that chose it. A NUMA allocation is an
  // `mmap` with a header and has to be unmapped; handing it to `free` would be the same class of
  // error the comment on this function already warns about for device memory.
  //
  // The id alone decides, and that is a claim about every allocator in this file rather than
  // about this function. Each one that can be handed a NUMA-banded id -- the staging transfer
  // and the peer handoff -- maps with a header, so a banded id always means an mmap. A failed
  // MAPPING aborts and a failed BIND keeps the mapping, so there is no third outcome.
  //
  // An earlier version of this comment asserted the same thing while `vx_plugin_transfer_peer`
  // still returned `malloc` memory, and the first program to transfer between two domains
  // aborted here. Anything added later that allocates for a banded id has to map too.
  if (vx_numa_node_of(topology_id) >= 0) {
    vx_numa_free(device_ptr);
    return;
  }
  free(device_ptr);
}

uint64_t vx_plugin_dispatch_async(const void *binary_payload,
                                  size_t payload_size, void **device_args,
                                  const int32_t *arg_tags, int64_t num_args) {
  const char *kernel_name = static_cast<const char *>(binary_payload);

  if (vx_routing_try_dispatch(binary_payload, payload_size, device_args,
                              arg_tags, num_args)) {
    return 1;
  }

  // Run where the data is.
  //
  // The outlined kernel is one `ffi_call` on this thread -- there is no pool to spread -- so if
  // its arguments were placed on a node, this thread should be on that node too. Otherwise the
  // placement bought nothing: the bytes are local to a socket the thread is not on, and every
  // access crosses the interconnect exactly as it would have unplaced.
  //
  // Measured single-threaded on a two-socket c5.metal, reading 2 GiB: 12.7-13.8 GB/s local
  // against 8.7-9.0 remote, so this is worth about 1.5x on memory-bound work. The larger figure
  // a placed multi-threaded run reaches (242 GB/s against 180 interleaved) is not available
  // here and will not be until the host backend runs a kernel on more than one thread.
  //
  // Whichever node holds the most placed bytes wins, because a kernel reading two tiles on
  // different nodes has to be somewhere and the bigger one is the better guess. Arguments that
  // were never placed do not vote.
  {
    // An argument is a memref DESCRIPTOR, not the data. The placed pointer is its aligned base,
    // one indirection in -- and a slot argument holds the descriptor itself by reference, so it
    // is two. Looking the descriptor up in the registry finds nothing, which is exactly what
    // happened the first time this ran: every transfer placed correctly and the dispatch pinned
    // nothing, with no error anywhere to say why.
    size_t by_node[VX_NUMA_MAX_NODE + 1] = {0};
    bool any = false;
    for (int64_t i = 0; i < num_args; ++i) {
      if (!device_args || !device_args[i]) {
        continue;
      }
      if (VX_ABI_KIND(arg_tags[i]) != VX_ABI_KIND_MEMREF) {
        continue;
      }
      const void *desc = device_args[i];
      if (VX_ABI_IS_SLOT(arg_tags[i])) {
        desc = *(void *const *)desc;
        if (!desc) {
          continue;
        }
      }
      int nd = vx_numa_registry_node_of(vx_memref_aligned(desc));
      if (nd >= 0 && (uint32_t)nd <= VX_NUMA_MAX_NODE) {
        by_node[nd] += 1;
        any = true;
      }
    }
    // An opt-out, because a process that manages its own affinity should not have it changed
    // underneath it -- a benchmark harness pinning threads itself, or a caller running several
    // Vx dispatches on a thread it placed deliberately. It is also how the effect of this is
    // measured at all: the same binary, one variable.
    if (any && !getenv("VX_NUMA_NO_AFFINITY")) {
      int best = 0;
      for (uint32_t nd = 1; nd <= VX_NUMA_MAX_NODE; nd++) {
        if (by_node[nd] > by_node[best]) {
          best = (int)nd;
        }
      }
      bool pinned = vx_numa_pin_to_node(best);
      if (vx_verbose()) {
        fprintf(stderr, "[Vx " VX_BACKEND_NAME "] %s to NUMA node %d for this dispatch\n",
                pinned ? "pinned" : "could NOT pin", best);
      }
    }
  }

  if (vx_verbose()) {
    const char *kind = vx_payload_field(binary_payload, payload_size, "kind=");
    const char *roles =
        vx_payload_field(binary_payload, payload_size, "roles=");
    const char *outkind =
        vx_payload_field(binary_payload, payload_size, "outkind=");
    fprintf(stderr,
            "[Vx " VX_BACKEND_NAME "] host execution of %s (%lld args), "
            "kind=%s roles=%s outkind=%s\n",
            kernel_name, static_cast<long long>(num_args),
            kind ? kind : "<unclassified>", roles ? roles : "-",
            outkind ? outkind : "-");
    for (int64_t i = 0; i < num_args; ++i) {
      vx_describe_arg(i, arg_tags[i], device_args[i]);
    }
  }

  /* A classified matmul runs here rather than through the outlined kernel, the
     same way the CUDA backend hands one to cuBLAS.
   *
   * This is not an optimisation -- the loop nest computes the same numbers, and
   * a reference GEMM is no faster. It is what makes this backend answer the
   * same question the accelerator ones do. Without it the two disagree about
   * whether a kernel is *routable*, and that difference is load-bearing in one
   * place: a remote worker (runtime/vx_worker_main.cpp) has no outlined kernel
   * to fall back to, because the artifact was never shipped to it. A CPU worker
   * that could not route a matmul would abort on the first one, which is
   * exactly what it did before this existed.
   *
   * So a CPU machine can serve dispatches, and a disaggregated program can be
   * developed and tested without a GPU anywhere -- the property the rest of
   this
   * ABI already has. */
  {
    vx_gemm_plan plan;
    if (vx_gemm_plan_decode(binary_payload, payload_size, device_args, arg_tags,
                            num_args, &plan) &&
        vx_host_run_gemm(plan)) {
      return 1;
    }
  }

  vx_host_refuse_coop(binary_payload, payload_size, kernel_name,
                      VX_BACKEND_NAME);
  void *kernel = vx_host_kernel_symbol(kernel_name);
  if (!kernel) {
    fprintf(stderr,
            "[Vx " VX_BACKEND_NAME "] FATAL: outlined kernel %s not found\n",
            kernel_name);
    abort();
  }

  if (!vx_host_call_kernel(kernel, device_args, arg_tags, num_args,
                           vx_host_worker_count_for(binary_payload,
                                                    payload_size))) {
    fprintf(stderr,
            "[Vx " VX_BACKEND_NAME "] FATAL: could not build a call for %s\n",
            kernel_name);
    abort();
  }
  return 1;
}

uint64_t vx_plugin_dispatch_async_flat(float *xout, float *x, float *w, int n,
                                       int d) {
  // Reference matmul: xout[i] = sum_j w[i * n + j] * x[j], matching the
  // signature the flat path emits (see #31).
  for (int i = 0; i < d; ++i) {
    float acc = 0.0f;
    for (int j = 0; j < n; ++j) {
      acc += w[i * n + j] * x[j];
    }
    xout[i] = acc;
  }
  return 1;
}

void vx_plugin_await_future(uint64_t future_id) {
  // Dispatch is synchronous on this path, so the result is ready on return.
  (void)future_id;
}

int32_t vx_plugin_transfer_device_to_host(void *device_ptr, void *host_ptr,
                                          size_t bytes, uint32_t topology_id) {
  if (vx_routing_try_fetch(device_ptr, host_ptr, bytes, topology_id)) {
    return 1;
  }
  vx_routing_refuse_handle("a read-back", device_ptr, topology_id);
  // Otherwise one memory, so the topology names it and nothing follows.
  memcpy(host_ptr, device_ptr, bytes);
  return 1;
}

void *vx_plugin_transfer_peer(void *src_device_ptr, uint32_t src_topology_id,
                              uint32_t dst_topology_id, size_t bytes) {
  void *routed = nullptr;
  if (vx_routing_try_peer(src_device_ptr, src_topology_id, dst_topology_id,
                          bytes, &routed)) {
    return routed;
  }
  // A host backend has one memory, so both topologies name it and the movement
  // between them is a copy. Answering rather than refusing is what lets a
  // disaggregated program be developed and tested on a laptop: the same source
  // that moves a KV cache between two GPUs runs here, and produces the same
  // tokens, which is how the two-device run gets an oracle to be checked
  // against (#347).
  vx_routing_refuse_handle("a peer handoff", src_device_ptr, src_topology_id);
  (void)src_topology_id;

  // A handoff INTO a NUMA domain places the destination there, for the same reason a staging
  // transfer does: the model named a node and the bytes should end up on it. `transfer(x,
  // Memory::PEER_HBM)` between two domains of one host is the ordinary way to say "move this to
  // the other socket", and a `malloc` here would leave it wherever the allocator felt like.
  //
  // It is also what keeps `vx_plugin_free` decidable. The free is handed the destination's id
  // and nothing else, so every allocation carrying a NUMA-banded id has to come from the same
  // allocator -- otherwise the free reads a header that was never written. That is not a
  // hypothetical: this function returned `malloc` memory while the free saw a banded id, and the
  // result was a SIGABRT inside `vx_plugin_free` on the first two-domain program that ran.
  int dst_node = vx_numa_node_of(dst_topology_id);
  void *dst = nullptr;
  if (dst_node >= 0) {
    int bound = 0;
    dst = vx_numa_alloc(bytes, dst_node, &bound);
    if (dst && bound) {
      vx_numa_registry_record(dst, bytes, dst_node);
    }
    if (dst && vx_verbose()) {
      fprintf(stderr, "[Vx " VX_BACKEND_NAME "] handed %zu bytes to NUMA node %d%s\n", bytes,
              dst_node, bound ? "" : " (bind failed; unplaced)");
    }
  } else {
    dst = malloc(bytes);
  }
  if (dst && src_device_ptr) {
    memcpy(dst, src_device_ptr, bytes);
  }
  return dst;
}

void vx_plugin_release_future(uint64_t future_id) { (void)future_id; }

int32_t vx_plugin_control(uint32_t opcode, void *payload) {
  (void)payload;
  if (opcode == VX_CTRL_GET_DEVICE_COUNT) {
    // The host itself, modelled as one device.
    return 1;
  }
  return 0;
}

} // extern "C"

#endif /* VX_HOST_DISPATCH_COMMON_H */
