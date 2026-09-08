//===- vx_device_pool.h - Reusing device allocations ------------*- C++ -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A free list for device buffers, so a dispatch does not pay `cudaMalloc` and
// `cudaFree` for operands whose shapes it has already seen.
//
// This is the larger half of the dispatch overhead, measured rather than
// assumed. On an A100, a 2048x2048 f32 GEMM through the Vx path took 11.2 ms,
// of which the cuBLAS call was 1.06 ms; three `cudaMalloc`/`cudaFree` pairs
// were ~5.7 ms of it (51%), and staging from pageable host memory ~4.5 ms
// (40%). The device reached 16,223 GFLOP/s resident and 1,574 through the
// dispatch, and the gap is almost entirely these two things -- not the kernel.
//
// Allocation sizes repeat, which is what makes a cache work here: a program
// dispatches the same shapes every iteration, and llama repeats a layer's
// shapes once per token. So buckets are keyed on the exact byte count. Rounding
// up to a size class would trade memory for hits on shapes that vary slightly,
// and the shapes here do not vary slightly -- they repeat exactly, or they are
// unrelated.
//
// The allocator is a template parameter rather than a CUDA call, so the
// bookkeeping -- which is where the bugs are -- is testable on any machine.
// tests/runtime/device_pool_test.cpp does that; a mistake here is a leak, a
// double free, or handing the same buffer to two live dispatches, and none of
// those announce themselves.
//
//===----------------------------------------------------------------------===//

#ifndef VX_DEVICE_POOL_H
#define VX_DEVICE_POOL_H

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <mutex>
#include <vector>

namespace vx {

/// Free-listed device allocations for one process.
///
/// `Alloc` is `void *(size_t)` returning null on failure; `Free` is
/// `void(void *, size_t)`. Both are only ever called with the pool's lock
/// *released*, because a driver call under a lock held across a dispatch is how
/// a stall becomes a deadlock.
template <typename Alloc, typename Free> class DevicePool {
public:
  DevicePool(Alloc alloc, Free free, size_t capacity_bytes)
      : alloc_(alloc), free_(free), capacity_(capacity_bytes) {}

  ~DevicePool() { drain(); }

  DevicePool(const DevicePool &) = delete;
  DevicePool &operator=(const DevicePool &) = delete;

  /// A buffer of at least `bytes`, from the free list if one is there.
  ///
  /// On allocation failure the whole cache is returned to the driver and the
  /// allocation retried once. Pooling makes OOM *reachable* in programs that
  /// did not have it before -- the cache is holding memory the driver would
  /// otherwise have back -- so the pool has to be able to give it up rather
  /// than fail while sitting on free buffers.
  void *acquire(size_t bytes) {
    if (bytes == 0)
      return nullptr;
    {
      std::lock_guard<std::mutex> guard(mu_);
      auto it = free_lists_.find(bytes);
      if (it != free_lists_.end() && !it->second.empty()) {
        void *p = it->second.back();
        it->second.pop_back();
        pooled_bytes_ -= bytes;
        ++hits_;
        ++live_;
        return p;
      }
      // Counted while the lock is still held. The allocator below runs without
      // it, so the count cannot move down there.
      ++misses_;
    }
    void *p = alloc_(bytes);
    if (!p) {
      drain();
      p = alloc_(bytes);
    }
    if (p) {
      std::lock_guard<std::mutex> guard(mu_);
      ++live_;
    }
    return p;
  }

  /// Give a buffer back. It is kept for reuse while the cache is under its
  /// capacity, and returned to the driver otherwise -- so a program that
  /// allocates a great many distinct shapes does not hoard all of VRAM in
  /// buffers it will never ask for again.
  void release(void *p, size_t bytes) {
    if (!p)
      return;
    bool give_back = false;
    {
      std::lock_guard<std::mutex> guard(mu_);
      if (live_ > 0)
        --live_;
      if (pooled_bytes_ + bytes <= capacity_) {
        free_lists_[bytes].push_back(p);
        pooled_bytes_ += bytes;
      } else {
        give_back = true;
      }
    }
    if (give_back)
      free_(p, bytes);
  }

  /// Return everything cached to the driver. Buffers currently handed out are
  /// untouched: they are not the pool's to free.
  void drain() {
    std::map<size_t, std::vector<void *>> taken;
    {
      std::lock_guard<std::mutex> guard(mu_);
      taken.swap(free_lists_);
      pooled_bytes_ = 0;
    }
    for (auto &entry : taken)
      for (void *p : entry.second)
        free_(p, entry.first);
  }

  struct Stats {
    uint64_t hits = 0;
    uint64_t misses = 0;
    size_t pooled_bytes = 0;
    size_t live = 0; // handed out and not yet released
  };

  Stats stats() {
    std::lock_guard<std::mutex> guard(mu_);
    Stats s;
    s.hits = hits_;
    s.misses = misses_;
    s.pooled_bytes = pooled_bytes_;
    s.live = live_;
    return s;
  }

private:
  Alloc alloc_;
  Free free_;
  size_t capacity_;

  std::mutex mu_;
  std::map<size_t, std::vector<void *>> free_lists_;
  size_t pooled_bytes_ = 0;
  size_t live_ = 0;
  uint64_t hits_ = 0;
  uint64_t misses_ = 0;
};

/// How many bytes the pool may hold, from `VX_CUDA_POOL_MAX_MB`.
///
/// Zero disables caching entirely: `release` then always returns the buffer to
/// the driver, which is the pre-pool behaviour and the switch to flip if the
/// pool is ever suspected of a fault on hardware.
inline size_t vx_pool_capacity_bytes() {
  const char *env = getenv("VX_CUDA_POOL_MAX_MB");
  if (!env)
    return (size_t)2048 * 1024 * 1024;
  char *end = nullptr;
  unsigned long long mb = strtoull(env, &end, 10);
  if (end == env)
    return (size_t)2048 * 1024 * 1024;
  return (size_t)mb * 1024 * 1024;
}

} // namespace vx

#endif // VX_DEVICE_POOL_H
