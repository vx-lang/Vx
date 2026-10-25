//===- vx_arrival_counts.h - Counts kept between launches -------*- C++ -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A kernel with code after a nest of blocks counts the passes that ended, in
// device memory that is 0 before the launch. The last pass of each group sets
// its count back to 0, so after a launch the count is 0 again. The program
// keeps the count here and gives it to the next launch of the same `spawn`,
// with no new allocation, copy or free (docs/spawn_on.md section 8).
//
// A kept count goes only to a launch on the same stream. Two streams can run
// at the same time, and two launches that use one count at the same time give
// a wrong answer with no error.
//
// This header only keeps the records. The backend allocates and frees, so
// tests/runtime/arrival_counts_test.cpp checks the records with no GPU.
//
//===----------------------------------------------------------------------===//

#ifndef VX_ARRIVAL_COUNTS_H
#define VX_ARRIVAL_COUNTS_H

#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <tuple>

namespace vx {

/// The counts that no launch uses now, one for each `spawn` site, device and
/// stream.
class ArrivalCounts {
public:
  /// A kept count of at least `bytes`, which the caller then owns. Null if
  /// there is none, and the caller then allocates a zeroed one.
  void *take(const void *site, uint32_t topology, const void *stream,
             size_t bytes) {
    std::lock_guard<std::mutex> guard(mu_);
    auto it = kept_.find(Key{site, topology, stream});
    if (it == kept_.end() || it->second.bytes < bytes)
      return nullptr;
    void *count = it->second.count;
    kept_.erase(it);
    return count;
  }

  /// Keeps `count` for the next launch. Returns a count that the caller must
  /// free, or null. Only the larger of two counts for one key stays.
  void *give(const void *site, uint32_t topology, const void *stream,
             void *count, size_t bytes) {
    std::lock_guard<std::mutex> guard(mu_);
    auto [it, added] =
        kept_.try_emplace(Key{site, topology, stream}, Kept{count, bytes});
    if (added)
      return nullptr;
    if (it->second.bytes >= bytes)
      return count;
    void *smaller = it->second.count;
    it->second = Kept{count, bytes};
    return smaller;
  }

  size_t size() {
    std::lock_guard<std::mutex> guard(mu_);
    return kept_.size();
  }

private:
  using Key = std::tuple<const void *, uint32_t, const void *>;
  struct Kept {
    void *count;
    size_t bytes;
  };
  std::mutex mu_;
  std::map<Key, Kept> kept_;
};

} // namespace vx

#endif /* VX_ARRIVAL_COUNTS_H */
