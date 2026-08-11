//===- device_pool_test.cpp - The device free list --------------*- C++ -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// Exercises runtime/vx_device_pool.h on any machine, with no GPU -- which is
// why the allocator is a template parameter rather than a CUDA call.
//
// Every failure here is silent on hardware. Handing one buffer to two live
// dispatches gives each the other's numbers; losing a buffer on the capacity
// path leaks VRAM a few megabytes at a time until an allocation fails hours
// later somewhere unrelated; freeing a buffer that is still handed out is a
// use-after-free inside the driver. None of them fault at the point of the
// mistake, and none appear in a dispatch trace.
//
// Driven by tests/integration_test/device_pool_test.rs.
//
//===----------------------------------------------------------------------===//

#include "../../runtime/vx_device_pool.h"

#include <cstdio>
#include <cstring>
#include <functional>
#include <set>
#include <thread>
#include <vector>

namespace {

int failures = 0;

void check(bool cond, const char *what) {
  if (!cond) {
    fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

/// A stand-in device allocator that records everything, so the test can assert
/// on the calls the pool made rather than only on what it returned.
struct FakeDevice {
  std::vector<size_t> allocated;
  std::vector<std::pair<void *, size_t>> freed;
  std::set<void *> live;
  size_t next = 0x1000;
  bool out_of_memory = false;

  void *alloc(size_t bytes) {
    if (out_of_memory)
      return nullptr;
    allocated.push_back(bytes);
    void *p = (void *)(next += 0x1000);
    live.insert(p);
    return p;
  }
  void free(void *p, size_t bytes) {
    freed.push_back({p, bytes});
    live.erase(p);
  }
};

using Pool = vx::DevicePool<std::function<void *(size_t)>,
                            std::function<void(void *, size_t)>>;

Pool make_pool(FakeDevice &dev, size_t capacity) {
  return Pool([&dev](size_t n) { return dev.alloc(n); },
              [&dev](void *p, size_t n) { dev.free(p, n); }, capacity);
}

// A released buffer is handed back rather than reallocated. This is the whole
// point: on an A100 three malloc/free pairs were 51% of an 11.2 ms dispatch.
void test_release_then_acquire_reuses() {
  FakeDevice dev;
  Pool pool = make_pool(dev, 1u << 20);

  void *first = pool.acquire(4096);
  pool.release(first, 4096);
  void *second = pool.acquire(4096);

  check(first == second, "the same buffer comes back");
  check(dev.allocated.size() == 1, "only one allocation reached the driver");
  check(dev.freed.empty(), "nothing was returned to the driver");
  check(pool.stats().hits == 1, "the reuse was counted as a hit");
}

// Buffers of a size never seen are not silently satisfied from another bucket.
// Handing back a smaller buffer would be a heap overflow on the first copy.
void test_a_different_size_is_a_different_bucket() {
  FakeDevice dev;
  Pool pool = make_pool(dev, 1u << 20);

  void *small = pool.acquire(1024);
  pool.release(small, 1024);
  void *large = pool.acquire(4096);

  check(small != large, "a larger request did not reuse the smaller buffer");
  check(dev.allocated.size() == 2, "the larger one was allocated");
}

// Two buffers alive at once are distinct. A pool that hands the same pointer to
// both leaves each dispatch reading the other's operands.
void test_two_live_buffers_are_distinct() {
  FakeDevice dev;
  Pool pool = make_pool(dev, 1u << 20);

  void *a = pool.acquire(2048);
  void *b = pool.acquire(2048);
  check(a != b, "concurrent acquisitions are distinct");

  pool.release(a, 2048);
  pool.release(b, 2048);
  void *c = pool.acquire(2048);
  void *d = pool.acquire(2048);
  check(c != d, "both came back and are still distinct");
  check(dev.allocated.size() == 2, "no further allocation was needed");
}

// Past its capacity the pool returns buffers to the driver instead of hoarding
// them. Without this a program with many distinct shapes holds all of VRAM in
// buffers it will never ask for again.
void test_capacity_returns_memory_to_the_driver() {
  FakeDevice dev;
  Pool pool = make_pool(dev, 4096); // room for exactly one

  void *a = pool.acquire(4096);
  void *b = pool.acquire(4096);
  pool.release(a, 4096);
  check(dev.freed.empty(), "the first fits in the cache");
  pool.release(b, 4096);
  check(dev.freed.size() == 1, "the second was returned to the driver");
  check(pool.stats().pooled_bytes == 4096,
        "the cache is at capacity, not over");
}

// A capacity of zero is the kill switch: every release goes straight back to
// the driver, which is the behaviour from before the pool existed.
void test_zero_capacity_caches_nothing() {
  FakeDevice dev;
  Pool pool = make_pool(dev, 0);

  void *a = pool.acquire(4096);
  pool.release(a, 4096);
  check(dev.freed.size() == 1, "released straight through");
  check(pool.acquire(4096) != nullptr, "and allocated again");
  check(dev.allocated.size() == 2, "which reached the driver a second time");
  check(pool.stats().hits == 0, "never a hit");
}

// Pooling makes exhaustion reachable in programs that never saw it, because the
// cache is holding memory the driver would otherwise have. Failing while
// sitting on free buffers is the bug; giving them up and retrying is the fix.
void test_allocation_failure_drains_and_retries() {
  FakeDevice dev;
  Pool pool = make_pool(dev, 1u << 20);

  void *cached = pool.acquire(4096);
  pool.release(cached, 4096); // now in the cache
  check(pool.stats().pooled_bytes == 4096, "one buffer cached");

  // A different size, with the device refusing until the cache is given back.
  dev.out_of_memory = true;
  size_t freed_before = dev.freed.size();
  void *p = pool.acquire(8192);

  check(p == nullptr, "the retry also failed, and that is reported");
  check(dev.freed.size() == freed_before + 1,
        "the cached buffer was returned to the driver before retrying");
  check(pool.stats().pooled_bytes == 0, "the cache is empty after draining");
}

// Draining frees what is cached and leaves what is handed out alone -- those
// belong to a live dispatch, and freeing one is a use-after-free in the driver.
void test_drain_leaves_live_buffers_alone() {
  FakeDevice dev;
  Pool pool = make_pool(dev, 1u << 20);

  void *handed_out = pool.acquire(4096);
  void *returned = pool.acquire(2048);
  pool.release(returned, 2048);

  pool.drain();
  check(dev.freed.size() == 1, "only the cached buffer was freed");
  check(dev.freed[0].first == returned, "and it was the cached one");
  check(dev.live.count(handed_out) == 1, "the live buffer is still allocated");
}

// The pool is reached from whatever thread a dispatch runs on.
void test_concurrent_use_conserves_buffers() {
  FakeDevice dev;
  std::mutex dev_mu;
  Pool pool(
      [&](size_t n) {
        std::lock_guard<std::mutex> g(dev_mu);
        return dev.alloc(n);
      },
      [&](void *p, size_t n) {
        std::lock_guard<std::mutex> g(dev_mu);
        dev.free(p, n);
      },
      1u << 24);

  std::vector<std::thread> threads;
  for (int t = 0; t < 8; ++t) {
    threads.emplace_back([&pool] {
      for (int i = 0; i < 200; ++i) {
        void *p = pool.acquire(4096);
        if (p)
          pool.release(p, 4096);
      }
    });
  }
  for (auto &th : threads)
    th.join();

  Pool::Stats s = pool.stats();
  check(s.live == 0, "every buffer was released");
  check(s.hits + s.misses == 8 * 200, "every acquisition was accounted for");
  check(dev.allocated.size() <= 8,
        "at most one allocation per thread reached the driver");
}

} // namespace

int main() {
  test_release_then_acquire_reuses();
  test_a_different_size_is_a_different_bucket();
  test_two_live_buffers_are_distinct();
  test_capacity_returns_memory_to_the_driver();
  test_zero_capacity_caches_nothing();
  test_allocation_failure_drains_and_retries();
  test_drain_leaves_live_buffers_alone();
  test_concurrent_use_conserves_buffers();

  if (failures) {
    fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  printf("device pool: all checks passed\n");
  return 0;
}
