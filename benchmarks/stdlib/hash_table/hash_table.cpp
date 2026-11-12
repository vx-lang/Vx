//===- hash_table.cpp - Vx Compiler -----------------------------*- C++ -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// What hash_table.vx computes, on one core: the same keys, the same hash and
// the same linear probing as std::bulk_hash, and std::unordered_map with the
// same keys.
//
// std::unordered_map runs only up to 2^24 slots. It keeps each key in its own
// allocation, so at 2^30 slots and load 0.85 it needs more memory than the M4
// has.
//
//===----------------------------------------------------------------------===//

#include "../bench.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <unordered_map>
#include <vector>

constexpr uint32_t kEmpty = 0xFFFFFFFFu;

// The hash of std::bulk_hash: the 32-bit finalizer of MurmurHash3.
static uint32_t fmix32(uint32_t h) {
  h ^= h >> 16;
  h *= 2246822507u;
  h ^= h >> 13;
  h *= 3266489909u;
  h ^= h >> 16;
  return h;
}

static uint32_t key_of(uint32_t x) { return x * 2654435761u; }

// As in hash_table.vx: the lookup of the keys of the table reads them in
// another order than the insert.
constexpr int64_t kLookupStride = 1000003;

static void make_batch(std::vector<uint32_t> &keys, std::vector<uint32_t> &vals,
                       uint32_t first, int64_t stride) {
  const int64_t n = (int64_t)keys.size();
  for (int64_t i = 0; i < n; ++i) {
    uint32_t x = first + 2u * (uint32_t)(i * stride % n);
    keys[i] = key_of(x);
    vals[i] = x;
  }
}

__attribute__((noinline)) static void
insert_all(uint32_t *slots, uint32_t *values, uint32_t mask,
           const uint32_t *keys, const uint32_t *vals, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    uint32_t k = keys[i];
    uint32_t s = fmix32(k) & mask;
    while (slots[s] != kEmpty && slots[s] != k)
      s = (s + 1) & mask;
    slots[s] = k;
    values[s] = vals[i];
  }
}

__attribute__((noinline)) static void
find_all(const uint32_t *slots, const uint32_t *values, uint32_t mask,
         const uint32_t *keys, uint32_t *out, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    uint32_t k = keys[i];
    uint32_t s = fmix32(k) & mask;
    uint32_t v = kEmpty;
    for (;;) {
      uint32_t held = slots[s];
      if (held == kEmpty)
        break;
      if (held == k) {
        v = values[s];
        break;
      }
      s = (s + 1) & mask;
    }
    out[i] = v;
  }
}

// The seconds that `f` takes. The memory clobber of vxbench::escape keeps the
// writes of `f` to `written` between the two clock reads.
template <class F> static double timed(const void *written, F f) {
  vxbench::escape(written);
  auto t0 = std::chrono::steady_clock::now();
  f();
  vxbench::escape(written);
  auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(t1 - t0).count();
}

// The fastest repetition of one quantity, and the sum of its answers.
struct Quantity {
  const char *name;
  double best = 1e300;
  double total = 0;

  void add(double seconds, uint64_t answer) {
    if (seconds < best)
      best = seconds;
    total += (double)answer;
  }
};

static uint64_t sum_of(const std::vector<uint32_t> &v) {
  uint64_t total = 0;
  for (uint32_t x : v)
    total += x;
  return total;
}

static uint64_t count_missing(const std::vector<uint32_t> &v) {
  uint64_t count = 0;
  for (uint32_t x : v)
    count += x == kEmpty;
  return count;
}

static size_t keys_for(int log_slots, int load_percent) {
  return (size_t)((int64_t(1) << log_slots) * load_percent / 100);
}

static void report(const Quantity &q) {
  std::printf("vx-bench %s seconds %.9g\n", q.name, q.best);
  std::printf("vx-bench %s result %.17g\n", q.name, q.total);
}

// As run() in hash_table.vx.
static void linear_probing(int log_slots, int load_percent, int reps,
                           Quantity insert, Quantity present,
                           Quantity missing) {
  const size_t size = size_t(1) << log_slots;
  const uint32_t mask = (uint32_t)(size - 1);
  const size_t n = keys_for(log_slots, load_percent);
  std::vector<uint32_t> slots(size), values(size), keys(n), vals(n);
  for (int rep = 0; rep < reps; ++rep) {
    std::fill(slots.begin(), slots.end(), kEmpty);
    make_batch(keys, vals, 2 * rep + 1, 1);
    double t = timed(slots.data(), [&] {
      insert_all(slots.data(), values.data(), mask, keys.data(), vals.data(),
                 n);
    });
    uint64_t held = 0;
    for (size_t i = 0; i < size; ++i)
      if (slots[i] != kEmpty)
        held += values[i];
    insert.add(t, held);

    make_batch(keys, vals, 2 * rep + 1, kLookupStride);
    t = timed(vals.data(), [&] {
      find_all(slots.data(), values.data(), mask, keys.data(), vals.data(), n);
    });
    present.add(t, sum_of(vals));

    make_batch(keys, vals, 2 * rep + 2, 1);
    t = timed(vals.data(), [&] {
      find_all(slots.data(), values.data(), mask, keys.data(), vals.data(), n);
    });
    missing.add(t, count_missing(vals));
  }
  report(insert);
  report(present);
  report(missing);
}

static void unordered_map(int log_slots, int load_percent, int reps,
                          Quantity insert, Quantity present, Quantity missing) {
  const size_t n = keys_for(log_slots, load_percent);
  std::vector<uint32_t> keys(n), vals(n);
  auto find_each = [&](const std::unordered_map<uint32_t, uint32_t> &map) {
    for (size_t i = 0; i < n; ++i) {
      auto it = map.find(keys[i]);
      vals[i] = it == map.end() ? kEmpty : it->second;
    }
  };
  for (int rep = 0; rep < reps; ++rep) {
    std::unordered_map<uint32_t, uint32_t> map;
    map.reserve(n);
    make_batch(keys, vals, 2 * rep + 1, 1);
    double t = timed(&map, [&] {
      for (size_t i = 0; i < n; ++i)
        map.insert_or_assign(keys[i], vals[i]);
    });
    uint64_t held = 0;
    for (const auto &kv : map)
      held += kv.second;
    insert.add(t, held);

    make_batch(keys, vals, 2 * rep + 1, kLookupStride);
    t = timed(vals.data(), [&] { find_each(map); });
    present.add(t, sum_of(vals));

    make_batch(keys, vals, 2 * rep + 2, 1);
    t = timed(vals.data(), [&] { find_each(map); });
    missing.add(t, count_missing(vals));
  }
  report(insert);
  report(present);
  report(missing);
}

int main() {
  linear_probing(20, 50, 5, {"insert_2^20_slots_load_0.5/linear_probing"},
                 {"find_present_2^20_slots_load_0.5/linear_probing"},
                 {"find_missing_2^20_slots_load_0.5/linear_probing"});
  linear_probing(20, 85, 5, {"insert_2^20_slots_load_0.85/linear_probing"},
                 {"find_present_2^20_slots_load_0.85/linear_probing"},
                 {"find_missing_2^20_slots_load_0.85/linear_probing"});
  linear_probing(24, 50, 5, {"insert_2^24_slots_load_0.5/linear_probing"},
                 {"find_present_2^24_slots_load_0.5/linear_probing"},
                 {"find_missing_2^24_slots_load_0.5/linear_probing"});
  linear_probing(24, 85, 5, {"insert_2^24_slots_load_0.85/linear_probing"},
                 {"find_present_2^24_slots_load_0.85/linear_probing"},
                 {"find_missing_2^24_slots_load_0.85/linear_probing"});
  linear_probing(28, 50, 3, {"insert_2^28_slots_load_0.5/linear_probing"},
                 {"find_present_2^28_slots_load_0.5/linear_probing"},
                 {"find_missing_2^28_slots_load_0.5/linear_probing"});
  linear_probing(28, 85, 3, {"insert_2^28_slots_load_0.85/linear_probing"},
                 {"find_present_2^28_slots_load_0.85/linear_probing"},
                 {"find_missing_2^28_slots_load_0.85/linear_probing"});
  linear_probing(30, 50, 2, {"insert_2^30_slots_load_0.5/linear_probing"},
                 {"find_present_2^30_slots_load_0.5/linear_probing"},
                 {"find_missing_2^30_slots_load_0.5/linear_probing"});
  linear_probing(30, 85, 2, {"insert_2^30_slots_load_0.85/linear_probing"},
                 {"find_present_2^30_slots_load_0.85/linear_probing"},
                 {"find_missing_2^30_slots_load_0.85/linear_probing"});

  unordered_map(20, 50, 5, {"insert_2^20_slots_load_0.5/std_unordered_map"},
                {"find_present_2^20_slots_load_0.5/std_unordered_map"},
                {"find_missing_2^20_slots_load_0.5/std_unordered_map"});
  unordered_map(20, 85, 5, {"insert_2^20_slots_load_0.85/std_unordered_map"},
                {"find_present_2^20_slots_load_0.85/std_unordered_map"},
                {"find_missing_2^20_slots_load_0.85/std_unordered_map"});
  unordered_map(24, 50, 5, {"insert_2^24_slots_load_0.5/std_unordered_map"},
                {"find_present_2^24_slots_load_0.5/std_unordered_map"},
                {"find_missing_2^24_slots_load_0.5/std_unordered_map"});
  unordered_map(24, 85, 5, {"insert_2^24_slots_load_0.85/std_unordered_map"},
                {"find_present_2^24_slots_load_0.85/std_unordered_map"},
                {"find_missing_2^24_slots_load_0.85/std_unordered_map"});
}
