//===- arrival_counts_test.cpp - Counts kept between launches ---*- C++ -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Checks runtime/vx_arrival_counts.h with no GPU. A count that goes to two
// launches at the same time, or a count that is too small, gives a wrong
// answer with no error.
//
// Driven by tests/integration_test/arrival_counts_test.rs.
//
//===----------------------------------------------------------------------===//

#include "../../runtime/vx_arrival_counts.h"

#include <cstdio>
#include <mutex>
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

// Stand-ins for a spawn site, a stream and a count. Only the addresses matter.
char site_a, site_b, stream_a, stream_b, count_a, count_b;

void test_a_count_given_back_goes_to_the_next_take() {
  vx::ArrivalCounts counts;
  check(!counts.take(&site_a, 1, &stream_a, 16), "nothing is kept at first");
  check(!counts.give(&site_a, 1, &stream_a, &count_a, 16),
        "the first count given back is kept");
  check(counts.take(&site_a, 1, &stream_a, 16) == &count_a,
        "the next take gets the kept count");
  check(!counts.take(&site_a, 1, &stream_a, 16),
        "a kept count goes to one take only");
}

void test_a_count_goes_only_to_its_own_site_device_and_stream() {
  vx::ArrivalCounts counts;
  counts.give(&site_a, 1, &stream_a, &count_a, 16);
  check(!counts.take(&site_b, 1, &stream_a, 16), "another site gets nothing");
  check(!counts.take(&site_a, 2, &stream_a, 16), "another device gets nothing");
  check(!counts.take(&site_a, 1, &stream_b, 16), "another stream gets nothing");
  check(!counts.take(&site_a, 1, nullptr, 16),
        "the default stream is another stream");
  check(counts.take(&site_a, 1, &stream_a, 16) == &count_a,
        "the count is still there for its own key");
}

void test_a_count_that_is_too_small_stays_kept() {
  vx::ArrivalCounts counts;
  counts.give(&site_a, 1, &stream_a, &count_a, 8);
  check(!counts.take(&site_a, 1, &stream_a, 16),
        "a take of more bytes gets nothing");
  check(counts.size() == 1, "the small count stays kept");
  check(counts.take(&site_a, 1, &stream_a, 4) == &count_a,
        "a take of fewer bytes gets the count");
}

void test_only_the_larger_of_two_counts_stays() {
  vx::ArrivalCounts counts;
  counts.give(&site_a, 1, &stream_a, &count_a, 8);
  check(counts.give(&site_a, 1, &stream_a, &count_b, 16) == &count_a,
        "a larger count replaces the kept one, which the caller frees");
  check(counts.give(&site_a, 1, &stream_a, &count_a, 8) == &count_a,
        "a smaller count is not kept, and the caller frees it");
  check(counts.size() == 1, "one count is kept for one key");
  check(counts.take(&site_a, 1, &stream_a, 16) == &count_b,
        "the kept count is the larger one");
}

// Threads take and give on one key. No count may go to two threads at once.
void test_no_count_goes_to_two_threads_at_once() {
  vx::ArrivalCounts counts;
  std::mutex mu;
  std::set<void *> in_use;
  bool shared = false;
  std::vector<std::thread> threads;
  for (int t = 0; t < 8; ++t)
    threads.emplace_back([&] {
      for (int i = 0; i < 2000; ++i) {
        void *count = counts.take(&site_a, 1, &stream_a, 4);
        if (!count)
          count = new char[4];
        {
          std::lock_guard<std::mutex> guard(mu);
          if (!in_use.insert(count).second)
            shared = true;
        }
        {
          std::lock_guard<std::mutex> guard(mu);
          in_use.erase(count);
        }
        delete[] static_cast<char *>(
            counts.give(&site_a, 1, &stream_a, count, 4));
      }
    });
  for (auto &th : threads)
    th.join();
  check(!shared, "no count went to two threads at once");
  check(counts.size() == 1, "one count is kept at the end");
  delete[] static_cast<char *>(counts.take(&site_a, 1, &stream_a, 4));
}

} // namespace

int main() {
  test_a_count_given_back_goes_to_the_next_take();
  test_a_count_goes_only_to_its_own_site_device_and_stream();
  test_a_count_that_is_too_small_stays_kept();
  test_only_the_larger_of_two_counts_stays();
  test_no_count_goes_to_two_threads_at_once();

  if (failures) {
    fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  printf("arrival counts: all checks passed\n");
  return 0;
}
