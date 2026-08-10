//===- remote_region_test.cpp - Naming remote memory ------------*- C++ -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// Exercises runtime/vx_remote_region.h on any machine, with no GPU and no
// network -- which is the point of the header living outside the backends.
//
// Every failure this checks for is silent. A handle that resolves to the wrong
// region returns plausible numbers from the wrong tensor; a stale one reads
// whatever was allocated next; a handle mistaken for a host pointer sends the
// plugin off to stage memory that is not there. None of them fault, and none of
// them are visible in a dispatch trace, so they have to be caught here.
//
// Driven by tests/integration_test/remote_region_test.rs.
//
//===----------------------------------------------------------------------===//

#include "../../runtime/vx_remote_region.h"

#include <cstdio>
#include <cstring>

namespace {

int failures = 0;

void check(bool ok, const char *what) {
  if (!ok) {
    fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

vx_remote_region storage[64];

vx_remote_table fresh() {
  vx_remote_table t;
  memset(storage, 0, sizeof(storage));
  vx_remote_table_init(&t, storage, sizeof(storage) / sizeof(storage[0]));
  return t;
}

// A minted address must be non-canonical: bits 63:47 neither all-zeros nor
// all-ones. That is what makes it impossible to confuse with a real pointer and
// what makes a stray host dereference fault instead of reading.
void test_addresses_are_non_canonical() {
  uint64_t a = vx_remote_addr(1, 0);
  uint64_t b = vx_remote_addr(255, VX_REMOTE_OFFSET_MASK);
  uint64_t top_a = a >> 47;
  uint64_t top_b = b >> 47;

  check(top_a != 0 && top_a != 0x1FFFF, "worker 1 address is non-canonical");
  check(top_b != 0 && top_b != 0x1FFFF, "worker 255 address is non-canonical");

  check(vx_remote_addr_is_handle(a), "minted address is recognised");
  check(vx_remote_addr_is_handle(b), "top-of-space address is recognised");

  // Things that must never be mistaken for handles.
  int on_stack = 0;
  check(!vx_remote_addr_is_handle(0), "zero is not a handle");
  check(!vx_remote_addr_is_handle((uint64_t)(uintptr_t)&on_stack),
        "a stack address is not a handle");
  check(!vx_remote_addr_is_handle((uint64_t)(uintptr_t)storage),
        "a heap/static address is not a handle");
  check(!vx_remote_addr_is_handle(0xFFFFFFFFFFFFFFFFULL),
        "a kernel-shaped address is not a handle");
  check(!vx_remote_addr_is_handle(0x00007FFFFFFFFFFFULL),
        "the top user address is not a handle");

  check(vx_remote_addr_worker(a) == 1, "worker decodes");
  check(vx_remote_addr_worker(b) == 255, "worker 255 decodes");
}

void test_round_trip() {
  vx_remote_table t = fresh();
  int backing = 7;
  uint64_t base = vx_remote_table_alloc(&t, 1, 4096, 0, &backing);
  vx_remote_ref ref;

  check(base != 0, "allocation succeeds");
  check(vx_remote_resolve(&t, base, &ref), "base resolves");
  check(ref.offset == 0, "base is offset zero");
  check(ref.region->remote == &backing, "region carries its backing pointer");
  check(ref.region->size == 4096, "region carries its size");

  check(vx_remote_resolve(&t, base + 4095, &ref), "last byte resolves");
  check(ref.offset == 4095, "last byte has the right offset");
}

// The bounds check, which is the actual mechanism. An address past a region's
// end must be refused even though the binary search finds that region.
void test_out_of_bounds_is_refused() {
  vx_remote_table t = fresh();
  int backing = 0;
  uint64_t base = vx_remote_table_alloc(&t, 1, 4096, 0, &backing);
  vx_remote_ref ref;

  check(!vx_remote_resolve(&t, base + 4096, &ref), "one past the end misses");
  check(!vx_remote_resolve(&t, base + 100000, &ref), "far past the end misses");
  check(!vx_remote_resolve(&t, base - 1, &ref),
        "before the first region misses");
}

// The case the gap exists for, and the reason it is not merely tidiness.
//
// `vx_advance_ptr(w.wq, l * dim * dim)` with `l` one too large overshoots by
// exactly one region. Were regions adjacent, that address would land on the
// next region's base, pass the bounds check against *that* region, and be
// indistinguishable from a correct handle -- a dispatch against the wrong
// tensor, with no fault anywhere.
void test_overrun_into_the_next_region() {
  vx_remote_table t = fresh();
  int wq = 0, wk = 0;
  const uint64_t size = 1u << 20;

  uint64_t wq_base = vx_remote_table_alloc(&t, 1, size, 0, &wq);
  uint64_t wk_base = vx_remote_table_alloc(&t, 1, size, 0, &wk);
  vx_remote_ref ref;

  check(wq_base != 0 && wk_base != 0, "two regions allocate");
  check(wk_base > wq_base + size, "the second region is not adjacent");

  // Overshooting wq by exactly its own size is the off-by-one.
  check(!vx_remote_resolve(&t, wq_base + size, &ref),
        "overrun by one region does not alias the next");

  // And it must not have quietly become wk either.
  if (vx_remote_resolve(&t, wq_base + size, &ref)) {
    check(ref.region->remote != &wk,
          "overrun must not resolve to the neighbour");
  }

  // wk itself still resolves, so the gap has not broken the good case.
  check(vx_remote_resolve(&t, wk_base, &ref),
        "the second region still resolves");
  check(ref.region->remote == &wk, "and resolves to itself");
}

// A freed region's space is not reused, so a handle held across the free
// resolves to something dead rather than to whatever came next.
void test_use_after_free() {
  vx_remote_table t = fresh();
  int first = 0, second = 0;
  uint64_t base = vx_remote_table_alloc(&t, 1, 4096, 0, &first);
  vx_remote_ref ref;

  check(vx_remote_resolve(&t, base, &ref), "live before free");
  check(vx_remote_table_free(&t, base), "free finds the region");
  check(!vx_remote_resolve(&t, base, &ref), "dead after free");

  uint64_t next = vx_remote_table_alloc(&t, 1, 4096, 0, &second);
  check(next != base, "freed space is not handed out again");
  check(!vx_remote_resolve(&t, base, &ref),
        "the stale handle stays dead after a later allocation");
}

// Two workers minting independently must never resolve into each other. This is
// the reason identity is (worker, address) and not the address alone.
void test_workers_do_not_alias() {
  vx_remote_table t = fresh();
  int a = 0, b = 0;
  uint64_t one = vx_remote_table_alloc(&t, 1, 4096, 0, &a);
  uint64_t two = vx_remote_table_alloc(&t, 2, 4096, 0, &b);
  vx_remote_ref ref;

  check(one != two, "the same offset on two workers is a different address");
  check(vx_remote_addr_offset(one) == vx_remote_addr_offset(two),
        "and they really do share an offset");

  check(vx_remote_resolve(&t, one, &ref) && ref.region->remote == &a,
        "worker 1 resolves to its own buffer");
  check(vx_remote_resolve(&t, two, &ref) && ref.region->remote == &b,
        "worker 2 resolves to its own buffer");
  check(vx_remote_resolve(&t, one, &ref) && ref.region->worker == 1,
        "and carries the worker it belongs to");
}

// Refusals that should not be reached by arithmetic.
void test_refusals() {
  vx_remote_table t = fresh();
  int backing = 0;
  vx_remote_ref ref;

  check(vx_remote_table_alloc(&t, 0, 4096, 0, &backing) == 0,
        "worker 0 is reserved");
  check(vx_remote_table_alloc(&t, 256, 4096, 0, &backing) == 0,
        "worker 256 is out of range");
  check(vx_remote_table_alloc(&t, 1, 0, 0, &backing) == 0,
        "a zero-size region is refused");
  check(vx_remote_table_alloc(&t, 1, VX_REMOTE_OFFSET_MASK, 0, &backing) == 0,
        "a region that would fill the space is refused rather than wrapped");

  check(!vx_remote_resolve(&t, 0, &ref), "the empty table resolves nothing");
}

// A region far larger than 4 GiB, because capping one at 4 GiB was the mistake
// this design already made once: fleet/admit.vx declares a single 120 GiB
// tensor, and llama2.vx's wq blob at 70B scale is 10.7 GB at f16.
void test_large_regions() {
  vx_remote_table t = fresh();
  int wq = 0, wk = 0;
  const uint64_t big = UINT64_C(120) << 30; /* 120 GiB */

  uint64_t base = vx_remote_table_alloc(&t, 1, big, 0, &wq);
  vx_remote_ref ref;

  check(base != 0, "a 120 GiB region allocates");
  check(vx_remote_resolve(&t, base + big - 1, &ref), "its last byte resolves");
  check(ref.offset == big - 1, "at the right offset");
  check(!vx_remote_resolve(&t, base + big, &ref), "and one past it does not");

  uint64_t next = vx_remote_table_alloc(&t, 1, big, 0, &wk);
  check(next >= base + big + big, "the gap after it scales with its size");
}

} // namespace

int main() {
  test_addresses_are_non_canonical();
  test_round_trip();
  test_out_of_bounds_is_refused();
  test_overrun_into_the_next_region();
  test_use_after_free();
  test_workers_do_not_alias();
  test_refusals();
  test_large_regions();

  if (failures) {
    fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  printf("all checks passed\n");
  return 0;
}
