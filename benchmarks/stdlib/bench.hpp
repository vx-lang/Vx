//===- bench.hpp - Vx Compiler ----------------------------------*- C++ -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The C++ side of a `benchmarks/stdlib` comparison. A benchmark reports the
// same two lines its Vx counterpart does, which `cargo vx-bench compare` reads:
//
//   vx-bench <quantity>/<implementation> seconds <fastest repetition>
//   vx-bench <quantity>/<implementation> result <sum of all answers>
//
//===----------------------------------------------------------------------===//

#pragma once

#include <chrono>
#include <cstdio>

namespace vxbench {

// Makes `v` look read and written by code the compiler cannot see, so the work
// that produced it has to finish here, before the clock is read again.
template <class T> inline void keep(T &v) {
  asm volatile("" : "+m"(v) : : "memory");
}

// Makes the memory behind `p` look visible to code the compiler cannot see.
// Without it clang proved the arrays private to `main` and moved read-only
// calls past the clock.
inline void escape(const void *p) { asm volatile("" : : "g"(p) : "memory"); }

// Runs `f(rep)` for `reps` repetitions and reports the fastest one. `f` must
// use `rep` so that each repetition computes something new: a compiler computes
// a repeated pure call once. Every answer is added into the reported result, so
// none can be dropped.
template <class F> void measure(const char *name, int reps, F f) {
  double best = 1e300;
  double total = 0;
  for (int rep = 0; rep < reps; ++rep) {
    auto t0 = std::chrono::steady_clock::now();
    auto r = f(rep);
    keep(r);
    auto t1 = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(t1 - t0).count();
    if (dt < best)
      best = dt;
    total += (double)r;
  }
  std::printf("vx-bench %s seconds %.9g\n", name, best);
  std::printf("vx-bench %s result %.17g\n", name, total);
}

} // namespace vxbench
