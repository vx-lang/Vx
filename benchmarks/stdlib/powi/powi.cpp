//===- powi.cpp - Vx Compiler -----------------------------------*- C++ -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The sum powi.vx computes, three ways: the squaring loop core::num uses,
// written out; the compiler's builtin; and std::pow.
//
//===----------------------------------------------------------------------===//

#include "../bench.hpp"

#include <cmath>

static inline double same_loop(double x, int n) {
  long long e = n < 0 ? -(long long)n : n;
  double result = 1.0, base = x;
  while (e != 0) {
    if ((e & 1) == 1)
      result = result * base;
    base = base * base;
    e = e >> 1;
  }
  return n < 0 ? 1.0 / result : result;
}

#define KERNEL(NAME, EXPR)                                                     \
  __attribute__((noinline)) static double NAME(int n_items, int seed) {        \
    double acc = 0.0;                                                          \
    for (int i = 0; i < n_items; ++i) {                                        \
      double x = 1.0 + (double)((i + seed) & 1023) * 0.000001;                 \
      int n = (i % 41) - 20;                                                   \
      acc += EXPR;                                                             \
    }                                                                          \
    return acc;                                                                \
  }

KERNEL(k_same_loop, same_loop(x, n))
KERNEL(k_builtin_powi, __builtin_powi(x, n))
KERNEL(k_std_pow, std::pow(x, n))

int main() {
  const int n_items = 100000000;
  vxbench::measure("powi/same_loop", 5,
                   [&](int rep) { return k_same_loop(n_items, rep); });
  vxbench::measure("powi/builtin_powi", 5,
                   [&](int rep) { return k_builtin_powi(n_items, rep); });
  vxbench::measure("powi/std_pow", 5,
                   [&](int rep) { return k_std_pow(n_items, rep); });
}
