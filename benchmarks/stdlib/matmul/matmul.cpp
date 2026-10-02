//===- matmul.cpp - Vx Compiler ---------------------------------*- C++ -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// What matmul.vx computes, with the loop order (i, k, j) a compiler turns into
// vector code: each element of `a` times a row of `b`, added to a row of `c`.
//
//===----------------------------------------------------------------------===//

#include "../bench.hpp"

#include <vector>

__attribute__((noinline)) static void ikj(const float *a, const float *b,
                                          float *c, int n) {
  for (int i = 0; i < n * n; ++i)
    c[i] = 0.0f;
  for (int i = 0; i < n; ++i)
    for (int k = 0; k < n; ++k) {
      float x = a[i * n + k];
      for (int j = 0; j < n; ++j)
        c[i * n + j] += x * b[k * n + j];
    }
}

static void run(int n, const char *name) {
  std::vector<float> a(n * n), b(n * n), c(n * n);
  vxbench::escape(a.data());
  vxbench::escape(b.data());
  vxbench::escape(c.data());
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      a[i * n + j] = (float)((i * 31 + j * 17) % 1000) * 0.001f;
      b[i * n + j] = (float)((i * 7 + j * 13) % 1000) * 0.001f;
    }
  // As in matmul.vx, each repetition first changes one element of row 1.
  vxbench::measure(name, 5, [&](int rep) {
    a[n + rep] = 3.0f + rep;
    ikj(a.data(), b.data(), c.data(), n);
    return c[n + rep];
  });
}

int main() {
  run(1024, "matmul_1024/loop_ikj");
  run(1000, "matmul_1000/loop_ikj");
}
