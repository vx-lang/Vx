//===- rank2.cpp - Vx Compiler ----------------------------------*- C++ -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// What rank2.vx computes, as plain C++ loops over a row-major 2048-by-2048
// array: the sum and the largest element, and the sums along each axis.
//
//===----------------------------------------------------------------------===//

#include "../bench.hpp"

#include <cmath>
#include <vector>

__attribute__((noinline)) static float sum(const float *a, int n) {
  float s = 0.0f;
  for (int i = 0; i < n * n; ++i)
    s += a[i];
  return s;
}

__attribute__((noinline)) static float largest(const float *a, int n) {
  float m = -INFINITY;
  for (int i = 0; i < n * n; ++i)
    m = std::fmax(m, a[i]);
  return m;
}

__attribute__((noinline)) static void columns(const float *a, int n,
                                              float *out) {
  for (int j = 0; j < n; ++j)
    out[j] = 0.0f;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      out[j] += a[i * n + j];
}

__attribute__((noinline)) static void rows(const float *a, int n, float *out) {
  for (int i = 0; i < n; ++i) {
    float s = 0.0f;
    for (int j = 0; j < n; ++j)
      s += a[i * n + j];
    out[i] = s;
  }
}

int main() {
  const int n = 2048;
  std::vector<float> a(n * n), out(n);
  vxbench::escape(a.data());
  vxbench::escape(out.data());
  auto fill = [&] {
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j)
        a[i * n + j] = (float)((i * 31 + j * 17) % 1000) * 0.001f;
  };
  // As in rank2.vx, each repetition first changes one element of the last row.
  auto touch = [&](int rep) { a[(n - 1) * n + (n - 1 - rep)] = 2.0f + rep; };
  fill();
  vxbench::measure("sum/loop", 20, [&](int rep) {
    touch(rep);
    return sum(a.data(), n);
  });
  fill();
  vxbench::measure("max/loop", 20, [&](int rep) {
    touch(rep);
    return largest(a.data(), n);
  });
  fill();
  vxbench::measure("sum_axis_0/loop", 20, [&](int rep) {
    touch(rep);
    columns(a.data(), n, out.data());
    return out[n - 1 - rep];
  });
  fill();
  vxbench::measure("sum_axis_1/loop", 20, [&](int rep) {
    touch(rep);
    rows(a.data(), n, out.data());
    return out[n - 1];
  });
}
