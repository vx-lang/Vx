//===- reduce.cpp - Vx Compiler ---------------------------------*- C++ -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The sums reduce.vx computes, the ways C++ offers: a plain loop, which must
// add in order, and the standard reductions, which may reorder.
// std::execution::unseq also allows the compiler to vectorize, but only under
// -ffast-math in practice.
//
//===----------------------------------------------------------------------===//

#include "../bench.hpp"

#include <execution>
#include <numeric>
#include <vector>

__attribute__((noinline)) static float loop_sum(const float *a, int m) {
  float s = 0.0f;
  for (int i = 0; i < m; ++i)
    s += a[i];
  return s;
}
__attribute__((noinline)) static float loop_dot(const float *a, const float *b,
                                                int m) {
  float s = 0.0f;
  for (int i = 0; i < m; ++i)
    s += a[i] * b[i];
  return s;
}
__attribute__((noinline)) static float std_reduce(const float *a, int m) {
  return std::reduce(a, a + m, 0.0f);
}
__attribute__((noinline)) static float std_reduce_unseq(const float *a, int m) {
  return std::reduce(std::execution::unseq, a, a + m, 0.0f);
}
__attribute__((noinline)) static float
std_transform_reduce(const float *a, const float *b, int m) {
  return std::transform_reduce(a, a + m, b, 0.0f);
}
__attribute__((noinline)) static float
std_transform_reduce_unseq(const float *a, const float *b, int m) {
  return std::transform_reduce(std::execution::unseq, a, a + m, b, 0.0f);
}

int main() {
  const int n = 1 << 24;
  std::vector<float> a(n), b(n);
  vxbench::escape(a.data());
  vxbench::escape(b.data());
  for (int i = 0; i < n; ++i) {
    a[i] = (float)((i % 1000) + 1) * 0.001f;
    b[i] = (float)(((i * 7) % 1000) + 1) * 0.001f;
  }
  // Each repetition is 64 elements shorter, as in reduce.vx.
  auto m = [&](int rep) { return n - rep * 64; };
  vxbench::measure("sum/loop", 20,
                   [&](int rep) { return loop_sum(a.data(), m(rep)); });
  vxbench::measure("sum/std_reduce", 20,
                   [&](int rep) { return std_reduce(a.data(), m(rep)); });
  vxbench::measure("sum/std_reduce_unseq", 20,
                   [&](int rep) { return std_reduce_unseq(a.data(), m(rep)); });
  vxbench::measure("dot/loop", 20, [&](int rep) {
    return loop_dot(a.data(), b.data(), m(rep));
  });
  vxbench::measure("dot/std_transform_reduce", 20, [&](int rep) {
    return std_transform_reduce(a.data(), b.data(), m(rep));
  });
  vxbench::measure("dot/std_transform_reduce_unseq", 20, [&](int rep) {
    return std_transform_reduce_unseq(a.data(), b.data(), m(rep));
  });
}
