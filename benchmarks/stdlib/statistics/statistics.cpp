//===- statistics.cpp - Vx Compiler -----------------------------*- C++ -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// What statistics.vx computes, as plain C++ loops doing the same arithmetic:
// variance in two passes, argmax keeping the first largest and skipping NaN,
// softmax subtracting the largest element first.
//
//===----------------------------------------------------------------------===//

#include "../bench.hpp"

#include <cmath>
#include <vector>

__attribute__((noinline)) static float sum(const float *a, int n) {
  float s = 0.0f;
  for (int i = 0; i < n; ++i)
    s += a[i];
  return s;
}

__attribute__((noinline)) static float variance(const float *a, int n) {
  float total = 0.0f;
  for (int i = 0; i < n; ++i)
    total += a[i];
  float mean = total / (float)n;
  float squares = 0.0f;
  for (int i = 0; i < n; ++i) {
    float d = a[i] - mean;
    squares += d * d;
  }
  return squares / (float)n;
}

__attribute__((noinline)) static int argmax(const float *a, int n) {
  int best = -1;
  float best_value = 0.0f;
  for (int i = 0; i < n; ++i) {
    float x = a[i];
    if (x == x && (best < 0 || x > best_value)) {
      best = i;
      best_value = x;
    }
  }
  return best;
}

__attribute__((noinline)) static void softmax(float *a, int n) {
  float largest = 0.0f;
  for (int i = 0; i < n; ++i)
    if (i == 0 || a[i] > largest)
      largest = a[i];
  float total = 0.0f;
  for (int i = 0; i < n; ++i) {
    float e = std::exp(a[i] - largest);
    a[i] = e;
    total += e;
  }
  for (int i = 0; i < n; ++i)
    a[i] = a[i] / total;
}

int main() {
  const int n = 1 << 22;
  const int soft_n = 1 << 20;
  std::vector<float> a(n), s(soft_n);
  vxbench::escape(a.data());
  vxbench::escape(s.data());
  for (int i = 0; i < n; ++i)
    a[i] = (float)((i * 37) % 1000) * 0.001f;
  // As in statistics.vx, each repetition first changes element `n - 1 - rep`.
  auto touch = [&](int rep) { a[n - 1 - rep] = 2.0f + (float)rep; };
  vxbench::measure("sum/loop", 20, [&](int rep) {
    touch(rep);
    return sum(a.data(), n);
  });
  for (int i = 0; i < n; ++i)
    a[i] = (float)((i * 37) % 1000) * 0.001f;
  vxbench::measure("variance/loop", 20, [&](int rep) {
    touch(rep);
    return variance(a.data(), n);
  });
  for (int i = 0; i < n; ++i)
    a[i] = (float)((i * 37) % 1000) * 0.001f;
  vxbench::measure("argmax/loop", 20, [&](int rep) {
    touch(rep);
    return argmax(a.data(), n);
  });
  for (int i = 0; i < n; ++i)
    a[i] = (float)((i * 37) % 1000) * 0.001f;
  vxbench::measure("softmax/loop", 20, [&](int rep) {
    touch(rep);
    for (int i = 0; i < soft_n; ++i)
      s[i] = a[i];
    softmax(s.data(), soft_n);
    return s[rep];
  });
}
