//===- math.cpp - Vx Compiler -----------------------------------*- C++ -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The sums math.vx computes, with the <cmath> counterpart of each core::num
// float method, over the same inputs. rem_euclid and div_euclid follow Rust's
// definitions, as core::num does; <cmath> has neither. round_ties_even is
// std::nearbyint in the default rounding mode, next_up and next_down are
// std::nextafter towards an infinity, and log(x, base) is log(x) / log(base),
// which is how core::num computes it.
//
//===----------------------------------------------------------------------===//

#include "../bench.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

template <class T> static inline T rem_euclid(T x, T y) {
  T r = std::fmod(x, y);
  return r < 0 ? r + std::fabs(y) : r;
}

template <class T> static inline T div_euclid(T x, T y) {
  T q = std::trunc(x / y);
  if (std::fmod(x, y) < 0)
    return y > 0 ? q - 1 : q + 1;
  return q;
}

#define UNARY(NAME, TYPE, LO, HI, EXPR)                                        \
  __attribute__((noinline)) static TYPE NAME(int n, int seed) {                \
    using T = TYPE;                                                            \
    T acc = 0;                                                                 \
    const T lo = (T)(LO);                                                      \
    const T step = ((T)(HI) - (T)(LO)) / (T)1024.0;                            \
    for (int i = 0; i < n; ++i) {                                              \
      T x = lo + (T)((i + seed) & 1023) * step;                                \
      acc += EXPR;                                                             \
    }                                                                          \
    return acc;                                                                \
  }

#define BINARY(NAME, TYPE, LO, HI, YLO, YHI, EXPR)                             \
  __attribute__((noinline)) static TYPE NAME(int n, int seed) {                \
    using T = TYPE;                                                            \
    T acc = 0;                                                                 \
    const T lo = (T)(LO);                                                      \
    const T step = ((T)(HI) - (T)(LO)) / (T)1024.0;                            \
    const T ylo = (T)(YLO);                                                    \
    const T ystep = ((T)(YHI) - (T)(YLO)) / (T)1024.0;                         \
    for (int i = 0; i < n; ++i) {                                              \
      T x = lo + (T)((i + seed) & 1023) * step;                                \
      T y = ylo + (T)((i * 7 + seed) & 1023) * ystep;                          \
      acc += EXPR;                                                             \
    }                                                                          \
    return acc;                                                                \
  }

UNARY(sqrt_f32, float, 0.5, 1000.0, std::sqrt(x))
UNARY(abs_f32, float, -100.0, 100.0, std::fabs(x))
UNARY(exp_f32, float, -10.0, 10.0, std::exp(x))
UNARY(exp2_f32, float, -10.0, 10.0, std::exp2(x))
UNARY(exp_m1_f32, float, -1.0, 1.0, std::expm1(x))
UNARY(ln_f32, float, 0.5, 1000.0, std::log(x))
UNARY(log2_f32, float, 0.5, 1000.0, std::log2(x))
UNARY(log10_f32, float, 0.5, 1000.0, std::log10(x))
UNARY(ln_1p_f32, float, -0.5, 10.0, std::log1p(x))
UNARY(sin_f32, float, -3.0, 3.0, std::sin(x))
UNARY(cos_f32, float, -3.0, 3.0, std::cos(x))
UNARY(tan_f32, float, -1.5, 1.5, std::tan(x))
UNARY(asin_f32, float, -0.99, 0.99, std::asin(x))
UNARY(acos_f32, float, -0.99, 0.99, std::acos(x))
UNARY(atan_f32, float, -100.0, 100.0, std::atan(x))
UNARY(sinh_f32, float, -5.0, 5.0, std::sinh(x))
UNARY(cosh_f32, float, -5.0, 5.0, std::cosh(x))
UNARY(tanh_f32, float, -5.0, 5.0, std::tanh(x))
UNARY(floor_f32, float, -100.0, 100.0, std::floor(x))
UNARY(ceil_f32, float, -100.0, 100.0, std::ceil(x))
UNARY(round_f32, float, -100.0, 100.0, std::round(x))
UNARY(trunc_f32, float, -100.0, 100.0, std::trunc(x))
UNARY(fract_f32, float, -100.0, 100.0, x - std::trunc(x))
UNARY(recip_f32, float, 0.5, 100.0, (T)1 / x)
UNARY(signum_f32, float, -100.0, 100.0,
      std::isnan(x) ? x : std::copysign((T)1, x))
UNARY(to_degrees_f32, float, -3.0, 3.0,
      x *((T)180.0 / (T)3.14159265358979323846))
UNARY(to_radians_f32, float, -180.0, 180.0,
      x *((T)3.14159265358979323846 / (T)180.0))
BINARY(powf_f32, float, 0.5, 2.0, -4.0, 4.0, std::pow(x, y))
BINARY(atan2_f32, float, -10.0, 10.0, -10.0, 10.0, std::atan2(x, y))
BINARY(copysign_f32, float, -100.0, 100.0, -1.0, 1.0, std::copysign(x, y))
BINARY(hypot_f32, float, -100.0, 100.0, -100.0, 100.0, std::hypot(x, y))
UNARY(asinh_f32, float, -10.0, 10.0, std::asinh(x))
UNARY(acosh_f32, float, 1.0, 100.0, std::acosh(x))
UNARY(atanh_f32, float, -0.99, 0.99, std::atanh(x))
UNARY(round_ties_even_f32, float, -100.0, 100.0, std::nearbyint(x))
UNARY(next_up_f32, float, -100.0, 100.0, std::nextafter(x, (T)INFINITY))
UNARY(next_down_f32, float, -100.0, 100.0, std::nextafter(x, -(T)INFINITY))
BINARY(log_f32, float, 0.5, 1000.0, 2.0, 10.0, std::log(x) / std::log(y))
BINARY(midpoint_f32, float, -100.0, 100.0, -100.0, 100.0, std::midpoint(x, y))
BINARY(rem_euclid_f32, float, -100.0, 100.0, 0.5, 10.0, rem_euclid(x, y))
BINARY(div_euclid_f32, float, -100.0, 100.0, 0.5, 10.0, div_euclid(x, y))
BINARY(max_f32, float, -10.0, 10.0, -10.0, 10.0, std::fmax(x, y))
BINARY(min_f32, float, -10.0, 10.0, -10.0, 10.0, std::fmin(x, y))
BINARY(mul_add_f32, float, -10.0, 10.0, -10.0, 10.0, std::fma(x, y, x))
BINARY(clamp_f32, float, -10.0, 10.0, 0.0, 1.0, std::clamp(x, (T)-1, (T)1))
UNARY(sqrt_f64, double, 0.5, 1000.0, std::sqrt(x))
UNARY(abs_f64, double, -100.0, 100.0, std::fabs(x))
UNARY(exp_f64, double, -10.0, 10.0, std::exp(x))
UNARY(exp2_f64, double, -10.0, 10.0, std::exp2(x))
UNARY(exp_m1_f64, double, -1.0, 1.0, std::expm1(x))
UNARY(ln_f64, double, 0.5, 1000.0, std::log(x))
UNARY(log2_f64, double, 0.5, 1000.0, std::log2(x))
UNARY(log10_f64, double, 0.5, 1000.0, std::log10(x))
UNARY(ln_1p_f64, double, -0.5, 10.0, std::log1p(x))
UNARY(sin_f64, double, -3.0, 3.0, std::sin(x))
UNARY(cos_f64, double, -3.0, 3.0, std::cos(x))
UNARY(tan_f64, double, -1.5, 1.5, std::tan(x))
UNARY(asin_f64, double, -0.99, 0.99, std::asin(x))
UNARY(acos_f64, double, -0.99, 0.99, std::acos(x))
UNARY(atan_f64, double, -100.0, 100.0, std::atan(x))
UNARY(sinh_f64, double, -5.0, 5.0, std::sinh(x))
UNARY(cosh_f64, double, -5.0, 5.0, std::cosh(x))
UNARY(tanh_f64, double, -5.0, 5.0, std::tanh(x))
UNARY(floor_f64, double, -100.0, 100.0, std::floor(x))
UNARY(ceil_f64, double, -100.0, 100.0, std::ceil(x))
UNARY(round_f64, double, -100.0, 100.0, std::round(x))
UNARY(trunc_f64, double, -100.0, 100.0, std::trunc(x))
UNARY(fract_f64, double, -100.0, 100.0, x - std::trunc(x))
UNARY(recip_f64, double, 0.5, 100.0, (T)1 / x)
UNARY(signum_f64, double, -100.0, 100.0,
      std::isnan(x) ? x : std::copysign((T)1, x))
UNARY(to_degrees_f64, double, -3.0, 3.0,
      x *((T)180.0 / (T)3.14159265358979323846))
UNARY(to_radians_f64, double, -180.0, 180.0,
      x *((T)3.14159265358979323846 / (T)180.0))
BINARY(powf_f64, double, 0.5, 2.0, -4.0, 4.0, std::pow(x, y))
BINARY(atan2_f64, double, -10.0, 10.0, -10.0, 10.0, std::atan2(x, y))
BINARY(copysign_f64, double, -100.0, 100.0, -1.0, 1.0, std::copysign(x, y))
BINARY(hypot_f64, double, -100.0, 100.0, -100.0, 100.0, std::hypot(x, y))
UNARY(asinh_f64, double, -10.0, 10.0, std::asinh(x))
UNARY(acosh_f64, double, 1.0, 100.0, std::acosh(x))
UNARY(atanh_f64, double, -0.99, 0.99, std::atanh(x))
UNARY(round_ties_even_f64, double, -100.0, 100.0, std::nearbyint(x))
UNARY(next_up_f64, double, -100.0, 100.0, std::nextafter(x, (T)INFINITY))
UNARY(next_down_f64, double, -100.0, 100.0, std::nextafter(x, -(T)INFINITY))
BINARY(log_f64, double, 0.5, 1000.0, 2.0, 10.0, std::log(x) / std::log(y))
BINARY(midpoint_f64, double, -100.0, 100.0, -100.0, 100.0, std::midpoint(x, y))
BINARY(rem_euclid_f64, double, -100.0, 100.0, 0.5, 10.0, rem_euclid(x, y))
BINARY(div_euclid_f64, double, -100.0, 100.0, 0.5, 10.0, div_euclid(x, y))
BINARY(max_f64, double, -10.0, 10.0, -10.0, 10.0, std::fmax(x, y))
BINARY(min_f64, double, -10.0, 10.0, -10.0, 10.0, std::fmin(x, y))
BINARY(mul_add_f64, double, -10.0, 10.0, -10.0, 10.0, std::fma(x, y, x))
BINARY(clamp_f64, double, -10.0, 10.0, 0.0, 1.0, std::clamp(x, (T)-1, (T)1))

int main() {
  vxbench::measure("sqrt_f32/std", 5,
                   [](int rep) { return sqrt_f32(1000000, rep); });
  vxbench::measure("abs_f32/std", 5,
                   [](int rep) { return abs_f32(1000000, rep); });
  vxbench::measure("exp_f32/std", 5,
                   [](int rep) { return exp_f32(1000000, rep); });
  vxbench::measure("exp2_f32/std", 5,
                   [](int rep) { return exp2_f32(1000000, rep); });
  vxbench::measure("exp_m1_f32/std", 5,
                   [](int rep) { return exp_m1_f32(1000000, rep); });
  vxbench::measure("ln_f32/std", 5,
                   [](int rep) { return ln_f32(1000000, rep); });
  vxbench::measure("log2_f32/std", 5,
                   [](int rep) { return log2_f32(1000000, rep); });
  vxbench::measure("log10_f32/std", 5,
                   [](int rep) { return log10_f32(1000000, rep); });
  vxbench::measure("ln_1p_f32/std", 5,
                   [](int rep) { return ln_1p_f32(1000000, rep); });
  vxbench::measure("sin_f32/std", 5,
                   [](int rep) { return sin_f32(1000000, rep); });
  vxbench::measure("cos_f32/std", 5,
                   [](int rep) { return cos_f32(1000000, rep); });
  vxbench::measure("tan_f32/std", 5,
                   [](int rep) { return tan_f32(1000000, rep); });
  vxbench::measure("asin_f32/std", 5,
                   [](int rep) { return asin_f32(1000000, rep); });
  vxbench::measure("acos_f32/std", 5,
                   [](int rep) { return acos_f32(1000000, rep); });
  vxbench::measure("atan_f32/std", 5,
                   [](int rep) { return atan_f32(1000000, rep); });
  vxbench::measure("sinh_f32/std", 5,
                   [](int rep) { return sinh_f32(1000000, rep); });
  vxbench::measure("cosh_f32/std", 5,
                   [](int rep) { return cosh_f32(1000000, rep); });
  vxbench::measure("tanh_f32/std", 5,
                   [](int rep) { return tanh_f32(1000000, rep); });
  vxbench::measure("floor_f32/std", 5,
                   [](int rep) { return floor_f32(1000000, rep); });
  vxbench::measure("ceil_f32/std", 5,
                   [](int rep) { return ceil_f32(1000000, rep); });
  vxbench::measure("round_f32/std", 5,
                   [](int rep) { return round_f32(1000000, rep); });
  vxbench::measure("trunc_f32/std", 5,
                   [](int rep) { return trunc_f32(1000000, rep); });
  vxbench::measure("fract_f32/std", 5,
                   [](int rep) { return fract_f32(1000000, rep); });
  vxbench::measure("recip_f32/std", 5,
                   [](int rep) { return recip_f32(1000000, rep); });
  vxbench::measure("signum_f32/std", 5,
                   [](int rep) { return signum_f32(1000000, rep); });
  vxbench::measure("to_degrees_f32/std", 5,
                   [](int rep) { return to_degrees_f32(1000000, rep); });
  vxbench::measure("to_radians_f32/std", 5,
                   [](int rep) { return to_radians_f32(1000000, rep); });
  vxbench::measure("powf_f32/std", 5,
                   [](int rep) { return powf_f32(1000000, rep); });
  vxbench::measure("atan2_f32/std", 5,
                   [](int rep) { return atan2_f32(1000000, rep); });
  vxbench::measure("copysign_f32/std", 5,
                   [](int rep) { return copysign_f32(1000000, rep); });
  vxbench::measure("hypot_f32/std", 5,
                   [](int rep) { return hypot_f32(1000000, rep); });
  vxbench::measure("rem_euclid_f32/std", 5,
                   [](int rep) { return rem_euclid_f32(1000000, rep); });
  vxbench::measure("div_euclid_f32/std", 5,
                   [](int rep) { return div_euclid_f32(1000000, rep); });
  vxbench::measure("max_f32/std", 5,
                   [](int rep) { return max_f32(1000000, rep); });
  vxbench::measure("min_f32/std", 5,
                   [](int rep) { return min_f32(1000000, rep); });
  vxbench::measure("mul_add_f32/std", 5,
                   [](int rep) { return mul_add_f32(1000000, rep); });
  vxbench::measure("clamp_f32/std", 5,
                   [](int rep) { return clamp_f32(1000000, rep); });
  vxbench::measure("asinh_f32/std", 5,
                   [](int rep) { return asinh_f32(1000000, rep); });
  vxbench::measure("acosh_f32/std", 5,
                   [](int rep) { return acosh_f32(1000000, rep); });
  vxbench::measure("atanh_f32/std", 5,
                   [](int rep) { return atanh_f32(1000000, rep); });
  vxbench::measure("round_ties_even_f32/std", 5,
                   [](int rep) { return round_ties_even_f32(1000000, rep); });
  vxbench::measure("next_up_f32/std", 5,
                   [](int rep) { return next_up_f32(1000000, rep); });
  vxbench::measure("next_down_f32/std", 5,
                   [](int rep) { return next_down_f32(1000000, rep); });
  vxbench::measure("log_f32/std", 5,
                   [](int rep) { return log_f32(1000000, rep); });
  vxbench::measure("midpoint_f32/std", 5,
                   [](int rep) { return midpoint_f32(1000000, rep); });
  vxbench::measure("sqrt_f64/std", 5,
                   [](int rep) { return sqrt_f64(1000000, rep); });
  vxbench::measure("abs_f64/std", 5,
                   [](int rep) { return abs_f64(1000000, rep); });
  vxbench::measure("exp_f64/std", 5,
                   [](int rep) { return exp_f64(1000000, rep); });
  vxbench::measure("exp2_f64/std", 5,
                   [](int rep) { return exp2_f64(1000000, rep); });
  vxbench::measure("exp_m1_f64/std", 5,
                   [](int rep) { return exp_m1_f64(1000000, rep); });
  vxbench::measure("ln_f64/std", 5,
                   [](int rep) { return ln_f64(1000000, rep); });
  vxbench::measure("log2_f64/std", 5,
                   [](int rep) { return log2_f64(1000000, rep); });
  vxbench::measure("log10_f64/std", 5,
                   [](int rep) { return log10_f64(1000000, rep); });
  vxbench::measure("ln_1p_f64/std", 5,
                   [](int rep) { return ln_1p_f64(1000000, rep); });
  vxbench::measure("sin_f64/std", 5,
                   [](int rep) { return sin_f64(1000000, rep); });
  vxbench::measure("cos_f64/std", 5,
                   [](int rep) { return cos_f64(1000000, rep); });
  vxbench::measure("tan_f64/std", 5,
                   [](int rep) { return tan_f64(1000000, rep); });
  vxbench::measure("asin_f64/std", 5,
                   [](int rep) { return asin_f64(1000000, rep); });
  vxbench::measure("acos_f64/std", 5,
                   [](int rep) { return acos_f64(1000000, rep); });
  vxbench::measure("atan_f64/std", 5,
                   [](int rep) { return atan_f64(1000000, rep); });
  vxbench::measure("sinh_f64/std", 5,
                   [](int rep) { return sinh_f64(1000000, rep); });
  vxbench::measure("cosh_f64/std", 5,
                   [](int rep) { return cosh_f64(1000000, rep); });
  vxbench::measure("tanh_f64/std", 5,
                   [](int rep) { return tanh_f64(1000000, rep); });
  vxbench::measure("floor_f64/std", 5,
                   [](int rep) { return floor_f64(1000000, rep); });
  vxbench::measure("ceil_f64/std", 5,
                   [](int rep) { return ceil_f64(1000000, rep); });
  vxbench::measure("round_f64/std", 5,
                   [](int rep) { return round_f64(1000000, rep); });
  vxbench::measure("trunc_f64/std", 5,
                   [](int rep) { return trunc_f64(1000000, rep); });
  vxbench::measure("fract_f64/std", 5,
                   [](int rep) { return fract_f64(1000000, rep); });
  vxbench::measure("recip_f64/std", 5,
                   [](int rep) { return recip_f64(1000000, rep); });
  vxbench::measure("signum_f64/std", 5,
                   [](int rep) { return signum_f64(1000000, rep); });
  vxbench::measure("to_degrees_f64/std", 5,
                   [](int rep) { return to_degrees_f64(1000000, rep); });
  vxbench::measure("to_radians_f64/std", 5,
                   [](int rep) { return to_radians_f64(1000000, rep); });
  vxbench::measure("powf_f64/std", 5,
                   [](int rep) { return powf_f64(1000000, rep); });
  vxbench::measure("atan2_f64/std", 5,
                   [](int rep) { return atan2_f64(1000000, rep); });
  vxbench::measure("copysign_f64/std", 5,
                   [](int rep) { return copysign_f64(1000000, rep); });
  vxbench::measure("hypot_f64/std", 5,
                   [](int rep) { return hypot_f64(1000000, rep); });
  vxbench::measure("rem_euclid_f64/std", 5,
                   [](int rep) { return rem_euclid_f64(1000000, rep); });
  vxbench::measure("div_euclid_f64/std", 5,
                   [](int rep) { return div_euclid_f64(1000000, rep); });
  vxbench::measure("max_f64/std", 5,
                   [](int rep) { return max_f64(1000000, rep); });
  vxbench::measure("min_f64/std", 5,
                   [](int rep) { return min_f64(1000000, rep); });
  vxbench::measure("mul_add_f64/std", 5,
                   [](int rep) { return mul_add_f64(1000000, rep); });
  vxbench::measure("clamp_f64/std", 5,
                   [](int rep) { return clamp_f64(1000000, rep); });
  vxbench::measure("asinh_f64/std", 5,
                   [](int rep) { return asinh_f64(1000000, rep); });
  vxbench::measure("acosh_f64/std", 5,
                   [](int rep) { return acosh_f64(1000000, rep); });
  vxbench::measure("atanh_f64/std", 5,
                   [](int rep) { return atanh_f64(1000000, rep); });
  vxbench::measure("round_ties_even_f64/std", 5,
                   [](int rep) { return round_ties_even_f64(1000000, rep); });
  vxbench::measure("next_up_f64/std", 5,
                   [](int rep) { return next_up_f64(1000000, rep); });
  vxbench::measure("next_down_f64/std", 5,
                   [](int rep) { return next_down_f64(1000000, rep); });
  vxbench::measure("log_f64/std", 5,
                   [](int rep) { return log_f64(1000000, rep); });
  vxbench::measure("midpoint_f64/std", 5,
                   [](int rep) { return midpoint_f64(1000000, rep); });
}
