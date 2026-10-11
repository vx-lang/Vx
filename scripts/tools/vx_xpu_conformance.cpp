//===- vx_xpu_conformance.cpp - one Vx kernel on two backends --*- C++ -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// One kernel, two backends, the same numbers expected.
//
// The Vx programs under scripts/tools/xpu_conformance/ each hold several
// placed kernels. `vxc prog.vx --emit-llvm` carries each kernel's dispatch
// payload, and the same LLVM module holds the host kernel (`_mlir_ciface_...`)
// the CPU backend calls. This program is linked against one dispatch backend at
// a time and drives the vx_plugin_* ABI the way a compiled Vx program does:
// allocate, dispatch, read back. Run it twice -- once linked against
// libvx_x86_dispatch (the CPU backend), once against libvx_xpu_dispatch (the
// Intel Arc) -- and compare the two result files.
//
// Nothing here writes a launch by hand. The argument descriptors, the tag
// layout and the payload all come from the compiler through the ABI, so a
// backend that reads them differently is what the comparison notices.
//
// Usage:
//
//   vx_xpu_conformance run <op> <dtype> <M> <N> <K> <reps> <payload>
//                        <kernel> <out.bin>
//   vx_xpu_conformance compare <dtype> <count> <tolerance> <a.bin> <b.bin>
//
// `op` is one of ew, reduce, transpose, matmul; `dtype` is one of f32, f16,
// i8, u8, i16, u16, i32, u32, i64, u64, bool.
//
// `run` dispatches the payload `reps` times and writes the result to out.bin;
// it also prints the best dispatch time and, when the work is small enough,
// how far the result is from an independent reference.
//
// `compare` reads two result files and prints how many elements differ beyond
// `tolerance`. A tolerance of 0 means "must be equal", which is the rule for
// every kernel whose values are exact integers in its element type.
//
// Set VX_CONFORMANCE_PERTURB=1 to change one input element before dispatching.
// That is how the check in `compare` is shown able to fail.
//
// Memory: the footprint is the tensors the kernel's arguments describe, checked
// against the shared ceiling before anything is allocated (vram_ceiling.h).
//
//===----------------------------------------------------------------------===//

#include "vram_ceiling.h"

#include <vx_hardware_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

// The GPU band's topology id (src/arch.rs): what a `transfer(t,
// Memory::GPU_HBM)` carries, and the id both backends decode.
static constexpr uint32_t kGpuTopology = 500;

static constexpr int kWarmup = 3;

// ------------------------------------------------------------- element types

struct DType {
  const char *name;
  int32_t code;
  size_t bytes;
  bool is_float;
};

static const DType kDtypes[] = {
    {"f32", VX_DTYPE_F32, 4, true},  {"f16", VX_DTYPE_F16, 2, true},
    {"i8", VX_DTYPE_I8, 1, false},   {"u8", VX_DTYPE_I8, 1, false},
    {"i16", VX_DTYPE_I16, 2, false}, {"u16", VX_DTYPE_I16, 2, false},
    {"i32", VX_DTYPE_I32, 4, false}, {"u32", VX_DTYPE_I32, 4, false},
    {"i64", VX_DTYPE_I64, 8, false}, {"u64", VX_DTYPE_I64, 8, false},
    {"bool", VX_DTYPE_I1, 1, false},
};

static const DType &dtypeOf(const char *name) {
  for (const DType &d : kDtypes)
    if (std::strcmp(d.name, name) == 0)
      return d;
  std::fprintf(stderr, "unknown element type '%s'\n", name);
  std::exit(2);
}

/// fp32 to the nearest fp16, by round-to-nearest-even. Written out rather than
/// taken from a compiler builtin so the fill is the same on any host.
static uint16_t halfOf(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, 4);
  const uint32_t sign = (bits >> 16) & 0x8000;
  int32_t exp = (int32_t)((bits >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = bits & 0x7FFFFF;
  if (exp <= 0) {
    // Subnormal or zero. Shift the implicit bit in and round once.
    if (exp < -10)
      return (uint16_t)sign;
    mant |= 0x800000;
    const uint32_t shift = (uint32_t)(14 - exp);
    uint32_t half = mant >> shift;
    if ((mant >> (shift - 1)) & 1)
      half += 1;
    return (uint16_t)(sign | half);
  }
  if (exp >= 31) {
    // Overflow to infinity or NaN. A NaN keeps a non-zero mantissa.
    if (((bits >> 23) & 0xFF) == 0xFF && mant)
      return (uint16_t)(sign | 0x7C00 | (mant >> 13) | (mant ? 1 : 0));
    return (uint16_t)(sign | 0x7C00);
  }
  uint32_t half = (uint32_t)(sign | (exp << 10) | (mant >> 13));
  if ((mant >> 12) & 1)
    half += 1;
  return (uint16_t)half;
}

static float floatOfHalf(uint16_t h) {
  const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      // Subnormal: normalise it.
      int shift = 0;
      while (!(mant & 0x400)) {
        mant <<= 1;
        ++shift;
      }
      mant &= 0x3FF;
      bits = sign | ((uint32_t)(127 - 15 - shift) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7F800000 | (mant << 13);
  } else {
    bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

static void storeElem(std::vector<char> &raw, size_t index, const DType &dt,
                      double value) {
  char *p = raw.data() + index * dt.bytes;
  switch (dt.code) {
  case VX_DTYPE_F32: {
    float f = (float)value;
    std::memcpy(p, &f, 4);
    break;
  }
  case VX_DTYPE_F16: {
    uint16_t h = halfOf((float)value);
    std::memcpy(p, &h, 2);
    break;
  }
  case VX_DTYPE_I1:
    *p = value != 0.0 ? 1 : 0;
    break;
  case VX_DTYPE_I8: {
    int8_t v = (int8_t)value;
    std::memcpy(p, &v, 1);
    break;
  }
  case VX_DTYPE_I16: {
    int16_t v = (int16_t)value;
    std::memcpy(p, &v, 2);
    break;
  }
  case VX_DTYPE_I32: {
    int32_t v = (int32_t)value;
    std::memcpy(p, &v, 4);
    break;
  }
  case VX_DTYPE_I64: {
    int64_t v = (int64_t)value;
    std::memcpy(p, &v, 8);
    break;
  }
  default:
    std::fprintf(stderr, "cannot store element type %s\n", dt.name);
    std::exit(2);
  }
}

static double loadElem(const std::vector<char> &raw, size_t index,
                       const DType &dt) {
  const char *p = raw.data() + index * dt.bytes;
  switch (dt.code) {
  case VX_DTYPE_F32: {
    float f;
    std::memcpy(&f, p, 4);
    return (double)f;
  }
  case VX_DTYPE_F16: {
    uint16_t h;
    std::memcpy(&h, p, 2);
    return (double)floatOfHalf(h);
  }
  case VX_DTYPE_I1:
    return *p != 0 ? 1.0 : 0.0;
  case VX_DTYPE_I8: {
    int8_t v;
    std::memcpy(&v, p, 1);
    return (double)v;
  }
  case VX_DTYPE_I16: {
    int16_t v;
    std::memcpy(&v, p, 2);
    return (double)v;
  }
  case VX_DTYPE_I32: {
    int32_t v;
    std::memcpy(&v, p, 4);
    return (double)v;
  }
  case VX_DTYPE_I64: {
    int64_t v;
    std::memcpy(&v, p, 8);
    return (double)v;
  }
  default:
    return 0.0;
  }
}

// Adds two values the way the element type does, so a narrow type wraps where
// the device wraps. Only the integer kernels need this; the float ones are
// exact at these values.
static double addTyped(double a, double b, const DType &dt) {
  if (dt.is_float)
    return a + b;
  if (dt.code == VX_DTYPE_I1)
    return (a != 0.0 || b != 0.0) ? 1.0 : 0.0;
  return a + b;
}

// ---------------------------------------------------------------- operands

struct Operand {
  std::vector<char> host; // host bytes, as the ABI hands them over
  void *device = nullptr;
  int64_t sizes[2] = {0, 0};
  int64_t strides[2] = {0, 0};
};

enum class Op { Elementwise, Reduce, Transpose, Matmul };

static Op opOf(const char *name) {
  if (std::strcmp(name, "ew") == 0)
    return Op::Elementwise;
  if (std::strcmp(name, "reduce") == 0)
    return Op::Reduce;
  if (std::strcmp(name, "transpose") == 0)
    return Op::Transpose;
  if (std::strcmp(name, "matmul") == 0)
    return Op::Matmul;
  std::fprintf(stderr, "unknown op '%s'\n", name);
  std::exit(2);
}

// The values each program fills its tensors with. These mirror the .vx files in
// scripts/tools/xpu_conformance/ by hand: a divergence shows up as a result the
// CPU backend does not match, which is why the harness re-derives them here.
static double inputA(const Op op, size_t i, size_t j) {
  switch (op) {
  case Op::Elementwise:
    return (double)((i + j) % 4);
  case Op::Reduce:
    return (double)((i * 7 + j) % 5);
  case Op::Transpose:
    return (double)((i * 13 + j * 7) % 101);
  case Op::Matmul:
    return (((double)(int)((i * 131 + j * 17) % 200)) - 100.0) / 97.0;
  }
  return 0.0;
}

static double inputB(const Op op, size_t i, size_t j) {
  switch (op) {
  case Op::Elementwise:
    return 2.0;
  case Op::Matmul:
    return (((double)(int)((i * 61 + j * 29) % 200)) - 100.0) / 89.0;
  case Op::Reduce:
  case Op::Transpose:
    return 0.0;
  }
  return 0.0;
}

static double inputC(const Op op, size_t i, size_t j) {
  (void)i;
  (void)j;
  return op == Op::Elementwise ? 1.0 : 0.0;
}

// ---------------------------------------------------------------- helpers

static uint64_t parseSize(const char *name, const char *text) {
  return parseByteCount(name, text, (uint64_t)1 << 60);
}

static std::vector<char> readFile(const char *path) {
  std::FILE *f = std::fopen(path, "rb");
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", path);
    std::exit(1);
  }
  std::vector<char> out;
  char chunk[65536];
  size_t got;
  while ((got = std::fread(chunk, 1, sizeof(chunk), f)) > 0)
    out.insert(out.end(), chunk, chunk + got);
  std::fclose(f);
  return out;
}

static void writeFile(const char *path, const std::vector<char> &bytes) {
  std::FILE *f = std::fopen(path, "wb");
  if (!f) {
    std::fprintf(stderr, "cannot write %s\n", path);
    std::exit(1);
  }
  if (!bytes.empty())
    std::fwrite(bytes.data(), 1, bytes.size(), f);
  std::fclose(f);
}

/// How many elements differ beyond `tolerance`, with the first few printed.
/// A tolerance of 0 is equality, which is what every integer kernel asks for.
static uint64_t countDifferent(const std::vector<char> &got,
                               const std::vector<char> &want, const DType &dt,
                               size_t count, double tolerance) {
  if (got.size() < count * dt.bytes || want.size() < count * dt.bytes) {
    std::fprintf(stderr, "a result file is shorter than %zu %s elements\n",
                 count, dt.name);
    std::exit(1);
  }
  uint64_t wrong = 0;
  double worst = 0.0;
  double worst_relative = 0.0;
  for (size_t i = 0; i < count; ++i) {
    const double a = loadElem(got, i, dt);
    const double b = loadElem(want, i, dt);
    const double diff = std::fabs(a - b);
    // Relative to the wanted value, with a floor of 1 so a difference near zero
    // is still an absolute one. This is the rule the tolerance is measured in.
    const double allowed =
        tolerance == 0.0 ? 0.0 : tolerance * std::max(1.0, std::fabs(b));
    worst = std::max(worst, diff);
    worst_relative =
        std::max(worst_relative, diff / std::max(1.0, std::fabs(b)));
    if (diff > allowed) {
      if (wrong < 4)
        std::printf("  element %zu: got %.9g, wanted %.9g\n", i, a, b);
      ++wrong;
    }
  }
  // Always printed, so the size of the gap is auditable rather than implied by
  // the verdict: a tolerance far above this number is not doing any work.
  std::printf("  largest difference: %.9g (%.9g relative), tolerance %g\n",
              worst, worst_relative, tolerance);
  return wrong;
}

// ---------------------------------------------------------------- dispatch

struct Descriptor {
  void *allocated;
  void *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
};

/// One kernel's operands, in the order its arguments arrive.
struct Call {
  std::vector<Operand> operands;
  std::vector<Descriptor> descriptors;
  std::vector<void *> descPtrs;
  std::vector<void *> deviceArgs;
  std::vector<int32_t> tags;
  /// Which operand the kernel writes. It is the last one for a region whose
  /// result is its final capture, and the payload's `out:` for a matmul, where
  /// the destination is argument 0.
  size_t outIndex = 0;
};

/// Build the operands for one case, matching each .vx program's spawn body.
///
/// The order is the order the spawn body reads its captures, which is the order
/// the compiler flattens the arguments in. A wrong order produces a wrong
/// number rather than an abort, so every case is also checked against the
/// independent reference below.
static Call buildCall(Op op, const DType &dt, int64_t M, int64_t N, int64_t K,
                      const std::vector<char> &payload) {
  Call call;
  auto add = [&](int64_t rows, int64_t cols,
                 const std::function<void(std::vector<char> &)> &fill) {
    Operand operand;
    operand.host.assign((size_t)rows * (size_t)cols * dt.bytes, 0);
    fill(operand.host);
    operand.sizes[0] = rows;
    operand.sizes[1] = cols;
    operand.strides[0] = cols;
    operand.strides[1] = 1;
    call.operands.push_back(std::move(operand));
  };

  if (op == Op::Elementwise) {
    add(M, N, [&](std::vector<char> &r) {
      for (int64_t i = 0; i < M; ++i)
        for (int64_t j = 0; j < N; ++j)
          storeElem(r, (size_t)(i * N + j), dt, inputA(op, i, j));
    });
    add(M, N, [&](std::vector<char> &r) {
      for (int64_t i = 0; i < M; ++i)
        for (int64_t j = 0; j < N; ++j)
          storeElem(r, (size_t)(i * N + j), dt, inputB(op, i, j));
    });
    add(M, N, [&](std::vector<char> &r) {
      for (int64_t i = 0; i < M; ++i)
        for (int64_t j = 0; j < N; ++j)
          storeElem(r, (size_t)(i * N + j), dt, inputC(op, i, j));
    });
    add(M, N, [](std::vector<char> &) {});
  } else if (op == Op::Reduce) {
    add(M, N, [&](std::vector<char> &r) {
      for (int64_t i = 0; i < M; ++i)
        for (int64_t j = 0; j < N; ++j)
          storeElem(r, (size_t)(i * N + j), dt, inputA(op, i, j));
    });
    Operand out;
    out.host.assign((size_t)M * (size_t)1 * dt.bytes, 0);
    out.sizes[0] = M;
    out.sizes[1] = 1;
    out.strides[0] = 1;
    out.strides[1] = 1;
    call.operands.push_back(std::move(out));
  } else if (op == Op::Transpose) {
    add(M, N, [&](std::vector<char> &r) {
      for (int64_t i = 0; i < M; ++i)
        for (int64_t j = 0; j < N; ++j)
          storeElem(r, (size_t)(i * N + j), dt, inputA(op, i, j));
    });
    Operand out;
    out.host.assign((size_t)N * (size_t)M * dt.bytes, 0);
    out.sizes[0] = N;
    out.sizes[1] = M;
    out.strides[0] = M;
    out.strides[1] = 1;
    call.operands.push_back(std::move(out));
  } else {
    // A classified matmul: the payload's `roles=` names which argument is
    // which, so the operand order is read off rather than assumed. `out` comes
    // first because that is what the compiler emitted for `c = a @ b`.
    int ai = -1, bi = -1, oi = -1;
    const char *roles =
        vx_payload_field(payload.data(), payload.size(), "roles=");
    if (!roles || std::sscanf(roles, "a:%d,b:%d,out:%d", &ai, &bi, &oi) != 3) {
      std::fprintf(stderr, "the matmul payload has no usable roles=\n");
      std::exit(2);
    }
    std::vector<Operand> byRole(3);
    auto fill = [&](Operand &operand, int64_t rows, int64_t cols, bool isA) {
      operand.host.assign((size_t)rows * (size_t)cols * dt.bytes, 0);
      for (int64_t i = 0; i < rows; ++i)
        for (int64_t j = 0; j < cols; ++j)
          storeElem(operand.host, (size_t)(i * cols + j), dt,
                    isA ? inputA(op, i, j) : inputB(op, i, j));
      operand.sizes[0] = rows;
      operand.sizes[1] = cols;
      operand.strides[0] = cols;
      operand.strides[1] = 1;
    };
    fill(byRole[(size_t)ai], M, K, true);
    fill(byRole[(size_t)bi], K, N, false);
    byRole[(size_t)oi].host.assign((size_t)M * (size_t)N * dt.bytes, 0);
    byRole[(size_t)oi].sizes[0] = M;
    byRole[(size_t)oi].sizes[1] = N;
    byRole[(size_t)oi].strides[0] = N;
    byRole[(size_t)oi].strides[1] = 1;
    for (Operand &operand : byRole)
      call.operands.push_back(std::move(operand));
    call.outIndex = (size_t)oi;
  }

  if (op != Op::Matmul)
    call.outIndex = call.operands.size() - 1;

  const size_t n = call.operands.size();
  call.descriptors.resize(n);
  call.descPtrs.resize(n);
  call.deviceArgs.resize(n);
  call.tags.resize(n);
  for (size_t i = 0; i < n; ++i) {
    call.descriptors[i].allocated = call.operands[i].device;
    call.descriptors[i].aligned = call.operands[i].device;
    call.descriptors[i].offset = 0;
    call.descriptors[i].sizes[0] = call.operands[i].sizes[0];
    call.descriptors[i].sizes[1] = call.operands[i].sizes[1];
    call.descriptors[i].strides[0] = call.operands[i].strides[0];
    call.descriptors[i].strides[1] = call.operands[i].strides[1];
    call.descPtrs[i] = &call.descriptors[i];
    call.deviceArgs[i] = &call.descPtrs[i];
    call.tags[i] = VX_ABI_MEMREF_TAG(dt.code, 2);
  }
  return call;
}

/// The answer the kernel should give, computed here so a wrong result on both
/// backends is still caught.
static std::vector<char> referenceOf(Op op, const DType &dt, int64_t M,
                                     int64_t N, int64_t K) {
  std::vector<char> out;
  if (op == Op::Elementwise) {
    out.assign((size_t)M * (size_t)N * dt.bytes, 0);
    for (int64_t i = 0; i < M; ++i)
      for (int64_t j = 0; j < N; ++j) {
        const double v =
            addTyped(inputA(op, i, j) * inputB(op, i, j), inputC(op, i, j), dt);
        storeElem(out, (size_t)(i * N + j), dt, v);
      }
  } else if (op == Op::Reduce) {
    out.assign((size_t)M * dt.bytes, 0);
    for (int64_t i = 0; i < M; ++i) {
      double sum = 0.0;
      for (int64_t j = 0; j < N; ++j)
        sum = addTyped(sum, inputA(op, i, j), dt);
      storeElem(out, (size_t)i, dt, sum);
    }
  } else if (op == Op::Transpose) {
    out.assign((size_t)N * (size_t)M * dt.bytes, 0);
    for (int64_t i = 0; i < M; ++i)
      for (int64_t j = 0; j < N; ++j)
        storeElem(out, (size_t)(j * M + i), dt, inputA(op, i, j));
  } else {
    out.assign((size_t)M * (size_t)N * dt.bytes, 0);
    for (int64_t i = 0; i < M; ++i)
      for (int64_t j = 0; j < N; ++j) {
        double sum = 0.0;
        for (int64_t k = 0; k < K; ++k)
          sum += inputA(op, i, k) * inputB(op, k, j);
        storeElem(out, (size_t)(i * N + j), dt, sum);
      }
  }
  return out;
}

static int runCase(Op op, const DType &dt, int64_t M, int64_t N, int64_t K,
                   int reps, const char *payloadPath, const char *kernel,
                   const char *outPath) {
  std::vector<char> payload = readFile(payloadPath);
  // The first entry is the kernel name, and the payload is bounded by its
  // own size rather than by a terminator: a device image may end on any byte.
  if (payload.empty() || std::strcmp(payload.data(), kernel) != 0) {
    std::fprintf(stderr, "%s is not the payload for %s\n", payloadPath, kernel);
    return 2;
  }

  // The documented host fallback, reached the way a backend reaches it: the
  // payload says its image is in a format this backend does not load, so
  // run_device_image returns false and the outlined kernel runs on the host.
  // Same arguments, same kernel, same expected numbers.
  if (std::getenv("VX_CONFORMANCE_HOST_FALLBACK")) {
    const char *entry = "format=spirv";
    const size_t entry_len = std::strlen(entry);
    const auto found =
        std::search(payload.begin(), payload.end(), entry, entry + entry_len);
    if (found == payload.end()) {
      std::fprintf(stderr, "%s has no format=spirv entry to change\n",
                   payloadPath);
      return 2;
    }
    std::memcpy(&*found + 7, "metal", 5);
    std::printf(
        "fallback: changed format=spirv to format=metal, so the device\n"
        "          image is declined and the outlined kernel runs on the\n"
        "          host\n");
  }

  Call call = buildCall(op, dt, M, N, K, payload);

  // The footprint is the device tensors, and it is known before anything is
  // allocated.
  uint64_t footprint = 0;
  for (const Operand &operand : call.operands)
    footprint += operand.host.size();
  requireWithinCeiling("vx_xpu_conformance", footprint, vramCeiling());
  std::printf("footprint: %llu bytes of memory (%zu tensors)\n",
              (unsigned long long)footprint, call.operands.size());

  // Stage every operand through the backend's allocator, which is the call a
  // compiled program makes.
  for (Operand &operand : call.operands)
    operand.device = vx_plugin_alloc_and_transfer(
        operand.host.size(), operand.host.data(), kGpuTopology,
        VX_SPACE_ACCESS_HOST_READABLE);

  const bool perturb = std::getenv("VX_CONFORMANCE_PERTURB") != nullptr;
  if (perturb) {
    // Change one input element, so the result this backend computes differs
    // from the other backend's. Never the output operand, which is argument 0
    // for a matmul.
    Operand &first = call.operands[call.outIndex == 0 ? 1 : 0];
    const double before = loadElem(first.host, 0, dt);
    storeElem(first.host, 0, dt, before + 1.0);
    std::memcpy(first.device, first.host.data(), first.host.size());
    std::printf("sabotage: changed input element 0 from %.9g to %.9g\n", before,
                loadElem(first.host, 0, dt));
  }

  for (size_t i = 0; i < call.operands.size(); ++i) {
    call.descriptors[i].allocated = call.operands[i].device;
    call.descriptors[i].aligned = call.operands[i].device;
  }

  auto dispatch = [&]() {
    vx_plugin_dispatch_async(payload.data(), payload.size(),
                             call.deviceArgs.data(), call.tags.data(),
                             (int64_t)call.operands.size());
  };

  // A big kernel's warm-up costs what the kernel costs, so it is overridable;
  // the CPU backend's reference GEMM at 8192 takes minutes and three of them
  // would be the whole run.
  int warmup = kWarmup;
  if (const char *env = std::getenv("VX_CONFORMANCE_WARMUP"))
    warmup = (int)parseSize("VX_CONFORMANCE_WARMUP", env);
  for (int i = 0; i < warmup; ++i)
    dispatch();
  double best = 1e30;
  for (int i = 0; i < reps; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    dispatch();
    auto t1 = std::chrono::steady_clock::now();
    best = std::min(best,
                    std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  std::printf("dispatch: best of %d after %d warm-up = %.4f ms\n", reps, warmup,
              best);
  if (op == Op::Matmul) {
    const double flops = 2.0 * (double)M * (double)N * (double)K;
    std::printf("matmul: %lldx%lldx%lld fp32 = %.2f GFLOP = %.4f ms = %.4f "
                "TFLOPS (%.2f GFLOP/s)\n",
                (long long)M, (long long)N, (long long)K, flops / 1e9, best,
                flops / (best / 1000.0) / 1e12, flops / best / 1e6);
  }

  // Read the result back through the ABI, exactly as a compiled program does.
  // Which operand that is comes from the payload: a matmul's destination is
  // argument 0, not the last capture.
  const Operand &result = call.operands[call.outIndex];
  std::vector<char> got(result.host.size(), 0);
  vx_plugin_transfer_device_to_host(result.device, got.data(), got.size(),
                                    kGpuTopology);
  writeFile(outPath, got);

  // A matmul's two summations run in different orders, so the comparison
  // against a hand reference needs the same relative tolerance the two backends
  // are compared with. Every other kernel here is exact.
  const double tolerance = (op == Op::Matmul && dt.is_float) ? 1e-4 : 0.0;

  // The independent reference is worth its cost only while it is cheap; past
  // that, a sum computed in a different order than the kernel's is the check.
  const double work = (double)M * (double)std::max<int64_t>(1, N) *
                      (double)std::max<int64_t>(1, K);
  if (work <= 2e9) {
    const std::vector<char> reference = referenceOf(op, dt, M, N, K);
    const uint64_t wrong =
        countDifferent(got, reference, dt, got.size() / dt.bytes, tolerance);
    std::printf("against reference: %llu of %zu elements differ (tolerance "
                "%g)\n",
                (unsigned long long)wrong, got.size() / dt.bytes, tolerance);
  } else {
    std::printf("against reference: skipped (too large to check by hand)\n");
  }

  // `sum(C) = sum_k rowsum(A,k) * colsum(B,k)` holds for any summation order,
  // so it checks a result too large to compare element by element. It is O(N^2)
  // where the GEMM is O(N^3), so it costs nothing at 8192.
  if (op == Op::Matmul) {
    double total = 0.0;
    const size_t count = got.size() / dt.bytes;
    for (size_t i = 0; i < count; ++i)
      total += loadElem(got, i, dt);
    double expected = 0.0;
    for (int64_t k = 0; k < K; ++k) {
      double column_of_a = 0.0, row_of_b = 0.0;
      for (int64_t i = 0; i < M; ++i)
        column_of_a += inputA(op, i, k);
      for (int64_t j = 0; j < N; ++j)
        row_of_b += inputB(op, k, j);
      expected += column_of_a * row_of_b;
    }
    const double relative =
        std::fabs(total - expected) / std::max(1.0, std::fabs(expected));
    std::printf("sum check: got %.6f, expected %.6f, relative difference "
                "%.3g\n",
                total, expected, relative);
  }

  for (Operand &operand : call.operands)
    vx_plugin_free(operand.device, kGpuTopology);
  return 0;
}

int main(int argc, char **argv) {
  if (argc >= 2 && std::strcmp(argv[1], "compare") == 0) {
    // compare <dtype> <count> <tolerance> <a.bin> <b.bin>
    if (argc != 7) {
      std::fprintf(stderr,
                   "usage: %s compare <dtype> <count> <tolerance> <a> <b>\n",
                   argv[0]);
      return 2;
    }
    const DType &dt = dtypeOf(argv[2]);
    const size_t count = (size_t)parseSize("count", argv[3]);
    const double tolerance = std::strtod(argv[4], nullptr);
    const std::vector<char> a = readFile(argv[5]);
    const std::vector<char> b = readFile(argv[6]);
    const uint64_t wrong = countDifferent(a, b, dt, count, tolerance);
    std::printf("result: %s (%llu of %zu elements differ)\n",
                wrong ? "DIFFERENT" : "equal", (unsigned long long)wrong,
                count);
    return wrong ? 1 : 0;
  }

  if (argc != 11 || std::strcmp(argv[1], "run") != 0) {
    std::fprintf(stderr,
                 "usage: %s run <op> <dtype> <M> <N> <K> <reps> <payload> "
                 "<kernel> <out.bin>\n",
                 argv[0]);
    return 2;
  }
  const Op op = opOf(argv[2]);
  const DType &dt = dtypeOf(argv[3]);
  const int64_t M = (int64_t)parseSize("M", argv[4]);
  const int64_t N = (int64_t)parseSize("N", argv[5]);
  const int64_t K = (int64_t)parseSize("K", argv[6]);
  const int reps = (int)parseSize("reps", argv[7]);
  return runCase(op, dt, M, N, K, reps, argv[8], argv[9], argv[10]);
}
