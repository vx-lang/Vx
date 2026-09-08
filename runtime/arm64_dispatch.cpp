//===- arm64_dispatch.cpp - Vx dispatch backend for AArch64 -----*- C++ -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The vx_plugin_* ABI on AArch64. A sibling of runtime/cuda_dispatch.cpp and
// runtime/x86_dispatch.cpp: the compiler emits the same calls and this answers
// them for a CPU.
//
// Used on non-Apple AArch64 -- Graviton, Ampere, an ARM development board. On
// Apple silicon runtime/npu_dispatch.mm claims the target instead, because
// there the interesting hardware is the ANE and the AMX units rather than the
// scalar cores.
//
// 64-byte alignment: the cache line on every AArch64 core Vx is likely to meet,
// and a comfortable multiple of the 16 bytes a NEON quadword load wants. SVE
// vectors are longer and runtime-sized, so an SVE-aware allocator would have to
// ask the hardware rather than assume -- which is a reason to leave it until
// there is something that uses it.
//
//===----------------------------------------------------------------------===//

#define VX_BACKEND_NAME "arm64"
#define VX_BACKEND_ALIGN 64

#include "host_dispatch_common.h"
