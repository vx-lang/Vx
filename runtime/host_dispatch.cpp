//===- host_dispatch.cpp - Portable Vx dispatch backend ---------*- C++ -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The vx_plugin_* ABI for a CPU target with no backend of its own.
//
// x86-64 and AArch64 have named backends (runtime/x86_dispatch.cpp,
// runtime/arm64_dispatch.cpp) and Apple silicon has runtime/npu_dispatch.mm.
// This is what a target gets before someone has looked at it: correct, and
// making no claim about the hardware beyond that it has a C library.
//
// Every target having *some* backend is what lets the compiler emit calls to
// the plugin ABI unconditionally -- `transfer` becomes an allocation and a copy
// here and cudaMalloc plus an H2D copy under the CUDA backend, and the program
// does not change. Without this file a program containing a non-CPU `spawn on`
// would have no definition for vx_plugin_dispatch_async and fail to link, which
// is the state Linux was in before it existed.
//
//===----------------------------------------------------------------------===//

#define VX_BACKEND_NAME "Dispatcher"
#define VX_BACKEND_ALIGN 16

#include "host_dispatch_common.h"
