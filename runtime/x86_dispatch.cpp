//===- x86_dispatch.cpp - Vx dispatch backend for x86-64 --------*- C++ -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The vx_plugin_* ABI on x86-64. A sibling of runtime/cuda_dispatch.cpp rather
// than a fallback from it: the compiler emits the same calls -- allocate,
// transfer, free, dispatch -- and this answers them for a CPU.
//
// Naming the target matters for what #319 is trying to show. A run across
// {A100, x86, H100, AMD} is a claim that one program text places work on
// whatever is there; that reads very differently if every non-GPU target is a
// single thing called "the portable shim". Here x86 is a target Vx supports,
// with its own backend, not the absence of one.
//
// What is actually specific to x86-64 today is the allocation alignment: 64
// bytes, which is both the cache-line size and the width an AVX-512 load wants.
// Routing a recognised GEMM to a blocked kernel over these buffers is the
// obvious next step and is deliberately not guessed at here -- an untuned
// "optimized" path that loses to the outlined loop nest would be worse than the
// loop nest.
//
//===----------------------------------------------------------------------===//

#define VX_BACKEND_NAME "x86"
#define VX_BACKEND_ALIGN 64

#include "host_dispatch_common.h"
