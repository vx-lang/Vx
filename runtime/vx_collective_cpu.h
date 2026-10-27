//===- vx_collective_cpu.h - Collectives with no GPU ------------*- C++ -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The collectives of docs/lang/abi.md section 3.7 for a CPU backend. Such a
// build has no NCCL, so it runs a job of one rank only.
//
//===----------------------------------------------------------------------===//

#ifndef VX_COLLECTIVE_CPU_H
#define VX_COLLECTIVE_CPU_H

#include "vx_collective.h"

static vx::OneRankCollective vx_one_rank;

static void vx_collective_cpu_check(const std::string &why) {
  if (!why.empty()) {
    fprintf(stderr, "[Vx] FATAL: %s\n", why.c_str());
    abort();
  }
}

extern "C" {

void vx_collective_init(int32_t rank, int32_t ranks) {
  vx_collective_cpu_check(vx_one_rank.init(rank, ranks));
}

void vx_collective_finish(void) {
  vx_collective_cpu_check(vx_one_rank.finish());
}

int32_t vx_collective_rank(void) { return vx_one_rank.rank(); }

int32_t vx_collective_ranks(void) { return vx_one_rank.ranks(); }

void vx_collective_all_reduce_sum_f32(const float *input, float *output,
                                      int64_t count) {
  vx_collective_cpu_check(vx_one_rank.copy(input, output, count, 4,
                                           "vx_collective_all_reduce_sum_f32"));
}

void vx_collective_all_reduce_sum_bf16(const uint16_t *input, uint16_t *output,
                                       int64_t count) {
  vx_collective_cpu_check(vx_one_rank.copy(
      input, output, count, 2, "vx_collective_all_reduce_sum_bf16"));
}

void vx_collective_all_gather_f32(const float *input, float *output,
                                  int64_t count) {
  vx_collective_cpu_check(vx_one_rank.copy(input, output, count, 4,
                                           "vx_collective_all_gather_f32"));
}

void vx_collective_all_gather_bf16(const uint16_t *input, uint16_t *output,
                                   int64_t count) {
  vx_collective_cpu_check(vx_one_rank.copy(input, output, count, 2,
                                           "vx_collective_all_gather_bf16"));
}

} // extern "C"

#endif // VX_COLLECTIVE_CPU_H
