//===- vx_stream_policy.h - What a wait does on a caller's stream -*- C++ -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A host program can give Vx a CUDA stream for the calling thread with
// `vx_cuda_set_stream` (#1986). Then Vx's kernels go on that stream, and a
// wait covers that stream only. While the stream records a CUDA graph, a wait
// or a copy from host memory cannot go into the graph, so the runtime stops
// with a message. This header holds those choices, so that a test can check
// them on a machine with no GPU (tests/runtime/kernel_launch_test.cpp).
//
//===----------------------------------------------------------------------===//

#ifndef VX_STREAM_POLICY_H
#define VX_STREAM_POLICY_H

/* A wait for the GPU, or a copy between host and device memory. */
typedef enum {
  /* No stream is set: wait for the whole device, as before. */
  VX_WAIT_DEVICE,
  /* Wait for the caller's stream only. */
  VX_WAIT_STREAM,
  /* The stream records a graph, which cannot hold a wait. */
  VX_WAIT_REFUSE,
} vx_wait_action;

static inline vx_wait_action vx_wait_action_for(int has_stream,
                                                int recording) {
  if (!has_stream) {
    return VX_WAIT_DEVICE;
  }
  return recording ? VX_WAIT_REFUSE : VX_WAIT_STREAM;
}

/* A free of device memory. */
typedef enum {
  /* Wait for the device, then `cudaFree`, as before. */
  VX_FREE_AFTER_DEVICE_WAIT,
  /* Wait for the caller's stream, then `cudaFree`. */
  VX_FREE_AFTER_STREAM_WAIT,
  /* `cudaFreeAsync` on the stream, with no wait. */
  VX_FREE_ON_STREAM,
  /* The stream records a graph, and the memory needs a wait first. */
  VX_FREE_REFUSE,
} vx_free_action;

/* `on_stream` says whether the memory came from `cudaMallocAsync` on a
   caller's stream. Only that memory is freed in stream order. */
static inline vx_free_action vx_free_action_for(int has_stream, int recording,
                                                int on_stream) {
  if (!has_stream) {
    return VX_FREE_AFTER_DEVICE_WAIT;
  }
  if (on_stream) {
    return VX_FREE_ON_STREAM;
  }
  return recording ? VX_FREE_REFUSE : VX_FREE_AFTER_STREAM_WAIT;
}

#endif // VX_STREAM_POLICY_H
