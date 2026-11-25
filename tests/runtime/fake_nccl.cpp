//===- fake_nccl.cpp - An NCCL that records its calls -----------*- C++ -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The NCCL functions that runtime/vx_collective.h calls. Each one writes its
// arguments to `fake_nccl_calls` and moves no data, so that
// tests/runtime/collective_test.cpp can check them with no GPU.
//
//===----------------------------------------------------------------------===//

#include <cstddef>
#include <cstring>
#include <mutex>

struct UniqueId {
  char internal[128];
};

struct FakeNcclCalls {
  int ids_made;
  int inits;
  int init_ranks[2];
  int init_nranks[2];
  UniqueId init_ids[2];
  int destroys;
  const char *last;
  const void *input;
  void *output;
  size_t count;
  int type;
  int op;
  void *stream;
  // The next call returns this result, and then 0 again.
  int fail_next;
};

extern "C" {

FakeNcclCalls fake_nccl_calls;
static std::mutex lock;
static char comms[2];

static int result() {
  const int r = fake_nccl_calls.fail_next;
  fake_nccl_calls.fail_next = 0;
  return r;
}

int ncclGetUniqueId(UniqueId *id) {
  std::lock_guard<std::mutex> guard(lock);
  memset(id, 0, sizeof *id);
  id->internal[0] = (char)(++fake_nccl_calls.ids_made);
  id->internal[127] = 'v';
  return result();
}

int ncclCommInitRank(void **comm, int nranks, UniqueId id, int rank) {
  std::lock_guard<std::mutex> guard(lock);
  const int i = fake_nccl_calls.inits++ % 2;
  fake_nccl_calls.init_ranks[i] = rank;
  fake_nccl_calls.init_nranks[i] = nranks;
  fake_nccl_calls.init_ids[i] = id;
  *comm = &comms[i];
  return result();
}

int ncclCommDestroy(void *) {
  ++fake_nccl_calls.destroys;
  return result();
}

static int record(const char *name, const void *input, void *output,
                  size_t count, int type, int op, void *stream) {
  fake_nccl_calls.last = name;
  fake_nccl_calls.input = input;
  fake_nccl_calls.output = output;
  fake_nccl_calls.count = count;
  fake_nccl_calls.type = type;
  fake_nccl_calls.op = op;
  fake_nccl_calls.stream = stream;
  return result();
}

int ncclAllReduce(const void *input, void *output, size_t count, int type,
                  int op, void *, void *stream) {
  return record("ncclAllReduce", input, output, count, type, op, stream);
}

int ncclAllGather(const void *input, void *output, size_t count, int type,
                  void *, void *stream) {
  return record("ncclAllGather", input, output, count, type, -1, stream);
}

const char *ncclGetErrorString(int r) {
  return r == 2 ? "fake system error" : "fake error";
}

} // extern "C"
