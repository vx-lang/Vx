//===- vx_collective.h - All-reduce and all-gather across GPUs --*- C++ -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The collectives of docs/lang/abi.md section 3.7. A job runs one process per
// GPU. Each process is one rank, and `vx_collective_init(rank, ranks)` joins
// it to the others. An all-reduce or an all-gather then goes on the stream
// from `vx_cuda_set_stream`, so a CUDA graph can record it.
//
// The CUDA backend calls NCCL, loaded with dlopen as the CUDA driver is. A
// build with no NCCL must still link every program. The CPU backends answer
// one rank only, where each collective is a copy.
//
// The functions here return an error message, or null. The backends stop the
// program on an error. A test drives this file against a fake NCCL with no GPU
// (tests/runtime/collective_test.cpp).
//
//===----------------------------------------------------------------------===//

#ifndef VX_COLLECTIVE_H
#define VX_COLLECTIVE_H

#include <dlfcn.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>

namespace vx {

/// The part of NCCL's ABI that Vx uses, as in nccl.h. It did not change
/// since NCCL 2.0.
struct NcclUniqueId {
  char internal[128];
};
using NcclComm = void *;
enum NcclDataType : int { NcclFloat32 = 7, NcclBfloat16 = 9 };
enum NcclRedOp : int { NcclSum = 0 };

struct NcclApi {
  int (*GetUniqueId)(NcclUniqueId *);
  int (*CommInitRank)(NcclComm *, int, NcclUniqueId, int);
  int (*CommDestroy)(NcclComm);
  int (*AllReduce)(const void *, void *, size_t, int, int, NcclComm, void *);
  int (*AllGather)(const void *, void *, size_t, int, NcclComm, void *);
  const char *(*GetErrorString)(int);
};

/// Loads NCCL from `path`. Returns null and sets `why` if it cannot.
inline std::unique_ptr<const NcclApi> load_nccl(const char *path,
                                                std::string *why) {
  void *lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (!lib) {
    *why = std::string("Vx cannot load NCCL: ") + dlerror() +
           ". Install NCCL, or set VX_NCCL_LIB to the path of libnccl.so.2.";
    return nullptr;
  }
  auto api = std::make_unique<NcclApi>();
  auto find = [&](auto &slot, const char *name) {
    slot = reinterpret_cast<std::remove_reference_t<decltype(slot)>>(
        dlsym(lib, name));
    if (!slot && why->empty()) {
      *why = std::string(path) + " has no " + name + ".";
    }
  };
  why->clear();
  find(api->GetUniqueId, "ncclGetUniqueId");
  find(api->CommInitRank, "ncclCommInitRank");
  find(api->CommDestroy, "ncclCommDestroy");
  find(api->AllReduce, "ncclAllReduce");
  find(api->AllGather, "ncclAllGather");
  find(api->GetErrorString, "ncclGetErrorString");
  if (!why->empty()) {
    return nullptr;
  }
  return api;
}

/// Rank 0 writes the NCCL id to `path`. It writes another file first and then
/// renames it, so a rank that waits never reads half an id. A file that is
/// already there may hold an old job's id, which another rank may have read.
inline std::string write_unique_id(const char *path, const NcclUniqueId &id) {
  if (access(path, F_OK) == 0) {
    return std::string("the file ") + path +
           " is already there. It may hold the id of an earlier job. Remove "
           "it before you start the ranks.";
  }
  const std::string partial = std::string(path) + ".partial";
  FILE *f = fopen(partial.c_str(), "wb");
  if (!f || fwrite(&id, sizeof id, 1, f) != 1 || fclose(f) != 0) {
    return "rank 0 cannot write " + partial + ": " + strerror(errno) + ".";
  }
  if (rename(partial.c_str(), path) != 0) {
    return "rank 0 cannot rename " + partial + ": " + strerror(errno) + ".";
  }
  return "";
}

/// The other ranks wait for the file of rank 0, at most `timeout`.
inline std::string read_unique_id(const char *path, NcclUniqueId *id,
                                  std::chrono::milliseconds timeout) {
  const auto end = std::chrono::steady_clock::now() + timeout;
  while (access(path, F_OK) != 0) {
    if (std::chrono::steady_clock::now() > end) {
      return std::string("rank 0 did not write ") + path + " in " +
             std::to_string(timeout.count() / 1000) + " s.";
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  FILE *f = fopen(path, "rb");
  const bool ok = f && fread(id, sizeof *id, 1, f) == 1;
  if (f) {
    fclose(f);
  }
  if (!ok) {
    return std::string("cannot read the NCCL id from ") + path + ".";
  }
  return "";
}

/// What a collective needs from the CUDA backend. A test gives its own.
struct CollectiveDevice {
  /// The stream from `vx_cuda_set_stream`, or null for the legacy stream.
  void *(*stream)();
  /// Whether the stream records a CUDA graph now.
  bool (*recording)();
  /// Whether `ptr` points into GPU memory.
  bool (*device_memory)(const void *ptr);
  int (*current_device)();
  void (*set_device)(int);
};

/// One process's rank in a job, through NCCL.
class NcclCollective {
public:
  explicit NcclCollective(CollectiveDevice device) : device_(device) {}

  /// Joins this process to the job as `rank` of `ranks`. `nccl_lib` is the
  /// library to load, and `id_file` passes the NCCL id from rank 0 to the
  /// others.
  std::string init(int32_t rank, int32_t ranks, const char *nccl_lib,
                   const char *id_file, std::chrono::milliseconds timeout) {
    if (comm_) {
      return "vx_collective_init ran a second time. Call vx_collective_finish "
             "first.";
    }
    if (rank < 0 || rank >= ranks) {
      return "vx_collective_init needs 0 <= rank < ranks, but rank is " +
             std::to_string(rank) + " and ranks is " + std::to_string(ranks) +
             ".";
    }
    if (device_.recording()) {
      return "vx_collective_init ran while the stream records a CUDA graph. "
             "Call it before the recording.";
    }
    std::string why;
    if (!api_ && !(api_ = load_nccl(nccl_lib, &why))) {
      return why;
    }
    NcclUniqueId id;
    if (ranks == 1) {
      if (int r = api_->GetUniqueId(&id)) {
        return nccl_error("ncclGetUniqueId", r);
      }
    } else if (!id_file || !*id_file) {
      return "a job with more than one rank needs VX_NCCL_ID_FILE, a path that "
             "every rank can read. Rank 0 writes the NCCL id there.";
    } else if (rank == 0) {
      if (int r = api_->GetUniqueId(&id)) {
        return nccl_error("ncclGetUniqueId", r);
      }
      if (!(why = write_unique_id(id_file, id)).empty()) {
        return why;
      }
    } else if (!(why = read_unique_id(id_file, &id, timeout)).empty()) {
      return why;
    }
    device_index_ = device_.current_device();
    if (int r = api_->CommInitRank(&comm_, ranks, id, rank)) {
      comm_ = nullptr;
      return nccl_error("ncclCommInitRank", r);
    }
    rank_ = rank;
    ranks_ = ranks;
    return "";
  }

  /// Leaves the job. A later `init` can join another one.
  std::string finish() {
    if (!comm_) {
      return "vx_collective_finish ran before vx_collective_init.";
    }
    const int r = api_->CommDestroy(comm_);
    comm_ = nullptr;
    rank_ = -1;
    ranks_ = 0;
    return r ? nccl_error("ncclCommDestroy", r) : "";
  }

  int32_t rank() const { return rank_; }
  int32_t ranks() const { return ranks_; }

  /// `output[i]` becomes the sum of `input[i]` over all ranks, for `count`
  /// elements. `input` and `output` can be the same.
  std::string all_reduce_sum(const void *input, void *output, int64_t count,
                             NcclDataType type, const char *name) {
    std::string why = check(input, output, count, name);
    if (!why.empty()) {
      return why;
    }
    return on_device(name, [&](void *stream) {
      return api_->AllReduce(input, output, (size_t)count, type, NcclSum, comm_,
                             stream);
    });
  }

  /// `output` becomes the `input` of rank 0, then of rank 1, and so on: `count`
  /// elements from each rank.
  std::string all_gather(const void *input, void *output, int64_t count,
                         NcclDataType type, const char *name) {
    std::string why = check(input, output, count, name);
    if (!why.empty()) {
      return why;
    }
    return on_device(name, [&](void *stream) {
      return api_->AllGather(input, output, (size_t)count, type, comm_, stream);
    });
  }

private:
  std::string check(const void *input, const void *output, int64_t count,
                    const char *name) const {
    const std::string fn = name;
    if (!comm_) {
      return fn + " ran before vx_collective_init.";
    }
    if (count < 0) {
      return fn + " got a count of " + std::to_string(count) + ".";
    }
    if (count > 0 &&
        !(device_.device_memory(input) && device_.device_memory(output))) {
      return fn + " needs tensors in GPU memory. Transfer them to the GPU "
                  "first.";
    }
    return "";
  }

  /// Runs `call` on the GPU that joined the job, and restores the current GPU.
  template <typename Call> std::string on_device(const char *name, Call call) {
    const int current = device_.current_device();
    if (current != device_index_) {
      device_.set_device(device_index_);
    }
    const int r = call(device_.stream());
    if (current != device_index_) {
      device_.set_device(current);
    }
    return r ? nccl_error(name, r) : "";
  }

  std::string nccl_error(const char *what, int r) const {
    return std::string(what) + " failed: " + api_->GetErrorString(r) + ".";
  }

  CollectiveDevice device_;
  std::unique_ptr<const NcclApi> api_;
  NcclComm comm_ = nullptr;
  int device_index_ = 0;
  int32_t rank_ = -1;
  int32_t ranks_ = 0;
};

/// A job of one rank, for a process with no GPU. Each collective copies its
/// input to its output.
class OneRankCollective {
public:
  std::string init(int32_t rank, int32_t ranks) {
    if (ranks_) {
      return "vx_collective_init ran a second time. Call vx_collective_finish "
             "first.";
    }
    if (ranks != 1 || rank != 0) {
      return "vx_collective_init got rank " + std::to_string(rank) + " of " +
             std::to_string(ranks) +
             " ranks, but this process has no GPU, so it runs a job of one "
             "rank only.";
    }
    ranks_ = 1;
    return "";
  }

  std::string finish() {
    if (!ranks_) {
      return "vx_collective_finish ran before vx_collective_init.";
    }
    ranks_ = 0;
    return "";
  }

  int32_t rank() const { return ranks_ ? 0 : -1; }
  int32_t ranks() const { return ranks_; }

  /// The all-reduce and the all-gather of one rank, for `count` elements of
  /// `element` bytes.
  std::string copy(const void *input, void *output, int64_t count,
                   size_t element, const char *name) const {
    if (!ranks_) {
      return std::string(name) + " ran before vx_collective_init.";
    }
    if (count < 0) {
      return std::string(name) + " got a count of " + std::to_string(count) +
             ".";
    }
    if (input != output) {
      memmove(output, input, (size_t)count * element);
    }
    return "";
  }

private:
  int32_t ranks_ = 0;
};

} // namespace vx

#endif // VX_COLLECTIVE_H
