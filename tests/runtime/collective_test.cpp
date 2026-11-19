//===- collective_test.cpp - All-reduce and all-gather, no GPU --*- C++ -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Checks runtime/vx_collective.h against tests/runtime/fake_nccl.cpp, an NCCL
// that records its calls. A wrong count, element type or stream gives a wrong
// sum with no error, and two ranks with different ids wait for each other
// forever.
//
// usage: collective_test <fake libnccl.so> <empty directory>
// Driven by tests/integration_test/collective_test.rs.
//
//===----------------------------------------------------------------------===//

#include "../../runtime/vx_collective.h"

#include <cstdio>
#include <mutex>
#include <set>
#include <sys/stat.h>
#include <thread>
#include <vector>

namespace {

int failures = 0;

void check(bool cond, const char *what) {
  if (!cond) {
    fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

bool has(const std::string &text, const char *part) {
  return text.find(part) != std::string::npos;
}

// The same layout as in fake_nccl.cpp.
struct FakeNcclCalls {
  int ids_made;
  int inits;
  int init_ranks[2];
  int init_nranks[2];
  vx::NcclUniqueId init_ids[2];
  int destroys;
  const char *last;
  const void *input;
  void *output;
  size_t count;
  int type;
  int op;
  void *stream;
  int fail_next;
};

FakeNcclCalls *calls = nullptr;

// The GPU of the test: memory at `gpu_memory` is device memory, and the
// current device and the stream are variables.
float gpu_memory[8];
char stream_a;
void *stream = nullptr;
bool recording = false;
int current_device = 0;
std::set<int> devices_set;

// What the device did for the buffers of vx_peer_buffer_create, in order, and
// which rank's thread did it. A handle holds the address it opens.
struct Event {
  int rank;
  std::string what;
  void *ptr;
};
std::mutex events_lock;
std::vector<Event> events;
thread_local int event_rank = 0;

void note(const char *what, void *ptr) {
  std::lock_guard<std::mutex> guard(events_lock);
  events.push_back({event_rank, what, ptr});
}

vx::CollectiveDevice test_device() {
  return vx::CollectiveDevice{
      [] { return stream; },
      [] { return recording; },
      [](const void *p) {
        return p >= (const void *)gpu_memory &&
               p < (const void *)(gpu_memory + 8);
      },
      [] { return current_device; },
      [](int d) {
        devices_set.insert(d);
        current_device = d;
      },
      [](size_t bytes) -> void * {
        void *p = calloc(1, bytes);
        note("allocate", p);
        return p;
      },
      [](void *p) {
        note("release", p);
        free(p);
      },
      [](void *device, const void *host, size_t bytes) {
        memcpy(device, host, bytes);
      },
      [](void *host, const void *device, size_t bytes) {
        memcpy(host, device, bytes);
      },
      [](void *s) { note("synchronize", s); },
      [](void *p, void *handle) -> std::string {
        memset(handle, 0, vx::kPeerHandleBytes);
        memcpy(handle, &p, sizeof p);
        return "";
      },
      [](const void *handle, void **p) -> std::string {
        memcpy(p, handle, sizeof *p);
        note("open", *p);
        return "";
      },
      [](void *p) -> std::string {
        // Rank 1 closes late, so a rank that does not wait for it frees its
        // part first.
        if (event_rank == 1) {
          std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        note("close", p);
        return "";
      },
  };
}

// The events of `what` on `ptr`.
size_t count_events(const char *what, void *ptr) {
  size_t n = 0;
  for (const Event &e : events) {
    n += e.what == what && e.ptr == ptr;
  }
  return n;
}

// The position of the first event of `what` on `ptr`, or events.size().
size_t first_event(const char *what, void *ptr) {
  for (size_t i = 0; i < events.size(); ++i) {
    if (events[i].what == what && events[i].ptr == ptr) {
      return i;
    }
  }
  return events.size();
}

const auto no_wait = std::chrono::milliseconds(0);
std::string nccl;
std::string dir;

void test_a_missing_nccl_names_the_variable_that_finds_it() {
  vx::NcclCollective c(test_device());
  std::string why = c.init(0, 1, "/no/such/libnccl.so.2", nullptr, no_wait);
  check(has(why, "VX_NCCL_LIB"), "a missing NCCL names VX_NCCL_LIB");
}

void test_a_rank_outside_the_job_is_refused() {
  vx::NcclCollective c(test_device());
  check(has(c.init(2, 2, nccl.c_str(), nullptr, no_wait), "0 <= rank < ranks"),
        "rank 2 of 2 is refused");
  check(has(c.init(0, 0, nccl.c_str(), nullptr, no_wait), "0 <= rank < ranks"),
        "a job of 0 ranks is refused");
  check(c.ranks() == 0, "a refused init leaves no job");
}

void test_init_in_a_recording_is_refused() {
  vx::NcclCollective c(test_device());
  recording = true;
  check(has(c.init(0, 1, nccl.c_str(), nullptr, no_wait), "records a CUDA"),
        "init while the stream records is refused");
  recording = false;
}

void test_two_ranks_need_the_id_file() {
  vx::NcclCollective c(test_device());
  check(has(c.init(0, 2, nccl.c_str(), nullptr, no_wait), "VX_NCCL_ID_FILE"),
        "two ranks with no id file are refused");
}

void test_both_ranks_join_with_the_id_of_rank_0() {
  const std::string file = dir + "/id";
  const int before = calls->inits;
  std::string why0, why1;
  // Rank 1 starts first and waits for the file of rank 0.
  std::thread one([&] {
    vx::NcclCollective c(test_device());
    why1 = c.init(1, 2, nccl.c_str(), file.c_str(), std::chrono::seconds(10));
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  vx::NcclCollective zero(test_device());
  why0 = zero.init(0, 2, nccl.c_str(), file.c_str(), no_wait);
  one.join();
  check(why0.empty() && why1.empty(), "two ranks join");
  check(calls->inits == before + 2, "each rank calls ncclCommInitRank");
  const int a = before % 2, b = (before + 1) % 2;
  check(calls->init_nranks[a] == 2 && calls->init_nranks[b] == 2,
        "each rank says the job has 2 ranks");
  check(calls->init_ranks[a] + calls->init_ranks[b] == 1,
        "the ranks are 0 and 1");
  check(memcmp(&calls->init_ids[a], &calls->init_ids[b],
               sizeof(vx::NcclUniqueId)) == 0,
        "both ranks join with the same id");
  check(calls->init_ids[a].internal[127] == 'v',
        "the id is the one ncclGetUniqueId made");
  check(zero.rank() == 0 && zero.ranks() == 2, "rank 0 knows its rank");
}

void test_an_old_id_file_is_refused() {
  // The file of the test above is still there.
  vx::NcclCollective c(test_device());
  std::string file = dir + "/id";
  check(has(c.init(0, 2, nccl.c_str(), file.c_str(), no_wait), "Remove it"),
        "rank 0 refuses an id file that is already there");
}

// A rank that waits takes the id file as soon as it is there. So rank 0 makes
// the file only with a rename of a full file. To show this, the test blocks the
// name of the other file with a directory. Then the write must fail and leave
// no id file.
void test_rank_0_writes_the_id_with_one_rename() {
  vx::NcclUniqueId id;
  memset(&id, 'v', sizeof id);
  const std::string blocked = dir + "/blocked";
  mkdir((blocked + ".partial").c_str(), 0700);
  check(!vx::write_unique_id(blocked.c_str(), id).empty(),
        "rank 0 stops when it cannot write the other file");
  check(access(blocked.c_str(), F_OK) != 0,
        "rank 0 makes no id file when it cannot write the other file");

  const std::string file = dir + "/renamed";
  vx::NcclUniqueId read;
  check(vx::write_unique_id(file.c_str(), id).empty() &&
            vx::read_unique_id(file.c_str(), &read, no_wait).empty() &&
            memcmp(&read, &id, sizeof id) == 0,
        "the id file holds the id after the rename");
  check(access((file + ".partial").c_str(), F_OK) != 0,
        "the rename leaves no other file");
}

void test_a_rank_that_waits_too_long_stops() {
  vx::NcclCollective c(test_device());
  std::string file = dir + "/never";
  check(has(c.init(1, 2, nccl.c_str(), file.c_str(),
                   std::chrono::milliseconds(30)),
            "did not write"),
        "rank 1 stops when rank 0 never writes the id");
}

void test_an_all_reduce_goes_on_the_stream_with_its_arguments() {
  current_device = 3;
  vx::NcclCollective c(test_device());
  check(c.init(0, 1, nccl.c_str(), nullptr, no_wait).empty(), "one rank joins");
  current_device = 1;
  devices_set.clear();
  stream = &stream_a;
  std::string why = c.all_reduce_sum(gpu_memory, gpu_memory + 4, 4,
                                     vx::NcclFloat32, "all_reduce_f32");
  check(why.empty(), "the all-reduce runs");
  check(!strcmp(calls->last, "ncclAllReduce"), "it calls ncclAllReduce");
  check(calls->input == gpu_memory && calls->output == gpu_memory + 4,
        "it passes the input and the output");
  check(calls->count == 4, "it passes the count");
  check(calls->type == 7 && calls->op == 0,
        "it passes ncclFloat32 and ncclSum");
  check(calls->stream == &stream_a, "it runs on the stream that is set");
  check(devices_set.count(3) == 1, "it runs on the GPU that joined the job");
  check(current_device == 1, "it restores the current GPU");

  stream = nullptr;
  c.all_gather(gpu_memory, gpu_memory, 2, vx::NcclBfloat16, "gather_bf16");
  check(!strcmp(calls->last, "ncclAllGather"), "it calls ncclAllGather");
  check(calls->type == 9, "it passes ncclBfloat16");
  check(calls->stream == nullptr, "no stream set is the legacy stream");
  current_device = 0;
}

void test_bad_arguments_are_refused() {
  vx::NcclCollective c(test_device());
  float host[4];
  check(has(c.all_reduce_sum(gpu_memory, gpu_memory, 4, vx::NcclFloat32, "f"),
            "before vx_collective_init"),
        "an all-reduce before init is refused");
  c.init(0, 1, nccl.c_str(), nullptr, no_wait);
  check(has(c.all_reduce_sum(host, gpu_memory, 4, vx::NcclFloat32, "f"),
            "GPU memory"),
        "a host input is refused");
  check(has(c.all_gather(gpu_memory, host, 4, vx::NcclFloat32, "f"),
            "GPU memory"),
        "a host output is refused");
  check(has(c.all_reduce_sum(gpu_memory, gpu_memory, -1, vx::NcclFloat32, "f"),
            "a count of -1"),
        "a negative count is refused");
  check(c.all_reduce_sum(host, host, 0, vx::NcclFloat32, "f").empty(),
        "an empty all-reduce runs anywhere");
  check(has(c.init(0, 1, nccl.c_str(), nullptr, no_wait), "second time"),
        "a second init is refused");
}

void test_an_nccl_error_stops_with_its_message() {
  vx::NcclCollective c(test_device());
  c.init(0, 1, nccl.c_str(), nullptr, no_wait);
  calls->fail_next = 2;
  std::string why =
      c.all_reduce_sum(gpu_memory, gpu_memory, 1, vx::NcclFloat32, "sum_f32");
  check(has(why, "sum_f32 failed: fake system error"),
        "an NCCL error names the call and NCCL's message");
}

void test_finish_leaves_the_job_and_a_new_init_joins_again() {
  vx::NcclCollective c(test_device());
  check(has(c.finish(), "before vx_collective_init"),
        "finish before init is refused");
  c.init(0, 1, nccl.c_str(), nullptr, no_wait);
  const int destroys = calls->destroys;
  check(c.finish().empty(), "finish runs");
  check(calls->destroys == destroys + 1, "finish destroys the communicator");
  check(c.ranks() == 0 && c.rank() == -1, "after finish there is no job");
  check(c.init(0, 1, nccl.c_str(), nullptr, no_wait).empty(),
        "init after finish joins again");
}

void test_one_rank_with_no_gpu_copies() {
  vx::OneRankCollective c;
  float in[3] = {1, 2, 3}, out[3] = {0, 0, 0};
  check(has(c.copy(in, out, 3, 4, "f"), "before vx_collective_init"),
        "a copy before init is refused");
  check(has(c.init(0, 2), "one rank only"),
        "two ranks with no GPU are refused");
  check(c.init(0, 1).empty() && c.ranks() == 1 && c.rank() == 0,
        "one rank joins");
  check(c.copy(in, out, 3, 4, "f").empty() && out[0] == 1 && out[2] == 3,
        "the output gets the input");
  check(has(c.copy(in, out, -2, 4, "f"), "a count of -2"),
        "a negative count is refused");
  check(c.finish().empty() && c.ranks() == 0, "finish leaves the job");
}

void test_a_peer_buffer_needs_a_job_a_size_and_no_recording() {
  vx::NcclCollective c(test_device());
  void *b = nullptr;
  check(has(c.peer_buffer_create(16, &b), "before vx_collective_init"),
        "a peer buffer before init is refused");
  c.init(0, 1, nccl.c_str(), nullptr, no_wait);
  check(has(c.peer_buffer_create(0, &b), "above 0, but got 0"),
        "a peer buffer of 0 bytes is refused");
  recording = true;
  check(has(c.peer_buffer_create(16, &b), "records a CUDA graph"),
        "a peer buffer during a recording is refused");
  recording = false;
  uint64_t out[1];
  int x = 0;
  check(has(c.peer_buffer_pointers(&x, out), "did not make"),
        "the pointers of an unknown buffer are refused");
  check(has(c.peer_buffer_destroy(&x), "did not make"),
        "freeing an unknown buffer is refused");
}

void test_a_peer_buffer_of_one_rank_is_its_own_memory() {
  vx::NcclCollective c(test_device());
  c.init(0, 1, nccl.c_str(), nullptr, no_wait);
  events.clear();
  const char *last = calls->last;
  void *b = nullptr;
  check(c.peer_buffer_create(16, &b).empty() && b, "one rank makes a buffer");
  uint64_t out[1] = {0};
  check(c.peer_buffer_pointers(b, out).empty(), "its pointers are there");
  void *own = (void *)(uintptr_t)out[0];
  check(own && count_events("allocate", own) == 1,
        "the pointer is the memory that the rank allocated");
  check(calls->last == last, "one rank exchanges nothing through NCCL");
  recording = true;
  check(has(c.peer_buffer_destroy(b), "records a CUDA graph"),
        "freeing a buffer during a recording is refused");
  recording = false;
  check(c.peer_buffer_destroy(b).empty() && count_events("release", own) == 1,
        "destroy frees the memory");
  check(has(c.peer_buffer_destroy(b), "did not make"),
        "freeing a buffer twice is refused");
  void *kept = nullptr;
  c.peer_buffer_create(8, &kept);
  c.peer_buffer_pointers(kept, out);
  // The allocator can give the freed address again, so count from here.
  void *kept_part = (void *)(uintptr_t)out[0];
  const size_t released = count_events("release", kept_part);
  check(c.finish().empty() &&
            count_events("release", kept_part) == released + 1,
        "finish frees a buffer that is still there");
}

void test_two_ranks_address_each_others_part() {
  const std::string file = dir + "/peer_id";
  vx::NcclCollective zero(test_device()), one(test_device());
  std::string why[2], made[2], gone[2];
  uint64_t out[2][2] = {{0, 0}, {0, 0}};
  void *b[2] = {nullptr, nullptr};
  events.clear();
  auto rank = [&](int r, vx::NcclCollective &c) {
    event_rank = r;
    why[r] = c.init(r, 2, nccl.c_str(), file.c_str(), std::chrono::seconds(10));
    if (!why[r].empty()) {
      return;
    }
    made[r] = c.peer_buffer_create(32, &b[r]);
    if (made[r].empty()) {
      c.peer_buffer_pointers(b[r], out[r]);
      gone[r] = c.peer_buffer_destroy(b[r]);
    }
  };
  std::thread t1([&] { rank(1, one); });
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  std::thread t0([&] { rank(0, zero); });
  t0.join();
  t1.join();
  check(why[0].empty() && why[1].empty(), "two ranks join");
  check(made[0].empty() && made[1].empty(), "each rank makes its part");
  check(gone[0].empty() && gone[1].empty(), "each rank frees its part");
  void *part0 = (void *)(uintptr_t)out[0][0];
  void *part1 = (void *)(uintptr_t)out[0][1];
  check(part0 && part1 && part0 != part1, "rank 0 has two different parts");
  check(out[1][0] == out[0][0] && out[1][1] == out[0][1],
        "both ranks see the same part at each rank's index");
  bool part0_by_0 = false, part1_by_1 = false;
  for (const Event &e : events) {
    part0_by_0 |= e.what == "allocate" && e.ptr == part0 && e.rank == 0;
    part1_by_1 |= e.what == "allocate" && e.ptr == part1 && e.rank == 1;
  }
  check(part0_by_0 && part1_by_1, "rank r allocated the part at index r");
  check(count_events("open", part1) == 1 && count_events("open", part0) == 1,
        "each rank opened the other rank's part once");
  const size_t closed =
      std::max(first_event("close", part0), first_event("close", part1));
  check(closed < events.size() && first_event("release", part0) > closed &&
            first_event("release", part1) > closed,
        "no rank frees its part before both ranks close their address of it");
  // Rank r closes the part of the other rank, 1 - r.
  void *other_part[2] = {part1, part0};
  for (int r = 0; r < 2; ++r) {
    const size_t close = first_event("close", other_part[r]);
    bool waited = false;
    for (size_t i = first_event("open", other_part[r]); i < close; ++i) {
      waited |= events[i].what == "synchronize" && events[i].rank == r;
    }
    check(close < events.size() && waited,
          "each rank waits for its stream before it closes another rank's "
          "part");
  }
}

void test_a_peer_buffer_with_no_gpu_is_host_memory() {
  vx::OneRankCollective c;
  void *b = nullptr;
  check(has(c.peer_buffer_create(8, &b), "before vx_collective_init"),
        "a peer buffer before init is refused");
  c.init(0, 1);
  check(has(c.peer_buffer_create(-1, &b), "above 0"),
        "a negative size is refused");
  check(c.peer_buffer_create(8, &b).empty() && b, "one rank makes a buffer");
  uint64_t out[1] = {0};
  check(c.peer_buffer_pointers(b, out).empty() && out[0],
        "its pointer is there");
  const char *bytes = (const char *)(uintptr_t)out[0];
  bool zero = true;
  for (int i = 0; i < 8; ++i) {
    zero &= bytes[i] == 0;
  }
  check(zero, "the buffer starts at 0");
  check(c.peer_buffer_destroy(b).empty(), "destroy frees it");
  check(has(c.peer_buffer_pointers(b, out), "did not make"),
        "a freed buffer is unknown");
  c.finish();
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: collective_test <fake libnccl.so> <directory>\n");
    return 2;
  }
  nccl = argv[1];
  dir = argv[2];
  void *lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  calls = lib ? (FakeNcclCalls *)dlsym(lib, "fake_nccl_calls") : nullptr;
  if (!calls) {
    fprintf(stderr, "cannot load %s: %s\n", argv[1], dlerror());
    return 2;
  }
  test_a_missing_nccl_names_the_variable_that_finds_it();
  test_a_rank_outside_the_job_is_refused();
  test_init_in_a_recording_is_refused();
  test_two_ranks_need_the_id_file();
  test_both_ranks_join_with_the_id_of_rank_0();
  test_an_old_id_file_is_refused();
  test_rank_0_writes_the_id_with_one_rename();
  test_a_rank_that_waits_too_long_stops();
  test_an_all_reduce_goes_on_the_stream_with_its_arguments();
  test_bad_arguments_are_refused();
  test_an_nccl_error_stops_with_its_message();
  test_finish_leaves_the_job_and_a_new_init_joins_again();
  test_one_rank_with_no_gpu_copies();
  test_a_peer_buffer_needs_a_job_a_size_and_no_recording();
  test_a_peer_buffer_of_one_rank_is_its_own_memory();
  test_two_ranks_address_each_others_part();
  test_a_peer_buffer_with_no_gpu_is_host_memory();
  if (failures) {
    fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  printf("all collective checks passed\n");
  return 0;
}
