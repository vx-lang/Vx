//===- remote_kernel_output_test.cpp - Remote copies ------------*- C++ -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The host calls vx_remote_dispatch on buffers in its own memory. The client
// copies each buffer to the worker, the worker runs the kernel on the copies,
// and the client must copy the written values back into the host buffers.
//
// These dispatches go over one socket to a fake worker in a child process:
//
//   * A kernel the compiler generated (the payload carries `image=`) writes two
//     host buffers. Both must hold the kernel's values afterwards.
//     tests/backend/pass/gpu_that_shares_host_memory_gets_host_tensors.vx
//     is a program that compiles to this launch.
//   * The same kernel, with one output already on the worker. That output must
//     stay on the worker, and the other output must still reach the host.
//   * A matmul with `outkind=buffer`, which fills one host buffer.
//   * A view with an empty dimension. A worker cannot hold 0 bytes, so the
//     client must stage exactly one element and must not copy anything back.
//   * A 3x4 transposed view. The client must copy all 12 elements it reaches.
//     The same view inside a slot must also copy all 12.
//   * A 2x4 broadcast view, whose rows are the same 4 elements. The client
//     must copy exactly those 4.
//   * A 2x2 view with padded rows. The client must copy only the elements from
//     the view's first element to its last, in both directions. The view ends
//     at a page that cannot be read, so a longer copy crashes the test.
//   * Two views that share memory: one buffer passed twice, and the even and
//     the odd columns of one tensor. The kernel doubles the first view. The
//     copy back of the second view must not undo that.
//   * An empty view at the end of a buffer on the worker. Its handle is one
//     past the end of the buffer, so the client must stage one element for it.
//
// More cases call vx_memref_view_bytes directly in a child process. A view
// with an element type the runtime cannot copy, a negative size, a negative
// stride, or a span that does not fit in 64 bits must stop the program.
//
// The fake worker does not load the image. It runs a fixed function in its
// place: out1[i] = 2 * in[i] and out2[i] = in[i] + 100. For the views it
// doubles each element of the view. For two views it doubles each element of
// the first one.
//
// Driven by tests/integration_test/remote_kernel_output_test.rs.
//
//===----------------------------------------------------------------------===//

#include "../../runtime/vx_remote_client.h"
#include "test_worker.h"

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

using namespace vx_test;

int failures = 0;

void check(bool ok, const char *what) {
  if (!ok) {
    fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

/* --- the child: a fake worker -------------------------------------------- */

uint8_t child_body[1 << 20];
uint8_t child_reply[4096];

float *resolve_f32(uint64_t handle) {
  vx_remote_ref ref;
  if (!vx_remote_resolve(&worker_table, handle, &ref)) {
    return nullptr;
  }
  return (float *)((char *)ref.region->remote + ref.offset);
}

// The stand-in for a generated kernel. Arguments are in, out1, out2, each
// four f32 values.
int run_fake_kernel(const vx_wire_arg *args, int64_t num_args) {
  if (num_args != 3) {
    return 0;
  }
  const float *in = resolve_f32(args[0].handle);
  float *out1 = resolve_f32(args[1].handle);
  float *out2 = resolve_f32(args[2].handle);
  if (!in || !out1 || !out2) {
    return 0;
  }
  for (int i = 0; i < 4; ++i) {
    out1[i] = 2 * in[i];
    out2[i] = in[i] + 100;
  }
  return 1;
}

// The stand-in for a kernel on one rank-2 f32 view. It doubles each element.
// It fails when the client staged more or fewer bytes than the view reaches,
// or for an empty view, more or fewer than one element.
int run_view_kernel(const vx_wire_arg *args, int64_t num_args) {
  if (num_args != 1 || args[0].rank != 2) {
    return 0;
  }
  const int64_t *sizes = args[0].sizes;
  const int64_t *strides = args[0].strides;
  vx_remote_ref ref;
  if (!vx_remote_resolve(&worker_table, args[0].handle, &ref)) {
    return 0;
  }
  if (sizes[0] == 0 || sizes[1] == 0) {
    return ref.region->size - ref.offset == sizeof(float); // nothing to double
  }
  uint64_t reach = (uint64_t)((sizes[0] - 1) * strides[0] +
                              (sizes[1] - 1) * strides[1] + 1) *
                   sizeof(float);
  if (ref.region->size - ref.offset != reach) {
    return 0;
  }
  float *view = (float *)((char *)ref.region->remote + ref.offset);
  for (int64_t i = 0; i < sizes[0]; ++i) {
    for (int64_t j = 0; j < sizes[1]; ++j) {
      view[i * strides[0] + j * strides[1]] *= 2;
    }
  }
  return 1;
}

// The stand-in for a kernel on two rank-1 f32 views. It doubles each element
// of the first view and does not touch the second.
int run_double_first_kernel(const vx_wire_arg *args, int64_t num_args) {
  if (num_args != 2 || args[0].rank != 1 || args[1].rank != 1 ||
      !resolve_f32(args[1].handle)) {
    return 0;
  }
  float *first = resolve_f32(args[0].handle);
  if (!first) {
    return 0;
  }
  for (int64_t i = 0; i < args[0].sizes[0]; ++i) {
    first[i * args[0].strides[0]] *= 2;
  }
  return 1;
}

void run_worker(int fd) {
  worker_reset();

  for (;;) {
    uint32_t type = 0;
    uint64_t len = 0;
    int rc = vx_transport_recv(fd, &type, child_body, sizeof(child_body), &len);
    if (rc == VX_TRANSPORT_EOF) {
      _exit(0);
    }
    if (rc != VX_TRANSPORT_OK) {
      _exit(3);
    }

    vx_wire_reader r;
    vx_wire_writer w;
    vx_wire_reader_init(&r, child_body, (size_t)len);
    vx_wire_writer_init(&w, child_reply, sizeof(child_reply));

    if (type == VX_WIRE_TRANSFER) {
      vx_wire_transfer t;
      uint64_t handle = vx_wire_get_transfer(&r, &t) ? worker_transfer(&t) : 0;
      vx_wire_put_u64(&w, handle);
      if (vx_transport_send(fd, VX_WIRE_TRANSFER, child_reply, w.len) !=
          VX_TRANSPORT_OK) {
        _exit(4);
      }
    } else if (type == VX_WIRE_DISPATCH) {
      vx_wire_dispatch d;
      vx_wire_arg args[8];
      vx_wire_result results[4];
      int64_t num_results = 0;
      int ok = vx_wire_get_dispatch(&r, &d) && d.num_args <= 8;
      for (int64_t i = 0; ok && i < d.num_args; ++i) {
        ok = vx_wire_get_arg(&r, &args[i]);
      }
      if (ok) {
        if (strcmp((const char *)d.payload, "vx_gpu_kernel_view") == 0) {
          ok = run_view_kernel(args, d.num_args);
        } else if (strcmp((const char *)d.payload,
                          "vx_gpu_kernel_double_first") == 0) {
          ok = run_double_first_kernel(args, d.num_args);
        } else if (vx_payload_field(d.payload, (size_t)d.payload_len,
                                    "image=")) {
          ok = run_fake_kernel(args, d.num_args);
        } else {
          ok = worker_dispatch(&d, args, results, &num_results);
        }
      }
      vx_wire_put_results(&w, ok ? 0 : -1, results, ok ? num_results : 0);
      if (vx_transport_send(fd, VX_WIRE_DISPATCH, child_reply, w.len) !=
          VX_TRANSPORT_OK) {
        _exit(5);
      }
    } else if (type == VX_WIRE_FETCH) {
      uint64_t handle = 0, nbytes = 0;
      vx_remote_ref ref;
      if (!vx_wire_get_fetch(&r, &handle, &nbytes) ||
          !vx_remote_resolve(&worker_table, handle, &ref) ||
          nbytes > ref.region->size - ref.offset) {
        rc = vx_transport_send(fd, VX_WIRE_FETCH, nullptr, 0);
      } else {
        rc = vx_transport_send(fd, VX_WIRE_FETCH,
                               (const char *)ref.region->remote + ref.offset,
                               nbytes);
      }
      if (rc != VX_TRANSPORT_OK) {
        _exit(6);
      }
    } else if (type == VX_WIRE_FREE) {
      // The test arena never reuses memory, so a FREE only needs its reply.
      if (vx_transport_send(fd, VX_WIRE_FREE, nullptr, 0) != VX_TRANSPORT_OK) {
        _exit(7);
      }
    } else {
      _exit(8);
    }
  }
}

/* --- the parent: the host ------------------------------------------------ */

// A memref descriptor as generated code lays it out, for rank 1 and rank 2.
struct Desc1 {
  void *allocated;
  void *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
};
struct Desc2 {
  void *allocated;
  void *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
};

Desc1 vector_of_four(void *data) {
  Desc1 d = {data, data, 0, {4}, {1}};
  return d;
}

Desc2 two_by_two(void *data) {
  Desc2 d = {data, data, 0, {2, 2}, {2, 1}};
  return d;
}

uint8_t host_scratch[1 << 20];

// A payload in the shape the compiler writes for a kernel it generated. The
// image bytes are not real PTX, because the fake worker never loads them.
const char kKernelPayload[] = "vx_gpu_kernel_0\0abi=1\0topo=500\0"
                              "format=ptx\0image=not-real-ptx\0";

void test_a_generated_kernel_writes_both_host_outputs(int fd) {
  float in[4] = {1, 2, 3, 4};
  float out1[4] = {0, 0, 0, 0};
  float out2[4] = {0, 0, 0, 0};
  Desc1 d_in = vector_of_four(in);
  Desc1 d_out1 = vector_of_four(out1);
  Desc1 d_out2 = vector_of_four(out2);
  void *p_in = &d_in, *p_out1 = &d_out1, *p_out2 = &d_out2;
  void *device_args[3] = {&p_in, &p_out1, &p_out2};
  const int32_t tags[3] = {VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 1),
                           VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 1),
                           VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 1)};
  vx_wire_arg args[3];

  check(vx_remote_dispatch(fd, kKernelPayload, sizeof(kKernelPayload),
                           device_args, tags, 3, args, host_scratch,
                           sizeof(host_scratch)) == 1,
        "the generated kernel's dispatch succeeds");
  for (int i = 0; i < 4; ++i) {
    char what[64];
    snprintf(what, sizeof(what), "first output [%d] == %g", i,
             (double)(2 * in[i]));
    check(out1[i] == 2 * in[i], what);
    snprintf(what, sizeof(what), "second output [%d] == %g", i,
             (double)(in[i] + 100));
    check(out2[i] == in[i] + 100, what);
  }
  check(in[0] == 1 && in[1] == 2 && in[2] == 3 && in[3] == 4,
        "the input is unchanged");
}

void test_an_output_on_the_worker_stays_there(int fd) {
  float in[4] = {1, 2, 3, 4};
  float zeros[4] = {0, 0, 0, 0};
  float out2[4] = {0, 0, 0, 0};

  // Put the first output on the worker before the dispatch, as `transfer`
  // does in a program. Its descriptor then holds a handle, not a pointer.
  uint64_t resident = vx_remote_transfer(fd, zeros, sizeof(zeros), VX_DTYPE_F32,
                                         host_scratch, sizeof(host_scratch));
  check(resident != 0, "the resident output was placed on the worker");

  Desc1 d_in = vector_of_four(in);
  Desc1 d_out1 = vector_of_four((void *)(uintptr_t)resident);
  Desc1 d_out2 = vector_of_four(out2);
  void *p_in = &d_in, *p_out1 = &d_out1, *p_out2 = &d_out2;
  void *device_args[3] = {&p_in, &p_out1, &p_out2};
  const int32_t tags[3] = {VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 1),
                           VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 1),
                           VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 1)};
  vx_wire_arg args[3];

  check(vx_remote_dispatch(fd, kKernelPayload, sizeof(kKernelPayload),
                           device_args, tags, 3, args, host_scratch,
                           sizeof(host_scratch)) == 1,
        "the dispatch with a resident output succeeds");
  check(d_out1.aligned == (void *)(uintptr_t)resident,
        "the resident output's descriptor still holds its handle");

  float fetched[4] = {0, 0, 0, 0};
  check(vx_remote_fetch(fd, resident, fetched, sizeof(fetched)),
        "the resident output can be fetched");
  for (int i = 0; i < 4; ++i) {
    char what[64];
    snprintf(what, sizeof(what), "resident output [%d] == %g", i,
             (double)(2 * in[i]));
    check(fetched[i] == 2 * in[i], what);
    snprintf(what, sizeof(what), "host output [%d] == %g", i,
             (double)(in[i] + 100));
    check(out2[i] == in[i] + 100, what);
  }
  check(vx_remote_free(fd, resident), "the resident output is freed");
}

void test_a_matmul_fills_a_host_buffer(int fd) {
  // A = [1 2; 3 4], B = [5 6; 7 8], A * B = [19 22; 43 50].
  float a[4] = {1, 2, 3, 4};
  float b[4] = {5, 6, 7, 8};
  float c[4] = {0, 0, 0, 0};
  const float expected[4] = {19, 22, 43, 50};
  Desc2 d_a = two_by_two(a);
  Desc2 d_b = two_by_two(b);
  Desc2 d_c = two_by_two(c);
  void *p_a = &d_a, *p_b = &d_b, *p_c = &d_c;
  void *device_args[3] = {&p_a, &p_b, &p_c};
  const int32_t tags[3] = {VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 2),
                           VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 2),
                           VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 2)};
  static const char payload[] = "vx_npu_kernel_1\0kind=matmul\0"
                                "roles=a:0,b:1,out:2\0outkind=buffer\0"
                                "topo=1113\0toponame=DecodeWorker\0";
  vx_wire_arg args[3];

  check(vx_remote_dispatch(fd, payload, sizeof(payload), device_args, tags, 3,
                           args, host_scratch, sizeof(host_scratch)) == 1,
        "the matmul's dispatch succeeds");
  for (int i = 0; i < 4; ++i) {
    char what[64];
    snprintf(what, sizeof(what), "C[%d] == %g", i, (double)expected[i]);
    check(c[i] == expected[i], what);
  }
}

const char kViewPayload[] = "vx_gpu_kernel_view\0abi=1\0topo=500\0"
                            "format=ptx\0image=not-real-ptx\0";

// Dispatch the view kernel on one rank-2 view over `data`. With `in_slot`, the
// argument is a slot that holds the view, as generated code passes a local
// tensor.
int dispatch_view(int fd, float *data, int64_t offset, int64_t rows,
                  int64_t cols, int64_t row_stride, int64_t col_stride = 1,
                  bool in_slot = false) {
  Desc2 d = {data, data, offset, {rows, cols}, {row_stride, col_stride}};
  Desc1 slot = {&d, &d, 0, {1}, {1}};
  void *p = in_slot ? (void *)&slot : (void *)&d;
  void *device_args[1] = {&p};
  const int32_t tags[1] = {in_slot ? VX_ABI_SLOT_TAG(VX_DTYPE_F32, 2)
                                   : VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 2)};
  vx_wire_arg args[1];
  return vx_remote_dispatch(fd, kViewPayload, sizeof(kViewPayload), device_args,
                            tags, 1, args, host_scratch, sizeof(host_scratch));
}

void test_an_empty_view_stages_one_element(int fd) {
  float data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  check(dispatch_view(fd, data, 0, 2, 0, 4) == 1,
        "the 2x0 view's dispatch succeeds");
  // A dimension with stride 0 repeats one element, as a broadcast does. When
  // its size is 0 the view is still empty.
  check(dispatch_view(fd, data, 0, 2, 0, 4, 0) == 1,
        "the dispatch of a 2x0 view whose empty dimension has stride 0 "
        "succeeds");
  for (int i = 0; i < 8; ++i) {
    char what[64];
    snprintf(what, sizeof(what), "element %d under the empty views == %d", i,
             i + 1);
    check(data[i] == (float)(i + 1), what);
  }
}

void check_transposed_view(int fd, bool in_slot) {
  // A 3x4 view with strides (1, 3) reaches all 12 elements. Element (i, j) is
  // data[i + 3 * j]. sizes[0] * strides[0] is only 3.
  const char *where = in_slot ? "in a slot" : "as a tensor";
  float data[12];
  for (int i = 0; i < 12; ++i) {
    data[i] = (float)i;
  }
  char what[96];
  snprintf(what, sizeof(what),
           "the dispatch of the transposed view %s succeeds", where);
  check(dispatch_view(fd, data, 0, 3, 4, 1, 3, in_slot) == 1, what);
  for (int i = 0; i < 12; ++i) {
    snprintf(what, sizeof(what), "transposed view %s: element %d == %d", where,
             i, 2 * i);
    check(data[i] == (float)(2 * i), what);
  }
}

void test_a_transposed_view_copies_all_it_reaches(int fd) {
  check_transposed_view(fd, false);
  check_transposed_view(fd, true);
}

void test_a_broadcast_view_copies_one_row(int fd) {
  // A 2x4 view with strides (0, 1) repeats the row data[0..3], as a broadcast
  // does. The fake worker fails unless it gets exactly those 4 elements. It
  // doubles each element once per row, so each one comes back 4 times larger.
  float data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  const float after[8] = {4, 8, 12, 16, 5, 6, 7, 8};
  check(dispatch_view(fd, data, 0, 2, 4, 0, 1) == 1,
        "the broadcast view's dispatch succeeds");
  for (int i = 0; i < 8; ++i) {
    char what[64];
    snprintf(what, sizeof(what), "broadcast view: element %d == %g", i,
             (double)after[i]);
    check(data[i] == after[i], what);
  }
}

const char kDoubleFirstPayload[] = "vx_gpu_kernel_double_first\0abi=1\0"
                                   "topo=500\0format=ptx\0image=not-real-ptx\0";

// Dispatch the kernel that doubles the first of two rank-1 views of `data`.
int dispatch_two_views(int fd, float *data, int64_t offset0, int64_t stride0,
                       int64_t offset1, int64_t stride1) {
  Desc1 first = {data, data, offset0, {4}, {stride0}};
  Desc1 second = {data, data, offset1, {4}, {stride1}};
  void *p_first = &first, *p_second = &second;
  void *device_args[2] = {&p_first, &p_second};
  const int32_t tags[2] = {VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 1),
                           VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 1)};
  vx_wire_arg args[2];
  return vx_remote_dispatch(fd, kDoubleFirstPayload,
                            sizeof(kDoubleFirstPayload), device_args, tags, 2,
                            args, host_scratch, sizeof(host_scratch));
}

void test_views_that_share_memory_keep_the_result(int fd) {
  // One buffer, passed as the first view and as the second view.
  float same[4] = {1, 2, 3, 4};
  const float same_after[4] = {2, 4, 6, 8};
  check(dispatch_two_views(fd, same, 0, 1, 0, 1) == 1,
        "the dispatch of one buffer passed twice succeeds");
  for (int i = 0; i < 4; ++i) {
    char what[64];
    snprintf(what, sizeof(what), "buffer passed twice: element %d == %g", i,
             (double)same_after[i]);
    check(same[i] == same_after[i], what);
  }

  // A 2x4 tensor. The first view is the even columns, elements 0, 2, 4 and 6.
  // The second view is the odd columns. Each view's bytes cover elements of
  // the other.
  float columns[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  const float columns_after[8] = {2, 2, 6, 4, 10, 6, 14, 8};
  check(dispatch_two_views(fd, columns, 0, 2, 1, 2) == 1,
        "the dispatch of the even and the odd columns succeeds");
  for (int i = 0; i < 8; ++i) {
    char what[64];
    snprintf(what, sizeof(what), "even and odd columns: element %d == %g", i,
             (double)columns_after[i]);
    check(columns[i] == columns_after[i], what);
  }
}

void test_an_empty_view_at_the_end_of_a_worker_buffer(int fd) {
  float four[4] = {1, 2, 3, 4};
  uint64_t resident = vx_remote_transfer(fd, four, sizeof(four), VX_DTYPE_F32,
                                         host_scratch, sizeof(host_scratch));
  check(resident != 0, "the buffer was placed on the worker");
  // A 0x4 view that starts at element 4, one past the end of the buffer.
  check(dispatch_view(fd, (float *)(uintptr_t)resident, 4, 0, 4, 4) == 1,
        "the dispatch of an empty view at the end of a worker buffer "
        "succeeds");
  check(vx_remote_free(fd, resident), "the worker buffer is freed");
}

// The exit code of a child that called abort(). The child catches SIGABRT and
// exits, so the system writes no crash report or core dump.
const int kAborted = 86;

void exit_on_abort(int) { _exit(kAborted); }

// Calls vx_memref_view_bytes in a child process. Returns true when the child
// called abort() and printed `message`.
bool view_bytes_stops_with(int32_t rank, const int64_t *sizes,
                           const int64_t *strides, const char *message,
                           int32_t elem = VX_DTYPE_F32) {
  int pipe_fds[2];
  if (pipe(pipe_fds) != 0) {
    return false;
  }
  pid_t pid = fork();
  if (pid < 0) {
    return false;
  }
  if (pid == 0) {
    signal(SIGABRT, exit_on_abort);
    dup2(pipe_fds[1], STDERR_FILENO);
    vx_memref_view_bytes(sizes, strides, rank, elem);
    _exit(0);
  }
  close(pipe_fds[1]);
  char out[256] = {0};
  size_t got = 0;
  ssize_t n;
  while (got < sizeof(out) - 1 &&
         (n = read(pipe_fds[0], out + got, sizeof(out) - 1 - got)) > 0) {
    got += (size_t)n;
  }
  close(pipe_fds[0]);
  int status = 0;
  waitpid(pid, &status, 0);
  return WIFEXITED(status) && WEXITSTATUS(status) == kAborted &&
         strstr(out, message) != nullptr;
}

bool vector_stops_with(int64_t size, int64_t stride, const char *message,
                       int32_t elem = VX_DTYPE_F32) {
  return view_bytes_stops_with(1, &size, &stride, message, elem);
}

void test_a_view_the_runtime_cannot_copy_stops_the_program() {
  // Code 0 is what the compiler writes for `index`, for pointers and for any
  // type it does not list. A valid program can make this view.
  check(vector_stops_with(4, 1,
                          "cannot copy a tensor with the element type "
                          "code 0",
                          0),
        "a view with element type code 0 stops the program");
}

void test_a_broken_view_stops_the_program() {
  check(vector_stops_with(-1, 1, "has the negative size -1"),
        "a view with a negative size stops the program");
  check(vector_stops_with(4, -1, "has the negative stride -1"),
        "a view with a negative stride stops the program");
}

void test_a_view_whose_span_overflows_stops_the_program() {
  const int64_t two_62 = (int64_t)1 << 62;
  // The last element is at index 2^62 * 4 = 2^64. The product overflows.
  check(vector_stops_with(two_62 + 1, 4, "spans 2^64 elements or more"),
        "a view whose step reaches element 2^64 stops the program");
  // Each dimension steps 2 * 2^62 = 2^63, which fits. The sum, 2^64, overflows.
  const int64_t sizes[2] = {3, 3};
  const int64_t strides[2] = {two_62, two_62};
  check(view_bytes_stops_with(2, sizes, strides, "spans 2^64 elements or more"),
        "a view whose steps add up to element 2^64 stops the program");
  // The last element is at index (2^32 + 1) * (2^32 - 1) = 2^64 - 1, which
  // fits. The count of elements, 2^64, does not.
  check(vector_stops_with(((int64_t)1 << 32) + 2, ((int64_t)1 << 32) - 1,
                          "spans 2^64 elements or more"),
        "a view of 2^64 elements stops the program");
  // The last element is at index 2^62, and 2^62 + 1 f32 values need more than
  // 2^64 bytes.
  check(vector_stops_with(two_62 + 1, 1, "spans 2^64 bytes or more"),
        "a view of more than 2^64 bytes stops the program");
}

void test_a_padded_view_copies_only_what_it_reaches(int fd) {
  // Seven elements that end where an unreadable page starts. The view starts
  // at element 1, has two rows of two, and its rows are 4 elements apart. It
  // reaches elements 1, 2, 5 and 6, so the copy is elements 1 to 6.
  long page = sysconf(_SC_PAGESIZE);
  char *pages = (char *)mmap(nullptr, 2 * page, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (pages == MAP_FAILED || mprotect(pages + page, page, PROT_NONE) != 0) {
    check(false, "the guard page was set up");
    return;
  }
  float *data = (float *)(pages + page) - 7;
  const float before[7] = {10, 11, 12, 13, 14, 15, 16};
  const float after[7] = {10, 22, 24, 13, 14, 30, 32};
  memcpy(data, before, sizeof(before));

  check(dispatch_view(fd, data, 1, 2, 2, 4) == 1,
        "the padded view's dispatch succeeds");
  for (int i = 0; i < 7; ++i) {
    char what[64];
    snprintf(what, sizeof(what), "element %d == %g", i, (double)after[i]);
    check(data[i] == after[i], what);
  }
  munmap(pages, 2 * page);
}

} // namespace

int main() {
  int sv[2];
  if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) {
    fprintf(stderr, "FAIL: socketpair\n");
    return 1;
  }
  pid_t pid = fork();
  if (pid < 0) {
    fprintf(stderr, "FAIL: fork\n");
    return 1;
  }
  if (pid == 0) {
    close(sv[0]);
    run_worker(sv[1]);
  }
  close(sv[1]);

  test_a_generated_kernel_writes_both_host_outputs(sv[0]);
  test_an_output_on_the_worker_stays_there(sv[0]);
  test_a_matmul_fills_a_host_buffer(sv[0]);
  test_an_empty_view_stages_one_element(sv[0]);
  test_a_transposed_view_copies_all_it_reaches(sv[0]);
  test_a_broadcast_view_copies_one_row(sv[0]);
  test_views_that_share_memory_keep_the_result(sv[0]);
  test_an_empty_view_at_the_end_of_a_worker_buffer(sv[0]);
  test_a_view_the_runtime_cannot_copy_stops_the_program();
  test_a_broken_view_stops_the_program();
  test_a_view_whose_span_overflows_stops_the_program();
  // Last, because the code before the fix crashes on it.
  test_a_padded_view_copies_only_what_it_reaches(sv[0]);

  close(sv[0]);
  int status = 0;
  waitpid(pid, &status, 0);
  check(WIFEXITED(status) && WEXITSTATUS(status) == 0,
        "the fake worker exited cleanly");

  if (failures) {
    fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  printf("all checks passed\n");
  return 0;
}
