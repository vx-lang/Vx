//===- transport_test.cpp - A dispatch across a real socket -----*- C++ -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// The same dispatch loopback_test.cpp runs in one process, run across a socket
// between two of them. `fork`, a `socketpair`, and a child that owns the
// worker's memory and never sees the parent's.
//
// What this adds over the in-process test is the transport, and specifically
// the two things that cannot be exercised without a real descriptor:
//
//   * **Short reads.** A stream socket delivers what has arrived, not what was
//     asked for. The 512 KiB transfer below is far past any socket buffer, so
//     `read` returns partial counts and a transport that treated one call as
//     one message loses the rest. Verified by removing the loop: the child
//     fails on the first large message and the parent dies of SIGPIPE.
//
//     Short *writes* are not exercised here, and the test does not pretend
//     otherwise -- on a blocking unix socket both platforms queue the whole
//     body or block, so removing the write loop changes nothing that this test
//     can see. The loop stays because a partial write is POSIX-legal and does
//     happen on TCP under memory pressure, which is step 3's transport, but it
//     is unproven code until something proves it.
//   * **Separate address spaces.** The child resolves handles in *its* region
//     table against *its* memory. A pointer that leaked across instead of a
//     handle would have worked in the loopback test and cannot work here, which
//     is the property that makes this worth the fork.
//
// What it deliberately does not test is the artifact assumption. Parent and
// child are the same binary, so the outlined kernel is trivially present on
// both sides. On two real machines that has to be arranged, and nothing here
// says it has been.
//
// Driven by tests/integration_test/transport_test.rs.
//
//===----------------------------------------------------------------------===//

#include "../../runtime/vx_transport.h"
#include "test_worker.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
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

const float kA[4] = {1, 2, 3, 4};
const float kB[4] = {5, 6, 7, 8};
const float kExpected[4] = {19, 22, 43, 50};

// Two sizes that have to differ, and did not on the first attempt.
//
// kBigElems is far larger than a socket buffer (8 KiB on macOS, ~208 KiB on
// Linux) so that write and read are forced to return partial counts, and still
// smaller than the child's receive buffer so the message is meant to succeed.
// kHugeElems is larger than that buffer, so it is meant to be refused. Setting
// both to 2 MiB made the first test exercise the second's path and fail.
const size_t kBigElems =
    128 * 1024; /* 512 KiB of f32, under the 1 MiB buffer */
const size_t kHugeElems = 512 * 1024; /* 2 MiB of f32, over it */

/* --- the child: a worker that owns its own memory ------------------------ */

// Replies are small; a dispatch body is bounded by its arguments.
uint8_t child_body[1 << 20];
uint8_t child_reply[4096];

void run_worker(int fd) {
  worker_reset();

  for (;;) {
    uint32_t type = 0;
    uint64_t len = 0;
    int rc = vx_transport_recv(fd, &type, child_body, sizeof(child_body), &len);

    if (rc == VX_TRANSPORT_EOF) {
      _exit(failures ? 1 : 0);
    }
    if (rc == VX_TRANSPORT_TOO_LARGE) {
      /* Drained, so the stream is still usable: answer with a zero handle and
         keep serving, which is what makes this recoverable rather than fatal.
       */
      vx_wire_writer w;
      vx_wire_writer_init(&w, child_reply, sizeof(child_reply));
      vx_wire_put_u64(&w, 0);
      if (vx_transport_send(fd, VX_WIRE_TRANSFER, child_reply, w.len) !=
          VX_TRANSPORT_OK) {
        _exit(2);
      }
      continue;
    }
    if (rc != VX_TRANSPORT_OK) {
      _exit(3);
    }

    if (type == VX_WIRE_TRANSFER) {
      vx_wire_reader r;
      vx_wire_transfer t;
      vx_wire_writer w;
      uint64_t handle = 0;

      vx_wire_reader_init(&r, child_body, (size_t)len);
      if (vx_wire_get_transfer(&r, &t)) {
        handle = worker_transfer(&t);
      }
      vx_wire_writer_init(&w, child_reply, sizeof(child_reply));
      vx_wire_put_u64(&w, handle);
      if (vx_transport_send(fd, VX_WIRE_TRANSFER, child_reply, w.len) !=
          VX_TRANSPORT_OK) {
        _exit(4);
      }
    } else if (type == VX_WIRE_DISPATCH) {
      vx_wire_reader r;
      vx_wire_dispatch d;
      vx_wire_arg args[8];
      vx_wire_result results[4];
      int64_t num_results = 0;
      int ok;
      vx_wire_writer w;

      vx_wire_reader_init(&r, child_body, (size_t)len);
      ok = vx_wire_get_dispatch(&r, &d) && d.num_args <= 8;
      for (int64_t i = 0; ok && i < d.num_args; ++i) {
        ok = vx_wire_get_arg(&r, &args[i]);
      }
      if (ok) {
        ok = worker_dispatch(&d, args, results, &num_results);
      }

      vx_wire_writer_init(&w, child_reply, sizeof(child_reply));
      vx_wire_put_results(&w, ok ? 0 : -1, results, ok ? num_results : 0);
      if (vx_transport_send(fd, VX_WIRE_DISPATCH, child_reply, w.len) !=
          VX_TRANSPORT_OK) {
        _exit(5);
      }
    } else if (type == VX_WIRE_FETCH) {
      vx_wire_reader r;
      uint64_t handle = 0, nbytes = 0;
      vx_remote_ref ref;

      vx_wire_reader_init(&r, child_body, (size_t)len);
      if (!vx_wire_get_fetch(&r, &handle, &nbytes) ||
          !vx_remote_resolve(&worker_table, handle, &ref) ||
          nbytes > ref.region->size - ref.offset) {
        /* An empty reply says "no", without inventing bytes for a handle that
           named nothing or a length that ran past what it named. */
        if (vx_transport_send(fd, VX_WIRE_FETCH, nullptr, 0) !=
            VX_TRANSPORT_OK) {
          _exit(7);
        }
      } else {
        const char *src = (const char *)ref.region->remote + ref.offset;
        if (vx_transport_send(fd, VX_WIRE_FETCH, src, nbytes) !=
            VX_TRANSPORT_OK) {
          _exit(8);
        }
      }
    } else {
      _exit(6);
    }
  }
}

/* --- the parent ---------------------------------------------------------- */

uint8_t host_body[1 << 22];
uint8_t host_reply[4096];

uint64_t stage(int fd, const void *data, uint64_t nbytes, int64_t rows,
               int64_t cols) {
  const int64_t sizes[2] = {rows, cols};
  vx_wire_writer w;
  vx_wire_reader r;
  uint32_t type = 0;
  uint64_t len = 0;
  uint64_t handle = 0;

  vx_wire_writer_init(&w, host_body, sizeof(host_body));
  vx_wire_put_transfer(&w, VX_DTYPE_F32, 2, sizes, data, nbytes);
  if (w.overflow) {
    return 0;
  }
  if (vx_transport_send(fd, VX_WIRE_TRANSFER, host_body, w.len) !=
      VX_TRANSPORT_OK) {
    return 0;
  }
  if (vx_transport_recv(fd, &type, host_reply, sizeof(host_reply), &len) !=
      VX_TRANSPORT_OK) {
    return 0;
  }
  vx_wire_reader_init(&r, host_reply, (size_t)len);
  return vx_wire_get_u64(&r, &handle) ? handle : 0;
}

void run_host(int fd) {
  // --- a 2x2 matmul, end to end across the socket ------------------------
  uint64_t ha = stage(fd, kA, sizeof(kA), 2, 2);
  uint64_t hb = stage(fd, kB, sizeof(kB), 2, 2);
  check(ha != 0 && hb != 0, "both operands staged across the socket");
  check(ha != hb, "and got different handles");

  static const char payload[] = "vx_npu_kernel_0\0kind=matmul\0"
                                "roles=a:0,b:1,out:2\0outkind=slot\0"
                                "topo=1113\0toponame=DecodeWorker\0";
  vx_wire_arg args[3];
  memset(args, 0, sizeof(args));
  args[0].tag = VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 2);
  args[0].handle = ha;
  args[0].rank = 2;
  args[0].sizes[0] = args[0].sizes[1] = 2;
  args[0].strides[0] = 2;
  args[0].strides[1] = 1;
  args[1] = args[0];
  args[1].handle = hb;
  args[2].tag = VX_ABI_SLOT_TAG(VX_DTYPE_F32, 2);

  {
    vx_wire_writer w;
    vx_wire_reader r;
    uint32_t type = 0;
    uint64_t len = 0;
    int32_t status = -1;
    int64_t count = 0;
    vx_wire_result got;

    vx_wire_writer_init(&w, host_body, sizeof(host_body));
    vx_wire_put_dispatch(&w, payload, sizeof(payload), args, 3);
    check(vx_transport_send(fd, VX_WIRE_DISPATCH, host_body, w.len) ==
              VX_TRANSPORT_OK,
          "the dispatch goes out");
    check(vx_transport_recv(fd, &type, host_reply, sizeof(host_reply), &len) ==
              VX_TRANSPORT_OK,
          "and a reply comes back");
    check(type == VX_WIRE_DISPATCH, "the reply is for the dispatch");

    vx_wire_reader_init(&r, host_reply, (size_t)len);
    check(vx_wire_get_results_header(&r, &status, &count), "the reply decodes");
    check(status == 0, "the worker reports success");
    check(count == 1, "one slot produced one result");
    if (count == 1 && vx_wire_get_result(&r, &got)) {
      check(got.rank == 2 && got.sizes[0] == 2 && got.sizes[1] == 2,
            "the result is 2x2");

      // A handle, minted in the child's address space, meaningful only there.
      check(vx_remote_addr_is_handle(got.handle),
            "the result is named by a handle, not an address");
      check(vx_remote_addr_worker(got.handle) == 3,
            "and it names the worker that produced it");

      // The parent has no way to read it, which is the point -- so ask the
      // child, by dispatching again with the result as an operand. If the
      // handle did not mean anything on that side, this cannot work.
      vx_wire_arg again[3];
      memset(again, 0, sizeof(again));
      again[0].tag = VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 2);
      again[0].handle = got.handle;
      again[0].rank = 2;
      again[0].sizes[0] = again[0].sizes[1] = 2;
      again[0].strides[0] = 2;
      again[0].strides[1] = 1;
      again[1] = again[0];
      again[1].handle = hb;
      again[2].tag = VX_ABI_SLOT_TAG(VX_DTYPE_F32, 2);

      vx_wire_writer w2;
      vx_wire_writer_init(&w2, host_body, sizeof(host_body));
      vx_wire_put_dispatch(&w2, payload, sizeof(payload), again, 3);
      check(vx_transport_send(fd, VX_WIRE_DISPATCH, host_body, w2.len) ==
                VX_TRANSPORT_OK,
            "a second dispatch naming the first result goes out");

      uint64_t len2 = 0;
      int32_t status2 = -1;
      int64_t count2 = 0;
      check(vx_transport_recv(fd, &type, host_reply, sizeof(host_reply),
                              &len2) == VX_TRANSPORT_OK,
            "and is answered");
      vx_wire_reader r2;
      vx_wire_reader_init(&r2, host_reply, (size_t)len2);
      check(vx_wire_get_results_header(&r2, &status2, &count2),
            "the second reply decodes");
      check(status2 == 0,
            "a handle returned by one dispatch is usable by the next");

      // And now read C back, which is the only way the parent can see it: it
      // has no access to the child's memory, and the handle is deliberately
      // not dereferenceable here.
      vx_wire_writer wf;
      vx_wire_writer_init(&wf, host_body, sizeof(host_body));
      vx_wire_put_fetch(&wf, got.handle, sizeof(kExpected));
      check(vx_transport_send(fd, VX_WIRE_FETCH, host_body, wf.len) ==
                VX_TRANSPORT_OK,
            "the fetch goes out");

      uint64_t flen = 0;
      check(vx_transport_recv(fd, &type, host_reply, sizeof(host_reply),
                              &flen) == VX_TRANSPORT_OK,
            "the fetch is answered");
      check(flen == sizeof(kExpected), "with the bytes that were asked for");
      if (flen == sizeof(kExpected)) {
        float c[4];
        memcpy(c, host_reply, sizeof(c));
        for (int i = 0; i < 4; ++i) {
          char what[80];
          snprintf(what, sizeof(what), "C[%d] == %g, across a socket", i,
                   (double)kExpected[i]);
          check(c[i] == kExpected[i], what);
        }
      }

      // A handle the worker never minted must not produce bytes.
      vx_wire_writer wb;
      vx_wire_writer_init(&wb, host_body, sizeof(host_body));
      vx_wire_put_fetch(&wb, vx_remote_addr(9, 1 << 20), sizeof(kExpected));
      check(vx_transport_send(fd, VX_WIRE_FETCH, host_body, wb.len) ==
                VX_TRANSPORT_OK,
            "a fetch for an unknown handle goes out");
      uint64_t blen = 1;
      check(vx_transport_recv(fd, &type, host_reply, sizeof(host_reply),
                              &blen) == VX_TRANSPORT_OK,
            "and is answered");
      check(blen == 0, "with nothing, rather than with invented bytes");
    }
  }

  // --- a body far larger than any socket buffer --------------------------
  //
  // 2 MiB in one message, so write and read are both forced to return partial
  // counts. A transport that treated one call as one message truncates here.
  {
    float *big = (float *)calloc(kBigElems, sizeof(float));
    check(big != nullptr, "the large operand allocates");
    if (big) {
      big[0] = 1.0f;
      big[kBigElems - 1] = 2.0f;
      uint64_t h =
          stage(fd, big, kBigElems * sizeof(float), 1, (int64_t)kBigElems);
      check(h != 0, "a multi-megabyte transfer survives the socket");
      free(big);
    }
  }

  // --- a message the peer cannot hold ------------------------------------
  //
  // The child's receive buffer is 1 MiB; this is larger. It must drain the body
  // and keep serving rather than leaving the stream positioned mid-message.
  {
    size_t huge_elems = kHugeElems;
    float *huge = (float *)calloc(huge_elems, sizeof(float));
    check(huge != nullptr, "the oversized operand allocates");
    if (huge) {
      uint64_t h =
          stage(fd, huge, huge_elems * sizeof(float), 1, (int64_t)huge_elems);
      check(h == 0, "an oversized message is refused rather than half-read");
      free(huge);
    }
  }

  // --- and the stream still works ----------------------------------------
  //
  // This is the check the drain exists for. Had the oversized body been left in
  // the socket, this transfer's header would be read out of the middle of it.
  {
    uint64_t h = stage(fd, kA, sizeof(kA), 2, 2);
    check(h != 0, "the stream is still usable after an oversized message");
  }
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
    _exit(0);
  }

  close(sv[1]);
  run_host(sv[0]);
  close(sv[0]); /* the child's recv returns EOF and it exits */

  int status = 0;
  waitpid(pid, &status, 0);
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    fprintf(stderr, "FAIL: worker exited with status %d\n", status);
    ++failures;
  }

  if (failures) {
    fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  printf("all checks passed\n");
  return 0;
}
