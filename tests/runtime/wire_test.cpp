//===- wire_test.cpp - Encoding a dispatch for another machine --*- C++ -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// Exercises runtime/vx_wire.h with no GPU and no network.
//
// Two kinds of check, and the second is the reason this file is worth more than
// the round trips in it.
//
// Round trips pin the format: an argument's shape on the wire is decided by its
// ABI tag, so these are the tests that fail if the tag encoding and the wire
// stop agreeing.
//
// Refusals pin the behaviour on input that is wrong. A short read on a socket
// is the normal case rather than the exceptional one, and a length field is
// exactly what a bug or an attacker uses to make a reader walk off the end. The
// decoder has to refuse a truncated message, a length longer than the bytes
// present, and a rank that would index past a fixed array -- and it has to do
// it by returning zero, not by reading and then noticing.
//
// Driven by tests/integration_test/wire_test.rs.
//
//===----------------------------------------------------------------------===//

#include "../../runtime/vx_remote_region.h"
#include "../../runtime/vx_wire.h"

#include <cstdio>
#include <cstring>

namespace {

int failures = 0;

void check(bool ok, const char *what) {
  if (!ok) {
    fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

uint8_t buf[4096];

vx_wire_writer writer() {
  vx_wire_writer w;
  memset(buf, 0, sizeof(buf));
  vx_wire_writer_init(&w, buf, sizeof(buf));
  return w;
}

void test_scalars_round_trip() {
  vx_wire_writer w = writer();
  vx_wire_reader r;
  vx_wire_arg a, out;

  memset(&a, 0, sizeof(a));
  a.tag = VX_ABI_KIND_I32;
  int32_t v = -12345;
  memcpy(a.scalar, &v, sizeof(v));
  vx_wire_put_arg(&w, &a);

  memset(&a, 0, sizeof(a));
  a.tag = VX_ABI_KIND_F64;
  double d = 2.5;
  memcpy(a.scalar, &d, sizeof(d));
  vx_wire_put_arg(&w, &a);

  check(!w.overflow, "scalars fit");

  vx_wire_reader_init(&r, buf, w.len);
  check(vx_wire_get_arg(&r, &out), "i32 decodes");
  check(out.tag == VX_ABI_KIND_I32, "i32 tag survives");
  int32_t got_i = 0;
  memcpy(&got_i, out.scalar, sizeof(got_i));
  check(got_i == -12345, "i32 value survives");

  check(vx_wire_get_arg(&r, &out), "f64 decodes");
  double got_d = 0;
  memcpy(&got_d, out.scalar, sizeof(got_d));
  check(got_d == 2.5, "f64 value survives");
  check(r.pos == w.len, "the two scalars consumed exactly what was written");
}

// A memref sends its handle and extents and *not* its bytes, because the buffer
// is already resident. Getting this wrong in the direction of sending bytes is
// the "fax machine" failure the design rejects.
void test_memref_round_trip() {
  vx_wire_writer w = writer();
  vx_wire_reader r;
  vx_wire_arg a, out;

  memset(&a, 0, sizeof(a));
  a.tag = VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 2);
  a.handle = vx_remote_addr(1, 4096);
  a.rank = 2;
  a.sizes[0] = 288;
  a.sizes[1] = 288;
  a.strides[0] = 288;
  a.strides[1] = 1;
  vx_wire_put_arg(&w, &a);

  // handle + rank + 2 sizes + 2 strides + tag, and nothing resembling 288*288
  // floats.
  check(w.len == 4 + 8 + 4 + 4 * 8,
        "a memref costs its metadata, not its data");

  vx_wire_reader_init(&r, buf, w.len);
  check(vx_wire_get_arg(&r, &out), "memref decodes");
  check(out.handle == a.handle, "handle survives");
  check(out.rank == 2, "rank survives");
  check(out.sizes[0] == 288 && out.sizes[1] == 288, "sizes survive");
  check(out.strides[0] == 288 && out.strides[1] == 1, "strides survive");
  check(VX_ABI_ELEM(out.tag) == VX_DTYPE_F32, "element type survives");
}

// A slot carries neither handle nor extents: it is storage the worker writes a
// descriptor into, and what it will hold is already in the tag.
void test_slot_round_trip() {
  vx_wire_writer w = writer();
  vx_wire_reader r;
  vx_wire_arg a, out;

  memset(&a, 0, sizeof(a));
  a.tag = VX_ABI_SLOT_TAG(VX_DTYPE_F32, 2);
  vx_wire_put_arg(&w, &a);

  check(w.len == 4, "a slot is its tag and nothing else");

  vx_wire_reader_init(&r, buf, w.len);
  check(vx_wire_get_arg(&r, &out), "slot decodes");
  check(VX_ABI_IS_SLOT(out.tag), "slot bit survives");
  check(VX_ABI_RANK(out.tag) == 2, "the rank it will hold survives");
}

void test_header_round_trip() {
  vx_wire_writer w = writer();
  vx_wire_reader r;
  vx_wire_header h;

  vx_wire_put_header(&w, VX_WIRE_DISPATCH, 16);
  uint8_t body[16] = {0};
  vx_wire_put(&w, body, sizeof(body));

  vx_wire_reader_init(&r, buf, w.len);
  check(vx_wire_get_header(&r, &h), "header decodes");
  check(h.type == VX_WIRE_DISPATCH, "type survives");
  check(h.body_len == 16, "body length survives");
}

void test_transfer_round_trip() {
  vx_wire_writer w = writer();
  vx_wire_reader r;
  vx_wire_transfer t;
  const int64_t sizes[2] = {4, 8};
  uint8_t payload[32];
  for (size_t i = 0; i < sizeof(payload); ++i) {
    payload[i] = (uint8_t)i;
  }

  vx_wire_put_transfer(&w, VX_DTYPE_F32, 2, sizes, payload, sizeof(payload));
  check(!w.overflow, "transfer fits");

  vx_wire_reader_init(&r, buf, w.len);
  check(vx_wire_get_transfer(&r, &t), "transfer decodes");
  check(t.dtype == VX_DTYPE_F32, "dtype survives");
  check(t.rank == 2 && t.sizes[0] == 4 && t.sizes[1] == 8, "shape survives");
  check(t.nbytes == sizeof(payload), "length survives");
  check(t.bytes != nullptr && memcmp(t.bytes, payload, sizeof(payload)) == 0,
        "bytes survive");
}

void test_dispatch_round_trip() {
  vx_wire_writer w = writer();
  vx_wire_reader r;
  vx_wire_dispatch d;
  vx_wire_arg args[3], out;

  // The payload the compiler actually emits, NULs and all.
  static const char payload[] = "vx_npu_kernel_0\0kind=matmul\0roles=a:0,b:1,"
                                "out:2\0outkind=buffer\0topo=1113\0"
                                "toponame=PrefillWorker\0";

  memset(args, 0, sizeof(args));
  args[0].tag = VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 2);
  args[0].handle = vx_remote_addr(1, 1 << 20);
  args[0].rank = 2;
  args[0].sizes[0] = 2;
  args[0].sizes[1] = 2;
  args[1].tag = VX_ABI_KIND_I32;
  args[2].tag = VX_ABI_SLOT_TAG(VX_DTYPE_F32, 2);

  vx_wire_put_dispatch(&w, payload, sizeof(payload), args, 3);
  check(!w.overflow, "dispatch fits");

  vx_wire_reader_init(&r, buf, w.len);
  check(vx_wire_get_dispatch(&r, &d), "dispatch decodes");
  check(d.payload_len == sizeof(payload), "payload length survives");
  check(memcmp(d.payload, payload, sizeof(payload)) == 0,
        "payload survives verbatim, NULs included");
  check(d.num_args == 3, "argument count survives");

  // The payload is passed through untouched, so the worker decodes it with the
  // same vx_dispatch_plan.h a local plugin uses -- one decoder, not two.
  check(vx_payload_topology(d.payload, (size_t)d.payload_len) == 1113,
        "the worker can read topo= out of the forwarded payload");

  for (int i = 0; i < 3; ++i) {
    check(vx_wire_get_arg(&r, &out), "each argument decodes");
    check(out.tag == args[i].tag, "each tag survives");
  }
  check(r.pos == w.len, "the dispatch consumed exactly what was written");
}

// Everything below is input that is wrong, which is the normal case on a
// socket. None of it may read past the buffer.
void test_truncation_is_refused() {
  vx_wire_writer w = writer();
  vx_wire_arg a;
  vx_wire_reader r;
  vx_wire_arg out;

  memset(&a, 0, sizeof(a));
  a.tag = VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 2);
  a.handle = vx_remote_addr(1, 4096);
  a.rank = 2;
  vx_wire_put_arg(&w, &a);

  // Every prefix shorter than the whole must be refused, not partly believed.
  for (size_t n = 0; n < w.len; ++n) {
    vx_wire_reader_init(&r, buf, n);
    if (vx_wire_get_arg(&r, &out)) {
      check(false, "a truncated memref argument was accepted");
      break;
    }
  }
  vx_wire_reader_init(&r, buf, w.len);
  check(vx_wire_get_arg(&r, &out), "the complete argument is still accepted");
}

void test_bad_lengths_are_refused() {
  vx_wire_reader r;
  vx_wire_header h;
  vx_wire_transfer t;
  vx_wire_dispatch d;

  {
    // A body length longer than the bytes that follow.
    vx_wire_writer w = writer();
    vx_wire_put_header(&w, VX_WIRE_DISPATCH, 1000);
    vx_wire_reader_init(&r, buf, w.len);
    check(!vx_wire_get_header(&r, &h),
          "a body longer than the buffer is refused");
  }
  {
    // Wrong magic.
    vx_wire_writer w = writer();
    vx_wire_put_u32(&w, 0xDEADBEEF);
    vx_wire_put_u32(&w, VX_WIRE_VERSION);
    vx_wire_put_u32(&w, VX_WIRE_DISPATCH);
    vx_wire_put_u64(&w, 0);
    vx_wire_reader_init(&r, buf, w.len);
    check(!vx_wire_get_header(&r, &h), "a bad magic is refused");
  }
  {
    // A version we do not speak.
    vx_wire_writer w = writer();
    vx_wire_put_u32(&w, VX_WIRE_MAGIC);
    vx_wire_put_u32(&w, VX_WIRE_VERSION + 1);
    vx_wire_put_u32(&w, VX_WIRE_DISPATCH);
    vx_wire_put_u64(&w, 0);
    vx_wire_reader_init(&r, buf, w.len);
    check(!vx_wire_get_header(&r, &h), "an unknown version is refused");
  }
  {
    // A TRANSFER claiming more bytes than are present -- the field a bug would
    // use to make the reader hand out a pointer past the end.
    vx_wire_writer w = writer();
    vx_wire_put_i32(&w, VX_DTYPE_F32);
    vx_wire_put_i32(&w, 1);
    vx_wire_put_i64(&w, 4);
    vx_wire_put_u64(&w, 1u << 30);
    vx_wire_put_u32(&w, 0);
    vx_wire_reader_init(&r, buf, w.len);
    check(!vx_wire_get_transfer(&r, &t), "an overlong transfer is refused");
  }
  {
    // A DISPATCH claiming more arguments than could possibly follow.
    vx_wire_writer w = writer();
    vx_wire_put_u64(&w, 0);
    vx_wire_put_i64(&w, 1000000);
    vx_wire_reader_init(&r, buf, w.len);
    check(!vx_wire_get_dispatch(&r, &d),
          "an impossible argument count is refused");
  }
  {
    // A negative argument count.
    vx_wire_writer w = writer();
    vx_wire_put_u64(&w, 0);
    vx_wire_put_i64(&w, -1);
    vx_wire_reader_init(&r, buf, w.len);
    check(!vx_wire_get_dispatch(&r, &d),
          "a negative argument count is refused");
  }
}

// Rank arrives from the wire and then indexes fixed arrays. This is the check
// whose absence is a buffer overflow rather than a wrong answer.
void test_bad_rank_is_refused() {
  vx_wire_reader r;
  vx_wire_arg out;
  vx_wire_transfer t;

  {
    // A rank past what any consumer must hold.
    vx_wire_writer w = writer();
    vx_wire_put_i32(&w, VX_ABI_MEMREF_TAG(VX_DTYPE_F32, VX_ABI_MAX_RANK + 1));
    vx_wire_put_u64(&w, vx_remote_addr(1, 4096));
    vx_wire_put_i32(&w, VX_ABI_MAX_RANK + 1);
    for (int i = 0; i < 2 * (VX_ABI_MAX_RANK + 1); ++i) {
      vx_wire_put_i64(&w, 1);
    }
    vx_wire_reader_init(&r, buf, w.len);
    check(!vx_wire_get_arg(&r, &out), "an over-large rank is refused");
  }
  {
    // A negative rank.
    vx_wire_writer w = writer();
    vx_wire_put_i32(&w, VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 2));
    vx_wire_put_u64(&w, vx_remote_addr(1, 4096));
    vx_wire_put_i32(&w, -1);
    vx_wire_reader_init(&r, buf, w.len);
    check(!vx_wire_get_arg(&r, &out), "a negative rank is refused");
  }
  {
    // The tag says one rank and the body another. Reconciling them would mean
    // choosing which to believe; refusing does not.
    vx_wire_writer w = writer();
    vx_wire_put_i32(&w, VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 2));
    vx_wire_put_u64(&w, vx_remote_addr(1, 4096));
    vx_wire_put_i32(&w, 3);
    vx_wire_put_i64(&w, 1);
    vx_wire_put_i64(&w, 1);
    vx_wire_put_i64(&w, 1);
    vx_wire_put_i64(&w, 1);
    vx_wire_put_i64(&w, 1);
    vx_wire_put_i64(&w, 1);
    vx_wire_reader_init(&r, buf, w.len);
    check(!vx_wire_get_arg(&r, &out),
          "tag and body disagreeing on rank is refused");
  }
  {
    // And on a TRANSFER.
    vx_wire_writer w = writer();
    vx_wire_put_i32(&w, VX_DTYPE_F32);
    vx_wire_put_i32(&w, VX_ABI_MAX_RANK + 1);
    vx_wire_reader_init(&r, buf, w.len);
    check(!vx_wire_get_transfer(&r, &t),
          "an over-large transfer rank is refused");
  }
}

void test_writer_overflow_is_sticky() {
  uint8_t small[8];
  vx_wire_writer w;
  vx_wire_writer_init(&w, small, sizeof(small));

  vx_wire_put_u64(&w, 1);
  check(!w.overflow, "the first write fits");
  vx_wire_put_u64(&w, 2);
  check(w.overflow, "the second does not");
  check(w.len == 8, "and nothing was partially written");
}

} // namespace

int main() {
  test_scalars_round_trip();
  test_memref_round_trip();
  test_slot_round_trip();
  test_header_round_trip();
  test_transfer_round_trip();
  test_dispatch_round_trip();
  test_truncation_is_refused();
  test_bad_lengths_are_refused();
  test_bad_rank_is_refused();
  test_writer_overflow_is_sticky();

  if (failures) {
    fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  printf("all checks passed\n");
  return 0;
}
