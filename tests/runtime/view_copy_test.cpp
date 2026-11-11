//===- view_copy_test.cpp - Copy the views of one call ----------*- C++ -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The CUDA plugin runs some regions on the CPU. It copies each tensor view in
// device memory to the host for the call, and back after it, with the plan
// from runtime/vx_view_copy.h. This test runs that plan with memcpy, and an
// array in host memory stands in for device memory. So it needs no GPU.
//
//   * Two rows of one tensor get one copy each. A write to row 0 comes back.
//   * The even and the odd columns of one tensor overlap, so they get one
//     copy. A write to the even columns comes back.
//   * One tensor passed twice gets one copy.
//   * One descriptor passed twice, with an offset, gets its offset back.
//   * The last row of a large tensor gets a copy of that row only.
//   * A descriptor that the region publishes from a view gets device memory
//     back, with its own offset.
//   * An empty view keeps its descriptor.
//   * One column of a matrix gets a dense copy of its elements only, in one
//     block. The gaps between them do not change.
//   * A row and a column that the region publishes from a dense copy map back
//     to the layout of the view. A view that flattens the copy has no exact
//     map, and is refused.
//   * A view that reaches one element twice, and dimensions that interleave,
//     get no dense copy.
//   * A negative offset, or an offset past the address space, is refused.
//
// Driven by tests/integration_test/view_copy_test.rs.
//
//===----------------------------------------------------------------------===//

#include "../../runtime/vx_view_copy.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char *what) {
  if (!ok) {
    fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

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

Desc1 vector_view(float *data, int64_t offset, int64_t size, int64_t stride) {
  Desc1 d = {data, data, offset, {size}, {stride}};
  return d;
}

vx_view_copy view_of(Desc1 *d, int32_t elem = VX_DTYPE_F32) {
  vx_view_copy v = {};
  v.desc = d;
  v.rank = 1;
  v.elem = elem;
  return v;
}

// The plan, the copies and the restore, as the CUDA plugin does them.
struct Call {
  std::vector<vx_view_copy> views;
  std::vector<vx_copy_span> spans;
  int64_t num_spans = 0;

  bool copy_in(std::vector<Desc1 *> descs) {
    for (Desc1 *d : descs) {
      views.push_back(view_of(d));
    }
    spans.resize(views.size());
    int64_t bad = 0;
    if (!vx_view_copy_plan(views.data(), (int64_t)views.size(), spans.data(),
                           &num_spans, &bad)) {
      return false;
    }
    return copy_planned();
  }

  bool copy_planned() {
    for (int64_t s = 0; s < num_spans; ++s) {
      spans[s].copy = malloc(spans[s].bytes);
      if (spans[s].view >= 0) {
        vx_view_copy_gather(&views[spans[s].view], spans[s].copy);
      } else {
        memcpy(spans[s].copy, (const void *)spans[s].start, spans[s].bytes);
      }
    }
    vx_view_copy_point_at(views.data(), (int64_t)views.size(), spans.data());
    return true;
  }

  void copy_out(std::vector<void *> descs, std::vector<int32_t> ranks = {}) {
    for (int64_t s = 0; s < num_spans; ++s) {
      if (spans[s].view >= 0) {
        vx_view_copy_scatter(&views[spans[s].view], spans[s].copy);
      } else {
        memcpy((void *)spans[s].start, spans[s].copy, spans[s].bytes);
      }
    }
    ranks.resize(descs.size(), 1);
    vx_view_copy_restore(views.data(), (int64_t)views.size(), spans.data(),
                         descs.data(), ranks.data(), (int64_t)descs.size());
    for (int64_t s = 0; s < num_spans; ++s) {
      free(spans[s].copy);
    }
  }
};

// What a region does with a view: double each element through its
// descriptor.
void double_view(const Desc1 &d) {
  float *data = (float *)d.aligned + d.offset;
  for (int64_t i = 0; i < d.sizes[0]; ++i) {
    data[i * d.strides[0]] *= 2;
  }
}

void check_elements(const float *got, const float *want, int n,
                    const char *name) {
  for (int i = 0; i < n; ++i) {
    char what[96];
    snprintf(what, sizeof(what), "%s: element %d == %g", name, i,
             (double)want[i]);
    check(got[i] == want[i], what);
  }
}

void test_two_rows_get_one_copy_each() {
  float device[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  const float after[8] = {2, 4, 6, 8, 5, 6, 7, 8};
  Desc1 row0 = vector_view(device, 0, 4, 1);
  Desc1 row1 = vector_view(device, 4, 4, 1);
  Call call;
  check(call.copy_in({&row0, &row1}), "the plan for two rows succeeds");
  check(call.num_spans == 2, "two rows get two copies");
  check(call.spans[1].bytes == 16, "the copy of row 1 holds only row 1");
  check(row1.aligned != device && row1.offset == 0,
        "during the call, row 1 points at its copy with the offset 0");
  double_view(row0);
  call.copy_out({&row0, &row1});
  check(row1.aligned == device && row1.offset == 4,
        "after the call, row 1 points at device memory with the offset 4");
  check_elements(device, after, 8, "two rows");
}

void test_interleaved_columns_get_one_copy() {
  // A 2x4 tensor. The even columns are elements 0, 2, 4 and 6. The odd columns
  // are elements 1, 3, 5 and 7. Each view's bytes cover elements of the other.
  float device[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  const float after[8] = {2, 2, 6, 4, 10, 6, 14, 8};
  Desc1 even = vector_view(device, 0, 4, 2);
  Desc1 odd = vector_view(device, 1, 4, 2);
  Call call;
  check(call.copy_in({&even, &odd}),
        "the plan for the even and the odd columns succeeds");
  check(call.num_spans == 1, "the even and the odd columns get one copy");
  check(call.spans[0].bytes == 32, "the copy holds all 8 elements");
  check(call.views[1].at == 4, "the odd columns start 4 bytes into the copy");
  double_view(even);
  call.copy_out({&even, &odd});
  check_elements(device, after, 8, "interleaved columns");

  // The same, with the odd columns first. The copy must still start at
  // element 0.
  float swapped[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  Desc1 odd_first = vector_view(swapped, 1, 4, 2);
  Desc1 even_second = vector_view(swapped, 0, 4, 2);
  Call call2;
  check(call2.copy_in({&odd_first, &even_second}),
        "the plan for the odd and the even columns succeeds");
  check(call2.num_spans == 1 && call2.spans[0].bytes == 32,
        "with the odd columns first, the copy holds all 8 elements");
  double_view(even_second);
  call2.copy_out({&odd_first, &even_second});
  check_elements(swapped, after, 8, "interleaved columns, odd first");
}

void test_a_tensor_passed_twice_gets_one_copy() {
  float device[4] = {1, 2, 3, 4};
  const float after[4] = {2, 4, 6, 8};
  Desc1 out = vector_view(device, 0, 4, 1);
  Desc1 in = vector_view(device, 0, 4, 1);
  Call call;
  check(call.copy_in({&out, &in}), "the plan for one tensor twice succeeds");
  check(call.num_spans == 1, "one tensor passed twice gets one copy");
  check(out.aligned == in.aligned,
        "during the call, both views point at the same copy");
  double_view(out);
  call.copy_out({&out, &in});
  check_elements(device, after, 4, "one tensor twice");
}

void test_one_descriptor_passed_twice_gets_its_offset_back() {
  // Both arguments are the same descriptor. The second view sees the
  // descriptor after the first view pointed it at the copy.
  float device[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  Desc1 row1 = vector_view(device, 4, 4, 1);
  Call call;
  check(call.copy_in({&row1, &row1}),
        "the plan for one descriptor twice succeeds");
  check(call.num_spans == 1, "one descriptor passed twice gets one copy");
  call.copy_out({&row1, &row1});
  check(row1.aligned == device && row1.allocated == device,
        "after the call, the descriptor points at device memory");
  check(row1.offset == 4, "after the call, the descriptor has the offset 4");
}

void test_the_last_row_copies_only_that_row() {
  std::vector<float> device(1024 * 4, 1.0f);
  Desc1 last = vector_view(device.data(), 1023 * 4, 4, 1);
  Call call;
  check(call.copy_in({&last}), "the plan for the last row succeeds");
  check(call.num_spans == 1 && call.spans[0].bytes == 16,
        "the copy of the last row of 1024 holds 16 bytes");
  double_view(last);
  call.copy_out({&last});
  check(device[1023 * 4] == 2 && device[1023 * 4 + 3] == 2,
        "the write to the last row comes back");
  check(device[1022 * 4 + 3] == 1, "the row before it is unchanged");
  check(last.offset == 1023 * 4, "the offset of the last row is restored");
}

void test_a_published_view_gets_device_memory_back() {
  // The region publishes element 1 onwards of row 1 into a slot. The published
  // descriptor points at the copy with its own offset, 1.
  float device[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  Desc1 row1 = vector_view(device, 4, 4, 1);
  Call call;
  check(call.copy_in({&row1}), "the plan for the published row succeeds");
  Desc1 published = row1;
  published.offset = 1;
  published.sizes[0] = 3;
  call.copy_out({&row1, &published});
  check(published.aligned == device && published.allocated == device,
        "the published descriptor points at device memory");
  check(published.offset == 5,
        "the published descriptor's offset is 4 + 1 = 5");
}

void test_an_empty_view_keeps_its_descriptor() {
  float device[4] = {1, 2, 3, 4};
  Desc1 empty = vector_view(device, 4, 0, 1);
  Call call;
  check(call.copy_in({&empty}), "the plan for an empty view succeeds");
  check(call.num_spans == 0, "an empty view gets no copy");
  check(call.views[0].span == -1, "an empty view is in no span");
  check(empty.aligned == device && empty.offset == 4,
        "during the call, an empty view keeps its descriptor");
  call.copy_out({&empty});
}

void test_a_bad_offset_is_refused() {
  float device[4] = {1, 2, 3, 4};
  Desc1 good = vector_view(device, 0, 4, 1);
  Desc1 negative = vector_view(device, -1, 2, 1);
  std::vector<vx_view_copy> views = {view_of(&good), view_of(&negative)};
  std::vector<vx_copy_span> spans(2);
  int64_t num_spans = 0, bad = -1;
  check(!vx_view_copy_plan(views.data(), 2, spans.data(), &num_spans, &bad),
        "a view with the offset -1 is refused");
  check(bad == 1, "the refusal names view 1");

  // As an unsigned number, -2^63 is 2^63. For an i8 view, 2^63 bytes after
  // the base fit in the address space, so only the sign shows the error.
  Desc1 most_negative = vector_view(device, INT64_MIN, 2, 1);
  views = {view_of(&most_negative, VX_DTYPE_I8)};
  bad = -1;
  check(!vx_view_copy_plan(views.data(), 1, spans.data(), &num_spans, &bad),
        "an i8 view with the offset -2^63 is refused");
  check(bad == 0, "the refusal names view 0");

  // 2^62 f32 elements are 2^64 bytes, which does not fit.
  Desc1 far = vector_view(device, (int64_t)1 << 62, 2, 1);
  views = {view_of(&far)};
  bad = -1;
  check(!vx_view_copy_plan(views.data(), 1, spans.data(), &num_spans, &bad),
        "a view with the offset 2^62 is refused");
  check(bad == 0, "the refusal names view 0");
}

} // namespace

void test_a_column_gets_a_dense_copy() {
  // A 64x64 matrix. Element [r][c] is 100 * r + c.
  std::vector<float> device(64 * 64);
  for (int r = 0; r < 64; ++r) {
    for (int c = 0; c < 64; ++c) {
      device[r * 64 + c] = (float)(100 * r + c);
    }
  }
  Desc1 column = vector_view(device.data(), 3, 64, 64);
  Call call;
  check(call.copy_in({&column}), "the plan for a column succeeds");
  check(call.num_spans == 1 && call.spans[0].view == 0,
        "the column gets a dense copy");
  check(call.spans[0].bytes == 256,
        "the dense copy of a column of 64 floats holds 256 bytes");
  check(column.offset == 0 && column.strides[0] == 1,
        "during the call, the column has the offset 0 and the stride 1");
  double_view(column);
  call.copy_out({&column});
  check(column.aligned == device.data() && column.offset == 3 &&
            column.strides[0] == 64,
        "after the call, the column has its offset and its stride back");
  bool right = true;
  for (int r = 0; r < 64; ++r) {
    for (int c = 0; c < 64; ++c) {
      float want = (float)(100 * r + c) * (c == 3 ? 2 : 1);
      right &= device[r * 64 + c] == want;
    }
  }
  check(right, "column 3 doubled, and the other columns did not change");
}

void test_a_column_of_rank_two_gets_its_strides_back() {
  std::vector<float> device(64 * 64);
  Desc2 column = {device.data(), device.data(), 3, {64, 1}, {64, 1}};
  vx_view_copy v = {};
  v.desc = &column;
  v.rank = 2;
  v.elem = VX_DTYPE_F32;
  Call call;
  call.views.push_back(v);
  call.spans.resize(1);
  int64_t bad = 0;
  vx_view_copy_plan(call.views.data(), 1, call.spans.data(), &call.num_spans,
                    &bad);
  call.copy_planned();
  call.copy_out({&column}, {2});
  check(column.offset == 3 && column.strides[0] == 64 && column.strides[1] == 1,
        "a 64x1 column gets back its offset 3 and its strides 64 and 1");
}

int blocks = 0;
uint64_t block_width = 0, block_pitch = 0, block_height = 0;

void count_block(void *, uint64_t, uintptr_t, uint64_t width, uint64_t pitch,
                 uint64_t height) {
  ++blocks;
  block_width = width;
  block_pitch = pitch;
  block_height = height;
}

void test_a_column_is_one_block() {
  std::vector<float> device(64 * 64);
  Desc1 column = vector_view(device.data(), 3, 64, 64);
  Call call;
  call.views.push_back(view_of(&column));
  call.spans.resize(1);
  int64_t bad = 0;
  check(vx_view_copy_plan(call.views.data(), 1, call.spans.data(),
                          &call.num_spans, &bad),
        "the plan for a column succeeds");
  vx_view_copy_blocks(&call.views[0], count_block, nullptr);
  check(blocks == 1 && block_width == 4 && block_pitch == 256 &&
            block_height == 64,
        "a column is one block of 64 runs of 4 bytes, 256 bytes apart");

  // Two rows of two elements, 4 elements apart: each row is one run.
  Desc2 rows = {device.data(), device.data(), 1, {2, 2}, {4, 1}};
  vx_view_copy v = {};
  v.desc = &rows;
  v.rank = 2;
  v.elem = VX_DTYPE_F32;
  vx_copy_span span = {};
  check(vx_view_copy_plan(&v, 1, &span, &call.num_spans, &bad) && v.packed,
        "two padded rows get a dense copy");
  blocks = 0;
  vx_view_copy_blocks(&v, count_block, nullptr);
  check(blocks == 1 && block_width == 8 && block_pitch == 16 &&
            block_height == 2,
        "two padded rows are one block of 2 runs of 8 bytes, 16 bytes apart");
}

// A 4x2 block of a 4x8 matrix: columns 0 and 1. Element [r][c] is 10 * r + c.
struct Block {
  float device[32];
  Desc2 block;
  Call call;

  Block() {
    for (int i = 0; i < 32; ++i) {
      device[i] = (float)(10 * (i / 8) + i % 8);
    }
    block = {device, device, 0, {4, 2}, {8, 1}};
    vx_view_copy v = {};
    v.desc = &block;
    v.rank = 2;
    v.elem = VX_DTYPE_F32;
    call.views.push_back(v);
    call.spans.resize(1);
    int64_t bad = 0;
    vx_view_copy_plan(call.views.data(), 1, call.spans.data(), &call.num_spans,
                      &bad);
    call.copy_planned();
  }
};

void test_a_published_row_and_column_map_back() {
  Block b;
  check(b.call.spans[0].view == 0 && b.call.spans[0].bytes == 32,
        "the 4x2 block gets a dense copy of 32 bytes");
  check(b.block.strides[0] == 2 && b.block.strides[1] == 1,
        "during the call, the block has the dense strides 2 and 1");
  // What a region publishes: row 2 of the block, and column 1 of it.
  Desc1 row = {b.block.aligned, b.block.aligned, 4, {2}, {1}};
  Desc1 column = {b.block.aligned, b.block.aligned, 1, {4}, {2}};
  b.call.copy_out({&b.block, &row, &column}, {2, 1, 1});
  check(b.block.offset == 0 && b.block.strides[0] == 8 &&
            b.block.strides[1] == 1,
        "the block has its strides back");
  check(row.aligned == b.device && row.offset == 16 && row.strides[0] == 1,
        "the published row 2 is at the offset 16 with the stride 1");
  check(column.aligned == b.device && column.offset == 1 &&
            column.strides[0] == 8,
        "the published column 1 is at the offset 1 with the stride 8");
}

void test_a_flat_view_of_a_dense_copy_has_no_map() {
  Block b;
  // All 8 elements as one row: in the matrix, that is no view.
  int64_t offset = 0;
  int64_t sizes[1] = {8}, strides[1] = {1};
  check(!vx_view_copy_unpack(&b.call.views[0], &offset, sizes, strides, 1),
        "a flat view of the 4x2 block has no layout in the matrix");
  b.call.copy_out({&b.block}, {2});
}

void test_views_that_reach_an_element_twice_get_no_dense_copy() {
  float device[64] = {0};
  // Stride 0: each element is element 0.
  Desc2 repeat = {device, device, 0, {4, 2}, {0, 8}};
  // Sizes 2 and 3 with strides 4 and 2: element 4 is [0][2] and [1][0]. A
  // dense copy of its 6 elements is smaller than the 9 that it spans.
  Desc2 interleave = {device, device, 0, {2, 3}, {4, 2}};
  for (Desc2 *d : {&repeat, &interleave}) {
    vx_view_copy v = {};
    v.desc = d;
    v.rank = 2;
    v.elem = VX_DTYPE_F32;
    vx_copy_span span = {};
    int64_t num_spans = 0, bad = 0;
    check(vx_view_copy_plan(&v, 1, &span, &num_spans, &bad),
          "the plan succeeds");
    check(span.view == -1 && !v.packed,
          "a view that reaches an element twice keeps its strides");
  }
}

int main() {
  test_two_rows_get_one_copy_each();
  test_interleaved_columns_get_one_copy();
  test_a_tensor_passed_twice_gets_one_copy();
  test_one_descriptor_passed_twice_gets_its_offset_back();
  test_the_last_row_copies_only_that_row();
  test_a_published_view_gets_device_memory_back();
  test_an_empty_view_keeps_its_descriptor();
  test_a_bad_offset_is_refused();
  test_a_column_gets_a_dense_copy();
  test_a_column_is_one_block();
  test_a_column_of_rank_two_gets_its_strides_back();
  test_a_published_row_and_column_map_back();
  test_a_flat_view_of_a_dense_copy_has_no_map();
  test_views_that_reach_an_element_twice_get_no_dense_copy();

  if (failures) {
    fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  printf("all checks passed\n");
  return 0;
}
