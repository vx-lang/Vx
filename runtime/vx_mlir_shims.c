/*===- vx_mlir_shims.c - Vx Compiler ------------------------------*- C -*-===*\
|*                                                                            *|
|* Part of the Vx Project, under the Apache License v2.0 with LLVM            *|
|* Exceptions.                                                                *|
|* See LICENSE for license information.                                       *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/
/* The half-precision memref printers MLIR does not export in the form emitted
 * code calls.
 *
 * `libmlir_runner_utils` exports printMemrefF32, F64, I32 and I64 both plainly
 * and packed as `_mlir_ciface_*`, but the half-precision printers ONLY packed --
 * `nm` on MLIR 22 shows `_mlir_ciface_printMemrefBF16` and no plain
 * `printMemrefBF16`. Codegen emits the plain call, so a bf16 program could not
 * link, run, or emit an object; its RUN line only emits MLIR, so nothing noticed.
 *
 * `llvm.emit_c_interface` on the declaration does not fix it: MLIR then gives the
 * function a body and the ExecutionEngine wants `_mlir_printMemrefBF16` instead,
 * which no library exports either. Supplying the plain symbol here is what closes
 * it, and it belongs beside the runtime rather than in vx_std_core -- the standard
 * library has no MLIR dependency and should not gain one to print a tensor.
 *
 * MLIR's unpacked convention for `memref<*xT>` passes (rank, pointer to the ranked
 * descriptor); the packed entry point takes a pointer to exactly that pair.
 */

#include <stdint.h>

typedef struct {
  int64_t rank;
  void *descriptor;
} VxUnrankedMemRef;

extern void _mlir_ciface_printMemrefBF16(void *unranked);
extern void _mlir_ciface_printMemrefF16(void *unranked);

void printMemrefBF16(int64_t rank, void *descriptor) {
  VxUnrankedMemRef packed = {rank, descriptor};
  _mlir_ciface_printMemrefBF16(&packed);
}

void printMemrefF16(int64_t rank, void *descriptor) {
  VxUnrankedMemRef packed = {rank, descriptor};
  _mlir_ciface_printMemrefF16(&packed);
}

/* The unsigned tensor printers. MLIR has no unsigned integer type, so a `u32` or
 * `u64` tensor is a `memref` of `i32` or `i64`, and MLIR's printMemrefI32/I64 read
 * the bits as signed: 4000000000 printed as -294967296. These print the same text
 * MLIR's do -- the header line, then the brackets, spaces and commas of
 * MemRefDataPrinter in MemRefUtils.h -- with each element read unsigned.
 */

#include <inttypes.h>
#include <stdio.h>

typedef struct {
  void *allocated;
  void *aligned;
  int64_t offset;
  int64_t sizes_and_strides[];
} VxRankedMemRef;

static void vx_print_spaces(int64_t count) {
  for (int64_t i = 0; i < count; ++i)
    putchar(' ');
}

static void vx_print_unsigned(const void *base, int64_t at, int wide) {
  if (wide)
    printf("%" PRIu64, ((const uint64_t *)base)[at]);
  else
    printf("%" PRIu32, ((const uint32_t *)base)[at]);
}

static void vx_print_data(const void *base, int wide, int64_t dim, int64_t rank,
                          int64_t offset, const int64_t *sizes,
                          const int64_t *strides) {
  if (dim == 0) {
    vx_print_unsigned(base, offset, wide);
    return;
  }
  printf("[");
  vx_print_data(base, wide, dim - 1, rank, offset, sizes + 1, strides + 1);
  if (sizes[0] <= 1) {
    printf("]");
    return;
  }
  printf(", ");
  if (dim > 1)
    printf("\n");
  for (int64_t i = 1; i + 1 < sizes[0]; ++i) {
    vx_print_spaces(rank - dim + 1);
    vx_print_data(base, wide, dim - 1, rank, offset + i * strides[0], sizes + 1,
                  strides + 1);
    printf(", ");
    if (dim > 1)
      printf("\n");
  }
  vx_print_spaces(rank - dim + 1);
  vx_print_data(base, wide, dim - 1, rank, offset + strides[0] * (sizes[0] - 1),
                sizes + 1, strides + 1);
  printf("]");
}

static void vx_print_memref_unsigned(int64_t rank, void *descriptor, int wide) {
  const VxRankedMemRef *m = (const VxRankedMemRef *)descriptor;
  const int64_t *sizes = m->sizes_and_strides;
  const int64_t *strides = m->sizes_and_strides + rank;
  printf("Unranked Memref base@ = %p rank = %" PRId64 " offset = %" PRId64
         " sizes = [",
         m->aligned, rank, m->offset);
  for (int64_t i = 0; i < rank; ++i)
    printf(i ? ", %" PRId64 : "%" PRId64, sizes[i]);
  printf("] strides = [");
  for (int64_t i = 0; i < rank; ++i)
    printf(i ? ", %" PRId64 : "%" PRId64, strides[i]);
  printf("] data = \n");
  if (rank == 0)
    printf("[");
  vx_print_data(m->aligned, wide, rank, rank, m->offset, sizes, strides);
  if (rank == 0)
    printf("]");
  printf("\n");
  fflush(stdout);
}

void printMemrefU32(int64_t rank, void *descriptor) {
  vx_print_memref_unsigned(rank, descriptor, 0);
}

void printMemrefU64(int64_t rank, void *descriptor) {
  vx_print_memref_unsigned(rank, descriptor, 1);
}
