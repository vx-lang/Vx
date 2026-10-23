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
 * and packed as `_mlir_ciface_*`, but the half-precision printers ONLY packed
 * -- `nm` on MLIR 22 shows `_mlir_ciface_printMemrefBF16` and no plain
 * `printMemrefBF16`. Codegen emits the plain call, so a bf16 program could not
 * link, run, or emit an object; its RUN line only emits MLIR, so nothing
 * noticed.
 *
 * `llvm.emit_c_interface` on the declaration does not fix it: MLIR then gives
 * the function a body and the ExecutionEngine wants `_mlir_printMemrefBF16`
 * instead, which no library exports either. Supplying the plain symbol here is
 * what closes it, and it belongs beside the runtime rather than in vx_std_core
 * -- the standard library has no MLIR dependency and should not gain one to
 * print a tensor.
 *
 * MLIR's unpacked convention for `memref<*xT>` passes (rank, pointer to the
 * ranked descriptor); the packed entry point takes a pointer to exactly that
 * pair.
 */

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  int64_t rank;
  void *descriptor;
} VxUnrankedMemRef;

extern void _mlir_ciface_printMemrefBF16(void *unranked);
extern void _mlir_ciface_printMemrefF16(void *unranked);
extern void _mlir_ciface_printMemrefI8(void *unranked);
extern void _mlir_ciface_printMemrefI16(void *unranked);

void printMemrefBF16(int64_t rank, void *descriptor) {
  VxUnrankedMemRef packed = {rank, descriptor};
  _mlir_ciface_printMemrefBF16(&packed);
}

void printMemrefF16(int64_t rank, void *descriptor) {
  VxUnrankedMemRef packed = {rank, descriptor};
  _mlir_ciface_printMemrefF16(&packed);
}

/* The integer tensor printers MLIR does not supply in the form codegen calls,
 * or prints wrongly. MLIR has no unsigned integer type: a `u32` tensor is a
 * memref of `i32`, and MLIR's printMemrefI32 reads 4000000000 as -294967296.
 * printMemrefU8/U16/U32/U64 read the elements unsigned. printMemrefI8 cannot
 * use MLIR's printer at all -- it prints each element as a character (`100`
 * prints as `d`) -- so it is ours, reading the bytes signed like a scalar
 * `i8`. All of them print the header line and the brackets, spaces and commas
 * of MemRefDataPrinter in MemRefUtils.h, so the text does not depend on which
 * printer a type chose.
 */

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

/* The compiler sends only the element kinds that vx_print_element prints, so
 * any other kind is a compiler bug. The program stops in every build: an
 * `assert` is compiled out when NDEBUG is set, and the tensor would then print
 * wrong values or none. */
_Noreturn static void vx_no_element_printer(int bytes, int is_signed) {
  fprintf(stderr, "[Vx] FATAL: no printer for a %d-byte %s tensor element\n",
          bytes, is_signed ? "signed" : "unsigned");
  abort();
}

/* `bytes` is the element size. The signed branch is only for `i8`: MLIR prints
 * `i16` itself, and the wide signed types keep MLIR's own printers. */
static void vx_print_element(const void *base, int64_t at, int bytes,
                             int is_signed) {
  if (is_signed) {
    if (bytes != 1)
      vx_no_element_printer(bytes, is_signed);
    printf("%d", (int)((const int8_t *)base)[at]);
    return;
  }
  switch (bytes) {
  case 1:
    printf("%u", (unsigned)((const uint8_t *)base)[at]);
    return;
  case 2:
    printf("%u", (unsigned)((const uint16_t *)base)[at]);
    return;
  case 4:
    printf("%" PRIu32, ((const uint32_t *)base)[at]);
    return;
  case 8:
    printf("%" PRIu64, ((const uint64_t *)base)[at]);
    return;
  }
  vx_no_element_printer(bytes, is_signed);
}

static void vx_print_data(const void *base, int bytes, int is_signed,
                          int64_t dim, int64_t rank, int64_t offset,
                          const int64_t *sizes, const int64_t *strides) {
  if (dim == 0) {
    vx_print_element(base, offset, bytes, is_signed);
    return;
  }
  printf("[");
  vx_print_data(base, bytes, is_signed, dim - 1, rank, offset, sizes + 1,
                strides + 1);
  if (sizes[0] <= 1) {
    printf("]");
    return;
  }
  printf(", ");
  if (dim > 1)
    printf("\n");
  for (int64_t i = 1; i + 1 < sizes[0]; ++i) {
    vx_print_spaces(rank - dim + 1);
    vx_print_data(base, bytes, is_signed, dim - 1, rank,
                  offset + i * strides[0], sizes + 1, strides + 1);
    printf(", ");
    if (dim > 1)
      printf("\n");
  }
  vx_print_spaces(rank - dim + 1);
  vx_print_data(base, bytes, is_signed, dim - 1, rank,
                offset + strides[0] * (sizes[0] - 1), sizes + 1, strides + 1);
  printf("]");
}

static void vx_print_memref(int64_t rank, void *descriptor, int bytes,
                            int is_signed) {
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
  vx_print_data(m->aligned, bytes, is_signed, rank, rank, m->offset, sizes,
                strides);
  if (rank == 0)
    printf("]");
  printf("\n");
  fflush(stdout);
}

void printMemrefI8(int64_t rank, void *descriptor) {
  vx_print_memref(rank, descriptor, 1, 1);
}

void printMemrefI16(int64_t rank, void *descriptor) {
  VxUnrankedMemRef packed = {rank, descriptor};
  _mlir_ciface_printMemrefI16(&packed);
}

void printMemrefU8(int64_t rank, void *descriptor) {
  vx_print_memref(rank, descriptor, 1, 0);
}

void printMemrefU16(int64_t rank, void *descriptor) {
  vx_print_memref(rank, descriptor, 2, 0);
}

void printMemrefU32(int64_t rank, void *descriptor) {
  vx_print_memref(rank, descriptor, 4, 0);
}

void printMemrefU64(int64_t rank, void *descriptor) {
  vx_print_memref(rank, descriptor, 8, 0);
}
