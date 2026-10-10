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

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

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

/* The byte and short printers are in the same position the half-precision ones
 * were: MLIR exports `_mlir_ciface_printMemrefI8`/`I16` and no plain
 * `printMemrefI8`/`I16`, and codegen emits the plain call. `nm` on MLIR 22 shows
 * plain exports for F32, F64, I32, I64, C32, C64 and Ind only.
 *
 * `printMemrefI16` goes through MLIR, which prints numbers. `printMemrefI8`
 * cannot: MLIR's I8 printer prints each element as a CHARACTER, so a tensor
 * holding 100 prints as `d` -- measured, and not what `print(t)` means in a
 * language whose scalar `i8` prints as a number (`scalar_print_narrow.vx`).
 * So the byte printer is ours: same header line MLIR writes, then the values as
 * signed decimals.
 *
 * Two things it does differently from MLIR's printers, and why:
 *
 *  - The layout is one line. MLIR breaks a rank-2 memref into one line per row
 *    and pads columns to a common width; matching that exactly is not worth
 *    reimplementing, and one line is what the tests can match.
 *  - It prints SIGNED values, and `u8` goes through it the way a `u32` tensor
 *    already goes through `printMemrefI32`. So a byte above 127 reads as its
 *    signed counterpart. That is how every unsigned tensor prints today;
 *    making one of them honest would be a change to all of them.
 *
 * The descriptor is MLIR's `StridedMemRefType`: two pointers (allocated, then
 * the aligned one the elements are addressed through), an offset, then `rank`
 * sizes and `rank` strides.
 */

static void print_memref_i8_at(int8_t *data, const int64_t *sizes,
                               const int64_t *strides, int64_t rank, int64_t dim,
                               int64_t linear) {
  if (dim == rank) {
    printf("%d", (int)data[linear]);
    return;
  }
  printf("[");
  for (int64_t i = 0; i < sizes[dim]; ++i) {
    if (i)
      printf(", ");
    print_memref_i8_at(data, sizes, strides, rank, dim + 1,
                       linear + i * strides[dim]);
  }
  printf("]");
}

void printMemrefI8(int64_t rank, void *descriptor) {
  int8_t **ptrs = (int8_t **)descriptor;
  int64_t *meta = (int64_t *)(ptrs + 2); /* the offset, then sizes, then strides */
  printf("Unranked Memref base@ = %p rank = %" PRId64 " offset = %" PRId64
         " sizes = [",
         (void *)descriptor, rank, meta[0]);
  for (int64_t i = 0; i < rank; ++i)
    printf("%s%" PRId64, i ? ", " : "", meta[1 + i]);
  printf("] strides = [");
  for (int64_t i = 0; i < rank; ++i)
    printf("%s%" PRId64, i ? ", " : "", meta[1 + rank + i]);
  printf("] data = \n");
  print_memref_i8_at(ptrs[1], meta + 1, meta + 1 + rank, rank, 0, meta[0]);
  printf("\n");
}

void printMemrefI16(int64_t rank, void *descriptor) {
  VxUnrankedMemRef packed = {rank, descriptor};
  _mlir_ciface_printMemrefI16(&packed);
}
