module {
  func.func private @printMemrefF32(memref<*xf32>)
  func.func private @printMemrefF64(memref<*xf64>)
  func.func private @printMemrefI32(memref<*xi32>)
  func.func private @printMemrefI64(memref<*xi64>)
  func.func private @printMemrefBF16(memref<*xbf16>)
  func.func @strip_pin_and_deref(%t: memref<?x?xf32>) -> memref<?x?xf32> {
    // extract pointer from %t -> %v0
    %v0 = llvm.mlir.undef : !llvm.ptr<0>
    %v1 = arith.constant 0 : i64
    return %t : memref<?x?xf32>
  }
  func.func @main() -> i32 {
    %v2 = arith.constant 1 : i64
    %v3 = func.call @strip_pin_and_deref(%v2) : (i64) -> memref<?x?xf32>
    %v4 = arith.constant 0 : i32
    return %v4 : i32
  }
}
