module {
  func.func private @printMemrefF32(memref<*xf32>)
  func.func private @printMemrefF64(memref<*xf64>)
  func.func private @printMemrefI32(memref<*xi32>)
  func.func private @printMemrefI64(memref<*xi64>)
  func.func private @printMemrefBF16(memref<*xbf16>)
  func.func @_closure_1(%arg0: i32) -> i32 {
    return %arg0 : i32
  }
  func.func @main() -> i32 attributes {llvm.emit_c_interface} {
    %c0_i32 = arith.constant 0 : i32
    return %c0_i32 : i32
  }
}

