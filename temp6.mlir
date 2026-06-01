module {
  llvm.func @printMemrefF32(i64, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @printMemrefF64(i64, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @printMemrefI32(i64, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @printMemrefI64(i64, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @printMemrefBF16(i64, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @_closure_1(%arg0: i32) -> i32 {
    llvm.return %arg0 : i32
  }
  llvm.func @main() -> i32 attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(0 : i32) : i32
    llvm.return %0 : i32
  }
  llvm.func @_mlir_ciface_main() -> i32 attributes {llvm.emit_c_interface} {
    %0 = llvm.call @main() : () -> i32
    llvm.return %0 : i32
  }
}

