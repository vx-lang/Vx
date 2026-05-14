module {
  func.func @main() -> i32 {
    %0 = llvm.mlir.undef : !llvm.struct<"Config", (f32, i32)>
    %1 = llvm.extractvalue %0[1] : !llvm.struct<"Config", (f32, i32)>
    return %1 : i32
  }
}
