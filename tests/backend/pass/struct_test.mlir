// RUN: vxc %s -x mlir --action emit-mlir 2>&1 | FileCheck %s
// CHECK: module
module {
  func.func @main() -> i32 {
    %0 = llvm.mlir.undef : !llvm.struct<"Config", (f32, i32)>
    %1 = llvm.extractvalue %0[1] : !llvm.struct<"Config", (f32, i32)>
    return %1 : i32
  }
}
