// RUN: akarc %s | FileCheck %s

extern "C" {
  fn malloc(size: Tensor<i32>) -> *mut Tensor<f32>;
}

fn main() -> Tensor<i32> {
  unsafe {
    let ptr = malloc(1024);
  }
  return 0;
}

// CHECK: func.func private @malloc(memref<?x?xi32>) -> !llvm.ptr<0>
// CHECK: arith.constant 1024 : i64
// CHECK: func.call @malloc(%v0) : (i64) -> memref<?x?xi32>
