// RUN: akarc %s | FileCheck %s

fn method_test(t: Tensor<f32>) -> Tensor<i64> {
    let ptr = t.as_ptr();
    let mut_ptr = t.as_mut_ptr();
    let len = t.len();
    return len;
}

// CHECK: func.func @method_test(%t: memref<?x?xf32>) -> memref<?x?xi64> {
// CHECK: // extract pointer from %t -> %v0
// CHECK: %v0 = llvm.mlir.undef : !llvm.ptr<0>
// CHECK: // extract pointer from %t -> %v1
// CHECK: %v1 = llvm.mlir.undef : !llvm.ptr<0>
// CHECK: // get len of %t -> %v2
// CHECK: %v2 = arith.constant 0 : i64
