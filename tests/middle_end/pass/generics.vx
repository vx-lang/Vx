// RUN: akarc %s | FileCheck %s

// Generic function with implicit type deduction
fn process<T>(val: T) -> T {
  return val;
}

fn main() -> Tensor<i32> {
  let t1: Tensor<f32> = 1;
  let t2: Tensor<i32> = 2;

  // Instantiate process twice, once with f32, once with i32
  let res1 = process(t1);
  let res2 = process(t2);

  return res2;
}

// CHECK: func.func @process_TensorF32_(%val: memref<?x?xf32>) -> memref<?x?xf32> {
// CHECK:   return %val : memref<?x?xf32>
// CHECK: }
// CHECK: func.func @process_TensorI32_(%val: memref<?x?xi32>) -> memref<?x?xi32> {
// CHECK:   return %val : memref<?x?xi32>
// CHECK: }
// CHECK: func.func @main() -> i32 {
// CHECK:   %v0 = arith.constant 1.0 : f32
// CHECK:   %v1 = arith.constant 2 : i32
// CHECK:   %v2 = func.call @process_TensorF32_(%v0) : (f32) -> memref<?x?xf32>
// CHECK:   %v3 = func.call @process_TensorI32_(%v1) : (i32) -> memref<?x?xi32>
// CHECK:   %v4 = arith.constant 0 : i32
// CHECK:   return %v4 : i32
// CHECK: }
