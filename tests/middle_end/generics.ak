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

// CHECK: func.func @process_TensorF32(%val: memref<?x?xf32>) -> memref<?x?xf32> {
// CHECK:   return %val : memref<?x?xf32>
// CHECK: }
// CHECK: func.func @process_TensorI32(%val: memref<?x?xi32>) -> memref<?x?xi32> {
// CHECK:   return %val : memref<?x?xi32>
// CHECK: }
