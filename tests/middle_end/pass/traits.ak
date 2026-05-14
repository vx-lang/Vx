// RUN: akarc %s | FileCheck %s

trait Dimension {
  fn dim(self: Tensor<f32>) -> Tensor<i32>;
}

impl Dimension for Tensor<f32> {
  fn dim(self: Tensor<f32>) -> Tensor<i32> {
    return 2;
  }
}

fn process<T: Dimension>(val: T) -> Tensor<i32> {
  let d = val.dim();
  return d;
}

fn main() -> Tensor<i32> {
  let t: Tensor<f32> = 1.0;
  let res = process(t);
  return res;
}

// CHECK: func.func @Tensor_F32_dim(%self: memref<?x?xf32>) -> memref<?x?xi32> {
// CHECK:   %v0 = arith.constant 2 : i32
// CHECK:   return %v0 : i32
// CHECK: }
// CHECK: func.func @process_TensorF32(%val: memref<?x?xf32>) -> memref<?x?xi32> {
