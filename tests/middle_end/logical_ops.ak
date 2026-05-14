// RUN: akarc %s | lli | FileCheck %s

fn main(d: Tensor<i32>) -> Tensor<i32> {
  let a: Tensor<i32> = 10;
  let b: Tensor<i32> = 20;
  let c: Tensor<i32> = 10;

  // Relational
  let is_less = a < b;
  let is_eq = a == c;
  let is_neq = b != c;

  // Logical
  let compound = is_less && is_eq;
  let inverted = !compound;
  let result = compound || inverted;

  return d;
}

// Check that MLIR code generates cmpi and logical ops correctly
// CHECK: arith.cmpi "slt"
// CHECK: arith.cmpi "eq"
// CHECK: arith.cmpi "ne"
// CHECK: arith.andi
// CHECK: arith.xori
// CHECK: arith.ori
