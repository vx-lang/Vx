// RUN: akarc %s | FileCheck %s

struct Point {
    x: Tensor<f32>,
    y: Tensor<f32>
}

impl Point {
    fn distance(self: Point) -> Tensor<f32> {
        return self.x + self.y;
    }
}

fn main() -> Tensor<f32> {
    let p = Point { x: 3.0, y: 4.0 };
    let d = p.distance();
    return d;
}

// CHECK: func.func @StructPoint_distance(%self: !llvm.struct<"Point">) -> memref<?x?xf32> {
// CHECK:   %v0 = arith.addf %self, %self : f32
// CHECK:   return %v0 : f32
// CHECK: }
// CHECK: func.func @main() -> i32 {
