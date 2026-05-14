// RUN: akarc %s | FileCheck %s

struct Point {
  x: f32,
  y: f32
}

impl Point {
  fn distance(self: Point) -> f32 {
    return self.x + self.y;
  }
}

fn main() -> i32 {
  let p = Point { x: 3.0, y: 4.0 };
  let d = p.distance();
  return 0;
}

// CHECK: func.func @StructPoint_distance(%self: !llvm.struct<"Point", (f32, f32)>) -> f32 {
// CHECK:   %v0 = llvm.extractvalue %self[0] : !llvm.struct<"Point", (f32, f32)>
// CHECK:   %v1 = llvm.extractvalue %self[1] : !llvm.struct<"Point", (f32, f32)>
// CHECK:   %v2 = arith.addf %v0, %v1 : f32
// CHECK:   return %v2 : f32
// CHECK: }
// CHECK: func.func @main() -> i32 {
