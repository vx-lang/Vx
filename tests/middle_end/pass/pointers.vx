// RUN: akarc %s | FileCheck %s

extern "C" {
    fn malloc(size: i32) -> *mut f32;
}

fn main() -> i32 {
    unsafe {
        let ptr: *mut f32 = malloc(1024);
        ptr[0] = 5.1;
        let val: f32 = ptr[0];
    }
    return 0;
}

// CHECK: func.func @main() -> i32 {
// CHECK: = func.call @malloc(%
// CHECK: = arith.constant 5.1 : f32
// CHECK: = llvm.getelementptr %
// CHECK: llvm.store %
// CHECK: = llvm.getelementptr %
// CHECK: = llvm.load %
// CHECK: return %
// CHECK: }
