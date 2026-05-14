// RUN: akarc %s -o %t.mlir
// RUN: FileCheck %s < %t.mlir

fn process_tensor(val: Tensor<f32, [128, 128]>) -> Tensor<f32, [256]> {
    // Fails exactly without PadMode::Pad, but succeeds with it!
    let padded = val.reshape([256], PadMode::Pad);
    return padded;
}

// CHECK: func.func @process_tensor
// CHECK: memref.alloc() : memref<256xf32>
// CHECK: return %

fn main() -> i32 {
    let v: Tensor<f32, [128, 128]> = 1.0;
    process_tensor(v);
    return 0;
}
