// RUN: akarc %s -o %t.mlir
// RUN: FileCheck %s < %t.mlir

fn process_tensor(val: Tensor<f32, [128, 128]>) -> Tensor<f32, [4, 64, 64]> {
    // 128 * 128 = 16384
    // 2 * 64 * 2 * 64 = 16384
    let temp = val.reshape([2, 64, 2, 64]);
    
    // 0, 2, 1, 3 permutation is valid for 4-dimensional tensor
    let permuted = temp.transpose([0, 2, 1, 3]);
    
    let final_view = permuted.reshape([4, 64, 64]);
    return final_view;
}

// CHECK: func.func @process_tensor
// CHECK: memref.cast %val : memref<128x128xf32> to memref<*xf32>
// CHECK: memref.cast %v1 : memref<*xf32> to memref<2x64x2x64xf32>
// CHECK: memref.cast %v0 : memref<2x64x2x64xf32> to memref<*xf32>
// CHECK: memref.cast %v3 : memref<*xf32> to memref<2x2x64x64xf32>
// CHECK: memref.cast %v2 : memref<2x2x64x64xf32> to memref<*xf32>
// CHECK: memref.cast %v5 : memref<*xf32> to memref<4x64x64xf32>
// CHECK: return %v4 : memref<4x64x64xf32>

fn main() -> i32 {
    let v: Tensor<f32, [128, 128]> = 1.0;
    process_tensor(v);
    return 0;
}
