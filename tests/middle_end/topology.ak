// RUN: akarc %s | FileCheck %s

fn main() -> Tensor<f32> {
    return 0.0;
}

fn process(t_host: Tensor<f32>) -> Tensor<f32> {
    let t_device = t_host.to_device();

    spawn on (Topology::NPU[0]) {
        let x = t_device;
    }

    return t_host;
}

// CHECK: func.func @process(%t_host: memref<?x?xf32>) -> memref<?x?xf32> {
// CHECK:   %v1 = memref.alloc() : memref<?x?xf32, 1>
// CHECK:   memref.copy %t_host, %v1 : memref<?x?xf32> to memref<?x?xf32, 1>
// CHECK:   // --- BEGIN SPAWN ON NPU ---
// CHECK:   // --- END SPAWN ---
// CHECK:   return %t_host : memref<?x?xf32>
// CHECK: }
