// RUN: akarc %s | FileCheck %s

// Middle-end test exposing potentially invalid memref copies
// (e.g. from memory space 1 to memory space 2 directly without host bounce)

fn main() -> Tensor<f32> {
    let t_host: Tensor<f32> = 1.0;
    let t_device: Tensor<f32> = t_host.to_device();

    // In actual hardware, NPU to NPU might require Host bounce, but we just generate memref.copy
    // which may be invalid MLIR depending on the target backend capabilities.
    spawn on (Topology::NPU[0]) {
        let t_core = transfer(t_device, Memory::NPU_HBM);
    }

    return t_host;
}

// CHECK: memref.copy %{{.*}}, %{{.*}} : memref<?x?xf32, 1> to memref<?x?xf32, 2>
