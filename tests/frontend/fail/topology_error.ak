// RUN: akarc %s | FileCheck %s

fn main() -> i32 {
    let t_host: Tensor<f32> = 1.0;
    
    spawn on (Topology::NPU[0]) {
        // This is illegal! Accessing Host memory natively from NPU.
        let invalid = t_host;
    }
    
    return 0;
}
