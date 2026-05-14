// RUN: akarc %s | mlir-opt --convert-scf-to-cf --expand-strided-metadata --lower-affine --finalize-memref-to-llvm --convert-func-to-llvm --convert-cf-to-llvm --convert-arith-to-llvm --reconcile-unrealized-casts | mlir-translate --mlir-to-llvmir | lli --load=./libakar_rt.dylib
// RUN: %t | FileCheck %s

extern "C" {
    fn start_benchmark() -> Tensor<f32>;
    fn end_benchmark() -> Tensor<i32>;
    fn trace_start() -> Tensor<i32>;
    fn trace_end() -> Tensor<i32>;
}

fn main() -> Tensor<i32> {
    unsafe {
        trace_start();
        let _dummy1: Tensor<f32> = start_benchmark();
        let _dummy2: Tensor<i32> = end_benchmark();
        trace_end();
    }

    return 0;
}

// CHECK: [TRACE START] Event ID: 100
// CHECK: [TRACE END] Event ID: 100
