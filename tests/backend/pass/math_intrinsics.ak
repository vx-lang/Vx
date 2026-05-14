// RUN: akarc %s | mlir-opt --convert-scf-to-cf --expand-strided-metadata --lower-affine --finalize-memref-to-llvm --convert-func-to-llvm --convert-cf-to-llvm --convert-arith-to-llvm --reconcile-unrealized-casts | mlir-translate --mlir-to-llvmir | lli --load=./libakar_rt.dylib
// RUN: %t | FileCheck %s

extern "C" {
    fn akar_sqrtf(x: f32) -> f32;
    fn akar_expf(x: f32) -> f32;
    fn print_f32(val: f32) -> i32;
}

fn main() -> i32 {
    let a: f32 = 16.0;
    let b: f32 = 0.0;
    
    unsafe {
        let root = akar_sqrtf(a);
        let exp = akar_expf(b);
        print_f32(root);
        print_f32(exp);
    }
    return 0;
}

// CHECK: 4.000000
// CHECK: 1.000000
