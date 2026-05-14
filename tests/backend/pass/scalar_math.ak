// RUN: akarc %s | mlir-opt --convert-scf-to-cf --expand-strided-metadata --lower-affine --finalize-memref-to-llvm --convert-func-to-llvm --convert-cf-to-llvm --convert-arith-to-llvm --reconcile-unrealized-casts | mlir-translate --mlir-to-llvmir | lli --load=./libakar_rt.dylib
// RUN: %t | FileCheck %s

extern "C" {
    fn print_f32(val: f32) -> i32;
}

fn main() -> i32 {
    let a: f32 = 10.5;
    let b: f32 = 2.0;

    let c: f32 = a + b;
    let d: f32 = a - b;
    let e: f32 = a * b;
    let f: f32 = a / b;

    unsafe {
        print_f32(c);
        print_f32(d);
        print_f32(e);
        print_f32(f);
    }

    return 0;
}

// CHECK: 12.500000
// CHECK: 8.500000
// CHECK: 21.000000
// CHECK: 5.250000
