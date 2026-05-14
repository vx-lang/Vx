// CHECK: memref.alloc
// CHECK: scf.for %i =
// CHECK: memref.store
// CHECK: return %v

fn test_loop(a: Tensor, b: Tensor) -> Tensor {
    let mut result = Tensor([10, 10]);
    for i in 0..10 {
        result[i][i] = a[i][i] * b[i][i];
    }
    return result;
}
