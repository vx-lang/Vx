fn test_mutability(t: Tensor<f32>) -> Tensor<i32> {
    let imm_borrow = &t;
    let mut_ptr = imm_borrow.as_mut_ptr(); // ERROR: Cannot get mutable pointer from immutable borrow
    return 0;
}
