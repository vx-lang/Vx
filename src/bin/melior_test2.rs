fn main() {
    let context = melior::Context::new();
    let _ = melior::ir::attribute::DenseI64ArrayAttribute::new(&context, &[1, 2, 3]);
}
