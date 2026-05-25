fn main() {
    let context = melior::Context::new();
    let raw = context.to_raw();
    println!("Has raw? {:?}", raw);
}
