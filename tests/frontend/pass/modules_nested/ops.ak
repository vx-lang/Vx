fn do_ops(a: f32, b: f32) -> f32 {
    let core = comptime {
        import("tests/frontend/pass/modules_nested/math_core.ak");
    };
    
    let prod = core.multiply(a, b);
    let diff = core.subtract(a, b);
    return prod + diff;
}
