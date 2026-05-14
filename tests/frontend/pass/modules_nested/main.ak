fn main() -> i32 {
    let ops = comptime {
        import("tests/frontend/pass/modules_nested/ops.ak");
    };
    
    let result = ops.do_ops(5.0, 3.0);
    return 0;
}
