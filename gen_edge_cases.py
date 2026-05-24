import os

FAIL_DIR = "tests/middle_end/fail"

tests = {
    "undefined_variable.vx": """
fn main() -> i32 {
    return unknown_var;
}
""",
    "arg_count_mismatch.vx": """
fn foo(a: i32) -> i32 { return a; }

fn main() -> i32 {
    return foo(1, 2);
}
""",
    "invalid_member.vx": """
struct MyStruct { a: i32 }

fn main() -> i32 {
    let s = MyStruct { a: 1 };
    return s.b;
}
""",
    "not_callable.vx": """
fn main() -> i32 {
    let a = 1;
    return a(2);
}
""",
    "incompatible_assignment.vx": """
fn main() {
    let mut a: i32 = 1;
    a = 2.0;
}
""",
    "return_type_mismatch.vx": """
fn main() -> i32 {
    return 1.0;
}
""",
    "binop_mismatch.vx": """
fn main() -> i32 {
    let a: i32 = 1;
    let b: f32 = 2.0;
    return a + b;
}
""",
    "struct_init_missing.vx": """
struct MyStruct { a: i32, b: f32 }

fn main() -> MyStruct {
    return MyStruct { a: 1 };
}
""",
    "struct_init_wrong_type.vx": """
struct MyStruct { a: i32 }

fn main() -> MyStruct {
    return MyStruct { a: 1.0 };
}
"""
}

os.makedirs(FAIL_DIR, exist_ok=True)
for name, content in tests.items():
    with open(os.path.join(FAIL_DIR, name), "w") as f:
        f.write(content.strip())

print(f"Generated {len(tests)} targeted fail tests.")
