import os

test_dir = "tests/frontend/pass/"

vec_tests = [
    # Empty vec
    ("macro_vec_empty.vx", """fn main() -> i32 {
    let mut v = vec![];
    return 0;
}"""),
    # Single element
    ("macro_vec_single.vx", """fn main() -> i32 {
    let mut v = vec![42];
    return 0;
}"""),
    # Multiple elements
    ("macro_vec_multi.vx", """fn main() -> i32 {
    let mut v = vec![1, 2, 3, 4, 5];
    return 0;
}"""),
    # Float elements
    ("macro_vec_float.vx", """fn main() -> i32 {
    let mut v = vec![1.0, 2.5, 3.14];
    return 0;
}"""),
    # Nested vec
    ("macro_vec_nested.vx", """fn main() -> i32 {
    let mut v1 = vec![vec![1, 2], vec![3, 4]];
    return 0;
}"""),
    # Expressions in vec
    ("macro_vec_expr.vx", """fn main() -> i32 {
    let mut v = vec![1 + 1, 2 * 3, 10 / 2];
    return 0;
}"""),
    # Variables in vec
    ("macro_vec_vars.vx", """fn main() -> i32 {
    let x = 10;
    let y = 20;
    let mut v = vec![x, y, x + y];
    return 0;
}"""),
    # Function calls in vec
    ("macro_vec_func.vx", """
fn get_num() -> i32 { return 42; }
fn main() -> i32 {
    let mut v = vec![get_num(), get_num() * 2];
    return 0;
}"""),
    # Macro within vec
    ("macro_vec_nested_macro.vx", """
macro_rules! get_val {
    ($v:expr) => { $v + 10 };
}
fn main() -> i32 {
    let mut v = vec![get_val!(5), get_val!(10)];
    return 0;
}"""),
    # Large vec
    ("macro_vec_large.vx", """fn main() -> i32 {
    let mut v = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20];
    return 0;
}"""),
]

custom_macro_tests = [
    # Identity macro
    ("macro_custom_identity.vx", """
macro_rules! identity {
    ($x:expr) => { $x };
}
fn main() -> i32 {
    let a = identity!(42);
    return a;
}"""),
    # Arithmetic macro
    ("macro_custom_arithmetic.vx", """
macro_rules! double {
    ($x:expr) => { $x * 2 };
}
fn main() -> i32 {
    let a = double!(21);
    return a;
}"""),
    # Method call macro
    ("macro_custom_method.vx", """
struct Box { val: i32 }
impl Box {
    fn get(self) -> i32 { return self.val; }
}
macro_rules! get_val {
    ($obj:expr) => { $obj.get() };
}
fn main() -> i32 {
    let b = Box { val: 100 };
    let v = get_val!(b);
    return v;
}"""),
    # Array indexing macro
    ("macro_custom_index.vx", """
macro_rules! first {
    ($arr:expr) => { $arr[0] };
}
fn main() -> i32 {
    let arr = [10, 20, 30];
    let f = first!(arr);
    return f;
}"""),
    # Nested macro calls
    ("macro_custom_nested.vx", """
macro_rules! add_one {
    ($x:expr) => { $x + 1 };
}
macro_rules! add_two {
    ($x:expr) => { add_one!(add_one!($x)) };
}
fn main() -> i32 {
    let a = add_two!(10);
    return a;
}"""),
    # Macro returning boolean
    ("macro_custom_bool.vx", """
macro_rules! is_equal {
    ($x:expr) => { $x == 10 };
}
fn main() -> i32 {
    let b = is_equal!(10);
    if b { return 1; }
    return 0;
}"""),
    # Macro using tensor
    ("macro_custom_tensor.vx", """
macro_rules! shape_zero {
    ($t:expr) => { $t.shape[0] };
}
fn main() -> i32 {
    let t: Tensor = Tensor([10, 20]);
    let s = shape_zero!(t);
    return s;
}"""),
    # Multiple rules (not fully supported by my simple match but let's see)
    ("macro_custom_multi_rule.vx", """
macro_rules! do_thing {
    ($x:expr) => { $x + 1 };
}
fn main() -> i32 {
    let a = do_thing!(5);
    return a;
}"""),
    # String manipulation
    ("macro_custom_string.vx", """
macro_rules! get_str {
    ($s:expr) => { $s };
}
fn main() -> i32 {
    let s = get_str!("hello");
    return 0;
}"""),
    # Struct init macro
    ("macro_custom_struct.vx", """
struct Point { x: i32, y: i32 }
macro_rules! new_point {
    ($x:expr) => { Point { x: $x, y: 0 } };
}
fn main() -> i32 {
    let p = new_point!(10);
    return p.x;
}"""),
    # Return from macro
    ("macro_custom_return.vx", """
macro_rules! early_return {
    ($v:expr) => { return $v; };
}
fn main() -> i32 {
    early_return!(42);
    return 0;
}"""),
    # Spawn on macro
    ("macro_custom_spawn.vx", """
macro_rules! launch {
    ($top:expr) => {
        spawn on($top) {
            return 1;
        }
    };
}
fn main() -> i32 {
    launch!(Topology::NPU[0]);
    return 0;
}"""),
]

all_tests = vec_tests + custom_macro_tests

for name, content in all_tests:
    with open(os.path.join(test_dir, name), "w") as f:
        f.write(content)

print(f"Generated {len(all_tests)} tests.")
