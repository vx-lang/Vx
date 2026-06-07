import os

base_dir = "/Users/adityak/go/Vx/tests/frontend/fail"

tests = {
    # 1. Const Generics Tests
    "const_generic_missing_const_kw.vx": """// RUN: not vxc %s
struct Foo<N: i32> {
  x: i32,
}
fn main() -> void {}
""",
    "const_generic_invalid_type.vx": """// RUN: not vxc %s
struct Foo<const N: String> {
  x: i32,
}
fn main() -> void {}
""",
    "const_generic_reassign.vx": """// RUN: not vxc %s
fn foo<const N: i32>() -> void {
  N = 5;
}
fn main() -> void {}
""",
    "const_generic_arg_count.vx": """// RUN: not vxc %s
struct Foo<const N: i32, const M: i32> {
  x: i32,
}
fn main() -> void {
  let mut a: Foo<1>; // Missing M
}
""",
    # "const_generic_type_mismatch.vx": Already exists
    "const_generic_runtime_expr.vx": """// RUN: not vxc %s
struct Foo<const N: i32> {
  x: i32,
}
fn main() -> void {
  let n = 5;
  let mut a: Foo<n>;
}
""",
    "const_generic_fn_call_missing.vx": """// RUN: not vxc %s
fn foo<const N: i32>() -> void {}
fn main() -> void {
  foo();
}
""",
    "const_generic_impl_mismatch.vx": """// RUN: not vxc %s
struct Foo<const N: i32> {
  x: i32,
}
impl Foo<const N: f32> { // Mismatch here
  fn bar(self: &Foo<N>) -> void {}
}
fn main() -> void {}
""",

    # 2. Inline MLIR Macros (`mlir!`) Tests
    "mlir_macro_missing_inputs.vx": """// RUN: not vxc %s
fn main() -> void {
  mlir!(
    clobbers: [],
    returns: void,
    dialects: ["linalg"]
  ) {
    macro.yield
  };
}
""",
    "mlir_macro_missing_clobbers.vx": """// RUN: not vxc %s
fn main() -> void {
  mlir!(
    inputs: (),
    returns: void,
    dialects: ["linalg"]
  ) {
    macro.yield
  };
}
""",
    "mlir_macro_missing_dialects.vx": """// RUN: not vxc %s
fn main() -> void {
  mlir!(
    inputs: (),
    clobbers: [],
    returns: void
  ) {
    macro.yield
  };
}
""",
    "mlir_macro_invalid_input_arg.vx": """// RUN: not vxc %s
fn main() -> void {
  let y = 1;
  mlir!(
    inputs: (x = y: i32), // Should be %x
    clobbers: [],
    returns: void,
    dialects: ["linalg"]
  ) {
    macro.yield
  };
}
""",
    "mlir_macro_missing_block.vx": """// RUN: not vxc %s
fn main() -> void {
  mlir!(
    inputs: (),
    clobbers: [],
    returns: void,
    dialects: ["linalg"]
  ); // Missing {} block
}
""",
    "mlir_macro_undefined_input.vx": """// RUN: not vxc %s
fn main() -> void {
  mlir!(
    inputs: (%x = undefined_var: i32),
    clobbers: [],
    returns: void,
    dialects: ["linalg"]
  ) {
    macro.yield
  };
}
""",
    "mlir_macro_undefined_clobber.vx": """// RUN: not vxc %s
fn main() -> void {
  mlir!(
    inputs: (),
    clobbers: [undefined_var],
    returns: void,
    dialects: ["linalg"]
  ) {
    macro.yield
  };
}
""",
    "mlir_macro_immutable_clobber.vx": """// RUN: not vxc %s
fn main() -> void {
  let x = 5; // Not mutable
  mlir!(
    inputs: (),
    clobbers: [x],
    returns: void,
    dialects: ["linalg"]
  ) {
    macro.yield
  };
}
""",

    # 3. Comptime Branching Tests
    "comptime_if_not_bool.vx": """// RUN: not vxc %s
fn main() -> void {
  if comptime 5 {
    let x = 1;
  }
}
""",
    "comptime_if_runtime_cond.vx": """// RUN: not vxc %s
fn main() -> void {
  let condition = true;
  if comptime condition {
    let x = 1;
  }
}
""",
    "comptime_if_expr_missing_else.vx": """// RUN: not vxc %s
fn main() -> void {
  let x = if comptime true { 1 };
}
""",
    "comptime_if_expr_type_mismatch.vx": """// RUN: not vxc %s
fn main() -> void {
  let x = if comptime true { 1 } else { 1.0 };
}
""",
    "comptime_if_pruned_syntax_error.vx": """// RUN: not vxc %s
fn main() -> void {
  if comptime false {
    let x = ; // Syntax error should be caught in parsing before comptime eval
  }
}
""",

    # 4. Topology Context Tests
    "topology_compare_type_mismatch.vx": """// RUN: not vxc %s
fn main() -> void {
  if Topology::Current == 5 {
    let x = 1;
  }
}
""",
    "topology_assign_type_mismatch.vx": """// RUN: not vxc %s
fn main() -> void {
  let x: i32 = Topology::Current;
}
""",
    "topology_undefined_variant.vx": """// RUN: not vxc %s
fn main() -> void {
  let x = Topology::Fake;
}
""",
    "topology_invalid_spawn.vx": """// RUN: not vxc %s
fn main() -> void {
  spawn on (5) {
    let x = 1;
  };
}
"""
}

for name, content in tests.items():
    filepath = os.path.join(base_dir, name)
    with open(filepath, "w") as f:
        f.write(content)
    print(f"Created {name}")
