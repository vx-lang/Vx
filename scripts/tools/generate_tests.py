import os

def generate_math_pass_tests():
    types = ["f32", "f64", "i32", "i64"]
    ops = [("+", "add"), ("-", "sub"), ("*", "mul"), ("/", "div")]
    pass_dir = "tests/frontend/pass"
    
    for t in types:
        for op, op_name in ops:
            filename = os.path.join(pass_dir, f"gen_math_{t}_{op_name}_pass.vx")
            with open(filename, "w") as f:
                f.write(f"""//===- {os.path.basename(filename)} ---------------------------------===//
// RUN: vxc %s

fn main() -> {t} {{
  let a: {t} = 10{t};
  let b: {t} = 5{t};
  return a {op} b;
}}
""")

def generate_math_coercion_pass_tests():
    types = ["f32", "f64", "i32", "i64"]
    ops = [("+", "add"), ("-", "sub"), ("*", "mul"), ("/", "div")]
    pass_dir = "tests/frontend/pass"
    
    for t1 in types:
        for t2 in types:
            if t1 == t2:
                continue
            for op, op_name in ops:
                filename = os.path.join(pass_dir, f"gen_math_coercion_{t1}_{t2}_{op_name}_pass.vx")
                with open(filename, "w") as f:
                    f.write(f"""//===- {os.path.basename(filename)} ---------------------------------===//
// RUN: vxc %s

fn main() -> {t1} {{
  let a: {t1} = 10{t1};
  let b: {t2} = 5{t2};
  return a {op} b;
}}
""")

def generate_bool_tests():
    ops = [("&&", "and"), ("||", "or")]
    pass_dir = "tests/frontend/pass"
    fail_dir = "tests/frontend/fail"
    
    # Pass
    for op, op_name in ops:
        filename = os.path.join(pass_dir, f"gen_bool_{op_name}_pass.vx")
        with open(filename, "w") as f:
            f.write(f"""//===- {os.path.basename(filename)} ---------------------------------===//
// RUN: vxc %s

fn main() -> Bool {{
  let a: Bool = true;
  let b: Bool = false;
  return a {op} b;
}}
""")

    # Fail
    types = ["f32", "i32"]
    for t in types:
        for op, op_name in ops:
            filename = os.path.join(fail_dir, f"gen_bool_mismatch_{t}_{op_name}_fail.vx")
            with open(filename, "w") as f:
                f.write(f"""//===- {os.path.basename(filename)} ---------------------------------===//
// RUN: vxc %s 2>&1 | FileCheck %s

fn main() -> Bool {{
  let a: Bool = true;
  let b: {t} = 1{t};
  // EXPECT: Type mismatch
  return a {op} b;
}}
""")

def generate_tensor_math_tests():
    ops = [("+", "add"), ("-", "sub"), ("*", "mul"), ("/", "div")]
    pass_dir = "tests/frontend/pass"
    fail_dir = "tests/frontend/fail"
    
    # Pass
    for op, op_name in ops:
        filename = os.path.join(pass_dir, f"gen_tensor_math_{op_name}_pass.vx")
        with open(filename, "w") as f:
            f.write(f"""//===- {os.path.basename(filename)} ---------------------------------===//
// RUN: vxc %s

fn main() -> Tensor<f32, [16, 16]> {{
  let a: Tensor<f32, [16, 16]> = Tensor_f32(16, 16);
  let b: Tensor<f32, [16, 16]> = Tensor_f32(16, 16);
  return a {op} b;
}}
""")

    # Fail: Tensor element type mismatch in multiply
    filename = os.path.join(fail_dir, f"gen_tensor_math_mismatch_element_fail.vx")
    with open(filename, "w") as f:
        f.write(f"""//===- {os.path.basename(filename)} ---------------------------------===//
// RUN: vxc %s 2>&1 | FileCheck %s

fn main() -> Tensor<f32, [16, 16]> {{
  let a: Tensor<f32, [16, 16]> = Tensor_f32(16, 16);
  let b: Tensor<i32, [16, 16]> = Tensor_i32(16, 16);
  // EXPECT: Tensor multiplication requires matching element types
  return a * b;
}}
""")

if __name__ == "__main__":
    generate_math_pass_tests()
    generate_math_coercion_pass_tests()
    generate_bool_tests()
    generate_tensor_math_tests()
    print("Generated test files successfully.")
