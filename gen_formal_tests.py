import os
import random
import shutil

PASS_DIR = "tests/backend/pass/formal_verification"
FAIL_DIR = "tests/frontend/fail/formal_verification"

# Clean up directories
if os.path.exists(PASS_DIR):
    shutil.rmtree(PASS_DIR)
if os.path.exists(FAIL_DIR):
    shutil.rmtree(FAIL_DIR)

os.makedirs(PASS_DIR, exist_ok=True)
os.makedirs(FAIL_DIR, exist_ok=True)

test_count = 0

def generate_file(dir_path, name, functions, is_pass):
    global test_count
    test_count += len(functions)
    content = f"//===- {name}.vx ---------------------------------===//\n"
    if is_pass:
        content += "// RUN: vxc %s\n\n"
    
    for func in functions:
        content += func + "\n"
        
    content += "fn main() -> i32 { return 0; }\n"
    
    path = os.path.join(dir_path, f"{name}.vx")
    with open(path, "w") as f:
        f.write(content)

print("Generating Math tests...")
math_pass_funcs = []
math_fail_funcs = []
for i in range(50):
    val = random.randint(10, 100)
    add_val = random.randint(1, 50)
    math_pass_funcs.append(f"""fn test_math_{i}(N: i32) -> i32
requires N > {val}
ensures return > {val + add_val}
{{
    let x = N + {add_val} + 1;
    return x;
}}
""")
    math_fail_funcs.append(f"""fn test_math_fail_{i}(N: i32) -> i32
requires N > {val}
ensures return > {val + add_val}
{{
    // FAILS because x might be less than val + add_val if N is exactly val+1
    let x = N + {add_val} - 1;
    return x;
}}
""")

generate_file(PASS_DIR, "math_pass", math_pass_funcs, True)
generate_file(FAIL_DIR, "math_fail", math_fail_funcs, False)

print("Generating Logic tests...")
logic_pass_funcs = []
logic_fail_funcs = []
for i in range(30):
    v1 = random.randint(1, 100)
    v2 = random.randint(1, 100)
    logic_pass_funcs.append(f"""fn test_logic_and_{i}(A: i32, B: i32) -> i32
requires A > {v1} && B > {v2}
ensures return > {v1 + v2}
{{
    let sum = A + B;
    return sum;
}}
""")
    logic_fail_funcs.append(f"""fn test_logic_or_fail_{i}(A: i32, B: i32) -> i32
requires A > {v1} || B > {v2}
ensures return > {v1 + v2}
{{
    // FAILS because one of them could be negative or less than the required amount
    let sum = A + B;
    return sum;
}}
""")

generate_file(PASS_DIR, "logic_and_pass", logic_pass_funcs, True)
generate_file(FAIL_DIR, "logic_or_fail", logic_fail_funcs, False)

print("Generating Complex Logic & Inequalities tests...")
complex_pass_funcs = []
complex_fail_funcs = []
for i in range(25):
    offset = random.randint(0, 20)
    complex_pass_funcs.append(f"""fn test_complex_{i}(X: i32, Y: i32, Z: i32) -> i32
requires X >= Y && Y >= Z + {offset}
ensures return >= {offset}
{{
    let diff = X - Z;
    return diff;
}}
""")
    complex_fail_funcs.append(f"""fn test_complex_fail_{i}(X: i32, Y: i32, Z: i32) -> i32
requires X > Y && Y > Z + {offset}
ensures return < {offset}
{{
    let diff = X - Z;
    return diff;
}}
""")

generate_file(PASS_DIR, "complex_ineq_pass", complex_pass_funcs, True)
generate_file(FAIL_DIR, "complex_ineq_fail", complex_fail_funcs, False)

print("Generating Loop Invariant tests...")
loop_pass_funcs = []
loop_fail_funcs = []
for i in range(20):
    bound = random.randint(1, 100)
    loop_pass_funcs.append(f"""fn test_loop_{i}(N: i32) -> i32
requires N > {bound}
ensures return > {bound - 5}
{{
    for j in 0..10 invariant(N > {bound}) {{
        let dummy = N;
    }}
    return N;
}}
""")
    loop_fail_funcs.append(f"""fn test_loop_fail_{i}(N: i32) -> i32
requires N > {bound}
ensures return > {bound}
{{
    // FAILS because invariant requires more than N provides
    for j in 0..10 invariant(N > {bound + 10}) {{
        let dummy = N;
    }}
    return N;
}}
""")

generate_file(PASS_DIR, "loop_inv_pass", loop_pass_funcs, True)
generate_file(FAIL_DIR, "loop_inv_fail", loop_fail_funcs, False)

print("Generating Topology tests...")
top_pass_funcs = []
top_fail_funcs = []
for i in range(15):
    idx = i % 8
    top_pass_funcs.append(f"""fn test_topology_npu_{i}(T: i32) -> i32
requires Topology::NPU[{idx}] == Topology::NPU[{idx}]
ensures return == 1
{{
    return 1;
}}
""")
    top_fail_funcs.append(f"""fn test_topology_host_fail_{i}(T: i32) -> i32
requires Topology::Host == Topology::NPU[{idx}]
ensures return == 1
{{
    return 1;
}}
""")

generate_file(PASS_DIR, "topology_pass", top_pass_funcs, True)
generate_file(FAIL_DIR, "topology_fail", top_fail_funcs, False)

print(f"Generated {test_count} formal verification tests.")
