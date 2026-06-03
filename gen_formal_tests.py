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

def generate_file(dir_path, name, code, is_pass):
    global test_count
    test_count += 1
    content = f"//===- {name}.vx ---------------------------------===//\n"
    if is_pass:
        content += "// RUN: vxc %s\n\n"
    else:
        # Failing tests don't have RUN lines usually, the test harness checks them.
        # But we'll add a RUN line with a 'fail' prefix just in case the harness needs it.
        # Vx test harness automatically runs tests in fail/ expecting failure.
        pass
    
    content += code + "\n"
    path = os.path.join(dir_path, f"{name}.vx")
    with open(path, "w") as f:
        f.write(content)

print("Generating Math tests...")
for i in range(50):
    val = random.randint(10, 100)
    add_val = random.randint(1, 50)
    code = f"""
fn test_math_{i}(N: i32) -> i32
requires N > {val}
ensures return > {val + add_val}
{{
    let x = N + {add_val} + 1;
    return x;
}}

fn main() -> i32 {{
    return 0;
}}
"""
    generate_file(PASS_DIR, f"math_pass_{i}", code, True)

for i in range(50):
    val = random.randint(10, 100)
    add_val = random.randint(1, 50)
    code = f"""
fn test_math_fail_{i}(N: i32) -> i32
requires N > {val}
ensures return > {val + add_val}
{{
    // FAILS because x might be less than val + add_val if N is exactly val+1
    let x = N + {add_val} - 1;
    return x;
}}

fn main() -> i32 {{
    return 0;
}}
"""
    generate_file(FAIL_DIR, f"math_fail_{i}", code, False)


print("Generating Logic tests...")
for i in range(30):
    code = f"""
fn test_logic_and_{i}(A: i32, B: i32) -> i32
requires A > 0 && B > 0
ensures return > 0
{{
    let sum = A + B;
    return sum;
}}

fn main() -> i32 {{ return 0; }}
"""
    generate_file(PASS_DIR, f"logic_and_pass_{i}", code, True)

for i in range(30):
    code = f"""
fn test_logic_or_fail_{i}(A: i32, B: i32) -> i32
requires A > 0 || B > 0
ensures return > 0
{{
    // FAILS because one of them could be negative
    let sum = A + B;
    return sum;
}}

fn main() -> i32 {{ return 0; }}
"""
    generate_file(FAIL_DIR, f"logic_or_fail_{i}", code, False)


print("Generating Complex Logic & Inequalities tests...")
for i in range(25):
    code = f"""
fn test_complex_{i}(X: i32, Y: i32, Z: i32) -> i32
requires X >= Y && Y >= Z
ensures return >= 0
{{
    let diff = X - Z;
    return diff;
}}

fn main() -> i32 {{ return 0; }}
"""
    generate_file(PASS_DIR, f"complex_ineq_pass_{i}", code, True)

for i in range(25):
    code = f"""
fn test_complex_fail_{i}(X: i32, Y: i32, Z: i32) -> i32
requires X > Y && Y > Z
ensures return < 0
{{
    let diff = X - Z;
    return diff;
}}

fn main() -> i32 {{ return 0; }}
"""
    generate_file(FAIL_DIR, f"complex_ineq_fail_{i}", code, False)

print("Generating Loop Invariant tests...")
for i in range(20):
    code = f"""
fn test_loop_{i}(N: i32) -> i32
requires N > 5
ensures return > 0
{{
    for i in 0..10 invariant(N > 5) {{
        let dummy = N;
    }}
    return N;
}}
fn main() -> i32 {{ return 0; }}
"""
    generate_file(PASS_DIR, f"loop_inv_pass_{i}", code, True)

for i in range(20):
    code = f"""
fn test_loop_fail_{i}(N: i32) -> i32
requires N > 5
ensures return > 0
{{
    // FAILS because invariant N > 10 is not guaranteed by N > 5
    for i in 0..10 invariant(N > 10) {{
        let dummy = N;
    }}
    return N;
}}
fn main() -> i32 {{ return 0; }}
"""
    generate_file(FAIL_DIR, f"loop_inv_fail_{i}", code, False)

print("Generating Topology tests...")
for i in range(15):
    code = f"""
fn test_topology_npu_{i}(T: i32) -> i32
requires Topology::NPU[0] == Topology::NPU[0]
ensures return == 1
{{
    return 1;
}}
fn main() -> i32 {{ return 0; }}
"""
    generate_file(PASS_DIR, f"topology_pass_{i}", code, True)

for i in range(15):
    code = f"""
fn test_topology_host_fail_{i}(T: i32) -> i32
requires Topology::Host == Topology::NPU[0]
ensures return == 1
{{
    return 1;
}}
fn main() -> i32 {{ return 0; }}
"""
    generate_file(FAIL_DIR, f"topology_fail_{i}", code, False)


print(f"Generated {test_count} formal verification tests.")
