import os
import shutil
import itertools

PASS_DIR = "tests/frontend/pass/formal_verification"
FAIL_DIR = "tests/frontend/fail/formal_verification"
UNSUPPORTED_DIR = "tests/frontend/fail/unimplemented_smt"

def init_dirs():
    os.makedirs(PASS_DIR, exist_ok=True)
    os.makedirs(FAIL_DIR, exist_ok=True)
    if os.path.exists(UNSUPPORTED_DIR):
        shutil.rmtree(UNSUPPORTED_DIR)
    os.makedirs(UNSUPPORTED_DIR, exist_ok=True)

# Categories
COMPTIME_CONSTRUCTS = ["Topology::NPU[0]", "Topology::Host"]
ORDERING_OPS = ["==", "<="]
COMPTIME_FUNCTIONS = ["cos(1)", "abs(-5)"]
DIFF_FUNCTIONS = ["grad(f)"]

def is_supported(expr):
    if any(unsupported in expr for unsupported in ["sin", "cos", "abs", "max", "grad"]):
        return False
    return True

def generate_tests():
    init_dirs()
    
    pass_content = "// RUN: vxc %s\n\n"
    unsupported_content = "// EXPECTED TO FAIL due to Unsupported expression in SMT solver\n\n"
    
    func_idx = 0
    
    # Group 1: Comptime Constructs
    for c1, op, c2 in zip(COMPTIME_CONSTRUCTS, ORDERING_OPS, COMPTIME_CONSTRUCTS[::-1]):
        func_name = f"test_group1_{func_idx}"
        code = f"""
fn {func_name}(X: i32) -> i32
requires {c1} {op} {c2}
ensures return == X
{{
    return X;
}}
"""
        if is_supported(code):
            pass_content += code
        else:
            unsupported_content += code
        func_idx += 1
        
    # Group 2: Comptime Functions
    for f, op in zip(COMPTIME_FUNCTIONS, ORDERING_OPS):
        func_name = f"test_group2_{func_idx}"
        code = f"""
fn {func_name}(X: i32) -> i32
requires {f} {op} 0
ensures return == X
{{
    return X;
}}
"""
        if is_supported(code):
            pass_content += code
        else:
            unsupported_content += code
        func_idx += 1

    # Group 3: Diff Functions
    for f, op, c in zip(DIFF_FUNCTIONS, ["=="], ["Topology::NPU[0]"]):
        func_name = f"test_group3_{func_idx}"
        code = f"""
fn {func_name}(X: i32) -> i32
requires {f} {op} {c}
ensures return == X
{{
    return X;
}}
"""
        if is_supported(code):
            pass_content += code
        else:
            unsupported_content += code
        func_idx += 1
        
    pass_content += "\nfn main() -> i32 { return 0; }\n"
    unsupported_content += "\nfn main() -> i32 { return 0; }\n"
    
    with open(f"{PASS_DIR}/perm_pass.vx", "w") as f:
        f.write(pass_content)
        
    with open(f"{UNSUPPORTED_DIR}/perm_unsupported_fail.vx", "w") as f:
        f.write(unsupported_content)

if __name__ == "__main__":
    generate_tests()
    print("Tests generated successfully.")
