import os
import shutil
import itertools

PASS_DIR = "tests/backend/pass/formal_verification"
FAIL_DIR = "tests/frontend/fail/formal_verification"
UNSUPPORTED_DIR = "tests/frontend/fail/unimplemented_smt"

def init_dirs():
    os.makedirs(PASS_DIR, exist_ok=True)
    os.makedirs(FAIL_DIR, exist_ok=True)
    if os.path.exists(UNSUPPORTED_DIR):
        shutil.rmtree(UNSUPPORTED_DIR)
    os.makedirs(UNSUPPORTED_DIR, exist_ok=True)

# Categories
COMPTIME_CONSTRUCTS = ["Topology::NPU[0]", "Topology::Host", "Topology::GPU"]
ORDERING_OPS = ["==", "!=", "<", "<=", ">", ">="]
COMPTIME_FUNCTIONS = ["sin(0)", "cos(1)", "abs(-5)", "max(10, 20)"]
LOGICAL_OPS = ["&&", "||"]
DIFF_FUNCTIONS = ["grad(f)"]

def is_supported(expr):
    # Only Topology is supported in SMT prover for now.
    # Comptime Functions (sin, cos) and Diff Functions (grad) are not supported.
    if any(unsupported in expr for unsupported in ["sin", "cos", "abs", "max", "grad"]):
        return False
    return True

def generate_tests():
    init_dirs()

    # We will generate permutations in chunks and write them to files.
    # To prevent huge files, we'll keep it under 50 functions per file.
    
    pass_content = "// RUN: vxc %s\n\n"
    unsupported_content = "// EXPECTED TO FAIL due to Unsupported expression in SMT solver\n\n"
    
    # We will create permutations of:
    # requires: Construct OP Construct
    # requires: Function OP Number
    # requires: DiffFunc OP Construct
    
    # Group 1: Comptime Constructs x Ordering Ops x Comptime Constructs
    chunk_idx = 0
    func_idx = 0
    
    for c1, op, c2 in itertools.product(COMPTIME_CONSTRUCTS, ORDERING_OPS, COMPTIME_CONSTRUCTS):
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
        
    # Group 2: Comptime Functions x Ordering Ops x Number
    for f, op, num in itertools.product(COMPTIME_FUNCTIONS, ORDERING_OPS, ["0", "1", "-1"]):
        func_name = f"test_group2_{func_idx}"
        code = f"""
fn {func_name}(X: i32) -> i32
requires {f} {op} {num}
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

    # Group 3: Diff Functions x Ordering Ops x Comptime Constructs
    for f, op, c in itertools.product(DIFF_FUNCTIONS, ORDERING_OPS, COMPTIME_CONSTRUCTS):
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
        
    # Add main functions
    pass_content += "\nfn main() -> i32 { return 0; }\n"
    unsupported_content += "\nfn main() -> i32 { return 0; }\n"
    
    with open(f"{PASS_DIR}/perm_pass.vx", "w") as f:
        f.write(pass_content)
        
    with open(f"{UNSUPPORTED_DIR}/perm_unsupported_fail.vx", "w") as f:
        f.write(unsupported_content)

if __name__ == "__main__":
    generate_tests()
    print("Tests generated successfully.")
