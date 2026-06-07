import os
import re
import glob
import subprocess

def extract_function_body(content, name):
    # Find `fn main() -> Type { ... }`
    # We will replace `main` with `name`
    # Also strip anything after `// FRONTEND-LABEL` or `// MLIR-LABEL`
    
    # Remove MLIR stuff
    content = re.sub(r'// FRONTEND-LABEL:.*', '', content, flags=re.DOTALL)
    
    # Extract fn main
    match = re.search(r'fn\s+main\s*\(\)\s*(->\s*[^{]+)?\s*\{', content)
    if not match:
        return ""
    
    start_idx = match.start()
    
    # To find the end of the block, we count braces
    brace_count = 0
    in_block = False
    end_idx = start_idx
    for i in range(start_idx, len(content)):
        if content[i] == '{':
            brace_count += 1
            in_block = True
        elif content[i] == '}':
            brace_count -= 1
        
        if in_block and brace_count == 0:
            end_idx = i + 1
            break
            
    func_str = content[start_idx:end_idx]
    func_str = func_str.replace("fn main()", f"fn {name}()", 1)
    
    return func_str + "\n"

def process_group(out_path, pattern, is_fail=False):
    files = glob.glob(pattern)
    if not files:
        return
        
    combined_content = ""
    
    for f in sorted(files):
        # Prevent matching the target file if it already exists or matches pattern
        if os.path.abspath(f) == os.path.abspath(out_path):
            continue
            
        with open(f, "r") as file:
            content = file.read()
            
        basename = os.path.basename(f)
        name = basename.replace(".vx", "").replace("_pass", "").replace("_fail", "")
        
        func_body = extract_function_body(content, name)
        combined_content += func_body + "\n"
        
        os.remove(f)
        print(f"Removed {f}")
        
    if is_fail:
        # We need to compile to get errors
        temp_path = out_path + ".tmp"
        with open(temp_path, "w") as file:
            file.write(combined_content)
        
        res = subprocess.run(["./target/debug/vxc", temp_path], capture_output=True, text=True)
        os.remove(temp_path)
        
        errors = res.stdout + res.stderr
        check_lines = []
        for line in errors.split("\n"):
            if "Type mismatch" in line or "error" in line.lower():
                check_lines.append(f"// CHECK: {line.strip()}")
                
        header = f"// RUN: not vxc %s 2>&1 | FileCheck %s\n"
        for cl in check_lines:
            header += cl + "\n"
            
        with open(out_path, "w") as file:
            file.write(header + "\n" + combined_content)
    else:
        header = f"// RUN: vxc %s --action emit-mlir 2>&1 | FileCheck %s\n// CHECK: module\n"
        with open(out_path, "w") as file:
            file.write(header + "\n" + combined_content)
            
    print(f"Created {out_path}")

groups = [
    ("tests/frontend/pass/gen_math_coercion_pass.vx", "tests/frontend/pass/gen_math_coercion_*_pass.vx", False),
    ("tests/frontend/pass/gen_math_pass.vx", "tests/frontend/pass/gen_math_*_pass.vx", False),
    ("tests/frontend/pass/gen_tensor_math_pass.vx", "tests/frontend/pass/gen_tensor_math_*_pass.vx", False),
    ("tests/frontend/pass/gen_bool_pass.vx", "tests/frontend/pass/gen_bool_*_pass.vx", False),
    ("tests/frontend/fail/gen/gen_bool_mismatch_fail.vx", "tests/frontend/fail/gen/gen_bool_mismatch_*_fail.vx", True),
]

for out_path, pattern, is_fail in groups:
    process_group(out_path, pattern, is_fail)
