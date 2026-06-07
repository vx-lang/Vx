import os
import subprocess

def process_file(path):
    with open(path, "r") as f:
        content = f.read()
        
    if "FileCheck" in content or "RUN:" in content:
        return
        
    print(f"Processing {path}")
    
    if "/fail/" in path or "fail" in os.path.basename(path):
        res = subprocess.run(["./target/debug/vxc", path], capture_output=True, text=True)
        output = res.stdout + res.stderr
        
        lines = output.strip().split("\n")
        check_str = "error"
        for i, line in enumerate(lines):
            if "failed on" in line or "failed to parse" in line:
                if ":" in line:
                    check_str = line.split(":", 1)[1].strip()
                    if not check_str and i + 1 < len(lines):
                        check_str = lines[i + 1].strip()
                    break

        if not check_str:
            check_str = "error"
            
        header = f"// RUN: not vxc %s 2>&1 | FileCheck %s\n// CHECK: {check_str}\n"
        with open(path, "w") as f:
            f.write(header + content)
            
    else:
        # Pass test
        header = "// RUN: vxc %s --action emit-mlir 2>&1 | FileCheck %s\n// CHECK: module\n"
        with open(path, "w") as f:
            f.write(header + content)

if __name__ == "__main__":
    for root, _, files in os.walk("tests"):
        for file in files:
            if file.endswith(".vx"):
                process_file(os.path.join(root, file))
