import os
import re
import subprocess

def process_file(path):
    with open(path, "r") as f:
        content = f.read()

    # We only care about tests directory
    if "FileCheck" in content:
        return

    print(f"Processing {path}")

    is_fail = "/fail/" in path or "fail" in os.path.basename(path)

    lines = content.split('\n')
    has_run = False
    has_check = "CHECK:" in content

    new_lines = []
    for line in lines:
        if line.strip().startswith("// RUN:"):
            has_run = True
            # if it doesn't already have FileCheck
            if "FileCheck" not in line:
                if is_fail and "not vxc" not in line and "vxc" in line:
                    line = line.replace("vxc", "not vxc")
                
                if not is_fail and line.strip() == "// RUN: vxc %s":
                    line = "// RUN: vxc %s --action emit-mlir"
                
                line = line + " 2>&1 | FileCheck %s"
            new_lines.append(line)
        else:
            new_lines.append(line)

    if not has_run:
        # Prepend a run line
        if is_fail:
            new_lines.insert(0, "// RUN: not vxc %s 2>&1 | FileCheck %s")
        else:
            new_lines.insert(0, "// RUN: vxc %s --action emit-mlir 2>&1 | FileCheck %s")

    if not has_check:
        if is_fail:
            res = subprocess.run(["./target/debug/vxc", path], capture_output=True, text=True)
            output = res.stdout + res.stderr
            lines_out = output.strip().split("\n")
            check_str = "error"
            for i, out_line in enumerate(lines_out):
                if "failed on" in out_line or "failed to parse" in out_line:
                    if ":" in out_line:
                        check_str = out_line.split(":", 1)[1].strip()
                        if not check_str and i + 1 < len(lines_out):
                            check_str = lines_out[i + 1].strip()
                        break
            if not check_str:
                check_str = "error"
            new_lines.append(f"// CHECK: {check_str}")
        else:
            new_lines.append("// CHECK: module")

    # remove trailing empty lines before writing back
    while new_lines and new_lines[-1] == "":
        new_lines.pop()

    with open(path, "w") as f:
        f.write("\n".join(new_lines) + "\n")


if __name__ == "__main__":
    for root, _, files in os.walk("tests"):
        for file in files:
            if file.endswith(".vx"):
                process_file(os.path.join(root, file))
