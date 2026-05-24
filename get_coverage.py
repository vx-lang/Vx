import os
import subprocess
import glob

def get_test_binaries():
    binaries = []
    # Find all executables in target/debug/deps
    for f in glob.glob("target/debug/deps/*"):
        if os.path.isfile(f) and os.access(f, os.X_OK) and not f.endswith(".dylib") and not f.endswith(".dSYM"):
            # Ensure it's a test binary from Vx (by checking if it has a hyphen and hash)
            if "-" in os.path.basename(f) and not f.endswith(".rlib") and not f.endswith(".rmeta"):
                binaries.append(f)
    return binaries

def main():
    profraw_files = glob.glob("vx-*.profraw")
    if not profraw_files:
        print("No profraw files found!")
        return

    print(f"Merging {len(profraw_files)} profraw files...")
    llvm_profdata = "/opt/homebrew/opt/llvm/bin/llvm-profdata"
    subprocess.run([llvm_profdata, "merge", "-sparse"] + profraw_files + ["-o", "vx.profdata"], check=True)

    binaries = get_test_binaries()
    if not binaries:
        print("No test binaries found!")
        return
    
    print(f"Generating coverage report for {len(binaries)} binaries...")
    llvm_cov = "/opt/homebrew/opt/llvm/bin/llvm-cov"
    cmd = [llvm_cov, "report", f"--instr-profile=vx.profdata", "--ignore-filename-regex=/.cargo/registry"]
    
    cmd.append(binaries[0])
    for b in binaries[1:]:
        cmd.extend(["-object", b])
        
    result = subprocess.run(cmd, capture_output=True, text=True)
    with open("coverage_report.txt", "w") as f:
        f.write(result.stdout)
    
    print("Report generated in coverage_report.txt")
    lines = result.stdout.splitlines()
    if lines:
        print(lines[-1]) # print summary line

if __name__ == "__main__":
    main()
