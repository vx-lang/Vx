#!/usr/bin/env python3
"""Compile each client in this directory with vxc, and check it with GenMC under RC11.

Each client is checked twice: compiled with -O0 and then optimized by LLVM at -O1, and compiled
with -O3 and optimized at -O3. GenMC cannot read the -O0 IR directly (see README.md).
The script exits with 1 when GenMC reports an error, or when it explored no execution.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
# The vxc level, the LLVM passes, and flags for opt. The SLP vectorizer joins plain loads of two
# 32-bit fields into one vector load, and GenMC cannot check accesses of mixed sizes. It never
# changes an atomic access.
LEVELS = [("-O0", "default<O1>", []), ("-O3", "default<O3>", ["-vectorize-slp=false"])]


def run(cmd, out=None):
    with open(out, "w") if out else open(os.devnull, "w") as f:
        r = subprocess.run(cmd, stdout=f if out else subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        sys.exit(f"{' '.join(map(str, cmd))} failed:\n{r.stderr}")


def use_harness_memset(ir):
    """Point each `llvm.memset` at `vx_genmc_memset` in harness.c. GenMC runs `llvm.memset`
    only on a local or a global."""
    ir = re.sub(r"^declare void @llvm\.memset\.p0\.i64\(.*$", "declare void @vx_genmc_memset(ptr, i8, i64, i1)",
                ir, flags=re.M)
    return ir.replace("@llvm.memset.p0.i64(", "@vx_genmc_memset(")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("clients", nargs="*", help="the .vx files to check (default: all in this directory)")
    p.add_argument("--vxc", default=str(REPO / "target/debug/vxc"))
    p.add_argument("--genmc", default=os.environ.get("GENMC", "genmc"))
    p.add_argument("--llvm-bin", default="/usr/lib/llvm-22/bin")
    p.add_argument("--vxc-flag", action="append", default=[], help="pass a flag to vxc, for example -fweak-relaxed-atomics")
    p.add_argument("--keep", metavar="DIR", help="keep the IR files in DIR")
    args = p.parse_args()

    genmc = Path(subprocess.run(["which", args.genmc], capture_output=True, text=True).stdout.strip() or args.genmc)
    include = genmc.resolve().parent.parent / "include/genmc/runtime"
    llvm = Path(args.llvm_bin)
    clients = [Path(c) for c in args.clients] or sorted(HERE.glob("*.vx"))
    failed = False
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(args.keep or tmp)
        tmp.mkdir(parents=True, exist_ok=True)
        harness = tmp / "harness.ll"
        run([llvm / "clang", "-fno-discard-value-names", "-Xclang", "-disable-O0-optnone", "-g",
             f"-I{include}", "-S", "-emit-llvm", "-o", harness, HERE / "harness.c"])
        for client in clients:
            for vx_level, llvm_passes, opt_flags in LEVELS:
                name = f"{client.stem} {vx_level}"
                mlir, raw, opt, linked = (tmp / f"{client.stem}{s}" for s in (".mlir", ".ll", ".opt.ll", ".all.ll"))
                run([args.vxc, vx_level, *args.vxc_flag, "--action", "emit-llvm", client], out=mlir)
                run(["mlir-translate", "--mlir-to-llvmir", mlir, "-o", raw])
                run([llvm / "opt", *opt_flags, "-mtriple=x86_64-unknown-linux-gnu", f"-passes={llvm_passes}", "-S", raw, "-o", opt])
                opt.write_text(use_harness_memset(opt.read_text()))
                run([llvm / "llvm-link", "-S", harness, opt, "-o", linked])
                r = subprocess.run([genmc, "--rc11", "--check-liveness", "--disable-estimation", linked],
                                   capture_output=True, text=True)
                text = r.stdout + r.stderr
                runs = re.search(r"complete executions explored: (\d+)", text)
                blocked = re.search(r"blocked executions seen: (\d+)", text)
                ok = r.returncode == 0 and "No errors were detected" in text and runs and int(runs.group(1)) > 0
                print(f"{'ok  ' if ok else 'FAIL'} {name}: {runs.group(1) if runs else 0} executions"
                      + (f", {blocked.group(1)} blocked" if blocked else ""), flush=True)
                if not ok:
                    failed = True
                    print(text)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
