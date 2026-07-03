#!/bin/bash

# Source the local configuration to put MLIR/LLVM tools in the PATH
source config.local

# 1. Find the directory containing the compiled libnpu_dispatch
NPU_LIB_DIR=$(find target/debug/build -name "libnpu_dispatch.a" | head -n 1 | xargs dirname)

if [ -z "$NPU_LIB_DIR" ]; then
    echo "Error: Could not find libnpu_dispatch.a"
    exit 1
fi
echo "Found npu_dispatch at: $NPU_LIB_DIR"

# 2. Lower the Vx test program to an object file
# We run the `convert-vx-to-standard` pass inside vxc before piping to mlir-opt!
target/debug/vxc tests/backend/pass/ane_matmul.vx --emit-mlir -X mlir=--pass-pipeline="builtin.module(convert-vx-to-standard)" \
  | mlir-opt -convert-scf-to-cf -convert-func-to-llvm="use-bare-ptr-memref-call-conv=true" -expand-strided-metadata -finalize-memref-to-llvm -reconcile-unrealized-casts \
  | mlir-translate -mlir-to-llvmir \
  | llc -filetype=obj -o ane_matmul.o

# 3. Link the object file with the Apple hardware dispatcher to create the final executable
clang ane_matmul.o -L"$NPU_LIB_DIR" -lnpu_dispatch -o ane_matmul

# 4. Verify the binary format
file ane_matmul

# 5. Run it!
./ane_matmul

