# Vx v2.0 Release Notes

We are thrilled to announce **Vx v2.0**!

This major release achieves one of the most critical milestones in the history of the Vx language: **True Heterogeneous Hardware Offloading**.

## Key Achievements

### 1. Native NPU/Accelerator Offloading
Vx can now dynamically dispatch computational blocks designated for non-host hardware (like `Topology::NPU`) directly to physical silicon. We achieved this by integrating Apple's Matrix Coprocessor (AMX) natively via the Accelerate framework, bypassing the need for heavy Python or ONNX translation layers!

### 2. Zero-Copy Execution (Unified Memory Architecture)
The compiler natively understands and takes advantage of Apple Silicon's **Unified Memory Architecture (UMA)**.
Memory allocated in standard RAM via `Tensor` (or `memref`) is accessed natively by the Matrix Coprocessor. This eliminates explicit data movement (like PCIe `cudaMemcpy` operations) required by legacy disaggregated architectures, proving that Vx's topology types map perfectly to modern high-performance hardware.

### 3. MLIR FFI C-Interface Bridging
We introduced the `llvm.emit_c_interface` attribute to the compiler's backend logic. This bridges the MLIR standard dialect's `memref` ABI perfectly to standard C struct pointers (`MemRef2D`), allowing seamless `Objective-C++` interoperability during runtime.

### 4. JIT Dynamic Loading
The Vx JIT engine has been upgraded to orchestrate native platform compilations (`clang++ -framework Accelerate`) and load dynamically linked libraries (`.dylib`) into the `lli` execution pipeline on the fly!

### Example

The following code now successfully executes natively on the Mac matrix accelerators:

```rust
fn main() -> Verified<Tensor> {
  let mut a = Tensor([4, 4]).with_memory(Memory::NPU_HBM);
  let mut b = Tensor([4, 4]).with_memory(Memory::NPU_HBM);
  let mut result = Tensor([4, 4]).with_memory(Memory::NPU_HBM);

  // ... Initialize memory ...

  spawn on(Topology::NPU[0]) {
    for i in 0..4 {
      for j in 0..4 {
        for k in 0..4 {
          result[i][j] += a[i][k] * b[k][j];
        }
      }
    }
  }

  // CPU reads the updated memory in-place with zero latency overhead
  print(result);

  return Verified(result);
}
```

This successfully yields the correct matrix multiplication result calculated by the accelerator: `[24.0, 24.0...]`!
