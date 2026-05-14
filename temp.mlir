module {
  func.func private @printMemrefF32(memref<*xf32>)
  func.func private @printMemrefF64(memref<*xf64>)
  func.func private @printMemrefI32(memref<*xi32>)
  func.func private @printMemrefI64(memref<*xi64>)
  func.func private @printMemrefBF16(memref<*xbf16>)
  func.func private @akar_dispatch_npu(memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> i1 attributes { llvm.emit_c_interface }
  func.func @custom_matmul(%a: memref<?x?xf64>, %b: memref<?x?xf64>) -> memref<?x?xf64> {
    // --- BEGIN SPAWN ON NPU ---
    %v0 = arith.constant 4 : index
    %v1 = arith.constant 4 : index
    %v2 = memref.alloc(%v0, %v1) : memref<?x?xf64>
    %v3 = arith.constant 0 : index
    %v4 = arith.constant 4 : index
    %v5 = arith.constant 1 : index
    scf.for %i = %v3 to %v4 step %v5 {
      %v6 = arith.constant 0 : index
      %v7 = arith.constant 4 : index
      %v8 = arith.constant 1 : index
      scf.for %j = %v6 to %v7 step %v8 {
        %v9 = arith.constant 0.0 : f64
        memref.store %v9, %v2[%i, %j] : memref<?x?xf64>
        %v10 = arith.constant 0 : index
        %v11 = arith.constant 4 : index
        %v12 = arith.constant 1 : index
        scf.for %k = %v10 to %v11 step %v12 {
          %v13 = memref.load %v2[%i, %j] : memref<?x?xf64>
          %v14 = memref.load %a[%i, %k] : memref<?x?xf64>
          %v15 = memref.load %b[%k, %j] : memref<?x?xf64>
          %v16 = arith.mulf %v14, %v15 : f64
          %v17 = arith.addf %v13, %v16 : f64
          memref.store %v17, %v2[%i, %j] : memref<?x?xf64>
        }
      }
    }
    return %v2 : memref<?x?xf64>
    // --- END SPAWN ---
  }
  func.func @main() -> i32 {
    %v18 = arith.constant 4 : index
    %v19 = arith.constant 4 : index
    %v20 = memref.alloc(%v18, %v19) : memref<?x?xf64>
    %v21 = arith.constant 4 : index
    %v22 = arith.constant 4 : index
    %v23 = memref.alloc(%v21, %v22) : memref<?x?xf64>
    %v24 = arith.constant 0 : index
    %v25 = arith.constant 4 : index
    %v26 = arith.constant 1 : index
    scf.for %i = %v24 to %v25 step %v26 {
      %v27 = arith.constant 0 : index
      %v28 = arith.constant 4 : index
      %v29 = arith.constant 1 : index
      scf.for %j = %v27 to %v28 step %v29 {
        %v30 = arith.constant 2.0 : f64
        memref.store %v30, %v20[%i, %j] : memref<?x?xf64>
        %v31 = arith.constant 3.0 : f64
        memref.store %v31, %v23[%i, %j] : memref<?x?xf64>
      }
    }
    %v32 = func.call @custom_matmul(%v20, %v23) : (memref<?x?xf64>, memref<?x?xf64>) -> memref<?x?xf64>
    %v33 = memref.cast %v32 : memref<?x?xf64> to memref<*xf64>
    func.call @printMemrefF64(%v33) : (memref<*xf64>) -> ()
    %v34 = arith.constant 0 : i32
    return %v34 : i32
  }
}
