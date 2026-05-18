Vx Compiler (vxc) - Bootstrap Phase (Rust)
============================================
module {
  func.func private @printMemrefF32(memref<*xf32>)
  func.func private @printMemrefF64(memref<*xf64>)
  func.func private @printMemrefI32(memref<*xi32>)
  func.func private @printMemrefI64(memref<*xi64>)
  func.func private @printMemrefBF16(memref<*xbf16>)
  func.func private @vx_dispatch_amx(!llvm.ptr<0>, !llvm.ptr<0>, !llvm.ptr<0>, i32, i32) -> i32
  func.func private @vx_dispatch_ane(!llvm.ptr<0>, !llvm.ptr<0>, !llvm.ptr<0>, i32, i32) -> i32
  func.func private @vx_dispatch_gpu(!llvm.ptr<0>, !llvm.ptr<0>, !llvm.ptr<0>, i32, i32) -> i32
  func.func private @vx_malloc_f32(i32) -> memref<?xf32>
  func.func private @free(memref<?xf32>) -> i32
  func.func private @printf(memref<?xi8>, f32) -> i32
  func.func private @putchar(i32) -> i32
  func.func private @vx_sqrtf(f32) -> f32
  func.func private @vx_expf(f32) -> f32
  func.func private @vx_cosf(f32) -> f32
  func.func private @vx_sinf(f32) -> f32
  func.func private @vx_simd_reduce_add(vector<4xf32>) -> f32
  func.func private @vx_powf(f32, f32) -> f32
  func.func private @vx_get_rope_freq(i32, i32, i32) -> f32
  func.func private @vx_load_config(memref<?xi8>) -> memref<?xi32>
  func.func private @vx_load_weights(memref<?xi8>) -> memref<?xf32>
  func.func private @vx_advance_ptr(memref<?xf32>, i32) -> memref<?xf32>
  func.func private @vx_build_tokenizer(memref<?xi8>, i32) -> memref<?xi8>
  func.func private @vx_decode_token(memref<?xi8>, i32, i32) -> memref<?xi8>
  func.func private @vx_safe_printf(memref<?xi8>) -> i32
  func.func private @vx_print_int(i32) -> i32
  func.func private @vx_print_float(f32) -> i32
  func.func private @vx_read_prompt_file(memref<?xi8>) -> memref<?xi8>
  func.func private @vx_encode_prompt(memref<?xi8>, memref<?xi8>) -> memref<?xi32>
  func.func private @vx_get_time() -> f32
  func.func private @vx_printf_i32(memref<?xi8>, i32) -> i32
  // Struct Config: {i32, i32, i32, i32, i32, i32, i32}
  // Struct TransformerWeights: {memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>}
  // Struct RunState: {memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>}
  func.func @malloc_run_state(%dim: i32, %hidden_dim: i32, %n_layers: i32, %n_heads: i32, %n_kv_heads: i32, %seq_len: i32, %vocab_size: i32) -> !llvm.struct<"RunState", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)> {
    %v0 = arith.muli %dim, %n_kv_heads : i32
    %v1 = arith.divsi %v0, %n_heads : i32
    %v2 = func.call @vx_malloc_f32(%dim) : (i32) -> memref<?xf32>
    %v3 = func.call @vx_malloc_f32(%dim) : (i32) -> memref<?xf32>
    %v4 = func.call @vx_malloc_f32(%dim) : (i32) -> memref<?xf32>
    %v5 = func.call @vx_malloc_f32(%hidden_dim) : (i32) -> memref<?xf32>
    %v6 = func.call @vx_malloc_f32(%hidden_dim) : (i32) -> memref<?xf32>
    %v7 = func.call @vx_malloc_f32(%dim) : (i32) -> memref<?xf32>
    %v8 = func.call @vx_malloc_f32(%dim) : (i32) -> memref<?xf32>
    %v9 = func.call @vx_malloc_f32(%dim) : (i32) -> memref<?xf32>
    %v10 = arith.muli %n_heads, %seq_len : i32
    %v11 = func.call @vx_malloc_f32(%v10) : (i32) -> memref<?xf32>
    %v12 = func.call @vx_malloc_f32(%vocab_size) : (i32) -> memref<?xf32>
    %v13 = arith.muli %n_layers, %seq_len : i32
    %v14 = arith.muli %v13, %v1 : i32
    %v15 = func.call @vx_malloc_f32(%v14) : (i32) -> memref<?xf32>
    %v16 = arith.muli %n_layers, %seq_len : i32
    %v17 = arith.muli %v16, %v1 : i32
    %v18 = func.call @vx_malloc_f32(%v17) : (i32) -> memref<?xf32>
    %v19 = llvm.mlir.undef : !llvm.struct<"RunState", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>
    %v20 = llvm.insertvalue %v2, %v19[0] : !llvm.struct<"RunState", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>
    %v21 = llvm.insertvalue %v3, %v20[1] : !llvm.struct<"RunState", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>
    %v22 = llvm.insertvalue %v4, %v21[2] : !llvm.struct<"RunState", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>
    %v23 = llvm.insertvalue %v5, %v22[3] : !llvm.struct<"RunState", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>
    %v24 = llvm.insertvalue %v6, %v23[4] : !llvm.struct<"RunState", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>
    %v25 = llvm.insertvalue %v7, %v24[5] : !llvm.struct<"RunState", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>
    %v26 = llvm.insertvalue %v8, %v25[6] : !llvm.struct<"RunState", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>
    %v27 = llvm.insertvalue %v9, %v26[7] : !llvm.struct<"RunState", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>
    %v28 = llvm.insertvalue %v11, %v27[8] : !llvm.struct<"RunState", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>
    %v29 = llvm.insertvalue %v12, %v28[9] : !llvm.struct<"RunState", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>
    %v30 = llvm.insertvalue %v15, %v29[10] : !llvm.struct<"RunState", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>
    %v31 = llvm.insertvalue %v18, %v30[11] : !llvm.struct<"RunState", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>
    return %v31 : !llvm.struct<"RunState", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>
  }
  func.func @memory_map_weights(%ptr: memref<?xf32>, %dim: i32, %hidden_dim: i32, %n_layers: i32, %n_heads: i32, %n_kv_heads: i32, %seq_len: i32, %vocab_size: i32) -> !llvm.struct<"TransformerWeights", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)> {
    %v32 = arith.divsi %dim, %n_heads : i32
    %v33 = memref.memory_space_cast %ptr : memref<?xf32> to memref<?x?xi32>
    %v34 = arith.muli %vocab_size, %dim : i32
    %v35 = func.call @vx_advance_ptr(%v33, %v34) : (memref<?x?xi32>, i32) -> memref<?xf32>
    %v36 = memref.memory_space_cast %v35 : memref<?xf32> to memref<?x?xi32>
    %v37 = arith.muli %n_layers, %dim : i32
    %v38 = func.call @vx_advance_ptr(%v36, %v37) : (memref<?x?xi32>, i32) -> memref<?xf32>
    %v39 = arith.muli %n_layers, %dim : i32
    %v40 = arith.muli %v39, %n_heads : i32
    %v41 = arith.muli %v40, %v32 : i32
    %v42 = memref.memory_space_cast %v38 : memref<?xf32> to memref<?x?xi32>
    %v43 = func.call @vx_advance_ptr(%v42, %v41) : (memref<?x?xi32>, i32) -> memref<?xf32>
    %v44 = arith.muli %n_layers, %dim : i32
    %v45 = arith.muli %v44, %n_kv_heads : i32
    %v46 = arith.muli %v45, %v32 : i32
    %v47 = memref.memory_space_cast %v43 : memref<?xf32> to memref<?x?xi32>
    %v48 = func.call @vx_advance_ptr(%v47, %v46) : (memref<?x?xi32>, i32) -> memref<?xf32>
    %v49 = arith.muli %n_layers, %dim : i32
    %v50 = arith.muli %v49, %n_kv_heads : i32
    %v51 = arith.muli %v50, %v32 : i32
    %v52 = memref.memory_space_cast %v48 : memref<?xf32> to memref<?x?xi32>
    %v53 = func.call @vx_advance_ptr(%v52, %v51) : (memref<?x?xi32>, i32) -> memref<?xf32>
    %v54 = arith.muli %n_layers, %n_heads : i32
    %v55 = arith.muli %v54, %v32 : i32
    %v56 = arith.muli %v55, %dim : i32
    %v57 = memref.memory_space_cast %v53 : memref<?xf32> to memref<?x?xi32>
    %v58 = func.call @vx_advance_ptr(%v57, %v56) : (memref<?x?xi32>, i32) -> memref<?xf32>
    %v59 = memref.memory_space_cast %v58 : memref<?xf32> to memref<?x?xi32>
    %v60 = arith.muli %n_layers, %dim : i32
    %v61 = func.call @vx_advance_ptr(%v59, %v60) : (memref<?x?xi32>, i32) -> memref<?xf32>
    %v62 = memref.memory_space_cast %v61 : memref<?xf32> to memref<?x?xi32>
    %v63 = arith.muli %n_layers, %dim : i32
    %v64 = arith.muli %v63, %hidden_dim : i32
    %v65 = func.call @vx_advance_ptr(%v62, %v64) : (memref<?x?xi32>, i32) -> memref<?xf32>
    %v66 = memref.memory_space_cast %v65 : memref<?xf32> to memref<?x?xi32>
    %v67 = arith.muli %n_layers, %hidden_dim : i32
    %v68 = arith.muli %v67, %dim : i32
    %v69 = func.call @vx_advance_ptr(%v66, %v68) : (memref<?x?xi32>, i32) -> memref<?xf32>
    %v70 = memref.memory_space_cast %v69 : memref<?xf32> to memref<?x?xi32>
    %v71 = arith.muli %n_layers, %dim : i32
    %v72 = arith.muli %v71, %hidden_dim : i32
    %v73 = func.call @vx_advance_ptr(%v70, %v72) : (memref<?x?xi32>, i32) -> memref<?xf32>
    %v74 = memref.memory_space_cast %v73 : memref<?xf32> to memref<?x?xi32>
    %v75 = func.call @vx_advance_ptr(%v74, %dim) : (memref<?x?xi32>, i32) -> memref<?xf32>
    %v76 = memref.memory_space_cast %v75 : memref<?xf32> to memref<?x?xi32>
    %v77 = arith.muli %seq_len, %v32 : i32
    %v78 = func.call @vx_advance_ptr(%v76, %v77) : (memref<?x?xi32>, i32) -> memref<?xf32>
    %v79 = llvm.mlir.undef : !llvm.struct<"TransformerWeights", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>
    %v80 = llvm.insertvalue %ptr, %v79[0] : !llvm.struct<"TransformerWeights", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>
    %v81 = llvm.insertvalue %v35, %v80[1] : !llvm.struct<"TransformerWeights", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>
    %v82 = llvm.insertvalue %v38, %v81[3] : !llvm.struct<"TransformerWeights", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>
    %v83 = llvm.insertvalue %v43, %v82[4] : !llvm.struct<"TransformerWeights", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>
    %v84 = llvm.insertvalue %v48, %v83[5] : !llvm.struct<"TransformerWeights", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>
    %v85 = llvm.insertvalue %v53, %v84[6] : !llvm.struct<"TransformerWeights", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>
    %v86 = llvm.insertvalue %v58, %v85[2] : !llvm.struct<"TransformerWeights", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>
    %v87 = llvm.insertvalue %v61, %v86[7] : !llvm.struct<"TransformerWeights", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>
    %v88 = llvm.insertvalue %v65, %v87[8] : !llvm.struct<"TransformerWeights", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>
    %v89 = llvm.insertvalue %v69, %v88[9] : !llvm.struct<"TransformerWeights", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>
    %v90 = llvm.insertvalue %v73, %v89[10] : !llvm.struct<"TransformerWeights", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>
    %v91 = llvm.insertvalue %ptr, %v90[11] : !llvm.struct<"TransformerWeights", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>
    return %v91 : !llvm.struct<"TransformerWeights", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>
  }
  func.func @transformer(%token: i32, %pos: i32, %p: memref<?x!llvm.struct<"Config", (i32, i32, i32, i32, i32, i32, i32)>>, %s: memref<?x!llvm.struct<"RunState", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>>, %w: memref<?x!llvm.struct<"TransformerWeights", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>>) -> memref<?xf32> {
    %v92 = llvm.extractvalue %s[0] : memref<?x!llvm.struct<"RunState", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>>
    %v93 = llvm.extractvalue %p[0] : memref<?x!llvm.struct<"Config", (i32, i32, i32, i32, i32, i32, i32)>>
    %v94 = llvm.extractvalue %p[0] : memref<?x!llvm.struct<"Config", (i32, i32, i32, i32, i32, i32, i32)>>
    %v95 = llvm.extractvalue %p[4] : memref<?x!llvm.struct<"Config", (i32, i32, i32, i32, i32, i32, i32)>>
    %v96 = arith.muli %v94, %v95 : i32
    %v97 = llvm.extractvalue %p[3] : memref<?x!llvm.struct<"Config", (i32, i32, i32, i32, i32, i32, i32)>>
    %v98 = arith.divsi %v96, %v97 : i32
    %v99 = llvm.extractvalue %p[3] : memref<?x!llvm.struct<"Config", (i32, i32, i32, i32, i32, i32, i32)>>
    %v100 = llvm.extractvalue %p[4] : memref<?x!llvm.struct<"Config", (i32, i32, i32, i32, i32, i32, i32)>>
    %v101 = arith.divsi %v99, %v100 : i32
    %v102 = llvm.extractvalue %p[1] : memref<?x!llvm.struct<"Config", (i32, i32, i32, i32, i32, i32, i32)>>
    %v103 = llvm.extractvalue %p[3] : memref<?x!llvm.struct<"Config", (i32, i32, i32, i32, i32, i32, i32)>>
    %v104 = arith.divsi %v93, %v103 : i32
    %v105 = llvm.extractvalue %w[0] : memref<?x!llvm.struct<"TransformerWeights", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>>
    %v106 = memref.memory_space_cast %v105 : memref<?xf32> to memref<?x?xi32>
    %v107 = arith.muli %token, %v93 : i32
    %v108 = func.call @vx_advance_ptr(%v106, %v107) : (memref<?x?xi32>, i32) -> memref<?xf32>
    %v109 = arith.constant 4 : i32
    %v110 = arith.divsi %v93, %v109 : i32
    %v111 = arith.constant 0 : index
    %v112 = arith.index_cast %v110 : i32 to index
    %v113 = arith.constant 1 : index
    scf.for %i = %v111 to %v112 step %v113 {
      %v114 = arith.index_cast %i : index to i32
      %v115 = arith.constant 4 : i32
      %v116 = arith.muli %v114, %v115 : i32
      %v117 = memref.memory_space_cast %v108 : memref<?xf32> to memref<?x?xi32>
      %v118 = func.call @vx_advance_ptr(%v117, %v116) : (memref<?x?xi32>, i32) -> memref<?xf32>
      %v119 = memref.memory_space_cast %v92 : memref<?xf32> to memref<?x?xi32>
      %v120 = func.call @vx_advance_ptr(%v119, %v116) : (memref<?x?xi32>, i32) -> memref<?xf32>
      // deref %v118 -> %v121
      %v122 = arith.constant 0 : index
      %v121 = vector.load %v118[%v122] : memref<?xf32>, vector<4xf32>
      %v123 = arith.constant 0 : index
      vector.store %v121, %v120[%v123] : memref<?xf32>, vector<4xf32>
    }
    %v124 = arith.constant 0 : i64
    %v125 = llvm.extractvalue %p[2] : memref<?x!llvm.struct<"Config", (i32, i32, i32, i32, i32, i32, i32)>>
    %v126 = llvm.extractvalue %p[6] : memref<?x!llvm.struct<"Config", (i32, i32, i32, i32, i32, i32, i32)>>
    %v127 = arith.constant 0 : index
    %v128 = arith.index_cast %v125 : i32 to index
    %v129 = arith.constant 1 : index
    scf.for %l = %v127 to %v128 step %v129 {
      %v130 = arith.index_cast %l : index to i32
      %v131 = arith.muli %v130, %v93 : i32
      %v132 = llvm.extractvalue %w[1] : memref<?x!llvm.struct<"TransformerWeights", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>>
      %v133 = memref.memory_space_cast %v132 : memref<?xf32> to memref<?x?xi32>
      %v134 = func.call @vx_advance_ptr(%v133, %v131) : (memref<?x?xi32>, i32) -> memref<?xf32>
      %v135 = arith.constant 256.0 : f32
      %v136 = arith.constant 512.0 : f32
      %v137 = llvm.extractvalue %s[1] : memref<?x!llvm.struct<"RunState", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>>
      %v138 = func.call @rmsnorm(%v137, %v92, %v134, %v93, %v135) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, i32, f32) -> i32
      %v139 = arith.index_cast %l : index to i32
      %v140 = arith.muli %v139, %v126 : i32
      %v141 = arith.muli %v140, %v98 : i32
      %v142 = arith.muli %pos, %v98 : i32
      %v143 = arith.addi %v141, %v142 : i32
      %v144 = llvm.extractvalue %s[10] : memref<?x!llvm.struct<"RunState", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>>
      %v145 = memref.memory_space_cast %v144 : memref<?xf32> to memref<?x?xi32>
      %v146 = func.call @vx_advance_ptr(%v145, %v143) : (memref<?x?xi32>, i32) -> memref<?xf32>
      %v147 = llvm.extractvalue %s[11] : memref<?x!llvm.struct<"RunState", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>>
      %v148 = memref.memory_space_cast %v147 : memref<?xf32> to memref<?x?xi32>
      %v149 = func.call @vx_advance_ptr(%v148, %v143) : (memref<?x?xi32>, i32) -> memref<?xf32>
      %v150 = arith.index_cast %l : index to i32
      %v151 = arith.muli %v150, %v93 : i32
      %v152 = arith.muli %v151, %v93 : i32
      %v153 = llvm.extractvalue %w[3] : memref<?x!llvm.struct<"TransformerWeights", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>>
      %v154 = memref.memory_space_cast %v153 : memref<?xf32> to memref<?x?xi32>
      %v155 = func.call @vx_advance_ptr(%v154, %v152) : (memref<?x?xi32>, i32) -> memref<?xf32>
      %v156 = llvm.extractvalue %s[5] : memref<?x!llvm.struct<"RunState", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>>
      %v157 = llvm.extractvalue %s[1] : memref<?x!llvm.struct<"RunState", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>>
      %v158 = func.call @matmul(%v156, %v157, %v155, %v93, %v93) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, i32, i32) -> i32
      %v159 = arith.index_cast %l : index to i32
      %v160 = arith.muli %v159, %v93 : i32
      %v161 = arith.muli %v160, %v98 : i32
      %v162 = llvm.extractvalue %w[4] : memref<?x!llvm.struct<"TransformerWeights", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>>
      %v163 = memref.memory_space_cast %v162 : memref<?xf32> to memref<?x?xi32>
      %v164 = func.call @vx_advance_ptr(%v163, %v161) : (memref<?x?xi32>, i32) -> memref<?xf32>
      %v165 = llvm.extractvalue %s[1] : memref<?x!llvm.struct<"RunState", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>>
      %v166 = func.call @matmul(%v146, %v165, %v164, %v93, %v98) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, i32, i32) -> i32
      %v167 = arith.index_cast %l : index to i32
      %v168 = arith.muli %v167, %v93 : i32
      %v169 = arith.muli %v168, %v98 : i32
      %v170 = llvm.extractvalue %w[5] : memref<?x!llvm.struct<"TransformerWeights", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>>
      %v171 = memref.memory_space_cast %v170 : memref<?xf32> to memref<?x?xi32>
      %v172 = func.call @vx_advance_ptr(%v171, %v169) : (memref<?x?xi32>, i32) -> memref<?xf32>
      %v173 = llvm.extractvalue %s[1] : memref<?x!llvm.struct<"RunState", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>>
      %v174 = func.call @matmul(%v149, %v173, %v172, %v93, %v98) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, i32, i32) -> i32
      %v175 = arith.constant 0 : index
      %v176 = arith.index_cast %v98 : i32 to index
      %v177 = arith.constant 1 : index
      scf.for %i = %v175 to %v176 step %v177 {
        %v178 = arith.index_cast %i : index to i32
        %v179 = arith.constant 2 : i32
        %v180 = arith.divsi %v178, %v179 : i32
        %v181 = arith.constant 2 : i32
        %v182 = arith.muli %v180, %v181 : i32
        %v183 = arith.index_cast %i : index to i32
        %v184 = arith.subi %v183, %v182 : i32
        %v185 = arith.constant 0 : i32
        %v187 = arith.cmpi "eq", %v184, %v185 : i32
        scf.if %v187 {
          %v188 = arith.index_cast %i : index to i32
          %v189 = func.call @vx_get_rope_freq(%pos, %v188, %v104) : (i32, i32, i32) -> f32
          %v190 = func.call @vx_cosf(%v189) : (f32) -> f32
          %v191 = func.call @vx_sinf(%v189) : (f32) -> f32
          %v192 = arith.index_cast %i : index to i32
          %v193 = arith.index_cast %i : index to i32
          %v194 = arith.constant 1 : i32
          %v195 = arith.addi %v193, %v194 : i32
          %v196 = llvm.extractvalue %s[5] : memref<?x!llvm.struct<"RunState", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>>
          %v197 = arith.index_cast %v192 : i32 to index
          %v198 = llvm.extractvalue %s[5] : memref<?x!llvm.struct<"RunState", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>>
          %v199 = arith.index_cast %v195 : i32 to index
          %v200 = arith.index_cast %v192 : i32 to index
          %v201 = memref.load %v146[%v200] : memref<?x?xf32>
          %v202 = arith.index_cast %v195 : i32 to index
          %v203 = memref.load %v146[%v202] : memref<?x?xf32>
          %v204 = arith.index_cast %v192 : i32 to index
          %v205 = arith.mulf %v201, %v190 : f32
          %v206 = arith.mulf %v203, %v191 : f32
          %v207 = arith.subf %v205, %v206 : f32
          memref.store %v207, %v146[%v204] : memref<?xf32>
          %v208 = arith.index_cast %v195 : i32 to index
          %v209 = arith.mulf %v201, %v191 : f32
          %v210 = arith.mulf %v203, %v190 : f32
          %v211 = arith.addf %v209, %v210 : f32
          memref.store %v211, %v146[%v208] : memref<?xf32>
        }
        %v212 = arith.constant 0 : i64
      }
      %v213 = arith.constant 1 : i32
      %v214 = arith.addi %pos, %v213 : i32
      %v215 = arith.constant 0 : index
      %v216 = llvm.extractvalue %p[3] : memref<?x!llvm.struct<"Config", (i32, i32, i32, i32, i32, i32, i32)>>
      %v217 = arith.index_cast %v216 : i32 to index
      %v218 = arith.constant 1 : index
      scf.for %h = %v215 to %v217 step %v218 {
        %v219 = arith.index_cast %h : index to i32
        %v220 = arith.muli %v219, %v104 : i32
        %v221 = arith.index_cast %h : index to i32
        %v222 = arith.muli %v221, %v126 : i32
        %v223 = llvm.extractvalue %s[8] : memref<?x!llvm.struct<"RunState", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>>
        %v224 = memref.memory_space_cast %v223 : memref<?xf32> to memref<?x?xi32>
        %v225 = func.call @vx_advance_ptr(%v224, %v222) : (memref<?x?xi32>, i32) -> memref<?xf32>
        %v226 = arith.constant 0 : index
        %v227 = arith.index_cast %v214 : i32 to index
        %v228 = arith.constant 1 : index
        scf.for %t = %v226 to %v227 step %v228 {
          %v229 = arith.index_cast %h : index to i32
          %v230 = arith.divsi %v229, %v101 : i32
          %v231 = arith.index_cast %l : index to i32
          %v232 = arith.muli %v231, %v126 : i32
          %v233 = arith.muli %v232, %v98 : i32
          %v234 = arith.index_cast %t : index to i32
          %v235 = arith.muli %v234, %v98 : i32
          %v236 = arith.addi %v233, %v235 : i32
          %v237 = arith.muli %v230, %v104 : i32
          %v238 = arith.addi %v236, %v237 : i32
          %v239 = llvm.extractvalue %s[10] : memref<?x!llvm.struct<"RunState", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>>
          %v240 = memref.memory_space_cast %v239 : memref<?xf32> to memref<?x?xi32>
          %v241 = func.call @vx_advance_ptr(%v240, %v238) : (memref<?x?xi32>, i32) -> memref<?xf32>
          %v242 = arith.constant 0.0 : f32
          %v243 = memref.alloca() : memref<f32>
          memref.store %v242, %v243[] : memref<f32>
          %v244 = arith.constant 0 : index
          %v245 = arith.index_cast %v104 : i32 to index
          %v246 = arith.constant 1 : index
          scf.for %i = %v244 to %v245 step %v246 {
            %v247 = llvm.extractvalue %s[5] : memref<?x!llvm.struct<"RunState", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>>
            %v248 = arith.index_cast %v220 : i32 to index
            %v249 = arith.addi %v248, %i : index
            %v250 = memref.load %v241[%i] : memref<?x?xf32>
            %v251 = arith.mulf , %v250 : 
            %v252 = memref.load %v243[] : memref<f32>
            %v253 = arith.addf %v252, %v251 : f32
            memref.store %v253, %v243[] : memref<f32>
          }
          %v254 = arith.constant 32.0 : f32
          %v255 = arith.constant 1.0 : f32
          %v256 = func.call @vx_sqrtf(%v254) : (f32) -> f32
          %v257 = arith.divf %v255, %v256 : f32
          %v258 = memref.load %v243[] : memref<f32>
          %v259 = arith.mulf %v258, %v257 : f32
          memref.store %v259, %v243[] : memref<f32>
          %v260 = memref.load %v243[] : memref<f32>
          memref.store %v260, %v225[%t] : memref<?xf32>
        }
        %v261 = arith.constant 0 : index
        %v262 = memref.load %v225[%v261] : memref<?x?xf32>
        %v263 = memref.alloca() : memref<f32>
        memref.store %v262, %v263[] : memref<f32>
        %v264 = arith.constant 1 : index
        %v265 = arith.index_cast %v214 : i32 to index
        %v266 = arith.constant 1 : index
        scf.for %t = %v264 to %v265 step %v266 {
          %v267 = memref.load %v225[%t] : memref<?x?xf32>
          %v268 = memref.load %v263[] : memref<f32>
          %v270 = arith.cmpf "ogt", %v267, %v268 : f32
          scf.if %v270 {
            memref.store %v267, %v263[] : memref<f32>
          }
          %v271 = arith.constant 0 : i64
        }
        %v272 = arith.constant 0.0 : f32
        %v273 = memref.alloca() : memref<f32>
        memref.store %v272, %v273[] : memref<f32>
        %v274 = arith.constant 0 : index
        %v275 = arith.index_cast %v214 : i32 to index
        %v276 = arith.constant 1 : index
        scf.for %t = %v274 to %v275 step %v276 {
          %v277 = memref.load %v225[%t] : memref<?x?xf32>
          %v278 = memref.load %v263[] : memref<f32>
          %v279 = arith.subf %v277, %v278 : f32
          %v280 = func.call @vx_expf(%v279) : (f32) -> f32
          memref.store %v280, %v225[%t] : memref<?xf32>
          %v281 = memref.load %v273[] : memref<f32>
          %v282 = arith.addf %v281, %v280 : f32
          memref.store %v282, %v273[] : memref<f32>
        }
        %v283 = arith.constant 0 : index
        %v284 = arith.index_cast %v214 : i32 to index
        %v285 = arith.constant 1 : index
        scf.for %t = %v283 to %v284 step %v285 {
          %v286 = memref.load %v225[%t] : memref<?x?xf32>
          %v287 = memref.load %v273[] : memref<f32>
          %v288 = arith.divf %v286, %v287 : f32
          memref.store %v288, %v225[%t] : memref<?xf32>
        }
        %v289 = arith.index_cast %h : index to i32
        %v290 = arith.muli %v289, %v104 : i32
        %v291 = arith.constant 0 : index
        %v292 = arith.index_cast %v104 : i32 to index
        %v293 = arith.constant 1 : index
        scf.for %i = %v291 to %v292 step %v293 {
          %v294 = arith.constant 0.0 : f32
          %v295 = memref.alloca() : memref<f32>
          memref.store %v294, %v295[] : memref<f32>
          %v296 = arith.constant 0 : index
          %v297 = arith.index_cast %v214 : i32 to index
          %v298 = arith.constant 1 : index
          scf.for %t = %v296 to %v297 step %v298 {
            %v299 = arith.index_cast %h : index to i32
            %v300 = arith.divsi %v299, %v101 : i32
            %v301 = arith.index_cast %l : index to i32
            %v302 = arith.muli %v301, %v126 : i32
            %v303 = arith.muli %v302, %v98 : i32
            %v304 = arith.index_cast %t : index to i32
            %v305 = arith.muli %v304, %v98 : i32
            %v306 = arith.addi %v303, %v305 : i32
            %v307 = arith.muli %v300, %v104 : i32
            %v308 = arith.addi %v306, %v307 : i32
            %v309 = llvm.extractvalue %s[11] : memref<?x!llvm.struct<"RunState", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>>
            %v310 = memref.memory_space_cast %v309 : memref<?xf32> to memref<?x?xi32>
            %v311 = func.call @vx_advance_ptr(%v310, %v308) : (memref<?x?xi32>, i32) -> memref<?xf32>
            %v312 = memref.load %v225[%t] : memref<?x?xf32>
            %v313 = memref.load %v311[%i] : memref<?x?xf32>
            %v314 = arith.mulf %v312, %v313 : f32
            %v315 = memref.load %v295[] : memref<f32>
            %v316 = arith.addf %v315, %v314 : f32
            memref.store %v316, %v295[] : memref<f32>
          }
        }
      }
      %v317 = arith.index_cast %l : index to i32
      %v318 = arith.muli %v317, %v93 : i32
      %v319 = arith.muli %v318, %v93 : i32
      %v320 = llvm.extractvalue %w[6] : memref<?x!llvm.struct<"TransformerWeights", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>>
      %v321 = memref.memory_space_cast %v320 : memref<?xf32> to memref<?x?xi32>
      %v322 = func.call @vx_advance_ptr(%v321, %v319) : (memref<?x?xi32>, i32) -> memref<?xf32>
      %v323 = llvm.extractvalue %s[2] : memref<?x!llvm.struct<"RunState", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>>
      %v324 = llvm.extractvalue %s[1] : memref<?x!llvm.struct<"RunState", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>>
      %v325 = func.call @matmul(%v323, %v324, %v322, %v93, %v93) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, i32, i32) -> i32
      %v326 = arith.constant 0 : index
      %v327 = arith.index_cast %v93 : i32 to index
      %v328 = arith.constant 1 : index
      scf.for %i = %v326 to %v327 step %v328 {
        %v329 = memref.load %v92[%i] : memref<?x?xf32>
        %v330 = llvm.extractvalue %s[2] : memref<?x!llvm.struct<"RunState", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>>
        %v331 = arith.addf %v329,  : f32
        memref.store %v331, %v92[%i] : memref<?xf32>
      }
      %v332 = arith.index_cast %l : index to i32
      %v333 = arith.muli %v332, %v93 : i32
      %v334 = llvm.extractvalue %w[2] : memref<?x!llvm.struct<"TransformerWeights", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>>
      %v335 = memref.memory_space_cast %v334 : memref<?xf32> to memref<?x?xi32>
      %v336 = func.call @vx_advance_ptr(%v335, %v333) : (memref<?x?xi32>, i32) -> memref<?xf32>
      %v337 = llvm.extractvalue %s[1] : memref<?x!llvm.struct<"RunState", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>>
      %v338 = func.call @rmsnorm(%v337, %v92, %v336, %v93, %v135) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, i32, f32) -> i32
      %v339 = arith.index_cast %l : index to i32
      %v340 = arith.muli %v339, %v93 : i32
      %v341 = arith.muli %v340, %v102 : i32
      %v342 = llvm.extractvalue %w[7] : memref<?x!llvm.struct<"TransformerWeights", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>>
      %v343 = memref.memory_space_cast %v342 : memref<?xf32> to memref<?x?xi32>
      %v344 = func.call @vx_advance_ptr(%v343, %v341) : (memref<?x?xi32>, i32) -> memref<?xf32>
      %v345 = arith.index_cast %l : index to i32
      %v346 = arith.muli %v345, %v93 : i32
      %v347 = arith.muli %v346, %v102 : i32
      %v348 = llvm.extractvalue %w[9] : memref<?x!llvm.struct<"TransformerWeights", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>>
      %v349 = memref.memory_space_cast %v348 : memref<?xf32> to memref<?x?xi32>
      %v350 = func.call @vx_advance_ptr(%v349, %v347) : (memref<?x?xi32>, i32) -> memref<?xf32>
      %v351 = llvm.extractvalue %s[3] : memref<?x!llvm.struct<"RunState", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>>
      %v352 = llvm.extractvalue %s[1] : memref<?x!llvm.struct<"RunState", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>>
      %v353 = func.call @matmul_ane(%v351, %v352, %v344, %v93, %v102) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, i32, i32) -> i32
      %v354 = llvm.extractvalue %s[4] : memref<?x!llvm.struct<"RunState", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>>
      %v355 = llvm.extractvalue %s[1] : memref<?x!llvm.struct<"RunState", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>>
      %v356 = func.call @matmul_ane(%v354, %v355, %v350, %v93, %v102) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, i32, i32) -> i32
      %v357 = arith.constant 0 : index
      %v358 = arith.index_cast %v102 : i32 to index
      %v359 = arith.constant 1 : index
      scf.for %i = %v357 to %v358 step %v359 {
        %v360 = llvm.extractvalue %s[3] : memref<?x!llvm.struct<"RunState", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>>
        %v361 = memref.alloca() : memref<>
        memref.store , %v361[] : memref<>
        %v362 = arith.constant 0.0 : f32
        %v363 = memref.load %v361[] : memref<>
        %v364 = arith.subf %v362, %v363 : f32
        %v365 = func.call @vx_expf(%v364) : (f32) -> f32
        %v366 = arith.constant 1.0 : f32
        %v367 = arith.addf %v366, %v365 : f32
        %v368 = arith.constant 1.0 : f32
        %v369 = arith.divf %v368, %v367 : f32
        %v370 = memref.load %v361[] : memref<>
        %v371 = arith.mulf %v370, %v369 : 
        memref.store %v371, %v361[] : memref<>
        %v372 = llvm.extractvalue %s[4] : memref<?x!llvm.struct<"RunState", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>>
      }
      %v373 = arith.index_cast %l : index to i32
      %v374 = arith.muli %v373, %v93 : i32
      %v375 = arith.muli %v374, %v102 : i32
      %v376 = llvm.extractvalue %w[8] : memref<?x!llvm.struct<"TransformerWeights", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>>
      %v377 = memref.memory_space_cast %v376 : memref<?xf32> to memref<?x?xi32>
      %v378 = func.call @vx_advance_ptr(%v377, %v375) : (memref<?x?xi32>, i32) -> memref<?xf32>
      %v379 = llvm.extractvalue %s[1] : memref<?x!llvm.struct<"RunState", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>>
      %v380 = llvm.extractvalue %s[3] : memref<?x!llvm.struct<"RunState", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>>
      %v381 = func.call @matmul_ane(%v379, %v380, %v378, %v102, %v93) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, i32, i32) -> i32
      %v382 = arith.constant 0 : index
      %v383 = arith.index_cast %v93 : i32 to index
      %v384 = arith.constant 1 : index
      scf.for %i = %v382 to %v383 step %v384 {
        %v385 = memref.load %v92[%i] : memref<?x?xf32>
        %v386 = llvm.extractvalue %s[1] : memref<?x!llvm.struct<"RunState", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>>
        %v387 = arith.addf %v385,  : f32
        memref.store %v387, %v92[%i] : memref<?xf32>
      }
    }
    %v388 = arith.constant 256.0 : f32
    %v389 = llvm.extractvalue %w[10] : memref<?x!llvm.struct<"TransformerWeights", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>>
    %v390 = func.call @rmsnorm(%v92, %v92, %v389, %v93, %v388) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, i32, f32) -> i32
    %v391 = llvm.extractvalue %s[9] : memref<?x!llvm.struct<"RunState", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>>
    %v392 = llvm.extractvalue %w[11] : memref<?x!llvm.struct<"TransformerWeights", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>>
    %v393 = llvm.extractvalue %p[0] : memref<?x!llvm.struct<"Config", (i32, i32, i32, i32, i32, i32, i32)>>
    %v394 = llvm.extractvalue %p[5] : memref<?x!llvm.struct<"Config", (i32, i32, i32, i32, i32, i32, i32)>>
    %v395 = func.call @matmul(%v391, %v92, %v392, %v393, %v394) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, i32, i32) -> i32
    %v396 = llvm.extractvalue %s[9] : memref<?x!llvm.struct<"RunState", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>>
    return %v396 : memref<?xf32>
  }
  func.func @sample_argmax(%probabilities: memref<?xf32>, %n: i32) -> i32 {
    %v397 = arith.constant 0 : i32
    %v398 = memref.alloca() : memref<i32>
    memref.store %v397, %v398[] : memref<i32>
    %v399 = arith.constant 0 : index
    %v400 = memref.load %probabilities[%v399] : memref<?x?xf32>
    %v401 = memref.alloca() : memref<f32>
    memref.store %v400, %v401[] : memref<f32>
    %v402 = arith.constant 1 : index
    %v403 = arith.index_cast %n : i32 to index
    %v404 = arith.constant 1 : index
    scf.for %i = %v402 to %v403 step %v404 {
      %v405 = memref.load %probabilities[%i] : memref<?x?xf32>
      %v406 = memref.load %v401[] : memref<f32>
      %v408 = arith.cmpf "ogt", %v405, %v406 : f32
      scf.if %v408 {
        %v409 = arith.index_cast %i : index to i32
        memref.store %v409, %v398[] : memref<i32>
        memref.store %v405, %v401[] : memref<f32>
      }
      %v410 = arith.constant 0 : i64
    }
    %v411 = memref.load %v398[] : memref<i32>
    return %v411 : i32
  }
  func.func @rmsnorm(%o: memref<?xf32>, %x: memref<?xf32>, %weight: memref<?xf32>, %size: i32, %size_f: f32) -> i32 {
    %v412 = arith.constant 0.0 : f32
    %v413 = memref.alloca() : memref<f32>
    memref.store %v412, %v413[] : memref<f32>
    %v414 = arith.constant 0 : index
    %v415 = arith.index_cast %size : i32 to index
    %v416 = arith.constant 1 : index
    scf.for %j = %v414 to %v415 step %v416 {
      %v417 = memref.load %x[%j] : memref<?x?xf32>
      %v418 = arith.mulf %v417, %v417 : f32
      %v419 = memref.load %v413[] : memref<f32>
      %v420 = arith.addf %v419, %v418 : f32
      memref.store %v420, %v413[] : memref<f32>
    }
    %v421 = memref.load %v413[] : memref<f32>
    %v422 = arith.divf %v421, %size_f : f32
    %v423 = arith.constant 0.00001 : f32
    %v424 = arith.addf %v422, %v423 : f32
    memref.store %v424, %v413[] : memref<f32>
    %v425 = arith.constant 1.0 : f32
    %v426 = memref.load %v413[] : memref<f32>
    %v427 = func.call @vx_sqrtf(%v426) : (f32) -> f32
    %v428 = arith.divf %v425, %v427 : f32
    memref.store %v428, %v413[] : memref<f32>
    %v429 = arith.constant 0 : i64
    %v430 = arith.constant 0 : index
    %v431 = arith.index_cast %size : i32 to index
    %v432 = arith.constant 1 : index
    scf.for %j = %v430 to %v431 step %v432 {
      %v433 = memref.load %x[%j] : memref<?x?xf32>
      %v434 = memref.load %weight[%j] : memref<?x?xf32>
      %v435 = memref.load %v413[] : memref<f32>
      %v436 = arith.mulf %v435, %v433 : f32
      %v437 = arith.mulf %v434, %v436 : f32
      memref.store %v437, %o[%j] : memref<?xf32>
    }
    %v438 = arith.constant 0 : i32
    return %v438 : i32
  }
  func.func @softmax(%x: memref<?xf32>, %size: i32) -> i32 {
    %v439 = arith.constant 0 : index
    %v440 = memref.load %x[%v439] : memref<?x?xf32>
    %v441 = memref.alloca() : memref<f32>
    memref.store %v440, %v441[] : memref<f32>
    %v442 = arith.constant 1 : index
    %v443 = arith.index_cast %size : i32 to index
    %v444 = arith.constant 1 : index
    scf.for %i = %v442 to %v443 step %v444 {
      %v445 = memref.load %x[%i] : memref<?x?xf32>
      %v446 = memref.load %v441[] : memref<f32>
      %v448 = arith.cmpf "ogt", %v445, %v446 : f32
      scf.if %v448 {
        memref.store %v445, %v441[] : memref<f32>
      }
      %v449 = arith.constant 0 : i64
    }
    %v450 = arith.constant 0.0 : f32
    %v451 = memref.alloca() : memref<f32>
    memref.store %v450, %v451[] : memref<f32>
    %v452 = arith.constant 0 : index
    %v453 = arith.index_cast %size : i32 to index
    %v454 = arith.constant 1 : index
    scf.for %i = %v452 to %v453 step %v454 {
      %v455 = memref.load %x[%i] : memref<?x?xf32>
      %v456 = memref.load %v441[] : memref<f32>
      %v457 = arith.subf %v455, %v456 : f32
      %v458 = func.call @vx_expf(%v457) : (f32) -> f32
      memref.store %v458, %x[%i] : memref<?xf32>
      %v459 = arith.constant 0 : i64
      %v460 = memref.load %x[%i] : memref<?x?xf32>
      %v461 = memref.load %v451[] : memref<f32>
      %v462 = arith.addf %v461, %v460 : f32
      memref.store %v462, %v451[] : memref<f32>
    }
    %v463 = arith.constant 0 : index
    %v464 = arith.index_cast %size : i32 to index
    %v465 = arith.constant 1 : index
    scf.for %i = %v463 to %v464 step %v465 {
      %v466 = memref.load %x[%i] : memref<?x?xf32>
      %v467 = memref.load %v451[] : memref<f32>
      %v468 = arith.divf %v466, %v467 : f32
      memref.store %v468, %x[%i] : memref<?xf32>
    }
    %v469 = arith.constant 0 : i32
    return %v469 : i32
  }
  func.func @matmul(%xout: memref<?xf32>, %x: memref<?xf32>, %w: memref<?xf32>, %n: i32, %d: i32) -> i32 {
    // --- BEGIN SPAWN ON GPU ---
    %v470 = func.call @vx_plugin_dispatch_async_flat(%xout, %x, %w, %n, %d) : (!llvm.ptr<0>, !llvm.ptr<0>, !llvm.ptr<0>, i32, i32) -> i64
    // scf.execute_region { ... } (elided for flat C-ABI)
    // --- END SPAWN ON GPU ---
    %v471 = arith.constant 0 : i32
    return %v471 : i32
  }
  func.func @matmul_ane(%xout: memref<?xf32>, %x: memref<?xf32>, %w: memref<?xf32>, %n: i32, %d: i32) -> i32 {
    // --- BEGIN SPAWN ON ANE ---
    %v472 = func.call @vx_plugin_dispatch_async_flat(%xout, %x, %w, %n, %d) : (!llvm.ptr<0>, !llvm.ptr<0>, !llvm.ptr<0>, i32, i32) -> i64
    // scf.execute_region { ... } (elided for flat C-ABI)
    // --- END SPAWN ON ANE ---
    %v473 = arith.constant 0 : i32
    return %v473 : i32
  }
  func.func @main() -> i32 {
    %v475 = llvm.mlir.addressof @str_474 : !llvm.ptr<0>
    %v476 = func.call @vx_load_config(%v475) : (!llvm.ptr<0>) -> memref<?xi32>
    %v477 = arith.constant 0 : index
    %v478 = memref.load %v476[%v477] : memref<?x?xf32>
    %v479 = arith.constant 1 : index
    %v480 = memref.load %v476[%v479] : memref<?x?xf32>
    %v481 = arith.constant 2 : index
    %v482 = memref.load %v476[%v481] : memref<?x?xf32>
    %v483 = arith.constant 3 : index
    %v484 = memref.load %v476[%v483] : memref<?x?xf32>
    %v485 = arith.constant 4 : index
    %v486 = memref.load %v476[%v485] : memref<?x?xf32>
    %v487 = arith.constant 5 : index
    %v488 = memref.load %v476[%v487] : memref<?x?xf32>
    %v489 = arith.constant 6 : index
    %v490 = memref.load %v476[%v489] : memref<?x?xf32>
    %v491 = llvm.mlir.undef : !llvm.struct<"Config", (i32, i32, i32, i32, i32, i32, i32)>
    %v492 = arith.constant 288 : i32
    %v493 = llvm.insertvalue %v492, %v491[0] : !llvm.struct<"Config", (i32, i32, i32, i32, i32, i32, i32)>
    %v494 = arith.constant 768 : i32
    %v495 = llvm.insertvalue %v494, %v493[1] : !llvm.struct<"Config", (i32, i32, i32, i32, i32, i32, i32)>
    %v496 = arith.constant 6 : i32
    %v497 = llvm.insertvalue %v496, %v495[2] : !llvm.struct<"Config", (i32, i32, i32, i32, i32, i32, i32)>
    %v498 = arith.constant 6 : i32
    %v499 = llvm.insertvalue %v498, %v497[3] : !llvm.struct<"Config", (i32, i32, i32, i32, i32, i32, i32)>
    %v500 = arith.constant 6 : i32
    %v501 = llvm.insertvalue %v500, %v499[4] : !llvm.struct<"Config", (i32, i32, i32, i32, i32, i32, i32)>
    %v502 = arith.constant 32000 : i32
    %v503 = llvm.insertvalue %v502, %v501[5] : !llvm.struct<"Config", (i32, i32, i32, i32, i32, i32, i32)>
    %v504 = arith.constant 10000 : i32
    %v505 = llvm.insertvalue %v504, %v503[6] : !llvm.struct<"Config", (i32, i32, i32, i32, i32, i32, i32)>
    %v507 = llvm.mlir.addressof @str_506 : !llvm.ptr<0>
    %v508 = func.call @vx_load_weights(%v507) : (!llvm.ptr<0>) -> memref<?xf32>
    %v509 = llvm.extractvalue %v505[0] : !llvm.struct<"Config", (i32, i32, i32, i32, i32, i32, i32)>
    %v510 = llvm.extractvalue %v505[1] : !llvm.struct<"Config", (i32, i32, i32, i32, i32, i32, i32)>
    %v511 = llvm.extractvalue %v505[2] : !llvm.struct<"Config", (i32, i32, i32, i32, i32, i32, i32)>
    %v512 = llvm.extractvalue %v505[3] : !llvm.struct<"Config", (i32, i32, i32, i32, i32, i32, i32)>
    %v513 = llvm.extractvalue %v505[4] : !llvm.struct<"Config", (i32, i32, i32, i32, i32, i32, i32)>
    %v514 = llvm.extractvalue %v505[6] : !llvm.struct<"Config", (i32, i32, i32, i32, i32, i32, i32)>
    %v515 = llvm.extractvalue %v505[5] : !llvm.struct<"Config", (i32, i32, i32, i32, i32, i32, i32)>
    %v516 = func.call @memory_map_weights(%v508, %v509, %v510, %v511, %v512, %v513, %v514, %v515) : (memref<?xf32>, i32, i32, i32, i32, i32, i32, i32) -> !llvm.struct<"TransformerWeights", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>
    %v517 = llvm.extractvalue %v505[0] : !llvm.struct<"Config", (i32, i32, i32, i32, i32, i32, i32)>
    %v518 = llvm.extractvalue %v505[1] : !llvm.struct<"Config", (i32, i32, i32, i32, i32, i32, i32)>
    %v519 = llvm.extractvalue %v505[2] : !llvm.struct<"Config", (i32, i32, i32, i32, i32, i32, i32)>
    %v520 = llvm.extractvalue %v505[3] : !llvm.struct<"Config", (i32, i32, i32, i32, i32, i32, i32)>
    %v521 = llvm.extractvalue %v505[4] : !llvm.struct<"Config", (i32, i32, i32, i32, i32, i32, i32)>
    %v522 = llvm.extractvalue %v505[6] : !llvm.struct<"Config", (i32, i32, i32, i32, i32, i32, i32)>
    %v523 = llvm.extractvalue %v505[5] : !llvm.struct<"Config", (i32, i32, i32, i32, i32, i32, i32)>
    %v524 = func.call @malloc_run_state(%v517, %v518, %v519, %v520, %v521, %v522, %v523) : (i32, i32, i32, i32, i32, i32, i32) -> !llvm.struct<"RunState", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>
    %v526 = llvm.mlir.addressof @str_525 : !llvm.ptr<0>
    %v527 = llvm.extractvalue %v505[5] : !llvm.struct<"Config", (i32, i32, i32, i32, i32, i32, i32)>
    %v528 = func.call @vx_build_tokenizer(%v526, %v527) : (!llvm.ptr<0>, i32) -> memref<?xi8>
    %v530 = llvm.mlir.addressof @str_529 : !llvm.ptr<0>
    %v531 = func.call @vx_read_prompt_file(%v530) : (!llvm.ptr<0>) -> memref<?xi8>
    %v532 = memref.memory_space_cast %v528 : memref<?xi8> to memref<?x?xf32>
    %v533 = memref.memory_space_cast %v531 : memref<?xi8> to memref<?x?xf32>
    %v534 = func.call @vx_encode_prompt(%v532, %v533) : (memref<?x?xf32>, memref<?x?xf32>) -> memref<?xi32>
    %v535 = arith.constant 0 : index
    %v536 = memref.load %v534[%v535] : memref<?x?xi32>
    %v537 = arith.constant 1 : index
    %v538 = memref.load %v534[%v537] : memref<?x?xi32>
    %v539 = memref.alloca() : memref<i32>
    memref.store %v538, %v539[] : memref<i32>
    %v540 = arith.constant 0 : i32
    %v541 = memref.alloca() : memref<i32>
    memref.store %v540, %v541[] : memref<i32>
    %v542 = func.call @vx_get_time() : () -> f32
    %v543 = memref.alloca() : memref<f32>
    memref.store %v542, %v543[] : memref<f32>
    %v544 = arith.constant 1000 : i32
    %v545 = arith.constant 0 : index
    %v546 = arith.index_cast %v544 : i32 to index
    %v547 = arith.constant 1 : index
    scf.for %step = %v545 to %v546 step %v547 {
      %v548 = memref.load %v539[] : memref<i32>
      %v549 = memref.load %v541[] : memref<i32>
      // borrow value %v505 -> %v550
      %v551 = arith.constant 1 : i32
      %v550 = llvm.alloca %v551 x !llvm.struct<"Config", (i32, i32, i32, i32, i32, i32, i32)> : (i32) -> !llvm.ptr<0>
      llvm.store %v505, %v550 : !llvm.struct<"Config", (i32, i32, i32, i32, i32, i32, i32)>, !llvm.ptr<0>
      // borrow value %v524 -> %v552
      %v553 = arith.constant 1 : i32
      %v552 = llvm.alloca %v553 x !llvm.struct<"RunState", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)> : (i32) -> !llvm.ptr<0>
      llvm.store %v524, %v552 : !llvm.struct<"RunState", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>, !llvm.ptr<0>
      // borrow value %v516 -> %v554
      %v555 = arith.constant 1 : i32
      %v554 = llvm.alloca %v555 x !llvm.struct<"TransformerWeights", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)> : (i32) -> !llvm.ptr<0>
      llvm.store %v516, %v554 : !llvm.struct<"TransformerWeights", (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>, !llvm.ptr<0>
      %v556 = func.call @transformer(%v548, %v549, %v550, %v552, %v554) : (i32, i32, !llvm.ptr<0>, !llvm.ptr<0>, !llvm.ptr<0>) -> memref<?xf32>
      %v557 = arith.constant 0 : i32
      %v558 = memref.alloca() : memref<i32>
      memref.store %v557, %v558[] : memref<i32>
      %v559 = memref.load %v541[] : memref<i32>
      %v560 = arith.constant 1 : i32
      %v561 = arith.subi %v536, %v560 : i32
      %v563 = arith.cmpi "slt", %v559, %v561 : i32
      %v564 = scf.if %v563 -> (f32) {
        %v565 = memref.load %v541[] : memref<i32>
        %v566 = arith.constant 2 : i32
        %v567 = arith.addi %v565, %v566 : i32
        %v568 = arith.index_cast %v567 : i32 to index
        %v569 = memref.load %v534[%v568] : memref<?x?xi32>
        memref.store %v569, %v558[] : memref<i32>
        %v570 = arith.constant 0.0 : f32
        scf.yield %v570 : f32
      } else {
        %v571 = llvm.extractvalue %v505[5] : !llvm.struct<"Config", (i32, i32, i32, i32, i32, i32, i32)>
        %v572 = func.call @sample_argmax(%v556, %v571) : (memref<?xf32>, i32) -> i32
        memref.store %v572, %v558[] : memref<i32>
        %v573 = arith.constant 0.0 : f32
        scf.yield %v573 : f32
      }
      %v574 = memref.memory_space_cast %v528 : memref<?xi8> to memref<?x?xi32>
      %v575 = memref.load %v539[] : memref<i32>
      %v576 = memref.load %v558[] : memref<i32>
      %v577 = func.call @vx_decode_token(%v574, %v575, %v576) : (memref<?x?xi32>, i32, i32) -> memref<?xi8>
      %v578 = memref.load %v558[] : memref<i32>
      memref.store %v578, %v539[] : memref<i32>
      %v579 = arith.constant 1 : i32
      %v580 = memref.load %v541[] : memref<i32>
      %v581 = arith.addi %v580, %v579 : i32
      memref.store %v581, %v541[] : memref<i32>
      %v582 = memref.load %v541[] : memref<i32>
      %v583 = arith.constant 100 : i32
      %v584 = arith.divsi %v582, %v583 : i32
      %v585 = arith.constant 100 : i32
      %v586 = arith.muli %v584, %v585 : i32
      %v587 = memref.load %v541[] : memref<i32>
      %v588 = arith.subi %v587, %v586 : i32
      %v589 = arith.constant 0 : i32
      %v591 = arith.cmpi "eq", %v588, %v589 : i32
      scf.if %v591 {
        %v592 = func.call @vx_get_time() : () -> f32
        %v593 = memref.load %v543[] : memref<f32>
        %v594 = arith.subf %v592, %v593 : f32
        %v596 = llvm.mlir.addressof @str_595 : !llvm.ptr<0>
        %v597 = memref.load %v541[] : memref<i32>
        %v598 = func.call @vx_printf_i32(%v596, %v597) : (!llvm.ptr<0>, i32) -> i32
        %v599 = func.call @vx_print_float(%v594) : (f32) -> i32
        %v601 = llvm.mlir.addressof @str_600 : !llvm.ptr<0>
        %v602 = func.call @vx_safe_printf(%v601) : (!llvm.ptr<0>) -> i32
      }
      %v603 = arith.constant 0 : i64
    }
    %v604 = arith.constant 0 : i64
    %v605 = arith.constant 0 : i32
    return %v605 : i32
  }
func.func private @vx_plugin_dispatch_async_flat(!llvm.ptr<0>, !llvm.ptr<0>, !llvm.ptr<0>, i32, i32) -> i64
  llvm.mlir.global internal constant @str_474("stories15M.bin\00") {addr_space = 0 : i32} : !llvm.array<15 x i8>
  llvm.mlir.global internal constant @str_506("stories15M.bin\00") {addr_space = 0 : i32} : !llvm.array<15 x i8>
  llvm.mlir.global internal constant @str_525("tokenizer.bin\00") {addr_space = 0 : i32} : !llvm.array<14 x i8>
  llvm.mlir.global internal constant @str_529("prompt.txt\00") {addr_space = 0 : i32} : !llvm.array<11 x i8>
  llvm.mlir.global internal constant @str_595("\0A[%d tokens] Time elapsed: \00") {addr_space = 0 : i32} : !llvm.array<28 x i8>
  llvm.mlir.global internal constant @str_600(" seconds\0A\00") {addr_space = 0 : i32} : !llvm.array<10 x i8>
}

