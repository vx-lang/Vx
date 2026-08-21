// Benchmark versions of the two dot lowerings: same bodies as
// scripts/dot_lowering_compare/two_dots.mlir, but grid-strided so the launch
// is representative of how the flat path actually runs (one thread per query
// row, each thread doing many sequential dots over the key loop).
//
// N rows of 64 f32. Every thread walks rows [start, N) by grid stride.
module attributes {gpu.container_module} {
  gpu.module @m {

    gpu.func @bench_wide(%q: memref<?x64xf32>, %k: memref<?x64xf32>,
                         %o: memref<?xf32>, %n: index) kernel {
      %c0 = arith.constant 0 : index
      %tid = gpu.thread_id x
      %bid = gpu.block_id x
      %bdim = gpu.block_dim x
      %gdim = gpu.grid_dim x
      %bo = arith.muli %bid, %bdim : index
      %start = arith.addi %bo, %tid : index
      %stride = arith.muli %gdim, %bdim : index
      scf.for %i = %start to %n step %stride {
        %a = vector.load %q[%i, %c0] {alignment = 16 : i64} : memref<?x64xf32>, vector<64xf32>
        %b = vector.load %k[%i, %c0] {alignment = 16 : i64} : memref<?x64xf32>, vector<64xf32>
        %p = arith.mulf %a, %b : vector<64xf32>
        %s = vector.reduction <add>, %p : vector<64xf32> into f32
        memref.store %s, %o[%i] : memref<?xf32>
      }
      gpu.return
    }

    gpu.func @bench_chunked(%q: memref<?x64xf32>, %k: memref<?x64xf32>,
                            %o: memref<?xf32>, %n: index) kernel {
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %z = arith.constant dense<0.000000e+00> : vector<8xf32>
      %tid = gpu.thread_id x
      %bid = gpu.block_id x
      %bdim = gpu.block_dim x
      %gdim = gpu.grid_dim x
      %bo = arith.muli %bid, %bdim : index
      %start = arith.addi %bo, %tid : index
      %stride = arith.muli %gdim, %bdim : index
      scf.for %i = %start to %n step %stride {
        %acc = scf.for %j = %c0 to %c64 step %c8
                       iter_args(%a = %z) -> (vector<8xf32>) {
          %x = vector.load %q[%i, %j] {alignment = 16 : i64} : memref<?x64xf32>, vector<8xf32>
          %y = vector.load %k[%i, %j] {alignment = 16 : i64} : memref<?x64xf32>, vector<8xf32>
          %nn = vector.fma %x, %y, %a : vector<8xf32>
          scf.yield %nn : vector<8xf32>
        }
        %s = vector.reduction <add>, %acc : vector<8xf32> into f32
        memref.store %s, %o[%i] : memref<?xf32>
      }
      gpu.return
    }

  }
}
