// Two kernels computing the SAME thing: o[i] = dot(q[i], k[i]) over 64-wide f32 rows.
//
//   @dot_wide     what src/codegen/flat.rs:2235 emits today -- load the whole row
//                 as vector<64xf32>, multiply, reduce.
//   @dot_chunked  what Vx#382 proposes -- walk the row in vector<8xf32> chunks,
//                 accumulating with vector.fma, and reduce the 8-lane accumulator once.
//
// The {alignment = 16} on each vector.load is what flat.rs emits (vector_align_attr);
// without it LLVM cannot form 128-bit loads and scalarizes to one ld per element.
// Same inputs, same output, same pipeline. The only difference is the shape of the
// reduction, which is exactly the change #382 asks for.
module attributes {gpu.container_module} {
  gpu.module @m {

    gpu.func @dot_wide(%q: memref<128x64xf32>, %k: memref<128x64xf32>,
                       %o: memref<128xf32>) kernel {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c128 = arith.constant 128 : index
      scf.for %i = %c0 to %c128 step %c1 {
        %a = vector.load %q[%i, %c0] {alignment = 16 : i64} : memref<128x64xf32>, vector<64xf32>
        %b = vector.load %k[%i, %c0] {alignment = 16 : i64} : memref<128x64xf32>, vector<64xf32>
        %p = arith.mulf %a, %b : vector<64xf32>
        %s = vector.reduction <add>, %p : vector<64xf32> into f32
        memref.store %s, %o[%i] : memref<128xf32>
      }
      gpu.return
    }

    gpu.func @dot_chunked(%q: memref<128x64xf32>, %k: memref<128x64xf32>,
                          %o: memref<128xf32>) kernel {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %z = arith.constant dense<0.000000e+00> : vector<8xf32>
      scf.for %i = %c0 to %c128 step %c1 {
        %acc = scf.for %j = %c0 to %c64 step %c8
                       iter_args(%a = %z) -> (vector<8xf32>) {
          %x = vector.load %q[%i, %j] {alignment = 16 : i64} : memref<128x64xf32>, vector<8xf32>
          %y = vector.load %k[%i, %j] {alignment = 16 : i64} : memref<128x64xf32>, vector<8xf32>
          %n = vector.fma %x, %y, %a : vector<8xf32>
          scf.yield %n : vector<8xf32>
        }
        %s = vector.reduction <add>, %acc : vector<8xf32> into f32
        memref.store %s, %o[%i] : memref<128xf32>
      }
      gpu.return
    }

  }
}
