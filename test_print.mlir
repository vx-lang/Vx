module {
  func.func private @printMemrefF32(memref<*xf32>)
  func.func @main() -> i32 {
    %0 = arith.constant 0 : index
    %1 = arith.constant 4 : index
    %2 = arith.constant 1 : index
    %3 = memref.alloc() : memref<4x4xf32>
    scf.for %i = %0 to %1 step %2 {
      scf.for %j = %0 to %1 step %2 {
        %val = arith.constant 42.0 : f32
        memref.store %val, %3[%i, %j] : memref<4x4xf32>
      }
    }
    %unranked = memref.cast %3 : memref<4x4xf32> to memref<*xf32>
    func.call @printMemrefF32(%unranked) : (memref<*xf32>) -> ()
    %zero = arith.constant 0 : i32
    return %zero : i32
  }
}
