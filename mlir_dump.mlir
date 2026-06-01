module {
  func.func private @printMemrefF32(memref<*xf32>)
  func.func private @printMemrefF64(memref<*xf64>)
  func.func private @printMemrefI32(memref<*xi32>)
  func.func private @printMemrefI64(memref<*xi64>)
  func.func private @printMemrefBF16(memref<*xbf16>)
  func.func private @malloc(i32) -> !llvm.ptr
  func.func private @realloc(!llvm.ptr, i32) -> !llvm.ptr
  func.func private @free(!llvm.ptr) -> i32
  func.func @Range_next(%arg0: !llvm.ptr) -> !llvm.struct<"Option_i32", (i32, i32)> {
    %c1_i32 = arith.constant 1 : i32
    %0 = llvm.mlir.undef : !llvm.struct<"Option_i32", (i32, i32)>
    %1 = llvm.insertvalue %c1_i32, %0[0] : !llvm.struct<"Option_i32", (i32, i32)> 
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.alloca %2 x !llvm.struct<"Option_i32", (i32, i32)> : (i32) -> !llvm.ptr
    llvm.store %1, %3 : !llvm.struct<"Option_i32", (i32, i32)>, !llvm.ptr
    %4 = llvm.getelementptr %arg0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"Range", (i32, i32)>
    %5 = llvm.load %4 : !llvm.ptr -> i32
    %6 = llvm.getelementptr %arg0[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"Range", (i32, i32)>
    %7 = llvm.load %6 : !llvm.ptr -> i32
    %8 = arith.cmpi slt, %5, %7 : i32
    scf.if %8 {
      %10 = llvm.getelementptr %arg0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"Range", (i32, i32)>
      %11 = llvm.load %10 : !llvm.ptr -> i32
      %12 = llvm.getelementptr %arg0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"Range", (i32, i32)>
      %13 = llvm.load %12 : !llvm.ptr -> i32
      %c1_i32_0 = arith.constant 1 : i32
      %14 = arith.addi %13, %c1_i32_0 : i32
      %15 = llvm.getelementptr %arg0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"Range", (i32, i32)>
      llvm.store %14, %15 : i32, !llvm.ptr
      %c0_i32_1 = arith.constant 0 : i32
      %16 = llvm.mlir.undef : !llvm.struct<"Option_i32", (i32, i32)>
      %17 = llvm.insertvalue %c0_i32_1, %16[0] : !llvm.struct<"Option_i32", (i32, i32)> 
      %18 = llvm.insertvalue %11, %17[1] : !llvm.struct<"Option_i32", (i32, i32)> 
      llvm.store %18, %3 : !llvm.struct<"Option_i32", (i32, i32)>, !llvm.ptr
    } else {
      %c1_i32_0 = arith.constant 1 : i32
      %10 = llvm.mlir.undef : !llvm.struct<"Option_i32", (i32, i32)>
      %11 = llvm.insertvalue %c1_i32_0, %10[0] : !llvm.struct<"Option_i32", (i32, i32)> 
      llvm.store %11, %3 : !llvm.struct<"Option_i32", (i32, i32)>, !llvm.ptr
    }
    %c0_i32 = arith.constant 0 : i32
    %9 = llvm.load %3 : !llvm.ptr -> !llvm.struct<"Option_i32", (i32, i32)>
    return %9 : !llvm.struct<"Option_i32", (i32, i32)>
  }
  func.func @Map_next(%arg0: !llvm.ptr) -> !llvm.struct<"Option_i32", (i32, i32)> {
    %0 = llvm.getelementptr %arg0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"Map", (struct<"Range", (i32, i32)>, ptr)>
    %1 = llvm.load %0 : !llvm.ptr -> !llvm.struct<"Range", (i32, i32)>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.alloca %2 x !llvm.struct<"Range", (i32, i32)> : (i32) -> !llvm.ptr
    llvm.store %1, %3 : !llvm.struct<"Range", (i32, i32)>, !llvm.ptr
    %4 = call @Range_next(%3) : (!llvm.ptr) -> !llvm.struct<"Option_i32", (i32, i32)>
    %c1_i32 = arith.constant 1 : i32
    %5 = llvm.mlir.undef : !llvm.struct<"Option_i32", (i32, i32)>
    %6 = llvm.insertvalue %c1_i32, %5[0] : !llvm.struct<"Option_i32", (i32, i32)> 
    %7 = llvm.mlir.constant(1 : i32) : i32
    %8 = llvm.alloca %7 x !llvm.struct<"Option_i32", (i32, i32)> : (i32) -> !llvm.ptr
    llvm.store %6, %8 : !llvm.struct<"Option_i32", (i32, i32)>, !llvm.ptr
    %c0_i32 = arith.constant 0 : i32
    %9 = llvm.extractvalue %4[0] : !llvm.struct<"Option_i32", (i32, i32)> 
    %10 = arith.cmpi eq, %9, %c0_i32 : i32
    scf.if %10 {
      %12 = llvm.extractvalue %4[1] : !llvm.struct<"Option_i32", (i32, i32)> 
      %13 = llvm.getelementptr %arg0[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"Map", (struct<"Range", (i32, i32)>, ptr)>
      %14 = llvm.load %13 : !llvm.ptr -> !llvm.ptr
      %c0_i32_1 = arith.constant 0 : i32
      %15 = llvm.mlir.undef : !llvm.struct<"Option_i32", (i32, i32)>
      %16 = llvm.insertvalue %c0_i32_1, %15[0] : !llvm.struct<"Option_i32", (i32, i32)> 
      %17 = builtin.unrealized_conversion_cast %14 : !llvm.ptr to (i32) -> i32
      %18 = func.call_indirect %17(%12) : (i32) -> i32
      %19 = llvm.insertvalue %18, %16[1] : !llvm.struct<"Option_i32", (i32, i32)> 
      llvm.store %19, %8 : !llvm.struct<"Option_i32", (i32, i32)>, !llvm.ptr
    } else {
      %c1_i32_1 = arith.constant 1 : i32
      %12 = llvm.extractvalue %4[0] : !llvm.struct<"Option_i32", (i32, i32)> 
      %13 = arith.cmpi eq, %12, %c1_i32_1 : i32
      scf.if %13 {
        %c1_i32_2 = arith.constant 1 : i32
        %14 = llvm.mlir.undef : !llvm.struct<"Option_i32", (i32, i32)>
        %15 = llvm.insertvalue %c1_i32_2, %14[0] : !llvm.struct<"Option_i32", (i32, i32)> 
        llvm.store %15, %8 : !llvm.struct<"Option_i32", (i32, i32)>, !llvm.ptr
      } else {
      }
    }
    %c0_i32_0 = arith.constant 0 : i32
    %11 = llvm.load %8 : !llvm.ptr -> !llvm.struct<"Option_i32", (i32, i32)>
    return %11 : !llvm.struct<"Option_i32", (i32, i32)>
  }
  func.func @map_range(%arg0: !llvm.struct<"Range", (i32, i32)>, %arg1: !llvm.ptr) -> !llvm.struct<"Map", (struct<"Range", (i32, i32)>, ptr)> {
    %0 = llvm.mlir.undef : !llvm.struct<"Map", (struct<"Range", (i32, i32)>, ptr)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<"Map", (struct<"Range", (i32, i32)>, ptr)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<"Map", (struct<"Range", (i32, i32)>, ptr)> 
    return %2 : !llvm.struct<"Map", (struct<"Range", (i32, i32)>, ptr)>
  }
  func.func @square(%arg0: i32) -> i32 {
    %0 = arith.muli %arg0, %arg0 : i32
    return %0 : i32
  }
  func.func @main() -> i32 attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.undef : !llvm.struct<"Range", (i32, i32)>
    %c0_i32 = arith.constant 0 : i32
    %1 = llvm.insertvalue %c0_i32, %0[0] : !llvm.struct<"Range", (i32, i32)> 
    %c5_i32 = arith.constant 5 : i32
    %2 = llvm.insertvalue %c5_i32, %1[1] : !llvm.struct<"Range", (i32, i32)> 
    %3 = llvm.mlir.constant(1 : i32) : i32
    %4 = llvm.alloca %3 x !llvm.struct<"Range", (i32, i32)> : (i32) -> !llvm.ptr
    llvm.store %2, %4 : !llvm.struct<"Range", (i32, i32)>, !llvm.ptr
    %5 = llvm.load %4 : !llvm.ptr -> !llvm.struct<"Range", (i32, i32)>
    %f = constant @square : (i32) -> i32
    %6 = builtin.unrealized_conversion_cast %f : (i32) -> i32 to !llvm.ptr
    %7 = call @map_range(%5, %6) : (!llvm.struct<"Range", (i32, i32)>, !llvm.ptr) -> !llvm.struct<"Map", (struct<"Range", (i32, i32)>, ptr)>
    %8 = llvm.mlir.constant(1 : i32) : i32
    %9 = llvm.alloca %8 x !llvm.struct<"Map", (struct<"Range", (i32, i32)>, ptr)> : (i32) -> !llvm.ptr
    llvm.store %7, %9 : !llvm.struct<"Map", (struct<"Range", (i32, i32)>, ptr)>, !llvm.ptr
    %c0_i32_0 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<i32>
    memref.store %c0_i32_0, %alloca[] : memref<i32>
    %alloca_1 = memref.alloca() : memref<1xi1>
    %alloca_2 = memref.alloca() : memref<1xi1>
    %false = arith.constant false
    %c0 = arith.constant 0 : index
    memref.store %false, %alloca_1[%c0] : memref<1xi1>
    scf.while : () -> () {
      memref.store %false, %alloca_2[%c0] : memref<1xi1>
      %10 = memref.load %alloca_1[%c0] : memref<1xi1>
      %true = arith.constant true
      %11 = arith.xori %10, %true : i1
      scf.condition(%11)
    } do {
      %10 = func.call @Map_next(%9) : (!llvm.ptr) -> !llvm.struct<"Option_i32", (i32, i32)>
      %c0_i32_4 = arith.constant 0 : i32
      %11 = llvm.extractvalue %10[0] : !llvm.struct<"Option_i32", (i32, i32)> 
      %12 = arith.cmpi eq, %11, %c0_i32_4 : i32
      scf.if %12 {
        %13 = llvm.extractvalue %10[1] : !llvm.struct<"Option_i32", (i32, i32)> 
        %14 = memref.load %alloca[] : memref<i32>
        %15 = arith.addi %14, %13 : i32
        memref.store %15, %alloca[] : memref<i32>
      } else {
        %c1_i32 = arith.constant 1 : i32
        %13 = llvm.extractvalue %10[0] : !llvm.struct<"Option_i32", (i32, i32)> 
        %14 = arith.cmpi eq, %13, %c1_i32 : i32
        scf.if %14 {
          %true = arith.constant true
          %c0_6 = arith.constant 0 : index
          memref.store %true, %alloca_1[%c0_6] : memref<1xi1>
        } else {
        }
      }
      %c0_i32_5 = arith.constant 0 : i32
      scf.yield
    }
    %c0_i32_3 = arith.constant 0 : i32
    return %c0_i32_3 : i32
  }
}
