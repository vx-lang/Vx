module {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @printMemrefF32(i64, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @main() -> i32 {
    %0 = llvm.mlir.constant(0 : i32) : i32
    %1 = llvm.mlir.constant(4.200000e+01 : f32) : f32
    %2 = llvm.mlir.constant(0 : index) : i64
    %3 = llvm.mlir.constant(4 : index) : i64
    %4 = llvm.mlir.constant(1 : index) : i64
    %5 = llvm.mlir.constant(4 : index) : i64
    %6 = llvm.mlir.constant(4 : index) : i64
    %7 = llvm.mlir.constant(1 : index) : i64
    %8 = llvm.mlir.constant(16 : index) : i64
    %9 = llvm.mlir.zero : !llvm.ptr
    %10 = llvm.getelementptr %9[%8] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %11 = llvm.ptrtoint %10 : !llvm.ptr to i64
    %12 = llvm.call @malloc(%11) : (i64) -> !llvm.ptr
    %13 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %14 = llvm.insertvalue %12, %13[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %15 = llvm.insertvalue %12, %14[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.mlir.constant(0 : index) : i64
    %17 = llvm.insertvalue %16, %15[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %18 = llvm.insertvalue %5, %17[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %19 = llvm.insertvalue %6, %18[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %20 = llvm.insertvalue %6, %19[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %21 = llvm.insertvalue %7, %20[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb1(%2 : i64)
  ^bb1(%22: i64):  // 2 preds: ^bb0, ^bb5
    %23 = llvm.icmp "slt" %22, %3 : i64
    llvm.cond_br %23, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%2 : i64)
  ^bb3(%24: i64):  // 2 preds: ^bb2, ^bb4
    %25 = llvm.icmp "slt" %24, %3 : i64
    llvm.cond_br %25, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %26 = llvm.extractvalue %21[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %27 = llvm.mlir.constant(4 : index) : i64
    %28 = llvm.mul %22, %27 overflow<nsw, nuw> : i64
    %29 = llvm.add %28, %24 overflow<nsw, nuw> : i64
    %30 = llvm.getelementptr inbounds|nuw %26[%29] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %1, %30 : f32, !llvm.ptr
    %31 = llvm.add %24, %4 : i64
    llvm.br ^bb3(%31 : i64)
  ^bb5:  // pred: ^bb3
    %32 = llvm.add %22, %4 : i64
    llvm.br ^bb1(%32 : i64)
  ^bb6:  // pred: ^bb1
    %33 = llvm.mlir.constant(1 : index) : i64
    %34 = llvm.alloca %33 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %21, %34 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %35 = llvm.mlir.constant(2 : index) : i64
    %36 = llvm.mlir.poison : !llvm.struct<(i64, ptr)>
    %37 = llvm.insertvalue %35, %36[0] : !llvm.struct<(i64, ptr)> 
    %38 = llvm.insertvalue %34, %37[1] : !llvm.struct<(i64, ptr)> 
    %39 = llvm.extractvalue %38[0] : !llvm.struct<(i64, ptr)> 
    %40 = llvm.extractvalue %38[1] : !llvm.struct<(i64, ptr)> 
    llvm.call @printMemrefF32(%39, %40) : (i64, !llvm.ptr) -> ()
    llvm.return %0 : i32
  }
}

