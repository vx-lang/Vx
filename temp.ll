; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @printMemrefF32(i64, ptr)

declare void @printMemrefF64(i64, ptr)

declare void @printMemrefI32(i64, ptr)

declare void @printMemrefI64(i64, ptr)

declare void @printMemrefBF16(i64, ptr)

define private i1 @akar_dispatch_npu(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr %7, ptr %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, ptr %14, ptr %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20) {
  %22 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %0, 0
  %23 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, ptr %1, 1
  %24 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %23, i64 %2, 2
  %25 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, i64 %3, 3, 0
  %26 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, i64 %5, 4, 0
  %27 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %26, i64 %4, 3, 1
  %28 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %27, i64 %6, 4, 1
  %29 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, ptr %29, align 8
  %30 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %7, 0
  %31 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, ptr %8, 1
  %32 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, i64 %9, 2
  %33 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, i64 %10, 3, 0
  %34 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %33, i64 %12, 4, 0
  %35 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, i64 %11, 3, 1
  %36 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, i64 %13, 4, 1
  %37 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, ptr %37, align 8
  %38 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %14, 0
  %39 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %38, ptr %15, 1
  %40 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, i64 %16, 2
  %41 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, i64 %17, 3, 0
  %42 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, i64 %19, 4, 0
  %43 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %42, i64 %18, 3, 1
  %44 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %43, i64 %20, 4, 1
  %45 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %44, ptr %45, align 8
  %46 = call i1 @_mlir_ciface_akar_dispatch_npu(ptr %29, ptr %37, ptr %45)
  ret i1 %46
}

declare i1 @_mlir_ciface_akar_dispatch_npu(ptr, ptr, ptr)

define { ptr, ptr, i64, [2 x i64], [2 x i64] } @custom_matmul(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr %7, ptr %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13) {
  %15 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %7, 0
  %16 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, ptr %8, 1
  %17 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %16, i64 %9, 2
  %18 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %17, i64 %10, 3, 0
  %19 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %18, i64 %12, 4, 0
  %20 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %19, i64 %11, 3, 1
  %21 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %20, i64 %13, 4, 1
  %22 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %0, 0
  %23 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, ptr %1, 1
  %24 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %23, i64 %2, 2
  %25 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, i64 %3, 3, 0
  %26 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, i64 %5, 4, 0
  %27 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %26, i64 %4, 3, 1
  %28 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %27, i64 %6, 4, 1
  %29 = call ptr @malloc(i64 128)
  %30 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %29, 0
  %31 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, ptr %29, 1
  %32 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, i64 0, 2
  %33 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, i64 4, 3, 0
  %34 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %33, i64 4, 3, 1
  %35 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, i64 4, 4, 0
  %36 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, i64 1, 4, 1
  br label %37

37:                                               ; preds = %82, %14
  %38 = phi i64 [ %83, %82 ], [ 0, %14 ]
  %39 = icmp slt i64 %38, 4
  br i1 %39, label %40, label %84

40:                                               ; preds = %37
  br label %41

41:                                               ; preds = %80, %40
  %42 = phi i64 [ %81, %80 ], [ 0, %40 ]
  %43 = icmp slt i64 %42, 4
  br i1 %43, label %44, label %82

44:                                               ; preds = %41
  %45 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, 1
  %46 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, 4, 0
  %47 = mul nuw nsw i64 %38, %46
  %48 = add nuw nsw i64 %47, %42
  %49 = getelementptr inbounds nuw double, ptr %45, i64 %48
  store double 0.000000e+00, ptr %49, align 8
  br label %50

50:                                               ; preds = %53, %44
  %51 = phi i64 [ %79, %53 ], [ 0, %44 ]
  %52 = icmp slt i64 %51, 4
  br i1 %52, label %53, label %80

53:                                               ; preds = %50
  %54 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, 1
  %55 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, 4, 0
  %56 = mul nuw nsw i64 %38, %55
  %57 = add nuw nsw i64 %56, %42
  %58 = getelementptr inbounds nuw double, ptr %54, i64 %57
  %59 = load double, ptr %58, align 8
  %60 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 1
  %61 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 4, 0
  %62 = mul nuw nsw i64 %38, %61
  %63 = add nuw nsw i64 %62, %51
  %64 = getelementptr inbounds nuw double, ptr %60, i64 %63
  %65 = load double, ptr %64, align 8
  %66 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 1
  %67 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 4, 0
  %68 = mul nuw nsw i64 %51, %67
  %69 = add nuw nsw i64 %68, %42
  %70 = getelementptr inbounds nuw double, ptr %66, i64 %69
  %71 = load double, ptr %70, align 8
  %72 = fmul double %65, %71
  %73 = fadd double %59, %72
  %74 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, 1
  %75 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, 4, 0
  %76 = mul nuw nsw i64 %38, %75
  %77 = add nuw nsw i64 %76, %42
  %78 = getelementptr inbounds nuw double, ptr %74, i64 %77
  store double %73, ptr %78, align 8
  %79 = add i64 %51, 1
  br label %50

80:                                               ; preds = %50
  %81 = add i64 %42, 1
  br label %41

82:                                               ; preds = %41
  %83 = add i64 %38, 1
  br label %37

84:                                               ; preds = %37
  ret { ptr, ptr, i64, [2 x i64], [2 x i64] } %36
}

define i32 @main() {
  %1 = call ptr @malloc(i64 128)
  %2 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %1, 0
  %3 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %2, ptr %1, 1
  %4 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, i64 0, 2
  %5 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %4, i64 4, 3, 0
  %6 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %5, i64 4, 3, 1
  %7 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %6, i64 4, 4, 0
  %8 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %7, i64 1, 4, 1
  %9 = call ptr @malloc(i64 128)
  %10 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %9, 0
  %11 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %10, ptr %9, 1
  %12 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %11, i64 0, 2
  %13 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %12, i64 4, 3, 0
  %14 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, i64 4, 3, 1
  %15 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %14, i64 4, 4, 0
  %16 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, i64 1, 4, 1
  br label %17

17:                                               ; preds = %36, %0
  %18 = phi i64 [ %37, %36 ], [ 0, %0 ]
  %19 = icmp slt i64 %18, 4
  br i1 %19, label %20, label %38

20:                                               ; preds = %17
  br label %21

21:                                               ; preds = %24, %20
  %22 = phi i64 [ %35, %24 ], [ 0, %20 ]
  %23 = icmp slt i64 %22, 4
  br i1 %23, label %24, label %36

24:                                               ; preds = %21
  %25 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, 1
  %26 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, 4, 0
  %27 = mul nuw nsw i64 %18, %26
  %28 = add nuw nsw i64 %27, %22
  %29 = getelementptr inbounds nuw double, ptr %25, i64 %28
  store double 2.000000e+00, ptr %29, align 8
  %30 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %16, 1
  %31 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %16, 4, 0
  %32 = mul nuw nsw i64 %18, %31
  %33 = add nuw nsw i64 %32, %22
  %34 = getelementptr inbounds nuw double, ptr %30, i64 %33
  store double 3.000000e+00, ptr %34, align 8
  %35 = add i64 %22, 1
  br label %21

36:                                               ; preds = %21
  %37 = add i64 %18, 1
  br label %17

38:                                               ; preds = %17
  %39 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, 0
  %40 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, 1
  %41 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, 2
  %42 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, 3, 0
  %43 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, 3, 1
  %44 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, 4, 0
  %45 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, 4, 1
  %46 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %16, 0
  %47 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %16, 1
  %48 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %16, 2
  %49 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %16, 3, 0
  %50 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %16, 3, 1
  %51 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %16, 4, 0
  %52 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %16, 4, 1
  %53 = call { ptr, ptr, i64, [2 x i64], [2 x i64] } @custom_matmul(ptr %39, ptr %40, i64 %41, i64 %42, i64 %43, i64 %44, i64 %45, ptr %46, ptr %47, i64 %48, i64 %49, i64 %50, i64 %51, i64 %52)
  %54 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %53, ptr %54, align 8
  %55 = insertvalue { i64, ptr } { i64 2, ptr poison }, ptr %54, 1
  %56 = extractvalue { i64, ptr } %55, 0
  %57 = extractvalue { i64, ptr } %55, 1
  call void @printMemrefF64(i64 %56, ptr %57)
  ret i32 0
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
