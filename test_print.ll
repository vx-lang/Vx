; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @printMemrefF32(i64, ptr)

define i32 @main() {
  %1 = call ptr @malloc(i64 64)
  %2 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %1, 0
  %3 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %2, ptr %1, 1
  %4 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, i64 0, 2
  %5 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %4, i64 4, 3, 0
  %6 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %5, i64 4, 3, 1
  %7 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %6, i64 4, 4, 0
  %8 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %7, i64 1, 4, 1
  br label %9

9:                                                ; preds = %22, %0
  %10 = phi i64 [ %23, %22 ], [ 0, %0 ]
  %11 = icmp slt i64 %10, 4
  br i1 %11, label %12, label %24

12:                                               ; preds = %9
  br label %13

13:                                               ; preds = %16, %12
  %14 = phi i64 [ %21, %16 ], [ 0, %12 ]
  %15 = icmp slt i64 %14, 4
  br i1 %15, label %16, label %22

16:                                               ; preds = %13
  %17 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, 1
  %18 = mul nuw nsw i64 %10, 4
  %19 = add nuw nsw i64 %18, %14
  %20 = getelementptr inbounds nuw float, ptr %17, i64 %19
  store float 4.200000e+01, ptr %20, align 4
  %21 = add i64 %14, 1
  br label %13

22:                                               ; preds = %13
  %23 = add i64 %10, 1
  br label %9

24:                                               ; preds = %9
  %25 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, ptr %25, align 8
  %26 = insertvalue { i64, ptr } { i64 2, ptr poison }, ptr %25, 1
  %27 = extractvalue { i64, ptr } %26, 0
  %28 = extractvalue { i64, ptr } %26, 1
  call void @printMemrefF32(i64 %27, ptr %28)
  ret i32 0
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
