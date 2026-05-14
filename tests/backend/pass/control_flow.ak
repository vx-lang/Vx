fn test_if(x: i32) -> i32 {
    let mut res: i32 = 0;
    if x > 10 {
        res = 1;
    } else if x == 10 {
        res = 2;
    } else {
        res = 3;
    }
    return res;
}

fn main() -> i32 {
    let a = test_if(15);
    let b = test_if(10);
    let c = test_if(5);
    return a + b + c; // 1 + 2 + 3 = 6
}

// CHECK: func.func @test_if(%arg0: i32) -> i32 {
// CHECK: = arith.cmp
// CHECK: scf.if %
// CHECK: } else {
// CHECK: = arith.cmp
// CHECK: scf.if %
// CHECK: } else {
// CHECK: }
// CHECK: }
