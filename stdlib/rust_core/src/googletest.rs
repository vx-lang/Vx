use googletest::prelude::*;

#[no_mangle]
pub extern "C" fn vx_googletest_expect_eq_f32(actual: f32, expected: f32) -> i32 {
    assert_that!(actual, eq(expected));
    0
}

#[no_mangle]
pub extern "C" fn vx_googletest_expect_eq_i32(actual: i32, expected: i32) -> i32 {
    assert_that!(actual, eq(expected));
    0
}
