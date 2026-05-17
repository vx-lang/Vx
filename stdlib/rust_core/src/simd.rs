use std::slice;

#[no_mangle]
pub extern "C" fn vx_simd_add_f32x4(a: *const f32, b: *const f32, out: *mut f32) -> i32 {
    if a.is_null() || b.is_null() || out.is_null() {
        return -1;
    }
    unsafe {
        let a_slice = slice::from_raw_parts(a, 4);
        let b_slice = slice::from_raw_parts(b, 4);
        let out_slice = slice::from_raw_parts_mut(out, 4);
        for i in 0..4 {
            out_slice[i] = a_slice[i] + b_slice[i];
        }
    }
    0
}

#[no_mangle]
pub extern "C" fn vx_simd_sub_f32x4(a: *const f32, b: *const f32, out: *mut f32) -> i32 {
    if a.is_null() || b.is_null() || out.is_null() {
        return -1;
    }
    unsafe {
        let a_slice = slice::from_raw_parts(a, 4);
        let b_slice = slice::from_raw_parts(b, 4);
        let out_slice = slice::from_raw_parts_mut(out, 4);
        for i in 0..4 {
            out_slice[i] = a_slice[i] - b_slice[i];
        }
    }
    0
}

#[no_mangle]
pub extern "C" fn vx_simd_mul_f32x4(a: *const f32, b: *const f32, out: *mut f32) -> i32 {
    if a.is_null() || b.is_null() || out.is_null() {
        return -1;
    }
    unsafe {
        let a_slice = slice::from_raw_parts(a, 4);
        let b_slice = slice::from_raw_parts(b, 4);
        let out_slice = slice::from_raw_parts_mut(out, 4);
        for i in 0..4 {
            out_slice[i] = a_slice[i] * b_slice[i];
        }
    }
    0
}

#[no_mangle]
pub extern "C" fn vx_simd_fma_f32x4(
    a: *const f32,
    b: *const f32,
    c: *const f32,
    out: *mut f32,
) -> i32 {
    if a.is_null() || b.is_null() || c.is_null() || out.is_null() {
        return -1;
    }
    unsafe {
        let a_slice = slice::from_raw_parts(a, 4);
        let b_slice = slice::from_raw_parts(b, 4);
        let c_slice = slice::from_raw_parts(c, 4);
        let out_slice = slice::from_raw_parts_mut(out, 4);
        for i in 0..4 {
            out_slice[i] = a_slice[i] * b_slice[i] + c_slice[i];
        }
    }
    0
}
