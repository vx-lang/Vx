#[no_mangle]
pub extern "C" fn vx_math_sin_f32(x: f32) -> f32 {
    x.sin()
}

#[no_mangle]
pub extern "C" fn vx_math_cos_f32(x: f32) -> f32 {
    x.cos()
}

#[no_mangle]
pub extern "C" fn vx_math_tan_f32(x: f32) -> f32 {
    x.tan()
}

#[no_mangle]
pub extern "C" fn vx_math_asin_f32(x: f32) -> f32 {
    x.asin()
}

#[no_mangle]
pub extern "C" fn vx_math_acos_f32(x: f32) -> f32 {
    x.acos()
}

#[no_mangle]
pub extern "C" fn vx_math_atan_f32(x: f32) -> f32 {
    x.atan()
}

#[no_mangle]
pub extern "C" fn vx_math_abs_f32(x: f32) -> f32 {
    x.abs()
}

#[no_mangle]
pub extern "C" fn vx_math_sqrt_f32(x: f32) -> f32 {
    x.sqrt()
}

#[no_mangle]
pub extern "C" fn vx_math_exp_f32(x: f32) -> f32 {
    x.exp()
}

#[no_mangle]
pub extern "C" fn vx_math_log_f32(x: f32) -> f32 {
    x.ln()
}

#[no_mangle]
pub extern "C" fn vx_math_log2_f32(x: f32) -> f32 {
    x.log2()
}

#[no_mangle]
pub extern "C" fn vx_math_log10_f32(x: f32) -> f32 {
    x.log10()
}

#[no_mangle]
pub extern "C" fn vx_math_sin_f64(x: f64) -> f64 {
    x.sin()
}

#[no_mangle]
pub extern "C" fn vx_math_cos_f64(x: f64) -> f64 {
    x.cos()
}

#[no_mangle]
pub extern "C" fn vx_math_tan_f64(x: f64) -> f64 {
    x.tan()
}

#[no_mangle]
pub extern "C" fn vx_math_abs_f64(x: f64) -> f64 {
    x.abs()
}

#[no_mangle]
pub extern "C" fn vx_math_sqrt_f64(x: f64) -> f64 {
    x.sqrt()
}

#[no_mangle]
pub extern "C" fn vx_math_exp_f64(x: f64) -> f64 {
    x.exp()
}

#[no_mangle]
pub extern "C" fn vx_math_log_f64(x: f64) -> f64 {
    x.ln()
}

#[no_mangle]
pub extern "C" fn vx_math_asin_f64(x: f64) -> f64 {
    x.asin()
}

#[no_mangle]
pub extern "C" fn vx_math_acos_f64(x: f64) -> f64 {
    x.acos()
}

#[no_mangle]
pub extern "C" fn vx_math_atan_f64(x: f64) -> f64 {
    x.atan()
}

#[no_mangle]
pub extern "C" fn vx_math_log2_f64(x: f64) -> f64 {
    x.log2()
}

#[no_mangle]
pub extern "C" fn vx_math_log10_f64(x: f64) -> f64 {
    x.log10()
}
