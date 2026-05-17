use crate::instantiate_vec_ffi;

// Instantiate FFI endpoints for primitive vectors
instantiate_vec_ffi!(i32, i32);
instantiate_vec_ffi!(f32, f32);
instantiate_vec_ffi!(i64, i64);
instantiate_vec_ffi!(f64, f64);
