use crate::instantiate_vec_ffi;

// Instantiate FFI endpoints for primitive vectors
instantiate_vec_ffi!(i32, i32);
instantiate_vec_ffi!(f32, f32);
instantiate_vec_ffi!(i64, i64);
instantiate_vec_ffi!(f64, f64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec_i32_ffi() {
        let ptr = vx_vec_new_i32();
        assert!(!ptr.is_null());
        assert_eq!(vx_vec_len_i32(ptr), 0);

        vx_vec_push_i32(ptr, 42);
        assert_eq!(vx_vec_len_i32(ptr), 1);

        vx_vec_push_i32(ptr, 100);
        assert_eq!(vx_vec_len_i32(ptr), 2);

        vx_vec_drop_i32(ptr);
    }

    #[test]
    fn test_vec_f32_ffi() {
        let ptr = vx_vec_new_f32();
        assert!(!ptr.is_null());
        assert_eq!(vx_vec_len_f32(ptr), 0);

        vx_vec_push_f32(ptr, 3.14);
        assert_eq!(vx_vec_len_f32(ptr), 1);

        vx_vec_drop_f32(ptr);
    }
}
