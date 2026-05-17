use crate::instantiate_result_ffi;

// Common primitive results. Error codes represented as i32.
instantiate_result_ffi!(i32, i32, i32, i32);
instantiate_result_ffi!(f32, f32, i32, i32);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_result_ok_i32_i32_ffi() {
        let ptr = vx_result_new_ok_i32_i32(42);
        assert!(!ptr.is_null());
        assert!(vx_result_is_ok_i32_i32(ptr));
        assert!(!vx_result_is_err_i32_i32(ptr));
        assert_eq!(vx_result_unwrap_i32_i32(ptr), 42);
        vx_result_drop_i32_i32(ptr);
    }

    #[test]
    fn test_result_err_i32_i32_ffi() {
        let ptr = vx_result_new_err_i32_i32(-1);
        assert!(!ptr.is_null());
        assert!(!vx_result_is_ok_i32_i32(ptr));
        assert!(vx_result_is_err_i32_i32(ptr));
        vx_result_drop_i32_i32(ptr);
    }
}
