use crate::instantiate_option_ffi;

// Instantiate FFI endpoints for primitive options
instantiate_option_ffi!(i32, i32);
instantiate_option_ffi!(f32, f32);
instantiate_option_ffi!(i64, i64);
instantiate_option_ffi!(f64, f64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_option_some_i32_ffi() {
        let ptr = vx_option_new_some_i32(42);
        assert!(!ptr.is_null());
        assert!(vx_option_is_some_i32(ptr));
        assert!(!vx_option_is_none_i32(ptr));
        assert_eq!(vx_option_unwrap_i32(ptr), 42);
        vx_option_drop_i32(ptr);
    }

    #[test]
    fn test_option_none_i32_ffi() {
        let ptr = vx_option_new_none_i32();
        assert!(!ptr.is_null());
        assert!(!vx_option_is_some_i32(ptr));
        assert!(vx_option_is_none_i32(ptr));
        vx_option_drop_i32(ptr);
    }
}
