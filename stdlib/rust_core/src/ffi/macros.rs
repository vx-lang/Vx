/// Macro to instantiate C-ABI compatible FFI wrappers for `Vec<T>`.
///
/// This generates `vx_vec_new_<name>`, `vx_vec_push_<name>`,
/// `vx_vec_len_<name>`, and `vx_vec_drop_<name>` functions
/// that operate on an opaque `*mut std::ffi::c_void`.
#[macro_export]
macro_rules! instantiate_vec_ffi {
    ($type_name:ident, $type:ty) => {
        paste::paste! {
            #[no_mangle]
            pub extern "C" fn [<vx_vec_new_ $type_name>]() -> *mut std::ffi::c_void {
                let vec: Box<Vec<$type>> = Box::new(Vec::new());
                Box::into_raw(vec) as *mut std::ffi::c_void
            }

            #[no_mangle]
            pub extern "C" fn [<vx_vec_push_ $type_name>](ptr: *mut std::ffi::c_void, val: $type) {
                if ptr.is_null() {
                    return;
                }
                let vec = unsafe { &mut *(ptr as *mut Vec<$type>) };
                vec.push(val);
            }

            #[no_mangle]
            pub extern "C" fn [<vx_vec_len_ $type_name>](ptr: *mut std::ffi::c_void) -> usize {
                if ptr.is_null() {
                    return 0;
                }
                let vec = unsafe { &*(ptr as *mut Vec<$type>) };
                vec.len()
            }

            #[no_mangle]
            pub extern "C" fn [<vx_vec_drop_ $type_name>](ptr: *mut std::ffi::c_void) {
                if !ptr.is_null() {
                    let _ = unsafe { Box::from_raw(ptr as *mut Vec<$type>) };
                }
            }
        }
    };
}

/// Macro to instantiate C-ABI compatible FFI wrappers for `Option<T>`.
#[macro_export]
macro_rules! instantiate_option_ffi {
    ($type_name:ident, $type:ty) => {
        paste::paste! {
            #[no_mangle]
            pub extern "C" fn [<vx_option_new_some_ $type_name>](val: $type) -> *mut std::ffi::c_void {
                let opt: Box<Option<$type>> = Box::new(Some(val));
                Box::into_raw(opt) as *mut std::ffi::c_void
            }

            #[no_mangle]
            pub extern "C" fn [<vx_option_new_none_ $type_name>]() -> *mut std::ffi::c_void {
                let opt: Box<Option<$type>> = Box::new(None);
                Box::into_raw(opt) as *mut std::ffi::c_void
            }

            #[no_mangle]
            pub extern "C" fn [<vx_option_is_some_ $type_name>](ptr: *mut std::ffi::c_void) -> bool {
                if ptr.is_null() { return false; }
                let opt = unsafe { &*(ptr as *mut Option<$type>) };
                opt.is_some()
            }

            #[no_mangle]
            pub extern "C" fn [<vx_option_is_none_ $type_name>](ptr: *mut std::ffi::c_void) -> bool {
                if ptr.is_null() { return true; }
                let opt = unsafe { &*(ptr as *mut Option<$type>) };
                opt.is_none()
            }

            #[no_mangle]
            pub extern "C" fn [<vx_option_unwrap_ $type_name>](ptr: *mut std::ffi::c_void) -> $type {
                let opt = unsafe { &*(ptr as *mut Option<$type>) };
                opt.clone().unwrap()
            }

            #[no_mangle]
            pub extern "C" fn [<vx_option_drop_ $type_name>](ptr: *mut std::ffi::c_void) {
                if !ptr.is_null() {
                    let _ = unsafe { Box::from_raw(ptr as *mut Option<$type>) };
                }
            }
        }
    };
}

/// Macro to instantiate C-ABI compatible FFI wrappers for `Result<T, E>`.
#[macro_export]
macro_rules! instantiate_result_ffi {
    ($t_name:ident, $t_type:ty, $e_name:ident, $e_type:ty) => {
        paste::paste! {
            #[no_mangle]
            pub extern "C" fn [<vx_result_new_ok_ $t_name _ $e_name>](val: $t_type) -> *mut std::ffi::c_void {
                let res: Box<Result<$t_type, $e_type>> = Box::new(Ok(val));
                Box::into_raw(res) as *mut std::ffi::c_void
            }

            #[no_mangle]
            pub extern "C" fn [<vx_result_new_err_ $t_name _ $e_name>](err: $e_type) -> *mut std::ffi::c_void {
                let res: Box<Result<$t_type, $e_type>> = Box::new(Err(err));
                Box::into_raw(res) as *mut std::ffi::c_void
            }

            #[no_mangle]
            pub extern "C" fn [<vx_result_is_ok_ $t_name _ $e_name>](ptr: *mut std::ffi::c_void) -> bool {
                if ptr.is_null() { return false; }
                let res = unsafe { &*(ptr as *mut Result<$t_type, $e_type>) };
                res.is_ok()
            }

            #[no_mangle]
            pub extern "C" fn [<vx_result_is_err_ $t_name _ $e_name>](ptr: *mut std::ffi::c_void) -> bool {
                if ptr.is_null() { return false; }
                let res = unsafe { &*(ptr as *mut Result<$t_type, $e_type>) };
                res.is_err()
            }

            #[no_mangle]
            pub extern "C" fn [<vx_result_unwrap_ $t_name _ $e_name>](ptr: *mut std::ffi::c_void) -> $t_type {
                let res = unsafe { &*(ptr as *mut Result<$t_type, $e_type>) };
                res.clone().unwrap()
            }

            #[no_mangle]
            pub extern "C" fn [<vx_result_drop_ $t_name _ $e_name>](ptr: *mut std::ffi::c_void) {
                if !ptr.is_null() {
                    let _ = unsafe { Box::from_raw(ptr as *mut Result<$t_type, $e_type>) };
                }
            }
        }
    };
}
