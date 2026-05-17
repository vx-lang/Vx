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
