/// The value of the environment variable that the Vx string `name` names, for
/// `Environment::var`. 0 and a new Vx string in `*value`, 1 when the environment has no such
/// variable, 2 when the value is not UTF-8. A name that is empty or holds `=` or NUL names no
/// variable, so it gives 1.
#[no_mangle]
pub extern "C" fn vx_env_try_var(
    name: *const std::ffi::c_void,
    value: *mut *mut std::ffi::c_void,
) -> i32 {
    assert!(!name.is_null() && !value.is_null());
    let name = unsafe { &*(name as *const String) };
    if name.is_empty() || name.contains(['=', '\0']) {
        return 1;
    }
    match std::env::var_os(name) {
        None => 1,
        Some(os) => match os.into_string() {
            Ok(text) => {
                unsafe { *value = Box::into_raw(Box::new(text)) as *mut std::ffi::c_void };
                0
            }
            Err(_) => 2,
        },
    }
}
