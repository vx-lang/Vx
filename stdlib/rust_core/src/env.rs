use std::ffi::CString;

#[no_mangle]
pub extern "C" fn vx_env_args_count() -> i32 {
    let args: Vec<String> = std::env::args().collect();
    let mut start_idx = 0;

    // When executing via the JIT engine (lli), the process name is `lli` and the script
    // is passed as a `.ll` file. We filter these out.
    // When running as a standalone compiled executable, args[0] is the binary itself.
    for (i, arg) in args.iter().enumerate() {
        if arg.ends_with(".ll") {
            start_idx = i + 1;
            break;
        }
    }

    args.len().saturating_sub(start_idx) as i32
}

/// The argument at `idx`, as a C string the caller may keep. `c_char`, not `i8`: that is what
/// `CString::into_raw` returns, and it is `u8` on aarch64 Linux.
#[no_mangle]
pub extern "C" fn vx_env_arg(idx: i32) -> *mut std::ffi::c_char {
    let args: Vec<String> = std::env::args().collect();
    let mut start_idx = 0;

    for (i, arg) in args.iter().enumerate() {
        if arg.ends_with(".ll") {
            start_idx = i + 1;
            break;
        }
    }

    let actual_idx = start_idx + idx as usize;
    if idx < 0 || actual_idx >= args.len() {
        return std::ptr::null_mut();
    }

    // We intentionally leak the memory here so that the C caller can access the string
    let c_str = CString::new(args[actual_idx].clone()).unwrap();
    c_str.into_raw()
}
