//===- macros.rs - Vx Compiler ---------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides macros used to simplify FFI definitions.
// It contains helper macros that automatically generate the boilerplate required
// to safely expose Rust functions to the Vx compiler's MLIR execution engine.
//
//===----------------------------------------------------------------------===//
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

/// Macro to instantiate C-ABI compatible FFI wrappers for `HashMap<K, V>`.
#[macro_export]
macro_rules! instantiate_hash_map_ffi {
    ($name:ident, $k:ty, $v:ty) => {
        paste::paste! {
            #[no_mangle]
            pub extern "C" fn [<vx_hash_map_new_ $name>]() -> *mut std::ffi::c_void {
                let map: Box<std::collections::HashMap<$k, $v>> = Box::new(std::collections::HashMap::new());
                Box::into_raw(map) as *mut std::ffi::c_void
            }

            #[no_mangle]
            pub extern "C" fn [<vx_hash_map_insert_ $name>](ptr: *mut std::ffi::c_void, key: $k, val: $v) {
                if !ptr.is_null() {
                    let map = unsafe { &mut *(ptr as *mut std::collections::HashMap<$k, $v>) };
                    map.insert(key, val);
                }
            }

            #[no_mangle]
            pub extern "C" fn [<vx_hash_map_get_ $name>](ptr: *mut std::ffi::c_void, key: $k) -> *mut std::ffi::c_void {
                if ptr.is_null() { return std::ptr::null_mut(); }
                let map = unsafe { &*(ptr as *mut std::collections::HashMap<$k, $v>) };
                let opt: Box<Option<$v>> = Box::new(map.get(&key).copied());
                Box::into_raw(opt) as *mut std::ffi::c_void
            }

            #[no_mangle]
            pub extern "C" fn [<vx_hash_map_contains_key_ $name>](ptr: *mut std::ffi::c_void, key: $k) -> bool {
                if ptr.is_null() { return false; }
                let map = unsafe { &*(ptr as *mut std::collections::HashMap<$k, $v>) };
                map.contains_key(&key)
            }

            #[no_mangle]
            pub extern "C" fn [<vx_hash_map_len_ $name>](ptr: *mut std::ffi::c_void) -> usize {
                if ptr.is_null() { return 0; }
                let map = unsafe { &*(ptr as *mut std::collections::HashMap<$k, $v>) };
                map.len()
            }

            #[no_mangle]
            pub extern "C" fn [<vx_hash_map_drop_ $name>](ptr: *mut std::ffi::c_void) {
                if !ptr.is_null() {
                    let _ = unsafe { Box::from_raw(ptr as *mut std::collections::HashMap<$k, $v>) };
                }
            }
        }
    };
}

/// Macro to instantiate C-ABI compatible FFI wrappers for `HashSet<T>`.
#[macro_export]
macro_rules! instantiate_hash_set_ffi {
    ($name:ident, $t:ty) => {
        paste::paste! {
            #[no_mangle]
            pub extern "C" fn [<vx_hash_set_new_ $name>]() -> *mut std::ffi::c_void {
                let set: Box<std::collections::HashSet<$t>> = Box::new(std::collections::HashSet::new());
                Box::into_raw(set) as *mut std::ffi::c_void
            }

            #[no_mangle]
            pub extern "C" fn [<vx_hash_set_insert_ $name>](ptr: *mut std::ffi::c_void, val: $t) {
                if !ptr.is_null() {
                    let set = unsafe { &mut *(ptr as *mut std::collections::HashSet<$t>) };
                    set.insert(val);
                }
            }

            #[no_mangle]
            pub extern "C" fn [<vx_hash_set_contains_ $name>](ptr: *mut std::ffi::c_void, val: $t) -> bool {
                if ptr.is_null() { return false; }
                let set = unsafe { &*(ptr as *mut std::collections::HashSet<$t>) };
                set.contains(&val)
            }

            #[no_mangle]
            pub extern "C" fn [<vx_hash_set_len_ $name>](ptr: *mut std::ffi::c_void) -> usize {
                if ptr.is_null() { return 0; }
                let set = unsafe { &*(ptr as *mut std::collections::HashSet<$t>) };
                set.len()
            }

            #[no_mangle]
            pub extern "C" fn [<vx_hash_set_drop_ $name>](ptr: *mut std::ffi::c_void) {
                if !ptr.is_null() {
                    let _ = unsafe { Box::from_raw(ptr as *mut std::collections::HashSet<$t>) };
                }
            }
        }
    };
}

/// Macro to instantiate C-ABI compatible FFI wrappers for `String`.
#[macro_export]
macro_rules! instantiate_string_ffi {
    () => {
        #[no_mangle]
        pub extern "C" fn vx_string_new() -> *mut std::ffi::c_void {
            let s: Box<String> = Box::new(String::new());
            Box::into_raw(s) as *mut std::ffi::c_void
        }

        #[no_mangle]
        pub extern "C" fn vx_i32_to_string(val: i32) -> *mut std::ffi::c_void {
            let s: Box<String> = Box::new(val.to_string());
            Box::into_raw(s) as *mut std::ffi::c_void
        }

        #[no_mangle]
        pub extern "C" fn vx_string_from_c_str(
            c_str: *const std::ffi::c_char,
        ) -> *mut std::ffi::c_void {
            if c_str.is_null() {
                return vx_string_new();
            }
            let c_str = unsafe { std::ffi::CStr::from_ptr(c_str) };
            let s: Box<String> = Box::new(c_str.to_string_lossy().into_owned());
            Box::into_raw(s) as *mut std::ffi::c_void
        }

        #[no_mangle]
        pub extern "C" fn vx_string_push_c_str(
            ptr: *mut std::ffi::c_void,
            c_str: *const std::ffi::c_char,
        ) {
            if ptr.is_null() || c_str.is_null() {
                return;
            }
            let s = unsafe { &mut *(ptr as *mut String) };
            let c_str = unsafe { std::ffi::CStr::from_ptr(c_str) };
            s.push_str(&c_str.to_string_lossy());
        }

        #[no_mangle]
        pub extern "C" fn vx_string_len(ptr: *mut std::ffi::c_void) -> usize {
            if ptr.is_null() {
                return 0;
            }
            let s = unsafe { &*(ptr as *mut String) };
            s.len()
        }

        #[no_mangle]
        pub extern "C" fn vx_string_as_c_str(
            ptr: *mut std::ffi::c_void,
        ) -> *const std::ffi::c_char {
            if ptr.is_null() {
                return std::ptr::null();
            }
            // Note: This forces an allocation of a CString, which we must leak or store.
            // For simplicity and safety in a quick binding, we could return a leaked ptr,
            // but that's a memory leak.
            // A better approach is to modify the string in place to append a null byte if it doesn't have one,
            // or just use `as_ptr()` if we know it won't be used past its lifetime, but it needs a null terminator.
            // For now, let's leak a CString.
            let s = unsafe { &*(ptr as *mut String) };
            let c_string = std::ffi::CString::new(s.clone()).unwrap();
            c_string.into_raw()
        }

        #[no_mangle]
        pub extern "C" fn vx_string_free_c_str(c_str: *mut std::ffi::c_char) {
            if !c_str.is_null() {
                let _ = unsafe { std::ffi::CString::from_raw(c_str) };
            }
        }

        #[no_mangle]
        pub extern "C" fn vx_string_drop(ptr: *mut std::ffi::c_void) {
            if !ptr.is_null() {
                let _ = unsafe { Box::from_raw(ptr as *mut String) };
            }
        }
    };
}

/// Macro to instantiate C-ABI compatible FFI wrappers for `std::fs::File`.
#[macro_export]
macro_rules! instantiate_file_ffi {
    () => {
        use std::io::{Read, Seek, SeekFrom, Write};

        // Private to the bridge, but part of its i32 ABI. Keep these values in sync with
        // the table in docs/implementation_plans/io_error_contract.md.
        const VX_FILE_STATUS_OK: i32 = 0;
        const VX_FILE_STATUS_INVALID_ARGUMENT: i32 = -1;
        const VX_FILE_ERROR_NOT_FOUND: i32 = 1;
        const VX_FILE_ERROR_PERMISSION_DENIED: i32 = 2;
        const VX_FILE_ERROR_ALREADY_EXISTS: i32 = 3;
        const VX_FILE_ERROR_INVALID_INPUT: i32 = 4;
        const VX_FILE_ERROR_INTERRUPTED: i32 = 5;
        const VX_FILE_ERROR_WOULD_BLOCK: i32 = 6;
        const VX_FILE_ERROR_UNEXPECTED_EOF: i32 = 7;
        const VX_FILE_ERROR_WRITE_ZERO: i32 = 8;
        const VX_FILE_ERROR_OUT_OF_MEMORY: i32 = 9;
        const VX_FILE_ERROR_LIMIT_EXCEEDED: i32 = 10;
        const VX_FILE_ERROR_UNSUPPORTED: i32 = 11;
        const VX_FILE_ERROR_OTHER: i32 = 12;

        fn vx_file_error_kind(kind: std::io::ErrorKind) -> i32 {
            match kind {
                std::io::ErrorKind::NotFound => VX_FILE_ERROR_NOT_FOUND,
                std::io::ErrorKind::PermissionDenied => VX_FILE_ERROR_PERMISSION_DENIED,
                std::io::ErrorKind::AlreadyExists => VX_FILE_ERROR_ALREADY_EXISTS,
                std::io::ErrorKind::InvalidInput => VX_FILE_ERROR_INVALID_INPUT,
                std::io::ErrorKind::Interrupted => VX_FILE_ERROR_INTERRUPTED,
                std::io::ErrorKind::WouldBlock => VX_FILE_ERROR_WOULD_BLOCK,
                std::io::ErrorKind::UnexpectedEof => VX_FILE_ERROR_UNEXPECTED_EOF,
                std::io::ErrorKind::WriteZero => VX_FILE_ERROR_WRITE_ZERO,
                std::io::ErrorKind::OutOfMemory => VX_FILE_ERROR_OUT_OF_MEMORY,
                std::io::ErrorKind::StorageFull
                | std::io::ErrorKind::QuotaExceeded
                | std::io::ErrorKind::FileTooLarge => VX_FILE_ERROR_LIMIT_EXCEEDED,
                std::io::ErrorKind::Unsupported => VX_FILE_ERROR_UNSUPPORTED,
                _ => VX_FILE_ERROR_OTHER,
            }
        }

        /// # Safety
        /// Every non-null output must be aligned, writable, and disjoint from the others.
        unsafe fn vx_file_clear_transfer_outputs(
            count: *mut u64,
            native_code: *mut i32,
            has_native_code: *mut bool,
        ) {
            // Clear every provided output even when another output pointer is null.
            unsafe {
                if !count.is_null() {
                    *count = 0;
                }
                if !native_code.is_null() {
                    *native_code = 0;
                }
                if !has_native_code.is_null() {
                    *has_native_code = false;
                }
            }
        }

        /// # Safety
        /// Both output pointers must be non-null, aligned, writable, and disjoint.
        unsafe fn vx_file_store_error(
            error: &std::io::Error,
            native_code: *mut i32,
            has_native_code: *mut bool,
        ) -> i32 {
            let captured_code = error.raw_os_error();
            unsafe {
                *has_native_code = captured_code.is_some();
                *native_code = captured_code.unwrap_or(0);
            }
            vx_file_error_kind(error.kind())
        }

        #[no_mangle]
        pub extern "C" fn vx_file_open(
            c_path: *const std::ffi::c_char,
            mode: i32,
        ) -> *mut std::ffi::c_void {
            // The Vx OpenMode variants map to these three values.
            let mut opts = std::fs::OpenOptions::new();
            match mode {
                0 => {
                    opts.read(true);
                }
                1 => {
                    opts.write(true).create(true).truncate(true);
                }
                2 => {
                    opts.read(true).write(true).create(true);
                }
                _ => {
                    eprintln!("vx_file_open received unknown mode {mode}");
                    std::process::abort();
                }
            }

            if c_path.is_null() {
                return std::ptr::null_mut();
            }
            let c_path = unsafe { std::ffi::CStr::from_ptr(c_path) };
            #[cfg(unix)]
            let path = {
                use std::os::unix::ffi::OsStrExt;
                std::ffi::OsStr::from_bytes(c_path.to_bytes())
            };
            #[cfg(not(unix))]
            let path = match c_path.to_str() {
                Ok(path) => path,
                Err(_) => return std::ptr::null_mut(),
            };

            if let Ok(file) = opts.open(path) {
                let boxed: Box<std::fs::File> = Box::new(file);
                Box::into_raw(boxed) as *mut std::ffi::c_void
            } else {
                std::ptr::null_mut()
            }
        }

        #[no_mangle]
        pub extern "C" fn vx_file_read(
            ptr: *mut std::ffi::c_void,
            buffer: *mut u8,
            len: u64,
        ) -> u64 {
            if ptr.is_null() || buffer.is_null() || len == 0 || len > isize::MAX as u64 {
                return 0;
            }
            let file = unsafe { &mut *(ptr as *mut std::fs::File) };
            let buf_slice = unsafe { std::slice::from_raw_parts_mut(buffer, len as usize) };
            file.read(buf_slice).unwrap_or(0) as u64
        }

        #[no_mangle]
        pub extern "C" fn vx_file_write(
            ptr: *mut std::ffi::c_void,
            buffer: *const u8,
            len: u64,
        ) -> u64 {
            if ptr.is_null() || buffer.is_null() || len == 0 || len > isize::MAX as u64 {
                return 0;
            }
            let file = unsafe { &mut *(ptr as *mut std::fs::File) };
            let buf_slice = unsafe { std::slice::from_raw_parts(buffer, len as usize) };
            file.write(buf_slice).unwrap_or(0) as u64
        }

        /// Zero is success, -1 is invalid arguments, and 1..=12 is a portable
        /// OS error category. Every non-null output is initialized on every path:
        /// count is zero except on success; native_code is meaningful only when
        /// has_native_code is true. See io_error_contract.md for the ABI table.
        /// The native code is copied before classification or other work.
        ///
        /// # Safety
        /// A null `ptr` or `buffer` is accepted and reports invalid arguments. A
        /// non-null `ptr` must point to a live bridge-owned `std::fs::File`, with no
        /// concurrent access or drop. When `ptr` and `buffer` are non-null and
        /// `0 < len <= isize::MAX`, `buffer` must point to `len` initialized,
        /// writable bytes. Every non-null output pointer must be aligned and
        /// writable, even if another argument is invalid. The file, buffer,
        /// and outputs must not overlap for the duration of the call.
        #[no_mangle]
        pub unsafe extern "C" fn vx_file_try_read(
            ptr: *mut std::ffi::c_void,
            buffer: *mut u8,
            len: u64,
            count: *mut u64,
            native_code: *mut i32,
            has_native_code: *mut bool,
        ) -> i32 {
            unsafe { vx_file_clear_transfer_outputs(count, native_code, has_native_code) };
            if count.is_null() || native_code.is_null() || has_native_code.is_null() {
                return VX_FILE_STATUS_INVALID_ARGUMENT;
            }
            if ptr.is_null() || len > isize::MAX as u64 || (len != 0 && buffer.is_null()) {
                return VX_FILE_STATUS_INVALID_ARGUMENT;
            }
            if len == 0 {
                return VX_FILE_STATUS_OK;
            }
            let file = unsafe { &mut *(ptr as *mut std::fs::File) };
            let buf_slice = unsafe { std::slice::from_raw_parts_mut(buffer, len as usize) };
            match file.read(buf_slice) {
                Ok(read_count) => {
                    unsafe { *count = read_count as u64 };
                    VX_FILE_STATUS_OK
                }
                Err(error) => unsafe { vx_file_store_error(&error, native_code, has_native_code) },
            }
        }

        /// Uses the same status and output contract as `vx_file_try_read`.
        ///
        /// # Safety
        /// A null `ptr` or `buffer` is accepted and reports invalid arguments. A
        /// non-null `ptr` must point to a live bridge-owned `std::fs::File`, with no
        /// concurrent access or drop. When `ptr` and `buffer` are non-null and
        /// `0 < len <= isize::MAX`, `buffer` must point to `len` initialized,
        /// readable bytes. Every non-null output pointer must be aligned and
        /// writable, even if another argument is invalid. The file, buffer,
        /// and outputs must not overlap for the duration of the call.
        #[no_mangle]
        pub unsafe extern "C" fn vx_file_try_write(
            ptr: *mut std::ffi::c_void,
            buffer: *const u8,
            len: u64,
            count: *mut u64,
            native_code: *mut i32,
            has_native_code: *mut bool,
        ) -> i32 {
            unsafe { vx_file_clear_transfer_outputs(count, native_code, has_native_code) };
            if count.is_null() || native_code.is_null() || has_native_code.is_null() {
                return VX_FILE_STATUS_INVALID_ARGUMENT;
            }
            if ptr.is_null() || len > isize::MAX as u64 || (len != 0 && buffer.is_null()) {
                return VX_FILE_STATUS_INVALID_ARGUMENT;
            }
            if len == 0 {
                return VX_FILE_STATUS_OK;
            }
            let file = unsafe { &mut *(ptr as *mut std::fs::File) };
            let buf_slice = unsafe { std::slice::from_raw_parts(buffer, len as usize) };
            match file.write(buf_slice) {
                Ok(written_count) => {
                    unsafe { *count = written_count as u64 };
                    VX_FILE_STATUS_OK
                }
                Err(error) => unsafe { vx_file_store_error(&error, native_code, has_native_code) },
            }
        }

        #[no_mangle]
        pub extern "C" fn vx_file_seek(
            ptr: *mut std::ffi::c_void,
            offset: i64,
            whence: i32,
        ) -> i64 {
            if ptr.is_null() {
                return -1;
            }
            let file = unsafe { &mut *(ptr as *mut std::fs::File) };
            let seek_from = match whence {
                0 => SeekFrom::Start(offset as u64),
                1 => SeekFrom::Current(offset),
                2 => SeekFrom::End(offset),
                _ => return -1,
            };
            file.seek(seek_from).map(|pos| pos as i64).unwrap_or(-1)
        }

        #[no_mangle]
        pub extern "C" fn vx_get_temp_file(
            c_filename: *const std::ffi::c_char,
        ) -> *mut std::ffi::c_char {
            let filename = unsafe { std::ffi::CStr::from_ptr(c_filename) }.to_string_lossy();
            let mut path = std::env::temp_dir();
            path.push(filename.as_ref());
            let path_str = path.to_string_lossy().into_owned();
            let c_string = std::ffi::CString::new(path_str).unwrap();
            c_string.into_raw()
        }

        #[no_mangle]
        pub extern "C" fn vx_free_temp_file(ptr: *mut std::ffi::c_char) -> i32 {
            if !ptr.is_null() {
                let _ = unsafe { std::ffi::CString::from_raw(ptr) };
            }
            0
        }

        #[no_mangle]
        pub extern "C" fn vx_file_drop(ptr: *mut std::ffi::c_void) {
            if !ptr.is_null() {
                let _ = unsafe { Box::from_raw(ptr as *mut std::fs::File) };
            }
        }
    };
}

/// Macro to instantiate C-ABI compatible FFI wrappers for `std::net::TcpStream`.
#[macro_export]
macro_rules! instantiate_tcp_stream_ffi {
    () => {
        use std::io::{Read, Write};

        #[no_mangle]
        pub extern "C" fn vx_tcp_stream_connect(
            c_addr: *const std::ffi::c_char,
        ) -> *mut std::ffi::c_void {
            if c_addr.is_null() {
                return std::ptr::null_mut();
            }
            let addr_str = unsafe { std::ffi::CStr::from_ptr(c_addr) }.to_string_lossy();
            if let Ok(stream) = std::net::TcpStream::connect(addr_str.as_ref()) {
                let boxed: Box<std::net::TcpStream> = Box::new(stream);
                Box::into_raw(boxed) as *mut std::ffi::c_void
            } else {
                std::ptr::null_mut()
            }
        }

        #[no_mangle]
        pub extern "C" fn vx_tcp_stream_read(
            ptr: *mut std::ffi::c_void,
            buffer: *mut u8,
            len: u64,
        ) -> u64 {
            if ptr.is_null() || buffer.is_null() || len == 0 || len > isize::MAX as u64 {
                return 0;
            }
            let stream = unsafe { &mut *(ptr as *mut std::net::TcpStream) };
            let buf_slice = unsafe { std::slice::from_raw_parts_mut(buffer, len as usize) };
            stream.read(buf_slice).unwrap_or(0) as u64
        }

        #[no_mangle]
        pub extern "C" fn vx_tcp_stream_write(
            ptr: *mut std::ffi::c_void,
            buffer: *const u8,
            len: u64,
        ) -> u64 {
            if ptr.is_null() || buffer.is_null() || len == 0 || len > isize::MAX as u64 {
                return 0;
            }
            let stream = unsafe { &mut *(ptr as *mut std::net::TcpStream) };
            let buf_slice = unsafe { std::slice::from_raw_parts(buffer, len as usize) };
            stream.write(buf_slice).unwrap_or(0) as u64
        }

        #[no_mangle]
        pub extern "C" fn vx_tcp_stream_drop(ptr: *mut std::ffi::c_void) {
            if !ptr.is_null() {
                let _ = unsafe { Box::from_raw(ptr as *mut std::net::TcpStream) };
            }
        }
    };
}

/// Macro to instantiate C-ABI compatible FFI wrappers for `std::net::UdpSocket`.
#[macro_export]
macro_rules! instantiate_udp_socket_ffi {
    () => {
        #[no_mangle]
        pub extern "C" fn vx_udp_socket_bind(
            c_addr: *const std::ffi::c_char,
        ) -> *mut std::ffi::c_void {
            if c_addr.is_null() {
                return std::ptr::null_mut();
            }
            let addr_str = unsafe { std::ffi::CStr::from_ptr(c_addr) }.to_string_lossy();
            if let Ok(socket) = std::net::UdpSocket::bind(addr_str.as_ref()) {
                let boxed: Box<std::net::UdpSocket> = Box::new(socket);
                Box::into_raw(boxed) as *mut std::ffi::c_void
            } else {
                std::ptr::null_mut()
            }
        }

        #[no_mangle]
        pub extern "C" fn vx_udp_socket_recv(
            ptr: *mut std::ffi::c_void,
            buffer: *mut u8,
            len: u64,
        ) -> u64 {
            if ptr.is_null() || buffer.is_null() || len == 0 || len > isize::MAX as u64 {
                return 0;
            }
            let socket = unsafe { &mut *(ptr as *mut std::net::UdpSocket) };
            let buf_slice = unsafe { std::slice::from_raw_parts_mut(buffer, len as usize) };
            // Note: We ignore the peer address for simplicity in the FFI.
            socket.recv(buf_slice).unwrap_or(0) as u64
        }

        #[no_mangle]
        pub extern "C" fn vx_udp_socket_send_to(
            ptr: *mut std::ffi::c_void,
            buffer: *const u8,
            len: u64,
            c_addr: *const std::ffi::c_char,
        ) -> u64 {
            if ptr.is_null()
                || buffer.is_null()
                || len == 0
                || len > isize::MAX as u64
                || c_addr.is_null()
            {
                return 0;
            }
            let socket = unsafe { &mut *(ptr as *mut std::net::UdpSocket) };
            let buf_slice = unsafe { std::slice::from_raw_parts(buffer, len as usize) };
            let addr_str = unsafe { std::ffi::CStr::from_ptr(c_addr) }.to_string_lossy();
            socket.send_to(buf_slice, addr_str.as_ref()).unwrap_or(0) as u64
        }

        #[no_mangle]
        pub extern "C" fn vx_udp_socket_drop(ptr: *mut std::ffi::c_void) {
            if !ptr.is_null() {
                let _ = unsafe { Box::from_raw(ptr as *mut std::net::UdpSocket) };
            }
        }
    };
}

/// Macro to instantiate C-ABI compatible FFI wrappers for `std::net::TcpListener`.
#[macro_export]
macro_rules! instantiate_tcp_listener_ffi {
    () => {
        #[no_mangle]
        pub extern "C" fn vx_tcp_listener_bind(
            c_addr: *const std::ffi::c_char,
        ) -> *mut std::ffi::c_void {
            if c_addr.is_null() {
                return std::ptr::null_mut();
            }
            let addr_str = unsafe { std::ffi::CStr::from_ptr(c_addr) }.to_string_lossy();
            if let Ok(listener) = std::net::TcpListener::bind(addr_str.as_ref()) {
                let boxed: Box<std::net::TcpListener> = Box::new(listener);
                Box::into_raw(boxed) as *mut std::ffi::c_void
            } else {
                std::ptr::null_mut()
            }
        }

        #[no_mangle]
        pub extern "C" fn vx_tcp_listener_accept(
            ptr: *mut std::ffi::c_void,
        ) -> *mut std::ffi::c_void {
            if ptr.is_null() {
                return std::ptr::null_mut();
            }
            let listener = unsafe { &mut *(ptr as *mut std::net::TcpListener) };
            if let Ok((stream, _)) = listener.accept() {
                let boxed: Box<std::net::TcpStream> = Box::new(stream);
                Box::into_raw(boxed) as *mut std::ffi::c_void
            } else {
                std::ptr::null_mut()
            }
        }

        #[no_mangle]
        pub extern "C" fn vx_tcp_listener_drop(ptr: *mut std::ffi::c_void) {
            if !ptr.is_null() {
                let _ = unsafe { Box::from_raw(ptr as *mut std::net::TcpListener) };
            }
        }
    };
}

/// Macro to instantiate C-ABI compatible FFI wrappers for Standard I/O Streams.
#[macro_export]
macro_rules! instantiate_stdio_ffi {
    () => {
        #[no_mangle]
        pub extern "C" fn vx_stdout_write(buffer: *const u8, len: u64) -> u64 {
            if buffer.is_null() || len == 0 || len > isize::MAX as u64 {
                return 0;
            }
            let buf_slice = unsafe { std::slice::from_raw_parts(buffer, len as usize) };
            std::io::Write::write(&mut std::io::stdout(), buf_slice).unwrap_or(0) as u64
        }

        #[no_mangle]
        pub extern "C" fn vx_stderr_write(buffer: *const u8, len: u64) -> u64 {
            if buffer.is_null() || len == 0 || len > isize::MAX as u64 {
                return 0;
            }
            let buf_slice = unsafe { std::slice::from_raw_parts(buffer, len as usize) };
            std::io::Write::write(&mut std::io::stderr(), buf_slice).unwrap_or(0) as u64
        }

        #[no_mangle]
        pub extern "C" fn vx_stdin_read(buffer: *mut u8, len: u64) -> u64 {
            if buffer.is_null() || len == 0 || len > isize::MAX as u64 {
                return 0;
            }
            let buf_slice = unsafe { std::slice::from_raw_parts_mut(buffer, len as usize) };
            std::io::Read::read(&mut std::io::stdin(), buf_slice).unwrap_or(0) as u64
        }
    };
}
