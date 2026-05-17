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

        #[no_mangle]
        pub extern "C" fn vx_file_open(
            c_path: *const std::ffi::c_char,
            mode: i32,
        ) -> *mut std::ffi::c_void {
            if c_path.is_null() {
                return std::ptr::null_mut();
            }
            let path_str = unsafe { std::ffi::CStr::from_ptr(c_path) }.to_string_lossy();

            // Mode flags: 0 = read, 1 = write, 2 = read/write (create)
            let mut opts = std::fs::OpenOptions::new();
            if mode == 0 {
                opts.read(true);
            } else if mode == 1 {
                opts.write(true).create(true).truncate(true);
            } else {
                opts.read(true).write(true).create(true);
            }

            if let Ok(file) = opts.open(path_str.as_ref()) {
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
            len: usize,
        ) -> usize {
            if ptr.is_null() || buffer.is_null() || len == 0 {
                return 0;
            }
            let file = unsafe { &mut *(ptr as *mut std::fs::File) };
            let buf_slice = unsafe { std::slice::from_raw_parts_mut(buffer, len) };
            file.read(buf_slice).unwrap_or(0)
        }

        #[no_mangle]
        pub extern "C" fn vx_file_write(
            ptr: *mut std::ffi::c_void,
            buffer: *const u8,
            len: usize,
        ) -> usize {
            if ptr.is_null() || buffer.is_null() || len == 0 {
                return 0;
            }
            let file = unsafe { &mut *(ptr as *mut std::fs::File) };
            let buf_slice = unsafe { std::slice::from_raw_parts(buffer, len) };
            file.write(buf_slice).unwrap_or(0)
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
            len: usize,
        ) -> usize {
            if ptr.is_null() || buffer.is_null() || len == 0 {
                return 0;
            }
            let stream = unsafe { &mut *(ptr as *mut std::net::TcpStream) };
            let buf_slice = unsafe { std::slice::from_raw_parts_mut(buffer, len) };
            stream.read(buf_slice).unwrap_or(0)
        }

        #[no_mangle]
        pub extern "C" fn vx_tcp_stream_write(
            ptr: *mut std::ffi::c_void,
            buffer: *const u8,
            len: usize,
        ) -> usize {
            if ptr.is_null() || buffer.is_null() || len == 0 {
                return 0;
            }
            let stream = unsafe { &mut *(ptr as *mut std::net::TcpStream) };
            let buf_slice = unsafe { std::slice::from_raw_parts(buffer, len) };
            stream.write(buf_slice).unwrap_or(0)
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
            len: usize,
        ) -> usize {
            if ptr.is_null() || buffer.is_null() || len == 0 {
                return 0;
            }
            let socket = unsafe { &mut *(ptr as *mut std::net::UdpSocket) };
            let buf_slice = unsafe { std::slice::from_raw_parts_mut(buffer, len) };
            // Note: We ignore the peer address for simplicity in the FFI.
            socket.recv(buf_slice).unwrap_or(0)
        }

        #[no_mangle]
        pub extern "C" fn vx_udp_socket_send_to(
            ptr: *mut std::ffi::c_void,
            buffer: *const u8,
            len: usize,
            c_addr: *const std::ffi::c_char,
        ) -> usize {
            if ptr.is_null() || buffer.is_null() || len == 0 || c_addr.is_null() {
                return 0;
            }
            let socket = unsafe { &mut *(ptr as *mut std::net::UdpSocket) };
            let buf_slice = unsafe { std::slice::from_raw_parts(buffer, len) };
            let addr_str = unsafe { std::ffi::CStr::from_ptr(c_addr) }.to_string_lossy();
            socket.send_to(buf_slice, addr_str.as_ref()).unwrap_or(0)
        }

        #[no_mangle]
        pub extern "C" fn vx_udp_socket_drop(ptr: *mut std::ffi::c_void) {
            if !ptr.is_null() {
                let _ = unsafe { Box::from_raw(ptr as *mut std::net::UdpSocket) };
            }
        }
    };
}
