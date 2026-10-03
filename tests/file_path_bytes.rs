// This test expands the same raw-pointer FFI macro as the production library.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

// A Unix filename is a byte sequence; opening it must not change invalid UTF-8 bytes.
include!("../stdlib/rust_core/src/ffi/macros.rs");
instantiate_file_ffi!();

#[cfg(unix)]
#[test]
fn file_open_preserves_non_utf8_path_bytes() -> Result<(), String> {
    use std::os::unix::ffi::OsStrExt;

    let dir = tempfile::tempdir().map_err(|error| format!("create test directory: {error}"))?;
    let mut bytes = dir.path().as_os_str().as_bytes().to_vec();
    bytes.extend_from_slice(b"/file_");
    bytes.push(0xff);
    let raw_path = std::path::Path::new(std::ffi::OsStr::from_bytes(&bytes));
    let replacement_path = dir.path().join("file_\u{fffd}");
    std::fs::write(&replacement_path, b"W")
        .map_err(|error| format!("create replacement-character file: {error}"))?;
    // Some Unix filesystems reject this name. In that case opening the replacement
    // file would still prove that the FFI changed the caller's path.
    let raw_path_exists = match std::fs::write(raw_path, b"R") {
        Ok(()) => true,
        Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied => false,
        Err(error) => return Err(format!("create file with a non-UTF-8 name: {error}")),
    };

    let c_path = std::ffi::CString::new(bytes)
        .map_err(|error| format!("path contains a NUL byte: {error}"))?;
    let file = vx_file_open(c_path.as_ptr(), 0);
    if !raw_path_exists {
        if !file.is_null() {
            vx_file_drop(file);
            return Err("vx_file_open changed the path bytes and opened a different file".into());
        }
        return Ok(());
    }
    if file.is_null() {
        return Err("vx_file_open could not open the raw path".into());
    }

    let mut byte = [0u8; 1];
    let count = vx_file_read(file, byte.as_mut_ptr(), byte.len() as u64);
    vx_file_drop(file);
    if count != 1 {
        return Err(format!(
            "the opened file should contain one byte, read {count}"
        ));
    }
    if byte != *b"R" {
        return Err(format!(
            "vx_file_open opened a different filename: read {byte:?}"
        ));
    }
    Ok(())
}
