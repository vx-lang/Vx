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
    let raw_path_exists = std::fs::write(raw_path, b"R").is_ok();

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

#[cfg(not(unix))]
#[test]
fn file_open_rejects_non_utf8_path_bytes() -> Result<(), String> {
    let dir = tempfile::tempdir().map_err(|error| format!("create test directory: {error}"))?;
    let path_prefix = dir.path().join("file_");
    let prefix = path_prefix
        .to_str()
        .ok_or_else(|| "test directory path is not UTF-8".to_string())?;
    let replacement_path = format!("{prefix}\u{fffd}");
    std::fs::write(&replacement_path, b"W")
        .map_err(|error| format!("create replacement-character file: {error}"))?;

    let control_path = std::ffi::CString::new(replacement_path.as_bytes())
        .map_err(|error| format!("control path contains a NUL byte: {error}"))?;
    let control = vx_file_open(control_path.as_ptr(), 0);
    if control.is_null() {
        return Err("vx_file_open could not open the replacement-character file".into());
    }
    vx_file_drop(control);

    let mut invalid_bytes = prefix.as_bytes().to_vec();
    invalid_bytes.push(0xff);
    let c_path = std::ffi::CString::new(invalid_bytes)
        .map_err(|error| format!("path contains a NUL byte: {error}"))?;
    let file = vx_file_open(c_path.as_ptr(), 0);
    if !file.is_null() {
        vx_file_drop(file);
        return Err("vx_file_open accepted an invalid UTF-8 path".into());
    }
    Ok(())
}

// vx_file_open stores a boxed Rust File behind its opaque pointer.
fn into_ffi_file(file: std::fs::File) -> *mut std::ffi::c_void {
    Box::into_raw(Box::new(file)).cast()
}

#[test]
fn file_read_rejects_a_length_larger_than_a_rust_slice() -> Result<(), String> {
    let mut file = tempfile::tempfile().map_err(|error| format!("create file: {error}"))?;
    file.write_all(b"R")
        .map_err(|error| format!("write test byte: {error}"))?;
    file.rewind()
        .map_err(|error| format!("rewind file: {error}"))?;

    let file = into_ffi_file(file);
    let mut byte = [0u8; 1];
    let rejected = vx_file_read(file, byte.as_mut_ptr(), u64::MAX);
    let valid = vx_file_read(file, byte.as_mut_ptr(), 1);
    vx_file_drop(file);

    if rejected != 0 {
        return Err(format!("oversized read returned {rejected} instead of 0"));
    }
    if valid != 1 || byte != *b"R" {
        return Err(format!(
            "a one-byte read after the rejected length returned {valid} bytes: {byte:?}"
        ));
    }
    Ok(())
}

#[test]
fn file_write_rejects_a_length_larger_than_a_rust_slice() -> Result<(), String> {
    let file = tempfile::tempfile().map_err(|error| format!("create file: {error}"))?;
    let inspection = file
        .try_clone()
        .map_err(|error| format!("clone file for inspection: {error}"))?;
    let file = into_ffi_file(file);
    let byte = *b"R";
    let rejected = vx_file_write(file, byte.as_ptr(), u64::MAX);
    let size_after_rejection = inspection
        .metadata()
        .map_err(|error| format!("inspect file after rejected write: {error}"))?
        .len();
    let valid = vx_file_write(file, byte.as_ptr(), 1);
    vx_file_drop(file);

    if rejected != 0 || size_after_rejection != 0 {
        return Err(format!(
            "oversized write returned {rejected} and left {size_after_rejection} bytes in the file"
        ));
    }
    if valid != 1 {
        return Err(format!(
            "a one-byte write after the rejected length returned {valid} bytes"
        ));
    }
    Ok(())
}
