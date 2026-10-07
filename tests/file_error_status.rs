// Fallible file calls must report counts and errors without losing native codes.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

include!("../stdlib/rust_core/src/ffi/macros.rs");
instantiate_file_ffi!();

// vx_file_open stores a boxed Rust File behind its opaque pointer.
fn into_ffi_file(file: std::fs::File) -> *mut std::ffi::c_void {
    Box::into_raw(Box::new(file)).cast()
}

fn expect_equal<T: std::fmt::Debug + PartialEq>(
    actual: T,
    expected: T,
    context: &str,
) -> Result<(), String> {
    if actual != expected {
        return Err(format!("{context}: expected {expected:?}, got {actual:?}"));
    }
    Ok(())
}

#[test]
fn file_error_classifier_maps_every_supported_kind_and_unknown_codes() -> Result<(), String> {
    use std::io::ErrorKind;

    let cases = [
        (ErrorKind::NotFound, 1),
        (ErrorKind::PermissionDenied, 2),
        (ErrorKind::AlreadyExists, 3),
        (ErrorKind::InvalidInput, 4),
        (ErrorKind::Interrupted, 5),
        (ErrorKind::WouldBlock, 6),
        (ErrorKind::UnexpectedEof, 7),
        (ErrorKind::WriteZero, 8),
        (ErrorKind::OutOfMemory, 9),
        (ErrorKind::StorageFull, 10),
        (ErrorKind::QuotaExceeded, 10),
        (ErrorKind::FileTooLarge, 10),
        (ErrorKind::Unsupported, 11),
    ];
    for (kind, expected) in cases {
        expect_equal(vx_file_error_kind(kind), expected, &format!("{kind:?}"))?;
    }

    let unknown = std::io::Error::from_raw_os_error(12345);
    expect_equal(unknown.raw_os_error(), Some(12345), "unknown raw OS code")?;
    expect_equal(vx_file_error_kind(unknown.kind()), 12, "unknown category")?;
    Ok(())
}

#[test]
fn file_error_without_a_native_code_keeps_the_code_absent() -> Result<(), String> {
    let error = std::io::Error::new(std::io::ErrorKind::InvalidInput, "synthetic failure");
    expect_equal(error.raw_os_error(), None, "synthetic raw OS code")?;
    let (mut code, mut has_code) = (99, true);
    // This is the shared error path used by both fallible file calls.
    let status = unsafe { vx_file_store_error(&error, &mut code, &mut has_code) };
    expect_equal(
        (status, code, has_code),
        (4, 0, false),
        "synthetic error fields",
    )?;
    Ok(())
}

#[test]
fn fallible_file_calls_clear_every_available_output_on_invalid_arguments() -> Result<(), String> {
    let file = tempfile::tempfile().map_err(|error| format!("create file: {error}"))?;
    let file = into_ffi_file(file);
    let mut byte = *b"X";
    for is_write in [false, true] {
        let mut call = |count: *mut u64, native_code: *mut i32, has_native_code: *mut bool| {
            // The handle and buffer are valid; non-null outputs are distinct locals.
            if is_write {
                unsafe {
                    vx_file_try_write(file, byte.as_ptr(), 1, count, native_code, has_native_code)
                }
            } else {
                unsafe {
                    vx_file_try_read(
                        file,
                        byte.as_mut_ptr(),
                        1,
                        count,
                        native_code,
                        has_native_code,
                    )
                }
            }
        };

        {
            let (mut code, mut has_code) = (99, true);
            let status = call(std::ptr::null_mut(), &mut code, &mut has_code);
            expect_equal(
                (status, code, has_code),
                (-1, 0, false),
                "null count output",
            )?;
        }
        {
            let (mut count, mut has_code) = (99, true);
            let status = call(&mut count, std::ptr::null_mut(), &mut has_code);
            expect_equal(
                (status, count, has_code),
                (-1, 0, false),
                "null code output",
            )?;
        }
        {
            let (mut count, mut code) = (99, 99);
            let status = call(&mut count, &mut code, std::ptr::null_mut());
            expect_equal((status, count, code), (-1, 0, 0), "null code flag output")?;
        }
    }
    vx_file_drop(file);
    Ok(())
}

#[test]
fn fallible_file_calls_reject_null_handles_and_nonempty_null_buffers() -> Result<(), String> {
    let file = tempfile::tempfile().map_err(|error| format!("create file: {error}"))?;
    let inspection = file
        .try_clone()
        .map_err(|error| format!("clone file: {error}"))?;
    let file = into_ffi_file(file);
    let mut byte = *b"X";
    for is_write in [false, true] {
        let (mut count, mut code, mut has_code) = (99, 99, true);
        let null_handle_status = if is_write {
            unsafe {
                vx_file_try_write(
                    std::ptr::null_mut(),
                    byte.as_ptr(),
                    1,
                    &mut count,
                    &mut code,
                    &mut has_code,
                )
            }
        } else {
            unsafe {
                vx_file_try_read(
                    std::ptr::null_mut(),
                    byte.as_mut_ptr(),
                    1,
                    &mut count,
                    &mut code,
                    &mut has_code,
                )
            }
        };
        expect_equal(
            (null_handle_status, count, code, has_code),
            (-1, 0, 0, false),
            "null file handle",
        )?;

        (count, code, has_code) = (99, 99, true);
        let null_buffer_status = if is_write {
            unsafe {
                vx_file_try_write(
                    file,
                    std::ptr::null(),
                    1,
                    &mut count,
                    &mut code,
                    &mut has_code,
                )
            }
        } else {
            unsafe {
                vx_file_try_read(
                    file,
                    std::ptr::null_mut(),
                    1,
                    &mut count,
                    &mut code,
                    &mut has_code,
                )
            }
        };
        expect_equal(
            (null_buffer_status, count, code, has_code),
            (-1, 0, 0, false),
            "null buffer",
        )?;
    }
    vx_file_drop(file);
    expect_equal(
        inspection
            .metadata()
            .map_err(|error| format!("inspect file: {error}"))?
            .len(),
        0,
        "file length after rejected calls",
    )?;
    Ok(())
}

#[test]
fn fallible_file_calls_reject_oversized_lengths_before_io() -> Result<(), String> {
    let mut file = tempfile::tempfile().map_err(|error| format!("create file: {error}"))?;
    file.write_all(b"R")
        .map_err(|error| format!("seed file: {error}"))?;
    file.rewind()
        .map_err(|error| format!("rewind file: {error}"))?;
    let inspection = file
        .try_clone()
        .map_err(|error| format!("clone file: {error}"))?;
    let file = into_ffi_file(file);
    let mut byte = [0u8; 1];
    let oversized = isize::MAX as u64 + 1;
    let (mut count, mut code, mut has_code) = (99, 99, true);
    let read_status = unsafe {
        vx_file_try_read(
            file,
            byte.as_mut_ptr(),
            oversized,
            &mut count,
            &mut code,
            &mut has_code,
        )
    };
    expect_equal(
        (read_status, count, code, has_code),
        (-1, 0, 0, false),
        "oversized read",
    )?;

    (count, code, has_code) = (99, 99, true);
    let normal_read = unsafe {
        vx_file_try_read(
            file,
            byte.as_mut_ptr(),
            1,
            &mut count,
            &mut code,
            &mut has_code,
        )
    };
    expect_equal(
        (normal_read, count, code, has_code, byte),
        (0, 1, 0, false, *b"R"),
        "read after oversized request",
    )?;

    (count, code, has_code) = (99, 99, true);
    let write_status = unsafe {
        vx_file_try_write(
            file,
            b"X".as_ptr(),
            oversized,
            &mut count,
            &mut code,
            &mut has_code,
        )
    };
    vx_file_drop(file);
    expect_equal(
        (write_status, count, code, has_code),
        (-1, 0, 0, false),
        "oversized write",
    )?;
    expect_equal(
        inspection
            .metadata()
            .map_err(|error| format!("inspect file: {error}"))?
            .len(),
        1,
        "file length after oversized write",
    )?;
    Ok(())
}

#[cfg(unix)]
#[test]
fn fallible_file_read_classifies_a_real_would_block_failure() -> Result<(), String> {
    use std::os::fd::{FromRawFd, IntoRawFd};
    use std::os::unix::net::UnixStream;

    let (reader, _writer) = UnixStream::pair().map_err(|error| format!("create pair: {error}"))?;
    reader
        .set_nonblocking(true)
        .map_err(|error| format!("set nonblocking: {error}"))?;
    let mut control = reader
        .try_clone()
        .map_err(|error| format!("clone reader: {error}"))?;
    let mut byte = [0u8; 1];
    let expected = control
        .read(&mut byte)
        .expect_err("an empty nonblocking stream should return WouldBlock");
    if expected.kind() != std::io::ErrorKind::WouldBlock || expected.raw_os_error().is_none() {
        return Err(format!(
            "expected a native WouldBlock error, got {expected}"
        ));
    }

    // Transfer ownership of the descriptor to File, then to the same opaque pointer as vx_file_open.
    let file = unsafe { std::fs::File::from_raw_fd(reader.into_raw_fd()) };
    let file = into_ffi_file(file);
    let mut count = 99;
    let mut code = 99;
    let mut has_code = false;
    let status = unsafe {
        vx_file_try_read(
            file,
            byte.as_mut_ptr(),
            byte.len() as u64,
            &mut count,
            &mut code,
            &mut has_code,
        )
    };
    vx_file_drop(file);
    if status != 6 || count != 0 || !has_code || Some(code) != expected.raw_os_error() {
        return Err(format!(
            "WouldBlock returned status {status}, count {count}, native code {has_code:?}/{code}; expected {:?}",
            expected.raw_os_error()
        ));
    }
    Ok(())
}

#[test]
fn fallible_file_transfer_reports_successful_byte_counts() -> Result<(), String> {
    let file = tempfile::tempfile().map_err(|error| format!("create file: {error}"))?;
    let file = into_ffi_file(file);
    let mut count = 99;
    let mut code = 99;
    let mut has_code = true;
    let written =
        unsafe { vx_file_try_write(file, b"R".as_ptr(), 1, &mut count, &mut code, &mut has_code) };
    if written != 0 || count != 1 || code != 0 || has_code {
        vx_file_drop(file);
        return Err(format!(
            "write returned status {written}, count {count}, native code {has_code}"
        ));
    }
    let position = vx_file_seek(file, 0, 0);
    let mut byte = [0u8; 1];
    count = 99;
    let read = unsafe {
        vx_file_try_read(
            file,
            byte.as_mut_ptr(),
            1,
            &mut count,
            &mut code,
            &mut has_code,
        )
    };
    vx_file_drop(file);
    if position != 0 || read != 0 || count != 1 || byte != *b"R" || code != 0 || has_code {
        return Err(format!("read returned position {position}, status {read}, count {count}, bytes {byte:?}, native code {has_code}"));
    }
    Ok(())
}

#[test]
fn fallible_file_read_reports_a_short_success_before_eof() -> Result<(), String> {
    let mut file = tempfile::tempfile().map_err(|error| format!("create file: {error}"))?;
    file.write_all(b"AB")
        .map_err(|error| format!("seed file: {error}"))?;
    file.rewind()
        .map_err(|error| format!("rewind file: {error}"))?;
    let file = into_ffi_file(file);
    let mut buffer = [0u8; 4];
    let (mut count, mut code, mut has_code) = (99, 99, true);
    let status = unsafe {
        vx_file_try_read(
            file,
            buffer.as_mut_ptr(),
            buffer.len() as u64,
            &mut count,
            &mut code,
            &mut has_code,
        )
    };
    if (status, count, code, has_code, &buffer[..2]) != (0, 2, 0, false, &b"AB"[..]) {
        vx_file_drop(file);
        return Err(format!(
            "short read returned {status}/{count}/{code}/{has_code}: {buffer:?}"
        ));
    }

    count = 99;
    let eof = unsafe {
        vx_file_try_read(
            file,
            buffer.as_mut_ptr(),
            buffer.len() as u64,
            &mut count,
            &mut code,
            &mut has_code,
        )
    };
    vx_file_drop(file);
    if (eof, count, code, has_code) != (0, 0, 0, false) {
        return Err(format!(
            "EOF after short read returned {eof}/{count}/{code}/{has_code}"
        ));
    }
    Ok(())
}

#[cfg(unix)]
#[test]
fn fallible_file_write_reports_a_short_success() -> Result<(), String> {
    use std::os::fd::{FromRawFd, IntoRawFd};
    use std::os::unix::net::UnixStream;

    let (mut receiver, sender) =
        UnixStream::pair().map_err(|error| format!("create pair: {error}"))?;
    sender
        .set_nonblocking(true)
        .map_err(|error| format!("set nonblocking: {error}"))?;
    let file = unsafe { std::fs::File::from_raw_fd(sender.into_raw_fd()) };
    let file = into_ffi_file(file);
    // A single nonblocking write cannot fill an 8 MiB request into an empty Unix socket.
    let buffer: Vec<u8> = (0..8 * 1024 * 1024)
        .map(|index| (index % 251) as u8)
        .collect();
    let (mut count, mut code, mut has_code) = (99, 99, true);
    let status = unsafe {
        vx_file_try_write(
            file,
            buffer.as_ptr(),
            buffer.len() as u64,
            &mut count,
            &mut code,
            &mut has_code,
        )
    };
    vx_file_drop(file);
    if status != 0 || count == 0 || count >= buffer.len() as u64 || code != 0 || has_code {
        return Err(format!(
            "short write returned {status}/{count}/{code}/{has_code} for {} bytes",
            buffer.len()
        ));
    }
    let mut received = Vec::new();
    receiver
        .read_to_end(&mut received)
        .map_err(|error| format!("drain socket: {error}"))?;
    if received.len() as u64 != count || received != buffer[..count as usize] {
        return Err(format!(
            "write reported {count} bytes, but the receiver got {} matching bytes: {}",
            received.len(),
            received == buffer[..received.len().min(buffer.len())]
        ));
    }
    Ok(())
}

#[cfg(unix)]
#[test]
fn fallible_file_read_separates_eof_from_an_os_error() -> Result<(), String> {
    let file = tempfile::tempfile().map_err(|error| format!("create empty file: {error}"))?;
    let file = into_ffi_file(file);
    let mut byte = [0u8; 1];
    let mut count = 99;
    let mut code = 99;
    let mut has_code = true;
    let eof = unsafe {
        vx_file_try_read(
            file,
            byte.as_mut_ptr(),
            1,
            &mut count,
            &mut code,
            &mut has_code,
        )
    };
    if eof != 0 || count != 0 || code != 0 || has_code {
        vx_file_drop(file);
        return Err(format!(
            "EOF returned status {eof}, count {count}, native code {has_code}"
        ));
    }
    count = 99;
    let empty = unsafe {
        vx_file_try_read(
            file,
            std::ptr::null_mut(),
            0,
            &mut count,
            &mut code,
            &mut has_code,
        )
    };
    vx_file_drop(file);
    if empty != 0 || count != 0 || code != 0 || has_code {
        return Err(format!(
            "empty read returned status {empty}, count {count}, native code {has_code}"
        ));
    }

    let dir = tempfile::tempdir().map_err(|error| format!("create test directory: {error}"))?;
    let path = dir.path().join("write_only");
    let mut control = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&path)
        .map_err(|error| format!("open control file: {error}"))?;
    let expected = control
        .read(&mut byte)
        .expect_err("reading a write-only file must fail");
    let file = std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .map_err(|error| format!("open write-only file: {error}"))?;
    let file = into_ffi_file(file);
    count = 99;
    let failure = unsafe {
        vx_file_try_read(
            file,
            byte.as_mut_ptr(),
            1,
            &mut count,
            &mut code,
            &mut has_code,
        )
    };
    vx_file_drop(file);
    // EBADF has no dedicated portable category in the Vx error contract.
    if failure != 12 || count != 0 || !has_code || Some(code) != expected.raw_os_error() {
        return Err(format!("read failure returned status {failure}, count {count}, native code {code}; expected kind {:?} and code {:?}", expected.kind(), expected.raw_os_error()));
    }
    Ok(())
}

#[cfg(unix)]
#[test]
fn fallible_file_write_preserves_the_os_error() -> Result<(), String> {
    let dir = tempfile::tempdir().map_err(|error| format!("create test directory: {error}"))?;
    let path = dir.path().join("read_only");
    std::fs::write(&path, b"R").map_err(|error| format!("create test file: {error}"))?;
    let mut control =
        std::fs::File::open(&path).map_err(|error| format!("open control file: {error}"))?;
    let expected = control
        .write(b"W")
        .expect_err("writing a read-only file must fail");
    let file =
        std::fs::File::open(&path).map_err(|error| format!("open read-only file: {error}"))?;
    let file = into_ffi_file(file);
    let mut count = 99;
    let mut code = 99;
    let mut has_code = true;
    let failure =
        unsafe { vx_file_try_write(file, b"W".as_ptr(), 1, &mut count, &mut code, &mut has_code) };
    if failure != 12 || count != 0 || !has_code || Some(code) != expected.raw_os_error() {
        vx_file_drop(file);
        return Err(format!("write failure returned status {failure}, count {count}, native code {code}; expected kind {:?} and code {:?}", expected.kind(), expected.raw_os_error()));
    }
    count = 99;
    let empty = unsafe {
        vx_file_try_write(
            file,
            std::ptr::null(),
            0,
            &mut count,
            &mut code,
            &mut has_code,
        )
    };
    vx_file_drop(file);
    if empty != 0 || count != 0 || code != 0 || has_code {
        return Err(format!(
            "empty write returned status {empty}, count {count}, native code {has_code}"
        ));
    }
    Ok(())
}
