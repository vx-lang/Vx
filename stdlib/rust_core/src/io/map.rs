//===- map.rs - Vx Compiler ------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//! The Rust core behind `File::map`, a read-only view of a whole file in memory.

use super::{vx_file_store_error, VX_FILE_STATUS_INVALID_ARGUMENT, VX_FILE_STATUS_OK};

/// Maps the whole of the open file `file` into memory, read-only, for `File::map`. On success it
/// gives 0, and stores the address in `*base` and the size in `*len`. An empty file gives 0, a
/// null `*base` and a `*len` of 0, because the system cannot map an empty range. Else it gives
/// the status of `vx_file_try_open`, with the error number of the system in `*native_code`.
///
/// # Safety
/// A non-null `file` must come from `vx_file_try_open` and must still be open. `base`, `len` and
/// `native_code` must be non-null, aligned and writable.
#[no_mangle]
pub unsafe extern "C" fn vx_file_try_map(
    file: *mut std::ffi::c_void,
    base: *mut *mut u8,
    len: *mut i64,
    native_code: *mut i32,
) -> i32 {
    unsafe {
        *base = std::ptr::null_mut();
        *len = 0;
        *native_code = 0;
    }
    if file.is_null() {
        return VX_FILE_STATUS_INVALID_ARGUMENT;
    }
    let file = unsafe { &*(file as *const std::fs::File) };
    match crate::ffi::io::off_the_worker(|| map_a_file(file)) {
        Ok((address, size)) => {
            unsafe {
                *base = address as *mut u8;
                *len = size;
            }
            VX_FILE_STATUS_OK
        }
        Err(error) => {
            let mut has_native_code = false;
            unsafe { vx_file_store_error(&error, native_code, &mut has_native_code) }
        }
    }
}

/// The address and the size of a read-only map of `file`. The address is a number, so that the
/// result can cross threads.
#[cfg(unix)]
fn map_a_file(file: &std::fs::File) -> std::io::Result<(usize, i64)> {
    use std::os::fd::AsRawFd;
    // A file open only for writing cannot be read, even when it is empty.
    let flags = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GETFL) };
    if flags < 0 {
        return Err(std::io::Error::last_os_error());
    }
    if flags & libc::O_ACCMODE == libc::O_WRONLY {
        return Err(std::io::Error::from_raw_os_error(libc::EACCES));
    }
    let size = file.metadata()?.len();
    if size == 0 {
        return Ok((0, 0));
    }
    let (Ok(length), Ok(size)) = (usize::try_from(size), i64::try_from(size)) else {
        return Err(std::io::Error::from(std::io::ErrorKind::FileTooLarge));
    };
    let address = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            length,
            libc::PROT_READ,
            libc::MAP_SHARED,
            file.as_raw_fd(),
            0,
        )
    };
    if address == libc::MAP_FAILED {
        return Err(std::io::Error::last_os_error());
    }
    Ok((address as usize, size))
}

#[cfg(not(unix))]
fn map_a_file(_file: &std::fs::File) -> std::io::Result<(usize, i64)> {
    Err(std::io::Error::from(std::io::ErrorKind::Unsupported))
}

/// Unmaps what `vx_file_try_map` mapped. A null `base` or a `len` of 0 does nothing.
///
/// # Safety
/// A non-null `base` and its `len` must come from one `vx_file_try_map`, and must not be
/// unmapped yet. Nothing may read the memory after this call.
#[no_mangle]
pub unsafe extern "C" fn vx_file_unmap(base: *mut u8, len: i64) {
    if base.is_null() || len <= 0 {
        return;
    }
    #[cfg(unix)]
    unsafe {
        libc::munmap(base.cast(), len as usize);
    }
}
