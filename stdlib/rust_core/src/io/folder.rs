//===- folder.rs - Vx Compiler ---------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//! The Rust core behind `std::fs::Folder`, the right to read the files of one folder.
//!
//! A name in a folder is a relative path. Each part of it opens with `openat` and `O_NOFOLLOW`,
//! from the folder, so a `..` part, an absolute path or a symbolic link cannot lead out of it.

use super::{vx_file_store_error, VX_FILE_STATUS_INVALID_ARGUMENT, VX_FILE_STATUS_OK};

/// The status of a name that does not name a file inside the folder: a name with no part, an
/// absolute path, a `..` part, or a part that is a symbolic link. The other statuses are those
/// of `vx_file_try_read`.
pub const VX_FOLDER_STATUS_NOT_IN_FOLDER: i32 = 13;

/// Why a name did not open.
#[cfg_attr(not(unix), allow(dead_code))]
enum OpenError {
    NotInFolder,
    Io(std::io::Error),
}

#[cfg(unix)]
mod below {
    use super::OpenError;
    use std::ffi::{CString, OsStr};
    use std::fs::File;
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
    use std::os::unix::ffi::OsStrExt;
    use std::path::{Component, Path};

    /// The parts of `name`, or `None` when a part is the root or `..`, or no part is left.
    pub(super) fn parts(name: &[u8]) -> Option<Vec<&OsStr>> {
        let mut parts = Vec::new();
        for component in Path::new(OsStr::from_bytes(name)).components() {
            match component {
                Component::Normal(part) => parts.push(part),
                Component::CurDir => {}
                Component::RootDir | Component::ParentDir | Component::Prefix(_) => return None,
            }
        }
        if parts.is_empty() {
            None
        } else {
            Some(parts)
        }
    }

    /// True when `part` in the folder `at` is a symbolic link.
    fn is_a_link(at: RawFd, part: &CString) -> bool {
        let mut info = std::mem::MaybeUninit::<libc::stat>::uninit();
        let found = unsafe {
            libc::fstatat(
                at,
                part.as_ptr(),
                info.as_mut_ptr(),
                libc::AT_SYMLINK_NOFOLLOW,
            )
        };
        found == 0 && (unsafe { info.assume_init() }.st_mode & libc::S_IFMT) == libc::S_IFLNK
    }

    /// Opens `name` below the folder `dir` for reading, a part at a time. Each part but the last
    /// must be a folder, and the last must be a folder too when `want_folder` is true.
    pub(super) fn open(dir: &File, name: &[u8], want_folder: bool) -> Result<File, OpenError> {
        let parts = parts(name).ok_or(OpenError::NotInFolder)?;
        let mut current: Option<OwnedFd> = None;
        for (index, part) in parts.iter().enumerate() {
            let last = index + 1 == parts.len();
            let mut flags = libc::O_RDONLY | libc::O_NOFOLLOW | libc::O_CLOEXEC;
            if !last || want_folder {
                flags |= libc::O_DIRECTORY;
            }
            // A part of a C string holds no NUL byte.
            let part = CString::new(part.as_bytes()).map_err(|_| OpenError::NotInFolder)?;
            let at = current
                .as_ref()
                .map_or(dir.as_raw_fd(), |fd| fd.as_raw_fd());
            let fd = unsafe { libc::openat(at, part.as_ptr(), flags) };
            if fd < 0 {
                let error = std::io::Error::last_os_error();
                // `O_NOFOLLOW` refuses a link with ELOOP, or with ENOTDIR on Linux when
                // `O_DIRECTORY` is set too.
                if is_a_link(at, &part) {
                    return Err(OpenError::NotInFolder);
                }
                return Err(OpenError::Io(error));
            }
            current = Some(unsafe { OwnedFd::from_raw_fd(fd) });
        }
        Ok(File::from(
            current.expect("a name with no part returned above"),
        ))
    }
}

/// Opens the folder at `c_path`, for `Folder::open`. On success it gives 0 and stores the
/// folder in `*folder`. Else it stores null in `*folder`, gives the status of
/// `vx_file_try_open`, and stores the error number of the system in `*native_code`.
///
/// # Safety
/// A non-null `c_path` must point at a NUL-terminated string. `folder` and `native_code` must
/// be non-null, aligned and writable.
#[no_mangle]
pub unsafe extern "C" fn vx_folder_open(
    c_path: *const std::ffi::c_char,
    folder: *mut *mut std::ffi::c_void,
    native_code: *mut i32,
) -> i32 {
    unsafe {
        *folder = std::ptr::null_mut();
        *native_code = 0;
    }
    if c_path.is_null() {
        return VX_FILE_STATUS_INVALID_ARGUMENT;
    }
    let opened = open_a_folder(unsafe { std::ffi::CStr::from_ptr(c_path) }.to_bytes());
    match opened {
        Ok(dir) => {
            unsafe { *folder = Box::into_raw(Box::new(dir)).cast() };
            VX_FILE_STATUS_OK
        }
        Err(error) => {
            let mut has_native_code = false;
            unsafe { vx_file_store_error(&error, native_code, &mut has_native_code) }
        }
    }
}

#[cfg(unix)]
fn open_a_folder(path: &[u8]) -> std::io::Result<std::fs::File> {
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::OpenOptionsExt;
    let path = std::ffi::OsStr::from_bytes(path);
    crate::ffi::io::off_the_worker(|| {
        std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_CLOEXEC)
            .open(path)
    })
}

#[cfg(not(unix))]
fn open_a_folder(_path: &[u8]) -> std::io::Result<std::fs::File> {
    Err(std::io::Error::from(std::io::ErrorKind::Unsupported))
}

/// Opens `c_name` inside `folder` for reading, for `Folder::open_file` and, with `want_folder`
/// not 0, for `Folder::open_folder`. On success it gives 0 and stores the file or the folder in
/// `*out`. A name that does not name a file inside the folder gives
/// `VX_FOLDER_STATUS_NOT_IN_FOLDER`. Another failure gives the status of `vx_file_try_open`,
/// with the error number of the system in `*native_code`.
///
/// # Safety
/// A non-null `folder` must come from `vx_folder_open` or this function, and must still be
/// open. A non-null `c_name` must point at a NUL-terminated string. `out` and `native_code` must
/// be non-null, aligned and writable.
#[no_mangle]
pub unsafe extern "C" fn vx_folder_open_file(
    folder: *mut std::ffi::c_void,
    c_name: *const std::ffi::c_char,
    want_folder: i32,
    out: *mut *mut std::ffi::c_void,
    native_code: *mut i32,
) -> i32 {
    unsafe {
        *out = std::ptr::null_mut();
        *native_code = 0;
    }
    if folder.is_null() || c_name.is_null() {
        return VX_FILE_STATUS_INVALID_ARGUMENT;
    }
    let dir = unsafe { &*(folder as *const std::fs::File) };
    let name = unsafe { std::ffi::CStr::from_ptr(c_name) }.to_bytes();
    #[cfg(unix)]
    let opened = crate::ffi::io::off_the_worker(|| below::open(dir, name, want_folder != 0));
    #[cfg(not(unix))]
    let opened: Result<std::fs::File, OpenError> = {
        let _ = (dir, name, want_folder);
        Err(OpenError::Io(std::io::Error::from(
            std::io::ErrorKind::Unsupported,
        )))
    };
    match opened {
        Ok(file) => {
            unsafe { *out = Box::into_raw(Box::new(file)).cast() };
            VX_FILE_STATUS_OK
        }
        Err(OpenError::NotInFolder) => VX_FOLDER_STATUS_NOT_IN_FOLDER,
        Err(OpenError::Io(error)) => {
            let mut has_native_code = false;
            unsafe { vx_file_store_error(&error, native_code, &mut has_native_code) }
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::ffi::CString;
    use std::io::Read;

    /// A new empty folder under the temporary folder of the system.
    fn new_folder(test: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("vx-folder-{test}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("make the test folder");
        path
    }

    fn open_folder(path: &std::path::Path) -> *mut std::ffi::c_void {
        let c_path = CString::new(path.to_str().expect("a UTF-8 path")).expect("no NUL");
        let (mut folder, mut code) = (std::ptr::null_mut(), 0);
        let status = unsafe { vx_folder_open(c_path.as_ptr(), &mut folder, &mut code) };
        assert_eq!((status, code), (0, 0), "open the folder {path:?}");
        folder
    }

    /// The status, the error number, and the text of the file when it opened.
    fn open_in(folder: *mut std::ffi::c_void, name: &str, want_folder: bool) -> (i32, i32, String) {
        let c_name = CString::new(name).expect("no NUL");
        let (mut out, mut code) = (std::ptr::null_mut(), 99);
        let status = unsafe {
            vx_folder_open_file(
                folder,
                c_name.as_ptr(),
                want_folder as i32,
                &mut out,
                &mut code,
            )
        };
        let mut text = String::new();
        if status == 0 {
            let mut file = unsafe { Box::from_raw(out as *mut std::fs::File) };
            if !want_folder {
                file.read_to_string(&mut text).expect("read the file");
            }
        } else {
            assert!(out.is_null(), "a failed open of {name} stores null");
        }
        (status, code, text)
    }

    fn close(folder: *mut std::ffi::c_void) {
        drop(unsafe { Box::from_raw(folder as *mut std::fs::File) });
    }

    #[test]
    fn a_folder_opens_the_files_inside_it() {
        let root = new_folder("inside");
        std::fs::create_dir(root.join("sub")).unwrap();
        std::fs::write(root.join("a.txt"), "abc").unwrap();
        std::fs::write(root.join("sub/b.txt"), "xy").unwrap();
        let folder = open_folder(&root);
        assert_eq!(open_in(folder, "a.txt", false), (0, 0, "abc".to_string()));
        assert_eq!(
            open_in(folder, "sub/b.txt", false),
            (0, 0, "xy".to_string())
        );
        assert_eq!(
            open_in(folder, "./sub//b.txt", false),
            (0, 0, "xy".to_string())
        );
        assert_eq!(open_in(folder, "sub", true).0, 0);
        // A file is not a folder: ENOTDIR has no category of its own.
        assert_eq!(open_in(folder, "a.txt", true).0, 12);
        let missing = std::fs::File::open(root.join("missing.txt")).unwrap_err();
        assert_eq!(open_in(folder, "missing.txt", false).0, 1);
        assert_eq!(
            Some(open_in(folder, "missing.txt", false).1),
            missing.raw_os_error()
        );
        close(folder);
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_name_cannot_lead_out_of_the_folder() {
        let root = new_folder("outside");
        let inner = root.join("inner");
        std::fs::create_dir_all(inner.join("sub")).unwrap();
        std::fs::write(root.join("secret.txt"), "secret").unwrap();
        std::fs::write(inner.join("sub/c.txt"), "c").unwrap();
        std::os::unix::fs::symlink("../secret.txt", inner.join("link.txt")).unwrap();
        std::os::unix::fs::symlink("sub", inner.join("linked_sub")).unwrap();
        let folder = open_folder(&inner);
        let not_in_folder = (VX_FOLDER_STATUS_NOT_IN_FOLDER, 0, String::new());
        let absolute = root.join("secret.txt");
        for name in [
            "../secret.txt",
            "sub/../../secret.txt",
            absolute.to_str().unwrap(),
            "",
            ".",
            "link.txt",
            "linked_sub/c.txt",
        ] {
            assert_eq!(
                open_in(folder, name, false),
                not_in_folder,
                "the name {name:?}"
            );
        }
        assert_eq!(open_in(folder, "linked_sub", true), not_in_folder);
        close(folder);
        std::fs::remove_dir_all(&root).unwrap();
    }
}
