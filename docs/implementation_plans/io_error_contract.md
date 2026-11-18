<!--
Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# `std::io::Error` contract

`std::io` provides `Error`, `ErrorKind`, and `Operation`. `File::open_raw`, `read_raw`, and `write_raw` return these errors through `Result`. Like Rust's [`std::io::Error`](https://doc.rust-lang.org/std/io/struct.Error.html), it gives callers a portable category through `kind()` and an optional platform code through `raw_os_error()`. Vx also records the failed `operation()`, because the same error kind can occur while opening, reading, writing, seeking, syncing, or closing a file. Future fallible filesystem operations will use the same error type.

The `ErrorKind` cases are `NotFound`, `PermissionDenied`, `AlreadyExists`, `InvalidInput`, `Interrupted`, `WouldBlock`, `UnexpectedEof`, `WriteZero`, `OutOfMemory`, `LimitExceeded`, `Unsupported`, and `Other`. `Operation` names `Open`, `Read`, `Write`, `Seek`, `Sync`, `Close`, and `Unspecified`. A backend maps an OS error with no matching category to `Other` and an operation not listed here to `Unspecified`. Neither enum's ordinal is an OS error code.

These variant sets remain fixed until Vx can enforce a fallback arm when another module matches an enum. Adding a variant could otherwise break an exhaustive match. Callers should include a fallback arm. New native codes would remain observable through `raw_os_error()` even when their category is `Other`.

`Error::from_kind(kind)` creates an error without a native code. Its operation is `Unspecified`.
`Error::from_raw_os_error(code)` preserves the exact code and classifies it for the current host.
The native-code constructor does not accept a category. Backend helpers remain visible because Vx lacks private function visibility. `with_operation(operation)` attaches context without changing the category or native code.
Backends capture errors immediately after failure, before logging, cleanup, or another host call.
`raw_os_error()` returns `Option<i32>`. Native codes support diagnostics, while portable control flow uses `kind()`.

`message_c_str()` returns a static category description as a C pointer. The caller must not free it.
This is a raw text escape hatch, pending safe text and formatting support.
`Display`, `Debug`, custom messages, and underlying causes remain gated on the formatting and error-trait prerequisites.
`Error::new` is reserved for a future custom-error constructor, rather than assigning it an incompatible meaning now.
The small built-in error remains allocation-free and `Copy`.

```vx
let unknown = Error::from_raw_os_error(12345).with_operation(Operation::Read());
let invalid = Error::from_kind(ErrorKind::InvalidInput()).with_operation(Operation::Seek());
```

The public test `file_structured_errors.vx` checks native-code preservation, unknown codes, and Vx-generated errors.


## Current fallible bridge status encoding

`vx_file_try_open`, `vx_file_try_read`, and `vx_file_try_write` return an `i32` status and write to `count : *mut u64`, `native_code : *mut i32`, and `has_native_code : *mut bool`. These integers are a private bridge ABI, not enum ordinals or native OS codes. All three calls use the same mapping. Open writes a handle through `out : *mut *mut i8` instead of a byte count.
Success transfers ownership to the caller, which must call `vx_file_drop` exactly once.
Failure writes a null handle. Invalid legacy modes, unknown option bits, and null paths return `-1`.
`vx_file_open_options` accepts bits 1=read, 2=write, 4=append, 8=create, 16=truncate, and 32=create_new.
The host validates option combinations and reports `InvalidInput` for invalid combinations.
`vx_file_try_seek` uses the same status and native-code outputs, with a `position : *mut u64` output.
Its internal whence values are 0=start, 1=current, and 2=end.
Start offsets preserve all 64 bits. Relative offsets use signed `i64` values.
These integers are backend details. Public callers use `SeekFrom`.

| Status | Meaning | Rust source |
| ---: | --- | --- |
| `0` | Success | Successful read/write, including EOF and an empty request |
| `-1` | Invalid bridge arguments | Null handle or output pointer; null buffer with nonzero length; length above `isize::MAX` |
| `1` | `NotFound` | `std::io::ErrorKind::NotFound` |
| `2` | `PermissionDenied` | `PermissionDenied` or `ReadOnlyFilesystem` |
| `3` | `AlreadyExists` | `AlreadyExists` |
| `4` | `InvalidInput` | `InvalidInput` returned by the host operation |
| `5` | `Interrupted` | `Interrupted` |
| `6` | `WouldBlock` | `WouldBlock` |
| `7` | `UnexpectedEof` | `UnexpectedEof` |
| `8` | `WriteZero` | `WriteZero` |
| `9` | `OutOfMemory` | `OutOfMemory` |
| `10` | `LimitExceeded` | Rust's `StorageFull`, `QuotaExceeded`, or `FileTooLarge` |
| `11` | `Unsupported` | `Unsupported` |
| `12` | `Other` | Every remaining or unknown Rust error kind |

The Rust entry points are `unsafe extern "C"`. A non-null handle must remain a live, exclusively accessed bridge-owned `File` for the call. A non-null read buffer used with a valid positive length must hold that many writable bytes; they need not be initialized. A write buffer must hold that many initialized, readable bytes. Every non-null output pointer must be aligned and writable, including on an invalid-argument path; the file, buffer, and outputs must not overlap. Null handle and buffer pointers are accepted as invalid arguments, but null checks cannot validate other pointer properties. The Vx wrappers keep these requirements inside their unsafe boundaries.

A read-only filesystem is a permission restriction and maps to `PermissionDenied`. A write through a read-only file handle fails with a bad-handle error on Unix and maps to `Other`; its native code remains available to callers.

On status `0`, `count` is the transferred byte count, from zero through the requested length; `has_native_code` is false and `native_code` is zero. For reads, only the first `count` buffer bytes contain data read from the file. Bytes in `buffer[count..len]` may also change, including at EOF; callers must not rely on their previous contents being preserved. On statuses `1` through `12`, `count` is zero. The bridge captures `raw_os_error()` from the failed call before classification: when present, `has_native_code` is true and `native_code` is the exact code; otherwise `has_native_code` is false and `native_code` is zero. The code may be zero even when `has_native_code` is true, so callers must check the flag.

On `-1`, `count` and `native_code` are zero and `has_native_code` is false. Every non-null output is initialized this way even if another output pointer is null; a null output itself has no value. Callers must supply valid, writable, non-overlapping output storage for every non-null pointer. No OS error is implied by `-1`. The Vx wrappers translate `-1` to `Error::from_kind(ErrorKind::InvalidInput()).with_operation(operation)` with no native code. Status `4`, by contrast, is an `InvalidInput` result from an actual read or write and retains any native code it carried. The Vx wrappers reject any status outside `-1..=12` as a bridge contract violation rather than silently treating it as `Other`.

The old raw file calls still return zero for both EOF and I/O failure. New bridge calls distinguish success from failure, capture the native code immediately from Rust's `std::io::Error`, and map its kind to a portable category. A successful short transfer returns its count; an empty request returns zero; failure returns zero bytes and a separate error status. `File::read_raw` and `write_raw` expose those calls as `Result<usize, Error>`. A file-specific fallible read uses this error type without a `Read` trait; a reusable trait can be added when another byte source needs the same operations. The public read signature needs a bounded byte buffer before it can be called safe. The remaining compiler and file API work is described in [File I/O usability](file_io_usability.md).
