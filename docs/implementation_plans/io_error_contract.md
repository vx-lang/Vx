<!--
Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# `std::io::Error` contract

This is a proposed contract, not a currently available Vx type. The earlier `Error`, `ErrorKind`, and `Operation` definitions were removed from [`stdlib/std/io.vx`](../../stdlib/std/io.vx) because no public I/O operation produced them. Reintroduce the value when Vx can lower `Result<u64, Error>` (Vx#570) and file APIs can return it. Like Rust's [`std::io::Error`](https://doc.rust-lang.org/std/io/struct.Error.html), it gives callers a portable category through `kind()` and an optional platform code through `raw_os_error()`. Vx also records the failed `operation()`, because the same error kind can occur while opening, reading, writing, seeking, syncing, or closing a file. Future fallible `std::fs` APIs will use this type; they will not need a separate filesystem error type.

The proposed `ErrorKind` cases are `NotFound`, `PermissionDenied`, `AlreadyExists`, `InvalidInput`, `Interrupted`, `WouldBlock`, `UnexpectedEof`, `WriteZero`, `OutOfMemory`, `LimitExceeded`, `Unsupported`, and `Other`. `Operation` names `Open`, `Read`, `Write`, `Seek`, `Sync`, `Close`, and `Unspecified`. A backend maps an OS error with no matching category to `Other` and an operation not listed here to `Unspecified`. Neither enum's ordinal is an OS error code.

When introduced, these variant sets should remain fixed until Vx can enforce a fallback arm when another module matches an enum. Adding a variant could otherwise break an exhaustive match. Callers should include a fallback arm. New native codes would remain observable through `raw_os_error()` even when their category is `Other`.

The proposed `Error::new(kind, operation)` creates an error raised by Vx, with no native code. `Error::from_native_error(kind, operation, code)` stores the exact platform code and category supplied by a backend, even when `kind` is `Other`. The backend must read the OS error state **immediately** after the failed call, before logging, allocation, cleanup, or any other call can change it. It then classifies the code for that host and creates the error. The constructor does not classify a code: errno values and meanings are platform specific. `raw_os_error()` returns `Option<i32>`: `None` for a Vx-generated error, `Some(code)` for a native error. Callers can inspect the kind, operation, and native code without allocating. Native codes are for diagnostics or host-specific integration, not portable control flow.

The proposed `message()` returns a static, human-readable description of the category. Error creation and inspection do not allocate or format text; callers request text only if they need it. The message is deliberately independent of host-specific error strings, which can depend on the OS and locale. Diagnostic output can add `operation()` and `raw_os_error()` when useful. A later formatting API may combine these on demand without changing the stored error value. The static pointer returned by the proposed `message()` must not be freed.

For example, these illustrative values would distinguish a missing file, an unknown OS failure, and a check performed by Vx. This code does not compile until the type is restored:

<!-- vx-doctest: skip; proposed error type is not implemented -->

```vx
let missing = Error::from_native_error(ErrorKind::NotFound(), Operation::Open(), 2);
let unknown = Error::from_native_error(ErrorKind::Other(), Operation::Read(), 12345);
let invalid = Error::new(ErrorKind::InvalidInput(), Operation::Seek());
```

Here `2` is only an example of a host's missing-file code. A real backend must use the code it actually captured. `missing.raw_os_error()` is `Some(2)`, `unknown.raw_os_error()` is `Some(12345)`, and `invalid.raw_os_error()` is `None`. Their categories remain `NotFound`, `Other`, and `InvalidInput` respectively. Add a public Vx test for these cases when the type is restored.

## Current fallible bridge status encoding

`vx_file_try_read` and `vx_file_try_write` return an `i32` status and write to `count : *mut u64`, `native_code : *mut i32`, and `has_native_code : *mut bool`. These integers are a private bridge ABI, not enum ordinals or native OS codes. Both calls use the same mapping:

The Rust entry points are `unsafe extern "C"`. A non-null handle must remain a live, exclusively accessed bridge-owned `File` for the call. A non-null buffer used with a valid positive length must contain that many initialized bytes, writable for read or readable for write. Every non-null output pointer must be aligned and writable, including on an invalid-argument path; the file, buffer, and outputs must not overlap. Null handle and buffer pointers are accepted as invalid arguments, but null checks cannot validate other pointer properties. The future Vx wrapper must keep these requirements inside its unsafe boundary.

| Status | Meaning | Rust source |
| ---: | --- | --- |
| `0` | Success | Successful read/write, including EOF and an empty request |
| `-1` | Invalid bridge arguments | Null handle or output pointer; null buffer with nonzero length; length above `isize::MAX` |
| `1` | `NotFound` | `std::io::ErrorKind::NotFound` |
| `2` | `PermissionDenied` | `PermissionDenied` |
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

On status `0`, `count` is the transferred byte count, from zero through the requested length; `has_native_code` is false and `native_code` is zero. On statuses `1` through `12`, `count` is zero. The bridge captures `raw_os_error()` from the failed call before classification: when present, `has_native_code` is true and `native_code` is the exact code; otherwise `has_native_code` is false and `native_code` is zero. The code may be zero even when `has_native_code` is true, so callers must check the flag.

On `-1`, `count` and `native_code` are zero and `has_native_code` is false. Every non-null output is initialized this way even if another output pointer is null; a null output itself has no value. Callers must supply valid, writable, non-overlapping output storage for every non-null pointer. No OS error is implied by `-1`. When the Vx error type is restored, a wrapper should translate `-1` to `Error::new(ErrorKind::InvalidInput(), Operation::Read())` or `Operation::Write()` with no native code. Status `4`, by contrast, is an `InvalidInput` result from an actual read or write and retains any native code it carried. A Vx wrapper should reject any status outside `-1..=12` as a bridge contract violation rather than silently treating it as `Other`.

The old raw file calls still return zero for both EOF and I/O failure. New bridge calls distinguish success from failure, capture the native code immediately from Rust's `std::io::Error`, and map its kind to a portable category. A successful short transfer returns its count; an empty request returns zero; failure returns zero bytes and a separate error status. `File` does not expose those calls yet because Vx cannot currently generate code for `Result<u64, Error>` with its differently sized variants. A file-specific fallible read can use the proposed error type without a `Read` trait; a reusable trait can be added when another byte source needs the same operations. The public read signature needs a bounded byte buffer before it can be called safe. The remaining compiler and file API work is described in [File I/O usability](file_io_usability.md).
