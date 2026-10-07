<!--
Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# File I/O usability in the Vx standard library

**Status:** partial implementation. The proposed `std::io::Error` is deferred until file operations can return it. The existing `std::fs::File` can open, read, write, seek, and close a host file through Rust FFI. This document records what a Vx caller must do to read a file today and the work needed to make ordinary file reading safe and convenient.

File I/O belongs in host `std`, not device-portable `core`. The purity rule in [the core library plan](core_library.md) forbids `extern` in `core`; host `std::fs` may call the operating system. The intended end state is for Vx to own the file API and its implementation above those operating-system calls, without the Rust file bridge. The [roadmap](../../ROADMAP.md) marks native file I/O complete, but that checkmark describes the basic host path, not an end-user file API.

## Current implementation status

- `std::io` currently exposes raw standard-stream calls. The [error contract](io_error_contract.md) records the proposed portable category, operation, optional native code, and deferred message; those public types are not implemented.
- There is no public Vx I/O error type or public error test yet. Rust bridge tests cover every supported error-kind mapping, unknown-code fallback, a real nonblocking `WouldBlock` read, native-code preservation, and a synthetic error with no native code through the shared error conversion path. Add a public Vx error test when the file API returns the proposed type.
- `File::open` accepts `OpenMode::Read`, `OpenMode::Write`, or `OpenMode::ReadWriteCreate`. The raw bridge reports an unknown mode integer and aborts, because a null return would look like an ordinary open failure. These choices keep the existing three opening behaviors; combinations such as append and exclusive create are still unspecified.
- The Rust bridge has separate fallible byte-transfer calls. They return a byte count on success and capture the native code and portable category on failure. Rust tests cover short reads and writes, EOF, empty requests, failed reads and writes, invalid pointers and lengths, and preservation of the native code. A Vx backend test calls both bridge functions directly to check their status and output types across success and failure. `std::fs::File` does not expose these calls yet: Vx currently fails to generate valid code for `Result<u64, std::io::Error>`, whose variants carry differently shaped values. Keep that compiler work separate from the file API.
- `std::fs::File` still has raw-pointer methods and returns a `File` or a byte count, rather than a fallible result. `File` now implements `Drop`, so its handle is released automatically. The Rust file bridge still handles host calls and can turn a read failure into a zero byte count.

The next step is to make Vx's `Result<u64, std::io::Error>` code generation work, then connect the fallible bridge calls to `File` methods. The raw-pointer and ownership work below is still needed before file reading can be safe for ordinary callers. Replacing the Rust bridge can follow as a separate backend change while keeping the public error contract.

## Reading a file today

This is the shape of a program that reads four bytes from a Parquet file using `std::fs::File`. The path and buffer size are examples; this code reads bytes, not Parquet rows.

```vx
import core::mem;
import std::fs;
import std::io;

extern {
  fn malloc(size : i64) -> *mut u8;
  fn free(ptr : *mut u8) -> void;
}

fn main() -> i32 {
  unsafe {
    let mut file = File::open("/path/to/example.parquet", OpenMode::Read());
    let buffer = malloc(4);

    let count = file.read(buffer, 4);
    print!("Bytes read: ");
    print!(count);
    print!("\nHeader: ");
    let _printed = stdout_write(buffer, count);
    print!("\n");

    drop(file);
    free(buffer);
  }
  return 0;
}
```

On a valid Parquet file, the first four bytes are `PAR1`. A four-byte read is useful for demonstrating the API; it does not establish that the rest of the file is valid Parquet.

The example uses one broad `unsafe` block for brevity. `File::open` takes a raw pointer to a NUL-terminated path, `File::read` takes a raw pointer and a separately supplied byte count, and a raw pointer retained after `File` drops can dangle. `malloc` and `free` are declared locally only to provide a four-byte buffer: `std::fs` does not allocate one for the caller. The repository also exposes C allocation functions in `std::alloc`, but callers still have to manage the buffer. `print!` and `seek` do not themselves need an unsafe block.

There is no reliable failure path in this example. `File::open` returns a `File` containing a null pointer when opening fails, but Vx cannot test that pointer for null (Vx#714). `read` returns zero for end of file, a null file or buffer, and an I/O error. The Rust FFI currently converts a read error to zero with `unwrap_or(0)`. A short read is possible and must be handled separately when an exact number of bytes is required. The `File` handle now closes when dropped; the caller must still free the raw buffer on every exit path. See [`stdlib/std/fs.vx`](../../stdlib/std/fs.vx) and the Rust [file FFI](../../stdlib/rust_core/src/ffi/macros.rs).

## Target caller experience

The intended API should let a caller read a whole binary file without raw pointers, C allocation declarations, or an `unsafe` block. For example, the following is **proposed API syntax, not code that compiles today**:

<!-- vx-doctest: skip; proposed API and pattern syntax do not compile yet -->

```vx
import std::fs;

fn main() -> i32 {
  match fs::read_bytes("/path/to/example.parquet") {
    Ok(bytes) => {
      print!("File size: ");
      print!(bytes.len());
      return 0;
    }
    Err(error) => {
      print!("Could not read file: ");
      print!(error.message());
      return 1;
    }
  }
}
```

The final names and signatures should follow Vx's evolving `Result`, byte collection, and path APIs. The user-facing contract matters more than these provisional names: success returns owned bytes; failure returns a useful error; resources are released exactly once.

## Work to reach that API

For this plan, assume three prerequisite fixes have merged: `vx_file_drop` has the same `void` return type on both sides of the FFI; read and write lengths are unsigned; and the Unix Rust bridge preserves non-UTF-8 path bytes. Those fixes repair specific defects. They do not make `File::open`, `File::read`, or resource cleanup safe.

### Define the byte I/O contract

- [x] Document the proposed `std::io::Error` contract: portable category, operation, optional native error code, and message on request. Public types remain deferred pending Vx#570 and file API integration.
- [x] Define the current three open modes as named choices in Vx. Reject an unknown integer passed directly to the Rust bridge instead of treating it as read/write/create. Richer open options remain future work.
- [x] Specify byte-transfer results: success returns a count from zero through the requested length; a short positive transfer succeeds. Zero means EOF for a nonempty regular-file read or success for an empty request. An error returns no count and carries a portable category and optional native code. The bridge follows this contract; the Vx `File` methods do not yet expose it.
- [ ] Expose the fallible result through `File` once Vx can generate code for `Result<u64, std::io::Error>`. Preserve the old unsafe count-only methods until callers can migrate.
- [ ] Specify what happens on an interrupted call, a write that makes no progress, an invalid handle, and a failed close. Document whether an explicit close consumes the handle even when the OS reports an error.
- [ ] Define positioned `read_at` and `write_at` using a file byte offset. Keep them independent of the shared file cursor so separate requests can run concurrently. Define how sequential `read`, `write`, and `seek` use that cursor.
- [ ] When more than one reader or writer needs shared operations, define reusable sequential and positioned interfaces over bounded byte views. Preserve implementation-specific errors, or use an explicit common error type until Vx can support associated error types. Keep `File` usable without requiring these interfaces first.
- [ ] Reject a negative start offset in `seek`; do not cast it to `u64`. Check that file positions and requested lengths fit the platform API and the Vx return type.

### Replace the Rust file bridge with a Vx host backend

- [ ] Put the `File` handle and file-operation logic in `std::fs` written in Vx. Start with the supported macOS and Linux hosts; name any unsupported host explicitly.
- [ ] Add small, platform-specific declarations for the OS or C library calls needed to open, read, write, seek, perform positioned I/O, and close. Check their integer widths and calling conventions on each host.
- [ ] Capture the platform error immediately after a failed call and convert it to `std::io::Error`. Do not rely on a later read of process or thread error state.
- [ ] Use real positioned operations for `read_at` and `write_at`; a seek followed by a read or write changes shared cursor state and is not equivalent.
- [ ] Keep raw host calls inside the backend. Remove the file-specific `vx_file_*` Rust wrappers only after the Vx backend passes the same behavior tests. This does not require removing unrelated Rust core services in the same change.

### Provide safe paths and byte storage

- [ ] Define the accepted path type and its encoding policy. On Unix, preserve path bytes; reject embedded NUL rather than truncating. Define a separate policy for any future Windows backend.
- [ ] Add a bounded borrowed byte view or an owned byte buffer with a checked length and capacity. A safe read must take the buffer's capacity from that value, not a caller-supplied raw pointer and unrelated count. Keep a clearly marked unsafe raw entry point for low-level use.
- [ ] Make allocation failure and growth failure reportable for the owned byte buffer, with checked size conversions and an explicit maximum size. Today's `std::vec::Vec` has an `i32` length, requires manual `free`, and its Rust allocator aborts on allocation failure.
- [ ] Resolve Vx raw-pointer null checks (Vx#714) where the native backend or native allocator needs them. Test empty buffers and zero-length calls without forming an invalid reference.
- [ ] If the goal is **no Rust bridge anywhere in this API**, use Vx-owned path and byte storage as well: today's `std::vec::Vec` allocator and `std::string::String` storage call Rust core functions.

### Make ownership and cleanup enforceable

- [ ] Decide which values own a file handle or an allocated byte buffer and which operations borrow them. Check that returning either value inside `Result` preserves its move rules; generic enum linearity was addressed by Vx#715; verify the complete `File` move and drop behavior once the fallible API exists.
- [ ] Make explicit `close` and buffer release safe against repeated calls on the same value. Do not claim that copying a raw handle is safe merely because one wrapper sets its pointer to null.
- [x] Implement `Drop` for `File` so its handle is released on scope exit. Verify release through `Result` when file-returning operations exist; keep an explicit close operation for callers that need to observe a close error.
- [ ] Until those guarantees exist, document explicit close/free as required. A scoped helper may promise cleanup only if the language prevents its borrowed handle from escaping the scope; otherwise it is a convenience, not a safety guarantee.

### Add helpers on top of the bounded operations

- [ ] Implement `read_exact` as a loop that handles short reads and reports premature EOF separately from an I/O error.
- [ ] Implement `write_all` as a loop that handles short writes and reports a zero-progress write as an error.
- [ ] Implement `read_bytes(path)` by reading in bounded chunks and growing its output with checked allocation. Do not assume an initial file-size query remains correct while the file is being read.
- [ ] Set a documented whole-file size limit and decide what happens when a file grows past it. Keep chunked and positioned reads available for model weights and query scans that should not load a whole file.
- [ ] Provide an opt-in buffered reader and writer over the sequential interfaces. Define `fill_buf`/`consume`, seek and `into_inner` behavior, explicit writer flush, and recovery of unwritten bytes after failure. Do not silently buffer positioned operations.

### Prove the public behavior and update its status

- [ ] Add Vx tests through the public `std::fs` API that compare bytes, including an empty file and a missing file. The existing [`ffi_fs.vx`](../../tests/backend/pass/ffi_fs.vx) tests raw calls and byte counts only.
- [ ] Test distinct read, write, seek, and close errors; offset boundaries; embedded NUL in a path; and cleanup on both success and failure. Use controlled inputs or an injected backend to force short reads and writes instead of assuming a regular file will produce them.
- [ ] Run the host tests on macOS and Linux. Check that the public file API no longer links to the file-specific Rust bridge once it is replaced.
- [ ] Document the supported API, error and size policies, cleanup rules, and a small binary-file example. Split the roadmap's basic host I/O checkmark from completion of the usable standard-library API.

## Native Vx end state and the OS boundary

The intended end state has `File`, the error type, path and byte types, I/O loops, and convenience functions implemented in Vx. A host program still needs to cross the operating system's interface to access a file. Calling that interface through `extern` declarations is technically FFI, but it does **not** require a Rust bridge or a Rust-owned `std::fs::File`. The boundary should be small and platform-specific; the public API should hide it.

If “no FFI” means no external calls at all, portable host file I/O cannot meet that requirement. Direct system calls through compiler intrinsics or inline assembly would still use an OS ABI and would need separate implementations for each supported target. The same distinction applies later to GPU storage backends: Vx can own the request and buffer contract while a backend calls a platform or vendor interface.

Removing only the current file wrappers would leave Rust beneath today's Vx `Vec` and `String`. Reaching a file API with no Rust bridge therefore includes replacing or avoiding those storage implementations, plus the Vx ownership and allocation work above. This is separate from moving file I/O into device-portable `core`, which is not proposed.
