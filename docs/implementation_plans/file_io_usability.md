<!--
Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# File I/O usability in the Vx standard library

**Status:** incubating, with the feature name `file_io_errors`. The `Result` file API and typed seeking work. Safe path and byte-buffer APIs wait for slices and text.

## Current API and migration

`File::open`, `File::read` and `File::write` are safe. `open` takes the right to open files (a `&Files`, which `main` gets in its `Root`), a `String` path and an `OpenMode`, and gives a `Result<File, FileError>`. `read` and `write` take a `Vec<u8>` and give a `Result<u64, FileError>`. `Folder` narrows the right to the files of one folder. The older raw-pointer methods are `open_c_str`, `read_raw` and `write_raw`.

The incubating `IoError` API adds the methods below. They take raw pointers, so each one is `unsafe`, and none of them needs a `&Files`.

- `File::open_raw(path)` opens a file that exists, for reading. It returns `Result<File, IoError>`.
- `File::create_raw(path)` makes a file for writing, or makes an existing file empty.
- `File::create_new_raw(path)` makes a new file for reading and writing, in one step. It fails if the file exists. This is Rust's `File::create_new`.
- `OpenOptions::new()` turns every choice off. Its setters `read`, `write`, `append`, `create`, `truncate` and `create_new` take a `bool`.
- `OpenOptions::open_raw(path)` checks the combination of choices, and returns an `IoError` when the open fails.
- `File::try_read_raw` and `try_write_raw` take a raw buffer and a `usize` length. They return `Result<usize, IoError>`.
- `File` implements `Seek::seek_to`. `SeekFrom::Start(u64)`, `Current(i64)` and `End(i64)` give the position.
- `Seek::stream_position` returns the position.
- `File` owns its handle, and `Drop` closes it. A failed open makes no `File`.
- `File` implements `HostOnly`, so device code cannot use it (E6038). The E6038 message gives the reason for a channel or a lock ("can wait for the host"), which is wrong for a file. Issue #3620 tracks the fix. Until then, `a_file_in_a_device_body.vx` checks only the start of the message.
- `IoError::from_kind` makes an error with no OS code. `from_raw_os_error` keeps the OS code and gets its kind on this host.
- `IoError::with_operation` sets the operation. `message_c_str` returns the name of the kind as a C string, until Vx has text formatting.

A raw path must point to a NUL-terminated string. On Unix, the path can hold bytes that are not UTF-8.
A read buffer must hold `len` bytes that the program can write. A write buffer must hold `len` bytes with values.
A buffer must not be null, also for an empty request. The `requires valid(buffer, len)` contract says this, and the compiler tells LLVM that the pointer is not null.
A short transfer is not an error. A read count of 0 is the end of the file, and an error is an `Err`.
On Unix, a read does not clear the buffer first. On macOS, a read asks for at most `i32::MAX - 1` bytes, because macOS refuses a larger request.
The Rust file bridge still does the work. [The error contract](io_error_contract.md) gives its status codes.

`IoError` is `Copy` today. A later message or cause, as in Rust's `io::Error`, cannot be `Copy`. So adding one removes `Copy`, which breaks a program that copies an `IoError`. The doc comment of `IoError` warns about this. RFC question Q3 decides the shape of the error type, and `Copy` must not decide it first.

## Raw-pointer APIs that hide errors

These methods hide errors: a failed `open_c_str` gives a `File` with a null handle, and a failed `read_raw` or `write_raw` gives 0. Use the safe methods, or the incubating `IoError` methods, in new code.

| Method that hides errors | Replacement |
| --- | --- |
| `File::open_c_str(path, OpenMode) -> File` | `File::open`, or `File::open_raw`, `create_raw` or `OpenOptions::open_raw` |
| `File::read_raw(buffer, u64) -> u64` | `File::read(&mut Vec<u8>)`, or `File::try_read_raw(buffer, usize) -> Result<usize, IoError>` |
| `File::write_raw(buffer, u64) -> u64` | `File::write(&Vec<u8>)`, or `File::try_write_raw(buffer, usize) -> Result<usize, IoError>` |
| `File::seek(offset, whence) -> i64` | `Seek::seek_to(SeekFrom) -> Result<u64, IoError>` |

New runtime declarations are in `std::fs::raw`. The old runtime and C declarations stay in `std::fs`, so their paths do not change.
Vx has no private functions yet, so a program can still call these declarations.

## Safe API gates

The names in the table below are placeholders. No API in it has an implementation yet.

| Prerequisite | API it enables | Required safety contract |
| --- | --- | --- |
| Safe paths and text | `File::open_path(path)`, `create(path)`, `OpenOptions::open(path)` | Preserve Unix path bytes and reject embedded NUL. |
| Bounded borrowed byte slices | `Read::read(&mut bytes)`, `Write::write(&bytes)` | Enforce bounds, initialization, exclusive mutable access, and owner lifetime. |
| Bounded slices and reusable byte traits | `read_exact`, `write_all`, buffered adapters | Handle short operations, interruption, EOF, and zero-progress writes consistently. |
| Safe text and formatting traits | `Display`, `Debug`, error messages and causes | Format without exposing C pointers or requiring eager allocation. |

Byte counts use `usize`; absolute file offsets use `u64`.
The intended basic `Read` and `Write` operations return `Result<usize, IoError>`.
`read_exact` and `write_all` return `Result<void, IoError>` and retry interrupted calls.
Basic operations report interruption. Exact reads report `UnexpectedEof`, and complete writes report `WriteZero` when progress stops.
`Seek::rewind` is also deferred until generic `Result<void, IoError>::unwrap` works on both code generators.
This branch does not implement those prerequisites or claim the gated interfaces are usable.

## Blockers and their issues

A program that makes a file and writes a string to it needs about 40 lines today (see the book chapter on files).
The table lists what makes it long, and what removes each part. "No issue yet" means that nobody filed one.

| Problem today | What removes it | Issue |
| --- | --- | --- |
| Each call needs `unsafe`, because a buffer is a raw pointer and its size is not checked. | Bounded byte slices | #534 |
| A path is a raw C string. | A `Path` type that keeps Unix path bytes and refuses an embedded NUL | No issue yet |
| A string literal is `*const i8`, but `try_write_raw` takes `*const u8`. A program must copy the text into a `Vec<u8>`. | `str` with `as_bytes()` | #532 |
| Vx cannot cast `*const i8` to `*const u8`. | A pointer cast between byte types. This is smaller than `str`, and helps before it. | No issue yet |
| A write needs a loop, because `try_write_raw` can take fewer bytes. A read of an exact size also needs a loop. | `write_all` and `read_exact` on slices. Before slices, `write_all_raw` and `read_exact_raw` on raw pointers can remove the loop. | No issue yet |
| An `IoError` cannot print itself, so a program calls `message_c_str`. | Text formatting traits, from the core-library plan | No issue yet |
| `Vec` lengths are `i32`. | `Vec` with 64-bit lengths | #3271 |
| Each program imports `core::result`, `std::fs` and `std::io`. | A prelude | #529 |
| Device code that uses a `File` gets E6038 with a reason for channels and locks. | A message that is true for each `HostOnly` type | #3620 |

Two compiler bugs made the example in the book chapter harder to write:

- #3621: `let text = "hi"; text[0]` gives a wrong value, with no error. The workaround is `let text : *const i8 = "hi";`.
- #3622: with the AST code generator, a `match` with a `return` arm fails after `import std::string`. The flat code generator compiles it. A program also reaches the AST code generator when the flat code generator declines a function, for example because of #3621.

These questions are open, and have no issue yet:

- Can `main` return a `Result`, so that `?` works in `main`?
- Inside a task, the Rust bridge sends each file call to a helper thread, so that the task's worker can run other tasks. That costs a heap allocation, atomic operations, a wake and a park for each call. A benchmark against Rust must measure it, from a plain thread and from inside a task.

[`file_read_and_write_make_one_runtime_call.vx`](../../tests/optimizations/pass/file_read_and_write_make_one_runtime_call.vx) checks that `try_read_raw` and `try_write_raw` make one runtime call and no other work after `opt -O3`.

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

- [x] Document the `std::io::Error` contract: portable category, operation, optional native error code, and message on request. Public types and fallible raw file methods are implemented.
- [x] Define the current three open modes as named choices in Vx. Reject an unknown integer passed directly to the Rust bridge instead of treating it as read/write/create. Richer open options remain future work.
- [x] Specify byte-transfer results: success returns a count from zero through the requested length; a short positive transfer succeeds. Zero means EOF for a nonempty regular-file read or success for an empty request. An error returns no count and carries a portable category and optional native code. The bridge follows this contract; the Vx `try_read_raw` and `try_write_raw` methods expose it.
- [x] Expose fallible open/read/write through `File::open_raw`, `try_read_raw`, and `try_write_raw`. Preserve the explicit legacy methods during migration.
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

The Vx backend needs these language and runtime features first. None of them has an issue yet:

- **Code and constants for each OS.** `O_CREAT` is `0x200` on macOS and `0x40` on Linux, and `EAGAIN` is 35 on macOS and 11 on Linux. The program reads `errno` through `__error()` on macOS and `__errno_location()` on Linux. Vx has no way to choose code or constants for each OS.
- **Variadic `extern` functions.** C declares `open(path, flags, ...)`, and the mode is a variadic argument. On Apple arm64, a variadic argument goes on the stack, so a declaration without `...` passes the mode in the wrong place. `stdlib/std/libc.vx` declares `open` with two parameters today.
- **No call between the system call and the `errno` read.** The generated code must not put a call there. The optimized IR of a file program has runtime calls such as `vx_cancel_on`. A test must check where they go.
- **A way to run a call off the task's worker.** `off_the_worker` is part of the Rust scheduler. A Vx backend needs an entry point that Vx can call, or it must accept that a file call blocks the worker.

Removing Rust from `Vec` and `String` is separate work: #3271 for `Vec`, and #846 for `String`.

### Provide safe paths and byte storage

- [ ] Define the accepted path type and its encoding policy. On Unix, preserve path bytes; reject embedded NUL rather than truncating. Define a separate policy for any future Windows backend.
- [ ] Add a bounded borrowed byte view or an owned byte buffer with a checked length and capacity. A safe read must take the buffer's capacity from that value, not a caller-supplied raw pointer and unrelated count. Keep a clearly marked unsafe raw entry point for low-level use.
- [ ] Make allocation failure and growth failure reportable for the owned byte buffer, with checked size conversions and an explicit maximum size. Today's `std::vec::Vec` has an `i32` length, requires manual `free`, and its Rust allocator aborts on allocation failure.
- [x] Resolve Vx raw-pointer null checks (Vx#714). `core::ptr::is_null_mut` works. Test empty buffers and zero-length calls without forming an invalid reference.
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
