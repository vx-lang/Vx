<!--
Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# File I/O usability in the Vx standard library

**Status:** experimental. Fallible raw file APIs and typed seeking are implemented. Safe path and byte-buffer APIs remain gated.

## Current API and migration

- `File::open_raw(path)` opens an existing file for reading and returns `Result<File, Error>`.
- `File::create_raw(path)` creates or truncates a writable file.
- `File::create_new_raw(path)` creates a writable file atomically and fails if it exists.
- `OpenOptions::new()` starts with every option disabled. Its `read`, `write`, `append`, `create`, `truncate`, and `create_new` setters accept booleans.
- `OpenOptions::open_raw(path)` validates combinations and returns a structured error on failure.
- `File::read_raw` and `write_raw` take raw buffers with `usize` lengths and return `Result<usize, Error>`.
- `File` implements `Seek::seek_to`, which keeps the existing `File::seek(offset, whence)` available. `SeekFrom::Start(u64)`, `Current(i64)`, and `End(i64)` provide typed offsets.
- `Seek::stream_position` returns the position. Seek to `SeekFrom::Start(0)` to return to the beginning.
- `File` owns its handle and closes it through `Drop`. A failed fallible open does not produce a `File`.
- `Error::from_kind` constructs a portable error. `from_raw_os_error` classifies the native code on this host.
- `Error::with_operation` attaches context. `message_c_str` explicitly exposes the temporary C-string representation.

Raw opening requires a live NUL-terminated path. Unix preserves non-UTF-8 bytes.
Raw reading requires writable memory. Raw writing requires initialized readable memory.
Short transfers succeed, EOF is distinct from failure, and empty requests succeed without accessing buffer memory.
On Unix, reads accept uninitialized writable memory without clearing the full request first.
The existing file bridge remains the backend. [The error contract](io_error_contract.md) specifies its status ABI.

The existing APIs retain their names, signatures, safety qualifiers, and behavior:

| Compatibility API | Recommended alternative for new code |
| --- | --- |
| `File::open(path, OpenMode) -> File` | `File::open_raw`, `create_raw`, or `OpenOptions::open_raw` |
| `File::read(buffer, u64) -> u64` | `File::read_raw(buffer, usize) -> Result<usize, Error>` |
| `File::write(buffer, u64) -> u64` | `File::write_raw(buffer, usize) -> Result<usize, Error>` |
| `File::seek(offset, whence) -> i64` | `File::seek_to(SeekFrom) -> Result<u64, Error>` or `Seek::seek_to` |

`OpenMode` and its three variants remain available. Existing callers need no migration.
The old operations retain their error-hiding behavior. New code should use the fallible alternatives.
Vx has no implemented deprecation annotations, so this recommendation is documentation-only.
Add compiler deprecation warnings when that facility exists. Do not remove or rename the compatibility APIs in a routine release.
The intermediate `try_*` names were introduced only during this branch and are not released compatibility APIs.

New runtime declarations live in `std::fs::raw`. Existing runtime and C declarations remain in `std::fs` to preserve their paths.
New application code should prefer the fallible `File` methods.
Vx does not yet provide Rust-style private function visibility, so module placement alone does not enforce backend privacy.

## Safe API gates

Existing `File::open`, `read`, `write`, and `seek` remain compatibility APIs.
Future safe APIs must be additive, through new methods, types, or trait-qualified operations.
Do not replace existing signatures to reclaim a preferred name. Gated APIs have no placeholder implementations.

| Prerequisite | API it enables | Required safety contract |
| --- | --- | --- |
| Safe paths and text | `File::open_path(path)`, `create(path)`, `OpenOptions::open(path)` | Preserve Unix path bytes and reject embedded NUL. |
| Bounded borrowed byte slices | `Read::read(&mut bytes)`, `Write::write(&bytes)` | Enforce bounds, initialization, exclusive mutable access, and owner lifetime. |
| Bounded slices and reusable byte traits | `read_exact`, `write_all`, buffered adapters | Handle short operations, interruption, EOF, and zero-progress writes consistently. |
| Safe text and formatting traits | `Display`, `Debug`, error messages and causes | Format without exposing C pointers or requiring eager allocation. |

Byte counts use `usize`; absolute file offsets use `u64`.
The intended basic `Read` and `Write` operations return `Result<usize, Error>`.
`read_exact` and `write_all` return `Result<void, Error>` and retry interrupted calls.
Basic operations report interruption. Exact reads report `UnexpectedEof`, and complete writes report `WriteZero` when progress stops.
Safe borrowed slices are tracked in #534, safe text in #532, and formatting in the core-library plan.
`Seek::rewind` is also deferred until generic `Result<void, Error>::unwrap` works on both code generators.
This branch does not implement those prerequisites or claim the gated interfaces are usable.

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
- [x] Specify byte-transfer results: success returns a count from zero through the requested length; a short positive transfer succeeds. Zero means EOF for a nonempty regular-file read or success for an empty request. An error returns no count and carries a portable category and optional native code. The bridge follows this contract; the Vx `read_raw` and `write_raw` methods expose it.
- [x] Expose fallible open/read/write through `File::open_raw`, `read_raw`, and `write_raw`. Preserve the explicit legacy methods during migration.
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
