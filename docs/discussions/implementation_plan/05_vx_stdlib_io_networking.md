# Vx StdLib: Completing System I/O & Networking

To fully complete the system I/O and networking capabilities, we will implement the missing TCP Server functionality and Standard Input/Output streams.

## Proposed Changes

### 1. TCP Server Capabilities (`TcpListener`)

Currently, Vx can only act as a TCP client. We will add `TcpListener` to allow Vx to host servers.

#### `stdlib/rust_core/src/ffi/macros.rs`
- Add `instantiate_tcp_listener_ffi!()` macro.
- Generates: `vx_tcp_listener_bind`, `vx_tcp_listener_accept`, `vx_tcp_listener_drop`.
- `accept` will return a new `*mut c_void` pointer representing the `TcpStream` handle for the accepted connection.

#### `stdlib/rust_core/src/net.rs`
- Instantiate the new macro: `instantiate_tcp_listener_ffi!();`

#### `stdlib/std/net.vx`
- Define the opaque `TcpListener` struct.
- Declare `extern` functions for `vx_tcp_listener_bind`, `vx_tcp_listener_accept`, and `vx_tcp_listener_drop`.
- Add wrappers in the `TcpListener` namespace.

### 2. Standard Streams (`stdin`, `stdout`, `stderr`)

Currently, Vx relies on `printf` for output. We will expose the standard streams as global `File` handles.

#### `stdlib/rust_core/src/ffi/macros.rs`
- Add macros or explicit `extern "C"` functions for:
  - `vx_io_stdin()` -> `*mut c_void`
  - `vx_io_stdout()` -> `*mut c_void`
  - `vx_io_stderr()` -> `*mut c_void`
- These functions will return pointers to `Box<std::io::Stdin>`, `Box<std::io::Stdout>`, etc. But since `std::io::Stdout` implements `Write`, we might need to expose generic stream reading/writing, or just map them directly. Actually, the simplest approach is to provide FFI functions like `vx_stdout_write` directly to avoid complex trait object FFI, OR map them to the same `File` FFI by boxing them as `Box<dyn Write>` (but our current `File` FFI strictly uses `std::fs::File`).
- **Simpler approach**: Implement specific `extern "C" fn vx_print_stdout(buffer: *const u8, len: usize)` and `vx_read_stdin(buffer: *mut u8, len: usize)`.

#### `stdlib/std/io.vx`
- Add `print(str: String)` and `read_line() -> String` high-level wrappers.

## Verification Plan
1. Create `tests/backend/pass/ffi_tcp_server.vx` that binds a listener, accepts a connection, and reads from it (can be tested by connecting a client to itself in the same process, or simulating a connection).
2. Create `tests/backend/pass/ffi_stdio.vx` to verify basic output to stdout.

## Open Questions
> [!IMPORTANT]
> The `TcpListener` accept call blocks until a client connects. In our automated test, we will need to spawn a background thread or process, or just use non-blocking sockets. How would you like the integration test for `TcpListener` to verify the server is working without hanging the test suite indefinitely? (e.g. Setting `set_nonblocking(true)` and polling, or simply omitting the end-to-end execution test for `accept`?)
