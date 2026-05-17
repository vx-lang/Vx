# Vx Standard Library: I/O and Networking Walkthrough

This document summarizes the newly added features to the Vx Standard Library, fully supporting basic File I/O, Standard Streams, and TCP/UDP Networking.

## Added Capabilities

### 1. Standard Streams (`std::io`)
You can now read from and write to standard operating system streams. This handles standard input, standard output, and standard error directly.

```rust
import std::io;

fn main() -> i32 {
    let msg = "Hello from Vx StdLib!\n";
    stdout_write(msg, 22);
    stderr_write("Error message\n", 14);

    // Reading is also supported using a pre-allocated buffer
    // let mut buffer: [u8; 1024];
    // stdin_read(buffer, 1024);

    return 0;
}
```

### 2. TCP Server (`std::net`)
Vx now supports hosting TCP servers through `TcpListener`. Combined with `TcpStream`, full bidirectional client-server applications can be written in Vx.

```rust
import std::net;
import std::io;

fn start_server() {
    let listener = TcpListener::bind("127.0.0.1:8080");
    let stream = listener.accept(); // Blocks until connection

    // Handle the stream
    // let mut buffer: [u8; 1024];
    // stream.read(buffer, 1024);

    tcp_stream_drop(stream);
    tcp_listener_drop(listener);
}
```

## Validation
- ✅ **Test Coverage**: Created integration tests `ffi_stdio.vx` and `ffi_tcp_server.vx`.
- ✅ **JIT Linker Compatibility**: Resolved linker errors by rebuilding `vx_std_core` and properly exporting `#[no_mangle]` symbols.
- ✅ **Parser Fixes**: Resolved implicit return bugs in `Vx` syntax parsing where missing semicolons in FFI declarations caused the `if` block parser to fail.
- ✅ **Unified Output Stream**: Updated the `execute_mlir` test infrastructure to capture both `stdout` and `stderr` so that tests can assert properly against them.

## Next Steps
The core system interoperability is completed. You can now build high-level web servers, logging systems, or general utilities natively in Vx!
