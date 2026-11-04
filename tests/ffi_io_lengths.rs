//===- ffi_io_lengths.rs - Vx Compiler ------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Exercise the same FFI macros that the Rust core uses.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

include!("../stdlib/rust_core/src/ffi/macros.rs");

// The sockets of these tests block, so a call never needs a second try.
fn call_once<T>(
    _fd: std::os::fd::RawFd,
    _write: bool,
    mut call: impl FnMut() -> std::io::Result<T>,
) -> std::io::Result<T> {
    call()
}

fn connect_and_block(addr: &str) -> std::io::Result<std::net::TcpStream> {
    std::net::TcpStream::connect(addr)
}

fn lookup(addr: &str) -> std::io::Result<Vec<std::net::SocketAddr>> {
    Ok(std::net::ToSocketAddrs::to_socket_addrs(addr)?.collect())
}

/// The status that a call with a null pointer or a length that is not a size gives.
const EINVAL: i32 = 22;

fn status(e: &std::io::Error) -> i32 {
    e.raw_os_error().unwrap_or(-6)
}

// Report a failed check through the test's Result, with the expression that failed.
macro_rules! check_eq {
    ($actual:expr, $expected:expr) => {{
        let actual = $actual;
        let expected = $expected;
        if actual != expected {
            return Err(std::io::Error::other(format!(
                "{}: expected {expected:?}, got {actual:?}",
                stringify!($actual)
            )));
        }
    }};
}

mod stdio {
    instantiate_stdio_ffi!();

    #[test]
    fn rejects_invalid_lengths_and_pointers() -> std::io::Result<()> {
        let mut byte = *b"X";
        let oversized = isize::MAX as u64 + 1;

        check_eq!(vx_stdout_write(byte.as_ptr(), oversized), 0);
        check_eq!(vx_stderr_write(byte.as_ptr(), oversized), 0);
        check_eq!(vx_stdin_read(byte.as_mut_ptr(), oversized), 0);
        check_eq!(vx_stdout_write(byte.as_ptr(), 0), 0);
        check_eq!(vx_stderr_write(byte.as_ptr(), 0), 0);
        check_eq!(vx_stdin_read(byte.as_mut_ptr(), 0), 0);
        check_eq!(vx_stdout_write(std::ptr::null(), 1), 0);
        check_eq!(vx_stderr_write(std::ptr::null(), 1), 0);
        check_eq!(vx_stdin_read(std::ptr::null_mut(), 1), 0);
        check_eq!(byte, *b"X");
        Ok(())
    }
}

mod tcp {
    instantiate_tcp_stream_ffi!(super::call_once, super::connect_and_block, super::status);

    /// The status of a read or write of `len` bytes, and the count that it gave.
    fn read(s: *mut std::ffi::c_void, p: *mut u8, len: u64) -> (i32, u64) {
        let mut n = 0;
        (vx_tcp_stream_read(s, p, len, &mut n), n)
    }

    fn write(s: *mut std::ffi::c_void, p: *const u8, len: u64) -> (i32, u64) {
        let mut n = 0;
        (vx_tcp_stream_write(s, p, len, &mut n), n)
    }

    #[test]
    fn rejects_invalid_lengths_then_transfers_one_byte() -> std::io::Result<()> {
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let client = std::net::TcpStream::connect(listener.local_addr()?)?;
        let (server, _) = listener.accept()?;
        let client = Box::into_raw(Box::new(client)).cast();
        let server = Box::into_raw(Box::new(server)).cast();
        let mut byte = *b"X";
        let oversized = isize::MAX as u64 + 1;

        check_eq!(write(client, byte.as_ptr(), oversized), (super::EINVAL, 0));
        check_eq!(
            read(server, byte.as_mut_ptr(), oversized),
            (super::EINVAL, 0)
        );
        check_eq!(write(client, byte.as_ptr(), 0), (0, 0));
        check_eq!(read(server, byte.as_mut_ptr(), 0), (0, 0));
        check_eq!(
            write(std::ptr::null_mut(), byte.as_ptr(), 1),
            (super::EINVAL, 0)
        );
        check_eq!(
            read(std::ptr::null_mut(), byte.as_mut_ptr(), 1),
            (super::EINVAL, 0)
        );
        check_eq!(write(client, std::ptr::null(), 1), (super::EINVAL, 0));
        check_eq!(read(server, std::ptr::null_mut(), 1), (super::EINVAL, 0));
        check_eq!(write(client, byte.as_ptr(), 1), (0, 1));
        byte[0] = 0;
        check_eq!(read(server, byte.as_mut_ptr(), 1), (0, 1));
        check_eq!(byte, *b"X");

        vx_tcp_stream_drop(client);
        vx_tcp_stream_drop(server);
        Ok(())
    }
}

mod udp {
    instantiate_udp_socket_ffi!(super::call_once, super::lookup, super::status);

    /// The status of a send or receive of `len` bytes, and the count that it gave.
    fn recv(s: *mut std::ffi::c_void, p: *mut u8, len: u64) -> (i32, u64) {
        let mut n = 0;
        (vx_udp_socket_recv(s, p, len, &mut n), n)
    }

    fn send_to(
        s: *mut std::ffi::c_void,
        p: *const u8,
        len: u64,
        to: *const std::ffi::c_char,
    ) -> (i32, u64) {
        let mut n = 0;
        (vx_udp_socket_send_to(s, p, len, to, &mut n), n)
    }

    #[test]
    fn rejects_invalid_lengths_then_transfers_one_byte() -> std::io::Result<()> {
        let receiver = std::net::UdpSocket::bind("127.0.0.1:0")?;
        let address = std::ffi::CString::new(receiver.local_addr()?.to_string())
            .map_err(std::io::Error::other)?;
        let sender = std::net::UdpSocket::bind("127.0.0.1:0")?;
        let receiver = Box::into_raw(Box::new(receiver)).cast();
        let sender = Box::into_raw(Box::new(sender)).cast();
        let to = address.as_ptr();
        let mut byte = *b"X";
        let oversized = isize::MAX as u64 + 1;

        check_eq!(
            send_to(sender, byte.as_ptr(), oversized, to),
            (super::EINVAL, 0)
        );
        check_eq!(
            recv(receiver, byte.as_mut_ptr(), oversized),
            (super::EINVAL, 0)
        );
        check_eq!(send_to(sender, byte.as_ptr(), 0, to), (0, 0));
        check_eq!(recv(receiver, byte.as_mut_ptr(), 0), (0, 0));
        check_eq!(
            send_to(std::ptr::null_mut(), byte.as_ptr(), 1, to),
            (super::EINVAL, 0)
        );
        check_eq!(
            recv(std::ptr::null_mut(), byte.as_mut_ptr(), 1),
            (super::EINVAL, 0)
        );
        check_eq!(send_to(sender, std::ptr::null(), 1, to), (super::EINVAL, 0));
        check_eq!(
            send_to(sender, byte.as_ptr(), 1, std::ptr::null()),
            (super::EINVAL, 0)
        );
        check_eq!(recv(receiver, std::ptr::null_mut(), 1), (super::EINVAL, 0));
        check_eq!(send_to(sender, byte.as_ptr(), 1, to), (0, 1));
        byte[0] = 0;
        check_eq!(recv(receiver, byte.as_mut_ptr(), 1), (0, 1));
        check_eq!(byte, *b"X");

        vx_udp_socket_drop(sender);
        vx_udp_socket_drop(receiver);
        Ok(())
    }
}
