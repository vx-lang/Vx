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
    instantiate_tcp_stream_ffi!();

    #[test]
    fn rejects_invalid_lengths_then_transfers_one_byte() -> std::io::Result<()> {
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let client = std::net::TcpStream::connect(listener.local_addr()?)?;
        let (server, _) = listener.accept()?;
        let client = Box::into_raw(Box::new(client)).cast();
        let server = Box::into_raw(Box::new(server)).cast();
        let mut byte = *b"X";
        let oversized = isize::MAX as u64 + 1;

        check_eq!(vx_tcp_stream_write(client, byte.as_ptr(), oversized), 0);
        check_eq!(vx_tcp_stream_read(server, byte.as_mut_ptr(), oversized), 0);
        check_eq!(vx_tcp_stream_write(client, byte.as_ptr(), 0), 0);
        check_eq!(vx_tcp_stream_read(server, byte.as_mut_ptr(), 0), 0);
        check_eq!(
            vx_tcp_stream_write(std::ptr::null_mut(), byte.as_ptr(), 1),
            0
        );
        check_eq!(
            vx_tcp_stream_read(std::ptr::null_mut(), byte.as_mut_ptr(), 1),
            0
        );
        check_eq!(vx_tcp_stream_write(client, std::ptr::null(), 1), 0);
        check_eq!(vx_tcp_stream_read(server, std::ptr::null_mut(), 1), 0);
        check_eq!(vx_tcp_stream_write(client, byte.as_ptr(), 1), 1);
        byte[0] = 0;
        check_eq!(vx_tcp_stream_read(server, byte.as_mut_ptr(), 1), 1);
        check_eq!(byte, *b"X");

        vx_tcp_stream_drop(client);
        vx_tcp_stream_drop(server);
        Ok(())
    }
}

mod udp {
    instantiate_udp_socket_ffi!();

    #[test]
    fn rejects_invalid_lengths_then_transfers_one_byte() -> std::io::Result<()> {
        let receiver = std::net::UdpSocket::bind("127.0.0.1:0")?;
        let address = std::ffi::CString::new(receiver.local_addr()?.to_string())
            .map_err(std::io::Error::other)?;
        let sender = std::net::UdpSocket::bind("127.0.0.1:0")?;
        let receiver = Box::into_raw(Box::new(receiver)).cast();
        let sender = Box::into_raw(Box::new(sender)).cast();
        let mut byte = *b"X";
        let oversized = isize::MAX as u64 + 1;

        check_eq!(
            vx_udp_socket_send_to(sender, byte.as_ptr(), oversized, address.as_ptr()),
            0
        );
        check_eq!(
            vx_udp_socket_recv(receiver, byte.as_mut_ptr(), oversized),
            0
        );
        check_eq!(
            vx_udp_socket_send_to(sender, byte.as_ptr(), 0, address.as_ptr()),
            0
        );
        check_eq!(vx_udp_socket_recv(receiver, byte.as_mut_ptr(), 0), 0);
        check_eq!(
            vx_udp_socket_send_to(std::ptr::null_mut(), byte.as_ptr(), 1, address.as_ptr()),
            0
        );
        check_eq!(
            vx_udp_socket_recv(std::ptr::null_mut(), byte.as_mut_ptr(), 1),
            0
        );
        check_eq!(
            vx_udp_socket_send_to(sender, std::ptr::null(), 1, address.as_ptr()),
            0
        );
        check_eq!(
            vx_udp_socket_send_to(sender, byte.as_ptr(), 1, std::ptr::null()),
            0
        );
        check_eq!(vx_udp_socket_recv(receiver, std::ptr::null_mut(), 1), 0);
        check_eq!(
            vx_udp_socket_send_to(sender, byte.as_ptr(), 1, address.as_ptr()),
            1
        );
        byte[0] = 0;
        check_eq!(vx_udp_socket_recv(receiver, byte.as_mut_ptr(), 1), 1);
        check_eq!(byte, *b"X");

        vx_udp_socket_drop(sender);
        vx_udp_socket_drop(receiver);
        Ok(())
    }
}
