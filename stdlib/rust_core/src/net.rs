//===- net.rs - Vx Compiler ------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the standard library networking primitives.
// It provides the Rust FFI bindings for TCP/UDP sockets, allowing Vx programs
// to create servers, handle incoming connections, and stream data across the
// network.
//
//===----------------------------------------------------------------------===//
//! FFI bindings for `std::net::TcpStream` and `std::net::UdpSocket`.

use crate::{instantiate_tcp_listener_ffi, instantiate_tcp_stream_ffi, instantiate_udp_socket_ffi};

instantiate_tcp_stream_ffi!();
instantiate_udp_socket_ffi!();
instantiate_tcp_listener_ffi!();
