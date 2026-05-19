//===- net.rs - Vx Compiler ------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//! FFI bindings for `std::net::TcpStream` and `std::net::UdpSocket`.

use crate::{instantiate_tcp_listener_ffi, instantiate_tcp_stream_ffi, instantiate_udp_socket_ffi};

instantiate_tcp_stream_ffi!();
instantiate_udp_socket_ffi!();
instantiate_tcp_listener_ffi!();
