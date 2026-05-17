//! FFI bindings for `std::net::TcpStream` and `std::net::UdpSocket`.

use crate::{instantiate_tcp_listener_ffi, instantiate_tcp_stream_ffi, instantiate_udp_socket_ffi};

instantiate_tcp_stream_ffi!();
instantiate_udp_socket_ffi!();
instantiate_tcp_listener_ffi!();
