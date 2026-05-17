//! FFI bindings for `std::fs::File`.

use crate::{instantiate_file_ffi, instantiate_stdio_ffi};

instantiate_file_ffi!();
instantiate_stdio_ffi!();
