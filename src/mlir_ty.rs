//===- mlir_ty.rs - Vx Compiler --------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// How a Vx type is spelled in MLIR.
//
// A leaf module. Both backends and the flattener need to agree on these spellings, and they used
// to agree by having the same code twice -- `flatten::element_mlir` and `flat::mlir_scalar` were
// identical, arm for arm, including the comment explaining why fp8 declines.
//
//===----------------------------------------------------------------------===//

use crate::syntax::ElementType;

/// The MLIR spelling of a scalar element type, or `None` when the flat path has no lowering for it.
pub fn mlir_scalar(elem: &ElementType) -> Option<&'static str> {
    use ElementType::*;
    Some(match elem {
        F16 => "f16",
        F32 => "f32",
        F64 => "f64",
        BF16 => "bf16",
        I4 | U4 => "i4",
        I8 | U8 => "i8",
        I16 | U16 => "i16",
        I32 | U32 => "i32",
        I64 | U64 => "i64",
        I128 | U128 => "i128",
        Bool => "i1",
        // fp8 is capacity/declaration-only for now: the JIT has no fp8 arithmetic,
        // so the flat path declines. Compute support is #249.
        F8E4M3 | F8E5M2 | F4E2M1 => return None,
        Generic(_) => return None,
    })
}

/// The MLIR spelling of `<N x T>`: `vector<NxT>`. `None` when the element has no MLIR spelling
/// (fp8, a generic), which is the same condition that makes a scalar of it decline.
pub fn mlir_vector(elem: &ElementType, lanes: usize) -> Option<String> {
    Some(format!("vector<{lanes}x{}>", mlir_scalar(elem)?))
}

/// The float element type an MLIR spelling names, for reading a vector type back apart.
/// Floats only: the vectorized slice ops are the only place a spelling is re-parsed, and
/// they are float-only.
pub fn float_elem_of_mlir(spelling: &str) -> Option<ElementType> {
    use ElementType::*;
    Some(match spelling {
        "f16" => F16,
        "f32" => F32,
        "f64" => F64,
        "bf16" => BF16,
        _ => return None,
    })
}
