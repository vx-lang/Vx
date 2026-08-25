//===- mlir_ty.rs - Vx Compiler --------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
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
        F8E4M3 | F8E5M2 => return None,
        Generic(_) => return None,
    })
}
