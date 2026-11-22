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
        I64 | U64 | USize => "i64",
        I128 | U128 => "i128",
        Bool => "i1",
        // LLVM has no type for these. The `vx-expand-narrow-floats` pass writes their
        // conversions as integer operations.
        F8E4M3 => "f8E4M3FN",
        F8E5M2 => "f8E5M2",
        F4E2M1 => "f4E2M1FN",
        F8E8M0 => "f8E8M0FNU",
        Generic(_) => return None,
    })
}

/// The MLIR spelling of `<N x T>`: `vector<NxT>`. `None` when the element has no MLIR spelling
/// (a generic), which is the same condition that makes a scalar of it decline.
pub fn mlir_vector(elem: &ElementType, lanes: usize) -> Option<String> {
    VectorTy::new(elem, lanes).map(|v| v.to_string())
}

/// The type of a vector value: a number of lanes of one element type. It prints as its MLIR
/// spelling, `vector<NxT>`.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorTy {
    lanes: usize,
    elem: ElementType,
}

impl VectorTy {
    /// `None` when the element type has no MLIR spelling (a generic).
    pub fn new(elem: &ElementType, lanes: usize) -> Option<Self> {
        mlir_scalar(elem)?;
        Some(Self {
            lanes,
            elem: elem.clone(),
        })
    }

    pub fn lanes(&self) -> usize {
        self.lanes
    }

    pub fn elem(&self) -> &ElementType {
        &self.elem
    }

    /// The same number of lanes of another element type.
    pub fn with_elem(&self, elem: &ElementType) -> Option<Self> {
        Self::new(elem, self.lanes)
    }
}

impl std::fmt::Display for VectorTy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let et = mlir_scalar(&self.elem).expect("VectorTy::new takes only a spellable element");
        write!(f, "vector<{}x{et}>", self.lanes)
    }
}

/// The float element type an MLIR spelling names. Floats only.
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
