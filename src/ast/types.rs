use super::*;

//===- ast.rs - Vx Compiler ------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// This file defines the Abstract Syntax Tree (AST) structures for the Vx language.
// It contains the enums and structs representing expressions, statements, types,
// and declarations, serving as the foundational data model for the entire frontend
// of the compiler.
//
//===----------------------------------------------------------------------===//
#[derive(Debug, PartialEq, Clone, Default)]
pub struct Span {
    pub line: usize,
    pub column: usize,
    pub length: usize,
}

#[derive(Debug, PartialEq, Clone)]
pub enum Topology {
    Host,
    NPU(Box<Expr>),
    AccCore(Box<Expr>),
    AMX,
    ANE,
    GPU,
    Slice(Box<Topology>, Box<Expr>, Box<Expr>), // For NPU[0..4] etc.
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum MemorySpace {
    HostDRAM,
    NPUHBM,
    LocalSRAM,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ElementType {
    F16,
    F32,
    F64,
    BF16,
    I4,
    U4,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    I128,
    U128,
    Bool,
    Generic(String),
}

#[derive(Debug, PartialEq, Clone)]
pub enum Type {
    Tensor(ElementType, Vec<Expr>, Option<Topology>),
    Matrix,
    Ref(Box<Type>, MemorySpace),
    Borrow(Box<Type>, Option<MemorySpace>, bool, usize), // (type, mem_space, is_mut, region_id)
    Pointer(Box<Type>, Option<MemorySpace>, bool),       // (type, mem_space, is_mut)
    Scalar(ElementType),
    Struct(String, Option<crate::gid::TypeId>),
    Enum(String, Option<crate::gid::TypeId>),
    Verified(Box<Type>),
    Pinned(Box<Type>, Topology),
    Generic(String, Option<crate::gid::TypeId>), // e.g. T
    GenericInstance(Box<Type>, Vec<Type>),       // e.g. Config<f32>
    Module(String, std::collections::HashMap<String, Type>), // (path, exported_symbols)
    Simd(ElementType, usize),                    // e.g. <4 x f32>
}

impl Type {
    pub fn is_linear(&self) -> bool {
        matches!(
            self,
            Type::Ref(_, _)
                | Type::Tensor(_, _, _)
                | Type::Matrix
                | Type::Verified(_)
                | Type::Pinned(_, _)
                | Type::Struct(_, _)
                | Type::Enum(_, _)
        )
    }

    pub fn substitute(&self, mapping: &std::collections::HashMap<String, Type>) -> Type {
        match self {
            Type::Generic(name, _) => {
                if let Some(concrete) = mapping.get(name) {
                    concrete.clone()
                } else {
                    self.clone()
                }
            }
            Type::GenericInstance(base, args) => {
                let new_base = base.substitute(mapping);
                let new_args = args.iter().map(|a| a.substitute(mapping)).collect();
                Type::GenericInstance(Box::new(new_base), new_args)
            }
            Type::Borrow(inner, mem, is_mut, region) => Type::Borrow(
                Box::new(inner.substitute(mapping)),
                mem.clone(),
                *is_mut,
                *region,
            ),
            Type::Pointer(inner, mem, is_mut) => {
                Type::Pointer(Box::new(inner.substitute(mapping)), mem.clone(), *is_mut)
            }
            Type::Tensor(el_ty, dims, top) => {
                let new_el_ty = if let ElementType::Generic(ref name) = el_ty {
                    if let Some(Type::Scalar(concrete_el)) = mapping.get(name) {
                        concrete_el.clone()
                    } else {
                        el_ty.clone()
                    }
                } else {
                    el_ty.clone()
                };
                let new_dims = dims.iter().map(|d| d.substitute(mapping)).collect();
                Type::Tensor(new_el_ty, new_dims, top.clone())
            }
            Type::Ref(inner, mem) => Type::Ref(Box::new(inner.substitute(mapping)), mem.clone()),
            Type::Verified(inner) => Type::Verified(Box::new(inner.substitute(mapping))),
            Type::Pinned(inner, top) => {
                Type::Pinned(Box::new(inner.substitute(mapping)), top.clone())
            }
            Type::Simd(el_ty, n) => {
                let new_el_ty = if let ElementType::Generic(ref name) = el_ty {
                    if let Some(Type::Scalar(concrete_el)) = mapping.get(name) {
                        concrete_el.clone()
                    } else {
                        el_ty.clone()
                    }
                } else {
                    el_ty.clone()
                };
                Type::Simd(new_el_ty, *n)
            }
            _ => self.clone(),
        }
    }
}
