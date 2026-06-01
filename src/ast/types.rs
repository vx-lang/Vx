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

impl std::fmt::Display for ElementType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ElementType::F16 => write!(f, "f16"),
            ElementType::F32 => write!(f, "f32"),
            ElementType::F64 => write!(f, "f64"),
            ElementType::BF16 => write!(f, "bf16"),
            ElementType::I4 => write!(f, "i4"),
            ElementType::U4 => write!(f, "u4"),
            ElementType::I8 => write!(f, "i8"),
            ElementType::U8 => write!(f, "u8"),
            ElementType::I16 => write!(f, "i16"),
            ElementType::U16 => write!(f, "u16"),
            ElementType::I32 => write!(f, "i32"),
            ElementType::U32 => write!(f, "u32"),
            ElementType::I64 => write!(f, "i64"),
            ElementType::U64 => write!(f, "u64"),
            ElementType::I128 => write!(f, "i128"),
            ElementType::U128 => write!(f, "u128"),
            ElementType::Bool => write!(f, "Bool"),
            ElementType::Generic(g) => write!(f, "{}", g),
        }
    }
}

impl std::str::FromStr for ElementType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "f16" => Ok(ElementType::F16),
            "f32" => Ok(ElementType::F32),
            "f64" => Ok(ElementType::F64),
            "bf16" => Ok(ElementType::BF16),
            "i4" => Ok(ElementType::I4),
            "u4" => Ok(ElementType::U4),
            "i8" => Ok(ElementType::I8),
            "u8" => Ok(ElementType::U8),
            "i16" => Ok(ElementType::I16),
            "u16" => Ok(ElementType::U16),
            "i32" => Ok(ElementType::I32),
            "u32" => Ok(ElementType::U32),
            "i64" => Ok(ElementType::I64),
            "u64" => Ok(ElementType::U64),
            "i128" => Ok(ElementType::I128),
            "u128" => Ok(ElementType::U128),
            "Bool" => Ok(ElementType::Bool),
            _ => Err(format!("Unknown element type '{}'", s)),
        }
    }
}

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::Scalar(el) => write!(f, "{}", el),
            Type::Struct(name, _) => write!(f, "{}", name),
            Type::Enum(name, _) => write!(f, "{}", name),
            Type::Pointer(inner, _, is_mut) => {
                if *is_mut {
                    write!(f, "*mut {}", inner)
                } else {
                    write!(f, "*const {}", inner)
                }
            }
            Type::Borrow(inner, _, is_mut, _) => {
                if *is_mut {
                    write!(f, "&mut {}", inner)
                } else {
                    write!(f, "&{}", inner)
                }
            }
            Type::GenericInstance(base, args) => {
                write!(f, "{}<", base)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ">")
            }
            Type::Generic(name, _) => write!(f, "{}", name),
            _ => write!(f, "{:?}", self), // Fallback for complex types
        }
    }
}
