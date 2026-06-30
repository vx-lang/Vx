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
use crate::symbol::Symbol;
use crate::syntax;
#[derive(Debug, PartialEq, Eq, Clone, Copy, Default, Hash)]
pub struct Span {
    pub line: usize,
    pub column: usize,
    pub length: usize,
}

#[allow(non_camel_case_types)]
#[derive(Debug, PartialEq, Clone)]
pub enum Topology {
    CPU,
    NPU(Box<Expr>),
    AccCore(Box<Expr>),
    AMX,
    ANE,
    GPU,
    CpuAvx512,
    CpuNeon,
    Slice(Box<Topology>, Box<Expr>, Box<Expr>), // For NPU[0..4] etc.
    Current,
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum MemorySpace {
    CPUDRAM,
    NPUHBM,
    GpuHbm,
    LocalSRAM,
    NicRam,
    RemoteHbm,
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum TopologyKind {
    CPU,
    NPU,
    AccCore,
    AMX,
    ANE,
    GPU,
    CpuAvx512,
    CpuNeon,
    Slice,
    Current,
}

impl Topology {
    pub fn kind(&self) -> TopologyKind {
        match self {
            Topology::CPU => TopologyKind::CPU,
            Topology::NPU(_) => TopologyKind::NPU,
            Topology::AccCore(_) => TopologyKind::AccCore,
            Topology::AMX => TopologyKind::AMX,
            Topology::ANE => TopologyKind::ANE,
            Topology::GPU => TopologyKind::GPU,
            Topology::CpuAvx512 => TopologyKind::CpuAvx512,
            Topology::CpuNeon => TopologyKind::CpuNeon,
            Topology::Slice(..) => TopologyKind::Slice,
            Topology::Current => TopologyKind::Current,
        }
    }

    pub fn is_same_kind(&self, other: &Self) -> bool {
        self.kind() == other.kind()
    }
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
    Generic(Symbol),
}

#[derive(Debug, PartialEq, Clone)]
pub enum Type {
    Tensor(ElementType, Vec<Expr>, Option<Topology>),
    Matrix,
    Ref(Box<Type>, MemorySpace),
    Borrow {
        inner: Box<Type>,
        mem_space: Option<MemorySpace>,
        is_mut: bool,
        region_id: usize,
    }, // (type, mem_space, is_mut, region_id)
    Pointer(Box<Type>, Option<MemorySpace>, bool), // (type, mem_space, is_mut)
    Scalar(ElementType),
    Struct(Symbol, Option<crate::gid::TypeId>),
    Enum(Symbol, Option<crate::gid::TypeId>),
    Verified(Box<Type>),
    Pinned(Box<Type>, Topology),
    Generic(Symbol, Option<crate::gid::TypeId>), // e.g. T
    GenericInstance(Box<Type>, Vec<Type>),       // e.g. Config<f32>
    Module(Symbol, std::collections::HashMap<Symbol, Type>), // (path, exported_symbols)
    Simd(ElementType, usize),                    // e.g. <4 x f32>
    Function(Vec<Type>, Box<Type>),              // e.g. fn(i32, f32) -> f32
    Closure(Vec<Type>, Box<Type>),               // Fat pointer closure type
    Const(Box<syntax::expr::Expr>),              // E.g., generic const argument like `10`
    Unknown,
}

pub trait Mangle {
    fn mangle_to(&self, w: &mut dyn std::fmt::Write) -> std::fmt::Result;
    fn mangle(&self) -> String {
        let mut s = String::new();
        let _ = self.mangle_to(&mut s);
        s
    }
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

    pub fn substitute(&self, mapping: &std::collections::HashMap<Symbol, Type>) -> Type {
        match self {
            Type::Generic(name, _) => {
                if let Some(concrete) = mapping.get(name) {
                    concrete.clone()
                } else {
                    self.clone()
                }
            }
            Type::Const(expr) => Type::Const(Box::new(expr.substitute(mapping))),
            Type::GenericInstance(base, args) => {
                let new_base = base.substitute(mapping);
                let new_args = args.iter().map(|a| a.substitute(mapping)).collect();
                Type::GenericInstance(Box::new(new_base), new_args)
            }
            Type::Borrow {
                inner,
                mem_space,
                is_mut,
                region_id,
            } => Type::Borrow {
                inner: Box::new(inner.substitute(mapping)),
                mem_space: mem_space.clone(),
                is_mut: *is_mut,
                region_id: *region_id,
            },
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
            Type::Function(params, ret) => {
                let new_params = params.iter().map(|p| p.substitute(mapping)).collect();
                let new_ret = Box::new(ret.substitute(mapping));
                Type::Function(new_params, new_ret)
            }
            Type::Closure(params, ret) => {
                let new_params = params.iter().map(|p| p.substitute(mapping)).collect();
                let new_ret = Box::new(ret.substitute(mapping));
                Type::Closure(new_params, new_ret)
            }
            Type::Unknown => Type::Unknown,
            _ => self.clone(),
        }
    }

    // Mangle trait is implemented below

    pub fn topology(&self) -> Option<Topology> {
        match self {
            Type::Tensor(_, _, top) => top.clone(),
            _ => None,
        }
    }
}

impl ElementType {}

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
            "bool" | "Bool" => Ok(ElementType::Bool),
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
            Type::Borrow { inner, is_mut, .. } => {
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
            Type::Const(expr) => {
                if let syntax::expr::Expr::Number(n) = &**expr {
                    write!(f, "{}", n.value)
                } else if let syntax::expr::Expr::StringLiteral(s) = &**expr {
                    write!(f, "\"{}\"", s.value)
                } else {
                    write!(f, "{{{:?}}}", expr)
                }
            }
            Type::Unknown => write!(f, "?"),
            Type::Function(params, ret) => {
                write!(f, "fn(")?;
                for (i, p) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", p)?;
                }
                write!(f, ") -> {}", ret)
            }
            Type::Closure(params, ret) => {
                write!(f, "|")?;
                for (i, p) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", p)?;
                }
                write!(f, "| -> {}", ret)
            }
            _ => write!(f, "{:?}", self), // Fallback for complex types
        }
    }
}

impl Mangle for Type {
    fn mangle_to(&self, w: &mut dyn std::fmt::Write) -> std::fmt::Result {
        match self {
            Type::Scalar(el) => el.mangle_to(w),
            Type::Simd(el, n) => {
                write!(w, "Simd$")?;
                el.mangle_to(w)?;
                write!(w, "${}", n)
            }
            Type::Pointer(inner, _, is_mut) => {
                write!(w, "ptr${}$", if *is_mut { "mut" } else { "const" })?;
                inner.mangle_to(w)
            }
            Type::Borrow { inner, is_mut, .. } => {
                write!(w, "ref${}$", if *is_mut { "mut" } else { "const" })?;
                inner.mangle_to(w)
            }
            Type::Ref(inner, _) => {
                write!(w, "ref$")?;
                inner.mangle_to(w)
            }
            Type::Tensor(el, dims, _) => {
                write!(w, "Tensor$")?;
                el.mangle_to(w)?;
                write!(w, "${}", dims.len())
            }
            Type::Matrix => write!(w, "Matrix"),
            Type::Struct(name, _) => write!(w, "{}", name),
            Type::Enum(name, _) => write!(w, "{}", name),
            Type::Generic(name, _) => write!(w, "{}", name),
            Type::GenericInstance(base, args) => {
                base.mangle_to(w)?;
                for arg in args {
                    write!(w, "$")?;
                    arg.mangle_to(w)?;
                }
                Ok(())
            }
            Type::Function(_, _) => write!(w, "fn"),
            Type::Closure(_, _) => write!(w, "closure"),
            Type::Verified(inner) => {
                write!(w, "Verified$")?;
                inner.mangle_to(w)
            }
            Type::Pinned(inner, _) => {
                write!(w, "Pinned$")?;
                inner.mangle_to(w)
            }
            Type::Const(expr) => {
                let debug_str = format!("{:?}", expr);
                let sanitized: String = debug_str
                    .chars()
                    .map(|c| if c.is_alphanumeric() { c } else { '_' })
                    .collect();
                write!(w, "const${}", sanitized)
            }
            Type::Module(name, _) => write!(w, "Module${}", name),
            Type::Unknown => write!(w, "Unknown"),
        }
    }
}

impl Mangle for ElementType {
    fn mangle_to(&self, w: &mut dyn std::fmt::Write) -> std::fmt::Result {
        match self {
            ElementType::Generic(g) => write!(w, "{}", g),
            _ => write!(w, "{}", self),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_type_substitute_generic_simple() {
        let ty = Type::Generic("T".into(), None);
        let mut mapping = HashMap::new();
        mapping.insert("T".into(), Type::Scalar(ElementType::I32));
        let result = ty.substitute(&mapping);
        assert_eq!(result, Type::Scalar(ElementType::I32));
    }

    #[test]
    fn test_type_substitute_no_match_passthrough() {
        let ty = Type::Generic("U".into(), None);
        let mut mapping = HashMap::new();
        mapping.insert("T".into(), Type::Scalar(ElementType::I32));
        let result = ty.substitute(&mapping);
        // U is not in the mapping, so it stays Generic("U")
        assert_eq!(result, Type::Generic("U".into(), None));
    }

    #[test]
    fn test_type_substitute_borrow_nested() {
        let ty = Type::Borrow {
            inner: Box::new(Type::Generic("T".into(), None)),
            mem_space: None,
            is_mut: true,
            region_id: 0,
        };
        let mut mapping = HashMap::new();
        mapping.insert("T".into(), Type::Scalar(ElementType::F32));
        let result = ty.substitute(&mapping);
        assert_eq!(
            result,
            Type::Borrow {
                inner: Box::new(Type::Scalar(ElementType::F32)),
                mem_space: None,
                is_mut: true,
                region_id: 0,
            }
        );
    }

    #[test]
    fn test_type_substitute_tensor_element() {
        let ty = Type::Tensor(ElementType::Generic("T".into()), vec![], None);
        let mut mapping = HashMap::new();
        mapping.insert("T".into(), Type::Scalar(ElementType::F32));
        let result = ty.substitute(&mapping);
        assert_eq!(result, Type::Tensor(ElementType::F32, vec![], None));
    }

    #[test]
    fn test_type_substitute_generic_instance() {
        let ty = Type::GenericInstance(
            Box::new(Type::Struct("Vec".into(), None)),
            vec![Type::Generic("T".into(), None)],
        );
        let mut mapping = HashMap::new();
        mapping.insert("T".into(), Type::Scalar(ElementType::I32));
        let result = ty.substitute(&mapping);
        assert_eq!(
            result,
            Type::GenericInstance(
                Box::new(Type::Struct("Vec".into(), None)),
                vec![Type::Scalar(ElementType::I32)],
            )
        );
    }

    #[test]
    fn test_type_substitute_function() {
        let ty = Type::Function(
            vec![Type::Generic("T".into(), None)],
            Box::new(Type::Generic("T".into(), None)),
        );
        let mut mapping = HashMap::new();
        mapping.insert("T".into(), Type::Scalar(ElementType::F64));
        let result = ty.substitute(&mapping);
        assert_eq!(
            result,
            Type::Function(
                vec![Type::Scalar(ElementType::F64)],
                Box::new(Type::Scalar(ElementType::F64)),
            )
        );
    }

    #[test]
    fn test_type_substitute_unknown_passthrough() {
        let ty = Type::Unknown;
        let mapping = HashMap::new();
        assert_eq!(ty.substitute(&mapping), Type::Unknown);
    }

    #[test]
    fn test_type_is_linear_true_cases() {
        assert!(Type::Tensor(ElementType::F32, vec![], None).is_linear());
        assert!(Type::Matrix.is_linear());
        assert!(Type::Ref(
            Box::new(Type::Scalar(ElementType::I32)),
            MemorySpace::CPUDRAM
        )
        .is_linear());
        assert!(Type::Verified(Box::new(Type::Scalar(ElementType::I32))).is_linear());
        assert!(Type::Pinned(Box::new(Type::Scalar(ElementType::I32)), Topology::CPU).is_linear());
        assert!(Type::Struct("Foo".into(), None).is_linear());
        assert!(Type::Enum("Bar".into(), None).is_linear());
    }

    #[test]
    fn test_type_is_linear_false_cases() {
        assert!(!Type::Scalar(ElementType::I32).is_linear());
        assert!(!Type::Unknown.is_linear());
        assert!(!Type::Function(vec![], Box::new(Type::Unknown)).is_linear());
        assert!(!Type::Generic("T".into(), None).is_linear());
    }

    #[test]
    fn test_type_topology_tensor_with_topology() {
        let ty = Type::Tensor(ElementType::F32, vec![], Some(Topology::GPU));
        assert_eq!(ty.topology(), Some(Topology::GPU));
    }

    #[test]
    fn test_type_topology_tensor_without_topology() {
        let ty = Type::Tensor(ElementType::F32, vec![], None);
        assert_eq!(ty.topology(), None);
    }

    #[test]
    fn test_type_topology_non_tensor_returns_none() {
        assert_eq!(Type::Scalar(ElementType::I32).topology(), None);
        assert_eq!(Type::Struct("X".into(), None).topology(), None);
    }

    #[test]
    fn test_topology_is_same_kind() {
        // CPU, CpuAvx512, CpuNeon are all distinct TopologyKinds
        assert!(Topology::CPU.is_same_kind(&Topology::CPU));
        assert!(!Topology::CPU.is_same_kind(&Topology::CpuAvx512));
        assert!(!Topology::CPU.is_same_kind(&Topology::GPU));
        assert!(Topology::GPU.is_same_kind(&Topology::GPU));
    }

    fn make_npu_expr(idx: &str) -> Box<Expr> {
        Box::new(Expr::Number(super::super::expr::NumberExpr::new(
            idx.to_string(),
            None,
            Span::default(),
        )))
    }

    #[test]
    fn test_topology_kind_all_variants() {
        assert_eq!(Topology::CPU.kind(), TopologyKind::CPU);
        assert_eq!(Topology::NPU(make_npu_expr("0")).kind(), TopologyKind::NPU);
        assert_eq!(
            Topology::AccCore(make_npu_expr("0")).kind(),
            TopologyKind::AccCore
        );
        assert_eq!(Topology::AMX.kind(), TopologyKind::AMX);
        assert_eq!(Topology::ANE.kind(), TopologyKind::ANE);
        assert_eq!(Topology::GPU.kind(), TopologyKind::GPU);
        assert_eq!(Topology::CpuAvx512.kind(), TopologyKind::CpuAvx512);
        assert_eq!(Topology::CpuNeon.kind(), TopologyKind::CpuNeon);
        assert_eq!(Topology::Current.kind(), TopologyKind::Current);
        let slice = Topology::Slice(
            Box::new(Topology::NPU(make_npu_expr("0"))),
            make_npu_expr("0"),
            make_npu_expr("4"),
        );
        assert_eq!(slice.kind(), TopologyKind::Slice);
    }

    #[test]
    fn test_topology_is_same_kind_npu_different_indices() {
        let npu0 = Topology::NPU(make_npu_expr("0"));
        let npu1 = Topology::NPU(make_npu_expr("1"));
        // Same kind (both NPU) even though different indices
        assert!(npu0.is_same_kind(&npu1));
        // But they are not PartialEq-equal
        assert_ne!(npu0, npu1);
    }

    #[test]
    fn test_topology_is_same_kind_acccore_different_indices() {
        let ac0 = Topology::AccCore(make_npu_expr("0"));
        let ac3 = Topology::AccCore(make_npu_expr("3"));
        assert!(ac0.is_same_kind(&ac3));
        assert_ne!(ac0, ac3);
    }

    #[test]
    fn test_type_substitute_preserves_topology() {
        let ty = Type::Tensor(
            ElementType::Generic("T".into()),
            vec![],
            Some(Topology::GPU),
        );
        let mut mapping = HashMap::new();
        mapping.insert("T".into(), Type::Scalar(ElementType::F32));
        let result = ty.substitute(&mapping);
        assert_eq!(
            result,
            Type::Tensor(ElementType::F32, vec![], Some(Topology::GPU))
        );
    }

    #[test]
    fn test_type_substitute_pinned_preserves_topology() {
        let ty = Type::Pinned(Box::new(Type::Generic("T".into(), None)), Topology::ANE);
        let mut mapping = HashMap::new();
        mapping.insert("T".into(), Type::Scalar(ElementType::I32));
        let result = ty.substitute(&mapping);
        assert_eq!(
            result,
            Type::Pinned(Box::new(Type::Scalar(ElementType::I32)), Topology::ANE)
        );
    }

    #[test]
    fn test_mangle_tensor_ignores_topology() {
        let ty_with = Type::Tensor(ElementType::F32, vec![], Some(Topology::GPU));
        let ty_without = Type::Tensor(ElementType::F32, vec![], None);
        // Topology is intentionally not included in mangling
        assert_eq!(ty_with.mangle(), ty_without.mangle());
        assert_eq!(ty_with.mangle(), "Tensor$f32$0");
    }

    #[test]
    fn test_mangle_pinned_type() {
        let ty = Type::Pinned(Box::new(Type::Scalar(ElementType::I32)), Topology::ANE);
        assert_eq!(ty.mangle(), "Pinned$i32");
    }
}
