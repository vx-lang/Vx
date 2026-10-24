//===- identity.rs - Vx Compiler ------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//! Generated layouts use declaration IDs for nominal types and erase function-local borrow regions.
//! Placement, access permissions, and reference kinds remain part of their identity.

use crate::gid::TypeId;
use crate::symbol::Symbol;
use crate::syntax::identity::{
    write_const_key, write_placement_key, write_topology_key, SemanticKey,
};
use crate::syntax::{Dim, Expr, NumberExpr, Type};
use rustc_hash::FxHashMap;

/// The declaration named by an aggregate use.
pub enum AggregateBase<'a> {
    Name(&'a str),
    Type(&'a Type),
    Declaration(TypeId),
}
impl<'a> From<&'a str> for AggregateBase<'a> {
    fn from(name: &'a str) -> Self {
        Self::Name(name)
    }
}
impl<'a> From<&'a Type> for AggregateBase<'a> {
    fn from(ty: &'a Type) -> Self {
        Self::Type(ty)
    }
}
impl<'a> From<TypeId> for AggregateBase<'a> {
    fn from(id: TypeId) -> Self {
        Self::Declaration(id)
    }
}

/// Declaration identity for generated layouts, shared across a compilation's workers.
#[derive(Debug, Clone, Default)]
pub struct AggregateIdentities {
    names: FxHashMap<Symbol, Option<TypeId>>,
    declarations: FxHashMap<TypeId, Symbol>,
}

impl AggregateIdentities {
    pub fn new(declarations: impl IntoIterator<Item = (TypeId, Symbol)>) -> Self {
        let mut result = Self::default();
        for (id, name) in declarations {
            result
                .names
                .entry(name.clone())
                .and_modify(|old| {
                    if *old != Some(id) {
                        *old = None;
                    }
                })
                .or_insert(Some(id));
            result
                .declarations
                .entry(id)
                .and_modify(|old| {
                    if name < *old {
                        *old = name.clone();
                    }
                })
                .or_insert(name);
        }
        result
    }

    pub fn from_layouts(layouts: &FxHashMap<TypeId, crate::registry::TypeDefinition>) -> Self {
        Self::new(
            layouts
                .iter()
                .map(|(id, def)| (*id, def.name.as_str().into())),
        )
    }

    /// An attached ID is authoritative. A missing ID requires an unambiguous name.
    pub fn resolve(&self, name: &Symbol, attached: Option<TypeId>) -> Option<(Symbol, TypeId)> {
        let id = match attached {
            Some(id) => id,
            None => self.names.get(name).copied().flatten()?,
        };
        Some((self.declarations.get(&id)?.clone(), id))
    }

    pub fn resolve_base(&self, base: AggregateBase<'_>) -> Option<(Symbol, TypeId)> {
        match base {
            AggregateBase::Name(name) => self.resolve(&name.into(), None),
            AggregateBase::Type(Type::Struct(name, id) | Type::Enum(name, id)) => {
                self.resolve(name, *id)
            }
            AggregateBase::Declaration(id) => Some((self.declarations.get(&id)?.clone(), id)),
            AggregateBase::Type(_) => None,
        }
    }

    pub fn len(&self) -> usize {
        self.declarations.len()
    }
    pub fn is_empty(&self) -> bool {
        self.declarations.is_empty()
    }

    /// The same semantic type key serves generated IDs and layout agreement checks.
    pub fn type_key(&self, ty: &Type) -> Option<String> {
        aggregate_type_key(ty, self)
    }

    /// Canonical metadata retains placement and access permissions but erases local borrow regions.
    pub fn normalize_type(&self, ty: &mut Type) -> Option<()> {
        fn normalize_constant(expr: &mut Expr) {
            use crate::syntax::Span;
            if let Some(value) = crate::syntax::types::const_value_text(expr) {
                *expr = Expr::Number(NumberExpr::new(value, None, Span::default()));
                return;
            }
            match expr {
                Expr::Identifier(expr) => expr.span = Span::default(),
                Expr::StringLiteral(expr) => expr.span = Span::default(),
                Expr::BinaryOp(expr) => {
                    normalize_constant(&mut expr.lhs);
                    normalize_constant(&mut expr.rhs);
                    expr.span = Span::default();
                    expr.operand_ty = None;
                }
                Expr::RelationalOp(expr) => {
                    normalize_constant(&mut expr.lhs);
                    normalize_constant(&mut expr.rhs);
                    expr.span = Span::default();
                    expr.operand_ty = None;
                }
                Expr::LogicalOp(expr) => {
                    normalize_constant(&mut expr.lhs);
                    normalize_constant(&mut expr.rhs);
                    expr.span = Span::default();
                }
                Expr::UnaryOp(expr) => {
                    normalize_constant(&mut expr.expr);
                    expr.span = Span::default();
                }
                Expr::MemberAccess(expr) => {
                    normalize_constant(&mut expr.base);
                    expr.span = Span::default();
                    expr.struct_name = None;
                }
                Expr::IndexAccess(expr) => {
                    normalize_constant(&mut expr.base);
                    normalize_constant(&mut expr.index);
                    expr.span = Span::default();
                }
                _ => {}
            }
        }

        fn normalize_topology(topology: &mut crate::syntax::Topology) {
            use crate::syntax::Topology;
            match topology {
                Topology::GPU(index) | Topology::NPU(index) | Topology::AccCore(index) => {
                    normalize_constant(index)
                }
                Topology::Slice(base, start, end) => {
                    normalize_topology(base);
                    normalize_constant(start);
                    normalize_constant(end);
                }
                _ => {}
            }
        }

        fn normalize_placement(placement: &mut Option<crate::syntax::Placement>) {
            if let Some(placement) = placement {
                normalize_topology(&mut placement.topology);
            }
        }
        if matches!(ty, Type::Struct(_, None)) && crate::syntax::is_void_ty(ty) {
            return Some(());
        }
        match ty {
            Type::Struct(name, id) | Type::Enum(name, id) => {
                let (name, id) = self.resolve(name, *id)?;
                // Struct is the common nominal form in retained layout metadata.
                *ty = Type::Struct(name, Some(id));
            }
            Type::Borrow {
                inner, region_id, ..
            } => {
                *region_id = 0;
                self.normalize_type(inner)?;
            }
            Type::Ref(inner, ..) | Type::Verified(inner) => {
                self.normalize_type(inner)?;
            }
            Type::Pointer(inner, placement, _) => {
                self.normalize_type(inner)?;
                normalize_placement(placement);
            }
            Type::Pinned(inner, topology) => {
                self.normalize_type(inner)?;
                normalize_topology(topology);
            }
            Type::GenericInstance(base, args) => {
                self.normalize_type(base)?;
                for arg in args {
                    self.normalize_type(arg)?;
                }
            }
            Type::Function(params, result, _) | Type::Closure(params, result) => {
                for param in params {
                    self.normalize_type(param)?;
                }
                self.normalize_type(result)?;
            }
            Type::Const(expr) => normalize_constant(expr),
            Type::Tensor(_, dims, placement) => {
                normalize_placement(placement);
                for dim in dims {
                    if let Dim::Static(expr) = dim {
                        normalize_constant(expr);
                    }
                }
            }
            Type::Module(..) | Type::Unknown => return None,
            _ => {}
        }
        Some(())
    }
}

fn aggregate_type_key(ty: &Type, identities: &AggregateIdentities) -> Option<String> {
    fn write(ty: &Type, identities: &AggregateIdentities, key: &mut SemanticKey) -> Option<()> {
        use crate::syntax::types::Mangle;
        match ty {
            Type::Tensor(elem, dims, placement) => {
                key.begin("tensor");
                key.atom("element", format_args!("{elem:?}"));
                key.begin("dimensions");
                for dim in dims {
                    match dim {
                        Dim::Dyn => key.atom("dynamic", ""),
                        Dim::Static(expr) => write_const_key(expr, key)?,
                    }
                }
                key.end();
                write_placement_key(placement, key)?;
                key.end();
            }
            Type::Pointer(inner, placement, mutable) => {
                key.begin("pointer");
                key.atom("mutable", mutable);
                write(inner, identities, key)?;
                write_placement_key(placement, key)?;
                key.end();
            }
            Type::Borrow {
                inner,
                mem_space,
                is_mut,
                ..
            } => {
                key.begin("borrow");
                key.atom("mutable", is_mut);
                key.atom("memory", format_args!("{mem_space:?}"));
                write(inner, identities, key)?;
                key.end();
            }
            Type::Ref(inner, space) => {
                key.begin("ref");
                key.atom("memory", format_args!("{space:?}"));
                write(inner, identities, key)?;
                key.end();
            }
            Type::Verified(inner) => {
                key.begin("verified");
                write(inner, identities, key)?;
                key.end();
            }
            Type::Pinned(inner, topology) => {
                key.begin("pinned");
                write(inner, identities, key)?;
                write_topology_key(topology, key)?;
                key.end();
            }
            Type::GenericInstance(base, args) => {
                key.begin("instance");
                write(base, identities, key)?;
                key.begin("arguments");
                for arg in args {
                    write(arg, identities, key)?;
                }
                key.end();
                key.end();
            }
            Type::Function(params, result, unsafe_fn) => {
                key.begin("function");
                key.atom("unsafe", unsafe_fn);
                key.begin("parameters");
                for param in params {
                    write(param, identities, key)?;
                }
                key.end();
                write(result, identities, key)?;
                key.end();
            }
            Type::Closure(params, result) => {
                key.begin("closure");
                key.begin("parameters");
                for param in params {
                    write(param, identities, key)?;
                }
                key.end();
                write(result, identities, key)?;
                key.end();
            }
            Type::Const(expr) => {
                key.begin("constant");
                write_const_key(expr, key)?;
                key.end();
            }
            Type::Struct(name, id) | Type::Enum(name, id) => {
                if id.is_none() && crate::syntax::is_void_ty(ty) {
                    key.atom("builtin", ty.mangle());
                } else {
                    let (_, id) = identities.resolve(name, *id)?;
                    key.atom("nominal", format_args!("{id:?}"));
                }
            }
            Type::Module(..) | Type::Unknown => return None,
            _ => key.atom("type", ty.mangle()),
        }
        Some(())
    }
    let mut key = SemanticKey::default();
    write(ty, identities, &mut key)?;
    Some(key.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rstest::rstest]
    #[case::pointers(false)]
    #[case::generic_instances(true)]
    fn nested_type_keys_have_linear_size_and_distinct_leaves(
        #[case] generic: bool,
        #[values(8, 20, 64)] depth: usize,
    ) {
        let declaration = TypeId::new(1, 1, 0, 0);
        let identities = AggregateIdentities::new([(declaration, "Box".into())]);
        let nested = |element| {
            let mut ty = Type::Scalar(element);
            for _ in 0..depth {
                ty = if generic {
                    Type::GenericInstance(
                        Box::new(Type::Struct("Box".into(), Some(declaration))),
                        vec![ty],
                    )
                } else {
                    Type::Pointer(Box::new(ty), None, false)
                };
            }
            identities.type_key(&ty).unwrap()
        };
        let first = nested(crate::syntax::ElementType::I32);
        let second = nested(crate::syntax::ElementType::I64);
        assert!(
            first.len() <= 256 * (depth + 1),
            "nested type key size must be linear: {} bytes at depth {depth}",
            first.len()
        );
        assert_ne!(first, second, "nested type keys must retain the leaf type");
    }
}
