//===- identity.rs - Vx Compiler ------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//! Source-independent keys shared by aggregate layouts and function specializations.

use super::types::Mangle;
use super::{Expr, Type};

/// Tagged nodes and length-delimited atoms keep nested keys unambiguous without escaping child keys.
#[derive(Default)]
pub(crate) struct SemanticKey(String);

impl SemanticKey {
    pub(crate) fn begin(&mut self, tag: &str) {
        self.0.push_str(tag);
        self.0.push('(');
    }
    pub(crate) fn end(&mut self) {
        self.0.push(')');
    }
    pub(crate) fn atom(&mut self, tag: &str, value: impl std::fmt::Display) {
        use std::fmt::Write;
        let value = value.to_string();
        write!(self.0, "{tag}{}:", value.len()).expect("writing to a String cannot fail");
        self.0.push_str(&value);
    }
    pub(crate) fn finish(self) -> String {
        self.0
    }
}

pub(crate) fn const_key(expr: &Expr) -> Option<String> {
    let mut key = SemanticKey::default();
    write_const_key(expr, &mut key)?;
    Some(key.finish())
}

pub(crate) fn write_const_key(expr: &Expr, key: &mut SemanticKey) -> Option<()> {
    if let Some(value) = crate::syntax::types::const_value_text(expr) {
        key.atom("number", value);
        return Some(());
    }
    match expr {
        Expr::Identifier(id) => key.atom("name", &id.name),
        Expr::StringLiteral(value) => key.atom("string", &value.value),
        Expr::BinaryOp(expr) => {
            key.begin("binary");
            key.atom("op", format_args!("{:?}", expr.op));
            write_const_key(&expr.lhs, key)?;
            write_const_key(&expr.rhs, key)?;
            key.end();
        }
        Expr::UnaryOp(expr) => {
            key.begin("unary");
            key.atom("op", format_args!("{:?}", expr.op));
            write_const_key(&expr.expr, key)?;
            key.end();
        }
        Expr::RelationalOp(expr) => {
            key.begin("compare");
            key.atom("op", format_args!("{:?}", expr.op));
            write_const_key(&expr.lhs, key)?;
            write_const_key(&expr.rhs, key)?;
            key.end();
        }
        Expr::LogicalOp(expr) => {
            key.begin("logical");
            key.atom("op", format_args!("{:?}", expr.op));
            write_const_key(&expr.lhs, key)?;
            write_const_key(&expr.rhs, key)?;
            key.end();
        }
        Expr::MemberAccess(expr) => {
            key.begin("member");
            write_const_key(&expr.base, key)?;
            key.atom("field", &expr.member);
            key.end();
        }
        Expr::IndexAccess(expr) => {
            key.begin("index");
            write_const_key(&expr.base, key)?;
            write_const_key(&expr.index, key)?;
            key.end();
        }
        _ => return None,
    }
    Some(())
}

pub(crate) fn topology_key(topology: &crate::syntax::Topology) -> Option<String> {
    let mut key = SemanticKey::default();
    write_topology_key(topology, &mut key)?;
    Some(key.finish())
}

pub(crate) fn write_topology_key(
    topology: &crate::syntax::Topology,
    key: &mut SemanticKey,
) -> Option<()> {
    use crate::syntax::Topology;
    key.begin("topology");
    key.atom("kind", format_args!("{:?}", topology.kind()));
    match topology {
        Topology::GPU(index) | Topology::NPU(index) | Topology::AccCore(index) => {
            key.atom("index", const_key(index)?)
        }
        Topology::Slice(base, start, end) => {
            write_topology_key(base, key)?;
            write_const_key(start, key)?;
            write_const_key(end, key)?;
        }
        _ => {}
    }
    key.end();
    Some(())
}

pub(crate) fn placement_key(placement: &Option<crate::syntax::Placement>) -> Option<String> {
    let mut key = SemanticKey::default();
    write_placement_key(placement, &mut key)?;
    Some(key.finish())
}

pub(crate) fn write_placement_key(
    placement: &Option<crate::syntax::Placement>,
    key: &mut SemanticKey,
) -> Option<()> {
    key.begin("placement");
    if let Some(p) = placement {
        write_topology_key(&p.topology, key)?;
        key.atom("space", format_args!("{:?}", p.space));
    }
    key.end();
    Some(())
}

/// Encode every byte, so punctuation and user names cannot collapse to the same symbol.
pub(crate) fn symbol_key(key: &str) -> String {
    use std::fmt::Write;
    let mut result = String::with_capacity(key.len() * 2);
    for byte in key.bytes() {
        write!(result, "{byte:02x}").expect("writing to a String cannot fail");
    }
    result
}

/// Keep existing type names and append locations with their positions in the type tree.
/// Local borrow regions and source spelling do not affect a specialization's identity.
pub(crate) fn specialization_type_name(ty: &Type) -> Option<String> {
    fn visit(ty: &Type, path: &str, entries: &mut Vec<(String, String)>) -> Option<()> {
        let mut record = |key: String| entries.push((path.to_owned(), key));
        match ty {
            Type::Tensor(_, _, Some(p)) | Type::Pointer(_, Some(p), _) => {
                record(placement_key(&Some(p.clone()))?);
            }
            Type::Borrow {
                mem_space: Some(space),
                ..
            }
            | Type::Ref(_, space) => {
                record(format!("memory({space:?})"));
            }
            Type::Pinned(_, topology) => record(topology_key(topology)?),
            _ => {}
        }
        match ty {
            Type::Pointer(inner, _, _)
            | Type::Borrow { inner, .. }
            | Type::Ref(inner, _)
            | Type::Pinned(inner, _)
            | Type::Verified(inner) => {
                visit(inner, &format!("{path}/inner"), entries)?;
            }
            Type::GenericInstance(base, args) => {
                visit(base, &format!("{path}/base"), entries)?;
                for (i, arg) in args.iter().enumerate() {
                    visit(arg, &format!("{path}/arg{i}"), entries)?;
                }
            }
            Type::Function(params, result, _) | Type::Closure(params, result) => {
                for (i, param) in params.iter().enumerate() {
                    visit(param, &format!("{path}/param{i}"), entries)?;
                }
                visit(result, &format!("{path}/result"), entries)?;
            }
            _ => {}
        }
        Some(())
    }
    let mut entries = Vec::new();
    visit(ty, "", &mut entries)?;
    let mut name = ty.mangle();
    if !entries.is_empty() {
        name.push_str("$locations$");
        name.push_str(&symbol_key(&format!("{entries:?}")));
    }
    Some(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::syntax::{BinaryOp, BinaryOpExpr, IdentifierExpr, Span, UnaryOp, UnaryOpExpr};

    #[rstest::rstest]
    #[case::unary(false)]
    #[case::binary(true)]
    fn nested_expression_keys_have_linear_size_and_distinct_leaves(
        #[case] binary: bool,
        #[values(8, 20, 64)] depth: usize,
    ) {
        let nested = |name: &str| {
            let mut expr = Expr::Identifier(IdentifierExpr::new(name.into(), Span::default()));
            for _ in 0..depth {
                expr = if binary {
                    Expr::BinaryOp(BinaryOpExpr::new(
                        Box::new(expr),
                        BinaryOp::Add,
                        Box::new(Expr::Identifier(IdentifierExpr::new(
                            "tail".into(),
                            Span::default(),
                        ))),
                        Span::default(),
                    ))
                } else {
                    Expr::UnaryOp(UnaryOpExpr::new(
                        UnaryOp::Neg,
                        Box::new(expr),
                        Span::default(),
                    ))
                };
            }
            const_key(&expr).unwrap()
        };
        let first = nested("first");
        let second = nested("second");
        assert!(
            first.len() <= 128 * (depth + 1),
            "nested expression key size must be linear: {} bytes at depth {depth}",
            first.len()
        );
        assert_ne!(
            first, second,
            "nested expression keys must retain the leaf name"
        );
    }

    #[test]
    fn semantic_key_atoms_preserve_boundaries_and_punctuation() {
        let encode = |parts: &[&str]| {
            let mut key = SemanticKey::default();
            key.begin("parts");
            for part in parts {
                key.atom("part", part);
            }
            key.end();
            key.finish()
        };
        assert_ne!(encode(&["a)\"\\", "b"]), encode(&["a)\"\\part1:b"]),
            "atom lengths must preserve punctuation and distinguish embedded tags from separate atoms");
    }
}
