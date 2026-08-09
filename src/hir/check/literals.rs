//===- check submodule - Vx Compiler ----------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// One expression family's type checks, split out of the former 5000-line `hir/expr.rs` along the
// `check_expr_type_flag` dispatch seam (frontend_refactoring_borrow_checker.md R2, #279). An additional
// `impl TypeChecker` block; moves only, zero logic change. Reaches the shared types and helpers via
// `use super::super::*`, exactly as `expr.rs` uses `use super::*`.
//
//===----------------------------------------------------------------------===//

use super::super::*;
use crate::hir::expr::expected_numeric_elem;
use std::collections::HashMap;

impl<'a> TypeChecker<'a> {
    pub(crate) fn check_enumvariant_expr(&mut self, expr: &mut Expr, consume: bool) -> Type {
        match expr {
            Expr::EnumVariant(EnumVariantExpr {
                enum_name,
                variant_name: variant,
                payload,
                span,
            }) => {
                let actual_enum_name = if let Some(idx) = enum_name.find('<') {
                    &enum_name[..idx]
                } else {
                    enum_name.as_ref()
                };

                if let Some(enum_decl) = self.env.enums.get(actual_enum_name) {
                    if let Some((_, expected_payload)) =
                        enum_decl.variants.iter().find(|(n, _)| n == variant)
                    {
                        if let Some(expr_payload) = payload {
                            if let Some(exp_types) = expected_payload {
                                if expr_payload.len() != exp_types.len() {
                                    if !self.speculating {
                                        self.errors.error_with_code(
                                            crate::diagnostic::DiagnosticCode::E3009,
                                            format!("Enum variant {}::{} expects {} payload arguments, got {}", actual_enum_name, variant, exp_types.len(), expr_payload.len()),
                                            Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
                                        );
                                    }
                                } else {
                                    let mut mapping = HashMap::new();
                                    if let Some(idx) = enum_name.find('<') {
                                        let ty_args_str = &enum_name[idx + 1..enum_name.len() - 1];
                                        let ty_args: Vec<&str> = ty_args_str.split(',').collect();
                                        for (i, param) in enum_decl.generics.iter().enumerate() {
                                            if i < ty_args.len() {
                                                let ty_arg = ty_args[i].trim();
                                                let parsed_ty = match ty_arg {
                                                    "i32" => Type::Scalar(ElementType::I32),
                                                    "f32" => Type::Scalar(ElementType::F32),
                                                    "f64" => Type::Scalar(ElementType::F64),
                                                    "i64" => Type::Scalar(ElementType::I64),
                                                    "Bool" => Type::Scalar(ElementType::Bool),
                                                    // A non-scalar type arg is a known struct/enum
                                                    // *or* a generic parameter still in scope (a
                                                    // generic body checked before monomorphization).
                                                    // Parsing `T` as `Struct("T")` made
                                                    // `Option<T>::Some(v: T)` mismatch its own
                                                    // `Generic("T")` payload (E3008 in `VecIter`); a
                                                    // name that isn't a declared nominal is generic.
                                                    // (#242)
                                                    other => {
                                                        let name: crate::symbol::Symbol =
                                                            other.to_string().into();
                                                        if self.env.structs.contains_key(&name)
                                                            || self.env.enums.contains_key(&name)
                                                        {
                                                            Type::Struct(name, None)
                                                        } else {
                                                            Type::Generic(name, None)
                                                        }
                                                    }
                                                };
                                                mapping.insert(param.name().into(), parsed_ty);
                                            }
                                        }
                                    }

                                    for (i, expr) in expr_payload.iter_mut().enumerate() {
                                        let expr_ty = self.check_expr_type_flag(expr, consume);
                                        let expected_ty = exp_types[i].substitute(&mapping);
                                        if !self.is_assignable(&expected_ty, &expr_ty)
                                            && !matches!(&expected_ty, Type::Generic(_, _))
                                            && !self.speculating
                                        {
                                            self.errors.error_with_code(
                                                crate::diagnostic::DiagnosticCode::E3008,
                                                format!("Type mismatch in payload argument {} for {}::{}: expected {:?}, got {:?}", i + 1, actual_enum_name, variant, expected_ty, expr_ty),
                                                Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
                                            );
                                        }
                                    }
                                }
                            } else if !self.speculating {
                                self.errors.error_with_code(
                                    crate::diagnostic::DiagnosticCode::E3009,
                                    format!(
                                        "Enum variant {}::{} does not take a payload",
                                        actual_enum_name, variant
                                    ),
                                    Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
                                );
                            }
                        } else if expected_payload.is_some() && !self.speculating {
                            self.errors.error_with_code(
                                crate::diagnostic::DiagnosticCode::E3009,
                                format!(
                                    "Enum variant {}::{} expects a payload",
                                    actual_enum_name, variant
                                ),
                                Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
                            );
                        }
                    } else if !self.speculating {
                        self.errors.error_with_code(
                            crate::diagnostic::DiagnosticCode::E2004,
                            format!(
                                "Enum {} does not have variant {}",
                                actual_enum_name, variant
                            ),
                            Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
                        );
                    }
                } else if !self.speculating {
                    self.errors.error_with_code(
                        crate::diagnostic::DiagnosticCode::E2003,
                        format!("Unknown enum {}", enum_name),
                        Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
                    );
                }

                if let Some(idx) = enum_name.find('<') {
                    if let Some(end_idx) = enum_name.find('>') {
                        let base = &enum_name[..idx];
                        let ty_arg = &enum_name[idx + 1..end_idx];
                        let parsed_ty = match ty_arg {
                            "i32" => Type::Scalar(ElementType::I32),
                            "f32" => Type::Scalar(ElementType::F32),
                            "f64" => Type::Scalar(ElementType::F64),
                            "i64" => Type::Scalar(ElementType::I64),
                            "Bool" => Type::Scalar(ElementType::Bool),
                            // A non-scalar type arg that isn't a declared nominal is a generic
                            // parameter in scope, not a struct named `T` — so `Option<T>::None`'s
                            // result type is `Option<Generic("T")>`, matching a `-> Option<T>` return
                            // (was `Option<Struct("T")>`, E3002 in `VecIter::next`). (#242)
                            other => {
                                let name: crate::symbol::Symbol = other.to_string().into();
                                if self.env.structs.contains_key(&name)
                                    || self.env.enums.contains_key(&name)
                                {
                                    Type::Struct(name, None)
                                } else {
                                    Type::Generic(name, None)
                                }
                            }
                        };
                        return Type::GenericInstance(
                            Box::new(Type::Struct(base.to_string().into(), None)),
                            vec![parsed_ty],
                        );
                    }
                }
                Type::Enum(enum_name.clone(), None)
            }
            _ => panic!("Expected IndexAccess, got {:?}", expr),
        }
    }

    /// Type an untyped numeric literal (#240). An unsuffixed literal reaches the checker with
    /// `ty: None`; it adopts the *expected* scalar type of its context when that is a compatible
    /// numeric type (integer literal → integer type, float literal → float type), and otherwise
    /// falls back to the spelling-based default. The inferred type is written back onto the node so
    /// every downstream consumer — the flat lowerer and the AST codegen alike — sees the concrete
    /// type. A suffixed literal already carries its type and is returned unchanged.
    pub(crate) fn check_number_literal(&mut self, n: &mut NumberExpr) -> Type {
        if let Some(el) = &n.ty {
            return Type::Scalar(el.clone());
        }
        let elem = self
            .expected_type
            .as_ref()
            .and_then(|t| expected_numeric_elem(t, &n.value))
            .unwrap_or_else(|| crate::parser::expr::default_number_elem(&n.value));
        n.ty = Some(elem.clone());
        Type::Scalar(elem)
    }

    /// The tensor an operand denotes, seen through the wrappers that do not
    /// change what it is: a borrow, a pointer, a placement, a reference.
    ///
    /// `&a @ &b` is the non-consuming spelling of `a @ b` -- both are matmuls
    /// over the same two tensors, and one reads its operands where the other
    /// moves them (#335). The wrappers are peeled in a loop rather than once,
    /// so a borrow of a placed tensor (`&Ref<Tensor<f32>, Memory::GPU_HBM>`,
    /// which is what a resident weight is) resolves too.
    pub(crate) fn as_tensor_operand(
        t: &Type,
    ) -> Option<(&ElementType, &Vec<crate::syntax::Expr>, &Option<Topology>)> {
        let mut inner = t;
        loop {
            match inner {
                Type::Tensor(elem, dims, top) => return Some((elem, dims, top)),
                Type::Borrow { inner: i, .. }
                | Type::Pointer(i, _, _)
                | Type::Pinned(i, _)
                | Type::Ref(i, _) => inner = i.as_ref(),
                _ => return None,
            }
        }
    }

    /// A rank-1 (or, permissively, any-rank) f32 tensor slice, as produced by `q[i]` (S1).
    pub(crate) fn is_f32_slice(t: &Type) -> bool {
        let inner = match t {
            Type::Borrow { inner, .. }
            | Type::Pointer(inner, _, _)
            | Type::Pinned(inner, _)
            | Type::Ref(inner, _) => inner.as_ref(),
            other => other,
        };
        matches!(inner, Type::Tensor(ElementType::F32, _, _))
    }

    pub(crate) fn check_array_expr(&mut self, expr: &mut Expr) -> Type {
        match expr {
            Expr::Array(ArrayExpr { elements, span: _ }) => {
                // The array's element type is its first element's — an integer array literal
                // (`[10, 20, 30]`) is `Tensor<i32>`, not `Tensor<f32>` (#240). Later elements are
                // checked expecting that type, so untyped literals adopt it. An empty literal keeps
                // the historical `f32` default.
                let mut elem_ty = ElementType::F32;
                for (i, el) in elements.iter_mut().enumerate() {
                    if i == 0 {
                        if let Type::Scalar(e) = self.check_expr_type(el) {
                            elem_ty = e;
                        }
                    } else {
                        self.check_expr_expecting(el, Some(Type::Scalar(elem_ty.clone())), true);
                    }
                }
                Type::Tensor(elem_ty, vec![], None)
            }
            _ => panic!("Expected IndexAccess, got {:?}", expr),
        }
    }

    pub(crate) fn check_structinit_expr(&mut self, expr: &mut Expr, consume: bool) -> Type {
        match expr {
            Expr::StructInit(StructInitExpr {
                name,
                fields,
                type_id,
                span: _,
            }) => {
                let resolved_name = name.clone();
                let mut base_name = resolved_name.clone();
                let mut generic_args = Vec::new();

                if let Type::GenericInstance(inner, args) = self.parse_ty_str(&resolved_name) {
                    if let Type::Struct(s, _) = *inner {
                        base_name = s;
                    }
                    generic_args = args;
                }

                // Attach the struct's resolved GID so downstream consumers (the flat-HIR lowerer)
                // reach its registry layout without re-resolving the name (#199). Resolved through the
                // registry-backed `ModuleInterface` -- the AST-free import oracle -- rather than a
                // borrowed-AST side map (#219). A generic instance's monomorph has a distinct identity,
                // so its GID is left `None` here; so is the empty-registry path (the sequential driver /
                // legacy AST harnesses), where only the AST codegen runs and never reads this field.
                let resolved_gid = if generic_args.is_empty() {
                    let mi: &dyn crate::registry::ModuleInterface = &*self.worker.global.registry;
                    mi.resolve_unique_nominal(&base_name)
                } else {
                    None
                };
                *type_id = resolved_gid;

                // The struct's generic parameter names + declared field types: a local AST struct (or
                // a generated one) first, else an *imported* struct from the frozen registry's
                // `structs` table (#219) — so constructing an imported struct type-checks with no AST.
                // A local definition shadows a same-named import.
                #[allow(clippy::type_complexity)]
                let struct_info: Option<(
                    Vec<String>,
                    Vec<(crate::symbol::Symbol, Type)>,
                )> = self
                    .env
                    .structs
                    .get(base_name.as_ref())
                    .map(|s| {
                        (
                            s.generics.iter().map(|g| g.name().to_string()).collect(),
                            s.fields.clone(),
                        )
                    })
                    .or_else(|| {
                        self.generated_structs
                            .iter()
                            .find(|s| s.name == base_name)
                            .map(|s| {
                                (
                                    s.generics.iter().map(|g| g.name().to_string()).collect(),
                                    s.fields.clone(),
                                )
                            })
                    })
                    .or_else(|| {
                        // GID-keyed since #291: resolve the base name to its unique GID (the
                        // instance-annotation `resolved_gid` when it was computed, else a fresh
                        // bare-name resolution for a generic instance's base). An ambiguous name
                        // declines to "unknown struct" rather than picking an arbitrary module's.
                        let reg = &self.worker.global.registry;
                        let gid =
                            resolved_gid.or_else(|| reg.resolve_unique_nominal(&base_name))?;
                        reg.structs.get(&gid).map(|sf| {
                            (
                                sf.generics.iter().map(|g| g.to_string()).collect(),
                                sf.fields.clone(),
                            )
                        })
                    });

                if let Some((generic_names, struct_fields)) = struct_info {
                    let mut mapping = std::collections::HashMap::new();
                    for (i, param) in generic_names.iter().enumerate() {
                        if i < generic_args.len() {
                            mapping.insert(param.as_str().into(), generic_args[i].clone());
                        }
                    }

                    // Check missing fields and type mismatch
                    for (expected_name, raw_expected_type) in &struct_fields {
                        let expected_type = &raw_expected_type.substitute(&mapping);
                        let mut found = false;
                        for (f_name, f_expr) in fields.iter_mut() {
                            if f_name == expected_name {
                                found = true;
                                let f_type = self.check_expr_type_flag(f_expr, consume);
                                if !self.is_assignable(expected_type, &f_type) && !self.speculating
                                {
                                    self.errors.push(format!(
                                            "Type mismatch in struct initialization for field '{}'. Expected {:?}, got {:?}",
                                            expected_name, expected_type, f_type
                                        ));
                                }
                                break;
                            }
                        }
                        if !found && !self.speculating {
                            self.errors.push(format!(
                                "Missing field '{}' in initialization of struct '{}'",
                                expected_name, resolved_name
                            ));
                        }
                    }
                    // Check extra fields
                    for (f_name, f_expr) in fields.iter_mut() {
                        if !struct_fields.iter().any(|(n, _)| n == f_name) {
                            if !self.speculating {
                                self.errors.push(format!(
                                    "Struct '{}' has no field '{}'",
                                    resolved_name, f_name
                                ));
                            }
                            self.check_expr_type_flag(f_expr, consume); // evaluate to find errors
                        }
                    }
                } else {
                    if !self.speculating {
                        // Distinguish "no such struct" from "two imported modules both define it":
                        // the latter used to resolve by coin flip before the bare name learned to
                        // decline, and "unknown" would send the user hunting the wrong bug (#291).
                        if self.worker.global.registry.is_ambiguous_nominal(&base_name) {
                            self.errors.push(crate::registry::ambiguous_import_message(
                                "Struct",
                                &resolved_name,
                            ));
                        } else {
                            self.errors
                                .push(format!("Unknown struct {} (expr.rs:2175)", resolved_name));
                        }
                    }
                    for (_, f_expr) in fields.iter_mut() {
                        self.check_expr_type_flag(f_expr, consume);
                    }
                }

                if !generic_args.is_empty() {
                    Type::GenericInstance(Box::new(Type::Struct(base_name, None)), generic_args)
                } else {
                    // The *returned* type keeps GID `None` (as before) so struct type-identity
                    // comparisons are unchanged; the resolved GID rides on the `type_id` field for
                    // the flat-HIR lowerer to consume (#199).
                    Type::Struct(resolved_name, None)
                }
            }
            _ => panic!("Expected IndexAccess, got {:?}", expr),
        }
    }

    pub(crate) fn check_vecmacro_expr(&mut self, expr: &mut Expr) -> Type {
        match expr {
            Expr::VecMacro(VecMacroExpr { elements, span }) => {
                let mut element_type = Type::Scalar(ElementType::I32); // Default
                if !elements.is_empty() {
                    let mut first = elements[0].clone();
                    element_type = self.check_expr_type(&mut first);
                }

                let var_name = format!("_vec_macro_tmp_{}", self.next_id);
                self.next_id += 1;

                let new_call = Expr::FunctionCall(FunctionCallExpr::new(
                    format!("Vec<{}>::new", element_type).into(),
                    None,
                    vec![],
                    *span,
                ));

                let decl = Statement::LetDecl(LetDeclStmt::new(
                    var_name.clone(),
                    true,
                    None,
                    new_call,
                    *span,
                ));

                let mut stmts = vec![decl];

                for el in elements.clone() {
                    let push_call = Expr::MethodCall(MethodCallExpr::new(
                        Box::new(Expr::Borrow(BorrowExpr {
                            expr: Box::new(Expr::Identifier(IdentifierExpr::new(
                                var_name.clone().into(),
                                Span::default(),
                            ))),
                            is_mut: true,
                            span: Span::default(),
                        })),
                        "push".to_string().into(),
                        None,
                        vec![el],
                        *span,
                    ));
                    stmts.push(Statement::ExprStmt(ExprStmtStmt::new(
                        push_call, true, *span,
                    )));
                }

                let ret_expr =
                    Expr::Identifier(IdentifierExpr::new(var_name.clone().into(), *span));

                let block =
                    Expr::UnsafeBlock(UnsafeBlockExpr::new(stmts, Some(Box::new(ret_expr)), *span));

                *expr = block;
                self.check_expr_type(expr)
            }
            _ => panic!("Expected IndexAccess, got {:?}", expr),
        }
    }
}
