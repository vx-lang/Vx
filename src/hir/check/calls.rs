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
use crate::hir::expr::{expected_numeric_elem, int_dim_expr};
use std::collections::HashMap;

/// Everything `check_functioncall_expr` computes *before* the argument loop so it can decide,
/// *after* it, which reference-argument reborrows persist past the call (#243). Built by
/// `prepare_reference_arg_reborrows`, consumed by `commit_reference_arg_reborrows`.
struct ReborrowPlan {
    /// Callee `(param types, return type)` if resolvable and not speculating; `None` disables all
    /// reborrow bookkeeping.
    callee_sig: Option<(Vec<Type>, Type)>,
    /// The callee returns a reference (only then can an argument borrow outlive the call).
    ret_is_ref: bool,
    /// The returned reference is mutable (the reborrow's mutability is the result's, not the param's).
    result_is_mut: bool,
    /// Parameter slots the returned reference derives from (in-compilation summary).
    return_prov: crate::hir::provenance::ReturnProvenance,
    /// Same, for an imported callee read from the registry inline code (`None` = use `return_prov`).
    imported_prov: Option<u8>,
    /// (arg index, base, path, param-is-mut, arg-is-a-`&x`-literal) for each reference argument.
    ref_args: Vec<(usize, String, Vec<String>, bool, bool)>,
    /// Pre-call borrow list of each distinct reference-argument base, for the selective revert.
    base_snapshots: HashMap<String, Option<Vec<BorrowRecord>>>,
}

impl<'a> TypeChecker<'a> {
    pub(crate) fn check_indirectcall_expr(&mut self, expr: &mut Expr, consume: bool) -> Type {
        let (callee, args) = match expr {
            Expr::IndirectCall(c) => (&mut c.callee, &mut c.args),
            _ => panic!("Expected IndexAccess, got {:?}", expr),
        };
        let callee_ty = self.check_expr_type_flag(callee, consume);
        let mut arg_types = Vec::new();
        for arg in args.iter_mut() {
            arg_types.push(self.check_expr_type_flag(arg, consume));
        }

        if let Type::Struct(struct_name, _) = callee_ty {
            if struct_name.starts_with("Closure_") {
                let call_name = format!("{}_call", struct_name);
                if let Some(func) = self.monomorphized_functions.iter().find(|f| {
                    f.0.name == <std::string::String as Clone>::clone(&call_name.clone()).into()
                }) {
                    let param_types: Vec<Type> = func
                        .0
                        .params
                        .iter()
                        .skip(1)
                        .map(|(_, t)| t.clone())
                        .collect();
                    if args.len() != param_types.len() {
                        if !self.speculating {
                            self.errors.push(format!(
                                "Closure expects {} arguments, got {}",
                                param_types.len(),
                                args.len()
                            ));
                        }
                    } else {
                        for (i, param_ty) in param_types.iter().enumerate() {
                            let arg_ty = &arg_types[i];
                            if !self.is_assignable(param_ty, arg_ty) && !self.speculating {
                                self.errors.push(format!(
                                    "Type mismatch in argument {} for closure. Expected {:?}, got {:?}",
                                    i + 1, param_ty, arg_ty
                                ));
                            }
                        }
                    }

                    let mut new_args = vec![Expr::Borrow(BorrowExpr {
                        expr: callee.clone(),
                        is_mut: true,
                        span: Span::default(),
                    })];
                    new_args.extend(args.clone());

                    *expr = Expr::FunctionCall(FunctionCallExpr {
                        name: call_name.into(),
                        type_args: None,
                        args: new_args,
                        span: Span::default(),
                    });

                    return func.0.return_type.clone();
                } else {
                    if !self.speculating {
                        self.errors.push(format!(
                            "Missing call method for closure struct '{}'",
                            struct_name
                        ));
                    }
                }
            } else {
                if !self.speculating {
                    self.errors
                        .push(format!("Cannot call non-closure struct '{}'", struct_name));
                }
            }
        } else if let Type::Closure(param_types, ret_ty) = &callee_ty {
            if args.len() != param_types.len() {
                if !self.speculating {
                    self.errors.push(format!(
                        "Closure fat pointer expects {} arguments, got {}",
                        param_types.len(),
                        args.len()
                    ));
                }
            } else {
                for (i, param_ty) in param_types.iter().enumerate() {
                    let arg_ty = &arg_types[i];
                    if !self.is_assignable(param_ty, arg_ty) && !self.speculating {
                        self.errors.push(format!(
                            "Type mismatch in argument {} for closure fat pointer. Expected {:?}, got {:?}",
                            i + 1, param_ty, arg_ty
                        ));
                    }
                }
            }

            if let Expr::IndirectCall(ref mut ic) = expr {
                ic.target_func_ty = Some(callee_ty.clone());
            }

            return *ret_ty.clone();
        } else if let Type::Function(_, _) = callee_ty {
            if !self.speculating {
                self.errors.push(
                    "Function pointers are not natively callable yet; use closure interfaces."
                        .to_string(),
                );
            }
        } else {
            if !self.speculating {
                self.errors
                    .push(format!("Cannot call expression of type {:?}", callee_ty));
            }
        }

        Type::Tensor(ElementType::F32, vec![], None)
    }

    /// Type-check `e` under an explicit expected type, restoring the previous one after. The
    /// mechanism by which an untyped numeric literal in a *checking position* adopts that
    /// position's type (#240): only [`Self::check_number_literal`] consults `expected_type`, so
    /// this affects a bare literal (and literals inside a nested checking position) and nothing
    /// else. The expected type does not leak past `e`.
    pub(crate) fn check_expr_expecting(
        &mut self,
        e: &mut Expr,
        expected: Option<Type>,
        consume: bool,
    ) -> Type {
        let prev = self.expected_type.take();
        self.expected_type = expected;
        let ty = self.check_expr_type_flag(e, consume);
        self.expected_type = prev;
        ty
    }

    /// Refine a call argument's type against its parameter, letting an untyped numeric literal
    /// adopt the parameter's scalar type — the call-argument checking position (#240). Arguments
    /// are type-checked eagerly (before the callee is resolved), so a literal has already taken its
    /// spelling default by the time the parameter type is known; this re-types it in place to the
    /// parameter (when kind-compatible: an integer literal to an integer parameter, a float literal
    /// to a float parameter). A non-literal argument is returned unchanged, so a genuine typed-value
    /// mismatch (`f(some_i64)` into an `i32` parameter) is left for the caller's `is_assignable`
    /// check to reject — the programmer writes an explicit `as`. Returns the argument's type after
    /// refinement.
    pub(crate) fn refine_literal_arg(
        &self,
        arg: &mut Expr,
        param_ty: &Type,
        arg_ty: &Type,
    ) -> Type {
        if let Expr::Number(n) = arg {
            if let Some(elem) = expected_numeric_elem(param_ty, &n.value) {
                n.ty = Some(elem.clone());
                return Type::Scalar(elem);
            }
        }
        arg_ty.clone()
    }

    /// Compute, *before* the argument loop, everything the post-call reborrow revert needs (#243):
    /// the callee signature, whether it returns a (mutable) reference, the parameter slots the
    /// return derives from (in-compilation summary + imported inline code), and — per reference
    /// argument — its base/path plus a snapshot of that base's pre-call borrow list. Consumed by
    /// `commit_reference_arg_reborrows`. Split out of `check_functioncall_expr` (R4, #279).
    fn prepare_reference_arg_reborrows(
        &mut self,
        resolved_name: crate::symbol::Symbol,
        args: &[Expr],
    ) -> ReborrowPlan {
        // Reborrow tracking (#243). Passing a reference to a reference parameter reborrows
        // the underlying storage; whether that reborrow *persists past the call* is decided
        // per argument by the callee's return-provenance summary — only an argument the
        // returned reference actually derives from stays borrowed once the call returns
        // (`pick(&x, &y)` returning from `b` must keep `y` borrowed but release `x`). We
        // snapshot each reference-argument base *before* the arguments are checked, let the
        // arguments record their borrows (so intra-call conflicts like `f(&mut x, &x)` still
        // fire), then revert the non-deriving bases to their pre-call state.
        let callee_sig = if self.speculating {
            None
        } else {
            self.resolve_callee_ref_signature(&resolved_name)
        };
        let return_prov = self.env.return_provenance_of(resolved_name.as_ref());
        // Cross-module refinement (#265 step 7): an *imported* callee has no AST body, so it is
        // absent from the `return_provenances` summary map and `return_provenance_of` falls to
        // the conservative `AnyParam`. Its per-parameter provenance instead travels in the
        // frozen registry's `FnSig.ret_prov` (populated by a `.vxlib` deserialize). Read it
        // only on a genuine summary *miss*, so a local definition's summary always wins; a hit
        // (any in-compilation function, even one that is legitimately `AnyParam`) is untouched.
        let imported_prov: Option<u8> = if self
            .env
            .return_provenances
            .contains_key(resolved_name.as_ref())
        {
            None
        } else {
            self.worker
                .global
                .registry
                .fn_sigs
                .get(&crate::symbol::Symbol::from(resolved_name.as_ref()))
                .map(|sig| sig.ret_prov)
        };
        // (#265) The same summary lowers into the return type's inline slot-0 provenance code,
        // which the cross-module path (step 7, `.vxlib`) will read straight from the `TypeId`
        // with no side table. Intra-compilation the side table above still drives the persist
        // decision; here we populate and *check* the inline form on every real call — proving
        // it round-trips through the `TypeId` API and stays a conservative refinement of the
        // summary it will replace, so wiring the consumer later can never silently read a
        // narrower (unsound) alias set. Debug-only: the release hot path is untouched.
        #[cfg(debug_assertions)]
        {
            let code = crate::hir::provenance::encode_return_provenance(&return_prov);
            let mut sig = crate::gid::TypeId::new(0, 0, 0, 0);
            sig.set_return_provenance(code);
            debug_assert_eq!(
                sig.extract_return_provenance(),
                code,
                "slot-0 provenance code must round-trip through the TypeId encoding"
            );
            debug_assert!(
                (0..32).all(|slot| {
                    !return_prov.includes(slot)
                        || crate::hir::provenance::inline_prov_includes(
                            sig.extract_return_provenance(),
                            slot,
                        )
                }),
                "inline provenance must conservatively refine the summary for '{}'",
                resolved_name.as_ref()
            );
        }
        // A reference argument's borrow can only outlive the call if the callee actually
        // returns a reference. Gating on the return *type* (not just the summary) keeps
        // void/value-returning callees correct even when their summary is absent — e.g. an
        // impl method, which `build` does not summarize (it defaults to the conservative
        // `AnyParam`). Without this, `foo(&mut x)` on a void method would wrongly persist a
        // mutable borrow of `x` and fire a spurious `E4004` at the next use.
        let ret_is_ref = callee_sig
            .as_ref()
            .map(|(_, ret)| Self::is_ref_type(ret))
            .unwrap_or(false);
        // A reborrow's mutability is the *result* reference's, not the parameter's: `found =
        // probe_mut(m)` where `probe_mut(m: &mut Map) -> &i32` yields a *shared* alias of
        // `*m`, so the reborrow is shared. Keying on the return type (rather than the param)
        // is both more correct and what keeps a shared-returning generic from marking its
        // argument mutably borrowed — which would collide with the callee body's own reads
        // of a same-named parameter during instantiation (#268).
        let result_is_mut = callee_sig
            .as_ref()
            .map(|(_, ret)| Self::is_mut_ref(ret))
            .unwrap_or(false);
        // (index, base, path, param-is-mut, arg-is-a-`&x`-literal) for each reference arg,
        // plus a snapshot of each distinct base's pre-call borrow list.
        let mut ref_args: Vec<(usize, String, Vec<String>, bool, bool)> = Vec::new();
        let mut base_snapshots: HashMap<String, Option<Vec<BorrowRecord>>> = HashMap::new();
        if let Some((param_types, _)) = &callee_sig {
            for (i, arg) in args.iter().enumerate() {
                match param_types.get(i) {
                    Some(pty) if Self::is_ref_type(pty) => {
                        if let Some((base, path)) = Self::arg_reborrow_base(arg) {
                            base_snapshots
                                .entry(base.clone())
                                .or_insert_with(|| self.borrow.snapshot_base(base.as_str()));
                            ref_args.push((
                                i,
                                base,
                                path,
                                Self::is_mut_ref(pty),
                                matches!(arg, Expr::Borrow(_)),
                            ));
                        }
                    }
                    _ => {}
                }
            }
        }

        ReborrowPlan {
            callee_sig,
            ret_is_ref,
            result_is_mut,
            return_prov,
            imported_prov,
            ref_args,
            base_snapshots,
        }
    }

    /// After the argument loop, decide which reference-argument reborrows persist past the call
    /// and revert the rest (#243). A reborrow outlives the call iff the callee returns a reference
    /// the argument's slot derives from; every other base is restored to the borrows it had before
    /// the call (releasing the `&x` literals `check_borrow_expr` over-recorded). No-op while
    /// speculating or when the callee signature was unresolved. Split out of
    /// `check_functioncall_expr` (R4, #279).
    fn commit_reference_arg_reborrows(
        &mut self,
        plan: &ReborrowPlan,
        arg_types: &[Type],
        span: &Span,
    ) {
        if self.speculating || plan.callee_sig.is_none() {
            return;
        }
        // An argument's reborrow outlives the call iff the callee returns a reference
        // and its result derives from that argument's parameter slot. For an imported
        // callee the summary comes from the registry's inline code (#265 step 7); for an
        // in-compilation callee, from the AST `return_provenances` summary.
        let persists = |i: usize| {
            plan.ret_is_ref
                && match plan.imported_prov {
                    Some(code) => crate::hir::provenance::inline_prov_includes(code, i),
                    None => plan.return_prov.includes(i),
                }
        };
        // (a) Record reborrows for bare-reference arguments (`foo(m)`); `&x` literals
        //     were already recorded by `check_borrow_expr` during the arg loop. A
        //     non-deriving argument still runs the conflict check but records nothing —
        //     so passing the same reference to two parameters of a non-reference-
        //     returning call (`rmsnorm(x, x, ..)`, an in-place reborrow) does not
        //     self-conflict, matching the pre-#243 behaviour.
        for (i, base, path, is_mut_param, is_borrow_lit) in &plan.ref_args {
            if *is_borrow_lit || !Self::is_ref_type(&arg_types[*i]) {
                continue;
            }
            self.track_reference_arg_borrow(
                base,
                path.clone(),
                *is_mut_param,
                plan.result_is_mut,
                persists(*i),
                span,
            );
        }
        // (b) Selective revert: a base keeps its borrow past the call iff at least one
        //     of its argument positions is one the return derives from. Non-deriving
        //     bases are restored to their pre-call state (call-duration borrow only).
        //     This is what releases the `&x` literals `check_borrow_expr` over-recorded.
        let deriving: std::collections::HashSet<&str> = plan
            .ref_args
            .iter()
            .filter(|(i, _, _, _, _)| persists(*i))
            .map(|(_, base, _, _, _)| base.as_str())
            .collect();
        for (base, snap) in &plan.base_snapshots {
            if deriving.contains(base.as_str()) {
                continue;
            }
            // Drop only the borrows *this call* added; keep exactly the records that
            // were present pre-call and still are. Restoring the raw snapshot instead
            // would resurrect borrows the arg loop legitimately NLL-released, firing
            // spurious conflicts later.
            let prev: &[BorrowRecord] = snap.as_deref().unwrap_or(&[]);
            self.borrow.retain_present(base.as_str(), prev);
        }
    }

    pub(crate) fn check_functioncall_expr(&mut self, expr: &mut Expr, consume: bool) -> Type {
        match expr {
            Expr::FunctionCall(FunctionCallExpr {
                name,
                type_args,
                args,
                span,
            }) => {
                let resolved_name = name.clone();
                let mut base_name = resolved_name.clone();
                let mut explicit_generic_args = type_args.clone().unwrap_or_default();
                if explicit_generic_args.is_empty() {
                    if let Some(start_idx) = resolved_name.find('<') {
                        if let Some(end_idx) = resolved_name.rfind('>') {
                            base_name = format!(
                                "{}{}",
                                &resolved_name[..start_idx],
                                &resolved_name[end_idx + 1..]
                            )
                            .into();
                            let args_str = &resolved_name[start_idx + 1..end_idx];
                            explicit_generic_args = args_str
                                .split(',')
                                .map(|s| self.parse_ty_str(s.trim()))
                                .collect();
                        }
                    }
                } else if let Some(idx) = resolved_name.find('<') {
                    base_name = resolved_name[..idx].to_string().into();
                }

                // Mocking built-ins
                let mut arg_types = Vec::new();
                // Slice reductions (S2) read their operands (a zero-copy view); they do not
                // consume the linear tensor, so the same slice can feed several reductions.
                let is_slice_reduction =
                    matches!(resolved_name.as_ref(), "dot" | "sum" | "max" | "min");
                let is_builtin_ref = resolved_name == "print".into()
                    || resolved_name == "Verified".into()
                    || is_slice_reduction;
                let arg_consume = if is_builtin_ref { false } else { consume };

                // Reference-argument reborrow bookkeeping (#243): snapshot each base before the
                // args are checked so the post-call revert can tell a call-duration borrow from
                // one the return keeps alive.
                let reborrow_plan =
                    self.prepare_reference_arg_reborrows(resolved_name.clone(), args.as_slice());
                for arg in args.iter_mut() {
                    arg_types.push(self.check_expr_type_flag(arg, arg_consume));
                }

                self.commit_reference_arg_reborrows(&reborrow_plan, &arg_types, span);
                if let Some(intrinsic_ty) = self.resolve_intrinsic_function(
                    &resolved_name,
                    args,
                    &arg_types,
                    &explicit_generic_args,
                ) {
                    return intrinsic_ty;
                }

                if let Some((Type::Function(param_types, ret_ty), _)) =
                    self.lookup(&resolved_name).cloned()
                {
                    if args.len() != param_types.len() && !self.speculating {
                        self.errors.push(format!(
                            "Function pointer '{}' expects {} arguments, got {}",
                            resolved_name,
                            param_types.len(),
                            args.len()
                        ));
                    } else {
                        for (i, param_ty) in param_types.iter().enumerate() {
                            let arg_ty = &arg_types[i];
                            if !self.is_assignable(param_ty, arg_ty) && !self.speculating {
                                self.errors.push(format!(
                                        "Type mismatch in argument {} for function pointer '{}'. Expected {:?}, got {:?}",
                                        i + 1, resolved_name, param_ty, arg_ty
                                    ));
                            }
                        }
                    }
                    *ret_ty
                } else if let Some((Type::Closure(param_types, ret_ty), _)) =
                    self.lookup(&resolved_name).cloned()
                {
                    if args.len() != param_types.len() && !self.speculating {
                        self.errors.push(format!(
                            "Closure '{}' expects {} arguments, got {}",
                            resolved_name,
                            param_types.len(),
                            args.len()
                        ));
                    } else {
                        for (i, param_ty) in param_types.iter().enumerate() {
                            let arg_ty = &arg_types[i];
                            if !self.is_assignable(param_ty, arg_ty) && !self.speculating {
                                self.errors.push(format!(
                                        "Type mismatch in argument {} for closure '{}'. Expected {:?}, got {:?}",
                                        i + 1, resolved_name, param_ty, arg_ty
                                    ));
                            }
                        }
                    }
                    *ret_ty
                } else if let Some((Type::Struct(struct_name, _), _)) =
                    self.lookup(&resolved_name).cloned()
                {
                    if struct_name.starts_with("Closure_") {
                        let call_name = format!("{}_call", struct_name);
                        if let Some(func) = self
                            .monomorphized_functions
                            .iter()
                            .find(|f| f.0.name == call_name.as_str().into())
                        {
                            let param_types: Vec<Type> = func
                                .0
                                .params
                                .iter()
                                .skip(1)
                                .map(|(_, t)| t.clone())
                                .collect();
                            if args.len() != param_types.len() {
                                if !self.speculating {
                                    self.errors.push(format!(
                                        "Closure '{}' expects {} arguments, got {}",
                                        resolved_name,
                                        param_types.len(),
                                        args.len()
                                    ));
                                }
                            } else {
                                for (i, param_ty) in param_types.iter().enumerate() {
                                    let arg_ty = &arg_types[i];
                                    if !self.is_assignable(param_ty, arg_ty) && !self.speculating {
                                        self.errors.push(format!(
                                                "Type mismatch in argument {} for closure '{}'. Expected {:?}, got {:?}",
                                                i + 1, resolved_name, param_ty, arg_ty
                                            ));
                                    }
                                }
                            }

                            let mut new_args = vec![Expr::Borrow(BorrowExpr {
                                expr: Box::new(Expr::Identifier(IdentifierExpr::new(
                                    resolved_name.clone(),
                                    Span::default(),
                                ))),
                                is_mut: true,
                                span: Span::default(),
                            })];
                            new_args.extend(args.clone());

                            *expr = Expr::FunctionCall(FunctionCallExpr {
                                name: call_name.into(),
                                type_args: None,
                                args: new_args,
                                span: Span::default(),
                            });

                            func.0.return_type.clone()
                        } else {
                            if !self.speculating {
                                self.errors.push(format!(
                                    "Missing call method for closure struct '{}'",
                                    struct_name
                                ));
                            }
                            Type::Tensor(ElementType::F32, vec![], None)
                        }
                    } else {
                        if !self.speculating {
                            self.errors.push(format!(
                                "Cannot call non-closure struct '{}'",
                                resolved_name
                            ));
                        }
                        Type::Tensor(ElementType::F32, vec![], None)
                    }
                } else if let Some((ret_ty, is_unsafe, param_types, req_topology, _, _)) =
                    self.env.functions.get(resolved_name.as_ref())
                {
                    if (*req_topology != self.active_topology) && !self.speculating {
                        self.errors.error_with_code(
                            crate::diagnostic::DiagnosticCode::E6001,
                            format!(
                                "Type error: Function '{}' requires topology '{:?}', but is called from '{:?}'",
                                resolved_name, req_topology, self.active_topology
                            ),
                            Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
                        );
                    }
                    if *is_unsafe && !self.in_unsafe_block && !self.speculating {
                        self.errors.error_with_code(
                            crate::diagnostic::DiagnosticCode::E5001,
                            format!("Call to unsafe function '{}' is unsafe and requires unsafe function or block", resolved_name),
                            Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
                        );
                    }
                    if args.len() != param_types.len() && !self.speculating {
                        self.errors.error_with_code(
                            crate::diagnostic::DiagnosticCode::E3010,
                            format!(
                                "Function '{}' expects {} arguments, got {}",
                                resolved_name,
                                param_types.len(),
                                args.len()
                            ),
                            Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
                        );
                    } else {
                        for (i, param_ty) in param_types.iter().enumerate() {
                            let arg_ty =
                                self.refine_literal_arg(&mut args[i], param_ty, &arg_types[i]);
                            if !self.is_assignable(param_ty, &arg_ty) && !self.speculating {
                                self.errors.error_with_code(
                                    crate::diagnostic::DiagnosticCode::E3003,
                                    format!(
                                        "Type mismatch in argument {} for function '{}'. Expected {:?}, got {:?}",
                                        i + 1, resolved_name, param_ty, arg_ty
                                    ),
                                    Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
                                );
                            }
                        }
                    }
                    ret_ty.clone()
                } else if let Some(func) = self
                    .monomorphized_functions
                    .iter()
                    .find(|f| f.0.name == resolved_name)
                {
                    if (func.0.topology != self.active_topology) && !self.speculating {
                        self.errors.error_with_code(
                            crate::diagnostic::DiagnosticCode::E6001,
                            format!(
                                "Type error: Function '{}' requires topology '{:?}', but is called from '{:?}'",
                                resolved_name, func.0.topology, self.active_topology
                            ),
                            Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
                        );
                    }
                    let param_types: Vec<Type> =
                        func.0.params.iter().map(|(_, t)| t.clone()).collect();
                    if args.len() != param_types.len() {
                        if !self.speculating {
                            self.errors.error_with_code(
                                crate::diagnostic::DiagnosticCode::E3010,
                                format!(
                                    "Function '{}' expects {} arguments, got {}",
                                    resolved_name,
                                    param_types.len(),
                                    args.len()
                                ),
                                Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
                            );
                        }
                    } else {
                        for (i, param_ty) in param_types.iter().enumerate() {
                            let arg_ty =
                                self.refine_literal_arg(&mut args[i], param_ty, &arg_types[i]);
                            if !self.is_assignable(param_ty, &arg_ty) && !self.speculating {
                                self.errors.error_with_code(
                                    crate::diagnostic::DiagnosticCode::E3003,
                                    format!(
                                        "Type mismatch in argument {} for function '{}'. Expected {:?}, got {:?}",
                                        i + 1, resolved_name, param_ty, arg_ty
                                    ),
                                    Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
                                );
                            }
                        }
                    }
                    func.0.return_type.clone()
                } else if let Some((generic_func, origin_hash)) =
                    self.env.generic_functions.get(base_name.as_ref()).cloned()
                {
                    self.instantiate_generic_function_call(
                        generic_func,
                        origin_hash,
                        &resolved_name,
                        name,
                        args,
                        &arg_types,
                        &explicit_generic_args,
                    )
                    .unwrap_or(Type::Tensor(ElementType::F32, vec![], None))
                } else if resolved_name.contains("::") {
                    self.check_static_method_call(&resolved_name, name)
                } else if let Some(sig) = self
                    .worker
                    .global
                    .registry
                    .fn_sigs
                    .get(&crate::symbol::Symbol::from(resolved_name.as_ref()))
                    .cloned()
                {
                    // Imported callee resolved from a merged `.vxlib` interface (#219 flip, phase 2):
                    // its AST is absent from this compile, so type-check the call against the registry
                    // `FnSig` — arg count + per-argument assignability against `params`, result type is
                    // `ret_ty`. The flat codegen links the body separately via `body_of`. In a normal
                    // compile the registry is empty, so this arm is inert and the call falls to E2002.
                    if args.len() != sig.params.len() && !self.speculating {
                        self.errors.error_with_code(
                            crate::diagnostic::DiagnosticCode::E3010,
                            format!(
                                "Function '{}' expects {} arguments, got {}",
                                resolved_name,
                                sig.params.len(),
                                args.len()
                            ),
                            Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
                        );
                    } else {
                        for (i, param_ty) in sig.params.iter().enumerate() {
                            let arg_ty =
                                self.refine_literal_arg(&mut args[i], param_ty, &arg_types[i]);
                            if !self.is_assignable(param_ty, &arg_ty) && !self.speculating {
                                self.errors.error_with_code(
                                    crate::diagnostic::DiagnosticCode::E3003,
                                    format!(
                                        "Type mismatch in argument {} for function '{}'. Expected {:?}, got {:?}",
                                        i + 1, resolved_name, param_ty, arg_ty
                                    ),
                                    Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
                                );
                            }
                        }
                    }
                    sig.ret_ty.clone()
                } else {
                    let mono_names: Vec<crate::symbol::Symbol> = self
                        .monomorphized_functions
                        .iter()
                        .map(|(f, _)| f.name.clone())
                        .collect();
                    if !self.speculating {
                        self.errors.error_with_code(
                            crate::diagnostic::DiagnosticCode::E2002,
                            format!(
                                "Undefined function '{}'. Available monos: {:?}",
                                resolved_name, mono_names
                            ),
                            Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
                        );
                    }
                    Type::Tensor(ElementType::F32, vec![], None)
                }
            }
            _ => panic!("Expected IndexAccess, got {:?}", expr),
        }
    }

    /// Resolve and instantiate a `Struct::method(...)` static call (an inherent-impl method
    /// named through its type). Parses any explicit type args on the struct or method, finds the
    /// matching `_inherent` impl method, instantiates it, rewrites `name` to the mangled instance
    /// and checks that instance once. Returns the instance return type, or an `f32` placeholder
    /// after E-undefined if no such method exists. Split out of `check_functioncall_expr`
    /// (frontend_refactoring_borrow_checker.md R4, #279).
    fn check_static_method_call(
        &mut self,
        resolved_name: &crate::symbol::Symbol,
        name: &mut crate::symbol::Symbol,
    ) -> Type {
        let idx = resolved_name.find("::").expect("caller guards on `::`");
        let mut struct_name = resolved_name[..idx].to_string();
        let mut method_name = resolved_name[idx + 2..].to_string();
        let mut explicit_ty_str = String::new();

        if let Some(lt) = struct_name.find('<') {
            if struct_name.ends_with('>') {
                explicit_ty_str = struct_name[lt + 1..struct_name.len() - 1].to_string();
                struct_name = struct_name[..lt].to_string();
            }
        }

        if let Some(lt) = method_name.find('<') {
            if method_name.ends_with('>') {
                let method_ty_str = method_name[lt + 1..method_name.len() - 1].to_string();
                if explicit_ty_str.is_empty() {
                    explicit_ty_str = method_ty_str;
                } else {
                    explicit_ty_str = format!("{}, {}", explicit_ty_str, method_ty_str);
                }
                method_name = method_name[..lt].to_string();
            }
        }

        let mut found_generic_func = None;
        let mut found_mapping = HashMap::new();

        if let Some(impl_blocks) = self.env.impls.get("_inherent") {
            for ib in impl_blocks {
                let mut matches = false;
                if let Type::Struct(n, _) = &ib.target_type {
                    if *struct_name == **n {
                        matches = true;
                    }
                } else if let Type::Enum(n, _) = &ib.target_type {
                    if *struct_name == **n {
                        matches = true;
                    }
                } else if let Type::Generic(n, _) = &ib.target_type {
                    if *struct_name == **n {
                        matches = true;
                    }
                } else if let Type::GenericInstance(inner, _) = &ib.target_type {
                    if let Type::Struct(n, _) = &**inner {
                        if *struct_name == **n {
                            matches = true;
                        }
                    } else if let Type::Enum(n, _) = &**inner {
                        if *struct_name == **n {
                            matches = true;
                        }
                    }
                }

                if matches {
                    for m in &ib.methods {
                        if m.name == method_name.as_str().into() {
                            found_generic_func = Some(m.clone());
                            if !explicit_ty_str.is_empty() {
                                let mut explicit_args = Vec::new();
                                let mut depth = 0;
                                let mut current = String::new();
                                for c in explicit_ty_str.chars() {
                                    if c == '<' {
                                        depth += 1;
                                        current.push(c);
                                    } else if c == '>' {
                                        depth -= 1;
                                        current.push(c);
                                    } else if c == ',' && depth == 0 {
                                        explicit_args.push(self.parse_ty_str(&current));
                                        current.clear();
                                    } else {
                                        current.push(c);
                                    }
                                }
                                if !current.trim().is_empty() {
                                    explicit_args.push(self.parse_ty_str(&current));
                                }

                                for (i, parsed_ty) in explicit_args.into_iter().enumerate() {
                                    if i < ib.generics.len() {
                                        found_mapping
                                            .insert(ib.generics[i].name().to_string(), parsed_ty);
                                    }
                                }
                            }
                            break;
                        }
                    }
                }
                if found_generic_func.is_some() {
                    break;
                }
            }
        }

        if let Some(generic_func) = found_generic_func {
            let mut modified_func = generic_func.clone();
            modified_func.name = format!("{}::{}", struct_name, method_name).into();
            modified_func.generics = found_mapping
                .keys()
                .map(|k| decl::GenericParam::Type {
                    name: k.clone().into(),
                    bound: None,
                })
                .collect();

            let mut inst_func = self.instantiate_function(
                &modified_func,
                &found_mapping
                    .into_iter()
                    .map(|(k, v)| (k.into(), v))
                    .collect(),
                &std::collections::HashMap::new(),
            );
            let inst_ret = inst_func.return_type.clone();
            let inst_name = inst_func.name.clone();

            *name = inst_name.clone();

            if !self.env.functions.contains_key(inst_name.as_ref())
                && !self
                    .monomorphized_functions
                    .iter()
                    .any(|(f, _)| f.name == inst_name)
            {
                self.check_function(&mut inst_func);
                self.monomorphized_functions.push((inst_func, 0));
            }

            inst_ret
        } else {
            if !self.speculating {
                self.errors
                    .push(format!("Undefined static method '{}'.", resolved_name));
            }
            Type::Tensor(ElementType::F32, vec![], None)
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn instantiate_generic_function_call(
        &mut self,
        generic_func: &Function,
        origin_hash: u64,
        resolved_name: &str,
        name: &mut crate::symbol::Symbol,
        args: &[Expr],
        arg_types: &[Type],
        explicit_generic_args: &[Type],
    ) -> Option<Type> {
        let mut mapping: std::collections::HashMap<crate::symbol::Symbol, Type> = HashMap::new();
        let mut success = true;
        // Topology generic parameters (`<D: Topology>`): make `unify_types` bind a
        // `Pinned<_, D>` param's topology to the argument's concrete topology.
        self.pending_topo_vars = generic_func
            .generics
            .iter()
            .filter_map(|g| match g {
                decl::GenericParam::Type {
                    name,
                    bound: Some(b),
                } if b.as_ref() == "Topology" => Some(name.clone()),
                _ => None,
            })
            .collect();
        self.pending_topo_bindings.clear();
        if args.len() != generic_func.params.len() {
            if !self.speculating {
                self.errors.push(format!(
                    "Generic function '{}' expects {} arguments, got {}",
                    resolved_name,
                    generic_func.params.len(),
                    args.len()
                ));
            }
            success = false;
        } else {
            for (i, param) in generic_func.generics.iter().enumerate() {
                if i < explicit_generic_args.len() {
                    mapping.insert(param.name().into(), explicit_generic_args[i].clone());
                }
            }
            for (i, _arg) in args.iter().enumerate() {
                let arg_ty = arg_types[i].clone();
                let param_ty = &generic_func.params[i].1;
                if !self.unify_types(param_ty, &arg_ty, &mut mapping) {
                    if !self.speculating {
                        self.errors.push(format!("Failed to deduce types for generic function '{}': Expected {:?}, got {:?}", name, param_ty, arg_ty));
                    }
                    success = false;
                }
            }
            // Deduce return-only variables (a topology `D` in `-> Pinned<T, D>`, or a
            // return-only type generic) from the call's expected type, if known. Best-effort:
            // the return-type unify binds topology variables into `pending_topo_bindings`
            // (via the `Pinned` arm) and type generics into `mapping`.
            if let Some(expected) = self.expected_type.clone() {
                let ret_ty = generic_func.return_type.clone();
                let _ = self.unify_types(&ret_ty, &expected, &mut mapping);
            }
        }

        if success {
            for param in &generic_func.generics {
                let g_name = param.name();
                let bound_opt = match param {
                    decl::GenericParam::Type { bound, .. } => bound.clone(),
                    _ => None,
                };
                if let Some(bound_name) = bound_opt {
                    if let Some(concrete_ty) = mapping.get(g_name) {
                        let mut implements_trait = false;
                        if let Some(impl_blocks) = self.env.impls.get(bound_name.as_ref()) {
                            for ib in impl_blocks {
                                if self.unify_types(
                                    &ib.target_type,
                                    concrete_ty,
                                    &mut HashMap::new(),
                                ) {
                                    implements_trait = true;
                                    break;
                                }
                            }
                        }
                        if !implements_trait {
                            if !self.speculating {
                                self.errors.push(format!(
                                    "Type '{:?}' does not implement trait '{}' required by parameter '{}'",
                                    concrete_ty, bound_name, g_name
                                ));
                            }
                            success = false;
                        }
                    }
                }
            }
        }

        if success {
            let topo_mapping = std::mem::take(&mut self.pending_topo_bindings);
            self.pending_topo_vars.clear();

            // Discharge `where Transfer<A, B>` now that the topology variables are bound:
            // a transfer path from A's memory to B's must exist in the cost graph.
            for (a, b) in &generic_func.where_transfers {
                if let (Some(ta), Some(tb)) = (topo_mapping.get(a), topo_mapping.get(b)) {
                    let ma = self.transfer_cost_graph.default_memory_for(ta);
                    let mb = self.transfer_cost_graph.default_memory_for(tb);
                    if self.transfer_cost_graph.transfer_path(&ma, &mb).is_none() {
                        if !self.speculating {
                            self.errors.push(format!(
                                "unsatisfied `where Transfer<{}, {}>` in call to '{}': no transfer \
                                 path from {:?} to {:?}",
                                a, b, name, ta, tb
                            ));
                        }
                        return None;
                    }
                }
            }

            let mut inst_func = self.instantiate_function(generic_func, &mapping, &topo_mapping);
            let inst_ret = inst_func.return_type.clone();
            let inst_name = inst_func.name.clone();

            *name = inst_name.clone();

            if !self.env.functions.contains_key(inst_name.as_ref())
                && !self
                    .monomorphized_functions
                    .iter()
                    .any(|(f, _)| f.name == inst_name)
            {
                // Check the instantiated body in an *isolated* borrow context. It shares
                // `active_borrows` with the caller otherwise, and a same-named parameter (`m` here,
                // `m` in the caller) makes the callee's own `&m.field` run the NLL dead-borrow
                // cleanup against the caller's records with the callee's liveness — wrongly
                // releasing the caller's live reborrow before the next statement is checked (#268).
                let saved_borrows = self.borrow.take();
                self.check_function(&mut inst_func);
                self.borrow.restore(saved_borrows);
                self.monomorphized_functions.push((inst_func, origin_hash));
            }
            Some(inst_ret)
        } else {
            None
        }
    }

    pub(crate) fn resolve_intrinsic_function(
        &mut self,
        resolved_name: &str,
        args: &[Expr],
        arg_types: &[Type],
        explicit_generic_args: &[Type],
    ) -> Option<Type> {
        if resolved_name == "Verified" {
            if args.len() != 1 {
                self.errors.push(format!(
                    "Function 'Verified' expects 1 argument, got {}",
                    args.len()
                ));
            }
            let inner_ty = arg_types[0].clone();
            Some(Type::Verified(Box::new(inner_ty)))
        } else if resolved_name.starts_with("Tensor") && resolved_name.ends_with("::from") {
            if args.len() != 2 {
                self.errors.push(format!(
                    "Function '{}' expects 2 arguments (pointer, shape), got {}",
                    resolved_name,
                    args.len()
                ));
            }
            if !self.in_unsafe_block {
                self.errors.push(format!("Call to '{}' is unsafe because it interprets raw memory. Requires unsafe block.", resolved_name));
            }
            let mut el_ty = ElementType::F32;
            if resolved_name.contains("_i32") {
                el_ty = ElementType::I32;
            } else if resolved_name.contains("_i64") {
                el_ty = ElementType::I64;
            } else if resolved_name.contains("_f64") {
                el_ty = ElementType::F64;
            }
            let mut dims = Vec::new();
            if args.len() == 2 {
                if let Expr::Array(arr) = &args[1] {
                    dims = arr.elements.clone();
                }
            }
            Some(Type::Tensor(el_ty, dims, None))
        } else if resolved_name.starts_with("Tensor")
            && !resolved_name.contains("$")
            && !resolved_name.contains("__")
        {
            let el_ty = if !explicit_generic_args.is_empty() {
                if let Type::Scalar(el) = &explicit_generic_args[0] {
                    el.clone()
                } else {
                    self.errors
                        .push("Generic argument to Tensor must be a scalar type.".to_string());
                    ElementType::F32
                }
            } else {
                self.errors
                    .push("Missing generic argument for Tensor initialization.".to_string());
                ElementType::F32
            };
            let mut dims = Vec::new();
            if !args.is_empty() {
                if let Expr::Array(arr) = &args[0] {
                    if let Some(shape) = arr.initializer_shape() {
                        // Initializer list `Tensor<T>([[..],[..]])`: shape from the nesting.
                        dims = shape.into_iter().map(int_dim_expr).collect();
                    } else {
                        dims = arr.elements.clone();
                    }
                } else {
                    dims = args.to_vec();
                }
            }
            Some(Type::Tensor(el_ty, dims, None))
        } else if resolved_name.starts_with("Math::") {
            if args.len() != 1 {
                self.errors.push(format!(
                    "Function '{}' expects 1 argument, got {}",
                    resolved_name,
                    args.len()
                ));
            }
            let inner_ty = arg_types[0].clone();
            if inner_ty != Type::Scalar(ElementType::F32) {
                self.errors.push(format!(
                    "Function '{}' expects f32 argument, got {:?}",
                    resolved_name, inner_ty
                ));
            }
            Some(Type::Scalar(ElementType::F32))
        } else if resolved_name == "print" {
            if args.len() != 1 {
                self.errors
                    .push("Function 'print' expects 1 argument".to_string());
            }
            Some(Type::Tensor(ElementType::F32, vec![], None))
        } else if resolved_name == "printf" || resolved_name == "vx_internal_printf" {
            if args.is_empty() {
                self.errors
                    .push("Function 'printf' expects at least 1 argument".to_string());
            }
            Some(Type::Scalar(ElementType::I32))
        } else if resolved_name == "Some" || resolved_name == "Option::Some" {
            if args.len() != 1 {
                self.errors
                    .push(format!("Function '{}' expects 1 argument", resolved_name));
            }
            Some(Type::Struct("Option".into(), None))
        } else if resolved_name == "dot" {
            // Slice reduction (S2): dot(a, b) over two rank-1 f32 slices -> scalar f32.
            // Lowers to vector.load + arith.mulf + vector.reduction<add> (SIMD by construction).
            if args.len() != 2 {
                self.errors
                    .push("Function 'dot' expects 2 slice arguments".to_string());
            }
            for t in arg_types.iter().take(2) {
                if !Self::is_f32_slice(t) {
                    self.errors.push(format!(
                        "Function 'dot' expects rank-1 f32 slices, got {:?}",
                        t
                    ));
                }
            }
            Some(Type::Scalar(ElementType::F32))
        } else if resolved_name == "sum" || resolved_name == "max" || resolved_name == "min" {
            // Slice reduction (S2): sum/max/min(a) over a rank-1 f32 slice -> scalar f32.
            // Lowers to vector.load + vector.reduction<add|maximumf|minimumf>.
            if args.len() != 1 {
                self.errors.push(format!(
                    "Function '{}' expects 1 slice argument",
                    resolved_name
                ));
            }
            if let Some(t) = arg_types.first() {
                if !Self::is_f32_slice(t) {
                    self.errors.push(format!(
                        "Function '{}' expects a rank-1 f32 slice, got {:?}",
                        resolved_name, t
                    ));
                }
            }
            Some(Type::Scalar(ElementType::F32))
        } else {
            None
        }
    }

    /// Resolve `base_ty`'s method `method` by walking every impl block (unifying the receiver,
    /// peeling references), filling `mapping` with the deduced impl-level generic bindings. Returns
    /// the matching method + its impl block, or `None`. In debug builds also asserts the frozen
    /// registry's `ModuleInterface` resolves the same concrete `(receiver GID, method)` — the #219
    /// keep-green parity gate. Split out of `check_methodcall_expr` (R4, #279).
    fn resolve_method_in_impls(
        &mut self,
        base_ty: &Type,
        method: &crate::symbol::Symbol,
        mapping: &mut std::collections::HashMap<crate::symbol::Symbol, Type>,
    ) -> Option<(Function, decl::ImplBlock)> {
        let mut found_method = None;
        for impl_blocks in self.env.impls.values() {
            for ib in impl_blocks {
                mapping.clear();
                let mut check_ty = base_ty.clone();
                while let Type::Borrow { inner, .. }
                | Type::Pointer(inner, _, _)
                | Type::Ref(inner, _) = &check_ty
                {
                    check_ty = *inner.clone();
                }

                if self.unify_types(&ib.target_type, &check_ty, mapping) {
                    for m in &ib.methods {
                        if m.name == *method {
                            found_method = Some((m.clone(), (*ib).clone()));
                            break;
                        }
                    }
                }
                if found_method.is_some() {
                    break;
                }
            }
            if found_method.is_some() {
                break;
            }
        }

        // Dual-run parity gate for the stdlib<->compiler decoupling (#219): when the AST
        // impl-walk above resolves a method on a *concrete* receiver via a non-generic impl,
        // the registry-backed `ModuleInterface` must resolve the same `(receiver GID, method)`.
        // This proves the frozen registry is a sufficient oracle at real resolution sites --
        // the keep-green gate before imported-symbol resolution stops consulting the borrowed
        // AST env. Generic impls and non-nominal / generic receivers are outside the registry
        // method table's scope (#218), so they are skipped rather than asserted. The gate only
        // runs when a frozen registry is actually in use: an *empty* method table means this
        // compilation never built one (the sequential driver / legacy AST-only harnesses use
        // `GlobalSession::new`), so there is nothing to dual-run against.
        #[cfg(debug_assertions)]
        if let Some((ref m, ref ib)) = found_method {
            if !self.worker.global.registry.methods.is_empty()
                && ib.generics.is_empty()
                && m.generics.is_empty()
            {
                let mut recv = base_ty.clone();
                while let Type::Borrow { inner, .. }
                | Type::Pointer(inner, _, _)
                | Type::Ref(inner, _) = &recv
                {
                    recv = (**inner).clone();
                }
                let recv_gid = match &recv {
                    Type::Scalar(ElementType::Generic(_)) => None,
                    Type::Scalar(e) => Some(crate::hir::flatten::scalar_gid(e)),
                    Type::Struct(_, Some(id)) | Type::Enum(_, Some(id)) => Some(*id),
                    _ => None,
                };
                if let Some(gid) = recv_gid {
                    let mi: &dyn crate::registry::ModuleInterface = &*self.worker.global.registry;
                    debug_assert!(
                        mi.resolve_method(gid, &m.name).is_some(),
                        "ModuleInterface missing a method the AST resolved: {}.{}",
                        recv.mangle(),
                        m.name
                    );
                }
            }
        }
        found_method
    }

    /// Instantiate the resolved generic method, register + type-check the monomorphized function,
    /// and build the `FunctionCall` node the method call rewrites to (prepending the receiver,
    /// borrowed to match a `&self`/`&mut self` first parameter). Probes the synthesized call
    /// speculatively (R3) to recover its return type without double-reporting. Returns `(return
    /// type, replacement node)`; the caller performs the `*expr = …` rewrite. Split out of
    /// `check_methodcall_expr` (R4, #279).
    #[allow(clippy::too_many_arguments)]
    fn instantiate_method_call_rewrite(
        &mut self,
        generic_method: Function,
        mut mapping: std::collections::HashMap<crate::symbol::Symbol, Type>,
        base_ty: &Type,
        obj: &Expr,
        args: &[Expr],
        checked_arg_types: &[Type],
        consume: bool,
    ) -> (Type, Expr) {
        // Infer method-level generics from argument types. Reuse the types from the
        // single check above: re-checking here would re-consume linear args (a closure
        // struct passed to `.map`) and yield `Unknown`, defeating the deduction.
        for (i, arg_ty) in checked_arg_types.iter().enumerate() {
            if i + 1 < generic_method.params.len() {
                let expected_param = &generic_method.params[i + 1].1;
                self.unify_types(expected_param, arg_ty, &mut mapping);
            }
        }

        // Provide generic mapping to the method itself by copying impl block generics
        let mut modified_func = generic_method.clone();
        modified_func.generics = mapping
            .keys()
            .map(|k| decl::GenericParam::Type {
                name: k.clone(),
                bound: None,
            })
            .collect();
        let mut method_func =
            self.instantiate_function(&modified_func, &mapping, &std::collections::HashMap::new());

        // Create a unique mangled name for the method based on the target type
        let mangled_name = format!("{}${}", base_ty.mangle(), method_func.name);

        method_func.name = mangled_name.clone().into();

        if !self.env.functions.contains_key(&*mangled_name)
            && !self
                .monomorphized_functions
                .iter()
                .any(|(f, _)| f.name == crate::symbol::Symbol::from(mangled_name.as_str()))
        {
            // Type check the instantiated method
            let mut func_to_check = method_func.clone();
            self.check_function(&mut func_to_check);
            self.monomorphized_functions.push((func_to_check, 0)); // 0 will fall back to caller_module_idx
        }

        // Rewrite AST from MethodCall to FunctionCall
        let mut call_args = vec![];
        if let Some(first_param) = method_func.params.first() {
            let param_is_ref =
                matches!(first_param.1, Type::Borrow { .. } | Type::Pointer(_, _, _));
            let is_mut = match &first_param.1 {
                Type::Borrow { is_mut: m, .. } => *m,
                Type::Pointer(_, _, m) => *m,
                _ => false,
            };

            let obj_is_ref = matches!(base_ty, Type::Borrow { .. } | Type::Pointer(_, _, _));

            if param_is_ref && !obj_is_ref {
                call_args.push(Expr::Borrow(BorrowExpr {
                    expr: Box::new((*obj).clone()),
                    is_mut,
                    span: Span::default(),
                }));
            } else {
                call_args.push((*obj).clone());
            }
        } else {
            call_args.push((*obj).clone());
        }

        for a in args.iter() {
            call_args.push(a.clone());
        }

        let mut func_call = Expr::FunctionCall(FunctionCallExpr {
            name: crate::symbol::Symbol::from(mangled_name.as_str()),
            type_args: None,
            args: call_args,
            span: Span::default(),
        });
        // Probe the synthesized call *speculatively* to recover its return type without
        // emitting diagnostics or committing borrow/move side effects: the method-call
        // node is only now being rewritten into this call, so a second (real) check
        // would double-report. This is the sole site that turns `speculating` on; the
        // "fresh check" entry points (`check_expr_type`, `check_block`) force it back off
        // for independent subtrees. Replaces the old `silent = true` argument (#279 R3).
        let saved_speculating = self.speculating;
        self.speculating = true;
        let ret_ty = self.check_expr_type_flag(&mut func_call, consume);
        self.speculating = saved_speculating;
        (ret_ty, func_call)
    }

    pub(crate) fn check_methodcall_expr(&mut self, expr: &mut Expr, consume: bool) -> Type {
        match expr {
            Expr::MethodCall(MethodCallExpr {
                base: obj,
                method_name: _method,
                type_args: _,
                args,
                span: method_span,
            }) => {
                let method_span = *method_span;
                let mut base_ty = self.check_expr_type_flag(obj, false);

                // Pre-infer closure argument types for specific intrinsics before type-checking them
                if let Type::Tensor(el_ty, _, _) = &base_ty {
                    if _method.as_ref() == "map" && args.len() == 1 {
                        if let Expr::Closure(c) = &mut args[0] {
                            if c.params.len() == 1 && c.params[0].1 == Type::Unknown {
                                c.params[0].1 = Type::Scalar(el_ty.clone());
                            }
                        }
                    }
                }

                // Check each argument once, keeping its type. A linear arg (e.g. a closure
                // struct passed to `.map`) is consumed by this pass, so re-checking it later for
                // generic deduction would see it moved and yield `Unknown` — reuse these instead.
                let mut checked_arg_types: Vec<Type> = Vec::with_capacity(args.len());
                for arg in args.iter_mut() {
                    checked_arg_types.push(self.check_expr_type(arg));
                }

                if _method.as_ref() == "drop" && args.is_empty() {
                    if let Expr::Identifier(id) = &**obj {
                        self.consume(&id.name);
                    }
                }

                if let Type::Module(ref path, ref exports) = base_ty {
                    if let Some(exported_ty) = exports.get(_method) {
                        let prefix = TypeChecker::mangle_path(path);
                        let mangled_name = format!("{}_{}", prefix, _method);
                        let func_call = Expr::FunctionCall(FunctionCallExpr {
                            name: crate::symbol::Symbol::from(mangled_name.as_str()),
                            type_args: None,
                            args: args.clone(),
                            span: Span::default(),
                        });
                        *expr = func_call;
                        return exported_ty.clone();
                    } else {
                        self.errors.push(format!(
                            "Module '{}' does not export function '{}'",
                            path, _method
                        ));
                        return Type::Tensor(ElementType::F32, vec![], None);
                    }
                }

                if let Some((ty, replace_with_obj)) =
                    self.resolve_intrinsic_method(&base_ty, _method, args)
                {
                    if replace_with_obj {
                        *expr = *obj.clone();
                    }
                    return ty;
                }

                // Dynamic Method Resolution
                let mut mapping = HashMap::new();
                let found_method = self.resolve_method_in_impls(&base_ty, _method, &mut mapping);
                if let Some((generic_method, _ib)) = found_method {
                    let (ret_ty, func_call) = self.instantiate_method_call_rewrite(
                        generic_method,
                        mapping,
                        &base_ty,
                        obj,
                        args.as_slice(),
                        &checked_arg_types,
                        consume,
                    );
                    *expr = func_call;
                    return ret_ty;
                }

                // Fallback for hardcoded mock methods
                if _method.as_ref() == "with_memory" {
                    base_ty = Type::Ref(Box::new(base_ty), MemorySpace::NPUHBM);
                } else if matches!(
                    _method.as_ref(),
                    "to_device"
                        | "to_device_relaxed"
                        | "to_sram"
                        | "to_sram_relaxed"
                        | "to_gpu"
                        | "to_gpu_relaxed"
                ) {
                    // Device-placement transfers. The `_relaxed` variants are the escape
                    // hatch that omits the synchronizing release / DMA-completion wait.
                    // Read before reassigning `*expr`, since `_method` borrows from it.
                    let m = _method.as_ref();
                    let is_relaxed = m.ends_with("_relaxed");
                    let target_mem = if m.starts_with("to_sram") {
                        MemorySpace::LocalSRAM // accelerator-core scratchpad
                    } else if m.starts_with("to_gpu") {
                        MemorySpace::GpuHbm // discrete-GPU device memory (NVPTX side)
                    } else {
                        MemorySpace::NPUHBM // to_device: default NPU[0]
                    };
                    *expr = Expr::Transfer(TransferExpr {
                        expr: obj.clone(),
                        space: target_mem,
                        cost: None,
                        span: method_span,
                    });
                    // Mark it so the per-seam obligation in `check_transfer_expr` sends
                    // published payloads to TOP (a stale read).
                    self.pending_transfer_relaxed = is_relaxed;
                    return self.check_transfer_expr(expr, consume);
                } else if _method.as_ref() == "to_host" {
                    let target_mem = MemorySpace::CPUDRAM;
                    base_ty = Type::Pinned(Box::new(base_ty), Topology::CPU);
                    *expr = Expr::Transfer(TransferExpr {
                        expr: obj.clone(),
                        space: target_mem,
                        cost: None,
                        span: Span::default(),
                    });
                } else if _method.as_ref() == "as_ptr" || **_method == *"as_mut_ptr" {
                    let is_mut = _method.as_ref() == "as_mut_ptr";
                    match &base_ty {
                        Type::Tensor(el_ty, dims, top) => {
                            base_ty = Type::Pointer(
                                Box::new(Type::Tensor(el_ty.clone(), dims.clone(), top.clone())),
                                None,
                                is_mut,
                            );
                        }
                        Type::Borrow {
                            inner,
                            mem_space: mem,
                            is_mut: mutability,
                            ..
                        } => {
                            if is_mut && !mutability {
                                self.errors.push(
                                    "Cannot get mutable pointer from immutable borrow".to_string(),
                                );
                            }
                            base_ty = Type::Pointer(inner.clone(), mem.clone(), is_mut);
                        }
                        Type::Pointer(_, _, _) => {
                            self.errors.push("Already a pointer".to_string());
                        }
                        _ => {
                            self.errors
                                .push(format!("Cannot call {} on {:?}", _method, base_ty));
                        }
                    }
                } else if _method.as_ref() == "len" {
                    match &base_ty {
                        Type::Tensor(_, _, _) | Type::Borrow { .. } | Type::Pointer(_, _, _) => {
                            base_ty = Type::Tensor(ElementType::I64, vec![], None);
                        }
                        _ => {
                            self.errors
                                .push(format!("Cannot call len on {:?}", base_ty));
                        }
                    }
                } else {
                    self.errors.push(format!(
                        "Method '{}' not found on type {:?}",
                        _method, base_ty
                    ));
                }
                base_ty
            }
            _ => panic!("Expected IndexAccess, got {:?}", expr),
        }
    }

    /// Resolve a callee's `(param types, return type)` for the reborrow analysis (#243, #268). The
    /// return type's *reference shape* (is it a reference? which parameters are references?) is all
    /// the reborrow decision needs; the summary (`return_provenance_of`) supplies which parameter
    /// the return derives from, defaulting to `AnyParam` where unknown.
    ///
    /// **Generics resolve to their declared signature** without instantiation: params/return may
    /// carry type variables, but `&T`/`&mut T` are still references, so a reborrow through a generic
    /// callee (bc9 through a generic, #268) is tracked conservatively instead of leaking untracked.
    /// A generic returning a non-reference has a non-reference declared return, so nothing persists —
    /// no false positive. Only genuinely unresolvable callees (closures resolved via other rules,
    /// `dyn`, intrinsics) still return `None`.
    ///
    /// **Function-pointer / closure *values*** (a parameter or local of `fn(..)->..` type, #268)
    /// carry their signature in their type, so a reborrow through `f(m)` is tracked even though which
    /// function `f` holds is unknown — the aliasing depends only on the signature. Checked first so a
    /// shadowing local wins over a same-named global function.
    pub(crate) fn resolve_callee_ref_signature(
        &self,
        resolved_name: &str,
    ) -> Option<(Vec<Type>, Type)> {
        if let Some((Type::Function(params, ret) | Type::Closure(params, ret), _)) =
            self.lookup(resolved_name)
        {
            return Some((params.clone(), (**ret).clone()));
        }
        // A closure *value* (`f` typed `Closure_N`): resolve its generated `Closure_N_call` and drop
        // the synthetic environment parameter (the leading `skip(1)`), so its reference arguments
        // line up with the call's arguments. Without this, a reborrow through a closure (`let r =
        // f(m); insert(m, ..)`) was tracked by nothing — the escape rule caught the unsound case only
        // incidentally, and over-rejected the sound one (#269).
        let closure_call: Option<String> = match self.lookup(resolved_name) {
            Some((Type::Struct(sname, _), _)) if sname.starts_with("Closure_") => {
                Some(format!("{}_call", sname))
            }
            _ => None,
        };
        if let Some(call_name) = closure_call {
            if let Some(f) = self
                .monomorphized_functions
                .iter()
                .find(|f| f.0.name.as_ref() == call_name)
            {
                return Some((
                    f.0.params.iter().skip(1).map(|(_, t)| t.clone()).collect(),
                    f.0.return_type.clone(),
                ));
            }
            if let Some(f) = self.env.syntax_functions.get(call_name.as_str()) {
                return Some((
                    f.params.iter().skip(1).map(|(_, t)| t.clone()).collect(),
                    f.return_type.clone(),
                ));
            }
        }
        if let Some(f) = self
            .monomorphized_functions
            .iter()
            .find(|f| f.0.name.as_ref() == resolved_name)
        {
            return Some((
                f.0.params.iter().map(|(_, t)| t.clone()).collect(),
                f.0.return_type.clone(),
            ));
        }
        if let Some(f) = self.env.syntax_functions.get(resolved_name) {
            if f.generics.is_empty() {
                return Some((
                    f.params.iter().map(|(_, t)| t.clone()).collect(),
                    f.return_type.clone(),
                ));
            }
        }
        // Generic callee: its declared signature (the name may carry explicit type args, e.g.
        // `pass<i32>`, so strip them to the base name the generic table is keyed by).
        let base = resolved_name.split('<').next().unwrap_or(resolved_name);
        if let Some((gf, _)) = self.env.generic_functions.get(base) {
            return Some((
                gf.params.iter().map(|(_, t)| t.clone()).collect(),
                gf.return_type.clone(),
            ));
        }
        // An *imported* callee resolved from a merged `.vxlib` interface (#219 flip, phase 2) has no
        // AST in this compile, so it is absent from the env tables above — but its signature lives in
        // the frozen registry's `fn_sigs`. Resolving it there is what lets the borrow checker
        // reborrow-track (and, via `ret_prov`, provenance-refine) a cross-module reference call.
        // Checked last so a local definition always shadows a same-named import.
        if let Some(sig) = self
            .worker
            .global
            .registry
            .fn_sigs
            .get(&crate::symbol::Symbol::from(resolved_name))
        {
            return Some((sig.params.clone(), sig.ret_ty.clone()));
        }
        None
    }

    pub(crate) fn resolve_intrinsic_method(
        &mut self,
        base_ty: &Type,
        _method: &str,
        args: &mut [Expr],
    ) -> Option<(Type, bool)> {
        if let Type::Pinned(_inner, _top) = base_ty {
            if _method == "topology" {
                if !args.is_empty() {
                    self.errors
                        .push("topology requires 0 arguments".to_string());
                }
                return Some((Type::Struct("Option".into(), None), false));
            }
        }

        if let Type::Tensor(el_ty, dims, top) = base_ty {
            if _method == "topology" {
                if !args.is_empty() {
                    self.errors
                        .push("topology requires 0 arguments".to_string());
                }
                return Some((Type::Struct("Option".into(), None), false));
            } else if _method == "reshape" {
                if args.is_empty() || args.len() > 3 {
                    self.errors
                        .push("reshape requires 1 to 3 arguments".to_string());
                    return Some((base_ty.clone(), false));
                }

                let mut is_exact = true;
                if args.len() >= 2 {
                    if let Expr::EnumVariant(EnumVariantExpr {
                        enum_name,
                        variant_name: variant,
                        payload: _,
                        span: _,
                    }) = &args[1]
                    {
                        if enum_name.as_ref() == "PadMode"
                            && (variant.as_ref() == "Pad" || variant.as_ref() == "Trim")
                        {
                            is_exact = false;
                        } else {
                            self.errors.push(
                                "reshape mode must be PadMode::Pad or PadMode::Trim".to_string(),
                            );
                        }
                    } else {
                        self.errors.push(
                            "reshape mode must be an enum variant (e.g. PadMode::Pad)".to_string(),
                        );
                    }
                }

                if let Expr::Array(ArrayExpr {
                    elements: new_dims,
                    span: _,
                }) = &args[0]
                {
                    let empty_env = HashMap::new();
                    let mut src_elements = 1.0;
                    for d in dims {
                        if let Some(Value::Number(v)) = self.eval_expr(d, &empty_env) {
                            src_elements *= v;
                        } else {
                            self.errors.push(
                                "Cannot statically evaluate source dimension for reshape"
                                    .to_string(),
                            );
                            return Some((base_ty.clone(), false));
                        }
                    }

                    let mut target_elements = 1.0;
                    for d in new_dims {
                        if let Some(Value::Number(v)) = self.eval_expr(d, &empty_env) {
                            target_elements *= v;
                        } else {
                            self.errors.push(
                                "Cannot statically evaluate target dimension for reshape"
                                    .to_string(),
                            );
                            return Some((base_ty.clone(), false));
                        }
                    }

                    if is_exact && (src_elements - target_elements).abs() > 1e-6 {
                        self.errors.push(format!(
                            "reshape arithmetic mismatch: source has {} elements, target has {}",
                            src_elements, target_elements
                        ));
                        return Some((base_ty.clone(), false));
                    }

                    return Some((
                        Type::Tensor(el_ty.clone(), new_dims.clone(), top.clone()),
                        false,
                    ));
                } else {
                    self.errors.push(
                        "reshape requires an array of dimensions as the first argument".to_string(),
                    );
                    return Some((base_ty.clone(), false));
                }
            } else if _method == "iter" {
                if !args.is_empty() {
                    self.errors.push("iter requires 0 arguments".to_string());
                }
                return Some((base_ty.clone(), true));
            } else if _method == "map" {
                if args.len() != 1 {
                    self.errors
                        .push("map requires exactly 1 argument (the closure)".to_string());
                    return Some((base_ty.clone(), false));
                }

                let arg_ty = self.check_expr_type(&mut args[0]);
                if let Type::Struct(name, _) = &arg_ty {
                    if !name.starts_with("Closure_") {
                        self.errors
                            .push(format!("map expects a closure, got {:?}", arg_ty));
                    }
                } else {
                    self.errors
                        .push(format!("map expects a closure, got {:?}", arg_ty));
                }
                return Some((base_ty.clone(), false));
            } else if _method == "transpose" {
                if args.len() != 1 {
                    self.errors.push(
                        "transpose requires exactly 1 argument (an array of permutation indices)"
                            .to_string(),
                    );
                    return Some((base_ty.clone(), false));
                }
                if let Expr::Array(ArrayExpr {
                    elements: perm,
                    span: _,
                }) = &args[0]
                {
                    let empty_env = HashMap::new();
                    let mut new_dims = vec![
                        Expr::Number(NumberExpr {
                            value: "0".into(),
                            ty: Some(ElementType::I32),
                            span: Span::default()
                        });
                        dims.len()
                    ];
                    if perm.len() != dims.len() {
                        self.errors.push(
                            "transpose permutation map length must match tensor rank".to_string(),
                        );
                        return Some((base_ty.clone(), false));
                    }
                    let mut seen = vec![false; dims.len()];
                    for (i, p) in perm.iter().enumerate() {
                        if let Some(Value::Number(v)) = self.eval_expr(p, &empty_env) {
                            let v = v as usize;
                            if v >= dims.len() {
                                self.errors
                                    .push("transpose index out of bounds".to_string());
                                return Some((base_ty.clone(), false));
                            }
                            if seen[v] {
                                self.errors.push(
                                    "transpose permutation map must not contain duplicates"
                                        .to_string(),
                                );
                                return Some((base_ty.clone(), false));
                            }
                            seen[v] = true;
                            new_dims[i] = dims[v].clone();
                        } else {
                            self.errors.push(
                                "Cannot statically evaluate transpose permutation index"
                                    .to_string(),
                            );
                            return Some((base_ty.clone(), false));
                        }
                    }
                    return Some((Type::Tensor(el_ty.clone(), new_dims, top.clone()), false));
                } else {
                    self.errors
                        .push("transpose requires an array of permutation indices".to_string());
                    return Some((base_ty.clone(), false));
                }
            }
        }
        None
    }
}
