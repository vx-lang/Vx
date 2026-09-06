//===- expr.rs - Vx Compiler -------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// Semantic analysis for expressions. Enforces type checking, const generics, and memory topology rules.
//
//===----------------------------------------------------------------------===//

use super::*;

/// Build an `i32` numeric dimension expression (used to synthesize a tensor shape from an
/// initializer list's nesting).
pub(crate) fn int_dim_expr(n: usize) -> Expr {
    Expr::Number(NumberExpr {
        value: n.to_string().into(),
        ty: Some(ElementType::I32),
        span: Span::default(),
    })
}

/// The scalar element an untyped numeric literal adopts from its expected type, when compatible:
/// an integer literal takes an integer expected type, a float literal a float expected type. A
/// `bool`, generic, or non-scalar expected type — or a literal/expected-kind mismatch such as
/// `let x: f64 = 5` — yields `None`, leaving the literal to its spelling default (#240).
pub(crate) fn expected_numeric_elem(expected: &Type, value: &str) -> Option<ElementType> {
    let Type::Scalar(el) = expected else {
        return None;
    };
    if matches!(el, ElementType::Bool | ElementType::Generic(_)) {
        return None;
    }
    let is_float_lit = value.contains('.');
    (el.is_float() == is_float_lit).then(|| el.clone())
}

impl<'a> TypeChecker<'a> {
    pub fn check_expr_type(&mut self, expr: &mut Expr) -> Type {
        // A "fresh", non-speculative check. Force `speculating` off for the duration so a probe
        // higher on the stack (the methodcall return-type probe in `check_methodcall_expr`) can't
        // leak into this independent subtree — reproducing the old hard-coded `silent = false`
        // argument this call used to pass (#279 R3).
        let saved = self.speculating;
        self.speculating = false;
        let ty = self.check_expr_type_flag(expr, true);
        self.speculating = saved;
        ty
    }

    pub(crate) fn check_expr_block(&mut self, stmts: &mut [Statement], consume: bool) -> Type {
        let mut ret_ty = Type::Struct("void".into(), None);
        let mut terminated = false;

        for s in stmts.iter_mut() {
            if terminated && !self.speculating {
                let stmt_span = s.span();
                self.errors.warn(
                    crate::diagnostic::DiagnosticCode::W1003,
                    "Unreachable code after return, break, or continue",
                    Some(crate::diagnostic::SourceSpan::from_ast_span(&stmt_span)),
                );
                break;
            }

            if let Statement::ExprStmt(ExprStmtStmt {
                ref mut expr,
                has_semi: _,
                span: _,
            }) = s
            {
                let saved_borrows = self.borrow.snapshot();
                ret_ty = self.check_expr_type_flag(expr, consume);
                self.borrow.restore(saved_borrows);
            } else {
                let expected_ret = self.current_return_type.clone().unwrap_or(Type::Tensor(
                    ElementType::F32,
                    vec![],
                    None,
                ));
                self.check_statement(s, &expected_ret, consume);
            }

            match s {
                Statement::Return(_) | Statement::Break(_) | Statement::Continue(_) => {
                    terminated = true;
                }
                _ => {}
            }
        }
        ret_ty
    }

    pub fn check_expr_type_flag(&mut self, expr: &mut Expr, consume: bool) -> Type {
        let mut is_enum_variant = false;
        if let Expr::FunctionCall(fc) = expr {
            if let Some((enum_name, _)) = fc.name.split_once("::") {
                let actual_enum_name = if let Some(idx) = enum_name.find('<') {
                    &enum_name[0..idx]
                } else {
                    enum_name
                };
                if self.env.enums.contains_key(actual_enum_name) {
                    is_enum_variant = true;
                }
            }
        }

        if is_enum_variant {
            if let Expr::FunctionCall(fc) = expr {
                if let Some((enum_name, variant)) = fc.name.split_once("::") {
                    let mut payload = None;
                    if !fc.args.is_empty() {
                        let mut args = Vec::new();
                        std::mem::swap(&mut args, &mut fc.args);
                        payload = Some(args);
                    }
                    *expr = Expr::EnumVariant(EnumVariantExpr {
                        enum_name: enum_name.into(),
                        variant_name: variant.to_string().into(),
                        payload,
                        span: fc.span,
                    });
                }
            }
        }

        match expr {
            Expr::Identifier(..) => self.check_identifier_expr(expr, consume),
            Expr::EnumVariant(..) => self.check_enumvariant_expr(expr, consume),
            Expr::Number(n) => self.check_number_literal(n),
            Expr::StringLiteral(StringLiteralExpr { .. }) => Type::Pointer(
                Box::new(Type::Scalar(ElementType::I8)),
                None,
                false, // const
            ),
            Expr::Transfer(..) => self.check_transfer_expr(expr, consume),
            // `Reachable<A, B>` is a comptime boolean.
            Expr::TransferPredicate(..) => Type::Scalar(ElementType::Bool),
            Expr::ComptimeBlock(..) => self.check_comptimeblock_expr(expr, consume),
            Expr::SpawnOn(..) => self.check_spawnon_expr(expr, consume),
            Expr::If(..) => self.check_if_expr(expr, consume),
            Expr::SizeOf(..) => Type::Scalar(ElementType::I64),
            Expr::FunctionCall(..) => self.check_functioncall_expr(expr, consume),
            Expr::IndirectCall(..) => self.check_indirectcall_expr(expr, consume),
            Expr::Array(..) => self.check_array_expr(expr),
            Expr::MemberAccess(..) => self.check_memberaccess_expr(expr),
            Expr::IndexAccess(..) => self.check_indexaccess_expr(expr),
            Expr::MethodCall(..) => self.check_methodcall_expr(expr, consume),
            Expr::BinaryOp(..) => self.check_binaryop_expr(expr, consume),
            Expr::RelationalOp(..) => self.check_relationalop_expr(expr),
            Expr::LogicalOp(..) => self.check_logicalop_expr(expr),
            // A memory space is not a value: `transfer(a, Memory::X)` destructures it at the
            // call site, and every other use carries it in a type. Answering with a dims-less
            // tensor let `let t : Tensor<f32, [2, 2]> = Memory::GPU_HBM` type-check, because a
            // source with no dims skips the comparison against the annotation.
            Expr::MemorySpace(MemorySpaceExpr { space, .. }) => {
                self.errors.push(format!(
                    "`Memory::{}` names a memory space, which is not a value; it belongs in a \
                     type or as the destination of a `transfer`",
                    space.name()
                ));
                Type::Unknown
            }
            Expr::Topology(TopologyExpr { top, span: _ }) => {
                if matches!(top, Topology::Current) {
                    *top = self.active_topology.clone();
                }
                // A topology used as a *value* (assigned, pushed into a `Vec<Topology>`,
                // compared) is its runtime dispatch id: an i32 discriminant. Placement
                // uses (`Pinned<T, Topology::X>`, `spawn on`) carry the topology in the
                // type/statement, not through this expression's value type.
                Type::Scalar(ElementType::I32)
            }
            Expr::UnaryOp(..) => self.check_unaryop_expr(expr),
            Expr::Borrow(..) => self.check_borrow_expr(expr),
            Expr::Dereference(..) => self.check_dereference_expr(expr, consume),
            Expr::UnsafeBlock(..) => self.check_unsafeblock_expr(expr, consume),
            Expr::StructInit(..) => self.check_structinit_expr(expr, consume),
            Expr::Grad(..) => self.check_grad_expr(expr),
            Expr::Vjp(..) => self.check_vjp_expr(expr),
            Expr::Jvp(..) => self.check_jvp_expr(expr),
            Expr::Range(..) => self.check_range_expr(expr),
            Expr::Match(..) => self.check_match_expr(expr, consume),
            Expr::VecMacro(..) => self.check_vecmacro_expr(expr),
            Expr::Closure(..) => self.check_closure_expr(expr, consume),
            Expr::AsCast(e) => self.check_ascast_expr(e, consume),
            Expr::Print(p) => {
                for arg in &mut p.args {
                    self.check_expr_type_flag(arg, consume);
                }
                Type::Scalar(ElementType::I32) // Assuming print returns 0 as i32 for C compatibility
            }
            Expr::Println(p) => {
                for arg in &mut p.args {
                    self.check_expr_type_flag(arg, consume);
                }
                Type::Scalar(ElementType::I32)
            }
            Expr::InlineMlir(e) => {
                // Typecheck inputs
                for (_, arg_expr, _) in &mut e.inputs {
                    self.check_expr_type_flag(arg_expr, consume);
                }
                // Typecheck clobbers and mark them as mutated if needed
                for clobber in &mut e.clobbers {
                    self.check_expr_type_flag(clobber, consume);
                }

                // Return the specified type or Unknown if void
                e.returns.clone().unwrap_or(Type::Unknown)
            }
            Expr::MacroCall(_) => {
                self.errors.push("Macro failed to expand".to_string());
                crate::syntax::Type::Scalar(crate::syntax::types::ElementType::I32)
                // Fallback type
            }
        }
    }

    /// Lowers an AST `Type` to a globally resolved `TypeId` structure.
    /// This integrates the AST semantic boundary with the hardware-level
    /// 256-bit FastPath borrow checking rules.
    pub fn lower_to_type_id(&self, ty: &Type) -> crate::gid::TypeId {
        // We use a dummy symbol_hash for local types, as we are only concerned
        // with the Lifetime Signature (Word 2) for borrow checking right now.
        let mut id = crate::gid::TypeId::new(0, 0, 0, 0);

        match ty {
            Type::Borrow {
                region_id: region, ..
            } => {
                // The lifetime of the borrow itself is Covariant (even for mutable borrows,
                // which allows reborrowing for shorter lifetimes during function calls).
                // (The inner type T would be invariant for mutable borrows, but we are
                // only hashing the outer lifetime here).
                let variance: u8 = 0x1;

                // Pack the region and variance directly into Param 0 (the return slot) of the
                // FastPath hash! Slot 0 reserves its top 3 region bits for the return-provenance code
                // (#265), so its region is 9 bits: `region_for_depth_slot0` preserves the "unset"
                // sentinel, clamps a real depth below it, and keeps the value inside the 9-bit field
                // so it can neither collide with the sentinel (#267) nor spill into the code bits.
                let region = crate::borrow::region_for_depth_slot0(*region as u64) as u16;
                let _ = id.try_set_fast_param(0, region, variance);
            }
            Type::Pointer(_inner, _mem, is_mut) => {
                let variance: u8 = if *is_mut { 0x0 } else { 0x1 };
                // Pointers don't have safe lifetimes, so we assign 'static (0)
                // which represents the unconstrained lifetime.
                let _ = id.try_set_fast_param(0, 0, variance);
            }
            // For other types, we just return the raw un-initialized hash
            _ => {}
        }
        id
    }

    /// A located tensor split into what it holds and where it is, for either
    /// spelling. `None` for a type that says nothing about where it lives.
    fn located_parts(t: &Type) -> Option<(Type, Topology)> {
        match t {
            Type::Pinned(inner, top) => Some(((**inner).clone(), top.clone())),
            Type::Tensor(e, d, Some(p)) => {
                Some((Type::Tensor(e.clone(), d.clone(), None), p.topology.clone()))
            }
            _ => None,
        }
    }

    pub(crate) fn is_assignable(&self, target: &Type, source: &Type) -> bool {
        if target == source {
            return true;
        }

        // `Pinned<T, D>` and a `T` carrying a placement on `D` are two spellings of
        // one fact: this value is on that device. Both are real -- a signature can
        // write `Pinned<Tensor<bf16, [4, 4]>, Topology::NPU[0]>`, and the checker
        // wraps a topology-bound return the same way, while `transfer` and a
        // `Memory::` annotation produce the placed tensor. Comparing them by
        // constructor made a transferred tensor unassignable to a parameter that
        // named the same device.
        if let (Some((t_inner, t_top)), Some((s_inner, s_top))) =
            (Self::located_parts(target), Self::located_parts(source))
        {
            return t_top == s_top && self.is_assignable(&t_inner, &s_inner);
        }

        if let Type::Struct(n_target, id_target) = target {
            if let Type::Struct(n_source, id_source) = source {
                if n_target == n_source {
                    if id_target.is_some() && id_source.is_some() {
                        return id_target == id_source;
                    }
                    return true;
                }
            }
        }

        if let Type::Enum(n_target, id_target) = target {
            if let Type::Enum(n_source, id_source) = source {
                if n_target == n_source {
                    if id_target.is_some() && id_source.is_some() {
                        return id_target == id_source;
                    }
                    return true;
                }
            }
        }

        if let Type::Struct(n_target, id_target) = target {
            if let Type::Enum(n_source, id_source) = source {
                if n_target == n_source {
                    if id_target.is_some() && id_source.is_some() {
                        return id_target == id_source;
                    }
                    return true;
                }
            }
        }

        if let Type::Enum(n_target, id_target) = target {
            if let Type::Struct(n_source, id_source) = source {
                if n_target == n_source {
                    if id_target.is_some() && id_source.is_some() {
                        return id_target == id_source;
                    }
                    return true;
                }
            }
        }

        if let Type::GenericInstance(inner_target, args_target) = target {
            if let Type::GenericInstance(inner_source, args_source) = source {
                if self.is_assignable(inner_target, inner_source)
                    && args_target.len() == args_source.len()
                {
                    let mut all_match = true;
                    for (at, asrc) in args_target.iter().zip(args_source.iter()) {
                        if !self.is_assignable(at, asrc) {
                            all_match = false;
                            break;
                        }
                    }
                    if all_match {
                        return true;
                    }
                }
            }
        }

        // Allow assigning a scalar ElementType to a Simd type (for loading from pointer)
        if let Type::Simd(el_target, _) = target {
            if let Type::Scalar(el_source) = source {
                if el_target == el_source {
                    return true;
                }
            }
        }

        // Allow assigning a Simd type to a scalar ElementType (for storing to pointer)
        if let Type::Scalar(el_target) = target {
            if let Type::Simd(el_source, _) = source {
                if el_target == el_source {
                    return true;
                }
            }
        }

        // Explicit Memory transfer enforcement:
        // We no longer allow implicit unwrapping of Ref<T> or Pinned<T> to T.
        // Users must use `transfer(expr, Memory::Space)` or `.to_host()` / `.to_device()`
        // to move data across memory boundaries.

        // Allow numeric coercions for scalar literals (mock behavior for now)
        if let Type::Tensor(t_target, dims_target, top_target) = target {
            if let Type::Tensor(t_source, dims_source, top_source) = &source {
                let mut el_match = false;
                if *t_target == *t_source {
                    el_match = true;
                }

                if !el_match {
                    return false;
                }

                if top_target.is_some() && top_target != top_source {
                    return false;
                }

                // An annotation that names no place asks for host data, so a value
                // sitting on a device does not satisfy it -- passing one to
                // `fn f(t : Tensor<f32, [4]>)` is exactly the mistake `transfer`
                // exists to make visible. A value placed in CPU_DRAM does satisfy
                // it: that is where an unplaced tensor lives.
                if top_target.is_none() {
                    if let Some(p) = top_source {
                        if p.space != MemorySpace::CPUDRAM {
                            return false;
                        }
                    }
                }

                // Rank is static and no cast changes it. Per dimension, a `?` in the target
                // accepts any extent and a static extent accepts only itself: `[512, 8]` widens
                // to `[?, 8]`, and `[?, 8]` does not narrow to `[512, 8]` without saying so.
                if !dims_target.is_empty() && !dims_source.is_empty() {
                    if dims_target.len() != dims_source.len() {
                        return false;
                    }
                    let empty_env = std::collections::HashMap::new();
                    for (dt, ds) in dims_target.iter().zip(dims_source.iter()) {
                        let Some(et) = dt.as_static() else { continue };
                        let Some(es) = ds.as_static() else {
                            return false;
                        };
                        let vt = self.eval_expr(et, &empty_env);
                        let vs = self.eval_expr(es, &empty_env);
                        if vt.is_some() && vs.is_some() {
                            if vt != vs {
                                return false;
                            }
                        } else if et != es {
                            return false;
                        }
                    }
                }
                return true;
            }
        }

        if let Type::Function(p_target, r_target) = target {
            if let Type::Function(p_source, r_source) = source {
                if p_target.len() == p_source.len() && self.is_assignable(r_target, r_source) {
                    let mut all_match = true;
                    for (pt, ps) in p_target.iter().zip(p_source.iter()) {
                        if !self.is_assignable(pt, ps) {
                            all_match = false;
                        }
                    }
                    if all_match {
                        return true;
                    }
                }
            }
        }

        if let Type::Closure(p_target, r_target) = target {
            if let Type::Closure(p_source, r_source) = source {
                if p_target.len() == p_source.len() && self.is_assignable(r_target, r_source) {
                    let mut all_match = true;
                    for (pt, ps) in p_target.iter().zip(p_source.iter()) {
                        if !self.is_assignable(pt, ps) {
                            all_match = false;
                        }
                    }
                    if all_match {
                        return true;
                    }
                }
            }
        }

        // Allow Closure to map to ClosureN struct (if tests use it)
        if let Type::GenericInstance(inner, args) = target {
            if let Type::Struct(name, _) = &**inner {
                if name.starts_with("Closure") {
                    if let Type::Closure(p_source, r_source) = source {
                        if args.len() == p_source.len() + 1 {
                            let mut all_match = true;
                            for (i, ps) in p_source.iter().enumerate() {
                                if !self.is_assignable(&args[i], ps) {
                                    all_match = false;
                                }
                            }
                            if !self.is_assignable(&args[args.len() - 1], r_source) {
                                all_match = false;
                            }
                            if all_match {
                                return true;
                            }
                        }
                    }
                }
            }
        }

        if let Type::Scalar(t_target) = target {
            if let Type::Scalar(t_source) = &source {
                // Scalars are assignable only when identical (#240): Vx has no implicit numeric
                // conversion. An untyped literal has already adopted its context's type in the
                // checking positions (`let`/`return`/assignment/operands/call args), so what reaches
                // here mismatched is a genuine typed-value conversion — the programmer writes `as`.
                return *t_target == *t_source;
            }
        }

        // A scalar is assignable to a RANK-0 tensor of the identical element, which wraps it.
        // A shaped tensor is not: that spelling was a splat, and it allocated and filled a whole
        // buffer from something that reads as an assignment. The two codegen paths did not even
        // agree on it -- one emitted the allocation and the fill, the other kept the bare
        // constant. `Tensor<T, [..]>::fill(v)` says it instead. The any-to-any arm went with the
        // rest of implicit numeric conversion (#240, Vx#396).
        if let Type::Tensor(t_target, dims_target, _) = target {
            if let Type::Scalar(t_source) = &source {
                return dims_target.is_empty() && *t_target == *t_source;
            }
        }

        // Semantic coercion rule: Verified<T> can only be assigned from another Verified<U> where is_assignable(T, U)
        if let Type::Verified(inner_target) = target {
            if let Type::Verified(inner_source) = source {
                if self.is_assignable(inner_target, inner_source) {
                    return true;
                }
            }
        }

        // Note: Verified<T> should NOT implicitly coerce to T if the user strictly expected T in tests,
        // or perhaps we shouldn't strip it here. Let's revert this coercion so type_mismatch fails again.

        // Allow coercing Borrow to Pointer (e.g. &mut T to *mut T)
        if let Type::Pointer(target_inner, target_mem, target_mut) = target {
            if let Type::Borrow {
                inner: source_inner,
                mem_space: source_mem,
                is_mut: source_mut,
                ..
            } = source
            {
                if target_mem == source_mem
                    && (!*target_mut || *source_mut)
                    && self.is_assignable(target_inner, source_inner)
                {
                    return true;
                }
            }
        }

        if let Type::Borrow {
            inner: target_inner,
            mem_space: target_mem,
            is_mut: target_mut,
            ..
        } = target
        {
            if let Type::Borrow {
                inner: source_inner,
                mem_space: source_mem,
                is_mut: source_mut,
                ..
            } = source
            {
                if target_mem == source_mem
                    && (!*target_mut || *source_mut)
                    && self.is_assignable(target_inner, source_inner)
                {
                    // Hook up 256-bit FastPath Borrow Checker algorithm from src/borrow.rs
                    let id_target = self.lower_to_type_id(target);
                    let id_source = self.lower_to_type_id(source);
                    if crate::borrow::verify_subtyping_bounds(&id_source, &id_target, self.worker) {
                        return true;
                    }
                }
            }
        }

        if let Type::Pointer(target_inner, target_mem, target_mut) = target {
            if let Type::Pointer(source_inner, source_mem, source_mut) = source {
                if target_mem == source_mem && (!*target_mut || *source_mut) {
                    if let Type::Scalar(ElementType::I8) = &**source_inner {
                        return true; // allow casting *mut i8 (void*) to any pointer
                    }
                    if let Type::Scalar(ElementType::I8) = &**target_inner {
                        return true; // allow casting any pointer to *mut i8 (void*)
                    }
                    if self.is_assignable(target_inner, source_inner) {
                        return true;
                    }
                }
            }
        }

        false
    }
    /// The `(element, dims)` of the tensor at the core of a (possibly wrapped) type, if any.
    pub(crate) fn tensor_of(ty: &Type) -> Option<(&ElementType, &[Dim])> {
        match ty {
            Type::Tensor(e, d, _) => Some((e, d.as_slice())),
            Type::Ref(inner, _) | Type::Pinned(inner, _) | Type::Verified(inner) => {
                Self::tensor_of(inner)
            }
            _ => None,
        }
    }
}
