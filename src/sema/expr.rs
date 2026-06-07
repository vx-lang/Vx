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

use std::collections::HashMap;

use super::*;

impl<'a> TypeChecker<'a> {
    pub(crate) fn check_expr(&mut self, expr: &mut Expr) -> (Type, u32) {
        // First perform semantic validation silently for HIR lowering
        let ty = self.check_expr_type_flag(expr, false, true);

        // Then perform AST to HIR lowering
        let type_idx = self.emit_type(&ty);

        // Determine opcode based on the AST expression
        let opcode = match expr {
            Expr::Number(NumberExpr { .. }) => crate::hir::OP_CONST,
            Expr::Identifier(IdentifierExpr { name: _, span: _ }) => crate::hir::OP_LOAD,
            Expr::BinaryOp(BinaryOpExpr {
                lhs: _,
                op,
                rhs: _,
                span: _,
            }) => match op {
                BinaryOp::Add => crate::hir::OP_ADD,
                BinaryOp::Sub => crate::hir::OP_SUB,
                BinaryOp::Mul => crate::hir::OP_MUL,
                BinaryOp::MatMul => crate::hir::OP_MATMUL,
                BinaryOp::Div => crate::hir::OP_DIV,
            },
            Expr::RelationalOp(RelationalOpExpr { .. }) => crate::hir::OP_NOP,
            Expr::LogicalOp(LogicalOpExpr { .. }) => crate::hir::OP_NOP,
            Expr::FunctionCall(FunctionCallExpr {
                name: _,
                args: _,
                span: _,
            }) => crate::hir::OP_CALL,
            _ => crate::hir::OP_NOP,
        };

        // In a full implementation, we would recursively call check_expr here
        // to get operand registers. For this bridge proof-of-concept, we emit
        // dummy operands and assign the result register.
        let reg = self.emit_inst(opcode, 0, 0, type_idx);
        (ty, reg)
    }

    pub fn check_expr_type(&mut self, expr: &mut Expr) -> Type {
        self.check_expr_type_flag(expr, true, false)
    }

    pub(crate) fn check_expr_block(
        &mut self,
        stmts: &mut [Statement],
        consume: bool,
        silent: bool,
    ) -> Type {
        let mut ret_ty = Type::Tensor(ElementType::F32, vec![], None);
        let mut terminated = false;

        for s in stmts.iter_mut() {
            if terminated && !silent {
                self.errors
                    .push_warning("Unreachable code after return, break, or continue".to_string());
                break;
            }

            if let Statement::ExprStmt(ExprStmtStmt {
                ref mut expr,
                has_semi: _,
                span: _,
            }) = s
            {
                let saved_borrows = self.active_borrows.clone();
                ret_ty = self.check_expr_type_flag(expr, consume, silent);
                self.active_borrows = saved_borrows;
            } else {
                let expected_ret = self.current_return_type.clone().unwrap_or(Type::Tensor(
                    ElementType::F32,
                    vec![],
                    None,
                ));
                self.check_statement(s, &expected_ret, consume, silent);
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

    pub fn check_expr_type_flag(&mut self, expr: &mut Expr, consume: bool, silent: bool) -> Type {
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
                        enum_name: enum_name.to_string(),
                        variant_name: variant.to_string(),
                        payload,
                        span: fc.span.clone(),
                    });
                }
            }
        }

        match expr {
            Expr::Identifier(..) => self.check_identifier_expr(expr, consume, silent),
            Expr::EnumVariant(..) => self.check_enumvariant_expr(expr, consume, silent),
            Expr::Number(NumberExpr {
                value: _,
                ty: Some(el_ty),
                span: _,
            }) => Type::Scalar(el_ty.clone()),
            Expr::Number(NumberExpr {
                value: _,
                ty: None,
                span: _,
            }) => Type::Scalar(ElementType::F32),
            Expr::StringLiteral(StringLiteralExpr { .. }) => Type::Pointer(
                Box::new(Type::Scalar(ElementType::I8)),
                None,
                false, // const
            ),
            Expr::Transfer(..) => self.check_transfer_expr(expr, consume, silent),
            Expr::ComptimeBlock(..) => self.check_comptimeblock_expr(expr, consume, silent),
            Expr::SpawnOn(..) => self.check_spawnon_expr(expr, consume, silent),
            Expr::If(..) => self.check_if_expr(expr, consume, silent),
            Expr::SizeOf(..) => Type::Scalar(ElementType::I64),
            Expr::FunctionCall(..) => self.check_functioncall_expr(expr, consume, silent),
            Expr::IndirectCall(..) => self.check_indirectcall_expr(expr, consume, silent),
            Expr::Array(..) => self.check_array_expr(expr, silent),
            Expr::MemberAccess(..) => self.check_memberaccess_expr(expr, silent),
            Expr::IndexAccess(..) => self.check_indexaccess_expr(expr, silent),
            Expr::MethodCall(..) => self.check_methodcall_expr(expr, consume, silent),
            Expr::BinaryOp(..) => self.check_binaryop_expr(expr, consume, silent),
            Expr::RelationalOp(..) => self.check_relationalop_expr(expr, silent),
            Expr::LogicalOp(..) => self.check_logicalop_expr(expr, silent),
            Expr::MemorySpace(MemorySpaceExpr { .. }) => {
                Type::Tensor(ElementType::F32, vec![], None)
            }
            Expr::Topology(TopologyExpr { top, span: _ }) => {
                if matches!(top, Topology::Current) {
                    *top = self.active_topology.clone();
                }
                Type::Tensor(ElementType::F32, vec![], None)
            }
            Expr::UnaryOp(..) => self.check_unaryop_expr(expr, silent),
            Expr::Borrow(..) => self.check_borrow_expr(expr, silent),
            Expr::Dereference(..) => self.check_dereference_expr(expr, consume, silent),
            Expr::UnsafeBlock(..) => self.check_unsafeblock_expr(expr, consume, silent),
            Expr::StructInit(..) => self.check_structinit_expr(expr, consume, silent),
            Expr::Grad(..) => self.check_grad_expr(expr, silent),
            Expr::Vjp(..) => self.check_vjp_expr(expr, silent),
            Expr::Jvp(..) => self.check_jvp_expr(expr, silent),
            Expr::Range(..) => self.check_range_expr(expr, silent),
            Expr::Match(..) => self.check_match_expr(expr, consume, silent),
            Expr::VecMacro(..) => self.check_vecmacro_expr(expr, silent),
            Expr::Closure(..) => self.check_closure_expr(expr, consume, silent),
            Expr::AsCast(e) => self.check_ascast_expr(e, consume, silent),
            Expr::Print(p) => {
                for arg in &mut p.args {
                    self.check_expr_type_flag(arg, consume, silent);
                }
                Type::Scalar(ElementType::I32) // Assuming print returns 0 as i32 for C compatibility
            }
            Expr::Println(p) => {
                for arg in &mut p.args {
                    self.check_expr_type_flag(arg, consume, silent);
                }
                Type::Scalar(ElementType::I32)
            }
            Expr::InlineMlir(e) => {
                // Typecheck inputs
                for (_, arg_expr, _) in &mut e.inputs {
                    self.check_expr_type_flag(arg_expr, consume, silent);
                }
                // Typecheck clobbers and mark them as mutated if needed
                for clobber in &mut e.clobbers {
                    self.check_expr_type_flag(clobber, consume, silent);
                }

                // Return the specified type or Unknown if void
                e.returns.clone().unwrap_or(Type::Unknown)
            }
            Expr::MacroCall(m) => panic!(
                "Macros should be expanded before type checking: macro `{}` at {:?}",
                m.name, m.span
            ),
        }
    }

    pub(crate) fn check_differentiability(&mut self, func: &Function) {
        match &func.return_type {
            Type::Tensor(_, _, _) | Type::Scalar(_) | Type::Simd(_, _) => {}
            _ => {
                self.errors.push(format!("Function '{}' cannot be differentiated because it returns a non-continuous type: {:?}", func.name, func.return_type));
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
            Type::Borrow(_inner, _mem, _is_mut, region) => {
                // The lifetime of the borrow itself is Covariant (even for mutable borrows,
                // which allows reborrowing for shorter lifetimes during function calls).
                // (The inner type T would be invariant for mutable borrows, but we are
                // only hashing the outer lifetime here).
                let variance: u8 = 0x1;

                // Pack the region and variance directly into Param 0 of the FastPath hash!
                // We use standard try_set_fast_param to pack the 16 bits.
                if let Err(e) = id.try_set_fast_param(0, *region as u16, variance) {
                    // If we exceed 4095 lexical scopes, we log but continue safely with max
                    // In a production compiler, this would trigger the SlowPath allocation.
                    println!("Warning: Region overflow during lowering: {}", e);
                    let _ = id.try_set_fast_param(0, 4095, variance);
                }
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

    pub(crate) fn is_assignable(&self, target: &Type, source: &Type) -> bool {
        if target == source {
            return true;
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

        if let Type::GenericInstance(inner_target, _args_target) = target {
            if let Type::Enum(n_source, _) = source {
                if let Type::Struct(n_target, _) = &**inner_target {
                    if n_source.starts_with(n_target) && n_source.contains('<') {
                        return true; // Weak check for Option<T>
                    }
                }
            }
        }

        if let Type::GenericInstance(inner_source, _args_source) = source {
            if let Type::Enum(n_target, _) = target {
                if let Type::Struct(n_source, _) = &**inner_source {
                    if n_target.starts_with(n_source) && n_target.contains('<') {
                        return true; // Weak check for Option<T>
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
                } else if *t_source == ElementType::F32 && t_target != &ElementType::Bool {
                    // Literals currently parse as f32, so we allow f32 to coerce
                    el_match = true;
                }

                if !el_match {
                    return false;
                }

                if top_target.is_some() && top_target != top_source {
                    return false;
                }

                if !dims_target.is_empty() && !dims_source.is_empty() {
                    if dims_target.len() != dims_source.len() {
                        return false;
                    }
                    let empty_env = std::collections::HashMap::new();
                    for (dt, ds) in dims_target.iter().zip(dims_source.iter()) {
                        let vt = self.eval_expr(dt, &empty_env);
                        let vs = self.eval_expr(ds, &empty_env);
                        if vt.is_some() && vs.is_some() {
                            if vt != vs {
                                return false;
                            }
                        } else if dt != ds {
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
                if *t_target == *t_source {
                    return true;
                }
                // Allow numeric coercions
                if *t_target != ElementType::Bool && t_source != &ElementType::Bool {
                    return true;
                }
            }
        }

        // Allow coercing Scalar to Tensor (e.g. 0.0 to Tensor<f32>) for backwards compatibility with tests
        if let Type::Tensor(t_target, _, _) = target {
            if let Type::Scalar(t_source) = &source {
                if *t_target == *t_source {
                    return true;
                }
                if *t_source != ElementType::Bool && t_target != &ElementType::Bool {
                    return true;
                }
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
            if let Type::Borrow(source_inner, source_mem, source_mut, _source_region) = source {
                if target_mem == source_mem
                    && (!*target_mut || *source_mut)
                    && self.is_assignable(target_inner, source_inner)
                {
                    return true;
                }
            }
        }

        if let Type::Borrow(target_inner, target_mem, target_mut, _target_region) = target {
            if let Type::Borrow(source_inner, source_mem, source_mut, _source_region) = source {
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
    fn check_identifier_expr(&mut self, expr: &mut Expr, consume: bool, silent: bool) -> Type {
        match expr {
            Expr::Identifier(IdentifierExpr { name, span: _ }) => {
                if name == "true" || name == "false" {
                    return Type::Scalar(ElementType::Bool);
                }

                if let Some(borrows) = self.active_borrows.get(name) {
                    for b in borrows {
                        if b.is_mut && !silent {
                            self.errors.push(format!(
                                "Cannot access '{}' because it is mutably borrowed.",
                                name
                            ));
                            break;
                        }
                    }
                }
                let lookup_depth_res = self
                    .lookup_with_depth(name)
                    .map(|(ty, top, d)| (ty.clone(), top.clone(), d));
                let lookup_res = lookup_depth_res
                    .as_ref()
                    .map(|(ty, top, _)| (ty.clone(), top.clone()));

                if let Some((ty, _, depth)) = &lookup_depth_res {
                    // If we are inside a closure and the variable is defined outside of it,
                    // we must capture it in all closures between the definition and usage.
                    for (i, closure_depth) in self.closure_depths.iter().enumerate() {
                        if depth < closure_depth {
                            self.closure_captures_stack[i].insert(name.clone(), ty.clone());
                        }
                    }
                }
                if name == "new_item" {
                    println!(
                        "lookup('new_item') = {:?}, is_moved = {}, consume = {}",
                        lookup_res,
                        self.is_moved(name),
                        consume
                    );
                    println!("scopes = {:?}", self.scopes.last());
                    println!("Backtrace:");
                    let bt = std::backtrace::Backtrace::force_capture();
                    println!("{}", bt);
                }

                if lookup_res.is_none() && self.is_moved(name) {
                    if !silent {
                        self.errors.push(format!(
                            "Use of moved or consumed linear variable: {}",
                            name
                        ));
                    }
                    return Type::Tensor(ElementType::F32, vec![], None);
                } else if lookup_res.is_none() {
                    if let Some((ret_ty, _, params, _, _, _)) = self.env.functions.get(name) {
                        return Type::Function(params.clone(), Box::new(ret_ty.clone()));
                    }
                    for (func, _) in &self.monomorphized_functions {
                        if &func.name == name {
                            let params = func.params.iter().map(|(_, t)| t.clone()).collect();
                            return Type::Function(params, Box::new(func.return_type.clone()));
                        }
                    }

                    if !silent {
                        self.errors.push(format!("Undefined variable '{}'", name));
                    }
                    return Type::Scalar(ElementType::F32); // Fallback to prevent panic
                }

                match lookup_res {
                    Some((ty, top)) => {
                        if consume && ty.is_linear() && !silent {
                            self.consume(name);
                        }

                        // Enforce Topology Boundaries!
                        let is_valid = self.hardware_graph.is_type_accessible(
                            &self.active_topology,
                            &top,
                            &ty,
                        );

                        if !is_valid {
                            let is_pinned_on_host = matches!(ty, Type::Pinned(_, _))
                                && matches!(self.active_topology, Topology::Host);
                            if !is_pinned_on_host && !silent {
                                let msg = format!(
                                                "Cross-topology access error: Variable '{}' belongs to {:?} (type: {:?}), but accessed from {:?}",
                                                name, top, ty, self.active_topology
                                            );
                                self.errors.push(msg);
                            }
                        }
                        ty.clone()
                    }
                    None => {
                        if !silent {
                            let msg = format!("Undefined variable '{}'", name);
                            self.errors.push(msg);
                        }
                        Type::Tensor(ElementType::F32, vec![], None) // Default placeholder on error
                    }
                }
            }
            _ => unreachable!(),
        }
    }

    fn check_enumvariant_expr(&mut self, expr: &mut Expr, consume: bool, silent: bool) -> Type {
        match expr {
            Expr::EnumVariant(EnumVariantExpr {
                enum_name,
                variant_name: variant,
                payload,
                span: _,
            }) => {
                let actual_enum_name = if let Some(idx) = enum_name.find('<') {
                    &enum_name[..idx]
                } else {
                    enum_name.as_str()
                };

                if let Some(enum_decl) = self.env.enums.get(actual_enum_name) {
                    if let Some((_, expected_payload)) =
                        enum_decl.variants.iter().find(|(n, _)| n == variant)
                    {
                        if let Some(expr_payload) = payload {
                            if let Some(exp_types) = expected_payload {
                                if expr_payload.len() != exp_types.len() {
                                    if !silent {
                                        self.errors.push(format!("Enum variant {}::{} expects {} payload arguments, got {}", actual_enum_name, variant, exp_types.len(), expr_payload.len()));
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
                                                    _ => Type::Struct(ty_arg.to_string(), None),
                                                };
                                                mapping.insert(param.name().to_string(), parsed_ty);
                                            }
                                        }
                                    }

                                    for (i, expr) in expr_payload.iter_mut().enumerate() {
                                        let expr_ty =
                                            self.check_expr_type_flag(expr, consume, silent);
                                        let expected_ty = exp_types[i].substitute(&mapping);
                                        if !self.is_assignable(&expected_ty, &expr_ty)
                                            && !matches!(&expected_ty, Type::Generic(_, _))
                                            && !silent
                                        {
                                            self.errors.push(format!("Type mismatch in payload argument {} for {}::{}: expected {:?}, got {:?}", i + 1, actual_enum_name, variant, expected_ty, expr_ty));
                                        }
                                    }
                                }
                            } else if !silent {
                                self.errors.push(format!(
                                    "Enum variant {}::{} does not take a payload",
                                    actual_enum_name, variant
                                ));
                            }
                        } else if expected_payload.is_some() && !silent {
                            self.errors.push(format!(
                                "Enum variant {}::{} expects a payload",
                                actual_enum_name, variant
                            ));
                        }
                    } else if !silent {
                        self.errors.push(format!(
                            "Enum {} does not have variant {}",
                            actual_enum_name, variant
                        ));
                    }
                } else if !silent {
                    self.errors.push(format!("Unknown enum {}", enum_name));
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
                            _ => Type::Struct(ty_arg.to_string(), None),
                        };
                        return Type::GenericInstance(
                            Box::new(Type::Struct(base.to_string(), None)),
                            vec![parsed_ty],
                        );
                    }
                }
                Type::Enum(enum_name.clone(), None)
            }
            _ => unreachable!(),
        }
    }

    fn check_ascast_expr(&mut self, expr: &mut AsCastExpr, consume: bool, silent: bool) -> Type {
        let source_ty = self.check_expr_type_flag(&mut expr.expr, consume, silent);
        let target_ty = expr.target_ty.clone();
        expr.source_ty = Some(source_ty.clone());

        match (&source_ty, &target_ty) {
            (Type::Struct(name, _), Type::Closure(target_args, target_ret))
                if name.starts_with("Closure_") =>
            {
                let call_method_name = format!("{}_call", name);

                let mut found_func = None;
                if let Some(func_type) = self.env.functions.get(&call_method_name) {
                    found_func = Some(func_type.0.clone());
                } else {
                    for (func, _) in &self.monomorphized_functions {
                        if func.name == call_method_name {
                            let params = func.params.iter().map(|(_, t)| t.clone()).collect();
                            found_func =
                                Some(Type::Function(params, Box::new(func.return_type.clone())));
                            break;
                        }
                    }
                }

                if let Some(func_type) = found_func {
                    let Type::Function(func_args, func_ret) = func_type else {
                        unreachable!()
                    };
                    let mut args_match = func_args.len() == target_args.len() + 1;
                    if args_match {
                        for (i, target_arg) in target_args.iter().enumerate() {
                            if target_arg != &func_args[i + 1] {
                                args_match = false;
                                break;
                            }
                        }
                    }

                    if !args_match || target_ret.as_ref() != func_ret.as_ref() {
                        self.errors.push(format!(
                                "Closure cast signature mismatch. Expected {:?} but found a function with args {:?} and ret {:?}",
                                target_ty, func_args, func_ret
                            ));
                        return Type::Unknown;
                    }

                    return target_ty;
                } else {
                    self.errors.push(format!(
                        "Closure call method '{}' not found for cast.",
                        call_method_name
                    ));
                    return Type::Unknown;
                }
            }
            (Type::Scalar(_), Type::Scalar(_)) => {
                return target_ty;
            }
            _ => {}
        }

        self.errors.push(format!(
            "Unsupported cast: cannot cast from {:?} to {:?}",
            source_ty, target_ty
        ));
        Type::Unknown
    }

    fn check_transfer_expr(&mut self, expr: &mut Expr, consume: bool, silent: bool) -> Type {
        match expr {
            Expr::Transfer(TransferExpr {
                expr: inner_expr,
                space: target_mem,
                cost: ref mut expr_cost,
                span: _,
            }) => {
                let inner_ty = self.check_expr_type_flag(inner_expr, false, silent);

                // Extract source memory space, default to HostDRAM if it's not explicitly a Ref
                let source_mem = match &inner_ty {
                    Type::Ref(_, mem) => mem.clone(),
                    Type::Pinned(_, top) => crate::arch::HardwareGraph::default_memory_for(top),
                    _ => MemorySpace::HostDRAM,
                };

                let calculated_cost = self.hardware_graph.transfer_cost(&source_mem, target_mem);
                if calculated_cost.is_none() {
                    if !silent {
                        self.errors.push(format!(
                            "Cannot transfer from {:?} to {:?}: no hardware path exists",
                            source_mem, target_mem
                        ));
                    }
                    return Type::Tensor(ElementType::F32, vec![], None);
                }
                *expr_cost = calculated_cost;

                match inner_ty {
                    Type::Ref(base_ty, _) => Type::Ref(base_ty, target_mem.clone()),
                    Type::Tensor(_, _, _) => {
                        let pinned_top = match &target_mem {
                            MemorySpace::NPUHBM => {
                                Topology::NPU(Box::new(Expr::Number(NumberExpr {
                                    value: "0".to_string(),
                                    ty: Some(ElementType::I32),
                                    span: Span::default(),
                                })))
                            }
                            MemorySpace::LocalSRAM => {
                                Topology::AccCore(Box::new(Expr::Number(NumberExpr {
                                    value: "0".to_string(),
                                    ty: Some(ElementType::I32),
                                    span: Span::default(),
                                })))
                            }
                            MemorySpace::HostDRAM => Topology::Host,
                        };
                        Type::Pinned(Box::new(inner_ty.clone()), pinned_top)
                    }
                    Type::Verified(_inner) => {
                        let inner_pinned = self.check_expr_type_flag(inner_expr, consume, silent);
                        Type::Verified(Box::new(inner_pinned))
                    }
                    Type::Pinned(base, _) => {
                        let pinned_top = match &target_mem {
                            MemorySpace::NPUHBM => {
                                Topology::NPU(Box::new(Expr::Number(NumberExpr {
                                    value: "0".to_string(),
                                    ty: Some(ElementType::I32),
                                    span: Span::default(),
                                })))
                            }
                            MemorySpace::LocalSRAM => {
                                Topology::AccCore(Box::new(Expr::Number(NumberExpr {
                                    value: "0".to_string(),
                                    ty: Some(ElementType::I32),
                                    span: Span::default(),
                                })))
                            }
                            MemorySpace::HostDRAM => Topology::Host,
                        };
                        Type::Pinned(base, pinned_top)
                    }
                    _ => {
                        if !silent {
                            self.errors.push(format!(
                                "Cannot transfer non-reference type: {:?}",
                                inner_ty
                            ));
                        }
                        Type::Tensor(ElementType::F32, vec![], None)
                    }
                }
            }

            _ => unreachable!(),
        }
    }

    fn check_comptimeblock_expr(&mut self, expr: &mut Expr, consume: bool, silent: bool) -> Type {
        match expr {
            Expr::ComptimeBlock(ComptimeBlockExpr {
                stmts,
                ret,
                span: _,
            }) => {
                self.push_scope();
                let mut ret_ty = self.check_expr_block(stmts, consume, silent);
                if let Some(r) = ret {
                    ret_ty = self.check_expr_type(r);
                }
                self.pop_scope();
                ret_ty
            }

            _ => unreachable!(),
        }
    }

    fn check_spawnon_expr(&mut self, expr: &mut Expr, consume: bool, silent: bool) -> Type {
        match expr {
            Expr::SpawnOn(SpawnOnExpr {
                top,
                stmts,
                ret,
                span: _,
            }) => {
                let mut actual_top = top.clone();
                if actual_top == Topology::Current {
                    actual_top = self.active_topology.clone();
                }

                let prev_top = self.active_topology.clone();
                let prev_mem = self.active_memory.clone();
                self.active_topology = actual_top.clone();
                self.active_memory = crate::arch::HardwareGraph::default_memory_for(&actual_top);

                *top = actual_top;

                self.push_scope();

                // Validate topology expression if it contains one
                match top {
                    Topology::NPU(expr) | Topology::AccCore(expr) => {
                        let _ty = self.check_expr_type(expr);
                    }
                    Topology::Slice(_, start, end) => {
                        let _t1 = self.check_expr_type(start);
                        let _t2 = self.check_expr_type(end);
                    }
                    Topology::Host
                    | Topology::AMX
                    | Topology::ANE
                    | Topology::GPU
                    | Topology::Host_AVX512
                    | Topology::Host_Neon
                    | Topology::Current => {}
                }

                self.check_expr_block(stmts, consume, silent);

                let mut ret_ty = Type::Tensor(ElementType::F32, vec![], None); // default void-like type
                if let Some(r) = ret {
                    ret_ty = self.check_expr_type_flag(r, consume, silent);
                }

                self.pop_scope();

                self.active_topology = prev_top;
                self.active_memory = prev_mem;

                ret_ty
            }
            _ => unreachable!(),
        }
    }

    fn check_if_expr(&mut self, expr: &mut Expr, consume: bool, silent: bool) -> Type {
        let if_expr = match expr {
            Expr::If(e) => e,
            _ => unreachable!(),
        };

        let cond_ty = self.check_expr_type(&mut if_expr.cond);
        if cond_ty != Type::Scalar(ElementType::Bool) {
            self.errors
                .push("Condition in if expression must be of type bool (i1)".to_string());
        }

        if if_expr.is_comptime {
            let mut tmp_env = std::collections::HashMap::new();
            for env in &self.eval_env {
                for (k, v) in env {
                    tmp_env.insert(k.clone(), v.clone());
                }
            }
            if let Some(Value::Bool(b)) = self.eval_expr(&if_expr.cond, &tmp_env) {
                if b {
                    if_expr.else_block = None;
                } else {
                    if_expr.then_block.clear();
                }
            } else {
                self.errors
                    .push("Cannot statically evaluate comptime if condition".to_string());
            }
        }

        self.push_scope();
        let mut then_ty = Type::Tensor(ElementType::F32, vec![], None);
        if !silent && !if_expr.then_block.is_empty() {
            then_ty = self.check_expr_block(&mut if_expr.then_block, consume, silent);
        }
        self.pop_scope();

        let mut else_ty = Type::Tensor(ElementType::F32, vec![], None);
        if let Some(else_b) = if_expr.else_block.as_mut() {
            if !else_b.is_empty() {
                self.push_scope();
                if !silent {
                    else_ty = self.check_expr_block(else_b, consume, silent);
                }
                self.pop_scope();

                if !if_expr.is_comptime && then_ty != else_ty {
                    self.errors.push(format!(
                        "If expression branches have incompatible types: {:?} and {:?}",
                        then_ty, else_ty
                    ));
                }
            }
        } else if !if_expr.is_comptime {
            // Without else block, it evaluates to unit (represented as dummy Tensor)
            then_ty = Type::Tensor(ElementType::F32, vec![], None);
        }

        // If it was comptime evaluated to false, the return type should just be the else block type
        if if_expr.is_comptime {
            if if_expr.then_block.is_empty() {
                return else_ty;
            } else {
                return then_ty;
            }
        }

        then_ty
    }

    fn check_indirectcall_expr(&mut self, expr: &mut Expr, consume: bool, silent: bool) -> Type {
        let (callee, args) = match expr {
            Expr::IndirectCall(c) => (&mut c.callee, &mut c.args),
            _ => unreachable!(),
        };
        let callee_ty = self.check_expr_type_flag(callee, consume, silent);
        let mut arg_types = Vec::new();
        for arg in args.iter_mut() {
            arg_types.push(self.check_expr_type_flag(arg, consume, silent));
        }

        if let Type::Struct(struct_name, _) = callee_ty {
            if struct_name.starts_with("Closure_") {
                let call_name = format!("{}_call", struct_name);
                if let Some(func) = self
                    .monomorphized_functions
                    .iter()
                    .find(|f| f.0.name == call_name)
                {
                    let param_types: Vec<Type> = func
                        .0
                        .params
                        .iter()
                        .skip(1)
                        .map(|(_, t)| t.clone())
                        .collect();
                    if args.len() != param_types.len() {
                        if !silent {
                            self.errors.push(format!(
                                "Closure expects {} arguments, got {}",
                                param_types.len(),
                                args.len()
                            ));
                        }
                    } else {
                        for (i, param_ty) in param_types.iter().enumerate() {
                            let arg_ty = &arg_types[i];
                            if !self.is_assignable(param_ty, arg_ty) && !silent {
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
                        name: call_name,
                        args: new_args,
                        span: Span::default(),
                    });

                    return func.0.return_type.clone();
                } else {
                    if !silent {
                        self.errors.push(format!(
                            "Missing call method for closure struct '{}'",
                            struct_name
                        ));
                    }
                }
            } else {
                if !silent {
                    self.errors
                        .push(format!("Cannot call non-closure struct '{}'", struct_name));
                }
            }
        } else if let Type::Closure(param_types, ret_ty) = &callee_ty {
            if args.len() != param_types.len() {
                if !silent {
                    self.errors.push(format!(
                        "Closure fat pointer expects {} arguments, got {}",
                        param_types.len(),
                        args.len()
                    ));
                }
            } else {
                for (i, param_ty) in param_types.iter().enumerate() {
                    let arg_ty = &arg_types[i];
                    if !self.is_assignable(param_ty, arg_ty) && !silent {
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
            if !silent {
                self.errors.push(
                    "Function pointers are not natively callable yet; use closure interfaces."
                        .to_string(),
                );
            }
        } else {
            if !silent {
                self.errors
                    .push(format!("Cannot call expression of type {:?}", callee_ty));
            }
        }

        Type::Tensor(ElementType::F32, vec![], None)
    }

    fn check_functioncall_expr(&mut self, expr: &mut Expr, consume: bool, silent: bool) -> Type {
        match expr {
            Expr::FunctionCall(FunctionCallExpr {
                name,
                args,
                span: _,
            }) => {
                let resolved_name = name.clone();
                let mut base_name = resolved_name.clone();
                let mut explicit_generic_args = Vec::new();
                if let Some(idx) = resolved_name.find('<') {
                    if resolved_name.ends_with('>') {
                        base_name = resolved_name[..idx].to_string();
                        let args_str = &resolved_name[idx + 1..resolved_name.len() - 1];
                        explicit_generic_args = args_str
                            .split(',')
                            .map(|s| self.parse_ty_str(s.trim()))
                            .collect();
                    }
                }

                // Mocking built-ins
                let mut arg_types = Vec::new();
                let is_builtin_ref = resolved_name == "print" || resolved_name == "Verified";
                let arg_consume = if is_builtin_ref { false } else { consume };
                for arg in args.iter_mut() {
                    arg_types.push(self.check_expr_type_flag(arg, arg_consume, silent));
                }

                if resolved_name == "Verified" {
                    if args.len() != 1 {
                        self.errors.push(format!(
                            "Function 'Verified' expects 1 argument, got {}",
                            args.len()
                        ));
                    }
                    let inner_ty = arg_types[0].clone();
                    Type::Verified(Box::new(inner_ty))
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
                    // Types already checked
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
                    Type::Tensor(el_ty, dims, None)
                } else if resolved_name.starts_with("Tensor") && !resolved_name.contains("__") {
                    let el_ty = match resolved_name.as_str() {
                        "Tensor_f64" => ElementType::F64,
                        "Tensor_bf16" => ElementType::BF16,
                        "Tensor_i32" => ElementType::I32,
                        "Tensor_i64" => ElementType::I64,
                        _ => {
                            if resolved_name.starts_with("Tensor_") {
                                let t_name = resolved_name.strip_prefix("Tensor_").unwrap();
                                if t_name != "f32" {
                                    ElementType::Generic(t_name.to_string())
                                } else {
                                    ElementType::F32
                                }
                            } else {
                                ElementType::F32
                            }
                        }
                    };
                    let mut dims = Vec::new();
                    if !args.is_empty() {
                        if let Expr::Array(arr) = &args[0] {
                            dims = arr.elements.clone();
                        } else {
                            dims = args.clone();
                        }
                    }
                    Type::Tensor(el_ty, dims, None)
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
                    Type::Scalar(ElementType::F32)
                } else if resolved_name == "print" {
                    if args.len() != 1 {
                        self.errors
                            .push("Function 'print' expects 1 argument".to_string());
                    }
                    Type::Tensor(ElementType::F32, vec![], None)
                } else if resolved_name == "printf" || resolved_name == "vx_internal_printf" {
                    if args.is_empty() {
                        self.errors
                            .push("Function 'printf' expects at least 1 argument".to_string());
                    }
                    Type::Scalar(ElementType::I32)
                } else if let Some((Type::Function(param_types, ret_ty), _)) =
                    self.lookup(&resolved_name).cloned()
                {
                    if args.len() != param_types.len() && !silent {
                        self.errors.push(format!(
                            "Function pointer '{}' expects {} arguments, got {}",
                            resolved_name,
                            param_types.len(),
                            args.len()
                        ));
                    } else {
                        for (i, param_ty) in param_types.iter().enumerate() {
                            let arg_ty = &arg_types[i];
                            if !self.is_assignable(param_ty, arg_ty) && !silent {
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
                    if args.len() != param_types.len() && !silent {
                        self.errors.push(format!(
                            "Closure '{}' expects {} arguments, got {}",
                            resolved_name,
                            param_types.len(),
                            args.len()
                        ));
                    } else {
                        for (i, param_ty) in param_types.iter().enumerate() {
                            let arg_ty = &arg_types[i];
                            if !self.is_assignable(param_ty, arg_ty) && !silent {
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
                            .find(|f| f.0.name == call_name)
                        {
                            let param_types: Vec<Type> = func
                                .0
                                .params
                                .iter()
                                .skip(1)
                                .map(|(_, t)| t.clone())
                                .collect();
                            if args.len() != param_types.len() {
                                if !silent {
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
                                    if !self.is_assignable(param_ty, arg_ty) && !silent {
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
                                name: call_name,
                                args: new_args,
                                span: Span::default(),
                            });

                            func.0.return_type.clone()
                        } else {
                            if !silent {
                                self.errors.push(format!(
                                    "Missing call method for closure struct '{}'",
                                    struct_name
                                ));
                            }
                            Type::Tensor(ElementType::F32, vec![], None)
                        }
                    } else {
                        if !silent {
                            self.errors.push(format!(
                                "Cannot call non-closure struct '{}'",
                                resolved_name
                            ));
                        }
                        Type::Tensor(ElementType::F32, vec![], None)
                    }
                } else if let Some((ret_ty, is_unsafe, param_types, req_topology, _, _)) =
                    self.env.functions.get(&resolved_name)
                {
                    if (*req_topology != self.active_topology) && !silent {
                        self.errors.push(format!(
                                        "Type error: Function '{}' requires topology '{:?}', but is called from '{:?}'",
                                        resolved_name, req_topology, self.active_topology
                                    ));
                    }
                    if *is_unsafe && !self.in_unsafe_block && !silent {
                        self.errors.push(format!("Call to unsafe function '{}' is unsafe and requires unsafe function or block", resolved_name));
                    }
                    if args.len() != param_types.len() && !silent {
                        self.errors.push(format!(
                            "Function '{}' expects {} arguments, got {}",
                            resolved_name,
                            param_types.len(),
                            args.len()
                        ));
                    } else {
                        for (i, param_ty) in param_types.iter().enumerate() {
                            let arg_ty = &arg_types[i];
                            if !self.is_assignable(param_ty, arg_ty) && !silent {
                                self.errors.push(format!(
                                        "Type mismatch in argument {} for function '{}'. Expected {:?}, got {:?}",
                                        i + 1, resolved_name, param_ty, arg_ty
                                    ));
                            }
                        }
                    }
                    ret_ty.clone()
                } else if let Some(func) = self
                    .monomorphized_functions
                    .iter()
                    .find(|f| f.0.name == resolved_name)
                {
                    if (func.0.topology != self.active_topology) && !silent {
                        self.errors.push(format!(
                                "Type error: Function '{}' requires topology '{:?}', but is called from '{:?}'",
                                resolved_name, func.0.topology, self.active_topology
                            ));
                    }
                    let param_types: Vec<Type> =
                        func.0.params.iter().map(|(_, t)| t.clone()).collect();
                    if args.len() != param_types.len() {
                        if !silent {
                            self.errors.push(format!(
                                "Function '{}' expects {} arguments, got {}",
                                resolved_name,
                                param_types.len(),
                                args.len()
                            ));
                        }
                    } else {
                        for (i, param_ty) in param_types.iter().enumerate() {
                            let arg_ty = &arg_types[i];
                            if !self.is_assignable(param_ty, arg_ty) && !silent {
                                self.errors.push(format!(
                                        "Type mismatch in argument {} for function '{}'. Expected {:?}, got {:?}",
                                        i + 1, resolved_name, param_ty, arg_ty
                                    ));
                            }
                        }
                    }
                    func.0.return_type.clone()
                } else if let Some((generic_func, origin_hash)) =
                    self.env.generic_functions.get(&base_name).cloned()
                {
                    // Type deduction
                    let mut mapping = HashMap::new();
                    let mut success = true;
                    if args.len() != generic_func.params.len() {
                        if !silent {
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
                                mapping.insert(
                                    param.name().to_string(),
                                    explicit_generic_args[i].clone(),
                                );
                            }
                        }
                        for (i, _arg) in args.iter_mut().enumerate() {
                            let arg_ty = arg_types[i].clone();
                            let param_ty = &generic_func.params[i].1;
                            if !self.unify_types(param_ty, &arg_ty, &mut mapping) {
                                if !silent {
                                    self.errors.push(format!("Failed to deduce types for generic function '{}': Expected {:?}, got {:?}", name, param_ty, arg_ty));
                                }
                                success = false;
                            }
                        }
                    }

                    if success {
                        // Trait Bounds Checking
                        for param in &generic_func.generics {
                            let g_name = param.name().to_string();
                            let bound_opt = match param {
                                decl::GenericParam::Type { bound, .. } => bound.clone(),
                                _ => None,
                            };
                            if let Some(bound_name) = bound_opt {
                                if let Some(concrete_ty) = mapping.get(&g_name) {
                                    let mut implements_trait = false;
                                    if let Some(impl_blocks) = self.env.impls.get(&bound_name) {
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
                                        if !silent {
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
                        // Instantiate
                        let mut inst_func = self.instantiate_function(generic_func, &mapping);
                        let inst_ret = inst_func.return_type.clone();
                        let inst_name = inst_func.name.clone();

                        // Rewrite AST name
                        *name = inst_name.clone();

                        if !self.env.functions.contains_key(&inst_name)
                            && !self
                                .monomorphized_functions
                                .iter()
                                .any(|(f, _)| f.name == inst_name)
                        {
                            // self.env is immutable, monomorphization tracks functions internally
                            self.check_function(&mut inst_func);
                            self.monomorphized_functions.push((inst_func, origin_hash));
                        }
                        inst_ret
                    } else {
                        Type::Tensor(ElementType::F32, vec![], None)
                    }
                } else if let Some(idx) = resolved_name.find("::") {
                    let mut struct_name = resolved_name[..idx].to_string();
                    let mut method_name = resolved_name[idx + 2..].to_string();
                    let mut explicit_ty_str = String::new();

                    if let Some(lt) = struct_name.find('<') {
                        if struct_name.ends_with('>') {
                            explicit_ty_str =
                                struct_name[lt + 1..struct_name.len() - 1].to_string();
                            struct_name = struct_name[..lt].to_string();
                        }
                    }

                    if let Some(lt) = method_name.find('<') {
                        if method_name.ends_with('>') {
                            let method_ty_str =
                                method_name[lt + 1..method_name.len() - 1].to_string();
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
                                if &struct_name == n {
                                    matches = true;
                                }
                            } else if let Type::Enum(n, _) = &ib.target_type {
                                if &struct_name == n {
                                    matches = true;
                                }
                            } else if let Type::Generic(n, _) = &ib.target_type {
                                if &struct_name == n {
                                    matches = true;
                                }
                            } else if let Type::GenericInstance(inner, _) = &ib.target_type {
                                if let Type::Struct(n, _) = &**inner {
                                    if &struct_name == n {
                                        matches = true;
                                    }
                                } else if let Type::Enum(n, _) = &**inner {
                                    if &struct_name == n {
                                        matches = true;
                                    }
                                }
                            }

                            if matches {
                                for m in &ib.methods {
                                    if m.name == method_name {
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

                                            for (i, parsed_ty) in
                                                explicit_args.into_iter().enumerate()
                                            {
                                                if i < ib.generics.len() {
                                                    found_mapping.insert(
                                                        ib.generics[i].name().to_string(),
                                                        parsed_ty,
                                                    );
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
                        modified_func.name = format!("{}::{}", struct_name, method_name);
                        modified_func.generics = found_mapping
                            .keys()
                            .map(|k| decl::GenericParam::Type {
                                name: k.clone(),
                                bound: None,
                            })
                            .collect();

                        let mut inst_func =
                            self.instantiate_function(&modified_func, &found_mapping);
                        let inst_ret = inst_func.return_type.clone();
                        let inst_name = inst_func.name.clone();

                        *name = inst_name.clone();

                        if !self.env.functions.contains_key(&inst_name)
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
                        if !silent {
                            self.errors
                                .push(format!("Undefined static method '{}'.", resolved_name));
                        }
                        Type::Tensor(ElementType::F32, vec![], None)
                    }
                } else {
                    let mono_names: Vec<String> = self
                        .monomorphized_functions
                        .iter()
                        .map(|(f, _)| f.name.clone())
                        .collect();
                    if !silent {
                        self.errors.push(format!(
                            "Undefined function '{}'. Available monos: {:?}",
                            resolved_name, mono_names
                        ));
                    }
                    Type::Tensor(ElementType::F32, vec![], None)
                }
            }
            _ => unreachable!(),
        }
    }

    fn check_array_expr(&mut self, expr: &mut Expr, _silent: bool) -> Type {
        match expr {
            Expr::Array(ArrayExpr { elements, span: _ }) => {
                for el in elements {
                    self.check_expr_type(el);
                }
                Type::Tensor(ElementType::F32, vec![], None)
            }
            _ => unreachable!(),
        }
    }

    fn check_memberaccess_expr(&mut self, expr: &mut Expr, silent: bool) -> Type {
        match expr {
            Expr::MemberAccess(MemberAccessExpr {
                base: obj,
                member,
                struct_name: struct_name_field,
                span: _,
            }) => {
                let obj_ty = self.check_expr_type_flag(obj, false, silent);
                let mut base_ty = obj_ty.clone();
                if let Type::Borrow(t, _, _, _) | Type::Pointer(t, _, _) = base_ty {
                    base_ty = *t;
                }

                let mut actual_struct_name = String::new();
                let mut struct_decl_opt = None;
                let mut mapping = HashMap::new();

                if let Type::Struct(struct_name, _) = &base_ty {
                    actual_struct_name = struct_name.clone();
                    struct_decl_opt = self
                        .env
                        .structs
                        .get(struct_name)
                        .map(|s| (*s).clone())
                        .or_else(|| {
                            self.generated_structs
                                .iter()
                                .find(|s| s.name == *struct_name)
                                .cloned()
                        });
                } else if let Type::GenericInstance(inner, args) = &base_ty {
                    if let Type::Struct(struct_name, _) = &**inner {
                        actual_struct_name = struct_name.clone();
                        struct_decl_opt = self
                            .env
                            .structs
                            .get(struct_name)
                            .map(|s| (*s).clone())
                            .or_else(|| {
                                self.generated_structs
                                    .iter()
                                    .find(|s| s.name == *struct_name)
                                    .cloned()
                            });
                        if let Some(decl) = &struct_decl_opt {
                            for (i, param) in decl.generics.iter().enumerate() {
                                if i < args.len() {
                                    mapping.insert(param.name().to_string(), args[i].clone());
                                }
                            }
                        }
                    }
                }

                if let Some(decl) = struct_decl_opt {
                    *struct_name_field = Some(base_ty.to_string());
                    for (f_name, f_type) in &decl.fields {
                        if f_name == member {
                            return f_type.substitute(&mapping);
                        }
                    }
                    self.errors.push(format!(
                        "Struct '{}' has no field '{}'",
                        actual_struct_name, member
                    ));
                } else if let Type::Struct(struct_name, _) = &base_ty {
                    self.errors
                        .push(format!("Unknown struct '{}' (expr.rs:1481)", struct_name));
                } else if let Type::GenericInstance(inner, _) = &base_ty {
                    if let Type::Struct(struct_name, _) = &**inner {
                        self.errors
                            .push(format!("Unknown struct '{}' (expr.rs:1485)", struct_name));
                    }
                } else if let Type::Module(ref path, ref exports) = base_ty {
                    if let Some(exported_ty) = exports.get(member) {
                        return exported_ty.clone();
                    } else {
                        self.errors
                            .push(format!("Module '{}' does not export '{}'", path, member));
                    }
                } else if member == "shape" {
                    return Type::Tensor(ElementType::I32, vec![], None);
                } else {
                    self.errors
                        .push("Member access on non-struct type".to_string());
                }
                Type::Tensor(ElementType::F32, vec![], None)
            }
            _ => unreachable!(),
        }
    }

    fn check_indexaccess_expr(&mut self, expr: &mut Expr, silent: bool) -> Type {
        match expr {
            Expr::IndexAccess(IndexAccessExpr {
                base: obj,
                index: idx,
                span: _,
            }) => {
                let obj_ty = self.check_expr_type_flag(obj, false, silent);
                self.check_expr_type(idx);
                if let Type::Pointer(inner, _, _) = obj_ty {
                    *inner
                } else if let Type::Borrow(inner, _, _, _) = obj_ty {
                    *inner
                } else if let Type::Tensor(el_ty, _, _) = obj_ty {
                    Type::Scalar(el_ty)
                } else {
                    Type::Scalar(ElementType::F32)
                }
            }
            _ => unreachable!(),
        }
    }

    fn check_methodcall_expr(&mut self, expr: &mut Expr, consume: bool, silent: bool) -> Type {
        match expr {
            Expr::MethodCall(MethodCallExpr {
                base: obj,
                method_name: _method,
                args,
                span: _,
            }) => {
                let mut base_ty = self.check_expr_type_flag(obj, false, silent);

                // Pre-infer closure argument types for specific intrinsics before type-checking them
                if let Type::Tensor(el_ty, _, _) = &base_ty {
                    if _method == "map" && args.len() == 1 {
                        if let Expr::Closure(c) = &mut args[0] {
                            if c.params.len() == 1 && c.params[0].1 == Type::Unknown {
                                c.params[0].1 = Type::Scalar(el_ty.clone());
                            }
                        }
                    }
                }

                for arg in args.iter_mut() {
                    self.check_expr_type(arg);
                }

                if let Type::Module(ref path, ref exports) = base_ty {
                    if let Some(exported_ty) = exports.get(_method) {
                        let prefix = TypeChecker::mangle_path(path);
                        let mangled_name = format!("{}_{}", prefix, _method);
                        let func_call = Expr::FunctionCall(FunctionCallExpr {
                            name: mangled_name,
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

                // --- COMPILER INTRINSICS ---
                if let Type::Tensor(el_ty, dims, top) = &base_ty {
                    if _method == "reshape" {
                        if args.is_empty() || args.len() > 3 {
                            self.errors
                                .push("reshape requires 1 to 3 arguments".to_string());
                            return base_ty;
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
                                if enum_name == "PadMode" && (variant == "Pad" || variant == "Trim")
                                {
                                    is_exact = false;
                                } else {
                                    self.errors.push(
                                        "reshape mode must be PadMode::Pad or PadMode::Trim"
                                            .to_string(),
                                    );
                                }
                            } else {
                                self.errors.push(
                                    "reshape mode must be an enum variant (e.g. PadMode::Pad)"
                                        .to_string(),
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
                                    return base_ty;
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
                                    return base_ty;
                                }
                            }

                            if is_exact && (src_elements - target_elements).abs() > 1e-6 {
                                self.errors.push(format!("reshape arithmetic mismatch: source has {} elements, target has {}", src_elements, target_elements));
                                return base_ty;
                            }

                            return Type::Tensor(el_ty.clone(), new_dims.clone(), top.clone());
                        } else {
                            self.errors.push(
                                "reshape requires an array of dimensions as the first argument"
                                    .to_string(),
                            );
                            return base_ty;
                        }
                    } else if _method == "iter" {
                        if !args.is_empty() {
                            self.errors.push("iter requires 0 arguments".to_string());
                        }
                        // We lower .iter() on Tensors to just evaluate to the tensor itself
                        // so the ForLoopStmt can catch it and emit a native scf.for loop
                        *expr = *obj.clone();
                        return base_ty;
                    } else if _method == "map" {
                        if args.len() != 1 {
                            self.errors
                                .push("map requires exactly 1 argument (the closure)".to_string());
                            return base_ty;
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
                        return base_ty;
                    } else if _method == "transpose" {
                        if args.len() != 1 {
                            self.errors.push("transpose requires exactly 1 argument (an array of permutation indices)".to_string());
                            return base_ty;
                        }
                        if let Expr::Array(ArrayExpr {
                            elements: perm,
                            span: _,
                        }) = &args[0]
                        {
                            let empty_env = HashMap::new();
                            let mut new_dims = vec![
                                Expr::Number(NumberExpr {
                                    value: "0".to_string(),
                                    ty: Some(ElementType::I32),
                                    span: Span::default()
                                });
                                dims.len()
                            ];
                            if perm.len() != dims.len() {
                                self.errors.push(
                                    "transpose permutation map length must match tensor rank"
                                        .to_string(),
                                );
                                return base_ty;
                            }
                            let mut seen = vec![false; dims.len()];
                            for (i, p) in perm.iter().enumerate() {
                                if let Some(Value::Number(v)) = self.eval_expr(p, &empty_env) {
                                    let v = v as usize;
                                    if v >= dims.len() {
                                        self.errors
                                            .push("transpose index out of bounds".to_string());
                                        return base_ty;
                                    }
                                    if seen[v] {
                                        self.errors.push(
                                            "transpose permutation map must not contain duplicates"
                                                .to_string(),
                                        );
                                        return base_ty;
                                    }
                                    seen[v] = true;
                                    new_dims[i] = dims[v].clone();
                                } else {
                                    self.errors.push(
                                        "Cannot statically evaluate transpose permutation index"
                                            .to_string(),
                                    );
                                    return base_ty;
                                }
                            }
                            return Type::Tensor(el_ty.clone(), new_dims, top.clone());
                        } else {
                            self.errors.push(
                                "transpose requires an array of permutation indices".to_string(),
                            );
                            return base_ty;
                        }
                    }
                }
                // --- END INTRINSICS ---

                // Dynamic Method Resolution
                let mut found_method = None;
                let mut mapping = HashMap::new();
                for impl_blocks in self.env.impls.values() {
                    for ib in impl_blocks {
                        mapping.clear();
                        let mut check_ty = base_ty.clone();
                        while let Type::Borrow(inner, _, _, _)
                        | Type::Pointer(inner, _, _)
                        | Type::Ref(inner, _) = &check_ty
                        {
                            check_ty = *inner.clone();
                        }

                        if self.unify_types(&ib.target_type, &check_ty, &mut mapping) {
                            for m in &ib.methods {
                                if m.name == *_method {
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

                if let Some((generic_method, _ib)) = found_method {
                    // Infer method-level generics from argument types
                    for (i, arg) in args.iter_mut().enumerate() {
                        let arg_ty = self.check_expr_type_flag(arg, false, true); // silent = true
                        if i + 1 < generic_method.params.len() {
                            let expected_param = &generic_method.params[i + 1].1;
                            self.unify_types(expected_param, &arg_ty, &mut mapping);
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
                    let mut method_func = self.instantiate_function(&modified_func, &mapping);

                    // Create a unique mangled name for the method based on the target type
                    let mangled_name = format!("{:?}_{}", base_ty, method_func.name)
                        .replace("(", "_")
                        .replace(")", "")
                        .replace(" ", "")
                        .replace("[", "")
                        .replace("]", "")
                        .replace(",", "_")
                        .replace("_None", "")
                        .replace("\"", "")
                        .replace("Tensor", "Tensor_")
                        .replace("GenericInstance_", "")
                        .replace("Struct_", "")
                        .replace("Scalar_", "");

                    println!(
                        "MethodCall: method={} base_ty={:?} mangled_name={}",
                        _method, base_ty, mangled_name
                    );

                    method_func.name = mangled_name.clone();

                    if !self.env.functions.contains_key(&mangled_name)
                        && !self
                            .monomorphized_functions
                            .iter()
                            .any(|(f, _)| f.name == mangled_name)
                    {
                        // Type check the instantiated method
                        let mut func_to_check = method_func.clone();
                        self.check_function(&mut func_to_check);
                        self.monomorphized_functions.push((func_to_check, 0)); // 0 will fall back to caller_module_idx
                    }

                    // Rewrite AST from MethodCall to FunctionCall
                    let mut call_args = vec![];
                    if let Some(first_param) = method_func.params.first() {
                        let param_is_ref = matches!(
                            first_param.1,
                            Type::Borrow(_, _, _, _) | Type::Pointer(_, _, _)
                        );
                        let is_mut = match &first_param.1 {
                            Type::Borrow(_, _, m, _) => *m,
                            Type::Pointer(_, _, m) => *m,
                            _ => false,
                        };

                        let obj_is_ref =
                            matches!(base_ty, Type::Borrow(_, _, _, _) | Type::Pointer(_, _, _));

                        if param_is_ref && !obj_is_ref {
                            call_args.push(Expr::Borrow(BorrowExpr {
                                expr: Box::new((**obj).clone()),
                                is_mut,
                                span: Span::default(),
                            }));
                        } else {
                            call_args.push((**obj).clone());
                        }
                    } else {
                        call_args.push((**obj).clone());
                    }

                    for a in args.iter() {
                        call_args.push(a.clone());
                    }

                    let mut func_call = Expr::FunctionCall(FunctionCallExpr {
                        name: mangled_name,
                        args: call_args,
                        span: Span::default(),
                    });
                    let ret_ty = self.check_expr_type_flag(&mut func_call, consume, silent);

                    // Replace the AST node in-place!
                    *expr = func_call;
                    return ret_ty;
                }

                // Fallback for hardcoded mock methods
                if _method == "with_memory" {
                    base_ty = Type::Ref(Box::new(base_ty), MemorySpace::NPUHBM);
                } else if _method == "to_device" {
                    let target_mem = MemorySpace::NPUHBM; // Can be enhanced later to parse arg
                    base_ty = Type::Pinned(
                        Box::new(base_ty),
                        Topology::NPU(Box::new(Expr::Number(NumberExpr {
                            value: "0".to_string(),
                            ty: Some(ElementType::I32),
                            span: Span::default(),
                        }))),
                    ); // Default to NPU[0]
                    *expr = Expr::Transfer(TransferExpr {
                        expr: obj.clone(),
                        space: target_mem,
                        cost: None,
                        span: Span::default(),
                    });
                } else if _method == "to_host" {
                    let target_mem = MemorySpace::HostDRAM;
                    base_ty = Type::Pinned(Box::new(base_ty), Topology::Host);
                    *expr = Expr::Transfer(TransferExpr {
                        expr: obj.clone(),
                        space: target_mem,
                        cost: None,
                        span: Span::default(),
                    });
                } else if _method == "as_ptr" || _method == "as_mut_ptr" {
                    let is_mut = _method == "as_mut_ptr";
                    match &base_ty {
                        Type::Tensor(el_ty, dims, top) => {
                            base_ty = Type::Pointer(
                                Box::new(Type::Tensor(el_ty.clone(), dims.clone(), top.clone())),
                                None,
                                is_mut,
                            );
                        }
                        Type::Borrow(inner, mem, mutability, _region) => {
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
                } else if _method == "len" {
                    match &base_ty {
                        Type::Tensor(_, _, _)
                        | Type::Borrow(_, _, _, _)
                        | Type::Pointer(_, _, _) => {
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
            _ => unreachable!(),
        }
    }

    fn check_binaryop_expr(&mut self, expr: &mut Expr, consume: bool, silent: bool) -> Type {
        match expr {
            Expr::BinaryOp(BinaryOpExpr {
                lhs,
                op,
                rhs,
                span: _,
            }) => {
                let lhs_ty = self.check_expr_type_flag(lhs, consume, silent);
                let rhs_ty = self.check_expr_type_flag(rhs, consume, silent);

                // Tensor operator overloading (A * B) -> Matmul
                if let (
                    Type::Tensor(el_ty_l, dims_l, top_l),
                    Type::Tensor(el_ty_r, dims_r, _top_r),
                ) = (&lhs_ty, &rhs_ty)
                {
                    if *op == BinaryOp::MatMul {
                        if el_ty_l != el_ty_r {
                            self.errors.push(format!("Tensor multiplication requires matching element types, got {:?} and {:?}", el_ty_l, el_ty_r));
                        }
                        let l_len = dims_l.len();
                        let r_len = dims_r.len();
                        if (l_len != 2 && l_len != 0) || (r_len != 2 && r_len != 0) {
                            self.errors.push(format!("Tensor multiplication (matmul) requires 2D tensors, got {}D and {}D", l_len, r_len));
                            return Type::Tensor(el_ty_l.clone(), vec![], top_l.clone());
                        }
                        return Type::Tensor(el_ty_l.clone(), vec![], top_l.clone());
                    }
                }

                if !self.is_assignable(&lhs_ty, &rhs_ty) {
                    self.errors.push(format!(
                        "Type mismatch in binary operation: {:?} vs {:?}",
                        lhs_ty, rhs_ty
                    ));
                }
                lhs_ty
            }
            _ => unreachable!(),
        }
    }

    fn check_relationalop_expr(&mut self, expr: &mut Expr, silent: bool) -> Type {
        match expr {
            Expr::RelationalOp(RelationalOpExpr {
                lhs,
                op: _,
                rhs,
                span: _,
            }) => {
                let lhs_ty = self.check_expr_type_flag(lhs, false, silent);
                let rhs_ty = self.check_expr_type_flag(rhs, false, silent);
                if !self.is_assignable(&lhs_ty, &rhs_ty) {
                    self.errors.push(format!(
                        "Type mismatch in relational operation: {:?} vs {:?}",
                        lhs_ty, rhs_ty
                    ));
                }
                Type::Scalar(ElementType::Bool)
            }
            _ => unreachable!(),
        }
    }

    fn check_logicalop_expr(&mut self, expr: &mut Expr, silent: bool) -> Type {
        match expr {
            Expr::LogicalOp(LogicalOpExpr {
                lhs,
                op: _,
                rhs,
                span: _,
            }) => {
                let lhs_ty = self.check_expr_type_flag(lhs, false, silent);
                let rhs_ty = self.check_expr_type_flag(rhs, false, silent);
                if !self.is_assignable(&lhs_ty, &rhs_ty) {
                    self.errors.push(format!(
                        "Type mismatch in logical operation: {:?} vs {:?}",
                        lhs_ty, rhs_ty
                    ));
                }
                Type::Scalar(ElementType::Bool)
            }
            _ => unreachable!(),
        }
    }

    fn check_unaryop_expr(&mut self, expr: &mut Expr, _silent: bool) -> Type {
        match expr {
            Expr::UnaryOp(UnaryOpExpr {
                op,
                expr: inner,
                span: _,
            }) => {
                let inner_ty = self.check_expr_type(inner);
                match op {
                    UnaryOp::Not => Type::Scalar(ElementType::Bool),
                    UnaryOp::Neg => inner_ty,
                }
            }
            _ => unreachable!(),
        }
    }

    fn check_borrow_expr(&mut self, expr: &mut Expr, silent: bool) -> Type {
        match expr {
            Expr::Borrow(BorrowExpr {
                expr: inner,
                is_mut,
                span: _,
            }) => {
                let inner_ty = self.check_expr_type_flag(inner, false, silent);

                // If the inner expression is an identifier, track the borrow
                if let Expr::Identifier(IdentifierExpr { name, span: _ }) = &**inner {
                    if let Some(borrows) = self.active_borrows.get(name) {
                        for b in borrows {
                            if b.is_mut {
                                if !silent {
                                    self.errors.push(format!("Cannot borrow '{}' because it is already borrowed as mutable.", name));
                                }
                            } else if *is_mut && !silent {
                                self.errors.push(format!("Cannot borrow '{}' as mutable because it is also borrowed as immutable.", name));
                            }
                        }
                    }
                    if !silent {
                        self.active_borrows
                            .entry(name.clone())
                            .or_default()
                            .push(BorrowRecord {
                                is_mut: *is_mut,
                                scope_depth: self.scopes.len(),
                                borrower_name: None,
                            });
                    }
                }

                Type::Borrow(Box::new(inner_ty), None, *is_mut, self.scopes.len())
            }
            _ => unreachable!(),
        }
    }

    fn check_dereference_expr(&mut self, expr: &mut Expr, consume: bool, silent: bool) -> Type {
        match expr {
            Expr::Dereference(e) => {
                let inner_ty = self.check_expr_type_flag(&mut e.expr, consume, silent);
                let resolved_ty = match inner_ty.clone() {
                    Type::Pointer(t, _, _) => {
                        if !self.in_unsafe_block && !silent {
                            println!("DEREF ERROR! inner_ty is {:?}", inner_ty);
                            let bt = std::backtrace::Backtrace::force_capture();
                            println!("{}", bt);
                            self.errors.push(
                                "Dereference of raw pointer outside of unsafe block!".to_string(),
                            );
                        }
                        *t
                    }
                    Type::Borrow(t, _, _, _) => *t,
                    _ => {
                        self.errors
                            .push("Cannot dereference non-pointer type".to_string());
                        inner_ty
                    }
                };
                e.ty = Some(resolved_ty.clone());
                resolved_ty
            }
            _ => unreachable!(),
        }
    }

    fn check_unsafeblock_expr(&mut self, expr: &mut Expr, consume: bool, silent: bool) -> Type {
        match expr {
            Expr::UnsafeBlock(UnsafeBlockExpr {
                stmts,
                ret: ret_expr,
                span: _,
            }) => {
                let prev_unsafe = self.in_unsafe_block;
                self.in_unsafe_block = true;
                self.push_scope();
                let mut ret_ty = self.check_expr_block(stmts, consume, silent);
                if let Some(r) = ret_expr {
                    ret_ty = self.check_expr_type_flag(r, consume, silent);
                }
                self.pop_scope();
                self.in_unsafe_block = prev_unsafe;
                ret_ty
            }
            _ => unreachable!(),
        }
    }

    fn check_structinit_expr(&mut self, expr: &mut Expr, consume: bool, silent: bool) -> Type {
        match expr {
            Expr::StructInit(StructInitExpr {
                name,
                fields,
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

                if let Some(struct_decl) = self
                    .env
                    .structs
                    .get(&base_name)
                    .map(|s| (*s).clone())
                    .or_else(|| {
                        self.generated_structs
                            .iter()
                            .find(|s| s.name == base_name)
                            .cloned()
                    })
                {
                    let mut mapping = std::collections::HashMap::new();
                    for (i, param) in struct_decl.generics.iter().enumerate() {
                        if i < generic_args.len() {
                            mapping.insert(param.name().to_string(), generic_args[i].clone());
                        }
                    }

                    // Check missing fields and type mismatch
                    for (expected_name, raw_expected_type) in &struct_decl.fields {
                        let expected_type = &raw_expected_type.substitute(&mapping);
                        let mut found = false;
                        for (f_name, f_expr) in fields.iter_mut() {
                            if f_name == expected_name {
                                found = true;
                                let f_type = self.check_expr_type_flag(f_expr, consume, silent);
                                if !self.is_assignable(expected_type, &f_type) && !silent {
                                    self.errors.push(format!(
                                            "Type mismatch in struct initialization for field '{}'. Expected {:?}, got {:?}",
                                            expected_name, expected_type, f_type
                                        ));
                                }
                                break;
                            }
                        }
                        if !found && !silent {
                            self.errors.push(format!(
                                "Missing field '{}' in initialization of struct '{}'",
                                expected_name, resolved_name
                            ));
                        }
                    }
                    // Check extra fields
                    for (f_name, f_expr) in fields.iter_mut() {
                        if !struct_decl.fields.iter().any(|(n, _)| n == f_name) {
                            if !silent {
                                self.errors.push(format!(
                                    "Struct '{}' has no field '{}'",
                                    resolved_name, f_name
                                ));
                            }
                            self.check_expr_type_flag(f_expr, consume, silent); // evaluate to find errors
                        }
                    }
                } else {
                    if !silent {
                        self.errors
                            .push(format!("Unknown struct {} (expr.rs:2175)", resolved_name));
                    }
                    for (_, f_expr) in fields.iter_mut() {
                        self.check_expr_type_flag(f_expr, consume, silent);
                    }
                }

                if !generic_args.is_empty() {
                    Type::GenericInstance(Box::new(Type::Struct(base_name, None)), generic_args)
                } else {
                    Type::Struct(resolved_name, None)
                }
            }
            _ => unreachable!(),
        }
    }

    fn check_grad_expr(&mut self, expr: &mut Expr, _silent: bool) -> Type {
        match expr {
            Expr::Grad(GradExpr {
                target_fn,
                args,
                span: _,
            }) => {
                let func = if let Some(&f) = self.env.ast_functions.get(target_fn) {
                    f.clone()
                } else {
                    self.errors.push(format!(
                        "Cannot differentiate unknown function '{}'",
                        target_fn
                    ));
                    return Type::Tensor(ElementType::F32, vec![], None);
                };
                self.check_differentiability(&func);

                if args.len() != func.params.len() {
                    self.errors.push(format!(
                        "Function {} expects {} arguments, but {} were provided",
                        target_fn,
                        func.params.len(),
                        args.len()
                    ));
                } else {
                    for (i, arg) in args.iter_mut().enumerate() {
                        let arg_type = self.check_expr_type(arg);
                        let param_type = &func.params[i].1;
                        if !self.is_assignable(param_type, &arg_type) {
                            self.errors.push(format!(
                                        "Type mismatch in argument {} for grad target {}: expected {:?}, got {:?}",
                                        i + 1, target_fn, param_type, arg_type
                                    ));
                        }
                    }
                }
                func.return_type.clone()
            }
            _ => unreachable!(),
        }
    }

    fn check_vjp_expr(&mut self, expr: &mut Expr, _silent: bool) -> Type {
        match expr {
            Expr::Vjp(VjpExpr {
                target_fn,
                args,
                cotangent,
                span: _,
            }) => {
                let func = if let Some(&f) = self.env.ast_functions.get(target_fn) {
                    f.clone()
                } else {
                    self.errors
                        .push(format!("Cannot vjp unknown function '{}'", target_fn));
                    return Type::Tensor(ElementType::F32, vec![], None);
                };
                self.check_differentiability(&func);

                if args.len() != func.params.len() {
                    self.errors.push(format!(
                        "Function {} expects {} arguments, but {} were provided",
                        target_fn,
                        func.params.len(),
                        args.len()
                    ));
                } else {
                    for (i, arg) in args.iter_mut().enumerate() {
                        let arg_type = self.check_expr_type(arg);
                        let param_type = &func.params[i].1;
                        if !self.is_assignable(param_type, &arg_type) {
                            self.errors.push(format!(
                                        "Type mismatch in argument {} for vjp target {}: expected {:?}, got {:?}",
                                        i + 1, target_fn, param_type, arg_type
                                    ));
                        }
                    }
                }
                self.check_expr_type(cotangent);
                func.return_type.clone()
            }
            _ => unreachable!(),
        }
    }

    fn check_jvp_expr(&mut self, expr: &mut Expr, _silent: bool) -> Type {
        match expr {
            Expr::Jvp(JvpExpr {
                target_fn,
                args,
                tangent,
                span: _,
            }) => {
                let func = if let Some(&f) = self.env.ast_functions.get(target_fn) {
                    f.clone()
                } else {
                    self.errors
                        .push(format!("Cannot jvp unknown function '{}'", target_fn));
                    return Type::Tensor(ElementType::F32, vec![], None);
                };
                self.check_differentiability(&func);

                if args.len() != func.params.len() {
                    self.errors.push(format!(
                        "Function {} expects {} arguments, but {} were provided",
                        target_fn,
                        func.params.len(),
                        args.len()
                    ));
                } else {
                    for (i, arg) in args.iter_mut().enumerate() {
                        let arg_type = self.check_expr_type(arg);
                        let param_type = &func.params[i].1;
                        if !self.is_assignable(param_type, &arg_type) {
                            self.errors.push(format!(
                                        "Type mismatch in argument {} for jvp target {}: expected {:?}, got {:?}",
                                        i + 1, target_fn, param_type, arg_type
                                    ));
                        }
                    }
                }
                self.check_expr_type(tangent);
                func.return_type.clone()
            }
            _ => unreachable!(),
        }
    }

    fn check_range_expr(&mut self, expr: &mut Expr, _silent: bool) -> Type {
        match expr {
            Expr::Range(RangeExpr {
                start,
                end,
                span: _,
            }) => {
                let start_ty = self.check_expr_type(start);
                let end_ty = self.check_expr_type(end);
                if start_ty != end_ty {
                    self.errors.push(format!(
                        "Range start and end types must match, got {:?} and {:?}",
                        start_ty, end_ty
                    ));
                }
                start_ty
            }
            _ => unreachable!(),
        }
    }

    fn bind_pattern_variables(&mut self, pattern: &Pattern, expr_ty: &Type) {
        match pattern {
            Pattern::Identifier(name) => {
                self.insert(name.clone(), expr_ty.clone());
            }
            Pattern::EnumVariant(enum_name, variant_name, Some(payloads)) => {
                let mut base_name = enum_name.clone();
                if let Some(idx) = enum_name.find('<') {
                    base_name = enum_name[..idx].to_string();
                }
                if let Some(enum_decl) = self.env.enums.get(&base_name) {
                    if let Some(variant) = enum_decl.variants.iter().find(|v| v.0 == *variant_name)
                    {
                        if let Some(payload_types) = &variant.1 {
                            let mut mapping = std::collections::HashMap::new();
                            if let Type::GenericInstance(_, args) = expr_ty {
                                for (i, param) in enum_decl.generics.iter().enumerate() {
                                    if i < args.len() {
                                        mapping.insert(param.name().to_string(), args[i].clone());
                                    }
                                }
                            }
                            for (i, p) in payloads.iter().enumerate() {
                                if let Pattern::Identifier(name) = p {
                                    if i < payload_types.len() {
                                        let p_ty = payload_types[i].substitute(&mapping);
                                        self.insert(name.clone(), p_ty);
                                    } else {
                                        self.insert(name.clone(), Type::Unknown);
                                    }
                                }
                            }
                        }
                    }
                } else {
                    for p in payloads {
                        if let Pattern::Identifier(name) = p {
                            self.insert(name.clone(), Type::Unknown);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn check_match_expr(&mut self, expr: &mut Expr, consume: bool, silent: bool) -> Type {
        match expr {
            Expr::Match(MatchExpr {
                expr: match_expr,
                arms,
                span: _,
            }) => {
                let expr_ty = self.check_expr_type(match_expr);

                for arm in arms {
                    self.push_scope();
                    self.bind_pattern_variables(&arm.pattern, &expr_ty);

                    let _arm_ty = if !silent {
                        self.check_expr_block(&mut arm.body, consume, silent)
                    } else {
                        Type::Tensor(ElementType::F32, vec![], None)
                    };
                    self.pop_scope();
                }

                Type::Tensor(ElementType::F32, vec![], None)
            }
            _ => unreachable!(),
        }
    }

    fn check_vecmacro_expr(&mut self, expr: &mut Expr, _silent: bool) -> Type {
        match expr {
            Expr::VecMacro(VecMacroExpr { elements, span }) => {
                let mut element_type = Type::Scalar(ElementType::I32); // Default
                if !elements.is_empty() {
                    let mut first = elements[0].clone();
                    element_type = self.check_expr_type(&mut first);
                }

                let var_name = format!("_vec_macro_tmp_{}", self.next_reg);
                self.next_reg += 1;

                let new_call = Expr::FunctionCall(FunctionCallExpr::new(
                    format!("Vec<{}>::new", element_type),
                    vec![],
                    span.clone(),
                ));

                let decl = Statement::LetDecl(LetDeclStmt::new(
                    var_name.clone(),
                    true,
                    None,
                    new_call,
                    span.clone(),
                ));

                let mut stmts = vec![decl];

                for el in elements.clone() {
                    let push_call = Expr::MethodCall(MethodCallExpr::new(
                        Box::new(Expr::Identifier(IdentifierExpr::new(
                            var_name.clone(),
                            span.clone(),
                        ))),
                        "push".to_string(),
                        vec![el],
                        span.clone(),
                    ));
                    stmts.push(Statement::ExprStmt(ExprStmtStmt::new(
                        push_call,
                        true,
                        span.clone(),
                    )));
                }

                let ret_expr =
                    Expr::Identifier(IdentifierExpr::new(var_name.clone(), span.clone()));

                let block = Expr::UnsafeBlock(UnsafeBlockExpr::new(
                    stmts,
                    Some(Box::new(ret_expr)),
                    span.clone(),
                ));

                *expr = block;
                self.check_expr_type(expr)
            }
            _ => unreachable!(),
        }
    }

    fn check_closure_expr(&mut self, expr: &mut Expr, _consume: bool, silent: bool) -> Type {
        match expr {
            Expr::Closure(e) => {
                let struct_name = format!("Closure_{}", self.next_reg);
                let func_name = format!("{}_call", struct_name);
                self.next_reg += 1;

                let cloned_params = e.params.clone();

                let closure_depth = self.scopes.len();
                self.closure_depths.push(closure_depth);
                self.closure_captures_stack.push(HashMap::new());

                let old_ret = self.current_return_type.take();
                self.current_return_type = Some(Type::Unknown);

                self.push_scope();
                for (name, ty) in &cloned_params {
                    self.insert(name.clone(), ty.clone());
                }
                let mut b = e.body.clone();
                let expr_ret_ty = self.check_expr_type(&mut b);

                let mut ret_ty = expr_ret_ty;
                if let Some(inferred) = &self.current_return_type {
                    if *inferred != Type::Unknown {
                        ret_ty = inferred.clone();
                    }
                }

                self.pop_scope();
                self.current_return_type = old_ret;

                let captured_vars_map = self.closure_captures_stack.pop().unwrap();
                self.closure_depths.pop();

                let mut captured_vars: Vec<(String, Type)> =
                    captured_vars_map.into_iter().collect();
                captured_vars.sort_by(|a, b| a.0.cmp(&b.0)); // Stable layout

                if !silent {
                    // Consume captured variables in the outer scope if they are linear
                    for (name, ty) in &captured_vars {
                        if matches!(ty, Type::Struct(_, _) | Type::Tensor(_, _, _)) {
                            self.consume(name);
                        }
                    }
                }

                // Create StructDecl for the environment
                let struct_decl = decl::StructDecl {
                    name: struct_name.clone(),
                    generics: vec![],
                    fields: captured_vars.clone(),
                };
                self.generated_structs.push(struct_decl);

                // Create the Function for calling the closure
                let mut env_params = vec![(
                    "_env".to_string(),
                    Type::Pointer(
                        Box::new(Type::Struct(struct_name.clone(), None)),
                        None,
                        true,
                    ), // &mut env
                )];

                for (name, ty) in &cloned_params {
                    env_params.push((name.clone(), ty.clone()));
                }

                let mut body_stmts = Vec::new();
                for (cap_name, cap_ty) in &captured_vars {
                    let env_access = Expr::MemberAccess(MemberAccessExpr {
                        base: Box::new(Expr::Identifier(IdentifierExpr::new(
                            "_env".to_string(),
                            e.span.clone(),
                        ))),
                        member: cap_name.clone(),
                        struct_name: Some(struct_name.clone()),
                        span: e.span.clone(),
                    });
                    body_stmts.push(Statement::LetDecl(LetDeclStmt {
                        name: cap_name.clone(),
                        is_mut: true,
                        ty_ann: Some(cap_ty.clone()),
                        expr: env_access,
                        span: e.span.clone(),
                    }));
                }

                body_stmts.push(Statement::Return(ReturnStmt {
                    expr: *b,
                    span: e.span.clone(),
                }));

                let call_func = decl::Function {
                    name: func_name.clone(),
                    generics: vec![],
                    params: env_params,
                    topology: self.active_topology.clone(),
                    return_type: ret_ty.clone(),
                    requires: vec![],
                    ensures: vec![],
                    body: body_stmts,
                };

                self.monomorphized_functions.push((call_func, 0));

                let mut fields = Vec::new();
                for (cap_name, _) in &captured_vars {
                    fields.push((
                        cap_name.clone(),
                        Expr::Identifier(IdentifierExpr::new(cap_name.clone(), e.span.clone())),
                    ));
                }

                *expr = Expr::StructInit(StructInitExpr {
                    name: struct_name.clone(),
                    fields,
                    span: e.span.clone(),
                });

                Type::Struct(struct_name, None)
            }
            _ => unreachable!(),
        }
    }
}
