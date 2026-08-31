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
use std::collections::HashMap;

impl<'a> TypeChecker<'a> {
    pub(crate) fn check_identifier_expr(&mut self, expr: &mut Expr, consume: bool) -> Type {
        match expr {
            Expr::Identifier(id) => {
                let name = id.name.clone();
                let span = id.span;

                if name.as_ref() == "true" || name.as_ref() == "false" {
                    return Type::Scalar(ElementType::Bool);
                }

                // Track variable usage for W1001/W1009 diagnostics
                self.used_vars.insert(name.clone());

                if !self.borrow.skip_borrow_check && !self.speculating {
                    // NLL: a borrow whose borrower is dead past this access no longer conflicts, so a
                    // semantically dead `&mut x` does not spuriously block reading `x` (#276). `live_borrows`
                    // sweeps first; the `!self.speculating` gate keeps speculative checks from mutating borrow state.
                    for b in self.borrow.live_borrows(name.as_ref()) {
                        if b.is_mut {
                            self.errors.error_with_code(
                                crate::diagnostic::DiagnosticCode::E4002,
                                format!("Cannot access '{}' because it is mutably borrowed.", name),
                                Some(crate::diagnostic::SourceSpan::from_ast_span(&span)),
                            );
                            break;
                        }
                    }
                }
                let lookup_depth_res = self
                    .lookup_with_depth(name.as_ref())
                    .map(|(ty, top, d)| (ty.clone(), top.clone(), d));
                let lookup_res = lookup_depth_res
                    .as_ref()
                    .map(|(ty, top, _)| (ty.clone(), top.clone()));

                if let Some((ty, _, depth)) = &lookup_depth_res {
                    // If we are inside a closure and the variable is defined outside of it,
                    // we must capture it in all closures between the definition and usage.
                    for (i, closure_depth) in self.mono.closure_depths.iter().enumerate() {
                        if depth < closure_depth {
                            self.mono.closure_captures_stack[i]
                                .insert(name.to_string().into(), ty.clone());
                        }
                    }
                }
                if lookup_res.is_none() && self.is_moved(name.as_ref()) {
                    if !self.speculating {
                        self.errors.error_with_code(
                            crate::diagnostic::DiagnosticCode::E4001,
                            format!("Use of moved or consumed linear variable: {}", name),
                            Some(crate::diagnostic::SourceSpan::from_ast_span(&span)),
                        );
                    }
                    // Poison the result rather than lying that it's an f32 tensor: the
                    // diagnostic above already fails compilation, and `Unknown` unifies with
                    // anything so it does not spawn cascade errors downstream.
                    return Type::Unknown;
                } else if lookup_res.is_none() {
                    if let Some((ret_ty, _, params, _, _, _)) =
                        self.env.functions.get(name.as_ref())
                    {
                        return Type::Function(params.clone(), Box::new(ret_ty.clone()));
                    }
                    for (func, _) in &self.mono.functions {
                        if func.name.as_ref() == name.as_ref() {
                            let params = func.params.iter().map(|(_, t)| t.clone()).collect();
                            return Type::Function(params, Box::new(func.return_type.clone()));
                        }
                    }

                    if !self.speculating {
                        // Collect all visible variable names for "did you mean?" suggestion
                        let mut candidate_names: Vec<String> = Vec::new();
                        for scope in &self.scopes {
                            for key in scope.keys() {
                                candidate_names.push(key.to_string());
                            }
                        }
                        for func_name in self.env.functions.keys() {
                            candidate_names.push(func_name.to_string());
                        }
                        let candidates: Vec<&str> =
                            candidate_names.iter().map(|s| s.as_str()).collect();
                        let suggestion = crate::suggest::suggest_name(name.as_ref(), &candidates);

                        self.errors.error_with_code(
                            crate::diagnostic::DiagnosticCode::E2001,
                            format!("Undefined variable '{}'", name),
                            Some(crate::diagnostic::SourceSpan::from_ast_span(&span)),
                        );
                        if let Some(suggested) = suggestion {
                            if let Some(last_diag) = self.errors.inner.last_mut() {
                                last_diag.notes.push(crate::diagnostic::Note {
                                    message: format!("Did you mean '{}'?", suggested).into(),
                                    span: None,
                                });
                            }
                        }
                    }
                    // An undefined variable has already been reported, so this only needs to
                    // avoid a panic. `Unknown` is the poison type for exactly that (#294) and
                    // suppresses downstream mismatches; claiming f32 instead invents a type the
                    // program never had and turns one root-cause error into a cascade of
                    // mismatches against it.
                    return Type::Unknown;
                }

                match lookup_res {
                    Some((ty, top)) => {
                        // Enforce Topology Boundaries!
                        let is_valid = self.transfer_cost_graph.is_type_accessible(
                            &self.active_topology,
                            &top,
                            &ty,
                        );

                        if !is_valid {
                            let is_pinned_on_host = matches!(ty, Type::Pinned(_, _))
                                && matches!(self.active_topology, Topology::CPU);
                            if !is_pinned_on_host && !self.speculating && !self.allow_cross_topology
                            {
                                let mut implements_transfer = false;
                                if let Some(impl_blocks) = self.env.impls.get("Relocatable") {
                                    for ib in impl_blocks {
                                        if self.unify_types(
                                            &ib.target_type,
                                            &ty,
                                            &mut std::collections::HashMap::new(),
                                        ) {
                                            implements_transfer = true;
                                            break;
                                        }
                                    }
                                }

                                if implements_transfer {
                                    // The type opted into implicit movement (it `impl`s
                                    // `Relocatable`), so we insert the `.relocate()` call for the
                                    // programmer. Surface it: a real data movement happens
                                    // silently at this use site, and the opt-in lives far away at
                                    // the type definition. Report the cost so it isn't a hidden
                                    // performance surprise (explicit-seam policy, "implicit but
                                    // visible"). Write `{name}.relocate()` explicitly to silence.
                                    //
                                    // This trait says a value MAY move implicitly. It does not say
                                    // how bytes move across a hardware edge -- that is
                                    // `impl Transfer<Memory::A, Memory::B> for Topology::X`, which
                                    // is a different question with a different key and a real
                                    // contract behind it (docs/custom_transfer_contract.md). Both
                                    // were spelled `Transfer` until Vx#353; the two names differed
                                    // only by a capital letter, and this one is the unchecked one.
                                    let cost_note = match self.transfer_cost_graph.reachable(
                                        &self.active_topology,
                                        &top,
                                        &ty,
                                    ) {
                                        crate::arch::Reachability::NeedsSeam { cost } => {
                                            format!(" (cost {cost})")
                                        }
                                        _ => String::new(),
                                    };
                                    self.errors.warn(
                                        crate::diagnostic::DiagnosticCode::W1024,
                                        format!(
                                            "implicit transfer of '{}' to {:?} inserted here via its \
                                             `Relocatable` impl{}; write `{}.relocate()` explicitly to silence",
                                            name, self.active_topology, cost_note, name
                                        ),
                                        Some(crate::diagnostic::SourceSpan::from_ast_span(&span)),
                                    );

                                    let _active_mem = self
                                        .transfer_cost_graph
                                        .default_memory_for(&self.active_topology);
                                    let method_name = crate::symbol::Symbol::from("relocate");
                                    let method_call =
                                        Expr::MethodCall(crate::syntax::expr::MethodCallExpr {
                                            base: Box::new(expr.clone()),
                                            method_name,
                                            type_args: None,
                                            args: vec![],
                                            span,
                                        });
                                    *expr = method_call;

                                    let old_allow = self.allow_cross_topology;
                                    self.allow_cross_topology = true;
                                    let ret_ty = self.check_methodcall_expr(expr, consume);
                                    self.allow_cross_topology = old_allow;
                                    return ret_ty;
                                } else {
                                    // USE-NEEDS-SEAM (explicit-seam policy): the value is not
                                    // visible from the active topology and has no `Transfer` impl,
                                    // so point at the fix. Distinguish "a transfer path exists,
                                    // write one" from "no path at all". See the hardware-monad doc.
                                    use crate::arch::Reachability;
                                    let reach = self.transfer_cost_graph.reachable(
                                        &self.active_topology,
                                        &top,
                                        &ty,
                                    );
                                    // M5: an implicit cross-space use is allowed when the value's
                                    // memory space is declared `managed: cached` (hardware-coherent)
                                    // and a path exists (`NeedsSeam`). `explicit`/undeclared spaces
                                    // and truly `Unreachable` ones still require an explicit transfer.
                                    let allowed = matches!(reach, Reachability::NeedsSeam { .. })
                                        && self
                                            .space_is_cached(&self.value_memory_space(&ty, &top));
                                    if !allowed {
                                        match reach {
                                        Reachability::NeedsSeam { cost } => {
                                            // E6003, naming the value's space, the topology's
                                            // visible set, and the costed fix — so a misplaced
                                            // handoff (e.g. an un-transferred KV cache in a
                                            // disaggregated prefill/decode split) is a compile
                                            // error carrying its own remedy (#253).
                                            let src = self.value_memory_space(&ty, &top);
                                            let dst = self
                                                .transfer_cost_graph
                                                .default_memory_for(&self.active_topology);
                                            let visible = self
                                                .transfer_cost_graph
                                                .descriptor(&self.active_topology.kind())
                                                .map(|d| {
                                                    d.visibility
                                                        .iter()
                                                        .map(|s| s.name())
                                                        .collect::<Vec<_>>()
                                                        .join(", ")
                                                })
                                                .unwrap_or_default();
                                            self.errors.error_with_code(
                                                crate::diagnostic::DiagnosticCode::E6003,
                                                format!(
                                                    "'{}' lives in {} but {} sees only [{}]; \
                                                     insert an explicit transfer to {} (cost {} \
                                                     on the declared path)",
                                                    name,
                                                    src.name(),
                                                    self.active_topology.display_name(),
                                                    visible,
                                                    dst.name(),
                                                    cost
                                                ),
                                                Some(crate::diagnostic::SourceSpan::from_ast_span(
                                                    &span,
                                                )),
                                            );
                                        }
                                        Reachability::Unreachable => self.errors.push(format!(
                                            "Cross-topology access error: '{}' (type: {:?}) is \
                                             unreachable from {:?}: no transfer path exists",
                                            name, ty, self.active_topology
                                        )),
                                        // Not visible here by construction; fall back to the plain message.
                                        Reachability::Visible => self.errors.push(format!(
                                            "Cross-topology access error: Variable '{}' belongs to {:?} \
                                             (type: {:?}), but accessed from {:?}",
                                            name, top, ty, self.active_topology
                                        )),
                                        }
                                    }
                                }
                            }
                        }
                        if consume && ty.is_linear() && !self.speculating {
                            self.consume(name.as_ref());
                        }
                        ty.clone()
                    }
                    None => {
                        if !self.speculating {
                            let msg = format!("Undefined variable '{}'", name);
                            self.errors.push(msg);
                        }
                        Type::Unknown // Poison, not a silent f32 tensor (see `Undefined variable`)
                    }
                }
            }
            _ => panic!("Expected Identifier, got {:?}", expr),
        }
    }

    pub(crate) fn check_memberaccess_expr(&mut self, expr: &mut Expr) -> Type {
        match expr {
            Expr::MemberAccess(MemberAccessExpr {
                base: obj,
                member,
                struct_name: struct_name_field,
                span: _,
            }) => {
                let old_skip = self.borrow.skip_borrow_check;
                self.borrow.skip_borrow_check = true;
                let obj_ty = self.check_expr_type_flag(obj, false);
                self.borrow.skip_borrow_check = old_skip;

                if !self.borrow.skip_borrow_check && !self.speculating {
                    if let Some((name, mut path)) = Self::extract_base_and_path(obj) {
                        path.push(member.to_string());
                        // NLL: `live_borrows` sweeps dead borrows of the base before testing path overlap,
                        // so reading a field after its `&mut p.x` borrow is dead is accepted — Example B's
                        // `return p.x` after `*r = 42` (#276). Gated on `!self.speculating` like the identifier arm.
                        for b in self.borrow.live_borrows(&name) {
                            if b.is_mut && crate::hir::places::paths_may_alias(&path, &b.path) {
                                self.errors.push(format!(
                                    "Cannot access '{}' because it is mutably borrowed.",
                                    name
                                ));
                                break;
                            }
                        }
                    }
                }
                let mut base_ty = obj_ty.clone();
                if let Type::Borrow { inner: t, .. } | Type::Pointer(t, _, _) = base_ty {
                    base_ty = *t;
                }

                let mut actual_struct_name = String::new();
                let mut struct_decl_opt = None;
                let mut mapping = HashMap::new();

                if let Type::Struct(struct_name, _) = &base_ty {
                    actual_struct_name = struct_name.to_string();
                    struct_decl_opt = self
                        .env
                        .structs
                        .get(struct_name.as_ref())
                        .map(|s| (*s).clone())
                        .or_else(|| {
                            self.mono
                                .generated_structs
                                .iter()
                                .find(|s| s.name == *struct_name)
                                .cloned()
                        });
                } else if let Type::GenericInstance(inner, args) = &base_ty {
                    if let Type::Struct(struct_name, _) = &**inner {
                        actual_struct_name = struct_name.to_string();
                        struct_decl_opt = self
                            .env
                            .structs
                            .get(struct_name.as_ref())
                            .map(|s| (*s).clone())
                            .or_else(|| {
                                self.mono
                                    .generated_structs
                                    .iter()
                                    .find(|s| s.name == *struct_name)
                                    .cloned()
                            });
                        if let Some(decl) = &struct_decl_opt {
                            for (i, param) in decl.generics.iter().enumerate() {
                                if i < args.len() {
                                    mapping.insert(param.name().into(), args[i].clone());
                                }
                            }
                        }
                    }
                }

                // An *imported* struct resolved from a `.vxlib` (no AST in this compile) is absent from
                // `env.structs`; its declared field types live in the frozen registry's `structs`
                // table (#219). Fall back to it so member access on an imported struct types the same
                // way a local one does. Checked only when the AST env misses, so a local definition
                // always shadows a same-named import.
                if struct_decl_opt.is_none() {
                    let imported_name = match &base_ty {
                        Type::Struct(n, _) => Some(n.clone()),
                        Type::GenericInstance(inner, _) => match &**inner {
                            Type::Struct(n, _) => Some(n.clone()),
                            _ => None,
                        },
                        _ => None,
                    };
                    if let Some(name) = imported_name {
                        // Resolved through the GID the type carries (else the unambiguous bare
                        // name), so two artifacts' same-named structs can't cross wires (#291).
                        // Clone out so the registry borrow ends before `self.errors` is touched.
                        let imported = self
                            .worker
                            .global
                            .registry
                            .struct_fields_of(&base_ty)
                            .map(|sf| (sf.generics.clone(), sf.fields.clone()));
                        if let Some((generics, fields)) = imported {
                            if let Type::GenericInstance(_, args) = &base_ty {
                                for (i, param) in generics.iter().enumerate() {
                                    if i < args.len() {
                                        mapping.insert(param.as_ref().into(), args[i].clone());
                                    }
                                }
                            }
                            *struct_name_field = Some(base_ty.to_string().into());
                            for (f_name, f_type) in &fields {
                                if f_name == member {
                                    return f_type.substitute(&mapping);
                                }
                            }
                            self.errors
                                .push(format!("Struct '{}' has no field '{}'", name, member));
                            return Type::Unknown;
                        }
                    }
                }

                if let Some(decl) = struct_decl_opt {
                    *struct_name_field = Some(base_ty.to_string().into());
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
                    self.errors.push(unknown_struct_message(
                        &self.worker.global.registry,
                        struct_name,
                        "expr.rs:1481",
                    ));
                } else if let Type::GenericInstance(inner, _) = &base_ty {
                    if let Type::Struct(struct_name, _) = &**inner {
                        self.errors.push(unknown_struct_message(
                            &self.worker.global.registry,
                            struct_name,
                            "expr.rs:1485",
                        ));
                    }
                } else if let Type::Module(ref path, ref exports) = base_ty {
                    if let Some(exported_ty) = exports.get(member) {
                        return exported_ty.clone();
                    } else {
                        self.errors
                            .push(format!("Module '{}' does not export '{}'", path, member));
                    }
                } else if member.as_ref() == "shape" {
                    return Type::Tensor(ElementType::I32, vec![], None);
                } else {
                    self.errors
                        .push("Member access on non-struct type".to_string());
                }
                Type::Unknown
            }
            _ => panic!("Expected IndexAccess, got {:?}", expr),
        }
    }

    /// The element type produced by indexing a user container such as `Vec<T>`: resolved from the
    /// container struct's backing `data : *mut T` field, with the struct's generic parameters
    /// substituted by the instance's type arguments. So `v[i]` on a `Vec<i32>` is `i32` — not the
    /// `f32` an unresolved index used to fall back to, a default that permissive coercion hid
    /// (#240). Returns `None` for anything that isn't such a container.
    pub(crate) fn container_element_type(&self, base_ty: &Type) -> Option<Type> {
        let (struct_name, args): (&str, &[Type]) = match base_ty {
            Type::Struct(n, _) => (n.as_ref(), &[]),
            Type::GenericInstance(inner, args) => match &**inner {
                Type::Struct(n, _) => (n.as_ref(), args.as_slice()),
                _ => return None,
            },
            _ => return None,
        };
        let decl = self
            .env
            .structs
            .get(struct_name)
            .map(|s| (*s).clone())
            .or_else(|| {
                self.mono
                    .generated_structs
                    .iter()
                    .find(|s| s.name.as_ref() == struct_name)
                    .cloned()
            })?;
        let mut mapping = HashMap::new();
        for (i, param) in decl.generics.iter().enumerate() {
            if let Some(a) = args.get(i) {
                mapping.insert(param.name().into(), a.clone());
            }
        }
        for (f_name, f_type) in &decl.fields {
            if f_name.as_ref() == "data" {
                if let Type::Pointer(inner, _, _) | Type::Borrow { inner, .. } =
                    f_type.substitute(&mapping)
                {
                    return Some(*inner);
                }
            }
        }
        None
    }

    pub(crate) fn check_indexaccess_expr(&mut self, expr: &mut Expr) -> Type {
        match expr {
            Expr::IndexAccess(IndexAccessExpr {
                base: obj,
                index: idx,
                span: ix_span,
            }) => {
                let ix_span = *ix_span;
                let obj_ty = self.check_expr_type_flag(obj, false);

                // Enforce topology boundary for Pinned types
                if let Type::Pinned(_, pinned_top) = &obj_ty {
                    if !self.transfer_cost_graph.is_type_accessible(
                        &self.active_topology,
                        pinned_top,
                        &obj_ty,
                    ) && !self.speculating
                    {
                        self.errors.push(format!(
                            "Cross-topology access error: Cannot access Pinned type on {:?} from {:?}",
                            pinned_top, self.active_topology
                        ));
                    }
                }

                self.check_expr_type(idx);
                // Look through a Pinned/Ref device wrapper to the underlying tensor so that
                // indexing a transferred tensor follows the same rule as a local one.
                let base = match obj_ty {
                    Type::Pinned(inner, _) | Type::Ref(inner, _) => *inner,
                    // A borrow indexes like what it borrows: `t[i]` on `&Tensor<f32, [N, D]>` is
                    // the row. Answering with the inner type applied the index to nothing, so
                    // `t[0][0]` was the row rather than the element (Vx#417).
                    Type::Borrow { inner, .. } => *inner,
                    other => other,
                };
                if let Type::Pointer(inner, _, _) = base {
                    *inner
                } else if let Type::Tensor(el_ty, dims, top) = base {
                    if dims.len() > 1 {
                        // Partial index (S1): a rank-reduced slice of the remaining dimensions,
                        // e.g. `q[i]` on Tensor<f32,[N,D]> is a Tensor<f32,[D]> row view.
                        Type::Tensor(el_ty, dims[1..].to_vec(), top)
                    } else {
                        Type::Scalar(el_ty)
                    }
                } else if let Type::DynTensor(el_ty, _) = base {
                    // A dynamic tensor is indexable; its rank is what is unknown. Indexing
                    // yields the element, which is what a dims-less tensor yielded before the
                    // shape-unknown reading had its own type (Vx#399).
                    Type::Scalar(el_ty)
                } else if let Some(elem) = self.container_element_type(&base) {
                    // A user container (`Vec<T>`): its element type, resolved from the backing
                    // `data` pointer. `v[i]` on a `Vec<i32>` is `i32`, not the `f32` this used to
                    // default to (a bug coercion hid, #240).
                    //
                    // In read position the sugar rewrites to the container's own `get`, so both
                    // backends lower one construct -- the AST index lowering emitted a
                    // `memref.load` on the struct value here (Vx#398). A store place
                    // (`v[i] = x`) and a speculative probe keep the typed-only answer: the
                    // first is not a read, the second must not mutate the AST.
                    if !self.speculating && !self.checking_assign_lhs {
                        let mut call = Expr::MethodCall(crate::syntax::MethodCallExpr {
                            base: obj.clone(),
                            method_name: "get".into(),
                            type_args: None,
                            args: vec![(**idx).clone()],
                            span: ix_span,
                        });
                        let t = self.check_methodcall_expr(&mut call, false);
                        *expr = call;
                        return t;
                    }
                    elem
                } else if let Type::Scalar(el_ty) = base {
                    // A dynamically shaped `Tensor<T>` carries no dims, so the branch above
                    // takes `dims.len() > 1` as false and the *first* index already yields
                    // `Scalar(T)`. The second index of `t[i][j]` therefore arrives here with a
                    // scalar base, and used to fall through to the F32 default below --
                    // silently retyping the element of every tensor whose element is not f32.
                    // f32 tensors never revealed it because the wrong answer matched the right
                    // one, which is why this reads as an f16 bug (#324) and is not one.
                    //
                    // Carrying the element type through is what a further index of an
                    // already-elemental value means. It does not model rank: a rank-unknown
                    // tensor cannot say how many indices exhaust it, and `t[i]` yielding a row
                    // view rather than an element is a separate question (#324).
                    Type::Scalar(el_ty)
                } else if base == Type::Unknown {
                    // Already-reported failure upstream; stay quiet (#294).
                    Type::Unknown
                } else {
                    // Never guess an element type. Defaulting to f32 here is what let #324
                    // hide: indexing produced f32 for every tensor whose element was not
                    // f32, and because f32 is the common case the wrong answer usually
                    // matched the right one. A wrong type that is right often enough is
                    // worse than an error, because it is discovered by the numbers being
                    // wrong somewhere else entirely.
                    if !self.speculating {
                        self.errors.error_with_code(
                            crate::diagnostic::DiagnosticCode::E3004,
                            format!("Cannot index a value of type {:?}", base),
                            None,
                        );
                    }
                    Type::Unknown
                }
            }
            _ => panic!("Expected IndexAccess, got {:?}", expr),
        }
    }

    pub(crate) fn check_borrow_expr(&mut self, expr: &mut Expr) -> Type {
        match expr {
            Expr::Borrow(BorrowExpr {
                expr: inner,
                is_mut,
                span,
            }) => {
                let inner_ty = self.check_expr_type_flag(inner, false);

                if let Some((name, path)) = Self::extract_base_and_path(inner) {
                    // NLL: `live_borrows` sweeps dead borrows before the shared-XOR-mutable conflict
                    // check, so a borrow whose borrower is dead no longer blocks a new one (#276). This
                    // was the hand-copied sweep duplicate the R1 refactor removed.
                    for b in self.borrow.live_borrows(&name) {
                        // Split borrows: skip a record whose path is disjoint from this borrow's.
                        if !crate::hir::places::paths_may_alias(&path, &b.path) {
                            continue;
                        }
                        if b.is_mut {
                            if !self.speculating {
                                self.errors.error_with_code(
                                    crate::diagnostic::DiagnosticCode::E4004,
                                    format!("Cannot borrow '{}' because it is already borrowed as mutable.", name),
                                    Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
                                );
                            }
                        } else if *is_mut && !self.speculating {
                            self.errors.error_with_code(
                                crate::diagnostic::DiagnosticCode::E4003,
                                format!("Cannot borrow '{}' as mutable because it is also borrowed as immutable.", name),
                                Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
                            );
                        }
                    }
                    if !self.speculating {
                        self.borrow.record(
                            &name,
                            BorrowRecord {
                                is_mut: *is_mut,
                                scope_depth: self.scopes.len(),
                                borrower_name: self.current_assignment_target.clone(),
                                path,
                            },
                        );
                    }
                }

                Type::Borrow {
                    inner: Box::new(inner_ty),
                    mem_space: None,
                    is_mut: *is_mut,
                    region_id: self.scopes.len(),
                }
            }
            _ => panic!("Expected IndexAccess, got {:?}", expr),
        }
    }

    /// Base variable and field path of an lvalue, in the checker's `String` representation (its
    /// `BorrowRecord.path` is `Vec<String>`). A thin adapter over the shared `places::base_and_path` —
    /// one derivation for both the checker and the flat lowerer (#275 §16.2).
    pub(crate) fn extract_base_and_path(expr: &Expr) -> Option<(String, Vec<String>)> {
        crate::hir::places::base_and_path(expr).map(|(root, path)| {
            (
                root.to_string(),
                path.iter().map(|s| s.to_string()).collect(),
            )
        })
    }

    /// Base variable and field path a reference *argument* reborrows, seeing through a leading `&`.
    /// `foo(m)` and `foo(&m.slot)` both reborrow storage rooted at `m`; the summary-driven persist
    /// decision (#243) keys on that base. Returns `None` for arguments with no nameable base
    /// (`foo(&5)`, `foo(g())`).
    pub(crate) fn arg_reborrow_base(arg: &Expr) -> Option<(String, Vec<String>)> {
        match arg {
            Expr::Borrow(b) => Self::extract_base_and_path(&b.expr),
            other => Self::extract_base_and_path(other),
        }
    }

    /// True for the reference-shaped types (`&T`, `&mut T`, raw pointers, `Ref<T>`). These are
    /// the types the return-escape and reborrow analyses (#243) reason about.
    pub(crate) fn is_ref_type(ty: &Type) -> bool {
        matches!(ty, Type::Borrow { .. } | Type::Pointer(..) | Type::Ref(..))
    }

    /// True for a *mutable* reference type (`&mut T` / `*mut T`).
    pub(crate) fn is_mut_ref(ty: &Type) -> bool {
        matches!(
            ty,
            Type::Borrow { is_mut: true, .. } | Type::Pointer(_, _, true)
        )
    }

    /// Return-escape provenance (#243): where does the reference produced by `expr` root?
    /// `None` when `expr` is not a reference (nothing to check). `External` when it roots in a
    /// caller-owned reference parameter (safe to return); `Local` when it roots in a
    /// function-local slot, a by-value binding, or a temporary (returning it would dangle).
    pub(crate) fn ref_provenance_of(&self, expr: &Expr) -> Option<crate::hir::env::RefProvenance> {
        use crate::hir::env::RefProvenance;
        match expr {
            // `&base` / `&base.field`: a fresh borrow. It is safe to return only when it reborrows
            // *through* a caller-owned reference parameter (e.g. `&m.slot` for `m: &Map`).
            // Borrowing a by-value parameter, a local, or a temporary all yield stack-local refs.
            Expr::Borrow(b) => {
                if let Some((base, _path)) = Self::extract_base_and_path(&b.expr) {
                    match self.borrow.current_params.get(base.as_str()) {
                        Some(pty) if Self::is_ref_type(pty) => Some(RefProvenance::External),
                        _ => Some(RefProvenance::Local),
                    }
                } else {
                    // `&5`, `&(a + b)`, `&f()` — borrows an unnamed temporary.
                    Some(RefProvenance::Local)
                }
            }
            // A bare reference variable: a reference parameter is external; a local binding carries
            // whatever provenance we recorded at its `let`. An untracked reference identifier is
            // left unresolved (`None`) rather than guessed, to avoid false escapes.
            Expr::Identifier(id) => {
                if let Some(pty) = self.borrow.current_params.get(id.name.as_ref()) {
                    if Self::is_ref_type(pty) {
                        return Some(RefProvenance::External);
                    }
                    return None;
                }
                self.borrow.ref_provenance.get(id.name.as_ref()).copied()
            }
            // A call yielding a reference reborrows from its reference arguments: local iff any
            // reference argument is local (e.g. `identity(&x)` for a local `x`).
            Expr::FunctionCall(fc) => {
                // A closure invocation is rewritten to `Closure_N_call(<env>, real_args..)`, where
                // the env (slot 0) carries the captures and is a local. Whether to count it toward
                // the result's provenance depends on what the closure body actually returns (#269):
                // a reference derived from a real *parameter* (`|q| &q.slot`) is safe — skip the env
                // and join the real arguments; a reference derived from the *env* (a captured local,
                // `|| &x`) or a body local is an escape — keep the env so its `Local` provenance is
                // seen. The closure's own return summary tells them apart.
                let name = fc.name.as_ref();
                let is_closure_call =
                    name.starts_with("Closure_") && name.ends_with("_call") && !fc.args.is_empty();
                let skip_env = is_closure_call
                    && self
                        .mono
                        .functions
                        .iter()
                        .find(|f| f.0.name.as_ref() == name)
                        .map(|f| crate::hir::provenance::compute_return_provenance(&f.0))
                        .is_some_and(|rp| {
                            // Skip the env only when the return derives from real parameters
                            // (slots >= 1), never the env (slot 0). Local / AnyParam / unknown keep
                            // the env, conservatively.
                            matches!(
                                rp,
                                crate::hir::provenance::ReturnProvenance::FromParams(bits)
                                    if bits & 1 == 0
                            )
                        });
                if skip_env {
                    self.join_arg_provenance(&fc.args[1..])
                } else {
                    self.join_arg_provenance(&fc.args)
                }
            }
            Expr::MethodCall(mc) => {
                let mut provs: Vec<&Expr> = vec![mc.base.as_ref()];
                provs.extend(mc.args.iter());
                self.join_arg_provenance_exprs(&provs)
            }
            _ => None,
        }
    }

    pub(crate) fn join_arg_provenance(
        &self,
        args: &[Expr],
    ) -> Option<crate::hir::env::RefProvenance> {
        let refs: Vec<&Expr> = args.iter().collect();
        self.join_arg_provenance_exprs(&refs)
    }

    /// Join provenance across a call's reference operands: `Local` if any is `Local`, otherwise
    /// `External` (a correct callee returns a reference derived from its reference inputs; a
    /// callee that fabricates one from a local is caught when *it* is checked).
    pub(crate) fn join_arg_provenance_exprs(
        &self,
        args: &[&Expr],
    ) -> Option<crate::hir::env::RefProvenance> {
        use crate::hir::env::RefProvenance;
        for a in args {
            if self.ref_provenance_of(a) == Some(RefProvenance::Local) {
                return Some(RefProvenance::Local);
            }
        }
        Some(RefProvenance::External)
    }

    /// Track a reborrow created by passing an existing reference *by name* to a reference
    /// parameter (#243, bc9). Mirrors `check_borrow_expr`'s NLL dead-borrow cleanup and
    /// shared-XOR-mutable conflict check, keyed on the underlying variable.
    ///
    /// Two distinct mutabilities (#268): `access_is_mut` is what the *call* does to the argument
    /// (the parameter's mutability — `insert(&mut Map)` mutates through it) and drives the conflict
    /// check; `record_is_mut` is what the persisted alias is (the *result* reference's mutability —
    /// `probe(m: &mut Map) -> &i32` yields a shared alias) and is the mutability of the record left
    /// behind. Persist only when the reborrow outlives the call (the callee returns a reference);
    /// a value/void call borrows only for its own duration.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn track_reference_arg_borrow(
        &mut self,
        base: &str,
        path: Vec<String>,
        access_is_mut: bool,
        record_is_mut: bool,
        persist: bool,
        span: &crate::syntax::Span,
    ) {
        if self.speculating {
            return;
        }
        // NLL: `live_borrows` drops records whose borrower is no longer used past this point (the same
        // sweep the access checks run, #276) before the shared-XOR-mutable conflict check.
        for b in self.borrow.live_borrows(base) {
            // Overlapping-path conflict, shared with `check_borrow_expr` (#275 §16.2).
            if !crate::hir::places::paths_may_alias(&path, &b.path) {
                continue;
            }
            if b.is_mut {
                self.errors.error_with_code(
                    crate::diagnostic::DiagnosticCode::E4004,
                    format!(
                        "Cannot borrow '{}' because it is already borrowed as mutable.",
                        base
                    ),
                    Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
                );
            } else if access_is_mut {
                self.errors.error_with_code(
                    crate::diagnostic::DiagnosticCode::E4003,
                    format!(
                        "Cannot borrow '{}' as mutable because it is also borrowed as immutable.",
                        base
                    ),
                    Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
                );
            }
        }
        if persist {
            self.borrow.record(
                base,
                BorrowRecord {
                    is_mut: record_is_mut,
                    scope_depth: self.scopes.len(),
                    borrower_name: self.current_assignment_target.clone(),
                    path,
                },
            );
        }
    }

    pub(crate) fn check_dereference_expr(&mut self, expr: &mut Expr, consume: bool) -> Type {
        match expr {
            Expr::Dereference(e) => {
                let inner_ty = self.check_expr_type_flag(&mut e.expr, consume);
                let resolved_ty = match inner_ty.clone() {
                    Type::Pointer(t, _, _) => {
                        if !self.in_unsafe_block && !self.speculating {
                            println!("DEREF ERROR! inner_ty is {:?}", inner_ty);
                            let bt = std::backtrace::Backtrace::force_capture();
                            println!("{}", bt);
                            self.errors.push(
                                "Dereference of raw pointer outside of unsafe block!".to_string(),
                            );
                        }
                        *t
                    }
                    Type::Borrow { inner: t, .. } => *t,
                    _ => {
                        self.errors
                            .push("Cannot dereference non-pointer type".to_string());
                        inner_ty
                    }
                };
                e.ty = Some(resolved_ty.clone());
                resolved_ty
            }
            _ => panic!("Expected IndexAccess, got {:?}", expr),
        }
    }
}

/// The member-access "unknown struct" message, upgraded when the name is *ambiguous* rather than
/// absent: two imported modules both defining it used to resolve by merge order before the bare
/// name learned to decline, and "unknown" would send the user hunting the wrong bug (#291).
fn unknown_struct_message(
    registry: &crate::registry::ImmutableGlobalRegistry,
    struct_name: &crate::symbol::Symbol,
    site: &str,
) -> String {
    if registry.is_ambiguous_nominal(struct_name) {
        crate::registry::ambiguous_import_message("Struct", struct_name)
    } else {
        format!("Unknown struct '{}' ({})", struct_name, site)
    }
}
