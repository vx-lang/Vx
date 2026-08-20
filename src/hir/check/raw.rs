//===- check/raw.rs - The raw:: transfer-lowering primitives ----*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// The eight `raw::` primitives an `impl transfer` lowering is written in, and the
// obligations each one carries (Vx#353 A2; the table lives in
// docs/custom_transfer_contract.md). The primitives are *indexed, not addressed*:
// every one takes a typed tile plus an element index, none takes or produces an
// address. That single restriction is what keeps the contract's placement and
// visibility constraints (C1, C6) mechanically checkable.
//
// Discharge follows the contract doc's split exactly:
//   * bounds        -> the existing `SmtProver` (`prove_expr`), refused when
//                      unprovable unless inside `unsafe`, which records the
//                      asserted obligation instead -- the same standing an
//                      unverified `spec:` figure has;
//   * spaces        -> a table lookup against the declaring topologies'
//                      `visible:` lists (no prover);
//   * exclusivity   -> the borrow checker, unchanged (`&mut` on the store side);
//   * capability    -> a machine-file lookup: `raw::async_copy` is legal only
//                      when a declared topology edge carries `copy_engine`;
//   * barrier shape -> a conservative control-flow check: `raw::barrier()` is
//                      legal only as a top-level statement of the body;
//   * async order   -> a forward walk whose reject is the seam obligation
//                      (an unwaited copy that gets read = a reachable stale
//                      read, the same fact `hir/seam.rs` refuses).
//
// Everything here runs on the surface AST. The walkers match `Expr` and
// `Statement` exhaustively -- no catch-all arm -- so adding a syntax node the
// walk does not know about fails the build instead of silently skipping a
// `raw::` call nested inside it.
//
//===----------------------------------------------------------------------===//

use super::super::*;
use crate::diagnostic::{DiagnosticCode, SourceSpan};
use crate::syntax::expr::{Expr, FunctionCallExpr, LogicalOpExpr, NumberExpr, RelationalOpExpr};
use crate::syntax::stmt::Statement;
use crate::syntax::{LogicalOp, RelationalOp, Span};

/// The eight primitive names, for the unknown-name diagnostic.
const RAW_PRIMITIVES: &[&str] = &[
    "extent",
    "lane",
    "lanes",
    "load",
    "store",
    "barrier",
    "async_copy",
    "async_wait",
];

/// What a tile argument resolved to: its element type and, when every declared
/// dimension is a literal, its element count.
struct TileInfo {
    elem: crate::syntax::ElementType,
    static_extent: Option<u64>,
    name: crate::symbol::Symbol,
}

impl<'a> TypeChecker<'a> {
    /// Type one `raw::<prim>` call. Called from `resolve_intrinsic_function`, so this
    /// runs before any user-code lookup: the `raw::` prefix is reserved whether or not
    /// the call is legal where it appears.
    pub(crate) fn check_raw_primitive(
        &mut self,
        prim: &str,
        args: &[Expr],
        arg_types: &[Type],
        call_span: Span,
    ) -> Type {
        let unit = || Type::Tensor(crate::syntax::ElementType::F32, vec![], None);
        let i64_ty = Type::Scalar(crate::syntax::ElementType::I64);
        let span = Some(SourceSpan::from_ast_span(&call_span));

        // A speculative probe (a method-call rewrite trying shapes) must not emit
        // diagnostics or run prover obligations -- it would double every error at the
        // real check. Answer with the primitive's type and nothing else.
        if self.speculating {
            return match prim {
                "extent" | "lane" | "lanes" => i64_ty,
                "load" => arg_types
                    .first()
                    .and_then(|t| Self::as_tensor_operand(t).map(|(e, _, _)| e.clone()))
                    .map(Type::Scalar)
                    .unwrap_or_else(unit),
                _ => unit(),
            };
        }

        // A closure runs when it is called, not when the lowering runs, so a primitive
        // inside one escapes every whole-body guarantee (barrier reachability, the async
        // walk). Refused at typing -- the only place that reliably sees closure bodies,
        // since they are lifted out before the whole-body scan runs.
        if !self.closure_captures_stack.is_empty() {
            self.errors.error_with_code(
                DiagnosticCode::E6017,
                format!(
                    "`raw::{prim}` cannot be used inside a closure: a lowering's moves \
                     must run when the lowering runs"
                ),
                span,
            );
            return unit();
        }

        // The floor property: outside an `impl Transfer` body the names do not resolve.
        let Some((from, to, machine)) = self.transfer_lowering_edge.clone() else {
            self.errors.error_with_code(
                DiagnosticCode::E6017,
                format!(
                    "`raw::{prim}` is a transfer-lowering primitive; it is only legal inside \
                     an `impl transfer` body (docs/custom_transfer_contract.md)"
                ),
                span,
            );
            return unit();
        };

        let arity = |n: usize, checker: &mut Self| {
            if args.len() != n {
                checker.errors.error_with_code(
                    DiagnosticCode::E6017,
                    format!("`raw::{prim}` expects {n} argument(s), got {}", args.len()),
                    Some(SourceSpan::from_ast_span(&call_span)),
                );
                false
            } else {
                true
            }
        };

        match prim {
            "extent" => {
                if arity(1, self) {
                    self.raw_tile_arg(prim, &args[0], &arg_types[0], false, &call_span);
                }
                i64_ty
            }
            "lane" | "lanes" => {
                arity(0, self);
                i64_ty
            }
            "load" => {
                if !arity(2, self) {
                    return unit();
                }
                self.raw_index_arg(prim, &arg_types[1], &call_span);
                match self.raw_tile_arg(prim, &args[0], &arg_types[0], false, &call_span) {
                    Some(tile) => {
                        let elem = tile.elem.clone();
                        self.raw_bounds_obligation(prim, &args[1], &[tile], &call_span);
                        Type::Scalar(elem)
                    }
                    None => unit(),
                }
            }
            "store" => {
                if !arity(3, self) {
                    return unit();
                }
                self.raw_index_arg(prim, &arg_types[1], &call_span);
                if let Some(tile) =
                    self.raw_tile_arg(prim, &args[0], &arg_types[0], true, &call_span)
                {
                    // #240 identical-only assignability: the stored value must be exactly the
                    // tile's element type, no widening.
                    if arg_types[2] != Type::Scalar(tile.elem.clone()) {
                        self.errors.error_with_code(
                            DiagnosticCode::E6017,
                            format!(
                                "`raw::store` writes a {:?} tile but the value has type {:?}",
                                tile.elem, arg_types[2]
                            ),
                            Some(SourceSpan::from_ast_span(&call_span)),
                        );
                    }
                    self.raw_bounds_obligation(prim, &args[1], &[tile], &call_span);
                }
                unit()
            }
            "barrier" => {
                arity(0, self);
                unit()
            }
            "async_copy" => {
                if !arity(3, self) {
                    return unit();
                }
                self.raw_index_arg(prim, &arg_types[2], &call_span);
                // Capability, not choice: THIS lowering's machine must declare the engine on
                // its own edge. Another machine declaring `copy_engine` on a like-named edge
                // is another machine's hardware -- the lookup used to scan every declared
                // topology, so one machine's declaration armed every machine's lowerings for
                // that edge pair (found by review, Vx#353). Absence of the declaration IS
                // absence of the capability.
                if !self.edge_has_copy_engine(&from, &to, &machine) {
                    self.errors.error_with_code(
                        DiagnosticCode::E6020,
                        format!(
                            "`raw::async_copy` needs a copy engine on {} -> {}, and \
                             Topology::{} does not declare one there. Add `copy_engine` to \
                             {}'s own edge (`transfer Memory::{} -> Memory::{} copy_engine`) \
                             if its hardware has the engine -- another machine declaring it \
                             on a like-named edge is not this machine's capability",
                            from.name(),
                            to.name(),
                            machine,
                            machine,
                            from.name(),
                            to.name()
                        ),
                        Some(SourceSpan::from_ast_span(&call_span)),
                    );
                }
                let dst = self.raw_tile_arg(prim, &args[0], &arg_types[0], true, &call_span);
                let src = self.raw_tile_arg(prim, &args[1], &arg_types[1], false, &call_span);
                let tiles: Vec<TileInfo> = dst.into_iter().chain(src).collect();
                if !tiles.is_empty() {
                    self.raw_bounds_obligation(prim, &args[2], &tiles, &call_span);
                }
                unit()
            }
            "async_wait" => {
                arity(0, self);
                unit()
            }
            _ => {
                self.errors.error_with_code(
                    DiagnosticCode::E6017,
                    format!(
                        "unknown primitive `raw::{prim}`; the primitive set is raw::{{{}}}",
                        RAW_PRIMITIVES.join(", ")
                    ),
                    span,
                );
                unit()
            }
        }
    }

    /// A tile argument must be a bare identifier (indexed, not addressed: no expression
    /// computes a tile) whose type peels to a tensor. `need_mut` additionally demands the
    /// `&mut` the store side of the contract requires -- exclusivity itself stays the
    /// borrow checker's job.
    fn raw_tile_arg(
        &mut self,
        prim: &str,
        arg: &Expr,
        arg_ty: &Type,
        need_mut: bool,
        call_span: &Span,
    ) -> Option<TileInfo> {
        let span = Some(SourceSpan::from_ast_span(call_span));
        let Expr::Identifier(id) = arg else {
            self.errors.error_with_code(
                DiagnosticCode::E6017,
                format!(
                    "the tile argument of `raw::{prim}` must be a bare name -- the primitives \
                     are indexed, not addressed, so no expression can compute a tile"
                ),
                span,
            );
            return None;
        };
        // The name must be one of the lowering's own parameters. A local alias
        // (`let d2 = dst`) would carry the same tensor type but a different name, and a
        // different name is invisible to the async-discipline walk -- reviewed and
        // reproduced: the alias read an in-flight destination without a diagnostic.
        // Locals also have no place in the edge's space mapping (a stack array is in
        // neither `from` nor `to`).
        if !self.transfer_lowering_params.contains(&id.name) {
            self.errors.error_with_code(
                DiagnosticCode::E6017,
                format!(
                    "`{}` is not a parameter of this lowering; `raw::{prim}` moves only \
                     the tiles the transfer is given, not local copies or aliases",
                    id.name
                ),
                span,
            );
            return None;
        }
        if need_mut && !matches!(arg_ty, Type::Borrow { is_mut: true, .. }) {
            self.errors.error_with_code(
                DiagnosticCode::E6017,
                format!(
                    "`raw::{prim}` writes into `{}`, which must be held by `&mut`",
                    id.name
                ),
                span.clone(),
            );
        }
        let Some((elem, dims, _)) = Self::as_tensor_operand(arg_ty) else {
            self.errors.error_with_code(
                DiagnosticCode::E6017,
                format!(
                    "`{}` is not a tile: `raw::{prim}` needs a tensor-typed parameter, \
                     got {:?}",
                    id.name, arg_ty
                ),
                span,
            );
            return None;
        };
        Some(TileInfo {
            elem: elem.clone(),
            static_extent: Self::static_extent_of_dims(dims),
            name: id.name.clone(),
        })
    }

    /// The element count when every declared dimension is a literal; `None` otherwise.
    fn static_extent_of_dims(dims: &[Expr]) -> Option<u64> {
        let mut product: u64 = 1;
        for d in dims {
            let Expr::Number(n) = d else { return None };
            let v: u64 = n.value.as_ref().parse().ok()?;
            product = product.checked_mul(v)?;
        }
        Some(product)
    }

    /// Indices are integers. Both i32 and i64 are accepted -- a bare literal defaults to
    /// i32 (#240) and the extent-driven form `for i in 0..raw::extent(t)` binds i64, and
    /// rejecting either would force a cast that proves nothing.
    fn raw_index_arg(&mut self, prim: &str, ty: &Type, call_span: &Span) {
        use crate::syntax::ElementType::*;
        if !matches!(ty, Type::Scalar(I32) | Type::Scalar(I64)) {
            self.errors.error_with_code(
                DiagnosticCode::E6017,
                format!("the index of `raw::{prim}` must be an integer, got {ty:?}"),
                Some(SourceSpan::from_ast_span(call_span)),
            );
        }
    }

    /// Discharge `0 <= i < extent(t)` for every tile the call touches, through the same
    /// `prove_expr` refutation loop `Verified<T>` uses. Unprovable is an error -- unless
    /// the call sits in an `unsafe` block, which records the asserted obligation as a
    /// warning instead, the same standing an unverified `spec:` figure has. Note the
    /// prover fails OPEN when z3 is absent (see `hir/prover.rs`), so this is a proof
    /// when the toolchain is complete and a recorded assumption when it is not.
    fn raw_bounds_obligation(
        &mut self,
        prim: &str,
        index: &Expr,
        tiles: &[TileInfo],
        call_span: &Span,
    ) {
        let span = Some(SourceSpan::from_ast_span(call_span));
        // The prover fails OPEN without z3 (see `hir/prover.rs`), which would turn every
        // bounds obligation into a silent yes. Say so once per obligation instead: the
        // program still compiles, but the compile record shows what was assumed.
        if !z3_available() {
            self.errors.push_warning(format!(
                "z3 is not installed, so the bounds of `raw::{prim}` are recorded as \
                 assumed, not proven"
            ));
            return;
        }
        let idx = self.fold_raw_extent(index);
        for tile in tiles {
            let Some(extent) = tile.static_extent else {
                let msg = format!(
                    "the extent of `{}` is not static, so the bounds of `raw::{prim}` \
                     cannot be proven",
                    tile.name
                );
                if self.in_unsafe_block {
                    self.errors
                        .push_warning(format!("{msg}; asserted, not proven (unsafe)"));
                } else {
                    self.errors
                        .error_with_code(DiagnosticCode::E6018, msg, span.clone());
                }
                continue;
            };
            let obligation = Expr::LogicalOp(LogicalOpExpr {
                lhs: Box::new(Expr::RelationalOp(RelationalOpExpr {
                    lhs: Box::new(idx.clone()),
                    op: RelationalOp::Ge,
                    rhs: Box::new(Self::number_expr(0)),
                    span: Span::default(),
                })),
                op: LogicalOp::And,
                rhs: Box::new(Expr::RelationalOp(RelationalOpExpr {
                    lhs: Box::new(idx.clone()),
                    op: RelationalOp::Lt,
                    rhs: Box::new(Self::number_expr(extent as i64)),
                    span: Span::default(),
                })),
                span: Span::default(),
            });
            if self.prove_expr(&obligation) {
                continue;
            }
            let msg = format!(
                "cannot prove the index of `raw::{prim}` stays inside `{}` \
                 (needs 0 <= index < {extent})",
                tile.name
            );
            if self.in_unsafe_block {
                self.errors
                    .push_warning(format!("{msg}; asserted, not proven (unsafe)"));
            } else {
                self.errors.error_with_code(
                    DiagnosticCode::E6018,
                    format!(
                        "{msg}. Prove it with a loop bound or invariant, or assert it \
                             in an `unsafe` block"
                    ),
                    span.clone(),
                );
            }
        }
    }

    fn number_expr(v: i64) -> Expr {
        Expr::Number(NumberExpr {
            value: v.to_string().into(),
            ty: None,
            span: Span::default(),
        })
    }

    /// Replace `raw::extent(t)` with its literal element count wherever `t` is in scope
    /// with a statically shaped tensor type. The prover cannot lower a function call, so
    /// without this fold every fact and obligation mentioning an extent is unprovable.
    /// Only the operator shapes facts are built from are walked; anything else is
    /// returned as written.
    pub(crate) fn fold_raw_extent(&self, e: &Expr) -> Expr {
        match e {
            Expr::FunctionCall(fc) if fc.name.as_ref() == "raw::extent" && fc.args.len() == 1 => {
                if let Expr::Identifier(id) = &fc.args[0] {
                    if let Some((ty, _)) = self.lookup(id.name.as_ref()) {
                        if let Some((_, dims, _)) = Self::as_tensor_operand(ty) {
                            if let Some(n) = Self::static_extent_of_dims(dims) {
                                return Self::number_expr(n as i64);
                            }
                        }
                    }
                }
                e.clone()
            }
            Expr::BinaryOp(b) => {
                let mut b = b.clone();
                b.lhs = Box::new(self.fold_raw_extent(&b.lhs));
                b.rhs = Box::new(self.fold_raw_extent(&b.rhs));
                Expr::BinaryOp(b)
            }
            Expr::RelationalOp(r) => {
                let mut r = r.clone();
                r.lhs = Box::new(self.fold_raw_extent(&r.lhs));
                r.rhs = Box::new(self.fold_raw_extent(&r.rhs));
                Expr::RelationalOp(r)
            }
            Expr::LogicalOp(l) => {
                let mut l = l.clone();
                l.lhs = Box::new(self.fold_raw_extent(&l.lhs));
                l.rhs = Box::new(self.fold_raw_extent(&l.rhs));
                Expr::LogicalOp(l)
            }
            Expr::UnaryOp(u) => {
                let mut u = u.clone();
                u.expr = Box::new(self.fold_raw_extent(&u.expr));
                Expr::UnaryOp(u)
            }
            _ => e.clone(),
        }
    }

    /// Does `machine`'s own declaration of the edge `from -> to` carry a copy engine?
    ///
    /// Scoped to one topology on purpose. The unscoped form ("does ANY declared topology's
    /// edge carry it") let a lowering for Hopper use `raw::async_copy` because Ampere
    /// declared `copy_engine` on its own like-named edge -- one machine's capability arming
    /// another machine's code, which is the conflation the `for Topology::X` clause exists
    /// to remove.
    pub(crate) fn edge_has_copy_engine(
        &self,
        from: &crate::syntax::MemorySpace,
        to: &crate::syntax::MemorySpace,
        machine: &str,
    ) -> bool {
        self.env
            .topologies
            .values()
            .filter(|t| t.name.as_ref() == machine)
            .any(|t| {
                t.descriptor
                    .transfers
                    .iter()
                    .any(|e| &e.from == from && &e.to == to && e.copy_engine)
            })
    }

    /// The whole-body half of the A2 obligations, run once after every lowering body has
    /// been type-checked (the per-call-site half lives in `check_raw_primitive`). Checks,
    /// per `impl transfer from -> to`:
    ///
    ///   * the edge exists on some declared topology (warning if not: the lowering is
    ///     carried but nothing can execute it), and every topology that declares it can
    ///     see both endpoint spaces (E6022, constraint C6);
    ///   * `raw::barrier()` appears only as a top-level statement (E6019) and never
    ///     inside a closure (E6017);
    ///   * async-copy discipline: no read or write of a destination while a copy into it
    ///     is outstanding, no copies leaking past the body or a loop iteration (E6021);
    ///   * a body that publishes (stores or async-copies) in a lowering for a
    ///     synchronizing edge ends with `raw::barrier()` (E6021) -- the syntactic form
    ///     of the seam obligation: without the trailing barrier the transfer is
    ///     `Relaxed {{ published }}` and `hir/seam.rs`'s stale read is reachable.
    pub fn check_transfer_impl_bodies(&mut self, impls: &[crate::syntax::TransferImplDecl]) {
        for t in impls {
            // The topology this lowering is FOR, and its declaration of this edge. Before the
            // `for` clause existed this had to search every declared topology for one that
            // declared the edge, and then hold every one of them to the visibility rule --
            // including machines the lowering was never written for. Now it names its machine,
            // so the check applies to that machine.
            let declaring: Vec<(
                &crate::symbol::Symbol,
                &Vec<crate::syntax::MemorySpace>,
                &crate::arch::TransferEdge,
            )> = self
                .env
                .topologies
                .values()
                .filter(|topo| topo.name.as_ref() == t.topology.display_name())
                .flat_map(|topo| {
                    topo.descriptor
                        .transfers
                        .iter()
                        .filter(|e| e.from == t.from && e.to == t.to)
                        .map(move |e| (&topo.name, &topo.descriptor.visibility, e))
                })
                .collect();

            // The whole-body checks and the edge-level checks below apply to lowerings
            // written against the primitives. An A1-era body that never says `raw::` is
            // carried code -- warning it (or refusing its edge) would fail every
            // existing program the moment a fleet file declares the edge.
            let scans: Vec<RawScan> = t
                .methods
                .iter()
                .map(|f| {
                    let mut scan = RawScan::default();
                    scan_stmts(&f.body, false, &mut scan);
                    scan
                })
                .collect();
            let uses_raw = scans.iter().any(|s| !s.calls.is_empty());

            if uses_raw && declaring.is_empty() {
                self.errors.push_warning(format!(
                    "`impl Transfer<Memory::{}, Memory::{}> for Topology::{}` matches no declared \
                     topology edge; the lowering is carried but nothing can execute it",
                    t.from.name(),
                    t.to.name(),
                    t.topology.display_name()
                ));
            }
            if uses_raw {
                for (topo_name, vis, _) in &declaring {
                    for space in [&t.from, &t.to] {
                        // The host side of a host link is exempt: every declared topology
                        // reaches CPU_DRAM by construction (the unified-memory rule in
                        // `arch.rs::is_type_accessible`), and the shipped fleet files all
                        // declare host edges without listing CPU_DRAM as visible.
                        if matches!(space, crate::syntax::MemorySpace::CPUDRAM) {
                            continue;
                        }
                        if !vis.contains(space) {
                            self.errors.error_with_code(
                                DiagnosticCode::E6022,
                                format!(
                                    "topology {} declares the edge {} -> {} but cannot see \
                                     Memory::{}; a lowering for this edge would run on a \
                                     part that cannot address the space it moves bytes \
                                     through (constraint C6 of the transfer contract)",
                                    topo_name,
                                    t.from.name(),
                                    t.to.name(),
                                    space.name()
                                ),
                                None,
                            );
                        }
                    }
                }
            }
            // The body requires a trailing barrier only when some declaring topology grades
            // the edge synchronizing (the default). An edge every topology declares relaxed
            // has opted out of the visibility guarantee -- and gets the seam engine's
            // W1027 treatment at the declaration instead.
            let edge_requires_sync =
                declaring.is_empty() || declaring.iter().any(|(_, _, e)| e.sync);

            for (f, scan) in t.methods.iter().zip(&scans) {
                if scan.calls.is_empty() {
                    continue; // Not written against the primitives (yet); A1-era body.
                }

                for (name, span, in_closure) in &scan.calls {
                    if *in_closure {
                        self.errors.error_with_code(
                            DiagnosticCode::E6017,
                            format!(
                                "`{name}` cannot appear inside a closure: a lowering's \
                                 moves must run when the lowering runs"
                            ),
                            Some(SourceSpan::from_ast_span(span)),
                        );
                    }
                }

                // "Top level" sees through `unsafe { }`: wrapping the body in unsafe is
                // the documented escape for unprovable bounds, and it must not demote the
                // barrier inside it to "under control flow".
                let top = top_level_view(&f.body);

                // Every barrier must be one of the body's own top-level statements.
                let top_level_barriers: Vec<*const Expr> = top
                    .iter()
                    .filter_map(|s| match s {
                        Statement::ExprStmt(es) if is_raw_call(&es.expr, "raw::barrier") => {
                            Some(&es.expr as *const Expr)
                        }
                        _ => None,
                    })
                    .collect();
                for (ptr, span) in &scan.barriers {
                    if !top_level_barriers.contains(ptr) {
                        self.errors.error_with_code(
                            DiagnosticCode::E6019,
                            "`raw::barrier()` must be reachable by every lane, and inside \
                             an `if`, `match`, or loop the compiler cannot guarantee \
                             that. Move the barrier to the top level of the body"
                                .to_string(),
                            Some(SourceSpan::from_ast_span(span)),
                        );
                    }
                }

                // A publishing lowering on a synchronizing edge has ONE exit, at the
                // bottom, past the barrier. An early return would leave a path that
                // publishes and exits unsynchronized -- and if the exit is taken by some
                // lanes only, the remaining lanes wait at a barrier the exited lanes
                // never reach. Reviewed and reproduced before this check existed.
                let final_return: Option<*const Statement> = match top.last() {
                    Some(s @ Statement::Return(_)) => Some(*s as *const Statement),
                    _ => None,
                };
                // Unconditional, for every lowering on every edge grade: the body is
                // INLINED at the transfer site, so a `return` anywhere but the bottom
                // returns from the function that contains the transfer. Reproduced
                // before this dropped its edge-grade guard: a lowering on a `relaxed`
                // edge returned 7 out of `main`, skipping the rest of the program, and
                // the same shape inside a kernel produced invalid IR. On a
                // synchronizing edge the rule is also what keeps every path past the
                // trailing barrier.
                for (ptr, span) in &scan.returns {
                    if Some(*ptr) != final_return {
                        self.errors.error_with_code(
                            DiagnosticCode::E6021,
                            "`return` before the end of a lowering: the body is inlined \
                             at the transfer site, so this would return from whatever \
                             function performs the transfer. A lowering returns once, \
                             at the bottom"
                                .to_string(),
                            Some(SourceSpan::from_ast_span(span)),
                        );
                    }
                }

                // Forward walk for the async discipline. Early exits are real exits: a
                // `return` records its in-flight state, and the leak check below unions
                // every way out of the body.
                let mut state = DiscState::default();
                let mut cx = WalkCx::default();
                self.discipline_stmts(&f.body, &mut state, &mut cx);
                let mut leaks = state.outstanding;
                for exit in cx.fn_exits {
                    for (tile, span) in exit {
                        leaks.entry(tile).or_insert(span);
                    }
                }
                for (tile, span) in &leaks {
                    self.errors.error_with_code(
                        DiagnosticCode::E6021,
                        format!(
                            "`raw::async_copy` into `{tile}` has no matching \
                             `raw::async_wait` on some path out of the body; the copy \
                             may still be in flight when the transfer returns"
                        ),
                        Some(SourceSpan::from_ast_span(span)),
                    );
                }

                // The trailing synchronization grade (C3, syntactic form).
                if edge_requires_sync && scan.publishes {
                    // Every Vx function ends with a `return` (the language has no void
                    // functions), and a return carries no synchronization -- the barrier
                    // is the last statement BEFORE it. The return's own expression must
                    // not publish either: it runs after the barrier.
                    let mut it = top.iter().rev();
                    let mut last = it.next();
                    let mut return_publishes = false;
                    if let Some(Statement::Return(r)) = last {
                        let mut ret_scan = RawScan::default();
                        scan_expr(&r.expr, false, &mut ret_scan);
                        return_publishes = ret_scan.publishes;
                        last = it.next();
                    }
                    let ends_with_barrier = !return_publishes
                        && matches!(
                            last,
                            Some(Statement::ExprStmt(es)) if is_raw_call(&es.expr, "raw::barrier")
                        );
                    if !ends_with_barrier {
                        let anchor = top.last().map(|s| SourceSpan::from_ast_span(&stmt_span(s)));
                        self.errors.error_with_code(
                            DiagnosticCode::E6021,
                            format!(
                                "lowering `{}` for the synchronizing edge {} -> {} writes \
                                 the destination but does not end with `raw::barrier()`, \
                                 so another lane can read stale data (the transfer \
                                 visibility rule). Add the barrier as the last statement \
                                 before the return, or declare the edge `relaxed` in the \
                                 machine file",
                                f.name,
                                t.from.name(),
                                t.to.name()
                            ),
                            anchor,
                        );
                    }
                }
            }
        }
    }

    /// The forward walk: statements in order, branches joined by union, and every way
    /// OUT of a block modeled -- a `return` snapshots its in-flight state for the
    /// end-of-body leak check, a `break` snapshots it for the loop-exit join, and a
    /// `continue` feeds the next iteration. Reviewed and reproduced: without these,
    /// a lane-guarded early return carried an unwaited copy straight past every check.
    fn discipline_stmts(&mut self, stmts: &[Statement], st: &mut DiscState, cx: &mut WalkCx) {
        for s in stmts {
            match s {
                Statement::LetDecl(l) => self.discipline_expr(&l.expr, st, cx),
                Statement::Return(r) => {
                    self.discipline_expr(&r.expr, st, cx);
                    cx.fn_exits.push(st.outstanding.clone());
                }
                Statement::ExprStmt(e) => self.discipline_expr(&e.expr, st, cx),
                Statement::Assign(a) => {
                    self.discipline_expr(&a.rhs, st, cx);
                    self.discipline_expr(&a.lhs, st, cx);
                }
                Statement::CompoundAssign(c) => {
                    self.discipline_expr(&c.rhs, st, cx);
                    self.discipline_expr(&c.lhs, st, cx);
                }
                Statement::Assert(a) => self.discipline_expr(&a.expr, st, cx),
                Statement::ForLoop(fl) => {
                    self.discipline_expr(&fl.iterable, st, cx);
                    self.discipline_loop_body(&fl.body, st, cx);
                }
                Statement::Loop(lp) => {
                    self.discipline_loop_body(&lp.body, st, cx);
                }
                Statement::Break(_) => {
                    if let Some(frame) = cx.break_exits.last_mut() {
                        frame.push(st.outstanding.clone());
                    }
                }
                Statement::Continue(_) => {
                    if let Some(frame) = cx.continue_heads.last_mut() {
                        frame.push(st.outstanding.clone());
                    }
                }
                Statement::MacroCall(_) => {} // Expanded before checking; nothing survives here.
                Statement::Error(_) => {}
            }
        }
    }

    /// Copies MAY stay outstanding across loop iterations -- issuing one copy per
    /// iteration and waiting once after the loop is exactly how a copy engine is used
    /// (Ampere's commit-group pattern). What a loop must not do is READ a destination a
    /// previous iteration left in flight, so the body is walked twice: the second pass
    /// starts from the first pass's exit state (with `continue` paths unioned in), which
    /// is where a read at the top of iteration N+1 meets a copy issued at the bottom of
    /// iteration N. The reported-site set keeps the repeat walk from doubling every
    /// diagnostic. The state after the loop is the union of every way out: the natural
    /// exit, every `break` snapshot, and the entry state (the loop may run zero times).
    fn discipline_loop_body(&mut self, body: &[Statement], st: &mut DiscState, cx: &mut WalkCx) {
        let entry = st.clone();
        cx.break_exits.push(Vec::new());
        cx.continue_heads.push(Vec::new());
        self.discipline_stmts(body, st, cx);
        for head in cx.continue_heads.last().cloned().unwrap_or_default() {
            for (tile, span) in head {
                st.outstanding.entry(tile).or_insert(span);
            }
        }
        self.discipline_stmts(body, st, cx);
        let breaks = cx.break_exits.pop().unwrap_or_default();
        cx.continue_heads.pop();
        for exit in breaks {
            for (tile, span) in exit {
                st.outstanding.entry(tile).or_insert(span);
            }
        }
        for (tile, span) in entry.outstanding {
            st.outstanding.entry(tile).or_insert(span);
        }
    }

    /// Expression half of the walk: recurse in evaluation order, apply each `raw::` call's
    /// effect, and join branches by union.
    fn discipline_expr(&mut self, e: &Expr, st: &mut DiscState, cx: &mut WalkCx) {
        match e {
            Expr::FunctionCall(fc) => {
                for a in &fc.args {
                    self.discipline_expr(a, st, cx);
                }
                self.apply_call_effect(fc, st);
            }
            Expr::If(ifx) => {
                self.discipline_expr(&ifx.cond, st, cx);
                let mut then_st = st.clone();
                self.discipline_stmts(&ifx.then_block, &mut then_st, cx);
                let mut else_st = st.clone();
                if let Some(eb) = &ifx.else_block {
                    self.discipline_stmts(eb, &mut else_st, cx);
                }
                *st = then_st.union(else_st);
            }
            Expr::Match(m) => {
                self.discipline_expr(&m.expr, st, cx);
                let entry = st.clone();
                let mut joined: Option<DiscState> = None;
                for arm in &m.arms {
                    let mut arm_st = entry.clone();
                    self.discipline_stmts(&arm.body, &mut arm_st, cx);
                    joined = Some(match joined {
                        None => arm_st,
                        Some(j) => j.union(arm_st),
                    });
                }
                if let Some(j) = joined {
                    *st = j;
                }
            }
            Expr::UnsafeBlock(u) => {
                self.discipline_stmts(&u.stmts, st, cx);
                if let Some(r) = &u.ret {
                    self.discipline_expr(r, st, cx);
                }
            }
            Expr::ComptimeBlock(c) => {
                self.discipline_stmts(&c.stmts, st, cx);
                if let Some(r) = &c.ret {
                    self.discipline_expr(r, st, cx);
                }
            }
            Expr::SpawnOn(sp) => {
                self.discipline_stmts(&sp.stmts, st, cx);
                if let Some(r) = &sp.ret {
                    self.discipline_expr(r, st, cx);
                }
            }
            // Closures are refused outright by the scan (E6017), so their bodies are not
            // part of the discipline.
            Expr::Closure(_) => {}
            Expr::EnumVariant(ev) => {
                for a in ev.payload.iter().flatten() {
                    self.discipline_expr(a, st, cx);
                }
            }
            Expr::Transfer(tr) => self.discipline_expr(&tr.expr, st, cx),
            Expr::IndirectCall(ic) => {
                self.discipline_expr(&ic.callee, st, cx);
                for a in &ic.args {
                    self.discipline_expr(a, st, cx);
                }
            }
            Expr::Array(a) => {
                for el in &a.elements {
                    self.discipline_expr(el, st, cx);
                }
            }
            Expr::MemberAccess(m) => self.discipline_expr(&m.base, st, cx),
            Expr::IndexAccess(ix) => {
                self.discipline_expr(&ix.base, st, cx);
                self.discipline_expr(&ix.index, st, cx);
            }
            Expr::MethodCall(mc) => {
                self.discipline_expr(&mc.base, st, cx);
                for a in &mc.args {
                    self.discipline_expr(a, st, cx);
                }
            }
            Expr::BinaryOp(b) => {
                self.discipline_expr(&b.lhs, st, cx);
                self.discipline_expr(&b.rhs, st, cx);
            }
            Expr::RelationalOp(r) => {
                self.discipline_expr(&r.lhs, st, cx);
                self.discipline_expr(&r.rhs, st, cx);
            }
            Expr::LogicalOp(l) => {
                self.discipline_expr(&l.lhs, st, cx);
                self.discipline_expr(&l.rhs, st, cx);
            }
            Expr::UnaryOp(u) => self.discipline_expr(&u.expr, st, cx),
            Expr::Borrow(b) => self.discipline_expr(&b.expr, st, cx),
            Expr::Dereference(d) => self.discipline_expr(&d.expr, st, cx),
            Expr::StructInit(si) => {
                for (_, fe) in &si.fields {
                    self.discipline_expr(fe, st, cx);
                }
            }
            Expr::Range(r) => {
                self.discipline_expr(&r.start, st, cx);
                self.discipline_expr(&r.end, st, cx);
            }
            Expr::Grad(g) => {
                for a in &g.args {
                    self.discipline_expr(a, st, cx);
                }
            }
            Expr::Vjp(v) => {
                for a in &v.args {
                    self.discipline_expr(a, st, cx);
                }
                self.discipline_expr(&v.cotangent, st, cx);
            }
            Expr::Jvp(j) => {
                for a in &j.args {
                    self.discipline_expr(a, st, cx);
                }
                self.discipline_expr(&j.tangent, st, cx);
            }
            Expr::VecMacro(v) => {
                for el in &v.elements {
                    self.discipline_expr(el, st, cx);
                }
            }
            Expr::AsCast(c) => self.discipline_expr(&c.expr, st, cx),
            Expr::Print(p) => {
                for a in &p.args {
                    self.discipline_expr(a, st, cx);
                }
            }
            Expr::Println(p) => {
                for a in &p.args {
                    self.discipline_expr(a, st, cx);
                }
            }
            Expr::InlineMlir(im) => {
                for (_, ie, _) in &im.inputs {
                    self.discipline_expr(ie, st, cx);
                }
                for cl in &im.clobbers {
                    self.discipline_expr(cl, st, cx);
                }
            }
            Expr::Identifier(_)
            | Expr::Number(_)
            | Expr::StringLiteral(_)
            | Expr::TransferPredicate(_)
            | Expr::MemorySpace(_)
            | Expr::Topology(_)
            | Expr::MacroCall(_)
            | Expr::SizeOf(_) => {}
        }
    }

    /// One `raw::` call's effect on the async state.
    fn apply_call_effect(&mut self, fc: &FunctionCallExpr, st: &mut DiscState) {
        let name = fc.name.as_ref();
        let tile_name = |i: usize| -> Option<crate::symbol::Symbol> {
            match fc.args.get(i) {
                Some(Expr::Identifier(id)) => Some(id.name.clone()),
                _ => None,
            }
        };
        match name {
            "raw::async_copy" => {
                if let Some(dst) = tile_name(0) {
                    st.outstanding.insert(dst, fc.span);
                }
                if let Some(src) = tile_name(1) {
                    st.outstanding_src.insert(src, fc.span);
                }
            }
            "raw::async_wait" => {
                st.outstanding.clear();
                st.outstanding_src.clear();
            }
            "raw::load" | "raw::store" => {
                if let Some(t) = tile_name(0) {
                    if name == "raw::store"
                        && st.outstanding_src.contains_key(&t)
                        && st
                            .reported
                            .borrow_mut()
                            .insert((fc.span.line, fc.span.column))
                    {
                        self.errors.error_with_code(
                            DiagnosticCode::E6021,
                            format!(
                                "`{t}` is written while it is the source of an in-flight \
                                 `raw::async_copy`; the engine is still reading it, so \
                                 `raw::async_wait()` must come first"
                            ),
                            Some(SourceSpan::from_ast_span(&fc.span)),
                        );
                    }
                    if st.outstanding.contains_key(&t)
                        && st
                            .reported
                            .borrow_mut()
                            .insert((fc.span.line, fc.span.column))
                    {
                        let verb = if name == "raw::load" {
                            "read"
                        } else {
                            "written"
                        };
                        self.errors.error_with_code(
                            DiagnosticCode::E6021,
                            format!(
                                "`{t}` is {verb} while a `raw::async_copy` into it is \
                                 still outstanding; `raw::async_wait()` must come first \
                                 (its contents are unspecified until the wait)"
                            ),
                            Some(SourceSpan::from_ast_span(&fc.span)),
                        );
                    }
                }
            }
            _ => {}
        }
    }
}

/// The async-discipline state: destination tiles with an outstanding copy, keyed to the
/// span of the `async_copy` that started it (so the leak diagnostic points at the copy).
/// `reported` is shared across clones (branch states, the loop's second pass) so one
/// hazard site is diagnosed once no matter how many paths reach it.
#[derive(Default, Clone)]
struct DiscState {
    outstanding: std::collections::HashMap<crate::symbol::Symbol, crate::syntax::Span>,
    /// Source tiles of in-flight copies: the engine is READING these, so the program
    /// must not write them until the wait (reading them is fine).
    outstanding_src: std::collections::HashMap<crate::symbol::Symbol, crate::syntax::Span>,
    reported: std::rc::Rc<std::cell::RefCell<std::collections::HashSet<(usize, usize)>>>,
}

impl DiscState {
    /// Branch join: outstanding on either path is outstanding after it.
    fn union(mut self, other: DiscState) -> DiscState {
        for (k, v) in other.outstanding {
            self.outstanding.entry(k).or_insert(v);
        }
        for (k, v) in other.outstanding_src {
            self.outstanding_src.entry(k).or_insert(v);
        }
        self
    }
}

/// The ways out of the walk: `return` snapshots feed the end-of-body leak check;
/// `break`/`continue` snapshots feed the enclosing loop's joins. One frame per loop
/// nesting level.
#[derive(Default)]
struct WalkCx {
    fn_exits: Vec<std::collections::HashMap<crate::symbol::Symbol, crate::syntax::Span>>,
    break_exits: Vec<Vec<std::collections::HashMap<crate::symbol::Symbol, crate::syntax::Span>>>,
    continue_heads: Vec<Vec<std::collections::HashMap<crate::symbol::Symbol, crate::syntax::Span>>>,
}

/// Is this lowering body written against the `raw::` primitives?
///
/// The whole-body contract in `check_transfer_impl_bodies` -- the trailing barrier (C3), the
/// early-return refusal, the async discipline -- is skipped for a body with no `raw::` call,
/// because such bodies predate the primitives and are CARRIED rather than emitted. So a body
/// that answers `false` here must never be emitted: doing so takes the skip and the emission
/// both, and the emitted transfer ends up with no barrier from either source.
pub(crate) fn body_uses_raw(body: &[Statement]) -> bool {
    let mut scan = RawScan::default();
    scan_stmts(body, false, &mut scan);
    !scan.calls.is_empty()
}

/// What one deep scan of a body collects.
#[derive(Default)]
struct RawScan {
    /// Every `raw::` call: name, span, and whether a closure encloses it.
    calls: Vec<(crate::symbol::Symbol, crate::syntax::Span, bool)>,
    /// Every `raw::barrier` call, identified by node address so the top-level check can
    /// tell the body's own statements from occurrences nested anywhere deeper.
    barriers: Vec<(*const Expr, crate::syntax::Span)>,
    /// Does the body write at all (store or async_copy)? A read-only body publishes
    /// nothing and needs no trailing barrier.
    publishes: bool,
    /// Every `return` statement, by node address and span. The whole-body pass allows
    /// exactly one, at the bottom of a publishing lowering on a synchronizing edge.
    returns: Vec<(*const Statement, crate::syntax::Span)>,
}

/// Is `e` exactly a call to `name`?
fn is_raw_call(e: &Expr, name: &str) -> bool {
    matches!(e, Expr::FunctionCall(fc) if fc.name.as_ref() == name)
}

/// Is z3 on PATH? Checked once per process; the answer decides whether a bounds
/// obligation is a proof or a recorded assumption.
fn z3_available() -> bool {
    static Z3: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *Z3.get_or_init(|| {
        std::process::Command::new("z3")
            .arg("--version")
            .output()
            .is_ok()
    })
}

/// Deep scan over statements. `depth` counts nested statement blocks (0 = the body's own
/// statement list); `in_closure` is sticky below a closure literal.
fn scan_stmts(stmts: &[Statement], in_closure: bool, out: &mut RawScan) {
    for s in stmts {
        match s {
            Statement::LetDecl(l) => scan_expr(&l.expr, in_closure, out),
            Statement::Return(r) => {
                out.returns.push((s as *const Statement, r.span));
                scan_expr(&r.expr, in_closure, out);
            }
            Statement::ExprStmt(e) => scan_expr(&e.expr, in_closure, out),
            Statement::Assign(a) => {
                scan_expr(&a.lhs, in_closure, out);
                scan_expr(&a.rhs, in_closure, out);
            }
            Statement::CompoundAssign(c) => {
                scan_expr(&c.lhs, in_closure, out);
                scan_expr(&c.rhs, in_closure, out);
            }
            Statement::Assert(a) => scan_expr(&a.expr, in_closure, out),
            Statement::ForLoop(fl) => {
                scan_expr(&fl.iterable, in_closure, out);
                for inv in &fl.invariants {
                    scan_expr(inv, in_closure, out);
                }
                scan_stmts(&fl.body, in_closure, out);
            }
            Statement::Loop(lp) => {
                for inv in &lp.invariants {
                    scan_expr(inv, in_closure, out);
                }
                scan_stmts(&lp.body, in_closure, out);
            }
            Statement::Break(_) | Statement::Continue(_) => {}
            Statement::MacroCall(_) => {}
            Statement::Error(_) => {}
        }
    }
}

/// Deep scan over one expression. Matches every `Expr` variant -- no catch-all -- so a
/// new syntax node cannot silently hide a `raw::` call from the whole-body checks.
fn scan_expr(e: &Expr, in_closure: bool, out: &mut RawScan) {
    match e {
        Expr::FunctionCall(fc) => {
            let name = fc.name.as_ref();
            if let Some(prim) = name.strip_prefix("raw::") {
                out.calls.push((fc.name.clone(), fc.span, in_closure));
                if prim == "barrier" {
                    out.barriers.push((e as *const Expr, fc.span));
                }
                if prim == "store" || prim == "async_copy" {
                    out.publishes = true;
                }
            }
            for a in &fc.args {
                scan_expr(a, in_closure, out);
            }
        }
        Expr::If(ifx) => {
            scan_expr(&ifx.cond, in_closure, out);
            scan_stmts(&ifx.then_block, in_closure, out);
            if let Some(eb) = &ifx.else_block {
                scan_stmts(eb, in_closure, out);
            }
        }
        Expr::Match(m) => {
            scan_expr(&m.expr, in_closure, out);
            for arm in &m.arms {
                scan_stmts(&arm.body, in_closure, out);
            }
        }
        Expr::UnsafeBlock(u) => {
            scan_stmts(&u.stmts, in_closure, out);
            if let Some(r) = &u.ret {
                scan_expr(r, in_closure, out);
            }
        }
        Expr::ComptimeBlock(c) => {
            scan_stmts(&c.stmts, in_closure, out);
            if let Some(r) = &c.ret {
                scan_expr(r, in_closure, out);
            }
        }
        Expr::SpawnOn(sp) => {
            scan_stmts(&sp.stmts, in_closure, out);
            if let Some(r) = &sp.ret {
                scan_expr(r, in_closure, out);
            }
        }
        Expr::Closure(cl) => scan_expr(&cl.body, true, out),
        Expr::EnumVariant(ev) => {
            for a in ev.payload.iter().flatten() {
                scan_expr(a, in_closure, out);
            }
        }
        Expr::Transfer(tr) => scan_expr(&tr.expr, in_closure, out),
        Expr::IndirectCall(ic) => {
            scan_expr(&ic.callee, in_closure, out);
            for a in &ic.args {
                scan_expr(a, in_closure, out);
            }
        }
        Expr::Array(a) => {
            for el in &a.elements {
                scan_expr(el, in_closure, out);
            }
        }
        Expr::MemberAccess(m) => scan_expr(&m.base, in_closure, out),
        Expr::IndexAccess(ix) => {
            scan_expr(&ix.base, in_closure, out);
            scan_expr(&ix.index, in_closure, out);
        }
        Expr::MethodCall(mc) => {
            scan_expr(&mc.base, in_closure, out);
            for a in &mc.args {
                scan_expr(a, in_closure, out);
            }
        }
        Expr::BinaryOp(b) => {
            scan_expr(&b.lhs, in_closure, out);
            scan_expr(&b.rhs, in_closure, out);
        }
        Expr::RelationalOp(r) => {
            scan_expr(&r.lhs, in_closure, out);
            scan_expr(&r.rhs, in_closure, out);
        }
        Expr::LogicalOp(l) => {
            scan_expr(&l.lhs, in_closure, out);
            scan_expr(&l.rhs, in_closure, out);
        }
        Expr::UnaryOp(u) => scan_expr(&u.expr, in_closure, out),
        Expr::Borrow(b) => scan_expr(&b.expr, in_closure, out),
        Expr::Dereference(d) => scan_expr(&d.expr, in_closure, out),
        Expr::StructInit(si) => {
            for (_, fe) in &si.fields {
                scan_expr(fe, in_closure, out);
            }
        }
        Expr::Range(r) => {
            scan_expr(&r.start, in_closure, out);
            scan_expr(&r.end, in_closure, out);
        }
        Expr::Grad(g) => {
            for a in &g.args {
                scan_expr(a, in_closure, out);
            }
        }
        Expr::Vjp(v) => {
            for a in &v.args {
                scan_expr(a, in_closure, out);
            }
            scan_expr(&v.cotangent, in_closure, out);
        }
        Expr::Jvp(j) => {
            for a in &j.args {
                scan_expr(a, in_closure, out);
            }
            scan_expr(&j.tangent, in_closure, out);
        }
        Expr::VecMacro(v) => {
            for el in &v.elements {
                scan_expr(el, in_closure, out);
            }
        }
        Expr::AsCast(c) => scan_expr(&c.expr, in_closure, out),
        Expr::Print(p) => {
            for a in &p.args {
                scan_expr(a, in_closure, out);
            }
        }
        Expr::Println(p) => {
            for a in &p.args {
                scan_expr(a, in_closure, out);
            }
        }
        Expr::InlineMlir(im) => {
            for (_, ie, _) in &im.inputs {
                scan_expr(ie, in_closure, out);
            }
            for cl in &im.clobbers {
                scan_expr(cl, in_closure, out);
            }
        }
        Expr::Identifier(_)
        | Expr::Number(_)
        | Expr::StringLiteral(_)
        | Expr::TransferPredicate(_)
        | Expr::MemorySpace(_)
        | Expr::Topology(_)
        | Expr::MacroCall(_)
        | Expr::SizeOf(_) => {}
    }
}

impl<'a> TypeChecker<'a> {
    /// Overwrite every recorded prover fact that mentions `name` with `0 == 0`.
    /// Called when `name` is rebound (a shadowing `let`, a loop induction variable):
    /// facts are keyed by bare symbol name, so a fact about the OUTER binding would
    /// otherwise keep "proving" things about the inner one -- reviewed and reproduced
    /// as a forged out-of-bounds proof. Overwritten in place rather than removed so
    /// the scope-exit truncation (`constraints.truncate(prev_len)`) keeps its meaning.
    pub(crate) fn neutralize_facts_mentioning(&mut self, name: &str) {
        for c in self.constraints.iter_mut() {
            if expr_mentions(c, name) {
                *c = Expr::RelationalOp(RelationalOpExpr {
                    lhs: Box::new(Expr::Number(NumberExpr {
                        value: "0".to_string().into(),
                        ty: None,
                        span: Span::default(),
                    })),
                    op: RelationalOp::Eq,
                    rhs: Box::new(Expr::Number(NumberExpr {
                        value: "0".to_string().into(),
                        ty: None,
                        span: Span::default(),
                    })),
                    span: Span::default(),
                });
            }
        }
    }
}

/// Does the expression mention identifier `name` anywhere? Conservative: an expression
/// shape the walk does not enumerate counts as mentioning it, so an unknown container
/// neutralizes rather than preserves.
pub(crate) fn expr_mentions(e: &Expr, name: &str) -> bool {
    match e {
        Expr::Identifier(id) => id.name.as_ref() == name,
        Expr::Number(_) | Expr::StringLiteral(_) | Expr::MemorySpace(_) | Expr::Topology(_) => {
            false
        }
        Expr::EnumVariant(ev) => ev.payload.iter().flatten().any(|a| expr_mentions(a, name)),
        Expr::FunctionCall(fc) => fc.args.iter().any(|a| expr_mentions(a, name)),
        Expr::BinaryOp(b) => expr_mentions(&b.lhs, name) || expr_mentions(&b.rhs, name),
        Expr::RelationalOp(r) => expr_mentions(&r.lhs, name) || expr_mentions(&r.rhs, name),
        Expr::LogicalOp(l) => expr_mentions(&l.lhs, name) || expr_mentions(&l.rhs, name),
        Expr::UnaryOp(u) => expr_mentions(&u.expr, name),
        Expr::Borrow(b) => expr_mentions(&b.expr, name),
        Expr::Dereference(d) => expr_mentions(&d.expr, name),
        Expr::MemberAccess(m) => expr_mentions(&m.base, name),
        Expr::IndexAccess(ix) => expr_mentions(&ix.base, name) || expr_mentions(&ix.index, name),
        Expr::Range(r) => expr_mentions(&r.start, name) || expr_mentions(&r.end, name),
        _ => true,
    }
}

/// Can the prover's QF_LIA lowering express this fact? Range facts with a bound the
/// prover cannot lower (a function call, a method call) must not be recorded: every
/// later `prove_expr` in the function would warn about the unlowerable constraint --
/// reviewed as new warnings on previously-clean programs.
pub(crate) fn prover_expressible(e: &Expr) -> bool {
    match e {
        Expr::Number(_) | Expr::Identifier(_) | Expr::EnumVariant(_) | Expr::Topology(_) => true,
        Expr::BinaryOp(b) => prover_expressible(&b.lhs) && prover_expressible(&b.rhs),
        Expr::RelationalOp(r) => prover_expressible(&r.lhs) && prover_expressible(&r.rhs),
        Expr::LogicalOp(l) => prover_expressible(&l.lhs) && prover_expressible(&l.rhs),
        Expr::UnaryOp(u) => prover_expressible(&u.expr),
        Expr::MemberAccess(_) | Expr::IndexAccess(_) => true,
        _ => false,
    }
}

/// The body's statements with top-level `unsafe { }` wrappers flattened away.
/// Wrapping in `unsafe` is the documented escape for unprovable bounds, and it must
/// not demote the statements inside it -- a trailing barrier is still trailing, and a
/// final return is still final, whether or not an unsafe block surrounds them.
fn top_level_view(body: &[Statement]) -> Vec<&Statement> {
    let mut out = Vec::new();
    for s in body {
        if let Statement::ExprStmt(es) = s {
            if let Expr::UnsafeBlock(u) = &es.expr {
                out.extend(top_level_view(&u.stmts));
                continue;
            }
        }
        out.push(s);
    }
    out
}

/// A statement's source span, for anchoring whole-body diagnostics.
fn stmt_span(s: &Statement) -> crate::syntax::Span {
    match s {
        Statement::LetDecl(x) => x.span,
        Statement::Return(x) => x.span,
        Statement::ExprStmt(x) => x.span,
        Statement::ForLoop(x) => x.span,
        Statement::Assign(x) => x.span,
        Statement::CompoundAssign(x) => x.span,
        Statement::Assert(x) => x.span,
        Statement::Loop(x) => x.span,
        Statement::Break(x) => x.span,
        Statement::Continue(x) => x.span,
        Statement::MacroCall(x) => x.span,
        Statement::Error(sp) => *sp,
    }
}

/// Does `body` (or anything nested in it) assign to `name`? Conservative in the safe
/// direction: an assignment through an index or member of `name` counts, and anything
/// the walk cannot classify counts. Used by the for-loop range facts -- a reassigned
/// induction variable must not keep its range facts, because the checker does not
/// version mutated symbols.
pub(crate) fn body_reassigns(body: &[Statement], name: &str) -> bool {
    fn base_is(e: &Expr, name: &str) -> bool {
        match e {
            Expr::Identifier(id) => id.name.as_ref() == name,
            Expr::IndexAccess(ix) => base_is(&ix.base, name),
            Expr::MemberAccess(m) => base_is(&m.base, name),
            Expr::Dereference(d) => base_is(&d.expr, name),
            _ => false,
        }
    }
    // Assignments are statements, so a statement-level walk sees them all; expressions
    // are entered only where they can carry statement blocks.
    fn walk(stmts: &[Statement], name: &str) -> bool {
        for s in stmts {
            match s {
                Statement::Assign(a) if base_is(&a.lhs, name) => return true,
                Statement::CompoundAssign(c) if base_is(&c.lhs, name) => return true,
                Statement::ForLoop(fl) => {
                    if walk(&fl.body, name) {
                        return true;
                    }
                }
                Statement::Loop(lp) => {
                    if walk(&lp.body, name) {
                        return true;
                    }
                }
                Statement::LetDecl(l) => {
                    if expr_blocks_reassign(&l.expr, name) {
                        return true;
                    }
                }
                Statement::ExprStmt(e) => {
                    if expr_blocks_reassign(&e.expr, name) {
                        return true;
                    }
                }
                Statement::Return(r) => {
                    if expr_blocks_reassign(&r.expr, name) {
                        return true;
                    }
                }
                Statement::Assign(a) => {
                    if expr_blocks_reassign(&a.rhs, name) {
                        return true;
                    }
                }
                Statement::CompoundAssign(c) => {
                    if expr_blocks_reassign(&c.rhs, name) {
                        return true;
                    }
                }
                Statement::Assert(a) => {
                    if expr_blocks_reassign(&a.expr, name) {
                        return true;
                    }
                }
                Statement::Break(_) | Statement::Continue(_) => {}
                Statement::MacroCall(_) => return true, // Cannot see inside: conservative.
                Statement::Error(_) => return true,
            }
        }
        false
    }
    fn expr_blocks_reassign(e: &Expr, name: &str) -> bool {
        match e {
            Expr::If(ifx) => {
                walk(&ifx.then_block, name)
                    || ifx.else_block.as_ref().is_some_and(|eb| walk(eb, name))
                    || expr_blocks_reassign(&ifx.cond, name)
            }
            Expr::Match(m) => {
                m.arms.iter().any(|arm| walk(&arm.body, name))
                    || expr_blocks_reassign(&m.expr, name)
            }
            Expr::UnsafeBlock(u) => {
                walk(&u.stmts, name)
                    || u.ret
                        .as_ref()
                        .is_some_and(|r| expr_blocks_reassign(r, name))
            }
            Expr::ComptimeBlock(c) => {
                walk(&c.stmts, name)
                    || c.ret
                        .as_ref()
                        .is_some_and(|r| expr_blocks_reassign(r, name))
            }
            Expr::SpawnOn(sp) => {
                walk(&sp.stmts, name)
                    || sp
                        .ret
                        .as_ref()
                        .is_some_and(|r| expr_blocks_reassign(r, name))
            }
            Expr::Closure(cl) => expr_blocks_reassign(&cl.body, name),
            // Leaf and operator shapes cannot contain a statement block on any path the
            // parser produces; if one grows a block later, the exhaustive scan_expr above
            // is the walker that must learn it first.
            _ => false,
        }
    }
    walk(body, name)
}

/// Bytes read and written per space, while a lowering body is being counted.
#[derive(Default, Clone, PartialEq, Eq)]
struct TrafficAcc {
    from_read: u64,
    from_written: u64,
    to_read: u64,
    to_written: u64,
}

/// An overflow is reported the way every other uncountable body is: as an absence with a
/// reason. Saturating instead published `u64::MAX` as an exact byte count -- a wrong number
/// where the entire discipline is that a published number can be trusted. The trip-count
/// multiply already refused honestly while the accumulation clamped, and the two sat one hop
/// apart in the same expression.
pub(crate) fn overflowed() -> String {
    "the byte count overflows a 64-bit counter".to_string()
}

impl TrafficAcc {
    fn add_read(&mut self, to_dst: bool, bytes: u64) -> Result<(), String> {
        let slot = if to_dst {
            &mut self.to_read
        } else {
            &mut self.from_read
        };
        *slot = slot.checked_add(bytes).ok_or_else(overflowed)?;
        Ok(())
    }

    fn add_written(&mut self, to_dst: bool, bytes: u64) -> Result<(), String> {
        let slot = if to_dst {
            &mut self.to_written
        } else {
            &mut self.from_written
        };
        *slot = slot.checked_add(bytes).ok_or_else(overflowed)?;
        Ok(())
    }

    /// The per-space maximum of two branch outcomes. A branch is counted as the worst
    /// arm, not the sum: one execution takes one arm, so the sum would overstate what
    /// the hardware does. The caller marks the result inexact.
    fn max_with(&self, other: &TrafficAcc) -> TrafficAcc {
        TrafficAcc {
            from_read: self.from_read.max(other.from_read),
            from_written: self.from_written.max(other.from_written),
            to_read: self.to_read.max(other.to_read),
            to_written: self.to_written.max(other.to_written),
        }
    }
}

impl<'a> TypeChecker<'a> {
    /// Count the bytes a user `impl transfer` body moves, per space (#353 A4).
    ///
    /// This is where "cost is derived, not declared" cashes out: the figures come from the
    /// body's own `raw::` calls multiplied by its static loop bounds, so a lowering that
    /// reads the source twice reports twice the traffic without anyone declaring anything.
    /// The counts carry no time and no bandwidth -- what they cost is the time model's
    /// separate, calibrated claim.
    ///
    /// `Err(reason)` rather than a guess whenever the body steps outside what can be
    /// counted exactly: a loop whose trip count is not static, an unbounded `loop`, a
    /// `break`/`continue` that makes trip counts a fiction. The reason travels into the
    /// record, because "uncountable" and "zero" must not look alike to a consumer.
    pub(crate) fn derive_lowering_traffic(
        &self,
        body: &[Statement],
        elem_bytes: u64,
        params: &[(crate::symbol::Symbol, Type)],
        from: &MemorySpace,
        to: &MemorySpace,
    ) -> Result<crate::hir::env::Traffic, String> {
        // `raw::extent(t)` is the natural loop bound in a lowering, and it must resolve
        // HERE, at the transfer site, where the lowering's parameters are not in scope --
        // the checker's own fold looks them up by name and finds nothing. The declared
        // parameter types carry the shape, so the extents come from there.
        let extent_of = |ty: &Type| -> Option<u64> {
            let (_, dims, _) = Self::as_tensor_operand(ty)?;
            let mut n: u64 = 1;
            for d in dims {
                let Expr::Number(lit) = d else { return None };
                n = n.checked_mul(lit.value.as_ref().parse::<u64>().ok()?)?;
            }
            Some(n)
        };
        let src_name = &params[0].0;
        let dst_name = &params[1].0;
        let exts = (
            extent_of(&params[0].1).ok_or_else(|| {
                "the lowering's source tile has no statically known extent".to_string()
            })?,
            extent_of(&params[1].1).ok_or_else(|| {
                "the lowering's destination tile has no statically known extent".to_string()
            })?,
        );
        let mut acc = TrafficAcc::default();
        let mut exact = true;
        self.traffic_stmts(
            body, 1, &mut acc, &mut exact, elem_bytes, src_name, dst_name, exts,
        )?;
        Ok(crate::hir::env::Traffic {
            per_space: vec![
                crate::hir::env::SpaceTraffic {
                    space: from.clone(),
                    read_bytes: acc.from_read,
                    written_bytes: acc.from_written,
                },
                crate::hir::env::SpaceTraffic {
                    space: to.clone(),
                    read_bytes: acc.to_read,
                    written_bytes: acc.to_written,
                },
            ],
            source: crate::hir::env::TrafficSource::LoweringBody,
            exact,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn traffic_stmts(
        &self,
        stmts: &[Statement],
        mult: u64,
        acc: &mut TrafficAcc,
        exact: &mut bool,
        elem: u64,
        src: &crate::symbol::Symbol,
        dst: &crate::symbol::Symbol,
        exts: (u64, u64),
    ) -> Result<(), String> {
        for s in stmts {
            match s {
                Statement::LetDecl(l) => {
                    self.traffic_expr(&l.expr, mult, acc, exact, elem, src, dst, exts)?
                }
                Statement::Return(r) => {
                    self.traffic_expr(&r.expr, mult, acc, exact, elem, src, dst, exts)?
                }
                Statement::ExprStmt(e) => {
                    self.traffic_expr(&e.expr, mult, acc, exact, elem, src, dst, exts)?
                }
                Statement::Assign(a) => {
                    self.traffic_expr(&a.rhs, mult, acc, exact, elem, src, dst, exts)?;
                    self.traffic_expr(&a.lhs, mult, acc, exact, elem, src, dst, exts)?;
                }
                Statement::CompoundAssign(c) => {
                    self.traffic_expr(&c.rhs, mult, acc, exact, elem, src, dst, exts)?;
                    self.traffic_expr(&c.lhs, mult, acc, exact, elem, src, dst, exts)?;
                }
                Statement::Assert(a) => {
                    self.traffic_expr(&a.expr, mult, acc, exact, elem, src, dst, exts)?
                }
                Statement::ForLoop(fl) => {
                    let trips = self
                        .static_trip_count(&fl.iterable, src, dst, exts)
                        .ok_or_else(|| {
                            "a loop bound is not statically known, so the number of times its \
                         body runs cannot be counted"
                                .to_string()
                        })?;
                    let next = mult.checked_mul(trips).ok_or_else(|| {
                        "the loop nest's trip count overflows a 64-bit counter".to_string()
                    })?;
                    self.traffic_stmts(&fl.body, next, acc, exact, elem, src, dst, exts)?;
                }
                Statement::Loop(_) => {
                    return Err("an unbounded `loop` runs an unknown number of times".to_string())
                }
                Statement::Break(_) | Statement::Continue(_) => {
                    return Err(
                        "`break`/`continue` makes the loop's trip count a fiction".to_string()
                    )
                }
                Statement::MacroCall(_) | Statement::Error(_) => {}
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn traffic_expr(
        &self,
        e: &Expr,
        mult: u64,
        acc: &mut TrafficAcc,
        exact: &mut bool,
        elem: u64,
        src: &crate::symbol::Symbol,
        dst: &crate::symbol::Symbol,
        exts: (u64, u64),
    ) -> Result<(), String> {
        match e {
            Expr::FunctionCall(fc) => {
                // Arguments first: `raw::store(dst, i, raw::load(src, i))` is a write and a
                // read, and the read is nested in the write's arguments.
                for a in &fc.args {
                    self.traffic_expr(a, mult, acc, exact, elem, src, dst, exts)?;
                }
                let tile = |i: usize| -> Option<&crate::symbol::Symbol> {
                    match fc.args.get(i) {
                        Some(Expr::Identifier(id)) => Some(&id.name),
                        _ => None,
                    }
                };
                let bytes = elem.checked_mul(mult).ok_or_else(overflowed)?;
                // Which side of the edge a tile name denotes: `false` = source space,
                // `true` = destination.
                let side = |t: Option<&crate::symbol::Symbol>| -> Option<bool> {
                    match t {
                        Some(t) if t == src => Some(false),
                        Some(t) if t == dst => Some(true),
                        _ => None,
                    }
                };
                match fc.name.as_ref() {
                    "raw::load" => match side(tile(0)) {
                        Some(to_dst) => acc.add_read(to_dst, bytes)?,
                        None => {
                            return Err("a `raw::load` names a tile that is not this \
                                         lowering's source or destination"
                                .to_string())
                        }
                    },
                    "raw::store" => match side(tile(0)) {
                        Some(to_dst) => acc.add_written(to_dst, bytes)?,
                        None => {
                            return Err("a `raw::store` names a tile that is not this \
                                         lowering's source or destination"
                                .to_string())
                        }
                    },
                    // One chunk out of the source and into the destination. The chunk is
                    // one element while the copy engine is unimplemented and `async_copy`
                    // lowers to a load and a store; when the real engine lands, its chunk
                    // size joins the machine file beside `copy_engine` and multiplies here.
                    "raw::async_copy" => {
                        match side(tile(1)) {
                            Some(to_dst) => acc.add_read(to_dst, bytes)?,
                            None => {
                                return Err("a `raw::async_copy` reads a tile that is not \
                                            this lowering's source or destination"
                                    .to_string())
                            }
                        }
                        match side(tile(0)) {
                            Some(to_dst) => acc.add_written(to_dst, bytes)?,
                            None => {
                                return Err("a `raw::async_copy` writes a tile that is not \
                                            this lowering's source or destination"
                                    .to_string())
                            }
                        }
                    }
                    // extent/lane/lanes/barrier/async_wait move nothing, and an ordinary
                    // call cannot appear here -- the primitives are the whole vocabulary a
                    // lowering body has for touching memory.
                    _ => {}
                }
                Ok(())
            }
            Expr::If(ifx) => {
                self.traffic_expr(&ifx.cond, mult, acc, exact, elem, src, dst, exts)?;
                // `if comptime` decides at compile time, so only one arm exists in the
                // emitted code and the traffic is exactly known. Weighing it as a runtime
                // branch reported the maximum of two arms -- twice the truth on a body
                // whose answer is not in doubt -- and flagged inexact a figure the
                // compiler had already resolved.
                if ifx.is_comptime {
                    if let Some(crate::hir::env::Value::Bool(taken)) =
                        self.eval_expr(&ifx.cond, &std::collections::HashMap::new())
                    {
                        let block = if taken {
                            Some(&ifx.then_block)
                        } else {
                            ifx.else_block.as_ref()
                        };
                        if let Some(b) = block {
                            self.traffic_stmts(b, mult, acc, exact, elem, src, dst, exts)?;
                        }
                        return Ok(());
                    }
                }
                let mut then_acc = acc.clone();
                self.traffic_stmts(
                    &ifx.then_block,
                    mult,
                    &mut then_acc,
                    exact,
                    elem,
                    src,
                    dst,
                    exts,
                )?;
                let mut else_acc = acc.clone();
                if let Some(eb) = &ifx.else_block {
                    self.traffic_stmts(eb, mult, &mut else_acc, exact, elem, src, dst, exts)?;
                }
                if then_acc != else_acc {
                    *exact = false;
                }
                *acc = then_acc.max_with(&else_acc);
                Ok(())
            }
            Expr::Match(m) => {
                self.traffic_expr(&m.expr, mult, acc, exact, elem, src, dst, exts)?;
                let entry = acc.clone();
                let mut outcomes: Vec<TrafficAcc> = Vec::new();
                for arm in &m.arms {
                    let mut arm_acc = entry.clone();
                    self.traffic_stmts(&arm.body, mult, &mut arm_acc, exact, elem, src, dst, exts)?;
                    outcomes.push(arm_acc);
                }
                // Inexact only when the arms DISAGREE, matching `if`. Flagging on "some arm
                // moved something" made a match whose arms move identical bytes report
                // inexact while the same program written as an `if` reported exact -- a
                // difference in the record with no difference in the code.
                if outcomes.windows(2).any(|w| w[0] != w[1]) {
                    *exact = false;
                }
                let mut worst = entry;
                for o in &outcomes {
                    worst = worst.max_with(o);
                }
                *acc = worst;
                Ok(())
            }
            // `unsafe` is the documented bounds escape, not control flow: transparent here.
            Expr::UnsafeBlock(u) => {
                self.traffic_stmts(&u.stmts, mult, acc, exact, elem, src, dst, exts)?;
                if let Some(r) = &u.ret {
                    self.traffic_expr(r, mult, acc, exact, elem, src, dst, exts)?;
                }
                Ok(())
            }
            Expr::ComptimeBlock(c) => {
                self.traffic_stmts(&c.stmts, mult, acc, exact, elem, src, dst, exts)?;
                if let Some(r) = &c.ret {
                    self.traffic_expr(r, mult, acc, exact, elem, src, dst, exts)?;
                }
                Ok(())
            }
            Expr::BinaryOp(b) => {
                self.traffic_expr(&b.lhs, mult, acc, exact, elem, src, dst, exts)?;
                self.traffic_expr(&b.rhs, mult, acc, exact, elem, src, dst, exts)
            }
            Expr::RelationalOp(r) => {
                self.traffic_expr(&r.lhs, mult, acc, exact, elem, src, dst, exts)?;
                self.traffic_expr(&r.rhs, mult, acc, exact, elem, src, dst, exts)
            }
            Expr::LogicalOp(l) => {
                self.traffic_expr(&l.lhs, mult, acc, exact, elem, src, dst, exts)?;
                self.traffic_expr(&l.rhs, mult, acc, exact, elem, src, dst, exts)
            }
            Expr::UnaryOp(u) => self.traffic_expr(&u.expr, mult, acc, exact, elem, src, dst, exts),
            Expr::Borrow(b) => self.traffic_expr(&b.expr, mult, acc, exact, elem, src, dst, exts),
            Expr::Dereference(d) => {
                self.traffic_expr(&d.expr, mult, acc, exact, elem, src, dst, exts)
            }
            Expr::AsCast(c) => self.traffic_expr(&c.expr, mult, acc, exact, elem, src, dst, exts),
            Expr::Range(r) => {
                self.traffic_expr(&r.start, mult, acc, exact, elem, src, dst, exts)?;
                self.traffic_expr(&r.end, mult, acc, exact, elem, src, dst, exts)
            }
            Expr::IndexAccess(ix) => {
                self.traffic_expr(&ix.base, mult, acc, exact, elem, src, dst, exts)?;
                self.traffic_expr(&ix.index, mult, acc, exact, elem, src, dst, exts)
            }
            Expr::MemberAccess(m) => {
                self.traffic_expr(&m.base, mult, acc, exact, elem, src, dst, exts)
            }
            Expr::Identifier(_) | Expr::Number(_) | Expr::StringLiteral(_) => Ok(()),
            // Anything else is refused rather than assumed empty -- but only when it could
            // hide a movement. A subtree with no `raw::` call in it moves nothing by
            // construction, because the primitives are the only way a body touches memory.
            other => {
                let mut scan = RawScan::default();
                scan_expr(other, false, &mut scan);
                if scan.calls.is_empty() {
                    Ok(())
                } else {
                    Err(
                        "a `raw::` call appears somewhere this counter does not know how \
                         to weigh"
                            .to_string(),
                    )
                }
            }
        }
    }

    /// Trip count of `for _ in a..b` when both bounds are static after the `raw::extent`
    /// fold; `None` otherwise. An empty or reversed range runs zero times.
    fn static_trip_count(
        &self,
        iterable: &Expr,
        src: &crate::symbol::Symbol,
        dst: &crate::symbol::Symbol,
        exts: (u64, u64),
    ) -> Option<u64> {
        let Expr::Range(r) = iterable else {
            return None;
        };
        let lit = |e: &Expr| -> Option<i64> {
            // `raw::extent(src)` / `raw::extent(dst)` first, from the declaration; then
            // the checker's ordinary fold for anything else in scope.
            if let Expr::FunctionCall(fc) = e {
                if fc.name.as_ref() == "raw::extent" && fc.args.len() == 1 {
                    if let Expr::Identifier(id) = &fc.args[0] {
                        if &id.name == src {
                            return Some(exts.0 as i64);
                        }
                        if &id.name == dst {
                            return Some(exts.1 as i64);
                        }
                    }
                }
            }
            match self.fold_raw_extent(e) {
                Expr::Number(n) => n.value.as_ref().parse::<i64>().ok(),
                _ => None,
            }
        };
        let (a, b) = (lit(&r.start)?, lit(&r.end)?);
        Some(if b > a { (b - a) as u64 } else { 0 })
    }
}
