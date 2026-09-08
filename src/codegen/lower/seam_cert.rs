//===- seam_cert.rs - Vx Compiler ---------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Seam certificate emission: transport a host-proven relation across a host->device
// `spawn` seam as an `llvm.intr.assume` at the kernel entry.
//
// The companion `crate::hir::seam` engine *verifies* that a boundary contract survives
// a transfer. This is the inverse, generative direction the boundary thesis calls for:
// a relation the host has *proven* (the condition of an `assert`, discharged by the
// host prover) is opaque to the device backend once it crosses the launch boundary as
// runtime kernel arguments. We re-materialize that relation inside the kernel body as
// `llvm.intr.assume(<relation>)`, so the device compiler's `-O3` can fold on it (e.g.
// prove a causal-mask guard constant-false and DCE the masked block) — a fold it
// provably cannot reach without the certificate.
//
// This is sound by construction: we only transport facts the host already proved (an
// `assert`), and only when every variable the relation names is one the kernel body
// itself references — so each operand of the re-lowered condition resolves to an SSA
// value the body can already see (and that dominates the assume). Gated behind
// `vxc --emit-seam-certs`; off by default, ordinary lowering is untouched.
//
//===----------------------------------------------------------------------===//

use super::*;
use crate::syntax::{Expr, Statement};
use melior::ir::operation::OperationBuilder;
use std::collections::HashSet;

/// Emit an `llvm.intr.assume` at the entry of a device kernel body for each host-proven
/// `assert` fact the kernel actually depends on. Returns the block to continue lowering
/// the body into (unchanged for the simple boolean facts we transport). Called at the
/// top of a device `spawn`'s region, before its statements are lowered, so the assume
/// dominates the guard it is meant to fold.
pub(crate) fn emit_seam_certificates<'c>(
    gen: &mut MeliorGenerator<'c>,
    stmts: &[Statement],
    ret: &Option<Box<Expr>>,
    mut body_block: melior::ir::BlockRef<'c, 'c>,
) -> Result<melior::ir::BlockRef<'c, 'c>, LowerError> {
    if gen.assert_facts.is_empty() {
        return Ok(body_block);
    }

    // Variables the kernel body references. A fact is only transportable if all of its
    // variables appear here: that guarantees each operand of the re-lowered relation
    // resolves to a captured, body-usable SSA value (see module note on soundness).
    let mut body_idents = HashSet::new();
    for s in stmts {
        collect_stmt_idents(s, &mut body_idents);
    }
    if let Some(r) = ret {
        collect_expr_idents(r, &mut body_idents);
    }

    // Clone out of `gen` so we can borrow `gen` mutably to lower each fact.
    let facts = gen.assert_facts.clone();
    for fact in &facts {
        let mut fact_idents = HashSet::new();
        collect_expr_idents(fact, &mut fact_idents);
        // A fact with no variables carries nothing across the boundary; a fact naming a
        // variable the kernel does not use is not this kernel's obligation to assume.
        if fact_idents.is_empty() || !fact_idents.iter().all(|id| body_idents.contains(id)) {
            continue;
        }

        // Re-lower the proven condition inside the kernel body to an `i1`, then assume it.
        let (cond, cond_ty, new_block) = gen.generate_expr(fact, body_block)?;
        body_block = new_block;
        if !cond_ty.to_string().contains("i1") {
            // Not a boolean relation we can hand to `llvm.intr.assume`; skip it rather
            // than emit an ill-typed op.
            continue;
        }
        // `llvm.intr.assume` carries optional LLVM operand bundles; the verifier requires
        // the (empty, here) `op_bundle_sizes` array even with no bundle operands.
        let empty_bundle_sizes = melior::ir::Attribute::parse(gen.context, "array<i32>")
            .ok_or_else(|| LowerError::from("failed to parse array<i32> for assume".to_string()))?;
        let assume = OperationBuilder::new("llvm.intr.assume", gen.loc())
            .add_operands(&[cond])
            .add_attributes(&[(
                melior::ir::Identifier::new(gen.context, "op_bundle_sizes"),
                empty_bundle_sizes,
            )])
            .build()?;
        body_block.append_operation(assume);
    }
    Ok(body_block)
}

/// Collect the names of every `Expr::Identifier` reachable from `e`. Best-effort over the
/// structurally-recursive variants that carry value expressions; variants that cannot name
/// a transported host variable (literals, type/topology nodes, nested spawns) contribute
/// nothing. Under-collecting is safe — it only suppresses a certificate, never emits a
/// wrong one.
pub(crate) fn collect_expr_idents(e: &Expr, out: &mut HashSet<String>) {
    match e {
        Expr::Identifier(i) => {
            out.insert(i.name.as_ref().to_string());
        }
        Expr::RelationalOp(b) => {
            collect_expr_idents(&b.lhs, out);
            collect_expr_idents(&b.rhs, out);
        }
        Expr::BinaryOp(b) => {
            collect_expr_idents(&b.lhs, out);
            collect_expr_idents(&b.rhs, out);
        }
        Expr::LogicalOp(b) => {
            collect_expr_idents(&b.lhs, out);
            collect_expr_idents(&b.rhs, out);
        }
        Expr::UnaryOp(u) => collect_expr_idents(&u.expr, out),
        Expr::Dereference(d) => collect_expr_idents(&d.expr, out),
        Expr::Borrow(b) => collect_expr_idents(&b.expr, out),
        Expr::AsCast(c) => collect_expr_idents(&c.expr, out),
        Expr::IndexAccess(a) => {
            collect_expr_idents(&a.base, out);
            collect_expr_idents(&a.index, out);
        }
        Expr::MemberAccess(m) => collect_expr_idents(&m.base, out),
        Expr::FunctionCall(c) => {
            for a in &c.args {
                collect_expr_idents(a, out);
            }
        }
        Expr::MethodCall(c) => {
            collect_expr_idents(&c.base, out);
            for a in &c.args {
                collect_expr_idents(a, out);
            }
        }
        Expr::If(i) => {
            collect_expr_idents(&i.cond, out);
            for s in &i.then_block {
                collect_stmt_idents(s, out);
            }
            if let Some(eb) = &i.else_block {
                for s in eb {
                    collect_stmt_idents(s, out);
                }
            }
        }
        _ => {}
    }
}

/// Collect identifiers referenced by a statement (recursing into its expressions and
/// nested blocks). Assignment targets are included: a variable written in the body is
/// still a variable the body depends on.
pub(crate) fn collect_stmt_idents(s: &Statement, out: &mut HashSet<String>) {
    match s {
        Statement::LetDecl(l) => collect_expr_idents(&l.expr, out),
        Statement::ExprStmt(e) => collect_expr_idents(&e.expr, out),
        Statement::Return(r) => {
            if let Some(e) = &r.expr {
                collect_expr_idents(e, out);
            }
        }
        Statement::Assert(a) => collect_expr_idents(&a.expr, out),
        Statement::Assign(a) => {
            collect_expr_idents(&a.lhs, out);
            collect_expr_idents(&a.rhs, out);
        }
        Statement::CompoundAssign(a) => {
            collect_expr_idents(&a.lhs, out);
            collect_expr_idents(&a.rhs, out);
        }
        Statement::ForLoop(f) => {
            collect_expr_idents(&f.iterable, out);
            for s in &f.body {
                collect_stmt_idents(s, out);
            }
        }
        Statement::Loop(l) => {
            for s in &l.body {
                collect_stmt_idents(s, out);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbol::Symbol;
    use crate::syntax::types::Span;
    use crate::syntax::{IdentifierExpr, RelationalOp, RelationalOpExpr};

    fn sp() -> Span {
        Span {
            line: 1,
            column: 1,
            length: 1,
        }
    }
    fn id(name: &str) -> Expr {
        Expr::Identifier(IdentifierExpr::new(Symbol::from(name), sp()))
    }
    fn gt(l: Expr, r: Expr) -> Expr {
        Expr::RelationalOp(RelationalOpExpr {
            lhs: Box::new(l),
            op: RelationalOp::Gt,
            rhs: Box::new(r),
            span: sp(),
        })
    }

    #[test]
    fn collects_both_operands_of_a_relation() {
        // The certificate condition `kblk_start > qblk_end` names exactly its two vars.
        let e = gt(id("kblk_start"), id("qblk_end"));
        let mut out = HashSet::new();
        collect_expr_idents(&e, &mut out);
        assert_eq!(out.len(), 2);
        assert!(out.contains("kblk_start") && out.contains("qblk_end"));
    }

    #[test]
    fn relevance_filter_matches_when_body_uses_the_fact_vars() {
        // A fact is transportable iff every var it names is referenced by the kernel body.
        let fact = gt(id("a"), id("b"));
        let mut fact_ids = HashSet::new();
        collect_expr_idents(&fact, &mut fact_ids);

        // Body that uses a and b: transportable.
        let uses_ab = gt(id("a"), id("b"));
        let mut body_ids = HashSet::new();
        collect_expr_idents(&uses_ab, &mut body_ids);
        assert!(fact_ids.iter().all(|v| body_ids.contains(v)));

        // Body that uses only an unrelated var c: not transportable.
        let uses_c = gt(id("c"), id("c"));
        let mut body_ids2 = HashSet::new();
        collect_expr_idents(&uses_c, &mut body_ids2);
        assert!(!fact_ids.iter().all(|v| body_ids2.contains(v)));
    }
}
