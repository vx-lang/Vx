//===- affine.rs - Vx Compiler --------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Counted loops that the code generator writes as `affine.for`, so that the affine passes can
// fuse and tile them. docs/discussions/loop_optimizations_in_the_frontend_and_in_mlir.md has the
// rules.
//
//===----------------------------------------------------------------------===//

use super::*;
use crate::bytecode::AFFINE_FOR_BRANCHES;

/// Whether the body of a loop can be the body of an `affine.for`: nothing in it leaves the loop,
/// and it holds no region that the MLIR passes lower by splitting its block.
pub(super) fn affine_body(body: &[Statement]) -> bool {
    stays_in(body, false)
}

/// Whether nothing in `stmts` leaves the loop. In a nested loop, `in_loop` is set, and a
/// `break` or `continue` stays in that loop.
fn stays_in(stmts: &[Statement], in_loop: bool) -> bool {
    stmts.iter().all(|s| match s {
        Statement::Return(_) => false,
        Statement::Break(_) | Statement::Continue(_) => in_loop,
        Statement::ForLoop(f) => {
            exprs_stay_in(std::slice::from_ref(s), in_loop) && stays_in(&f.body, true)
        }
        Statement::Loop(l) => {
            exprs_stay_in(std::slice::from_ref(s), in_loop) && stays_in(&l.body, true)
        }
        // An unexpanded macro may hold anything.
        Statement::MacroCall(_) | Statement::Error(_) => false,
        _ => exprs_stay_in(std::slice::from_ref(s), in_loop),
    })
}

/// Whether the expressions of `stmts`, and the blocks in them, stay in the loop.
fn exprs_stay_in(stmts: &[Statement], in_loop: bool) -> bool {
    let mut ok = true;
    let mut check = |e: &Expr| {
        ok &= match e {
            Expr::If(i) => {
                stays_in(&i.then_block, in_loop)
                    && i.else_block.as_ref().is_none_or(|b| stays_in(b, in_loop))
            }
            Expr::Match(m) => m.arms.iter().all(|a| stays_in(&a.body, in_loop)),
            Expr::UnsafeBlock(b) => stays_in(&b.stmts, in_loop),
            Expr::ComptimeBlock(b) => stays_in(&b.stmts, in_loop),
            Expr::SpawnOn(_)
            | Expr::TaskBlock(_)
            | Expr::InlineMlir(_)
            | Expr::Closure(_)
            | Expr::MacroCall(_) => false,
            _ => true,
        }
    };
    for s in stmts {
        // A nested loop's body is checked by `stays_in`. Only its own expressions are here.
        match s {
            Statement::ForLoop(f) => {
                crate::syntax::visit_expr(&f.iterable, &mut check);
                f.invariants
                    .iter()
                    .for_each(|e| crate::syntax::visit_expr(e, &mut check));
            }
            Statement::Loop(l) => l
                .invariants
                .iter()
                .for_each(|e| crate::syntax::visit_expr(e, &mut check)),
            _ => crate::syntax::visit_exprs(std::slice::from_ref(s), &mut check),
        }
    }
    ok
}

impl Lowerer<'_> {
    /// Whether `v` can be a bound of an `affine.for` here: a constant, a value from before the
    /// outermost open affine loop, or the variable of an open affine loop.
    fn affine_bound(&self, v: &Val) -> bool {
        let r = v.reg.0 as usize;
        match self.affine_open.first() {
            None => true,
            Some(&outer) => {
                r < outer
                    || self.affine_open.contains(&r)
                    || self.code.get(r).is_some_and(|i| i.opcode == Opcode::Const)
            }
        }
    }

    /// Whether the loop `f` over `start..end` of type `elem` lowers as an `affine.for`.
    pub(super) fn affine_loop(
        &self,
        f: &crate::syntax::ForLoopStmt,
        start: &Val,
        end: &Val,
        elem: &ElementType,
    ) -> bool {
        // The GPU and host-worker lowerings read the loops of a region in their branch form.
        self.spawn_depth == 0
            && matches!(elem, ElementType::I32 | ElementType::I64)
            && self.affine_bound(start)
            && self.affine_bound(end)
            && affine_body(&f.body)
    }

    /// `for i in start..end { body }` as an `AffineFor`. The loop variable is the instruction's
    /// result.
    pub(super) fn lower_affine_for(
        &mut self,
        f: &crate::syntax::ForLoopStmt,
        start: Val,
        end: Val,
        elem: ElementType,
    ) -> Lowered<()> {
        let open = self.code.len();
        let i = self.emit_value(Opcode::AffineFor, start.reg, end.reg, elem, 0);
        self.affine_open.push(open);
        self.bind_local(f.iter.as_str().into(), i);
        let lowered = self.lower_block(&f.body);
        self.affine_open.pop();
        lowered?;
        // A block of the body, outside the affine loops in it, needs a region of its own.
        let mut depth = 0u32;
        let mut branches = self.block_terminated();
        for ins in &self.code[open + 1..] {
            match ins.opcode {
                Opcode::AffineFor => depth += 1,
                Opcode::AffineForEnd => depth -= 1,
                Opcode::BlockStart | Opcode::Br | Opcode::CondBr | Opcode::Ret if depth == 0 => {
                    branches = true
                }
                _ => {}
            }
        }
        if branches {
            self.code[open].imm |= AFFINE_FOR_BRANCHES;
        }
        self.emit_effect(Opcode::AffineForEnd, Register(0), Register(0), 0);
        Ok(())
    }
}
