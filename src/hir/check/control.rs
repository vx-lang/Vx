//===- check submodule - Vx Compiler ----------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// One expression family's type checks, split out of the former 5000-line `hir/expr.rs` along the
// `check_expr_type_flag` dispatch seam. An additional
// `impl TypeChecker` block; moves only, zero logic change. Reaches the shared types and helpers via
// `use super::super::*`, exactly as `expr.rs` uses `use super::*`.
//
//===----------------------------------------------------------------------===//

use super::super::*;
use crate::hir::check_state::{
    ComptimeCheck, ComptimeCheckProblem, ComptimeCheckProblemKind, ComptimeEvalFlow,
};
use std::collections::HashMap;

/// What running a `comptime` block produced.
enum ComptimeFold {
    /// A value, which replaces the block. Boxed because it dwarfs the other two.
    Folded(Box<Expr>),
    /// It ran and left no value, so the block goes away entirely.
    NoValue,
    /// It could not be run. Reported already, unless this is a closure body -- which is not
    /// asked to fold where it is written.
    Refused,
}

/// The names that `pattern` binds, in order.
pub(crate) fn pattern_names(pattern: &Pattern, out: &mut Vec<crate::symbol::Symbol>) {
    match pattern {
        Pattern::Identifier(n) => out.push(n.clone()),
        Pattern::EnumVariant(_, _, Some(parts)) => {
            for part in parts {
                pattern_names(part, out);
            }
        }
        _ => {}
    }
}

impl<'a> TypeChecker<'a> {
    /// The constant environment as one map, the innermost scope winning.
    pub(crate) fn consteval_snapshot(&self) -> HashMap<crate::symbol::Symbol, Value> {
        let mut env = HashMap::new();
        for scope in &self.consteval.env {
            for (k, v) in scope {
                env.insert(k.clone(), v.clone());
            }
        }
        env
    }

    /// Run a `comptime` block and answer the constant it produced, if it has one.
    ///
    /// A block that cannot be run is reported here. It exists to run during compilation and
    /// leave nothing behind, so there is no such thing as one that half-ran: until now those
    /// were emitted as ordinary run-time code, which is how a loop or a call the evaluator
    /// skipped stayed invisible.
    /// `return_is_value`: the block is a closure's whole body or the value of a `return`, so a
    /// `return` in it gives the block's value.
    fn fold_comptime_block(
        &mut self,
        stmts: &[Statement],
        ret: Option<&Expr>,
        block_span: Span,
        before: &HashMap<crate::symbol::Symbol, Value>,
        return_is_value: bool,
    ) -> ComptimeFold {
        let interpretation = self.observe_comptime_block(stmts, ret, before);
        // At the block's last expression; at the `comptime` keyword when there is none, when it
        // has no place, or when it is a number, which cannot be what went wrong.
        let span = ret
            .filter(|r| !matches!(r, Expr::Number(_)))
            .map(|r| r.span())
            .filter(|span| *span != Span::default())
            .unwrap_or(block_span);

        // A check that fails every time the block runs stops the build. Each one is reported,
        // so a block of asserts reports every assert that fails.
        let failed: Vec<_> = interpretation
            .check_problems
            .iter()
            .filter(|problem| problem.kind == ComptimeCheckProblemKind::Fails)
            .collect();
        if !failed.is_empty() {
            for problem in failed {
                self.report_comptime_check_failure(problem);
            }
            return ComptimeFold::Refused;
        }
        // Anything it writes that outlives it would have to survive, and the block does not.
        if let Some(name) = interpretation.escaping_write {
            self.report_comptime_block_failure(
                &format!(
                    "it writes to '{}', which is declared outside it -- the block disappears, \
                     so the write would have to disappear with it",
                    name
                ),
                &span,
            );
            return ComptimeFold::Refused;
        }
        if interpretation.call_depth_exceeded {
            self.report_comptime_depth_exceeded(&span);
            return ComptimeFold::Refused;
        }
        if let Some(loop_span) = interpretation.loop_limit_reached_at {
            self.report_comptime_loop_limit(&loop_span);
            return ComptimeFold::Refused;
        }
        if interpretation.outcome.requires_refusal || !interpretation.outcome.support.is_supported()
        {
            self.report_comptime_block_failure(
                "it holds a statement the evaluator cannot run",
                &span,
            );
            return ComptimeFold::Refused;
        }
        // Any other check would disappear with the block, and never run.
        if !interpretation.check_problems.is_empty() {
            for problem in &interpretation.check_problems {
                self.report_comptime_check_failure(problem);
            }
            return ComptimeFold::Refused;
        }
        // A `return` inside the block, where the block is a closure or function body, is
        // that body's value -- `|| comptime { ..; return x; }` is how the closure fixtures
        // are written. Answer with it, the same as a trailing expression.
        if interpretation.outcome.flow == ComptimeEvalFlow::Return {
            if !return_is_value {
                self.report_comptime_block_failure(
                    "a `return` in it leaves the function, which a block that runs while \
                     compiling cannot do",
                    &span,
                );
                return ComptimeFold::Refused;
            }
            return self.fold_value(interpretation.outcome.value.concrete, &span);
        }
        if interpretation.outcome.flow != ComptimeEvalFlow::Normal {
            self.report_comptime_block_failure(
                "its control flow does not complete normally",
                &span,
            );
            return ComptimeFold::Refused;
        }
        if ret.is_none() {
            return ComptimeFold::NoValue;
        }
        self.fold_value(interpretation.outcome.value.concrete, &span)
    }

    fn report_comptime_depth_exceeded(&mut self, span: &Span) {
        if self.speculating
            || self.consteval.closure_body_depth > 0
            || self
                .errors
                .has_error(crate::diagnostic::DiagnosticCode::E8004)
        {
            return;
        }
        self.errors.error_with_code(
            crate::diagnostic::DiagnosticCode::E8004,
            format!(
                "compile-time evaluation went more than {} calls deep and was stopped. A \
                 recursive function whose base case is never reached is the usual cause.",
                crate::hir::check_state::MAX_CALL_DEPTH
            ),
            Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
        );
    }

    fn fold_value(&mut self, value: Option<Value>, span: &Span) -> ComptimeFold {
        let Some(value) = value else {
            self.report_comptime_block_failure("its value cannot be worked out", span);
            return ComptimeFold::Refused;
        };
        match Self::value_to_expr(&value, span) {
            Some(expr) => ComptimeFold::Folded(Box::new(expr)),
            None => {
                self.report_comptime_block_failure(
                    "its value is not one that can be written as a constant",
                    span,
                );
                ComptimeFold::Refused
            }
        }
    }

    /// A false assert on a path that always runs is E8002. Every other check that a block
    /// cannot show to pass is E3033.
    pub(crate) fn report_comptime_check_failure(&mut self, problem: &ComptimeCheckProblem) {
        use ComptimeCheckProblemKind::*;
        let why = match (&problem.check, problem.kind) {
            (ComptimeCheck::Assert(message), Fails) => {
                if self.speculating || self.consteval.closure_body_depth > 0 {
                    return;
                }
                let message = message.as_deref().unwrap_or("Comptime assertion failed");
                self.errors.error_with_code(
                    crate::diagnostic::DiagnosticCode::E8002,
                    format!("Comptime assert failed: {}", message),
                    Some(crate::diagnostic::SourceSpan::from_ast_span(&problem.span)),
                );
                return;
            }
            (ComptimeCheck::Assert(_), MayFail) => {
                "this assert is false on a path that depends on a value known only when the \
                 program runs"
            }
            (ComptimeCheck::Assert(_), Undecided) => {
                "this assert depends on a value known only when the program runs"
            }
            (ComptimeCheck::Assert(_), OutOfRange) => {
                "this assert has no value, because an operation it depends on gives a result \
                 outside its type, such as an overflow"
            }
            (ComptimeCheck::Division, Fails) => {
                "this division stops the program: it divides by zero, or it divides the \
                 minimum value by -1"
            }
            (ComptimeCheck::Division, MayFail) => {
                "this division stops the program on a path that depends on a value known only \
                 when the program runs"
            }
            (ComptimeCheck::Division, OutOfRange) => {
                "this division has no value, because an operation it depends on gives a result \
                 outside its type, such as an overflow"
            }
            (ComptimeCheck::Division, Undecided) => {
                "this division depends on a value known only when the program runs, and it may \
                 stop the program"
            }
        };
        self.report_comptime_block_failure(why, &problem.span);
    }

    /// E8005, once. The checker's own evaluator may have reported it at the same loop.
    pub(crate) fn report_comptime_loop_limit(&mut self, span: &Span) {
        if self.speculating
            || self.consteval.closure_body_depth > 0
            || self
                .errors
                .has_error(crate::diagnostic::DiagnosticCode::E8005)
        {
            return;
        }
        self.errors.error_with_code(
            crate::diagnostic::DiagnosticCode::E8005,
            format!(
                "compile-time evaluation ran more than {} loop iterations and was stopped. A \
                 loop whose end condition is never reached is the usual cause.",
                crate::hir::check_state::MAX_LOOP_STEPS
            ),
            Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
        );
    }

    fn report_comptime_block_failure(&mut self, why: &str, span: &Span) {
        // A closure body is not asked to fold where it is written -- its parameters have no
        // values yet. The call is what has to fold, and that is reported at the call.
        if self.speculating || self.consteval.closure_body_depth > 0 {
            return;
        }
        self.errors.error_with_code(
            crate::diagnostic::DiagnosticCode::E3033,
            format!("this `comptime` block cannot be evaluated: {}", why),
            Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
        );
    }

    /// Whether a statement binds a lambda whose body is a `comptime` block.
    ///
    /// Every call to one has folded by now, so nothing is left to call and the binding is
    /// dropped. Keeping it would emit the closure and its generated body, which is exactly
    /// the run-time code a `comptime` block must not leave behind.
    pub(crate) fn binds_a_comptime_lambda(stmt: &Statement) -> bool {
        let Statement::LetDecl(decl) = stmt else {
            return false;
        };
        let Expr::Closure(closure) = &decl.expr else {
            return false;
        };
        matches!(&*closure.body, Expr::ComptimeBlock(_))
    }

    /// Write a computed value back as a constant expression.
    ///
    /// Only immutable values: numbers, booleans and arrays of them. Anything else has no
    /// spelling that survives to run time on its own.
    /// Give each untyped number in a folded value the element type of `ty`. A whole number
    /// such as 0.0 is written `0`, which is an `i32` without a type, and in an array that
    /// first element made the whole array an `i32` array.
    pub(crate) fn give_numbers_the_type(value: &mut Expr, ty: &Type) {
        let elem = match ty {
            Type::Scalar(elem) | Type::Tensor(elem, _, _) => elem,
            _ => return,
        };
        match value {
            Expr::Number(n) => {
                n.ty.get_or_insert_with(|| elem.clone());
            }
            Expr::Array(array) => {
                for element in &mut array.elements {
                    Self::give_numbers_the_type(element, ty);
                }
            }
            _ => {}
        }
    }

    pub(crate) fn value_to_expr(value: &Value, span: &Span) -> Option<Expr> {
        match value {
            Value::Int(i) => Some(Expr::Number(NumberExpr {
                written: None,
                value: i.to_string().into(),
                ty: None,
                span: *span,
            })),
            Value::Number(n) => Some(Expr::Number(NumberExpr {
                written: None,
                value: n.to_string().into(),
                ty: None,
                span: *span,
            })),
            Value::CStr(text) => Some(Expr::StringLiteral(StringLiteralExpr {
                value: text.as_str().into(),
                span: *span,
            })),
            Value::Bool(b) => Some(Expr::Identifier(IdentifierExpr {
                name: if *b { "true".into() } else { "false".into() },
                span: *span,
            })),
            Value::Array(items) => {
                let mut elements = Vec::with_capacity(items.len());
                for item in items {
                    elements.push(Self::value_to_expr(item, span)?);
                }
                Some(Expr::Array(ArrayExpr {
                    elements,
                    span: *span,
                }))
            }
            _ => None,
        }
    }

    pub(crate) fn check_comptimeblock_expr(&mut self, expr: &mut Expr, consume: bool) -> Type {
        let unused = std::mem::replace(&mut self.value_unused, false);
        match expr {
            Expr::ComptimeBlock(ComptimeBlockExpr {
                stmts,
                ret,
                span: block_span,
            }) => {
                let block_span = *block_span;
                let return_is_value = std::mem::take(&mut self.consteval.comptime_return_is_value);
                // What is known before the block runs. Its own `let`s live in the scope
                // pushed below and are gone by the time the fold needs them, so the block
                // is run again from here rather than read out of the checker's environment.
                // One `comptime` inside another asks for nothing the outer one does not
                // already do, and it is what put a closure's body out of the evaluator's
                // reach: the inner block's value needed a call to a closure declared in the
                // outer block.
                if self.consteval.comptime_depth > 0 && !self.speculating {
                    self.errors.error_with_code(
                        crate::diagnostic::DiagnosticCode::E3034,
                        "a `comptime` block inside another one: the outer block already runs \
                         while compiling, so remove the inner `comptime`"
                            .to_string(),
                        Some(crate::diagnostic::SourceSpan::from_ast_span(&block_span)),
                    );
                }
                let before = self.consteval_snapshot();
                self.push_scope();
                // Each block has its own loop budget, as in the `comptime` interpreter.
                let outer_steps = self.consteval.loop_steps.replace(0);
                self.consteval.comptime_depth += 1;
                self.value_unused = unused;
                let mut ret_ty = self.check_expr_block(stmts, ret.as_deref(), consume);
                if let Some(r) = ret {
                    ret_ty = self.check_expr_type(r);
                }
                self.consteval.comptime_depth -= 1;
                self.consteval.loop_steps.set(outer_steps);
                self.pop_scope();
                let folded = self.fold_comptime_block(
                    stmts,
                    ret.as_deref(),
                    block_span,
                    &before,
                    return_is_value,
                );
                // An assertion the placement fold answered `true` is discharged here: nothing
                // at run time holds a placement, so nothing is left to check. (The fold
                // reported a false one.) Asserts on anything else stay, as the program wrote
                // them.
                stmts.retain(|s| {
                    !matches!(s, Statement::Assert(a)
                        if matches!(&*a.expr, Expr::Identifier(id) if id.name.as_ref() == "true"))
                });
                stmts.retain(|s| !Self::binds_a_comptime_lambda(s));
                // The block ran while compiling and must leave nothing behind, so what it
                // worked out replaces it. A block with no value becomes an empty one, which
                // the statement walk then drops.
                match folded {
                    // The value it worked out replaces it, as a literal of the block's type:
                    // an untyped literal is an `i32` to the legacy code generator, which cut
                    // an `i64` like 20! down to its low 32 bits.
                    ComptimeFold::Folded(mut value) => {
                        Self::give_numbers_the_type(&mut value, &ret_ty);
                        *expr = *value
                    }
                    // It ran and produced nothing, so nothing is left to emit.
                    ComptimeFold::NoValue => {
                        *expr = Expr::ComptimeBlock(ComptimeBlockExpr {
                            stmts: Vec::new(),
                            ret: None,
                            span: block_span,
                        })
                    }
                    // Left as written: either the error above stops the build, or this is a
                    // closure body, which folds at its call rather than here.
                    ComptimeFold::Refused => {}
                }
                ret_ty
            }

            _ => panic!("Expected IndexAccess, got {:?}", expr),
        }
    }

    /// Whether a block leaves the function, so that nothing after the `if` it belongs to is
    /// reached along this path.
    ///
    /// Only `return` counts. `break` and `continue` also stop the code right after the `if`
    /// from running, but they go somewhere else in the same function -- the statement after
    /// the loop, or its next turn -- and a value moved on the way there is just as moved when
    /// it arrives. Treating them like `return` would drop a mark that is still true.
    ///
    /// A block that ends some other way is answered `false`, including one whose last
    /// statement is an `if` both of whose arms return. That is the conservative direction:
    /// the mark is kept and the program refused, rather than a moved value let through.
    /// The moves at the end of a branch that was just checked, when it reaches the code after
    /// it. A branch that ends in `break` or `continue` instead hands its moves to the loop it
    /// leaves or restarts; one that ends in `return` hands them to nobody.
    pub(crate) fn branch_end(
        &mut self,
        stmts: &[Statement],
    ) -> Option<Vec<std::collections::HashSet<String>>> {
        let exit = stmts.iter().find_map(|s| match s {
            // The leave of a failed join goes out of a `tasks` block, not out of a loop.
            Statement::Return(_) => Some(0),
            Statement::Break(b) if b.tasks_value.is_some() => Some(0),
            Statement::Break(_) => Some(1),
            Statement::Continue(_) => Some(2),
            _ => None,
        });
        let marks = self.moved_snapshot();
        let slot = match (exit, self.borrow.loop_exits.last_mut()) {
            (None, _) => return Some(marks),
            (Some(1), Some(l)) => &mut l.at_break,
            (Some(2), Some(l)) => &mut l.at_continue,
            _ => return None,
        };
        *slot = Some(match slot.take() {
            Some(j) => Self::union_moved(j, marks),
            None => marks,
        });
        None
    }

    /// Whether control never reaches the end of `stmts`: it returns, breaks, continues, panics,
    /// aborts or calls a function declared `-> !` on every path.
    pub(crate) fn branch_ends_early(&self, stmts: &[Statement]) -> bool {
        let functions = &self.env.functions;
        let never = |name: &str| {
            functions
                .get(name)
                .is_some_and(|f| crate::syntax::is_never_ty(&f.0))
        };
        stmts.iter().any(|s| {
            matches!(
                s,
                Statement::Return(_) | Statement::Break(_) | Statement::Continue(_)
            ) || crate::syntax::expr::statement_always_exits(s, &never)
        })
    }

    /// The values known after paths join: a value stays known only when every path leaves it
    /// the same. Each path is the checker's scopes of known values at the join.
    fn join_known_values(
        paths: Vec<Vec<HashMap<crate::symbol::Symbol, Value>>>,
    ) -> Vec<HashMap<crate::symbol::Symbol, Value>> {
        let mut paths = paths.into_iter();
        let mut joined = paths.next().unwrap_or_default();
        for path in paths {
            assert_eq!(
                joined.len(),
                path.len(),
                "paths join at the same scope depth"
            );
            for (scope, other) in joined.iter_mut().zip(&path) {
                scope.retain(|name, value| other.get(name) == Some(value));
            }
        }
        joined
    }

    /// The value of an `if` condition when the checker knows it. A condition that calls a
    /// function is not evaluated: the call could run a whole function body while checking.
    fn known_condition(&self, cond: &Expr) -> Option<bool> {
        fn calls_nothing(expr: &Expr) -> bool {
            match expr {
                Expr::Identifier(_) | Expr::Number(_) => true,
                Expr::BinaryOp(op) => calls_nothing(&op.lhs) && calls_nothing(&op.rhs),
                Expr::RelationalOp(op) => calls_nothing(&op.lhs) && calls_nothing(&op.rhs),
                Expr::LogicalOp(op) => calls_nothing(&op.lhs) && calls_nothing(&op.rhs),
                Expr::UnaryOp(op) => calls_nothing(&op.expr),
                _ => false,
            }
        }
        if !calls_nothing(cond) {
            return None;
        }
        match self.eval_expr(cond, &self.consteval_snapshot()) {
            Some(Value::Bool(b)) => Some(b),
            _ => None,
        }
    }

    /// Check the condition of an `if` or the value a `match` tests, with `branch_uses` still in
    /// use: the branches run after it, so a borrow they use must not end while it is checked.
    fn check_head_of_branches(
        &mut self,
        branch_uses: std::collections::HashSet<String>,
        check: impl FnOnce(&mut Self) -> Type,
    ) -> Type {
        self.borrow.enter_block(crate::hir::borrow_cx::BlockUses {
            last_use: branch_uses.into_iter().map(|n| (n.into(), 1)).collect(),
            ..Default::default()
        });
        let ty = check(self);
        self.borrow.exit_block();
        ty
    }

    pub(crate) fn check_if_expr(&mut self, expr: &mut Expr, consume: bool) -> Type {
        let unused = std::mem::replace(&mut self.value_unused, false);
        let if_expr = match expr {
            Expr::If(e) => e,
            _ => panic!("Expected IndexAccess, got {:?}", expr),
        };

        // A name that a branch declares with its own `let` is a different variable there.
        let mut branch_uses = std::collections::HashSet::new();
        Self::extract_uses_block(&if_expr.then_block, None, &mut branch_uses);
        if let Some(else_block) = &if_expr.else_block {
            Self::extract_uses_block(else_block, None, &mut branch_uses);
        }
        let cond_ty =
            self.check_head_of_branches(branch_uses, |c| c.check_expr_type(&mut if_expr.cond));
        if cond_ty != Type::Scalar(ElementType::Bool) {
            self.errors
                .push("Condition in if expression must be of type bool (i1)".to_string());
        }

        let mut taken = None;
        if if_expr.is_comptime {
            let tmp_env = self.consteval_snapshot();
            if let Some(Value::Bool(b)) = self.eval_expr(&if_expr.cond, &tmp_env) {
                taken = Some(b);
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

        // Each branch starts from the moves made before the `if`. Afterwards a variable is
        // moved if a branch that reaches the code after the `if` moved it; a branch that
        // returns or breaks does not count.
        let marks_before = self.moved_snapshot();
        // The same for borrows: the `else` branch does not see what the `then` branch borrowed,
        // and still sees what the `then` branch ended.
        let borrows_before = self.borrow.snapshot();
        // The prover's facts follow the paths: the `then` branch knows the condition and the
        // `else` branch its negation, and what a branch learns ends with it.
        let facts_before = self.consteval.constraints.len();
        let prefix_before = self.consteval.constraints.clone();
        let pointers_before = self.pointer_facts.clone();
        let cond_fact = (!if_expr.is_comptime && crate::hir::prover::is_modelled(&if_expr.cond))
            .then(|| (*if_expr.cond).clone());
        if let Some(c) = &cond_fact {
            self.consteval.constraints.push(c.clone());
        }
        // Each branch starts from the values known before the `if`, and the paths join after it.
        let taken = taken.or_else(|| self.known_condition(&if_expr.cond));
        let values_before = self.consteval.env.clone();
        let (then_name, else_name) = Self::pointer_branch_names(if_expr).unzip();
        let named = then_name.is_some();
        self.pointer_branches.extend(then_name);
        self.push_releasing_scope();
        let mut then_ty = Type::Struct("void".into(), None);
        if !self.speculating && !if_expr.then_block.is_empty() {
            self.value_unused = unused;
            then_ty = self.check_expr_block(&mut if_expr.then_block, None, consume);
        }
        self.pop_scope();
        if named {
            self.pointer_branches.pop();
        }
        let then_facts = self.consteval.constraints.split_off(facts_before);
        // An assignment in the `then` branch renames the facts from before the `if` in place, so
        // the `else` branch starts from a copy of them.
        let then_prefix = std::mem::replace(&mut self.consteval.constraints, prefix_before.clone());
        let pointers_then = std::mem::replace(&mut self.pointer_facts, pointers_before);
        if let Some(c) = &cond_fact {
            self.consteval
                .constraints
                .push(crate::hir::prover::negate(c));
        }
        let marks_after_then = self.branch_end(&if_expr.then_block);
        let values_after_then = std::mem::replace(&mut self.consteval.env, values_before);
        self.restore_moved(marks_before.clone());
        let then_borrows = self.borrow.next_branch(&borrows_before);
        let mut marks_after_else = Some(marks_before.clone());

        let mut else_ty = Type::Struct("void".into(), None);
        if let Some(else_b) = if_expr.else_block.as_mut() {
            if !else_b.is_empty() {
                self.pointer_branches.extend(else_name);
                self.push_releasing_scope();
                if !self.speculating {
                    self.value_unused = unused;
                    else_ty = self.check_expr_block(else_b, None, consume);
                }
                self.pop_scope();
                if named {
                    self.pointer_branches.pop();
                }
                marks_after_else = self.branch_end(else_b);

                // A branch that never finishes, `todo()`, has no value: the other branch decides.
                // So does a branch that ends in `continue` when the value is used. A `while` is
                // an unused `if` with an `else { break }`, and its body must still give no value.
                if crate::syntax::is_never_ty(&then_ty)
                    || (!unused && self.branch_ends_early(&if_expr.then_block))
                {
                    then_ty = else_ty.clone();
                } else if crate::syntax::is_never_ty(&else_ty)
                    || (!unused && self.branch_ends_early(else_b))
                {
                    else_ty = then_ty.clone();
                }
                if !if_expr.is_comptime && then_ty != else_ty {
                    self.errors.error_with_code(
                        crate::diagnostic::DiagnosticCode::E3007,
                        format!(
                            "If expression branches have incompatible types: {} and {}",
                            then_ty, else_ty
                        ),
                        Some(crate::diagnostic::SourceSpan::from_ast_span(&if_expr.span)),
                    );
                }
            }
        } else if !if_expr.is_comptime {
            // After the error the `if` keeps its `then` branch's type, so that one mistake
            // gives one error.
            if !unused {
                self.errors.error_with_code(
                    crate::diagnostic::DiagnosticCode::E3048,
                    "this `if` has no `else`, so it has no value when its condition is false; \
                     add an `else`, or use the `if` as a statement"
                        .to_string(),
                    Some(crate::diagnostic::SourceSpan::from_ast_span(&if_expr.span)),
                );
            } else {
                then_ty = Type::Struct("void".into(), None);
            }
        }
        let else_facts = self.consteval.constraints.split_off(facts_before);
        let pointers_else = std::mem::take(&mut self.pointer_facts);
        // When one branch ends early, the code after the `if` is reached only through the other,
        // so what that branch knew at its end still holds. When both end early, it is not reached.
        let then_ends = self.branch_ends_early(&if_expr.then_block);
        let else_ends = if_expr
            .else_block
            .as_ref()
            .is_some_and(|b| self.branch_ends_early(b));
        let values_after_else = std::mem::take(&mut self.consteval.env);
        self.consteval.env = match (taken, then_ends, else_ends) {
            (Some(true), _, _) | (None, false, true) => values_after_then,
            (Some(false), _, _) | (None, true, false) => values_after_else,
            (None, _, _) => Self::join_known_values(vec![values_after_then, values_after_else]),
        };
        self.pointer_facts = match (then_ends, else_ends) {
            (true, false) => pointers_else,
            (false, true) => pointers_then,
            _ => pointers_then.join(&pointers_else),
        };
        let else_prefix = std::mem::take(&mut self.consteval.constraints);
        let ends = [
            BranchEnd {
                before: then_prefix,
                learned: then_facts,
            },
            BranchEnd {
                before: else_prefix,
                learned: else_facts,
            },
        ];
        // A branch that ends early does not reach the code after the `if`.
        let reaching: Vec<BranchEnd> = ends
            .into_iter()
            .zip([then_ends, else_ends])
            .filter(|(_, ends_early)| !ends_early)
            .map(|(end, _)| end)
            .collect();
        self.consteval.constraints = facts_after_branches(prefix_before, reaching);
        let joined = match (marks_after_then, marks_after_else) {
            (Some(a), Some(b)) => Self::union_moved(a, b),
            (Some(a), None) | (None, Some(a)) => a,
            (None, None) => marks_before,
        };
        self.restore_moved(joined);
        self.borrow.add_back(then_borrows);

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

    /// Notes that an operation needing `unsafe` was found (W1010 reads this), and returns true
    /// when nothing around the operation allows it, so the caller should report an error.
    pub(crate) fn unsafe_is_missing(&mut self) -> bool {
        self.unsafe_op_seen = true;
        !self.in_unsafe_block && !self.speculating
    }

    pub(crate) fn check_unsafeblock_expr(&mut self, expr: &mut Expr, consume: bool) -> Type {
        let unused = std::mem::replace(&mut self.value_unused, false);
        match expr {
            Expr::UnsafeBlock(UnsafeBlockExpr {
                stmts,
                ret: ret_expr,
                span,
                generated,
                closure_body,
                ..
            }) => {
                let block_span = *span;
                let closure_body = *closure_body;
                // The block body of a closure is a plain block: it allows nothing more than the
                // code around it.
                let written = !*generated && !closure_body;
                let prev_unsafe = self.in_unsafe_block;
                self.in_unsafe_block = prev_unsafe || !closure_body;
                // A block the compiler made is transparent here: what it holds counts for the
                // written block around it.
                let prev_written = self.in_written_unsafe_block;
                let prev_op_seen = self.unsafe_op_seen;
                if written {
                    self.in_written_unsafe_block = true;
                    self.unsafe_op_seen = false;
                }
                let errors_before = self.errors.error_count();
                // Vx has no plain `{ .. }` block, so one that declares a variable may be there to
                // end the variable's life early. Removing it would change the program.
                let declares_a_variable = stmts
                    .iter()
                    .any(|s| matches!(s, Statement::LetDecl(l) if !l.name.starts_with("$value")));
                // A value nothing takes is the block's last statement, which the block drops
                // as it drops `make();`.
                if unused {
                    if let Some(value) = ret_expr.take() {
                        let span = value.span();
                        stmts.push(Statement::ExprStmt(crate::syntax::stmt::ExprStmtStmt {
                            expr: *value,
                            has_semi: false,
                            span,
                        }));
                    }
                }
                // `unsafe { let v = ..; v }`, as `vec![..]` writes: the block's drops are placed
                // when its statements are checked, so the value is moved out of `v` by one of
                // them, into a `$value` that owns nothing.
                if let Some(Expr::Identifier(id)) = ret_expr.as_deref() {
                    if !id.name.starts_with("$value") && !stmts.is_empty() {
                        let value = Statement::LetDecl(crate::syntax::stmt::LetDeclStmt {
                            name: "$value".into(),
                            is_mut: false,
                            ty_ann: None,
                            expr: Expr::Identifier(id.clone()),
                            span: id.span,
                        });
                        stmts.push(value);
                        *ret_expr = Some(Box::new(Expr::Identifier(IdentifierExpr {
                            name: "$value".into(),
                            span: id.span,
                        })));
                    }
                }
                self.push_scope();
                self.value_unused = unused;
                // A `return` of the closure, or its written return type, gives the last value of
                // its body its type, as the return type of a function does.
                self.closure_body_next = closure_body;
                let ret_ty = match ret_expr {
                    Some(r) => self.check_expr_block_with_value(stmts, r, unused, consume),
                    None => self.check_expr_block(stmts, None, consume),
                };
                self.closure_body_next = false;
                self.pop_scope();
                if written {
                    // An error inside may hide the operation that needed `unsafe`, so only a
                    // block with no errors in it warns.
                    if !self.speculating
                        && !declares_a_variable
                        && self.errors.error_count() == errors_before
                    {
                        let message = if !self.unsafe_op_seen {
                            Some("this `unsafe` block holds nothing that needs `unsafe`, so it can go")
                        } else if prev_written {
                            Some(
                                "this `unsafe` block is inside another one, which already allows \
                                 what it holds",
                            )
                        } else {
                            None
                        };
                        if let Some(message) = message {
                            self.errors.warn(
                                crate::diagnostic::DiagnosticCode::W1010,
                                message,
                                Some(crate::diagnostic::SourceSpan::from_ast_span(&block_span)),
                            );
                        }
                    }
                    // What this block holds also needs the block around it, if there is one.
                    self.unsafe_op_seen |= prev_op_seen;
                    self.in_written_unsafe_block = prev_written;
                }
                self.in_unsafe_block = prev_unsafe;
                ret_ty
            }
            _ => panic!("Expected IndexAccess, got {:?}", expr),
        }
    }

    pub(crate) fn check_range_expr(&mut self, expr: &mut Expr) -> Type {
        match expr {
            Expr::Range(RangeExpr {
                start,
                end,
                span: _,
            }) => {
                // Reconcile the bounds so an untyped literal adopts the other bound's type
                // (`0..n` with `n: i64` → `0` becomes i64), mirroring binary-operand inference (#240).
                let (start_ty, end_ty) = self.check_operand_pair(start, end, true);
                if start_ty != end_ty {
                    self.errors.push(format!(
                        "Range start and end types must match, got {:?} and {:?}",
                        start_ty, end_ty
                    ));
                }
                start_ty
            }
            _ => panic!("Expected IndexAccess, got {:?}", expr),
        }
    }

    /// The variant names of `ty`'s enum declaration, or `None` when it is not an enum this
    /// compilation declares (so exhaustiveness cannot be decided by enumeration). A reference
    /// to an enum has the variants of the enum.
    fn enum_variants_of(&self, mut ty: &Type) -> Option<Vec<crate::symbol::Symbol>> {
        while let Type::Borrow { inner, .. } = ty {
            ty = inner;
        }
        let name = match ty {
            Type::Enum(n, _) | Type::Struct(n, _) => n.clone(),
            Type::GenericInstance(inner, _) => match &**inner {
                Type::Enum(n, _) | Type::Struct(n, _) => n.clone(),
                _ => return None,
            },
            _ => return None,
        };
        let base = match name.find('<') {
            Some(i) => crate::symbol::Symbol::from(&name[..i]),
            None => name,
        };
        self.env
            .enums
            .get(base.as_ref())
            .map(|d| d.variants.iter().map(|v| v.0.clone()).collect())
    }

    /// Refuse a `match` arm whose integer literal cannot be represented in the scrutinee's type.
    ///
    /// Such an arm can never be selected. Codegen parses the literal with a zero fallback, which
    /// silently turns it into a comparison against 0 -- so the arm fires for scrutinee 0 instead
    /// of never. Reported here, where the scrutinee's type is known.
    pub(crate) fn check_literal_pattern_range(
        &mut self,
        pattern: &Pattern,
        expr_ty: &Type,
        span: &Span,
    ) {
        if self.speculating {
            return;
        }
        let Pattern::Literal(lit) = pattern else {
            return;
        };
        let Expr::Number(n) = &**lit else {
            return;
        };
        let elem = match expr_ty {
            Type::Scalar(e) => e,
            _ => return,
        };
        if elem.accepts_integer_literal(n.value.as_ref()) == Some(false) {
            self.errors.error_with_code(
                crate::diagnostic::DiagnosticCode::E3019,
                format!(
                    "match arm literal '{}' is not representable in the scrutinee's type '{}', \
                     so the arm can never be selected",
                    n.value, elem
                ),
                Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
            );
        }
    }

    /// Whether `pattern`, matched against a value of type `ty`, takes a part that has to move:
    /// it binds a name to something linear that is not `Copy`. The names get their types as
    /// in `bind_pattern_variables`.
    fn pattern_moves(&mut self, pattern: &Pattern, ty: &Type) -> bool {
        match pattern {
            Pattern::Identifier(_) => ty.is_linear() && !self.is_copy(ty),
            Pattern::EnumVariant(enum_name, variant, Some(parts)) => self
                .payload_types(enum_name, variant, ty)
                .is_some_and(|types| {
                    parts
                        .iter()
                        .zip(&types)
                        .any(|(p, t)| self.pattern_moves(p, t))
                }),
            _ => false,
        }
    }

    /// The names of `pattern`, just bound by a `match` on a reference, are parts of a value that
    /// someone else owns.
    fn mark_bound_behind_ref(&mut self, pattern: &Pattern) {
        let mut bound = Vec::new();
        pattern_names(pattern, &mut bound);
        let scope = self
            .borrow
            .bound_behind_ref
            .last_mut()
            .expect("a match arm has a scope");
        scope.extend(bound.iter().map(|n| n.to_string()));
    }

    /// E4008 when `place` is, or is part of, a name that a `match` on a reference bound. The
    /// value stays with its owner, which drops it. Answers whether it refused.
    pub(crate) fn refuse_move_behind_ref(&mut self, root: &str, place: &str, span: &Span) -> bool {
        if self.speculating || !self.is_bound_behind_ref(root) {
            return false;
        }
        self.errors.error_with_code(
            crate::diagnostic::DiagnosticCode::E4008,
            format!(
                "`{place}` cannot be moved out: `{root}` is part of a value behind a reference, and \
                 the reference does not own that value. Borrow it (`&{place}`) or clone it"
            ),
            Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
        );
        true
    }

    /// An arm of `match scrutinee` took its value: the variable is moved, and what a
    /// reference points at cannot be.
    fn match_moves(&mut self, scrutinee: &Expr) {
        match scrutinee {
            Expr::Identifier(id) => self.consume(&id.name),
            Expr::Dereference(d) => {
                let inner = self.check_expr_type_flag(&mut (*d.expr).clone(), false);
                if let Type::Borrow { inner, .. } = inner {
                    if !self.is_copy(&inner) {
                        self.errors.error_with_code(
                            crate::diagnostic::DiagnosticCode::E4008,
                            format!(
                                "this moves a value of type `{inner}` out from behind a \
                                 reference, which does not own it; copy or clone it instead"
                            ),
                            Some(crate::diagnostic::SourceSpan::from_ast_span(&d.span)),
                        );
                    }
                }
            }
            _ => {}
        }
    }

    /// E3050: a pattern inside a variant's payload other than a name or `_`. Neither code
    /// generator tests it, so `Val(3)` was taken for every `Val`.
    fn refuse_payload_subpattern(&mut self, pattern: &Pattern, span: &Span) {
        let Pattern::EnumVariant(_, variant, Some(parts)) = pattern else {
            return;
        };
        let refutable = parts
            .iter()
            .any(|p| !matches!(p, Pattern::Identifier(_) | Pattern::Wildcard));
        if refutable && !self.speculating {
            self.errors.error_with_code(
                crate::diagnostic::DiagnosticCode::E3050,
                format!(
                    "a payload of `{variant}` can be matched only by a name or `_`; a value or \
                     variant inside it is not tested, so the arm would be taken for any payload. \
                     Bind the payload to a name and test it in the arm"
                ),
                Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
            );
        }
    }

    /// A variant pattern must name the matched value's enum and one of its variants. An arm
    /// is chosen by its variant name alone, so `Other::Red` was taken for a `Color::Red`.
    fn check_variant_pattern(&mut self, pattern: &Pattern, expr_ty: &Type, span: &Span) {
        let Pattern::EnumVariant(enum_name, variant, _) = pattern else {
            return;
        };
        if self.speculating {
            return;
        }
        let at = Some(crate::diagnostic::SourceSpan::from_ast_span(span));
        let base = |name: &str| -> crate::symbol::Symbol {
            name.find('<').map_or(name, |i| &name[..i]).into()
        };
        let pattern_enum = base(enum_name);
        let Some(decl) = self.env.enums.get(pattern_enum.as_ref()) else {
            let poisoned = self.worker.global.registry.poisoned_reason(&pattern_enum)
                == Some(crate::registry::PoisonKind::EnumVariant);
            self.errors.error_with_code(
                crate::diagnostic::DiagnosticCode::E2003,
                if poisoned {
                    crate::registry::ambiguous_import_message("Enum", &pattern_enum)
                } else {
                    format!("Unknown enum {}", enum_name)
                },
                at,
            );
            return;
        };
        if !decl.variants.iter().any(|v| v.0 == *variant) {
            self.errors.error_with_code(
                crate::diagnostic::DiagnosticCode::E2004,
                format!("Enum {} does not have variant {}", pattern_enum, variant),
                at,
            );
            return;
        }
        let mut matched = expr_ty;
        while let Type::Borrow { inner, .. } = matched {
            matched = inner;
        }
        let matched_enum = match matched {
            Type::Enum(n, _) | Type::Struct(n, _) => base(n),
            Type::GenericInstance(inner, _) => match &**inner {
                Type::Enum(n, _) | Type::Struct(n, _) => base(n),
                _ => return,
            },
            _ => return,
        };
        if matched_enum != pattern_enum && self.env.enums.contains_key(matched_enum.as_ref()) {
            self.errors.error_with_code(
                crate::diagnostic::DiagnosticCode::E3051,
                format!(
                    "this pattern is a variant of `{pattern_enum}`, but the value matched is a \
                     `{matched_enum}`; write `{matched_enum}::{variant}`"
                ),
                at,
            );
        }
    }

    /// Refuse `true` or `false` in a `match` on a value that is not a `bool`, a number in a
    /// `match` on a `bool`, and a float in a `match` on an integer. The arm would compare values
    /// of two types.
    fn check_literal_pattern_kind(&mut self, pattern: &Pattern, expr_ty: &Type, span: &Span) {
        let Pattern::Literal(lit) = pattern else {
            return;
        };
        if self.speculating || matches!(expr_ty, Type::Unknown | Type::Generic(..)) {
            return;
        }
        let is_bool = *expr_ty == Type::Scalar(ElementType::Bool);
        let is_integer = !is_bool && matches!(expr_ty, Type::Scalar(e) if e.int_bits().is_some());
        let message = match (bool_literal(lit), &**lit) {
            (Some(b), _) if !is_bool => {
                format!("`{b}` is a `bool`, but the value matched has the type `{expr_ty}`")
            }
            (None, Expr::Number(n)) if is_bool => format!(
                "`{}` is a number, but the value matched is a `bool`; write `true` or `false`",
                n.value
            ),
            (None, Expr::Number(n))
                if is_integer && crate::syntax::types::is_float_literal_spelling(&n.value) =>
            {
                format!(
                    "`{}` is a float, but the value matched has the type `{expr_ty}`",
                    n.value
                )
            }
            _ => return,
        };
        self.errors.error_with_code(
            crate::diagnostic::DiagnosticCode::E3051,
            message,
            Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
        );
    }

    pub(crate) fn bind_pattern_variables(&mut self, pattern: &Pattern, expr_ty: &Type) {
        match pattern {
            Pattern::Identifier(name) => {
                self.insert(name.to_string(), expr_ty.clone());
            }
            Pattern::EnumVariant(enum_name, variant_name, Some(payloads)) => {
                let base = enum_name
                    .find('<')
                    .map_or(enum_name.as_ref(), |i| &enum_name[..i]);
                if self.env.enums.contains_key(base) {
                    // A pattern inside a payload is refused (E3050), but its names are still
                    // bound, so that error is the only one.
                    let Some(types) = self.payload_types(enum_name, variant_name, expr_ty) else {
                        return;
                    };
                    for (i, p) in payloads.iter().enumerate() {
                        let p_ty = types.get(i).cloned().unwrap_or(Type::Unknown);
                        self.bind_pattern_variables(p, &p_ty);
                    }
                } else {
                    for p in payloads {
                        if let Pattern::Identifier(name) = p {
                            self.insert(name.to_string(), Type::Unknown);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    /// The type of each part of the payload of `enum_name::variant_name`, in a value of type
    /// `expr_ty`. `None` when the enum or the variant is unknown, or the variant has no payload.
    fn payload_types(
        &self,
        enum_name: &crate::symbol::Symbol,
        variant_name: &crate::symbol::Symbol,
        expr_ty: &Type,
    ) -> Option<Vec<Type>> {
        let base = enum_name
            .find('<')
            .map_or(enum_name.as_ref(), |i| &enum_name[..i]);
        let enum_decl = self.env.enums.get(base)?;
        let variant = enum_decl.variants.iter().find(|v| v.0 == *variant_name)?;
        let mut mapping = std::collections::HashMap::new();
        let matched = match expr_ty {
            Type::Borrow { inner, .. } => inner,
            other => other,
        };
        if let Type::GenericInstance(_, args) = matched {
            for (param, arg) in enum_decl.generics.iter().zip(args) {
                mapping.insert(param.name().into(), arg.clone());
            }
        }
        let types = variant.1.as_ref()?;
        Some(types.iter().map(|t| t.substitute(&mapping)).collect())
    }

    /// An arm that takes its value owns all of it, also the parts that its pattern does not
    /// name: a `_`, or the payload of `One::It` written with no `(..)`. Each such part that
    /// needs a drop gets a hidden name, so that the arm drops it with its other names.
    fn name_the_parts_to_drop(&mut self, pattern: &mut Pattern, expr_ty: &Type) {
        match pattern {
            Pattern::Wildcard if !self.drops_glue(expr_ty).is_empty() => {
                let name = self.drops_fresh("rest");
                self.insert(name.clone(), expr_ty.clone());
                *pattern = Pattern::Identifier(name.into());
            }
            Pattern::EnumVariant(enum_name, variant, payloads) => {
                let Some(types) = self.payload_types(enum_name, variant, expr_ty) else {
                    return;
                };
                let written = payloads.is_some();
                let parts = payloads.get_or_insert_with(|| vec![Pattern::Wildcard; types.len()]);
                for (part, ty) in parts.iter_mut().zip(&types) {
                    if matches!(part, Pattern::Wildcard) {
                        self.name_the_parts_to_drop(part, ty);
                    }
                }
                if !written && parts.iter().all(|p| matches!(p, Pattern::Wildcard)) {
                    *payloads = None;
                }
            }
            _ => {}
        }
    }

    /// A `match` with string literal arms as a chain of `if`s, since each arm compares the
    /// text with `==`. `match s { "a" => A, _ => B }` is `if s == "a" { A } else { B }`. A subject
    /// that is not a variable is bound to one first, by a match arm that only binds it, so it
    /// runs once. `None` for any other `match`.
    fn string_match_as_ifs(expr: &Expr) -> Option<Expr> {
        let Expr::Match(MatchExpr {
            expr: subject,
            arms,
            span,
        }) = expr
        else {
            return None;
        };
        let span = *span;
        if !arms
            .iter()
            .any(|arm| crate::syntax::is_string_pattern(&arm.pattern))
            || !arms.iter().all(|arm| {
                crate::syntax::is_string_pattern(&arm.pattern)
                    || matches!(arm.pattern, Pattern::Wildcard | Pattern::Identifier(_))
            })
        {
            return None;
        }
        // A first arm of `_` or a name is the only arm that can run.
        if !crate::syntax::is_string_pattern(&arms[0].pattern) {
            return Some(Expr::Match(MatchExpr::new(
                subject.clone(),
                vec![arms[0].clone()],
                span,
            )));
        }
        let name: crate::symbol::Symbol = match &**subject {
            Expr::Identifier(id) => id.name.clone(),
            _ => "match_subject".into(),
        };
        let value = || Expr::Identifier(IdentifierExpr::new(name.clone(), span));
        // From the last arm to the first: each arm is an `if` whose `else` is the arms after it.
        let mut rest: Option<Vec<Statement>> = None;
        for arm in arms.iter().rev() {
            rest = Some(match &arm.pattern {
                Pattern::Wildcard => arm.body.clone(),
                Pattern::Identifier(bound) => {
                    let mut body = vec![Statement::LetDecl(crate::syntax::LetDeclStmt::new(
                        bound.to_string(),
                        false,
                        None,
                        value(),
                        span,
                    ))];
                    body.extend(arm.body.iter().cloned());
                    body
                }
                Pattern::Literal(text) => {
                    let cond = Expr::RelationalOp(crate::syntax::RelationalOpExpr::new(
                        Box::new(value()),
                        crate::syntax::RelationalOp::Eq,
                        text.clone(),
                        span,
                    ));
                    let chain = IfExpr::new(false, Box::new(cond), arm.body.clone(), rest, span);
                    vec![Statement::ExprStmt(crate::syntax::ExprStmtStmt::new(
                        Expr::If(chain),
                        false,
                        span,
                    ))]
                }
                Pattern::EnumVariant(..) => unreachable!("only string, `_` and name arms"),
            });
        }
        // The first arm is a string literal, so `rest` is one `if`.
        let first = match rest?.pop()? {
            Statement::ExprStmt(e) => e.expr,
            _ => return None,
        };
        if matches!(&**subject, Expr::Identifier(_)) {
            return Some(first);
        }
        let arm = crate::syntax::MatchArm {
            pattern: Pattern::Identifier(name),
            body: vec![Statement::ExprStmt(crate::syntax::ExprStmtStmt::new(
                first, false, span,
            ))],
            span: Span::default(),
        };
        Some(Expr::Match(MatchExpr::new(
            subject.clone(),
            vec![arm],
            span,
        )))
    }

    pub(crate) fn check_match_expr(&mut self, expr: &mut Expr, consume: bool) -> Type {
        if let Some(chain) = Self::string_match_as_ifs(expr) {
            *expr = chain;
            return self.check_expr_type_flag(expr, consume);
        }
        let unused = std::mem::replace(&mut self.value_unused, false);
        match expr {
            Expr::Match(MatchExpr {
                expr: match_expr,
                arms,
                span: match_span,
            }) => {
                // Matching a variable, or what a reference points at, reads it: only an arm
                // that binds a value which has to move moves it (below). Any other scrutinee
                // is a value the match takes.
                let place = matches!(**match_expr, Expr::Identifier(_) | Expr::Dereference(_));
                let mut arm_uses = std::collections::HashSet::new();
                for arm in arms.iter() {
                    let mut uses = std::collections::HashSet::new();
                    Self::extract_uses_block(&arm.body, None, &mut uses);
                    let mut bound = Vec::new();
                    pattern_names(&arm.pattern, &mut bound);
                    for name in &bound {
                        uses.remove(name.as_ref());
                    }
                    arm_uses.extend(uses);
                }
                let expr_ty = self.check_head_of_branches(arm_uses, |c| {
                    if place {
                        c.check_expr_type_flag(match_expr, false)
                    } else {
                        c.check_expr_type(match_expr)
                    }
                });
                // A `match` on a reference moves nothing. Its names are parts of a value that
                // someone else owns, and so are the names of a `match` on one of them.
                self.warn_wait_again_after_cancel(&expr_ty, arms);
                let behind_ref = matches!(expr_ty, Type::Borrow { .. })
                    || matches!(&**match_expr, Expr::Identifier(id)
                        if self.is_bound_behind_ref(id.name.as_ref()));
                // When only some arms move a part, the value may still be there after the
                // `match`, as after an `if` that moves it in one branch.
                let every_arm_moves = place
                    && !self.speculating
                    && arms
                        .iter()
                        .all(|arm| self.pattern_moves(&arm.pattern, &expr_ty));

                let mut match_ty: Option<Type> = None;
                let mut covered: Vec<crate::symbol::Symbol> = Vec::new();
                let mut has_wildcard = false;
                // Each arm starts from the moves made before the `match`; afterwards a
                // variable is moved if an arm that reaches the following code moved it.
                let marks_before = self.moved_snapshot();
                let borrows_before = self.borrow.snapshot();
                let mut arm_borrows = Vec::new();
                let mut joined: Option<Vec<std::collections::HashSet<String>>> = None;
                // Each arm starts from the facts before the `match`, and the arms join after it.
                let facts_before = self.consteval.constraints.len();
                let prefix_before = self.consteval.constraints.clone();
                let mut ends_after = Vec::new();
                let mut every_arm_ends = !arms.is_empty();
                let values_before = self.consteval.env.clone();
                let mut values_after_arms = Vec::new();
                let pointers_before = self.pointer_facts.clone();
                let mut pointers_after: Option<crate::hir::check::pointer_facts::PointerFacts> =
                    None;
                // Arms that are only a number with no suffix, such as `_ => { 0 }`. Their number
                // takes the type of the other arms once those are known.
                let mut literal_arms = Vec::new();
                let mut literal_ty = None;
                // A `match` on an integer that the prover models: each arm knows which pattern
                // matched, and which literal patterns above it did not.
                let integer = matches!(&expr_ty, Type::Scalar(e)
                    if e.int_bits().is_some() && *e != ElementType::Bool);
                let matched_number = (integer && crate::hir::prover::is_modelled(match_expr))
                    .then(|| (**match_expr).clone());
                let mut literals_above: Vec<Expr> = Vec::new();
                for (i, arm) in arms.iter_mut().enumerate() {
                    let literal_arm = self.expected_type.is_none()
                        && unsuffixed_number_tail(&mut arm.body).is_some_and(|n| n.ty.is_none());
                    self.consteval.env = values_before.clone();
                    self.pointer_facts = pointers_before.clone();
                    // An arm renames the facts from before the `match` in place, so each arm
                    // starts from a copy of them.
                    self.consteval.constraints = prefix_before.clone();
                    self.restore_moved(marks_before.clone());
                    if i > 0 {
                        arm_borrows.extend(self.borrow.next_branch(&borrows_before));
                    }
                    match &arm.pattern {
                        Pattern::Wildcard | Pattern::Identifier(_) => has_wildcard = true,
                        Pattern::EnumVariant(_, variant, _) => covered.push(variant.clone()),
                        Pattern::Literal(lit) => covered.extend(bool_literal(lit)),
                    }
                    // `_ => {}` and an empty arm for an enum variant or a `bool` are there to make
                    // the match cover every case, so they stay quiet. Every arm the compiler
                    // writes is one of them.
                    if arm.body.is_empty()
                        && !matches!(arm.pattern, Pattern::Wildcard | Pattern::EnumVariant(..))
                        && !matches!(&arm.pattern, Pattern::Literal(l) if bool_literal(l).is_some())
                        && !self.speculating
                    {
                        self.errors.warn(
                            crate::diagnostic::DiagnosticCode::W1008,
                            "this match arm's body is empty, so this case does nothing",
                            Some(crate::diagnostic::SourceSpan::from_ast_span(&arm.span)),
                        );
                    }
                    self.refuse_payload_subpattern(&arm.pattern, &arm.span);
                    self.check_variant_pattern(&arm.pattern, &expr_ty, &arm.span);
                    self.check_literal_pattern_kind(&arm.pattern, &expr_ty, &arm.span);
                    self.pointer_branches
                        .push(Self::pointer_arm_name(&arm.pattern));
                    self.push_scope();
                    self.check_literal_pattern_range(&arm.pattern, &expr_ty, &arm.span);
                    self.bind_pattern_variables(&arm.pattern, &expr_ty);
                    let mut takes_value = !place;
                    if behind_ref {
                        self.mark_bound_behind_ref(&arm.pattern);
                        takes_value = false;
                    } else if place
                        && !self.speculating
                        && self.pattern_moves(&arm.pattern, &expr_ty)
                    {
                        // Moved before the arm's own names exist, which may hide it.
                        self.pop_scope();
                        if every_arm_moves {
                            self.match_moves(match_expr);
                        } else {
                            self.drops_arm_takes(|c| c.match_moves(match_expr));
                        }
                        self.push_scope();
                        self.bind_pattern_variables(&arm.pattern, &expr_ty);
                        // `match *p` takes nothing: through a reference the move is refused
                        // (E4008), and through a raw pointer the value stays with its owner.
                        takes_value = matches!(**match_expr, Expr::Identifier(_));
                    }
                    if takes_value && !self.speculating {
                        if self.drops_rewriting() {
                            self.name_the_parts_to_drop(&mut arm.pattern, &expr_ty);
                        }
                        let mut bound = Vec::new();
                        pattern_names(&arm.pattern, &mut bound);
                        // Listed first, the parts with no name drop last. In Rust too, the end
                        // of a `match` drops the rest of the value that it took.
                        bound.sort_by_key(|n| !n.starts_with("__vx_rest_"));
                        self.drops_bind_arm(&bound);
                    }
                    if let Some(value) = &matched_number {
                        let facts = facts_of_a_pattern(value, &literals_above, &arm.pattern);
                        self.consteval.constraints.extend(facts);
                    }
                    if let Pattern::Literal(lit) = &arm.pattern {
                        literals_above.push((**lit).clone());
                    }

                    let arm_ty = if !self.speculating {
                        self.value_unused = unused;
                        self.check_expr_block(&mut arm.body, None, consume)
                    } else {
                        Type::Struct("void".into(), None)
                    };
                    self.pop_scope();
                    self.pointer_branches.pop();
                    every_arm_ends &= self.branch_ends_early(&arm.body);
                    if !self.branch_ends_early(&arm.body) {
                        values_after_arms.push(self.consteval.env.clone());
                        let mut before = std::mem::take(&mut self.consteval.constraints);
                        let learned = before.split_off(facts_before);
                        ends_after.push(BranchEnd { before, learned });
                        let arm_pointers = std::mem::take(&mut self.pointer_facts);
                        pointers_after = Some(match pointers_after.take() {
                            Some(joined) => joined.join(&arm_pointers),
                            None => arm_pointers,
                        });
                    }
                    if let Some(after) = self.branch_end(&arm.body) {
                        joined = Some(match joined.take() {
                            Some(j) => Self::union_moved(j, after),
                            None => after,
                        });
                    }

                    // An arm contributes to the match's value type only if its block
                    // ends in a tail expression (a non-semicolon `ExprStmt`). Arms that
                    // end in `return`/`break`/`continue`, or in a statement like `assert`,
                    // diverge or yield unit and do not constrain the match value — this is
                    // what lets `Option::unwrap` (Some arm `return v`, None arm asserts)
                    // type-check as `-> T`.
                    let yields_value = matches!(
                        arm.body.last(),
                        Some(Statement::ExprStmt(ExprStmtStmt {
                            has_semi: false,
                            ..
                        }))
                    );
                    if literal_arm {
                        literal_arms.push(i);
                        literal_ty.get_or_insert(arm_ty);
                    } else if yields_value && match_ty.is_none() {
                        match_ty = Some(arm_ty);
                    }
                }
                match &match_ty {
                    Some(ty) => {
                        for i in literal_arms {
                            let n = unsuffixed_number_tail(&mut arms[i].body)
                                .expect("a literal arm ends in a number");
                            if let Some(e) = crate::hir::expr::expected_numeric_elem(ty, &n.value) {
                                n.ty = Some(e);
                            }
                        }
                    }
                    None => match_ty = literal_ty,
                }
                self.restore_moved(joined.unwrap_or(marks_before));
                self.borrow.add_back(arm_borrows);
                // With no `_` arm, the value may match no arm at all.
                if !has_wildcard || values_after_arms.is_empty() {
                    values_after_arms.push(values_before);
                }
                self.pointer_facts = match pointers_after {
                    Some(joined) if has_wildcard => joined,
                    Some(joined) => joined.join(&pointers_before),
                    None => pointers_before,
                };
                self.consteval.env = Self::join_known_values(values_after_arms);
                // With no `_` arm, the code after the `match` can come from before it.
                if !has_wildcard {
                    ends_after.push(BranchEnd {
                        before: prefix_before.clone(),
                        learned: Vec::new(),
                    });
                }
                if every_arm_ends {
                    ends_after.clear();
                }
                self.consteval.constraints = facts_after_branches(prefix_before, ends_after);

                // Coverage. An enum or a `bool` can be enumerated, so a missing case is a fact
                // about the match wherever it sits: the uncovered value falls through, and if
                // every written arm returns, the function falls off its end carrying whatever
                // was in the return slot. A scrutinee that cannot be enumerated has no such
                // fact, so asking it for a wildcard stays limited to value position, where the
                // fall-through edge would otherwise have no value to carry.
                if !has_wildcard && !self.speculating {
                    let at = Some(crate::diagnostic::SourceSpan::from_ast_span(match_span));
                    let cases = if expr_ty == Type::Scalar(ElementType::Bool) {
                        Some(vec!["true".into(), "false".into()])
                    } else {
                        self.enum_variants_of(&expr_ty)
                    };
                    match cases {
                        Some(all) => {
                            let missing: Vec<String> = all
                                .iter()
                                .filter(|v| !covered.contains(v))
                                .map(|v| v.to_string())
                                .collect();
                            if !missing.is_empty() {
                                self.errors.error_with_code(
                                    crate::diagnostic::DiagnosticCode::E3020,
                                    format!(
                                        "this `match` does not cover {}; add an arm for each, \
                                         or a `_` arm",
                                        missing.join(", ")
                                    ),
                                    at,
                                );
                            }
                        }
                        None => {
                            if match_ty.is_some() {
                                self.errors.error_with_code(
                                    crate::diagnostic::DiagnosticCode::E3020,
                                    "this `match` produces a value but has no arm matching every \
                                     value; add a `_` arm so every path yields one"
                                        .to_string(),
                                    at,
                                );
                            }
                        }
                    }
                }

                // When no arm yields a value the match sits in diverging/statement
                // position (every arm returns or aborts). Type it as the expected return
                // type so an implicit `return match { ... }` type-checks; fall back to the
                // historical placeholder when there is no expected return type. A match
                // whose value nothing uses is a statement, and has no value at all.
                if unused && match_ty.is_none() {
                    return Type::Struct("void".into(), None);
                }
                match_ty.unwrap_or_else(|| {
                    self.current_return_type.clone().unwrap_or(Type::Tensor(
                        ElementType::F32,
                        vec![],
                        None,
                    ))
                })
            }
            _ => panic!("Expected IndexAccess, got {:?}", expr),
        }
    }

    /// The end of a closure with a block body, whose `return` or written type gives `returned`:
    /// its last value, of type `ty`, must have that type, and it must not end with no value.
    fn check_closure_end(&mut self, body: &Expr, ty: &Type, returned: &Type, span: &Span) {
        if *ty == Type::Unknown || crate::syntax::is_never_ty(ty) {
            return;
        }
        // Only a block body can end in another way than its value: a `comptime` body returns.
        let Expr::UnsafeBlock(u) = body else {
            return;
        };
        if !u.closure_body {
            return;
        }
        let message = match body {
            Expr::UnsafeBlock(u) if u.ret.is_none() => {
                let functions = &self.env.functions;
                let never = |name: &str| {
                    functions
                        .get(name)
                        .is_some_and(|f| crate::syntax::is_never_ty(&f.0))
                };
                let exits = crate::syntax::block_always_exits(&u.stmts, &never);
                (!exits && !crate::syntax::types::is_void_ty(returned)).then(|| {
                    format!(
                        "this closure returns '{returned}', so its body must end with a value \
                         of that type or with a `return`"
                    )
                })
            }
            _ => (!self.is_assignable(returned, ty)).then(|| {
                format!(
                    "this closure returns '{returned}', so its last value must have that type. \
                     It gives '{ty}'"
                )
            }),
        };
        if let Some(message) = message {
            self.errors.error_with_code(
                crate::diagnostic::DiagnosticCode::E3002,
                message,
                Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
            );
        }
    }

    pub(crate) fn check_closure_expr(&mut self, expr: &mut Expr, _consume: bool) -> Type {
        match expr {
            Expr::Closure(e) => {
                let struct_name = format!("Closure_{}", self.next_id);
                let func_name = format!("{}_call", struct_name);
                self.next_id += 1;

                let cloned_params = e.params.clone();
                // The `let` whose value this closure is. A `let` in the body changes the field.
                let in_let = self
                    .let_closure
                    .take()
                    .zip(self.current_assignment_target.clone());
                // The variable that holds the closure. A closure in another place, such as a
                // call argument, holds its borrows until the end of its statement.
                let holder = in_let.as_ref().map(|(_, closure)| closure.clone());

                let closure_depth = self.scopes.len();
                self.mono.closure_depths.push(closure_depth);
                self.mono.closure_captures_stack.push(HashMap::new());
                self.mono.closure_capture_uses.push(Vec::new());
                // What the body changes decides which captures the closure borrows `&mut`.
                let changed_outside = std::mem::take(&mut self.changed_vars);

                let old_ret = self.current_return_type.take();
                // `|x| -> T { .. }` returns `T`. Otherwise the first `return` gives the type.
                self.current_return_type = Some(e.ret_ty.clone().unwrap_or(Type::Unknown));
                // An `ensures` of the function is not about a `return` of the closure.
                let outer_ensures = std::mem::take(&mut self.consteval.current_ensures);

                self.push_scope();
                for (name, ty) in &cloned_params {
                    self.insert(name.to_string(), ty.clone());
                }
                // The closure's call has its own parameters, which a returned reference may use.
                let outer_params = self.borrow.current_params.clone();
                self.borrow
                    .current_params
                    .extend(cloned_params.iter().cloned());
                let mut b = e.body.clone();
                // A closure is a function, so its body starts a fresh compile-time context.
                // `comptime` directly inside `comptime` is the thing being refused; a lambda
                // whose body is a `comptime` block, written inside another one, is two
                // separate functions and is ordinary.
                let outer_comptime = std::mem::take(&mut self.consteval.comptime_depth);
                self.consteval.closure_body_depth += 1;
                // A `return` in the body leaves the closure, not the function around it.
                self.drops_enter_function(Vec::new(), &[]);
                // A `break` in the body needs a loop of the body.
                let outer_loops = std::mem::replace(
                    &mut self.borrow.loops_outside_the_closure,
                    self.borrow.loop_exits.len(),
                );
                self.traffic.closure_depth += 1;
                self.consteval.comptime_return_is_value = matches!(*b, Expr::ComptimeBlock(_));
                // The body runs when a call runs it, so it starts with no fact of its parent.
                let outer_pointers = self.pointer_facts_enter_body();
                // A borrow made in the body lasts for one call of the closure. What the closure
                // holds between calls is the borrow of its environment, made below.
                let outer_borrows = self.borrow.snapshot();
                // What the body learns, such as from an `assert`, holds only in a call.
                let outer_facts = self.consteval.constraints.len();
                let in_within = self.within_bodies.contains(&e.span);
                self.within_depth += usize::from(in_within);
                let expr_ret_ty = self.check_expr_type(&mut b);
                self.within_depth -= usize::from(in_within);
                self.consteval.constraints.truncate(outer_facts);
                self.borrow.restore(outer_borrows);
                self.pointer_facts_leave_body(outer_pointers);
                self.consteval.comptime_return_is_value = false;
                self.traffic.closure_depth -= 1;
                self.borrow.loops_outside_the_closure = outer_loops;
                self.drops_exit_function();
                self.consteval.closure_body_depth -= 1;
                self.consteval.comptime_depth = outer_comptime;

                self.consteval.current_ensures = outer_ensures;
                let mut ret_ty = expr_ret_ty;
                if let Some(returned) = self
                    .current_return_type
                    .clone()
                    .filter(|t| *t != Type::Unknown)
                {
                    self.check_closure_end(&b, &ret_ty, &returned, &e.span);
                    ret_ty = returned;
                }
                if !self.speculating
                    && self.type_can_hold_reference(&ret_ty)
                    && matches!(
                        self.ref_provenance_of(&b),
                        Some(crate::hir::env::RefProvenance::Local(d)) if d >= closure_depth
                    )
                {
                    self.errors.error_with_code(
                        crate::diagnostic::DiagnosticCode::E4005,
                        "this closure returns a value that points into the closure's own call: \
                         at a parameter, or at the copy the closure keeps of a variable it uses. \
                         That copy is gone once the call returns. Return the value itself, or a \
                         reference made outside the closure"
                            .to_string(),
                        Some(crate::diagnostic::SourceSpan::from_ast_span(&e.span)),
                    );
                }
                self.borrow.current_params = outer_params;

                self.pop_scope();
                self.current_return_type = old_ret;

                let captured_vars_map = self.mono.closure_captures_stack.pop().unwrap_or_default();
                let capture_uses = self.mono.closure_capture_uses.pop().unwrap_or_default();
                self.mono.closure_depths.pop();
                let changed_in_body = std::mem::replace(&mut self.changed_vars, changed_outside);
                self.changed_vars.extend(changed_in_body.iter().cloned());

                let mut captured_vars: Vec<(crate::symbol::Symbol, Type)> =
                    captured_vars_map.into_iter().collect();
                captured_vars.sort_by(|a, b| a.0.cmp(&b.0)); // Stable layout

                // How the closure takes each capture: by value, or through `&` or `&mut` (C1 in
                // docs/discussions/closures_that_change_captured_locals.md).
                let borrows: HashMap<crate::symbol::Symbol, bool> = captured_vars
                    .iter()
                    .filter_map(|(name, ty)| {
                        if matches!(ty, Type::Tensor(..)) || self.is_moved(name.as_ref()) {
                            None
                        } else if changed_in_body.contains(name.as_ref()) {
                            Some((name.clone(), true))
                        } else if is_struct_like(ty) && !self.is_copy(ty) {
                            Some((name.clone(), false))
                        } else {
                            None
                        }
                    })
                    .collect();

                if !self.speculating {
                    // A closure whose body moves a capture takes it, so the code around the
                    // closure must not drop it too. The move in the body stopped at the body.
                    for (name, ty) in &captured_vars {
                        if !matches!(ty, Type::Tensor(..)) && self.is_moved(name.as_ref()) {
                            self.drops_note_move(name.as_ref(), false);
                        }
                    }
                    // A tensor is moved into the closure, as before.
                    for (name, ty) in &captured_vars {
                        if matches!(ty, Type::Tensor(_, _, _))
                            && (self.is_copy(ty)
                                || !self.refuse_move_behind_ref(name, name, &e.span))
                        {
                            self.consume(name.as_ref());
                        }
                    }
                    let copied: Vec<String> = captured_vars
                        .iter()
                        .filter(|(name, ty)| {
                            !borrows.contains_key(name)
                                && !matches!(ty, Type::Struct(_, _) | Type::Tensor(_, _, _))
                        })
                        .map(|(name, _)| name.to_string())
                        .collect();
                    if let Some((at, closure)) = in_let {
                        self.note_closure_copies(closure, &copied, &at);
                    }
                }

                // The field of a borrowed capture holds the reference.
                let env_fields: Vec<(crate::symbol::Symbol, Type)> = captured_vars
                    .iter()
                    .map(|(name, ty)| match borrows.get(name) {
                        Some(&is_mut) => (
                            name.clone(),
                            Type::Borrow {
                                inner: Box::new(ty.clone()),
                                mem_space: None,
                                is_mut,
                                region_id: 0,
                            },
                        ),
                        None => (name.clone(), ty.clone()),
                    })
                    .collect();
                // Each use of a borrowed capture in the body reads through the reference. A field
                // access or a method call reads through a reference by itself, as on `self`.
                let mut through_reference: std::collections::HashSet<(
                    crate::symbol::Symbol,
                    crate::syntax::Span,
                )> = capture_uses
                    .into_iter()
                    .filter(|(name, _)| borrows.contains_key(name))
                    .collect();
                crate::syntax::visit_expr(&b, &mut |e| {
                    let base = match e {
                        Expr::MemberAccess(m) => &*m.base,
                        Expr::MethodCall(m) => &*m.base,
                        _ => return,
                    };
                    if let Expr::Identifier(id) = base {
                        through_reference.remove(&(id.name.clone(), id.span));
                    }
                });
                crate::syntax::visit_expr_mut(&mut b, &mut |use_| {
                    if let Expr::Identifier(id) = use_ {
                        if through_reference.contains(&(id.name.clone(), id.span)) {
                            let ty = captured_vars.iter().find(|(n, _)| *n == id.name);
                            *use_ = Expr::Dereference(crate::syntax::expr::DereferenceExpr {
                                expr: Box::new(Expr::Identifier(id.clone())),
                                ty: ty.map(|(_, t)| t.clone()),
                                span: id.span,
                            });
                        }
                    }
                });

                // Create StructDecl for the environment
                let struct_decl = decl::StructDecl {
                    name: struct_name.clone().into(),
                    generics: vec![],
                    fields: env_fields.clone(),
                    invariants: Vec::new(),
                    place: Default::default(),
                    doc_comment: None,
                };
                self.mono.generated_structs.push(struct_decl);

                // Create the Function for calling the closure
                let mut env_params: Vec<(crate::symbol::Symbol, Type)> = vec![(
                    "_env".to_string().into(),
                    Type::Pointer(
                        Box::new(Type::Struct(struct_name.clone().into(), None)),
                        None,
                        true,
                    ), // &mut env
                )];

                for (name, ty) in &cloned_params {
                    env_params.push((name.clone(), ty.clone()));
                }

                let mut body_stmts = Vec::new();
                for (cap_name, cap_ty) in &env_fields {
                    let env_access = Expr::MemberAccess(MemberAccessExpr {
                        base: Box::new(Expr::Identifier(IdentifierExpr::new(
                            "_env".to_string().into(),
                            e.span,
                        ))),
                        member: cap_name.clone(),
                        struct_name: Some(struct_name.clone().into()),
                        span: e.span,
                    });
                    // Only a moved capture that the body changes needs `mut`: a tensor. The
                    // others are borrowed, or copied and only read. A reference to a tensor is
                    // `mut` too, because the AST code generator reads it from the environment
                    // only into the cell of a `let mut`.
                    let tensor_reference = matches!(cap_ty, Type::Borrow { inner, .. }
                        if matches!(**inner, Type::Tensor(..)));
                    body_stmts.push(Statement::LetDecl(LetDeclStmt {
                        name: cap_name.clone(),
                        is_mut: !borrows.contains_key(cap_name)
                            && (changed_in_body.contains(cap_name.as_ref()) || tensor_reference),
                        ty_ann: Some(cap_ty.clone()),
                        expr: env_access,
                        span: e.span,
                    }));
                }

                body_stmts.push(Statement::Return(ReturnStmt {
                    expr: Some(*b),
                    span: e.span,
                }));

                let call_func = decl::Function {
                    is_unsafe: false,
                    is_pub: false,
                    name: func_name.clone().into(),
                    generics: vec![],
                    params: env_params,
                    mut_params: Vec::new(),
                    topology: self.active_topology.clone(),
                    return_type: ret_ty.clone(),
                    requires: vec![],
                    ensures: vec![],
                    where_transfers: vec![],
                    body: body_stmts,
                    doc_comment: None,
                    span: e.span,
                    file: self.errors.file.clone(),
                };

                self.mono.functions.push((call_func, 0));

                // The environment is built where the closure is written, and the closure holds
                // the borrow of each borrowed capture from there. The body already reported each
                // conflict with a borrow made before the closure.
                let mut fields = Vec::new();
                for (cap_name, _) in &captured_vars {
                    let mut value = Expr::Identifier(IdentifierExpr::new(cap_name.clone(), e.span));
                    if let Some(&is_mut) = borrows.get(cap_name) {
                        value = Expr::Borrow(BorrowExpr {
                            expr: Box::new(value),
                            is_mut,
                            span: e.span,
                        });
                        if !self.speculating {
                            self.borrow.record(
                                cap_name.as_ref(),
                                crate::hir::env::BorrowRecord {
                                    carried_from: None,
                                    is_mut,
                                    scope_depth: self.scopes.len(),
                                    borrower_name: holder.clone(),
                                    path: Vec::new(),
                                },
                            );
                            if is_mut {
                                self.consteval_forget(cap_name.as_ref());
                            }
                            if holder.is_none() {
                                self.mono
                                    .closure_literal_borrows
                                    .entry(struct_name.clone().into())
                                    .or_default()
                                    .push((cap_name.clone(), is_mut));
                            }
                        }
                    }
                    // In the body of a closure around this one, the value is a use of that
                    // closure's capture.
                    let outer = self.mono.closure_depths.last().copied();
                    let depth = self.lookup_with_depth(cap_name.as_ref()).map(|(_, _, d)| d);
                    if matches!((depth, outer), (Some(d), Some(o)) if d < o) {
                        if let Some(uses) = self.mono.closure_capture_uses.last_mut() {
                            uses.push((cap_name.clone(), e.span));
                        }
                    }
                    fields.push((cap_name.clone(), value));
                }

                *expr = Expr::StructInit(StructInitExpr {
                    name: struct_name.clone().into(),
                    fields,
                    type_id: None,
                    span: e.span,
                });

                // Record the closure's call signature so a later `unify_types` can recover it:
                // `Struct("Closure_N")` erases the args/ret, but matching it against a
                // `ClosureK<Args.., Ret>` parameter (e.g. `.map`'s `Closure1<T, NewItem>`) needs
                // them to bind the method's generics. Keyed by the generated struct name.
                let param_tys: Vec<Type> = cloned_params.iter().map(|(_, ty)| ty.clone()).collect();
                self.mono
                    .closure_signatures
                    .insert(struct_name.clone().into(), (param_tys, ret_ty));

                Type::Struct(struct_name.into(), None)
            }
            _ => panic!("Expected IndexAccess, got {:?}", expr),
        }
    }
}

/// What the prover knows where a branch reaches the code after the branches.
struct BranchEnd {
    /// The facts from before the branches, as the branch changed them.
    before: Vec<Expr>,
    /// The facts that the branch learned, such as its condition.
    learned: Vec<Expr>,
}

/// The prover's facts after branches. `before` holds the facts from before the branches, and
/// `reaching` the branches that reach the code after them.
fn facts_after_branches(before: Vec<Expr>, mut reaching: Vec<BranchEnd>) -> Vec<Expr> {
    match reaching.len() {
        0 => {
            let mut facts = before;
            facts.push(crate::hir::prover::unreachable_fact());
            facts
        }
        1 => {
            let end = reaching.pop().expect("one branch");
            let mut facts = end.before;
            facts.extend(end.learned);
            facts
        }
        _ => join_facts_of_branches(&before, reaching),
    }
}

/// The prover's facts after branches that reach the code after them, `ends`. `before` holds the
/// facts from before the branches. A branch changes such a fact in place when it changes a
/// variable of the fact: an assignment renames the old value to `$old_N`, and any other change
/// ends the fact.
///
/// - A fact that no branch changed holds after the branches.
/// - A fact that one branch changed takes the version of that branch. It names the old values,
///   and the other branches changed no variable of it. Those branches also keep the fact as it was.
/// - The other facts hold only in their branch. So the code after the branches knows "the facts
///   of the first branch, or the facts of the second".
fn join_facts_of_branches(before: &[Expr], ends: Vec<BranchEnd>) -> Vec<Expr> {
    assert!(
        ends.iter().all(|e| e.before.len() == before.len()),
        "a branch keeps the facts before it"
    );
    let no_fact = crate::hir::check::raw::no_fact();
    let mut joined = Vec::with_capacity(before.len());
    let mut either: Vec<Vec<Expr>> = vec![Vec::new(); ends.len()];
    let keep = |facts: &mut Vec<Expr>, fact: &Expr| {
        if *fact != no_fact && is_a_condition(fact) {
            facts.push(fact.clone());
        }
    };
    for (k, fact) in before.iter().enumerate() {
        let changed: Vec<usize> = (0..ends.len())
            .filter(|&j| ends[j].before[k] != *fact)
            .collect();
        match changed[..] {
            [] => joined.push(fact.clone()),
            [one] => {
                joined.push(ends[one].before[k].clone());
                for (j, facts) in either.iter_mut().enumerate() {
                    if j != one {
                        keep(facts, fact);
                    }
                }
            }
            _ => {
                joined.push(no_fact.clone());
                for (j, facts) in either.iter_mut().enumerate() {
                    keep(facts, &ends[j].before[k]);
                }
            }
        }
    }
    for (j, end) in ends.into_iter().enumerate() {
        for fact in end.learned {
            // A marker is a note for a later error, and the prover skips it.
            if crate::hir::stmt::read_lost_invariant(&fact).is_some() {
                joined.push(fact);
            } else {
                keep(&mut either[j], &fact);
            }
        }
    }
    // A branch that knows nothing new makes "one branch or another" say nothing.
    if either.iter().all(|facts| !facts.is_empty()) {
        let all = |facts: Vec<Expr>| join_with(facts, LogicalOp::And);
        joined.push(join_with(
            either.into_iter().map(all).collect(),
            LogicalOp::Or,
        ));
    }
    joined
}

/// Whether the prover can read the fact `e` as true or false. It knows only numbers, so a `bool`
/// variable such as the `c` of `if c` is not one. Inside "one branch or another", such a fact
/// would make the prover ignore the whole fact.
fn is_a_condition(e: &Expr) -> bool {
    match e {
        Expr::RelationalOp(_) => true,
        Expr::LogicalOp(l) => is_a_condition(&l.lhs) && is_a_condition(&l.rhs),
        Expr::UnaryOp(u) if matches!(u.op, UnaryOp::Not) => is_a_condition(&u.expr),
        _ => false,
    }
}

/// `true` or `false`, when the literal of a pattern is one of them.
fn bool_literal(lit: &Expr) -> Option<crate::symbol::Symbol> {
    match lit {
        Expr::Identifier(id) if matches!(id.name.as_ref(), "true" | "false") => {
            Some(id.name.clone())
        }
        _ => None,
    }
}

/// The facts that an arm of a `match` on the number `value` knows from its pattern: `value == 5`
/// for the literal `5`, `n == value` for the name `n`, and `value != l` for each literal `l` of an
/// arm above it. A name that hides a variable of `value` would change what `value` means, so that
/// arm knows nothing.
fn facts_of_a_pattern(value: &Expr, literals_above: &[Expr], pattern: &Pattern) -> Vec<Expr> {
    let number = |e: &Expr| matches!(e, Expr::Number(_));
    let relation = |lhs: Expr, op: RelationalOp, rhs: Expr| {
        Expr::RelationalOp(RelationalOpExpr {
            lhs: Box::new(lhs),
            op,
            rhs: Box::new(rhs),
            span: Span::default(),
            operand_ty: None,
        })
    };
    if let Pattern::Identifier(n) = pattern {
        if crate::hir::check::raw::expr_mentions(value, n.as_ref()) {
            return Vec::new();
        }
    }
    let mut facts: Vec<Expr> = literals_above
        .iter()
        .filter(|l| number(l))
        .map(|l| relation(value.clone(), RelationalOp::NotEq, l.clone()))
        .collect();
    match pattern {
        Pattern::Literal(l) if number(l) => {
            facts.push(relation(value.clone(), RelationalOp::Eq, (**l).clone()))
        }
        Pattern::Identifier(n) => {
            let name = Expr::Identifier(IdentifierExpr {
                name: n.clone(),
                span: Span::default(),
            });
            facts.push(relation(name, RelationalOp::Eq, value.clone()))
        }
        _ => {}
    }
    facts
}

/// `facts` joined with `op`. Each list here has at least one fact.
fn join_with(facts: Vec<Expr>, op: LogicalOp) -> Expr {
    facts
        .into_iter()
        .reduce(|lhs, rhs| {
            Expr::LogicalOp(LogicalOpExpr {
                lhs: Box::new(lhs),
                op: op.clone(),
                rhs: Box::new(rhs),
                span: Span::default(),
            })
        })
        .expect("a list with a fact")
}

/// The number that a block's value is, as in `{ 0 }` or `{ -1 }`.
fn unsuffixed_number_tail(body: &mut [Statement]) -> Option<&mut NumberExpr> {
    let Some(Statement::ExprStmt(ExprStmtStmt {
        expr,
        has_semi: false,
        ..
    })) = body.last_mut()
    else {
        return None;
    };
    match expr {
        Expr::Number(n) => Some(n),
        Expr::UnaryOp(UnaryOpExpr {
            op: UnaryOp::Neg,
            expr,
            ..
        }) => match expr.as_mut() {
            Expr::Number(n) => Some(n),
            _ => None,
        },
        _ => None,
    }
}

/// A struct, or an instance of a generic struct such as `Receiver<i64>`.
fn is_struct_like(ty: &Type) -> bool {
    match ty {
        Type::Struct(..) => true,
        Type::GenericInstance(base, _) => matches!(**base, Type::Struct(..)),
        _ => false,
    }
}
