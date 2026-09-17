//===- check submodule - Vx Compiler ----------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
use crate::hir::stmt::EvalFlow;
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
    fn fold_comptime_block(
        &mut self,
        stmts: &[Statement],
        ret: Option<&Expr>,
        before: &HashMap<crate::symbol::Symbol, Value>,
    ) -> ComptimeFold {
        // Anything it writes that outlives it would have to survive, and the block does not.
        if let Some(name) = Self::escaping_write(stmts) {
            self.report_comptime_block_failure(
                &format!(
                    "it writes to '{}', which is declared outside it -- the block disappears, \
                     so the write would have to disappear with it",
                    name
                ),
                &ret.map(|r| r.span()).unwrap_or_default(),
            );
            return ComptimeFold::Refused;
        }
        let mut env = before.clone();
        let outer_unsupported = self.consteval.unsupported_stmt.replace(false);
        let flow = self.eval_block(stmts, &mut env);
        let ran = !self.consteval.unsupported_stmt.get();
        self.consteval.unsupported_stmt.set(outer_unsupported);

        let span = ret.map(|r| r.span()).unwrap_or_default();
        if !ran {
            self.report_comptime_block_failure(
                "it holds a statement the evaluator cannot run",
                &span,
            );
            return ComptimeFold::Refused;
        }
        // A `return` inside the block, where the block is a closure or function body, is
        // that body's value -- `|| comptime { ..; return x; }` is how the closure fixtures
        // are written. Answer with it, the same as a trailing expression.
        if let EvalFlow::Return(returned) = flow {
            return self.fold_value(returned, &span);
        }
        let Some(ret) = ret else {
            return ComptimeFold::NoValue;
        };
        let value = self.eval_expr(ret, &env);
        self.fold_value(value, &span)
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

    /// A name the block writes that was declared outside it.
    ///
    /// The block disappears, so anything it did has to disappear with it. Writing to a
    /// variable that outlives the block is an effect that cannot: the write would simply
    /// stop happening, which is how this turned a program that printed 4 into one that
    /// printed 0.
    fn escaping_write(stmts: &[Statement]) -> Option<crate::symbol::Symbol> {
        fn walk(
            stmts: &[Statement],
            declared: &mut std::collections::HashSet<String>,
            written: &mut Vec<crate::symbol::Symbol>,
        ) {
            for stmt in stmts {
                match stmt {
                    Statement::LetDecl(d) => {
                        declared.insert(d.name.to_string());
                    }
                    Statement::Assign(a) => {
                        if let Some(root) = TypeChecker::place_root(&a.lhs) {
                            written.push(root.clone());
                        }
                    }
                    Statement::CompoundAssign(c) => {
                        if let Some(root) = TypeChecker::place_root(&c.lhs) {
                            written.push(root.clone());
                        }
                    }
                    Statement::ForLoop(f) => {
                        declared.insert(f.iter.to_string());
                        walk(&f.body, declared, written);
                    }
                    Statement::Loop(l) => walk(&l.body, declared, written),
                    Statement::ExprStmt(e) => {
                        if let Expr::If(i) = &e.expr {
                            walk(&i.then_block, declared, written);
                            if let Some(otherwise) = &i.else_block {
                                walk(otherwise, declared, written);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        let mut declared = std::collections::HashSet::new();
        let mut written = Vec::new();
        walk(stmts, &mut declared, &mut written);
        written
            .into_iter()
            .find(|name| !declared.contains(name.as_ref()))
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
    pub(crate) fn value_to_expr(value: &Value, span: &Span) -> Option<Expr> {
        match value {
            Value::Int(i) => Some(Expr::Number(NumberExpr {
                value: i.to_string().into(),
                ty: None,
                span: *span,
            })),
            Value::Number(n) => Some(Expr::Number(NumberExpr {
                value: n.to_string().into(),
                ty: None,
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
        match expr {
            Expr::ComptimeBlock(ComptimeBlockExpr {
                stmts,
                ret,
                span: block_span,
            }) => {
                let block_span = *block_span;
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
                self.consteval.comptime_depth += 1;
                let mut ret_ty = self.check_expr_block(stmts, consume);
                if let Some(r) = ret {
                    ret_ty = self.check_expr_type(r);
                }
                self.consteval.comptime_depth -= 1;
                self.pop_scope();
                let folded = self.fold_comptime_block(stmts, ret.as_deref(), &before);
                // An assertion the placement fold answered `true` is discharged here: nothing
                // at run time holds a placement, so nothing is left to check. (A false one
                // was reported by the assert check.) Asserts on anything else stay, as the
                // program wrote them.
                stmts.retain(|s| {
                    !matches!(s, Statement::Assert(a)
                        if matches!(&*a.expr, Expr::Identifier(id) if id.name.as_ref() == "true"))
                });
                stmts.retain(|s| !Self::binds_a_comptime_lambda(s));
                // The block ran while compiling and must leave nothing behind, so what it
                // worked out replaces it. A block with no value becomes an empty one, which
                // the statement walk then drops.
                match folded {
                    // The value it worked out replaces it.
                    ComptimeFold::Folded(value) => *expr = *value,
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
    fn block_returns(stmts: &[Statement]) -> bool {
        stmts.iter().any(|s| matches!(s, Statement::Return(_)))
    }

    pub(crate) fn check_if_expr(&mut self, expr: &mut Expr, consume: bool) -> Type {
        let if_expr = match expr {
            Expr::If(e) => e,
            _ => panic!("Expected IndexAccess, got {:?}", expr),
        };

        let cond_ty = self.check_expr_type(&mut if_expr.cond);
        if cond_ty != Type::Scalar(ElementType::Bool) {
            self.errors
                .push("Condition in if expression must be of type bool (i1)".to_string());
        }

        if if_expr.is_comptime {
            let tmp_env = self.consteval_snapshot();
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

        let marks_before_then = self.moved_snapshot();
        self.push_releasing_scope();
        let mut then_ty = Type::Struct("void".into(), None);
        if !self.speculating && !if_expr.then_block.is_empty() {
            then_ty = self.check_expr_block(&mut if_expr.then_block, consume);
        }
        self.pop_scope();
        if Self::block_returns(&if_expr.then_block) {
            self.restore_moved(marks_before_then);
        }

        let mut else_ty = Type::Struct("void".into(), None);
        if let Some(else_b) = if_expr.else_block.as_mut() {
            if !else_b.is_empty() {
                let marks_before_else = self.moved_snapshot();
                self.push_releasing_scope();
                if !self.speculating {
                    else_ty = self.check_expr_block(else_b, consume);
                }
                self.pop_scope();
                if Self::block_returns(else_b) {
                    self.restore_moved(marks_before_else);
                }

                if !if_expr.is_comptime && then_ty != else_ty {
                    self.errors.error_with_code(
                        crate::diagnostic::DiagnosticCode::E3007,
                        format!(
                            "If expression branches have incompatible types: {:?} and {:?}",
                            then_ty, else_ty
                        ),
                        Some(crate::diagnostic::SourceSpan::from_ast_span(&if_expr.span)),
                    );
                }
            }
        } else if !if_expr.is_comptime {
            // Without else block, it evaluates to unit (represented as dummy Tensor)
            then_ty = Type::Struct("void".into(), None);
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

    pub(crate) fn check_unsafeblock_expr(&mut self, expr: &mut Expr, consume: bool) -> Type {
        match expr {
            Expr::UnsafeBlock(UnsafeBlockExpr {
                stmts,
                ret: ret_expr,
                span: _,
            }) => {
                let prev_unsafe = self.in_unsafe_block;
                self.in_unsafe_block = true;
                self.push_scope();
                let mut ret_ty = self.check_expr_block(stmts, consume);
                if let Some(r) = ret_expr {
                    ret_ty = self.check_expr_type_flag(r, consume);
                }
                self.pop_scope();
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
    /// compilation declares (so exhaustiveness cannot be decided by enumeration).
    fn enum_variants_of(&self, ty: &Type) -> Option<Vec<crate::symbol::Symbol>> {
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
    pub(crate) fn check_literal_pattern_range(&mut self, pattern: &Pattern, expr_ty: &Type) {
        if self.speculating {
            return;
        }
        let Pattern::Literal(Expr::Number(n)) = pattern else {
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
                Some(crate::diagnostic::SourceSpan::from_ast_span(&n.span)),
            );
        }
    }

    pub(crate) fn bind_pattern_variables(&mut self, pattern: &Pattern, expr_ty: &Type) {
        match pattern {
            Pattern::Identifier(name) => {
                self.insert(name.to_string(), expr_ty.clone());
            }
            Pattern::EnumVariant(enum_name, variant_name, Some(payloads)) => {
                let mut base_name = enum_name.clone();
                if let Some(idx) = enum_name.find('<') {
                    base_name = enum_name[..idx].to_string().into();
                }
                if let Some(enum_decl) = self.env.enums.get(base_name.as_ref()) {
                    if let Some(variant) = enum_decl.variants.iter().find(|v| v.0 == *variant_name)
                    {
                        if let Some(payload_types) = &variant.1 {
                            let mut mapping = std::collections::HashMap::new();
                            if let Type::GenericInstance(_, args) = expr_ty {
                                for (i, param) in enum_decl.generics.iter().enumerate() {
                                    if i < args.len() {
                                        mapping.insert(param.name().into(), args[i].clone());
                                    }
                                }
                            }
                            for (i, p) in payloads.iter().enumerate() {
                                if let Pattern::Identifier(name) = p {
                                    if i < payload_types.len() {
                                        let p_ty = payload_types[i].substitute(&mapping);
                                        self.insert(name.to_string(), p_ty);
                                    } else {
                                        self.insert(name.to_string(), Type::Unknown);
                                    }
                                }
                            }
                        }
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

    pub(crate) fn check_match_expr(&mut self, expr: &mut Expr, consume: bool) -> Type {
        match expr {
            Expr::Match(MatchExpr {
                expr: match_expr,
                arms,
                span: _,
            }) => {
                let expr_ty = self.check_expr_type(match_expr);

                let mut match_ty: Option<Type> = None;
                let mut covered: Vec<crate::symbol::Symbol> = Vec::new();
                let mut has_wildcard = false;
                for arm in arms {
                    match &arm.pattern {
                        Pattern::Wildcard | Pattern::Identifier(_) => has_wildcard = true,
                        Pattern::EnumVariant(_, variant, _) => covered.push(variant.clone()),
                        _ => {}
                    }
                    self.push_scope();
                    self.check_literal_pattern_range(&arm.pattern, &expr_ty);
                    self.bind_pattern_variables(&arm.pattern, &expr_ty);

                    let arm_ty = if !self.speculating {
                        self.check_expr_block(&mut arm.body, consume)
                    } else {
                        Type::Struct("void".into(), None)
                    };
                    self.pop_scope();

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
                    if yields_value && match_ty.is_none() {
                        match_ty = Some(arm_ty);
                    }
                }

                // A match in value position has to produce one on every path. Without a
                // wildcard arm or full variant coverage there is a path through it that
                // yields nothing, which codegen cannot lower and used to answer with zero.
                if match_ty.is_some() && !has_wildcard && !self.speculating {
                    let variants = self.enum_variants_of(&expr_ty);
                    let exhaustive = match &variants {
                        Some(all) => all.iter().all(|v| covered.contains(v)),
                        None => false,
                    };
                    if !exhaustive {
                        let missing = match &variants {
                            Some(all) => {
                                let m: Vec<String> = all
                                    .iter()
                                    .filter(|v| !covered.contains(v))
                                    .map(|v| v.to_string())
                                    .collect();
                                format!("does not cover {}", m.join(", "))
                            }
                            None => "has no arm matching every value".to_string(),
                        };
                        self.errors.error_with_code(
                            crate::diagnostic::DiagnosticCode::E3020,
                            format!(
                                "this `match` produces a value but {missing}; add a `_` arm so \
                                 every path yields one"
                            ),
                            None,
                        );
                    }
                }

                // When no arm yields a value the match sits in diverging/statement
                // position (every arm returns or aborts). Type it as the expected return
                // type so an implicit `return match { ... }` type-checks; fall back to the
                // historical placeholder when there is no expected return type.
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

    pub(crate) fn check_closure_expr(&mut self, expr: &mut Expr, _consume: bool) -> Type {
        match expr {
            Expr::Closure(e) => {
                let struct_name = format!("Closure_{}", self.next_id);
                let func_name = format!("{}_call", struct_name);
                self.next_id += 1;

                let cloned_params = e.params.clone();

                let closure_depth = self.scopes.len();
                self.mono.closure_depths.push(closure_depth);
                self.mono.closure_captures_stack.push(HashMap::new());

                let old_ret = self.current_return_type.take();
                self.current_return_type = Some(Type::Unknown);

                self.push_scope();
                for (name, ty) in &cloned_params {
                    self.insert(name.to_string(), ty.clone());
                }
                let mut b = e.body.clone();
                // A closure is a function, so its body starts a fresh compile-time context.
                // `comptime` directly inside `comptime` is the thing being refused; a lambda
                // whose body is a `comptime` block, written inside another one, is two
                // separate functions and is ordinary.
                let outer_comptime = std::mem::take(&mut self.consteval.comptime_depth);
                self.consteval.closure_body_depth += 1;
                let expr_ret_ty = self.check_expr_type(&mut b);
                self.consteval.closure_body_depth -= 1;
                self.consteval.comptime_depth = outer_comptime;

                let mut ret_ty = expr_ret_ty;
                if let Some(inferred) = &self.current_return_type {
                    if *inferred != Type::Unknown {
                        ret_ty = inferred.clone();
                    }
                }

                self.pop_scope();
                self.current_return_type = old_ret;

                let captured_vars_map = self.mono.closure_captures_stack.pop().unwrap_or_default();
                self.mono.closure_depths.pop();

                let mut captured_vars: Vec<(crate::symbol::Symbol, Type)> =
                    captured_vars_map.into_iter().collect();
                captured_vars.sort_by(|a, b| a.0.cmp(&b.0)); // Stable layout

                if !self.speculating {
                    // Consume captured variables in the outer scope if they are linear
                    for (name, ty) in &captured_vars {
                        if matches!(ty, Type::Struct(_, _) | Type::Tensor(_, _, _)) {
                            self.consume(name.as_ref());
                        }
                    }
                }

                // Create StructDecl for the environment
                let struct_decl = decl::StructDecl {
                    name: struct_name.clone().into(),
                    generics: vec![],
                    fields: captured_vars.clone(),
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
                for (cap_name, cap_ty) in &captured_vars {
                    let env_access = Expr::MemberAccess(MemberAccessExpr {
                        base: Box::new(Expr::Identifier(IdentifierExpr::new(
                            "_env".to_string().into(),
                            e.span,
                        ))),
                        member: cap_name.clone(),
                        struct_name: Some(struct_name.clone().into()),
                        span: e.span,
                    });
                    body_stmts.push(Statement::LetDecl(LetDeclStmt {
                        name: cap_name.clone(),
                        is_mut: true,
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
                    name: func_name.clone().into(),
                    generics: vec![],
                    params: env_params,
                    topology: self.active_topology.clone(),
                    return_type: ret_ty.clone(),
                    requires: vec![],
                    ensures: vec![],
                    where_transfers: vec![],
                    body: body_stmts,
                    doc_comment: None,
                };

                self.mono.functions.push((call_func, 0));

                let mut fields = Vec::new();
                for (cap_name, _) in &captured_vars {
                    fields.push((
                        cap_name.clone(),
                        Expr::Identifier(IdentifierExpr::new(cap_name.clone(), e.span)),
                    ));
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
