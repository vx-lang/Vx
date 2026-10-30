//===- check submodule - Vx Compiler ----------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// `tasks { .. }` and `spawn { .. }`: T1 to T5 and S5 in
// docs/discussions/cpu_tasks_send_sync_and_shared_atomics.md.
//
// After the checks a `tasks` block and a task body are blocks with a `task` mark,
// `let h = spawn { e }` is `let h = { e }`, and `h.join()` is `h`. For a task that can fail, it is
// a `match` on `h` whose `Err` arm leaves the `tasks` block. Each pass treats them as plain
// blocks, which T7 allows. The flat code generator runs each task on a thread of its own, and a
// read of `h` waits for the task. The checks are for the tasks that run at the same time: a task
// holds a borrow of each variable that it uses, from its `spawn` until its join.
//
//===----------------------------------------------------------------------===//

use super::super::*;
use crate::diagnostic::DiagnosticCode;
use crate::hir::check::thread_safety::ThreadTrait;
use crate::syntax::{Span, TaskBlockExpr, TaskBlockKind};
use std::collections::{BTreeMap, HashSet};

/// What the checker knows about the `tasks` blocks and tasks around the code it checks.
#[derive(Default)]
pub(crate) struct TaskCx {
    /// The open `tasks` blocks and task bodies, innermost last.
    frames: Vec<Frame>,
    /// The task handles that `let h = spawn { .. }` made, in the order of their `let`s.
    handles: Vec<Handle>,
    next_id: usize,
    /// The `spawn`s whose task a later statement of the same block always joins.
    joined_in_block: HashSet<Span>,
    /// The `h.join()` calls that always run after the `spawn` of `h`, in the same block. Such a
    /// join ends the borrows of the task. Any other join leaves them until the `tasks` block ends.
    ending_joins: HashSet<Span>,
    /// Where the `spawn` about to be checked is. Only these two places may hold one (T3).
    spawn_place: Option<SpawnPlace>,
    /// The result type of the last `spawn` in a `let`, with the name of its task.
    last_spawn: Option<(String, Type)>,
    /// True while the receiver of `h.join()` is checked.
    joining: bool,
    /// The tasks started in a loop that is in their `tasks` block.
    loop_tasks: Vec<LoopTask>,
}

enum Frame {
    Tasks {
        /// `scopes.len()` in the block. A task's borrow made at this depth lasts until the end
        /// of the block.
        scope_len: usize,
        closures: usize,
        loops: usize,
        tasks: Vec<String>,
        /// The tasks of the block that can fail.
        failures: Vec<Failure>,
    },
    Spawn {
        /// `scopes.len()` before the body. A variable of a lower scope is from outside the task.
        depth: usize,
        closures: usize,
        loops: usize,
        /// Each variable from outside that the body names, with its type.
        used: BTreeMap<crate::symbol::Symbol, Type>,
    },
}

impl TaskCx {
    /// The scope of the `spawn` of the task body whose top scope is `scope`.
    pub(crate) fn spawn_scope_of_body(&self, scope: usize) -> Option<usize> {
        self.frames.iter().find_map(|f| match f {
            Frame::Spawn { depth, .. } if *depth == scope && *depth > 0 => Some(depth - 1),
            _ => None,
        })
    }
}

struct Handle {
    name: crate::symbol::Symbol,
    /// The scope of the `let`, so that a later variable of the same name is not the handle.
    depth: usize,
    task: String,
    result: Type,
    /// The `T` and `E` of a task whose body gives `Result<T, E>`: a task that can fail (F1).
    fails: Option<(Type, Type)>,
}

/// A task that can fail, in its `tasks` block.
struct Failure {
    handle: crate::symbol::Symbol,
    err: Type,
    spawn: Span,
    /// A join of the task always runs after its `spawn`.
    joined: bool,
}

/// The `T` and `E` of core's `Result<T, E>`. The compiler knows `Result` by its name.
fn result_parts(ty: &Type) -> Option<(Type, Type)> {
    let Type::GenericInstance(base, args) = ty else {
        return None;
    };
    let named_result =
        matches!(&**base, Type::Enum(n, _) | Type::Struct(n, _) if n.as_ref() == "Result");
    match args.as_slice() {
        [t, e] if named_result => Some((t.clone(), e.clone())),
        _ => None,
    }
}

/// How a task takes a variable from outside it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Take {
    Moves,
    Copies,
    Writes,
    Reads,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum SpawnPlace {
    Let,
    Statement,
}

struct LoopTask {
    task: String,
    /// The loops around the `spawn`, counted as `loop_exits.len()`.
    loops: usize,
    /// Each variable the task borrows: its name, the scope that declares it, and whether the
    /// task writes it.
    uses: Vec<(String, usize, bool)>,
}

impl TaskCx {
    /// Start the check of a function body, and return the state of the code around it. The
    /// task names stay unique, so that a task of the function never ends one of its caller.
    pub(crate) fn enter_function(&mut self) -> TaskCx {
        let next_id = self.next_id;
        let outer = std::mem::take(self);
        self.next_id = next_id;
        outer
    }

    pub(crate) fn exit_function(&mut self, outer: TaskCx) {
        let next_id = self.next_id;
        *self = outer;
        self.next_id = next_id;
    }

    fn handle(&self, name: &str, depth: Option<usize>) -> Option<&Handle> {
        let depth = depth?;
        self.handles
            .iter()
            .rev()
            .find(|h| h.name.as_ref() == name && h.depth == depth)
    }

    /// The innermost task body, if no closure starts between it and the code being checked.
    fn innermost_spawn(&self, closures: usize) -> Option<&Frame> {
        self.frames
            .iter()
            .rev()
            .find(|f| matches!(f, Frame::Spawn { .. }))
            .filter(|f| matches!(f, Frame::Spawn { closures: c, .. } if *c == closures))
    }
}

/// The name of the borrower that a task's borrows have.
const TASK_PREFIX: &str = "task#";

/// The end of the borrower name of a task's hold on a variable that it copied.
const COPY_SUFFIX: &str = "/copy";

/// A sentence to add to a borrow error when the borrow `b` belongs to a task.
pub(crate) fn held_by_task_note(b: &crate::hir::env::BorrowRecord) -> &'static str {
    match &b.borrower_name {
        Some(name) if name.starts_with(TASK_PREFIX) && name.ends_with(COPY_SUFFIX) => {
            " A task that is not joined yet copied it, so it must not change until the task \
             ends. To give the task the value that it has now, copy it into a new variable \
             before the `spawn`. To share a value that changes, use an atomic."
        }
        Some(name) if name.starts_with(TASK_PREFIX) => {
            " A task that is not joined yet uses it. A task holds what it uses until its \
             `join()`, or until the end of its `tasks` block."
        }
        _ => "",
    }
}

/// Whether `e` is `spawn { .. }`.
pub(crate) fn is_spawn(e: &Expr) -> bool {
    matches!(
        e,
        Expr::TaskBlock(TaskBlockExpr {
            kind: TaskBlockKind::Spawn,
            ..
        })
    )
}

impl TypeChecker<'_> {
    pub(crate) fn check_task_block_expr(&mut self, expr: &mut Expr, consume: bool) -> Type {
        let Expr::TaskBlock(tb) = expr else {
            unreachable!("only a task block reaches here")
        };
        match tb.kind {
            TaskBlockKind::Tasks => self.check_tasks_block(expr, consume),
            TaskBlockKind::Spawn => self.check_spawn(expr, consume),
        }
    }

    /// Check the statements and the last value of a task block in a scope of its own.
    fn check_task_body(
        &mut self,
        expr: &mut Expr,
        consume: bool,
        before: impl FnOnce(&mut Self),
    ) -> Type {
        let unused = std::mem::replace(&mut self.value_unused, false);
        let Expr::TaskBlock(TaskBlockExpr { stmts, ret, .. }) = expr else {
            unreachable!("only a task block reaches here")
        };
        // As in an `unsafe` block: the block's drops are placed when its statements are checked,
        // so a last value that a variable of the block holds moves into a `$value` first.
        let held = match ret.as_deref() {
            Some(Expr::Identifier(id)) => !id.name.starts_with("$value"),
            Some(Expr::MethodCall(m)) => {
                m.method_name.as_ref() == "join" && matches!(&*m.base, Expr::Identifier(_))
            }
            _ => false,
        };
        if held && !stmts.is_empty() {
            let value = ret.take().expect("matched above");
            let span = value.span();
            stmts.push(Statement::LetDecl(crate::syntax::stmt::LetDeclStmt {
                name: "$value".into(),
                is_mut: false,
                ty_ann: None,
                expr: *value,
                span,
            }));
            *ret = Some(Box::new(Expr::Identifier(IdentifierExpr::new(
                "$value".into(),
                span,
            ))));
        }
        self.push_scope();
        before(self);
        self.value_unused = unused;
        let mut ty = self.check_expr_block(stmts, ret.as_deref(), consume);
        if let Some(r) = ret {
            self.value_unused = unused;
            ty = self.check_expr_type_flag(r, consume);
        }
        ty
    }

    /// Make the checked task block `expr` a plain block for the code generators.
    fn task_block_to_plain_block(
        &self,
        expr: &mut Expr,
        ret: Option<Expr>,
        fails_as: Option<Type>,
    ) {
        if self.speculating {
            return;
        }
        let Expr::TaskBlock(tb) = expr else {
            unreachable!("only a task block reaches here")
        };
        let stmts = std::mem::take(&mut tb.stmts);
        let ret = ret.map(Box::new).or_else(|| tb.ret.take());
        *expr = Expr::UnsafeBlock(UnsafeBlockExpr {
            stmts,
            ret,
            span: tb.span,
            generated: true,
            task: Some(tb.kind),
            fails_as,
        });
    }

    fn check_tasks_block(&mut self, expr: &mut Expr, consume: bool) -> Type {
        if let Expr::TaskBlock(tb) = &*expr {
            find_ending_joins(
                &tb.stmts,
                &mut self.tasks.joined_in_block,
                &mut self.tasks.ending_joins,
            );
        }
        let closures = self.mono.closure_depths.len();
        let loops = self.borrow.loop_exits.len();
        self.drops_enter_tasks();
        let ty = self.check_task_body(expr, consume, |me| {
            me.tasks.frames.push(Frame::Tasks {
                scope_len: me.scopes.len(),
                closures,
                loops,
                tasks: Vec::new(),
                failures: Vec::new(),
            });
        });
        self.drops_exit_tasks();
        // The end of the block joins each task that the program did not join (T2).
        let Some(Frame::Tasks {
            scope_len,
            tasks,
            failures,
            ..
        }) = self.tasks.frames.pop()
        else {
            unreachable!("pushed above")
        };
        for task in &tasks {
            self.borrow.end_task(task);
        }
        self.tasks.loop_tasks.retain(|t| !tasks.contains(&t.task));
        self.tasks.handles.retain(|h| h.depth < scope_len - 1);
        self.pop_scope();
        let fails_as = self.check_block_failures(&failures, &ty, &expr.span());
        self.task_block_to_plain_block(expr, None, fails_as);
        ty
    }

    /// F1: a `tasks` block with a task that can fail gives `Result<U, E>`, with the `E` of each
    /// such task. Each of these tasks has a join that always runs, which gives the block the
    /// task's failure. So no failure is lost. Returns the type of the block when it can fail.
    fn check_block_failures(
        &mut self,
        failures: &[Failure],
        block_ty: &Type,
        span: &Span,
    ) -> Option<Type> {
        if failures.is_empty() {
            return None;
        }
        for f in failures.iter().filter(|f| !f.joined) {
            self.task_error(
                DiagnosticCode::E3060,
                format!(
                    "the task of '{}' can fail, so the `tasks` block must join it. Write `{}.join()` \
                     after the `spawn`, in the same block, where it always runs",
                    f.handle, f.handle
                ),
                &f.spawn,
            );
        }
        let block_err = result_parts(block_ty).map(|(_, e)| e);
        for f in failures {
            if block_err.as_ref() != Some(&f.err) {
                self.task_error(
                    DiagnosticCode::E3060,
                    format!(
                        "the task of '{}' can fail with '{}', so its `tasks` block must give a \
                         `Result<_, {}>`. The block gives '{block_ty}'",
                        f.handle, f.err, f.err
                    ),
                    span,
                );
                return None;
            }
        }
        Some(block_ty.clone())
    }

    fn check_spawn(&mut self, expr: &mut Expr, consume: bool) -> Type {
        let span = expr.span();
        let place = self.tasks.spawn_place.take();
        let closures = self.mono.closure_depths.len();
        let context = match self.tasks.frames.last() {
            Some(Frame::Tasks {
                scope_len,
                closures: c,
                loops,
                ..
            }) if *c == closures => Ok((*scope_len, *loops)),
            Some(Frame::Spawn { closures: c, .. }) if *c == closures => Err(
                "a `spawn` directly in a task is not in a `tasks` block. Put it in a \
                 `tasks { .. }` block in the task",
            ),
            Some(_) => Err(
                "a `spawn` in a closure is not in a `tasks` block, also when the closure is. Put \
                 it in a `tasks { .. }` block in the closure",
            ),
            None => Err(
                "`spawn { .. }` must be in a `tasks { .. }` block. The end of the block waits for \
                 the task",
            ),
        };
        let (tasks_scope_len, tasks_loops) = match context {
            Ok(found) => found,
            Err(message) => {
                self.task_error(DiagnosticCode::E3056, message.to_string(), &span);
                let ty = self.check_task_body(expr, consume, |_| {});
                self.pop_scope();
                self.task_block_to_plain_block(expr, None, None);
                // A handle all the same, so that its `join()` reports nothing more.
                if place == Some(SpawnPlace::Let) {
                    let id = format!("{TASK_PREFIX}{}", self.tasks.next_id);
                    self.tasks.next_id += 1;
                    self.tasks.last_spawn = Some((id, ty));
                }
                return Type::Unknown;
            }
        };
        if place.is_none() {
            self.task_error(
                DiagnosticCode::E3057,
                "a task must not leave its `tasks` block, so a `spawn` must be the value of a \
                 `let` or a statement of its own: `let h = spawn { .. };`"
                    .to_string(),
                &span,
            );
        }

        // Nothing reads the result of a task that is a statement, so its last value is a
        // statement too. A statement drops its value. The last value of a block does not.
        if place != Some(SpawnPlace::Let) {
            // A task that can fail needs a handle, so that its join gives the failure to the
            // block. The type of a tail that names a variable of the task is not known here.
            let tail_ty = match &*expr {
                Expr::TaskBlock(TaskBlockExpr {
                    ret: Some(tail), ..
                }) => Some(self.operand_type(tail)),
                _ => None,
            };
            if let Some((_, err)) = tail_ty.as_ref().and_then(result_parts) {
                self.task_error(
                    DiagnosticCode::E3060,
                    format!(
                        "this task can fail with '{err}', so it needs a handle that the `tasks` \
                         block joins: `let h = spawn {{ .. }};` and then `h.join()`"
                    ),
                    &span,
                );
            }
            if let Expr::TaskBlock(tb) = &mut *expr {
                if let Some(tail) = tb.ret.take() {
                    let span = tail.span();
                    tb.stmts.push(Statement::ExprStmt(ExprStmtStmt {
                        expr: *tail,
                        has_semi: true,
                        span,
                    }));
                }
            }
        }
        let ended_by_join = self.tasks.joined_in_block.contains(&span);
        if !ended_by_join {
            self.move_locals_that_end_first(expr, tasks_scope_len);
        }
        let id = format!("{TASK_PREFIX}{}", self.tasks.next_id);
        self.tasks.next_id += 1;
        let changed_outside = std::mem::take(&mut self.changed_vars);
        let depth = self.scopes.len();
        let loops = self.borrow.loop_exits.len();
        self.tasks.frames.push(Frame::Spawn {
            depth,
            closures,
            loops,
            used: BTreeMap::new(),
        });
        let mut ty = self.check_task_body(expr, consume, |_| {});
        self.pop_scope();
        let Some(Frame::Spawn { mut used, .. }) = self.tasks.frames.pop() else {
            unreachable!("pushed above")
        };
        let changed_in_task = std::mem::replace(&mut self.changed_vars, changed_outside);
        self.changed_vars.extend(changed_in_task.iter().cloned());
        // An assignment writes a variable without reading it.
        for name in &changed_in_task {
            if let Some((t, _, d)) = self.lookup_with_depth(name) {
                if d < depth {
                    used.entry(name.as_str().into())
                        .or_insert_with(|| t.clone());
                }
            }
        }

        let mut uses = Vec::new();
        for (name, var_ty) in &used {
            let Some(var_depth) = self.lookup_with_depth(name).map(|(_, _, d)| d) else {
                continue;
            };
            if self.tasks.handle(name, Some(var_depth)).is_some() {
                continue;
            }
            let moved = self.is_moved(name);
            let writes = changed_in_task.contains(name.as_ref())
                || matches!(var_ty, Type::Borrow { is_mut: true, .. });
            // The `spawn` evaluates what the task takes (T4): a `Copy` value that the task only
            // reads is copied into it.
            let take = if moved {
                Take::Moves
            } else if writes {
                Take::Writes
            } else if self.is_copy(var_ty) {
                Take::Copies
            } else {
                Take::Reads
            };
            self.check_task_takes(name, var_ty, take, &span);
            if take == Take::Moves {
                continue;
            }
            if take == Take::Copies {
                // A copied reference keeps what it points at borrowed until the task ends.
                for (base, _) in self.borrow.borrowed_by(name) {
                    let base_depth = self.lookup_with_depth(&base).map(|(_, _, d)| d);
                    if base_depth.is_some_and(|d| d + 1 > tasks_scope_len) && !ended_by_join {
                        self.report_ends_before_task(name, Some(&base), &span);
                    }
                }
                self.borrow.copy_borrows(name, &id, tasks_scope_len);
                // The code around the task must not write the variable until the task ends, so
                // the copy is always the current value. The variable may end first.
                self.borrow.record(
                    name,
                    crate::hir::env::BorrowRecord {
                        is_mut: false,
                        scope_depth: tasks_scope_len.max(var_depth + 1),
                        borrower_name: Some(format!("{id}{COPY_SUFFIX}")),
                        path: Vec::new(),
                    },
                );
                uses.push((name.to_string(), var_depth, false));
                continue;
            }
            // The variable's scope ends before the `tasks` block, and nothing joins the task
            // before then. The variable of scope `i` lives while `scopes.len() > i`.
            if var_depth + 1 > tasks_scope_len && !ended_by_join {
                self.report_ends_before_task(name, None, &span);
                continue;
            }
            self.borrow.record(
                name,
                crate::hir::env::BorrowRecord {
                    is_mut: writes,
                    scope_depth: tasks_scope_len,
                    borrower_name: Some(id.clone()),
                    path: Vec::new(),
                },
            );
            uses.push((name.to_string(), var_depth, writes));
        }
        self.borrow.start_task(&id);
        if let Some(Frame::Tasks { tasks, .. }) = self.tasks.frames.last_mut() {
            tasks.push(id.clone());
        }
        if loops > tasks_loops {
            self.tasks.loop_tasks.push(LoopTask {
                task: id.clone(),
                loops,
                uses,
            });
        }

        // A `let` of a task with no result holds a `bool` that nothing reads.
        let mut ret = None;
        if place == Some(SpawnPlace::Let) {
            self.tasks.last_spawn = Some((id, ty.clone()));
            if crate::syntax::types::is_void_ty(&ty) {
                ret = Some(Expr::Identifier(IdentifierExpr::new("false".into(), span)));
                ty = Type::Scalar(ElementType::Bool);
            }
        } else {
            ty = Type::Struct("void".into(), None);
        }
        self.task_block_to_plain_block(expr, ret, None);
        ty
    }

    /// K2: a task moves each local that is not `Copy` and that ends before the task's join. The
    /// body then starts with `let v = v;`, so the checks of moves and drops see the move.
    fn move_locals_that_end_first(&mut self, expr: &mut Expr, tasks_scope_len: usize) {
        let Expr::TaskBlock(tb) = expr else {
            unreachable!("only a task block reaches here")
        };
        let mut names = std::collections::BTreeSet::new();
        let mut note = |e: &Expr| {
            if let Expr::Identifier(id) = e {
                names.insert(id.name.clone());
            }
        };
        crate::syntax::visit_exprs(&tb.stmts, &mut note);
        if let Some(r) = &tb.ret {
            crate::syntax::visit_expr(r, &mut note);
        }
        let mut moves = Vec::new();
        for name in names {
            let Some((ty, depth)) = self
                .lookup_with_depth(&name)
                .map(|(t, _, d)| (t.clone(), d))
            else {
                continue;
            };
            // A reference keeps what it points at borrowed, so the task cannot take it away.
            let movable = depth + 1 > tasks_scope_len
                && !self.is_copy(&ty)
                && !matches!(ty, Type::Borrow { .. } | Type::Pointer(..) | Type::Ref(..))
                && self.tasks.handle(&name, Some(depth)).is_none()
                && !binds(&tb.stmts, &name)
                && !tb.ret.as_deref().is_some_and(|r| binds_in_expr(r, &name));
            if movable {
                moves.push(Statement::LetDecl(LetDeclStmt {
                    name: name.clone(),
                    is_mut: self.is_mutable(&name),
                    ty_ann: None,
                    expr: Expr::Identifier(IdentifierExpr::new(name, tb.span)),
                    span: tb.span,
                }));
            }
        }
        tb.stmts.splice(0..0, moves);
    }

    /// E4014: the task borrows `name`, or what the reference `name` points at, `base`. That
    /// variable ends before the `tasks` block does.
    fn report_ends_before_task(&mut self, name: &str, base: Option<&str>, at: &Span) {
        let ends = base.unwrap_or(name);
        let what = match base {
            Some(base) => format!("'{name}', which points at '{base}'"),
            None => format!("'{name}'"),
        };
        self.task_error(
            DiagnosticCode::E4014,
            format!(
                "the task uses {what}, but '{ends}' ends before its `tasks` block does, and the \
                 task can still run then. Join the task in the block that declares '{ends}': \
                 `let h = spawn {{ .. }};` and then `h.join();`"
            ),
            at,
        );
    }

    /// S5: a task that moves or copies a value into itself, or writes a variable, needs its type
    /// to be `Send`. A task that reads a variable in place shares it, so its type must be `Sync`.
    fn check_task_takes(&mut self, name: &str, ty: &Type, take: Take, at: &Span) {
        let (which, what) = match take {
            Take::Moves => (ThreadTrait::Send, format!("moves '{name}' into itself")),
            Take::Copies => (ThreadTrait::Send, format!("copies '{name}' into itself")),
            Take::Writes => (ThreadTrait::Send, format!("writes '{name}'")),
            Take::Reads => (
                ThreadTrait::Sync,
                format!("reads '{name}', which the code around the task can read at the same time"),
            ),
        };
        if let Some(why) = self.why_not_thread_trait(ty, which) {
            let trait_name = if which == ThreadTrait::Send {
                "Send"
            } else {
                "Sync"
            };
            self.task_error(
                DiagnosticCode::E3055,
                format!("the task {what}, so its type '{ty}' must be `{trait_name}`: {why}"),
                at,
            );
        }
    }

    /// `let h = spawn { .. };`: check it, and remember `h` as the handle of the task.
    pub(crate) fn check_spawn_let(&mut self, decl: &mut LetDeclStmt, consume: bool) {
        let annotation = decl.ty_ann.take();
        self.tasks.spawn_place = Some(SpawnPlace::Let);
        self.tasks.last_spawn = None;
        self.check_let_decl_stmt(decl, consume);
        self.tasks.spawn_place = None;
        let Some((task, result)) = self.tasks.last_spawn.take() else {
            return;
        };
        if let Some(ann) = annotation {
            let matches = match &ann {
                Type::GenericInstance(base, args) => {
                    matches!(&**base, Type::Struct(n, _) if n.as_ref() == "Task")
                        && args.len() == 1
                        && args[0] == result
                }
                _ => false,
            };
            if !matches {
                self.task_error(
                    DiagnosticCode::E3001,
                    format!(
                        "'{}' is a task handle of type 'Task<{result}>', not '{ann}'",
                        decl.name
                    ),
                    &decl.span,
                );
            }
        }
        let depth = self.lookup_with_depth(&decl.name).map(|(_, _, d)| d);
        let fails = result_parts(&result);
        if let (Some((_, err)), Some(Frame::Tasks { failures, .. })) =
            (&fails, self.tasks.frames.last_mut())
        {
            failures.push(Failure {
                handle: decl.name.clone(),
                err: err.clone(),
                spawn: decl.span,
                joined: false,
            });
        }
        if let Some(depth) = depth {
            self.tasks.handles.push(Handle {
                name: decl.name.clone(),
                depth,
                task,
                result,
                fails,
            });
        }
    }

    /// The statement `spawn { .. };` is one of the two places that may hold a `spawn`.
    pub(crate) fn allow_spawn_statement(&mut self, expr: &Expr) {
        if is_spawn(expr) {
            self.tasks.spawn_place = Some(SpawnPlace::Statement);
        }
    }

    /// A use of the variable `name`. A task handle has one use, `h.join()` (T3). In a task body,
    /// a variable from outside is something the task takes.
    pub(crate) fn note_task_use(&mut self, name: &str, span: &Span) {
        let found = self.lookup_with_depth(name).map(|(t, _, d)| (t.clone(), d));
        let Some((ty, depth)) = found else {
            return;
        };
        if !self.tasks.joining && self.tasks.handle(name, Some(depth)).is_some() {
            self.task_error(
                DiagnosticCode::E3057,
                format!(
                    "'{name}' is a task handle. A task must not leave its `tasks` block, so the \
                     only use of a handle is `{name}.join()`"
                ),
                span,
            );
        }
        for frame in &mut self.tasks.frames {
            if let Frame::Spawn { depth: d, used, .. } = frame {
                if depth < *d {
                    used.entry(name.into()).or_insert_with(|| ty.clone());
                }
            }
        }
    }

    /// `h.join()` on a task handle `h`: the value of the task. `None` for any other call.
    /// `consume` is false when the code around the join does not take its value.
    pub(crate) fn check_task_join(&mut self, expr: &mut Expr, consume: bool) -> Option<Type> {
        let Expr::MethodCall(MethodCallExpr {
            base,
            method_name,
            args,
            span,
            ..
        }) = &*expr
        else {
            return None;
        };
        let Expr::Identifier(id) = &**base else {
            return None;
        };
        if method_name.as_ref() != "join" || !args.is_empty() {
            return None;
        }
        let depth = self.lookup_with_depth(&id.name).map(|(_, _, d)| d);
        let handle = self.tasks.handle(&id.name, depth)?;
        let (task, result, handle_depth, fails) = (
            handle.task.clone(),
            handle.result.clone(),
            handle.depth,
            handle.fails.clone(),
        );
        let (span, mut receiver) = (*span, (**base).clone());
        let closures = self.mono.closure_depths.len();
        let in_other_task = matches!(
            self.tasks.innermost_spawn(closures),
            Some(Frame::Spawn { depth, .. }) if handle_depth < *depth
        );
        if in_other_task {
            self.task_error(
                DiagnosticCode::E3058,
                format!(
                    "a task cannot join another task yet. Join '{}' in the `tasks` block",
                    id.name
                ),
                &span,
            );
        }
        if let (Some((ok_ty, _)), false) = (fails, in_other_task) {
            let name = id.name.clone();
            return Some(self.check_failing_join(expr, &name, handle_depth, &task, &result, ok_ty));
        }
        // The join becomes `h`. Only code that takes the value moves it out of `h`. Otherwise
        // `h` still owns the value and drops it at the end of its scope.
        self.tasks.joining = true;
        self.check_identifier_expr(&mut receiver, consume);
        self.tasks.joining = false;
        // A second join of the same task is a use after a move, also for a `Copy` result.
        if !consume {
            self.mark_used_keep_drop(&id.name);
        }
        if self.tasks.ending_joins.contains(&span) && !in_other_task {
            self.borrow.end_task(&task);
            self.tasks.loop_tasks.retain(|t| t.task != task);
        }
        if !self.speculating {
            *expr = receiver;
        }
        Some(result)
    }

    /// `h.join()` of a task that can fail (F1): `match h { Ok(v) => { v } Err(e) => .. }`, where
    /// the `Err` arm leaves the `tasks` block, and `e` is the failure of the block. Its type is
    /// the `T` of the task's `Result<T, E>`.
    fn check_failing_join(
        &mut self,
        expr: &mut Expr,
        handle: &crate::symbol::Symbol,
        handle_depth: usize,
        task: &str,
        result: &Type,
        ok_ty: Type,
    ) -> Type {
        let span = expr.span();
        let closures = self.mono.closure_depths.len();
        // The `Err` arm leaves the innermost `tasks` block, so that must be the task's own block.
        let in_own_block = self.tasks.frames.iter().rev().find_map(|f| match f {
            Frame::Tasks {
                scope_len,
                closures: c,
                ..
            } => Some(*c == closures && handle_depth + 1 >= *scope_len),
            Frame::Spawn { .. } => None,
        });
        if in_own_block != Some(true) {
            self.task_error(
                DiagnosticCode::E3060,
                format!(
                    "the task of '{handle}' can fail, so a join of it must be in its own `tasks` \
                     block, not in a `tasks` block or a closure inside that block"
                ),
                &span,
            );
            // One error for the handle is enough.
            self.note_failure_joined(handle);
            return ok_ty;
        }
        let n = self.tasks.next_id;
        self.tasks.next_id += 1;
        let (ok, err): (crate::symbol::Symbol, crate::symbol::Symbol) =
            (format!("$ok{n}").into(), format!("$err{n}").into());
        let ident = |name: &crate::symbol::Symbol| {
            Expr::Identifier(IdentifierExpr::new(name.clone(), span))
        };
        let arm = |variant: &str, bind: &crate::symbol::Symbol, body: Statement| MatchArm {
            pattern: Pattern::EnumVariant(
                result.to_string().into(),
                variant.into(),
                Some(vec![Pattern::Identifier(bind.clone())]),
            ),
            body: vec![body],
            span: Span::default(),
        };
        let mut leave = BreakStmt::new(span);
        leave.tasks_value = Some(Box::new(ident(&err)));
        let mut joined = Expr::Match(MatchExpr::new(
            Box::new(ident(handle)),
            vec![
                arm(
                    "Ok",
                    &ok,
                    Statement::ExprStmt(ExprStmtStmt {
                        expr: ident(&ok),
                        has_semi: false,
                        span,
                    }),
                ),
                arm("Err", &err, Statement::Break(leave)),
            ],
            span,
        ));
        self.tasks.joining = true;
        let ty = self.check_expr_type_flag(&mut joined, true);
        self.tasks.joining = false;
        if self.tasks.ending_joins.contains(&span) {
            self.borrow.end_task(task);
            self.tasks.loop_tasks.retain(|t| t.task != task);
            self.note_failure_joined(handle);
        }
        if !self.speculating {
            *expr = joined;
        }
        ty
    }

    /// The task of `handle`, which can fail, has a join that always runs.
    fn note_failure_joined(&mut self, handle: &crate::symbol::Symbol) {
        let failures = self.tasks.frames.iter_mut().rev().find_map(|f| match f {
            Frame::Tasks { failures, .. } if failures.iter().any(|f| f.handle == *handle) => {
                Some(failures)
            }
            _ => None,
        });
        if let Some(f) = failures.and_then(|fs| fs.iter_mut().find(|f| f.handle == *handle)) {
            f.joined = true;
        }
    }

    /// A `return`, `break` or `continue` that leaves a task body (E3058).
    pub(crate) fn check_task_exit(&mut self, stmt: &Statement) {
        let closures = self.mono.closure_depths.len();
        let Some(Frame::Spawn { loops, .. }) = self.tasks.innermost_spawn(closures) else {
            return;
        };
        let (what, span) = match stmt {
            Statement::Return(r) => ("`return`", r.span),
            Statement::Break(b) if *loops == self.borrow.loop_exits.len() => ("`break`", b.span),
            Statement::Continue(c) if *loops == self.borrow.loop_exits.len() => {
                ("`continue`", c.span)
            }
            _ => return,
        };
        self.task_error(
            DiagnosticCode::E3058,
            format!(
                "a {what} cannot leave a task. The last expression of the task body is its result"
            ),
            &span,
        );
    }

    /// At the end of a loop: a task that a pass started and did not join runs at the same time
    /// as the next pass (T5). `outer_depth` is `scopes.len()` before the loop, and `changed`
    /// holds the variables that the loop body changes.
    pub(crate) fn check_tasks_in_loop(
        &mut self,
        outer_depth: usize,
        changed: &HashSet<String>,
        span: &Span,
    ) {
        let level = self.borrow.loop_exits.len() + 1;
        let mut found = Vec::new();
        for t in self
            .tasks
            .loop_tasks
            .iter_mut()
            .filter(|t| t.loops == level)
        {
            for (name, depth, writes) in &t.uses {
                if *depth >= outer_depth {
                    continue;
                }
                if *writes {
                    found.push(format!(
                        "a task that this loop starts writes '{name}', and the pass that starts \
                         it does not join it. So the task of the next pass can use '{name}' at \
                         the same time"
                    ));
                } else if changed.contains(name) {
                    found.push(format!(
                        "a task that this loop starts reads '{name}', and the pass that starts it \
                         does not join it. So the next pass can change '{name}' while the task \
                         reads it"
                    ));
                }
            }
            t.loops -= 1;
        }
        found.sort();
        found.dedup();
        for message in found {
            self.task_error(DiagnosticCode::E4014, message, span);
        }
    }

    /// The tasks whose ending join is in `rhs`. An assignment runs its right side first, so
    /// these tasks end before the assignment writes.
    pub(crate) fn tasks_joined_in(&self, rhs: &Expr) -> Vec<String> {
        self.tasks
            .handles
            .iter()
            .filter(|h| {
                self.lookup_with_depth(&h.name).map(|(_, _, d)| d) == Some(h.depth)
                    && join_in(rhs, &h.name).is_some_and(|s| self.tasks.ending_joins.contains(&s))
            })
            .map(|h| h.task.clone())
            .collect()
    }

    fn task_error(&mut self, code: DiagnosticCode, message: String, at: &Span) {
        if self.speculating {
            return;
        }
        self.errors.error_with_code(
            code,
            message,
            Some(crate::diagnostic::SourceSpan::from_ast_span(at)),
        );
    }
}

/// Find the joins that end a task's borrows: for `let h = spawn { .. }` in a block, the first
/// later statement of that block that always runs `h.join()`. Blocks inside `if`, loops and
/// `unsafe` are searched too, but not the bodies of tasks and closures.
fn find_ending_joins(stmts: &[Statement], spawns: &mut HashSet<Span>, joins: &mut HashSet<Span>) {
    for (i, stmt) in stmts.iter().enumerate() {
        if let Statement::LetDecl(l) = stmt {
            if is_spawn(&l.expr) {
                for later in &stmts[i + 1..] {
                    if let Some(join) = join_that_always_runs(later, &l.name) {
                        spawns.insert(l.expr.span());
                        joins.insert(join);
                        break;
                    }
                    if matches!(later, Statement::LetDecl(l2) if l2.name == l.name) {
                        break;
                    }
                }
            }
        }
        for block in nested_blocks(stmt) {
            find_ending_joins(block, spawns, joins);
        }
    }
}

/// The blocks of statements directly inside `stmt` that `find_ending_joins` searches.
fn nested_blocks(stmt: &Statement) -> Vec<&[Statement]> {
    let mut blocks: Vec<&[Statement]> = Vec::new();
    match stmt {
        Statement::ForLoop(f) => blocks.push(&f.body),
        Statement::Loop(l) => blocks.push(&l.body),
        Statement::ExprStmt(ExprStmtStmt { expr, .. }) => match expr {
            Expr::If(i) => {
                blocks.push(&i.then_block);
                if let Some(e) = &i.else_block {
                    blocks.push(e);
                }
            }
            Expr::Match(m) => blocks.extend(m.arms.iter().map(|a| a.body.as_slice())),
            Expr::UnsafeBlock(u) => blocks.push(&u.stmts),
            _ => {}
        },
        _ => {}
    }
    blocks
}

/// The span of `name.join()` in `stmt` if it runs each time `stmt` runs.
fn join_that_always_runs(stmt: &Statement, name: &str) -> Option<Span> {
    match stmt {
        Statement::ExprStmt(ExprStmtStmt { expr, .. }) => join_in(expr, name),
        Statement::LetDecl(l) => join_in(&l.expr, name),
        Statement::Assign(a) => join_in(&a.rhs, name),
        Statement::CompoundAssign(a) => join_in(&a.rhs, name),
        Statement::Return(r) => r.expr.as_ref().and_then(|e| join_in(e, name)),
        _ => None,
    }
}

/// The span of `name.join()` in `e` if it runs each time `e` runs. The other side of `&&` and
/// `||`, the arms of `if` and `match`, and closures may not run, so they are not searched.
fn join_in(e: &Expr, name: &str) -> Option<Span> {
    let first = |es: Vec<&Expr>| es.into_iter().find_map(|e| join_in(e, name));
    match e {
        Expr::MethodCall(m) => {
            if m.method_name.as_ref() == "join"
                && m.args.is_empty()
                && matches!(&*m.base, Expr::Identifier(id) if id.name.as_ref() == name)
            {
                return Some(m.span);
            }
            first(std::iter::once(&*m.base).chain(m.args.iter()).collect())
        }
        Expr::FunctionCall(c) => first(c.args.iter().collect()),
        Expr::BinaryOp(b) => first(vec![&*b.lhs, &*b.rhs]),
        Expr::RelationalOp(r) => first(vec![&*r.lhs, &*r.rhs]),
        Expr::LogicalOp(l) => join_in(&l.lhs, name),
        Expr::UnaryOp(u) => join_in(&u.expr, name),
        Expr::AsCast(a) => join_in(&a.expr, name),
        Expr::If(i) => join_in(&i.cond, name),
        Expr::Print(p) => first(p.args.iter().collect()),
        Expr::Println(p) => first(p.args.iter().collect()),
        _ => None,
    }
}

/// Whether `stmts` declare a variable `name` anywhere, so that a use of `name` in them may not
/// be the variable outside.
fn binds(stmts: &[Statement], name: &str) -> bool {
    let here = stmts.iter().any(|s| match s {
        Statement::LetDecl(l) => l.name.as_ref() == name,
        Statement::ForLoop(l) => l.iter == name,
        _ => false,
    });
    let mut found = false;
    crate::syntax::visit_exprs(stmts, &mut |e| found |= binds_in_expr(e, name));
    here || found
}

fn binds_in_expr(e: &Expr, name: &str) -> bool {
    match e {
        Expr::If(i) => {
            binds(&i.then_block, name) || i.else_block.as_ref().is_some_and(|b| binds(b, name))
        }
        Expr::Match(m) => m
            .arms
            .iter()
            .any(|a| pattern_binds(&a.pattern, name) || binds(&a.body, name)),
        Expr::UnsafeBlock(b) => binds(&b.stmts, name),
        Expr::ComptimeBlock(b) => binds(&b.stmts, name),
        Expr::SpawnOn(b) => binds(&b.stmts, name),
        Expr::TaskBlock(b) => binds(&b.stmts, name),
        Expr::Closure(c) => c.params.iter().any(|(p, _)| p.as_ref() == name),
        _ => false,
    }
}

fn pattern_binds(p: &crate::syntax::Pattern, name: &str) -> bool {
    use crate::syntax::Pattern;
    match p {
        Pattern::Identifier(n) => n.as_ref() == name,
        Pattern::EnumVariant(_, _, Some(ps)) => ps.iter().any(|p| pattern_binds(p, name)),
        _ => false,
    }
}
