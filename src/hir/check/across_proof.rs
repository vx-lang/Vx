//===- across_proof.rs - Vx Compiler --------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// L4 of #252: no pass of an `across` loop may access a place that another pass writes. The
// proof looks at each tensor that the loop writes. Every access to it must share one dimension
// whose index has the same affine form in both accesses, and that form must fix the loop's
// variable. A form fixes it when its terms are the digits of a mixed-radix number: each
// coefficient is larger than the most that the smaller terms can add up to.
//
// An atomic access, through a call or an `mlir!` block with an `atomic(t)` clause, races with no
// other atomic access. A tensor that the loop changes atomically has only atomic accesses.
//
//===----------------------------------------------------------------------===//

use crate::hir::flatten::is_tensor_alloc_call;
use crate::syntax::stmt::{ForLoopStmt, Statement};
use crate::syntax::{BinaryOp, Expr, Span, UnaryOp};
use std::collections::{BTreeMap, HashMap};

/// `Σ coefficient · variable + constant`. A variable is a loop index of this pass, by a name that
/// is unique to its loop, or a value from outside the loop, which is the same in every pass.
#[derive(Clone, Debug, PartialEq)]
struct Affine {
    terms: BTreeMap<String, i64>,
    constant: i64,
}

impl Affine {
    fn constant(c: i64) -> Self {
        Affine {
            terms: BTreeMap::new(),
            constant: c,
        }
    }
    fn var(name: String) -> Self {
        Affine {
            terms: BTreeMap::from([(name, 1)]),
            constant: 0,
        }
    }
    fn add(mut self, other: &Affine, sign: i64) -> Option<Self> {
        for (v, c) in &other.terms {
            let e = self.terms.entry(v.clone()).or_insert(0);
            *e = e.checked_add(c.checked_mul(sign)?)?;
        }
        self.terms.retain(|_, c| *c != 0);
        self.constant = self
            .constant
            .checked_add(other.constant.checked_mul(sign)?)?;
        Some(self)
    }
    fn scale(mut self, k: i64) -> Option<Self> {
        for c in self.terms.values_mut() {
            *c = c.checked_mul(k)?;
        }
        self.terms.retain(|_, c| *c != 0);
        self.constant = self.constant.checked_mul(k)?;
        Some(self)
    }
    fn as_constant(&self) -> Option<i64> {
        self.terms.is_empty().then_some(self.constant)
    }
}

/// What a name means inside the pass.
#[derive(Clone)]
enum Binding {
    /// A loop index of this pass, with its range when the bounds are numbers.
    Index { key: String },
    /// A `let` whose value is this form, the same each time it is read.
    Value(Affine),
    /// A tensor that the pass makes, so no other pass sees it.
    Tensor,
    /// Any other name of the pass.
    Other,
}

/// One access to a tensor from outside the loop.
struct Access {
    write: bool,
    /// An atomic operation, which races with no other one.
    atomic: bool,
    indices: Vec<Option<Affine>>,
    span: Span,
}

struct Proof<'c> {
    /// The parameters with an `atomic` clause, for each function that has one.
    contracts: &'c HashMap<String, Vec<usize>>,
    /// The key of the loop's own variable.
    own: String,
    scopes: Vec<HashMap<String, Binding>>,
    /// The ranges of the loop indices, by key.
    ranges: HashMap<String, Option<(i64, i64)>>,
    accesses: BTreeMap<String, Vec<Access>>,
    /// Tensors from outside the loop that the pass uses whole, not by element.
    whole: HashMap<String, Span>,
    next_key: usize,
    error: Option<(String, Span)>,
    /// The statement being walked, for an expression with no position of its own.
    at: Span,
}

/// Why the passes of `f` may touch a place that another pass writes, or `None` when the proof
/// holds. `level` names the loop's level in the message.
pub(crate) fn across_loop_race(
    f: &ForLoopStmt,
    level: &str,
    contracts: &HashMap<String, Vec<usize>>,
) -> Option<(String, Span)> {
    let range = match &*f.iterable {
        Expr::Range(r) => int(&r.start).zip(int(&r.end)),
        _ => None,
    };
    // One pass races with nothing.
    if matches!(range, Some((lo, hi)) if hi - lo <= 1) {
        return None;
    }
    let mut p = Proof {
        contracts,
        own: String::new(),
        scopes: vec![HashMap::new()],
        ranges: HashMap::new(),
        accesses: BTreeMap::new(),
        whole: HashMap::new(),
        next_key: 0,
        error: None,
        at: f.span,
    };
    p.own = p.bind_index(&f.iter, range);
    p.stmts(&f.body);
    if let Some(e) = p.error {
        return Some(e);
    }
    p.check(&f.iter, level)
}

fn int(e: &Expr) -> Option<i64> {
    match e {
        Expr::Number(n) => n.value.parse().ok(),
        _ => None,
    }
}

impl Proof<'_> {
    fn fail(&mut self, why: String, at: Span) {
        let at = if at == Span::default() { self.at } else { at };
        if self.error.is_none() {
            self.error = Some((why, at));
        }
    }

    fn lookup(&self, name: &str) -> Option<&Binding> {
        self.scopes.iter().rev().find_map(|s| s.get(name))
    }

    fn bind(&mut self, name: &str, b: Binding) {
        self.scopes
            .last_mut()
            .expect("a scope")
            .insert(name.to_string(), b);
    }

    fn bind_index(&mut self, name: &str, range: Option<(i64, i64)>) -> String {
        let key = format!("{name}#{}", self.next_key);
        self.next_key += 1;
        self.ranges.insert(key.clone(), range);
        self.bind(name, Binding::Index { key: key.clone() });
        key
    }

    fn block(&mut self, stmts: &[Statement]) {
        self.scopes.push(HashMap::new());
        self.stmts(stmts);
        self.scopes.pop();
    }

    fn stmts(&mut self, stmts: &[Statement]) {
        for s in stmts {
            self.stmt(s);
        }
    }

    fn stmt(&mut self, s: &Statement) {
        if s.span() != Span::default() {
            self.at = s.span();
        }
        match s {
            Statement::LetDecl(l) => {
                self.expr(&l.expr);
                let value = crate::syntax::bare_unsafe_value(&l.expr);
                let b = match value {
                    Expr::FunctionCall(fc) if is_tensor_alloc_call(fc) => Binding::Tensor,
                    _ if !l.is_mut => self.affine(&l.expr).map_or(Binding::Other, Binding::Value),
                    _ => Binding::Other,
                };
                self.bind(l.name.as_ref(), b);
            }
            Statement::Assign(a) => {
                self.target(&a.lhs, a.span);
                self.expr(&a.rhs);
            }
            Statement::CompoundAssign(a) => {
                self.target(&a.lhs, a.span);
                // `x += v` reads `x` too.
                self.expr(&a.lhs);
                self.expr(&a.rhs);
            }
            Statement::ForLoop(inner) => {
                self.expr(&inner.iterable);
                let range = match &*inner.iterable {
                    Expr::Range(r) => int(&r.start).zip(int(&r.end)),
                    _ => None,
                };
                self.scopes.push(HashMap::new());
                self.bind_index(&inner.iter, range);
                self.stmts(&inner.body);
                self.scopes.pop();
            }
            Statement::Loop(l) => self.block(&l.body),
            Statement::ExprStmt(e) => self.expr(&e.expr),
            Statement::Return(r) => {
                if let Some(e) = &r.expr {
                    self.expr(e);
                }
            }
            Statement::Assert(a) => self.expr(&a.expr),
            Statement::Break(_)
            | Statement::Continue(_)
            | Statement::Drop(_)
            | Statement::MacroCall(_)
            | Statement::Error(_) => {}
        }
    }

    /// The tensor and the indices of `a[i][j]`, outermost index first.
    fn element(e: &Expr) -> Option<(&str, Vec<&Expr>)> {
        let mut indices = Vec::new();
        let mut at = e;
        while let Expr::IndexAccess(ix) = at {
            indices.push(&*ix.index);
            at = &ix.base;
        }
        indices.reverse();
        match at {
            Expr::Identifier(id) if !indices.is_empty() => Some((id.name.as_ref(), indices)),
            _ => None,
        }
    }

    fn outside(&self, name: &str) -> bool {
        self.lookup(name).is_none()
    }

    fn record(&mut self, name: &str, indices: &[&Expr], write: bool, at: Span) {
        let indices = indices.iter().map(|i| self.affine(i)).collect();
        self.accesses
            .entry(name.to_string())
            .or_default()
            .push(Access {
                write,
                atomic: false,
                indices,
                span: at,
            });
    }

    /// An atomic access to the tensor that `arg` passes. A tensor that the pass makes needs
    /// nothing.
    fn atomic_arg(&mut self, arg: &Expr) {
        match crate::hir::check::atomic_contracts::passed_tensor(arg) {
            Some(name) if self.outside(name) => {
                self.accesses
                    .entry(name.to_string())
                    .or_default()
                    .push(Access {
                        write: true,
                        atomic: true,
                        indices: Vec::new(),
                        span: arg.span(),
                    });
            }
            Some(_) => {}
            None => self.fail(
                "the checker cannot see which tensor this atomic operation changes. Pass the \
                 tensor by name"
                    .to_string(),
                arg.span(),
            ),
        }
    }

    fn target(&mut self, lhs: &Expr, at: Span) {
        if let Some((name, indices)) = Self::element(lhs) {
            for i in &indices {
                self.expr(i);
            }
            if self.outside(name) {
                self.record(name, &indices, true, at);
            }
            return;
        }
        match lhs {
            Expr::Identifier(id) if self.outside(id.name.as_ref()) => self.fail(
                format!(
                    "each pass writes `{}`, which is declared outside the loop, so the passes \
                     race. Declare it in the loop, or give each pass an element of a tensor",
                    id.name
                ),
                at,
            ),
            Expr::Identifier(_) => {}
            Expr::MemberAccess(m) => self.target(&m.base, at),
            other => self.fail(
                "the checker cannot see which place this assignment writes".to_string(),
                other.span(),
            ),
        }
    }

    fn expr(&mut self, e: &Expr) {
        if self.error.is_some() {
            return;
        }
        if let Some((name, indices)) = Self::element(e) {
            for i in &indices {
                self.expr(i);
            }
            if self.outside(name) {
                self.record(name, &indices, false, e.span());
            }
            return;
        }
        match e {
            Expr::Identifier(id) => {
                if self.outside(id.name.as_ref()) {
                    self.whole
                        .entry(id.name.as_ref().to_string())
                        .or_insert(id.span);
                }
            }
            Expr::FunctionCall(fc) => {
                // Making a tensor writes only the new tensor.
                // A lane reduction reads its tensor and writes nothing. The type checker
                // refuses a name after `lanes::` that is not one.
                let pure = crate::hir::check::parallel::parallel_pure_call(&fc.name)
                    || is_tensor_alloc_call(fc)
                    || fc.name.as_ref().starts_with("lanes::");
                // A call with an atomic contract changes only the tensors that it names.
                if let Some(params) = (!pure)
                    .then(|| self.contracts.get(fc.name.as_ref()))
                    .flatten()
                {
                    let params = params.clone();
                    for (i, a) in fc.args.iter().enumerate() {
                        if params.contains(&i) {
                            self.atomic_arg(a);
                        } else {
                            self.expr(a);
                        }
                    }
                    return;
                }
                if !pure {
                    let why = if fc.name.as_ref() == "barrier" {
                        "`barrier()` may not be in an `across` loop: the end of a lane loop is \
                         the barrier"
                            .to_string()
                    } else {
                        format!(
                            "the call to `{}` may write through an argument, and the checker \
                             does not read its body. Write its work in the loop",
                            fc.name
                        )
                    };
                    self.fail(why, fc.span);
                    return;
                }
                fc.args.iter().for_each(|a| self.expr(a));
            }
            Expr::Borrow(b) => {
                if b.is_mut {
                    self.fail(
                        "a `&mut` borrow in an `across` loop may write any element".to_string(),
                        b.span,
                    );
                    return;
                }
                self.expr(&b.expr);
            }
            Expr::If(i) => {
                self.expr(&i.cond);
                self.block(&i.then_block);
                if let Some(b) = &i.else_block {
                    self.block(b);
                }
            }
            Expr::Match(m) => {
                self.expr(&m.expr);
                for arm in &m.arms {
                    self.block(&arm.body);
                }
            }
            Expr::UnsafeBlock(b) => {
                self.scopes.push(HashMap::new());
                self.stmts(&b.stmts);
                if let Some(r) = &b.ret {
                    self.expr(r);
                }
                self.scopes.pop();
            }
            // A block with clauses changes only the tensors that they name.
            Expr::InlineMlir(m) if !m.ensures.is_empty() => {
                let named = crate::hir::check::atomic_contracts::atomic_inputs(m);
                for (_, input, _) in &m.inputs {
                    match crate::hir::check::atomic_contracts::passed_tensor(input) {
                        Some(n) if named.contains(n) => self.atomic_arg(input),
                        _ => self.expr(input),
                    }
                }
            }
            Expr::Print(_) | Expr::Println(_) => self.fail(
                "printing in an `across` loop: the passes run at the same time, so the order of \
                 the lines is not known"
                    .to_string(),
                e.span(),
            ),
            Expr::MethodCall(_)
            | Expr::IndirectCall(_)
            | Expr::InlineMlir(_)
            | Expr::SpawnOn(_)
            | Expr::TaskBlock(_)
            | Expr::Closure(_)
            | Expr::ComptimeBlock(_)
            | Expr::Grad(_)
            | Expr::Vjp(_)
            | Expr::Jvp(_) => self.fail(
                format!(
                    "{} in an `across` loop may write memory that the checker cannot see",
                    what(e)
                ),
                e.span(),
            ),
            // The rest hold only expressions.
            _ => {
                let mut subs: Vec<Expr> = Vec::new();
                crate::syntax::visit_expr(e, &mut |s| {
                    if !std::ptr::eq(s, e) {
                        subs.push(s.clone());
                    }
                });
                // Only the direct operands: `visit_expr` also yields what they hold.
                let mut direct: Vec<Expr> = Vec::new();
                for s in subs {
                    if !direct.iter().any(|d| contains(d, &s)) {
                        direct.push(s);
                    }
                }
                for s in &direct {
                    self.expr(s);
                }
            }
        }
    }

    fn affine(&self, e: &Expr) -> Option<Affine> {
        match e {
            Expr::Number(n) => n.value.parse().ok().map(Affine::constant),
            Expr::Identifier(id) => match self.lookup(id.name.as_ref()) {
                Some(Binding::Index { key, .. }) => Some(Affine::var(key.clone())),
                Some(Binding::Value(a)) => Some(a.clone()),
                Some(Binding::Tensor | Binding::Other) => None,
                // From outside the loop: the same in every pass.
                None => Some(Affine::var(format!("{}#outside", id.name))),
            },
            Expr::AsCast(c) => self.affine(&c.expr),
            Expr::UnaryOp(u) if u.op == UnaryOp::Neg => self.affine(&u.expr)?.scale(-1),
            Expr::BinaryOp(b) => {
                let (l, r) = (self.affine(&b.lhs)?, self.affine(&b.rhs)?);
                match b.op {
                    BinaryOp::Add => l.add(&r, 1),
                    BinaryOp::Sub => l.add(&r, -1),
                    BinaryOp::Mul => match (l.as_constant(), r.as_constant()) {
                        (Some(k), _) => r.scale(k),
                        (_, Some(k)) => l.scale(k),
                        _ => None,
                    },
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Whether two passes that compute `form` to the same value are the same pass.
    fn fixes_own(&self, form: &Affine) -> bool {
        if !form.terms.contains_key(&self.own) {
            return false;
        }
        // A value from outside the loop is the same in both passes, so it cancels.
        let mut digits: Vec<(i64, Option<i64>)> = form
            .terms
            .iter()
            .filter(|(v, _)| !v.ends_with("#outside"))
            .map(|(v, c)| {
                let span = self
                    .ranges
                    .get(v)
                    .copied()
                    .flatten()
                    .map(|(lo, hi)| hi - lo);
                (c.abs(), span)
            })
            .collect();
        // For equal coefficients, a smaller range goes first, and a range of unknown size last.
        digits.sort_by_key(|&(c, span)| (c, span.unwrap_or(i64::MAX)));
        let last = digits.len() - 1;
        let mut most: i64 = 0;
        for (i, (c, span)) in digits.into_iter().enumerate() {
            if c <= most {
                return false;
            }
            match span {
                Some(s) if s >= 1 => match c.checked_mul(s - 1).and_then(|x| most.checked_add(x)) {
                    Some(m) => most = m,
                    None => return false,
                },
                _ if i == last => {}
                _ => return false,
            }
        }
        true
    }

    fn check(&self, own: &str, level: &str) -> Option<(String, Span)> {
        for (name, accesses) in &self.accesses {
            if accesses.iter().any(|a| a.atomic) {
                let plain = accesses
                    .iter()
                    .find(|a| !a.atomic)
                    .map(|a| a.span)
                    .or_else(|| self.whole.get(name).copied());
                if let Some(at) = plain {
                    return Some((
                        format!(
                            "the passes of this {level} loop change `{name}` with atomic \
                             operations, so every access to it in the loop must be one"
                        ),
                        at,
                    ));
                }
                continue;
            }
            if let Some(&at) = self.whole.get(name) {
                if accesses.iter().any(|a| a.write) {
                    return Some((
                        format!(
                            "a pass uses all of `{name}`, which the passes of this {level} loop \
                             write"
                        ),
                        at,
                    ));
                }
            }
            for w in accesses.iter().filter(|a| a.write) {
                for other in accesses {
                    let apart = w.indices.iter().zip(&other.indices).any(|(x, y)| {
                        matches!((x, y), (Some(x), Some(y)) if x == y && self.fixes_own(x))
                    });
                    if apart {
                        continue;
                    }
                    let why = if std::ptr::eq(w, other) {
                        format!(
                            "two passes of this {level} loop may write the same element of \
                             `{name}`: an index of the write must fix `{own}`"
                        )
                    } else if other.write {
                        format!(
                            "two passes of this {level} loop may write the same element of \
                             `{name}`: the writes must use the same index that fixes `{own}`"
                        )
                    } else {
                        format!(
                            "a pass of this {level} loop may read an element of `{name}` that \
                             another pass writes: the read must use the same index as the \
                             write, one that fixes `{own}`"
                        )
                    };
                    return Some((why, other.span));
                }
            }
        }
        None
    }
}

/// The name of an expression that the proof cannot see into, for a message.
fn what(e: &Expr) -> &'static str {
    match e {
        Expr::MethodCall(_) => "a method call",
        Expr::IndirectCall(_) => "a call through a function value",
        Expr::InlineMlir(_) => "an `mlir!` block",
        Expr::SpawnOn(_) => "a `spawn on` region",
        Expr::TaskBlock(_) => "a task",
        Expr::Closure(_) => "a closure",
        Expr::ComptimeBlock(_) => "a `comptime` block",
        _ => "a derivative",
    }
}

/// Whether `inner` is inside `outer`.
fn contains(outer: &Expr, inner: &Expr) -> bool {
    let mut found = false;
    crate::syntax::visit_expr(outer, &mut |e| found |= e == inner);
    found
}
