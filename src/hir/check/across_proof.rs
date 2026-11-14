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
// Two accesses whose index forms differ by a number are apart when that number, as one more
// digit, keeps the form a mixed-radix number: `2 * j` and `2 * j + 1` never meet.
//
// Two accesses whose forms have the same coefficient for the loop's variable are apart when the
// rest of the two forms always differ by less than it: `4 * j + c`, with `c` below 3, and
// `4 * j + 3` never meet.
//
// `v / n` and `v % n`, where `v` is the loop's variable, are two more variables of the pass. When
// one index fixes `v / n` and another fixes `v % n`, the two together fix `v`. `n` is a number above
// zero, or a value from outside the loop that is a factor of the loop's end, as in `0..m * n`: when
// `n` is zero, the loop has no passes.
//
// An access in `if v == c`, where `v` is the loop's variable and `c` a number, is made by one
// pass only. Two such accesses with the same `c` are made by the same pass, so they never race.
//
//===----------------------------------------------------------------------===//

use crate::hir::comptime_interpreter::ComptimeInterpreter;
use crate::hir::flatten::is_tensor_alloc_call;
use crate::syntax::stmt::{AcrossClause, ForLoopStmt, Statement};
use crate::syntax::types::{scalar_of, ElementType};
use crate::syntax::{BinaryOp, Expr, LogicalOp, RelationalOp, Span, UnaryOp};
use std::collections::{BTreeMap, HashMap, HashSet};

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
    Index {
        key: String,
        ty: Option<ElementType>,
    },
    /// A `let` whose value is this form, of this type, the same each time it is read.
    Value(Affine, Option<ElementType>),
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
    /// `c`, when only the pass whose variable is `c` makes the access.
    pass: Option<i64>,
}

struct Proof<'c> {
    /// The parameters with an `atomic` clause, for each function that has one.
    contracts: &'c HashMap<String, Vec<usize>>,
    /// The names of the generic functions.
    templates: &'c HashSet<String>,
    /// The key of the loop's own variable.
    own: String,
    /// The values from outside the loop that are factors of its end, by key.
    end_factors: HashSet<String>,
    scopes: Vec<HashMap<String, Binding>>,
    /// The ranges of the loop indices, by key.
    ranges: HashMap<String, Option<(i64, i64)>>,
    /// The least and the greatest value of each loop index whose range is not numbers, from its
    /// type and from its range.
    type_ranges: HashMap<String, (i128, i128)>,
    /// The most passes of a loop whose size is not a number, by key, for a launch check.
    limits: HashMap<String, i128>,
    /// The `across` loops whose size is not a number and whose range starts at a number: the key,
    /// the position of the `across` clause, the start, and the type of the index.
    open: Vec<(String, Span, i128, ElementType)>,
    accesses: BTreeMap<String, Vec<Access>>,
    /// Tensors from outside the loop that the pass uses whole, not by element.
    whole: HashMap<String, Span>,
    next_key: usize,
    error: Option<(String, Span)>,
    /// The statement being walked, for an expression with no position of its own.
    at: Span,
    /// `c` in the `then` block of an `if` that tests that the loop's variable is `c`.
    pass: Option<i64>,
    /// Each slot that `let at = atomic_fetch_add_..(count, i, k, ..)` takes, by its key: the
    /// element of the counter, and `k`.
    slots: HashMap<String, (String, i64)>,
    /// The tensor argument of each call that takes a slot.
    slot_calls: HashSet<Span>,
}

/// The position of an `across` clause, the largest end that its loop may have, and the type of
/// the loop's index.
pub(crate) type LargestEnd = (Span, i128, ElementType);

/// Why the passes of `f` may touch a place that another pass writes, as an error. When the proof
/// holds, the largest end that each `across` loop in `f` may have, by the position of its
/// `across` clause, with the type of its index. A loop that is not there may have any end.
/// `level` names the loop's level in the message.
///
/// An index of a loop whose size is not a number can have each value of its type, so arithmetic
/// on it may wrap. When the proof fails, it runs again with fewer passes for each such `across`
/// loop. If it then holds, the loops get the most passes that it allows, and the program checks
/// each one at launch. Fewer passes only let more arithmetic fit its type: two indices that are
/// the same in two passes stay the same, so a race stays an error.
pub(crate) fn across_loop_race(
    f: &ForLoopStmt,
    level: &str,
    contracts: &HashMap<String, Vec<usize>>,
    templates: &HashSet<String>,
) -> Result<Vec<LargestEnd>, (String, Span)> {
    let range = match &*f.iterable {
        Expr::Range(r) => int(&r.start).zip(int(&r.end)),
        _ => None,
    };
    // One pass races with nothing.
    if matches!(range, Some((lo, hi)) if hi - lo <= 1) {
        return Ok(Vec::new());
    }
    let prove = |limits: HashMap<String, i128>| {
        let mut p = Proof {
            contracts,
            templates,
            own: String::new(),
            end_factors: match &*f.iterable {
                Expr::Range(r) => factors(&r.end),
                _ => HashSet::new(),
            },
            scopes: vec![HashMap::new()],
            ranges: HashMap::new(),
            type_ranges: HashMap::new(),
            limits,
            open: Vec::new(),
            accesses: BTreeMap::new(),
            whole: HashMap::new(),
            next_key: 0,
            error: None,
            at: f.span,
            pass: None,
            slots: HashMap::new(),
            slot_calls: HashSet::new(),
        };
        p.own = p.bind_index(&f.iter, &f.iterable, range, f.across.as_ref());
        p.stmts(&f.body);
        let error = p.error.take().or_else(|| p.check(&f.iter, level));
        (error, p.open)
    };
    let (error, open) = prove(HashMap::new());
    let Some(error) = error else {
        return Ok(Vec::new());
    };
    // The proof holds for fewer passes, so the most passes that it allows is the largest `k`.
    let holds = |k: i128| {
        let limits = open.iter().map(|(key, ..)| (key.clone(), k)).collect();
        prove(limits).0.is_none()
    };
    if open.is_empty() || !holds(1) {
        return Err(error);
    }
    let (mut ok, mut bad) = (1i128, 1i128 << 64);
    while bad - ok > 1 {
        let mid = ok + (bad - ok) / 2;
        if holds(mid) {
            ok = mid;
        } else {
            bad = mid;
        }
    }
    Ok(open
        .into_iter()
        .map(|(_, at, start, ty)| (at, start + ok, ty))
        .collect())
}

/// The names in a product such as `m * n`, as keys of values from outside the loop.
fn factors(e: &Expr) -> HashSet<String> {
    match e {
        Expr::Identifier(id) => HashSet::from([format!("{}#outside", id.name)]),
        Expr::BinaryOp(b) if b.op == BinaryOp::Mul => {
            let mut all = factors(&b.lhs);
            all.extend(factors(&b.rhs));
            all
        }
        _ => HashSet::new(),
    }
}

fn int(e: &Expr) -> Option<i64> {
    match e {
        Expr::Number(n) => n.value.parse().ok(),
        _ => None,
    }
}

/// The type of a loop's index, from a number at either end of its range.
fn range_type(iterable: &Expr) -> Option<ElementType> {
    match iterable {
        Expr::Range(r) => [&*r.start, &*r.end].into_iter().find_map(|e| match e {
            Expr::Number(n) => n.ty.clone(),
            _ => None,
        }),
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

    fn bind_index(
        &mut self,
        name: &str,
        iterable: &Expr,
        range: Option<(i64, i64)>,
        across: Option<&AcrossClause>,
    ) -> String {
        let key = format!("{name}#{}", self.next_key);
        self.next_key += 1;
        self.ranges.insert(key.clone(), range);
        let ty = range_type(iterable);
        if range.is_none() {
            if let Some((mut lo, mut hi)) = ty.as_ref().and_then(ComptimeInterpreter::integer_range)
            {
                if let Expr::Range(r) = iterable {
                    let full = Some((lo, hi));
                    if let Some((a, _)) = self.value_bounds(&r.start, full) {
                        lo = lo.max(a);
                    }
                    if let Some((_, b)) = self.value_bounds(&r.end, full) {
                        hi = hi.min(b - 1);
                    }
                    if let (Some(a), Some(start), Some(ty)) = (across, int(&r.start), &ty) {
                        let start = start as i128;
                        self.open.push((key.clone(), a.span, start, ty.clone()));
                        if let Some(&k) = self.limits.get(&key) {
                            hi = hi.min(start + k - 1);
                        }
                    }
                }
                self.type_ranges.insert(key.clone(), (lo, hi));
            }
        }
        self.bind(
            name,
            Binding::Index {
                key: key.clone(),
                ty,
            },
        );
        key
    }

    /// The least and the greatest value that `e` can have, when the program computes it with
    /// integers. An operation that may wrap can give any value of its type. A name from outside
    /// the loop can have any value of `ty`.
    fn value_bounds(&self, e: &Expr, ty: Option<(i128, i128)>) -> Option<(i128, i128)> {
        match e {
            Expr::Number(n) => n.value.parse::<i128>().ok().map(|v| (v, v)),
            Expr::Identifier(id) => match self.lookup(id.name.as_ref()) {
                Some(Binding::Index { key, .. }) => match self.ranges.get(key).copied().flatten() {
                    Some((a, b)) if b > a => Some((a as i128, b as i128 - 1)),
                    _ => self.type_ranges.get(key).copied(),
                },
                None => ty,
                _ => None,
            },
            Expr::BinaryOp(b) => {
                let t = b
                    .operand_ty
                    .as_ref()
                    .and_then(ComptimeInterpreter::integer_range)?;
                let wrapped = |(lo, hi): (i128, i128)| {
                    if t.0 <= lo && hi <= t.1 {
                        (lo, hi)
                    } else {
                        t
                    }
                };
                let l = self.value_bounds(&b.lhs, Some(t)).unwrap_or(t);
                let r = self.value_bounds(&b.rhs, Some(t)).unwrap_or(t);
                let corners = |f: fn(i128, i128) -> Option<i128>| {
                    let v = [f(l.0, r.0), f(l.0, r.1), f(l.1, r.0), f(l.1, r.1)];
                    let v: Option<Vec<i128>> = v.into_iter().collect();
                    v.map_or(t, |v| (*v.iter().min().unwrap(), *v.iter().max().unwrap()))
                };
                Some(match b.op {
                    BinaryOp::Add => wrapped(corners(i128::checked_add)),
                    BinaryOp::Sub => wrapped(corners(i128::checked_sub)),
                    BinaryOp::Mul => wrapped(corners(i128::checked_mul)),
                    // A division by a positive number only brings a value closer to zero.
                    BinaryOp::Div if r.0 > 0 => corners(i128::checked_div),
                    _ => t,
                })
            }
            _ => None,
        }
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
                let slot = match value {
                    Expr::FunctionCall(fc) if !l.is_mut => self.slot_of(fc),
                    _ => None,
                };
                let b = match value {
                    _ if slot.is_some() => {
                        let key = format!("{}#slot{}", l.name, self.next_key);
                        self.next_key += 1;
                        let (counter, step, ty) = slot.expect("matched above");
                        self.slots.insert(key.clone(), (counter, step));
                        // A slot is any value of its type.
                        if let Some(range) = ComptimeInterpreter::integer_range(&ty) {
                            self.type_ranges.insert(key.clone(), range);
                        }
                        Binding::Value(Affine::var(key), Some(ty))
                    }
                    Expr::FunctionCall(fc) if is_tensor_alloc_call(fc) => Binding::Tensor,
                    _ if !l.is_mut => self.affine(&l.expr).map_or(Binding::Other, |a| {
                        Binding::Value(a, self.int_type(&l.expr))
                    }),
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
                self.bind_index(&inner.iter, &inner.iterable, range, inner.across.as_ref());
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

    /// `atomic_fetch_add_..(count, i, k, ..)` of an integer, with a number `k` above 0 and an index
    /// `i` that is the same in every pass: the element of the counter, and `k`. Each call gets the
    /// old value, so the calls of all passes get the slots `[old, old + k)`, and no two of them
    /// meet while only such calls change the counter.
    fn slot_of(
        &mut self,
        fc: &crate::syntax::expr::FunctionCallExpr,
    ) -> Option<(String, i64, ElementType)> {
        // A generic instance is the name with `$` and its type arguments after it.
        let name = fc.name.split('$').next().unwrap_or_default();
        let ty = match name {
            "atomic_fetch_add_i32" => ElementType::I32,
            "atomic_fetch_add_u32" => ElementType::U32,
            "atomic_fetch_add_i64" => ElementType::I64,
            "atomic_fetch_add_u64" => ElementType::U64,
            _ => return None,
        };
        if fc.args.len() < 3 {
            return None;
        }
        let tensor = crate::hir::check::atomic_contracts::passed_tensor(&fc.args[0])?;
        if !self.outside(tensor) {
            return None;
        }
        let index = self.affine(&fc.args[1])?;
        if !index.terms.keys().all(|k| k.ends_with("#outside")) {
            return None;
        }
        let step = self.affine(&fc.args[2])?.as_constant().filter(|k| *k > 0)?;
        self.slot_calls.insert(fc.args[0].span());
        Some((format!("{tensor}[{index:?}]"), step, ty))
    }

    /// Whether `x` and `y` are slots of one counter, each plus a number below its `k`, while only
    /// calls that take slots change the counter. Two such places never meet, in one pass or in
    /// two.
    fn slots_apart(&self, x: &Affine, y: &Affine) -> bool {
        let slot = |a: &Affine| {
            let (key, coefficient) = a.terms.iter().next().filter(|_| a.terms.len() == 1)?;
            let (counter, step) = self.slots.get(key).filter(|_| *coefficient == 1)?;
            (0..*step).contains(&a.constant).then_some(counter.as_str())
        };
        let trusted = |counter: &str| {
            let tensor = counter.split('[').next().unwrap_or_default();
            self.accesses.get(tensor).is_some_and(|all| {
                all.iter()
                    .all(|a| a.atomic && self.slot_calls.contains(&a.span))
            })
        };
        match (slot(x), slot(y)) {
            (Some(a), Some(b)) => a == b && trusted(a),
            _ => false,
        }
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
                pass: self.pass,
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
                        pass: self.pass,
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
                        let name =
                            crate::hir::check::atomic_contracts::shown_function(&fc.name, |t| {
                                self.templates.contains(t)
                            });
                        format!(
                            "the call to `{name}` may write through an argument, and the checker \
                             does not read its body. If `{name}` writes no memory, give it \
                             `ensures writes_only()`. If it only reads a tensor parameter `t`, \
                             give it `ensures reads_only(t)`. Or write its work in the loop"
                        )
                    };
                    self.fail(why, fc.span);
                    return;
                }
                // A lane sort and a lane copy write the tensor that they borrow first: a row
                // `t[w]`, or a tensor of this pass.
                let sort = crate::hir::check::lanes::lane_routine(&fc.name)
                    .is_some_and(|(op, _)| op.writes_first());
                if let (true, Some(Expr::Borrow(b))) = (sort, fc.args.first()) {
                    if let Some((name, indices)) = Self::element(&b.expr) {
                        for i in &indices {
                            self.expr(i);
                        }
                        if self.outside(name) {
                            self.record(name, &indices, true, b.span);
                        }
                    } else {
                        self.target(&b.expr, b.span);
                    }
                    fc.args[1..].iter().for_each(|a| self.expr(a));
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
                let outer = self.pass;
                self.pass = self.pass.or(self.one_pass(&i.cond));
                self.block(&i.then_block);
                self.pass = outer;
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

    /// The form of `e`, when the program computes it as an exact integer. A cast that may change
    /// the value, or arithmetic that may wrap, has no form: two passes could then meet.
    fn affine(&self, e: &Expr) -> Option<Affine> {
        match e {
            Expr::Number(n) => n.value.parse().ok().map(Affine::constant),
            Expr::Identifier(id) => match self.lookup(id.name.as_ref()) {
                Some(Binding::Index { key, .. }) => Some(Affine::var(key.clone())),
                Some(Binding::Value(a, _)) => Some(a.clone()),
                Some(Binding::Tensor | Binding::Other) => None,
                // From outside the loop: the same in every pass.
                None => Some(Affine::var(format!("{}#outside", id.name))),
            },
            Expr::AsCast(c) => {
                let form = self.affine(&c.expr)?;
                let (min, max) = scalar_of(&c.target_ty)
                    .as_ref()
                    .and_then(ComptimeInterpreter::integer_range)?;
                let holds = |(lo, hi): (i128, i128)| min <= lo && hi <= max;
                let source = c
                    .source_ty
                    .as_ref()
                    .and_then(scalar_of)
                    .as_ref()
                    .and_then(ComptimeInterpreter::integer_range);
                (source.is_some_and(holds) || self.bounds(&form, false, None).is_some_and(holds))
                    .then_some(form)
            }
            Expr::UnaryOp(u) if u.op == UnaryOp::Neg => {
                let form = self.affine(&u.expr)?.scale(-1)?;
                self.fits(&form, self.int_type(&u.expr).as_ref())
                    .then_some(form)
            }
            Expr::BinaryOp(b) => {
                let (l, r) = (self.affine(&b.lhs)?, self.affine(&b.rhs)?);
                let outside = |f: &Affine| f.terms.keys().all(|v| v.ends_with("#outside"));
                let form = match b.op {
                    BinaryOp::Add => l.add(&r, 1),
                    BinaryOp::Sub => l.add(&r, -1),
                    BinaryOp::Mul => match (l.as_constant(), r.as_constant()) {
                        (Some(k), _) => r.scale(k),
                        (_, Some(k)) => l.scale(k),
                        _ => None,
                    },
                    BinaryOp::Div | BinaryOp::Rem if outside(&l) && outside(&r) => {
                        // Made only of values from outside the loop, so the same in every pass.
                        let op = if b.op == BinaryOp::Div { '/' } else { '%' };
                        return Some(Affine::var(format!("({l:?}){op}({r:?})#outside")));
                    }
                    BinaryOp::Div | BinaryOp::Rem => self.part(&l, &r, b.op == BinaryOp::Div),
                    _ => None,
                }?;
                self.fits(&form, b.operand_ty.as_ref()).then_some(form)
            }
            _ => None,
        }
    }

    /// `v / n` or `v % n` as a variable of its own, when `v` is the loop's variable, `v` is never
    /// below zero, and `n` is a number above zero or a factor of the loop's end. Its key is the key
    /// of `v`, then `/` or `%`, then `n` in parentheses.
    fn part(&self, v: &Affine, n: &Affine, quotient: bool) -> Option<Affine> {
        if *v != Affine::var(self.own.clone()) || self.index_range(&self.own)?.0 < 0 {
            return None;
        }
        let n = match n.as_constant() {
            Some(c) if c > 0 => c.to_string(),
            Some(_) => return None,
            None => {
                let (key, &c) = n.terms.iter().next()?;
                if n.terms.len() != 1
                    || c != 1
                    || n.constant != 0
                    || !self.end_factors.contains(key)
                {
                    return None;
                }
                key.clone()
            }
        };
        let op = if quotient { '/' } else { '%' };
        // The key cannot end with `#outside`, which marks a value from outside the loop.
        Some(Affine::var(format!("{}{op}({n})", self.own)))
    }

    /// The key of the loop's variable, whether the key is of `v / n`, and `n`, for a key that
    /// `part` made.
    fn part_of(key: &str) -> Option<(&str, bool, &str)> {
        let at = key.find(['/', '%'])?;
        let n = key[at + 1..].strip_prefix('(')?.strip_suffix(')')?;
        Some((&key[..at], &key[at..at + 1] == "/", n))
    }

    /// `c` when `cond` holds only in the pass whose variable is `c`: `v == c` or `c == v`, also
    /// as one side of `&&`.
    fn one_pass(&self, cond: &Expr) -> Option<i64> {
        match cond {
            Expr::RelationalOp(r) if r.op == RelationalOp::Eq => {
                let own = Affine::var(self.own.clone());
                let (l, r) = (self.affine(&r.lhs)?, self.affine(&r.rhs)?);
                if l == own {
                    r.as_constant()
                } else if r == own {
                    l.as_constant()
                } else {
                    None
                }
            }
            Expr::LogicalOp(l) if l.op == LogicalOp::And => {
                self.one_pass(&l.lhs).or_else(|| self.one_pass(&l.rhs))
            }
            _ => None,
        }
    }

    /// The integer type of `e`, when the checker recorded it.
    fn int_type(&self, e: &Expr) -> Option<ElementType> {
        match e {
            Expr::Number(n) => n.ty.clone(),
            Expr::BinaryOp(b) => b.operand_ty.clone(),
            Expr::AsCast(c) => scalar_of(&c.target_ty),
            Expr::UnaryOp(u) => self.int_type(&u.expr),
            Expr::Identifier(id) => match self.lookup(id.name.as_ref()) {
                Some(Binding::Index { ty, .. } | Binding::Value(_, ty)) => ty.clone(),
                _ => None,
            },
            _ => None,
        }
    }

    /// The least and the greatest value of the loop index `v`, from its range or its type.
    fn index_range(&self, v: &str) -> Option<(i128, i128)> {
        // `v` is never below zero. A remainder is below `n`, and neither part is above `v`.
        if let Some((own, quotient, n)) = Self::part_of(v) {
            let (lo, hi) = self.index_range(own)?;
            return Some(match (n.parse::<i128>(), quotient) {
                (Ok(n), true) => (lo / n, hi / n),
                (Ok(n), false) => (0, hi.min(n - 1)),
                (Err(_), true) => (-hi, hi),
                (Err(_), false) => (0, hi),
            });
        }
        match self.ranges.get(v).copied().flatten() {
            Some((a, b)) if b > a => Some((a as i128, b as i128 - 1)),
            _ => self.type_ranges.get(v).copied(),
        }
    }

    /// The least and the greatest value of `form`, when each of its variables is a loop index
    /// with known bounds, or `fallback` gives them. With `passes_only`, a value from outside the
    /// loop counts as zero.
    fn bounds(
        &self,
        form: &Affine,
        passes_only: bool,
        fallback: Option<(i128, i128)>,
    ) -> Option<(i128, i128)> {
        let (mut lo, mut hi) = (form.constant as i128, form.constant as i128);
        for (v, &c) in &form.terms {
            if passes_only && v.ends_with("#outside") {
                continue;
            }
            let (a, b) = self.index_range(v).or(fallback)?;
            let (x, y) = (c as i128 * a, c as i128 * b);
            lo += x.min(y);
            hi += x.max(y);
        }
        Some((lo, hi))
    }

    /// Whether the program computes `form` in `ty` without a wrap that can make two passes meet.
    /// A coefficient or the constant must fit the type. A value from outside the loop is the same
    /// in every pass, so it moves every pass by the same amount. Then the passes must spread over
    /// fewer values than the type has. Without such a value, each value must fit the type.
    fn fits(&self, form: &Affine, ty: Option<&ElementType>) -> bool {
        let Some((min, max)) = ty.and_then(ComptimeInterpreter::integer_range) else {
            return false;
        };
        let small = |c: i64| (c.unsigned_abs() as i128) <= max;
        if !(form.terms.values().all(|&c| small(c)) && small(form.constant)) {
            return false;
        }
        // An index whose type the proof does not know has the type of the arithmetic.
        let Some((lo, hi)) = self.bounds(form, true, Some((min, max))) else {
            return false;
        };
        if form.terms.keys().any(|v| v.ends_with("#outside")) {
            hi - lo <= max - min
        } else {
            min <= lo && hi <= max
        }
    }

    /// Whether two passes that compute `form` to the same value are the same pass.
    fn fixes_own(&self, form: &Affine) -> bool {
        self.fixes_own_with(form, None)
    }

    /// As `fixes_own`, for two forms that differ by the constant `gap`: `form` and
    /// `form + gap`. The gap is one more digit, `gap * d` with `d` 0 or 1. So two passes never
    /// meet when the passes and the digit together are a mixed-radix number, as for
    /// `out[2 * j]` and `out[2 * j + 1]`, or `out[j]` and `out[j + 32]` with `j` below 32.
    fn fixes_own_with(&self, form: &Affine, gap: Option<i64>) -> bool {
        self.fixes_with(form, &self.own, gap)
    }

    /// As `fixes_own_with`, for the variable `key` of the pass.
    fn fixes_with(&self, form: &Affine, key: &str, gap: Option<i64>) -> bool {
        if !form.terms.contains_key(key) {
            return false;
        }
        // A value from outside the loop is the same in both passes, so it cancels.
        let mut digits: Vec<(i64, Option<i64>)> = form
            .terms
            .iter()
            .filter(|(v, _)| !v.ends_with("#outside"))
            .map(|(v, c)| {
                // The number of values that `v` can have.
                let span = match Self::part_of(v) {
                    Some((own, ..)) if self.ranges.get(own).copied().flatten().is_some() => self
                        .index_range(v)
                        .and_then(|(lo, hi)| i64::try_from(hi - lo + 1).ok()),
                    Some(_) => None,
                    None => self
                        .ranges
                        .get(v)
                        .copied()
                        .flatten()
                        .map(|(lo, hi)| hi - lo),
                };
                (c.abs(), span)
            })
            .collect();
        if let Some(gap) = gap {
            digits.push((gap, Some(2)));
        }
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

    /// Whether two passes that make `x` and `y` equal are the same pass, when the loop's
    /// variable has the same coefficient `a` in both. Passes `v` and `v'` meet only when
    /// `a * (v - v')` equals what the other terms differ by. When that difference is always
    /// smaller than `a`, `v` is `v'`: `out[4 * x + c]` with `c` below 3 and `out[4 * x + 3]`.
    fn own_term_decides(&self, x: &Affine, y: &Affine) -> bool {
        let (Some(&a), Some(&b)) = (x.terms.get(&self.own), y.terms.get(&self.own)) else {
            return false;
        };
        // A value from outside the loop is the same in both passes. It cancels only when both
        // forms have it with the same coefficient.
        let outside = |f: &Affine| {
            f.terms
                .iter()
                .filter(|(v, _)| v.ends_with("#outside"))
                .map(|(v, c)| (v.clone(), *c))
                .collect::<Vec<_>>()
        };
        if a != b || outside(x) != outside(y) {
            return false;
        }
        // Each pass gives its other loop indices their own values.
        let mut lo = y.constant as i128 - x.constant as i128;
        let mut hi = lo;
        for (form, sign) in [(y, 1i128), (x, -1i128)] {
            for (v, &c) in &form.terms {
                if *v == self.own || v.ends_with("#outside") {
                    continue;
                }
                let Some((p, q)) = self.index_range(v) else {
                    return false;
                };
                let (s, t) = (sign * c as i128 * p, sign * c as i128 * q);
                lo += s.min(t);
                hi += s.max(t);
            }
        }
        let a = (a as i128).abs();
        -a < lo && hi < a
    }

    /// Whether one index of `x` and `y` fixes `v / n` and another fixes `v % n`, for one `n`, so
    /// that the two together fix `v`.
    fn parts_fix_own(&self, x: &[Option<Affine>], y: &[Option<Affine>]) -> bool {
        let fixed = |key: &str| {
            x.iter().zip(y).any(|(x, y)| match (x, y) {
                (Some(x), Some(y)) if x == y => self.fixes_with(x, key, None),
                (Some(x), Some(y)) if x.terms == y.terms => x
                    .constant
                    .checked_sub(y.constant)
                    .and_then(i64::checked_abs)
                    .is_some_and(|gap| self.fixes_with(x, key, Some(gap))),
                _ => false,
            })
        };
        x.iter()
            .flatten()
            .flat_map(|f| f.terms.keys())
            .filter_map(|k| Self::part_of(k))
            .filter(|(own, quotient, _)| *own == self.own && *quotient)
            .any(|(own, _, n)| fixed(&format!("{own}/({n})")) && fixed(&format!("{own}%({n})")))
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
                    // One pass makes both.
                    if w.pass.is_some() && w.pass == other.pass {
                        continue;
                    }
                    let apart = w
                        .indices
                        .iter()
                        .zip(&other.indices)
                        .any(|(x, y)| match (x, y) {
                            (Some(x), Some(y)) if self.slots_apart(x, y) => true,
                            (Some(x), Some(y)) if x == y => {
                                self.fixes_own(x) || self.own_term_decides(x, y)
                            }
                            (Some(x), Some(y)) if x.terms == y.terms => {
                                x.constant
                                    .checked_sub(y.constant)
                                    .and_then(i64::checked_abs)
                                    .is_some_and(|gap| self.fixes_own_with(x, Some(gap)))
                                    || self.own_term_decides(x, y)
                            }
                            (Some(x), Some(y)) => self.own_term_decides(x, y),
                            _ => false,
                        })
                        || self.parts_fix_own(&w.indices, &other.indices);
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
                             `{name}`: the writes must use the same index that fixes `{own}`, \
                             or indices that no two passes make equal, such as `2 * {own}` and \
                             `2 * {own} + 1`"
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
