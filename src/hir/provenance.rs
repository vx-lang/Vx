//===- provenance.rs - Vx Compiler -----------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Return-provenance summary: for each reference-returning function, which parameter slot(s)
// its returned reference roots in (#243). It is computed once, structurally, from the body —
// a pure function of the AST with no `TypeChecker` and no shared state — and frozen into the
// immutable `GlobalAstEnv`. The per-function parallel checker only ever *reads* it, so it does
// not interfere with the lock-free `type_check_phase`. Consumed at call sites to decide, per
// argument, whether a reborrow persists past the call (a non-deriving argument's borrow is
// call-duration only). Every unknown falls to `AnyParam` — today's conservative behaviour — so
// an incomplete analysis is only ever a precision loss, never a soundness one.
//
//===----------------------------------------------------------------------===//
use crate::syntax::{Expr, Function, Statement, Type};

/// Which parameter(s) a function's returned reference roots in — the per-function summary that
/// makes the call-site reborrow decision per-argument instead of all-or-nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReturnProvenance {
    /// The function does not return a reference — no argument reborrow persists past the call.
    NotAReference,
    /// The returned reference roots in exactly the parameter slots in this bitset (bit `i` set ⇒
    /// derives from parameter `i`). Exact for functions with up to 32 parameters.
    FromParams(u32),
    /// Conservative top: the return may derive from any reference parameter. The default for every
    /// unknown — a call in the return position (v1), recursion, `unsafe`, more than 32 parameters,
    /// or a missing summary. Reproduces the pre-precision (#243) behaviour exactly.
    AnyParam,
    /// The returned reference roots in a function-local — already an `E4005` in the callee; callers
    /// treat it conservatively.
    Local,
}

impl ReturnProvenance {
    /// Does a call whose result is bound (and kept live) reborrow — and therefore persist a borrow
    /// on — the argument at position `slot`? `AnyParam`/`Local` say "assume yes" (conservative);
    /// `NotAReference` says "no argument reborrow outlives the call."
    pub fn includes(&self, slot: usize) -> bool {
        match self {
            ReturnProvenance::NotAReference => false,
            ReturnProvenance::FromParams(bits) => slot < 32 && (bits & (1u32 << slot)) != 0,
            ReturnProvenance::AnyParam | ReturnProvenance::Local => true,
        }
    }
}

/// Pack a return-provenance summary into the 3-bit slot-0 code used by the inline `TypeId` encoding
/// (#265; see [`crate::gid::TypeId::set_return_provenance`]). The inline field can name a single
/// parameter (`1..=4`) or fall back to the conservative top (`7`); a *multi-parameter* union or a
/// slot `>= 4` does not fit the fixed width and encodes as top. It therefore always yields a
/// **conservative refinement** of the summary — for every parameter the summary flags as an alias
/// source, the packed code flags it too — so an over-budget signature degrades gracefully to today's
/// all-arguments behaviour rather than becoming unsound. `NotAReference` (no reference returned)
/// encodes as `0`; `Local` matches the summary's conservative-top reading (it is an `E4005` in the
/// callee, so a caller never actually sees the reference, but the encoding must not read narrower
/// than [`ReturnProvenance::includes`]).
pub fn encode_return_provenance(prov: &ReturnProvenance) -> u8 {
    match prov {
        ReturnProvenance::NotAReference => 0,
        ReturnProvenance::AnyParam | ReturnProvenance::Local => 7,
        ReturnProvenance::FromParams(bits) => {
            // Exactly one parameter in slots 0..=3 fits the inline field; anything else (multiple
            // sources, or a slot beyond the fourth) degrades to the conservative top.
            if bits.count_ones() == 1 {
                let slot = bits.trailing_zeros();
                if slot < 4 {
                    return (slot + 1) as u8;
                }
            }
            7
        }
    }
}

/// Does the packed slot-0 code `prov` (see [`encode_return_provenance`]) treat parameter `slot` as a
/// possible alias source? The packed-form mirror of [`ReturnProvenance::includes`]: `0` excludes all,
/// `7` (and the reserved `5`/`6`) includes all, `1..=4` includes exactly parameter slot `0..=3`.
pub fn inline_prov_includes(prov: u8, slot: usize) -> bool {
    match prov {
        0 => false,
        1..=4 => slot == (prov as usize - 1),
        _ => true, // 7 = conservative top; 5/6 are reserved and treated conservatively.
    }
}

fn type_is_ref(ty: &Type) -> bool {
    matches!(ty, Type::Borrow { .. } | Type::Pointer(..) | Type::Ref(..))
}

/// Base variable and field path of a place expression (`x`, `x.f`, `x[i].g`) — the borrow
/// checker's `extract_base_and_path`, duplicated here so the pre-pass needs no `TypeChecker`.
fn base_of(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Identifier(id) => Some(id.name.to_string()),
        Expr::MemberAccess(ma) => base_of(&ma.base),
        Expr::IndexAccess(idx) => base_of(&idx.base),
        _ => None,
    }
}

/// Provenance of a single (reference-producing) expression within the callee's body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExprProv {
    /// Roots in exactly these parameter slots (bitset).
    Params(u32),
    /// Roots in a function-local slot or a temporary (would dangle if returned).
    Local,
    /// Unknown — a call result, or a reference we cannot classify. Conservative.
    Any,
    /// Not a reference — a value expression.
    NotRef,
}

struct ProvWalk<'a> {
    /// parameter name -> (slot index, is the parameter itself a reference type).
    params: std::collections::HashMap<&'a str, (usize, bool)>,
    /// provenance recorded for each reference-typed `let` binding, as we walk in order.
    locals: std::collections::HashMap<String, ExprProv>,
    /// union of every return site's parameter-slot bitset.
    acc_bits: u32,
    /// a return position we could not pin to specific parameters ⇒ the whole summary is `AnyParam`.
    saw_any: bool,
    /// at least one return rooted in a local (an `E4005` in the callee).
    saw_local: bool,
    /// at least one return rooted in a parameter.
    saw_param: bool,
    /// the body contains an `unsafe` block ⇒ a parameter-derived pointer may be stashed somewhere a
    /// structural walk cannot see, so the summary must be conservative.
    saw_unsafe: bool,
}

impl<'a> ProvWalk<'a> {
    fn expr_prov(&self, expr: &Expr) -> ExprProv {
        match expr {
            // `&base` / `&base.field`: a reborrow. Through a reference *parameter* it roots in
            // caller memory (that slot); of a by-value parameter, a local, or a temporary it is a
            // stack-local reference.
            Expr::Borrow(b) => match base_of(&b.expr) {
                Some(base) => {
                    if let Some((slot, is_ref)) = self.params.get(base.as_str()) {
                        if *is_ref {
                            ExprProv::Params(1u32 << slot)
                        } else {
                            ExprProv::Local
                        }
                    } else {
                        // `&local` / `&temp` — borrows storage that dies with the frame.
                        ExprProv::Local
                    }
                }
                None => ExprProv::Local,
            },
            // A bare reference: a reference parameter is that slot; a local binding carries whatever
            // provenance we recorded at its `let`; an unclassifiable reference is conservative.
            Expr::Identifier(id) => {
                if let Some((slot, is_ref)) = self.params.get(id.name.as_ref()) {
                    if *is_ref {
                        ExprProv::Params(1u32 << slot)
                    } else {
                        ExprProv::NotRef
                    }
                } else if let Some(p) = self.locals.get(id.name.as_ref()) {
                    *p
                } else {
                    ExprProv::Any
                }
            }
            // A call in the return position: v1 is conservative (cross-function summaries are a
            // deferred fixpoint). A correct callee derives its result from its reference inputs, so
            // this is sound as `Any`.
            Expr::FunctionCall(_) | Expr::MethodCall(_) => ExprProv::Any,
            // Value expressions carry no reference provenance.
            Expr::Number(_) | Expr::StringLiteral(_) => ExprProv::NotRef,
            _ => ExprProv::Any,
        }
    }

    fn note_return(&mut self, expr: &Expr) {
        match self.expr_prov(expr) {
            ExprProv::Params(bits) => {
                self.acc_bits |= bits;
                self.saw_param = true;
            }
            ExprProv::Local => self.saw_local = true,
            // Unknown or a non-reference in a reference-returning function ⇒ conservative.
            ExprProv::Any | ExprProv::NotRef => self.saw_any = true,
        }
    }

    fn walk_block(&mut self, stmts: &'a [Statement]) {
        for s in stmts {
            self.walk_stmt(s);
        }
    }

    fn walk_stmt(&mut self, stmt: &'a Statement) {
        match stmt {
            Statement::Return(r) => {
                self.note_return(&r.expr);
                self.walk_expr(&r.expr);
            }
            Statement::LetDecl(l) => {
                // Record a reference binding's provenance so a later `return name` resolves through
                // it. Non-reference bindings record `NotRef` (harmless).
                let prov = self.expr_prov(&l.expr);
                self.locals.insert(l.name.to_string(), prov);
                self.walk_expr(&l.expr);
            }
            Statement::ExprStmt(e) => self.walk_expr(&e.expr),
            Statement::Assign(a) => {
                self.walk_expr(&a.lhs);
                self.walk_expr(&a.rhs);
            }
            Statement::CompoundAssign(a) => {
                self.walk_expr(&a.lhs);
                self.walk_expr(&a.rhs);
            }
            Statement::ForLoop(f) => self.walk_block(&f.body),
            Statement::Loop(l) => self.walk_block(&l.body),
            Statement::Assert(a) => self.walk_expr(&a.expr),
            _ => {}
        }
    }

    /// Descend into control-flow *expressions* so returns nested in `if`/`match` branches (and
    /// `unsafe` blocks) are seen.
    fn walk_expr(&mut self, expr: &'a Expr) {
        match expr {
            Expr::If(i) => {
                self.walk_expr(&i.cond);
                self.walk_block(&i.then_block);
                if let Some(eb) = &i.else_block {
                    self.walk_block(eb);
                }
            }
            Expr::Match(m) => {
                self.walk_expr(&m.expr);
                for arm in &m.arms {
                    self.walk_block(&arm.body);
                }
            }
            Expr::UnsafeBlock(_) => {
                // A parameter-derived pointer can be stashed through raw-pointer stores inside an
                // `unsafe` block where a structural walk cannot follow it. Refuse to narrow.
                self.saw_unsafe = true;
            }
            Expr::ComptimeBlock(c) => {
                self.walk_block(&c.stmts);
                if let Some(r) = &c.ret {
                    self.walk_expr(r);
                }
            }
            _ => {}
        }
    }
}

/// Compute the return-provenance summary for `func` (#243). Pure and body-only — safe to call from
/// any worker on any function, in any order.
pub fn compute_return_provenance(func: &Function) -> ReturnProvenance {
    if !type_is_ref(&func.return_type) {
        return ReturnProvenance::NotAReference;
    }
    // The exact bitset holds 32 parameters; wider signatures fall back to conservative.
    if func.params.len() > 32 {
        return ReturnProvenance::AnyParam;
    }

    let mut walk = ProvWalk {
        params: func
            .params
            .iter()
            .enumerate()
            .map(|(i, (n, t))| (n.as_ref(), (i, type_is_ref(t))))
            .collect(),
        locals: std::collections::HashMap::new(),
        acc_bits: 0,
        saw_any: false,
        saw_local: false,
        saw_param: false,
        saw_unsafe: false,
    };
    walk.walk_block(&func.body);

    if walk.saw_unsafe || walk.saw_any {
        return ReturnProvenance::AnyParam;
    }
    if walk.saw_param {
        // Some return derived from a parameter; `acc_bits` is the exact union. (Local return paths
        // are E4005s in the callee and contribute no alias source.)
        return ReturnProvenance::FromParams(walk.acc_bits);
    }
    if walk.saw_local {
        return ReturnProvenance::Local;
    }
    // A reference-returning function with no classifiable return (e.g. an `extern` with no body):
    // conservative.
    ReturnProvenance::AnyParam
}

/// Build the name → summary map for a set of functions whose bodies are present. Functions with an
/// empty body (signature-stripped, or genuinely bodiless like `extern`) are skipped, so the map
/// carries an entry only where the summary is meaningful; a missing lookup means `AnyParam`.
pub fn provenance_map<'a>(
    funcs: impl Iterator<Item = &'a Function>,
) -> std::collections::HashMap<crate::symbol::Symbol, ReturnProvenance> {
    let mut map = std::collections::HashMap::new();
    for f in funcs {
        if f.body.is_empty() {
            continue;
        }
        map.insert(f.name.clone(), compute_return_provenance(f));
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    fn summarize(src: &str, fname: &str) -> ReturnProvenance {
        let mut lexer = crate::lexer::Lexer::new(src);
        let tokens = lexer.tokenize();
        let mut parser = crate::parser::Parser::new(&tokens, src);
        let program = parser.parse().expect("parses");
        let f = program
            .functions
            .iter()
            .find(|f| f.name.as_ref() == fname)
            .expect("function present");
        compute_return_provenance(f)
    }

    #[test]
    fn single_source_names_only_its_param() {
        // `pick` returns from `b` (slot 1) — the signature Rust cannot elide.
        let src = "fn pick(a : &i32, b : &i32) -> &i32 { return b; }";
        assert_eq!(summarize(src, "pick"), ReturnProvenance::FromParams(0b10));
    }

    #[test]
    fn reborrow_through_param_field_names_that_param() {
        let src = "struct M { slot : i32 } fn f(a : &M, b : &M) -> &i32 { return &b.slot; }";
        assert_eq!(summarize(src, "f"), ReturnProvenance::FromParams(0b10));
    }

    #[test]
    fn value_return_is_not_a_reference() {
        let src = "fn f(a : &i32) -> i32 { return 3; }";
        assert_eq!(summarize(src, "f"), ReturnProvenance::NotAReference);
    }

    #[test]
    fn multi_source_unions_the_params() {
        // params c=0, a=1, b=2 -> {1, 2}
        let src =
            "fn choose(c : i32, a : &i32, b : &i32) -> &i32 { if c == 0 { return a; } return b; }";
        assert_eq!(
            summarize(src, "choose"),
            ReturnProvenance::FromParams(0b110)
        );
    }

    #[test]
    fn transitive_local_binding_is_threaded() {
        let src =
            "struct M { slot : i32 } fn via(a : &M, b : &M) -> &i32 { let t = &b.slot; return t; }";
        assert_eq!(summarize(src, "via"), ReturnProvenance::FromParams(0b10));
    }

    #[test]
    fn returning_a_local_reference_is_local() {
        let src = "fn dangle() -> &i32 { let x = 5; return &x; }";
        assert_eq!(summarize(src, "dangle"), ReturnProvenance::Local);
    }

    #[test]
    fn unsafe_body_falls_back_to_conservative() {
        let src =
            "struct M { slot : i32 } fn u(a : &M, b : &M) -> &i32 { unsafe { let p = 0; } return &b.slot; }";
        assert_eq!(summarize(src, "u"), ReturnProvenance::AnyParam);
    }

    #[test]
    fn call_in_return_position_is_conservative_in_v1() {
        let src = "fn probe(m : &i32) -> &i32 { return m; } fn wrap(a : &i32, b : &i32) -> &i32 { return probe(b); }";
        assert_eq!(summarize(src, "wrap"), ReturnProvenance::AnyParam);
    }

    /// The inline slot-0 code (#265) names a single in-budget parameter as `slot + 1` and degrades
    /// every wider or unknown case to the conservative top; a non-reference return encodes as none.
    #[test]
    fn encode_maps_single_param_and_degrades_the_rest() {
        assert_eq!(
            encode_return_provenance(&ReturnProvenance::FromParams(0b0001)),
            1
        );
        assert_eq!(
            encode_return_provenance(&ReturnProvenance::FromParams(0b0010)),
            2
        );
        assert_eq!(
            encode_return_provenance(&ReturnProvenance::FromParams(0b0100)),
            3
        );
        assert_eq!(
            encode_return_provenance(&ReturnProvenance::FromParams(0b1000)),
            4
        );
        // Over budget -> conservative top.
        assert_eq!(
            encode_return_provenance(&ReturnProvenance::FromParams(0b0110)),
            7
        ); // two sources
        assert_eq!(
            encode_return_provenance(&ReturnProvenance::FromParams(1 << 4)),
            7
        ); // slot >= 4
        assert_eq!(encode_return_provenance(&ReturnProvenance::AnyParam), 7);
        assert_eq!(encode_return_provenance(&ReturnProvenance::Local), 7);
        // No reference returned -> no alias source.
        assert_eq!(
            encode_return_provenance(&ReturnProvenance::NotAReference),
            0
        );
    }

    /// The packed code is always a *conservative refinement* of the summary it replaces: for every
    /// parameter the summary flags as an alias source, the packed code flags it too. This is the
    /// soundness property that makes the fixed-width inline encoding safe — it may lose precision (a
    /// multi-source union widens to "any"), never soundness (#265).
    #[test]
    fn inline_code_conservatively_refines_the_summary() {
        let cases = [
            ReturnProvenance::NotAReference,
            ReturnProvenance::Local,
            ReturnProvenance::AnyParam,
            ReturnProvenance::FromParams(0b0001),
            ReturnProvenance::FromParams(0b1000),
            ReturnProvenance::FromParams(0b0110), // multi-source -> widens to top
            ReturnProvenance::FromParams(1 << 5), // out-of-budget slot -> top
        ];
        for prov in cases {
            let code = encode_return_provenance(&prov);
            for slot in 0..32 {
                assert!(
                    !prov.includes(slot) || inline_prov_includes(code, slot),
                    "summary {prov:?} flags slot {slot} but the packed code {code} does not"
                );
            }
        }
    }
}
