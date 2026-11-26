//===- purity.rs - Vx Compiler --------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Whether a function writes only plain variables in its own frame, and so can run in every
// pass of a `spawn` loop at once. Stricter than "writes nothing that a caller sees": a callee
// that only *reads* another pass's row still races that pass's write, and a call shows only its
// arguments, not its body. A function counts only when
//
//   - no parameter can reach memory at all,
//   - the body writes only plain names in its own frame, and
//   - every call in it is to another function that counts.
//
// So `fn exp_helper(x : f32) -> f32` that wraps `math.exp` in an `mlir!` block counts, and one
// that takes `&mut Tensor` does not.
//
// This is a fact about a body, so like the return provenance in `provenance.rs` it is read from
// the syntax of every body before the checker checks any body, and then frozen. The checker
// only reads it, so bodies are still checked in parallel.
//
//===----------------------------------------------------------------------===//

use crate::hir::provenance::Decls;
use crate::symbol::Symbol;
use crate::syntax::{Expr, Function, Statement};
use crate::syntax::{LetDeclStmt, Type};
use std::collections::HashSet;

/// Whether every parameter of `f` is a value that cannot reach memory, and its result is one
/// too. A type parameter counts as reaching memory: at this point nothing says what it becomes.
/// A parameter whose type has a `Drop` impl does not count either: dropping it runs that impl,
/// which is code this walk cannot see.
fn signature_is_pure(f: &Function, decls: &Decls) -> bool {
    f.params
        .iter()
        .all(|(_, ty)| !decls.reaches_memory(ty, 0) && !decls.has_drop(ty, 0))
        && !decls.reaches_memory(&f.return_type, 0)
}

/// Whether `f` writes only plain variables in its own frame, reads nothing it could share, and
/// calls only functions that `callee_pure` accepts.
///
/// `f` may be an unchecked body (at the freeze point) or a checked instance (whose method calls
/// are already calls by name). An empty body is an `extern` or a signature-stripped copy, and
/// nothing can be shown about it.
pub fn writes_only_its_frame(
    f: &Function,
    decls: &Decls,
    callee_pure: &dyn Fn(&str) -> bool,
) -> bool {
    if f.body.is_empty() || !signature_is_pure(f, decls) {
        return false;
    }
    let walk = Walk {
        decls,
        callee_pure,
        inside_unsafe: false,
    };
    walk.stmts(&f.body)
}

/// The declarations that an `mlir!` block names as its input types, checked as text: a block
/// that takes a memory value can write what the caller sees.
fn mlir_input_reaches_memory(ty: &str) -> bool {
    ty.contains("memref") || ty.contains("tensor<") || ty.contains("ptr") || ty.starts_with('!')
}

#[derive(Clone, Copy)]
struct Walk<'a> {
    decls: &'a Decls<'a>,
    callee_pure: &'a dyn Fn(&str) -> bool,
    /// Inside an `unsafe { }` block, where an `mlir!` block may do anything the program says.
    inside_unsafe: bool,
}

impl Walk<'_> {
    fn stmts(&self, stmts: &[Statement]) -> bool {
        stmts.iter().all(|s| self.stmt(s))
    }

    fn stmt(&self, s: &Statement) -> bool {
        match s {
            Statement::LetDecl(l) => !self.binds_a_dropped_value(l) && self.expr(&l.expr),
            Statement::Return(r) => r.expr.as_ref().is_none_or(|e| self.expr(e)),
            Statement::ExprStmt(e) => self.expr(&e.expr),
            Statement::ForLoop(f) => self.expr(&f.iterable) && self.stmts(&f.body),
            Statement::Assign(a) => self.plain_target(&a.lhs) && self.expr(&a.rhs),
            Statement::CompoundAssign(a) => self.plain_target(&a.lhs) && self.expr(&a.rhs),
            Statement::Assert(a) => self.expr(&a.expr),
            Statement::Loop(l) => self.stmts(&l.body),
            Statement::Break(_) | Statement::Continue(_) => true,
            // A macro has not expanded here, so its body is unknown. A drop frees memory that
            // the function owns, which is a write the caller can see.
            Statement::MacroCall(_) | Statement::Drop(_) | Statement::Error(_) => false,
        }
    }

    /// Whether this `let` binds a value whose type has a `Drop` impl. Dropping it at the end of
    /// the block runs that impl, which is code this walk cannot see. A `let` with no written type
    /// is judged by what builds the value: a struct or an enum variant the declarations know.
    fn binds_a_dropped_value(&self, l: &LetDeclStmt) -> bool {
        if l.ty_ann
            .as_ref()
            .is_some_and(|ty| self.decls.has_drop(ty, 0))
        {
            return true;
        }
        let built = match &l.expr {
            Expr::StructInit(s) => Some(Type::Struct(s.name.clone(), s.type_id)),
            Expr::EnumVariant(v) => Some(Type::Enum(v.enum_name.clone(), None)),
            _ => None,
        };
        built.is_some_and(|ty| self.decls.has_drop(&ty, 0))
    }

    /// Whether an assignment writes a name of the function's own frame. An element, a field or
    /// a place behind a pointer is memory that another pass may reach.
    fn plain_target(&self, lhs: &Expr) -> bool {
        matches!(lhs, Expr::Identifier(_))
    }

    fn exprs(&self, es: &[Expr]) -> bool {
        es.iter().all(|e| self.expr(e))
    }

    fn expr(&self, e: &Expr) -> bool {
        match e {
            Expr::Identifier(_)
            | Expr::Number(_)
            | Expr::StringLiteral(_)
            | Expr::MemorySpace(_)
            | Expr::Topology(_)
            | Expr::SizeOf(_) => true,
            Expr::EnumVariant(v) => v.payload.as_ref().is_none_or(|p| self.exprs(p)),
            Expr::Array(a) => self.exprs(&a.elements),
            Expr::StructInit(s) => s.fields.iter().all(|(_, e)| self.expr(e)),
            Expr::MemberAccess(m) => self.expr(&m.base),
            Expr::IndexAccess(i) => self.expr(&i.base) && self.expr(&i.index),
            Expr::BinaryOp(b) => self.expr(&b.lhs) && self.expr(&b.rhs),
            Expr::RelationalOp(r) => self.expr(&r.lhs) && self.expr(&r.rhs),
            Expr::LogicalOp(l) => self.expr(&l.lhs) && self.expr(&l.rhs),
            Expr::UnaryOp(u) => self.expr(&u.expr),
            Expr::AsCast(c) => self.expr(&c.expr),
            Expr::Range(r) => self.expr(&r.start) && self.expr(&r.end),
            Expr::If(i) => {
                self.expr(&i.cond)
                    && self.stmts(&i.then_block)
                    && i.else_block.as_ref().is_none_or(|b| self.stmts(b))
            }
            Expr::Match(m) => self.expr(&m.expr) && m.arms.iter().all(|arm| self.stmts(&arm.body)),
            // A block is no more than its statements, and inside `unsafe { }` the walk knows that
            // an `mlir!` block there is not one the compiler verified (see `InlineMlir`).
            Expr::UnsafeBlock(b) => Walk {
                inside_unsafe: true,
                ..*self
            }
            .stmts(&b.stmts),
            Expr::ComptimeBlock(b) => self.stmts(&b.stmts),
            Expr::FunctionCall(fc) => (self.callee_pure)(fc.name.as_ref()) && self.exprs(&fc.args),
            // An `mlir!` block is opaque, but a block that takes no memory value and names no
            // clobber cannot write memory that a caller sees: the program has no globals. Inside
            // `unsafe { }` the compiler does not verify the block, so it counts only when it says
            // `ensures : [writes_only()]`, which is the same rule the contract check applies.
            Expr::InlineMlir(m) => {
                (!self.inside_unsafe
                    || m.ensures
                        .iter()
                        .any(crate::hir::check::atomic_contracts::is_writes_nothing))
                    && m.clobbers.is_empty()
                    && m.inputs
                        .iter()
                        .all(|(_, e, ty)| !mlir_input_reaches_memory(ty) && self.expr(e))
                    && m.returns
                        .as_ref()
                        .is_none_or(|ty| !self.decls.reaches_memory(ty, 0))
            }
            // A method call has not been rewritten into a call by name at the freeze point, so
            // the walk cannot say which function it is. At the freeze point nothing that reads
            // a body has run, so refusing here costs only precision.
            Expr::MethodCall(_)
            | Expr::Borrow(_)
            | Expr::Dereference(_)
            | Expr::Transfer(_)
            | Expr::TransferPredicate(_)
            | Expr::IndirectCall(_)
            | Expr::Grad(_)
            | Expr::Vjp(_)
            | Expr::Jvp(_)
            | Expr::SpawnOn(_)
            | Expr::TaskBlock(_)
            | Expr::VecMacro(_)
            | Expr::Closure(_)
            | Expr::MacroCall(_)
            | Expr::Print(_)
            | Expr::Println(_) => false,
        }
    }
}

/// Every function that writes only its own frame, by name, from the syntax of `funcs`. `pure`
/// starts empty and grows: a function joins only once every call in it already has, so a
/// recursive function never does. That is conservative and always sound.
///
/// `builtin` accepts the scalar intrinsics that the compiler already trusts without reading a
/// body, such as `f32::sqrt`.
pub fn functions_writing_only_their_frame<'a>(
    funcs: impl Iterator<Item = (Symbol, &'a Function)>,
    decls: &Decls,
    builtin: &dyn Fn(&str) -> bool,
) -> HashSet<Symbol> {
    // Two functions may share a name. A name counts only when every function under it counts, so
    // a name that any of them fails never joins. The key is the name a call site uses, which is
    // the mangled one for a method.
    let mut by_name: std::collections::HashMap<Symbol, Vec<&Function>> =
        std::collections::HashMap::new();
    for (key, f) in funcs {
        if !f.body.is_empty() {
            by_name.entry(key).or_default().push(f);
        }
    }
    let mut pure: HashSet<Symbol> = HashSet::new();
    loop {
        let mut changed = false;
        for (name, group) in &by_name {
            if pure.contains(name) {
                continue;
            }
            let callee = |callee: &str| pure.contains(callee) || builtin(callee);
            if group
                .iter()
                .all(|f| writes_only_its_frame(f, decls, &callee))
            {
                pure.insert(name.clone());
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    pure
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parse, then expand macros, as the compiler does before it builds the environment: an
    /// `mlir!` block is a `MacroCall` in the parse tree and an `InlineMlir` expression after.
    fn parse(src: &str) -> crate::syntax::Program {
        let mut lexer = crate::lexer::Lexer::new(src);
        let tokens = lexer.tokenize();
        let mut parser = crate::parser::Parser::new(&tokens, src);
        let mut program = parser.parse().expect("parses");
        let macros = crate::parser::MacroTable::new(std::iter::once(&program));
        let defaults = crate::resolver::collect_trait_defaults(std::iter::once(&program));
        crate::parser::MacroExpander::new(&macros, &defaults)
            .expand_module(&mut program)
            .expect("macros expand");
        program
    }

    fn pure_names(src: &str) -> HashSet<Symbol> {
        let program = parse(src);
        let decls = Decls::new(&program.structs, &program.enums);
        let funcs: Vec<_> = program
            .functions
            .iter()
            .map(|f| (f.name.clone(), f))
            .collect();
        functions_writing_only_their_frame(funcs.into_iter(), &decls, &|_| false)
    }

    #[test]
    fn a_scalar_function_that_writes_its_own_variables_counts() {
        let pure = pure_names("fn bump(x : f32) -> f32 { let mut y = x; y = y + 1.0; return y; }");
        assert!(pure.contains("bump"));
    }

    #[test]
    fn a_parameter_that_reaches_memory_does_not_count() {
        // The body only reads the tensor, so what refuses this function is the parameter
        // alone: a pass could still read another pass's row through it.
        let pure =
            pure_names("fn peek(t : &Tensor<f32, [4], Memory::CPU_DRAM>) -> f32 { return t[0]; }");
        assert!(!pure.contains("peek"));
    }

    #[test]
    fn a_call_follows_its_callee() {
        let pure = pure_names(
            "fn inner(x : f32) -> f32 { return x + 1.0; }
             fn outer(x : f32) -> f32 { return inner(x); }
             fn writer(x : f32) -> f32 { let mut s = x; s = inner(s); return s; }",
        );
        assert!(pure.contains("inner"));
        assert!(pure.contains("outer"));
        assert!(pure.contains("writer"));
    }

    #[test]
    fn a_function_that_calls_one_that_writes_memory_does_not_count() {
        let pure = pure_names(
            "fn writer(t : &mut Tensor<f32, [4], Memory::CPU_DRAM>) -> void { t[0] = 1.0; }
             fn through(t : &mut Tensor<f32, [4], Memory::CPU_DRAM>) -> void { writer(t); }
             fn caller() -> f32 { return 1.0; }",
        );
        // `caller` calls nothing, so it counts; a function that reaches the tensor or calls one
        // that does, does not.
        assert!(!pure.contains("writer"));
        assert!(!pure.contains("through"));
        assert!(pure.contains("caller"));
    }

    #[test]
    fn a_write_through_an_element_stops_a_function() {
        let pure = pure_names(
            "fn via_local() -> f32 {
                 let mut a = [1.0, 2.0];
                 a[0] = 3.0;
                 return a[0];
             }",
        );
        // Writing an element of a local array is memory, not a plain name.
        assert!(!pure.contains("via_local"));
    }

    #[test]
    fn a_recursive_function_is_not_proven() {
        let pure =
            pure_names("fn down(n : i32) -> i32 { if n == 0 { return 0; } return down(n - 1); }");
        assert!(!pure.contains("down"));
    }

    #[test]
    fn an_mlir_block_inside_unsafe_counts_only_with_the_clause() {
        // The compiler does not verify a block inside `unsafe { }`, so one that stores through a
        // pointer it makes up can write anywhere, and the loop must not split.
        let without = pure_names(
            "fn poke(x : f32) -> f32 {
                 unsafe { mlir!( inputs : (), dialects : [\"llvm\"]) {
                   %p = arith.constant 0 : i64
                   %q = llvm.inttoptr %p : i64 to !llvm.ptr
                   %one = arith.constant 1.0 : f32
                   llvm.store %one, %q : f32, !llvm.ptr
                   macro.yield
                 } };
                 return x;
             }",
        );
        assert!(!without.contains("poke"));
        let with = pure_names(
            "fn fence(x : f32) -> f32 {
                 unsafe { mlir!( inputs : (), dialects : [\"llvm\"], ensures : [writes_only()]) {
                   llvm.fence release
                   macro.yield
                 } };
                 return x;
             }",
        );
        // The clause says the block writes no memory, which is what the contract check trusts.
        assert!(with.contains("fence"));
    }

    #[test]
    fn a_local_whose_type_has_a_drop_does_not_count() {
        let program = parse(
            "struct Noisy { n : i32 }
             impl Noisy { fn drop(self : &mut Noisy) -> void { println!(\"{}\", self.n); } }
             fn noisy(x : i32) -> i32 { let s = Noisy { n : x }; return x; }",
        );
        let decls = Decls::new(&program.structs, &program.enums).with_drop_impls(&program.impls);
        let funcs: Vec<_> = program
            .functions
            .iter()
            .map(|f| (f.name.clone(), f))
            .collect();
        let pure = functions_writing_only_their_frame(funcs.into_iter(), &decls, &|_| false);
        // Dropping `s` runs `drop`, which prints.
        assert!(!pure.contains("noisy"));
    }

    #[test]
    fn two_functions_under_one_name_count_only_when_both_do() {
        let program = parse(
            "fn f(x : f32) -> f32 { return x; }
             fn g(t : &mut Tensor<f32, [4], Memory::CPU_DRAM>) -> void { t[0] = 1.0; }",
        );
        let decls = Decls::new(&program.structs, &program.enums);
        let mut second = program.functions[1].clone();
        second.name = "f".into();
        // The second function is the impure body, under the same name as the first.
        let key: Symbol = "f".into();
        let pure = functions_writing_only_their_frame(
            [(key.clone(), &program.functions[0]), (key, &second)].into_iter(),
            &decls,
            &|_| false,
        );
        assert!(!pure.contains("f"));
    }
}
