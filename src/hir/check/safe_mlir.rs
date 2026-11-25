//===- safe_mlir.rs - Vx Compiler ----------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// An `mlir!` block outside `unsafe` must be one that the compiler can verify as safe: each
// operation is on a list of operations that give a defined result for every input, and the
// block takes and gives only numbers. Any other block goes in `unsafe { }`.
//
//===----------------------------------------------------------------------===//

use crate::hir::TypeChecker;
use crate::syntax::*;
use melior::ir::attribute::TypeAttribute;
use melior::ir::operation::{OperationLike, WalkOrder, WalkResult};
use melior::ir::r#type::{FunctionType, TypeLike};
use std::cell::RefCell;
use std::collections::HashMap;

/// The operations that give a defined result for every input when they carry no `overflow` or
/// `fastmath` flags. Division and remainder of integers and shifts are not on the list: a zero
/// divisor or a large shift gives undefined behavior or poison. A conversion from a float to an
/// integer gives poison when the value does not fit.
const VERIFIED_OPERATIONS: &[&str] = &[
    "arith.addf",
    "arith.addi",
    "arith.andi",
    "arith.bitcast",
    "arith.cmpf",
    "arith.cmpi",
    "arith.constant",
    "arith.divf",
    "arith.extf",
    "arith.extsi",
    "arith.extui",
    "arith.maximumf",
    "arith.maxsi",
    "arith.maxui",
    "arith.minimumf",
    "arith.minsi",
    "arith.minui",
    "arith.mulf",
    "arith.muli",
    "arith.mulsi_extended",
    "arith.mului_extended",
    "arith.negf",
    "arith.ori",
    "arith.select",
    "arith.sitofp",
    "arith.subf",
    "arith.subi",
    "arith.truncf",
    "arith.trunci",
    "arith.uitofp",
    "arith.xori",
    "math.absf",
    "math.ceil",
    "math.copysign",
    "math.cos",
    "math.ctlz",
    "math.ctpop",
    "math.cttz",
    "math.exp",
    "math.exp2",
    "math.floor",
    "math.fma",
    "math.log",
    "math.log2",
    "math.round",
    "math.roundeven",
    "math.rsqrt",
    "math.sin",
    "math.sqrt",
    "math.tanh",
    "math.trunc",
    "func.return",
];

/// The reason for a block that does not parse with only the `arith` and `math` dialects. Code
/// generation can still parse it, with an operation of another dialect, so it is not verified.
const NOT_VALID_MLIR: &str = "does not parse as MLIR of the `arith` and `math` dialects";

thread_local! {
    /// A context with only the dialects of the list, so that an operation of any other dialect
    /// does not parse.
    static CONTEXT: melior::Context = {
        let context = melior::Context::new();
        // SAFETY: each handle comes from the MLIR that vxc links.
        unsafe {
            for handle in [
                mlir_sys::mlirGetDialectHandle__arith__(),
                mlir_sys::mlirGetDialectHandle__math__(),
                mlir_sys::mlirGetDialectHandle__func__(),
            ] {
                melior::dialect::DialectHandle::from_raw(handle).load_dialect(&context);
            }
        }
        context
    };
    /// The answer for each wrapper text, because the checker reads a generic function once for
    /// each type that a call gives it.
    static ANSWERS: RefCell<HashMap<String, Option<String>>> = RefCell::new(HashMap::new());
}

/// Why the compiler cannot verify that `m` is safe, as words that follow "the block",
/// or `None` for a block that it verifies.
pub(crate) fn unverified(m: &InlineMlirExpr) -> Option<String> {
    if !m.clobbers.is_empty() {
        return Some("writes the variables of its `clobbers` list".into());
    }
    if m.threads.is_some() {
        return Some("runs on each thread of a threadgroup (`threads`)".into());
    }
    if !m.ensures.is_empty() {
        return Some("has `ensures` clauses, which the compiler trusts".into());
    }
    if let Some(d) = m
        .dialects
        .iter()
        .find(|d| !["arith", "math"].contains(&d.as_str()))
    {
        let op = m.block_str.lines().find_map(|line| {
            crate::hir::check::mlir_dialects::line_ops(line)
                .find_map(|(op, dialect)| (dialect == d).then(|| op.to_string()))
        });
        return Some(match op {
            Some(op) => format!("uses `{op}`"),
            None => format!("names the dialect `{d}`"),
        });
    }
    let result = match &m.returns {
        None => None,
        Some(Type::Scalar(e)) => Some(crate::mlir_ty::mlir_scalar(e)?.to_string()),
        Some(Type::Simd(e, n)) => Some(crate::mlir_ty::mlir_vector(e, *n)?),
        Some(t) => {
            return Some(format!(
                "gives a value of type `{t}`, which is not a number"
            ))
        }
    };
    let params: Vec<String> = m
        .inputs
        .iter()
        .map(|(n, _, t)| format!("{n}: {t}"))
        .collect();
    let text = format!(
        "func.func @block({}){} {{\n{}\n}}",
        params.join(", "),
        result.map(|r| format!(" -> {r}")).unwrap_or_default(),
        m.block_str.replace("macro.yield", "return"),
    );
    if let Some(answer) = ANSWERS.with(|a| a.borrow().get(&text).cloned()) {
        return answer;
    }
    let answer = CONTEXT.with(|context| verify(context, &text));
    ANSWERS.with(|a| a.borrow_mut().insert(text, answer.clone()));
    answer
}

/// Why the function in `text` is not one that the compiler verifies, or `None`.
fn verify(context: &melior::Context, text: &str) -> Option<String> {
    let errors = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let seen = errors.clone();
    let id = context.attach_diagnostic_handler(move |d| {
        seen.borrow_mut().push(d.to_string());
        true
    });
    let parsed = melior::ir::Module::parse(context, text);
    context.detach_diagnostic_handler(id);
    let Some(module) = parsed else {
        let message = errors.borrow().first().map(|m| m.trim().to_string());
        let message = message.unwrap_or_default();
        return Some(format!("{NOT_VALID_MLIR}: {message}"));
    };
    let mut why = None;
    module.as_operation().walk(WalkOrder::PreOrder, |op| {
        let name = op.name().as_string_ref().as_str().unwrap_or("").to_string();
        if name == "builtin.module" {
            return WalkResult::Advance;
        }
        if name == "func.func" {
            let ty = op
                .attribute("function_type")
                .ok()
                .and_then(|a| TypeAttribute::try_from(a).ok())
                .and_then(|a| FunctionType::try_from(a.value()).ok());
            let Some(ty) = ty else {
                why = Some(NOT_VALID_MLIR.into());
                return WalkResult::Interrupt;
            };
            let inputs = (0..ty.input_count()).filter_map(|i| ty.input(i).ok());
            let results = (0..ty.result_count()).filter_map(|i| ty.result(i).ok());
            if let Some(t) = inputs.chain(results).find(|t| !is_number(*t)) {
                why = Some(format!("takes or gives a `{t}`, which is not a number"));
                return WalkResult::Interrupt;
            }
            return WalkResult::Advance;
        }
        if !VERIFIED_OPERATIONS.contains(&name.as_str()) {
            why = Some(format!("uses `{name}`"));
            return WalkResult::Interrupt;
        }
        for (key, attr) in op.attributes() {
            let key = key.as_string_ref().as_str().unwrap_or("").to_string();
            let flag = match key.as_str() {
                "overflowFlags" => "an `overflow` flag",
                "fastmath" => "a `fastmath` flag",
                _ => continue,
            };
            if !attr.to_string().ends_with("<none>") {
                why = Some(format!("uses `{name}` with {flag}, which can give poison"));
                return WalkResult::Interrupt;
            }
        }
        WalkResult::Advance
    });
    why
}

/// Whether `t` is an integer, a float, an `index` or a vector of them.
fn is_number(t: melior::ir::Type) -> bool {
    let t = if t.is_vector() {
        // SAFETY: a vector type is a shaped type, and its element type lives in the context.
        unsafe { melior::ir::Type::from_raw(mlir_sys::mlirShapedTypeGetElementType(t.to_raw())) }
    } else {
        t
    };
    t.is_integer() || t.is_float() || t.is_index()
}

impl TypeChecker<'_> {
    /// E8016: an `mlir!` block outside `unsafe` that the compiler cannot verify as safe. Inside
    /// `unsafe`, such a block is what the `unsafe` is for (W1010).
    pub(crate) fn check_mlir_is_verified(&mut self, m: &InlineMlirExpr) {
        if self.speculating {
            return;
        }
        let Some(why) = unverified(m) else {
            return;
        };
        if self.in_unsafe_block || self.in_unsafe_fn {
            self.unsafe_op_seen = true;
            return;
        }
        self.errors.error_with_code(
            crate::diagnostic::DiagnosticCode::E8016,
            format!(
                "the compiler cannot verify that this `mlir!` block is safe: the block {why}. \
                 Put the block in `unsafe {{ }}`"
            ),
            Some(crate::diagnostic::SourceSpan::from_ast_span(&m.span)),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each `mlir!` block of `program` outside `unsafe` that the compiler cannot verify, with why.
    fn unverified_blocks(program: &Program) -> Vec<(usize, String)> {
        let methods = program.impls.iter().flat_map(|ib| ib.methods.iter());
        let mut found = Vec::new();
        for f in program
            .functions
            .iter()
            .chain(methods)
            .filter(|f| !f.is_unsafe)
        {
            let mut inside = std::collections::HashSet::new();
            crate::syntax::visit_exprs(&f.body, &mut |e| {
                if let Expr::UnsafeBlock(_) = e {
                    crate::syntax::visit_expr(e, &mut |e| {
                        if let Expr::InlineMlir(m) = e {
                            inside.insert(m.span);
                        }
                    });
                }
            });
            crate::syntax::visit_exprs(&f.body, &mut |e| {
                if let Expr::InlineMlir(m) = e {
                    if !inside.contains(&m.span) {
                        if let Some(why) = unverified(m) {
                            found.push((m.span.line, why));
                        }
                    }
                }
            });
        }
        found
    }

    /// The type checker reads only the library functions that a program uses, so this test reads
    /// every `mlir!` block of `core` and `std`.
    #[test]
    fn each_library_block_outside_unsafe_is_verified() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("stdlib");
        let mut files = Vec::new();
        let mut dirs = vec![root.join("core"), root.join("std")];
        while let Some(dir) = dirs.pop() {
            for entry in std::fs::read_dir(&dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    dirs.push(path);
                } else if path.extension().is_some_and(|e| e == "vx") {
                    files.push(path);
                }
            }
        }
        let mut bad = Vec::new();
        for path in files {
            let text = std::fs::read_to_string(&path).unwrap();
            let tokens = crate::lexer::Lexer::new(&text).tokenize();
            let mut program = crate::parser::Parser::new(&tokens, &text).parse().unwrap();
            let macros = crate::parser::MacroTable::new(std::iter::once(&program));
            let defaults = crate::resolver::collect_trait_defaults(std::iter::once(&program));
            crate::parser::MacroExpander::new(&macros, &defaults)
                .expand_module(&mut program)
                .unwrap();
            for (line, why) in unverified_blocks(&program) {
                bad.push(format!("{}:{line}: the block {why}", path.display()));
            }
        }
        assert!(
            bad.is_empty(),
            "blocks that need `unsafe`:\n{}",
            bad.join("\n")
        );
    }
}
