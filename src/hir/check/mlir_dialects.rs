//===- mlir_dialects.rs - Vx Compiler ------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The `dialects` list of an `mlir!` block names the dialects that its operations come from.
//
//===----------------------------------------------------------------------===//

use crate::hir::TypeChecker;
use crate::syntax::*;

/// The dialects that MLIR registers (`mlir-opt --show-dialects`, LLVM 22), and `vx`, the
/// compiler's own. A test checks the list against the MLIR that vxc links.
pub(crate) const KNOWN_DIALECTS: &[&str] = &[
    "acc",
    "affine",
    "amdgpu",
    "amx",
    "arith",
    "arm_neon",
    "arm_sme",
    "arm_sve",
    "async",
    "bufferization",
    "builtin",
    "cf",
    "complex",
    "dlti",
    "emitc",
    "func",
    "gpu",
    "index",
    "irdl",
    "linalg",
    "llvm",
    "math",
    "memref",
    "ml_program",
    "mpi",
    "nvgpu",
    "nvvm",
    "omp",
    "pdl",
    "pdl_interp",
    "ptr",
    "quant",
    "rocdl",
    "scf",
    "shape",
    "shard",
    "smt",
    "sparse_tensor",
    "spirv",
    "tensor",
    "tosa",
    "transform",
    "ub",
    "vector",
    "wasmssa",
    "x86vector",
    "xegpu",
    "xevm",
    "vx",
];

/// The operation that a line of MLIR starts, and its dialect: `math.ctpop` and `math` for
/// `%r = math.ctpop %a : i64` and for `%r = "math.ctpop"(%a)`. `None` for a line that starts
/// no operation, such as `}` or `^bb0(%x: f32):`.
fn line_op(line: &str) -> Option<(&str, &str)> {
    let mut rest = line.split("//").next()?.trim();
    // The results come first: `%r =`, `%a, %b =` or `%r:2 =`.
    if rest.starts_with('%') {
        let (results, op) = rest.split_once('=')?;
        if !results.split(',').all(|r| r.trim().starts_with('%')) {
            return None;
        }
        rest = op.trim_start();
    }
    let rest = rest.strip_prefix('"').unwrap_or(rest);
    let (dialect, op) = rest.split_once('.')?;
    let is_name = |s: &str| {
        s.starts_with(|c: char| c.is_ascii_lowercase() || c == '_')
            && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
    };
    if !is_name(dialect) || !op.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_') {
        return None;
    }
    let op_len = op
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '.'))
        .unwrap_or(op.len());
    Some((&rest[..dialect.len() + 1 + op_len], dialect))
}

impl TypeChecker<'_> {
    /// E8010: a name in the `dialects` list of an `mlir!` block that is not a dialect, or an
    /// operation of a dialect that the list does not name. The block gives its value with
    /// `macro.yield`, or with `func.return` from the function that Vx wraps it in, and no list
    /// names either. The first operation that is missing is reported.
    pub(crate) fn check_block_dialects(&mut self, m: &InlineMlirExpr) {
        if self.speculating {
            return;
        }
        let at = Some(crate::diagnostic::SourceSpan::from_ast_span(&m.span));
        for name in &m.dialects {
            if !KNOWN_DIALECTS.contains(&name.as_str()) {
                self.errors.error_with_code(
                    crate::diagnostic::DiagnosticCode::E8010,
                    format!(
                        "the `dialects` list of this `mlir!` block names `{name}`, which is not \
                         a dialect"
                    ),
                    at.clone(),
                );
            }
        }
        let missing = m.block_str.lines().find_map(|line| {
            let (op, dialect) = line_op(line)?;
            let gives_the_value = op == "macro.yield" || op == "func.return";
            (!gives_the_value && !m.dialects.iter().any(|d| d == dialect))
                .then(|| (line.trim().to_string(), dialect.to_string()))
        });
        if let Some((line, dialect)) = missing {
            self.errors.error_with_code(
                crate::diagnostic::DiagnosticCode::E8010,
                format!(
                    "this `mlir!` block uses `{line}`, but its `dialects` list does not name \
                     `{dialect}`. Add \"{dialect}\" to the list"
                ),
                at,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_line_names_its_operation_and_dialect() {
        let math = Some(("math.ctpop", "math"));
        assert_eq!(line_op("%r = math.ctpop %a : i64"), math);
        assert_eq!(line_op("  %r = \"math.ctpop\"(%a) : (i64) -> i64"), math);
        let scf_if = Some(("scf.if", "scf"));
        assert_eq!(line_op("%a, %b = scf.if %c -> (i32, i32) {"), scf_if);
        let scf_while = Some(("scf.while", "scf"));
        assert_eq!(
            line_op("%r:2 = scf.while (%x = %y) : (i32) -> (i32, i32) {"),
            scf_while
        );
        let intr = Some(("llvm.intr.ctpop", "llvm"));
        assert_eq!(line_op("%r = llvm.intr.ctpop(%a) : (i32) -> i32"), intr);
        assert_eq!(
            line_op("func.return %s : i32"),
            Some(("func.return", "func"))
        );
        assert_eq!(line_op("}"), None);
        assert_eq!(line_op("^bb0(%in: f32, %out: f32):"), None);
        assert_eq!(line_op("// math.ctpop in a comment"), None);
        assert_eq!(line_op("iterator_types = [\"parallel\"]"), None);
    }

    /// The list is what the MLIR that vxc links registers: each name loads, and the count
    /// matches. An LLVM upgrade that adds, removes or renames a dialect fails here.
    #[test]
    fn the_known_dialects_are_the_ones_mlir_registers() {
        let registry = melior::dialect::DialectRegistry::new();
        melior::utility::register_all_dialects(&registry);
        let context = melior::Context::new();
        context.append_dialect_registry(&registry);
        let upstream: Vec<&str> = KNOWN_DIALECTS
            .iter()
            .copied()
            .filter(|d| *d != "vx")
            .collect();
        for name in &upstream {
            let name_ref = melior::StringRef::new(name);
            // SAFETY: the context is live, and the call returns a null dialect for an unknown name.
            let dialect = unsafe {
                mlir_sys::mlirContextGetOrLoadDialect(context.to_raw(), name_ref.to_raw())
            };
            assert!(
                !dialect.ptr.is_null(),
                "MLIR does not register the dialect `{name}`"
            );
        }
        assert_eq!(
            context.registered_dialect_count(),
            upstream.len(),
            "MLIR registers a dialect that KNOWN_DIALECTS does not list: compare it with \
             `mlir-opt --show-dialects`"
        );
    }
}
