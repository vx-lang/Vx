# Code Review: `src/sema/stmt.rs`

This file governs the semantic analysis of statements. Its most critical role is acting as the bridge between standard type-checking and **Formal Verification**.

## Core Responsibilities

- **Scope Management**: Pushes and pops semantic lexical scopes.
- **HIR Lowering**: Directly emits dummy `hir::OP_STORE` and `hir::OP_RET` instructions to the active `TypeChecker` registry when it processes `Assign`, `Return`, or `LetDecl`.
- **Compile-Time Evaluation**: Uses `eval_expr` and `eval_statement` to interpret statements at compile time (e.g. for `comptime` blocks or array sizing).
- **Mathematical Constraints**: Populates the `self.constraints` vector to build a logical model of the program for the SMT Prover.

## Formal Verification & Contracts

Vx is deeply integrated with Design-by-Contract principles. `stmt.rs` translates statements into mathematical proofs:

1. **Loop Invariants**: When encountering a `ForLoopStmt` or `LoopStmt`, the checker evaluates the `invariants`. It attempts to statically prove them upon loop entry using `prove_expr`. If valid, it pushes them into the `constraints` pool as assumptions for the rest of the block, and verifies them again at the end.
1. **Return Verification**: A `ReturnStmt` pushes a synthetic equality constraint: `return == expr`. This allows `ensures` clauses declared on the function signature to mathematically reference the `return` value and prove post-conditions!
1. **Assert & Verified<T>**:
   - `AssertStmt` is dynamically evaluated at compile time if possible.
   - If inside a function returning `Verified<T>`, the compiler invokes the SMT solver (`prove_expr`) to guarantee the assertion holds across all mathematical inputs.
   - If it's just a standard function, the assertion is *assumed* true and added to `self.constraints` to aid in downstream proofs.

## Non-Lexical Lifetimes (NLL) Support

The checker maintains a `lookahead_stack` for statement execution. When checking a statement, it peers into the upcoming statements in the block. It populates `current_assignment_target` so that `expr.rs` knows *who* is taking a borrow, which allows the borrow checker to release dead borrows before the scope closes if the variable isn't used again.

## Design Critique & Actionable Items

1. **Coupling**: Emitting HIR instructions (`self.emit_inst(crate::hir::OP_STORE, ...)`) directly from the type checker violates the separation of concerns. The type checker should output a validated, typed AST, and a separate lowering pass (`codegen/lower.rs`) should translate it to HIR.
1. **SMT Prover Robustness**: `prove_expr` prints raw `println!("Warning: ...")` when the prover fails to lower an expression. This should be captured formally into `self.errors` or `self.warnings`.
1. **Iterator Lowering**: The `ForLoop` logic synthesizes a `.next()` method call to deduce the loop variable type for generics. This is clever but slightly brittle; it relies on string prefix matching `name.starts_with("Option<")`.
