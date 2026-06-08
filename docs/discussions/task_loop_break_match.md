# Task List: Phase 1 CFG Refactoring

## 1. Update Generator State

- `[ ]` Add `break_blocks: Vec<*const melior::ir::Block<'c>>` to `MeliorGenerator`.
- `[ ]` Add `continue_blocks: Vec<*const melior::ir::Block<'c>>` to `MeliorGenerator`.
- `[ ]` Update `MeliorGenerator::new` to initialize these vectors.
- `[ ]` Remove `break_flags` from `MeliorGenerator`.

## 2. Update Codegen Signatures

- `[ ]` Modify `LowerToMelior<'c>` to take `block: &'c melior::ir::Block<'c>` or figure out the signature for `generate_statement`.
  Wait, let's use the explicit lifetime `'a` for block references:
  `fn generate_statement<'a>(&mut self, stmt: &Statement, block: &'a melior::ir::Block<'c>) -> Option<&'a melior::ir::Block<'c>>`
- `[ ]` Update `Expr` lowering to just take `block: &'a Block<'c>`. Since expressions don't currently terminate blocks, they don't need to return `Option<&'a Block>`. Wait, `Expr::Match` or `Expr::If` DO terminate blocks if they return values via branches! Oh, an expression can branch?
  Wait. `Expr::If` and `Expr::Match` currently return `(Value, Type)`. If they use `cf.cond_br`, they must create a `merge_block` where the branches jump to and pass block arguments (phi nodes) to yield the result!
  So `generate_expr` might also need to return the new active block?
  `fn generate_expr<'a>(&mut self, expr: &Expr, block: &'a Block<'c>) -> (Value<'c, 'c>, Type<'c>, &'a Block<'c>)`

## 3. Lowering Control Flow Statements

- `[ ]` Rewrite `LoopStmt` to use `cf.br`.
- `[ ]` Rewrite `BreakStmt` to use `gen.break_blocks`.
- `[x]` Run all `compile_test` tests to ensure the JIT execution doesn't crash anymore. completeness).
- `[ ]` Rewrite `IfStmt` (if it exists as a statement) and `Expr::If` to use `cf.cond_br` and block arguments.
- `[ ]` Rewrite `Expr::Match` to use chained `cf.cond_br` and block arguments.

## 4. Unreachable Code Warning

- `[ ]` Add unreachable warning in `BlockStmt` or `generate_function` when `generate_statement` returns `None` but there are more statements in the block.

## 5. Cleanup

- `[ ]` Delete `src/codegen/break_utils.rs`.
- `[ ]` Test and verify.
