# MLIR Codegen for Control Flow: `cf` Dialect

## Overview

We initially implemented `loop`, `break`, and `match` using Structured Control Flow (`scf.if` and `scf.while`). However, `scf` regions are strictly isolated, making it difficult to support arbitrarily nested `break` and `continue` statements out of `if` blocks or deeply nested loops.

To support true C/Rust-like control flow, we are migrating to **Option A**: using Basic Blocks and the `cf` dialect (`cf.br`, `cf.cond_br`).

## Phase 1: CFG Refactoring (Option A)

### Goal

Refactor all control flow operations (`IfStmt`, `LoopStmt`, `BreakStmt`, `MatchExpr`) to use `cf` basic blocks.

### Proposed Changes

#### [MODIFY] `src/codegen/generator.rs`

- Add `break_blocks: Vec<*const melior::ir::Block<'c>>` to track the loop end blocks.
- Add `continue_blocks: Vec<*const melior::ir::Block<'c>>` to track loop condition blocks.
- Change `generate_statement` to return `Option<&'c melior::ir::Block<'c>>`. If a statement emits an unconditional branch (like `break`), it returns `None`. The block generator (`{ ... }`) will stop emitting operations if `None` is returned, effectively handling unreachable code.

#### [MODIFY] `src/codegen/lower.rs`

- **LoopStmt**: Create `cond_block`, `body_block`, `end_block` using `parent_region.append_block()`. Push target pointers. Lower body. Emit `cf.br(&cond_block)` at the end of the body.
- **BreakStmt**: Peek `break_blocks`, emit `cf.br(end_block)`. Return `None` to indicate block termination.
- **IfStmt**: Create `then_block`, `else_block`, `merge_block`. Emit `cf.cond_br`. Lower bodies. If a body doesn't terminate early, emit `cf.br(&merge_block)`. Return `Some(merge_block)`.
- **MatchExpr**: Refactor from chained `scf.if` to chained `cf.cond_br` to basic blocks.

#### [DELETE] `src/codegen/break_utils.rs`

- We no longer need AST lookahead to conditionally inject `scf.if(!break_flag)`. The CFG handles it natively!

> [!IMPORTANT]
> This refactoring will break existing tests temporarily until all control flow nodes (`if`, `loop`, `match`) are rewritten, because we cannot mix `cf.br` escaping from inside `scf` regions.

## Phase 2: Compiler Warnings Infrastructure (Unreachable Code)

### Goal

As identified by the user, if a programmer writes a statement after a `break` or `return`, it is dead code. We should emit a compiler warning.

### Proposed Changes

- Introduce a diagnostic/warning system during semantic analysis or codegen.
- When traversing a block (`{ stmt1; break; stmt2; }`), if we detect a statement *after* a terminator, we emit a warning: `Warning: Unreachable code`.
- *(Data structures for this will be fleshed out after Phase 1)*.

## Phase 3: Traits & Generics Polish

### Goal

Formalize the trait system to support `Iterator` and `IntoIterator`.

### Proposed Changes

- **AST/Sema**: Add `TraitDecl` and `ImplDecl`. Add trait bounds to generics. Type check `impl` blocks.
- **Parser**: Parse `trait` and `impl`.

## Phase 4: `For` Loop Desugaring using Iterables

- Implement `for x in iter {}` using the `Iterator` trait, desugaring to a `loop` + `match`.
