# Rust-Like Macro System Implementation

This document outlines the plan to implement a full-fledged declarative macro system in Vx, analogous to Rust's `macro_rules!`.

## Goal Description

Introduce declarative macros (`macro_rules!`) to enable metaprogramming in Vx. This will allow developers to write custom syntactic extensions like `vec![1, 2, 3]`, which parse custom token streams and expand them into standard AST nodes before type checking. The system will support token trees, repetition operators (`$(...)*`), and fragment specifiers (e.g., `expr`, `ident`, `stmt`).

## User Review Required

> [!IMPORTANT]
> The macro expansion pass requires parsing token streams *back* into AST nodes after expansion. This implies the parser must be able to parse expressions/statements from arbitrary token sequences instead of just the root source file. Does this align with your expectations?
>
> We will implement declarative macros (`macro_rules!`) first. Procedural macros (which run arbitrary Rust/Vx code at compile time) are significantly more complex and out of scope for this initial phase.

## Proposed Changes

### Lexer

- Add `TokenType::Dollar` (`$`) to support capture variables (e.g., `$x`).
- Add `TokenType::MacroRules` or parse `macro_rules` as an identifier combined with `Bang` (`!`).

### Parser & Token Trees

- Introduce `TokenTree` (TT) structures in `ast/mod.rs` to represent raw tokens enclosed in `()`, `{}`, or `[]`.
- Update `Parser` to support parsing macro definitions:
  ```rust
  macro_rules! name {
      ( pattern ) => { expansion };
  }
  ```
- Update `Expr` and `Stmt` parsing to intercept `ident!` and parse the subsequent tokens as a `MacroCallExpr` or `MacroCallStmt` containing a `TokenTree`.

### AST

- **`MacroDefDecl`**: Represents a parsed `macro_rules!` definition, containing rules (Pattern -> Expansion TT).
- **`MacroCall`**: Represents an invocation like `vec![1, 2, 3]`.

### `src/ast/macro_expand.rs` (COMPLETED & NEXT PHASE)

- **Status:** File created, basic integration in `src/pipeline.rs` and `src/driver.rs` completed.
- **Next Phase:**
  - Add a full `match_rule` engine that supports `$name:expr` and `$name:ident`.
  - Add a `transcribe` engine to replace `$name` with the captured tokens.
  - Recursively parse the transcribed token stream back into expressions/statements.

#### [MODIFY] \[src/ast/macro_expand.rs\](file:///Users/adityak/go/Vx/src/ast/macro_expand.rs)

- Implement `match_rule` and `transcribe`.

#### [MODIFY] \[src/parser/decl.rs\](file:///Users/adityak/go/Vx/src/parser/decl.rs)

- The parsing for `macro_rules!` is already complete.

- **Matcher**: Takes a macro invocation TT and matches it against the macro definition's pattern, extracting captured fragments (e.g., `$x:expr` captures an AST expression).

- **Expander**: Transcribes the expansion TT, substituting the captured fragments and expanding repetitions `$(...)*`.

- **Re-parsing**: Converts the expanded TokenTree back into Vx AST nodes (`Expr`, `Stmt`, `Decl`) and injects them into the AST, replacing the macro call.

## Verification Plan

### Automated Tests

- Create `scratch/test_macro.vx` demonstrating the `vec!` macro:
  ```rust
  macro_rules! vec {
      ( $( $x:expr ),* ) => {
          {
              let mut temp_vec = Vec::new();
              $( temp_vec.push($x); )*
              temp_vec
          }
      };
  }
  let v = vec![10, 20, 30];
  ```
- Verify that `v` is correctly expanded, type-checked, and lowered by MLIR, resulting in correct runtime execution.

### Integration

- Ensure that macro expansion errors produce meaningful diagnostics with accurate source line numbers (spanning the macro invocation).
