# Macro System Implementation Tasks

- `[/]` **Lexer Updates**

  - `[/]` Add `TokenType::Dollar` (`$`).
  - `[/]` Add parsing for `TokenType::Dollar` in `lexer.rs`.
  - `[/]` Ensure `macro_rules` is handled gracefully (either as keyword or parsed as identifier).

- `[ ]` **AST Nodes**

  - `[ ]` Define `TokenTree` structures (`Token`, `Group`, `Delimited`).
  - `[ ]` Define `MacroDefDecl` in `ast/decl.rs` (patterns and expansions).
  - `[ ]` Define `MacroCallExpr` in `ast/expr.rs`.
  - `[ ]` Define `MacroCallStmt` in `ast/stmt.rs`.

- `[ ]` **Parser Updates**

  - `[ ]` Implement `parse_token_tree()` to handle nested `()`, `{}`, `[]` and collect raw tokens.
  - `[ ]` Update `parse_decl()` to intercept `macro_rules! name { ... }` and parse into `MacroDefDecl`.
  - `[ ]` Update expression parsing to intercept `ident! ( ... )` and parse into `MacroCallExpr`.
  - `[x]` Implement Phase 1: AST and Parsing

- `[x]` Implement Phase 2: Intercept in sema/resolver/codegen

- `[x]` Implement Phase 3: Macro Expansion Pass (`macro_expand.rs`)

- `[x]` Implement `match_rule` and `transcribe` logic for variables.

  - `[ ]` Implement `Matcher`: matching TokenTrees against rules and binding `$name:frag`.
  - `[ ]` Implement `Expander`: transcribing tokens, expanding `$name` and repetitions `$(...)*`.
  - `[ ]` Implement `Re-parser`: convert expanded TokenTree back into `Expr`, `Stmt`, etc.

- `[ ]` **Compiler Integration**

  - `[ ]` Wire up the expansion pass in `main.rs` before Semantic Analysis.
  - `[ ]` Ensure `macro_rules!` definitions are hoisted or tracked in scope.

- `[ ]` **Verification**

  - `[ ]` Create `scratch/test_macro.vx` with `vec!` macro.
  - `[ ]` Test nested repetitions and variable substitution.
  - `[ ]` Ensure MLIR lowering and JIT execution work correctly for expanded code.
