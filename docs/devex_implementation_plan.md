# Akar Developer Experience (Dev-Ex) Tools Implementation Plan

This document outlines the approach for building robust developer tooling for the Akar systems programming language. The goal is to dramatically improve the developer experience by adding VSCode syntax highlighting, an AST text pretty-printer, and Clang-style precise compiler error squigglies.

## 1. VSCode Extension (Syntax Highlighter)
We will create a lightweight VSCode extension inside a new `vscode-akar/` directory in the repository.

**Components:**
- `package.json`: Registers the `.ak` extension and defines the language configuration.
- `language-configuration.json`: Configures auto-closing brackets, comments (`//`), and folding.
- `syntaxes/akar.tmLanguage.json`: A TextMate grammar that correctly identifies:
  - **Keywords**: `fn`, `let`, `mut`, `spawn`, `on`, `unsafe`, `comptime`, `match`, etc.
  - **Topology / Hardware Identifiers**: `Topology`, `MemorySpace`, `Host`, `GPU`, `AMX`, `ANE`.
  - **Types**: `i32`, `f32`, `memref`, `bool`.
  - **Operators & Literals**: Strings, numbers, and basic symbols.

## 2. AST Pretty Printer
We will implement a custom AST formatter to print the Abstract Syntax Tree in a structured, human-readable text format, aiding in compiler debugging.

**Approach:**
- Create `src/ast_printer.rs` containing an `AstPrinter` struct.
- Implement recursive `print_stmt` and `print_expr` methods that utilize indentation tracking to render tree nodes (e.g., `├─ BinaryOp` or `└─ Identifier`).
- Hook this into `src/main.rs` via a new `--print-ast` compiler flag so developers can visualize the parsed code before it lowers to MLIR.

## 3. Clang-Style Compiler Error Indicators
To provide precise squiggly-line error reporting (e.g., `~~~~^~~~`), we need to enhance the error tracking pipeline from the Lexer to the Compiler frontend.

**Approach:**
- **Lexer Upgrade**: Enhance the `Token` struct in `src/lexer.rs` to track `length` (or the raw `&str` slice) so we know exactly how many `~` to draw.
- **Error Formatter**: Create an `AkarError` struct or utility function `format_compiler_error(source, line, column, length, message)` that takes the original source code, extracts the specific line, and renders the pointer string beneath it.
- **Parser & Sema Integration**: Update `parser.rs` and `sema.rs` to pass along the source code string (or a line index array) and the token locations so they can emit these rich, visually formatted errors instead of basic `"Error at L:C"` strings.

## User Review Required
> [!IMPORTANT]
> 1. **AST Formatting:** Do you want the AST to look like Lisp S-Expressions `(BinaryOp + (Ident a) (Number 1))` or like a graphical tree `├─ BinaryOp (+)`?
> 2. **Sema Tracking:** Semantic errors often involve variables that were parsed earlier. To point a squiggly line at a variable during *Semantic Analysis*, we will need to store `line` and `column` inside every AST Node (e.g., `Expr::Identifier(String, usize, usize)`). This requires a minor refactor of `ast.rs`. Is this acceptable?
