# Extend Parser to Support `test.ak`

This plan details the implementation of a recursive descent parser in `src/parser.rs` to successfully parse the `test.ak` snippet and related constructs defined in `src/ast.rs`.

The goal is to generate a full `Program` AST from the token stream for basic constructs like functions, variables, basic types, and the unique `spawn on` construct.

## Open Questions

> [!NOTE]
> - Do you want to support arbitrary expressions in array index locations for topology (e.g. `Topology::NPU[i + 1]`) right now, or just integer literals (e.g. `Topology::NPU[0]`)?
> - Should `transfer` be a built-in expression type as currently defined in the AST (`Expr::Transfer`), or parsed as a standard function call and resolved during semantic analysis? (I will implement it as `Expr::Transfer` since it is already in `ast.rs`).

## Proposed Changes

### AST
No significant changes are needed in `src/ast.rs` right now, as it already supports `LetDecl`, `Return`, `SpawnOn`, `ExprStmt`, `Transfer`, and topologies.

### Parser
We will implement the following parsing methods in `src/parser.rs`.

#### [MODIFY] [parser.rs](file:///Users/adityak/go/akar/src/parser.rs)
Add recursive descent methods to `Parser`:
1. **`parse_function`**:
   - Expects `Fn` token, followed by an identifier.
   - Parses parameters: `(`, comma-separated `parse_param` calls, `)`.
   - Parses return type: `->` followed by `parse_type()`.
   - Parses body: `{` followed by a list of `parse_statement()` calls until `}`.
2. **`parse_type`**:
   - Parses base types like `Tensor`, `Matrix`.
   - Parses generic-like syntax for smart types: `Ref<Type, MemorySpace>`, `Verified<Type>`, `Pinned<Type, Topology>`.
3. **`parse_statement`**:
   - `LetDecl`: Expects `Let`, `Identifier`, `=`, `parse_expr()`, `;`.
   - `Return`: Expects `Return`, `parse_expr()`, `;`.
   - `SpawnOn`: Expects `Spawn`, `On`, `(`, `parse_topology()`, `)`, `{`, body statements, `}`.
   - `ExprStmt`: `parse_expr()`, `;`.
4. **`parse_expr`**:
   - Uses a basic lookahead to differentiate between identifiers and function calls (e.g. `custom_matmul(...)`).
   - Specifically handles the `transfer(expr, memory)` keyword as an `Expr::Transfer`.
5. **`parse_topology`** and **`parse_memory_space`**:
   - Parsers for the enum variations (e.g. `Memory::Host_DRAM`, `Topology::NPU[0]`).

## Verification Plan

### Automated Tests
- I will write a unit test in `src/parser.rs` that tokenizes and parses the exact content of `test.ak`.
- I will run `cargo test` to verify that `parser::parse` produces the correct AST.

### Manual Verification
- We can add a simple `-p` (parse-only) flag to `src/main.rs` that prints out the AST using `println!("{:#?}", program)` to verify the parser handles the `test.ak` file correctly.
