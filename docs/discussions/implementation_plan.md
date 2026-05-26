# Split Binary Operators

The current `BinaryOp` enum is overloaded with arithmetic, relational, and logical operators. We will separate these into distinct enums and `Expr` variants to better reflect their semantic differences.

## Proposed Changes

### AST (`src/ast.rs`)

#### [MODIFY] \[ast.rs\](file:///Users/adityak/go/Vx/src/ast.rs)

- Remove relational and logical operators from `BinaryOp`.
- Define `pub enum RelationalOp { Eq, NotEq, Lt, Gt, Le, Ge }` and `pub struct RelationalOpExpr`.
- Define `pub enum LogicalOp { And, Or }` and `pub struct LogicalOpExpr`.
- Add `RelationalOp(RelationalOpExpr)` and `LogicalOp(LogicalOpExpr)` to the `Expr` enum.
- Update `Expr::span()`.

### Parser (`src/parser.rs`)

#### [MODIFY] \[parser.rs\](file:///Users/adityak/go/Vx/src/parser.rs)

- Update `parse_binary_expr` to check the operator type and emit `Expr::BinaryOp`, `Expr::RelationalOp`, or `Expr::LogicalOp` accordingly.
- Fix any parser tests that manually assert `BinaryOp` variants.

### Semantic Analysis (`src/sema.rs`)

#### [MODIFY] \[sema.rs\](file:///Users/adityak/go/Vx/src/sema.rs)

- Add match arms in `type_check_expr` for `Expr::RelationalOp` and `Expr::LogicalOp`.
- Retain the existing type-checking logic for these operators but adapt it to the new `Expr` variants.

### Code Generation (`src/melior_codegen.rs`)

#### [MODIFY] \[melior_codegen.rs\](file:///Users/adityak/go/Vx/src/melior_codegen.rs)

- Implement `LowerToMelior` for `RelationalOpExpr` and `LogicalOpExpr`.
- Move the MLIR predicate generation logic into the `RelationalOpExpr` lowering, and logical operation generation into `LogicalOpExpr` lowering.

## Verification Plan

- Run `cargo fmt` and `cargo test`.
- Ensure all tests (frontend, middle-end, backend) pass without any changes to the test files themselves.
