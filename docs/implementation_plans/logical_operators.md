# Implementation Plan: Logical Operators

We will implement logical (`&&`, `||`, `!`) and relational (`==`, `!=`, `<`, `>`, `<=`, `>=`) operators into the Akar compiler. This feature touches the entire pipeline from lexing to MLIR generation.

## User Review Required

> [!IMPORTANT]
> Akar currently lacks an explicit boolean element type in `ElementType`. I propose adding `ElementType::Bool` (lowering to MLIR's `i1`) so that comparisons evaluate to `Tensor<Bool>` (or a scalar boolean). Alternatively, we could reuse `I8`. Please approve the addition of `ElementType::Bool` or suggest an alternative.

## Proposed Changes

### 1. Lexer (`src/lexer.rs`)
- [NEW] Add tokens: `EqEq`, `NotEq`, `LessEq`, `GreaterEq`, `AndAnd`, `OrOr`, `Bang`.
- (Note: `LeftAngle` and `RightAngle` already exist for `<` and `>` and will be reused for less-than/greater-than in expression context).

### 2. AST (`src/ast.rs`)
- [MODIFY] Add `ElementType::Bool`.
- [MODIFY] Add `BinaryOp` variants: `Eq`, `NotEq`, `Lt`, `Gt`, `Le`, `Ge`, `And`, `Or`.
- [NEW] Add `UnaryOp` enum with `Not` variant.
- [MODIFY] Add `Expr::UnaryOp(UnaryOp, Box<Expr>)`.

### 3. Parser (`src/parser.rs`)
- [MODIFY] Implement precedence-based expression parsing (Pratt parsing or recursive descent precedence levels).
  - Precedence order (highest to lowest): Unary `!`, `* /`, `+ -`, `< > <= >=`, `== !=`, `&&`, `||`.
  - Handle `LeftAngle` and `RightAngle` interchangeably as `<` and `>` in expressions, disambiguating them from generic type parameters.

### 4. Semantic Analysis (`src/sema.rs`)
- [MODIFY] Typecheck binary operations:
  - Relational ops require operands to be of the same type. The result type is `Tensor(ElementType::Bool)`.
  - Logical ops (`&&`, `||`, `!`) require boolean operands.

### 5. MLIR Codegen (`src/codegen.rs`)
- [MODIFY] Map logical operations to MLIR:
  - Integer relational: `arith.cmpi` (eq, ne, slt, sgt, sle, sge)
  - Float relational: `arith.cmpf` (oeq, one, olt, ogt, ole, oge)
  - Logical boolean ops: `arith.andi`, `arith.ori`.
- [MODIFY] Ensure `ElementType::Bool` lowers to `i1`.

### 6. Documentation & Tests
- [MODIFY] Update `docs/syntax.md` to document the new operators.
- [NEW] Add `docs/tutorial/logical_operators.md` as required by our project guidelines.
- [MODIFY] Update `docs/semantics/operational.md` to formally define boolean evaluation transitions.
- [NEW] Create `tests/backend/logical_ops.ak` to test MLIR codegen and execution natively.

## Verification Plan
1. Run `cargo test` to ensure `Lexer`, `Parser`, and `Sema` validate logical structures.
2. Run backend test (`logical_ops.ak`) through `lli` to ensure the runtime behaves mathematically correctly when evaluating combinations of `&&` and `<`.
