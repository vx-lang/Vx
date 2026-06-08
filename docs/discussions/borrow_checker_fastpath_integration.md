# Borrow Checker FastPath Integration

## Overview

Traditional borrow checkers (like Rust's Non-Lexical Lifetimes) rely on complex constraint graph traversal and heavy AST iterations. This causes massive slowdowns during compilation when checking deep lifetime relationships.

The Vx compiler solves this by leveraging a **256-bit Creative Hashing Strategy**, encoding lifetimes and subtyping variance directly into global `TypeId` structures and executing hardware-level bitwise register math to prove safety instantly.

## The 256-bit TypeId Bitpacking

The `TypeId` is a 256-bit identifier split into 4 x 64-bit words.
**Word 2** is exclusively reserved for the "FastPath Lifetime Hash", which stores up to 4 generic parameters/lifetimes simultaneously into 16-bit slots:
`[ Param 3 (16) | Param 2 (16) | Param 1 (16) | Param 0 (16) ]`

Each 16-bit payload contains:
`[ Variance Flags (4 bits) | Region ID (12 bits) ]`

## Region ID Convention (Math Rules)

The Region ID fundamentally represents the lifetime of a borrow as the depth of its lexical scope.

- **`Region 0`**: The `'static` lifetime (Global scope, lives forever).
- **`Region N`**: A local nested block at depth N.

**Mathematical Rule**: A smaller Region ID represents a *longer* lifetime.
Therefore, `Region A <= Region B` proves that Lifetime A outlives or equals Lifetime B.

## Variance Check Math

In `src/borrow.rs`, the `verify_subtyping_bounds` method evaluates coercion rules using the 4-bit variance flags:

1. **Covariant (`0x1`)**: Common for immutable references (`&T`).
   Allows a longer lifetime to be coerced/assigned to a shorter lifetime.
   *FastPath check*: `region_a <= region_b`

1. **Invariant (`0x0`)**: Strictly enforced.
   *FastPath check*: `region_a == region_b`

1. **Contravariant (`0x2`)**: Common for function parameters.
   Target must outlive source.
   *FastPath check*: `region_a >= region_b`

## AST Lowering to FastPath

During semantic analysis (`src/sema.rs`), the AST representations of References (`Type::Borrow`) are dynamically lowered to `TypeId`s by:

1. Deriving `Region ID` directly from `self.scopes.len()`.
1. Resolving the variance (`0x1` for reference lifetimes, allowing reborrowing).
1. Packing this data into the FastPath word via `try_set_fast_param`.

When assigning references or passing function arguments, `verify_subtyping_bounds` validates these packed words against each other to safely catch lifetime mismatches or strict aliasing violations globally.
