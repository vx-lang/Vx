# Global Borrow Checker & 256-bit FastPath Integration

You are absolutely right. While the Lexical Borrow Checker prevents immediate aliasing conflicts within a single function scope, it lacks the cross-module capability to verify if references passed through generic signatures or function bounds satisfy lifetime rules (`'a : 'b`). 

To claim full victory, we need to utilize the **256-bit Creative Hashing Strategy** (`TypeId`) to perform high-speed hardware-level register checks for subtype and variance bounding.

## Proposed Changes

### 1. Document the Hashing Algorithm (`src/borrow.rs`)
I will rewrite the top-level documentation of `src/borrow.rs` to thoroughly detail the algorithm.
- **The Data Structure**: The `TypeId` is a 256-bit (4-word) structure. Word 2 handles generic lifetimes.
- **Fast Path Bitpacking**: Instead of iterating through graph edges, we pack up to 4 lifetime boundaries directly into Word 2 (16 bits per parameter).
- **Variance Math**: The 16 bits use 4 bits for Variance Flags (Covariant, Contravariant, Invariant) and 12 bits for the Region ID. 
- **Subtyping Evaluation**: A single bitwise evaluation and numerical comparison checks if `Region A <= Region B` (where lower Region IDs live longer, with Region 0 being `'static`).

### 2. Lowering AST `Type` to `TypeId`
Currently, `is_assignable` in `sema.rs` operates purely on AST enums. I will introduce a helper method `lower_to_type_id(&Type, scope_depth) -> TypeId`.
- For `Type::Borrow(&mut T)`, it will assign the variance flag to `Invariant` and extract the current `scope_depth` as the `region_id`.
- For `Type::Borrow(&T)`, it will assign the variance flag to `Covariant`.
- It will pack these directly into a `TypeId` using `tid.try_set_fast_param(...)`.

### 3. Hooking up `verify_subtyping_bounds` (`src/sema.rs`)
In `sema.rs`, we will modify `is_assignable()` to utilize the bitwise FastPath when assigning references:
```rust
if let Type::Borrow(...) = target {
    if let Type::Borrow(...) = source {
        // ... (Memory space checks) ...
        
        let id_target = self.lower_to_type_id(target);
        let id_source = self.lower_to_type_id(source);
        
        // Execute the 256-bit register hash strategy!
        if !crate::borrow::verify_subtyping_bounds(&id_source, &id_target, self.worker) {
            return false;
        }
    }
}
```

## User Review Required

> [!NOTE]
> Currently, the FastPath bitwise math assumes "smaller region IDs live longer" (e.g. Scope Depth 0 > Scope Depth 1). Is this the exact mathematical convention you envisioned for the numerical `region_a <= region_b` check in `verify_subtyping_bounds`? I will proceed using AST scope depth as the literal Region ID unless instructed otherwise!
