# Brittleness Hardening Plan

> Codebase review + prioritized fixes for hardcoded values, shortcuts, and missing
> paths to MLIR codegen. Each item cites `file:line` evidence and a fix approach.
> Status is tracked inline (`[ ]` / `[x]`) as items land.

## Tier 1 — crash risk / correctness

### 1. `generate_expr` panics on unhandled AST nodes `[ ]`

`src/codegen/generator.rs:840` is `_ => todo!("{:?}", expr)`. Five `Expr` variants fall
through and **panic at codegen** instead of erroring: `Range`, `VecMacro`, `MemorySpace`,
`MacroCall`, `TransferPredicate` (a non-`comptime` use of `Transfer<A,B>`). A program with
a bare range or `vec![]` in an unexpected position crashes the compiler.

**Fix:** replace the catch-all with explicit arms that return a `LowerError` naming the
unsupported construct (or lower `Range`/`VecMacro` properly); `MacroCall` stays an
`unreachable!` (must be expanded before codegen) but with a clear message.

### 2. Type-checker masks errors with a placeholder `[ ]`

`src/hir/expr.rs:760` returns `Type::Tensor(F32, [], None)` as a "Default placeholder on
error"; a failed lookup silently becomes an f32 tensor → wrong downstream codegen. Similar
"mock behavior for now" coercions (`expr.rs:332`) and "hardcoded mock methods" (`expr.rs:2810`).

**Fix:** emit a diagnostic and return a dedicated poison/unknown type that suppresses
cascade errors rather than an f32 tensor that silently type-checks.

### 3. Internal-invariant panics on the codegen hot path `[ ]`

~15 `panic!` / `unwrap_or_else(|| panic!)` in `generator.rs` ("Failed to parse MLIR type",
"Generic … should be instantiated", "Matrix type not supported"). Each is a hard crash
where a diagnostic belongs. (Tracked together with #4.)

## Tier 2 — hardcoded / duplicated (divergence risk)

### 4. Three separate hardcoded topology→int maps `[ ]`

`topology_to_i32` (`src/codegen/lower/mod.rs:146`), a *different* `target_topology_id`
(`src/codegen/lower/tensors.rs:120`), and two `addr_space` matches
(`src/codegen/generator.rs:889,1188`) — none consult the topology registry that is now the
single source of truth. `Custom` gets an FNV-hash id (`mod.rs:162`, collision-prone).

**Fix:** add `dispatch_id()` / `address_space()` to the topology registry (or derive from
the descriptor), route all four sites through it, delete the duplicated matches.

## Tier 3 — stubs / unwired paths (device codegen honesty)

### 5. `VxHardwarePlugin` is not wired into the pipeline `[ ]`

`apple_npe.lower_to_binary` returns MLIR-string bytes (`src/plugin/apple_npe.rs:38`), and
nothing in `driver.rs`/`codegen/` calls the plugin trait. Architecture-only.

### 6. `--emit-llvm` / `--target` are shallow `[ ]`

The VX `--emit-llvm` path prints **LLVM-dialect MLIR, not real `.ll`**
(`src/driver.rs:465`); `translate_to_llvm_ir` (real `mlir-translate`) is only used for the
MLIR-language input path. `--target` **text-injects** the triple. Fix: optional real `.ll`
emission + set the triple as a real module attribute.

### 7. ANE dispatcher hardcodes shapes `[ ]`

`runtime/npu_dispatch.mm:300-337` assumes `memrefs [res,a,b]` order and `sizes == 4`.
Generalize (or keep as a documented 4×4 demo primitive).

## Tier 4 — limits

### 8. Seam value contracts capped at 0-255 `[ ]`

`VAL_BITS = 8` (`src/hir/seam.rs:42`) + the `(0.0..256.0)` guard (`src/hir/expr.rs:1080`)
silently downgrade any contract with a constant > 255 to the coarse visibility check.
**Fix:** widen `VAL_BITS` (e.g. 64) and drop the guard.

### 9. Global topology registry `[ ]`

Parse-time registration mutates process-global state; coherence is per-program-scoped now,
but name lookups remain global. **Fix (later):** thread a per-compilation registry, or
snapshot at `TypeChecker::new`.

## Recommended order

1 → 4 → 2 → 3 → 8 → 6, with 5/7/9 as separate scoped efforts (real device backends /
architecture). #1 and #4 build directly on the topology work already landed and are the
highest value-per-effort.
