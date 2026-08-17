# Brittleness Hardening Plan

> Codebase review + prioritized fixes for hardcoded values, shortcuts, and missing
> paths to MLIR codegen. Each item cites `file:line` evidence and a fix approach.
> Status is tracked inline (`[ ]` / `[x]`) as items land.

## Tier 1 — crash risk / correctness

### 1. `generate_expr` panics on unhandled AST nodes `[x]`

`src/codegen/generator.rs:840` is `_ => todo!("{:?}", expr)`. Five `Expr` variants fall
through and **panic at codegen** instead of erroring: `Range`, `VecMacro`, `MemorySpace`,
`MacroCall`, `TransferPredicate` (a non-`comptime` use of `Reachable<A,B>`). A program with
a bare range or `vec![]` in an unexpected position crashes the compiler.

**Fix:** replace the catch-all with explicit arms that return a `LowerError` naming the
unsupported construct (or lower `Range`/`VecMacro` properly); `MacroCall` stays an
`unreachable!` (must be expanded before codegen) but with a clear message.

### 2. Type-checker masks errors with a placeholder `[x]`

`src/hir/expr.rs:760` returns `Type::Tensor(F32, [], None)` as a "Default placeholder on
error"; a failed lookup silently becomes an f32 tensor → wrong downstream codegen. Similar
"mock behavior for now" coercions (`expr.rs:332`) and "hardcoded mock methods" (`expr.rs:2810`).

**Fix:** emit a diagnostic and return a dedicated poison/unknown type that suppresses
cascade errors rather than an f32 tensor that silently type-checks.

**Landed:** the three genuine error paths (undefined variable, use-of-moved `E4001`, and
"no hardware transfer path") now return `Type::Unknown` after their diagnostic instead of a
fake f32 tensor. `Unknown` already prints as `?` and unifies with anything, so it poisons
the result without spawning cascade errors. The `expr.rs:332` numeric coercion and
`expr.rs:2810` built-in transfer methods are load-bearing *policy* (literal→f32 coercion,
`.to_device()`/`.with_memory()` lowering), not error masking, and are intentionally kept.

### 3. Internal-invariant panics on the codegen hot path `[x]`

~15 `panic!` / `unwrap_or_else(|| panic!)` in `generator.rs` ("Failed to parse MLIR type",
"Generic … should be instantiated", "Matrix type not supported"). Each is a hard crash
where a diagnostic belongs. (Tracked together with #4.)

**Landed:** `lower_type` / `lower_type_str` / `lower_tensor_type` already return
`Result<_, LowerError>`, so every reachable panic in them now `return Err(LowerError::…)`:
`Matrix` type, unparseable lowered MLIR/memref/enum-layout strings, wrong generic arity,
missing generic struct/enum, and const/unmonomorphized-generic leaks. The messages tagged
`internal:` are should-never-happen invariants that now degrade to a diagnostic instead of
aborting the process. The two debug `println!`s that dumped struct/enum keys were removed.
`Type::parse` on hardcoded constant strings (`i32`, `!llvm.ptr`, …) stays `.unwrap()` — it
cannot fail — and `Statement::MacroCall` stays a `panic!` (must be expanded before codegen).

## Tier 2 — hardcoded / duplicated (divergence risk)

### 4. Three separate hardcoded topology→int maps `[x]`

`topology_to_i32` (`src/codegen/lower/mod.rs:146`), a *different* `target_topology_id`
(`src/codegen/lower/tensors.rs:120`), and two `addr_space` matches
(`src/codegen/generator.rs:889,1188`) — none consult the topology registry that is now the
single source of truth. `Custom` gets an FNV-hash id (`mod.rs:162`, collision-prone).

**Fix:** add `dispatch_id()` / `address_space()` to the topology registry (or derive from
the descriptor), route all four sites through it, delete the duplicated matches.

**Landed:** `arch` now owns all four mappings as the single source of truth —
`topology_dispatch_id` / `memory_space_dispatch_id` (runtime `vx.spawn`/`vx.transfer` ids,
sharing the FNV scheme for `Custom` so a topology and its canonical memory space agree) and
`memory_space_address_space` / `topology_address_space` (the coarse `memref<…, N>` /
`!llvm.ptr<N>` annotation). The topology address space now *derives* from the topology's
default memory space, fixing a latent divergence where `Pinned<T, GPU>` got address space 5
but `Ref<T, GpuHbm>` got 1 for the same buffer. All four codegen sites are thin delegates.

## Tier 3 — stubs / unwired paths (device codegen honesty)

### 5. `VxHardwarePlugin` is not wired into the pipeline `[~]`

`apple_npe.lower_to_binary` returns MLIR-string bytes (`src/plugin/apple_npe.rs:38`), and
nothing in `driver.rs`/`codegen/` calls the plugin trait. Architecture-only.

**Partially landed (selection + consultation):** the trait is now consulted during
compilation. `plugin::plugin_for(dispatch_id)` exposes a static built-in `PluginRegistry`
(seeded with `AppleNPEPlugin` on macOS), and the `vx.spawn` lowering
(`codegen/lower/tensors.rs`) looks it up by the region's topology dispatch id — stamping a
`plugin = "<name>"` attribute on the op so the emitted IR records which backend owns the
region. `AppleNPEPlugin::target_topology()` was reconciled to the real ANE dispatch id
(`arch::topology_dispatch_id(ANE)` = 400) instead of the stale hardcoded `3`. Tests:
`plugin::tests::apple_npe_is_selected_by_ane_dispatch_id` and the macOS-gated
`ane_plugin_dispatch.vx` (`run_middle_end_test` now honors `// REQUIRES: macos`).

**Still open (issue #171):** `lower_to_binary` is still a byte passthrough (honestly
documented now, not a fake "compiled model"); the trait's MLIR types are mocks
(`mlir::Module { text: String }`) rather than real melior handles, so `is_op_supported` /
`register_passes` / a genuine device-binary lowering path remain future work.

### 6. `--emit-llvm` / `--target` are shallow `[x]`

The VX `--emit-llvm` path prints **LLVM-dialect MLIR, not real `.ll`**
(`src/driver.rs:465`); `translate_to_llvm_ir` (real `mlir-translate`) is only used for the
MLIR-language input path. `--target` **text-injects** the triple. Fix: optional real `.ll`
emission + set the triple as a real module attribute.

**Landed:** `--emit-llvm` alone still prints the portable, target-independent LLVM-dialect
MLIR (a legitimate view many backend tests assert against). Pairing it with a concrete
backend — `--emit-llvm --target <x86_64|aarch64|nvptx64|amdgcn>` — now sets that target's
`llvm.target_triple` / `llvm.data_layout` as **real module attributes** (via
`Operation::set_attribute`, replacing the fragile string-injection `tag_llvm_target`, which
was deleted) and runs `translate_to_llvm_ir`, so the output is genuine `.ll` with `target triple` / `target datalayout` / `define`. `llvm_backends.vx` checks the real `.ll` for
x86_64/aarch64/nvptx64; amdgcn is documented as a clean failure (valid AMDGPU `.ll` needs
addrspace(5) allocas, i.e. target-specific lowering that is out of scope for now).

### 7. ANE dispatcher hardcodes shapes `[ ]`

`runtime/npu_dispatch.mm:300-337` assumes `memrefs [res,a,b]` order and `sizes == 4`.
Generalize (or keep as a documented 4×4 demo primitive).

## Tier 4 — limits

### 8. Seam value contracts capped at 0-255 `[x]`

`VAL_BITS = 8` (`src/hir/seam.rs:42`) + the `(0.0..256.0)` guard (`src/hir/expr.rs:1080`)
silently downgrade any contract with a constant > 255 to the coarse visibility check.
**Fix:** widen `VAL_BITS` (e.g. 64) and drop the guard.

**Landed:** `VAL_BITS = 64`, and `hex()` masks with `u64::MAX` at full width to avoid the
`1 << 64` overflow. `extract_pins` now parses the literal directly as `u64` (exact; rejects
negatives/floats with no cap), and `const_value_of` accepts any non-negative integer up to
`2^53` (the f64 exact-integer limit, since its source value is an f64). New unit tests
`value_contract_holds_for_value_above_old_8bit_field` (70_000 now pins instead of masking to
`0x70`) and `hex_is_full_width_64_bit`.

### 9. Global topology registry `[~]`

Parse-time registration mutates process-global state; coherence is per-program-scoped now,
but name lookups remain global. **Fix (later):** thread a per-compilation registry, or
snapshot at `TypeChecker::new`.

**Partially landed (snapshot-reset):** `arch::reset_topology_registry()` restores the
built-in baseline, and the driver calls it at the start of each compilation
(`execute_vx_pipeline`, before `load_and_expand` parses). This closes the concrete leak — a
`topology` declared while compiling one program no longer bleeds into the next when several
are compiled in one process (the Rust test binary, a build server, an LSP). New unit test
`reset_clears_custom_topologies_but_keeps_builtins`; the additive registry tests now share a
mutex so they can't race the wipe. **Still open:** a fully thread-isolated per-compilation
registry (concurrent in-process compilations still share one global) is the larger,
separately-scoped change.

## Recommended order

1 → 4 → 2 → 3 → 8 → 6, with 5/7/9 as separate scoped efforts (real device backends /
architecture). #1 and #4 build directly on the topology work already landed and are the
highest value-per-effort.

**Status:** items 1, 2, 3, 4, 6, 8 are landed (see the `[x]` sections above); #5 and #9 are
partially landed (plugin selection/consultation, and the per-compilation snapshot-reset).
Remaining — larger, separately-scoped device/architecture work:

- **#5** (remainder) a real device-binary `lower_to_binary` + porting the trait onto real
  melior MLIR handles (so `is_op_supported` / `register_passes` can run).
- **#7** generalize the ANE dispatcher beyond the 4×4 demo shapes (bounded by fixed-shape
  CoreML primitives — realistically a documented demo limit until real device models exist).
- **#9** (remainder) thread a fully thread-isolated per-compilation registry rather than the
  process-global one.
