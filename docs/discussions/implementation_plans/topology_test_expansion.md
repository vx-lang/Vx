# Topology Test Coverage Expansion & Feature Implementation

## Date: 2026-06-20

## Summary

Comprehensive audit and expansion of topology-related test coverage across the
Vx compiler, including parser, sema, arch, codegen, registry, and `.vx`
integration tests. Also includes implementation of NPU Slice syntax parsing and
a sema bug fix for dynamic topology indices.

______________________________________________________________________

## Problem Statement

The Vx compiler's topology system (spawn, transfer, memory spaces, accessibility
checks, Dijkstra-based transfer cost routing) had minimal test coverage. Several
features existed in the codebase but lacked `.vx` integration tests, and some
parser features (NPU Slice syntax) were defined in the AST but never wired up.

______________________________________________________________________

## Changes Made

### 1. Unit Tests Added (45 new, total: 111 → 156)

#### `src/arch.rs` — TransferCostGraph (+19 tests)

- **Transfer costs**: same-space (zero), direct hop (CPUDRAM→NPUHBM, cost 50),
  multi-hop (CPUDRAM→LocalSRAM, cost 60 via NPUHBM)
- **NIC shortcut**: NPU→NicRam→RemoteHbm (cost 25) cheaper than CPU→RemoteHbm
- **Accessibility matrix (positive)**: GPU/ANE/AMX see CPUDRAM, NPU sees NPUHBM,
  AccCore sees LocalSRAM, CpuAvx512/CpuNeon see CPUDRAM
- **Accessibility matrix (negative)**: AccCore cannot reach NPUHBM, GPU cannot
  reach LocalSRAM or NPUHBM, NPU cannot reach LocalSRAM
- **Default memory**: CpuAvx512/CpuNeon → CPUDRAM, Slice → NPUHBM,
  Current → panics

#### `src/ast/types.rs` — Type & Topology (+7 tests)

- `kind()` mapping for all topology variants
- `is_same_kind()` for NPU[0] vs NPU[3]
- `substitute()` replaces Topology::Current with concrete topology
- Tensor topology extraction

#### `src/codegen/lower/mod.rs` — topology_to_i32 (+3 tests)

- Exhaustive mapping: CPU=0, NPU=100+idx, AccCore=200+idx, AMX=300, ANE=400,
  GPU=500, CpuAvx512=600, CpuNeon=700, Current=0, Slice=900
- Non-numeric NPU index fallback

#### `src/parser/mod.rs` — Parser topology (+12 tests)

- All 9 static topology variants
- NPU[index] and AccCore[index] with numeric indices
- NPU[0..4] → Topology::Slice parsing (new feature)
- Error cases: unknown variant, missing NPU index
- Memory space: all 5 variants + unknown error
- Type parsing: Tensor, Tensor<i64>, Pinned\<i32, GPU>, Ref\<f32, NPU_HBM>

#### `src/registry.rs` — ImmutableGlobalRegistry (+7 tests)

- Empty, single-def, valid dependency chain builds
- Mutual cycle and self-referential cycle detection
- Unresolved dependency errors
- Module index grouping

### 2. Integration Tests (.vx files)

#### Pass tests (consolidated into unified files)

| File | Tests |
|------|-------|
| `topology_spawn.vx` | GPU/ANE unified access, AMX, CpuAvx512, CpuNeon, nested spawn, NPU Slice, dynamic NPU index |
| `topology_transfer.vx` | NIC_RAM transfer, Remote_HBM multi-hop, CPUDRAM↔LocalSRAM round-trip |
| `topology_current_comparison.vx` | Topology::Current resolution via print-ast |

#### Fail tests (one error scenario per file)

| File | Tests |
|------|-------|
| `acccore_access_dram.vx` | AccCore cannot access CPU DRAM |
| `cpuavx512_access_cpu_dram.vx` | CpuAvx512 distinct from CPU |
| `pinned_cross_topology.vx` | Pinned\<T, NPU> inaccessible from GPU |

### 3. Features Implemented

#### NPU Slice Syntax Parsing (`src/parser/types.rs`)

**Before**: `Topology::NPU[0..4]` was parsed as `Topology::NPU(Range(0, 4))` — wrong.
**After**: Parser detects `Range` expression inside `NPU[expr]` and converts to
`Topology::Slice(NPU(start), start, end)`.

```rust
// In parse_topology(), NPU branch:
if let Expr::Range(RangeExpr { start, end, .. }) = expr {
    Ok(Topology::Slice(
        Box::new(Topology::NPU(start.clone())),
        start,
        end,
    ))
} else {
    Ok(Topology::NPU(Box::new(expr)))
}
```

#### Dynamic NPU Index Bug Fix (`src/sema/expr.rs`)

**Bug**: `spawn on(Topology::NPU[i])` inside a `for i in 0..4` loop failed
because the sema validated the index expression `i` *after* switching to NPU
context, where `i` is undefined.

**Fix**: Move topology index validation *before* the context switch in
`check_spawnon_expr`. The index expression belongs to the outer (caller) scope.

```diff
-                let prev_top = self.active_topology.clone();
-                ...
-                self.active_topology = actual_top.clone();
-                ...
-                self.push_scope();
-                // Validate topology expression if it contains one
-                match top {
+                // Validate topology index expressions BEFORE switching context
+                match &mut actual_top {
                     Topology::NPU(expr) | Topology::AccCore(expr) => {
                         let _ty = self.check_expr_type(expr);
                     }
                     ...
                 }
+                let prev_top = self.active_topology.clone();
+                ...
+                self.active_topology = actual_top.clone();
+                ...
+                self.push_scope();
```

______________________________________________________________________

## Verification

All changes verified with:

- `cargo test --lib` — 156 unit tests pass
- `cargo test` — full suite (156 unit + 45 integration tests) all green
- Individual `.vx` file testing with `vxc --action emit-mlir` and `not vxc`

## Commits

| Hash | Description |
|------|-------------|
| `ea57667` | Topology unit tests (arch, types, codegen) |
| `501bf9d` | Parser/registry tests + .vx integration tests |
| `adef46f` | Consolidate topology .vx tests into unified files |
| `8e1dfc4` | Remaining accessibility & transfer path tests |
| `aa4bddf` | Slice parsing, dynamic NPU fix, Pinned fail test |
