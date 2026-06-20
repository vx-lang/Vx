# Language Documentation Test Coverage Analysis

This document outlines the strategy for applying the `test_skills.md` guidelines to the features documented in `docs/lang/`. We audited the language specification (`syntax.md`, `types.md`, `semantics.md`) against the actual compiler implementation and test suite.

## 1. Goal Description

Ensure that every language feature documented in the specification has a corresponding presence in the codebase and test suite, following the "Feature Coverage Triangle" (unit, pass, fail). During the audit, we also discovered several features that are documented but not yet implemented in the parser/AST.

## 2. Open Questions & User Review Required

> [!WARNING]
> **Specification vs. Implementation Gaps Discovered**
> The following features are extensively documented in `docs/lang/` but are completely absent from the compiler's source code (parser, AST, or Sema).
>
> 1. `unroll across(Topology)` (Syntax §6)
> 1. `safe fn` FFI annotation (Syntax §7)
> 1. `HardwareState` and `try_pin` (Types §4)
> 1. `effects(DataMovement(...))` function annotations (Types §5)
> 1. Topology Allocation e.g., `Tensor::new() with Topology::NPU` (Types §6)
>
> **Action Required**: Do you want me to *implement* these missing features in the parser/AST/Sema, or just document them as `TODO`s in the `docs/lang/` files for now and focus strictly on testing the implemented features?

## 3. Proposed Changes (Testing)

The following features *are* implemented in the compiler but lack the required pass/fail test coverage as per our guidelines.

### Missing Tests for Implemented Features

#### [NEW] \[topology_signature_mismatch.vx\](file:///Users/adityak/go/Vx/tests/frontend/fail/topology/topology_signature_mismatch.vx)

- **Feature**: Topology-Aware Function Signatures (`on Topology`).
- **Test Level**: Fail.
- **Scenario**: Calling an `on Topology::GPU` function from a CPU context.
- **Expected Error**: `Type error: Function '{}' requires topology '{:?}', but is called from '{:?}'`.

#### [NEW] \[topology_dynamic_index.vx\](file:///Users/adityak/go/Vx/tests/frontend/pass/topology_dynamic_index.vx)

- **Feature**: Dynamic Topology Index (`NPU[i]`).
- **Test Level**: Pass.
- **Scenario**: Using a `for` loop variable as the index for a `spawn on(Topology::NPU[i])` block to scatter work dynamically.

#### [NEW] \[unsafe_ffi_call.vx\](file:///Users/adityak/go/Vx/tests/frontend/fail/unsafe_ffi_call.vx)

- **Feature**: Foreign Function Interface (FFI) Safety.
- **Test Level**: Fail.
- **Scenario**: Calling an `extern` function without wrapping it in an `unsafe { }` block.
- **Expected Error**: `Call to unsafe function '{}' is unsafe and requires unsafe function or block`.

#### [NEW] \[topology_nic_transfer.vx\](file:///Users/adityak/go/Vx/tests/frontend/pass/topology_nic_transfer.vx)

- **Feature**: Multi-Hop NIC Transfers.
- **Test Level**: Pass.
- **Scenario**: Transferring a tensor from `Memory::CPUDRAM` -> `Memory::NIC_RAM` -> `Memory::RemoteHbm` to exercise the complex `TransferCostGraph` pathing.

## 4. Verification Plan

### Automated Tests

- Run `cargo run --bin vx-format -- <file>` on the new `.vx` tests.
- Run `source config.local && cargo test --test compile_test` to verify the FileCheck conditions for both the pass and fail tests match the exact compiler output strings.
