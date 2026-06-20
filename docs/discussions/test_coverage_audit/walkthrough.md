# Language Specs Test Coverage

## Overview

Based on the `test_skills.md` guidelines, we conducted an audit comparing the official Vx language specifications (`docs/lang/`) against the compiler's actual implementation and test suite coverage.

During the audit, we categorized the features into three buckets:

1. **Implemented and Tested**: Most features (like `spawn on`, `Verified<T>`, `Pinned<T>`, macros, math).
1. **Implemented but Untested**: Features we have in AST/Sema but lacked pass/fail integration tests.
1. **Unimplemented**: Features described in the specs that don't exist in the compiler yet.

## What Was Completed

### 1. New Test Coverage

We added integration tests for 4 features that were implemented but untested:

- **Topology Mismatch (Fail)**: \[topology_signature_mismatch.vx\](file:///Users/adityak/go/Vx/tests/frontend/fail/topology/topology_signature_mismatch.vx) verifies that the compiler accurately rejects calls to `on Topology` functions from an invalid context (e.g., calling an NPU-bound function from the CPU).
- **Dynamic Topology Indexing (Pass)**: \[topology_dynamic_index.vx\](file:///Users/adityak/go/Vx/tests/frontend/pass/topology_dynamic_index.vx) ensures `spawn on(Topology::NPU[i])` works correctly with dynamic loop indices, verifying `i` is properly resolved across scoping boundaries.
- **Unsafe FFI Enforcement (Fail)**: \[unsafe_ffi_call.vx\](file:///Users/adityak/go/Vx/tests/frontend/fail/unsafe_ffi_call.vx) guarantees the compiler enforces `unsafe { ... }` blocks when invoking foreign functions.
- **Multi-Hop Transfers (Pass)**: \[topology_nic_transfer.vx\](file:///Users/adityak/go/Vx/tests/frontend/pass/topology_nic_transfer.vx) verifies type inference and multi-hop paths like `Memory::NIC_RAM` -> `Memory::Remote_HBM`.

### 2. Specification Accuracy (Disclaimers)

To prevent confusion between planned features and current capabilities, we added explicit GitHub alerts (`> [!WARNING] Experimental / Unimplemented Feature`) to the documentation for the following 5 upcoming features:

- **`unroll across(Topology)`**: Missing from parser. (\[docs/lang/syntax.md\](file:///Users/adityak/go/Vx/docs/lang/syntax.md))
- **`safe fn` FFI keyword**: Missing from parser. (\[docs/lang/syntax.md\](file:///Users/adityak/go/Vx/docs/lang/syntax.md))
- **`HardwareState` & `try_pin`**: Missing from typestates. (\[docs/lang/types.md\](file:///Users/adityak/go/Vx/docs/lang/types.md))
- **`effects(...)`**: Missing from function signatures. (\[docs/lang/types.md\](file:///Users/adityak/go/Vx/docs/lang/types.md))
- **Topology Allocation (`with Topology`)**: Missing from parser. (\[docs/lang/types.md\](file:///Users/adityak/go/Vx/docs/lang/types.md))

## Verification Results

- `vx-format` ran successfully on all new `.vx` tests.
- `cargo test --test compile_test` successfully passed with 13/13 test runners succeeding, verifying our new frontend pass/fail `.vx` tests match their `FileCheck` expected outputs.
