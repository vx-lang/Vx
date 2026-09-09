# Diagnostic index

Every diagnostic the Vx compiler can emit, with the code it reports and what it means.

Codes are grouped by the compilation stage that raises them, and the group is readable off the
number: `E1xxx` is the parser, `E3xxx` the type checker, `E6xxx` the placement and capacity rules,
and so on. A `W` prefix is a warning rather than an error.

> This file is generated from `src/diagnostic.rs` by `scripts/tools/gen_error_index.py`. Edit the
> doc comments on the codes there, not this file.

## Contents

- [Warnings](#warnings) — `W1001`–`W1031` (23 codes)
- [Parser Errors](#parser-errors) — `E1001`–`E1013` (13 codes)
- [Name Resolution Errors](#name-resolution-errors) — `E2001`–`E2007` (7 codes)
- [Type Errors](#type-errors) — `E3001`–`E3029` (29 codes)
- [Borrow/Ownership Errors](#borrowownership-errors) — `E4001`–`E4005` (5 codes)
- [Safety Errors](#safety-errors) — `E5001`–`E5002` (2 codes)
- [Topology/Hardware Errors](#topologyhardware-errors) — `E6001`–`E6028` (28 codes)
- [Tensor/Math Errors](#tensormath-errors) — `E7001`–`E7004` (4 codes)
- [Contract/Verification Errors](#contractverification-errors) — `E8001`–`E8002` (2 codes)

## Warnings

Reported without stopping the compile. A warning means the program is accepted but something in it is probably not what was intended.

| Code | Meaning |
| --- | --- |
| `W1001` | Unused variable binding |
| `W1002` | Unused function definition |
| `W1003` | Unreachable code after return, break, or continue |
| `W1004` | Unnecessary mutable binding (`let mut x` where x is never reassigned) |
| `W1005` | Shadowed variable in same scope |
| `W1006` | Redundant borrow (`&&x`) |
| `W1007` | Implicit type widening in `as` cast |
| `W1008` | Empty match arm body |
| `W1009` | Unused function parameter |
| `W1010` | Unnecessary unsafe block (no unsafe ops inside) |
| `W1013` | Redundant `as` cast to same type |
| `W1014` | Narrowing cast loses precision |
| `W1020` | Immediately dereferenced borrow (`*&x`) |
| `W1022` | Transfer to same memory space (no-op) |
| `W1023` | Spawn on Topology::Current (no-op) |
| `W1024` | Implicit cross-topology transfer inserted via a `Relocatable` impl (a real data movement happens silently at the use site; write the transfer explicitly to silence). `Relocatable` answers "may this value move implicitly?" and is keyed on a user type. That is a different question from "what code moves bytes across this hardware edge?", which is `impl Transfer<Memory::A, Memory::B> for Topology::X`. Both were called `Transfer` before Vx#353. |
| `W1025` | Use of a user-defined topology with no registered descriptor (not declared via `Topology <Name> { ... }` and not registered by a plugin). Often a typo of a built-in; defaults to host-like placement. |
| `W1026` | A user-defined topology's memory is unreachable from the host (no transfer path), so data can never be moved to it. See the topology coherence check. |
| `W1027` | A declared `relaxed` transfer edge does not preserve visibility (the seam engine shows a consumer may read stale data). See the topology coherence check. |
| `W1028` | The working set of a memory space exceeds `capacity`, but the space is declared `overcommit`, so the cumulative-budget errors (E6010, and the cross-call E6027/E6028) are downgraded to this warning. |
| `W1029` | A tensor placed in a memory space that declares a `capacity` has a *dynamic* (non- literal) shape, so the capacity check (E6009/E6010) could not run — the placement is unverified. Silence by making the shape static, or bounding it (see P1-1). Emitted only when the destination space actually declares a capacity. |
| `W1030` | A topology's device index is not a compile-time constant (`GPU[i]` for a runtime `i`), so it cannot be resolved to a device instance and falls back to index 0. Every such spawn therefore targets the same device. Vx models one representative device per declared kind (#284), so a fleet program should index with constants or const generics. |
| `W1031` | A proof obligation could not be discharged because no SMT solver was available, so the property is **unverified** rather than proved. Distinct from W1027, which means the solver ran and found a violation. Emitted only under `VX_ALLOW_UNVERIFIED`; without it a missing solver is an error, because silence used to be indistinguishable from success (Vx#374). |

## Parser Errors

Raised while turning source text into an AST. The program is not syntactically valid Vx.

| Code | Meaning |
| --- | --- |
| `E1001` | Unexpected token |
| `E1002` | Unexpected end of file |
| `E1003` | Expected identifier |
| `E1004` | Expected type |
| `E1005` | Expected expression |
| `E1006` | Unclosed delimiter (paren/brace/bracket) |
| `E1007` | Missing semicolon |
| `E1008` | Missing comma |
| `E1009` | Invalid operator |
| `E1010` | Unknown topology variant |
| `E1011` | Unknown memory space |
| `E1012` | Unknown element type |
| `E1013` | Invalid macro invocation syntax |

## Name Resolution Errors

Raised when a name cannot be resolved to a declaration, or resolves to something of the wrong kind.

| Code | Meaning |
| --- | --- |
| `E2001` | Undefined variable |
| `E2002` | Undefined function |
| `E2003` | Unknown enum |
| `E2004` | Unknown enum variant |
| `E2005` | Unknown struct field |
| `E2006` | Module does not export function |
| `E2007` | Method not found on type |

## Type Errors

Raised by the type checker. Vx performs no implicit numeric conversion, so many of these are mismatches that a language with coercion would have silently accepted.

| Code | Meaning |
| --- | --- |
| `E3001` | Type mismatch in variable declaration |
| `E3002` | Type mismatch in return |
| `E3003` | Type mismatch in function argument |
| `E3004` | Type mismatch in binary operation |
| `E3005` | Type mismatch in relational operation |
| `E3006` | Type mismatch in logical operation |
| `E3007` | If branch type mismatch |
| `E3008` | Enum payload type mismatch |
| `E3009` | Enum payload arity mismatch |
| `E3010` | Function argument count mismatch |
| `E3011` | Unsupported cast |
| `E3012` | Type mismatch in struct field initialization |
| `E3013` | Missing struct field in initialization |
| `E3014` | Range type mismatch |
| `E3015` | Trait not implemented |
| `E3016` | Generic type deduction failure |
| `E3017` | Closure argument count or type mismatch |
| `E3018` | An array literal whose elements are not scalars, or which is empty. An array literal lowers to `tensor.from_elements`, whose element type must be a scalar, so `[a, b]` for tensors -- placed or not -- has nothing to lower to, and an empty literal has no element type to give it. Both used to be accepted by the checker (the element type silently stayed at its `f32` default) and then crash codegen with an internal error rather than a diagnostic. See Vx#354. |
| `E3019` | A `match` arm whose integer literal cannot be represented in the scrutinee's type. The arm can never be selected, so the program does not mean what it says. Codegen used to parse the literal with a zero fallback, which turned an unrepresentable arm into a comparison against 0 -- so the arm fired for scrutinee 0, the most common value there is, with no diagnostic. |
| `E3020` | A `match` used as a value that no arm is guaranteed to match. A value-position match must produce a value on every path, so it needs a wildcard arm or must name every variant of its scrutinee's enum. Without that the fall-through edge has no value to carry, and codegen used to paper over it by evaluating the whole match to a constant zero. |
| `E3021` | An enum variant whose payload is a tensor. A payload is stored into the variant's tagged-union slot with `llvm.insertvalue`, which takes primitive operands, and a tensor is a memref descriptor. The AST path dropped such a payload silently: the construction emitted the tag and nothing else, so a program carrying a tensor through an enum compiled, ran, and lost it with no diagnostic. A struct field holding a tensor is the same representational gap in another position. |
| `E3022` | An `extern` function whose signature mentions a tensor. A tensor is a memref, and lowering expands a memref parameter into the seven scalars of its descriptor -- allocated pointer, aligned pointer, offset, and a size and stride per rank. So `fn c_take(t : Tensor<f32, [?, ?]>) -> i32` declares a C symbol taking seven arguments, which is not a signature anyone writes on the C side; the call links by name and passes something the callee never agreed to. Take a raw pointer and build the tensor in Vx (`Tensor<f32, [?, ?]>::from_ptr_2d`), which is what the corpus already does. |
| `E3023` | A shaped tensor initialized from a scalar. `let a : Tensor<f32, [128, 64]> = 1.0` allocated and filled a whole buffer from something that reads as an assignment, and the two codegen paths disagreed about it: the AST path emitted the allocation and a `linalg.fill`, the flat path kept the bare constant and handed an `f32` to a call expecting a memref. `Tensor<T, [..]>::fill(v)` is the spelling. The rank-0 wrap (`Tensor<f32, []> = 1.0`) is a different thing and stays legal. |
| `E3024` | `.shape[i]` on a tensor. It answered on any value, not only a tensor, and typed its answer as a rank-0 tensor. `extent(i)` is the read of a run-time extent. |
| `E3025` | `extent(i)` with an index that is not a literal below the tensor's rank. Rank is static, so the index is checked here rather than read past the descriptor at run time. |
| `E3026` | A placement query (`.topology()`) the checker cannot decide. Placement is a fact of the receiver's type, compared with `Some(Topology::..)` or `None`; it has no run-time value. |
| `E3027` | A function whose return type is a closure. A closure value points into the frame that made it, so it cannot outlive that frame yet. |
| `E3028` | A function with a non-void return type whose body can complete without returning. Reported here rather than left to codegen, where it surfaced as an MLIR verifier message naming an operation, with no source location. |
| `E3029` | A type name in a signature that names no declaration. An unknown name in type position parses as a user nominal, so without this a typo -- or a type constructor removed from the language -- compiled silently and did nothing. |

## Borrow/Ownership Errors

Raised by the borrow checker and the linear-type rules. These rule out use-after-move, aliasing violations and lifetimes that outlive what they point at.

| Code | Meaning |
| --- | --- |
| `E4001` | Use of moved or consumed linear variable |
| `E4002` | Cannot access mutably borrowed variable |
| `E4003` | Cannot borrow as mutable (already immutably borrowed) |
| `E4004` | Cannot borrow (already mutably borrowed) |
| `E4005` | A returned reference escapes the function borrowing a function-local (dangling return) |

## Safety Errors

Raised where an operation needs an `unsafe` context and does not have one.

| Code | Meaning |
| --- | --- |
| `E5001` | Unsafe function call outside unsafe block |
| `E5002` | Unsafe memory operation outside unsafe block |

## Topology/Hardware Errors

Raised by the placement and capacity rules — the checks that make Vx different from a single-address-space language. A value in the wrong memory space, a region on the wrong device, a working set that does not fit, or a transfer with no declared route.

| Code | Meaning |
| --- | --- |
| `E6001` | Topology mismatch in function call |
| `E6002` | Cannot transfer between memory spaces (no hardware path) |
| `E6003` | A value is used from a topology that cannot see the memory space it lives in. The diagnostic names the value's space, the visible set of the topology reading it, and the cost of the transfer that would fix it -- so a misplaced handoff (an un-transferred KV cache in a disaggregated prefill/decode split, say) is a compile error that carries its own remedy. A `managed: cached` space the topology can reach across a declared seam is coherent in hardware and is not reported here. |
| `E6004` | Transfer violates the boundary contract at a seam (per-seam local-completeness / soundness obligation is `sat`; a stale read can violate the contract). |
| `E6005` | A user-defined topology declaration is incoherent: it cannot see its own default memory space (`default_space ∉ visibility`). See the topology coherence check. |
| `E6006` | A `Memory` declaration's `within:` hierarchy forms a cycle (a space contains itself). |
| `E6007` | A `Memory` sub-space's `capacity` exceeds its parent's capacity (a child cannot be larger than what contains it). |
| `E6008` | A `Memory` declaration has a non-positive `capacity`, `bandwidth`, or `granule`. |
| `E6009` | A statically-shaped tensor placed in a memory space exceeds that space's `capacity`. |
| `E6010` | The working set placed in a memory space (the sum of its tiles) exceeds `capacity`. Downgraded to W1028 when the space is declared `overcommit`. |
| `E6011` | A sub-space's `scope` is broader than its parent's (locality must narrow down `within:`). |
| `E6012` | The same `Memory` or `Topology` name is declared by two compilation inputs (e.g. a `--machine` file and the program). Declarations are name-keyed, so one would silently shadow the other and the machine model in force would depend on load order (#281). |
| `E6013` | A declared `transfer` edge carries an explicit cost *and* has one derivable from its endpoints' `bandwidth:` figures. An edge gets exactly one cost source, because two answers to "what does this hop cost" is not a model: the compiler routed by the declared number and reported the derived one, and nothing detected the disagreement. |
| `E6014` | A program stages through host memory while a machine model is in force, and no host was declared. `--machine` describes an accelerator and says nothing about the machine it hangs off, so the host end of that seam was being reasoned about without anything describing it. `--host <file>` names one; `--host default` names the machine compiling the program. A host declares no capacity -- host memory is virtual, and a hard limit would reject programs that page rather than fail -- so this is about the host being *stated* rather than assumed, not about a budget. |
| `E6015` | A structurally invalid transfer lowering (`impl transfer A -> B { ... }`): the same edge implemented twice in one compilation (which one is in force would be load order), or a lowering with no functions (an empty body cannot move anything, and accepting it would make `impl transfer` an inert annotation rather than code). |
| `E6016` | A `Topology` or `Memory` declaration whose identity cannot be relied on. Two forms: the declared name shadows a built-in topology (every use of `Topology::<Name>` resolves to the built-in, so the declaration is silently ignored -- including its `arch:`); or two declared names collide on one dispatch id (custom ids are derived from the name by hashing), in which case which declaration is in force would be hash-iteration order -- observed as the same program getting a device image on some runs and not others for a topology, and as a `transfer` carrying the other space's capacity and granule for a memory space. |
| `E6017` | A misuse of the `raw::` transfer-lowering primitives (Vx#353 A2): a `raw::` call outside an `impl transfer` body, an unknown primitive name, a tile argument that is not a bare parameter name (the primitives are indexed, not addressed), a store into a tile not held by `&mut`, or a wrongly typed index/value. |
| `E6018` | A `raw::` bounds obligation (`0 <= index < extent`) that could not be proven. Prove it with a loop bound or invariant the SMT prover can see, or assert it in an `unsafe` block -- which records the obligation as asserted-not-proven, the same standing an unverified `spec:` figure has. |
| `E6019` | `raw::barrier()` anywhere but a top-level statement of the lowering body. The barrier's contract requires every lane to reach it; under a conditional or a loop that cannot be guaranteed syntactically, so it is rejected outright (conservative by design -- restructure the body so the barrier is unconditional). |
| `E6020` | `raw::async_copy` in a lowering for an edge no declared topology equips with a copy engine. The capability lives in the machine file (`transfer A -> B copy_engine`); using an absent primitive is a compile error, not a fallback. |
| `E6021` | A violation of the async/synchronization discipline in a lowering body: a destination read while an `async_copy` into it is still outstanding, a body that ends with copies no `async_wait` covers, or a lowering for a synchronizing edge whose body does not end with `raw::barrier()` (the seam obligation of `hir/seam.rs`: a relaxed publication makes a stale read reachable). |
| `E6022` | An `impl transfer` lowering whose edge endpoints are not visible to a topology that declares the edge -- the lowering would execute on a part that cannot address the spaces it moves bytes between (contract constraint C6). |
| `E6023` | An `impl transfer` lowering whose declared tile shape is not the shape the transfer at hand actually moves. A lowering is selected by edge, so nothing else relates the two, and the `raw::` primitives take their extents from the declaration: a smaller declaration copies part of the tile and leaves the rest uninitialised, a larger one stores past the end (observed as a SIGSEGV). |
| `E6024` | A proof obligation could not be discharged because no SMT solver was available. Fails the compilation by default: an undischarged obligation is not a proved one, and treating the two alike is what let a missing z3 certify every seam in silence (Vx#374). Set `VX_ALLOW_UNVERIFIED=1` to downgrade this to W1031 and compile anyway. |
| `E6025` | A placement naming a location the machine does not have: a memory space no declared topology holds, written either as the space or as the device that would hold it. The derivation between the two spellings has a like-named fallback, so an undeclared name resolves to a space that exists only in the placement that mentions it. |
| `E6026` | A tensor whose element type the target hardware cannot represent, placed on it anyway. The machine model states what a device has (`dtypes: [f32, f16, ...]`); this is the check that a placement stays inside it. An H100 has no fp4, so an fp4 tensor placed on one asks for silicon that is not there -- and the placement is in the type, so the question is answerable here rather than at a kernel launch on the machine that lacks the type. Only fires against a topology that declares `dtypes:`. An undeclared machine constrains nothing, which is what keeps every machine file written before the field kept working. |
| `E6027` | A working set that overflows a space only across call boundaries: the peak along some call path -- what each caller still holds when it calls, plus the deepest callee's own peak -- exceeds the space's declared capacity, while every function on the path fits by itself (that case is E6010's). Computed by folding per-function capacity summaries over the call graph, after the per-function checks. Downgraded to W1028 when the space is declared `overcommit`. |
| `E6028` | A recursive cycle that places tiles in a space with a declared capacity. The recursion depth is not known at compile time, so the true peak is unbounded and the placement is refused conservatively. Downgraded to W1028 when the space is declared `overcommit`. |

## Tensor/Math Errors

Raised on tensor shapes and numeric operations, including shape mismatches that are decided at compile time.

| Code | Meaning |
| --- | --- |
| `E7001` | Matmul dimension mismatch |
| `E7002` | Matmul element type mismatch |
| `E7003` | Reshape arithmetic mismatch |
| `E7004` | Non-differentiable return type (autodiff) |

## Contract/Verification Errors

Raised when a `requires`, `ensures` or `invariant` clause cannot be discharged, or when a seam obligation is left unproven.

| Code | Meaning |
| --- | --- |
| `E8001` | Cannot prove postcondition |
| `E8002` | Comptime assert failed |

______________________________________________________________________

113 diagnostics.
