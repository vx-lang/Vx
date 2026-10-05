# Drop semantics: freeing what a program owns when its owner goes away

**Status:** phases 0 to 3 done, 2026-10-06: every program frees its tensors at their drops, on
both code generators, and `vx-free-heap-buffers` is gone. Placed tensors and `spawn` regions
(phase 5) and the `Drop` trait (phase 4) remain. Proposed 2026-10-03. Tracking issue: Vx#1041, one issue per phase (Vx#1049, then Vx#1042 to Vx#1046). Decides Vx#495.

______________________________________________________________________

## 0. Summary

Vx frees heap memory today in three unrelated ways, and none of them uses what the language
already knows about ownership:

- **compiler-allocated tensors** are freed by MLIR's `buffer-deallocation-pipeline`, run near the
  end of lowering (`vx-free-heap-buffers`). It reconstructs ownership from the IR, frees after the
  last use, and skips any module it cannot follow;
- **`transfer` results** are freed by a second pass, `placeTransferFrees`, after their last use;
- **standard-library values** (`Vec`, `Box`, `String`, files, sockets) are freed by hand, with
  `v.free()`, or not at all.

The first of these keeps producing bugs, because it rebuilds facts the checker had from IR that has
lost them: a buffer read back through a cell (#993), a placeholder for an early return (#1014), a
buffer in a struct field (leaked by design), any program that places data in a memory space
(leaked wholesale). Each fix has been either "skip the module and leak" or "teach the analysis one
more shape".

This plan gives Vx Rust's model instead: **a value that owns resources is dropped when its owner
goes out of scope, unless it was moved first.** The decisions:

1. **Views are borrows, checked first.** A row `q[i]`, a tensor field read or a reshape borrows
   its owner, as `&q` does: the owner cannot be moved or written while the view is used, and the
   view cannot outlive the owner's block. Nothing below is sound until the borrow checker
   enforces this (phase 0).
1. **Memory is freed after its last use; other drops wait for the end of the scope.** An owner
   whose drop only frees memory (a tensor, a `Vec` with no `Drop` of its own) is freed after the
   last use of the owner *and of every view of it*. The borrow checker is what makes that safe: a
   free too early is a borrow checker bug. A type with a `Drop` implementation is dropped at the
   end of its block, in reverse declaration order, because its drop has effects whose timing is
   part of the program (`RefCell` guards, locks, files; A19 in `core_library.md`). So is an
   owner a raw pointer was taken from (`as_ptr`, `from_ptr`), which no borrow checker sees.
1. **The checker decides, the code generators obey.** The checker already tracks moves per scope
   (`BorrowCx.moved_vars`). It gains the step it is missing, "which owners are still live when this
   scope ends", and writes explicit drop operations into the program. Both code generators lower
   those; neither infers ownership from IR.
1. **Owners and views are told apart by the checker, not by the type.** A row `q[i]`, a
   `tensor_view_2d` and a `reshape` have type `Tensor` but must never be freed. The checker
   records where each tensor value came from; only a value it can prove is an owner is dropped,
   and anything else is treated as a view. The seed exists: `owned_tensors` and
   `is_tensor_construction`. A separate view type (#400) can replace this later; it is not needed
   first.
1. **A conditional move uses a drop flag.** A value moved on some paths and not others gets a
   run-time boolean, set when it is moved and tested where it would be dropped, as Rust does.
1. **A by-value parameter is owned by the callee.** Today the checker marks the argument moved
   while the caller still frees it at run time. Under this plan the callee drops it, so moved means
   owned in both places.
1. **`Drop` is a trait.** `impl Drop for T { fn drop(self : &mut Self) }` runs before T's fields
   are dropped. `Vec`, `Box`, `String` and the handle types implement it, and their `free()` methods
   go. `core::mem::drop`, `forget`, `needs_drop` and `ManuallyDrop` get their real meaning.
1. **MLIR's deallocation leaves when drops are complete.** The transition runs both schemes behind
   a flag, compares them, then removes `vx-free-heap-buffers`. `placeTransferFrees` becomes the
   drop of a placed tensor.

______________________________________________________________________

## 1. What exists today

**Moves.** `consume(name)` (`src/hir/env.rs`) marks a name moved in the scope that declared it, and
a read of a moved name is refused. Branches join by union: a value moved on any path is moved
(`moved_snapshot`, `restore_moved`, `union_moved`; `branch_end` in `src/hir/check/control.rs`).
Loops refuse moving an outer value on a path that goes round again (E4001, `settle_loop_moves`).
`pop_scope` throws the scope's moved set away, so nothing ever asks which owners were still live:
that is the hook this plan adds.

**Linear types** (`Type::is_linear`): `Tensor`, `Struct`, `Enum`, and the checker-only wrappers
(`Ref`, `Pinned`, `Verified`). `impl Copy` opts a type out. Scalars, pointers and closures are not
linear.

**Gaps the plan has to close.**

- A field or index read of a linear type (`let x = s.field`, `let r = q[i]`) moves nothing and is
  not tracked: an untracked alias of the same type.
- By-value parameters: "moved" in the checker, "still owned by the caller" in codegen.
- Struct fields holding tensors go through `unrealized_conversion_cast`, which disables freeing.
- Tensors allocated inside device regions are never freed unless they are small and static.
- `.drop()` with no arguments consumes its receiver, whatever `drop` means for that type.

**Where tensors are allocated.** Flat path: `Opcode::TensorAlloc` from `src/hir/flatten.rs`
(constructors, initializer lists, array literals), and the elementwise, transpose, map and `@`
outputs in `src/codegen/flat/emit/`. AST path: the corresponding sites in
`src/codegen/lower/expr.rs` and `mod.rs`. Statically shaped returns go through a slot the caller
allocates on its stack, so they need no free.

## 2. The design

### 2.1 Owners and views

The checker gives every tensor-valued expression a provenance:

| Provenance | Examples | Dropped? |
| --- | --- | --- |
| **owned** | `Tensor<T>([n])`, `::new()`, initializer and array literals, `a @ b`, `a + b` on tensors, `transfer(..)`, a call returning a tensor by value, a by-value parameter | yes, by its owner |
| **view** | `q[i]`, `tensor_view_2d(..)`, `reshape(..)`, a field read, anything through a reference | never |

A local takes the provenance of its initializer. Assigning a view to a local declared from an owner
(or the reverse) is an error until a view type exists, because the local could not then be dropped
correctly. This is the one place the plan restricts programs that compile today; the fuzzers and
the corpus will measure how often it fires (phase 1).

### 2.2 Drop points

For an owner that only holds memory, the drop point is after the last use of the owner or of any
view of it, on each path. For a type with `Drop`, and for an owner a raw pointer was taken from,
it is the end of its block. In both cases the checker computes the owners declared in a block
that are not definitely moved:

- **definitely live**: dropped unconditionally, at the drop point above;
- **maybe moved** (moved on some paths reaching the end): dropped under a drop flag;
- **definitely moved**: not dropped.

The same applies on early exits: `return`, `break` and `continue` drop the owners of every block
they leave, innermost first, before jumping. A value being returned is moved, so it is not dropped.

Temporaries: an owned value produced in an expression and not bound or moved (`print(a @ b)`) is
dropped at the end of the statement.

The checker records drops as explicit statements in the AST (a `Drop(name)` statement and a
`DropFlag` declaration), so each code generator only lowers what it is given. This is Rust's "drop
elaboration", done once, ahead of both code generators.

### 2.3 What dropping does

- **A tensor in host memory**: `memref.dealloc` of its buffer.
- **A tensor placed in a memory space**: `vx.free`, which lowers to a host dealloc or to
  `vx_plugin_free` as today.
- **A struct**: its `Drop` implementation if it has one, then each owning field, in declaration
  order.
- **An enum**: the payload of the active variant.
- **A type with nothing to drop** (scalars, pointers, `Copy` types): nothing. `TYPE_NEEDS_DROP`
  (src/gid.rs) is set on exactly the types for which dropping does something, so the rest cost
  nothing.

### 2.4 Moves the checker must start tracking

- **Moving out of a field** (`let x = s.field` where the field owns): either a move of the field
  (a partial move, after which `s` may only have its other fields used and is dropped field by
  field), or refused. The plan starts with **refused** (an error suggesting a reference or a
  swap), and adds partial moves only if programs need them.
- **Indexing a tensor of owners** does not arise yet: `Tensor` elements are scalars.

### 2.5 Calls, returns and closures

- **By-value argument**: moved into the callee, which owns and drops it. The caller does not free
  it after the call.
- **By-reference argument**: a view in the callee; nothing changes.
- **Return**: the returned value moves to the caller. A slot return writes into the caller's
  buffer as today.
- **Closure capture of an owner**: moves it into the closure's environment, which then owns it and
  drops it when the closure is dropped. Closures cannot escape their frame today (E4005), which
  keeps this simple.
- **`spawn` regions**: a captured owner stays owned by the enclosing function; a tensor allocated
  inside a region is dropped at the end of the region like any block.

## 3. Phases

Each phase is one or a few PRs, and every phase leaves the compiler working.

**Phase 0: views are borrows.** `let r = q[i]`, a tensor field read and a reshape record a
borrow of the owner, mutable when the view is declared `mut`. The existing errors then apply:
moving the owner while the view is used (E4007), writing it (E4009), and letting the view
outlive the owner's block or function (E4005). Assigning a view to an owning tensor
(`keep = q[1]`) is refused until it can mean a copy; today it crashes both code generators.

**Phase 1: provenance and drop points in the checker, reported only.** The checker classifies
owners and views (§2.1), computes drop points and drop flags (§2.2), and refuses moving out of a
field (§2.4). Nothing is emitted yet: a debug flag prints the drop points, and tests check them. The
new view/owner and field-move errors run against the test corpus and the fuzzers to measure what
they reject.

**Phase 2: tensor drops in both code generators, behind `-X drop=scope`.** Lower `Drop` and drop
flags for tensors (§2.3), flip by-value parameters to callee-owned, and drop temporaries. With the
flag on, `vx-free-heap-buffers` and `placeTransferFrees` are off. With it off, nothing changes.

**Phase 3: prove it, then make it the default.** The differential fuzzer gains a leak check (count
`malloc` and `free` at run time with a preload library, and require that a program frees all it
allocates) and runs under glibc's checking allocator. The corpus is compared both ways: no program
may leak more, crash, or print differently. Then the flag becomes the default and
`vx-free-heap-buffers` is deleted; `placeTransferFrees` becomes the drop of a placed tensor.

**Phase 4: the `Drop` trait.** `impl Drop for T`, drop glue for structs and enums, `TYPE_NEEDS_DROP`
set on the types that need it. `Vec`, `Box`, `String`, `File` and the sockets implement `Drop`;
their `free()` and `*_drop` functions are removed and their callers updated. `core::mem::drop`,
`forget`, `needs_drop` and `ManuallyDrop` get their real meaning, and `RefCell` guards (A19) become
possible.

**Phase 5: device regions and placed memory.** Tensors allocated inside device regions are dropped
on the device; placed owners use `vx.free`. This is where `useStackScratchIn` and the plugin free
paths are revisited.

## 4. Testing

- **Drop-point tests (phase 1):** for each kind of block and exit, the drop points the checker
  computes, printed by the debug flag.
- **Run-time counts (phases 2 to 4):** every program in `tests/backend/pass` frees everything it
  allocates beyond what the runtime itself keeps, on both code generators. The preload counter
  used for #1039 becomes a test helper.
- **Fuzzing:** the two existing generators, plus one that builds owners and moves them through
  calls, structs, branches and loops, each with a Rust twin. Run with the leak check and glibc's
  checking allocator.
- **Errors:** the new errors (a view assigned where an owner is expected, a move out of a field,
  use after a conditional move) each get a failing test.

## 5. Open questions

1. **Partial moves out of struct fields:** refuse first (this plan), or allow as Rust does?
1. **Reassigning an owner** (`t = Tensor<f32>([n])` when `t` already owns a buffer): drop the old
   value first, as Rust does. Planned as yes; it changes the meaning of programs that today
   overwrite without freeing.
1. **Drop order across a `spawn` boundary** when a region yields an owner: moved out to the
   enclosing function, like a return. To confirm in phase 5.
1. **A view passed by value** (`norm(q[i])` for `fn norm(v : Tensor<f32, [4]>)`): once the
   callee owns its by-value parameters, the caller must copy the view or the call is refused.
   Phase 0 measures how common this is.
1. **The view type (#400):** once it exists, provenance becomes a type distinction, and the
   restriction in §2.1 can go.
