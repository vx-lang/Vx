# Contracts for `extern` and `unsafe` code

Status: proposed. Issue: #1902. Tracking: #1870. Builds on section 7 of
`waiting_tasks_channels_and_task_errors.md` (the trusted base) and on W8, W9, M5 and admission in
that note.

In Vx, a signature is the whole contract (`docs/language_rules_for_a_parallel_compiler.md`). An
`extern` declaration breaks this rule today. It can carry only `safe` and doc comments
(`docs/lang/grammar.md`). Its signature does not say whether the function:

- blocks the thread (W8)
- keeps a pointer that it gets (M5, E4005)
- calls back into Vx (W9)
- is safe to call from many threads
- allocates memory (admission)
- writes through each pointer, or only reads

This note adds those facts to the declaration. A programmer who knows a library can state them
exactly. A programmer who does not know it writes nothing and gets safe defaults.

## 1. The rule: no clause means the worst case

**X1.** With no clause, the compiler assumes that an `extern` function:

- blocks its thread
- keeps each pointer that it gets
- calls back into Vx if it gets a function pointer
- must not run at the same time as another call into the same library
- writes everything that it can reach through its pointer arguments
- allocates an unknown amount of memory
- can stop the program

Each clause narrows one of these assumptions, and then the compiler can do more. So a missing
clause costs speed. It never costs safety. This is the escape hatch: write nothing.

**X2.** `safe fn` keeps its meaning: a call needs no `unsafe` block. The clauses say what the call
does, and `safe` says that the call cannot break memory safety when the clauses hold.

## 2. The clauses

<!-- vx-doctest: skip (proposed syntax) -->

```vx
strict extern "C" thread_safe {
  /// Reads up to `n` bytes into `buf`. Returns the count, or -1.
  fn vx_tcp_read(s : *mut i8, buf : *mut u8, n : u64) -> i64
      requires n > 0
      ensures return >= -1
      reads(s)
      writes(buf, n)
      blocks
      allocates(Memory::Host, 0);
}
```

The clauses come after the return type, as `requires` and `ensures` do. Vx has no attribute
syntax, so this keeps one style.

| Clause | What the compiler does with it | How it is enforced |
| --- | --- | --- |
| `requires e` | Proves `e` at each call site with z3, as for a Vx function (E8006). | Statically, at the caller. |
| `ensures e` | Uses `e` as a fact after the call. | Trusted. A guarded build checks it (G1). |
| `reads(p)`, `reads(p, n)` | The call does not change what `p` points to. | Trusted. A guarded build checks it with a copy. |
| `writes(p, n)` | The call changes only the first `n` elements at `p`. The rest of the reachable memory stays the same. | Trusted. A guarded build puts guard bytes after `n`. |
| `keeps(p)` | The function keeps `p` after it returns. Without `keeps`, the program can pass a reference to a local, which E4005 refuses today. | Trusted. Listed in the audit (A1). |
| `frees(p)` | The call ends the life of the handle `p`. The program cannot use `p` after the call. | Statically, on the Vx side. |
| `returns owned(f)` | The returned handle must go to `f`, which `frees` it. The handle type gets a `drop` that calls `f`. So the program cannot leak it or free it twice. | Statically, on the Vx side. |
| `blocks`, `nonblocking` | A `nonblocking` call needs no help from the monitor (W8). The default is `blocks`. | Trusted. A guarded build reports a `nonblocking` call that takes too long. |
| `calls back` | The task stays on its thread for the call (W9). A function-pointer argument without this clause is a compile error. | Statically. |
| `allocates(Memory::X, n)` | Admission counts `n` in `Memory::X`. Without it, the verdict is "unknown". | Trusted. Listed in the audit. |
| `-> !` | The call does not return. It exists today. | Statically, at the caller. |

**X3. `thread_safe`** is a clause of the whole `extern` block, because a library is safe from many
threads or it is not. Without it, the runtime sends one call at a time to that library. The
`static` map in `cublas_handle()` (`runtime/cuda_dispatch.cpp`) is the bug that this default
prevents.

**X4. An unwind never enters Vx.** A foreign unwind that reaches a Vx frame stops the program at
the boundary. This is not a clause, because no clause can make it safe. Vx code never unwinds
(T6), so its state is never half-changed.

**X5. Targets.** An `extern` block can name the targets that have the library, for example
`extern "C" for(CPU)`. A device body that calls a function of another target gets the error of F8.

## 3. Strict blocks

**B1.** In a `strict extern { .. }` block, each function must state each clause, also when the
answer is the default: `blocks` or `nonblocking`, `keeps` or no `keeps`, and so on. A missing
clause is a compile error that names it. A team that must control each foreign call uses strict
blocks, and a reviewer sees the whole contract in the declaration.

**B2.** `strict` also forbids `safe fn` without `requires` on each pointer argument, for example
`requires n > 0` for a buffer of `n` elements.

## 4. A policy for `unsafe` and `extern`

**P1.** `vxc --unsafe=deny|audit|allow` sets the policy of a build:

- `deny`: an `unsafe` block, an `unsafe fn`, an `unsafe impl` or an `extern` block is a compile
  error, except in the modules on the allow list (P2).
- `audit`: each one is allowed, and the build writes the audit (A1).
- `allow`: today's behavior. It is the default.

**P2.** The allow list is a file that names the modules that may hold `unsafe` or `extern` code.
A fleet build can then forbid `unsafe` code outside of `stdlib` and a few reviewed modules.

**P3.** `--require-strict` makes each `extern` block outside `stdlib` a strict block (B1).

## 5. The audit of the trusted base

**A1.** `vxc --audit` prints the trusted base of the program:

- each `extern` function, with its clauses, and the clauses that are trusted
- each `unsafe` block, `unsafe fn` and `unsafe impl`, with its line
- each `pub fn` that foreign code can call
- each linked library, with its ELF marks for shadow stacks and IBT or BTI (section 7 of the
  waiting-tasks note)

Section 7 of the waiting-tasks note asks for this list. The compiler has all of it after the
freeze, so the list is complete and does not depend on a person.

**A2.** The audit is a stable text format. So a review can compare the audit of two versions and
see each new trusted item.

## 6. A guarded build

**G1.** `vxc --guarded` puts a shim around each `extern` call. The shim checks the clauses that a
run can check:

- `ensures`, after the call
- `writes(p, n)`, with guard bytes after `n`, and `reads(p)`, with a copy
- `nonblocking`, with a time limit that the monitor watches
- that a task without `calls back` is on the same thread after the call

A failed check stops the program, and the report names the function and the clause. A guarded
build is for tests and for deployments that take the cost.

## 7. Later: isolation

**I1.** `extern "isolated" { .. }` runs a library in its own process, and each call goes over IPC.
This is for a library that the program does not trust at all. The clauses then become checks at
the process boundary.

## The parallel compiler

| Rule | This design |
| --- | --- |
| A signature is the whole contract | The clauses make it true for `extern` declarations too. |
| Every layout is known at the freeze | The handle types of `returns owned` are library structs. |
| A body declares nothing global | Clauses are on declarations. The policy is a flag and a file. |
| No implicit conversions | No clause converts a value. |

## Decisions

1. **The syntax of clauses.** Keywords after the signature, as `requires` (section 2), or a new
   attribute syntax. Recommended: keywords, so Vx keeps one style.
1. **The default for `thread_safe`.** One call at a time (X3), or many calls at once. Recommended:
   one at a time. It is slow only for a library that did not say `thread_safe`.
1. **Where the policy lives.** A flag and an allow-list file now (P1, P2), or a package manifest
   later. Recommended: the flag and the file now. Vx has no manifest yet.
1. **`safe fn` in strict blocks (B2).** Require a `requires` for each pointer argument, or not.
   Recommended: require it.

## The work, in order

1. The clauses in the parser and in `ExternDecl`, with `requires` and `ensures` (X1, X2).
1. `vxc --audit` (A1, A2). It is useful before the other parts exist.
1. `keeps`, `frees`, `returns owned` and `calls back`, in the borrow checker and E4005.
1. `thread_safe` and the one-call-at-a-time default, with the first runtime of the tasks design.
1. Strict blocks and the policy (B1, B2, P1 to P3).
1. `nonblocking` and `allocates`, with the monitor and admission.
1. The guarded build (G1).
1. Isolation (I1).
