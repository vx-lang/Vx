# Contracts for `extern` and `unsafe` code

Status: agreed. Issue: #1902. Tracking: #1870. Builds on section 7 of
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
strict extern "C" ensures thread_safe {
  /// Reads up to `n` bytes into `buf`. Returns the count, or -1.
  fn vx_tcp_read(s : *mut i8, buf : *mut u8, n : u64) -> i64
      requires n > 0
      ensures return >= -1
      ensures reads_only(s)
      ensures writes_only(buf, n)
      ensures !keeps(s)
      ensures !keeps(buf)
      ensures blocks
      ensures allocates(Memory::Host) == 0;
}
```

**X3. The clauses are `requires` and `ensures`.** They are the contracts that Vx functions already
have, so Vx gets no new keyword and no attribute syntax. A clause about the effect of a call is a
predicate that only a contract can use:

| Predicate | What the compiler does with it | Check in the wrapper (B3) |
| --- | --- | --- |
| `requires e` on a value | Proves `e` at each call site with z3, as for a Vx function (E8006). | Checks `e` before the call. |
| `ensures e` on a value | Uses `e` as a fact after the call. | Checks `e` after the call. |
| `ensures reads_only(p)` | The call does not change what `p` points to. | Compares a copy. |
| `ensures writes_only(p, n)` | The call changes only the first `n` elements at `p`. The rest of the reachable memory stays the same. | Guard bytes after `n`. |
| `ensures !keeps(p)` | The function does not keep `p` after it returns. Then the program can pass a reference to a local, which E4005 refuses today. | None. Trusted, and listed in the audit (A1). |
| `ensures freed(p)` | The call ends the life of the handle `p`. The program cannot use `p` after the call. | None. The Vx side checks it statically. |
| `ensures owned_by(return, f)` | The returned handle must go to `f`, which frees it. The handle type gets a `drop` that calls `f`. So the program cannot leak it or free it twice. | None. The Vx side checks it statically. |
| `ensures !blocks` | The call does not block its thread, so it needs no help from the monitor (W8). `ensures blocks` states the default. | A time limit that the monitor watches. |
| `ensures !calls_back` | The task does not need to stay on its thread for the call (W9). Without it, a call with a function-pointer argument pins the task. | The task is on the same thread after the call. |
| `ensures allocates(Memory::X) <= n` | Admission counts `n` in `Memory::X`. Without it, the verdict is "unknown". | None. Trusted, and listed in the audit. |
| `-> !` | The call does not return. It exists today. | Stops the program if the call returns. |

**X4. `ensures thread_safe`** is a contract of the whole `extern` block, because a library is safe
from many threads or it is not. Without it, the runtime sends one call at a time to that library.
The `static` map in `cublas_handle()` (`runtime/cuda_dispatch.cpp`) is the bug that this default
prevents.

**X5. An unwind never enters Vx.** A foreign unwind that reaches a Vx frame stops the program at
the boundary. No contract can make it safe, so none exists. Vx code never unwinds (T6), so its
state is never half-changed.

**X6. Targets.** An `extern` block can name the targets that have the library, for example
`extern "C" for(CPU)`. A device body that calls a function of another target gets the error of F8.

## 3. Strict blocks

**B1. Each contract is written.** In a `strict extern { .. }` block, each function must state each
predicate of X3, also when the answer is the default, for example `ensures keeps(p)`. A missing
one is a compile error that names it. A reviewer then sees the whole contract in the declaration.

**B2.** A `safe fn` in a strict block needs a `requires` for each pointer argument, for example
`requires n > 0` for a buffer of `n` elements.

**B3. The compiler wraps each call.** For each function of a strict block, the compiler generates a
wrapper, and each call goes to the wrapper:

1. Check each `requires` at run time. z3 already proved it at the call site, but a call through a
   function pointer or from foreign code may still break it.
1. Call the `extern` function.
1. Check each `ensures` that the third column of X3 can check.

A failed check stops the program. The report names the function, the contract, and the line of
the call. A contract that no run can check, such as `!keeps(p)`, stays trusted, and the audit
lists it (A1).

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

**G1.** `vxc --guarded` gives each `extern` block the wrapper of B3, also a block that is not
strict. It checks the contracts that the block states. It is for tests, and for deployments that
take the cost.

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

We chose these on 2026-10-19.

1. **The syntax.** Clauses are `requires` and `ensures`, with the predicates of X3. Vx gets no new
   keyword and no attribute syntax.
1. **Strict blocks check at run time.** The compiler wraps each call of a strict block, and the
   wrapper checks `requires` before the call and `ensures` after it (B3).
1. **The default for `thread_safe`.** One call at a time (X4). It is slow only for a library that
   does not state `ensures thread_safe`.
1. **Where the policy lives.** A flag and an allow-list file now (P1, P2). Vx has no package
   manifest yet. The policy can move into one later.
1. **`safe fn` in strict blocks (B2).** It needs a `requires` for each pointer argument.

## The work, in order

1. `requires` and `ensures` on `extern` functions, in the parser and in `ExternDecl` (X1 to X3).
   z3 proves each `requires` at the call site, as for a Vx function.
1. `vxc --audit` (A1, A2). It is useful before the other parts exist.
1. `keeps`, `freed`, `owned_by` and `calls_back`, in the borrow checker and E4005.
1. `thread_safe` and the one-call-at-a-time default, with the first runtime of the tasks design.
1. Strict blocks and their wrappers (B1 to B3), and the policy (P1 to P3).
1. `blocks` and `allocates`, with the monitor and admission.
1. The guarded build (G1). It reuses the wrapper of B3.
1. Isolation (I1).
