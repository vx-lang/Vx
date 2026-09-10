# Working on the Vx compiler

This guide is for people changing the compiler, not people using the language. If you want to
*use* Vx, read <https://vxlang.org/docs/> instead.

It describes the compiler as it is today. Where a detail here disagrees with the code, the code is
right and this file is a bug — please fix it.

## Before you start

Build the compiler once, so the rest of this makes sense with something running in front of you.
[`INSTALL.md`](INSTALL.md) has the full setup; the short version is:

```bash
source config.local     # puts the pinned LLVM 22 on PATH
cargo build
```

`source config.local` is needed before *every* `cargo` command. Without it the build script cannot
find `llvm-config` and fails late, after a long dependency rebuild.

Then run something:

```bash
./target/debug/vxc tests/backend/pass/control_flow.vx --run
```

## Where the code lives

The compiler is one Rust crate, `Vx`, plus a small Rust core for the standard library.

| Path | What it holds |
| --- | --- |
| `src/lexer.rs` | Source text to tokens. The keyword table is here. |
| `src/parser/` | Tokens to AST. Split by what it parses: `decl`, `stmt`, `expr`, `types`, `macro_expand`. |
| `src/syntax/` | The AST itself, and name resolution over it. |
| `src/hir/` | The typed representation, and every check that runs on it. |
| `src/hir/check/` | Type checking, split by concern: `calls`, `operators`, `control`, `literals`, `access`, `transfer`, `raw`, `autodiff`, `capacity_fold`, `region_traffic`. |
| `src/hir/decl_check/` | Checks over declarations: `topology`, `memory`, `placement`, `conflicts`. |
| `src/borrow.rs`, `src/hir/borrow_cx.rs` | Borrow checking. |
| `src/codegen/flat/` | The default code generator. |
| `src/codegen/lower/` | The older AST-walking code generator. |
| `src/pipeline.rs` | The orchestrator that runs all of the above, in parallel. |
| `src/jit.rs` | Running a compiled program, and emitting an object file. |
| `src/diagnostic.rs` | Every error and warning code. |
| `src/plugin/` | Hardware plugins, which let a vendor claim a topology. |
| `stdlib/std/` | The standard library, written in Vx. |
| `stdlib/rust_core/` | The Rust code the standard library is built on. |

`src/pipeline.rs` and `src/arch.rs` are the two largest files and the two worth reading first.

## How a compile runs

`src/pipeline.rs` runs the frontend as a sequence of phases. The phases are numbered in the code,
and the numbers are worth knowing because comments and issues refer to them.

Modules are compiled **in parallel** with `rayon`. Most phases run once per module across a thread
pool, and a few are barriers where every module has to arrive before anything continues.

| Phase | Function | What happens |
| --- | --- | --- |
| — | `parse_phase` | Each file is lexed and parsed into an AST. Parallel, one module per task. |
| — | `macro_expansion_phase` | `macro_rules!` definitions are expanded. |
| — | `name_resolution_phase` | Names inside each module are resolved. |
| 2 | `build_frozen_registry` | **A barrier.** Every module's declarations are merged into one registry, which is then frozen. Cycle detection runs here, so an infinitely sized recursive struct is caught at this point. |
| 3 | (in `type_check_phase`) | Each function's type references are lowered to a flat stream of GIDs. |
| — | `declaration_check_phase` | Checks over declarations — topologies, memory spaces, placement. |
| — | `type_check_phase` | The AST is lowered to HIR and type checked. |
| 5 | `deduplication_phase` | Each worker's local type arena is merged into one global arena. |
| 6 | `simd_patch_phase` | Worker-local arena indices are rewritten to global ones. |
| 7 | `codegen_mlir_phase` | MLIR is generated. |

### GIDs

A GID is a 256-bit identifier for a declaration, computed as a **content hash** of the module and
symbol name. It is not a counter.

That distinction matters. Because a GID does not depend on the order work was scheduled in, the
same source files produce the same GID stream no matter how `rayon` happens to schedule the
threads. That is what makes a parallel frontend reproducible, and there is a test that compiles the
same input twice and compares the streams.

`src/gid.rs` has the implementation. `docs/parallel_compiler_architecture.md` explains the design
at length.

## The two code generators

There are two, and knowing which one ran explains a large fraction of confusing behaviour.

**The flat path** (`src/codegen/flat/`) is the default. It works over the flat type stream and can
be run in parallel.

**The AST path** (`src/codegen/lower/`) walks the AST directly. It is older. Use `--legacy-codegen`
to force it.

The flat path does not cover the whole language. When it meets a construct it cannot lower, it
**declines** — it returns "no MLIR" rather than failing. `vxc` answers a decline by quietly falling
back to the AST path, so the program still compiles. `src/decline.rs` records the reasons.

Two consequences worth remembering:

- A program can compile through either generator, and they have different bugs. Many open issues
  are titled "AST codegen: ..." or "flat codegen: ..." for exactly this reason.
- A test that asserts on emitted IR is asserting about *one* of the two. Test files that check the
  AST path's exact output pin it with `--legacy-codegen` on their `RUN` line.

## From MLIR to a running program

MLIR passes run **inside the compiler process**, through melior's `PassManager` — the compiler does
not shell out to `mlir-opt`. Two of the passes are custom C++, declared in `src/codegen/mod.rs` and
implemented in the C++ sources: `addVxLoweringPass` and `addVxToLLVMPass`.

After that, `--run` shells out to the LLVM tools in turn:

```
MLIR  --mlir-translate-->  LLVM IR  --opt-->  --llc-->  object  --clang-->  executable  -->  run
```

Each tool's path can be overridden by an environment variable — `MLIR_TRANSLATE_PATH`, `OPT_PATH`,
`LLC_PATH`, `CLANG_PATH` — which is how a packaged toolchain points at its own copies rather than
whatever is on `PATH`.

`--action emit-obj` does not use that chain. It builds an MLIR execution engine in process and asks
it to write an object file (`ObjectEmitter` in `src/jit.rs`).

## What `vxc` can be asked to do

`--action` selects the output:

| Action | Result |
| --- | --- |
| `parse-only` | Parse and check, produce nothing. The fastest way to ask "is this legal Vx?" |
| `print-ast` | Print the AST. |
| `emit-mlir` | Print the generated MLIR. |
| `emit-llvm` | Print LLVM IR. |
| `emit-obj` | Write an object file. |
| `emit-interface` | Write the module's public interface. |
| `run` | Compile and execute. `--run` is the shorthand. |

Useful while debugging:

```bash
vxc file.vx --action emit-mlir                    # what the default path produced
vxc file.vx --action emit-mlir --legacy-codegen   # what the AST path produced
vxc file.vx --action parse-only                   # does it even parse
```

## Tests

```bash
source config.local
cargo test
```

Most of the suite is **fixture tests**: `.vx` files under `tests/`, each carrying a `// RUN:` line
saying what to do with it and `// CHECK:` lines saying what the output must contain. This is the
same style LLVM uses.

| Directory | Meaning |
| --- | --- |
| `tests/frontend/pass/` | Must compile. |
| `tests/frontend/fail/` | Must be rejected, with the diagnostic the file names. |
| `tests/middle_end/` | Checks generated IR. |
| `tests/backend/pass/` | Compiles and runs end to end. |
| `tests/optimizations/` | Checks that a transformation happened. |

`CHECK` lines are matched by the real LLVM `FileCheck`, so every directive works: `{{regex}}` holes,
`[[VAR:pattern]]` captures, `CHECK-NEXT`, `CHECK-SAME`, `CHECK-DAG`, `CHECK-NOT`, `CHECK-COUNT-n`.
`FileCheck` must be on `PATH`, which `config.local` handles.

Two ways a file is run, and the difference matters when a test behaves oddly:

- `tests/frontend/`, `tests/optimizations/` and `tests/backend/` run the `RUN:` line through a
  shell, exactly as written. `%s` is the test file, `%t` a scratch path.
- `tests/middle_end/` compiles **in process**, because it drives passes the CLI has no flag for,
  then pipes the result into `FileCheck`.

Markers a fixture can carry:

| Marker | Effect |
| --- | --- |
| `// REQUIRES: macos` | Skipped on other systems. |
| `// REQUIRES: ane` | Skipped without Apple Neural Engine models. |
| `// REQUIRES: z3` | Skipped without the z3 prover. |
| `// REQUIRES: flat-codegen` | Only meaningful on the flat path. |
| `// XFAIL: *` | Known to fail because the compiler is wrong. The command still runs, and the day it passes the test reports that the marker should go. |

One trap: `FileCheck` reads `CHECK:` **anywhere** on a line, including inside prose. Do not write
the word followed by a colon in a comment unless you mean it.

To regenerate `CHECK` lines after changing what the compiler emits:

```bash
cargo run --bin update_mlir_test_checks -- <file.vx>
```

Be careful which path you regenerate against: the updater rewrites the checks for whatever the
`RUN` line actually runs. If the checks describe the AST path, the `RUN` line needs
`--legacy-codegen`, or the update will silently replace them with flat-path output.

## Adding a language feature

The path through the compiler, in order. A new statement or expression usually touches all of it.

1. **Lexer** — `src/lexer.rs`. Add the keyword to the `KEYWORDS` table if the feature needs one.
1. **AST** — `src/syntax/`. Add the node to the right enum (`stmt.rs`, `expr.rs`, `decl.rs`,
   `types.rs`).
1. **Parser** — `src/parser/`. Build the node.
1. **HIR** — `src/hir/flatten.rs` lowers the AST node to HIR; `src/hir/check/` type checks it.
1. **Code generation** — `src/codegen/flat/emit/` for the default path. If the flat path cannot
   handle it yet, record a decline in `src/decline.rs` so it falls back rather than miscompiles.
1. **Diagnostics** — any new error gets a code in `src/diagnostic.rs`. Codes are grouped by area;
   put a new one with its neighbours.
1. **Grammar** — update `docs/lang/grammar.md`. It is normative, so a feature that is not in the
   grammar is not in the language.
1. **Tests** — a `pass` fixture for the feature working, and a `fail` fixture for each way it can
   be used wrongly.
1. **Documentation** — the book under `www/book/src/` teaches the language. CI compiles every
   example in it, so an example that does not build fails the build.

Prefer an `assert` over a silent fallback. This is a compiler: crashing is better than emitting
wrong code quietly.

## Conventions

The full rules are in `agents/AGENTS.md`. The ones that catch people out:

- **Formatters run before the commit, not after.** `cargo fmt` for Rust, `cargo run --bin vx-format -- <file>` for `.vx` files, `clang-format` for C++. Never run `clang-format` on a `.vx` file. If you
  format after staging, the commit captures the unformatted version and CI fails.
- **No issue numbers or codewords in source comments.** Git already records why a line changed.
  (There is a backlog of existing violations — see Vx#423.)
- **Commit messages carry the issue.** Use `Fixes: #<id>` when the commit closes it.
- **Simple English.** Comments should read for someone who has not seen the code before.

## Where to start

Issues labelled [**good first issue**](https://github.com/vx-lang/Vx/labels/good%20first%20issue)
are self-contained, and each one states the problem with a reproduction you can run.

They are written to be picked up cold: the issue says what is wrong, shows the current behaviour,
and says what correct behaviour would look like. If one is unclear, that is worth a comment on the
issue — the description is the thing at fault.

[**help wanted**](https://github.com/vx-lang/Vx/labels/help%20wanted) holds larger pieces that are
still well specified.

## Further reading

| Document | Covers |
| --- | --- |
| [`parallel_compiler_architecture.md`](parallel_compiler_architecture.md) | The phase model, GIDs and epochs, at length |
| [`architecture_executive_summary.md`](architecture_executive_summary.md) | The same in brief |
| [`ast_reference.md`](ast_reference.md) | The AST node by node |
| [`generics_design.md`](generics_design.md) | Monomorphization |
| [`adding_a_topology.md`](adding_a_topology.md) | Supporting new hardware |
| [`scalable_plugin_system.md`](scalable_plugin_system.md) | How a vendor extends the compiler |
| [`lang/grammar.md`](lang/grammar.md) | The grammar, which is normative |
