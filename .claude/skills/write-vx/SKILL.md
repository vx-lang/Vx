---
name: write-vx
description: Write, fix or explain a Vx program (a .vx file). Use when adding a test fixture, an example, a benchmark or library code in Vx, or when a .vx file does not compile.
---

# Writing Vx

Vx looks like Rust, and Rust habits are the most common reason Vx does not compile. Before
writing Vx, read these two pages. They are short, and their programs are compiled by the test
suite, so they match the compiler:

- `www/book/src/quick-reference.md`: the language on one page, with complete programs.
- `www/book/src/for-rust-programmers.md`: what to change when writing Vx the Rust way.

For more on one topic, read its chapter in `www/book/src/` (`SUMMARY.md` lists them). For a
standard library type, read its source in `stdlib/core/` or `stdlib/std/`. Do not guess a
method's name or signature.

## Compile, read the error, fix

Build the compiler once per session, and again after changing its source:

```bash
source config.local && cargo build -j <cores minus 2> --bin vxc
```

Then compile and run a program:

```bash
source config.local && ./target/debug/vxc program.vx
```

- Lines starting `[JIT]` or `[flat-codegen]` are the compiler's own progress, not the program's
  output.
- `--action parse-only` checks the syntax only. `--legacy-codegen` uses the second code
  generator. A program should print the same with and without it.
- `import core::...` paths resolve against the repository, so run `vxc` from the repository root.

When it fails, fix the first error and compile again: later errors often follow from the first.
For an error with a code, such as `E3004`, `www/book/src/error-index.md` gives its meaning in a
line, the doc comment on the code in `src/diagnostic.rs` says more when there is more to say, and
the tests that report it show programs that reach it (`grep -rl E3004 tests/`).

## When the compiler is wrong

An error that names a compiler source file, a `Codegen Error`, or a `Vx Compiler Internal Error`
is a compiler bug, not a mistake in the program. So is a program that compiles but prints the
wrong answer. Reduce it to a few lines, check the open issues, and report it. Do not leave a
workaround in the program without saying which issue it works around.

## Tests

A program that should run goes in `tests/backend/pass/`, with its expected output in an
`// EXPECT: ...` line. A program that should be refused goes in `tests/frontend/fail/`, with a
`RUN:` line and `CHECK:` lines for the error. Copy the header and layout of a neighbouring file.
