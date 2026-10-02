# Differential fuzzing of the two code generators

Random Vx programs, each with a Rust twin, run on the default (flat) code generator, on
`--legacy-codegen`, and through `rustc`. Any difference in what they print is a bug in one of
them. This found the block-scoping bugs of #990, and #1001, #1012 and #1014.

- `shadowing.py SEED OUTDIR` writes `pSEED.vx` and `pSEED.rs`: one function whose blocks
  (`if`/`else`, `for`, `while`, `loop`, `match`, `unsafe`, value `if`) reuse a few names, with a
  4-element tensor, a `&mut i32` helper, and `break`/`continue`/`return`.
- `tensors.py SEED OUTDIR`: a function that builds tensors in nested blocks and returns one,
  often early. Aimed at buffer freeing.
- `run_one.sh GENERATOR SEED VXC OUTDIR` runs one program three ways and prints `SEED ok` or
  `SEED MISMATCH ...`, keeping the files of a mismatch.
- `reduce_shadowing.py` / `reduce_tensors.py SEED VXC "MESSAGE" [flags]` delete statements from
  a failing program while `vxc [flags]` still prints MESSAGE, and print what is left.

Many seeds, 12 at a time, with the LLVM tools and `rustc` on PATH (`source config.local`):

```
mkdir -p /tmp/fuzz
seq 1 1000 | xargs -P 12 -I{} scripts/tools/differential_fuzz/run_one.sh \
    shadowing.py {} target/debug/vxc /tmp/fuzz > /tmp/fuzz/results.txt
grep -v ' ok' /tmp/fuzz/results.txt
```
