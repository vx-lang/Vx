# Differential fuzzing

Run the same Vx program several ways and compare what it prints. Each way is a
*configuration*: a set of `vxc` flags. By default these are the flat code generator (`flat`)
and `--legacy-codegen` (`ast`). A generated program can also have a Rust twin, built with
`rustc`, which must print exactly the same thing. Any difference is a bug in a code generator,
or in the generator that wrote the program.

This found #990 (a `let` in a block taking over an outer variable), #1001, #1012, #1014 (a
double free), #1017, and #1096 (a row chosen by an `if`, inside another `if`).

Run it with the LLVM tools and `rustc` on PATH: `source config.local` first.

## Generated programs

```
fuzz.py list                                # the generators
fuzz.py run shadowing --seeds 1-1000        # 1000 programs, every configuration and Rust
fuzz.py run tensors --seeds 1-300 --keep /tmp/failures
fuzz.py run views --seeds 1-1000            # rows and reshapes, used as the borrow rules allow
fuzz.py run owners --seeds 1-1000           # tensors moved into calls, in branches and loops
fuzz.py show tensors 1133                   # the Vx program for a seed (--rust: its twin)
```

`run` prints how many programs agree, then groups the rest by symptom, such as
`flat+ast: wrong output` or `ast: 'llvm.load' op operand ## must be ...`, with seeds for each.
`--keep DIR` writes each failing program, its twin and every output.

To shrink a failing seed:

```
fuzz.py reduce shadowing 52                 # while the same runs still disagree
fuzz.py reduce tensors 1133 --message "double free" --repeat 5
```

The reducer deletes statements as long as the program still fails. "Fails" means, by default,
that the same set of configurations still disagrees with Rust; with `--message`, that some run
still prints the text. `--repeat N` runs each candidate up to N times, for a failure that does
not happen on every run.

## Existing programs

```
fuzz.py compare tests/backend/pass/*.vx
fuzz.py compare bench.vx --ignore '[0-9]+\.[0-9]+e-[0-9]+'
```

`compare` runs each file under every configuration and reports the files whose outputs differ,
with the first line that differs. Addresses (`0x5b78...`, from printing a tensor) are ignored,
and `--ignore REGEX` hides anything else that changes from run to run, such as a timing. A file
marked `// REQUIRES: flat-codegen` is expected to fail on `ast`, and the report says so.

## Configurations

```
fuzz.py run shadowing --config flat= --config o0=-O0
fuzz.py compare tests/backend/pass/*.vx --config flat= --config ast=--legacy-codegen --config j4="-j 4"
```

Giving any `--config` replaces the defaults. `--vxc PATH` picks the compiler (by default
`$CARGO_TARGET_DIR/debug/vxc`), `--jobs N` how many programs run at once, and `--timeout S` the
seconds one run may take.

Words like `VAR=value` before the flags set environment variables for that configuration:

```
fuzz.py run tensors --config flat= --config o0="RUST_BACKTRACE=0 -O0"
```

## Freeing what a program allocates

```
fuzz.py run owners --seeds 1-1000 --heap
fuzz.py compare tests/backend/pass/*.vx --heap
```

`--heap` counts the heap blocks each compiled program allocates and frees, with a small
library (`heap_count.c`) preloaded into it. An empty program leaves a block or so that the
runtime keeps for itself, so a run that leaves a different number than an empty program does
is reported as `flat:heap: leaks # blocks`, beside any difference in output. A block freed twice
or a stack address freed usually stops the program in glibc's own checks. This needs Linux,
glibc and `cc`.

## Writing a generator

A generator is a module in `generators/`. Its docstring describes it, and its
`generate(seed)` returns a `fuzzlib.Program`, made from:

- **a statement tree:** a list of tuples whose first element is the kind of statement. A list
  of tuples inside a statement is a block, and a list of blocks (a `match`'s arms) works too.
  The reducer deletes statements from blocks, and needs nothing else;
- **`source(tree, lang)`:** the program as `"vx"`, or as `"rs"` for the Rust twin. Return
  `None` for `"rs"` when there is no twin; the configurations are then compared with each
  other only;
- **`keep(stmt, block)`, optional:** the statements the reducer must not delete, such as a
  declaration everything else uses.

A small generator, for integer arithmetic:

```python
"""Small products plus one, printed on one line."""
import random
from fuzzlib import Program

def generate(seed):
    rng = random.Random(seed)
    tree = [("print", rng.randint(0, 99), rng.randint(0, 99)) for _ in range(5)]

    def source(stmts, lang):
        if lang == "vx":
            body = "".join(f'  print({a} * {b} + 1);\n  print!(" ");\n' for _, a, b in stmts)
            return "fn main() -> i32 {\n" + body + "  return 0;\n}\n"
        body = "".join(f'  print!("{{}} ", {a} * {b} + 1);\n' for _, a, b in stmts)
        return "fn main() {\n" + body + "}\n"

    return Program(tree, source)
```

Vx's `print` adds no newline, so print the same separators on both sides. Keep a generator's
values small enough that Vx and Rust agree on overflow, and write each
expression so that one string is valid in both languages. `shadowing.py` indexes its tensor
through a small Rust wrapper type for that reason.
