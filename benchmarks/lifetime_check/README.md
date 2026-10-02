# The lifetime check, measured

This crate holds the numbers in the blog post
[Can the borrow checker algorithm be auto-vectorized?](https://vxlang.org/blog/can-the-borrow-checker-be-auto-vectorized.html)

It copies the lifetime check from `src/borrow.rs` (the fast path of `verify_subtyping_bounds`)
in two forms:

- **The old loop**, which returns at the first slot that fails.
- **The branch-free check**, which `src/borrow.rs` now runs after a whole-word equality test.

Keeping them in a crate of their own lets them be compared on billions of inputs and timed apart
from the rest of the compiler. The crate is not part of the compiler's workspace, so building Vx
never builds it.

## Running it

From this directory:

```sh
cargo run --release -- exhaustive   # every 16-bit value pair of each slot, old loop vs branch-free
cargo run --release -- agree        # 50 million whole-word pairs of each input, every version
cargo run --release -- bench        # nanoseconds per check, best of 30 runs, one thread
```

`exhaustive` uses every core but two. Both `exhaustive` and `agree` print the number of
disagreements, which should be 0.

`bench` uses four kinds of input, in the order of the post's tables:

| Name | Pairs |
|------|-------|
| `passing` | pairs that pass every slot, with no two words identical |
| `made_up` | words with variances 0 to 2 and small regions, almost all failing somewhere |
| `random` | random 64-bit words, which almost always fail at slot 0 |
| `identical` | the same word twice |

For each input it prints two sets of times:

- **A batch through one loop**, where LLVM can vectorize across pairs.
- **One call per check**, the way the compiler calls it.

The figures in the post come from an Apple M4 with rustc 1.95.0. Other machines will give other
numbers.

## Reading the generated code

```sh
cargo rustc --release --lib -- --emit asm             # assembly, under target/release/deps/
cargo rustc --release --lib -- -C remark=all -C debuginfo=1 2>&1 | grep vectoriz
```

The second command lists what LLVM's vectorizers did and did not vectorize, with source lines. `single_branch_free` is
the function whose assembly the post shows.
