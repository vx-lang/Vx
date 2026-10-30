# Formatter architecture review fixes

These fixes address the four findings from the formatter architecture review.
They are part of issue 2343.

## Changes

The registration test now proves dispatch through a distinctive layout instead of an atomic counter.
The exact CI atomic-use check passes without exemptions.

Child documents retain their shape and remaining width budget through composition.
Parents select expanded groups through the document interface without inspecting child variants.
A zero-budget child now selects its expanded alternative in the active pipeline.

Grammar probes omit excluded contents while retaining delimiters connected to surrounding code.
A connected opening brace receives a neutral block prefix in the probe only.
Original source tokens, excluded bytes, and renderer delimiter transitions remain unchanged.
Unsupported excluded statements no longer disable unrelated declarations or the supported remainder of a function.
Uncertain structural boundaries retain the conservative preservation policy.

Recognition builds one significant-token buffer and computes scope context in one traversal.
Declarations borrow that buffer instead of copying its remaining suffix.
Body probes, rendering, and state updates operate on indexed source intervals.
Local validation allocates positions for its interval rather than the complete file.
Syntax ownership uses a tree index, and protected intervals use sorted range indexes.
Generic closing-bracket checks use indexed type ranges and expression boundaries.
One final full-file preservation check remains after rendering.

## Validation

The current working tree passed 54 formatter tests and 16 CLI tests.
Six deliberate regressions failed at their intended assertions before restoration.
They cover child budgets, excluded root syntax, excluded body syntax, registration, token intervals, and legacy ownership.
Focused Clippy passed with warnings denied.
The formatter binary built successfully.
The exact CI atomic-use shell check passed.

All 1,786 corpus files completed formatting and check operations on temporary copies.
Every copy stayed unchanged after a second formatting pass.
Hash comparisons confirmed that repository fixtures remained untouched.
The original corpus check still reports 959 formatting differences and returns one.
That existing style migration remains separate from these fixes.

Commands sourced `config.local` and used eight build workers and eight test threads.
The focused targets were:

```sh
cargo test --offline --locked -p vxc --lib formatter:: -- --test-threads=8
cargo test --offline --locked -p vxc --test vx_format_cli -- --test-threads=8
cargo clippy --offline --locked -p vxc --lib --bin vx-format --test vx_format_cli -- -D warnings
cargo build --offline --locked -p vxc --bin vx-format
```

Logs use the `build/vx-format-review-fix-` prefix.
Full workspace and backend tests were not run.
Markdown formatting remains unverified because `mdformat` is absent from the project environment.

## Scaling measurements

Measurements use the debug CLI, one Rayon worker, and the median of three runs.
Every generated file returned zero without warnings or formatting differences.
The original review reported 21.2 seconds for 12,800 simple declarations occupying 411,290 bytes.
These local timings do not establish release performance.

| Declarations | Simple declaration seconds | Nested generic declaration seconds |
| --- | --- | --- |
| 1,600 | 0.067 | 0.118 |
| 3,200 | 0.123 | 0.219 |
| 6,400 | 0.237 | 0.434 |
| 12,800 | 0.462 | 0.846 |

The largest simple file contains 411,290 bytes.
The largest generic file contains 667,290 bytes.
Both series show approximately doubled execution time when declaration counts double.
