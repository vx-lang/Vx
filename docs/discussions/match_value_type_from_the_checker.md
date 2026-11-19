# A `match` takes its value type from the checker

Status: experimental. The branch `checker-stores-block-value-types` does all four kinds of block in
one change, in a draft pull request.
Related: #3622, #3665, #3680, PR #3681.

## The problem

The type checker works out the type of every `match` that gives a value. `check_match_expr` in
`src/hir/check/control.rs` returns it. But the checker does not store the type. So each code
generator works out the type again from the source code:

- The legacy code generator (`--legacy-codegen`) uses `infer_ast_type`, then a chain of
  `.or_else(...)` fallbacks in the `MatchExpr` lowering in `src/codegen/lower/expr.rs`.
- The flat code generator uses `infer_expr_ty` and `infer_block_ty` in `src/hir/flatten.rs`.

To do this, each code generator copies the scoping rules of the language by hand, and the two
copies are separate. Each bug so far was one missed scoping rule:

| Issue | Program | Missed rule |
|---|---|---|
| #3622 | `Ok(v) => { v }` after `import std::string` | A local `v` of another function leaked in. |
| #3665 | `let v : i64 = ..; match .. { Ok(v) => { v } }` | The payload `v` hides the outer `v`. |
| #3680 | `Some(v) => { let v = v as f32; v }` | A `let v` in the arm hides the payload `v`. |

The fix for #3680 in PR #3681 works for that program, but it is incomplete. It looks only for a `let` at the top
level of the arm. Other forms, such as `v * 2.0` or `v.field`, go through other guesses. Each
new kind of expression or scope needs a new rule, in two places.

## The goal

The checker is the one place that knows the scoping rules. It stores the type of each `match`
on the `match` node. Both code generators read that type and do not guess.

The repository already does this for other nodes:

- `AsCastExpr.source_ty`, which the checker sets in `src/hir/check/operators.rs`.
- The `value_ty` of a `tasks` block. The flat code generator reads it and declines the function
  if it is missing.

## Facts checked before this plan

- **Generic functions are safe.** The checker makes a copy of a generic function for each list
  of type arguments, and then it checks the copy again. All three ways to make a copy do this:
  `instantiate_generic_function_call_inner`, `instantiate_impl_method`, and the path near the
  first `instantiate_function` call in `src/hir/check/calls.rs`. Each one calls `check_instance`.
  So each copy gets its own type, for example `i32` in one copy and `f32` in another.
- **Some code builds a `match` after the checker.** These places must set the type themselves:
  - `src/hir/check/drops.rs` (the code that frees values at the end of a scope)
  - `src/hir/check/tasks.rs` (two places)
- **Other places build a `match` before the checker checks it**, so the checker sets the type:
  - `src/parser/tuple_match.rs`
  - `src/hir/check/control.rs` (two places)

  Check each of these again during the work.

## The work

### Pull request 1: store the type, and read it in both code generators

1. Add `value_ty: Option<Type>` to `MatchExpr` in `src/syntax/expr.rs`. Use `None` for a `match`
   that gives no value.
2. In `check_match_expr`, set `value_ty` from the type that the function returns.
3. In each place that builds a `match` after the checker, set `value_ty`. If the code there does
   not know the type, it must get the type from the checker. It must not guess.
4. In the legacy code generator, take the type of a `match` from `value_ty`. Add an `assert` that
   says which `match` had no type.
5. In the flat code generator, do the same in `infer_expr_ty` and in `lower_match`.
6. Delete the code that guesses the type of a `match`:
   - `payload_binding_type` in `src/codegen/lower/expr.rs`
   - `infer_payload_binding_ty` in `src/hir/flatten.rs`
   - the `match` part of the `.or_else(...)` chain in the `MatchExpr` lowering
   - `MatchArm::body_declares` from PR #3681

   Keep `variant_payload_type` if other code still calls it.
7. Check every caller of `infer_ast_type` and `infer_expr_ty` that can get a `match`. An example
   is a `let` with no type annotation. Its result may change. The tests must show that each
   new result is right.

Estimate: 6 to 8 files, about 80 lines added and 150 lines deleted, 1 to 2 days.

### Later pull requests: other blocks that give a value

These blocks guess their type in the same way:

- `if` that gives a value: `infer_if_ty` and the `Expr::If` part of `infer_ast_type`
- `spawn` that gives a value (patched in #3665)
- `unsafe` block that gives a value

Do one kind of block in each pull request, with the same steps as above. Each one is smaller than
`match`, about half a day.

## Testing strategy

### Tests to keep

- `tests/frontend/pass/match_arm_value_type_ignores_other_locals.vx`. It has one case for each
  issue in the table above. It must pass with no change to its checks.

### New test files

Each test file has two RUN lines, one for each code generator. Each case prints a value that
shows its type, for example `4.5` for an `f32`.

Name each file for what it checks. Some possible names:

- `match_arm_redefines_its_payload_in_a_nested_block.vx`: the `let` is inside an inner `{ }` or
  an `if` in the arm, and the arm's value is the payload after that block.
- `match_arm_value_uses_a_redefined_payload.vx`: `let v = v as f32; v * 2.0` and
  `let v = Point { .. }; v.x`.
- `generic_function_with_a_match_on_two_types.vx`: one generic function, called with `i32` and
  with `f32` in the same program.
- `match_value_in_each_position.vx`: the `match` is a function argument, a return value, a `let`
  with no type annotation, and a struct field.
- `match_with_an_arm_that_returns.vx`: one arm leaves the function with `return`. The type comes
  from the other arms.
- `match_on_strings_and_tuples.vx`: the checker rewrites these into other code. The test checks
  that the rewritten code still has a type.
- `match_that_frees_a_value.vx`: a `match` that the drop code builds, and a `match` in a `tasks`
  block that gives a value.

Also run each new test with a parallel compile (`-j`). The type is part of the function's own
body, so it must not change with `-j`. See `docs/language_rules_for_a_parallel_compiler.md`.

### Prove that the tests check the right thing

Follow the rule in `agents/AGENTS.md`: break each assertion separately, and read which check
failed.

- Put the old guessing code back for one code generator. The new tests for that code generator
  must fail. Do this for each code generator separately.
- Delete the line that sets `value_ty` in one place that builds a `match`. The `assert` must fire
  and name that `match`. Do this for each place.
- Set a wrong `value_ty` in the checker, such as `i64` for every `match`. Tests must fail.

### Check the whole compiler

- The integration tests (`cargo test --test integration_test`) and the library tests.
  Especially check `flat_codegen_differential` and `flat_corpus_sweep`. They compare the two
  code generators and count the functions that the flat code generator declines.
- Compare the generated MLIR before and after the change, for every file in `tests/frontend/pass`
  and `tests/backend`. For a program that compiled before, the MLIR must be the same. List each
  difference and explain it in the pull request.
- The number of functions that the flat code generator declines must not go up. It may go down.

## Open questions

- Should `value_ty` hold the type before or after the checker resolves type aliases and number
  literals? The code generators need the final type.
- Does a `.vxlib` library store function bodies? If it does, `value_ty` must go into the library
  too, or the code that reads a library must check the body again.
- After the `if`, `spawn` and `unsafe` blocks, is `infer_ast_type` still needed? If only a few
  callers remain, they can also use types from the checker.

## Progress

One change does all four kinds of block, on the branch `checker-stores-block-value-types`.

- [x] `match`
- [x] `if` that gives a value
- [x] `spawn` that gives a value
- [x] `unsafe` block that gives a value
- [ ] Decide what to do with the rest of `infer_ast_type`

### How it differs from the plan

- The checker sets the type in one helper, `store_value_type` in `src/hir/expr.rs`. The main
  dispatch of the checker calls it after it checks each of the four kinds of block. This also
  covers a `match` on strings, which the checker makes into an `if`.
- The legacy code generator still lowers both branches of an `if` before it chooses the type of
  the value. It did not guess from the source there. Only its `infer_ast_type` changed for `if`.
- The flat code generator needs `lower_ty_synth`, not `lowered_ty`, for the stored type. Only
  `lower_ty_synth` records the layout of an enum or a closure.
- `UnsafeBlockExpr` already had `value_ty` for blocks made from tasks. The helper sets it only
  for a block that is not a task, so the code for tasks keeps its own meaning of the field.
- Generic functions needed no work. The checker checks each copy of a generic function again.

### What the checks found

- On `main`, the flat code generator also crashed on an `if` whose branch redefines a name.
  `if_and_block_value_type_follows_the_inner_scope.vx` checks it.
- MLIR before and after, for 1141 programs in `tests/frontend/pass` and `tests/backend/pass`,
  with each code generator:
  - 1768 were the same. 464 differed only in the path of the standard library.
  - 3 compile now and did not compile before: the regression tests.
  - 13 differed in one way: a `transfer` of the value of a `spawn` block now names its source
    (`source_topology`). The old guess read the last expression of the block, a local of the
    block, and lost the placement of the value.
  - 2 differed in the order of two loads before a GPU region. The uses changed to match.
- Breaking each check separately:
  - With no type stored on a `match`, every `match` test fails with the new assert.
  - With a wrong type stored on a `match` or an `if`, the tests for it fail.
  - With no type stored on an `if`, the assert in the flat code generator fires.
  - With no type stored on a `spawn`, `transfer_of_a_spawn_value_names_its_source.vx` fails.
  - With no type stored on an `unsafe` block, no test fails. No test found a program where the
    old guess for an `unsafe` block was wrong and something used that type.
- `match_arm_redefines_its_payload.vx` and
  `match_value_in_generic_functions_and_each_position.vx` pass on `main` too. They check the new
  code, not a bug in the old code.
