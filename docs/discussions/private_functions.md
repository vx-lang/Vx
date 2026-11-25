# Private functions

Status: proposed, for review. Issue: #3890. Needed for #3884 (safe functions of `std` that take a
raw pointer) and #3750 (capability types).

`docs/lang/visibility.md` (#3934) is now the spec of visibility. It changes two
points of this note. The level `pkg` answers the first open question: the helpers of #3884 that
five files of `std::sync` share become `pkg fn`. And a name that a macro writes has the
privileges of the file that defines the macro, not of the file that calls it (rule 7).

Today each function of a file is visible to each file that imports it. So a helper of `std`, such
as `word(words : *mut AtomicI64, i : i64)` in `std::sync::channel`, is part of the API of `std`.
The helper reads through its raw pointer, and any program can give it a wrong one, so safe code can
corrupt memory (#3884). Rust makes such a helper sound by keeping it private to its module. This
note gives Vx the same rule: a function with no `pub` is private to its file.

## 1. The rules

1. A function, an associated function or a method is private unless its declaration starts with
   `pub`.
1. Only code in the file that declares a private function can call it, or use it as a value.
   Code in another file gets an error that names the function and its file, and says to write
   `pub fn` there.
1. A method of a trait impl has no `pub`. Each one is public, because the trait is public.
1. A function in an `extern` block is private too, unless its declaration starts with `pub`. So
   `std::libc` writes `pub fn malloc(size : usize) -> *mut i8;`.
1. `main` needs no `pub`. The runtime calls it, and a program cannot (E5004).
1. `extern fn` exports a function to C with its own name, as `pub fn` does today
   (`docs/lang/abi.md`, section 4). Its `pub` says only which Vx files can call it, so a
   kernel that C and other Vx files call is a `pub extern fn`. Only a top-level function that
   is not generic can be `extern`. A `pub fn` keeps a hidden symbol, as each other Vx function
   does. An `extern fn` has a body in Vx. An `extern` block declares the functions that C
   defines.
1. Code that a macro writes belongs to the file that calls the macro, as for an `impl` (#3754). So
   the code of a macro can call only the `pub` functions of another file.
1. The check reads the text of a call. A call in the body of a generic function is a call from the
   file of that body, in each instantiation. A call that the compiler makes, such as `drop` or
   `Iterator::next` for a `for` loop, is not text of the program, and gets no check.

Types, fields, traits and constants stay visible to each file in this step. Private fields can
come next, as in Rust. A capability type would then need only a private field, and no
`invariant true`.

## 2. The rules of a parallel compiler

The check of a call reads two facts: the `pub` of the declaration of the callee, and the file of
the declaration. Both are in the signature. So no global fact depends on a function body
([language_rules_for_a_parallel_compiler.md](../language_rules_for_a_parallel_compiler.md)).

## 3. What changes

- **`std` and `core`.** Each function of the API gets `pub`. A helper stays private. The helpers
  of #3884, such as `word`, `release` and `lock_channel`, then stay safe functions, because only
  their file can give them a pointer.
- **Tests.** 54 test files import a file that is not in `std` or `core`, such as
  `tests::modules::iter`. The functions that they call get `pub`.
- **C exports.** About 560 functions in the tests are `pub fn`. A test that reads the C name of
  one, for example in its symbols or as a kernel entry, changes it to `extern fn`.
- **Apps.** qwen-vx and glm-vx add `pub` to the functions that their other files call, and
  `extern` to the functions that C calls. Vx is at 0.0.5, so this change keeps no old form.

## 4. Steps

Each step leaves main green.

1. Parse `pub` on a method and on an associated function, and parse `extern fn`. In this step,
   `pub fn` and `extern fn` both export to C.
1. Change the tests that read a C name to `extern fn`. Then `pub fn` stops exporting to C.
1. Add `pub` to the API of `std` and `core`, and to the test files that other files import.
1. The check of rule 2, with a new error code.
1. #3884: the helpers of `std::sync` stay private. The other functions of #3884 get a pointer
   clause, take a reference, or become `unsafe fn`.

## 5. Open questions

1. Rust has `pub(crate)`: visible in the package, and not outside it. Vx has packages for
   `vxc --no-trusted-code NAME`. Does a package need such a level, for helpers that two files of
   `std` share? Until it does, such a helper is `pub`, and it must be sound as a public function.
1. Does a macro need a way to call a private function of its own file, as `$crate` does in Rust?
