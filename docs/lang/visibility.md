# Visibility

This document says which code can name an item of a Vx program, and what that means for the
object file and for the version of a library. The design comes from the discussion
[Visibility in Rust and problems associated with it](https://github.com/vx-lang/Vx/discussions/1849)
and the post
[Visibility is a versioning decision](https://hiraditya.github.io/posts/visibility-is-a-versioning-decision/).
Section 8 lists the steps that build it, in order.

## 1. The four levels

Each level opens one boundary more than the level before it.

| Level | Written | Code that can name the item |
| --------- | ------------ | ----------------------------------------------- |
| private | `fn f()` | The file that declares it. A file is a module. |
| package | `pkg fn f()` | Each file of the package of that file. |
| workspace | `ws fn f()` | Each package of the workspace group of that package. |
| public | `pub fn f()` | Any file. This is the API that the package exports. |

The levels have an order: private < `pkg` < `ws` < `pub`. Section 6 uses this order.

Rust has `pub(crate)`, `pub(super)` and `pub(in path)`. Vx has none of them. `pkg` is the level of
`pub(crate)`. A Vx module is one file and has no parent module, so `pub(super)` has no meaning.

## 2. Packages and workspace groups

A package is one of these:

- `core` or `std`.
- Another library under `stdlib/`, such as `graph` for `import graph::traversal`.
- A package that `vxc --package NAME=DIR` gives.
- The program: the files that `vxc` compiles, and the files that they import from beside them.

The compiler finds the package of a module from the root that its import path resolved against.
It knows this before it reads any function body.

A workspace group is a set of packages that a manifest declares, and that always release
together. Vx has no manifest yet, so `ws` waits for it (section 8).

## 3. Where a level can be written

- A function, an associated function, a method of an inherent `impl`, and a function in an
  `extern` block can have each of the four levels. So `std::libc` writes
  `pub fn malloc(size : usize) -> *mut i8;`.
- A method of a trait, and a method of a trait impl, have no level of their own. They have the
  level of the trait. Otherwise an implementor and a caller could disagree about what the trait
  contains.
- `main` has no level. The runtime calls it, and a program cannot (E5004).
- The level comes first: `pub unsafe fn`, `pkg extern fn`. The parser refuses `unsafe pub fn`.
- Structs, enums, traits and constants are visible to each file today. Section 7 gives them levels.

## 4. The rules

1. A call of a function, or its use as a value, must be in code that can name the function.
   Otherwise the call is the error E5005, which names the function and its file.
1. The body of a generic function belongs to the file that declares it, in each copy that the
   compiler makes for a type. So a `pub` generic function can call the private functions of its
   file, wherever a program uses it.
1. A name that a macro definition writes has the privileges of the file that defines the macro. A
   name that the caller gives the macro as an argument has the privileges of the caller. So `std`
   can write a macro that calls a `pkg` function of `std`. A caller cannot use the macro to reach
   a function that its own code cannot reach. Code that an MLIR dialect generates follows the same
   rule.
1. A call that the compiler makes, such as `drop`, or `Iterator::next` for a `for` loop, is not
   text of the program. It gets no check.
1. A `pub` item must be reachable from outside its package. Today each module of a package can be
   imported, so each `pub` item is reachable. If Vx gets a module that only its own package can
   import, a `pub` item in it is an error.
1. A signature names only types that are at least as visible as the signature. A `pkg` type
   cannot be in the signature of a `pub` function. An opaque return type, such as `-> impl Trait`,
   is the exception. Today each type is public, so the rule holds for each signature.
1. Visibility is separate from placement. A `pkg` kernel can start from the host, as a `pub`
   kernel can.

The check of a call reads the level and the file of the callee, and the packages of the two
files. Each of these comes from a declaration or an import path, so no global fact depends on a
function body
([language_rules_for_a_parallel_compiler.md](../language_rules_for_a_parallel_compiler.md)).

## 5. Visibility and the object file

The level of an item sets the most that its symbol can be exported. The kind of output can only
narrow it. A package can be built as an object file, a static library, a shared library or an
executable, and the level of an item is the same in each.

| Level | Object file or static library | Shared library | Executable |
| ------------ | ----------------------------- | ------------------------------------ | -------------- |
| private | internal linkage | internal linkage | internal linkage |
| `pkg` | hidden | hidden | hidden |
| `ws` | hidden | exported, with a private version tag | not exported |
| `pub` | hidden | exported, with a public version | not exported |
| `extern fn` | exported | exported | exported |

- `extern fn` is the entry point for C. Each kind of output exports it under its C name
  ([abi.md](abi.md), section 4). Its level says only which Vx files can call it, so a function that C and
  other Vx files call is a `pub extern fn`.
- Each kind of output exports the entry point of a kernel to the dispatch runtime, as for an
  `extern fn`.
- A `ws` symbol of a shared library has the version tag `VX_<group>_<fingerprint>_PRIVATE`. The
  fingerprint is a hash of the sources and the compiler of the group build. So a library built
  against another group build fails to load, with a symbol that the loader cannot find. Without
  the fingerprint, it could run with an ABI that does not match. Packages that deploy on their own
  must talk through `pub`.
- A `pub` symbol of a shared library needs a stable Vx ABI: the names of symbols, the calling
  convention, and generic functions that each caller copies for its types. `pkg` and `ws` need
  none of this.

Today Vx builds executables and object files, and it links Vx code statically. So today the
compiler exports only `extern fn`, `main` and the entry points of kernels. Each other function is
`hidden` and `dso_local`. Internal linkage for private functions, and shared libraries that
export `pub`, come later (section 8).

## 6. Visibility and versions

A release of a library must raise its version by the most that any change in it needs. The
level of the item that a change touches sets what the change needs.

- The obligations have an order: none < patch < minor < major.
- `b(kind, level)` is the obligation of one change of the given kind to an item of the given
  level. It only grows as the level grows. A change to a private item needs no new version. The
  same change to a `pub` item can need a major one.
- The bump of a release is the largest `b` of its changes:
  `bump(release) = max { b(kind(c), level(c)) | c in diff(old, new) }`.
- A change can keep the source of a caller working and still break the ABI. For example, a new
  private field changes the layout of a `pub` struct. So there are two functions: `b_src` for a
  caller that compiles again, and `b_abi` for a binary that does not.
- With separate levels to read and to write a field (section 7), and with sealed traits, the
  levels are a partial order, not a line. `b` must still only grow along it.

`src/api_diff.rs` compares the `.vxlib` files of two versions and reports patch, minor or major
(#2251). The interface file keeps no level yet, so it counts each item as `pub` (#3900).

## 7. Levels for other items

**Fields** get one level to read and one to write. The level to write is never more than the level
to read. Any file can read `len` below, and only this file can write it. Any file can read
`cursor`, and each file of the package can write it.

<!-- vx-doctest: skip (the syntax of a later step: a field has no level yet) -->

```rust
struct Stream {
  pub get len : u64,
  pub get pkg set cursor : u64,
  pkg buf : *mut u8,
}
```

The other items:

- **Traits** can be `sealed`. Any file can use a `pub sealed trait`, but only its own package can
  implement it.

- **Impls** can have a level. Only the package can use a `pkg impl Debug for Plan`. The impl still
  holds its place for coherence in the whole program, so no other package can write a second one.

- **Types and constants** get the four levels, and rule 6 of section 4 applies to them.

A capability type then needs only a private field, and no `invariant true`.

## 8. Steps

1. A function with no level is private to its file, `pub` makes it public, and `extern fn`
   exports it to C (#3895, #3896, #3901). The error is E5005.
1. `pkg fn`.
1. The helpers of `std::sync` in #3884 become private or `pkg`. So they stay safe functions.
1. Macros get the privileges of the file that defines them (rule 3).
1. The interface file keeps the level of each item (#3900). The API diff then uses it (#2251).
1. Private functions get internal linkage.
1. Types, constants, fields, traits and impls get levels, with rule 6 of section 4.
1. A manifest declares workspace groups. `ws` and its fingerprint follow.
1. A shared library built from Vx exports its `pub` symbols, with version tags.

## 9. Open questions

1. An `impl` that a macro writes belongs to the file that calls the macro (#3754). So a macro
   cannot write an impl of a type with an invariant outside the file of the type (E3066). Must a
   macro in the file of the type be able to? Rule 3 suggests yes.
1. The identifier of a function keeps no level, as it keeps no `unsafe` (#489). Must two functions
   that differ only in level have two identifiers?
