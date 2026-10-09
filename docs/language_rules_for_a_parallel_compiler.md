# Language rules for a parallel compiler

Vx checks every function body in parallel, with no locks. That works because, before any body
is checked, the compiler can collect every global fact about the program — every signature,
every type's layout, every impl — and freeze it. A body only reads that frozen table.

This page lists the language rules that make the freeze possible. **Use it as a checklist for
new language features:** a feature that breaks one of these rules makes a global fact depend on
a function body, and then bodies can no longer be checked independently.

## The rules

| Vx rule | What breaks it in C++ | What breaks it in Rust |
| --- | --- | --- |
| **A signature is the whole contract.** Every parameter and return type is written out; nothing about a return type is read from the body. | `auto f()`: checking a caller needs the callee's body | `-> impl Trait` and `async fn`: whether the result is `Send` comes from the body |
| **Every layout is known at the freeze.** A recursive type goes through a box, and no size is computed by running a function. | | `[u8; n()]`: a struct's size comes from running a `const fn`, whose body must be checked first |
| **A generic closes at its definition.** `Vec<i32>` means the base plus its arguments, nothing else. | `template<> struct hash<MyType>` after use; point-of-instantiation lookup and ADL | Specialization (nightly; `std` uses it): a more specific impl changes an already-written call |
| **One name, one function.** A call is resolved by name; there is no overloading. | Open overload sets: any header can add a candidate to an existing call | |
| **No implicit conversions.** Every conversion is an explicit `as`; a number literal takes its type from where it is used. | Converting constructors and `operator T()`: which overload wins depends on every conversion in scope | |
| **Copy and drop belong to the type.** They are settled at the freeze. | `operator=` and other special members are generated on first use, inside whichever body needs them first | |
| **Macros expand in their own phase**, before any name resolves. | The preprocessor: a header means something different in each file that includes it | Macro expansion and name resolution run as one loop until nothing changes; procedural macros run arbitrary code |
| **A body declares nothing global.** Nothing inside a function body, including code under `if comptime`, adds a declaration. | | `impl Trait for S` written inside a function body applies to the whole crate |

One global fact can still appear inside a body: a **new generic instantiation**. That one
exception has its own machinery (deferred identity, in
[`parallel_compiler_architecture.md`](parallel_compiler_architecture.md)).

## What breaking them costs

Rust shows the cost. rustc has worked on a parallel front end since 2018 (rust-lang/rust#48685,
then #113349, still open). It has to discover dependencies while it checks bodies, so it uses a
query engine with locks and cycle detection. The result, with 8 threads, is up to 50% less
front-end time and up to 35% more memory
([Rust blog, Nov 2023](https://blog.rust-lang.org/2023/11/09/parallel-rustc/)). In 2026, name
resolution and macro expansion are still serial
([project goal](https://goals.rust-lang.org/2026/parallel-front-end.html)).
