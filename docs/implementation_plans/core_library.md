# The Vx core library: a plan for parity with Rust's `core`

**Status:** proposal, 2026-09-13. Tracking issue: Vx#451.
**Scope:** the first layer of a rewritten standard library, `core`, and the compiler work it needs.
The `alloc`/`std` layers above it are sketched only where a decision here constrains them.

______________________________________________________________________

## 0. Summary

Rust's standard library is three crates stacked on each other: `core` (no operating system, no
allocator, no threads), `alloc` (heap types over an allocator), and `std` (the operating system).
Vx today has one 1,358-line `std` in which the layers are mixed: `Option<T>` and the `Iterator`
trait sit beside `TcpStream`. This plan adds a `core` layer with Rust's structure and Rust's public
API as its checklist, and moves the pieces of today's `std` that belong there.

The decisions this document makes:

1. **`core` lives at `stdlib/core/` and is imported as `import core::<module>;`.** The module
   loader already treats `stdlib/` as a search root, so `core::iter` resolves to
   `stdlib/core/iter.vx` with no compiler change. Nested paths such as `core::sync::atomic`
   resolve too (verified, §9).
1. **`core` contains no `extern` block.** That single rule is what makes it usable inside a
   `spawn on(...)` region on a device that has no libc. Where Rust's `core` reaches for
   `core::intrinsics`, Vx's `core` reaches for `mlir!`, which is portable across backends by
   construction. A test enforces the rule (§7).
1. **Parity is defined against Rust's stable public API, module by module**, with a checked-in
   matrix (Appendix A) that every `core` PR updates. "At par" means every stable item in a Rust
   `core` module has a Vx counterpart or an entry in the exclusions list with a reason.
1. **The library cannot be written before the language grows.** Fourteen probes against the
   current compiler (§4, §9) show that the constructs Rust's `core` is made of are missing or
   broken: trait bounds on `impl` generics are not usable inside the impl, traits cannot have
   generic methods or default bodies, there are no associated types, no `%` or bitwise operators,
   no multiple bounds, and the `for x in iter` protocol does not run. The plan is therefore two
   interleaved tracks: **Track A**, language enablers ordered by how much of `core` each unlocks,
   and **Track B**, library modules phased so each phase needs only what Track A has landed.
1. **Existing `std` stays working throughout.** Each `core` module lands with the `std` module it
   replaces deleted in the same PR, and every `import std::x` in the tree updated. Nothing is
   duplicated for longer than one PR.

What this plan does not do: it does not design `alloc` (`Vec`, `String`, `Box`, `HashMap` in Vx)
or `std` beyond saying what they will need from `core`. It does not decide `Drop`, but it says
where the decision bites (§8).

______________________________________________________________________

## 1. What "at par with Rust's `core`" means

Rust's `core` is large: about forty public modules, roughly seventy-five provided methods on
`Iterator` alone, around sixty on `Option`, several hundred on the integer types once the
`checked_*`/`wrapping_*`/`saturating_*`/`overflowing_*` families are counted. A claim of parity has
to be checkable or it is a slogan. The definition used here:

- **The unit of parity is a Rust `core` module** (`core::cmp`, `core::iter`, ...). For each, the
  spec is the module's stable public API in the current Rust release, read from the Rust docs, not
  from memory.
- **A module is at par when** every stable public item (trait, method, function, type, constant)
  either has a Vx counterpart with the same name and meaning, or is listed in that module's
  exclusions with a one-line reason. Exclusions are legitimate: `core::future` has no meaning in a
  language without `async`; `core::any` needs runtime type identity Vx does not have.
- **Behaviour parity is tested, not asserted.** Each module ships with `stdlib/core/tests/<module>.vx`
  whose cases are ported from Rust's own `coretests` where they apply, run with `vxc --run`.
- **Appendix A is the scoreboard.** It is updated in the same PR as the code. The tracking issue
  links to it. A module moves to "at par" only when the exclusions list is reviewed.

Two things parity does **not** require: matching Rust's implementation strategy, and matching Rust
where Vx has a better answer. Float math is the example: Rust's `core` cannot provide `f32::exp`
because it needs libm, so `exp` is in `std`. Vx's `core` can provide it through `mlir!` and the MLIR
`math` dialect, which every backend lowers. `core::num` in Vx will therefore be a superset of Rust's
on floats, and the matrix records that.

______________________________________________________________________

## 2. Ground truth: what exists today

Read with §9, which is the probe transcript.

**The library.** 21 modules, 1,358 lines, under `stdlib/std/`. Five of them are `extern`
declaration lists with no Vx type or function at all (`alloc`, `hash_map`, `hash_set`, `libc`,
`mmap`, `result`). `result.vx` declares a Rust-backed `Result<i32, i32>` reached through six C
symbols; there is no Vx `Result` enum. Two modules do not type-check on their own and are on an
exemption list in the test suite (`iter.vx`, `tensor.vx`, Vx#487).

The pieces that already belong to `core`, and will move:

| Today | Lines | Note |
| --- | --- | --- |
| `std/option.vx` | 51 | `enum Option<T>`, `is_some`, `is_none`, `unwrap`. Pure Vx. |
| `std/iter.vx` | 45 | `trait Iterator<T, Item>` with `Item` as a trait parameter, a `Map` adaptor whose struct is never declared (why Vx#487 lists it). |
| `std/closure.vx` | 22 | `Closure0..3<..>`, the nominal record the compiler lowers a closure literal into. Vx's `Fn` traits, in effect. |
| `std/math.vx` | 130 | `trait Math` over f32/f64, every method a libm `extern`. Moved: the methods are inherent on `f32`/`f64` in `core::num` now, over `math` dialect ops, and this module is gone. |
| `std/googletest.vx` | 35 | Stays in `std` (it calls into Rust) but is the assertion vocabulary `core`'s tests use. |

What the repo already decided, in the design document that preceded this one (removed from the
tree in `b30fabe7`, recoverable from history): drop to native only at irreducible primitives
(transcendental math, syscalls, raw allocation, opaque data-structure cores); methods via traits
by default, free functions for symmetric operations and constructors; the stdlib is the flat
pipeline's dogfood; and an open question, "a `no_std`-like core/full split for accelerator
targets", which this document answers.

**The compiler, as it bears on a library.** The full inventory is in §9. The short version: generic
structs, enums, functions and impls work and are monomorphized; a trait bound on a *function's*
generic is enforced and usable; a trait bound on an *impl block's* generic is parsed and then
ignored, so a conditional impl cannot call the method it is conditional on; traits have neither
default bodies, nor generic methods, nor associated types; there are no tuples, slices, `char`,
`str`, `usize`, `const` items, attributes, or `pub`; operators stop at `+ - * / @` and the
comparisons; closures are values of `Closure0..3` and are invoked by hand through `.func`/`.env`;
`for x in iter` over a `next()` method does not compile. Imports are whole-module and transitive.

______________________________________________________________________

## 3. The shape of `core`

### 3.1 Layout

```text
stdlib/
  core/                 <- this plan
    prelude.vx          re-declares nothing; imports the modules a program always wants (§3.4)
    marker.vx  cmp.vx  ops.vx  clone.vx  default.vx  convert.vx
    option.vx  result.vx
    num.vx              (or num/int.vx, num/float.vx if it grows past ~1,500 lines)
    mem.vx  ptr.vx  hint.vx  panic.vx
    iter.vx  slice.vx  str.vx  char.vx  ascii.vx
    fmt.vx  hash.vx
    cell.vx  time.vx  alloc.vx  ffi.vx  error.vx
    sync/atomic.vx
    tests/<module>.vx   one runnable test file per module, `vxc stdlib/core/tests/cmp.vx --run`
  std/                  today's modules, shrinking as core absorbs them; later split into alloc/ and std/
  graph/                unchanged
  rust_core/            unchanged; core never links it
```

Import path: `import core::cmp;`. No leading-component rewrite is needed: `std::x` is special-cased
to `stdlib/std/x.vx`, and every other path is taken verbatim under the search roots, so
`core::sync::atomic` is `stdlib/core/sync/atomic.vx` (verified, §9 A2).

### 3.2 The purity rule

**No `extern` block anywhere under `stdlib/core/`.** Everything `core` needs from below the
language it gets from `mlir!`. This is the property that lets a `spawn on(Topology::GPU)` body call
`x.exp()` or `a.checked_add(b)` without the dispatcher falling back to libffi on the host. It is
enforced by a test (§7.2) and it is the reason `core::num`'s float functions are written over the
`math` dialect rather than over `expf`.

Two consequences worth stating:

- `core` has no I/O, not even `print`. Its tests use `std::googletest`, which is fine, because
  tests run on the host.
- `core` cannot panic through the runtime's `vx_panic`, and it cannot panic at all yet.
  `assert(false, "msg")` does not compile: a statically false assert is reported at check time
  whether or not anything calls the function holding it, so a function that always panics
  cannot be written (Vx#526). `Option::unwrap` works because its condition comes from a method
  call the checker cannot fold, which is a shape only a method with a receiver has. So
  `core::panic` waits on the intrinsics rather than shipping in a reduced form.

### 3.3 Traits before types

Rust's `core` is mostly traits, and the concrete types (`Option`, `Ordering`, `Duration`) are
small. The same is true here. The order of work inside a phase is always: the trait, its impls for
the primitives (stamped out with `macro_rules` once macros can produce items, by hand until then),
its impls for `core`'s own types, then the free functions.

### 3.4 The prelude

Every Rust program sees `Option`, `Some`, `None`, `Result`, `Ok`, `Err`, `Iterator`, `Clone`,
`PartialEq`, ... without importing them. Vx has no `use` and no glob, but imports are transitive
(§9 A1): a module that imports `core::option` exposes `Option` to whoever imports it. So
`core/prelude.vx` is a module with no declarations and a list of imports, and a program writes
`import core::prelude;` once. Making that import implicit is a small compiler item (A12).

Transitivity cuts the other way: with no `pub`, every helper function in `core` is visible to every
program that imports it, and a collision is a hard error. Until `pub` exists (Vx#489), `core`'s
private helpers are named with a `core_` prefix and its public names are exactly Rust's. That is a
convention, and the parity review checks it.

### 3.5 What `core` is not

- Not the tensor library. `Tensor<T, [..]>` is a built-in type with compiler-known operators; the
  methods in today's `std/tensor.vx` are placement-aware kernels. They stay in `std`. `core` may
  gain a `Tensor` impl of a trait (`Default`, later `Index`) but defines no tensor operation.
- Not `alloc`. `Box`, `Vec`, `String`, `HashMap` need an allocator and a destructor story; they are
  the next plan. `core` defines the `Allocator` trait and `Layout` they will be written against
  (phase 3), which is where Rust puts them too.
- Not async. `core::future`, `core::task`, `core::pin` are excluded with the reason "no `async` in
  the language".

______________________________________________________________________

## 4. Track A: what the language must grow, and in what order

Each item is ordered by how much of `core` it unlocks divided by how much compiler it costs.
Sizes: **S** is one focused PR; **M** is a short series; **L** deserves its own tracking issue.
"Evidence" points at §9 or an existing issue. Items marked **P1** must land before phase 1 of
Track B can be completed; the rest are tagged with the phase that needs them.

| # | Feature | What in `core` needs it | Today | Size | Needed by |
| --- | --- | --- | --- | --- | --- |
| A1 | **Trait bounds on `impl` generics are enforced and usable inside the impl.** `impl<T : PartialEq> PartialEq for Option<T> { .. self.v.eq(other.v) .. }` | Every conditional impl: `PartialEq`/`Ord`/`Clone`/`Default`/`Hash` for `Option<T>`, `Result<T, E>`, every iterator adaptor's `Iterator` impl. Without it `core` is impls on primitives only. | Bound parsed, then ignored; the body fails with "Method not found on type T" (§9 D2, D5). The check at `calls.rs:1087` runs over function generics only. | M | **P1** |
| A2 | **Default method bodies in traits.** | `Iterator` (about 75 provided methods over one required `next`), `PartialOrd` (`lt`/`le`/`gt`/`ge` over `partial_cmp`), `Ord` (`max`/`min`/`clamp`), `Hasher` (`write_u8`.. over `write`), `fmt::Write`. Without it every impl re-implements the whole surface. | `parse_trait_decl` requires `;` after the signature (`parser/decl.rs:896`). | M | **P1** |
| A3 | **Generic methods in traits.** `trait Iterator { fn map<B>(self, f : Closure1<Item, B>) -> Map<Self, B>; }` | `Iterator::map/filter/fold/zip/...`, `Option::map/and_then`, `Result::map_err`, `Hash::hash<H : Hasher>`. | Parse error at the `<` (§9 M2). Generic methods in *inherent* impls work (`VecIter::map`). | S–M | **P1** |
| A4 | **Multiple bounds** `T : A + B` **and trait `where` clauses.** | `fn sort<T : Ord + Copy>`, `fn sum<T : Add + Default>`, `impl<I : Iterator + ExactSizeIterator>`. | Parse error at `+` (§9 M5); `where` accepts only `Reachable<A, B>`. | S | **P1** |
| A5 | **Integer operators** `% & \| ^ << >>` **and the compound assignments** `-= *= /= %= &= \|= ^= <<= >>=`. | All of `core::num`: `pow`, `leading_zeros`, `count_ones`, `rotate_left`, `is_power_of_two`, `from_str_radix`, `rem_euclid`; every hasher; `char` decoding. | Not in the lexer (§9 E1, E2). Only `+=` exists. | S | **P1** |
| A6 | **`Copy` as a marker trait the move checker honours.** `impl Copy for Ordering {}`; a struct or enum all of whose fields are `Copy` may be declared `Copy`. | `Ordering`, `Option<i32>`, `Duration`, `Range<i32>`, `Char`. Today a struct is linear by construction (`Type::is_linear`), so `let b = a;` consumes `a` for every `core` type. | M | **P1** |
| A7 | **Item-position macros.** `macro_rules` expanding to `impl`/`fn`/`struct` items. | Stamping `impl Ord for i8/i16/.../u128`, the `checked_*` families, `From` between widths. Rust's `core::num` is written this way. Without it the integer module is around 4,000 hand-written lines instead of 400. | Expression-position only (§9 G). | M | **P1** for `num` |
| A8 | **`panic(msg)`, `unreachable()`, `todo()` intrinsics** lowering to `cf.assert`/trap, with a message. | `unwrap`, `expect`, `unreachable_unchecked`'s safe sibling, every bounds check `core` writes. | `assert(cond, "msg")` exists; `assert(false, ..)` is the workaround. | S | P1 (nice), P2 (needed for `expect`) |
| A9 | **The `for x in iter` protocol.** Lower `for` over any value with `fn next(self : &mut Self) -> Option<Item>`; take the iterable by `&mut`, not by move. | `core::iter` is unusable from a `for` loop without it; today the only way to drive an iterator is a hand-written `loop { match it.next() .. }`. | Three errors (§9 I): the iterable is consumed, `next` is not found, and the induction variable is typed `i64`. The AST path resolves `next` by string-matching symbol names. | M | P2 |
| A10 | **Associated types in traits.** `trait Iterator { type Item; fn next(..) -> Option<Self::Item>; }` | `Iterator::Item`, `IntoIterator::IntoIter`, `Add::Output`, `Deref::Target`, `Index::Output`. Phase 1 encodes these as extra trait parameters (`Iterator<Self, Item>`, the encoding today's `std::iter` already uses); phase 2 replaces it. Keeping the encoding is possible but it leaks into every bound a user writes. | `Self::Item` landed in Vx#527: a trait declares `type Item;` and each impl binds it. `I::Item` landed in Vx#727: resolved when a generic is instantiated, from the impl for what `I` is. A struct field cannot name one (E3041), so an adaptor takes its closure as a type parameter, as Rust's do. | L | P2 |
| A11 | **Operator dispatch to traits.** `a + b` on a non-builtin type calls `Add::add`; same for the comparison operators over `PartialEq`/`PartialOrd`, and `a[i]` over `Index`. | `core::ops` means nothing without it; `Wrapping<T>`, `Duration`, `Saturating<T>`, and `==` on `Option<T>` all depend on it. The tensor operators are already dispatched specially in `check/operators.rs`; this generalizes that hook. | Not present. | M | P2 |
| A12 | **Implicit prelude import**, off with `--no-prelude`. | Ergonomics only; §3.4. | Imports are transitive, so a manual `import core::prelude;` works today. | S | P2 |
| A13 | **Generic struct with an enum field lowers.** `struct Peekable<I, T> { it : I, peeked : Option<T> }` | `Peekable`, `Fuse`, `Chain`, `Cycle`, `Rev`; `Cell<Option<T>>`; `OnceCell`. | Flat path declines ("struct with no GID"); AST path fails MLIR verification (§9 M3). | M | P2 |
| A14 | **A generic enum constructed at a method-level type parameter lowers.** `fn map<U>(..) -> Option<U> { return Option<U>::Some(..); }` | `Option::map`, `Result::map`, `Iterator::map`'s `next`. | `unrealized_conversion_cast i32 to Opt_i32` on both code generators (§9 M1). | M | **P1** for `option` |
| A15 | **`char` and `str` as language types**, or the compiler treating a string literal as `(ptr, len)`. | `core::char`, `core::str`. Phase 2 ships them as library types (`Char { code : u32 }`, `Str { ptr : *const u8, len : i64 }`), which is faithful to what Rust's are underneath. What the library cannot do is know a literal's length without scanning for NUL, or write `'a'`. | String literal types as `*const i8`; indexing one and comparing the byte fails in the AST code generator (§9 M4). | M | P2 |
| A16 | **Tuples** (types, literals, patterns). | `zip`, `enumerate`, `split_at`, `Option::zip`, `unzip`, `overflowing_add -> (T, bool)`. Phase 1 uses `Pair<A, B>` / `Triple<A, B, C>` structs in `core::tuple`; tuples replace them. | None. | L | P3 |
| A17 | **Slices `&[T]` / `&mut [T]` as fat references.** | `core::slice` is a quarter of Rust's `core`. Phase 2 ships `Slice<T>` / `SliceMut<T>` structs over `(ptr, len)` with a `get`/`set` API; `&[T]` sugar and `s[i]` indexing come with A11 and A17. | `Vec::as_slice` returns `&T`. | L | P3 |
| A18 | **Const generics on methods.** | `core::array`: `[T; N]::map`, `from_fn`, `IntoIter`. | `const_generics_methods.vx` is `XFAIL`. | M | P3 |
| A19 | **`Drop`.** | Not for `core` itself (its owning types are `Cell` and `ManuallyDrop`, which are trivial), but `mem::drop`, `mem::forget`, `ManuallyDrop` and `MaybeUninit` only mean something once it exists, and every `alloc` type needs it. Vx#495 is the decision. | Reserved bit, nothing set. | L | P3 / alloc |
| A20 | **`pub` visibility** (Vx#489) and **`while`** (Vx#506), **`if let`**, **`?`**. | Quality of life for writing `core`; none is required for its API. `?` desugars to a `match` on `Result`; `if let` to a two-arm `match`. | None. | S each | any |

Explicitly **not** required: `dyn Trait` (Rust's `core` uses it only for `fmt::Arguments` and
`Any`; the Vx `fmt` design in §5 avoids it), attributes (`#[derive]` is replaced by item macros
plus hand-written impls, as Rust's `core` itself is), lifetimes syntax (the region checker infers),
`usize` (Vx indexes with `i32`/`i64`; §8 decides which `core` standardizes on).

The P1 set is A1–A7 and A14. Of those, A4, A5 and A7 are parser-and-lowering work of a day or two
each; A1, A2, A3 and A14 are the substantive ones and are where the type checker and both code
generators are touched. A6 needs a decision (§8.2) before it needs code.

**Progress.** A5 is done, in four commits — one per operator family, plus the compound
assignments. A4 is done. Both turned up something the table did not predict, which is recorded
here because it changes what the remaining items are worth: `%` was the only way to spell an MLIR
value name, so making it an operator stopped every module with an inline `mlir!` block from
parsing; and a bound constrains *callers* only. A generic body is checked after monomorphization
against the concrete type, so an unbounded `T` can already call any method that type has. That
last point narrows A1: what fails there is method resolution inside a generic `impl` block, which
may have nothing to do with the bound written on it. Confirm that before sizing the work.

______________________________________________________________________

## 5. Track B: the modules, phase by phase

For each module: the Rust module it mirrors, what lands, the key signatures in today's Vx spelling
(`self : Self` written out, no `&self` shorthand), what is excluded, and what it depends on. Where
the Vx encoding differs from Rust's the difference is called out; everything not called out is a
straight port and the Rust docs are the spec.

Code in this section is the proposed API, not compilable today.

### Phase 1: foundations

Needs A1–A7, A14. Delivers the traits every later module is written against, and the two enums.
Nothing in this phase iterates, formats, or touches memory beyond `mem`/`ptr`.

#### `core::marker`

`Copy`, `Sized` (a no-op marker kept for signature compatibility), `PhantomData<T>` (an empty
generic struct; empty structs lower today, §9 C). `Send`/`Sync` are declared but mean nothing until
the language has threads; the matrix marks them "declared, not enforced".

#### `core::cmp`

<!-- vx-doctest: skip -- proposed API, not yet compilable -->

```rust
enum Ordering { Less, Equal, Greater, }          // Copy

trait PartialEq<Rhs> {
  fn eq(self : &Self, other : &Rhs) -> bool;
  fn ne(self : &Self, other : &Rhs) -> bool { return !self.eq(other); }     // A2
}
trait Eq<Rhs> {}                                  // marker over PartialEq
trait PartialOrd<Rhs> {
  fn partial_cmp(self : &Self, other : &Rhs) -> Option<Ordering>;
  fn lt(self : &Self, other : &Rhs) -> bool { .. }  fn le .. fn gt .. fn ge ..   // A2
}
trait Ord {
  fn cmp(self : &Self, other : &Self) -> Ordering;
  fn max(self : Self, other : Self) -> Self { .. }  fn min .. fn clamp ..      // A2
}
fn max<T : Ord>(a : T, b : T) -> T;   fn min<T : Ord>(a : T, b : T) -> T;
fn max_by<T>(a : T, b : T, f : Closure2<&T, &T, Ordering>) -> T;  // and min_by, max_by_key, min_by_key
struct Reverse<T> { v : T, }         // with the inverted Ord impl (A1)
impl Ordering { fn reverse, then, then_with, is_lt, is_le, is_gt, is_ge, is_eq, is_ne }
```

Impls for every scalar width and `bool`, stamped by an item macro (A7). `Option<T>` and
`Result<T, E>` impls in `option`/`result` (A1). `Rhs` is a trait parameter defaulting to `Self` in
Rust; Vx has no defaults, so phase 1 writes `PartialEq<Self>` everywhere and A10's work decides
whether trait parameters get defaults.

Exclusions: none. This module is the first parity target and the one the A1/A2 work is validated
against.

#### `core::ops`

Phase 1 declares the traits and implements them for the scalars; they are callable as methods
(`a.add(b)`) and become operator-dispatched in phase 2 (A11).

`Add Sub Mul Div Rem Neg Not BitAnd BitOr BitXor Shl Shr` and the `*Assign` forms; `Index`,
`IndexMut` (used by `Slice` in phase 2); `Range<T>`, `RangeInclusive<T>`, `RangeFrom<T>`,
`RangeTo<T>`, `RangeFull` as plain structs (the compiler's `a..b` stays a built-in loop bound until
A9 lets a `Range` be iterated; then `a..b` constructs a `core::ops::Range`); `ControlFlow<B, C>`;
`Fn`/`FnMut`/`FnOnce` are **not** traits here: the callable types are `Closure0..3<..>`, which
move into this module from `std::closure`, plus a `call` method on each so a body writes
`f.call(x)` instead of `let p = f.func; p(f.env, x)`.

`Output` is an associated type in Rust. Until A10, `Add<Rhs, Output>` carries it as a parameter,
which is why `core::ops` is declared in phase 1 but is not marked at par until phase 2.

Exclusions: `Deref`/`DerefMut` (no auto-deref in the language; revisit with A11), `Drop` (A19),
`Fn*` traits (closures are nominal), `Coroutine`, `AsyncFn*`.

#### `core::clone`, `core::default`

`trait Clone { fn clone(self : &Self) -> Self; fn clone_from(..) { .. } }` and
`trait Default { fn default() -> Self; }` (a static method: check that a trait method with no
`self` parses; if not, that is a one-line parser change bundled with A2). Impls for scalars, `bool`,
`Option<T : Clone>`, `Ordering`, the `Range*` types.

#### `core::convert`

Landed: `From<T>` stamped per source-and-target pair over the lossless widenings
(`i8 -> i16 -> i32 -> i64`, unsigned likewise, `u8 -> i16` and so on, the integer-to-float
conversions that do not round, `f32 -> f64`, and `bool` to every width), the reflexive
`From<T> for T` stamped per type, `From<T> for Option<T>`, `Infallible` as an enum with no
variants, and `fn identity<T>(x : T) -> T`. This is the module where A7 pays for itself first,
and it is the first one that needed a trait's arguments in the mangled method name (Vx#686):
without that every `From` impl for one target claimed the same symbol.

Three things this section predicted did not survive contact.

**`Into` and `TryInto` are excluded, and not for the reason given here.** The plan assumed a
blanket `impl<T, U : From<T>> Into<U> for T` was the only obstacle and that stamping per pair
would do instead. It does not: `x.into()` carries no argument, so two `Into` impls for one
source type are indistinguishable at the call site, and the checker refuses the call as
ambiguous (E3035). Vx does not resolve a call from the type its result is assigned to, which
is what makes `into()` work in Rust. One target per source would resolve, and is not worth a
trait.

**A blanket impl whose `Self` is a bare type parameter is never found**, so the reflexive case
is stamped per type. `impl<T> From<T> for T` alone gives "Undefined static method 'i32::from'".
Worth knowing before any other module reaches for a blanket impl.

**`TryFrom` is blocked by Vx#570, not by anything in this section.** It returns
`Result<Self, TryFromIntError>`, and a `Result` whose two payloads have different layouts does
not lower on the flat path when one of them is a struct. The narrowings land with it.

#### `core::option`

Moves from `std/option.vx`. Today's `is_some`, `is_none`, `unwrap` plus the rest of Rust's surface:
`expect`, `unwrap_or`, `unwrap_or_else`, `unwrap_or_default`, `map`, `map_or`, `map_or_else`,
`and`, `and_then`, `or`, `or_else`, `xor`, `filter`, `take`, `replace`, `insert`, `get_or_insert`,
`get_or_insert_with`, `zip` (returns `Pair<T, U>` until A16), `ok_or`, `ok_or_else`, `is_some_and`,
`is_none_or`, `as_ref`, `as_mut`, `iter` (phase 2), `flatten` (on `Option<Option<T>>`, needs A1),
`copied`/`cloned` (on `Option<&T>`), `transpose` (with `Result`). `unwrap_unchecked` is an
`unsafe fn`.

Trait impls: `PartialEq`, `Eq`, `PartialOrd`, `Ord`, `Clone`, `Copy` (when `T : Copy`), `Default`,
`From<T>`, `Hash` (phase 2).

Needs A14 for `map` and everything that returns `Option<U>`, which is why A14 is P1.

#### `core::result`

New. `enum Result<T, E> { Ok(T), Err(E), }` in Vx, replacing the six `vx_result_*_i32_i32`
externs, which are deleted from `stdlib/rust_core` in the same PR along with their Vx
declarations and the `ffi_result.vx` fixture (rewritten against the enum). Surface: Rust's
(`is_ok`, `is_err`, `is_ok_and`, `is_err_and`, `ok`, `err`, `map`, `map_err`, `map_or`,
`map_or_else`, `and`, `and_then`, `or`, `or_else`, `unwrap`, `unwrap_err`, `expect`, `expect_err`,
`unwrap_or`, `unwrap_or_else`, `unwrap_or_default`, `as_ref`, `as_mut`, `iter`, `copied`,
`cloned`, `transpose`, `flatten`, `inspect`, `inspect_err`), same trait impls as `Option`.

#### `core::num` (integers)

The largest phase-1 module and the one that most needs A5 and A7. For each of
`i8 i16 i32 i64 i128 u8 u16 u32 u64 u128` (the `i4`/`u4` and the 8- and 4-bit float types the
language also has are covered by whatever subset lowers; the matrix records each width):

- constants as functions until `const` items exist: `fn i32_min() -> i32`, `fn i32_max() -> i32`,
  `fn i32_bits() -> u32`. (Rust: `i32::MIN`, `i32::MAX`, `i32::BITS`.) This is the one place the
  Vx spelling is visibly worse than Rust's, and it is on the A20 list.
- `checked_add sub mul div rem neg shl shr pow`, `wrapping_*`, `saturating_*`, `overflowing_*`
  (returning `Pair<T, bool>` until A16), `abs`, `signum`, `is_positive`, `is_negative`, `pow`,
  `isqrt`, `ilog2`, `ilog10`, `div_euclid`, `rem_euclid`, `abs_diff`, `count_ones`, `count_zeros`,
  `leading_zeros`, `trailing_zeros`, `rotate_left`, `rotate_right`, `swap_bytes`,
  `reverse_bits`, `to_be`, `to_le`, `from_be`, `from_le`, `is_power_of_two`,
  `next_power_of_two`, `min`/`max`/`clamp` via `Ord`, `from_str_radix` (phase 2, needs `Str`),
  `midpoint`, `unsigned_abs`, `cast_signed`/`cast_unsigned`.
- The overflow-detecting forms are written once over `mlir!`: `arith.addui_extended`,
  `arith.mulsi_extended`, `arith.mului_extended` exist and lower on every backend (verified for
  `addui_extended`, §9 H2); signed add/sub overflow is the sign test over the wrapped result.
  `leading_zeros` and friends are `math.ctlz`/`math.cttz`/`math.ctpop`.
- `Wrapping<T>` and `Saturating<T>` newtypes with the `ops` impls (operator-dispatched in
  phase 2).
- `NonZero<T>`: phase 2 (it wants `Option<NonZero<T>>` to be niche-optimized in Rust; here it is a
  plain struct with a checked constructor, and the matrix says so).

Exclusions: `ParseIntError` until `str` (phase 2), the `f16`/`f128` unstable surface.

#### `core::mem`, `core::ptr`, `core::hint`

`mem::size_of<T>()` (over the existing `sizeof<T>()` intrinsic), `align_of<T>()` (needs a one-line
intrinsic beside `sizeof`), `swap`, `replace`, `take` (needs `Default`), `forget` and `drop`
(no-ops with a doc comment until A19, and the matrix marks them "declared, semantics pending
Drop"), `zeroed<T>()` (unsafe), `transmute` (unsafe, via `mlir!` bitcast where sizes match),
`ManuallyDrop<T>`, `MaybeUninit<T>` (a struct holding `T` with `uninit`, `write`, `assume_init`;
meaningful only after A19 but the API is stable now), `discriminant` (phase 2), `needs_drop`
(false until A19).

`ptr::null<T>()`, `null_mut<T>()` (today spelled `0 as *mut T`), `read`, `write`,
`read_unaligned`, `write_unaligned`, `read_volatile`, `write_volatile` (via `mlir!`
`llvm.load volatile`), `copy`, `copy_nonoverlapping` (via `llvm.intr.memmove`/`memcpy`), `swap`,
`replace`, `eq`, `NonNull<T>`, `addr_of` (excluded: needs a place expression form), `write_bytes`.

`hint::black_box` (an `mlir!` with an empty inline-asm sideeffect, or an identity call the
optimizer cannot see through; test that `-O` does not fold it), `spin_loop` (`llvm.intr.x86.pause`
where available, no-op elsewhere), `unreachable_unchecked` (`llvm.unreachable`),
`assert_unchecked`, `must_use` (identity).

#### `core::panic`

`panic`, `unreachable`, `todo`, `unimplemented`, `assert_eq`/`assert_ne` as functions over `Ord`
and `PartialEq` (macros later), `Location` (excluded until the compiler can pass a call-site span
into a function). Until A8 lands, `panic(msg)` is `assert(false, msg)`.

**Phase 1 acceptance.** `core::cmp`, `core::option`, `core::result`, `core::clone`,
`core::default`, `core::convert` at par (exclusions reviewed); `core::num` integers at par for
`i32`/`i64`/`u8`/`u32`/`u64` with the other widths tracked per width; `core::ops` and `core::mem`
declared. `std/option.vx`, `std/result.vx`, `std/closure.vx` deleted. `tests/backend/pass/*` that
imported them updated. The purity gate (§7.2) and the standalone-check gate green over
`stdlib/core`. The graph library still passes with no change to its source (it imports
`std::vec`, which imports `core::option` transitively).

### Phase 2: iteration, sequences, text, formatting, hashing

Needs A8–A15. This is where `core` becomes something a program is written *with* rather than a set
of definitions.

#### `core::iter`

The trait, with `Item` as an associated type once A10 lands (as a trait parameter before):

<!-- vx-doctest: skip -- proposed API, not yet compilable -->

```rust
trait Iterator {
  type Item;                                                   // A10
  fn next(self : &mut Self) -> Option<Self::Item>;
  fn size_hint(self : &Self) -> Pair<i64, Option<i64>> { .. }
  fn count(self : Self) -> i64 { .. }
  fn last(self : Self) -> Option<Self::Item> { .. }
  fn nth(self : &mut Self, n : i64) -> Option<Self::Item> { .. }
  fn step_by(self : Self, step : i64) -> StepBy<Self> { .. }
  fn chain<U : Iterator>(self : Self, other : U) -> Chain<Self, U> { .. }
  fn zip<U : Iterator>(self : Self, other : U) -> Zip<Self, U> { .. }
  fn map<B>(self : Self, f : Closure1<Self::Item, B>) -> Map<Self, B> { .. }
  fn for_each(self : Self, f : Closure1<Self::Item, void>) { .. }
  fn filter(self : Self, p : Closure1<&Self::Item, bool>) -> Filter<Self> { .. }
  fn filter_map<B>(..) -> FilterMap<Self, B>;  fn enumerate(..) -> Enumerate<Self>;
  fn peekable(..) -> Peekable<Self>;  fn skip_while(..);  fn take_while(..);  fn map_while(..);
  fn skip(..);  fn take(..);  fn scan(..);  fn flat_map(..);  fn flatten(..);  fn fuse(..);
  fn inspect(..);  fn by_ref(..);  fn collect<B : FromIterator<Self::Item>>(self : Self) -> B;
  fn partition(..);  fn fold<B>(self : Self, init : B, f : Closure2<B, Self::Item, B>) -> B;
  fn reduce(..);  fn all(..);  fn any(..);  fn find(..);  fn find_map(..);  fn position(..);
  fn rposition(..);  fn max(..);  fn min(..);  fn max_by_key(..);  fn max_by(..);  fn min_by_key(..);
  fn min_by(..);  fn rev(..);  fn unzip(..);  fn copied(..);  fn cloned(..);  fn cycle(..);
  fn sum<S : Sum<Self::Item>>(..) -> S;  fn product<P : Product<Self::Item>>(..) -> P;
  fn cmp(..);  fn partial_cmp(..);  fn eq(..);  fn ne(..);  fn lt(..);  fn le(..);  fn gt(..);
  fn ge(..);  fn is_sorted(..);  fn is_sorted_by(..);  fn is_sorted_by_key(..);
}
trait IntoIterator { type Item; type IntoIter : Iterator; fn into_iter(self : Self) -> Self::IntoIter; }
trait FromIterator<A> { fn from_iter<I : IntoIterator>(iter : I) -> Self; }
trait DoubleEndedIterator : Iterator { fn next_back(..); fn rfold(..); fn rfind(..); .. }
trait ExactSizeIterator : Iterator { fn len(self : &Self) -> i64 { .. } }
trait Extend<A>, Sum<A>, Product<A>, FusedIterator
```

Sources: `empty`, `once`, `once_with`, `repeat`, `repeat_n`, `repeat_with`, `from_fn`,
`successors`, `zip`. The adaptor structs, each with its `Iterator` impl conditional on the inner
iterator's (A1) and several holding an `Option<Item>` (A13: `Peekable`, `Fuse`, `Chain`, `Cycle`).

`collect` needs a target: `FromIterator` is implemented for `Vec<T>` in `std` (later `alloc`), and
`core` provides it for `Option<C>`, `Result<C, E>`, and nothing else, which is Rust's split too.

`for x in it` (A9) desugars to `let mut it = IntoIterator::into_iter(x); loop { match it.next() ..`
and `a..b` becomes a `core::ops::Range<T>` that implements `Iterator`. The compiler's existing
range loop stays as the fast path for a literal range so `for i in 0..n` keeps lowering to the same
`scf`/`cf` it does now; a differential test proves the two agree.

#### `core::slice`

`Slice<T> { ptr : *const T, len : i64 }` and `SliceMut<T> { ptr : *mut T, len : i64 }` as library
types until A17 gives them the `&[T]` spelling. Constructors are `unsafe fn from_raw_parts`.
Surface: `len`, `is_empty`, `first`, `last`, `get`, `get_mut`, `get_unchecked`, `split_at`,
`split_first`, `split_last`, `iter`, `iter_mut`, `windows`, `chunks`, `chunks_exact`, `rchunks`,
`split`, `contains`, `starts_with`, `ends_with`, `binary_search`, `binary_search_by`,
`binary_search_by_key`, `sort_unstable`, `sort_unstable_by`, `sort_unstable_by_key`,
`select_nth_unstable`, `reverse`, `swap`, `fill`, `fill_with`, `copy_from_slice`,
`clone_from_slice`, `copy_within`, `rotate_left`, `rotate_right`, `iter().rev()`, `concat`/`join`
(alloc), `to_vec` (alloc), `partition_point`, `is_sorted`. `sort` (stable) is in `alloc` in Rust
because it allocates; same here.

`sort_unstable` is a real algorithm (pattern-defeating quicksort in Rust; heapsort is acceptable
for parity with a matrix note, ipnsort later). It is also the first serious pure-Vx program in
`core` and is the phase-2 dogfood for the flat code generator.

`Vec::as_slice`/`as_mut_slice` in `std` change to return these types in the same PR.

#### `core::char`, `core::str`, `core::ascii`

`Char { code : u32 }` (Copy) with `from_u32`, `from_u32_unchecked`, `from_digit`, `to_digit`,
`is_alphabetic`, `is_numeric`, `is_alphanumeric`, `is_whitespace`, `is_control`, `is_ascii_*`,
`to_ascii_uppercase`/`lowercase`, `eq_ignore_ascii_case`, `len_utf8`, `encode_utf8`,
`is_uppercase`/`is_lowercase` and `to_uppercase`/`to_lowercase` restricted to ASCII plus
Latin-1 in phase 2 (full Unicode tables are a phase-4 item; the matrix says "ASCII + Latin-1").

`Str { ptr : *const u8, len : i64 }` with the UTF-8 invariant; `from_utf8 -> Result<Str, Utf8Error>`,
`from_utf8_unchecked`, `len`, `is_empty`, `as_bytes`, `chars`, `char_indices`, `bytes`, `lines`,
`split`, `splitn`, `rsplit`, `split_whitespace`, `trim`, `trim_start`, `trim_end`,
`trim_matches`, `starts_with`, `ends_with`, `contains`, `find`, `rfind`, `strip_prefix`,
`strip_suffix`, `parse<F : FromStr>`, `eq_ignore_ascii_case`, `is_char_boundary`, `get`,
`split_at`, `repeat`/`to_uppercase`/`to_owned` (alloc). `FromStr` for every integer width and for
`bool`; `f32`/`f64` parsing is phase 3 (correctly rounded decimal parsing is its own project; the
matrix says so).

Until A15, a string literal reaches `core` as `*const i8` and `Str::from_c_str` scans for the NUL.
The `s[0] == 104` failure in §9 M4 has to be fixed in the AST code generator before this module
can be tested; it is filed as part of A15.

`ascii::Char`, `escape_default`, and the `AsciiExt`-style methods live on `u8` and `Char`.

#### `core::fmt`

Rust's `fmt` is built on `dyn Write` and `Arguments`. Vx has neither `dyn` nor variadics, so the
design differs and the matrix records the difference explicitly:

<!-- vx-doctest: skip -- proposed API, not yet compilable -->

```rust
trait Write { fn write_str(self : &mut Self, s : Str) -> Result<void, Error>; fn write_char(..) { .. } }
struct Formatter<W : Write> { out : W, width : Option<i64>, precision : Option<i64>, fill : Char, align : Alignment, flags : u32, }
trait Display  { fn fmt<W : Write>(self : &Self, f : &mut Formatter<W>) -> Result<void, Error>; }
trait Debug    { fn fmt<W : Write>(self : &Self, f : &mut Formatter<W>) -> Result<void, Error>; }
trait LowerHex, UpperHex, Binary, Octal, LowerExp, UpperExp, Pointer  // same shape
impl<W : Write> Formatter<W> { fn pad, pad_integral, write_str, write_fmt?, debug_struct, debug_tuple, debug_list, debug_map, debug_set, alternate, width, precision, fill, sign_plus, .. }
```

Generic over the writer instead of dynamic; `print!`/`println!` in `std` become a `Formatter` over
a stdout `Write`. Integer formatting (all radixes, padding, sign) and float formatting (Grisu or
Ryu for shortest round-trip; `{:.N}` fixed) are pure Vx and the second and third serious programs
in `core`. The compiler's `print(x)` and `println!` keep working unchanged until `std` switches
them over; the switch is what closes Vx#323's "String printing" item.

Exclusions: `Arguments`, `format_args!`, `write!` as a macro (needs a variadic or item macro that
can see a format string: phase 4, once `macro_rules` can produce statements from a literal).

#### `core::hash`

`trait Hasher { fn finish(self : &Self) -> u64; fn write(self : &mut Self, bytes : Slice<u8>); fn write_u8 .. write_i128, write_usize? (excluded), write_length_prefix }`,
`trait Hash { fn hash<H : Hasher>(self : &Self, state : &mut H); fn hash_slice .. }`,
`trait BuildHasher { type Hasher; fn build_hasher(..); fn hash_one(..) }`, `BuildHasherDefault<H>`.
Impls for scalars, `bool`, `Char`, `Str`, `Slice<T : Hash>`, `Option<T>`, `Result<T, E>`,
`Ordering`, the `Range*` types. A concrete `SipHasher13` (Rust's default) and an `FxHasher`, both
pure Vx over A5's bitwise operators. `HashMap` in `alloc` is written against these, and the
hand-monomorphized `vx_hash_map_*_i32_i32` externs are deleted when it lands.

**Phase 2 acceptance.** `core::iter`, `core::slice`, `core::str`, `core::char`, `core::hash` at
par (exclusions reviewed); `core::fmt` at par under its documented design difference; `core::ops`
at par once A11 dispatches operators. `for x in v.iter()` runs on both code generators.
`std::iter`, `std::string`'s three pointer-walking helpers, and `std::math`'s trait deleted or
moved. The llama example's tokenizer decode prints text (Vx#323 item 3).

### Phase 3: cells, atomics, time, the allocator interface

Needs A16–A18 where noted; mostly needs nothing new.

- **`core::cell`**: `Cell<T : Copy>` (`get`, `set`, `replace`, `take`, `swap`, `update`),
  `RefCell<T>` (`borrow`, `borrow_mut`, `try_borrow*`, returning `Ref`/`RefMut` guards, which
  are only sound with A19; until then `RefCell` ships with `borrow_mut` returning `&mut T` and a
  runtime flag that is *not* released, and the matrix marks it "needs Drop"), `UnsafeCell<T>`,
  `OnceCell<T>` (A13), `LazyCell<T>` (holds a `Closure0<T>`).
- **`core::sync::atomic`**: `AtomicBool`, `AtomicI8..I64`, `AtomicU8..U64`, `AtomicPtr<T>`, with
  `load`, `store`, `swap`, `compare_exchange`, `compare_exchange_weak`, `fetch_add/sub/and/or/xor/ max/min/nand`, `fetch_update`; `enum Ordering { Relaxed, Release, Acquire, AcqRel, SeqCst }`;
  `fence`, `compiler_fence`. Written over `mlir!` (`llvm.atomicrmw`, `llvm.cmpxchg`, `llvm.fence`,
  `llvm.load atomic`). This is the ROADMAP §1 "Atomics" item and is expressible today; a `Mutex`
  over it is `std`'s job. Tested on the host with two threads once `std` has a thread, and with a
  single-threaded sequence test until then.
- **`core::time`**: `Duration` (Copy) with `from_secs/millis/micros/nanos`, `as_*`, `subsec_*`,
  `checked_add/sub/mul/div`, `saturating_*`, `mul_f32/f64`, `from_secs_f32/f64`, `is_zero`,
  `ZERO`/`MAX`/`SECOND`/... as functions, the `ops` impls (A11), `Ord`, `Hash`, `Display` via
  `Debug`. `std::time::now()` returns one.
- **`core::alloc`**: `Layout` (`from_size_align`, `new<T>`, `array<T>(n)`, `size`, `align`,
  `pad_to_align`, `extend`, `repeat`), `LayoutError`, `trait Allocator { fn allocate(self : &Self, l : Layout) -> Result<*mut u8, AllocError>; fn deallocate(..); fn grow(..); fn shrink(..) }`,
  `GlobalAlloc` (excluded: no `#[global_allocator]`; `std` provides the one `Global` allocator over
  `vx_vec_alloc`/`grow`/`free`). `Vec<T>` in `alloc` is rewritten against this, which retires the
  byte math in `vx_vec_grow` back into Vx once A5 lets Vx check the overflow itself.
- **`core::ffi`**: `c_void` (an empty enum), `CStr` (a `Str`-like over NUL-terminated `*const i8`
  with `from_ptr`, `to_bytes`, `to_str`, `count_bytes`), `c_char`/`c_int`/... **excluded** until
  the language has type aliases (there is no `type` keyword).
- **`core::error`**: `trait Error : Debug + Display { fn source(..) -> Option<&dyn Error> }` becomes
  `fn description(self : &Self) -> Str` plus `fn source(self : &Self) -> Option<&Self>` (no `dyn`;
  documented difference), implemented for `Utf8Error`, `ParseIntError`, `ParseFloatError`,
  `TryFromIntError`, `LayoutError`, `fmt::Error`, `cell::BorrowError`.
- **`core::array`** (A18): `[T; N]` does not exist as a type; the array literal is a `Tensor`. The
  matrix lists `core::array` as "excluded pending A18 and the Tensor-as-array decision" (§8.5).
- **`core::tuple`** is a Vx-only module holding `Pair`/`Triple` until A16, then deleted.
- **`core::net`**: `Ipv4Addr`, `Ipv6Addr`, `IpAddr`, `SocketAddr` with parsing and `Display`;
  pure data, straightforward, low priority.

**Phase 3 acceptance.** `core::cell` (under its Drop caveat), `core::sync::atomic`, `core::time`,
`core::alloc`, `core::error`, `core::net` at par. `alloc`'s plan is written against `core::alloc`
and `core::hash`.

### Phase 4: convergence

Not new modules; the state that makes `core` finished rather than added:

1. `std` is split into `stdlib/alloc/` (`Box`, `Vec`, `String`, `HashMap`, `HashSet`,
   `BTreeMap`, `VecDeque`, `Rc`) and `stdlib/std/` (I/O, fs, net, time's clocks, env, process,
   thread). The Rust-backed collection shims in `stdlib/rust_core/src/collections/` are deleted;
   `rust_core` shrinks to the OS surface and the print helpers.
1. The compiler's name-recognized `dot`/`sum`/`max`/`min` (Vx#223) become `Iterator::sum` and
   `Slice` methods, with the `vector.reduction` lowering reached through ordinary code generation.
1. `core` is precompiled to a `.vxlib` as part of the build (Vx#220/#221) so a user compile does
   not re-parse it. The purity rule makes `core` the easiest module set to precompile: no externs
   to declare, no link line.
1. The book's `stdlib-reference.md` generator (`scripts/tools/gen_stdlib_reference.py`) covers
   `stdlib/core`, and a `core` chapter explains the layer split and the purity rule.
1. `if comptime Topology::Current == ...` tests in a `spawn on` fixture prove `core::num` float
   math lowers on a non-host topology (`REQUIRES: macos` for the ANE, and the CPU
   `async.execute` path on Linux).

______________________________________________________________________

## 6. What happens to today's `std`

| Module | Fate | When |
| --- | --- | --- |
| `option.vx` | moves to `core::option`, extended | phase 1 |
| `result.vx` | deleted; `core::result` is a Vx enum; the `vx_result_*` externs and their Rust macro instantiation go | phase 1 |
| `closure.vx` | moves to `core::ops` with a `call` method | phase 1 |
| `iter.vx` | **deleted.** `core::iter` replaces it, and `std::vec`'s two iterators implement its trait. `tests/modules/iter.vx`, a third copy four fixtures import, is still there: none of them imports `core::iter`, so it does not collide yet | done; `tests/modules` pending |
| `math.vx` | **deleted.** The methods are `core::num`'s, over `math` dialect ops, and the 39 callers import `core::num` instead. The two `ffi_math` fixtures declare the libm externs themselves, since a scalar float across the C ABI is what they exist to test. | done |
| `string.vx` | `string_length`, `string_compare`, `parse_int` deleted in favour of `core::str`; `String` itself moves to `alloc` in phase 4 | phase 2, 4 |
| `vec.vx` | `as_slice`/`as_mut_slice` return `Slice`/`SliceMut`; `iter()` implements `core::iter::Iterator`; `VecIter`/`VecMap` deleted in favour of the generic adaptors; moves to `alloc` | phase 2, 4 |
| `hash_map.vx`, `hash_set.vx` | deleted; rewritten in Vx in `alloc` over `core::hash` | phase 4 |
| `googletest.vx` | stays; gains `expect_eq` over `PartialEq + Debug` so a test can compare any `core` type | phase 1 |
| `tensor.vx`, `simd.vx` | unchanged by this plan | |
| `io`, `fs`, `net`, `time`, `mmap`, `libc`, `alloc`, `llama` | unchanged until phase 4; `time::now` returns a `Duration` in phase 3 | |

The `graph` library is the regression canary: it must keep compiling and passing with **no source
change** through phases 1 and 2, because everything it uses (`Vec`, `googletest`, integer
arithmetic) is either untouched or reached transitively.

______________________________________________________________________

## 7. Testing and gates

Coding.md's rule applies to every one of these: a test is not evidence until it has been watched
to fail.

1. **Standalone check gate.** `every_stdlib_module_checks_on_its_own` in
   `tests/integration_test/shipped_programs_compile.rs` walks `stdlib/std`. Extend it to
   `stdlib/core`, with the same two-directional `KNOWN_BROKEN` list. A `core` module is never
   added to that list; if it does not check on its own it does not merge.
1. **Purity gate.** A new integration test reads every `.vx` under `stdlib/core/` and fails if any
   contains an `extern` block or an `import std::` line. It is a text scan on purpose: the rule is
   about the source, and a scan is the thing a contributor can run in their head.
1. **Per-module runnable tests.** `stdlib/core/tests/<module>.vx`, each a `main` that exercises
   the module through `std::googletest` and exits 0. A `tests/backend/pass/core_<module>.vx`
   fixture with `// RUN: vxc %s --run` and `// EXPECT:` lines drives each through the existing lit
   harness, so `cargo test` runs them and CI sees them. Cases are ported from Rust's `coretests`
   where the semantics match, and the file says which ones.
1. **Both code generators.** Every `core` fixture runs on the flat path by default and once more
   with `--legacy-codegen`; a decline on the flat path is a test failure for `core`, not a fallback,
   because `core` is the convergence corpus (Vx#197). This is stricter than the rest of the tree
   and is the point.
1. **Differential tests** in `tests/integration_test/flat_codegen_differential.rs` for the two
   places `core` replaces a compiler special case: `for i in 0..n` as a built-in loop versus as
   `Range::next`, and the slice reductions as builtins versus as `Iterator::sum`.
1. **The parity matrix** (Appendix A) is updated in the PR that changes a module's status, and
   the PR description names the Rust module it was checked against.
1. **Doc examples.** `scripts/tools/check_doc_examples.py` compiles every `rust`/`vx` fence in
   tracked markdown. `core`'s book chapter and this document keep proposed-syntax blocks marked
   `vx-doctest: skip` until the syntax lands, then unmark them, so the docs are tested too.

______________________________________________________________________

## 8. Decisions this plan needs from a human

1. **Associated types (A10) before or after phase 1?** Encoding `Item` as a trait parameter works
   today and is what `std::iter` does; it leaks into every user bound as `I : Iterator<I, T>`.
   Recommendation: phase 1 uses the encoding so `cmp`/`option`/`result` are not blocked on an L
   item; A10 lands at the start of phase 2, and `iter` is written against it from the first line.

1. **`Copy` semantics (A6). Decided: Rust's rule.** A type is `Copy` if it is declared so and all
   its fields are, enforced in the linear checker, with `Copy` a marker trait in `core::marker`.
   The alternative that was rejected is structural auto-Copy for all-scalar aggregates: it makes
   `Ordering` and `Option<i32>` copyable with no declaration, and it means adding a pointer field
   to a struct silently changes its move semantics.

   Being opt-in is what keeps it away from placement. A tensor, a `Pinned<T, Topology>` and a
   device buffer are linear because moving them is the whole discipline, and none of them can
   declare itself `Copy` -- so they stay linear without a special case. A user struct holding one
   cannot be `Copy` either, because the field is not. Duplicating placed data stays an explicit
   `transfer`, and a copy-like trait of one's own can call `transfer` in its body.

   Vx today: scalars copy and every struct and enum moves, though §9 D1 shows the checker does not
   flag a moved enum of scalars, so the rule is not evenly enforced now either.

1. **The index/length type.** Rust uses `usize`. Vx has none; `Vec::len` returns `i32`, `Slice`
   above uses `i64`, `sizeof` returns `i64`. Recommendation: `i64` everywhere in `core`, since it
   is what `memref.dim` and `sizeof` produce and it removes a class of overflow; `Vec` changes to
   match in phase 4. Adding `usize` as an alias for the target pointer width is a later language
   item.

1. **Panic on a device.** `assert`/`cf.assert` traps. On a GPU or the ANE there is no message
   channel back. Recommendation: `core` panics are `cf.assert` with the message, the host prints
   it, a device traps silently, and `panic::set_hook` is `std`-only. Document it; do not design
   around it now.

1. **Is `Tensor` Vx's array?** `[1, 2, 3]` is a `Tensor<i32, [3]>`. If yes, `core::array` is
   excluded permanently and `Tensor` gets `Index`, `IntoIterator`, `Default` impls in `std`. If no,
   A18 plus a real `[T; N]` type is an L language item. Recommendation: yes for now; revisit when
   A17 slices exist and the cost of a second array-like type is visible.

1. **Two layers or three?** Rust's `alloc` exists so that `no_std` programs with a heap can have
   `Vec`. Vx's device regions have tensor allocation but no heap, so the same split is real here.
   Recommendation: three, with `alloc` created in phase 4, not before.

1. **Where does this document live, and does ROADMAP §8 point at it?** `agents/Proposals.md` says
   `docs/implementation_plans/`, which is where it is. A one-line link from ROADMAP §8 would help a
   reader find it.

______________________________________________________________________

## 9. Probe transcript

Fourteen probes against `target/debug/vxc` at `e58697ed` on 2026-09-13, x86-64 Linux, run with
`config.local` sourced. Each was a single file in a scratch directory; the constructs are the ones
`core` is made of. Recorded so the Track A table can be re-verified when the compiler changes.

| Id | Construct | Result |
| --- | --- | --- |
| A1 | `main` imports `mylib::mid`, which imports `mylib::sub::base`; `main` calls `make_foo()` from `base` without importing it | Runs, returns 14. **Imports are transitive.** |
| A2 | `import mylib::sub::base;` with the file at `mylib/sub/base.vx` under a search root | Resolves. **Nested directories work with no loader change.** |
| C | `struct Marker<T> { }` and `let m = Marker<i32> { };` | Compiles and runs. Empty structs are fine (`PhantomData`). |
| D1 | `let a = Opt<i32>::Some(1); let b = a; match a { .. }` | Accepted. An enum of scalars is not treated as consumed on move, despite `is_linear` including `Enum`. |
| D2 | `impl<T : Eq2> Eq2 for Wrap<T> { fn eq2(..) { return self.v.eq2(other.v); } }` | `Method 'eq2' not found on type T`; `E3002 Expected bool, got T`. **Impl-level bounds are not usable inside the impl.** (A1) |
| D3 | Same impl, instantiated at `Wrap<f32>` with no `Eq2 for f32` | Errors, but with the same "not found on type T" message rather than an unmet-bound diagnostic. |
| D4 | `fn eq_wrap<T : Eq2>(a : Wrap<T>, b : Wrap<T>) -> bool { return a.v.eq2(b.v); }` | Runs, returns 1. **Function-level bounds work, including through a field.** |
| D5 | D2 with the field copied to a local first | Same failure as D2. It is the bound, not the field access. |
| D6 | `fn pick<T : Less>(a : T, b : T) -> T { if a.less(b) { return a; } return b; }` | Runs, returns 4. |
| E1 | `a % b` | `Unexpected token Unknown('%')`. (A5) |
| E2 | `(a & b) \| (a ^ b) << 1` | `Unexpected token Ampersand`. (A5) |
| G | `macro_rules impl_zero { ($t : ty) => { impl Zero for $t { .. } }; }  impl_zero!(i32);` | Parse error at the invocation. **Macros are expression-position only.** (A7) |
| H1 | `mlir!( inputs: (%a = x: f32), returns: f32, dialects: ["math"] ) { %r = math.exp %a : f32  macro.yield %r : f32 }` in a scalar function | Runs, prints 1. **Scalar `mlir!` over the `math` dialect works.** This is the mechanism for `core::num` floats. |
| H2 | `arith.addui_extended %x, %y : i32, i1` in `mlir!`, yielding the overflow bit | Runs (2147483647 + 1 unsigned correctly reports no overflow). The `checked_*` family is expressible. |
| I | `for x in r { sum += x; }` where `r` has `fn next(self : &mut Range2) -> Opt` | `E4001 Use of moved linear variable: r`; `Method 'next' not found on type ?`; `cannot assign i64 to i32`. (A9) |
| K | `enum Ordering { Less, Equal, Greater }` returned from a function, copied, matched | Runs, returns 1. |
| M1 | `impl<T> Opt<T> { fn map<U>(self : Opt<T>, f : Closure1<T, U>) -> Opt<U> { .. Opt<U>::Some(fp(f.env, v)) .. } }` called with a closure literal | Type-checks; both code generators fail with `LLVM Translation failed: builtin.unrealized_conversion_cast` on constructing `Opt<U>`. (A14) |
| M2 | `trait Folder<T> { fn fold<B>(self : Self, init : B, f : Closure2<B, T, B>) -> B; }` | `Unexpected token LeftAngle. Expected '('`. **No generic methods in traits.** (A3) |
| M3 | `struct Peek<T> { cur : i32, peeked : Opt<T> }`, assign the field, match on it | Flat path declines ("a struct with no GID"); AST path fails MLIR verification (`insertvalue` type mismatch). (A13) |
| M4 | `let s = "hi"; unsafe { if s[0] == 104 { .. } }` with `u8` arithmetic and casts | AST code generator: `'llvm.load' op result #0 must be LLVM type with size, but got 'none'`. Byte indexing of a literal in a comparison does not lower. (A15) |
| M5 | `fn both<T : A + B>(x : T)` | `Unexpected token Plus`. (A4) |

______________________________________________________________________

## 10. Issues to file

One per Track A item, so a PR can carry `Issue: #NNN` in its footer. Suggested titles, in landing
order:

1. Trait bounds on `impl` generics are parsed and then ignored inside the impl (A1)
1. Traits: default method bodies (A2)
1. Traits: generic methods (A3)
1. Multiple trait bounds `T : A + B` and trait `where` clauses (A4)
1. Integer operators `% & | ^ << >>` and the remaining compound assignments (A5)
1. `Copy` as a marker trait the move checker honours (A6; decision §8.2)
1. `macro_rules` in item position (A7)
1. A generic enum constructed at a method-level type parameter fails to lower (A14)
1. `panic`/`unreachable`/`todo` intrinsics with a message (A8)
1. `for x in iter` over the `next()` protocol (A9)
1. Associated types (A10)
1. Operator dispatch to `core::ops` traits (A11)
1. A generic struct with an enum field fails to lower on both code generators (A13)
1. `char`/`str`: string literal as `(ptr, len)`; byte indexing of a literal miscompiles (A15)
1. Implicit `core::prelude` import (A12)

And for Track B, one tracking issue per phase under Vx#451, each listing its modules as checkboxes
and linking the matrix. Phase 1 is Vx#720 and phase 2 is Vx#721; phases 3 and 4 are far enough out
that a list of them now would be fiction, and are described in Vx#451 instead. The Track A items
above are all filed and are tracked together in Vx#536.

______________________________________________________________________

## Appendix A: parity matrix

Status values: **par** (every stable item present or in the reviewed exclusions), **partial**
(some items; the module's test file lists what is missing), **declared** (traits/types exist,
semantics or dispatch pending a Track A item), **excluded** (with reason), **—** (not started).
Rust module names are the stable ones as of the current release; the reviewer checks against the
live docs, not this table.

| Rust `core` module | Vx module | Phase | Status | Blocking Track A items | Notes / exclusions |
| --- | --- | --- | --- | --- | --- |
| `marker` | `core::marker` | 1 | partial | Vx#715 | `Copy` declared and stamped for the scalars, `PhantomData<T>`; `Send`/`Sync`/`Sized` declared, not enforced. `Copy` is enforced for a struct and a payload-free enum; a generic enum is not treated as linear at all, so `Option`'s and `Result`'s impls are written and unenforced |
| `cmp` | `core::cmp` | 1 | partial | Vx#712 for `Reverse`, Vx#223 for free `max`/`min` | `PartialEq` and `Ord` over every integer width and `bool`; `PartialOrd` over those and the floats, which cannot be `Ord`; `Ordering` with `then_with`; `max_by`/`min_by`. No `Reverse` (its `Ord` impl declines on the flat path), no free `max`/`min` (the names are the compiler's tensor reductions), no `max_by_key`/`min_by_key`, no `Eq`. `Rhs` is `Self` by convention until trait-parameter defaults exist |
| `ops` | `core::ops` | 1→2 | declared | A11 (dispatch), A10 (`Output`) | `Deref`, `Drop`, `Fn*`, coroutine traits excluded |
| `clone` | `core::clone` | 1 | par | | `Clone` for the scalars, `bool`, `Ordering`, `Option<T : Clone>` and `Result<T : Clone, E : Clone>`; `clone_from` as the trait's one default. An impl over two bounded parameters turned out to be a shape the checker takes. `Result::clone` does not run at an instantiation mixing a float with an integer, which is `Result<f32, i32>` failing to lower rather than the impl |
| `default` | `core::default` | 1 | par | | `Default` for the scalars, `bool`, `Option<T>`; a static trait method dispatches since Vx#684 |
| `convert` | `core::convert` | 1 | partial | Vx#570 for `TryFrom` | `From` stamped per pair, the reflexive case among them, since a blanket impl over `Self` is never found; `Into`/`TryInto` excluded (a call carries no argument to choose an impl by, and Vx does not resolve from the expected type); `AsRef`/`AsMut` pending `str` and slices |
| `option` | `core::option` | 1 | partial | A16 for `zip`, Vx#526 for `expect`, Vx#711 for `inspect` | the combinators plus `is_none_or`, `or_else`, `map_or_else`, `take`, `replace`; `ok_or`/`ok_or_else` live in `core::result` to avoid an import cycle; no `zip`, `flatten`, `unwrap_or_default` (`T::default()` on a bounded parameter is not resolved), or the reference-returning methods |
| `result` | `core::result` | 1 | partial | Vx#526 for `expect`, Vx#711 for `inspect` | replaced the Rust-backed shims; `ok`, `err`, `map`, `map_err`, `and_then`, `unwrap_or_else`, `unwrap_err`, `is_ok_and`, `is_err_and`, `and`, `or`, `map_or`, `map_or_else`; no `or_else` (answers with a `Result` built from a closure, the shape the flat path declines hardest), `transpose`, `flatten`, or the reference-returning methods |
| `num` (integers) | `core::num` | 1 | partial | A16 for `overflowing_*` | every width, signed and unsigned; bit ops, rotates, `pow`, `ilog2`, `next_power_of_two`, `leading_ones`/`trailing_ones`, the checked, saturating and wrapping families; no `overflowing_*` (wants a pair type), `from_str_radix` (wants `str`), `to_be`/`to_le` (want the target's byte order), `isqrt`, `midpoint`, `div_ceil`; constants as functions until `const` items |
| `num` (floats) | `core::num` | 2 | partial | | the superset this table promised, over `f32` and `f64`: `sqrt`, `abs`, `exp`, `exp2`, `exp_m1`, `ln`, `log2`, `log10`, `ln_1p`, the six trigonometric and three hyperbolic functions, `floor`, `ceil`, `round`, `trunc`, `fract`, `powf`, `atan2`, `copysign`, `recip`, `to_degrees`, `to_radians`, `is_nan`, `is_finite`, `is_infinite`, `signum` -- each one `math` dialect op, no libm named in the source. `cbrt` and `erf` excluded: their ops lower to a libm declaration the pipeline rejects, and Rust's `core` has neither. No `max`/`min` (their NaN rule differs from `Ord`'s), `mul_add`, `hypot`, `rem_euclid`, `powi`, or the constants. Parsing in phase 3. `std::math` is deleted and its callers moved across |
| `mem` | `core::mem` | 1 | partial | A19 for semantics | `size_of`, `swap`, `replace`, `drop`, `forget`, `needs_drop`. No `align_of` (wants an intrinsic beside `sizeof`), no `take` (`T::default()` on a bounded parameter is not resolved), no `zeroed`/`transmute`/`ManuallyDrop`/`MaybeUninit`/`discriminant` |
| `ptr` | `core::ptr` | 1 | partial | Vx#714 | `null`, `null_mut`, `read`, `write`. No `eq`/`is_null`: two raw pointers cannot be compared and a pointer cannot be cast to an integer, so a null test cannot be spelled. No pointer arithmetic, `copy`, or the volatile forms. `addr_of` excluded |
| `hint` | `core::hint` | 1 | — | — | |
| `panic` | `core::panic` | 1 | — | A8 (Vx#526) | blocked outright, not merely reduced: `assert(false, ..)` is folded at check time, so no function that always panics compiles. `Location`, `PanicInfo`, hooks excluded |
| `iter` | `core::iter` | 2 | partial | A16 (`zip`, `enumerate`), Vx#649 | `Iterator` with `type Item` and nineteen defaults, including `fold`, `sum`, `product`, `collect`, `reduce` and `max`/`min` with their `_by` and `_by_key` forms; `FromIterator`, `Sum` and `Product`, the last two for every number type; `Range`, `map`/`filter`/`take`/`skip`, chained to any depth; no `rev`, `zip`, `enumerate` (tuples, Vx#533). `sum` and `product` answer `Self::Item` rather than Rust's caller-chosen `S : Sum`, and `FromIterator` is not a bound `collect` can state: a bound with type arguments does not parse. The adaptors name their inner item type as `I::Item`; `Map` and `Filter` take their closure as a type parameter. `std::vec`'s `VecIter`/`VecMap` implement this trait. The trait itself is `core::iter::traits`, which `std::vec` imports alone so the rest of `core::iter`'s names do not reach every program with a `Vec` (Vx#731) |
| `slice` | `core::slice` | 2 | — | A17 for `&[T]` spelling | library `Slice`/`SliceMut` first; `sort` (stable) is alloc |
| `str` | `core::str` | 2 | — | A15 | float `parse` in phase 3 |
| `char` | `core::char` | 2 | — | A15 | Unicode case tables phase 4; ASCII + Latin-1 first |
| `ascii` | `core::ascii` | 2 | — | — | |
| `fmt` | `core::fmt` | 2 | — | — | generic writer, not `dyn`; `Arguments`/`format_args!`/`write!` excluded until a format-string macro |
| `hash` | `core::hash` | 2 | — | A5 | SipHash-1-3 and Fx in Vx |
| `cell` | `core::cell` | 3 | — | A13, A19 for guards | `RefCell` guards need Drop |
| `sync::atomic` | `core::sync::atomic` | 3 | — | — | via `mlir!` `llvm.atomicrmw`/`cmpxchg`/`fence` |
| `time` | `core::time` | 3 | — | A11 | |
| `alloc` | `core::alloc` | 3 | — | — | `GlobalAlloc` excluded (no global allocator attribute) |
| `ffi` | `core::ffi` | 3 | — | — | `c_*` aliases excluded (no type aliases); `CStr` in |
| `error` | `core::error` | 3 | — | — | `source` returns `Option<&Self>`, no `dyn` |
| `net` | `core::net` | 3 | — | — | |
| `array` | — | — | excluded | A18, §8.5 | `Tensor` is the array type for now |
| `tuple` (Rust: primitive) | `core::tuple` | 1 | — | A16 | Vx-only `Pair`/`Triple`, deleted when tuples land |
| `borrow` | `core::borrow` | 3 | — | — | `Borrow`/`BorrowMut`/`ToOwned` (alloc) |
| `any` | — | — | excluded | | no runtime type identity |
| `future`, `task`, `pin` | — | — | excluded | | no `async` |
| `intrinsics` | (`mlir!`) | — | excluded | | `mlir!` is the intrinsic layer |
| `prelude` | `core::prelude` | 1 | — | A12 for implicitness | |
| `primitive`, `unicode`, `arch`, `simd` (unstable) | — | — | excluded | | `<N x T>` vectors are a language feature (Vx#482), not a library |
| `f32`/`f64` (consts) | `core::num` | 2 | — | — | `consts::PI` etc. as functions until `const` items |
| `i8`..`u128` (legacy modules) | — | — | excluded | | deprecated in Rust |
