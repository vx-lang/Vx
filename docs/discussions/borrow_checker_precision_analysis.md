# Borrow Checker Precision Analysis: NLL Problem Case #3

**Status:** analysis
**Question:** Does Vx's borrow checker accept the `get`-or-`insert` map pattern that Rust's NLL rejects and Polonius is designed to accept?
**Design of record:** [`borrow_checker_architecture.md`](borrow_checker_architecture.md) — this document measures that design, it does not replace it. Tracked in [#243](https://github.com/hiraditya/Vx/issues/243).

## 1. The question

The canonical shape, in Rust:

```rust
fn get_or_insert(map: &mut HashMap<K, V>, key: K) -> &V {
    match map.get(&key) {
        Some(v) => v,                       // the borrow escapes into the return
        None => {
            map.insert(key, V::default());  // ERROR under NLL: `map` is still
            map.get(&key).unwrap()          // considered immutably borrowed
        }
    }
}
```

Under NLL the loan created by `map.get()` must outlive the function, because the `Some(v) => v` arm returns it. NLL regions are sets of program points with no path discrimination, so the loan is treated as live at *every* point the borrow could reach — including the `None` arm, where it is in fact dead. This is **NLL Problem Case #3**, and it is the headline motivation for [Polonius](https://github.com/rust-lang/polonius), whose location-sensitive formulation asks "is loan L live at point P *on this path*?" and accepts the program.

## 2. Answer

**No.** Vx does not have Polonius-grade precision — the discriminating case is rejected exactly as NLL rejects it. Separately, the `get_or_insert` shape *does* compile in Vx, but **for the wrong reason**: reborrows through a reference parameter create no borrow record at all, so the same code path also accepts a program that is genuinely unsound.

Two soundness holes were found in the process. They matter more than the precision question, because "accepts more sound programs" is not currently distinguishable from "does not model this class of program."

## 3. Method

Nine programs compiled with the checked-in `target/debug/vxc`, each reduced to the smallest form that exercises one property. The `HashMap` in `stdlib/std/hash_map.vx` is an `extern` shim over `*mut i8`, so it is invisible to the borrow checker; a two-field `struct Map` with `probe(&Map) -> &i32` and `insert(&mut Map, i32)` reproduces the same aliasing structure in checkable code.

Full sources in Appendix A.

## 4. Results

| # | Shape | Sound? | Rust NLL | Polonius | **Vx** | |
|---|---|---|---|---|---|---|
| bc5 | borrow, last use, *then* mutate | ✅ | accept | accept | **accept** | ✓ |
| bc3 | borrow, mutate, then read | ❌ | reject | reject | **reject** `E4003` | ✓ |
| bc7 | mutate inside a branch, read after | ❌ | reject | reject | **reject** `E4003` | ✓ |
| **bc8** | **mutate in a branch that returns; read only on the other path** | ✅ | reject | **accept** | **reject** `E4003` | ✗ |
| bc2 | `get_or_insert`, across functions | ✅ | reject | accept | accept | (see §5.2) |
| bc6 | same, in `match` form | ✅ | reject | accept | accept, then ICE | (see §7) |
| bc9 | reborrow through `&mut` param, read after mutation | ❌ | reject | reject | **accept** | ✗ |
| bc4 | return a reference to a local | ❌ | reject | reject | **accept** | ✗ |

## 5. Analysis

### 5.1 Named locals: NLL-grade, and better than the docs claim

bc5 releases the loan at the last *use* of the borrowing binding, not at the end of its lexical scope:

```rust
let found = probe(&m);
let copied = *found;      // last use of `found`
insert(&mut m, 42);       // accepted
```

Pre-NLL lexical borrow checking rejects this. [`borrow_checker_architecture.md`](borrow_checker_architecture.md) describes exactly that lexical model — *"When an AST block scope ends, `TypeChecker` automatically iterates through `active_borrows` and pops any records where the `scope_depth` matches the exiting block"* — and [`src/borrow.rs`](../../src/borrow.rs) documents Region IDs as *"the depth of the lexical block it was instantiated in."*

The implementation is doing more than that. bc3 and bc5 differ only in whether the last read of `found` precedes or follows the mutation, and they get different verdicts. **The architecture doc is stale and understates the checker.**

bc7 further shows the conflict search descends into branches, so this is genuine use-based liveness over program order — which is, to a first approximation, NLL.

### 5.2 Reborrows through reference parameters are untracked

This is the hole that makes bc2 pass. `active_borrows` is keyed by local variable name ([`sema.rs`](../../src/borrow.rs), per the architecture doc), and a reborrow that never spells `&x` on a named local creates no record:

```rust
fn bad(m : &mut Map) -> &i32 {
  let found = probe(m);   // reborrow of *m — no borrow record created
  insert(m, 42);          // mutates through m while `found` aliases it
  return found;           // UNSOUND — accepted by Vx
}
```

bc9 is accepted. It is rejected by NLL *and* Polonius. So bc2's acceptance is not precision — the analysis is not looking at that class of borrow.

This is the single most important finding in this document. Any claim about Vx's borrow-checking precision is unmeasurable until it is fixed, because the pass/fail signal on the interesting cases is dominated by this hole.

### 5.3 Escapes through return values are untracked

```rust
fn dangle() -> &i32 {
  let x = 5;
  return &x;      // accepted by Vx
}
```

Rejected by every Rust borrow checker since 1.0. There is no escape analysis tying a returned reference's region to the function's inputs. In a language whose stated pitch includes memory safety, this is a soundness hole rather than a precision gap.

### 5.4 No location sensitivity

bc8 is the minimal Polonius case:

```rust
let found = probe(&m);
if m.present == 0 {
  insert(&mut m, 42);     // on this path, `found` is dead
  return 0;
}
return *found;            // `found` read only on the other path
```

Vx rejects with `E4003`, matching NLL. The loan is treated as live from creation to its last use *in program order*, with no discrimination between the path that reads it and the path that does not. That is precisely the imprecision Polonius exists to remove.

## 6. The representational constraint

This is a decision, not an oversight, and it should be made consciously before release.

[`src/borrow.rs`](../../src/borrow.rs) and [`borrow_checker_architecture.md`](borrow_checker_architecture.md) state the bet explicitly: Vx *"explicitly avoids building heavy, whole-program constraint graphs (like NLL or Polonius)"* in favour of a 12-bit Region ID equal to lexical scope depth, bitpacked into word 2 of the 256-bit `TypeId`, compared with `region_a <= region_b`.

That encoding **forecloses Polonius.** A region that is a scope depth is a single scalar; location sensitivity requires a region to be a *set of program points*, so that "live at P₁, dead at P₂" is expressible. The fast-path representation cannot hold that, by construction.

So the choice is:

- **Keep the bitpacked regions.** Constant-time subtyping checks, a real compile-speed argument, and NLL-grade precision at best. Polonius-grade acceptance is permanently off the table for the fast path.
- **Change the region representation.** Regions become point sets (or a datalog relation), the fast path becomes a pre-filter rather than the decision procedure, and Polonius-grade precision becomes reachable.

Both are defensible. What is not defensible is claiming the first and expecting the second.

## 7. Unrelated defect found

bc6 panics the compiler after passing semantic analysis:

```console
$ vxc bc6.vx --action emit-mlir
[flat-codegen] program outside the flat subset; using the AST path
Vx Compiler Internal Error: panicked at src/codegen/lower/mod.rs:314:14:
Unsupported pattern in codegen
```

The trigger is a `_ =>` wildcard arm in a `match` over an integer where every arm returns ([`src/codegen/lower/mod.rs:314`](../../src/codegen/lower/mod.rs#L314)). Not borrow-related; ordinary code, and an ICE rather than a diagnostic. Worth its own issue.

Incidentally, it also proves bc6 cleared the borrow checker — the panic is downstream of semantic analysis.

## 8. Recommendation

**Precision is the wrong thing to chase first.** bc2 and bc9 have the same structure and opposite soundness, and Vx accepts both. Until §5.2 and §5.3 are closed, "Vx accepts programs NLL rejects" is not a capability claim.

Suggested ordering:

1. **Track reborrows through reference parameters** (§5.2). Closes bc9. Highest value: it is the hole that makes every other measurement unreliable.
1. **Escape analysis on returned references** (§5.3). Closes bc4. Straightforward for the common case — a returned reference must derive from a parameter, not a local.
1. **Re-measure.** Re-run this matrix. bc2 and bc6 are expected to flip to *reject* once (1) lands, because Vx will then be NLL-grade and NLL rejects them. That is the correct outcome, not a regression.
1. **Then, and only then, decide about §6.** With a sound baseline, "should Vx accept `get_or_insert`?" becomes a real question with a measurable answer.

If (1) and (2) do not fit before release, state them as known limitations. That costs far less than a user discovering bc4 on their own.

**Test fixtures.** The nine cases in Appendix A are a natural regression suite. Four of them (bc2, bc4, bc6, bc9) are currently accepted but should eventually be rejected, so they belong in `tests/middle_end/fail/` only once the holes are closed — until then they document the gap.

## Appendix A — the nine cases

Common preamble for bc2–bc9:

```rust
struct Map { slot : i32, present : i32 }
fn probe(m : &Map) -> &i32 { return &m.slot; }
fn insert(m : &mut Map, v : i32) -> void { m.slot = v; m.present = 1; }
```

**bc1 — a borrow of a parameter escapes into the return.** Compiles; establishes that the shape is expressible.

```rust
fn probe(m : &Map) -> &i32 { return &m.slot; }
```

**bc2 — `get_or_insert`.** Accepted. Sound, but see §5.2.

```rust
fn get_or_insert(m : &mut Map) -> &i32 {
  let found = probe(m);
  if m.present == 1 { return found; }
  insert(m, 42);
  return probe(m);
}
```

**bc3 — straight-line conflict.** Rejected: `E4003`. Correct.

```rust
let found = probe(&m);
insert(&mut m, 42);
return *found;
```

**bc4 — dangling return.** Accepted. **Unsound** (§5.3).

```rust
fn dangle() -> &i32 { let x = 5; return &x; }
```

**bc5 — loan dead before the mutation.** Accepted. Correct, and the NLL-vs-lexical discriminator (§5.1).

```rust
let found = probe(&m);
let copied = *found;
insert(&mut m, 42);
return copied + m.slot;
```

**bc6 — `match` form of bc2.** Clears the borrow checker, then ICEs (§7).

```rust
let found = probe(&m);
match m.present {
  1 => { return *found; }
  _ => { insert(&mut m, 42); return m.slot; }
}
```

**bc7 — mutation inside a branch, read after.** Rejected: `E4003`. Correct.

```rust
let found = probe(&m);
if m.present == 1 { insert(&mut m, 42); }
return *found;
```

**bc8 — the Polonius discriminator.** Rejected: `E4003`. Sound program, conservatively refused (§5.4).

```rust
let found = probe(&m);
if m.present == 0 {
  insert(&mut m, 42);
  return 0;
}
return *found;
```

**bc9 — reborrow through a `&mut` parameter.** Accepted. **Unsound** (§5.2).

```rust
fn bad(m : &mut Map) -> &i32 {
  let found = probe(m);
  insert(m, 42);
  return found;
}
```

## Appendix B — evidence index

| Fact | Location |
|---|---|
| Borrow checker implementation | [`src/borrow.rs`](../../src/borrow.rs) |
| Two-half split; lexical model described | [`borrow_checker_architecture.md`](borrow_checker_architecture.md) |
| Region ID = lexical scope depth, 12 bits, bitpacked | [`src/borrow.rs`](../../src/borrow.rs) (header comment) |
| `active_borrows` keyed by variable name | [`borrow_checker_architecture.md`](borrow_checker_architecture.md) §1 |
| `E4003` conflicting-borrow diagnostic | [`tests/middle_end/fail/borrow_mut_imm.vx`](../../tests/middle_end/fail/borrow_mut_imm.vx) |
| Existing lexical-release test | [`tests/middle_end/pass/borrow_lexical.vx`](../../tests/middle_end/pass/borrow_lexical.vx) |
| `HashMap` is a raw-pointer `extern` shim | [`stdlib/std/hash_map.vx`](../../stdlib/std/hash_map.vx) |
| Codegen ICE on wildcard match arm | [`src/codegen/lower/mod.rs:314`](../../src/codegen/lower/mod.rs#L314) |
