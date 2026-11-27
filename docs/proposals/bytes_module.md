# Proposal: byte buffers and byte cursors (`core::bytes` and `std::bytes`)

- **Status:** draft
- **Issue:** #4161
- **Date:** 26 November 2026
- **Builds on:** the standard library RFC in PR #3514
  (`docs/proposals/standard_library_tenets_and_layout.md`). Tenet names such as A3 and C3 refer
  to that RFC.
- **Depends on:** slices (#534), a `Vec` with 64-bit lengths (#3271)

## Summary

Programs that read files, sockets or binary formats need four things:

1. Integers in a fixed byte order, read from and written to bytes.
1. A cursor that reads values from a byte buffer and moves forward.
1. A cursor that writes values into a byte buffer and grows it.
1. A shared buffer that many parts of a program can view without copying.

Vx has none of these today. Rust's standard library has the first one. Rust programs get the
other three from the `bytes` crate, and most of the Rust network and storage ecosystem agrees on
its types. This proposal puts the same functionality into the Vx standard library from the start.

The proposal has four parts:

| Part | Layer | Contents |
|---|---|---|
| `core::num` | core | `to_be_bytes`, `from_le_bytes` and the other conversions |
| `core::slice` | core | Byte functions on `Slice<u8>`: search, compare, copy |
| `core::bytes` | core | The `Buf` and `BufMut` traits, and the `Take`, `Chain` and `Limit` adapters |
| `std::bytes` | allocation | `Bytes` (shared, read-only) and `BytesMut` (unique, growable) |

The `Reader` and `Writer` adapters connect these types to `std::io`. They live in the OS layer.

## Terms

- **Byte order:** the order of the bytes of a number in memory. Big-endian (`be`) puts the most
  significant byte first. Little-endian (`le`) puts it last. Native (`ne`) is the order of the
  machine that runs the code.
- **Read cursor:** a value that holds a position in some bytes. Each read returns a value and
  moves the position forward. The trait for it is `Buf`.
- **Write cursor:** a value that writes bytes at a position and moves the position forward. The
  trait for it is `BufMut`.
- **Chunk:** the bytes that a cursor can give at once without copying. A cursor over one buffer
  has one chunk. A cursor over two joined buffers has two.
- **Shared buffer:** one allocation that several `Bytes` values view. A count of the views
  decides when to free the allocation.

## Motivation

### The first users

The RFC's "first user" rule asks for a real program before a module becomes stable. These kinds
of programs need this module:

- Vinyl, the Parquet reader. It reads little-endian values, and pages that point into one large
  file.
- Readers of database files. They read big-endian page headers and records from memory-mapped
  pages.
- Network clients and servers. They parse headers from socket buffers, and they pass parts of a
  buffer to other code without copying.

### What Vx has today

Checked on 26 November 2026 in `stdlib/`:

- `Vec<u8>` has `push`, `get`, `set` and an unsafe `extend_from_raw`. Its length is `i32`, so it
  holds at most 2 GiB.
- `Mapping` (from `File::map`) has `len`, `as_ptr` and a checked `byte(i)`.
- The integer types have `swap_bytes`. `f64` has `from_bits` and `to_bits`.
- `i64::parse` and the other parse functions take `(bytes, from, to)`, because there is no slice
  type.

So a program that reads a big-endian `u32` builds it from four calls to `byte(i)`, or it reads
through a raw pointer. Each program writes its own version of this code.

`docs/implementation_plans/file_io_usability.md` asks for "a bounded borrowed byte view or an owned
byte buffer". It does not design one. This proposal is that design.

### Why Rust needed a separate crate

Rust's standard library has `&[u8]`, `Vec<u8>` and `u32::from_be_bytes`. It lacks two things:

1. **Cursor traits.** `std::io::Read` reads into a buffer that the caller gives. It cannot
   return a `u32`, and every read can fail with an I/O error, even on memory. A parser wants
   `buf.get_u32()` on memory, with no I/O error.
1. **Cheap shared views.** A `Vec<u8>` has one owner. To give a part of it to another task, a
   program copies the part or uses `Arc<Vec<u8>>` with an offset and a length. `Bytes` does
   this in one type.

Tokio, Hyper, Tonic, Prost and many storage libraries take and return `bytes::Bytes`. That is
the shared vocabulary that tenet S1 describes. Two Vx libraries that pass byte buffers to each
other need the same type. So the type belongs in the standard library.

### The inclusion test

- **S1 (shared vocabulary):** yes. `Buf`, `BufMut` and `Bytes` sit between I/O code and parsers.
- **S4 (not a format):** the module knows no file format. Varints and other encodings stay in
  the libraries that use them (see "What this proposal leaves out").

## Design decisions

### B1. Every multi-byte method names its byte order

The `bytes` crate reads big-endian by default: `get_u32` is big-endian and `get_u32_le` is
little-endian. A reader who does not know the default can misread the code. Vx spells the order
every time: `get_u32_be`, `get_u32_le`, `get_u32_ne`. One-byte methods (`get_u8`, `get_i8`) have
no suffix.

This differs from the `bytes` crate. Tenet A1 allows the difference, and the module header
records it. It matches Rust's own `u32::from_be_bytes`, so the names agree inside Vx (A2).

### B2. A read has three forms

Tenets A3, A4 and C3 decide the forms.

| Form | Example | When the bytes run out |
|---|---|---|
| Checked | `try_get_u32_be` | Returns `Err(BufError)` |
| Panicking | `get_u32_be` | Panics, with a "Panics" line in its doc comment |
| Proven | name open (RFC Q20) | Cannot happen: `requires self.remaining() >= 4` |

Too few bytes in a file or a packet is an expected failure (A3). So a parser of outside data
uses the `try_` form. The panicking form is for code that already checked `remaining()`, and for
data the program made itself. A short input is then a programmer error (A4).

The proven form has no run-time check and needs no `unsafe`. It suits a hot loop in which the
caller checks the length of a whole page once. Then it reads many fields.

`BufError` holds the number of bytes the read needed and the number that remained, as
`bytes::TryGetError` does. Its final shape follows the RFC's answer to Q3 (errors).

### B3. Sizes and positions are 64-bit

Tenet A6 says host containers do not use `i32` for a length. Every length, position and count in
this module is `usize`, or `i64` if the RFC's Q11 decides so. A memory-mapped file of 10 GiB
then works as one `Bytes`.

### B4. Where the code runs

- `Buf`, `BufMut`, their adapters, and their impls for `Slice<u8>` and `SliceMut<u8>` are in the
  core layer. Each method is `on Topology::Any` (rule R1). A GPU kernel can then decode bytes, for
  example the pages of a Parquet file.
- If a `Slice` carries its memory space (RFC Q1), the impls are for `Slice<u8, M>` in any memory
  space `M`.
- `Bytes` and `BytesMut` hold a host allocation and a shared count. They are `HostOnly` (H3)
  until the RFC answers Q1 for allocators.

### B5. `Bytes` works without dynamic dispatch

The `bytes` crate stores a table of functions in each `Bytes` (a vtable). The table says how to
clone and free the memory under it. Vx has no trait objects (RFC Q4). So a `Bytes` holds a
pointer, a length and a storage value. The storage value is one of three kinds:

| Kind | Made by | Frees |
|---|---|---|
| Static | `Bytes::from_static` | Nothing. The bytes live as long as the program. |
| Shared | `Bytes::from(Vec<u8>)`, `BytesMut::freeze`, `clone` | The allocation, after its last view drops |
| Owner | `Bytes::from_owner(owner)` | The owner, after its last view drops |

The owner kind lets a file mapping become a `Bytes` without a copy. The `std::bytes` module
cannot import `std::fs`, because the allocation layer does not depend on the OS layer. So the
owner kind stores the owner with a function that drops it. That needs a function pointer or a
way to erase a type. If Vx cannot express that, it is a language gap under rule L2 (open
question B-Q2).

### B6. No hidden allocation and no hidden copy

Tenet A8 applies. Each method states in its doc comment if it allocates or copies:

- `Bytes::clone`, `slice`, `split_to` and `split_off` never allocate or copy.
- `Buf::copy_to_bytes` copies, except on `Bytes`, where it returns a view.
- `BytesMut::reserve` allocates only when it cannot reuse the space of views that dropped.
- `BytesMut::freeze` never copies.

### B7. Two import paths, until re-exports exist

The traits are in `core::bytes` and the buffers are in `std::bytes`. Vx has no re-export (RFC
Q7), so a program that uses both imports both. If Q7 adds re-exports, `std::bytes` re-exports the
traits.

## The API

The tables list each item with its `bytes` crate equivalent. "Same" means the same name and
meaning, apart from B1 and B3.

### `core::num` additions

The RFC already lists these in Section 3.1. This module depends on them.

| Item | Rust equivalent | Purpose |
|---|---|---|
| `T::to_be_bytes`, `to_le_bytes`, `to_ne_bytes` | Same, in Rust `core` | A number as a fixed-size byte array |
| `T::from_be_bytes`, `from_le_bytes`, `from_ne_bytes` | Same, in Rust `core` | A number from a fixed-size byte array |

`T` is each integer type from 8 to 128 bits, `f32` and `f64`.

### `core::slice` byte functions

The RFC lists `split_at`, `chunks`, `windows` and `copy_from_slice` for every `Slice<T>`. These
functions are for `Slice<u8>` only:

| Item | Purpose |
|---|---|
| `find_byte(b)`, `rfind_byte(b)` | The position of the first or last `b`. It uses SIMD, like C's `memchr`. |
| `find(needle)` | The position of the first copy of a byte sequence |
| `starts_with`, `ends_with` | Compare a prefix or a suffix |
| `eq_ignore_ascii_case` | Compare and treat ASCII upper case and lower case as equal |
| `ct_eq(other)` | Compare in constant time. RFC Section 7 keeps this one in `std`. |

### `core::bytes::Buf`, the read cursor

A type implements three required methods. The trait gives the others.

| Method | Rust `bytes` | Purpose |
|---|---|---|
| `remaining()` | Same | Bytes left to read |
| `chunk()` | Same | The next chunk, as a `Slice<u8>` |
| `advance(n)` | Same | Move forward `n` bytes. Panics when `n > remaining()`. |
| `try_advance(n)` | none | Move forward `n` bytes, or return `Err(BufError)` |
| `has_remaining()` | Same | `remaining() > 0` |
| `copy_to_slice(dst)` | Same | Fill `dst` and move forward |
| `get_u8()`, `get_i8()` | Same | One byte |
| `get_u16_be()` … `get_i128_le()` | `get_u16` … `get_i128_le` | Every integer width, in all three byte orders |
| `get_uint_be(n)`, `get_int_le(n)` … | Same, with B1 suffixes | An integer of `n` bytes, 1 to 8, for odd widths such as 3 or 6 |
| `get_f32_be()` … `get_f64_ne()` | `get_f32` … | Floats |
| `try_get_*` for each `get_*` | Same | The checked form of B2 |
| `chunks_vectored(dst)` | Same | Fill `dst` with several chunks, for a vectored write. It needs `std::io::IoSlice`, so the trait gives a version that fills core slices. |
| `copy_to_bytes(n)` | Same | The next `n` bytes as a `Bytes`. Allocates, except on `Bytes`. This method is in `std::bytes` as an extension, because `Bytes` is not in `core`. |
| `take(n)` | Same | A cursor that stops after `n` bytes |
| `chain(next)` | Same | A cursor that reads `self`, then `next` |

Impls: `Slice<u8>`, `Bytes`, `BytesMut`, `Take<B>`, `Chain<A, B>`, and `&mut B` for any
`B : Buf`. `VecDeque<u8>` gets an impl when `VecDeque` exists.

### `core::bytes::BufMut`, the write cursor

| Method | Rust `bytes` | Purpose |
|---|---|---|
| `remaining_mut()` | Same | Bytes that can still be written |
| `chunk_mut()` | Same | The next writable chunk. It may hold bytes that were never written. |
| `advance_mut(n)` | Same | `unsafe`. Mark `n` bytes as written. Safety: the caller wrote them. |
| `has_remaining_mut()` | Same | `remaining_mut() > 0` |
| `put(src : B)` where `B : Buf` | Same | Copy all of another cursor |
| `put_slice(src)` | Same | Copy a byte slice |
| `put_bytes(value, count)` | Same | Write `value` `count` times |
| `put_u8`, `put_i8`, `put_u16_be` … `put_f64_ne` | `put_u16` … | Every number type, in all three byte orders |
| `put_uint_be(value, n)` … | Same, with B1 suffixes | The low `n` bytes of an integer |
| `limit(n)` | Same | A write cursor that stops after `n` bytes |
| `chain_mut(next)` | Same | A write cursor that fills `self`, then `next` |

A `put_*` on a full buffer panics. A growable buffer, such as `Vec<u8>` or `BytesMut`, grows
instead and never panics. `try_put_*` forms return `Err(BufError)` on a full fixed buffer.

Impls: `SliceMut<u8>`, `Vec<u8>`, `BytesMut`, `Limit<B>`, `Chain<A, B>`, and `&mut B`.

`chunk_mut` returns uninitialized memory. It needs `MaybeUninit` (RFC Section 3.1, `core::mem`)
or an equivalent `UninitSlice` type.

### `std::bytes::Bytes`, a shared read-only buffer

| Method | Rust `bytes` | Purpose |
|---|---|---|
| `new()` | Same | An empty `Bytes`. Does not allocate. |
| `from_static(s)` | Same | A view of bytes that live for the whole program |
| `from_owner(owner)` | Same | A view of bytes that `owner` keeps alive, such as a file mapping (B5) |
| `copy_from_slice(s)` | Same | A new allocation with a copy of `s` |
| `len()`, `is_empty()` | Same | |
| `as_slice()` | `Deref` to `[u8]` | The bytes as a `Slice<u8>` |
| `slice(range)` | Same | A view of a part, with no copy |
| `slice_ref(sub)` | Same | A view of `sub`, which must point into `self`. Panics otherwise. |
| `split_to(at)`, `split_off(at)` | Same | Cut in two at `at`, with no copy |
| `truncate(len)`, `clear()` | Same | Make the view shorter |
| `is_unique()` | Same | True when no other view shares the allocation |
| `try_into_mut()` | Same | A `BytesMut` without a copy, when the view is unique |
| `clone()` | Same | A new view. Adds 1 to the shared count. Never copies. |

Traits: `Buf`, `Clone`, `PartialEq`, `Eq`, `Ord` and `Hash` by content, and comparison with
`Slice<u8>`, `Vec<u8>` and `str`. Conversions from `Vec<u8>`, `String`, `BytesMut` and a static
slice, and into `Vec<u8>` (without a copy when unique). Iteration over the bytes. Debug output
prints printable ASCII as text and other bytes as `\xNN`.

### `std::bytes::BytesMut`, a unique growable buffer

| Method | Rust `bytes` | Purpose |
|---|---|---|
| `new()`, `with_capacity(n)`, `zeroed(n)` | Same | |
| `len()`, `is_empty()`, `capacity()` | Same | |
| `reserve(n)` | Same | Room for `n` more bytes. Reuses freed space first. |
| `try_reclaim(n)` | Same | Reuse freed space for `n` bytes without allocating, or return false |
| `extend_from_slice(s)` | Same | Append a copy of `s` |
| `resize(len, value)`, `truncate(len)`, `clear()` | Same | |
| `set_len(len)` | Same | `unsafe`. Safety: the caller wrote the first `len` bytes. |
| `spare_capacity_mut()` | Same | The unwritten space after `len`, for a direct read from a file or socket |
| `split()`, `split_to(at)`, `split_off(at)` | Same | Cut off a part as its own `BytesMut`, with no copy |
| `unsplit(other)` | Same | Join two parts that came from one split. Copies only when they are not next to each other. |
| `freeze()` | Same | Turn into a `Bytes` with no copy |

Traits: `Buf`, `BufMut`, `PartialEq`, `Eq`, `Ord`, `Hash` and Debug output, as for `Bytes`.

The `split` and `freeze` pair is the main pattern for network code. A program reads from a
socket into the spare capacity, cuts off one complete message with `split_to`, freezes it and
sends it to another task. The next read reuses the same allocation when the old views dropped.

### `std::io` adapters

| Item | Rust `bytes` | Purpose |
|---|---|---|
| `buf.reader()` | `Buf::reader` | An `io::Read` over a `Buf` |
| `buf.writer()` | `BufMut::writer` | An `io::Write` over a `BufMut` |
| `Read::read_buf(dst : &mut B)` where `B : BufMut` | Tokio's `read_buf` | Read into the spare capacity of `dst`, and grow it |
| `File::map_bytes()` | none (`from_owner` with a map) | A file mapping as a `Bytes`, with no copy |

These live in the OS layer because they use `std::io`. The `Read` and `Write` traits come from
the RFC's Section 3.3.

## Coverage of the `bytes` crate

| `bytes` crate item | In this proposal |
|---|---|
| `Bytes`, `BytesMut` | Yes |
| `Buf`, `BufMut`, all `get_*` and `put_*` | Yes, with byte order suffixes (B1) |
| `try_get_*`, `TryGetError` | Yes, as `try_get_*` and `BufError` |
| `Take`, `Chain`, `Limit` | Yes |
| `Reader`, `Writer` | Yes, in `std::io` |
| `UninitSlice` | Yes, or `MaybeUninit` slices, after `core::mem` has them |
| `IntoIter` | Yes, as an `Iterator` impl |
| `Bytes::from_owner` | Yes, if B-Q2 finds a way to store the owner |
| `Buf for VecDeque<u8>` | Later, with `VecDeque` |
| `Buf for std::io::Cursor` | No. A `Slice<u8>` is already a cursor. |
| `serde` support | No. Vx has no serde. |
| A vtable for custom storage | No. Three fixed storage kinds (B5). |
| Big-endian as the default | No (B1) |

## What this proposal leaves out

- **Varints and other encodings.** LEB128 (Protobuf, WebAssembly, DWARF) and SQLite's varint are
  different formats. Under S4 they belong in the libraries for those formats. The `bytes` crate
  leaves them out too.
- **Text.** `Bytes` is not text. `str` (#532) is the UTF-8 view. A conversion from `Bytes` to
  `str` checks UTF-8 and copies nothing.
- **Buffer pools.** A pool of reusable buffers is an application choice, not shared vocabulary.
- **Compression and checksums.** CRC32 and similar functions are format code (S4).

## Life cycle and benchmarks

- **Stage:** incubating, with the feature name `bytes`.
- **Order of work:**
  1. Prototype a byte view and a read cursor in a real program, before slices exist. This tests
     the API early.
  1. Add the `core::num` byte conversions.
  1. After slices land (#534), add `core::bytes` with `Buf` and `BufMut` for slices.
  1. After the 64-bit `Vec` lands (#3271), add `std::bytes`.
  1. Add the `std::io` adapters with the `Read` and `Write` traits.
- **Benchmarks (P3), each against the Rust `bytes` crate:**
  - Read a million `u32_be` values from a `Slice<u8>`. The goal is one load and one byte swap for
    each value, with the length check moved out of the loop.
  - `Bytes::clone` and `split_to` in a loop.
  - Parse HTTP-style messages with `BytesMut` from a stream of reads of different sizes.
- **To become stable:** the RFC's list in Section 6, and at least one first user of each kind
  in the Motivation section.

## A sketch of use

The syntax is a sketch. It assumes slices.

```vx
import core::bytes;

struct MessageHeader {
  kind : u8,
  flags : u16,
  length : u32,
}

/// Reads the 8-byte header of a message: a kind, flags, one unused byte and a length.
fn read_message_header<B : Buf>(buf : &mut B) -> Result<MessageHeader, BufError> {
  let kind = buf.try_get_u8()?;
  let flags = buf.try_get_u16_be()?;
  buf.try_advance(1)?;
  let length = buf.try_get_u32_be()?;
  return Ok(MessageHeader { kind : kind, flags : flags, length : length });
}
```

## Open questions

- **B-Q1. The name of the proven form.** This follows the RFC's Q20. With `at` as the name, the
  proven read is `at_u32_be`. Is that clear, or does it need a different pattern?
- **B-Q2. Owners without trait objects.** How does `Bytes::from_owner` store an owner of any type
  and drop it later? A function pointer field is one way. A type-erasing language feature is
  another.
- **B-Q3. One module or two.** Should the traits live in `core::slice` instead of a new
  `core::bytes`, to keep the number of modules small?
- **B-Q4. `Bytes` on a device.** If `Slice` carries a memory space (RFC Q1), should `Bytes` carry
  one too? A shared count in GPU memory needs device atomics.
- **B-Q5. Growth failure.** `BytesMut::reserve` allocates. It follows the RFC's answer to Q14.
  Does it also need a `try_reserve` that returns an error?
- **B-Q6. Panicking reads.** Should `get_u32_be` (the panicking form) exist at all, or should a
  parser always choose between the checked and the proven form?
