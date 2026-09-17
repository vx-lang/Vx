# Standard library reference

Every public type and function in the 21 `std` modules, taken from their signatures.

Import a module with its path, then use the names it declares:

```rust
import std::vec;

fn main() -> i32 {
    let mut v = Vec<i32>::new();
    v.push(10);
    v.push(32);
    return v.get(0) + v.get(1);
}
```

The toolchain also ships a `graph` library outside `std`, imported as `graph::traversal` and
friends.

> This page is generated from `stdlib/core/*.vx` and `stdlib/std/*.vx` by
> `scripts/tools/gen_stdlib_reference.py`. Signatures are exactly what the source declares.

## Contents

- [`core::cmp`](#corecmp) —
- [`core::iter`](#coreiter) — The `Iterator` trait and its adaptors, which `for` loops and `.map` build on.
- [`core::num`](#corenum) — The integer methods, on `i32`.
- [`core::ops`](#coreops) —
- [`core::option`](#coreoption) — `Option<T>`, for a value that may be absent.
- [`core::result`](#coreresult) — `Result<T, E>`, for an operation that may fail.
- [`std::alloc`](#stdalloc) — Raw allocation and deallocation.
- [`std::box`](#stdbox) — `Box<T>`, a single-owner heap allocation. Required for recursive types.
- [`std::fs`](#stdfs) — Files and directories.
- [`std::googletest`](#stdgoogletest) — Assertions for tests written in Vx.
- [`std::hash_map`](#stdhash_map) — `HashMap<K, V>`.
- [`std::hash_set`](#stdhash_set) — `HashSet<T>`.
- [`std::io`](#stdio) — Standard input, output and error.
- [`std::iter`](#stditer) — The `Iterator` trait and its adaptors, which `for` loops and `.map` build on.
- [`std::libc`](#stdlibc) — Direct bindings to the C library.
- [`std::llama`](#stdllama) — Helpers used by the Llama 2 example.
- [`std::math`](#stdmath) — Mathematical functions and constants.
- [`std::mmap`](#stdmmap) — Memory-mapped files.
- [`std::net`](#stdnet) — TCP and UDP sockets.
- [`std::simd`](#stdsimd) — SIMD vector types and operations.
- [`std::string`](#stdstring) — `String` and text manipulation.
- [`std::tensor`](#stdtensor) — Operations on `Tensor`, including shape queries and elementwise maths.
- [`std::time`](#stdtime) — Clocks and durations.
- [`std::vec`](#stdvec) — `Vec<T>`, a growable array.

## `core::cmp`

**Types**

- `enum Ordering`
- `trait PartialEq`
- `trait Ord`

**`Ordering` methods**

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
fn is_lt(self : Ordering) -> bool
fn is_gt(self : Ordering) -> bool
fn is_eq(self : Ordering) -> bool
fn is_ne(self : Ordering) -> bool
fn is_le(self : Ordering) -> bool
fn is_ge(self : Ordering) -> bool
fn reverse(self : Ordering) -> Ordering
fn then(self : Ordering, other : Ordering) -> Ordering
fn eq(self : &Self, other : &Self) -> bool
fn ne(self : &Self, other : &Self) -> bool
fn cmp(self : &Self, other : &Self) -> Ordering
fn lt(self : &Self, other : &Self) -> bool
fn le(self : &Self, other : &Self) -> bool
fn gt(self : &Self, other : &Self) -> bool
fn ge(self : &Self, other : &Self) -> bool
fn max(self : Self, other : Self) -> Self
fn min(self : Self, other : Self) -> Self
fn clamp(self : Self, lo : Self, hi : Self) -> Self
```

**`PartialEq for $t` methods**

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
fn eq(self : &$t, other : &$t) -> bool
```

**`Ord for $t` methods**

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
fn cmp(self : &$t, other : &$t) -> Ordering
```

## `core::iter`

The `Iterator` trait and its adaptors, which `for` loops and `.map` build on.

**Types**

- `trait Iterator<Item>`
- `struct Range`
- `struct Map<I, Item, U>`
- `struct Filter<I, Item>`
- `struct Take<I, Item>`
- `struct Skip<I, Item>`

**Functions**

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
fn next(self : &mut Self) -> Option<Item>
fn count(self : &mut Self) -> i64
fn last(self : &mut Self) -> Option<Item>
fn nth(self : &mut Self, n : i64) -> Option<Item>
fn any(self : &mut Self, f : Closure1<Item, bool>) -> bool
fn all(self : &mut Self, f : Closure1<Item, bool>) -> bool
fn find(self : &mut Self, f : Closure1<Item, bool>) -> Option<Item>
fn position(self : &mut Self, f : Closure1<Item, bool>) -> Option<i64>
fn for_each(self : &mut Self, f : Closure1<Item, i32>) -> i32
```

**`Iterator<i64> for Range` methods**

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
fn next(self : &mut Range) -> Option<i64>
```

**`Iterator<U> for Map<I, Item, U>` methods**

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
fn next(self : &mut Map<I, Item, U>) -> Option<U>
```

**`Iterator<Item> for Filter<I, Item>` methods**

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
fn next(self : &mut Filter<I, Item>) -> Option<Item>
```

**`Iterator<Item> for Take<I, Item>` methods**

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
fn next(self : &mut Take<I, Item>) -> Option<Item>
```

**`Iterator<Item> for Skip<I, Item>` methods**

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
fn next(self : &mut Skip<I, Item>) -> Option<Item>
fn map<I, Item, U>(inner : I, f : Closure1<Item, U>) -> Map<I, Item, U>
fn filter<I, Item>(inner : I, keep : Closure1<Item, bool>) -> Filter<I, Item>
fn take<I, Item>(inner : I, left : i64) -> Take<I, Item>
fn skip<I, Item>(inner : I, drop : i64) -> Skip<I, Item>
fn range(at : i64, end : i64) -> Range
```

## `core::num`

The integer methods, on `i32`.

**`$t` methods**

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
fn min_value(self : $t) -> $t
fn max_value(self : $t) -> $t
fn bits(self : $t) -> $t
fn count_ones(self : $t) -> $t
fn count_zeros(self : $t) -> $t
fn leading_zeros(self : $t) -> $t
fn trailing_zeros(self : $t) -> $t
fn is_power_of_two(self : $t) -> bool
fn is_positive(self : $t) -> bool
fn is_negative(self : $t) -> bool
fn abs(self : $t) -> $t
fn signum(self : $t) -> $t
fn abs_diff(self : $t, other : $t) -> $t
fn pow(self : $t, exp : $t) -> $t
fn rem_euclid(self : $t, rhs : $t) -> $t
fn div_euclid(self : $t, rhs : $t) -> $t
fn ilog2(self : $t) -> $t
fn next_power_of_two(self : $t) -> $t
fn rotate_left(self : $t, n : $t) -> $t
fn rotate_right(self : $t, n : $t) -> $t
fn swap_bytes(self : $t) -> $t
fn reverse_bits(self : $t) -> $t
fn checked_add(self : $t, rhs : $t) -> Option<$t>
fn checked_sub(self : $t, rhs : $t) -> Option<$t>
fn checked_mul(self : $t, rhs : $t) -> Option<$t>
fn checked_div(self : $t, rhs : $t) -> Option<$t>
fn checked_rem(self : $t, rhs : $t) -> Option<$t>
fn checked_neg(self : $t) -> Option<$t>
fn saturating_add(self : $t, rhs : $t) -> $t
fn saturating_sub(self : $t, rhs : $t) -> $t
```

## `core::ops`

**Types**

- `struct Closure0<Ret>`
- `struct Closure1<Arg, Ret>`
- `struct Closure2<Arg1, Arg2, Ret>`
- `struct Closure3<Arg1, Arg2, Arg3, Ret>`

## `core::option`

`Option<T>`, for a value that may be absent.

**Types**

- `enum Option<T>`

**`Option<T>` methods**

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
fn is_some(self : &Option<T>) -> Bool
fn is_none(self : &Option<T>) -> Bool
fn unwrap(self : Option<T>) -> T
fn unwrap_or(self : Option<T>, default : T) -> T
fn or(self : Option<T>, other : Option<T>) -> Option<T>
fn and(self : Option<T>, other : Option<T>) -> Option<T>
fn xor(self : Option<T>, other : Option<T>) -> Option<T>
fn map<U>(self : Option<T>, f : Closure1<T, U>) -> Option<U>
fn and_then<U>(self : Option<T>, f : Closure1<T, Option<U>>) -> Option<U>
fn filter(self : Option<T>, p : Closure1<T, bool>) -> Option<T>
fn map_or<U>(self : Option<T>, default : U, f : Closure1<T, U>) -> U
fn unwrap_or_else(self : Option<T>, f : Closure0<T>) -> T
fn is_some_and(self : Option<T>, p : Closure1<T, bool>) -> bool
```

## `core::result`

`Result<T, E>`, for an operation that may fail.

**Types**

- `enum Result<T, E>`

**`Result<T, E>` methods**

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
fn is_ok(self : &Result<T, E>) -> bool
fn is_err(self : &Result<T, E>) -> bool
fn unwrap(self : Result<T, E>) -> T
fn unwrap_or(self : Result<T, E>, default : T) -> T
fn ok(self : Result<T, E>) -> Option<T>
fn err(self : Result<T, E>) -> Option<E>
fn map<U>(self : Result<T, E>, f : Closure1<T, U>) -> Result<U, E>
fn map_err<F>(self : Result<T, E>, f : Closure1<E, F>) -> Result<T, F>
fn and_then<U>(self : Result<T, E>, f : Closure1<T, Result<U, E>>) -> Result<U, E>
fn unwrap_or_else(self : Result<T, E>, f : Closure1<E, T>) -> T
```

## `std::alloc`

Raw allocation and deallocation.

**Functions** *(bound directly to C)*

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
fn malloc(size : i64) -> *mut i8
fn realloc(ptr : *mut i8, size : i64) -> *mut i8
fn free(ptr : *mut i8) -> i32
```

## `std::box`

`Box<T>`, a single-owner heap allocation. Required for recursive types.

**Types**

- `struct Box<T>`

**`Box<T>` methods**

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
fn new(val : T) -> Box<T>
fn free(self : &mut Box<T>) -> i32
```

## `std::fs`

Files and directories.

**Types**

- `struct File`

**`File` methods**

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
unsafe fn open(path : *const i8, mode : i32) -> File
unsafe fn read(self : *mut File, buffer : *mut u8, len : i64) -> i64
unsafe fn write(self : *mut File, buffer : *const u8, len : i64) -> i64
fn seek(self : *mut File, offset : i64, whence : i32) -> i64
unsafe fn file_drop(file : *mut File) -> void
```

**C bindings** *(the native functions this module is built on)*

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
fn vx_file_open(c_path : *const i8, mode : i32) -> *mut i8
fn vx_file_read(ptr : *mut i8, buffer : *mut u8, len : i64) -> i64
fn vx_file_write(ptr : *mut i8, buffer : *const u8, len : i64) -> i64
fn vx_file_seek(ptr : *mut i8, offset : i64, whence : i32) -> i64
fn vx_file_drop(ptr : *mut i8) -> i32
fn fopen(path : *const i8, mode : *const i8) -> *mut i8
fn fread(ptr : *mut u8, size : i64, nmemb : i64, stream : *mut i8) -> i64
fn fclose(stream : *mut i8) -> i32
fn fileno(f : *mut i8) -> i32
fn fseek(f : *mut i8, offset : i64, whence : i32) -> i32
fn ftell(f : *mut i8) -> i64
fn mmap(addr : *mut i8, len : i64, prot : i32, flags : i32, fd : i32, offset : i64) -> *mut i8
fn munmap(addr : *mut i8, len : i64) -> i32
```

## `std::googletest`

Assertions for tests written in Vx.

**Types**

- `trait GoogletestEq`

**Functions**

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
fn expect_eq(self : Self, expected : Self) -> i32
```

**`GoogletestEq for f32` methods**

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
fn expect_eq(self : f32, expected : f32) -> i32
```

**`GoogletestEq for i32` methods**

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
fn expect_eq(self : i32, expected : i32) -> i32
fn expect_eq<T : GoogletestEq>(actual : T, expected : T) -> i32
```

**C bindings** *(the native functions this module is built on)*

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
fn vx_googletest_expect_eq_f32(actual : f32, expected : f32) -> i32
fn vx_googletest_expect_eq_i32(actual : i32, expected : i32) -> i32
```

## `std::hash_map`

`HashMap<K, V>`.

**Functions** *(bound directly to C)*

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
fn vx_hash_map_new_i32_i32() -> *mut i8
fn vx_hash_map_insert_i32_i32(ptr : *mut i8, key : i32, val : i32) -> i32
fn vx_hash_map_get_i32_i32(ptr : *mut i8, key : i32) -> *mut i8
fn vx_hash_map_contains_key_i32_i32(ptr : *mut i8, key : i32) -> Bool
fn vx_hash_map_len_i32_i32(ptr : *mut i8) -> i32
fn vx_hash_map_drop_i32_i32(ptr : *mut i8) -> i32
fn vx_hash_map_new_i32_f32() -> *mut i8
fn vx_hash_map_insert_i32_f32(ptr : *mut i8, key : i32, val : f32) -> i32
fn vx_hash_map_get_i32_f32(ptr : *mut i8, key : i32) -> *mut i8
fn vx_hash_map_contains_key_i32_f32(ptr : *mut i8, key : i32) -> Bool
fn vx_hash_map_len_i32_f32(ptr : *mut i8) -> i32
fn vx_hash_map_drop_i32_f32(ptr : *mut i8) -> i32
```

## `std::hash_set`

`HashSet<T>`.

**Functions** *(bound directly to C)*

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
fn vx_hash_set_new_i32() -> *mut i8
fn vx_hash_set_insert_i32(ptr : *mut i8, val : i32) -> i32
fn vx_hash_set_contains_i32(ptr : *mut i8, val : i32) -> Bool
fn vx_hash_set_len_i32(ptr : *mut i8) -> i32
fn vx_hash_set_drop_i32(ptr : *mut i8) -> i32
```

## `std::io`

Standard input, output and error.

**Functions**

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
unsafe fn stdout_write(buffer : *const u8, len : i64) -> i64
unsafe fn stderr_write(buffer : *const u8, len : i64) -> i64
unsafe fn stdin_read(buffer : *mut u8, len : i64) -> i64
```

**C bindings** *(the native functions this module is built on)*

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
fn vx_stdout_write(buffer : *const u8, len : i64) -> i64
fn vx_stderr_write(buffer : *const u8, len : i64) -> i64
fn vx_stdin_read(buffer : *mut u8, len : i64) -> i64
```

## `std::iter`

The `Iterator` trait and its adaptors, which `for` loops and `.map` build on.

**Types**

- `trait Iterator<T, Item>`

**Functions**

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
fn next(self : &mut T) -> Option<Item>
```

**`Iterator<Map<I, F, Item, NewItem>, NewItem> for Map<I, F, Item, NewItem>` methods**

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
fn next(self : &mut Map<I, F, Item, NewItem>) -> Option<NewItem>
```

**`Map<I, F, Item, NewItem>` methods**

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
fn collect(self : &mut Map<I, F, Item, NewItem>) -> Vec<NewItem>
```

## `std::libc`

Direct bindings to the C library.

**Functions** *(bound directly to C)*

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
fn open(path : *const i8, flags : i32) -> i32
fn close(fd : i32) -> i32
fn lseek(fd : i32, offset : i64, whence : i32) -> i64
```

## `std::llama`

Helpers used by the Llama 2 example.

**Types**

- `struct LlamaConfig`
- `struct TransformerWeightOffsets`
- `struct Tokenizer`

**`LlamaConfig` methods**

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
fn load(filepath : *const i8) -> LlamaConfig
```

**`TransformerWeightOffsets` methods**

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
fn calculate(c : &LlamaConfig) -> TransformerWeightOffsets
fn load_all_weights(filepath : *const i8, c : &LlamaConfig) -> Tensor<f32, [?, ?]>
```

**`Tokenizer` methods**

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
fn load(filepath : *const i8, vocab_size : i32) -> Tokenizer
fn decode(self : &Tokenizer, prev_token : i32, token : i32) -> String
```

**C bindings** *(the native functions this module is built on)*

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
fn vx_load_config(filepath : *const i8) -> *mut i32
fn vx_load_weights(filepath : *const i8) -> *mut f32
fn vx_build_tokenizer(filepath : *const i8, vocab_size : i32) -> *mut i8
fn vx_decode_token(tokenizer_ptr : *mut i8, prev_token : i32, token : i32) -> *const i8
fn vx_encode_prompt(tokenizer_ptr : *mut i8, text_ptr : *const i8) -> *mut i32
fn vx_read_prompt_file(filepath : *const i8) -> *const i8
fn vx_get_llama_config() -> *mut i32
```

## `std::math`

Mathematical functions and constants.

**Types**

- `trait Math`

**Functions**

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
fn sin(self : Self) -> Self
fn cos(self : Self) -> Self
fn tan(self : Self) -> Self
fn abs(self : Self) -> Self
fn sqrt(self : Self) -> Self
fn exp(self : Self) -> Self
fn ln(self : Self) -> Self
fn asin(self : Self) -> Self
fn acos(self : Self) -> Self
fn atan(self : Self) -> Self
fn log2(self : Self) -> Self
fn log10(self : Self) -> Self
```

**`Math for f32` methods**

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
fn sin(self : f32) -> f32
fn cos(self : f32) -> f32
fn tan(self : f32) -> f32
fn abs(self : f32) -> f32
fn sqrt(self : f32) -> f32
fn exp(self : f32) -> f32
fn ln(self : f32) -> f32
fn asin(self : f32) -> f32
fn acos(self : f32) -> f32
fn atan(self : f32) -> f32
fn log2(self : f32) -> f32
fn log10(self : f32) -> f32
```

**`Math for f64` methods**

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
fn sin(self : f64) -> f64
fn cos(self : f64) -> f64
fn tan(self : f64) -> f64
fn abs(self : f64) -> f64
fn sqrt(self : f64) -> f64
fn exp(self : f64) -> f64
fn ln(self : f64) -> f64
fn asin(self : f64) -> f64
fn acos(self : f64) -> f64
fn atan(self : f64) -> f64
fn log2(self : f64) -> f64
fn log10(self : f64) -> f64
```

**C bindings** *(the native functions this module is built on)*

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
fn sinf(x : f32) -> f32
fn cosf(x : f32) -> f32
fn tanf(x : f32) -> f32
fn asinf(x : f32) -> f32
fn acosf(x : f32) -> f32
fn atanf(x : f32) -> f32
fn fabsf(x : f32) -> f32
fn sqrtf(x : f32) -> f32
fn expf(x : f32) -> f32
fn logf(x : f32) -> f32
fn log2f(x : f32) -> f32
fn log10f(x : f32) -> f32
fn sin(x : f64) -> f64
fn cos(x : f64) -> f64
fn tan(x : f64) -> f64
fn asin(x : f64) -> f64
fn acos(x : f64) -> f64
fn atan(x : f64) -> f64
fn fabs(x : f64) -> f64
fn sqrt(x : f64) -> f64
fn exp(x : f64) -> f64
fn log(x : f64) -> f64
fn log2(x : f64) -> f64
fn log10(x : f64) -> f64
```

## `std::mmap`

Memory-mapped files.

**Functions** *(bound directly to C)*

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
fn mmap(addr : *mut i8, length : i64, prot : i32, flags : i32, fd : i32, offset : i64) -> *mut i8
fn munmap(addr : *mut i8, length : i64) -> i32
```

## `std::net`

TCP and UDP sockets.

**Types**

- `struct TcpStream`
- `struct UdpSocket`
- `struct TcpListener`

**`TcpStream` methods**

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
unsafe fn connect(addr : *const i8) -> TcpStream
unsafe fn read(self : *mut TcpStream, buffer : *mut u8, len : i64) -> i64
unsafe fn write(self : *mut TcpStream, buffer : *const u8, len : i64) -> i64
unsafe fn tcp_stream_drop(stream : *mut TcpStream) -> void
```

**`UdpSocket` methods**

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
unsafe fn bind(addr : *const i8) -> UdpSocket
unsafe fn recv(self : *mut UdpSocket, buffer : *mut u8, len : i64) -> i64
unsafe fn send_to(self : *mut UdpSocket, buffer : *const u8, len : i64, addr : *const i8) -> i64
unsafe fn udp_socket_drop(socket : *mut UdpSocket) -> void
```

**`TcpListener` methods**

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
unsafe fn bind(addr : *const i8) -> TcpListener
fn accept(self : *mut TcpListener) -> TcpStream
unsafe fn tcp_listener_drop(listener : *mut TcpListener) -> void
```

**C bindings** *(the native functions this module is built on)*

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
fn vx_tcp_stream_connect(c_addr : *const i8) -> *mut i8
fn vx_tcp_stream_read(ptr : *mut i8, buffer : *mut u8, len : i64) -> i64
fn vx_tcp_stream_write(ptr : *mut i8, buffer : *const u8, len : i64) -> i64
fn vx_tcp_stream_drop(ptr : *mut i8) -> i32
fn vx_udp_socket_bind(c_addr : *const i8) -> *mut i8
fn vx_udp_socket_recv(ptr : *mut i8, buffer : *mut u8, len : i64) -> i64
fn vx_udp_socket_send_to(ptr : *mut i8, buffer : *const u8, len : i64, c_addr : *const i8) -> i64
fn vx_udp_socket_drop(ptr : *mut i8) -> i32
fn vx_tcp_listener_bind(c_addr : *const i8) -> *mut i8
fn vx_tcp_listener_accept(ptr : *mut i8) -> *mut i8
fn vx_tcp_listener_drop(ptr : *mut i8) -> i32
```

## `std::simd`

SIMD vector types and operations.

**Functions**

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
unsafe fn simd_add_f32x4(a : *const f32, b : *const f32, out : *mut f32) -> i32
unsafe fn simd_sub_f32x4(a : *const f32, b : *const f32, out : *mut f32) -> i32
unsafe fn simd_mul_f32x4(a : *const f32, b : *const f32, out : *mut f32) -> i32
unsafe fn simd_div_f32x4(a : *const f32, b : *const f32, out : *mut f32) -> i32
unsafe fn simd_fma_f32x4(a : *const f32, b : *const f32, c : *const f32, out : *mut f32) -> i32
```

**C bindings** *(the native functions this module is built on)*

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
fn vx_simd_add_f32x4(a : *const f32, b : *const f32, out : *mut f32) -> i32
fn vx_simd_sub_f32x4(a : *const f32, b : *const f32, out : *mut f32) -> i32
fn vx_simd_mul_f32x4(a : *const f32, b : *const f32, out : *mut f32) -> i32
fn vx_simd_div_f32x4(a : *const f32, b : *const f32, out : *mut f32) -> i32
fn vx_simd_fma_f32x4(a : *const f32, b : *const f32, c : *const f32, out : *mut f32) -> i32
```

## `std::string`

`String` and text manipulation.

**Types**

- `struct String`

**`String` methods**

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
fn new() -> String
unsafe fn from_c_str(c_str : *const i8) -> String
unsafe fn push_c_str(self : *mut String, c_str : *const i8) -> i32
fn len(self : *mut String) -> i32
fn as_c_str(self : *mut String) -> *const i8
fn drop(self : *mut String) -> i32
```

**`i32` methods**

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
fn to_string(self : i32) -> String
unsafe fn string_length(s : *const i8) -> i32
unsafe fn string_compare(s1 : *const i8, s2 : *const i8) -> i32
unsafe fn parse_int(s : *const i8) -> i32
```

**C bindings** *(the native functions this module is built on)*

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
fn vx_string_new() -> *mut i8
fn vx_string_from_c_str(ptr : *const i8) -> *mut i8
fn vx_string_push_c_str(ptr : *mut i8, c_str : *const i8) -> i32
fn vx_string_len(ptr : *mut i8) -> i32
fn vx_string_as_c_str(ptr : *mut i8) -> *const i8
fn vx_string_free_c_str(ptr : *const i8) -> i32
fn vx_string_drop(ptr : *mut i8) -> i32
fn vx_i32_to_string(val : i32) -> *mut i8
```

## `std::tensor`

Operations on `Tensor`, including shape queries and elementwise maths.

**`Tensor<T, [?, ?]>` methods**

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
fn from_ptr_1d(ptr : *mut T, d1 : i32) -> Tensor<T, [?, ?]>
fn from_ptr_2d(ptr : *mut T, d1 : i32, d2 : i32) -> Tensor<T, [?, ?]>
fn slice_2d(self : &Tensor<T, [?, ?]>, row : i32, d1 : i32, d2 : i32) -> Tensor<T, [?, ?]>
fn slice_2d_from_1d(self : &Tensor<T, [?, ?]>, start : i32, d1 : i32, d2 : i32) -> Tensor<T, [?, ?]>
fn slice_1d(self : &Tensor<T, [?, ?]>, start : i32, d1 : i32) -> Tensor<T, [?, ?]>
fn fill(self : &mut Tensor<T, [?, ?]>, val : T) -> void
fn copy(self : &mut Tensor<T, [?, ?]>, src : &Tensor<T, [?, ?]>) -> void
fn assign(self : &mut Tensor<T, [?, ?]>, val : T) -> void
fn compare(self : &Tensor<T, [?, ?]>, other : &Tensor<T, [?, ?]>) -> bool
```

**`Tensor<T, [N, M]>` methods**

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
fn fill_static(self : &mut Tensor<T, [ N, M ]>, val : T) -> void
```

## `std::time`

Clocks and durations.

**Functions**

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
fn now() -> f32
fn sleep(seconds : f32) -> i32
fn unix_timestamp() -> f64
unsafe fn bench_report(name : *const i8, unit : *const i8, value : f32) -> i32
```

**C bindings** *(the native functions this module is built on)*

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
fn vx_get_time() -> f32
fn vx_sleep(seconds : f32) -> i32
fn vx_unix_timestamp() -> f64
fn vx_bench_report(name : *const i8, unit : *const i8, value : f32) -> i32
```

## `std::vec`

`Vec<T>`, a growable array.

**Types**

- `struct Vec<T>`
- `struct VecIter<T>`
- `struct VecMap<T, NewItem>`

**`Vec<T>` methods**

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
fn new() -> Vec<T>
fn with_capacity(capacity : i32) -> Vec<T>
fn free(self : &mut Vec<T>) -> i32
fn as_mut_ptr(self : &Vec<T>) -> *mut T
fn as_mut_slice(self : &mut Vec<T>) -> &mut T
fn as_slice(self : &Vec<T>) -> &T
fn push(self : &mut Vec<T>, val : T) -> i32
fn get(self : &Vec<T>, index : i32) -> T
fn set(self : &mut Vec<T>, index : i32, val : T) -> i32
fn len(self : &Vec<T>) -> i32
fn iter(self : &Vec<T>) -> VecIter<T>
```

**`Iterator<VecIter<T>, T> for VecIter<T>` methods**

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
fn next(self : &mut VecIter<T>) -> Option<T>
```

**`VecIter<T>` methods**

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
fn map<NewItem>(self : VecIter<T>, f : Closure1<T, NewItem>) -> VecMap<T, NewItem>
```

**`Iterator<VecMap<T, NewItem>, NewItem> for VecMap<T, NewItem>` methods**

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
fn next(self : &mut VecMap<T, NewItem>) -> Option<NewItem>
```

**`VecMap<T, NewItem>` methods**

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
fn collect(self : &mut VecMap<T, NewItem>) -> Vec<NewItem>
```

**C bindings** *(the native functions this module is built on)*

<!-- vx-doctest: skip -- signature listing, not a program -->

```rust
fn vx_vec_alloc(elem_size : i64, cap : i64) -> *mut i8
fn vx_vec_grow(ptr : *mut i8, old_cap : i64, new_cap : i64, elem_size : i64) -> *mut i8
fn vx_vec_free(ptr : *mut i8, cap : i64, elem_size : i64) -> i32
fn vx_vec_bounds_check(index : i64, len : i64) -> i32
```

______________________________________________________________________

311 functions across 24 modules.
