//===- vec.rs - Vx Compiler ------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// This file implements the FFI bindings for the dynamic Vector type.
// It manages heap allocations, dynamic resizing, and safe pointer arithmetic for
// variable-length arrays used within the Vx standard library.
//
// Two surfaces live here:
//
//   * `instantiate_vec_ffi!` — opaque-handle `Vec<T>` endpoints (`vx_vec_*_i32`
//     etc.), used by the FFI demo/backend tests.
//
//   * A *type-erased byte allocator* (`vx_vec_alloc`/`grow`/`free`/`bounds_check`)
//     that backs `stdlib/std/vec.vx`. The Vx `Vec<T>` keeps a typed `data: *mut T`
//     view and does its own typed loads/stores, but delegates every *dangerous*
//     operation — alignment, growth doubling, `capacity * elem_size` overflow,
//     out-of-memory, and bounds enforcement — to Rust here. That way the hand-
//     rolled correctness pitfalls (silent i32 overflow, unchecked malloc, and the
//     compile-time-only `assert` bounds "checks") become Rust's well-defined,
//     abort-on-error behavior instead.
//
//===----------------------------------------------------------------------===//
use crate::instantiate_vec_ffi;
use std::alloc::{alloc, dealloc, realloc, Layout};

// Instantiate FFI endpoints for primitive vectors
instantiate_vec_ffi!(i32, i32);
instantiate_vec_ffi!(f32, f32);
instantiate_vec_ffi!(i64, i64);
instantiate_vec_ffi!(f64, f64);

/// Fixed over-alignment for Vx `Vec<T>` backing storage. 16 bytes covers every
/// scalar Vx element type (i8..i64, f32/f64, raw pointers) and matches the
/// alignment `malloc` previously provided, so typed loads/stores through the
/// Vx-side `data: *mut T` view stay well-aligned regardless of `T`.
const VX_VEC_ALIGN: usize = 16;

/// Build the allocation `Layout` for `bytes` at the fixed vector alignment.
/// Panics (aborts the process) if the size overflows the address space — the
/// same failure the hand-rolled Vx code would have hit as silent UB.
#[inline]
fn vx_vec_layout(bytes: usize) -> Layout {
    Layout::from_size_align(bytes, VX_VEC_ALIGN)
        .expect("vx_vec: capacity in bytes overflows the address space")
}

/// Convert an element count / element size pair coming across the C ABI (as
/// `i64`, since that is what Vx's `sizeof<T>()` yields) into a byte count,
/// rejecting negatives and multiplication overflow deterministically.
#[inline]
fn vx_vec_bytes(cap: i64, elem_size: i64) -> usize {
    let cap = usize::try_from(cap).expect("vx_vec: negative capacity");
    let elem = usize::try_from(elem_size).expect("vx_vec: negative element size");
    cap.checked_mul(elem)
        .expect("vx_vec: capacity * element size overflows usize")
}

/// Allocate a 16-byte-aligned buffer for `cap` elements of `elem_size` bytes.
///
/// Returns null for a zero-byte request; `vx_vec_grow`/`vx_vec_free` treat null
/// as "no live allocation", so an empty `Vec` never owns memory.
#[no_mangle]
pub extern "C" fn vx_vec_alloc(elem_size: i64, cap: i64) -> *mut u8 {
    let bytes = vx_vec_bytes(cap, elem_size);
    if bytes == 0 {
        return std::ptr::null_mut();
    }
    let ptr = unsafe { alloc(vx_vec_layout(bytes)) };
    assert!(!ptr.is_null(), "vx_vec: allocation of {bytes} bytes failed");
    ptr
}

/// Grow (or initially allocate) the buffer from `old_cap` to `new_cap` elements,
/// preserving existing contents. Rust owns the realloc and the overflow check,
/// so the Vx side never has to compute a byte size that could silently wrap.
///
/// `ptr`/`old_cap` must describe the buffer's current allocation (as produced by
/// a previous `vx_vec_alloc`/`vx_vec_grow`); a null `ptr` or non-positive
/// `old_cap` is treated as a fresh allocation.
#[no_mangle]
pub extern "C" fn vx_vec_grow(ptr: *mut u8, old_cap: i64, new_cap: i64, elem_size: i64) -> *mut u8 {
    let new_bytes = vx_vec_bytes(new_cap, elem_size);
    if new_bytes == 0 {
        return ptr;
    }
    let new_ptr = if ptr.is_null() || old_cap <= 0 {
        unsafe { alloc(vx_vec_layout(new_bytes)) }
    } else {
        let old_bytes = vx_vec_bytes(old_cap, elem_size);
        unsafe { realloc(ptr, vx_vec_layout(old_bytes), new_bytes) }
    };
    assert!(
        !new_ptr.is_null(),
        "vx_vec: reallocation to {new_bytes} bytes failed"
    );
    new_ptr
}

/// Free a buffer previously returned by `vx_vec_alloc`/`vx_vec_grow`. `cap` and
/// `elem_size` must match the buffer's current allocation so the `Layout` lines
/// up with the one used to allocate it. A null pointer is a no-op.
#[no_mangle]
pub extern "C" fn vx_vec_free(ptr: *mut u8, cap: i64, elem_size: i64) -> i32 {
    if ptr.is_null() {
        return 0;
    }
    let bytes = vx_vec_bytes(cap, elem_size);
    if bytes > 0 {
        unsafe { dealloc(ptr, vx_vec_layout(bytes)) };
    }
    0
}

/// Runtime bounds check for `Vec::get`/`Vec::set`. Aborts — rather than reading
/// or writing out of bounds — when `index` is not in `0..len`. Vx's `assert` is
/// compile-time only and silently skips runtime-valued conditions, so this is
/// where real, always-on bounds enforcement lives.
#[no_mangle]
pub extern "C" fn vx_vec_bounds_check(index: i64, len: i64) -> i32 {
    if index < 0 || index >= len {
        eprintln!("vx_vec: index {index} out of bounds for length {len}");
        std::process::abort();
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec_i32_ffi() {
        let ptr = vx_vec_new_i32();
        assert!(!ptr.is_null());
        assert_eq!(vx_vec_len_i32(ptr), 0);

        vx_vec_push_i32(ptr, 42);
        assert_eq!(vx_vec_len_i32(ptr), 1);

        vx_vec_push_i32(ptr, 100);
        assert_eq!(vx_vec_len_i32(ptr), 2);

        vx_vec_drop_i32(ptr);
    }

    #[test]
    fn test_vec_f32_ffi() {
        let ptr = vx_vec_new_f32();
        assert!(!ptr.is_null());
        assert_eq!(vx_vec_len_f32(ptr), 0);

        vx_vec_push_f32(ptr, 1.5);
        assert_eq!(vx_vec_len_f32(ptr), 1);

        vx_vec_drop_f32(ptr);
    }

    // -- type-erased byte allocator (backs stdlib/std/vec.vx) ------------------

    /// Exercise the full alloc → grow → typed store/load → free cycle the Vx
    /// `Vec<i32>` performs, and confirm the buffer is usable and 16-aligned.
    #[test]
    fn byte_alloc_grow_and_store_i32() {
        let elem = std::mem::size_of::<i32>() as i64;

        // with_capacity(2)
        let mut cap: i64 = 2;
        let mut data = vx_vec_alloc(elem, cap);
        assert!(!data.is_null());
        assert_eq!(data as usize % VX_VEC_ALIGN, 0, "buffer must be 16-aligned");

        // Fill and grow the way push() does, checking data survives realloc.
        for i in 0..10i32 {
            let len = i as i64;
            if len == cap {
                let new_cap = cap * 2;
                data = vx_vec_grow(data, cap, new_cap, elem);
                assert!(!data.is_null());
                assert_eq!(data as usize % VX_VEC_ALIGN, 0);
                cap = new_cap;
            }
            unsafe { *(data as *mut i32).offset(i as isize) = i * 7 };
        }
        for i in 0..10i32 {
            let got = unsafe { *(data as *const i32).offset(i as isize) };
            assert_eq!(got, i * 7);
        }
        assert_eq!(vx_vec_free(data, cap, elem), 0);
    }

    /// A zero-capacity vector must not own memory, and growing from empty must
    /// behave like a fresh allocation.
    #[test]
    fn byte_alloc_zero_capacity_is_null() {
        let elem = std::mem::size_of::<f64>() as i64;
        let empty = vx_vec_alloc(elem, 0);
        assert!(empty.is_null());
        // free of null / zero-cap is a no-op.
        assert_eq!(vx_vec_free(empty, 0, elem), 0);
        // grow from empty allocates.
        let grown = vx_vec_grow(empty, 0, 4, elem);
        assert!(!grown.is_null());
        assert_eq!(vx_vec_free(grown, 4, elem), 0);
    }

    #[test]
    fn bounds_check_accepts_in_range() {
        assert_eq!(vx_vec_bounds_check(0, 3), 0);
        assert_eq!(vx_vec_bounds_check(2, 3), 0);
    }
}
