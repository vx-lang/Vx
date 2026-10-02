//===- lib.rs - Vx Compiler ------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Copies of the lifetime check in src/borrow.rs (the fast path of verify_subtyping_bounds), so
// they can be compared on billions of inputs and timed apart from the rest of the compiler.
//
// A word packs four 16-bit slots: [variance (4 bits) | region (12 bits)]. Slot 0 is the return
// slot, and only the low 9 bits of its region field are the region. A region with every bit of
// its mask set is "unset" and matches anything. Variance 0 needs equal regions, 1 needs
// a <= b, and 2 needs a >= b.
//
//===----------------------------------------------------------------------===//

pub const VARIANCE_MASK: u64 = 0xF000;
pub const PARAM_MASK: u64 = 0xFFFF;
pub const REGION_MASK: u64 = 0x0FFF;
pub const REGION_UNSET: u64 = REGION_MASK;
pub const REGION_MASK_0: u64 = 0x01FF;
pub const REGION_UNSET_0: u64 = REGION_MASK_0;

/// The old loop, as src/borrow.rs had it before it was made branch-free: it returns at the first
/// slot that fails.
pub fn check_old_loop(bits_a: u64, bits_b: u64) -> bool {
    if bits_a == bits_b {
        return true;
    }
    for i in 0..4 {
        let shift = i * 16;
        let slot_a = (bits_a >> shift) & PARAM_MASK;
        let slot_b = (bits_b >> shift) & PARAM_MASK;
        let variance_a = (slot_a & VARIANCE_MASK) >> 12;
        let variance_b = (slot_b & VARIANCE_MASK) >> 12;
        if variance_a != variance_b {
            return false;
        }
        let (region_mask, region_unset) = if i == 0 {
            (REGION_MASK_0, REGION_UNSET_0)
        } else {
            (REGION_MASK, REGION_UNSET)
        };
        let region_a = slot_a & region_mask;
        let region_b = slot_b & region_mask;
        if region_a == region_unset || region_b == region_unset {
            continue;
        }
        let valid = match variance_a {
            0x0 => region_a == region_b,
            0x1 => region_a <= region_b,
            0x2 => region_a >= region_b,
            _ => false,
        };
        if !valid {
            return false;
        }
    }
    true
}

const MASKS: [u16; 4] = [
    REGION_MASK_0 as u16,
    REGION_MASK as u16,
    REGION_MASK as u16,
    REGION_MASK as u16,
];

/// One slot, with no branches: every condition is computed, then combined with `&` and `|`.
#[inline(always)]
fn slot_ok(a: u16, b: u16, mask: u16) -> bool {
    let (va, vb) = (a >> 12, b >> 12);
    let (ra, rb) = (a & mask, b & mask);
    let wildcard = (ra == mask) | (rb == mask);
    let order = ((va == 0) & (ra == rb)) | ((va == 1) & (ra <= rb)) | ((va == 2) & (ra >= rb));
    (va == vb) & (wildcard | order)
}

/// The check with no early exit: all four slots are checked and the answers ANDed.
// Indexed as it was when the post's figures were measured.
#[allow(clippy::needless_range_loop)]
#[inline(always)]
pub fn check_branch_free(bits_a: u64, bits_b: u64) -> bool {
    let mut all = true;
    for i in 0..4 {
        let a = (bits_a >> (16 * i)) as u16;
        let b = (bits_b >> (16 * i)) as u16;
        all &= slot_ok(a, b, MASKS[i]);
    }
    (bits_a == bits_b) | all
}

/// A batch through the old loop.
pub fn batch_old_loop(a: &[u64], b: &[u64], out: &mut [bool]) {
    for ((x, y), o) in a.iter().zip(b).zip(out.iter_mut()) {
        *o = check_old_loop(*x, *y);
    }
}

/// The first branch-free attempt: one flat loop over every slot of every pair, looking up each
/// slot's mask, then a second loop that ANDs each group of four. It is slow; it is kept because
/// the post reports it.
pub fn batch_two_loops(a: &[u64], b: &[u64], out: &mut [bool]) {
    const CHUNK: usize = 256;
    let mut ok = [0u16; CHUNK * 4];
    for ((ca, cb), co) in a
        .chunks(CHUNK)
        .zip(b.chunks(CHUNK))
        .zip(out.chunks_mut(CHUNK))
    {
        let n = ca.len();
        // A u64 read as four u16 slots, low slot first, matches `>> (16 * i)` on little-endian
        // machines; main.rs refuses to build anywhere else.
        let sa: &[u16] = unsafe { std::slice::from_raw_parts(ca.as_ptr() as *const u16, n * 4) };
        let sb: &[u16] = unsafe { std::slice::from_raw_parts(cb.as_ptr() as *const u16, n * 4) };
        for j in 0..n * 4 {
            ok[j] = if slot_ok(sa[j], sb[j], MASKS[j % 4]) {
                0xFFFF
            } else {
                0
            };
        }
        for k in 0..n {
            let all = (ok[4 * k] & ok[4 * k + 1] & ok[4 * k + 2] & ok[4 * k + 3]) == 0xFFFF;
            co[k] = (ca[k] == cb[k]) | all;
        }
    }
}

/// A batch through the branch-free check: the loop LLVM vectorizes.
pub fn batch_branch_free(a: &[u64], b: &[u64], out: &mut [bool]) {
    for ((x, y), o) in a.iter().zip(b).zip(out.iter_mut()) {
        *o = check_branch_free(*x, *y);
    }
}

/// One old-loop check, kept as a call so nothing is shared between pairs.
#[inline(never)]
pub fn single_old_loop(a: u64, b: u64) -> bool {
    check_old_loop(a, b)
}

/// One branch-free check, kept as a call. Its assembly is the listing in the post.
#[inline(never)]
pub fn single_branch_free(a: u64, b: u64) -> bool {
    check_branch_free(a, b)
}

/// The equality test first, then the branch-free check: what src/borrow.rs now runs.
#[inline(never)]
pub fn single_equality_first(a: u64, b: u64) -> bool {
    if a == b {
        return true;
    }
    check_branch_free(a, b)
}

/// A batch made of separate calls to `f`, the way the compiler makes one call per assignment.
pub fn calls(f: fn(u64, u64) -> bool, a: &[u64], b: &[u64], out: &mut [bool]) {
    for ((x, y), o) in a.iter().zip(b).zip(out.iter_mut()) {
        *o = f(*x, *y);
    }
}
