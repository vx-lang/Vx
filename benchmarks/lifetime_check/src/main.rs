//===- main.rs - Vx Compiler -----------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Compares and times the copies of the lifetime check in lib.rs. See README.md.
//
//===----------------------------------------------------------------------===//

use lifetime_check::*;
use std::hint::black_box;
use std::time::Instant;

#[cfg(not(target_endian = "little"))]
compile_error!("batch_two_loops reads a u64 as four u16 slots, low slot first");

/// The four kinds of input in the post's tables, in the order they appear there.
const INPUTS: [&str; 4] = ["passing", "made_up", "random", "identical"];

/// A batch of checks: two slices of words in, one answer per pair out.
type Batch<'a> = &'a dyn Fn(&[u64], &[u64], &mut [bool]);

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// A word like the ones Vx builds: variances 0 to 2, small regions, sometimes unset.
fn made_up_word(r: &mut Rng) -> u64 {
    let mut w = 0u64;
    for i in 0..4 {
        let v = r.next() % 3;
        let unset = if i == 0 { REGION_UNSET_0 } else { REGION_UNSET };
        let region = if r.next().is_multiple_of(16) {
            unset
        } else {
            r.next() % 8
        };
        w |= ((v << 12) | region) << (16 * i);
    }
    w
}

fn input(kind: &str, n: usize) -> (Vec<u64>, Vec<u64>) {
    let mut r = Rng(0x9E3779B97F4A7C15);
    let mut a = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    for _ in 0..n {
        let (x, y) = match kind {
            "random" => (r.next(), r.next()),
            "made_up" => (made_up_word(&mut r), made_up_word(&mut r)),
            // Pairs that pass every slot: covariant regions moved up by one where they can be.
            "passing" => loop {
                let x = made_up_word(&mut r);
                let mut y = x;
                for i in 0..4 {
                    let v = (x >> (16 * i + 12)) & 0xF;
                    if v == 1 && (y >> (16 * i)) & 0xFFF < 7 {
                        y += 1 << (16 * i);
                    }
                }
                // Identical words would take the old loop's first return, so leave them out.
                if x != y {
                    break (x, y);
                }
            },
            "identical" => {
                let x = made_up_word(&mut r);
                (x, x)
            }
            _ => unreachable!("unknown input {kind}"),
        };
        a.push(x);
        b.push(y);
    }
    (a, b)
}

/// Every pair of 16-bit values in one slot, with the other three slots zero: 2^32 pairs per slot.
fn exhaustive() {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get().saturating_sub(2).max(1))
        .unwrap_or(1) as u32;
    for slot in 0..4u32 {
        let workers: Vec<_> = (0..threads)
            .map(|t| {
                std::thread::spawn(move || {
                    let mut bad = 0u64;
                    let mut a = t;
                    while a < 65536 {
                        let wa = (a as u64) << (16 * slot);
                        for b in 0..65536u64 {
                            let wb = b << (16 * slot);
                            if check_old_loop(wa, wb) != check_branch_free(wa, wb) {
                                bad += 1;
                            }
                        }
                        a += threads;
                    }
                    bad
                })
            })
            .collect();
        let bad: u64 = workers.into_iter().map(|w| w.join().unwrap()).sum();
        println!("slot {slot}: 4294967296 pairs, {bad} disagreements");
    }
}

/// 50 million pairs of each input through every version, compared with the old loop.
fn agree() {
    let n = 50_000_000;
    for kind in INPUTS {
        let (a, b) = input(kind, n);
        let mut expected = vec![false; n];
        batch_old_loop(&a, &b, &mut expected);
        let mut bad = 0;
        let mut got = vec![false; n];
        for f in [batch_two_loops, batch_branch_free] {
            f(&a, &b, &mut got);
            bad += (0..n).filter(|&i| got[i] != expected[i]).count();
        }
        for f in [single_branch_free, single_equality_first] {
            calls(f, &a, &b, &mut got);
            bad += (0..n).filter(|&i| got[i] != expected[i]).count();
        }
        let passed = expected.iter().filter(|x| **x).count();
        println!("{kind:>9}: {n} pairs, {passed} pass, {bad} disagreements");
    }
}

/// Nanoseconds per check: 1,048,576 pairs, best of 30 runs, on one thread.
fn bench() {
    let n = 1 << 20;
    for kind in INPUTS {
        let (a, b) = input(kind, n);
        let mut out = vec![false; n];
        let mut time = |f: Batch| {
            let mut best = f64::MAX;
            for _ in 0..30 {
                let t = Instant::now();
                f(black_box(&a), black_box(&b), black_box(&mut out));
                best = best.min(t.elapsed().as_secs_f64());
            }
            best * 1e9 / n as f64
        };
        let old = time(&batch_old_loop);
        let two_loops = time(&batch_two_loops);
        let branch_free = time(&batch_branch_free);
        let one_old = time(&|a, b, o| calls(single_old_loop, a, b, o));
        let one_free = time(&|a, b, o| calls(single_branch_free, a, b, o));
        let one_first = time(&|a, b, o| calls(single_equality_first, a, b, o));
        println!(
            "{kind:>9}  batch: old loop {old:.2}, branch-free {branch_free:.2}, \
             two loops {two_loops:.2}  |  one call each: old loop {one_old:.2}, \
             branch-free {one_free:.2}, equality test first {one_first:.2}"
        );
    }
}

fn main() {
    match std::env::args().nth(1).as_deref() {
        Some("exhaustive") => exhaustive(),
        Some("agree") => agree(),
        Some("bench") => bench(),
        _ => eprintln!("usage: lifetime_check exhaustive | agree | bench"),
    }
}
