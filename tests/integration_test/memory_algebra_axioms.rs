//===- memory_algebra_axioms.rs - Vx Compiler ------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// S3 (vx-review#11): the cost model's AXIOMS, as tests.
//
// These separate two failure modes that look alike in a measurement campaign and have completely
// different responses:
//
//   * the model is internally inconsistent -- fatal, and cheap to find here;
//   * the model diverges from hardware -- the study's actual subject, measured on real machines.
//
// Additivity is the sharpest example, and it is the one that changed. "Path cost = sum of leg
// costs" was asserted here as an axiom. It is now false in the model on purpose: a containment hop
// charges both endpoints (vx-review#22), so staging through a space counts that space twice while
// a direct walk streams through it once. What survives is the inequality -- staging is never
// cheaper than streaming -- and that is what Axiom 1 now asserts.
//
// The hardware moves the same way and further: a staged HBM->L2->SMEM measured 0.60x the sum of
// its legs on an H100, because copy engines and TMA overlap the legs. That was M2's pre-registered
// expectation. So the model and the hardware now agree on the SIGN of the gap and disagree on its
// size, which is a residual to quantify rather than a bug to fix.
//
//===----------------------------------------------------------------------===//

use vxc::hir::memory::{granule_round, MemoryHierarchy};
use vxc::syntax::{MemorySpace, RatePer};

/// A three-level hierarchy with distinct bandwidths at every level, so a wrong leg cannot hide
/// behind an equal one, plus a granule to exercise the rounding axioms.
const MACHINE: &str = "
Memory HBM { capacity: 40 GiB, bandwidth: 3 TB/s }
Memory L2 { within: Memory::HBM, capacity: 50 MiB, bandwidth: 12 TB/s }
Memory SMEM { within: Memory::L2, capacity: 228 KiB, bandwidth: 128 B/cyc, granule: 1 KiB, scope: sm }
fn main() -> i32 { return 0; }
";

/// A hierarchy whose every level is a time rate, so costs are summable across the whole chain.
/// (`MACHINE` mixes `B/s` and `B/cyc`, which the model deliberately refuses to add.)
const TIME_MACHINE: &str = "
Memory DRAM { capacity: 1 TiB, bandwidth: 100 GB/s }
Memory HBM { within: Memory::DRAM, capacity: 80 GiB, bandwidth: 3 TB/s }
Memory L2 { within: Memory::HBM, capacity: 50 MiB, bandwidth: 12 TB/s }
Memory SMEM { within: Memory::L2, capacity: 228 KiB, bandwidth: 20 TB/s }
fn main() -> i32 { return 0; }
";

fn hierarchy(src: &str) -> vxc::syntax::VxModule {
    vxc::parse_module(src).expect("machine parses")
}

fn space(n: &str) -> MemorySpace {
    MemorySpace::from_name(n)
}

/// **Axiom 1 (revised): staging costs at least as much as streaming.**
///
/// This was strict additivity — `cost(A->C) == cost(A->B) + cost(B->C)` — asserted as an axiom.
/// It is now false, deliberately: since a containment hop charges both endpoints (vx-review#22),
/// staging through B writes into B and then reads back out of it, so B is counted twice, while the
/// direct walk streams through it once. The inequality is what survives.
///
/// The hardware agrees, and by a wide margin in the same direction. Measured on an H100, a staged
/// `HBM->L2->SMEM` costs 0.60x the sum of its legs (vx-review#16, measurements/EDGES.md) because
/// copy engines and TMA overlap the legs. M2 pre-registered exactly this, so the old axiom failing
/// is a confirmed prediction rather than a surprise — but note the model and the hardware disagree
/// on *how much*: the model now says staging is dearer by the doubled intermediates, and the
/// hardware says it is dearer still than that, because streaming overlaps and staging does not.
#[test]
fn staging_through_a_space_costs_at_least_the_direct_walk() {
    let m = hierarchy(TIME_MACHINE);
    let h = MemoryHierarchy::build(m.memories.iter());
    for &bytes in &[4096u64, 65_536, 1 << 20, 64 << 20] {
        let whole = h
            .derived_transfer_cost(&space("DRAM"), &space("SMEM"), bytes)
            .expect("DRAM -> SMEM is derivable");
        let legs: u64 = [("DRAM", "HBM"), ("HBM", "L2"), ("L2", "SMEM")]
            .iter()
            .map(|(a, b)| {
                h.derived_transfer_cost(&space(a), &space(b), bytes)
                    .unwrap_or_else(|| panic!("{a} -> {b} not derivable"))
                    .value
            })
            .sum();
        assert!(
            whole.value <= legs,
            "the direct walk must not cost more than staging through every level: \
             DRAM->SMEM is {} but its legs sum to {legs} at {bytes} bytes",
            whole.value
        );
        // The gap is exactly the intermediates, counted twice by the staged form and once by the
        // direct one. Assert it is real rather than a tie, so a regression to strict additivity
        // would fail here.
        assert!(
            whole.value < legs,
            "staging should cost strictly more than streaming at {bytes} bytes"
        );
    }
}

/// **Axiom 2: cost is monotone in bytes.** More data never costs less to move.
///
/// Checked across a wide sweep and at the rounding boundaries, where an off-by-one in the
/// `div_ceil` would show up as a cost that dips.
#[test]
fn cost_is_monotone_in_bytes() {
    let m = hierarchy(TIME_MACHINE);
    let h = MemoryHierarchy::build(m.memories.iter());
    let cost = |b: u64| {
        h.derived_transfer_cost(&space("DRAM"), &space("SMEM"), b)
            .map(|c| c.value)
    };

    let mut sizes: Vec<u64> = (0..24).map(|i| 1u64 << i).collect();
    // Neighbours of each power of two: a rounding error is likeliest to flip a comparison here.
    sizes.extend((1..24).flat_map(|i| [(1u64 << i) - 1, (1u64 << i) + 1]));
    sizes.sort_unstable();
    sizes.dedup();

    let mut prev: Option<(u64, u64)> = None;
    for b in sizes {
        let Some(c) = cost(b) else { continue };
        if let Some((pb, pc)) = prev {
            assert!(
                c >= pc,
                "cost is not monotone: {pb} bytes cost {pc}, but {b} bytes cost {c}"
            );
        }
        prev = Some((b, c));
    }
    assert!(
        prev.is_some(),
        "no costs were computed -- the sweep was vacuous"
    );
}

/// **Axiom 3: granule rounding is idempotent**, and it only ever rounds up.
///
/// Idempotence is what makes it safe for the capacity check and the working-set sum to apply the
/// rounding independently -- which they do, at two different call sites. Were it not idempotent,
/// the answer would depend on how many times a tile had been accounted for.
#[test]
fn granule_rounding_is_idempotent_and_never_shrinks() {
    for &g in &[None, Some(1u64), Some(64), Some(1024), Some(1 << 20)] {
        for &b in &[0u64, 1, 63, 64, 65, 1023, 1024, 1025, 16 * 1024, 1 << 30] {
            let once = granule_round(b, g);
            assert_eq!(
                once,
                granule_round(once, g),
                "granule_round is not idempotent for bytes={b} granule={g:?}"
            );
            assert!(
                once >= b,
                "granule_round shrank {b} to {once} (granule={g:?}) -- rounding must be up"
            );
            if let Some(g) = g.filter(|g| *g > 0) {
                assert_eq!(once % g, 0, "rounded size {once} is not a multiple of {g}");
                assert!(
                    once < b + g,
                    "granule_round over-rounded {b} to {once} (granule={g})"
                );
            }
        }
    }
    // A zero granule must mean "no granule" rather than a division by zero.
    assert_eq!(granule_round(1234, Some(0)), 1234);
}

/// **Axiom 4: a working set is invariant under placement order.**
///
/// The sum of granule-rounded tiles must not depend on the order the tiles were placed. It is a
/// sum, so this looks trivial -- but the rounding is per-tile, and the tempting simplification
/// (round the total once) is *not* order-invariant in the same way and gives a different, smaller
/// answer. Pinning it here keeps that optimisation from being made by accident.
#[test]
fn working_set_is_invariant_under_placement_order() {
    let granule = Some(1024u64);
    let tiles: Vec<u64> = vec![1, 1023, 1024, 1025, 4096, 5000, 16 * 1024];

    let total = |order: &[u64]| -> u64 { order.iter().map(|&b| granule_round(b, granule)).sum() };

    let baseline = total(&tiles);
    let mut permuted = tiles.clone();
    permuted.reverse();
    assert_eq!(
        baseline,
        total(&permuted),
        "reversing placement order changed the working set"
    );
    permuted.sort_unstable();
    assert_eq!(
        baseline,
        total(&permuted),
        "sorting placement order changed the working set"
    );
    permuted.sort_unstable_by(|a, b| b.cmp(a));
    assert_eq!(
        baseline,
        total(&permuted),
        "descending order changed the working set"
    );

    // Per-tile rounding is strictly more conservative than rounding the total, which is the whole
    // reason the model rounds per tile: a space holding seven 1-byte tiles at a 1 KiB granule
    // really has consumed 7 KiB, not 1.
    let rounded_total = granule_round(tiles.iter().sum::<u64>(), granule);
    assert!(
        baseline > rounded_total,
        "per-tile rounding ({baseline}) should exceed rounding the total ({rounded_total}); if \
         these are equal the test data no longer distinguishes the two rules"
    );
}

/// Realistic transfer sizes must produce a cost at all.
///
/// Regression for a bug introduced by the picosecond rescaling itself: the intermediate
/// `bytes * 1e12` was computed in `u64`, which overflows at ~18 MB, so every transfer at or above
/// that size silently returned "not derivable" instead of a number. Nothing caught it because the
/// probes in use were 16 KiB tiles. The sizes below bracket the old limit and run up to a KV-cache
/// -scale transfer, which is the regime the calibration study actually measures.
#[test]
fn large_transfers_still_have_a_cost() {
    let m = hierarchy(TIME_MACHINE);
    let h = MemoryHierarchy::build(m.memories.iter());
    for &bytes in &[
        17 << 20, // just under the old u64 overflow point
        19 << 20, // just over it
        1 << 30,  // 1 GiB
        64 << 30, // 64 GiB -- a full HBM's worth
    ] {
        let c = h
            .derived_transfer_cost(&space("DRAM"), &space("HBM"), bytes)
            .unwrap_or_else(|| panic!("{bytes} bytes produced no derived cost"));
        assert!(c.value > 0, "{bytes} bytes cost zero");
    }

    // A parent->child move charges BOTH: the data has to come out of DRAM as well as into HBM.
    // So 64 GiB from DRAM into HBM is 64 GiB/100 GB/s + 64 GiB/3 TB/s.
    //
    // This test previously asserted the opposite -- the child's bandwidth alone -- and called the
    // enclosing space's rate a modelling choice. Hardware settled it: pricing the destination alone
    // scored -85.1% on the H100 `HBM->L2` seam, and charging both halves brought it to -31.7%
    // (vx-review#15, measurements/EDGES.md). The enclosing space is on the critical path, and
    // leaving it out was under-pricing by omission.
    let c = h
        .derived_transfer_cost(&space("DRAM"), &space("HBM"), 64 << 30)
        .unwrap();
    // `div_ceil` per hop, not truncating division: a partial unit of time is still spent, and the
    // rounding is applied per term rather than to the sum.
    let bytes: u128 = 64u128 << 30;
    let expected = (bytes * 1_000_000_000_000).div_ceil(100_000_000_000)
        + (bytes * 1_000_000_000_000).div_ceil(3_000_000_000_000);
    assert_eq!(
        c.value as u128, expected,
        "DRAM->HBM should charge DRAM's 100 GB/s and HBM's 3 TB/s"
    );
}

/// The model converts between rate dimensions only when the clock is declared.
///
/// `MACHINE` mixes a `B/s` L2 with a `B/cyc` SMEM. Converting between them needs a clock the
/// declarations do not carry, so the only correct answer is "not derivable" -- and it must not be
/// a number, because a number here would silently add seconds to cycles. The fleet is exactly this
/// shape, so this is the live case rather than a hypothetical.
#[test]
fn mixed_rate_units_need_a_declared_clock() {
    // MACHINE declares SMEM in B/cyc with no `clock:`, so the L2(B/s) + SMEM(B/cyc) path cannot be
    // summed. Refusing beats inventing a frequency -- that is PREDICTIONS.md decision 5.
    let m = hierarchy(MACHINE);
    let h = MemoryHierarchy::build(m.memories.iter());
    assert_eq!(
        h.derived_transfer_cost(&space("L2"), &space("SMEM"), 16 * 1024),
        None,
        "a path mixing B/s and B/cyc with no declared clock must not produce a summed cost"
    );

    // Declare the clock and the same path converts exactly, in picoseconds.
    let clocked = hierarchy(
        "
Memory HBM { capacity: 40 GiB, bandwidth: 3 TB/s }
Memory L2 { within: Memory::HBM, capacity: 50 MiB, bandwidth: 12 TB/s }
Memory SMEM { within: Memory::L2, capacity: 228 KiB, bandwidth: 128 B/cyc, clock: 1.98 GHz, \
granule: 1 KiB, scope: sm }
fn main() -> i32 { return 0; }
",
    );
    let hc = MemoryHierarchy::build(clocked.memories.iter());
    let c = hc
        .derived_transfer_cost(&space("L2"), &space("SMEM"), 16 * 1024)
        .expect("a declared clock makes the mixed path derivable");
    assert_eq!(c.per, RatePer::Second, "a converted path is reported in ps");
    // SMEM: 16384/128 = 128 cycles, converted at 1.98 GHz. L2: 16384 B at 12 TB/s.
    // Spelled as arithmetic rather than a constant so the conversion is auditable, and rounded up
    // per term the way the model does it — 128 cycles is 64646.46 ps, and a partial picosecond is
    // still spent.
    let smem_ps = (128u128 * 1_000_000_000_000).div_ceil(1_980_000_000);
    let l2_ps = (16_384u128 * 1_000_000_000_000).div_ceil(12_000_000_000_000);
    assert_eq!(c.value as u128, smem_ps + l2_ps);
}
