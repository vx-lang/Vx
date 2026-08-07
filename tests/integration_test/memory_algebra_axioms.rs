//===- memory_algebra_axioms.rs - Vx Compiler ------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
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
// Additivity is the sharpest example. "Path cost = sum of leg costs" is an *axiom*, not a
// prediction: copy engines and TMA overlap staged legs, so real staged cost is expected to come in
// UNDER the sum, and quantifying that residual is experiment M2. What must hold now is that the
// model says what it means -- if the implementation is not even self-consistently additive, the M2
// residual would be measuring a bug rather than an overlap.
//
//===----------------------------------------------------------------------===//

use vxc::hir::memory::{granule_round, MemoryHierarchy};
use vxc::syntax::MemorySpace;

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

/// **Axiom 1: additivity.** The cost of a staged path is the sum of its legs.
///
/// Stated over a chain where every hop is a time rate, so the legs are commensurable. This is the
/// axiom M2 exists to violate on real hardware (overlap), so a failure *here* means the model does
/// not agree with itself, which is a different and much worse thing.
#[test]
fn path_cost_is_the_sum_of_its_legs() {
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
        assert_eq!(
            whole.value, legs,
            "additivity fails at {bytes} bytes: DRAM->SMEM is {} but its legs sum to {legs}",
            whole.value
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

    // A parent->child move is costed at the CHILD's bandwidth alone, because the nearest common
    // ancestor -- here the parent itself -- is the shared reservoir and contributes no bandwidth
    // term. So 64 GiB from DRAM into HBM is 64 GiB / 3 TB/s = 2.29e10 ps, NOT 64 GiB / 100 GB/s.
    //
    // Worth stating explicitly because it is easy to expect the other answer, and because it has a
    // consequence the study should know: an enclosing space's declared `bandwidth:` never
    // participates in a move into or out of the space it encloses. That is defensible as a model
    // (you fill the inner space at the inner space's rate) but it is a modelling choice, not a
    // physical law, and it is precisely why a host<->device link needs its bandwidth declared on
    // the *edge* rather than inferred from the memories it connects.
    let c = h
        .derived_transfer_cost(&space("DRAM"), &space("HBM"), 64 << 30)
        .unwrap();
    // `div_ceil`, not truncating division: a partial unit of time is still spent, so the roofline
    // rounds up. (The exact quotient here is …245.33, so the two differ by one and a truncating
    // expectation would fail.)
    let expected = ((64u128 << 30) * 1_000_000_000_000).div_ceil(3_000_000_000_000);
    assert_eq!(
        c.value as u128, expected,
        "64 GiB into HBM should be costed at HBM's 3 TB/s"
    );
}

/// The model refuses to add costs of different dimension.
///
/// `MACHINE` mixes a `B/s` L2 with a `B/cyc` SMEM. Converting between them needs a clock the
/// declarations do not carry, so the only correct answer is "not derivable" -- and it must not be
/// a number, because a number here would silently add seconds to cycles. The fleet is exactly this
/// shape, so this is the live case rather than a hypothetical.
#[test]
fn mixed_rate_units_are_refused_not_added() {
    let m = hierarchy(MACHINE);
    let h = MemoryHierarchy::build(m.memories.iter());
    assert_eq!(
        h.derived_transfer_cost(&space("HBM"), &space("SMEM"), 16 * 1024),
        None,
        "a path mixing B/s and B/cyc must not produce a summed cost"
    );
    // Each single-unit hop on its own is still derivable, so the `None` above is about the mixing
    // and not about a missing declaration.
    assert!(h
        .derived_transfer_cost(&space("HBM"), &space("L2"), 16 * 1024)
        .is_some());
    assert!(h
        .derived_transfer_cost(&space("L2"), &space("SMEM"), 16 * 1024)
        .is_some());
}
