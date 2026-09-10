//===- report.rs - Vx Compiler ---------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// What the memory-algebra analysis found, in the shape a consumer reads it in: resident sets,
// transfer routes and their costs, and counted traffic.
//
// These are the checker's output, not the checker's working state, which is why they live outside
// hir/. --diagnostics-json is a report format, and a report format that imports the type checker's
// internals cannot change without the checker changing too.
//
//===----------------------------------------------------------------------===//

/// One memory space's working set for a function: the granule-rounded sum of the tiles placed
/// there, against the space's declared capacity. Recorded for *every* placed space, not only
/// those that overflow — an admitted program emits no capacity diagnostic, so this is the only
/// place its resident total appears. Downstream that total is what turns a capacity verdict into
/// an engine's required memory utilization (#285). See [`crate::hir::env::TypeChecker::resident_sets`].
#[derive(Debug, Clone, PartialEq)]
pub struct ResidentSet {
    pub space: crate::syntax::MemorySpace,
    /// Granule-rounded sum of tiles placed in this space by one function.
    pub total_bytes: u64,
    pub capacity_bytes: u64,
    pub tiles: usize,
    /// Whether the space is declared `overcommit` (an overflow is W1028, not E6010).
    pub overcommit: bool,
}

/// A resolved `transfer`: the chain of memory spaces the value actually moves through and the
/// declared cost of each hop. A direct transfer has one hop; a multi-hop route (synthesized when
/// no direct edge exists) has one per staging step. `derived_cost` is the bandwidth-roofline cost
/// when the declarations make it computable, which is what the emitted `vx.transfer` carries.
/// See [`crate::hir::env::TypeChecker::staging_routes`] (#282).
#[derive(Debug, Clone, PartialEq)]
pub struct StagingRoute {
    /// The spaces traversed, source first: `[CPU_DRAM, HBM3e, SMEM]`.
    pub path: Vec<crate::syntax::MemorySpace>,
    /// Per-edge declared cost, one shorter than `path`; `None` for an edge with no declared cost.
    pub edge_costs: Vec<Option<u32>>,
    /// The cost graph's total for the whole route.
    pub total_cost: u32,
    /// Bandwidth-derived (roofline) cost, when the declarations supply bandwidths.
    ///
    /// `u64`, not `u32`: a picosecond-resolution time cost overflows `u32` at 4.3 ms, which a
    /// multi-gigabyte host transfer exceeds, and a saturated cost would read as a slow transfer
    /// rather than a missing one.
    pub derived_cost: Option<u64>,
    /// Bytes this transfer moves, when the shape is statically known.
    ///
    /// Without it a harvested cost cannot be interpreted: "260064 ps" is not a prediction unless
    /// the size it is a cost *of* travels with it, and comparing predicted against measured needs
    /// both. `None` for a dynamically-shaped tensor, where no cost is derivable either.
    pub bytes: Option<u64>,
    /// Which cost source produced `derived_cost`. The two are mutually exclusive per edge (E6013),
    /// so this names the one that applied rather than a precedence winner — and it is what tells a
    /// consumer whether a residual is attributable to a declared link figure or to an inferred
    /// containment roofline.
    pub cost_source: Option<CostSource>,
    /// The unit `derived_cost` is in — cycles, or picoseconds. Carried rather than dropped because
    /// one program's routes legitimately mix them: an on-die hop declared `B/cyc` and a host link
    /// declared `GB/s` produce costs of different *dimension*, and a harvested prediction that does
    /// not say which is not a prediction. (The fleet's `HBM->L2` and `L2->SMEM` are exactly this
    /// pair.)
    pub derived_unit: Option<crate::syntax::RatePer>,
    /// Which composition law priced a containment route: `sum` or `bottleneck`.
    ///
    /// `None` for a link-rate hop, which has one leg and so composes nothing. Carried because the
    /// two laws differ by ~2x on a multi-hop walk and a harvested prediction that does not say
    /// which one applied cannot be re-scored later -- the same reason `derived_unit` is carried.
    pub composition: Option<crate::syntax::Crossing>,
    /// Bytes moved per space, derived from code (#353 A4). `None` when the movement cannot be
    /// counted statically, in which case `traffic_absent_reason` says why -- a guess here would
    /// be indistinguishable from a measurement in the record a campaign harvests, which is the
    /// same reason `derived_cost` is `None` rather than 0 when no bandwidth is declared.
    pub traffic: Option<Traffic>,
    /// Why `traffic` is `None`. `None` when traffic is present.
    pub traffic_absent_reason: Option<String>,
}

/// Bytes read and written against one placed buffer by a `spawn` region (#353 A4 T4).
///
/// Per BUFFER, not just per space, because the amplification this stage exists to show is a
/// fact about a particular tensor: in an attention kernel K and V are re-read on every query
/// iteration while Q is read once per query. Summed into their shared space those three
/// become one number that hides which of them is the problem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferTraffic {
    pub buffer: String,
    pub space: crate::syntax::MemorySpace,
    pub read_bytes: u64,
    pub written_bytes: u64,
}

/// What one `spawn on(...)` region moves, counted from its indexed accesses and the static
/// trip counts of the loops around them (#353 A4 T4).
///
/// The transfer-hop counts say what it costs to GET a tile to a space. This says what the
/// kernel then does with it, which is where re-reading lives: a tile staged once and read
/// sixteen times is sixteen reads, and no declared edge cost can say so because the movement
/// across the edge happened exactly once.
#[derive(Debug, Clone, PartialEq)]
pub struct SpawnRegionTraffic {
    /// The function the `spawn` appears in.
    pub function: String,
    /// The topology the region runs on, as its display name.
    pub topology: String,
    /// Aggregate per space, or `None` when the region could not be counted exactly.
    pub traffic: Option<Traffic>,
    /// Per placed buffer, sorted by name. Empty when `traffic` is `None`.
    pub by_buffer: Vec<BufferTraffic>,
    /// Why `traffic` is `None`; `None` when traffic is present. Never both absent.
    pub traffic_absent_reason: Option<String>,
}

/// Bytes read and written against ONE memory space by a single transfer hop (#353 A4).
///
/// Read and written are kept apart because they are different facts about the hardware: a
/// space's read bandwidth and its write bandwidth are separate figures, and a plan that
/// re-reads the same tile is wasteful in a way a combined total cannot show.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpaceTraffic {
    pub space: crate::syntax::MemorySpace,
    pub read_bytes: u64,
    pub written_bytes: u64,
}

/// What the traffic figures were derived FROM. The distinction matters to a consumer for the
/// same reason `CostSource` does: a count read off a body is evidence about the code, while a
/// count asserted of the builtin copy is evidence about this compiler's own lowering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrafficSource {
    /// The builtin copy: one read of the whole tile, one write of it.
    BuiltinCopy,
    /// Counted from a user `impl transfer` body's `raw::` calls and static loop bounds.
    LoweringBody,
    /// Counted from the indexed reads and writes of a `spawn` region's static loop nest
    /// (#353 A4 T4). Appears on `spawn_regions` records, never on a transfer route: a route
    /// is one hop across one edge, while this is what a kernel does once the tile has
    /// arrived.
    SpawnRegion,
}

impl TrafficSource {
    pub fn as_str(self) -> &'static str {
        match self {
            TrafficSource::BuiltinCopy => "builtin_copy",
            TrafficSource::LoweringBody => "lowering_body",
            TrafficSource::SpawnRegion => "spawn_region",
        }
    }
}

/// The data movement one transfer hop performs, per space.
///
/// This is the *derived* half of "cost is derived, not declared" (#353 A4): bytes counted
/// from code, carrying no bandwidth, no time, and no opinion about how long they take. The
/// time model consumes these; it does not produce them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Traffic {
    pub per_space: Vec<SpaceTraffic>,
    pub source: TrafficSource,
    /// `false` when a branch forced a per-space maximum rather than a sum — the figures are
    /// then an upper bound on one execution, not the count of one. Anything less certain than
    /// a bound is not reported at all; see `StagingRoute::traffic_absent_reason`.
    pub exact: bool,
}

/// Where a hop's predicted cost came from. Exactly one applies per edge (E6013).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CostSource {
    /// A bandwidth declared on the link itself (`transfer A -> B : 64 GB/s`). Used where the
    /// endpoints do not nest, so containment can derive nothing — the host↔device seam.
    LinkRate,
    /// The roofline derived from the endpoints' own `bandwidth:` figures along the `within:` tree.
    Containment,
}

impl CostSource {
    pub fn as_str(self) -> &'static str {
        match self {
            CostSource::LinkRate => "link_rate",
            CostSource::Containment => "containment",
        }
    }
}
