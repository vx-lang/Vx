//===- diagnostics_json.rs - Vx Compiler ------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// The `--diagnostics-json` writer: one machine-readable admission verdict per compile, for
// harvesting an (config x SKU) accept/reject matrix without parsing human-readable prose.
// Hand-rolled because the tree carries no serde (only `bytemuck`, for the POD GID arrays).
//
//===----------------------------------------------------------------------===//

use crate::diagnostic::{Diagnostic, DiagnosticFacts, DiagnosticLevel, DiagnosticsVec};
use crate::report::{ResidentSet, SpawnRegionTraffic, StagingRoute};

/// The schema version embedded in every record. **Bump on any incompatible change** — a consumer
/// pins this, and the rental campaign (#289) will be reading artifacts produced over a period of
/// months, possibly by different compiler revisions.
///
/// # Schema (`vx-diagnostics-v1`)
///
/// ```json
/// {
///   "schema": "vx-diagnostics-v1",
///   "file": "program.vx",
///   "machine": "fleet/h100-sxm.vx",     // null when --machine was not given
///   "verdict": "admitted" | "rejected", // rejected iff error_count > 0
///   "error_count": 0,
///   "warning_count": 1,
///   "diagnostics": [
///     {
///       "level": "error" | "warning",
///       "code": "E6009",                // null for an uncoded diagnostic
///       "message": "...",
///       "line": 12, "column": 5,        // omitted when the diagnostic carries no span
///       "capacity": {                   // present on E6009 / E6010 / W1028
///         "space": "SMEM",
///         "required_bytes": 4194304,
///         "available_bytes": 1048576,
///         "margin_bytes": -3145728,     // available - required; negative means over
///         "tiles": 3                    // null for a single-tile verdict (E6009)
///       }
///     }
///   ],
///   "routes": [                         // the accept side: what an admitted program costs
///     {
///       "path": ["CPU_DRAM", "HBM3e"],
///       "edges": [{"from": "CPU_DRAM", "to": "HBM3e", "cost": 300}],
///       "total_cost": 300,              // ROUTE-SELECTION weight, not a prediction: it is
///                                       // unitless, size-independent, and 1 for an edge that
///                                       // declares no cost. Harvest `derived_cost` instead.
///       "bytes": 16384,                 // what moved; null for a dynamic shape
///       "derived_cost": 128,            // bandwidth roofline, null when not computable
///       "derived_unit": "cyc",          // "cyc" | "ps"; null iff derived_cost is null
///       "cost_source": "containment",   // "link_rate" | "containment"; null iff no cost
///       "composition": "sum",           // "sum" | "bottleneck"; null for a link-rate hop
///       "traffic": [                    // bytes DERIVED from code; null when uncountable
///         {"space": "GPU_HBM", "read_bytes": 16, "written_bytes": 0},
///         {"space": "SMEM",    "read_bytes": 0,  "written_bytes": 16}
///       ],
///       "traffic_source": "builtin_copy", // "builtin_copy" | "lowering_body"
///       "traffic_exact": true,          // false = a branch forced a per-space upper bound
///       "traffic_absent_reason": null   // non-null iff traffic is null; says why
///     }
///   ],
///   "resident_sets": [                  // working set per space, emitted even when admitted
///     {
///       "space": "HBM",
///       "total_bytes": 139653545984,
///       "capacity_bytes": 206158430208,
///       "utilization": 0.6774,          // total / capacity, rounded to 4 dp
///       "tiles": 3,
///       "overcommit": false
///     }
///   ],
///   "spawn_regions": [                  // what each kernel MOVES, counted from its own code
///     {
///       "function": "main",
///       "topology": "GPU[0]",
///       "traffic": [                    // per space; null when the region is uncountable
///         {"space": "GPU_HBM", "read_bytes": 608, "written_bytes": 224}
///       ],
///       "traffic_source": "spawn_region",
///       "traffic_exact": true,
///       "by_buffer": [                  // per placed tensor, sorted by name; [] when absent
///         {"buffer": "k", "space": "GPU_HBM", "read_bytes": 128, "written_bytes": 0}
///       ],
///       "traffic_absent_reason": null   // non-null iff traffic is null; says why
///     }
///   ]
/// }
/// ```
///
/// The `traffic_*` keys were added for #353 A4 and are **additive** in the same sense. They
/// carry bytes, never time: traffic is what the code moves, and what that costs is the time
/// model's separate (and calibrated) claim. `traffic` is null rather than zero when a movement
/// cannot be counted statically, because a harvested zero is indistinguishable from a
/// measurement of nothing.
///
/// `spawn_regions` was added for #353 A4 T4 and is **additive** likewise. It answers the
/// question the `routes` traffic cannot: a route says what it cost to GET a tile to a space,
/// while a region says what the kernel then does with it. Those differ by the loop nest, which
/// is where re-reading lives — an attention kernel re-reads K on every query iteration, so K's
/// reads exceed its footprint by a factor no edge cost can express, because the movement
/// across the edge happened exactly once. `traffic_source` is `"spawn_region"` here and never
/// on a route; conversely `"builtin_copy"`/`"lowering_body"` appear only on routes.
///
/// Three limits a consumer must know.
///
/// 1. Only PLACED tensors are counted (values carrying a `Pinned`/`Ref` type, which is what a
///    `transfer` produces): a scratch tile declared inside the kernel is not device traffic
///    anyone staged, and counting it would inflate the figure with kernel-local storage.
/// 2. The count is TOUCHES, not distinct elements — re-reading one element a thousand times is
///    a thousand reads, which is the whole point.
/// 3. It is PER LAUNCH, not per program run. One entry describes one `spawn` site, and says
///    what one execution of that region moves. A `spawn` inside a host loop still gets one
///    entry with one launch's bytes, because the host loop's own trip count is a property of
///    the caller and is not always static. A consumer totalling a program's device traffic
///    must multiply by how often each site runs; summing the entries as they stand
///    undercounts exactly by that factor.
///
/// `resident_sets` was added after the initial release of this schema. It is **additive** — a
/// consumer reading only the original keys is unaffected — so the version is deliberately not
/// bumped, per this constant's own rule of bumping on incompatible change. It exists because a
/// verdict alone does not carry the resident total: an admitted program emits no capacity
/// diagnostic, and the total is what a downstream consumer needs to compute the memory
/// utilization an engine must be given (#285).
pub const SCHEMA_VERSION: &str = "vx-diagnostics-v1";

/// Escape a string for a JSON string literal (RFC 8259): the two mandatory escapes, the standard
/// short forms, and `\u00XX` for the remaining control characters.
fn esc(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn opt_str(v: Option<&str>) -> String {
    match v {
        Some(s) => format!("\"{}\"", esc(s)),
        None => "null".to_string(),
    }
}

fn opt_num<T: std::fmt::Display>(v: Option<T>) -> String {
    match v {
        Some(n) => n.to_string(),
        None => "null".to_string(),
    }
}

fn diagnostic_json(d: &Diagnostic) -> String {
    let level = match d.level {
        DiagnosticLevel::Error => "error",
        DiagnosticLevel::Warning => "warning",
    };
    let code = d.code.map(|c| format!("{:?}", c));
    let mut fields = vec![
        format!("\"level\": \"{}\"", level),
        format!("\"code\": {}", opt_str(code.as_deref())),
        format!("\"message\": \"{}\"", esc(&d.message)),
    ];
    if let Some(sp) = &d.source_span {
        fields.push(format!("\"line\": {}", sp.line));
        fields.push(format!("\"column\": {}", sp.column));
    }
    if let Some(DiagnosticFacts::Capacity {
        space,
        required_bytes,
        available_bytes,
        tiles,
    }) = &d.facts
    {
        // Signed margin: negative is the amount by which the placement overflows. Computed here
        // rather than stored so it cannot disagree with the two operands.
        let margin = *available_bytes as i128 - *required_bytes as i128;
        fields.push(format!(
            "\"capacity\": {{\"space\": \"{}\", \"required_bytes\": {}, \"available_bytes\": {}, \
             \"margin_bytes\": {}, \"tiles\": {}}}",
            esc(space),
            required_bytes,
            available_bytes,
            margin,
            opt_num(*tiles)
        ));
    }
    format!("{{{}}}", fields.join(", "))
}

fn route_json(r: &StagingRoute) -> String {
    let path = r
        .path
        .iter()
        .map(|s| format!("\"{}\"", esc(&s.name())))
        .collect::<Vec<_>>()
        .join(", ");
    let edges = r
        .path
        .windows(2)
        .zip(r.edge_costs.iter())
        .map(|(pair, cost)| {
            format!(
                "{{\"from\": \"{}\", \"to\": \"{}\", \"cost\": {}}}",
                esc(&pair[0].name()),
                esc(&pair[1].name()),
                opt_num(*cost)
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    // The unit travels with the number. One program's routes legitimately mix dimensions -- an
    // on-die hop declared `B/cyc` yields cycles, a link declared `GB/s` yields picoseconds -- so a
    // bare integer is not a prediction. `null` unit iff `null` cost.
    let unit = match r.derived_unit {
        Some(crate::syntax::RatePer::Cycle) => "\"cyc\"",
        Some(crate::syntax::RatePer::Second) => "\"ps\"",
        None => "null",
    };
    let source = match r.cost_source {
        Some(s) => format!("\"{}\"", s.as_str()),
        None => "null".to_string(),
    };
    // Which composition law priced the walk. Named by what it DOES, not by the declaration that
    // selected it: a consumer re-scoring a harvested prediction needs "these legs were summed",
    // and `sequenced`/`streamed` is a statement about the hardware rather than the arithmetic.
    let composition = match r.composition {
        Some(crate::syntax::Crossing::Sequenced) => "\"sum\"",
        Some(crate::syntax::Crossing::Streamed) => "\"bottleneck\"",
        None => "null",
    };
    // Derived traffic (#353 A4). Present or explicitly absent-with-a-reason; never a
    // silent zero, which a consumer could not tell from a real count of nothing.
    let (traffic, traffic_source, traffic_exact) = match &r.traffic {
        Some(t) => (
            format!(
                "[{}]",
                t.per_space
                    .iter()
                    .map(space_traffic_json)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            format!("\"{}\"", t.source.as_str()),
            if t.exact { "true" } else { "false" }.to_string(),
        ),
        None => ("null".to_string(), "null".to_string(), "null".to_string()),
    };
    let traffic_absent_reason = match &r.traffic_absent_reason {
        Some(why) => format!("\"{}\"", esc(why)),
        None => "null".to_string(),
    };
    format!(
        "{{\"path\": [{}], \"edges\": [{}], \"total_cost\": {}, \"bytes\": {}, \
         \"derived_cost\": {}, \"derived_unit\": {}, \"cost_source\": {}, \
         \"composition\": {}, \"traffic\": {}, \"traffic_source\": {}, \
         \"traffic_exact\": {}, \"traffic_absent_reason\": {}}}",
        path,
        edges,
        r.total_cost,
        opt_num(r.bytes),
        opt_num(r.derived_cost),
        unit,
        source,
        composition,
        traffic,
        traffic_source,
        traffic_exact,
        traffic_absent_reason
    )
}

/// One space's read/written pair. Shared by the transfer-route records and the spawn-region
/// records so a consumer parses one shape, not two that happen to look alike today.
fn space_traffic_json(st: &crate::report::SpaceTraffic) -> String {
    format!(
        "{{\"space\": \"{}\", \"read_bytes\": {}, \"written_bytes\": {}}}",
        esc(&st.space.name()),
        st.read_bytes,
        st.written_bytes
    )
}

fn resident_set_json(r: &ResidentSet) -> String {
    // Utilization is derived here rather than stored so it cannot disagree with its operands.
    // Rounded to 4 dp: enough to distinguish campaign cells, short enough to read.
    let util = if r.capacity_bytes == 0 {
        0.0
    } else {
        (r.total_bytes as f64 / r.capacity_bytes as f64 * 10_000.0).round() / 10_000.0
    };
    format!(
        "{{\"space\": \"{}\", \"total_bytes\": {}, \"capacity_bytes\": {}, \"utilization\": {}, \
         \"tiles\": {}, \"overcommit\": {}}}",
        esc(&r.space.name()),
        r.total_bytes,
        r.capacity_bytes,
        util,
        r.tiles,
        r.overcommit
    )
}

/// One `spawn` region's derived traffic (#353 A4 T4).
///
/// `by_buffer` is carried beside the per-space aggregate because the amplification this
/// record exists to show is a fact about a particular tensor: summed into their shared space,
/// a re-read K and a read-once Q become one number that hides which of them is the problem.
/// Both are omitted (and `traffic_absent_reason` is non-null) when the region could not be
/// counted -- an uncountable kernel must not be readable as an efficient one.
fn spawn_region_json(s: &SpawnRegionTraffic) -> String {
    let (per_space, source, exact) = match &s.traffic {
        Some(t) => (
            format!(
                "[{}]",
                t.per_space
                    .iter()
                    .map(space_traffic_json)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            format!("\"{}\"", t.source.as_str()),
            t.exact.to_string(),
        ),
        None => ("null".to_string(), "null".to_string(), "null".to_string()),
    };
    let by_buffer = s
        .by_buffer
        .iter()
        .map(|b| {
            format!(
                "{{\"buffer\": \"{}\", \"space\": \"{}\", \"read_bytes\": {}, \
                 \"written_bytes\": {}}}",
                esc(&b.buffer),
                esc(&b.space.name()),
                b.read_bytes,
                b.written_bytes
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    let reason = match &s.traffic_absent_reason {
        Some(r) => format!("\"{}\"", esc(r)),
        None => "null".to_string(),
    };
    format!(
        "{{\"function\": \"{}\", \"topology\": \"{}\", \"traffic\": {}, \"traffic_source\": {}, \
         \"traffic_exact\": {}, \"by_buffer\": [{}], \"traffic_absent_reason\": {}}}",
        esc(&s.function),
        esc(&s.topology),
        per_space,
        source,
        exact,
        by_buffer,
        reason
    )
}

/// Render one compile's admission verdict. `file` is the program compiled and `machine` the
/// `--machine` model it was admitted against, so a harvested record identifies its own
/// (config, SKU) cell without the caller having to correlate it back to the invocation.
pub fn render(
    diagnostics: &DiagnosticsVec,
    routes: &[StagingRoute],
    residents: &[ResidentSet],
    spawn_regions: &[SpawnRegionTraffic],
    file: &str,
    machine: Option<&str>,
) -> String {
    let error_count = diagnostics.error_count();
    let warning_count = diagnostics
        .iter()
        .filter(|d| d.level == DiagnosticLevel::Warning)
        .count();
    let verdict = if error_count > 0 {
        "rejected"
    } else {
        "admitted"
    };
    let diags = diagnostics
        .iter()
        .map(diagnostic_json)
        .collect::<Vec<_>>()
        .join(",\n    ");
    let routes_json = routes
        .iter()
        .map(route_json)
        .collect::<Vec<_>>()
        .join(",\n    ");
    let residents_json = residents
        .iter()
        .map(resident_set_json)
        .collect::<Vec<_>>()
        .join(",\n    ");
    let spawn_json = spawn_regions
        .iter()
        .map(spawn_region_json)
        .collect::<Vec<_>>()
        .join(",\n    ");
    format!(
        "{{\n  \"schema\": \"{}\",\n  \"file\": \"{}\",\n  \"machine\": {},\n  \"verdict\": \
         \"{}\",\n  \"error_count\": {},\n  \"warning_count\": {},\n  \"diagnostics\": \
         [{}{}{}],\n  \"routes\": [{}{}{}],\n  \"resident_sets\": [{}{}{}],\n  \
         \"spawn_regions\": [{}{}{}]\n}}",
        SCHEMA_VERSION,
        esc(file),
        opt_str(machine),
        verdict,
        error_count,
        warning_count,
        if diags.is_empty() { "" } else { "\n    " },
        diags,
        if diags.is_empty() { "" } else { "\n  " },
        if routes_json.is_empty() { "" } else { "\n    " },
        routes_json,
        if routes_json.is_empty() { "" } else { "\n  " },
        if residents_json.is_empty() {
            ""
        } else {
            "\n    "
        },
        residents_json,
        if residents_json.is_empty() {
            ""
        } else {
            "\n  "
        },
        if spawn_json.is_empty() { "" } else { "\n    " },
        spawn_json,
        if spawn_json.is_empty() { "" } else { "\n  " },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diagnostic::DiagnosticCode;

    #[test]
    fn escapes_json_string_metacharacters() {
        assert_eq!(esc(r#"a"b\c"#), r#"a\"b\\c"#);
        assert_eq!(esc("line\nnext\ttab"), "line\\nnext\\ttab");
        // A control character that has no short form becomes \u00XX.
        assert_eq!(esc("\u{1}"), "\\u0001");
    }

    /// A reject carries the numbers a campaign keys on: code, space, required/available, and a
    /// negative margin -- not prose to be parsed back.
    #[test]
    fn reject_record_carries_capacity_facts() {
        let mut diags = DiagnosticsVec::default();
        diags
            .error_with_code(DiagnosticCode::E6009, "tile too big", None)
            .facts = Some(DiagnosticFacts::Capacity {
            space: "SMEM".to_string(),
            required_bytes: 4194304,
            available_bytes: 1048576,
            tiles: None,
        });
        let out = render(&diags, &[], &[], &[], "prog.vx", Some("fleet/h100.vx"));
        assert!(out.contains("\"schema\": \"vx-diagnostics-v1\""), "{out}");
        assert!(out.contains("\"verdict\": \"rejected\""), "{out}");
        assert!(out.contains("\"code\": \"E6009\""), "{out}");
        assert!(out.contains("\"space\": \"SMEM\""), "{out}");
        assert!(out.contains("\"required_bytes\": 4194304"), "{out}");
        assert!(out.contains("\"available_bytes\": 1048576"), "{out}");
        assert!(out.contains("\"margin_bytes\": -3145728"), "{out}");
        assert!(out.contains("\"tiles\": null"), "{out}");
        assert!(out.contains("\"machine\": \"fleet/h100.vx\""), "{out}");
    }

    /// A working-set reject (E6010) reports its tile count; the overcommit warning (W1028)
    /// carries the same facts but leaves the verdict admitted.
    #[test]
    fn overcommit_warning_is_admitted_but_reports_its_numbers() {
        let mut diags = DiagnosticsVec::default();
        diags
            .warn(DiagnosticCode::W1028, "working set over capacity", None)
            .facts = Some(DiagnosticFacts::Capacity {
            space: "TMEM".to_string(),
            required_bytes: 300,
            available_bytes: 256,
            tiles: Some(3),
        });
        let out = render(&diags, &[], &[], &[], "prog.vx", None);
        assert!(out.contains("\"verdict\": \"admitted\""), "{out}");
        assert!(out.contains("\"warning_count\": 1"), "{out}");
        assert!(out.contains("\"code\": \"W1028\""), "{out}");
        assert!(out.contains("\"tiles\": 3"), "{out}");
        assert!(out.contains("\"margin_bytes\": -44"), "{out}");
        assert!(out.contains("\"machine\": null"), "{out}");
    }

    /// An admitted program's evidence is its routes: the staging chain and per-edge costs.
    #[test]
    fn accept_record_carries_route_and_edge_costs() {
        use crate::syntax::MemorySpace;
        let route = StagingRoute {
            path: vec![
                MemorySpace::CPUDRAM,
                MemorySpace::Custom("HBM3e".into()),
                MemorySpace::Custom("SMEM".into()),
            ],
            edge_costs: vec![Some(300), Some(40)],
            total_cost: 340,
            bytes: Some(16384),
            cost_source: Some(crate::report::CostSource::Containment),
            derived_cost: Some(128),
            derived_unit: Some(crate::syntax::RatePer::Cycle),
            composition: Some(crate::syntax::Crossing::Sequenced),
            traffic: Some(crate::report::Traffic {
                per_space: vec![
                    crate::report::SpaceTraffic {
                        space: MemorySpace::CPUDRAM,
                        read_bytes: 16384,
                        written_bytes: 0,
                    },
                    crate::report::SpaceTraffic {
                        space: MemorySpace::Custom("SMEM".into()),
                        read_bytes: 0,
                        written_bytes: 16384,
                    },
                ],
                source: crate::report::TrafficSource::BuiltinCopy,
                exact: true,
            }),
            traffic_absent_reason: None,
        };
        let out = render(
            &DiagnosticsVec::default(),
            &[route],
            &[],
            &[],
            "prog.vx",
            None,
        );
        // Traffic is bytes, per space, with the side it moved them on (#353 A4).
        assert!(
            out.contains("{\"space\": \"CPU_DRAM\", \"read_bytes\": 16384, \"written_bytes\": 0}"),
            "{out}"
        );
        assert!(
            out.contains("{\"space\": \"SMEM\", \"read_bytes\": 0, \"written_bytes\": 16384}"),
            "{out}"
        );
        assert!(
            out.contains("\"traffic_source\": \"builtin_copy\""),
            "{out}"
        );
        assert!(out.contains("\"traffic_exact\": true"), "{out}");
        assert!(out.contains("\"traffic_absent_reason\": null"), "{out}");
        // A containment route must say which law priced it: the two differ by ~2x on a multi-hop
        // walk, so a harvested prediction that omits it cannot be re-scored (vx-review#26).
        assert!(out.contains("\"composition\": \"sum\""), "{out}");
        assert!(out.contains("\"verdict\": \"admitted\""), "{out}");
        assert!(out.contains("\"error_count\": 0"), "{out}");
        assert!(
            out.contains("\"path\": [\"CPU_DRAM\", \"HBM3e\", \"SMEM\"]"),
            "{out}"
        );
        assert!(
            out.contains("{\"from\": \"CPU_DRAM\", \"to\": \"HBM3e\", \"cost\": 300}"),
            "{out}"
        );
        assert!(
            out.contains("{\"from\": \"HBM3e\", \"to\": \"SMEM\", \"cost\": 40}"),
            "{out}"
        );
        assert!(out.contains("\"total_cost\": 340"), "{out}");
        assert!(out.contains("\"derived_cost\": 128"), "{out}");
    }

    /// The empty case still parses as an object with both arrays present.
    #[test]
    fn clean_compile_renders_empty_arrays() {
        let out = render(&DiagnosticsVec::default(), &[], &[], &[], "prog.vx", None);
        assert!(out.contains("\"diagnostics\": []"), "{out}");
        assert!(out.contains("\"routes\": []"), "{out}");
        assert!(out.contains("\"verdict\": \"admitted\""), "{out}");
    }
}
