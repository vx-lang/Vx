//===- traffic_test.rs - derived traffic, counted from code ---------------===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// "Cost is derived, not declared" (#353 A4), made checkable. Every figure here
// is counted from a body's own `raw::` calls and static loop bounds -- nothing
// in these programs declares a byte count anywhere.
//
// Two tests carry the weight, and they are different KINDS of claim:
//
//   * the calibration: a faithful copy's traffic must EQUAL the bytes it moves.
//     If that ever fails the counter is simply wrong, and every other number it
//     produces is worthless. It has already earned its place -- it caught the
//     env's signature clone stripping lowering bodies, which made the count come
//     back as a confident zero.
//
//   * the payoff: a lowering that computes the identical answer while reading
//     the source twice must REPORT twice the reads. No declared edge cost can
//     tell those two programs apart -- same spaces, same bandwidths, same tile,
//     same result -- because the waste is in the plan, not the edge.
//
// Through the binary and its `--diagnostics-json`, because that record is the
// artifact a measurement campaign harvests; testing the internal struct would
// not catch a schema mistake that makes the numbers unreadable.
//
//===----------------------------------------------------------------------===//

use std::path::{Path, PathBuf};
use std::process::Command;

fn corpus(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/backend/pass")
        .join(name)
}

/// Compile with `--diagnostics-json` and return the record.
fn record(name: &str) -> String {
    let out_dir = std::env::temp_dir().join(format!("vx_traffic_{}", std::process::id()));
    std::fs::create_dir_all(&out_dir).expect("temp dir");
    let json = out_dir.join(format!("{name}.json"));
    let out = Command::new(env!("CARGO_BIN_EXE_vxc"))
        .arg(corpus(name))
        .arg("--diagnostics-json")
        .arg(&json)
        .arg("--emit-mlir")
        .output()
        .unwrap_or_else(|e| panic!("could not run vxc: {e}"));
    assert!(
        out.status.success(),
        "compiling {name} failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    std::fs::read_to_string(&json).unwrap_or_else(|e| panic!("no record for {name}: {e}"))
}

/// Compile a source string written on the fly. Some traffic cases must never reach
/// the executed corpus -- an overflow probe needs a loop nest with 10^18 iterations,
/// which is instant to *count* and would hang a runner that ran it.
fn record_source(stem: &str, source: &str) -> String {
    let dir = std::env::temp_dir().join(format!("vx_traffic_src_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let vx = dir.join(format!("{stem}.vx"));
    std::fs::write(&vx, source).expect("write source");
    let json = dir.join(format!("{stem}.json"));
    let out = Command::new(env!("CARGO_BIN_EXE_vxc"))
        .arg(&vx)
        .arg("--diagnostics-json")
        .arg(&json)
        .arg("--emit-mlir")
        .output()
        .unwrap_or_else(|e| panic!("could not run vxc: {e}"));
    assert!(
        !String::from_utf8_lossy(&out.stderr).contains("Internal Error"),
        "the compiler crashed on {stem}:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    std::fs::read_to_string(&json).unwrap_or_else(|e| panic!("no record for {stem}: {e}"))
}

/// The corpus fixture with `move_tile`'s body swapped, so a traffic case is one
/// quoted body rather than a copy of the whole program.
fn with_body(body: &str) -> String {
    let base = std::fs::read_to_string(corpus("custom_topology_user_lowering.vx"))
        .expect("read base fixture");
    let start = base.find("  fn move_tile").expect("fixture has move_tile");
    let end = base[start..].find("\n}").expect("fixture body ends") + start;
    format!("{}{}{}", &base[..start], body, &base[end..])
}

/// The `GPU_HBM -> SMEM` route object, sliced out of the record. Hand-rolled
/// because the tree carries no JSON parser (the writer is hand-rolled too).
fn smem_route(rec: &str) -> String {
    let key = "\"path\": [\"GPU_HBM\", \"SMEM\"]";
    let start = rec
        .find(key)
        .unwrap_or_else(|| panic!("no GPU_HBM -> SMEM route in:\n{rec}"));
    let head = rec[..start].rfind('{').expect("route start");
    // Brace matching rather than a terminator guess: the route object contains a
    // nested array of objects, so the first `}` after the key is the end of a
    // per-space entry, not the route. (A guessed terminator is how this helper
    // failed the first time it ran.)
    let bytes = rec.as_bytes();
    let mut depth = 0usize;
    let mut tail = head;
    for (i, b) in bytes.iter().enumerate().skip(head) {
        match b {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    tail = i + 1;
                    break;
                }
            }
            _ => {}
        }
    }
    assert!(tail > head, "unterminated route object in:\n{rec}");
    rec[head..tail].to_string()
}

/// A faithful copy moves what it moves: derived traffic must equal `bytes`.
#[test]
fn a_faithful_lowering_reports_exactly_the_bytes_it_moves() {
    let route = smem_route(&record("custom_topology_user_lowering.vx"));
    assert!(
        route.contains("\"bytes\": 16"),
        "the 2x2 f32 tile is 16 bytes:\n{route}"
    );
    assert!(
        route.contains("\"traffic_source\": \"lowering_body\""),
        "the count must come from the user's body, not the builtin:\n{route}"
    );
    assert!(
        route.contains("\"traffic_exact\": true"),
        "a straight loop nest is countable exactly:\n{route}"
    );
    assert!(
        route.contains("{\"space\": \"GPU_HBM\", \"read_bytes\": 16, \"written_bytes\": 0}"),
        "a faithful copy reads the source once:\n{route}"
    );
    assert!(
        route.contains("{\"space\": \"SMEM\", \"read_bytes\": 0, \"written_bytes\": 16}"),
        "a faithful copy writes the destination once:\n{route}"
    );
}

/// The payoff: same answer, twice the reads, and the record says so. This is the
/// first thing in the compiler that sees a *plan* being wasteful.
#[test]
fn a_wasteful_lowering_reports_the_amplification() {
    let route = smem_route(&record("user_lowering_waste.vx"));
    assert!(
        route.contains("\"bytes\": 16"),
        "the tile is the same 16 bytes as the faithful version:\n{route}"
    );
    assert!(
        route.contains("{\"space\": \"GPU_HBM\", \"read_bytes\": 32, \"written_bytes\": 0}"),
        "reading the source twice must report 2x the reads -- this is the whole \
         point of deriving traffic instead of declaring cost:\n{route}"
    );
    // The write side is untouched: the waste is in the reads alone, which is
    // what makes the amplification legible rather than a general "it's bigger".
    assert!(
        route.contains("{\"space\": \"SMEM\", \"read_bytes\": 0, \"written_bytes\": 16}"),
        "the destination is still written exactly once:\n{route}"
    );
}

/// The builtin copy is counted too, and identically -- so a program that uses no
/// user lowering still gets a traffic figure, and the two sources are
/// comparable. Same tile, same numbers, different `traffic_source`.
#[test]
fn the_builtin_copy_is_counted_as_one_pass() {
    let route = smem_route(&record("custom_topology_device_image.vx"));
    assert!(
        route.contains("\"traffic_source\": \"builtin_copy\""),
        "no user lowering here, so the builtin is what moved the bytes:\n{route}"
    );
    assert!(
        route.contains("{\"space\": \"GPU_HBM\", \"read_bytes\": 16, \"written_bytes\": 0}")
            && route.contains("{\"space\": \"SMEM\", \"read_bytes\": 0, \"written_bytes\": 16}"),
        "the builtin reads the tile once and writes it once:\n{route}"
    );
}

/// The honesty case: a movement that cannot be counted statically is reported as
/// uncounted, with a reason -- never as a plausible number.
///
/// The cooperative copy this fixture uses is the shape the contract document
/// itself sketches, and it is correct code; its trip count just depends on how
/// many lanes the launch has. A count that quietly assumed one lane would be
/// indistinguishable, in a harvested record, from a measurement of a one-lane
/// machine. The same rule already governs `derived_cost`, which is null rather
/// than 0 when nothing declares a bandwidth.
#[test]
fn an_uncountable_body_reports_absence_with_a_reason() {
    let route = smem_route(&record("user_lowering_uncountable.vx"));
    assert!(
        route.contains("\"traffic\": null"),
        "an uncountable body must not produce numbers:\n{route}"
    );
    assert!(
        route.contains("\"traffic_source\": null") && route.contains("\"traffic_exact\": null"),
        "absent traffic carries no source and no exactness:\n{route}"
    );
    assert!(
        route.contains("\"traffic_absent_reason\": \"an unbounded `loop` runs an unknown"),
        "the record must say WHY it could not count:\n{route}"
    );
    // Still a real transfer of a known size -- absence of traffic is not absence
    // of the hop, and a consumer must be able to tell those apart.
    assert!(route.contains("\"bytes\": 16"), "{route}");
}

/// A count that overflows 64 bits is an ABSENCE, not a clamp.
///
/// Saturating published `u64::MAX` -- 58% of the true figure -- as an exact byte
/// count with no reason, on an admitted program. Worse, the accumulation was a
/// bare `+=`, so a debug build crashed the compiler outright and a release build
/// wrapped silently. The trip-count multiply next door already refused honestly;
/// the two sat one hop apart in the same expression.
#[test]
fn a_count_that_overflows_is_reported_as_absent_not_clamped() {
    let rec = record_source(
        "overflow",
        &with_body(
            "  fn move_tile(src: &Tensor<f32, [2, 2]>, dst: &mut Tensor<f32, [2, 2]>) -> i32 {\n\
             \x20   for i in 0..2000000 {\n\
             \x20     for j in 0..2000000 {\n\
             \x20       for k in 0..2000000 {\n\
             \x20         raw::store(dst, 0, raw::load(src, 0) + raw::load(src, 0));\n\
             \x20       }\n\
             \x20     }\n\
             \x20   }\n\
             \x20   raw::barrier();\n\
             \x20   return 0;\n\
             \x20 }",
        ),
    );
    let route = smem_route(&rec);
    assert!(
        route.contains("\"traffic\": null"),
        "an overflowed count must not be published:\n{route}"
    );
    assert!(
        route.contains("overflows a 64-bit counter"),
        "the record must say the count overflowed:\n{route}"
    );
    assert!(
        !route.contains("18446744073709551615"),
        "u64::MAX must never appear as a byte count:\n{route}"
    );
}

/// A sub-byte element type is refused rather than rounded up to a byte.
///
/// Rounding made a FAITHFUL i4 copy report twice its tile size -- the same ratio
/// `a_wasteful_lowering_reports_the_amplification` uses as proof that a plan is
/// wasteful. A figure that cannot be told apart from the thing it exists to
/// detect is worse than no figure.
#[test]
fn a_sub_byte_element_type_is_refused_rather_than_rounded() {
    let rec = record_source(
        "i4tile",
        &with_body(
            "  fn move_tile(src: &Tensor<i4, [2, 2]>, dst: &mut Tensor<i4, [2, 2]>) -> i32 {\n\
             \x20   for i in 0..raw::extent(src) {\n\
             \x20     raw::store(dst, i, raw::load(src, i));\n\
             \x20   }\n\
             \x20   raw::barrier();\n\
             \x20   return 0;\n\
             \x20 }",
        )
        .replace("Tensor<f32>([ 2, 2 ])", "Tensor<i4>([ 2, 2 ])"),
    );
    let route = smem_route(&rec);
    assert!(
        route.contains("\"traffic\": null") && route.contains("whole number of bytes"),
        "a sub-byte element type must refuse, not round:\n{route}"
    );
}

/// `if comptime` is decided at compile time, so only one arm exists in the emitted
/// code and the traffic is exactly known -- not the maximum of two arms, and not
/// flagged inexact.
#[test]
fn a_comptime_branch_is_counted_as_the_arm_that_survives() {
    let rec = record_source(
        "comptime",
        &with_body(
            "  fn move_tile(src: &Tensor<f32, [2, 2]>, dst: &mut Tensor<f32, [2, 2]>) -> i32 {\n\
             \x20   for i in 0..raw::extent(src) {\n\
             \x20     if comptime 1 < 2 {\n\
             \x20       raw::store(dst, i, raw::load(src, i));\n\
             \x20     } else {\n\
             \x20       raw::store(dst, i, raw::load(src, i) + raw::load(src, i));\n\
             \x20     }\n\
             \x20   }\n\
             \x20   raw::barrier();\n\
             \x20   return 0;\n\
             \x20 }",
        ),
    );
    let route = smem_route(&rec);
    assert!(
        route.contains("{\"space\": \"GPU_HBM\", \"read_bytes\": 16, \"written_bytes\": 0}"),
        "the dead arm must not be counted:\n{route}"
    );
    assert!(
        route.contains("\"traffic_exact\": true"),
        "a branch the compiler already resolved is exactly known:\n{route}"
    );
}

/// Two machines declare the same edge and move it differently. Which lowering
/// runs is decided by the topology in force at the site -- and the reported
/// traffic is what says which one ran, because the counter reads the body that
/// was chosen.
///
/// This program was a hard error until the `for Topology::X` clause existed
/// (Vx#353): a lowering was keyed on the edge pair alone, so the second
/// `Memory::GPU_HBM -> Memory::SMEM` implementation in a compilation was
/// refused no matter which part it was written for. That is the shape the
/// fleet directory is built to describe -- Ampere fills shared memory with
/// cp.async, Hopper with a different engine -- so it had to become writable.
///
/// The two bodies compute a bit-identical answer (x + x halved is exact in
/// IEEE for any value), so nothing but the byte count distinguishes them. A
/// selection bug that took the first candidate rather than the active
/// topology's would report 32 where this asserts 16.
fn two_machines(active: &str) -> String {
    let spaces = "\
Memory CPU_DRAM {}
Memory GPU_HBM {
  within: Memory::CPU_DRAM, capacity: 40 GiB, bandwidth: 3 TB/s, managed: cached
}
Memory SMEM {
  within: Memory::GPU_HBM, capacity: 228 KiB, bandwidth: 128 B/cyc,
  granule: 1 KiB, managed: explicit, scope: sm
}
Topology Thrifty {
  arch: nvptx64,
  memory: Memory::GPU_HBM,
  visible: [Memory::GPU_HBM, Memory::SMEM],
  transfer Memory::CPU_DRAM -> Memory::GPU_HBM : 63 GB/s,
  transfer Memory::GPU_HBM -> Memory::SMEM
}
Topology Wasteful {
  arch: nvptx64,
  memory: Memory::GPU_HBM,
  visible: [Memory::GPU_HBM, Memory::SMEM],
  transfer Memory::CPU_DRAM -> Memory::GPU_HBM : 63 GB/s,
  transfer Memory::GPU_HBM -> Memory::SMEM
}
impl Transfer<Memory::GPU_HBM, Memory::SMEM> for Topology::Thrifty {
  fn move_tile(src: &Tensor<f32, [2, 2]>, dst: &mut Tensor<f32, [2, 2]>) -> i32 {
    for i in 0..raw::extent(src) {
      raw::store(dst, i, raw::load(src, i));
    }
    raw::barrier();
    return 0;
  }
}
impl Transfer<Memory::GPU_HBM, Memory::SMEM> for Topology::Wasteful {
  fn move_tile(src: &Tensor<f32, [2, 2]>, dst: &mut Tensor<f32, [2, 2]>) -> i32 {
    for i in 0..raw::extent(src) {
      raw::store(dst, i, (raw::load(src, i) + raw::load(src, i)) * 0.5);
    }
    raw::barrier();
    return 0;
  }
}
";
    format!(
        "{spaces}
fn main() -> i32 {{
  let mut a = Tensor<f32>([ 2, 2 ]);
  let mut o = Tensor<f32>([ 2, 2 ]);
  for i in 0..2 {{
    for d in 0..2 {{
      a[i][d] = ((i + d) as f32) * 0.5;
      o[i][d] = 0.0;
    }}
  }}
  let ad = transfer(a, Memory::GPU_HBM);
  let mut od = transfer(o, Memory::GPU_HBM);
  spawn on(Topology::{active}) {{
    let tile = transfer(ad, Memory::SMEM);
    for i in 0..2 {{
      for d in 0..2 {{
        od[i][d] = tile[i][d] * 2.0;
      }}
    }}
  }}
  let home = transfer(od, Memory::CPU_DRAM);
  return 0;
}}
"
    )
}

#[test]
fn the_active_topology_picks_among_lowerings_for_one_edge() {
    let thrifty = smem_route(&record_source("sel_thrifty", &two_machines("Thrifty")));
    assert!(
        thrifty.contains("{\"space\": \"GPU_HBM\", \"read_bytes\": 16, \"written_bytes\": 0}"),
        "on Thrifty the single-read body must be the one counted:\n{thrifty}"
    );

    let wasteful = smem_route(&record_source("sel_wasteful", &two_machines("Wasteful")));
    assert!(
        wasteful.contains("{\"space\": \"GPU_HBM\", \"read_bytes\": 32, \"written_bytes\": 0}"),
        "on Wasteful the double-read body must be the one counted:\n{wasteful}"
    );

    // Same tile, same edge, same declared bandwidths, same answer. Only the
    // machine differs, and only the count reports it.
    assert!(
        thrifty.contains("{\"space\": \"SMEM\", \"read_bytes\": 0, \"written_bytes\": 16}")
            && wasteful.contains("{\"space\": \"SMEM\", \"read_bytes\": 0, \"written_bytes\": 16}"),
        "both machines write the same tile:\nthrifty {thrifty}\nwasteful {wasteful}"
    );
}

/// A machine that declares an edge but supplies NO lowering for it gets the
/// builtin copy -- never a peer machine's body.
///
/// HasImpl and NoImpl declare the same edge; only HasImpl implements it, with
/// a body that reads the source twice. The transfer runs inside
/// `spawn on(Topology::NoImpl)`. NoImpl chose not to supply a lowering, so the
/// builtin must move its tile: one read, `builtin_copy`.
///
/// The review found the opposite (Vx#353): the single-candidate fallback --
/// which exists so a host-driven transfer can find the one machine that owns a
/// device edge -- also fired when the site WAS on a machine, handing NoImpl
/// the only lowering in scope. HasImpl's body may use instructions NoImpl
/// never declared (that is what `copy_engine` gates), so borrowing it is not a
/// default, it is a miscompile. The fallback is now conditional on the active
/// topology not declaring the edge itself.
fn declined_machine() -> String {
    let two = two_machines("Thrifty");
    // Same program as the selection test, with the spawn moved onto a machine
    // that declares the edge and implements nothing: drop Thrifty's impl and
    // retarget the spawn at Thrifty, keeping Wasteful's double-read lowering
    // as the only candidate in scope.
    let start = two
        .find("impl Transfer<Memory::GPU_HBM, Memory::SMEM> for Topology::Thrifty")
        .expect("thrifty impl present");
    let end = two[start..].find("\n}\n").expect("impl ends") + start + 3;
    format!("{}{}", &two[..start], &two[end..])
}

#[test]
fn a_machine_without_a_lowering_gets_the_builtin_not_a_peers_body() {
    let route = smem_route(&record_source("declined", &declined_machine()));
    assert!(
        route.contains("\"traffic_source\": \"builtin_copy\""),
        "Thrifty supplied no lowering, so the builtin must move its tile -- \
         Wasteful's body belongs to Wasteful:\n{route}"
    );
    assert!(
        route.contains("{\"space\": \"GPU_HBM\", \"read_bytes\": 16, \"written_bytes\": 0}"),
        "the builtin reads the tile once:\n{route}"
    );
}
