//===- traffic_test.rs - derived traffic, counted from code ---------------===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
    let src = with_body(
        "  fn move_tile(src: &Tensor<i4, [2, 2]>, dst: &mut Tensor<i4, [2, 2]>) -> i32 {\n\
         \x20   for i in 0..raw::extent(src) {\n\
         \x20     raw::store(dst, i, raw::load(src, i));\n\
         \x20   }\n\
         \x20   raw::barrier();\n\
         \x20   return 0;\n\
         \x20 }",
    );
    // The fixture's own tensors have to be retyped too -- the substituted body borrows them.
    // Checked rather than replaced blind: a respelling in the corpus file would otherwise leave
    // this test measuring f32 while claiming to measure i4, which is exactly what happened when
    // the constructor syntax changed (Vx#429).
    const F32_TILE: &str = "Tensor<f32, [2, 2]>";
    assert!(
        src.contains(F32_TILE),
        "the fixture no longer spells its tensors `{F32_TILE}`, so this test would measure f32"
    );
    let rec = record_source("i4tile", &src.replace(F32_TILE, "Tensor<i4, [2, 2]>"));
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

// ---------------------------------------------------------------------------
// Spawn-region traffic (#353 A4 T4): what a KERNEL moves, as opposed to what it
// cost to stage the tile it moves. The two are different questions and only one
// of them is reachable from an edge cost.
// ---------------------------------------------------------------------------

/// The `spawn_regions` record for `func`, sliced out by brace matching.
///
/// Hand-rolled for the same reason `smem_route` is: the record nests arrays of
/// objects, so the first `}` after the key ends a per-buffer entry, not the
/// region. A guessed terminator is how that helper failed the first time.
fn spawn_region(rec: &str, func: &str) -> String {
    let key = format!("\"function\": \"{func}\"");
    let start = rec
        .find(&key)
        .unwrap_or_else(|| panic!("no spawn region for {func} in:\n{rec}"));
    let head = rec[..start].rfind('{').expect("region start");
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
    assert!(tail > head, "unterminated region object in:\n{rec}");
    rec[head..tail].to_string()
}

/// THE READOUT. FlashAttention's kernel re-reads K and V on every query
/// iteration, and this is that fact as a number nobody declared.
///
/// `flash_attention_placed.vx` stages Q[2,4], K[4,4], V[4,4], O[2,4] into
/// GPU_HBM and runs the online-softmax loop in one `spawn`. The arithmetic,
/// worked out by hand BEFORE this counter existed and confirmed by three
/// independent recomputations:
///
///   q[i][d]  in the dot loop      i2 * t2 * jj2 * d4 = 32 reads  = 128 B
///   k[j][d]  in the dot loop      i2 * t2 * jj2 * d4 = 32 reads  = 128 B
///   v[j][d]  in the accumulate    i2 * t2 * jj2 * d4 = 32 reads  = 128 B
///   o[i][d]  rescale  (r+w)       i2 * t2 * d4       = 16 each
///   o[i][d]  accumulate (r+w)     i2 * t2 * jj2 * d4 = 32 each
///   o[i][d]  final divide (r+w)   i2 * d4            =  8 each
///   -> o = 56 reads + 56 writes = 224 B + 224 B
///   aggregate GPU_HBM: 608 B read, 224 B written
///
/// The amplification is the point. K's footprint is 4*4*4 = 64 B and the kernel
/// reads 128 B of it: every key is fetched once per query. Q is worse -- 32 B
/// staged, 128 B read, a factor of 4. No declared edge cost can say any of this,
/// because each tensor crossed CPU_DRAM -> GPU_HBM exactly once; the re-reading
/// is a property of the loop nest, which lives entirely on the far side of the
/// transfer.
#[test]
fn the_flash_kernel_reports_its_k_and_v_re_reads() {
    let region = spawn_region(&record("flash_attention_placed.vx"), "main");
    assert!(
        region.contains("\"traffic_source\": \"spawn_region\""),
        "counted from the region's own accesses:\n{region}"
    );
    // K is staged once (64 B) and read twice over (128 B): re-read per query.
    assert!(
        region.contains(
            "{\"buffer\": \"k\", \"space\": \"GPU_HBM\", \"read_bytes\": 128, \
             \"written_bytes\": 0}"
        ),
        "K's reads are queries(2) x keys(4) x d(4) x 4 B = 128:\n{region}"
    );
    // V is read on the same schedule as K.
    assert!(
        region.contains(
            "{\"buffer\": \"v\", \"space\": \"GPU_HBM\", \"read_bytes\": 128, \
             \"written_bytes\": 0}"
        ),
        "V is re-read exactly as K is:\n{region}"
    );
    // Q: 32 B staged, 128 B read -- the largest amplification in the kernel.
    assert!(
        region.contains(
            "{\"buffer\": \"q\", \"space\": \"GPU_HBM\", \"read_bytes\": 128, \
             \"written_bytes\": 0}"
        ),
        "Q's row is re-read for every key block:\n{region}"
    );
    // O is the only tensor written, and it is read back as often as written --
    // the online-softmax rescale is a read-modify-write.
    assert!(
        region.contains(
            "{\"buffer\": \"o\", \"space\": \"GPU_HBM\", \"read_bytes\": 224, \
             \"written_bytes\": 224}"
        ),
        "O is rescaled and accumulated in place: 56 reads and 56 writes:\n{region}"
    );
    assert!(
        region.contains("{\"space\": \"GPU_HBM\", \"read_bytes\": 608, \"written_bytes\": 224}"),
        "aggregate over the four buffers:\n{region}"
    );
    assert!(
        region.contains("\"traffic_exact\": true"),
        "every loop bound is a literal and no branch touches a placed tensor:\n{region}"
    );
}

/// Calibration, the same shape T1 used and for the same reason: a count that
/// cannot be checked against an independently-known answer is not evidence.
///
/// One placed 2x2 f32 tile, read once per element under a literal nest and
/// written once. 4 elements x 4 B = 16 B each way, which is also exactly the
/// tile's footprint because nothing here is re-read. If this ever disagrees the
/// counter is simply wrong and every other figure it produces is worthless.
fn one_pass_kernel(body: &str) -> String {
    format!(
        "Memory CPU_DRAM {{}}
Memory GPU_HBM {{
  within: Memory::CPU_DRAM, capacity: 40 GiB, bandwidth: 3 TB/s, managed: cached
}}
Topology Dev {{
  arch: nvptx64,
  memory: Memory::GPU_HBM,
  visible: [Memory::GPU_HBM],
  transfer Memory::CPU_DRAM -> Memory::GPU_HBM : 63 GB/s
}}
fn main() -> i32 {{
  let mut a = Tensor<f32>([ 2, 2 ]);
  let mut o = Tensor<f32>([ 2, 2 ]);
  for i in 0..2 {{
    for d in 0..2 {{
      a[i][d] = 1.0;
      o[i][d] = 0.0;
    }}
  }}
  let ad = transfer(a, Memory::GPU_HBM);
  let mut od = transfer(o, Memory::GPU_HBM);
  spawn on(Topology::Dev) {{
{body}
  }}
  return 0;
}}
"
    )
}

#[test]
fn a_single_pass_kernel_reports_exactly_the_tile_it_touches() {
    let src = one_pass_kernel(
        "    for i in 0..2 {
      for d in 0..2 {
        od[i][d] = ad[i][d];
      }
    }",
    );
    let region = spawn_region(&record_source("region_calib", &src), "main");
    assert!(
        region.contains(
            "{\"buffer\": \"ad\", \"space\": \"GPU_HBM\", \"read_bytes\": 16, \
             \"written_bytes\": 0}"
        ),
        "4 elements x 4 B, read once each:\n{region}"
    );
    assert!(
        region.contains(
            "{\"buffer\": \"od\", \"space\": \"GPU_HBM\", \"read_bytes\": 0, \
             \"written_bytes\": 16}"
        ),
        "the destination is written, NOT read: an assignment's lhs is a store, and \
         booking it as a load would invent a read the program never performs:\n{region}"
    );
}

/// `+=` reads its destination as well as writing it. Written as a separate test
/// because `Statement::Assign` and `Statement::CompoundAssign` are distinct AST
/// nodes: a walker that handles only the first, or that treats both as
/// write-only, silently undercounts every accumulation in every kernel -- and a
/// silent undercount is indistinguishable from an efficient program.
#[test]
fn a_compound_assignment_counts_the_read_it_performs() {
    let src = one_pass_kernel(
        "    for i in 0..2 {
      for d in 0..2 {
        od[i][d] += ad[i][d];
      }
    }",
    );
    let region = spawn_region(&record_source("region_compound", &src), "main");
    assert!(
        region.contains(
            "{\"buffer\": \"od\", \"space\": \"GPU_HBM\", \"read_bytes\": 16, \
             \"written_bytes\": 16}"
        ),
        "`+=` is a read-modify-write, so the destination shows both:\n{region}"
    );
}

/// Re-reading is what the stage exists to detect, and it must show up as a
/// number larger than the footprint. Same tile, same result shape, one extra
/// pass over the source.
#[test]
fn a_kernel_that_re_reads_reports_more_than_the_tile_holds() {
    let src = one_pass_kernel(
        "    for pass in 0..3 {
      for i in 0..2 {
        for d in 0..2 {
          od[i][d] = ad[i][d];
        }
      }
    }",
    );
    let region = spawn_region(&record_source("region_reread", &src), "main");
    assert!(
        region.contains(
            "{\"buffer\": \"ad\", \"space\": \"GPU_HBM\", \"read_bytes\": 48, \
             \"written_bytes\": 0}"
        ),
        "three passes over a 16 B tile is 48 B of reads -- 3x the footprint, and \
         the tile crossed the edge exactly once:\n{region}"
    );
}

/// Scratch declared inside the kernel is not traffic against the device memory
/// anyone staged. `ts` is an ordinary `Tensor`, never transferred; the placement
/// rule keys on the TYPE carrying a placement, not on the ambient topology --
/// which inside `spawn on(Topology::Dev)` would have reported GPU_HBM for it and
/// inflated the figure with kernel-local storage.
#[test]
fn kernel_local_scratch_is_not_counted_as_device_traffic() {
    let src = one_pass_kernel(
        "    let mut ts = Tensor<f32>([ 2, 2 ]);
    for i in 0..2 {
      for d in 0..2 {
        ts[i][d] = ad[i][d];
        od[i][d] = ts[i][d];
      }
    }",
    );
    let region = spawn_region(&record_source("region_scratch", &src), "main");
    assert!(
        !region.contains("\"buffer\": \"ts\""),
        "an unplaced scratch tile contributes nothing:\n{region}"
    );
    assert!(
        region.contains(
            "{\"buffer\": \"ad\", \"space\": \"GPU_HBM\", \"read_bytes\": 16, \
             \"written_bytes\": 0}"
        ) && region.contains(
            "{\"buffer\": \"od\", \"space\": \"GPU_HBM\", \"read_bytes\": 0, \
             \"written_bytes\": 16}"
        ),
        "the placed tensors are still counted normally:\n{region}"
    );
}

/// The honesty rule, three ways. A region the counter cannot weigh exactly is
/// reported as absent WITH A REASON -- never as zero, and never partially. A
/// consumer that cannot tell "uncountable" from "moved nothing" would read a
/// walker limitation as a measurement of an efficient kernel.
#[test]
fn an_uncountable_region_reports_absence_with_a_reason() {
    // A dynamic loop bound: the number of times the body runs is not known here.
    let dynamic = one_pass_kernel(
        "    let n = ad[0][0] as i32;
    for i in 0..n {
      od[0][0] = ad[0][0];
    }",
    );
    let region = spawn_region(&record_source("region_dynamic", &dynamic), "main");
    assert!(
        region.contains("\"traffic\": null") && region.contains("\"by_buffer\": []"),
        "a dynamic bound makes the whole region uncountable, not partly counted:\n{region}"
    );
    assert!(
        region.contains("not statically known"),
        "and it says why:\n{region}"
    );

    // A call that receives a placed tensor: the callee's own accesses are not
    // walked, so counting only what is visible here would undercount. `print`
    // is the smallest such call that needs no signature of its own.
    //
    // This guard was found DISABLED (`let hit = false && ...`) while the suite
    // was green, because the case it protects was not covered by a compiling
    // program -- the first attempt passed a placed tensor to a user function,
    // which Vx currently rejects on a topology-index type mismatch unrelated to
    // traffic. A safety check with no test that compiles is a check that can be
    // switched off without anything going red.
    let opaque = "\
Memory CPU_DRAM {}
Memory GPU_HBM {
  within: Memory::CPU_DRAM, capacity: 40 GiB, bandwidth: 3 TB/s
}
fn main() -> i32 {
  let mut a = Tensor<f32>([ 2, 2 ]);
  for i in 0..2 {
    for d in 0..2 {
      a[i][d] = 1.0;
    }
  }
  let ad = transfer(a, Memory::GPU_HBM);
  spawn on(Topology::GPU) {
    print(ad);
  }
  return 0;
}
";
    let region = spawn_region(&record_source("region_opaque", opaque), "main");
    assert!(
        region.contains("\"traffic\": null") && region.contains("placed tensor to a call"),
        "a placed tensor crossing a call boundary is refused, not silently ignored:\n{region}"
    );
}

/// An overflowing count is an absence, not a saturated number. The A4 review
/// established the rule after saturation published `u64::MAX` as an EXACT byte
/// count; the nest here is instant to count and must never be run.
#[test]
fn a_region_count_that_overflows_is_reported_as_absent() {
    let src = one_pass_kernel(
        "    for a1 in 0..4000000000 {
      for a2 in 0..4000000000 {
        for a3 in 0..4000000000 {
          od[0][0] = ad[0][0];
        }
      }
    }",
    );
    let region = spawn_region(&record_source("region_overflow", &src), "main");
    assert!(
        region.contains("\"traffic\": null"),
        "an overflowed count must not be published as a number:\n{region}"
    );
    assert!(
        region.contains("overflow"),
        "and the reason names the overflow:\n{region}"
    );
}

/// One region, two spaces. `custom_topology_user_lowering.vx` stages a tile from
/// GPU_HBM into SMEM inside the kernel and writes its result back out, so the
/// same `spawn` touches both spaces and they must not be pooled.
///
/// This is what makes the record answer a locality question. The SMEM reads are
/// reads of a tile that was staged precisely so they would be cheap; the GPU_HBM
/// writes are not. Summed into one figure the distinction disappears, and the
/// distinction is the reason anyone stages a tile at all.
///
/// It also pins body-local placement: `tile` is bound INSIDE the region by
/// `let tile = transfer(ad, Memory::SMEM)`, so its space comes from that
/// initializer rather than from the enclosing scope, and its element size comes
/// from the value transferred -- a transfer moves bytes, it does not convert
/// them.
#[test]
fn a_region_keeps_its_two_spaces_apart() {
    let region = spawn_region(&record("custom_topology_user_lowering.vx"), "main");
    assert!(
        region.contains(
            "{\"buffer\": \"tile\", \"space\": \"SMEM\", \"read_bytes\": 16, \
             \"written_bytes\": 0}"
        ),
        "the staged tile is read from SMEM, not from the space it came from:\n{region}"
    );
    assert!(
        region.contains(
            "{\"buffer\": \"od\", \"space\": \"GPU_HBM\", \"read_bytes\": 0, \
             \"written_bytes\": 16}"
        ),
        "the result is written to device memory:\n{region}"
    );
    assert!(
        region.contains("{\"space\": \"GPU_HBM\", \"read_bytes\": 0, \"written_bytes\": 16}")
            && region.contains("{\"space\": \"SMEM\", \"read_bytes\": 16, \"written_bytes\": 0}"),
        "and the per-space aggregate keeps them apart:\n{region}"
    );
}

/// Every refusal path gets a test, because a guard with no test is a guard that
/// can be switched off without anything going red.
///
/// That is not hypothetical. The call-opacity guard in this counter was found
/// disabled -- `let hit = false && (...)` -- while this suite was green, because
/// the only test covering it used a program that did not compile. These cases
/// exist so the same thing cannot happen quietly to the others: each one names a
/// distinct reason string, so a disabled guard changes the record and fails here.
fn region_body(body: &str) -> String {
    format!(
        "Memory CPU_DRAM {{}}
Memory GPU_HBM {{
  within: Memory::CPU_DRAM, capacity: 40 GiB, bandwidth: 3 TB/s
}}
fn main() -> i32 {{
  let mut a = Tensor<f32>([ 2, 2 ]);
  for i in 0..2 {{
    for d in 0..2 {{
      a[i][d] = 1.0;
    }}
  }}
  let ad = transfer(a, Memory::GPU_HBM);
  spawn on(Topology::GPU) {{
{body}
  }}
  return 0;
}}
"
    )
}

#[test]
fn an_unbounded_loop_in_a_region_is_refused() {
    let src = region_body(
        "    let mut n = 0;
    loop {
      n = n + 1;
      let z = ad[0][0];
      if n > 2 { break; }
    }",
    );
    let region = spawn_region(&record_source("region_loop", &src), "main");
    assert!(
        region.contains("\"traffic\": null") && region.contains("unbounded `loop`"),
        "a loop with no static trip count cannot be weighed:\n{region}"
    );
}

#[test]
fn a_break_in_a_region_is_refused() {
    let src = region_body(
        "    for i in 0..2 {
      let z = ad[i][0];
      break;
    }",
    );
    let region = spawn_region(&record_source("region_break", &src), "main");
    assert!(
        region.contains("\"traffic\": null") && region.contains("trip counts a fiction"),
        "`break` means the loop's declared trip count is not what runs:\n{region}"
    );
}

#[test]
fn an_unrecognised_construct_that_indexes_is_refused() {
    let src = region_body("    let arr = [ ad[0][0] ];");
    let region = spawn_region(&record_source("region_catchall_idx", &src), "main");
    assert!(
        region.contains("\"traffic\": null") && region.contains("indexes something"),
        "an array literal is not walked, and this one holds a read:\n{region}"
    );
}

/// The opaque-handoff guard (`mentions_any`): a construct the walker does not
/// recognise that hands a placed tensor somewhere WITHOUT indexing it. This test
/// was impossible to write until this campaign's own fixes -- the array-literal
/// form crashed codegen (#354) and the struct form needed a Pinned annotation to
/// unify (#355). Both fixed; the "uncoverable" note this replaces is retired.
///
/// The guard family's history is the reason for the test: its call-opacity
/// sibling shipped disabled (`let hit = false && ...`) with a green suite,
/// because its only test did not compile.
#[test]
fn an_unrecognised_construct_that_hands_off_a_placed_tensor_is_refused() {
    let src = "\
Memory CPU_DRAM {}
Memory GPU_HBM {
  within: Memory::CPU_DRAM, capacity: 40 GiB, bandwidth: 3 TB/s
}
struct Holder {
  t: Pinned<Tensor<f32, [2, 2]>, Topology::GPU[0]>,
}
fn main() -> i32 {
  let mut a = Tensor<f32>([ 2, 2 ]);
  a[0][0] = 1.0;
  let ad = transfer(a, Memory::GPU_HBM);
  spawn on(Topology::GPU) {
    let h = Holder { t: ad };
  }
  return 0;
}
";
    let region = spawn_region(&record_source("region_handoff", src), "main");
    assert!(
        region.contains("\"traffic\": null") && region.contains("mentions a placed tensor"),
        "a struct initializer swallowing a placed tensor is a handoff the counter \
         cannot follow:\n{region}"
    );
}

/// The codegen review found three ways a placed tensor slipped past the call
/// guard or the binding resolver and produced `traffic: [], exact: true` -- an
/// exact-zero claim while the kernel read device memory. Each gets its own test,
/// asserting the corrected verdict.
///
/// A dereferenced reference is still the tensor: `peek(*r)` where `r = &ad`.
#[test]
fn a_placed_tensor_passed_through_a_dereference_is_refused() {
    let src = "\
Memory CPU_DRAM {}
Memory GPU_HBM {
  within: Memory::CPU_DRAM, capacity: 40 GiB, bandwidth: 3 TB/s
}
fn peek(t: Pinned<Tensor<f32, [2, 2]>, Topology::GPU[0]>) -> i32 {
  return 0;
}
fn main() -> i32 {
  let mut a = Tensor<f32>([ 2, 2 ]);
  a[0][0] = 1.0;
  let ad = transfer(a, Memory::GPU_HBM);
  spawn on(Topology::GPU) {
    let r = &ad;
    let z = peek(*r);
  }
  return 0;
}
";
    let region = spawn_region(&record_source("region_deref_arg", src), "main");
    assert!(
        region.contains("\"traffic\": null") && region.contains("placed tensor to a call"),
        "*r IS ad; unwrapping the reference must not launder the handoff:\n{region}"
    );
}

/// A placed STRUCT FIELD passed to a call: `h` is not placed, `h.t` is. The
/// classifier resolves the member chain through the struct environment, so a
/// scalar field (`f(cfg.max)`) stays legal while a placed one refuses.
#[test]
fn a_placed_struct_field_passed_to_a_call_is_refused() {
    let src = "\
Memory CPU_DRAM {}
Memory GPU_HBM {
  within: Memory::CPU_DRAM, capacity: 40 GiB, bandwidth: 3 TB/s
}
struct Holder {
  t: Pinned<Tensor<f32, [2, 2]>, Topology::GPU[0]>,
}
fn peek(t: Pinned<Tensor<f32, [2, 2]>, Topology::GPU[0]>) -> i32 {
  return 0;
}
fn main() -> i32 {
  let mut a = Tensor<f32>([ 2, 2 ]);
  a[0][0] = 1.0;
  let ad = transfer(a, Memory::GPU_HBM);
  let h = Holder { t: ad };
  spawn on(Topology::GPU) {
    let z = peek(h.t);
  }
  return 0;
}
";
    let region = spawn_region(&record_source("region_member_arg", src), "main");
    assert!(
        region.contains("\"traffic\": null") && region.contains("placed tensor to a call"),
        "the field is placed even though the struct is not:\n{region}"
    );
}

/// A binding initialized by a CALL that returns a placed tensor. The initializer
/// form is opaque to the resolver, but at the region's top level the checker has
/// already typed the binding and its scope is still live -- so this COUNTS,
/// which is strictly better than refusing: 4 elements x 4 B read once each.
#[test]
fn a_call_returned_placed_binding_is_counted_from_its_checked_type() {
    let src = "\
Memory CPU_DRAM {}
Memory GPU_HBM {
  within: Memory::CPU_DRAM, capacity: 40 GiB, bandwidth: 3 TB/s
}
fn pick() -> Pinned<Tensor<f32, [2, 2]>, Topology::GPU[0]> {
  let mut a = Tensor<f32>([ 2, 2 ]);
  a[0][0] = 1.0;
  let ad = transfer(a, Memory::GPU_HBM);
  return ad;
}
fn main() -> i32 {
  let mut s : f32 = 0.0;
  spawn on(Topology::GPU) {
    let t = pick();
    for i in 0..2 {
      for d in 0..2 {
        s = s + t[i][d];
      }
    }
  }
  return 0;
}
";
    let region = spawn_region(&record_source("region_callret", src), "main");
    assert!(
        region.contains(
            "{\"buffer\": \"t\", \"space\": \"GPU_HBM\", \"read_bytes\": 16, \
             \"written_bytes\": 0}"
        ),
        "the binding's checked type says GPU_HBM, and the loop reads it once per \
         element -- before the fix this region published traffic: [] exact:true:\n{region}"
    );
}

/// Sub-byte elements are refused rather than rounded up, in the REGION counter
/// as well as the lowering one. Rounding i4 to a byte made a faithful copy
/// report the 2.0 ratio that is this stage's evidence of waste; the same
/// argument applies to a kernel's reads.
#[test]
fn a_sub_byte_placed_tensor_in_a_region_is_refused() {
    let src = "Memory CPU_DRAM {}
Memory GPU_HBM {
  within: Memory::CPU_DRAM, capacity: 40 GiB, bandwidth: 3 TB/s
}
fn main() -> i32 {
  let mut a = Tensor<i4>([ 2, 2 ]);
  for i in 0..2 {
    for d in 0..2 {
      a[i][d] = 1;
    }
  }
  let ad = transfer(a, Memory::GPU_HBM);
  spawn on(Topology::GPU) {
    let z = ad[0][0];
  }
  return 0;
}
";
    let region = spawn_region(&record_source("region_subbyte", src), "main");
    assert!(
        region.contains("\"traffic\": null") && region.contains("sub-byte elements"),
        "a packed element width this counter does not model is refused, not rounded:\n{region}"
    );
}

/// A write THROUGH a dereference is uncountable, not zero.
///
/// `let q = &mut ad[i][d]; *q = 3.0;` writes device memory that this walk cannot
/// attribute -- following a reference to its referent is exactly what it does not
/// do. Before the fix the region published `written_bytes: 0, exact: true` while
/// booking the address computation's reads: wrong in both directions at once, and
/// the silent-exact-zero class the module header forbids.
#[test]
fn a_write_through_a_dereference_is_refused() {
    let src = "\
Memory CPU_DRAM {}
Memory GPU_HBM {
  within: Memory::CPU_DRAM, capacity: 40 GiB, bandwidth: 3 TB/s
}
fn main() -> i32 {
  let mut a = Tensor<f32>([ 2, 2 ]);
  a[0][0] = 1.0;
  let mut ad = transfer(a, Memory::GPU_HBM);
  spawn on(Topology::GPU) {
    for i in 0..2 {
      for d in 0..2 {
        let q = &mut ad[i][d];
        *q = 3.0;
      }
    }
  }
  return 0;
}
";
    let region = spawn_region(&record_source("region_deref_write", src), "main");
    assert!(
        region.contains("\"traffic\": null") && region.contains("through a dereference"),
        "an unfollowable write is uncountable, never a confident zero:\n{region}"
    );
}
