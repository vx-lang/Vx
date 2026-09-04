//===- working_set_peak_test.rs - Vx Compiler ------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
// What the capacity check believes is resident at once, shape by shape.
//
// A placed tile is released at the end of the block that placed it, so two tiles
// coexist only when one's block encloses the other's. The working set is the peak
// over that relation rather than the sum of every placement, and the difference is
// the whole point: a sum describes a moment the program never has.
//
// Each row states a program shape and the peak the checker should compute for it.
// Asserting the number rather than only the verdict is deliberate -- the figure
// reaches `--diagnostics-json` resident sets, which a downstream consumer reads to
// size an engine, so a shape can be admitted for the right reason and still report
// the wrong residency.
//
// See docs/working_set_peak_implementation_plan.md.

use std::path::PathBuf;
use std::process::Command;

/// 1024*768*4.
const TILE: u64 = 3_145_728;

/// Roomy on purpose. The matrix below measures the peak the checker computes, so every
/// shape in it has to compile -- a refusal would stop the figure being reported at all
/// and the case would say nothing about which tiles coexist.
const MACHINE: &str = "\
Memory CPU_DRAM {}
Memory W {
  within: Memory::CPU_DRAM, capacity: 64 MiB, managed: explicit
}
Topology Dev {
  arch: nvptx64,
  memory: Memory::W,
  visible: [Memory::W],
  transfer Memory::CPU_DRAM -> Memory::W : 10
}
";

/// A placement of one tile, bound to `name`.
fn place(name: &str) -> String {
    format!(
        "  let {name}_t = Tensor<f32, [1024, 768]>::uninit();\n  \
         let _{name} = transfer({name}_t, Memory::W);\n"
    )
}

struct Case {
    /// What the shape is, in the failure message.
    what: &'static str,
    /// Statements for the body of `main`.
    body: String,
    /// Expected peak residency in W, in bytes.
    peak: u64,
    /// Expected number of tiles in that peak.
    tiles: usize,
}

fn cases() -> Vec<Case> {
    let one = |n| TILE * n;
    vec![
        Case {
            what: "one placement",
            body: place("a"),
            peak: one(1),
            tiles: 1,
        },
        Case {
            what: "two in the same block coexist until the function ends",
            body: place("a") + &place("b"),
            peak: one(2),
            tiles: 2,
        },
        Case {
            what: "a block's tile is gone before the next one is placed",
            body: format!("  if true {{\n{}  }}\n{}", place("a"), place("b")),
            peak: one(1),
            tiles: 1,
        },
        Case {
            what: "sibling blocks never share the space",
            body: format!(
                "  if true {{\n{}  }}\n  if true {{\n{}  }}\n",
                place("a"),
                place("b")
            ),
            peak: one(1),
            tiles: 1,
        },
        Case {
            what: "the arms of one `if` are siblings too",
            body: format!(
                "  if true {{\n{}  }} else {{\n{}  }}\n",
                place("a"),
                place("b")
            ),
            peak: one(1),
            tiles: 1,
        },
        Case {
            what: "an enclosing tile is still resident inside the block",
            body: place("a") + &format!("  if true {{\n{}  }}\n", place("b")),
            peak: one(2),
            tiles: 2,
        },
        Case {
            what: "an outer placement after the block does not meet the block's tile",
            body: format!("  if true {{\n{}  }}\n", place("a")) + &place("b"),
            peak: one(1),
            tiles: 1,
        },
        Case {
            // The two outer tiles are two, not three: the block's tile was released when
            // the block closed and never meets `c`. A sum over the function would say 3.
            what: "outer, then a block, then outer again: the two outer ones meet",
            body: place("a") + &format!("  if true {{\n{}  }}\n", place("b")) + &place("c"),
            peak: one(2),
            tiles: 2,
        },
        Case {
            what: "nesting accumulates down a chain",
            body: place("a")
                + &format!(
                    "  if true {{\n{}    if true {{\n{}    }}\n  }}\n",
                    place("b"),
                    place("c")
                ),
            peak: one(3),
            tiles: 3,
        },
        Case {
            what: "a deep chain whose branches are siblings",
            body: format!(
                "  if true {{\n{}    if true {{\n{}    }}\n    if true {{\n{}    }}\n  }}\n",
                place("a"),
                place("b"),
                place("c")
            ),
            peak: one(2),
            tiles: 2,
        },
        Case {
            what: "a loop body is one residency, not one per iteration",
            body: format!("  for i in 0..4 {{\n{}  }}\n", place("a")),
            peak: one(1),
            tiles: 1,
        },
        Case {
            what: "a tile enclosing a loop meets the loop's own",
            body: place("a") + &format!("  for i in 0..4 {{\n{}  }}\n", place("b")),
            peak: one(2),
            tiles: 2,
        },
    ]
}

fn compile(root: &PathBuf, src: &std::path::Path) -> (bool, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_vxc"))
        .current_dir(root)
        .args([
            src.to_str().unwrap(),
            "--action",
            "emit-mlir",
            "-o",
            "/dev/null",
            "--diagnostics-json",
        ])
        .output()
        .expect("failed to run vxc");
    (
        out.status.success(),
        format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ),
    )
}

/// The `total_bytes` and `tiles` the compiler recorded for space `W`.
fn resident(log: &str) -> Option<(u64, usize)> {
    // Route entries name the same space in their traffic arrays, so the resident-set line
    // is the one that also carries a total.
    let line = log
        .lines()
        .find(|l| l.contains("\"space\": \"W\"") && l.contains("\"total_bytes\""))?;
    let field = |k: &str| -> Option<u64> {
        let at = line.find(&format!("\"{k}\": "))? + k.len() + 4;
        let rest = &line[at..];
        let end = rest.find(|c: char| !c.is_ascii_digit())?;
        rest[..end].parse().ok()
    };
    Some((field("total_bytes")?, field("tiles")? as usize))
}

#[test]
fn the_working_set_is_the_peak_of_what_coexists() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dir = std::env::temp_dir().join(format!("vx-working-set-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("failed to create the scratch directory");

    let mut wrong = Vec::new();
    for (i, case) in cases().iter().enumerate() {
        let src = dir.join(format!("case{i}.vx"));
        std::fs::write(
            &src,
            format!(
                "{MACHINE}\nfn main() -> i32 {{\n{}  return 0;\n}}\n",
                case.body
            ),
        )
        .expect("failed to write the case");

        let (ok, log) = compile(&root, &src);
        // Every case here fits: the point is the figure, not the refusal. A case that
        // does not compile is a broken fixture and says nothing about the peak.
        if !ok {
            wrong.push(format!("  {}: did not compile\n{}", case.what, log));
            continue;
        }
        match resident(&log) {
            None => wrong.push(format!("  {}: no resident set was recorded", case.what)),
            Some((bytes, tiles)) if bytes != case.peak || tiles != case.tiles => {
                wrong.push(format!(
                    "  {}: expected {} bytes over {} tile(s), got {} over {}",
                    case.what, case.peak, case.tiles, bytes, tiles
                ))
            }
            Some(_) => {}
        }
    }
    let _ = std::fs::remove_dir_all(&dir);

    assert!(
        wrong.is_empty(),
        "the working set does not match what coexists:\n{}",
        wrong.join("\n")
    );
}

/// The other half: a peak that genuinely exceeds the space is still refused. Without
/// this, an analysis that reported zero for everything would satisfy the matrix above.
#[test]
fn a_peak_over_capacity_is_still_refused() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dir = std::env::temp_dir().join(format!("vx-working-set-over-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("failed to create the scratch directory");

    // Three tiles in one block: 9 MiB into a 4 MiB space.
    let small = MACHINE.replace("capacity: 64 MiB", "capacity: 4 MiB");
    let src = dir.join("over.vx");
    std::fs::write(
        &src,
        format!(
            "{small}\nfn main() -> i32 {{\n{}{}{}  return 0;\n}}\n",
            place("a"),
            place("b"),
            place("c")
        ),
    )
    .expect("failed to write the case");
    let (ok, log) = compile(&root, &src);
    assert!(!ok, "three tiles in one block must not fit 4 MiB:\n{log}");
    assert!(
        log.contains("E6010") && log.contains(&(TILE * 3).to_string()),
        "the refusal must name the peak it computed:\n{log}"
    );

    // The same three, each in its own block: never more than one at a time.
    let src = dir.join("spread.vx");
    std::fs::write(
        &src,
        format!(
            "{small}\nfn main() -> i32 {{\n  if true {{\n{}  }}\n  if true {{\n{}  }}\n  if true {{\n{}  }}\n  return 0;\n}}\n",
            place("a"),
            place("b"),
            place("c")
        ),
    )
    .expect("failed to write the case");
    let (ok, log) = compile(&root, &src);
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        ok,
        "the same three tiles in sibling blocks never coexist and must fit:\n{log}"
    );
}
