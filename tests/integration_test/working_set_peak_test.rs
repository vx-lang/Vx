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

/// One tile: 1024*768*4 bytes.
const TILE: u64 = 3_145_728;

struct Case {
    /// What the shape is, in the failure message.
    what: &'static str,
    /// The body of `main`, written out so the shape can be read here rather than
    /// assembled from fragments.
    body: &'static str,
    /// Expected peak residency in W, in tiles. Each tile is TILE bytes.
    tiles: usize,
}

/// Every case places 3 MiB tiles into `W`. What differs is the blocks they sit in,
/// which is what decides how many of them are resident at once.
const CASES: &[Case] = &[
    Case {
        what: "one placement",
        body: "
  let a = Tensor<f32, [1024, 768]>::uninit();
  let _sa = transfer(a, Memory::W);
",
        tiles: 1,
    },
    Case {
        what: "two in the same block are resident together until the function ends",
        body: "
  let a = Tensor<f32, [1024, 768]>::uninit();
  let _sa = transfer(a, Memory::W);
  let b = Tensor<f32, [1024, 768]>::uninit();
  let _sb = transfer(b, Memory::W);
",
        tiles: 2,
    },
    Case {
        what: "a block's tile is released before the next one is placed",
        body: "
  if true {
    let a = Tensor<f32, [1024, 768]>::uninit();
    let _sa = transfer(a, Memory::W);
  }
  let b = Tensor<f32, [1024, 768]>::uninit();
  let _sb = transfer(b, Memory::W);
",
        tiles: 1,
    },
    Case {
        what: "sibling blocks never share the space",
        body: "
  if true {
    let a = Tensor<f32, [1024, 768]>::uninit();
    let _sa = transfer(a, Memory::W);
  }
  if true {
    let b = Tensor<f32, [1024, 768]>::uninit();
    let _sb = transfer(b, Memory::W);
  }
",
        tiles: 1,
    },
    Case {
        what: "the two arms of one `if` are siblings too",
        body: "
  if true {
    let a = Tensor<f32, [1024, 768]>::uninit();
    let _sa = transfer(a, Memory::W);
  } else {
    let b = Tensor<f32, [1024, 768]>::uninit();
    let _sb = transfer(b, Memory::W);
  }
",
        tiles: 1,
    },
    Case {
        what: "an enclosing tile is still resident inside the block",
        body: "
  let a = Tensor<f32, [1024, 768]>::uninit();
  let _sa = transfer(a, Memory::W);
  if true {
    let b = Tensor<f32, [1024, 768]>::uninit();
    let _sb = transfer(b, Memory::W);
  }
",
        tiles: 2,
    },
    Case {
        what: "a tile placed after a block does not meet the block's own",
        body: "
  if true {
    let a = Tensor<f32, [1024, 768]>::uninit();
    let _sa = transfer(a, Memory::W);
  }
  let b = Tensor<f32, [1024, 768]>::uninit();
  let _sb = transfer(b, Memory::W);
",
        tiles: 1,
    },
    Case {
        // Two, not three: `b` is gone by the time `c` is placed. A sum would say three,
        // and this is the shape that needs the program-order bound as well as the
        // prefix test -- `a` and `c` share a scope chain with `b`'s prefix.
        what: "outer, then a block, then outer again: only the two outer ones meet",
        body: "
  let a = Tensor<f32, [1024, 768]>::uninit();
  let _sa = transfer(a, Memory::W);
  if true {
    let b = Tensor<f32, [1024, 768]>::uninit();
    let _sb = transfer(b, Memory::W);
  }
  let c = Tensor<f32, [1024, 768]>::uninit();
  let _sc = transfer(c, Memory::W);
",
        tiles: 2,
    },
    Case {
        what: "nesting accumulates down a chain",
        body: "
  let a = Tensor<f32, [1024, 768]>::uninit();
  let _sa = transfer(a, Memory::W);
  if true {
    let b = Tensor<f32, [1024, 768]>::uninit();
    let _sb = transfer(b, Memory::W);
    if true {
      let c = Tensor<f32, [1024, 768]>::uninit();
      let _sc = transfer(c, Memory::W);
    }
  }
",
        tiles: 3,
    },
    Case {
        what: "a deep chain whose two inner branches are siblings",
        body: "
  if true {
    let a = Tensor<f32, [1024, 768]>::uninit();
    let _sa = transfer(a, Memory::W);
    if true {
      let b = Tensor<f32, [1024, 768]>::uninit();
      let _sb = transfer(b, Memory::W);
    }
    if true {
      let c = Tensor<f32, [1024, 768]>::uninit();
      let _sc = transfer(c, Memory::W);
    }
  }
",
        tiles: 2,
    },
    Case {
        what: "a loop body is one residency, not one per iteration",
        body: "
  for i in 0..4 {
    let a = Tensor<f32, [1024, 768]>::uninit();
    let _sa = transfer(a, Memory::W);
  }
",
        tiles: 1,
    },
    Case {
        what: "a tile enclosing a loop meets the loop's own",
        body: "
  let a = Tensor<f32, [1024, 768]>::uninit();
  let _sa = transfer(a, Memory::W);
  for i in 0..4 {
    let b = Tensor<f32, [1024, 768]>::uninit();
    let _sb = transfer(b, Memory::W);
  }
",
        tiles: 2,
    },
];

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
    for (i, case) in CASES.iter().enumerate() {
        let program = format!(
            "{MACHINE}\nfn main() -> i32 {{{}  return 0;\n}}\n",
            case.body
        );
        let src = dir.join(format!("case{i}.vx"));
        std::fs::write(&src, &program).expect("failed to write the case");

        let (ok, log) = compile(&root, &src);
        // Every case here fits: the point is the figure, not the refusal. A case that does
        // not compile is a broken fixture and says nothing about which tiles coexist.
        if !ok {
            wrong.push(format!("{}: did not compile\n{program}\n{log}", case.what));
            continue;
        }
        let expected = TILE * case.tiles as u64;
        match resident(&log) {
            None => wrong.push(format!(
                "{}: no resident set was recorded\n{program}",
                case.what
            )),
            Some((bytes, tiles)) if bytes != expected || tiles != case.tiles => {
                wrong.push(format!(
                    "{}: expected {} bytes over {} tile(s), got {} over {}\n{program}",
                    case.what, expected, case.tiles, bytes, tiles
                ))
            }
            Some(_) => {}
        }
    }
    let _ = std::fs::remove_dir_all(&dir);

    assert!(
        wrong.is_empty(),
        "the working set does not match what coexists:\n\n{}",
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
    // 3 MiB tiles into a 4 MiB space, so two of them already do not fit.
    let small = MACHINE.replace("capacity: 64 MiB", "capacity: 4 MiB");

    let over = format!(
        "{small}
fn main() -> i32 {{
  let a = Tensor<f32, [1024, 768]>::uninit();
  let _sa = transfer(a, Memory::W);
  let b = Tensor<f32, [1024, 768]>::uninit();
  let _sb = transfer(b, Memory::W);
  let c = Tensor<f32, [1024, 768]>::uninit();
  let _sc = transfer(c, Memory::W);
  return 0;
}}
"
    );
    let src = dir.join("over.vx");
    std::fs::write(&src, &over).expect("failed to write the case");
    let (ok, log) = compile(&root, &src);
    assert!(
        !ok,
        "three tiles in one block must not fit 4 MiB:\n{over}\n{log}"
    );
    assert!(
        log.contains("E6010") && log.contains(&(TILE * 3).to_string()),
        "the refusal must name the peak it computed:\n{log}"
    );

    // The same three tiles, each in its own block: never more than one at a time.
    let spread = format!(
        "{small}
fn main() -> i32 {{
  if true {{
    let a = Tensor<f32, [1024, 768]>::uninit();
    let _sa = transfer(a, Memory::W);
  }}
  if true {{
    let b = Tensor<f32, [1024, 768]>::uninit();
    let _sb = transfer(b, Memory::W);
  }}
  if true {{
    let c = Tensor<f32, [1024, 768]>::uninit();
    let _sc = transfer(c, Memory::W);
  }}
  return 0;
}}
"
    );
    let src = dir.join("spread.vx");
    std::fs::write(&src, &spread).expect("failed to write the case");
    let (ok, log) = compile(&root, &src);
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        ok,
        "the same three tiles in sibling blocks never coexist and must fit:\n{spread}\n{log}"
    );
}
