//===- assert_codegen_test.rs - `assert` emits no runtime check ----------===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// `assert(cond)` / `assert(cond, "msg")` emits a real runtime check (Vx#361).
//
// A condition the checker can fold is still decided at compile time -- E8002 on a
// false one -- and a folded-true condition costs a check the optimiser removes.
// A condition over runtime values now lowers to `cf.assert`, which the host
// pipeline's `convert-cf-to-llvm` turns into a branch onto `puts` + `abort`.
// Both backends emit it: the AST path in `generator.rs::emit_runtime_assert`,
// the flat path via `Opcode::Assert`. They must agree, because either may claim
// a given function.
//
// Before this, `assert` lowered to nothing in both paths, so a false assertion
// over runtime values was not a failure but silence. These tests were written
// while that was true -- one pinning the silence, one (ignored) demanding the
// trap -- and both are now inverted, which was their purpose.
//
// NOT EMITTED INSIDE A DEVICE KERNEL. `cf` is device-lowerable, so a `cf.assert`
// in a kernel would pass the dialect gate and then lower to a call to the host's
// `abort` in PTX: an unresolved symbol that costs the kernel its device image,
// turning an assertion into the silent loss of the whole kernel. Device-side
// trapping needs `__assertfail`, tracked on Vx#361.
//
//===----------------------------------------------------------------------===//

use std::io::Write;
use std::process::Command;

/// A program whose assertion is false at run time and cannot be folded at compile
/// time: `n` is a parameter, and the argument arrives through a tensor element,
/// so the const-evaluator has no value for it.
const FALSE_AT_RUNTIME: &str = r#"
fn checked(n: i32) -> i32 {
  assert(n < 5);
  return n * 10;
}
fn main() -> i32 {
  let mut t = Tensor<i32>([ 2 ]);
  t[0] = 1000;
  t[1] = 7;
  let big = t[0];
  print(checked(big));
  return 0;
}
"#;

fn write_probe(stem: &str, source: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("vx_assert_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join(format!("{stem}.vx"));
    let mut f = std::fs::File::create(&path).expect("write probe");
    f.write_all(source.as_bytes()).expect("write probe");
    path
}

/// The check reaches the IR, and it is the condition the programmer wrote.
///
/// `checked` compiles to the comparison, the assert, and the multiply:
///
///   func.func @checked(%arg0: i32) -> i32 {
///     %c5_i32 = arith.constant 5 : i32
///     %0 = arith.cmpi slt, %arg0, %c5_i32 : i32
///     cf.assert %0, "assertion failed"
///     ...
///
/// This test asserted the OPPOSITE until Vx#361 -- that no such op existed --
/// and was written to go red on exactly this change.
#[test]
fn a_runtime_assert_emits_a_check() {
    let probe = write_probe("assert_emits_check", FALSE_AT_RUNTIME);
    let ir = Command::new(env!("CARGO_BIN_EXE_vxc"))
        .arg(&probe)
        .arg("--action")
        .arg("emit-mlir")
        .output()
        .expect("run vxc");
    let ir_text =
        String::from_utf8_lossy(&ir.stdout).to_string() + &String::from_utf8_lossy(&ir.stderr);
    assert!(
        ir_text.contains("cf.assert"),
        "the assertion must survive into the IR as a check:\n{ir_text}"
    );
    assert!(
        ir_text.contains("arith.cmpi slt"),
        "and it must test the condition the programmer wrote, not a stand-in:\n{ir_text}"
    );
}

/// A false assertion stops the program.
///
/// This was the `#[ignore]`d wish; it is now a test. `checked(1000)` violates
/// `assert(n < 5)`, so the process must fail instead of printing 10000. The
/// probe's condition cannot be folded -- `n` is a parameter and its argument
/// arrives through a tensor element -- so this exercises the runtime check and
/// not E8002.
#[test]
fn a_false_runtime_assert_aborts() {
    let probe = write_probe("assert_should_abort", FALSE_AT_RUNTIME);
    let run = Command::new(env!("CARGO_BIN_EXE_vxc"))
        .arg(&probe)
        .output()
        .expect("run vxc");
    let out = String::from_utf8_lossy(&run.stdout).to_string();

    assert!(
        !out.contains("10000"),
        "a program whose assertion is false must not produce a result: \
         `assert(n < 5)` held n = 1000, and the program printed 10000 anyway"
    );
    assert!(
        !run.status.success(),
        "a violated assertion must fail the process, not be silence"
    );
}
