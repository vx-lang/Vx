//===- assert_codegen_test.rs - `assert` emits no runtime check ----------===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// `assert(cond)` is a COMPILE-TIME fact in Vx, and nothing else. When the
// condition folds, the checker decides it (E8002 on a false one). When it does
// not fold -- the ordinary case, a condition over runtime values -- **no code is
// emitted at all**. Both backends say so in as many words:
//
//   src/codegen/generator.rs, Statement::Assert:
//     // TODO: Lower to `scf.if` with panic/abort for runtime checks.
//
//   src/hir/flatten.rs, Statement::Assert:
//     // the AST codegen emits no runtime check for it. Match that exactly:
//     // lower it to nothing, so flat and AST agree at runtime.
//
// So a false assertion over runtime values is not a failure; it is silence. That
// is a defensible state for a fact used by the seam certificates, and an
// indefensible one the moment anything TRUSTS an assert -- which is exactly what
// the proposed topology-identity work would do (a merged device identity resting
// on `assert(i == j)` becomes a wrong-device miscompile if the assert is false at
// run time, with nothing to catch it). Filed as Vx#361.
//
// Two tests below, deliberately of opposite kinds:
//
//   * `a_false_runtime_assert_does_not_stop_the_program` PINS TODAY'S BEHAVIOUR
//     and passes. It exists so the day someone implements the trap, it goes red
//     and forces a deliberate decision rather than a silent semantic change.
//
//   * `a_false_runtime_assert_should_abort` states what the language OUGHT to do
//     and FAILS today. It is `#[ignore]`d so CI stays green; run it with
//     `cargo test --test integration_test -- --ignored assert` to see the gap.
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

/// PINS TODAY'S BEHAVIOUR: `assert(1000 < 5)` neither stops the program nor
/// leaves any trace in the emitted IR.
///
/// The emitted function is the whole evidence -- the assertion is not weakened
/// or hoisted, it is absent:
///
///   func.func @checked(%arg0: i32) -> i32 {
///     %c10_i32 = arith.constant 10 : i32
///     %0 = arith.muli %arg0, %c10_i32 : i32
///     return %0 : i32
///   }
///
/// When runtime checks land, this test breaks. That is its job: the change is a
/// semantic one and should not be able to happen quietly.
#[test]
fn a_false_runtime_assert_does_not_stop_the_program() {
    let probe = write_probe("assert_false_runtime", FALSE_AT_RUNTIME);

    // 1. The IR carries no check of any kind.
    let ir = Command::new(env!("CARGO_BIN_EXE_vxc"))
        .arg(&probe)
        .arg("--action")
        .arg("emit-mlir")
        .output()
        .expect("run vxc");
    let ir_text =
        String::from_utf8_lossy(&ir.stdout).to_string() + &String::from_utf8_lossy(&ir.stderr);
    for op in ["llvm.trap", "cf.assert", "llvm.intr.assume"] {
        assert!(
            !ir_text.contains(op),
            "`assert` is not code-generated today, so no `{op}` should appear. \
             If this fires, runtime checks have landed and this test (plus the \
             ignored one beside it) needs updating:\n{ir_text}"
        );
    }
    assert!(
        ir_text.contains("func.func @checked"),
        "the probe must actually have compiled:\n{ir_text}"
    );

    // 2. And the program runs to completion, printing the value the assertion
    //    was supposed to forbid.
    let run = Command::new(env!("CARGO_BIN_EXE_vxc"))
        .arg(&probe)
        .output()
        .expect("run vxc");
    let out = String::from_utf8_lossy(&run.stdout).to_string();
    assert!(
        out.contains("10000"),
        "checked(1000) returns 10000 -- the false assertion changed nothing:\n{out}"
    );
    assert!(
        run.status.success(),
        "and the process exits cleanly despite the violated assertion"
    );
}

/// STATES THE DESIRED BEHAVIOUR, AND FAILS TODAY.
///
/// A false assertion should stop the program. It does not: `assert` lowers to
/// nothing in both backends, so this is a demonstration of the gap rather than a
/// regression guard. Ignored so it does not break CI; run it deliberately:
///
///   cargo test --test integration_test -- --ignored assert
///
/// Delete the `#[ignore]` when runtime checks land -- that is the moment this
/// stops being a wish and starts being a test.
#[test]
#[ignore = "assert emits no runtime check (see module comment); run with --ignored \
            to see the gap"]
fn a_false_runtime_assert_should_abort() {
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
