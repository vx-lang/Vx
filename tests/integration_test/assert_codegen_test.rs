//===- assert_codegen_test.rs - `assert` emits no runtime check ----------===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// `assert(cond)` / `assert(cond, "msg")` emits a real runtime check (Vx#361),
// built on `abort()` -- the language's own termination primitive.
//
// `assert` is a CONDITIONAL ABORT and is lowered as exactly that: the flat path
// emits the pieces (a branch, the message, `Opcode::Abort`), and the AST path
// emits one `cf.assert`, which `convert-cf-to-llvm` expands into the identical
// branch-print-abort shape. Same behaviour, different amount written by hand.
//
// `abort()` is callable directly and is SAFE: ending a process violates no
// memory-safety property, which is why `std::process::abort` needs no `unsafe`
// in Rust either.
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
/// The two paths spell it differently, and that is by design: `assert` is a
/// CONDITIONAL ABORT, so the flat path lowers it as the pieces -- a branch, the
/// message, and `Opcode::Abort` -- while the AST path emits one `cf.assert`,
/// which `convert-cf-to-llvm` expands into that same branch-print-abort shape.
/// So this asserts the SHAPE, not one spelling: the condition is tested, and the
/// failing edge reaches `abort`.
///
/// It asserted the opposite until Vx#361 -- that no check existed at all -- and
/// was written to go red on exactly this change.
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
        ir_text.contains("arith.cmpi slt"),
        "the condition the programmer wrote must be tested, not a stand-in:\n{ir_text}"
    );
    assert!(
        ir_text.contains("cf.assert") || ir_text.contains("@abort"),
        "and the failing edge must reach abort -- as one `cf.assert` (AST path) or as \
         an explicit branch onto `@abort` (flat path):\n{ir_text}"
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

/// `abort()` is callable on its own, and it is SAFE.
///
/// The primitive is the point: a program should be able to terminate itself, and
/// `assert` is then one use of it rather than a construct the backend has to know
/// about specially. Safety follows the same reasoning Rust uses for
/// `std::process::abort` -- ending a process violates no memory-safety property --
/// so this compiles with no `unsafe` block. (Reaching it through an `extern`
/// declaration instead WOULD require one, but that is the FFI boundary's rule,
/// not termination's.)
#[test]
fn abort_is_callable_and_needs_no_unsafe() {
    let src = r#"
fn main() -> i32 {
  let mut t = Tensor<i32>([ 1 ]);
  t[0] = 3;
  if t[0] > 1 {
    abort();
  }
  print(999);
  return 0;
}
"#;
    let probe = write_probe("abort_builtin", src);

    // Both backends: terminates, and never reaches the print.
    for flags in [vec![], vec!["--legacy-codegen"]] {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_vxc"));
        cmd.arg(&probe);
        for f in &flags {
            cmd.arg(f);
        }
        let run = cmd.output().expect("run vxc");
        let out = String::from_utf8_lossy(&run.stdout).to_string();
        let err = String::from_utf8_lossy(&run.stderr).to_string();
        let path = if flags.is_empty() { "flat" } else { "AST" };
        assert!(
            !out.contains("999"),
            "{path}: abort() must not fall through to the print:\n{out}{err}"
        );
        assert!(
            !run.status.success(),
            "{path}: abort() must fail the process:\n{out}{err}"
        );
        assert!(
            !err.contains("unsafe") && !err.contains("E5001"),
            "{path}: terminating is safe -- no unsafe block should be demanded:\n{err}"
        );
    }
}

/// An assert inside a `spawn` body is a REAL assert, on both paths.
///
/// It must reach device code as `cf.assert` and must NOT reach it as a host
/// call. The distinction is the whole point: `cf` is device-lowerable, and
/// `convert-gpu-to-nvvm` (already in the device pipeline) expands `cf.assert`
/// into `__assertfail` with the message, file, line and `noreturn`. A
/// hand-desugared `print_str` + `abort` pair is `func` ops, which are NOT
/// device-lowerable -- a kernel containing one is classified not-device-ready
/// and dropped from GPU compilation entirely, silently.
///
/// This test has been three things in three commits, which is the record worth
/// keeping. First it asserted asserts emit nothing (true then). Then it asserted
/// they emit nothing IN KERNELS, after a hand-desugared lowering made kernel
/// asserts unsafe -- correct for that lowering, but the guard was reasoning from
/// the HOST pipeline's behaviour applied to device code. Now it asserts what the
/// portable op actually delivers (Vx#362).
#[test]
fn an_assert_inside_a_kernel_is_a_real_assert() {
    let src = r#"
Memory CPU_DRAM {}
Memory GPU_HBM {
  within: Memory::CPU_DRAM, capacity: 40 GiB, bandwidth: 3 TB/s
}
fn main() -> i32 {
  let mut a = Tensor<f32>([ 2, 2 ]);
  a[0][0] = 1.0;
  let mut ad = transfer(a, Memory::GPU_HBM);
  spawn on(Topology::GPU) {
    let n = 3;
    assert(n < 5, "kernel assert");
    ad[0][0] = 2.0;
  }
  return 0;
}
"#;
    let probe = write_probe("kernel_assert", src);
    for flags in [vec![], vec!["--legacy-codegen"]] {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_vxc"));
        cmd.arg(&probe).arg("--action").arg("emit-mlir");
        for f in &flags {
            cmd.arg(f);
        }
        let out = cmd.output().expect("run vxc");
        let ir = String::from_utf8_lossy(&out.stdout).to_string()
            + &String::from_utf8_lossy(&out.stderr);
        let path = if flags.is_empty() { "flat" } else { "AST" };
        assert!(
            ir.contains("cf.assert"),
            "{path}: a kernel assertion must survive as the device-lowerable op:\n{ir}"
        );
        for host_only in ["func.call @abort", "func.call @print_str"] {
            assert!(
                !ir.contains(host_only),
                "{path}: `{host_only}` is a `func` op -- not device-lowerable, so a kernel \
                 holding one is dropped from GPU compilation entirely:\n{ir}"
            );
        }
    }
}
