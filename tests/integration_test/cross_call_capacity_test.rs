//===- cross_call_capacity_test.rs - Vx Compiler ----------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// The cross-call capacity fold on the *parallel* frontend. The frontend tests
// exercise it through `vxc` (the sequential driver); these run the same two
// programs through `compile_pipeline_type_stream`, whose per-function phase is
// the rayon fan-out -- proving the fold reads worker-exported summaries, not
// state that only accumulates on a single sequential checker.
//
//===----------------------------------------------------------------------===//

use std::fs;

const HEADER: &str = r#"
Memory CPU_DRAM {}
Memory W {
  within: Memory::CPU_DRAM, capacity: 4 MiB, managed: explicit
}
Topology Dev {
  arch: nvptx64,
  memory: Memory::W,
  visible: [Memory::W],
  transfer Memory::CPU_DRAM -> Memory::W : 10
  transfer Memory::W -> Memory::CPU_DRAM : 10
}
"#;

fn compile(name: &str, body: &str) -> Result<(), String> {
    let dir = std::env::temp_dir().join("vx_cross_call_capacity");
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    fs::write(&path, format!("{HEADER}{body}")).unwrap();
    vxc::pipeline::compile_pipeline_type_stream(&[path.to_string_lossy().into_owned()])
        .map(|_| ())
        .map_err(|e| format!("{e:?}"))
}

/// The admit behind the fold: each function fits alone, their composition does
/// not, and the parallel frontend must refuse it too.
#[test]
fn the_parallel_frontend_refuses_a_tile_held_across_a_call() {
    let r = compile(
        "nested.vx",
        r#"
fn g() -> i32 {
  let y = Tensor<f32, [1024, 768]>::uninit();
  let _sy = transfer(y, Memory::W);
  return 1;
}
fn f() -> i32 {
  let x = Tensor<f32, [1024, 768]>::uninit();
  let sx = transfer(x, Memory::W);
  let r = g();
  let _back = transfer(sx, Memory::CPU_DRAM);
  return r;
}
fn main() -> i32 { return f(); }
"#,
    );
    assert!(
        r.is_err(),
        "two 3 MiB tiles coexist across the call in a 4 MiB space; the parallel \
         frontend admitted it"
    );
}

/// The mirror obligation: sequential calls compose by max, and the fold must
/// not turn this correct admission into a refusal on either frontend.
#[test]
fn the_parallel_frontend_still_admits_sequential_calls() {
    let r = compile(
        "sequential.vx",
        r#"
fn g() -> i32 {
  let y = Tensor<f32, [1024, 768]>::uninit();
  let _sy = transfer(y, Memory::W);
  return 1;
}
fn f() -> i32 {
  let x = Tensor<f32, [1024, 768]>::uninit();
  let _sx = transfer(x, Memory::W);
  return 2;
}
fn main() -> i32 {
  let a = f();
  let b = g();
  return a + b;
}
"#,
    );
    assert!(
        r.is_ok(),
        "f returns before g runs, the peak is one tile: {r:?}"
    );
}
