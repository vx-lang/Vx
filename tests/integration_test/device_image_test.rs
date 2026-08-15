//===- device_image_test.rs - the compiler emits a real device kernel -----===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// A GPU-placed region that no vendor library can stand in for is compiled to
// PTX and carried in the dispatch payload (#251).
//
// scripts/flash_kernel_to_ptx.sh drives the same kernel through the same passes
// from outside the compiler, and the two agree byte for byte -- that comparison
// is what established the pipeline. It cannot run in CI, though: it needs
// `mlir-opt` on PATH. These tests are the part that can, and they check the two
// facts that matter for the next step: an image exists, and it is the kernel
// rather than something shaped like one.
//
// The negative is as load-bearing as the positive. A classified matmul is left
// at `linalg` level on purpose so the plugin can route it to cuBLAS, which
// beats anything we would emit; giving it a device twin would be both wasted
// compile time and a worse kernel. That case broke six backend tests when the
// image compile first landed, because the `gpu.func` for it was never
// compilable and dropping it unexamined had hidden that. `test_backend` in
// compile_test.rs is what covers the whole corpus for that regression; these
// two cover the mechanism.
//
// Through the binary rather than the library, because the two codegen paths do
// not lower this region alike. The flat path -- the default, and what ships --
// writes the result into a buffer the caller passed in, so the PTX has 48
// `st.global`. The AST path takes an output slot instead: it allocates on the
// device and stores a descriptor into the slot, so the same program's PTX has
// none, and the result lives in memory only that descriptor names. That is the
// slot-output shape, the part of #251 with a design question still in it, and a
// test that built the module in-process would silently measure whichever path
// the harness happened to use.
//
//===----------------------------------------------------------------------===//

use std::path::{Path, PathBuf};
use std::process::Command;

fn corpus(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/backend/pass")
        .join(name)
}

/// Compile a corpus program to LLVM IR, which is where the dispatch payload has
/// become a global and the device image with it.
fn emit_llvm(name: &str) -> String {
    let path = corpus(name);
    let out = Command::new(env!("CARGO_BIN_EXE_vxc"))
        .arg(&path)
        .arg("--emit-llvm")
        .output()
        .unwrap_or_else(|e| panic!("could not run vxc: {e}"));
    assert!(
        out.status.success(),
        "vxc --emit-llvm failed on {}:\n{}",
        path.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// A placed region with no library equivalent reaches the payload as PTX.
///
/// The assertions are on the *content* rather than on the presence of an
/// `image=` field, because an empty or truncated image would satisfy the field
/// and then fail at the worker, a machine and a wire message away from the
/// cause.
#[test]
fn a_placed_kernel_is_compiled_to_ptx_and_carried_in_the_payload() {
    let ir = emit_llvm("flash_attention_placed_verified.vx");

    assert!(
        ir.contains("image="),
        "the launch payload carries no device image"
    );
    // The PTX lands in the payload constant escaped, but only its
    // non-printables are: directives survive as themselves.
    for want in [
        ".target sm_80",
        ".visible .entry ",
        // The loop, not a stub: operands read, the result written back to the
        // caller's memory, branches kept. A kernel that only writes to a device
        // allocation of its own has computed nothing anyone can collect.
        "ld.global",
        "st.global",
        "bra",
    ] {
        assert!(
            ir.contains(want),
            "the device image has no `{want}` -- this is not the kernel"
        );
    }

    // The payload leads with the kernel name, and that name is what selects an
    // entry point out of a loaded module, so the two have to agree. Read rather
    // than asserted literally: the outliner's counter says which
    // `vx_npu_kernel_N` this is, and that is not this test's business.
    let entry = ir
        .split(".visible .entry ")
        .nth(1)
        .and_then(|rest| rest.split('(').next())
        .expect("no entry point in the device image")
        .to_string();
    assert!(
        entry.starts_with("vx_"),
        "the entry point is named `{entry}`, which is not one of ours"
    );
    assert!(
        ir.contains(&format!("@{entry}_str")),
        "the image's entry point is `{entry}`, but no dispatch payload names it \
         -- a worker would load the module and then ask for a function that is \
         not in it"
    );

    // The signature the kernel declares, counted the way vx_kernel_launch.h
    // counts it: within the parentheses, so `ld.param` uses and externs' return
    // slots are not mistaken for parameters.
    //
    // 28 is four rank-2 memrefs at seven parameters each -- two pointers, an
    // offset, two sizes, two strides. The other half of that equality is
    // asserted in tests/runtime/kernel_launch_test.cpp, which builds a
    // parameter list for this exact signature and gets 28 from the argument
    // side. A launch is only safe while the two agree, and they are computed
    // from different things: this one from the compiler's PTX, that one from
    // the ABI tags.
    let signature = ir
        .split(".visible .entry ")
        .nth(1)
        .and_then(|rest| rest.split(')').next())
        .expect("no entry signature in the device image");
    assert_eq!(
        signature.matches(".param").count(),
        28,
        "the kernel's signature changed; tests/runtime/kernel_launch_test.cpp \
         builds 28 parameters for four rank-2 memrefs and the two must agree"
    );
}

/// A classified matmul carries no image, and still says it is a matmul.
#[test]
fn a_matmul_is_left_to_the_vendor_library() {
    let ir = emit_llvm("gpu_matmul_roles.vx");

    assert!(
        ir.contains("kind=matmul"),
        "the launch is no longer classified as a matmul, so this test no longer \
         covers the case it was written for"
    );
    assert!(
        !ir.contains("image="),
        "a classified matmul was given a device image; cuBLAS beats anything we \
         emit today (#321) and its `linalg` body cannot compile for a device \
         anyway"
    );
}
