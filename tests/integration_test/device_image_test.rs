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
/// The PTX out of the payload global: `image=` up to the terminating NUL, unescaped.
fn extract_image(ir: &str) -> String {
    let start = ir.find("image=").expect("payload carries image=") + "image=".len();
    let rest = &ir[start..];
    let end = rest.find("\\00").unwrap_or(rest.len());
    rest[..end]
        .replace("\\0A", "\n")
        .replace("\\09", "\t")
        .replace("\\22", "\"")
}

/// A USER-DECLARED topology whose machine file says `arch: nvptx64` gets a device image,
/// carried in the payload like the built-in GPU's -- and the tile its body places in
/// `Memory::SMEM` materialises as real shared memory in the PTX (Vx#352, #353).
///
/// Before the declared arch travelled onto the outlined kernel, this was impossible by
/// arithmetic, not by omission: eligibility was the dispatch-id band [500, 600), and a custom
/// topology's id is a hash at or above 3000 -- no name can land in the band.
#[test]
fn a_custom_topology_with_a_declared_arch_gets_a_device_image() {
    let ir = emit_llvm("custom_topology_device_image.vx");
    assert!(
        ir.contains("image="),
        "a custom nvptx64 topology must produce a device image"
    );
    let image = extract_image(&ir);
    assert!(
        image.contains(".target sm_80"),
        "the image is real PTX for the default chip:\n{image}"
    );
    // The SMEM placement is not an annotation: the tile lives in `.shared` and is read
    // from there. This is #352's first done-when item, observable.
    assert!(
        image.contains(".shared"),
        "a tile placed in Memory::SMEM must materialise as shared memory:\n{image}"
    );
    assert!(
        image.contains("ld.shared"),
        "the body must READ the tile from shared memory, not re-read global:\n{image}"
    );
    // The C3 barrier, closed in #353 A3: the builtin copy publishes the tile to every
    // lane before any lane reads it. Exactly one -- the count is what separates the
    // builtin from a user lowering (see the user-lowering test below, which pins two).
    assert_eq!(
        image.matches("bar.sync").count(),
        1,
        "the builtin SMEM copy must end in exactly one barrier:\n{image}"
    );
    // What the image must NOT contain yet, pinned so the day it appears the change is
    // deliberate: no cp.async (nothing emits it -- the fact that falsified
    // `crossing: streamed`, vx-review#27; raw::async_copy lowers synchronously until
    // the nvgpu route exists).
    assert!(
        !image.contains("cp.async"),
        "nothing emits cp.async today; if this appears, vx-review#26 wants re-scoring"
    );
}

/// A user-supplied `impl transfer` fills the SMEM tile, and the emitted image proves it
/// ran: TWO `bar.sync` where the builtin has one (Vx#353 A3).
///
/// The discriminator is structural on purpose. A transfer is a move, not a conversion
/// (contract C2), so a faithful user lowering computes exactly what the builtin computes
/// -- the answer cannot distinguish them. The fixture's body says `raw::barrier()` twice,
/// both top-level, and the count survives to PTX.
#[test]
fn a_user_supplied_lowering_is_emitted_instead_of_the_builtin_copy() {
    let ir = emit_llvm("custom_topology_user_lowering.vx");
    assert!(
        ir.contains("image="),
        "a user-lowered transfer must still produce a device image"
    );
    let image = extract_image(&ir);
    // The storage half is unchanged: the site keeps the builtin's allocation, and the
    // static-shape promotion to a real `.shared` global is what made the A100 stop
    // faulting (2b9d4190). A user lowering must not cost that.
    assert!(
        image.contains(".shared .align"),
        "the user-lowered tile must still get real shared STORAGE, not just \
         shared-typed instructions:\n{image}"
    );
    assert!(
        image.contains("ld.shared"),
        "the body must read the tile from shared memory:\n{image}"
    );
    assert_eq!(
        image.matches("bar.sync").count(),
        2,
        "the user lowering's two raw::barrier() calls must both reach the image -- \
         one barrier means the builtin copy ran instead:\n{image}"
    );
}

/// The AST-fallback path now gets a real device image for a statically shaped tile --
/// the flip its predecessor pre-authorized, earned in #353 A3.
///
/// It used to be refused, and the refusal was right at the time: this path typed the SMEM
/// tile as a dynamic memref (`?x?`), a dynamic shared tile cannot become a `.shared`
/// global, and the image it produced carried shared-typed instructions against LOCAL
/// storage -- measured on an A100 as CUDA_ERROR_ILLEGAL_ADDRESS. A3 fixed the cause
/// rather than the symptom: when the source tensor's dims are literals, the transfer's
/// result type is now static here too, so the alloca is static and promotes to real
/// `.shared` storage exactly as on the flat path.
///
/// A tile whose shape stays genuinely dynamic is still refused, and still keeps the
/// program on the host path, which computes the right answer.
#[test]
fn the_ast_codegen_path_materialises_a_static_shared_tile() {
    let path = corpus("custom_topology_device_image.vx");
    let out = Command::new(env!("CARGO_BIN_EXE_vxc"))
        .arg(&path)
        .arg("--legacy-codegen")
        .arg("--emit-llvm")
        .output()
        .unwrap_or_else(|e| panic!("could not run vxc: {e}"));
    assert!(
        out.status.success(),
        "legacy path failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let ir = String::from_utf8_lossy(&out.stdout);
    assert!(
        ir.contains("image="),
        "a statically shaped SMEM tile must now materialise on the AST path too"
    );
    let image = extract_image(&ir);
    // The storage declaration, not the substring: shared-typed instructions over local
    // storage is what faulted on the A100, and a `.shared`-substring check waved it through.
    assert!(
        image.contains(".shared .align"),
        "the AST path's image must declare real shared STORAGE:\n{image}"
    );
}

/// The same program whose topology declares NO arch produces no image: the compiler refuses
/// to guess what code to emit for a machine that did not say. The band fallback covers the
/// built-in GPU; a silent guess here would be the same category error the band was --
/// deciding device compilation on something other than the declaration.
#[test]
fn a_custom_topology_without_an_arch_stays_on_the_host() {
    let ir = emit_llvm("custom_topology_no_arch.vx");
    assert!(
        !ir.contains("image="),
        "no declared arch, no device image -- refusing to guess"
    );
}

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
