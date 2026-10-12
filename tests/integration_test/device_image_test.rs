//===- device_image_test.rs - the compiler emits a real device kernel -----===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A GPU-placed region that no vendor library can stand in for is compiled to
// PTX and carried in the dispatch payload (#251).
//
// These tests check the two facts that matter: an image exists, and it is the
// kernel rather than something shaped like one.
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
    let ir = emit_llvm("placed_kernel_four_operands.vx");

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

/// A region that calls functions -- `ln` reaches `core::libm`'s `logf`, which reads two
/// constant tables -- still gets a device image: the functions and the tables are copied into
/// the device module beside the kernel. Before, any call kept the region on the host.
#[test]
fn a_placed_kernel_that_calls_functions_gets_a_device_image() {
    let ir = emit_llvm("placed_kernel_calls_core_libm.vx");
    assert!(
        ir.contains("image="),
        "the launch payload carries no device image"
    );
    let image = extract_image(&ir);
    for table in ["__vx_const_LIBM_COMMON_R", "__vx_const_LIBM_COMMON_LOG_R"] {
        assert!(
            image.contains(table),
            "the device image does not hold `{table}`, which `logf` reads"
        );
    }
    // Every call left in the image is to libdevice, which the device pipeline links when the
    // machine has it; none is to a host function the device could not reach.
    for line in image.lines().filter(|l| l.contains(".extern .func")) {
        assert!(
            line.contains("__nv_"),
            "the device image calls a function it does not contain: {line}"
        );
    }
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
    // The visibility barrier, closed in #353: the builtin copy publishes the tile to every
    // lane before any lane reads it. Exactly one -- the count is what separates the
    // builtin from a user lowering (see the user-lowering test below, which pins two).
    assert_eq!(
        image.matches("bar.sync").count(),
        1,
        "the builtin SMEM copy must end in exactly one barrier:\n{image}"
    );
    // What the image must NOT contain yet, pinned so the day it appears the change is
    // deliberate: no cp.async (nothing emits it -- the fact that falsified
    // `crossing: streamed`; raw::async_copy lowers synchronously until
    // the nvgpu route exists).
    assert!(
        !image.contains("cp.async"),
        "nothing emits cp.async today; if this appears, the crossing model wants re-scoring"
    );
}

/// A user-supplied `impl transfer` fills the SMEM tile, and the emitted image proves it
/// ran: TWO `bar.sync` where the builtin has one (Vx#353).
///
/// The discriminator is structural on purpose. A transfer is a move, not a conversion
/// (value preservation), so a faithful user lowering computes exactly what the builtin computes
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
/// the flip its predecessor pre-authorized, earned in #353.
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

/// The dispatch payload global's bytes, decoded from the emitted LLVM IR.
///
/// The payload is one string constant, printed escaped: `\XX` for a byte that
/// is not printable, and `\\` for a backslash, which would otherwise end the
/// literal. A quote inside is printed `\22`, so the literal's own closing quote
/// is the first `"` after it. What is under test is the payload's own format,
/// not MLIR's escaping, so decode it back to bytes rather than match on the
/// text.
fn payload_bytes(ir: &str, kernel: &str) -> Vec<u8> {
    let marker = format!("@{kernel}_str(\"");
    let start = ir
        .find(&marker)
        .unwrap_or_else(|| panic!("no dispatch payload global for {kernel}"))
        + marker.len();
    let rest = &ir[start..];
    let end = rest
        .find('"')
        .expect("the payload global is not terminated");
    let escaped = &rest.as_bytes()[..end];

    let mut bytes = Vec::new();
    let mut i = 0;
    while i < escaped.len() {
        if escaped[i] == b'\\' {
            match &escaped[i + 1..] {
                [b'\\', ..] => {
                    bytes.push(b'\\');
                    i += 2;
                    continue;
                }
                [a, b, ..] => {
                    let digits = [*a, *b];
                    let hex = std::str::from_utf8(&digits).unwrap_or("");
                    if let Ok(byte) = u8::from_str_radix(hex, 16) {
                        bytes.push(byte);
                        i += 3;
                        continue;
                    }
                }
                _ => {}
            }
        }
        bytes.push(escaped[i]);
        i += 1;
    }
    bytes
}

/// The payload's section, walked the way a dispatch library walks it
/// (`vx_payload_text_end` and `vx_payload_section` in
/// include/vx_hardware_runtime.h): the text part is the kernel name and the
/// `key=value` entries, an empty entry ends it, and the section is a
/// little-endian 64-bit length followed by that many bytes.
///
/// Written out here rather than shared with the runtime, so that the two are
/// two statements of the same format rather than one.
fn payload_section(payload: &[u8]) -> &[u8] {
    let mut pos = payload
        .iter()
        .position(|&b| b == 0)
        .expect("the payload's kernel name is unterminated")
        + 1;
    loop {
        let len = payload[pos..]
            .iter()
            .position(|&b| b == 0)
            .expect("the payload's entries are unterminated");
        if len == 0 {
            break; // the empty entry: the section starts after it
        }
        pos += len + 1;
    }
    let start = pos + 1;

    let length = u64::from_le_bytes(
        payload[start..start + 8]
            .try_into()
            .expect("the payload is too short to hold the section's length"),
    ) as usize;
    assert_eq!(
        length,
        payload.len() - start - 8,
        "the section's length does not account for the rest of the payload"
    );
    &payload[start + 8..]
}

/// Ask `spirv-val` about a module, or `None` when this machine has no
/// `spirv-val` to ask.
///
/// The module goes in on standard input rather than through a file: the SPIR-V
/// tests run in parallel, and one shared file name would let either overwrite
/// the other's module between the write and the check.
fn spirv_val(module: &[u8]) -> Option<Result<(), String>> {
    use std::io::Write;
    use std::process::Stdio;

    let mut child = Command::new("spirv-val")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .ok()?;
    // Taking the handle and letting it drop closes the pipe, so the validator
    // sees the end of the module rather than waiting for more.
    child
        .stdin
        .take()
        .expect("spirv-val's stdin was piped")
        .write_all(module)
        .ok()?;
    let out = child.wait_with_output().ok()?;
    if out.status.success() {
        Some(Ok(()))
    } else {
        Some(Err(String::from_utf8_lossy(&out.stderr).into_owned()))
    }
}

/// A topology whose machine model declares `arch: spirv64` gets a SPIR-V
/// module, and it is carried in the payload's section rather than in an
/// `image=` entry (#1137).
///
/// The assertions are on the module's *content*, because a section that held
/// four right bytes and then nothing, or a module that never named the kernel
/// it is entered through, would satisfy "a section exists" and then fail at a
/// loader in another process. SPIR-V's own validator is the strongest claim
/// available here -- not "these bytes look like SPIR-V" but "this is a
/// module", which is the thing the memref-carried form of this kernel failed.
///
/// The section is the only shape that can carry the image: its first word is
/// `0x07230203`, which holds NUL bytes, so an `image=` entry would end at the
/// module's first word.
#[test]
fn a_spirv_topology_gets_a_spirv_module_in_the_payload_section() {
    let ir = emit_llvm("spirv_device_image.vx");
    let kernel = "vx_npu_kernel_0";
    let payload = payload_bytes(&ir, kernel);

    // The text part: the kernel name first, then the entries.
    assert!(
        payload.starts_with(format!("{kernel}\0").as_bytes()),
        "the payload does not lead with the kernel name, which is what selects \
         an entry point out of a loaded module"
    );
    let has = |needle: &[u8]| payload.windows(needle.len()).any(|w| w == needle);
    assert!(
        has(b"format=spirv\0"),
        "the payload does not say the image is SPIR-V, so a dispatch library \
         has to guess"
    );
    assert!(
        !has(b"image="),
        "the image is in a NUL-terminated `image=` entry, where its own first \
         word would end it"
    );

    let section = payload_section(&payload);
    assert_eq!(
        &section[..4],
        &[0x03, 0x02, 0x23, 0x07],
        "the section does not start with SPIR-V's magic number 0x07230203"
    );
    assert!(
        section
            .windows(kernel.len())
            .any(|w| w == kernel.as_bytes()),
        "the module does not name `{kernel}`, so a loader would load it and then \
         ask for a function that is not in it"
    );

    match spirv_val(section) {
        Some(Ok(())) => {}
        Some(Err(report)) => {
            panic!("spirv-val rejected the module the compiler emitted:\n{report}")
        }
        None => println!("spirv-val is not installed; the module was not validated"),
    }
}

/// The shape a real placed kernel has -- four rank-2 operands, a loop nest, and
/// each row read through a view -- also reaches the payload's section as a
/// SPIR-V module the validator accepts (#1137).
///
/// `spirv_device_image.vx` proves the mechanism on two 2x2 tensors. This is the
/// corpus shape `placed_kernel_four_operands.vx` has: four rank-2 operands, 28
/// flat parameters once expanded, so the rewrite has to keep four expansions
/// and the accesses into each apart. The loops' counters start in
/// `memref<i32>` slots (`{vx.parallel_init}` / `{vx.parallel_bound}`, which
/// `mem2reg` has to promote before SPIR-V sees them), and every row is read
/// through a `memref.reinterpret_cast` whose offset has to reach the address
/// arithmetic the flat arguments are addressed through.
#[test]
fn a_loop_and_views_kernel_gets_a_spirv_module_in_the_payload_section() {
    let ir = emit_llvm("spirv_device_image_loop_kernel.vx");
    let kernel = "vx_npu_kernel_0";
    let payload = payload_bytes(&ir, kernel);

    let has = |needle: &[u8]| payload.windows(needle.len()).any(|w| w == needle);
    assert!(
        has(b"format=spirv\0"),
        "the payload does not say the image is SPIR-V, so a dispatch library \
         has to guess"
    );
    assert!(
        !has(b"image="),
        "the image is in a NUL-terminated `image=` entry, where its own first \
         word would end it"
    );

    let section = payload_section(&payload);
    assert_eq!(
        &section[..4],
        &[0x03, 0x02, 0x23, 0x07],
        "the section does not start with SPIR-V's magic number 0x07230203"
    );
    assert!(
        section
            .windows(kernel.len())
            .any(|w| w == kernel.as_bytes()),
        "the module does not name `{kernel}`, so a loader would load it and then \
         ask for a function that is not in it"
    );

    match spirv_val(section) {
        Some(Ok(())) => {}
        Some(Err(report)) => {
            panic!("spirv-val rejected the module the compiler emitted:\n{report}")
        }
        None => println!("spirv-val is not installed; the module was not validated"),
    }
}

/// A kernel that reads a view of a view -- a strided row -- or a row of a
/// run-time-shaped tensor also reaches the payload's section as a SPIR-V module
/// the validator accepts (#1137).
///
/// These are the shapes the flat emitter builds with
/// `memref.extract_strided_metadata` and `memref.dim`, neither of which the
/// SPIR-V argument flattening knew how to rewrite. It refused them, and the
/// refusal was fatal: the whole compile failed rather than the region taking
/// the host path. The fix teaches the flattening both ops, so the kernel stays
/// on the device and the image has to be a real one.
#[test]
fn a_strided_and_dynamic_row_kernel_gets_a_spirv_module_in_the_payload_section() {
    let ir = emit_llvm("spirv_device_image_strided_row.vx");
    let kernel = "vx_npu_kernel_0";
    let payload = payload_bytes(&ir, kernel);

    let has = |needle: &[u8]| payload.windows(needle.len()).any(|w| w == needle);
    assert!(
        has(b"format=spirv\0"),
        "the payload does not say the image is SPIR-V, so a dispatch library \
         has to guess"
    );
    assert!(
        !has(b"image="),
        "the image is in a NUL-terminated `image=` entry, where its own first \
         word would end it"
    );

    let section = payload_section(&payload);
    assert_eq!(
        &section[..4],
        &[0x03, 0x02, 0x23, 0x07],
        "the section does not start with SPIR-V's magic number 0x07230203"
    );
    assert!(
        section
            .windows(kernel.len())
            .any(|w| w == kernel.as_bytes()),
        "the module does not name `{kernel}`, so a loader would load it and then \
         ask for a function that is not in it"
    );

    match spirv_val(section) {
        Some(Ok(())) => {}
        Some(Err(report)) => {
            panic!("spirv-val rejected the module the compiler emitted:\n{report}")
        }
        None => println!("spirv-val is not installed; the module was not validated"),
    }
}

/// A `spirv64` region that yields a value gets no `format=spirv` image, and
/// says why: the value is handed back through a host stack slot, which a device
/// cannot address (#1137).
///
/// The memref-cell warning the legacy path gets has a test of its own; this is
/// its sibling on the flat path, and it is the one the compiler has to make for
/// a value-yielding kernel rather than a memref-shaped one. The message names
/// the host stack slot so the author can tell this refusal from the others.
#[test]
fn a_yielding_spirv_region_says_it_runs_on_the_host() {
    let path = corpus("spirv_device_image_result_slot.vx");
    let out = Command::new(env!("CARGO_BIN_EXE_vxc"))
        .arg(&path)
        .arg("--emit-llvm")
        .output()
        .unwrap_or_else(|e| panic!("could not run vxc: {e}"));
    assert!(
        out.status.success(),
        "the flat path failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let ir = String::from_utf8_lossy(&out.stdout);
    assert!(
        !ir.contains("format=spirv"),
        "a kernel that yields a host-stack value must have no SPIR-V image \
         rather than one that is wrong"
    );

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("host stack slot") && stderr.contains("run on the host"),
        "the compiler dropped this kernel's device twin without saying why, so \
         the author believes it runs on the card; stderr was:\n{stderr}"
    );
}

/// A `--legacy-codegen` compile of the same `spirv64` topology still gets no
/// image, and now says so out loud.
///
/// The AST code generator represents a kernel's tensors as memrefs of memrefs
/// (`memref<memref<...>>`), and SPIR-V has no form for one, so the kernel keeps
/// the host path -- which computes the right answer, one thread at a time, and
/// never touches the card. That is a decision the program's author has to be
/// able to see; before, the twin was simply absent and the output said nothing.
///
/// NVPTX compiles the cell shape (it is the AST path's own PTX image), so the
/// refusal this reports is the SPIR-V arm alone and nothing about NVPTX changes.
#[test]
fn a_legacy_compiled_spirv_kernel_says_it_runs_on_the_host() {
    let path = corpus("spirv_device_image_loop_kernel.vx");
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
        !ir.contains("format=spirv"),
        "the AST path cannot express a memref of memrefs, so there must be no \
         SPIR-V image rather than one that is wrong"
    );

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("memref of memrefs") && stderr.contains("run on the host"),
        "the compiler dropped this kernel's device twin without saying why, so \
         the author believes it runs on the card; stderr was:\n{stderr}"
    );
}
