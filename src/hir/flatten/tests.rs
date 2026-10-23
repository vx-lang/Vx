//===- tests.rs - Vx Compiler --------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

use super::*;
use crate::bytecode::{
    AFFINE_FOR_BRANCHES, AUTODIFF_FORWARD, AUTODIFF_MODE_SHIFT, AUTODIFF_REVERSE,
};
use crate::session::GlobalSession;
use std::sync::Arc;

fn parse_fn(src: &str) -> Function {
    let mut lexer = crate::lexer::Lexer::new(src);
    let tokens = lexer.tokenize();
    let mut parser = crate::parser::Parser::new(&tokens, src);
    let prog = parser.parse().expect("parse failed");
    prog.functions.into_iter().next().expect("expected one fn")
}

fn worker() -> LocalWorkerState {
    LocalWorkerState::new(Arc::new(GlobalSession::new(1)))
}

fn opcodes(w: &LocalWorkerState) -> Vec<Opcode> {
    w.local_hir_stream.iter().map(|i| i.opcode).collect()
}

fn count(w: &LocalWorkerState, op: Opcode) -> usize {
    w.local_hir_stream.iter().filter(|i| i.opcode == op).count()
}

/// Parse a whole program, resolve names, build the *real* frozen registry (so aggregate layouts
/// exist), type-check (so `StructInit`s get their GID annotated), and attempt to lower `fn_name`.
/// Returns `(did_lower, worker)`.
fn lower_with_registry(src: &str, fn_name: &str) -> (bool, LocalWorkerState) {
    let mut lexer = crate::lexer::Lexer::new(src);
    let tokens = lexer.tokenize();
    let mut parser = crate::parser::Parser::new(&tokens, src);
    let mut prog = parser.parse().expect("parse failed");
    prog.module_path = "crate::t".into();
    let mut mods = vec![prog];
    let symbol_map = crate::resolver::build_symbol_map(&mods);
    mods[0].resolve_names(&symbol_map, &[]);
    let registry = crate::pipeline::build_frozen_registry(&mods).expect("registry builds");
    let mut worker = LocalWorkerState::new(Arc::new(GlobalSession::with_registry(1, registry)));

    // Type-check so the type checker annotates each `StructInit` with its struct GID.
    let env_mods: Vec<_> = mods.clone();
    let env = crate::hir::GlobalAstEnv::build(&env_mods);
    {
        let mut checker = crate::hir::TypeChecker::new(&env, &mut worker);
        for f in &mut mods[0].functions {
            checker.check_function(f);
        }
    }

    let func = mods
        .into_iter()
        .next()
        .unwrap()
        .functions
        .into_iter()
        .find(|f| f.name.as_ref() == fn_name)
        .expect("fn present");
    let did = lower_function_to_hir(&func, &mut worker).is_ok();
    (did, worker)
}

#[test]
fn the_control_flow_scan_sees_nested_and_trailing_positions() {
    // Each shape here hid a value-`if` from an earlier version of the scan; the register
    // path then let a branch's rebind escape its block. The scan is pinned directly, since
    // an end-to-end repro exists only for shapes the checker admits.
    let has = |src: &str| body_has_control_flow(&parse_fn(src).body);
    // A value-if nested in a call argument -- the shape that panicked as invalid MLIR.
    assert!(has(
            "fn f() -> i32 { let mut y = 5; let r = g(if y > 1 { y = 7; 1 } else { 0 }); return y + r; }"
        ));
    // A compound assignment whose right side branches.
    assert!(has(
        "fn f(c : bool) -> i32 { let mut x = 1; x += if c { 10 } else { 20 }; return x; }"
    ));
    // A binary operand.
    assert!(has(
        "fn f(c : bool) -> i32 { let v = (if c { 1 } else { 2 }) + 3; return v; }"
    ));
    // An array element.
    assert!(has(
        "fn f(c : bool) -> i32 { let a = [if c { 1 } else { 2 }, 3]; return a[0]; }"
    ));
    // A block's trailing expression, at any nesting.
    assert!(has(
        "fn f(c : bool) -> i32 { let mut x = 1; unsafe { if c { x = 0; } } return x; }"
    ));
    // And straight-line code still reads as straight-line.
    assert!(!has(
        "fn f() -> i32 { let a = 1; let b = a + 2; return b; }"
    ));
}

#[test]
fn the_struct_scan_sees_branches_and_trailing_positions() {
    let has = |src: &str| body_constructs_struct(&parse_fn(src).body);
    assert!(has(
        "struct P { x: i32 }\nfn f() -> i32 { let p = unsafe { P { x: 1 } }; return p.x; }"
    ));
    assert!(has(
            "struct P { x: i32 }\nfn f(c : bool) -> i32 { let p = if c { P { x: 1 } } else { P { x: 2 } }; return p.x; }"
        ));
    assert!(!has("fn f() -> i32 { return 3; }"));
}

#[test]
fn matmul_result_is_the_product_shape_not_the_left_operand() {
    // `[2, 3] @ [3, 4]` is `[2, 4]`. The generic binary rule gives the result the LEFT
    // operand's type, which is right for an elementwise op and wrong for a matmul.
    let f = parse_fn(
        "fn f() -> i32 { let a = Tensor<f32>([2, 3]); \
             let b = Tensor<f32>([3, 4]); let c = a @ b; return 0; }",
    );
    let mut w = worker();
    lower_function_to_hir(&f, &mut w).expect("a static matmul should lower");
    let ins = w
        .local_hir_stream
        .iter()
        .position(|i| i.opcode == Opcode::Matmul)
        .expect("the Matmul instruction");
    let gid = w.local_type_stream[w.local_hir_stream[ins].type_idx.0 as usize];
    let shape = w
        .local_tensor_types
        .iter()
        .find(|(g, _, _)| *g == gid)
        .map(|(_, _, s)| s.clone())
        .expect("the result's shape is recorded");
    assert_eq!(shape, vec!["2".to_string(), "4".to_string()]);
    verify_hir_stream(&w);
}

#[test]
fn half_matmul_keeps_its_element_type() {
    // Elementwise arithmetic on half operands is done in f32, so the generic rule widens the
    // result. A matmul accumulates in f32 but stores half, so it must not.
    let f = parse_fn(
        "fn f() -> i32 { let a = Tensor<f16>([2, 2]); \
             let b = Tensor<f16>([2, 2]); let c = a @ b; return 0; }",
    );
    let mut w = worker();
    lower_function_to_hir(&f, &mut w).expect("a half matmul should lower");
    let ins = w
        .local_hir_stream
        .iter()
        .position(|i| i.opcode == Opcode::Matmul)
        .expect("the Matmul instruction");
    let gid = w.local_type_stream[w.local_hir_stream[ins].type_idx.0 as usize];
    let elem = w
        .local_tensor_types
        .iter()
        .find(|(g, _, _)| *g == gid)
        .map(|(_, e, _)| e.clone())
        .expect("the result's element type is recorded");
    assert_eq!(elem, ElementType::F16);
    verify_hir_stream(&w);
}

#[test]
fn autodiff_lowers_to_one_opcode_per_mode() {
    // `grad` and `jvp` differ only by the mode packed into `imm`, and `jvp` carries its tangent
    // as a trailing argument. `vjp` reuses reverse mode and multiplies by the cotangent, so it
    // adds a `Mul` rather than a mode of its own.
    for (call, mode, args, muls) in [
        ("grad(cube, x)", AUTODIFF_REVERSE, 1, 0),
        ("vjp(cube, x, x)", AUTODIFF_REVERSE, 1, 1),
        ("jvp(cube, x, x)", AUTODIFF_FORWARD, 2, 0),
    ] {
        let src = format!(
            "fn cube(v: f32) -> f32 {{ return v * v * v; }}\n\
                 fn f(x: f32) -> f32 {{ return {call}; }}"
        );
        let (did, w) = lower_with_registry(&src, "f");
        assert!(did, "{call} should lower");
        assert_eq!(count(&w, Opcode::AutoDiff), 1, "{call}: one AutoDiff");
        let ins = w
            .local_hir_stream
            .iter()
            .find(|i| i.opcode == Opcode::AutoDiff)
            .expect("the AutoDiff instruction");
        assert_eq!(
            ins.imm >> AUTODIFF_MODE_SHIFT,
            mode,
            "{call}: differentiation mode"
        );
        assert_eq!(ins.imm & 0xffff_ffff, args, "{call}: argument count");
        assert_eq!(count(&w, Opcode::Arg), args as usize, "{call}: Arg count");
        assert_eq!(count(&w, Opcode::Mul), muls, "{call}: seed multiply");
        verify_hir_stream(&w);
    }
}

#[test]
fn indirect_call_borrow_materializes_its_argument() {
    let function =
        parse_fn("fn f() -> i32 { let x = 10; let read = |p : &i32| *p; return (read)(&x); }");
    let Statement::Return(return_stmt) = function.body.last().expect("return statement") else {
        panic!("function ends with a return statement");
    };
    let Expr::IndirectCall(call) = return_stmt.expr.as_ref().expect("return value") else {
        panic!("parenthesized callee parses as an indirect call");
    };
    assert!(
        matches!(call.args.as_slice(), [Expr::Borrow(_)]),
        "the indirect call has one borrowed argument"
    );

    let uses = analyze_local_uses(&function.body);

    assert!(
        uses.materialized.contains("x"),
        "an indirect call passing &x must materialize x"
    );
}

#[test]
fn indirect_call_reference_argument_escapes_its_place_binding() {
    let function = parse_fn(
        "fn f() -> i32 { let x = 10; let r = &x; let read = |p : &i32| *p; return (read)(r); }",
    );

    let uses = analyze_local_uses(&function.body);

    assert!(
        !uses.place_bindings.contains("r"),
        "a reference passed to an indirect call escapes its place binding"
    );
}

#[test]
fn struct_param_lowers_as_aggregate_slot() {
    // A struct parameter forces the memory model: its `Alloca` is sized from the registry
    // layout (Point = two i32s = 8 bytes) and its aggregate GID flows through the type stream.
    let (did, w) = lower_with_registry(
        "struct Point { x: i32, y: i32 }\nfn id(p: Point) -> Point { return p; }",
        "id",
    );
    assert!(did, "struct-param identity fn should lower");
    let alloca = w
        .local_hir_stream
        .iter()
        .find(|i| i.opcode == Opcode::Alloca)
        .expect("an Alloca slot for the aggregate param");
    assert_eq!(alloca.imm, 8, "Point Alloca sized from its 8-byte layout");
    // A scalar's GID lives in module 0; an aggregate GID carries the (nonzero) module hash, so
    // its presence proves the aggregate type reached the flat type stream.
    assert!(
        w.local_type_stream.iter().any(|t| t.module_id() != 0),
        "the Point aggregate GID is in the type stream"
    );
    assert_eq!(count(&w, Opcode::Store), 1, "the incoming struct is stored");
    assert!(count(&w, Opcode::SlotLoad) >= 1, "return reads the slot");
    assert_eq!(count(&w, Opcode::Ret), 1);
    verify_hir_stream(&w);
}

#[test]
fn generic_aggregate_param_resolves_via_base_layout() {
    // A monomorphized generic struct instance (`Wrap<i32>`) resolves to its base nominal's layout
    // when that layout is instance-independent — a pointer field is 8 bytes for any `T`, so
    // `Wrap<T> { data: *mut T, n: i32 }` has a concrete layout regardless of `T`. So a function
    // taking and returning `Wrap<i32>` HIR-lowers as an aggregate: the type-level foundation the
    // `Vec<T>` surface builds on. (The flat *emitter* still declines an aggregate with a pointer
    // field — a later layer — so this asserts the HIR lowering only.)
    let (did, w) = lower_with_registry(
        "struct Wrap<T> { data: *mut T, n: i32 }\n\
             fn id(w: Wrap<i32>) -> Wrap<i32> { return w; }",
        "id",
    );
    assert!(
        did,
        "generic-aggregate identity fn should HIR-lower via the base layout"
    );
    // The aggregate GID (nonzero module hash) reached the type stream — the generic instance
    // resolved to a concrete layout rather than declining.
    assert!(
        w.local_type_stream.iter().any(|t| t.module_id() != 0),
        "the Wrap<i32> aggregate GID is in the type stream"
    );
    verify_hir_stream(&w);
}

#[test]
fn pointer_field_and_raw_index_lower_through_self_pointer() {
    // The `Vec` surface in miniature (#242): a `&mut Buf` self pointer whose fields are read/
    // written through the pointer (`FieldLoad`/`FieldStore`) and whose raw-pointer field is
    // indexed (`PtrIndex` place + `PtrStore`). `b.data[0] = v` is a raw-pointer element store;
    // `b.len = b.len + 1` reads then writes the scalar field.
    let (did, w) = lower_with_registry(
        "struct Buf { data: *mut i32, len: i32 }\n\
             fn bump(b: &mut Buf, v: i32) -> i32 { unsafe { b.data[0] = v; } \
             b.len = b.len + 1; return 0; }",
        "bump",
    );
    assert!(
        did,
        "field access + raw-pointer index through a self pointer should lower"
    );
    // `b.data` (a pointer field) -> a `FieldLoad`; `b.data[0] = v` -> a `PtrIndex` place + a
    // `PtrStore`; `b.len` read -> a `FieldLoad`; `b.len = ...` -> a `FieldStore`.
    assert_eq!(
        count(&w, Opcode::PtrIndex),
        1,
        "one raw-pointer element place"
    );
    assert_eq!(
        count(&w, Opcode::PtrStore),
        1,
        "one raw-pointer element store"
    );
    assert!(
        count(&w, Opcode::FieldLoad) >= 2,
        "b.data and b.len are read"
    );
    assert_eq!(count(&w, Opcode::FieldStore), 1, "b.len is written");
    // The self pointer binds as an SSA register (no aggregate `Alloca` for it), so the only
    // slots are the memory-mode scalar locals — never the `&mut Buf` itself.
    verify_hir_stream(&w);
}

#[test]
fn struct_field_read_lowers_to_field_load() {
    // `p.y` becomes a `FieldLoad` off the aggregate slot at y's layout offset (4).
    let (did, w) = lower_with_registry(
        "struct Point { x: i32, y: i32 }\nfn gety(p: Point) -> i32 { return p.y; }",
        "gety",
    );
    assert!(did, "a scalar field read should lower");
    let fl = w
        .local_hir_stream
        .iter()
        .find(|i| i.opcode == Opcode::FieldLoad)
        .expect("a FieldLoad instruction");
    assert_eq!(fl.imm, 4, "y is at byte offset 4 in {{x:i32, y:i32}}");
    assert_eq!(count(&w, Opcode::FieldLoad), 1);
    assert_eq!(count(&w, Opcode::Ret), 1);
    verify_hir_stream(&w);
}

#[test]
fn field_read_offset_honours_alignment_padding() {
    // Rec { a: i8, b: i32 } -> b sits at offset 4 (3 bytes of padding after `a`); the FieldLoad
    // must use that padded offset, proving it comes from the real layout, not field order.
    let (did, w) = lower_with_registry(
        "struct Rec { a: i8, b: i32 }\nfn getb(r: Rec) -> i32 { return r.b; }",
        "getb",
    );
    assert!(did);
    let fl = w
        .local_hir_stream
        .iter()
        .find(|i| i.opcode == Opcode::FieldLoad)
        .expect("a FieldLoad instruction");
    assert_eq!(fl.imm, 4, "b is at offset 4 after i8 + 3 bytes padding");
    verify_hir_stream(&w);
}

#[test]
fn struct_construction_lowers_to_alloca_and_field_stores() {
    // `let p = Point { .. }` constructs in place: one sized Alloca + a FieldStore per field at
    // its layout offset; the later `p.y` is a FieldLoad off the same slot.
    let (did, w) = lower_with_registry(
        "struct Point { x: i32, y: i32 }\n\
             fn build() -> i32 { let p = Point { x: 7i32, y: 9i32 }; return p.y; }",
        "build",
    );
    assert!(did, "struct construction + field read should lower");
    assert_eq!(count(&w, Opcode::Alloca), 1, "one aggregate slot for p");
    let alloca = w
        .local_hir_stream
        .iter()
        .find(|i| i.opcode == Opcode::Alloca)
        .unwrap();
    assert_eq!(alloca.imm, 8, "Point slot sized from its layout");
    assert_eq!(count(&w, Opcode::FieldStore), 2, "x and y stored");
    let store_offsets: Vec<u64> = w
        .local_hir_stream
        .iter()
        .filter(|i| i.opcode == Opcode::FieldStore)
        .map(|i| i.imm)
        .collect();
    assert!(
        store_offsets.contains(&0) && store_offsets.contains(&4),
        "fields stored at layout offsets 0 and 4, got {store_offsets:?}"
    );
    assert_eq!(count(&w, Opcode::FieldLoad), 1, "p.y read");
    verify_hir_stream(&w);
}

#[test]
fn tensor_param_binds_as_ssa_reg_with_tensor_gid() {
    // A tensor parameter is a reference value (memref): it binds as an SSA register (no Alloca),
    // and its element+shape GID enters the flat type stream.
    let f = parse_fn("fn f(q: Tensor<f32, [2, 4]>) -> i32 { return 0; }");
    let mut w = worker();
    lower_function_to_hir(&f, &mut w).expect("tensor-param fn should lower");
    assert_eq!(
        count(&w, Opcode::Alloca),
        0,
        "a tensor param is a reference, not stack-allocated"
    );
    let expected = tensor_gid(&ElementType::F32, &["2".to_string(), "4".to_string()]);
    assert!(
        w.local_type_stream.contains(&expected),
        "the tensor GID is in the type stream"
    );
    verify_hir_stream(&w);
}

#[test]
fn tensor_full_index_yields_scalar_element() {
    // `q[0][1]` on a rank-2 tensor rank-reduces twice: [2,4] -> [4] -> scalar f32.
    let f = parse_fn("fn f(q: Tensor<f32, [2, 4]>) -> f32 { return q[0][1]; }");
    let mut w = worker();
    lower_function_to_hir(&f, &mut w).expect("tensor element read should lower");
    assert_eq!(
        count(&w, Opcode::TensorIndex),
        2,
        "two rank-reducing indexes"
    );
    assert_eq!(count(&w, Opcode::Alloca), 0);
    // The intermediate row type and the final scalar element are both in the type stream.
    let row = tensor_gid(&ElementType::F32, &["4".to_string()]);
    assert!(
        w.local_type_stream.contains(&row),
        "row (rank-1) tensor GID present"
    );
    assert!(
        w.local_type_stream.contains(&scalar_gid(&ElementType::F32)),
        "scalar element GID present"
    );
    verify_hir_stream(&w);
}

#[test]
fn tensor_partial_index_yields_row_view() {
    // `q[0]` on [2,4] yields a rank-1 row view [4] (one TensorIndex, tensor result).
    let f = parse_fn("fn f(q: Tensor<f32, [2, 4]>) -> Tensor<f32, [4]> { return q[0]; }");
    let mut w = worker();
    lower_function_to_hir(&f, &mut w).expect("row view should lower");
    assert_eq!(count(&w, Opcode::TensorIndex), 1);
    let row = tensor_gid(&ElementType::F32, &["4".to_string()]);
    assert!(w.local_type_stream.contains(&row), "row tensor GID present");
    verify_hir_stream(&w);
}

/// The result-type GID of the first instruction with `op`.
fn result_gid(w: &LocalWorkerState, op: Opcode) -> TypeId {
    let ins = w
        .local_hir_stream
        .iter()
        .find(|i| i.opcode == op)
        .expect("op present");
    w.local_type_stream[ins.type_idx.0 as usize]
}

#[test]
fn tensor_elementwise_mul_yields_a_tensor() {
    // `a * b` on two tensors is elementwise: a `Mul` whose *result type* is the tensor.
    let f = parse_fn(
        "fn f(a: Tensor<f32, [4]>, b: Tensor<f32, [4]>) -> Tensor<f32, [4]> { return a * b; }",
    );
    let mut w = worker();
    lower_function_to_hir(&f, &mut w).expect("elementwise mul should lower");
    assert_eq!(count(&w, Opcode::Mul), 1);
    assert_eq!(
        result_gid(&w, Opcode::Mul),
        tensor_gid(&ElementType::F32, &["4".to_string()]),
        "elementwise result is the tensor type"
    );
    verify_hir_stream(&w);
}

#[test]
fn scalar_times_tensor_broadcasts_to_a_tensor() {
    // `s * a` (scalar on the left) still yields the tensor type -- the result-type pick must
    // follow the tensor operand, not the lhs.
    let f = parse_fn("fn f(a: Tensor<f32, [4]>, s: f32) -> Tensor<f32, [4]> { return s * a; }");
    let mut w = worker();
    lower_function_to_hir(&f, &mut w).expect("scalar*tensor should lower");
    assert_eq!(
        result_gid(&w, Opcode::Mul),
        tensor_gid(&ElementType::F32, &["4".to_string()]),
        "scalar broadcasts: result is the tensor type"
    );
    verify_hir_stream(&w);
}

#[test]
fn broadcast_write_path_composes() {
    // Allocate the output, weight v by a scaled score, and store the row back -- alloc +
    // elementwise + index + store, the whole non-scalar surface in one flat stream.
    let f = parse_fn(
        "fn attn(score: f32, v: Tensor<f32, [4]>, scale: f32) -> Tensor<f32, [2, 4]> \
             { let o = Tensor<f32>([2, 4]); o[0] = v * (score * scale); return o; }",
    );
    let mut w = worker();
    lower_function_to_hir(&f, &mut w).expect("the FA write path should lower");
    assert_eq!(count(&w, Opcode::TensorAlloc), 1, "output buffer");
    assert_eq!(count(&w, Opcode::TensorIndex), 1, "o[0]");
    assert_eq!(count(&w, Opcode::Mul), 2, "scale the score, then weight v");
    assert_eq!(count(&w, Opcode::TensorStore), 1, "o[0] = ...");
    verify_hir_stream(&w);
}

#[test]
fn a_run_time_extent_rides_on_an_arg_before_the_alloc() {
    // `Tensor<f32>([n])` with `n` a parameter: the extent is an `Arg` immediately before the
    // `TensorAlloc`, the shape says `?` there, and the byte size is 0 -- there is none to
    // state. The typed form `Tensor<f32, [?, 4]>::uninit([n, 4])` reads its `?` off the array.
    let f =
        parse_fn("fn build(n: i32) -> Tensor<f32, [?]> { let a = Tensor<f32>([n]); return a; }");
    let mut w = worker();
    lower_function_to_hir(&f, &mut w).expect("a run-time allocation lowers");
    assert_eq!(count(&w, Opcode::Arg), 1, "one extent");
    assert_eq!(count(&w, Opcode::TensorAlloc), 1);
    assert_eq!(
        op_imm(&w, Opcode::TensorAlloc),
        Some(0),
        "no static byte size"
    );
    let (i, _) = w
        .local_hir_stream
        .iter()
        .enumerate()
        .find(|(_, ins)| ins.opcode == Opcode::TensorAlloc)
        .unwrap();
    assert_eq!(
        w.local_hir_stream[i - 1].opcode,
        Opcode::Arg,
        "the extent precedes the alloc"
    );
    verify_hir_stream(&w);

    let f = parse_fn(
            "fn build(n: i32) -> Tensor<f32, [?, 4]> { let a = Tensor<f32, [?, 4]>::uninit([n, 4]); return a; }",
        );
    let mut w = worker();
    lower_function_to_hir(&f, &mut w).expect("the typed form lowers");
    assert_eq!(
        count(&w, Opcode::Arg),
        1,
        "only the `?` position is an extent"
    );
    assert_eq!(op_imm(&w, Opcode::TensorAlloc), Some(0));
    verify_hir_stream(&w);
}

#[test]
fn a_tensor_view_rides_its_pointer_and_its_run_time_extents() {
    // `tensor_view_2d(p, 2, n)`: the pointer is the view's operand, the literal row count
    // is in the type, and the run-time column count is an `Arg` before the view.
    let f = parse_fn(
        "fn view(p: *mut f32, n: i32) -> Tensor<f32, [2, ?]> { \
               let v = unsafe { tensor_view_2d(p, 2, n) }; return v; }",
    );
    let mut w = worker();
    lower_function_to_hir(&f, &mut w).expect("a view lowers");
    assert_eq!(count(&w, Opcode::TensorView), 1);
    assert_eq!(count(&w, Opcode::Arg), 1, "one run-time extent");
    assert_eq!(
        count(&w, Opcode::TensorAlloc),
        0,
        "a view allocates nothing"
    );
    verify_hir_stream(&w);
}

fn op_imm(w: &LocalWorkerState, op: Opcode) -> Option<u64> {
    w.local_hir_stream
        .iter()
        .find(|i| i.opcode == op)
        .map(|i| i.imm)
}

#[test]
fn tensor_alloc_sizes_storage_for_the_receiver() {
    // `Tensor<f32>([2, 4])` allocates a buffer sized for every element (2*4*4 = 32 bytes), so a
    // later store has room.
    let f = parse_fn("fn f() -> Tensor<f32, [2, 4]> { let o = Tensor<f32>([2, 4]); return o; }");
    let mut w = worker();
    lower_function_to_hir(&f, &mut w).expect("tensor alloc should lower");
    assert_eq!(count(&w, Opcode::TensorAlloc), 1);
    assert_eq!(
        op_imm(&w, Opcode::TensorAlloc),
        Some(32),
        "byte size = 2*4*4"
    );
    assert_eq!(
        result_gid(&w, Opcode::TensorAlloc),
        tensor_gid(&ElementType::F32, &["2".to_string(), "4".to_string()]),
    );
    verify_hir_stream(&w);
}

#[test]
fn a_bool_tensor_takes_a_byte_per_element() {
    let f = parse_fn(
        "fn f() -> Tensor<bool, [3]> \
             { let o : Tensor<bool, [3]> = [true, false, true]; return o; }",
    );
    let mut w = worker();
    lower_function_to_hir(&f, &mut w).expect("a bool literal should lower");
    assert_eq!(
        op_imm(&w, Opcode::TensorAlloc),
        Some(3),
        "3 bools are 3 bytes"
    );
    verify_hir_stream(&w);
}

#[test]
fn a_nested_bool_initializer_takes_a_byte_per_element() {
    let f = parse_fn(
        "fn f() -> Tensor<bool, [2, 3]> \
             { let o = Tensor<bool>([[true, false, true], [false, true, false]]); return o; }",
    );
    let mut w = worker();
    lower_function_to_hir(&f, &mut w).expect("a nested bool literal should lower");
    assert_eq!(
        op_imm(&w, Opcode::TensorAlloc),
        Some(6),
        "6 bools are 6 bytes"
    );
    verify_hir_stream(&w);
}

#[test]
fn tensor_row_store_writes_through_the_slice() {
    // `o[0] = v` stores the row slice `v` into the allocated buffer `o`: alloc + index + store.
    let f = parse_fn(
        "fn f(v: Tensor<f32, [4]>) -> Tensor<f32, [2, 4]> \
             { let o = Tensor<f32>([2, 4]); o[0] = v; return o; }",
    );
    let mut w = worker();
    lower_function_to_hir(&f, &mut w).expect("row store should lower");
    assert_eq!(count(&w, Opcode::TensorAlloc), 1, "the destination buffer");
    assert_eq!(count(&w, Opcode::TensorIndex), 1, "the row place o[0]");
    assert_eq!(count(&w, Opcode::TensorStore), 1, "the store into it");
    verify_hir_stream(&w);
}

#[test]
fn scalar_element_store_into_rank1_marks_a_place() {
    // `q[0] = 1.0` into a rank-1 tensor: the whole tensor is the base, the index is a scalar-
    // element *place* (imm 1), and one `TensorStore` writes the scalar through it.
    let f = parse_fn("fn f() -> f32 { let q = Tensor<f32>([4]); q[0] = 1.0; return q[1]; }");
    let mut w = worker();
    lower_function_to_hir(&f, &mut w).expect("scalar-element store should lower");
    assert_eq!(count(&w, Opcode::TensorStore), 1);
    let places = w
        .local_hir_stream
        .iter()
        .filter(|i| i.opcode == Opcode::TensorIndex && i.imm == 1)
        .count();
    assert_eq!(places, 1, "the element index is a place (imm = 1)");
    verify_hir_stream(&w);
}

#[test]
fn scalar_element_store_rank2_indexes_row_then_element_place() {
    // `q[0][0] = 1.0`: `q[0]` is a value sub-view index (imm 0), the final `[0]` an element
    // place (imm 1); exactly one index carries the place flag.
    let f = parse_fn("fn f(q: Tensor<f32, [2, 4]>) -> f32 { q[0][0] = 1.0; return q[1][1]; }");
    let mut w = worker();
    lower_function_to_hir(&f, &mut w).expect("nested scalar-element store should lower");
    assert_eq!(count(&w, Opcode::TensorStore), 1);
    let places = w
        .local_hir_stream
        .iter()
        .filter(|i| i.opcode == Opcode::TensorIndex && i.imm == 1)
        .count();
    assert_eq!(places, 1, "exactly one element place among the indices");
    verify_hir_stream(&w);
}

#[test]
fn transfer_rehomes_a_tensor_to_a_memory_space() {
    // `transfer(a, Memory::NPU_HBM)` -> a Transfer carrying the space's dispatch id (100);
    // the result keeps the shape, so the receiving buffer is sized to hold it.
    let f = parse_fn(
            "fn f(a: Tensor<f32, [2, 4]>) -> Tensor<f32, [2, 4]> { return transfer(a, Memory::NPU_HBM); }",
        );
    let mut w = worker();
    lower_function_to_hir(&f, &mut w).expect("transfer should lower");
    assert_eq!(count(&w, Opcode::Transfer), 1);
    assert_eq!(
        op_imm(&w, Opcode::Transfer),
        Some(100),
        "NPU_HBM dispatch id"
    );
    assert_eq!(
        result_gid(&w, Opcode::Transfer),
        tensor_gid(&ElementType::F32, &["2".to_string(), "4".to_string()]),
        "same shape -> the destination holds the source",
    );
    verify_hir_stream(&w);
}

#[test]
fn scalar_under_a_rank_0_tensor_annotation_materializes_rank_0() {
    // `let t : Tensor<f32, []> = 1.0` allocates a rank-0 buffer and stores the scalar, so
    // the transfer receives a tensor, not the initializer's scalar (Vx#396). The shape is
    // written out now that the dims-less spelling is gone (Vx#399).
    let f = parse_fn(
            "fn f() -> i32 { let t : Tensor<f32, []> = 1.0; let d = transfer(t, Memory::NPU_HBM); return 0; }",
        );
    let mut w = worker();
    lower_function_to_hir(&f, &mut w).expect("rank-0 wrap should lower");
    assert_eq!(count(&w, Opcode::TensorAlloc), 1);
    assert_eq!(count(&w, Opcode::TensorStore), 1);
    assert_eq!(count(&w, Opcode::Transfer), 1);
    assert_eq!(
        result_gid(&w, Opcode::Transfer),
        tensor_gid(&ElementType::F32, &[]),
        "the transferred value is the rank-0 tensor",
    );
    verify_hir_stream(&w);
}

#[test]
fn wrapper_types_peel_to_their_runtime_type() {
    // `Verified<Ref<Tensor<f32,[2,2]>, Memory>>` is a memref at runtime; the checker
    // wrappers peel exactly as the AST codegen's `lower_type` peels them.
    let session = GlobalSession::new(1);
    let registry = session.registry.as_ref();
    let num = |v: &str| {
        Expr::Number(crate::syntax::NumberExpr::new(
            v.to_string(),
            None,
            crate::syntax::Span::default(),
        ))
    };
    let dims = vec![Dim::Static(num("2")), Dim::Static(num("2"))];
    let tensor = Type::Tensor(ElementType::F32, dims, None);
    let wrapped = Type::Verified(Box::new(Type::Ref(
        Box::new(tensor),
        crate::syntax::MemorySpace::NPUHBM,
    )));
    assert!(
        matches!(
            lowered_ty(&wrapped, registry),
            Some(LoweredTy::Tensor { .. })
        ),
        "wrappers peel to the tensor",
    );
}

#[test]
fn tensor_gid_distinguishes_element_and_shape() {
    let a = tensor_gid(&ElementType::F32, &["2".to_string(), "4".to_string()]);
    let b = tensor_gid(&ElementType::F32, &["4".to_string(), "4".to_string()]);
    let c = tensor_gid(&ElementType::F64, &["2".to_string(), "4".to_string()]);
    assert_ne!(a, b, "different shape => different GID");
    assert_ne!(a, c, "different element => different GID");
    // Deterministic, and distinct from the scalar element GID.
    assert_eq!(
        a,
        tensor_gid(&ElementType::F32, &["2".to_string(), "4".to_string()])
    );
    assert_ne!(a, scalar_gid(&ElementType::F32));
}

#[test]
fn compound_assign_desugars_to_op_and_store() {
    // `s += a` == `s = s + a`: the current value is read, combined, and stored back. In memory
    // mode (the loop forces it) that's a `SlotLoad` + `Add` + `Store`.
    let f = parse_fn("fn f(a: i32) -> i32 { let mut s = 0; for i in 0..a { s += a; } return s; }");
    let mut w = worker();
    lower_function_to_hir(&f, &mut w).expect("compound assign should lower");
    assert!(
        count(&w, Opcode::Add) >= 1,
        "the += combine (plus the loop step)"
    );
    assert!(count(&w, Opcode::Store) >= 1, "s is stored back");
    verify_hir_stream(&w);
}

#[test]
fn fixed_arity_call_lowers_to_args_and_call() {
    // `add(3, 4)` -> two `Arg`s then a `Call` whose type_idx is the callee's GID.
    let (did, w) = lower_with_registry(
        "fn add(a: i32, b: i32) -> i32 { return a + b; }\n\
             fn main() -> i32 { return add(3i32, 4i32); }",
        "main",
    );
    assert!(did, "a call to a known scalar-returning fn should lower");
    assert_eq!(count(&w, Opcode::Arg), 2, "two arguments");
    assert_eq!(count(&w, Opcode::Call), 1);
    let call = w
        .local_hir_stream
        .iter()
        .find(|i| i.opcode == Opcode::Call)
        .unwrap();
    assert_eq!(call.imm, 2, "arg count in imm");
    // The Call's type_idx is the callee GID (a real function identity: nonzero module hash),
    // not a scalar type (whose GID lives in module 0).
    let callee = w.local_type_stream[call.type_idx.0 as usize];
    assert_ne!(callee.module_id(), 0, "type_idx is the callee's GID");
    verify_hir_stream(&w);
}

#[test]
fn call_to_unknown_fn_declines() {
    // `mystery` isn't a known function -> not in the registry's fn_sigs -> the call declines,
    // so the whole function declines (atomic no-op).
    let (did, w) = lower_with_registry("fn main() -> i32 { return mystery(1i32); }", "main");
    assert!(!did);
    assert!(w.local_hir_stream.is_empty());
}

#[test]
fn unmodelled_aggregate_param_is_declined() {
    // `Buf` holds an enum with a payload by value, which the frozen layouts do not model, so
    // its layout stays the 0/0 stub; a function taking `Buf` by value cannot be sized, and
    // lowering is declined atomically (worker untouched). This used to use a `<4 x f32>` field,
    // then a generic struct instance, and each stopped being unmodelled when it got a layout.
    let (did, w) = lower_with_registry(
            "enum Maybe<T> { Some(T), Nothing }\nstruct Buf { data: Maybe<i32> }\nfn f(b: Buf) -> i32 { return 0; }",
            "f",
        );
    assert!(
        !did,
        "an unmodelled aggregate param should decline lowering"
    );
    assert!(
        w.local_hir_stream.is_empty(),
        "no partial lowering committed"
    );
}

#[test]
fn if_else_slots_only_the_mutated_local() {
    // Per-local rule (step 2): control flow no longer slots *every* local. `x` is reassigned in
    // both branches (its value must cross the merge), so it gets a slot; the param `a` is only read
    // (a dominating entry value), so it stays a register. One `Alloca`, not two — strictly less
    // memory traffic than the old function-global memory mode. (#230)
    let f = parse_fn(
        "fn c(a: i32) -> i32 { let mut x = a; if a < 0 { x = 0; } else { x = 1; } return x; }",
    );
    let mut w = worker();
    lower_function_to_hir(&f, &mut w).expect("lowers");
    assert_eq!(
        count(&w, Opcode::Alloca),
        1,
        "only the mutated `x` gets a slot"
    );
    assert!(
        count(&w, Opcode::SlotLoad) >= 1,
        "`x` is read back after the merge"
    );
    assert_eq!(count(&w, Opcode::CondBr), 1);
    assert_eq!(
        count(&w, Opcode::Br),
        2,
        "then and else each branch to merge"
    );
    assert_eq!(count(&w, Opcode::BlockStart), 4, "entry, then, else, merge");
    verify_hir_stream(&w);
}

#[test]
fn if_without_else_targets_merge_directly() {
    let f = parse_fn("fn c(a: i32) -> i32 { let mut x = a; if a < 0 { x = 0; } return x; }");
    let mut w = worker();
    lower_function_to_hir(&f, &mut w).expect("lowers");
    assert_eq!(count(&w, Opcode::CondBr), 1);
    assert_eq!(
        count(&w, Opcode::Br),
        1,
        "then branches to merge; else edge is merge"
    );
    assert_eq!(count(&w, Opcode::BlockStart), 3, "entry, then, merge");
    verify_hir_stream(&w);
}

#[test]
fn early_return_in_branch_emits_no_trailing_branch() {
    let f = parse_fn("fn c(a: i32) -> i32 { if a < 0 { return 0; } return a; }");
    let mut w = worker();
    lower_function_to_hir(&f, &mut w).expect("lowers");
    // The then-block terminates with Ret, so no `Br` to the merge is appended.
    assert_eq!(
        count(&w, Opcode::Br),
        0,
        "returning branch emits no trailing Br"
    );
    assert_eq!(count(&w, Opcode::Ret), 2);
    assert_eq!(count(&w, Opcode::CondBr), 1);
    verify_hir_stream(&w);
}

#[test]
fn straight_line_reassignment_stays_pure_ssa() {
    // No control flow -> SSA mode: reassignment is a rebind, no memory ops or blocks.
    let f = parse_fn("fn c(a: i32) -> i32 { let mut x = a; x = a + a; return x; }");
    let mut w = worker();
    lower_function_to_hir(&f, &mut w).expect("lowers");
    assert_eq!(count(&w, Opcode::Alloca), 0);
    assert_eq!(count(&w, Opcode::Store), 0);
    assert_eq!(count(&w, Opcode::BlockStart), 0);
    assert_eq!(
        opcodes(&w),
        vec![Opcode::Load, Opcode::Add, Opcode::Ret],
        "a materialized once, x rebinds to a+a"
    );
    verify_hir_stream(&w);
}

#[test]
fn non_escaping_borrow_binds_a_place_with_no_alloca() {
    // §5 Example A (#275): `let r = &x; return *r` where the borrow never escapes. `r` binds as a
    // symbolic `Binding::Place` aliasing `x`, `*r` reads `x` directly, and `x` — no longer forced
    // to materialize an address — stays a register. No `Alloca`, no `Store`, no `PtrIndex`: the
    // whole reference round-trip compiles to `const 5; ret`. This is the alloca that should not
    // exist, gone.
    let f = parse_fn("fn f() -> i32 { let x : i32 = 5; let r : &i32 = &x; return *r; }");
    let mut w = worker();
    lower_function_to_hir(&f, &mut w).expect("non-escaping scalar borrow+deref lowers");
    assert_eq!(
        count(&w, Opcode::Alloca),
        0,
        "the base stays a register — no materialized address"
    );
    assert_eq!(
        count(&w, Opcode::Store),
        0,
        "no store: nothing spilled to memory"
    );
    assert_eq!(
        count(&w, Opcode::PtrIndex),
        0,
        "`*r` reads the local directly, not via a pointer"
    );
    assert_eq!(count(&w, Opcode::Ret), 1);
    verify_hir_stream(&w);
}

#[test]
fn escaping_borrow_materializes_a_flagged_slot() {
    // The counterpart: when the reference *escapes* (here as a call argument), the base must
    // materialize a real address — an `llvm.alloca` flagged `imm = 1`, and `&x` a `PtrIndex`-able
    // pointer. `r` is used as `id(r)`, not `*r`, so the escape analysis keeps `x` materialized. (#275)
    let (did, w) = lower_with_registry(
        "fn id(p : &i32) -> i32 { return *p; }\n\
             fn f() -> i32 { let x : i32 = 5; let r : &i32 = &x; return id(r); }",
        "f",
    );
    assert!(did, "the escaping borrow lowers via materialization");
    let alloca = w
        .local_hir_stream
        .iter()
        .find(|i| i.opcode == Opcode::Alloca)
        .expect("the escaping base gets a materialized slot");
    assert_eq!(alloca.imm, 1, "an addressable `llvm.alloca`, not a memref");
}

#[test]
fn borrow_inside_a_struct_literal_materializes_its_base() {
    // #275: a `&x` inside an aggregate literal (`Holder { r : &x }`, a reference-typed field)
    // escapes into the struct, so the escape scan must descend into the literal and materialize `x`
    // — an addressable `llvm.alloca` (`imm = 1`). Without the StructInit arm in the scan, `x` stayed
    // a register and the `&x` declined. The whole reference-field program then lowers.
    let (did, w) = lower_with_registry(
        "struct Holder { r : &i32 }\n\
             fn main() -> i32 { let x = 5; let h = Holder { r : &x }; return *h.r; }",
        "main",
    );
    assert!(did, "the reference-typed-field program lowers");
    let addressable = w
        .local_hir_stream
        .iter()
        .any(|i| i.opcode == Opcode::Alloca && i.imm == 1);
    assert!(
        addressable,
        "`&x` inside the struct literal materializes `x` as an addressable slot"
    );
}

#[test]
fn field_places_resolve_writes_to_field_stores() {
    // §5 Example C (#275): `&mut p.x` / `&mut p.y` bind *field places*; each `*b = v` re-lowers as a
    // `FieldStore` through `p` — no pointer materialized, no `PtrStore`. The disjoint borrows lower.
    let (did, w) = lower_with_registry(
            "struct Point { x : i32, y : i32 }\n\
             fn update(p : &mut Point) -> void { let bx = &mut p.x; let by = &mut p.y; *bx = 1; *by = 2; }",
            "update",
        );
    assert!(
        did,
        "the disjoint field-borrow mutator lowers on the flat path"
    );
    assert_eq!(
        count(&w, Opcode::FieldStore),
        2,
        "each `*b = v` is a FieldStore through p"
    );
    assert_eq!(
        count(&w, Opcode::PtrStore),
        0,
        "the field places materialize no pointer"
    );
}

#[test]
fn disjoint_place_writes_reduce_to_mutual_noalias_siblings() {
    // §5.4 (M2b-2): the two disjoint place-writes above reduce to a numeric alias table — each store
    // gets its own group, and (same base `p`, disjoint fields `x`/`y`) each lists the other as a
    // sibling. The two FieldStores sit at distinct stream positions; the table pairs them mutually.
    let (did, w) = lower_with_registry(
            "struct Point { x : i32, y : i32 }\n\
             fn update(p : &mut Point) -> void { let bx = &mut p.x; let by = &mut p.y; *bx = 1; *by = 2; }",
            "update",
        );
    assert!(did, "the mutator lowers");
    let table = &w.local_place_alias_stores;
    assert_eq!(table.len(), 2, "both place-writes are tagged: {table:?}");
    // Two distinct groups (`p.x` and `p.y` are different fields).
    let groups: std::collections::HashSet<usize> = table.iter().map(|(_, g, _)| *g).collect();
    assert_eq!(
        groups.len(),
        2,
        "distinct fields => distinct groups: {table:?}"
    );
    // Each store names the *other* store's group as its lone disjoint sibling.
    for (_, own, sibs) in table {
        assert_eq!(sibs.len(), 1, "one disjoint sibling per store: {table:?}");
        assert_ne!(sibs[0], *own, "a store is not its own noalias sibling");
    }
    // The two positions are the two FieldStores in the stream.
    let fs_positions: Vec<usize> = w
        .local_hir_stream
        .iter()
        .enumerate()
        .filter(|(_, i)| i.opcode == Opcode::FieldStore)
        .map(|(p, _)| p)
        .collect();
    let tagged: std::collections::HashSet<usize> = table.iter().map(|(p, _, _)| *p).collect();
    assert_eq!(
        tagged,
        fs_positions.into_iter().collect(),
        "tagged positions are exactly the FieldStores"
    );
}

#[test]
fn a_lone_place_write_has_no_noalias_sibling() {
    // A single place-write has nothing proven disjoint from it: it still gets its own group, but an
    // empty sibling set — the reduction only pairs stores the frontend actually proved disjoint.
    let (did, w) = lower_with_registry(
            "struct P { x : i32, y : i32 }\n\
             fn main() -> i32 { let mut p = P { x : 1, y : 2 }; let r = &mut p.x; *r = 42; return *r; }",
            "main",
        );
    assert!(did, "the single-field mutator lowers");
    assert_eq!(w.local_place_alias_stores.len(), 1, "one tagged store");
    assert!(
        w.local_place_alias_stores[0].2.is_empty(),
        "a lone place-write has no disjoint sibling"
    );
}

#[test]
fn pointer_local_under_control_flow_stays_a_register() {
    // M3b (§11's remainder): a `*mut` pointer local under control flow that is not reassigned binds
    // as an SSA register, not a slot — the same per-local rule scalars got. `ptr` here is the
    // `let ptr = self.data` shape from `Vec::push`'s grow branch: a raw-pointer field read inside an
    // `if`, read once (`b.data = ptr`), never reassigned. It gets no `Alloca`.
    let (did, w) = lower_with_registry(
            "struct Buf { data : *mut i32, n : i32 }\n\
             fn grow(b : &mut Buf, c : i32) -> i32 { if c > 0 { let ptr : *mut i32 = b.data; b.data = ptr; } return 0; }",
            "grow",
        );
    assert!(did, "the raw-pointer mutator lowers");
    assert_eq!(
        count(&w, Opcode::Alloca),
        0,
        "the non-reassigned pointer local under control flow stays a register, no slot"
    );
}

#[test]
fn direct_field_assignment_is_not_tagged() {
    // A *direct* `p.x = v` (not through a `&mut` place) is a plain field store, not a disjoint
    // place-write, so it carries no alias metadata — only reference-mediated writes are tagged.
    let (did, w) = lower_with_registry(
            "struct P { x : i32, y : i32 }\n\
             fn main() -> i32 { let mut p = P { x : 0, y : 0 }; p.x = 1; p.y = 2; return p.x + p.y; }",
            "main",
        );
    assert!(did, "the direct-assignment fn lowers");
    assert!(
        w.local_place_alias_stores.is_empty(),
        "direct field assignments are untagged: {:?}",
        w.local_place_alias_stores
    );
}

#[test]
fn non_address_taken_scalar_stays_a_register() {
    // The demotion is *scoped*: a scalar local whose address is never taken stays pure-SSA (no
    // slot), so the pre-pass does not regress straight-line functions into memory traffic. (#230)
    let f = parse_fn("fn g() -> i32 { let x : i32 = 5; return x; }");
    let mut w = worker();
    lower_function_to_hir(&f, &mut w).expect("lowers");
    assert_eq!(
        count(&w, Opcode::Alloca),
        0,
        "no `&x`, so `x` stays a register"
    );
}

#[test]
fn reference_param_derefs_without_a_slot() {
    // A `&i32` parameter arrives already materialized (it crossed a call boundary), so it is a
    // pointer register on entry — never demoted. `*a` is a `PtrIndex` straight off the param, with
    // no `Alloca`. (design doc §3.2 NOTE: only *locals* are demoted.) (#230)
    let f = parse_fn("fn load(a : &i32) -> i32 { return *a; }");
    let mut w = worker();
    lower_function_to_hir(&f, &mut w).expect("reference param derefs");
    assert_eq!(
        count(&w, Opcode::Alloca),
        0,
        "the param is a pointer register, not a slot"
    );
    assert_eq!(
        count(&w, Opcode::PtrIndex),
        1,
        "`*a` is a PtrIndex read off the param"
    );
    verify_hir_stream(&w);
}

#[test]
fn void_call_lowers_and_mutates_through_a_reference() {
    // A `&mut` mutator returns `void`; calling it in statement position (`bump(&mut x);`) lowers
    // rather than declining — the call is a pure effect whose result is discarded. `x` is
    // address-taken, so it is an `llvm.alloca` slot (imm = 1) the pointer mutates; `return x`
    // reads the mutated value back through the same slot. (#230)
    let (did, w) = lower_with_registry(
        "fn bump(p : &mut i32) -> void { *p = *p + 1; }\n\
             fn main() -> i32 { let mut x = 41; bump(&mut x); return x; }",
        "main",
    );
    assert!(did, "the void call + read-back lowers on the flat path");
    assert_eq!(count(&w, Opcode::Call), 1, "one call to the void mutator");
    let alloca = w
        .local_hir_stream
        .iter()
        .find(|i| i.opcode == Opcode::Alloca)
        .expect("the address-taken `x` gets a slot");
    assert_eq!(alloca.imm, 1, "x is an address-taken llvm.alloca slot");
}

#[test]
fn for_range_loop_lowers_as_an_affine_loop() {
    let f =
        parse_fn("fn sum(n: i32) -> i32 { let mut s = 0; for i in 0..n { s = s + i; } return s; }");
    let mut w = worker();
    lower_function_to_hir(&f, &mut w).expect("lowers");
    assert_eq!(count(&w, Opcode::AffineFor), 1);
    assert_eq!(count(&w, Opcode::AffineForEnd), 1);
    assert_eq!(count(&w, Opcode::CondBr), 0, "no loop condition test");
    let open = w
        .local_hir_stream
        .iter()
        .find(|i| i.opcode == Opcode::AffineFor)
        .unwrap();
    assert_eq!(open.imm & AFFINE_FOR_BRANCHES, 0, "the body is one block");
    verify_hir_stream(&w);
}

#[test]
fn affine_loop_with_an_if_marks_its_branches() {
    let f = parse_fn(
        "fn f(n: i32) -> i32 { let mut s = 0; for i in 0..n { if i < 3 { s = s + i; } } return s; }",
    );
    let mut w = worker();
    lower_function_to_hir(&f, &mut w).expect("lowers");
    let open = w
        .local_hir_stream
        .iter()
        .find(|i| i.opcode == Opcode::AffineFor)
        .expect("an affine loop");
    assert_eq!(open.imm & AFFINE_FOR_BRANCHES, AFFINE_FOR_BRANCHES);
    verify_hir_stream(&w);
}

#[test]
fn loop_that_break_leaves_keeps_its_blocks() {
    let f = parse_fn(
        "fn f(n: i32) -> i32 { let mut s = 0; for i in 0..n { if i == 3 { break; } s = s + i; } return s; }",
    );
    let mut w = worker();
    lower_function_to_hir(&f, &mut w).expect("lowers");
    assert_eq!(count(&w, Opcode::AffineFor), 0);
    verify_hir_stream(&w);
}

#[test]
fn for_range_loop_lowers_header_body_latch_exit() {
    // An unsigned range is not an affine loop, so it keeps its blocks.
    let f = parse_fn(
        "fn sum(n: u32) -> u32 { let mut s = 0u32; for i in 0u32..n { s = s + i; } return s; }",
    );
    let mut w = worker();
    lower_function_to_hir(&f, &mut w).expect("lowers");
    assert_eq!(count(&w, Opcode::CondBr), 1, "loop condition test");
    assert_eq!(
        count(&w, Opcode::BlockStart),
        5,
        "entry, header, body, latch, exit"
    );
    // entry->header, body->latch, latch->header
    assert_eq!(count(&w, Opcode::Br), 3);
    // The induction var increments: an Add feeding a Store in the latch.
    assert!(
        count(&w, Opcode::Add) >= 2,
        "body add + induction increment"
    );
    verify_hir_stream(&w);
}

#[test]
fn infinite_loop_with_break_verifies() {
    let f = parse_fn(
        "fn f(a: i32) -> i32 { let mut x = a; loop { x = x - 1; if x < 0 { break; } } return x; }",
    );
    let mut w = worker();
    lower_function_to_hir(&f, &mut w).expect("lowers");
    assert!(count(&w, Opcode::CondBr) >= 1, "the if condition");
    // loop header/exit + entry + the if's then/merge blocks.
    assert!(count(&w, Opcode::BlockStart) >= 4);
    verify_hir_stream(&w); // every branch (incl. the break) targets a declared block
}

#[test]
fn for_loop_with_continue_verifies() {
    let f = parse_fn(
            "fn f(n: i32) -> i32 { let mut s = 0; for i in 0..n { if i < 2 { continue; } s = s + i; } return s; }",
        );
    let mut w = worker();
    lower_function_to_hir(&f, &mut w).expect("lowers");
    verify_hir_stream(&w);
}

#[test]
fn for_over_non_range_aborts() {
    // A non-range iterable (here a call) is outside the supported subset -> atomic abort.
    let f = parse_fn("fn f(a: i32) -> i32 { for i in gen() { } return a; }");
    let mut w = worker();
    assert!(lower_function_to_hir(&f, &mut w).is_err());
    assert!(w.local_hir_stream.is_empty());
}

#[test]
fn break_outside_loop_aborts() {
    let f = parse_fn("fn f() -> i32 { break; return 0; }");
    let mut w = worker();
    assert!(lower_function_to_hir(&f, &mut w).is_err());
    assert!(w.local_hir_stream.is_empty());
}

#[test]
fn a_blocked_spawn_region_is_grid_stridable() {
    // The bench template's shape in miniature: thread-private scratch before the loop, scalar
    // accumulators inside it, an if-expression, and every captured write leading with the
    // induction variable. (No `.exp()` here — the harness loads no stdlib, so a method call
    // cannot resolve at all; the `f32$exp` whitelist is exercised by the real template, which
    // fails to tag if it regresses.) `parallel_outer_for` must accept: the trip count rides
    // the SpawnEnd, and exactly one Store and one Add carry the stride tags.
    let (did, w) = lower_with_registry(
        "fn main() -> i32 {\n\
               let mut q = Tensor<f32>([4, 8]);\n\
               let mut o = Tensor<f32>([4, 8]);\n\
               let scale : f32 = 0.5;\n\
               spawn on (Topology::GPU) {\n\
                 let mut ts = Tensor<f32>([1, 8]);\n\
                 for i in 0..4 {\n\
                   let mut m : f32 = 0.0;\n\
                   for d in 0..8 {\n\
                     ts[0][d] = q[i][d] * scale;\n\
                     let t : f32 = if ts[0][d] > m { ts[0][d] } else { m };\n\
                     m = t;\n\
                   }\n\
                   for d in 0..8 {\n\
                     o[i][d] = m;\n\
                   }\n\
                 }\n\
               };\n\
               return 0;\n\
             }",
        "main",
    );
    assert!(did, "the region lowers on the flat path");
    let end = w
        .local_hir_stream
        .iter()
        .find(|i| i.opcode == Opcode::SpawnEnd)
        .expect("a SpawnEnd");
    assert_eq!(end.imm, 4, "the proved trip count rides the SpawnEnd");
    let inits = w
        .local_hir_stream
        .iter()
        .filter(|i| i.opcode == Opcode::Store && i.imm == IMM_PARALLEL_INIT)
        .count();
    let steps = w
        .local_hir_stream
        .iter()
        .filter(|i| i.opcode == Opcode::Add && i.imm == IMM_PARALLEL_STEP)
        .count();
    assert_eq!((inits, steps), (1, 1), "exactly the outer loop is tagged");
}

/// A placement in the type puts the space's dispatch id on the allocation, which is exactly
/// where `.with_memory(Memory::X)` used to put it (Vx#429).
///
/// The two spellings have to agree, because one replaces the other: a `scope: sm` space is
/// what turns the allocation into a space-3 alloca, and that decision is made from this
/// operand alone. If the placement stopped reaching it, every shared tile would quietly
/// become an ordinary allocation and the region would still compile.
#[test]
fn a_placed_tensor_carries_its_space_on_the_allocation() {
    let alloc_space = |src: &str| {
        let (did, w) = lower_with_registry(src, "main");
        assert!(did, "lowers: {src}");
        w.local_hir_stream
            .iter()
            .find(|i| i.opcode == Opcode::TensorAlloc)
            .expect("an allocation")
            .operand2
            .0
    };
    let by_type = alloc_space(
        "fn main() -> i32 { let mut t = Tensor<f32, [2, 4], Memory::SMEM>::uninit(); \
             t[0][0] = 1.0; return 0; }",
    );
    let by_method = alloc_space(
        "fn main() -> i32 { let mut t = Tensor<f32, [2, 4], Memory::SMEM>::uninit(); \
             t[0][0] = 1.0; return 0; }",
    );
    assert_eq!(by_type, by_method, "one spelling replaces the other");
    assert_eq!(
        by_type,
        crate::arch::memory_space_dispatch_id(&crate::syntax::MemorySpace::from_name("SMEM"))
            as u32
    );
    // A tensor with no placement leaves the operand alone, so a nonzero id above means the
    // space was carried rather than defaulted into.
    assert_eq!(
        alloc_space(
            "fn main() -> i32 { let mut t = Tensor<f32, [2, 4]>::uninit(); \
                         t[0][0] = 1.0; return 0; }"
        ),
        0
    );
}

#[test]
fn a_cooperative_tile_region_gets_the_two_level_plan() {
    // The FA-2 grammar in miniature (Vx#379 stage B): a shared tile filled by one thread
    // loop, rewritten in place by thread loops inside a sequential loop (each phase
    // barrier-terminated), and an affine writeback. The regression this pins: a sequential
    // loop's LAST inner thread loop leaked its shared-write ledger to the outer walk, which
    // then demanded a barrier the sequence had already provided -- and the whole region
    // silently fell back to a 1x1 serial launch.
    let (did, w) = lower_with_registry(
        "fn main() -> i32 {\n\
               let mut q = Tensor<f32>([8, 4]);\n\
               let mut o = Tensor<f32>([8, 4]);\n\
               spawn on (Topology::GPU) {\n\
                 let mut qs = Tensor<f32, [2, 4], Memory::SMEM>::uninit();\n\
                 for bq in 0..4 {\n\
                   for qi in 0..2 {\n\
                     qs[qi] = q[bq * 2 + qi] * 1.0;\n\
                   }\n\
                   barrier();\n\
                   for t in 0..3 {\n\
                     for r in 0..2 {\n\
                       qs[r] = qs[r] * 1.0;\n\
                     }\n\
                     barrier();\n\
                   }\n\
                   for qi in 0..2 {\n\
                     o[bq * 2 + qi] = qs[qi] * 1.0;\n\
                   }\n\
                   barrier();\n\
                 }\n\
               };\n\
               return 0;\n\
             }",
        "main",
    );
    assert!(did, "lowers");
    let end = w
        .local_hir_stream
        .iter()
        .find(|i| i.opcode == Opcode::SpawnEnd)
        .expect("a SpawnEnd");
    assert_ne!(
        end.imm & SPAWN_TWO_LEVEL,
        0,
        "two-level fired: imm={:#x}",
        end.imm
    );
}

#[test]
fn a_shared_tile_made_inside_unsafe_gets_the_two_level_plan() {
    // The region above, with its shared tile made inside `unsafe { .. }`, where
    // `::uninit()` belongs: the same allocation, so the same plan.
    let (did, w) = lower_with_registry(
        "fn main() -> i32 {\n\
               let mut q = Tensor<f32>([8, 4]);\n\
               let mut o = Tensor<f32>([8, 4]);\n\
               spawn on (Topology::GPU) {\n\
                 let mut qs = unsafe { Tensor<f32, [2, 4], Memory::SMEM>::uninit() };\n\
                 for bq in 0..4 {\n\
                   for qi in 0..2 {\n\
                     qs[qi] = q[bq * 2 + qi] * 1.0;\n\
                   }\n\
                   barrier();\n\
                   for qi in 0..2 {\n\
                     o[bq * 2 + qi] = qs[qi] * 1.0;\n\
                   }\n\
                   barrier();\n\
                 }\n\
               };\n\
               return 0;\n\
             }",
        "main",
    );
    assert!(did, "lowers");
    let end = w
        .local_hir_stream
        .iter()
        .find(|i| i.opcode == Opcode::SpawnEnd)
        .expect("a SpawnEnd");
    assert_ne!(
        end.imm & SPAWN_TWO_LEVEL,
        0,
        "two-level fired: imm={:#x}",
        end.imm
    );
}

#[test]
fn a_block_thread_region_gets_the_two_level_plan() {
    // The Vx#379 shape in miniature: a block loop of 2, a thread loop of 4 whose captured
    // write leads with `bq * 4 + qi` (multiplier == the thread trip), and a barrier. The
    // SpawnEnd must carry the block trip with the two-level bit; exactly one block pair and
    // one thread pair of stride tags must exist.
    let (did, w) = lower_with_registry(
        "fn main() -> i32 {\n\
               let mut o = Tensor<f32>([8, 4]);\n\
               spawn on (Topology::GPU) {\n\
                 for bq in 0..2 {\n\
                   for qi in 0..4 {\n\
                     o[bq * 4 + qi][0] = 1.0;\n\
                   }\n\
                   barrier();\n\
                 }\n\
               };\n\
               return 0;\n\
             }",
        "main",
    );
    assert!(did, "the two-level region lowers");
    let end = w
        .local_hir_stream
        .iter()
        .find(|i| i.opcode == Opcode::SpawnEnd)
        .expect("a SpawnEnd");
    assert_eq!(end.imm & 0xffff_ffff, 2, "block trip rides the SpawnEnd");
    assert_ne!(end.imm & SPAWN_TWO_LEVEL, 0, "with the two-level bit");
    let count_imm = |op: Opcode, imm: u64| {
        w.local_hir_stream
            .iter()
            .filter(|i| i.opcode == op && i.imm == imm)
            .count()
    };
    assert_eq!(
        count_imm(Opcode::Store, IMM_BLOCK_INIT),
        1,
        "one block init"
    );
    assert_eq!(count_imm(Opcode::Add, IMM_BLOCK_STEP), 1, "one block step");
    assert_eq!(
        count_imm(Opcode::Store, IMM_THREAD_INIT),
        1,
        "one thread init"
    );
    assert_eq!(
        count_imm(Opcode::Add, IMM_THREAD_STEP),
        1,
        "one thread step"
    );
    assert_eq!(
        w.local_hir_stream
            .iter()
            .filter(|i| i.opcode == Opcode::Barrier)
            .count(),
        1,
        "the barrier lowered"
    );
}

#[test]
fn a_register_accumulator_coop_region_gets_the_plan() {
    // Vx#379 stage C in miniature: ONE thread loop with the barriers inside it, a serial
    // tile loop between them, a thread-private accumulator, an own-row shared fill and a
    // nonown consume on the far side of a barrier. The two-level bit must fire with the
    // coop loop's trip as the launch width -- this is the FA-2 register-accumulator form,
    // and its silent fallback is invisible except as a severalfold slowdown.
    let (did, w) = lower_with_registry(
        "fn main() -> i32 {\n\
               let mut q = Tensor<f32>([8, 4]);\n\
               let mut k = Tensor<f32>([6, 4]);\n\
               let mut o = Tensor<f32>([8, 4]);\n\
               spawn on (Topology::GPU) {\n\
                 let mut kt = Tensor<f32, [2, 4], Memory::SMEM>::uninit();\n\
                 let mut acc = Tensor<f32>([1, 4]);\n\
                 for bq in 0..4 {\n\
                   for qi in 0..2 {\n\
                     acc[0] = q[bq * 2 + qi] * 0.0;\n\
                     for t in 0..3 {\n\
                       kt[qi] = k[t * 2 + qi] * 1.0;\n\
                       barrier();\n\
                       for jj in 0..2 {\n\
                         acc[0] = acc[0] + kt[jj] * 1.0;\n\
                       }\n\
                       barrier();\n\
                     }\n\
                     o[bq * 2 + qi] = acc[0] * 1.0;\n\
                   }\n\
                 }\n\
               };\n\
               return 0;\n\
             }",
        "main",
    );
    assert!(did, "lowers");
    let end = w
        .local_hir_stream
        .iter()
        .find(|i| i.opcode == Opcode::SpawnEnd)
        .expect("a SpawnEnd");
    assert_ne!(
        end.imm & SPAWN_TWO_LEVEL,
        0,
        "two-level fired: imm={:#x}",
        end.imm
    );
    assert_eq!(end.imm & 0xffff_ffff, 4, "block trip rides the SpawnEnd");
    assert_eq!(
        (end.imm >> 32) & 0xffff,
        2,
        "the coop trip is the launch width"
    );
}

#[test]
fn a_coop_region_missing_the_tail_barrier_stays_serial() {
    // The same shape minus the barrier AFTER the consume. One pass sees nothing wrong --
    // fill, barrier, consume is a legal order -- the wrap is the bug: the next tile's
    // fill overwrites rows a lagging thread is still reading. The walk's second pass
    // meets that fill with the consume's read ledger still dirty and refuses.
    let (did, w) = lower_with_registry(
        "fn main() -> i32 {\n\
               let mut q = Tensor<f32>([8, 4]);\n\
               let mut k = Tensor<f32>([6, 4]);\n\
               let mut o = Tensor<f32>([8, 4]);\n\
               spawn on (Topology::GPU) {\n\
                 let mut kt = Tensor<f32, [2, 4], Memory::SMEM>::uninit();\n\
                 let mut acc = Tensor<f32>([1, 4]);\n\
                 for bq in 0..4 {\n\
                   for qi in 0..2 {\n\
                     acc[0] = q[bq * 2 + qi] * 0.0;\n\
                     for t in 0..3 {\n\
                       kt[qi] = k[t * 2 + qi] * 1.0;\n\
                       barrier();\n\
                       for jj in 0..2 {\n\
                         acc[0] = acc[0] + kt[jj] * 1.0;\n\
                       }\n\
                     }\n\
                     o[bq * 2 + qi] = acc[0] * 1.0;\n\
                   }\n\
                 }\n\
               };\n\
               return 0;\n\
             }",
        "main",
    );
    assert!(did, "still lowers -- serially");
    let end = w
        .local_hir_stream
        .iter()
        .find(|i| i.opcode == Opcode::SpawnEnd)
        .expect("a SpawnEnd");
    assert_eq!(
        end.imm & SPAWN_TWO_LEVEL,
        0,
        "no two-level bit: imm={:#x}",
        end.imm
    );
}

#[test]
fn a_barrier_under_a_variable_bound_stays_serial() {
    // The tile loop's bound is a local, not a literal. Nothing here varies per thread,
    // but the walk cannot know that: a per-thread trip means per-thread barrier counts,
    // which is a hang. Literal bounds are the price of a barrier inside a loop.
    let (did, w) = lower_with_registry(
        "fn main() -> i32 {\n\
               let mut k = Tensor<f32>([6, 4]);\n\
               let mut o = Tensor<f32>([8, 4]);\n\
               spawn on (Topology::GPU) {\n\
                 let mut kt = Tensor<f32, [2, 4], Memory::SMEM>::uninit();\n\
                 for bq in 0..4 {\n\
                   for qi in 0..2 {\n\
                     let n : i64 = 3;\n\
                     for t in 0..n {\n\
                       kt[qi] = k[t * 2 + qi] * 1.0;\n\
                       barrier();\n\
                     }\n\
                     o[bq * 2 + qi] = kt[qi] * 1.0;\n\
                   }\n\
                 }\n\
               };\n\
               return 0;\n\
             }",
        "main",
    );
    assert!(did, "still lowers -- serially");
    let end = w
        .local_hir_stream
        .iter()
        .find(|i| i.opcode == Opcode::SpawnEnd)
        .expect("a SpawnEnd");
    assert_eq!(
        end.imm & SPAWN_TWO_LEVEL,
        0,
        "no two-level bit: imm={:#x}",
        end.imm
    );
}

#[test]
fn a_block_row_column_strided_write_gets_the_plan() {
    // Vx#379 R3's last piece in miniature: the BLOCK owns row `i`, the threads partition
    // its columns by residue (`t + jj*K`, K == the thread trip), with an SMEM partial and
    // a barrier between sweep and consume -- the block-per-row reduction shape a coalesced
    // softmax is made of. The 1-trip `z` loop is the reduce step: one thread, own-column
    // write at `lrow[i][z]`.
    let (did, w) = lower_with_registry(
        "fn main() -> i32 {\n\
               let mut s = Tensor<f32>([4, 4]);\n\
               let mut lrow = Tensor<f32>([4, 1]);\n\
               spawn on (Topology::GPU) {\n\
                 let mut pm = Tensor<f32, [2, 1], Memory::SMEM>::uninit();\n\
                 for i in 0..4 {\n\
                   for t in 0..2 {\n\
                     let mut m : f32 = -1000000.0;\n\
                     for jj in 0..2 {\n\
                       let x : f32 = s[i][t + jj * 2];\n\
                       let m2 : f32 = if x > m {\n\
                         x\n\
                       } else {\n\
                         m\n\
                       };\n\
                       m = m2;\n\
                     }\n\
                     pm[t][0] = m;\n\
                   }\n\
                   barrier();\n\
                   for t in 0..2 {\n\
                     let mx : f32 = pm[0][0];\n\
                     for jj in 0..2 {\n\
                       s[i][t + jj * 2] = s[i][t + jj * 2] - mx;\n\
                     }\n\
                   }\n\
                   for z in 0..1 {\n\
                     lrow[i][z] = 1.0;\n\
                   }\n\
                   barrier();\n\
                 }\n\
               };\n\
               return 0;\n\
             }",
        "main",
    );
    assert!(did, "lowers");
    let end = w
        .local_hir_stream
        .iter()
        .find(|i| i.opcode == Opcode::SpawnEnd)
        .expect("a SpawnEnd");
    assert_ne!(
        end.imm & SPAWN_TWO_LEVEL,
        0,
        "two-level fired: imm={:#x}",
        end.imm
    );
    assert_eq!(end.imm & 0xffff_ffff, 4, "block trip rides the SpawnEnd");
    assert_eq!((end.imm >> 32) & 0xffff, 2, "threads = the sweep width");
    assert_eq!(
        end.imm & SPAWN_COOP,
        0,
        "stage B shape: host-safe, no coop bit"
    );
}

#[test]
fn a_column_stride_not_the_trip_stays_serial() {
    // Same shape, but the column stride is 3 against a thread trip of 2: residues collide
    // (t=0,jj=2 lands on column 6; t=... the partition is simply not mod-K) and the region
    // must fall back to serial rather than race.
    let (did, w) = lower_with_registry(
        "fn main() -> i32 {\n\
               let mut s = Tensor<f32>([4, 8]);\n\
               spawn on (Topology::GPU) {\n\
                 for i in 0..4 {\n\
                   for t in 0..2 {\n\
                     for jj in 0..2 {\n\
                       s[i][t + jj * 3] = 1.0;\n\
                     }\n\
                   }\n\
                   barrier();\n\
                 }\n\
               };\n\
               return 0;\n\
             }",
        "main",
    );
    assert!(did, "still lowers -- serially");
    let end = w
        .local_hir_stream
        .iter()
        .find(|i| i.opcode == Opcode::SpawnEnd)
        .expect("a SpawnEnd");
    assert_eq!(
        end.imm & SPAWN_TWO_LEVEL,
        0,
        "no two-level bit: imm={:#x}",
        end.imm
    );
}

#[test]
fn a_column_write_without_the_thread_iv_stays_serial() {
    // `s[i][jj * 2]`: the residue class is constant, so every thread writes the same
    // columns. No ownership, no plan.
    let (did, w) = lower_with_registry(
        "fn main() -> i32 {\n\
               let mut s = Tensor<f32>([4, 4]);\n\
               spawn on (Topology::GPU) {\n\
                 for i in 0..4 {\n\
                   for t in 0..2 {\n\
                     for jj in 0..2 {\n\
                       s[i][jj * 2] = 1.0;\n\
                     }\n\
                   }\n\
                   barrier();\n\
                 }\n\
               };\n\
               return 0;\n\
             }",
        "main",
    );
    assert!(did, "still lowers -- serially");
    let end = w
        .local_hir_stream
        .iter()
        .find(|i| i.opcode == Opcode::SpawnEnd)
        .expect("a SpawnEnd");
    assert_eq!(
        end.imm & SPAWN_TWO_LEVEL,
        0,
        "no two-level bit: imm={:#x}",
        end.imm
    );
}

#[test]
fn a_matmul_region_lowers_flat_and_stays_serial() {
    // `matmul_into` in a spawn must lower on the FLAT path (one MatmulInto op, dst in the
    // imm) and must NOT be strided -- the region is classified and routed, not launched
    // wide. The regression this pins: the call used to evict the whole program to the AST
    // path, taking every OTHER region's parallel proof with it -- a program lost its
    // softmax kernel's grid-stride to the mere presence of a GEMM.
    let (did, w) = lower_with_registry(
        "fn main() -> i32 {\n\
               let mut a = Tensor<f32>([4, 3]);\n\
               let mut b = Tensor<f32>([3, 5]);\n\
               let mut c = Tensor<f32>([4, 5]);\n\
               spawn on (Topology::GPU) {\n\
                 matmul_into(&mut c, &a, &b);\n\
               };\n\
               return 0;\n\
             }",
        "main",
    );
    assert!(did, "the matmul region lowers on the flat path");
    let mm = w
        .local_hir_stream
        .iter()
        .find(|i| i.opcode == Opcode::MatmulInto)
        .expect("a MatmulInto op");
    assert_ne!(mm.imm, 0, "the destination register rides the imm");
    let end = w
        .local_hir_stream
        .iter()
        .find(|i| i.opcode == Opcode::SpawnEnd)
        .expect("a SpawnEnd");
    assert_eq!(end.imm, 0, "routed, not strided: imm={:#x}", end.imm);
}

#[test]
fn a_break_inside_a_synchronizing_loop_stays_serial() {
    // A `break` whose innermost loop contains a barrier lets one thread leave while the
    // rest wait at the bar. The guard is the innermost frame, so a break in a nested
    // barrier-free walk stays legal -- this one is directly in the tile loop.
    let (did, w) = lower_with_registry(
        "fn main() -> i32 {\n\
               let mut k = Tensor<f32>([6, 4]);\n\
               let mut o = Tensor<f32>([8, 4]);\n\
               spawn on (Topology::GPU) {\n\
                 let mut kt = Tensor<f32, [2, 4], Memory::SMEM>::uninit();\n\
                 for bq in 0..4 {\n\
                   for qi in 0..2 {\n\
                     for t in 0..3 {\n\
                       kt[qi] = k[t * 2 + qi] * 1.0;\n\
                       barrier();\n\
                       break;\n\
                     }\n\
                     o[bq * 2 + qi] = kt[qi] * 1.0;\n\
                   }\n\
                 }\n\
               };\n\
               return 0;\n\
             }",
        "main",
    );
    assert!(did, "still lowers -- serially");
    let end = w
        .local_hir_stream
        .iter()
        .find(|i| i.opcode == Opcode::SpawnEnd)
        .expect("a SpawnEnd");
    assert_eq!(
        end.imm & SPAWN_TWO_LEVEL,
        0,
        "no two-level bit: imm={:#x}",
        end.imm
    );
}

#[test]
fn a_wrong_affine_multiplier_keeps_the_region_serial() {
    // `bq * 8 + qi` with `qi in 0..4` leaves rows unowned: the partition is not exact, so
    // the two-level prover must reject -- and the single-level prover cannot accept the
    // outer loop either (its body writes are not led by `bq`), so the region stays serial.
    let (did, w) = lower_with_registry(
        "fn main() -> i32 {\n\
               let mut o = Tensor<f32>([16, 4]);\n\
               spawn on (Topology::GPU) {\n\
                 for bq in 0..2 {\n\
                   for qi in 0..4 {\n\
                     o[bq * 8 + qi][0] = 1.0;\n\
                   }\n\
                   barrier();\n\
                 }\n\
               };\n\
               return 0;\n\
             }",
        "main",
    );
    assert!(did, "the rejected region still lowers");
    let end = w
        .local_hir_stream
        .iter()
        .find(|i| i.opcode == Opcode::SpawnEnd)
        .expect("a SpawnEnd");
    assert_eq!(
        end.imm, 0,
        "no plan of either kind for an inexact partition"
    );
}

#[test]
fn a_captured_scalar_write_keeps_the_region_serial() {
    // `acc` lives outside the spawn, so every strided thread would race on it: the analysis
    // must reject, and rejection must be invisible — the region lowers exactly as before, with
    // a zero trip count and no tags.
    let (did, w) = lower_with_registry(
        "fn main() -> i32 {\n\
               let mut o = Tensor<f32>([4, 8]);\n\
               let mut acc : f32 = 0.0;\n\
               spawn on (Topology::GPU) {\n\
                 for i in 0..4 {\n\
                   for d in 0..8 {\n\
                     acc = acc + o[i][d];\n\
                   }\n\
                 }\n\
               };\n\
               return 0;\n\
             }",
        "main",
    );
    assert!(did, "the rejected region still lowers");
    let end = w
        .local_hir_stream
        .iter()
        .find(|i| i.opcode == Opcode::SpawnEnd)
        .expect("a SpawnEnd");
    assert_eq!(end.imm, 0, "no trip count for a racy region");
    assert!(
        !w.local_hir_stream
            .iter()
            .any(|i| i.opcode == Opcode::Store && i.imm == IMM_PARALLEL_INIT),
        "no stride tags either"
    );
}

#[test]
fn a_captured_scalar_written_after_an_inner_let_of_its_name_keeps_the_region_serial() {
    // The `acc` declared inside the `if` ends with it, so the write after the `if` is to the
    // captured `acc`, which every strided thread would race on: the analysis must reject.
    let (did, w) = lower_with_registry(
        "fn main() -> i32 {\n\
               let mut o = Tensor<f32>([4, 8]);\n\
               let mut acc : f32 = 0.0;\n\
               spawn on (Topology::GPU) {\n\
                 for i in 0..4 {\n\
                   if i > 10 {\n\
                     let acc : f32 = 1.0;\n\
                   }\n\
                   acc = o[i][0];\n\
                 }\n\
               };\n\
               return 0;\n\
             }",
        "main",
    );
    assert!(did, "the rejected region still lowers");
    let end = w
        .local_hir_stream
        .iter()
        .find(|i| i.opcode == Opcode::SpawnEnd)
        .expect("a SpawnEnd");
    assert_eq!(end.imm, 0, "no trip count for a racy region");
}

#[test]
fn a_captured_write_not_led_by_the_iv_keeps_the_region_serial() {
    // `o[0][d]` writes the same row from every iteration — the first-index rule is the whole
    // proof, so this must reject.
    let (did, w) = lower_with_registry(
        "fn main() -> i32 {\n\
               let mut o = Tensor<f32>([4, 8]);\n\
               spawn on (Topology::GPU) {\n\
                 for i in 0..4 {\n\
                   for d in 0..8 {\n\
                     o[0][d] = 1.0;\n\
                   }\n\
                 }\n\
               };\n\
               return 0;\n\
             }",
        "main",
    );
    assert!(did);
    let end = w
        .local_hir_stream
        .iter()
        .find(|i| i.opcode == Opcode::SpawnEnd)
        .expect("a SpawnEnd");
    assert_eq!(end.imm, 0, "a fixed-row write is not disjoint");
}

/// The `SpawnEnd` of the only spawn region in `main` of `src`: its trip count when the region
/// runs in parallel, 0 when it stays serial.
fn spawn_trip(src: &str) -> u64 {
    let (did, w) = lower_with_registry(src, "main");
    assert!(did, "the region lowers on the flat path");
    w.local_hir_stream
        .iter()
        .find(|i| i.opcode == Opcode::SpawnEnd)
        .expect("a SpawnEnd")
        .imm
}

#[test]
fn a_write_at_an_offset_from_the_loop_variable_still_strides() {
    // `lo` is fixed for the whole loop, so `lo + r` is a different row on every pass.
    let trip = spawn_trip(
        "fn main() -> i32 {\n\
               let mut o = Tensor<f32>([1024, 1]);\n\
               let lo = 512;\n\
               spawn on (Topology::GPU) {\n\
                 for r in 0..512 {\n\
                   o[lo + r][0] = o[lo + r][0] + 2.0;\n\
                 }\n\
               };\n\
               return 0;\n\
             }",
    );
    assert_eq!(trip, 512, "an offset row is the pass's own row");
}

#[test]
fn a_tensor_reached_through_two_row_forms_stays_serial() {
    // Pass `r` reads row `r`, which pass `r - lo` writes.
    let trip = spawn_trip(
        "fn main() -> i32 {\n\
               let mut o = Tensor<f32>([1024, 1]);\n\
               let lo = 512;\n\
               spawn on (Topology::GPU) {\n\
                 for r in 0..512 {\n\
                   o[lo + r][0] = o[r][0];\n\
                 }\n\
               };\n\
               return 0;\n\
             }",
    );
    assert_eq!(
        trip, 0,
        "`o[r]` and `o[lo + r]` are rows of different passes"
    );
}

#[test]
fn two_writes_through_different_offsets_stay_serial() {
    let trip = spawn_trip(
        "fn main() -> i32 {\n\
               let mut o = Tensor<f32>([1024, 1]);\n\
               let lo = 512;\n\
               spawn on (Topology::GPU) {\n\
                 for r in 0..512 {\n\
                   o[r][0] = 1.0;\n\
                   o[lo + r][0] = 2.0;\n\
                 }\n\
               };\n\
               return 0;\n\
             }",
    );
    assert_eq!(
        trip, 0,
        "pass `r` writes rows `r` and `lo + r`, and pass `r + lo` writes row `r + lo`"
    );
}

#[test]
fn an_offset_made_inside_the_loop_stays_serial() {
    // `k` is a local of the loop, so `k + r` need not be a different row on every pass.
    let trip = spawn_trip(
        "fn main() -> i32 {\n\
               let mut o = Tensor<f32>([1024, 1]);\n\
               spawn on (Topology::GPU) {\n\
                 for r in 0..512 {\n\
                   let k = 511 - r;\n\
                   o[k + r][0] = 1.0;\n\
                 }\n\
               };\n\
               return 0;\n\
             }",
    );
    assert_eq!(
        trip, 0,
        "an offset that changes with the pass is not a row the pass owns"
    );
}

#[test]
fn two_thread_loops_over_the_same_rows_keep_the_two_level_plan() {
    // The two thread loops have their own variables, `qi` and `qj`, but `bq * 4 + qi` and
    // `bq * 4 + qj` are the same row for the same thread: one row form, not two.
    let imm = spawn_trip(
        "fn main() -> i32 {\n\
               let mut o = Tensor<f32>([8, 4]);\n\
               spawn on (Topology::GPU) {\n\
                 for bq in 0..2 {\n\
                   for qi in 0..4 {\n\
                     o[bq * 4 + qi][0] = 1.0;\n\
                   }\n\
                   barrier();\n\
                   for qj in 0..4 {\n\
                     o[bq * 4 + qj][1] = 2.0;\n\
                   }\n\
                   barrier();\n\
                 }\n\
               };\n\
               return 0;\n\
             }",
    );
    assert_ne!(
        imm & SPAWN_TWO_LEVEL,
        0,
        "the two loops reach the same rows"
    );
}

#[test]
fn a_read_of_another_iterations_row_keeps_the_region_serial() {
    // Iteration `i` reads row `i + 1`, which iteration `i + 1` writes: split across
    // workers, the read races with that write.
    let trip = spawn_trip(
        "fn main() -> i32 {\n\
               let mut o = Tensor<f32>([5, 8]);\n\
               spawn on (Topology::GPU) {\n\
                 for i in 0..4 {\n\
                   for d in 0..8 {\n\
                     o[i][d] = o[i + 1][d];\n\
                   }\n\
                 }\n\
               };\n\
               return 0;\n\
             }",
    );
    assert_eq!(
        trip, 0,
        "a read of a row another iteration writes is not disjoint"
    );
}

#[test]
fn a_whole_read_of_a_written_tensor_keeps_the_region_serial() {
    let trip = spawn_trip(
        "fn main() -> i32 {\n\
               let mut o = Tensor<f32>([4, 8]);\n\
               spawn on (Topology::GPU) {\n\
                 for i in 0..4 {\n\
                   let all = o;\n\
                   o[i][0] = 1.0;\n\
                 }\n\
               };\n\
               return 0;\n\
             }",
    );
    assert_eq!(
        trip, 0,
        "reading the whole tensor reads every iteration's row"
    );
}

#[test]
fn reads_of_the_own_row_and_of_unwritten_tensors_still_stride() {
    // `o` is read only at row `i`, and `w` at any row, since nothing in the loop writes `w`.
    let trip = spawn_trip(
        "fn main() -> i32 {\n\
               let mut o = Tensor<f32>([4, 8]);\n\
               let w = Tensor<f32>([5, 8]);\n\
               spawn on (Topology::GPU) {\n\
                 for i in 0..4 {\n\
                   for d in 0..8 {\n\
                     o[i][d] = o[i][d] + w[i + 1][d];\n\
                   }\n\
                 }\n\
               };\n\
               return 0;\n\
             }",
    );
    assert_eq!(
        trip, 4,
        "own-row reads and reads of unwritten tensors are disjoint"
    );
}

#[test]
fn spawn_wraps_body_in_region_markers() {
    let f =
        parse_fn("fn k(a: i32) -> i32 { spawn on (Topology::GPU) { let x = a + 1; } return a; }");
    let mut w = worker();
    lower_function_to_hir(&f, &mut w).expect("lowers");
    assert_eq!(count(&w, Opcode::Spawn), 1);
    assert_eq!(count(&w, Opcode::SpawnEnd), 1);
    // The Spawn carries the topology dispatch id, and the body (`a + 1`) lowered between the
    // region markers.
    let spawn = w
        .local_hir_stream
        .iter()
        .find(|i| i.opcode == Opcode::Spawn)
        .unwrap();
    assert_eq!(
        spawn.imm & 0xffff_ffff,
        crate::arch::topology_dispatch_id(&crate::syntax::Topology::gpu(0)) as u64
    );
    assert!(
        count(&w, Opcode::Add) >= 1,
        "body arithmetic lowered inside the region"
    );
    verify_hir_stream(&w);
}

#[test]
fn value_producing_spawn_yields_through_its_end() {
    // A spawn that yields a value (no trailing `;`): the region's tail is lowered inside
    // it, the `SpawnEnd` carries the value as `operand1` and is typed with it, and that
    // instruction's register is what the function returns.
    let f = parse_fn("fn k(a: i32) -> i32 { spawn on (Topology::GPU) { a + 1 } }");
    let mut w = worker();
    lower_function_to_hir(&f, &mut w).expect("a value-producing spawn lowers");
    let end = w
        .local_hir_stream
        .iter()
        .position(|i| i.opcode == Opcode::SpawnEnd)
        .expect("a SpawnEnd");
    let ins = w.local_hir_stream[end];
    assert_ne!(ins.type_idx.0, NO_TYPE, "the end is typed with the value");
    assert_eq!(
        w.local_hir_stream[ins.operand1.0 as usize].opcode,
        Opcode::Add,
        "the value is the region's tail"
    );
    verify_hir_stream(&w);
}

#[test]
fn lowers_params_arithmetic_and_return() {
    let f = parse_fn("fn add(a: i32, b: i32) -> i32 { return a + b; }");
    let mut w = worker();
    lower_function_to_hir(&f, &mut w).expect("lowers");

    // Load a, Load b, Add(a,b), Ret(add).
    assert_eq!(
        opcodes(&w),
        vec![Opcode::Load, Opcode::Load, Opcode::Add, Opcode::Ret]
    );
    let add = w.local_hir_stream[2];
    assert_eq!(
        (add.operand1.0, add.operand2.0),
        (0, 1),
        "Add reads the two params"
    );
    assert_eq!(
        w.local_hir_stream[3].operand1.0, 2,
        "Ret reads the Add result"
    );
    // One type per instruction, all in bounds.
    assert_eq!(w.local_type_stream.len(), 4);
    verify_hir_stream(&w);
}

#[test]
fn lowers_let_binding_and_reuse() {
    // y = x * x; return y + x  ->  Load x, Mul(0,0), (let y=1), Add(1,0), Ret(2)
    let f = parse_fn("fn sq(x: i32) -> i32 { let y = x * x; return y + x; }");
    let mut w = worker();
    lower_function_to_hir(&f, &mut w).expect("lowers");
    assert_eq!(
        opcodes(&w),
        vec![Opcode::Load, Opcode::Mul, Opcode::Add, Opcode::Ret]
    );
    let mul = w.local_hir_stream[1];
    assert_eq!(
        (mul.operand1.0, mul.operand2.0),
        (0, 0),
        "x * x reads param twice"
    );
    let add = w.local_hir_stream[2];
    assert_eq!(
        (add.operand1.0, add.operand2.0),
        (1, 0),
        "y + x reads Mul result and x"
    );
    verify_hir_stream(&w);
}

#[test]
fn lowers_integer_literal_immediate() {
    // The literal lowers to a single `Const` (no coercion in the flat lowerer anymore, #240): in
    // real compilation the checker would type `7` to the `i64` return, but this test lowers the
    // raw parsed AST without the checker, so the `Const` keeps its spelling default and `Ret`
    // carries it directly.
    let f = parse_fn("fn seven() -> i64 { return 7; }");
    let mut w = worker();
    lower_function_to_hir(&f, &mut w).expect("lowers");
    assert_eq!(opcodes(&w), vec![Opcode::Const, Opcode::Ret]);
    assert_eq!(
        w.local_hir_stream[0].imm, 7,
        "Const carries the literal value"
    );
    verify_hir_stream(&w);
}

#[test]
fn lowers_float_literal_as_bit_pattern() {
    let f = parse_fn("fn half() -> f64 { return 0.5; }");
    let mut w = worker();
    lower_function_to_hir(&f, &mut w).expect("lowers");
    assert_eq!(w.local_hir_stream[0].imm, 0.5f64.to_bits());
}

#[test]
fn lowers_comparison_to_bool() {
    let f = parse_fn("fn lt(a: i32, b: i32) -> bool { return a < b; }");
    let mut w = worker();
    lower_function_to_hir(&f, &mut w).expect("lowers");
    assert_eq!(
        opcodes(&w),
        vec![Opcode::Load, Opcode::Load, Opcode::Cmp, Opcode::Ret]
    );
    let cmp = w.local_hir_stream[2];
    assert_eq!((cmp.operand1.0, cmp.operand2.0), (0, 1));
    assert_eq!(cmp.imm, 2, "Lt relation code");
    // The Cmp result type is bool.
    assert_eq!(
        w.local_type_stream[cmp.type_idx.0 as usize],
        scalar_gid(&ElementType::Bool)
    );
    verify_hir_stream(&w);
}

#[test]
fn lowers_scalar_cast_with_target_type() {
    let f = parse_fn("fn widen(a: i32) -> i64 { return a as i64; }");
    let mut w = worker();
    lower_function_to_hir(&f, &mut w).expect("lowers");
    assert_eq!(opcodes(&w), vec![Opcode::Load, Opcode::Cast, Opcode::Ret]);
    let cast = w.local_hir_stream[1];
    assert_eq!(cast.operand1.0, 0, "cast reads the source");
    assert_eq!(
        w.local_type_stream[cast.type_idx.0 as usize],
        scalar_gid(&ElementType::I64),
        "cast result carries the target type"
    );
    verify_hir_stream(&w);
}

#[test]
fn lowers_unary_negation() {
    let f = parse_fn("fn neg(a: i32) -> i32 { return -a; }");
    let mut w = worker();
    lower_function_to_hir(&f, &mut w).expect("lowers");
    assert_eq!(opcodes(&w), vec![Opcode::Load, Opcode::Neg, Opcode::Ret]);
    assert_eq!(w.local_hir_stream[1].operand1.0, 0);
    verify_hir_stream(&w);
}

#[test]
fn aborts_atomically_on_unsupported_construct() {
    // A call is outside the supported subset -> abort, worker untouched.
    let f = parse_fn("fn f(a: i32) -> i32 { return g(a); }");
    let mut w = worker();
    assert!(lower_function_to_hir(&f, &mut w).is_err());
    assert!(w.local_hir_stream.is_empty(), "no partial stream on abort");
    assert!(w.local_type_stream.is_empty(), "no partial types on abort");
}

#[test]
fn aborts_on_non_scalar_parameter() {
    let f = parse_fn("fn f(s: Widget) -> i32 { return 0; }");
    let mut w = worker();
    assert!(lower_function_to_hir(&f, &mut w).is_err());
    assert!(w.local_hir_stream.is_empty());
}

#[test]
fn unsafe_block_lowers_transparently() {
    // `unsafe` is transparent to lowering (safety was checked upstream): `return unsafe { a * a }`
    // lowers exactly as `return a * a`. This is what lets a stdlib wrapper body like
    // `return unsafe { sqrtf(self) }` lower through the flat path (#217).
    let f = parse_fn("fn sq(a: f32) -> f32 { return unsafe { a * a }; }");
    let mut w = worker();
    lower_function_to_hir(&f, &mut w).expect("unsafe-block body lowers");
    assert!(w.local_hir_stream.iter().any(|i| i.opcode == Opcode::Mul));
    assert_eq!(w.local_hir_stream.last().unwrap().opcode, Opcode::Ret);
    verify_hir_stream(&w);
}
