//! Tensors: allocation, indexing, stores, and the reduction and matmul kernels.
//!
//! One `impl` block on the per-function emitter; the dispatch that reaches these
//! lives in `step`, next to the struct.

use super::super::*;

/// The element type of a memref spelling: `memref<8x16xf32>` -> `f32`.
///
/// The element is the segment after the last `x`, shorn of the closing `>` and of a memory-space
/// suffix (`memref<4x4xf32, 3>`). A rank-0 memref has no `x` and is all element.
fn memref_element(memty: &str) -> Option<&str> {
    let inner = memty.strip_prefix("memref<")?;
    let inner = inner.split(',').next()?.trim_end_matches('>');
    Some(inner.rsplit_once('x').map_or(inner, |(_, el)| el))
}

impl FnEmit<'_> {
    /// Give `val`, of memref type `src_ty`, the type `dst_ty`, and return the new value's name.
    ///
    /// A `memref.cast` is enough unless `val` is a strided view (a row, `memref<4xf32,
    /// strided<[1], offset: ?>>`) and `dst_ty` has no layout. A plain memref's offset is 0 by its
    /// type, so the code using it never reads the descriptor's offset, and a cast would hand it
    /// the start of the tensor rather than the row. Instead the descriptor's data pointer is
    /// moved to the row and its offset set to 0. The view must be contiguous: a plain memref has
    /// row-major strides. A view in a GPU memory space is refused, as its pointers are not `ptr`.
    pub(crate) fn cast_memref_value(
        &mut self,
        tag: &str,
        val: &str,
        src_ty: &str,
        dst_ty: &str,
    ) -> Lowered<String> {
        if src_ty == dst_ty {
            return Ok(val.to_string());
        }
        if !src_ty.contains("strided<") || dst_ty.contains("strided<") {
            let c = format!("%{tag}");
            self.body += &format!("  {c} = memref.cast {val} : {src_ty} to {dst_ty}\n");
            return Ok(c);
        }
        if !src_ty.ends_with("offset: ?>>") && !src_ty.ends_with("offset: 0>>") {
            return Err(crate::emitter_gap!());
        }
        let (dims_x, et) = memref_lead_dims_and_elem(src_ty).ok_or(crate::emitter_gap!())?;
        let dims: Vec<&str> = dims_x.split('x').filter(|d| !d.is_empty()).collect();
        let strides: Vec<&str> = src_ty
            .split("strided<[")
            .nth(1)
            .and_then(|s| s.split(']').next())
            .ok_or(crate::emitter_gap!())?
            .split(',')
            .map(|s| s.trim())
            .collect();
        if dims.is_empty() || strides.len() != dims.len() {
            return Err(crate::emitter_gap!());
        }
        // Each stride must be the product of the sizes after it. A `?` cannot be checked here;
        // only indexing a row out of a tensor makes one, and a row of a tensor is contiguous.
        let mut expect = Some(1i64);
        for k in (0..dims.len()).rev() {
            if let (Some(e), Ok(s)) = (expect, strides[k].parse::<i64>()) {
                if s != e {
                    return Err(crate::emitter_gap!());
                }
            }
            expect = expect.zip(dims[k].parse::<i64>().ok()).map(|(e, d)| e * d);
        }
        let desc = memref_descriptor_ty(dims.len());
        let plain = format!("memref<{dims_x}{et}>");
        let d = format!("%{tag}_d");
        let al = format!("%{tag}_al");
        let off = format!("%{tag}_off");
        let p = format!("%{tag}_p");
        let z = format!("%{tag}_z");
        let d1 = format!("%{tag}_d1");
        let d2 = format!("%{tag}_d2");
        let r = format!("%{tag}_r");
        self.body +=
            &format!("  {d} = builtin.unrealized_conversion_cast {val} : {src_ty} to {desc}\n");
        self.body += &format!("  {al} = llvm.extractvalue {d}[1] : {desc}\n");
        self.body += &format!("  {off} = llvm.extractvalue {d}[2] : {desc}\n");
        self.body += &format!(
            "  {p} = llvm.getelementptr {al}[{off}] : (!llvm.ptr, i64) -> !llvm.ptr, {et}\n"
        );
        self.body += &format!("  {z} = llvm.mlir.constant(0 : i64) : i64\n");
        self.body += &format!("  {d1} = llvm.insertvalue {p}, {d}[1] : {desc}\n");
        self.body += &format!("  {d2} = llvm.insertvalue {z}, {d1}[2] : {desc}\n");
        self.body +=
            &format!("  {r} = builtin.unrealized_conversion_cast {d2} : {desc} to {plain}\n");
        if plain == dst_ty {
            return Ok(r);
        }
        let c = format!("%{tag}");
        self.body += &format!("  {c} = memref.cast {r} : {plain} to {dst_ty}\n");
        Ok(c)
    }

    // `t.clone()`: a new buffer of the source's shape and a `memref.copy` into it. The copy reads
    // the source through its own layout, so cloning a row copies that row.
    /// `t.as_ptr()`: the address of the first element, as a bare `!llvm.ptr`. The descriptor's
    /// aligned pointer plus its offset in bytes, so a row view answers with the row's start and
    /// not the buffer's. A tensor in GPU memory is an ordinary descriptor whose pointer is the
    /// device address, so the same code hands a C function what cuBLAS expects.
    pub(crate) fn op_tensor_data_ptr(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let s = ins.operand1.0 as usize;
        let src = self.names.get(s).ok_or(crate::emitter_gap!())?.clone();
        let src_mem = self
            .mem_of
            .get(s)
            .cloned()
            .flatten()
            .ok_or(crate::emitter_gap!())?;
        let (dims_x, et) = memref_lead_dims_and_elem(&src_mem).ok_or(crate::emitter_gap!())?;
        let rank = dims_x.split('x').filter(|d| !d.is_empty()).count();
        let bytes = crate::codegen::generator::scalar_type_bits(et)
            .ok_or(crate::emitter_gap!())?
            .div_ceil(8);
        let space_sfx = if src_mem.ends_with(", 3>") { ", 3" } else { "" };
        let buf = format!("%dp{idx}_b");
        let off = format!("%dp{idx}_o");
        let rest: Vec<String> = (0..2 * rank).map(|k| format!("%dp{idx}_r{k}")).collect();
        let results = [vec![buf, off.clone()], rest].concat().join(", ");
        let index_tys = vec!["index"; 1 + 2 * rank].join(", ");
        self.body += &format!(
            "  {results} = memref.extract_strided_metadata {src} : {src_mem} -> memref<{et}{space_sfx}>, {index_tys}\n"
        );
        self.body += &format!(
            "  %dp{idx}_a = memref.extract_aligned_pointer_as_index {src} : {src_mem} -> index\n"
        );
        self.body += &format!("  %dp{idx}_e = arith.constant {bytes} : index\n");
        self.body += &format!("  %dp{idx}_m = arith.muli {off}, %dp{idx}_e : index\n");
        self.body += &format!("  %dp{idx}_s = arith.addi %dp{idx}_a, %dp{idx}_m : index\n");
        self.body += &format!("  %dp{idx}_i = arith.index_cast %dp{idx}_s : index to i64\n");
        let n = format!("%v{idx}");
        self.body += &format!("  {n} = llvm.inttoptr %dp{idx}_i : i64 to !llvm.ptr\n");
        self.names[idx] = n;
        self.ptr_of[idx] = true;
        Ok(())
    }

    pub(crate) fn op_tensor_clone(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let gid = *self
            .types
            .get(ins.type_idx.0 as usize)
            .ok_or(crate::emitter_gap!())?;
        let (elem, shape) = self
            .ctx
            .tensors
            .get(&gid)
            .ok_or(crate::emitter_gap!())?
            .clone();
        let memty = tensor_memref_ty(&elem, &shape).ok_or(crate::emitter_gap!())?;
        let s = ins.operand1.0 as usize;
        let src = self.names.get(s).ok_or(crate::emitter_gap!())?.clone();
        let src_mem = self
            .mem_of
            .get(s)
            .cloned()
            .flatten()
            .ok_or(crate::emitter_gap!())?;
        let n = match self.nrvo_slot(idx, &memty) {
            Some(slot) => slot,
            None => {
                let mut sizes = Vec::new();
                for (k, d) in shape.iter().enumerate() {
                    if d == DYN_DIM {
                        let c = format!("%tcl{idx}_c{k}");
                        let v = format!("%tcl{idx}_d{k}");
                        self.body += &format!("  {c} = arith.constant {k} : index\n");
                        self.body += &format!("  {v} = memref.dim {src}, {c} : {src_mem}\n");
                        sizes.push(v);
                    }
                }
                let n = format!("%v{idx}");
                self.body += &format!("  {n} = memref.alloc({}) : {memty}\n", sizes.join(", "));
                n
            }
        };
        self.body += &format!("  memref.copy {src}, {n} : {src_mem} to {memty}\n");
        self.names[idx] = n;
        self.mem_of[idx] = Some(memty);
        Ok(())
    }

    // Allocate a tensor buffer (`Tensor<T>([..])`): a static `memref` of the shape recovered
    // from the side table by GID. Its register is tracked in `mem_of` for later index/store.
    pub(crate) fn op_tensor_alloc(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let gid = *self
            .types
            .get(ins.type_idx.0 as usize)
            .ok_or(crate::emitter_gap!())?;
        let (elem, shape) = self.ctx.tensors.get(&gid).ok_or(crate::emitter_gap!())?;
        let memty = tensor_memref_ty(elem, shape).ok_or(crate::emitter_gap!())?;
        // This is the buffer the function returns, and the caller already allocated it: build in
        // place and allocate nothing. Named return value optimization, the same move C++ and Rust
        // make. `op_ret` then returns without copying, because the value already *is* the slot.
        if let Some(slot) = self.nrvo_slot(idx, &memty) {
            self.names[idx] = slot;
            self.mem_of[idx] = Some(memty);
            return Ok(());
        }
        let n = format!("%v{idx}");
        // A `?` dimension's extent is the `Arg` before this instruction, one per `?` in order:
        // `memref.alloc(%d0, %d1)` takes them as indices.
        let dyn_count = shape.iter().filter(|d| *d == DYN_DIM).count();
        if self.pending_args.len() < dyn_count {
            return Err(crate::emitter_gap!());
        }
        let extents = self
            .pending_args
            .split_off(self.pending_args.len() - dyn_count);
        let mut sizes = Vec::with_capacity(dyn_count);
        for (k, r) in extents.iter().enumerate() {
            let v = self
                .names
                .get(*r as usize)
                .ok_or(crate::emitter_gap!())?
                .clone();
            let vt = mlir_scalar(&self.elem_at(*r).ok_or(crate::emitter_gap!())?)
                .ok_or(crate::emitter_gap!())?;
            let ix = format!("%tai{idx}_{k}");
            self.body += &format!("  {ix} = arith.index_cast {v} : {vt} to index\n");
            sizes.push(ix);
        }
        let sizes = sizes.join(", ");
        // `operand2` may carry a memory-space dispatch id from the placement in the type (Vx#379
        // stage B). A `scope: sm` space becomes a space-3 ALLOCA: on the host that is a
        // stack slot like any other, and in the device clone materializeGpuKernels turns
        // a static space-3 alloca into `.shared` storage -- the machinery #352 built.
        // Any other space stays a plain allocation; the annotation was advisory.
        let sm = ins.operand2.0 != 0
            && self
                .ctx
                .subspaces
                .get(&(ins.operand2.0 as u64))
                .and_then(|s| s.scope.as_deref())
                == Some("sm");
        if sm {
            // Shared storage is packed by the driver from static sizes; a run-time one has none.
            if dyn_count > 0 {
                return Err(Decline::TypeNotModelled {
                    what: "a tile of run-time size in shared memory",
                });
            }
            let smty = format!(
                "{}, 3>",
                memty.strip_suffix('>').ok_or(crate::emitter_gap!())?
            );
            // alignment 16, explicitly. The alloca becomes a `.shared` global under
            // convert-gpu-to-nvvm, and the DRIVER packs those globals by their declared
            // alignment -- an unannotated 4-byte-aligned f32 array landed at offset
            // 0x204, and LLVM's loop vectorizer (which assumes natural vector alignment
            // when it widens a serial walk over the tile) issued a 16-byte
            // `ld.shared.v4` into it: "misaligned address", device-fatal, found by
            // compute-sanitizer on the block-per-row softmax (Vx#379 R3).
            // Entry block: a tensor declared inside a loop would otherwise take a
            // fresh stack slot per iteration and never give one back.
            self.emit_slot(&format!(
                "  {n} = memref.alloca() {{alignment = 16 : i64}} : {smty}\n"
            ));
            self.names[idx] = n;
            self.mem_of[idx] = Some(smty);
        } else {
            self.body += &format!("  {n} = memref.alloc({sizes}) : {memty}\n");
            self.names[idx] = n;
            self.mem_of[idx] = Some(memty);
        }
        Ok(())
    }

    /// A rank-2 view over memory the caller owns (`tensor_view_2d(ptr, rows, cols)`): the memref
    /// descriptor built by hand over `operand1`'s pointer -- both pointers at the storage, a zero
    /// offset, the extents as sizes, row-major strides -- and cast into memref-typed IR the way
    /// the memref-to-LLVM conversion materializes one. A `?` extent is the `Arg` before this
    /// instruction; a static one is the type's. Nothing here checks the extents against the
    /// memory: that is the caller's claim, which is why the surface form needs `unsafe`.
    // The result tensor of a view-producing op, and the source's name, memref type and static
    // sizes; a source with a run-time extent or a strided layout is not reinterpreted here.
    fn view_operands(&self, ins: &HirInstruction) -> Lowered<ViewSource> {
        let gid = *self
            .types
            .get(ins.type_idx.0 as usize)
            .ok_or(crate::emitter_gap!())?;
        let (elem, shape) = self
            .ctx
            .tensors
            .get(&gid)
            .ok_or(crate::emitter_gap!())?
            .clone();
        let s = ins.operand1.0 as usize;
        let src = self.names.get(s).ok_or(crate::emitter_gap!())?.clone();
        let src_mem = self
            .mem_of
            .get(s)
            .cloned()
            .flatten()
            .ok_or(crate::emitter_gap!())?;
        if src_mem.contains("strided") {
            return Err(crate::emitter_gap!());
        }
        let sizes: Vec<i64> = src_mem
            .trim_start_matches("memref<")
            .split('x')
            .map_while(|t| t.parse().ok())
            .collect();
        Ok(ViewSource {
            elem,
            shape,
            src,
            src_mem,
            src_sizes: sizes,
        })
    }

    // `TensorReshape`: the source buffer reinterpreted with the result's sizes and contiguous
    // strides, the way the oracle lowers `reshape`.
    pub(crate) fn op_tensor_reshape(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let ViewSource {
            elem,
            shape,
            src,
            src_mem,
            ..
        } = self.view_operands(ins)?;
        let sizes: Vec<i64> = shape
            .iter()
            .map(|d| d.parse().map_err(|_| crate::emitter_gap!()))
            .collect::<Lowered<_>>()?;
        let memty = tensor_memref_ty(&elem, &shape).ok_or(crate::emitter_gap!())?;
        let n = format!("%v{idx}");
        self.body += &format!(
            "  {n} = memref.reinterpret_cast {src} to offset: [0], sizes: [{}], strides: [{}] : {src_mem} to {memty}\n",
            i64_list(&sizes),
            i64_list(&contiguous_strides(&sizes))
        );
        self.names[idx] = n;
        self.mem_of[idx] = Some(memty);
        Ok(())
    }

    // `TensorTranspose`: a strided view with the axes permuted, copied into a fresh contiguous
    // buffer of the result shape, the way the oracle lowers `transpose`.
    pub(crate) fn op_tensor_transpose(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let ViewSource {
            elem,
            shape,
            src,
            src_mem,
            src_sizes,
        } = self.view_operands(ins)?;
        let rank = shape.len();
        if src_sizes.len() != rank {
            return Err(crate::emitter_gap!());
        }
        let src_strides = contiguous_strides(&src_sizes);
        let perm: Vec<usize> = (0..rank)
            .map(|i| ((ins.imm >> (4 * i)) & 0xF) as usize)
            .collect();
        if perm.iter().any(|&p| p >= rank) {
            return Err(crate::emitter_gap!());
        }
        let view_sizes: Vec<i64> = perm.iter().map(|&p| src_sizes[p]).collect();
        let view_strides: Vec<i64> = perm.iter().map(|&p| src_strides[p]).collect();
        let el = mlir_scalar(&elem).ok_or(crate::emitter_gap!())?;
        let dims: Vec<String> = view_sizes.iter().map(|d| d.to_string()).collect();
        let view_ty = format!(
            "memref<{}x{el}, strided<[{}], offset: 0>>",
            dims.join("x"),
            i64_list(&view_strides)
        );
        let memty = tensor_memref_ty(&elem, &shape).ok_or(crate::emitter_gap!())?;
        let v = format!("%tvw{idx}");
        let n = format!("%v{idx}");
        self.body += &format!(
            "  {v} = memref.reinterpret_cast {src} to offset: [0], sizes: [{}], strides: [{}] : {src_mem} to {view_ty}\n",
            i64_list(&view_sizes),
            i64_list(&view_strides)
        );
        // A returned transpose copies out of the permuted view into the caller's buffer, which is
        // the one copy this op always needed -- it just no longer needs a buffer to make it into.
        let n = match self.nrvo_slot(idx, &memty) {
            Some(slot) => slot,
            None => {
                self.body += &format!("  {n} = memref.alloc() : {memty}\n");
                n
            }
        };
        self.body += &format!("  memref.copy {v}, {n} : {view_ty} to {memty}\n");
        self.names[idx] = n;
        self.mem_of[idx] = Some(memty);
        Ok(())
    }

    // `TensorMap`: a fresh buffer of the source's shape, each element the closure adapter
    // applied to the source's, as a `linalg.generic` over the two, the way the oracle lowers
    // `map`. A `?` extent is read off the source.
    pub(crate) fn op_tensor_map(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let gid = *self
            .types
            .get(ins.type_idx.0 as usize)
            .ok_or(crate::emitter_gap!())?;
        let (elem, shape) = self
            .ctx
            .tensors
            .get(&gid)
            .ok_or(crate::emitter_gap!())?
            .clone();
        let memty = tensor_memref_ty(&elem, &shape).ok_or(crate::emitter_gap!())?;
        let el = mlir_scalar(&elem).ok_or(crate::emitter_gap!())?;
        let s = ins.operand1.0 as usize;
        let src = self.names.get(s).ok_or(crate::emitter_gap!())?.clone();
        let src_mem = self
            .mem_of
            .get(s)
            .cloned()
            .flatten()
            .ok_or(crate::emitter_gap!())?;
        let env = self
            .names
            .get(ins.operand2.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        let callee_gid = *self
            .types
            .get(ins.imm as usize)
            .ok_or(crate::emitter_gap!())?;
        let callee = self
            .ctx
            .callees
            .get(&callee_gid)
            .ok_or(crate::emitter_gap!())?;
        let (params, ret) = self
            .ctx
            .func_sigs
            .get(&callee_gid)
            .ok_or(crate::emitter_gap!())?;
        let (name, params, ret) = (callee.name.clone(), params.join(", "), ret.clone());
        let mut sizes: Vec<String> = Vec::new();
        for (k, d) in shape.iter().enumerate() {
            if d == DYN_DIM {
                let c = format!("%tmc{idx}_{k}");
                let sz = format!("%tms{idx}_{k}");
                self.body += &format!("  {c} = arith.constant {k} : index\n");
                self.body += &format!("  {sz} = memref.dim {src}, {c} : {src_mem}\n");
                sizes.push(sz);
            }
        }
        // A returned map writes its elements straight into the caller's buffer.
        let n = match self.nrvo_slot(idx, &memty) {
            Some(slot) => slot,
            None => {
                let n = format!("%v{idx}");
                self.body += &format!("  {n} = memref.alloc({}) : {memty}\n", sizes.join(", "));
                n
            }
        };
        let dims: Vec<String> = (0..shape.len()).map(|i| format!("d{i}")).collect();
        let map = format!("affine_map<({0}) -> ({0})>", dims.join(", "));
        let iters: Vec<&str> = shape.iter().map(|_| "\"parallel\"").collect();
        self.body += &format!(
            "  linalg.generic {{indexing_maps = [{map}, {map}], iterator_types = [{}]}} ins({src} : {src_mem}) outs({n} : {memty}) {{\n",
            iters.join(", ")
        );
        self.body += &format!("  ^bb0(%tmi{idx}: {el}, %tmo{idx}: {el}):\n");
        self.body += &format!(
            "    %tmr{idx} = func.call {}({env}, %tmi{idx}) : ({params}) -> {ret}\n",
            sym_ref(&name)
        );
        self.body += &format!("    linalg.yield %tmr{idx} : {el}\n  }}\n");
        self.names[idx] = n;
        self.mem_of[idx] = Some(memty);
        Ok(())
    }

    // `TensorReduce`: every element combined with the closure adapter into one scalar, starting
    // from the preceding `Arg`. The accumulator is a rank-0 buffer the `linalg.generic` reduces
    // into. `vx.reassoc` marks it for `vx-reorderable-reductions`, which lets the closure's
    // arithmetic happen in any order, as `reduce` allows.
    pub(crate) fn op_tensor_reduce(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let result_gid = *self
            .types
            .get(ins.type_idx.0 as usize)
            .ok_or(crate::emitter_gap!())?;
        let elem = elem_of_gid(result_gid).ok_or(crate::emitter_gap!())?;
        let el = mlir_scalar(&elem).ok_or(crate::emitter_gap!())?;
        let init_reg = self.pending_args.pop().ok_or(crate::emitter_gap!())?;
        let init = self
            .names
            .get(init_reg as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        let s = ins.operand1.0 as usize;
        let src = self.names.get(s).ok_or(crate::emitter_gap!())?.clone();
        let src_mem = self
            .mem_of
            .get(s)
            .cloned()
            .flatten()
            .ok_or(crate::emitter_gap!())?;
        let rank = super::aggregate::memref_rank(&src_mem).ok_or(crate::emitter_gap!())?;
        let env = self
            .names
            .get(ins.operand2.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        let callee_gid = *self
            .types
            .get(ins.imm as usize)
            .ok_or(crate::emitter_gap!())?;
        let callee = self
            .ctx
            .callees
            .get(&callee_gid)
            .ok_or(crate::emitter_gap!())?;
        let (params, ret) = self
            .ctx
            .func_sigs
            .get(&callee_gid)
            .ok_or(crate::emitter_gap!())?;
        let (name, params, ret) = (callee.name.clone(), params.join(", "), ret.clone());
        let acc = format!("%tra{idx}");
        self.body += &format!("  {acc} = memref.alloca() : memref<{el}>\n");
        self.body += &format!("  memref.store {init}, {acc}[] : memref<{el}>\n");
        let dims: Vec<String> = (0..rank).map(|i| format!("d{i}")).collect();
        let in_map = format!("affine_map<({0}) -> ({0})>", dims.join(", "));
        let out_map = format!("affine_map<({}) -> ()>", dims.join(", "));
        let iters: Vec<&str> = (0..rank).map(|_| "\"reduction\"").collect();
        self.body += &format!(
            "  linalg.generic {{indexing_maps = [{in_map}, {out_map}], iterator_types = [{}]}} ins({src} : {src_mem}) outs({acc} : memref<{el}>) attrs = {{vx.reassoc}} {{\n",
            iters.join(", ")
        );
        self.body += &format!("  ^bb0(%tri{idx}: {el}, %tro{idx}: {el}):\n");
        self.body += &format!(
            "    %trr{idx} = func.call {}({env}, %tro{idx}, %tri{idx}) : ({params}) -> {ret}\n",
            sym_ref(&name)
        );
        self.body += &format!("    linalg.yield %trr{idx} : {el}\n  }}\n");
        let n = format!("%v{idx}");
        self.body += &format!("  {n} = memref.load {acc}[] : memref<{el}>\n");
        self.names[idx] = n;
        self.etypes[idx] = Some(elem);
        Ok(())
    }

    pub(crate) fn op_tensor_view(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let gid = *self
            .types
            .get(ins.type_idx.0 as usize)
            .ok_or(crate::emitter_gap!())?;
        let (elem, shape) = self
            .ctx
            .tensors
            .get(&gid)
            .ok_or(crate::emitter_gap!())?
            .clone();
        if shape.len() != 2 {
            return Err(crate::emitter_gap!());
        }
        let memty = tensor_memref_ty(&elem, &shape).ok_or(crate::emitter_gap!())?;
        let p = ins.operand1.0 as usize;
        if !self.ptr_of.get(p).copied().unwrap_or(false) {
            return Err(crate::emitter_gap!());
        }
        let ptr = self.names.get(p).ok_or(crate::emitter_gap!())?.clone();
        let dyn_count = shape.iter().filter(|d| *d == DYN_DIM).count();
        if self.pending_args.len() < dyn_count {
            return Err(crate::emitter_gap!());
        }
        let extents = self
            .pending_args
            .split_off(self.pending_args.len() - dyn_count);
        let mut ext = extents.into_iter();
        let mut dims_i64: Vec<String> = Vec::with_capacity(2);
        for (k, d) in shape.iter().enumerate() {
            let v = format!("%tv{idx}_{k}");
            if d == DYN_DIM {
                let r = ext.next().ok_or(crate::emitter_gap!())?;
                let a = self
                    .names
                    .get(r as usize)
                    .ok_or(crate::emitter_gap!())?
                    .clone();
                let at = mlir_scalar(&self.elem_at(r).ok_or(crate::emitter_gap!())?)
                    .ok_or(crate::emitter_gap!())?;
                if at == "i64" {
                    dims_i64.push(a);
                    continue;
                }
                self.body += &format!("  {v} = arith.extsi {a} : {at} to i64\n");
            } else {
                self.body += &format!("  {v} = arith.constant {d} : i64\n");
            }
            dims_i64.push(v);
        }
        let zero = format!("%tvz{idx}");
        let one = format!("%tvo{idx}");
        self.body += &format!("  {zero} = arith.constant 0 : i64\n");
        self.body += &format!("  {one} = arith.constant 1 : i64\n");
        let dty = "!llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>";
        let mut d = format!("%tvd{idx}_0");
        self.body += &format!("  {d} = llvm.mlir.undef : {dty}\n");
        // Fields in order: allocated, aligned, offset, sizes[0..1], strides[0..1]. Row-major, so
        // the row stride is the column count and the column stride 1.
        let fields: [(&str, &str); 7] = [
            (&ptr, "0"),
            (&ptr, "1"),
            (&zero, "2"),
            (&dims_i64[0], "3, 0"),
            (&dims_i64[1], "3, 1"),
            (&dims_i64[1], "4, 0"),
            (&one, "4, 1"),
        ];
        for (k, (val, pos)) in fields.iter().enumerate() {
            let nd = format!("%tvd{idx}_{}", k + 1);
            self.body += &format!("  {nd} = llvm.insertvalue {val}, {d}[{pos}] : {dty}\n");
            d = nd;
        }
        let n = format!("%v{idx}");
        self.body +=
            &format!("  {n} = builtin.unrealized_conversion_cast {d} : {dty} to {memty}\n");
        self.names[idx] = n;
        self.mem_of[idx] = Some(memty);
        Ok(())
    }

    /// Zero a freshly allocated tensor (`Tensor<T, [..]>::new()`): `linalg.fill` with a zero of
    /// the element type, which is the fill `MatmulInto` already emits before it accumulates.
    ///
    /// Integer elements are spelled with an integer zero rather than declined -- `linalg.fill`
    /// writes the value it is given, so there is no integer semantics to improvise here, unlike
    /// the multiply-accumulate `MatmulInto` guards against.
    pub(crate) fn op_tensor_zero(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let t = self
            .names
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        let memty = self
            .mem_of
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone()
            .ok_or(crate::emitter_gap!())?;
        let et = memref_element(&memty).ok_or(crate::emitter_gap!())?;
        let zero = if et.starts_with('f') || et.starts_with("bf") {
            "0.0"
        } else {
            "0"
        };
        self.body += &format!("  %z{idx} = arith.constant {zero} : {et}\n");
        self.body += &format!("  linalg.fill ins(%z{idx} : {et}) outs({t} : {memty})\n");
        Ok(())
    }

    /// Fill a freshly allocated tensor with a value (`Tensor<T, [..]>::fill(v)`): `linalg.fill`
    /// with the value the source wrote, where `TensorZero` supplies a zero of its own.
    pub(crate) fn op_tensor_fill(&mut self, _idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let t = self
            .names
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        let memty = self
            .mem_of
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone()
            .ok_or(crate::emitter_gap!())?;
        let et = memref_element(&memty).ok_or(crate::emitter_gap!())?;
        let v = self
            .names
            .get(ins.operand2.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        self.body += &format!("  linalg.fill ins({v} : {et}) outs({t} : {memty})\n");
        Ok(())
    }

    // The runtime extent of one dimension: `memref.dim %t, %k` (`t.shape[k]`). The index
    // operand arrives as a scalar and is cast to `index`; the result is cast back to `i32`,
    // the type the checker gives the expression.
    pub(crate) fn op_tensor_dim(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let t = self
            .names
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        let memty = self
            .mem_of
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone()
            .ok_or(crate::emitter_gap!())?;
        let k = self
            .names
            .get(ins.operand2.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        let kt = mlir_scalar(&self.elem_at(ins.operand2.0).ok_or(crate::emitter_gap!())?)
            .ok_or(crate::emitter_gap!())?;
        let ki = format!("%tdi{idx}");
        let d = format!("%tdd{idx}");
        let n = format!("%v{idx}");
        self.body += &format!("  {ki} = arith.index_cast {k} : {kt} to index\n");
        self.body += &format!("  {d} = memref.dim {t}, {ki} : {memty}\n");
        self.body += &format!("  {n} = arith.index_cast {d} : index to i32\n");
        self.names[idx] = n;
        self.etypes[idx] = Some(ElementType::I32);
        Ok(())
    }

    // Read a rank-0 tensor's element: `memref.load %t[]`. Only rank-0 bases emit this
    // (flatten's `read_rank0`); anything shaped is a wrong stream and declines.
    pub(crate) fn op_tensor_load(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let src = self
            .names
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        let memty = self
            .mem_of
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone()
            .ok_or(crate::emitter_gap!())?;
        let rank0 = memref_lead_dims_and_elem(&memty).is_some_and(|(d, _)| d.is_empty());
        if !rank0 {
            return Err(crate::emitter_gap!());
        }
        let e = elem_of_gid(
            *self
                .types
                .get(ins.type_idx.0 as usize)
                .ok_or(crate::emitter_gap!())?,
        )
        .ok_or(crate::emitter_gap!())?;
        let n = format!("%v{idx}");
        self.body += &format!(
            "  {n} = memref.load {src}[] : {memty}
"
        );
        self.names[idx] = n;
        self.etypes[idx] = Some(e);
        Ok(())
    }

    // Index a tensor along its outermost dimension. `operand1` is the base tensor (memref),
    // `operand2` the index (`arith.index_cast` to `index`). A scalar-element result
    // (`type_idx` is a scalar GID) is a value read (`imm = 0` → `memref.load`) or an element
    // *place* (`imm = 1` → recorded for the following `TensorStore`). A sub-view result (a
    // tensor GID) rank-reduces the base to a row via `memref.reinterpret_cast` (contiguous
    // base only; a further sub-view of a strided row is deferred).
    pub(crate) fn op_tensor_index(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let result_gid = *self
            .types
            .get(ins.type_idx.0 as usize)
            .ok_or(crate::emitter_gap!())?;
        let base = self
            .names
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        let base_memty = self
            .mem_of
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone()
            .ok_or(crate::emitter_gap!())?;
        let imt = mlir_scalar(&self.elem_at(ins.operand2.0).ok_or(crate::emitter_gap!())?)
            .ok_or(crate::emitter_gap!())?;
        let iname = self
            .names
            .get(ins.operand2.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        let ic = format!("%ic{idx}");
        self.body += &format!("  {ic} = arith.index_cast {iname} : {imt} to index\n");

        if let Some(e) = elem_of_gid(result_gid) {
            // Scalar element: a value read or a store place (works on a contiguous or a
            // strided-row base — `memref.load`/`store` handle both).
            if ins.imm == 1 {
                self.place_of[idx] = Some((base, ic, base_memty));
            } else {
                let n = format!("%v{idx}");
                self.body += &format!("  {n} = memref.load {base}[{ic}] : {base_memty}\n");
                self.names[idx] = n;
                self.etypes[idx] = Some(e);
            }
        } else {
            // Row sub-view: reinterpret the contiguous base as the row at flat offset
            // `index * product(row dims)`, with row-major strides over the remaining dims.
            // A row of a row: the base is already a strided view, so its own offset, sizes and
            // strides come back through `memref.extract_strided_metadata`, and the next row is
            // that offset plus the index times the leading stride.
            if base_memty.contains("strided") {
                return self.strided_row(idx, &base, &base_memty, &ic);
            }
            let (elem, shape) = self
                .ctx
                .tensors
                .get(&result_gid)
                .ok_or(crate::emitter_gap!())?;
            let et = mlir_scalar(elem).ok_or(crate::emitter_gap!())?;
            // A row with a run-time extent anywhere: each `?` size is read off the base with
            // `memref.dim`, the strides are the row-major products, and the offset is the index
            // times the row's element count. A static position stays a literal in the type.
            if shape.iter().any(|d| d == DYN_DIM) {
                let et = et.to_string();
                let shape = shape.clone();
                return self.dynamic_row(idx, &base, &base_memty, &ic, &et, &shape);
            }
            let dims: Vec<i64> = shape
                .iter()
                .map(|d| d.parse::<i64>().ok())
                .collect::<Option<_>>()
                .ok_or(Decline::TypeNotModelled {
                    what: "a row whose extents are not all literals",
                })?;
            let stride0: i64 = dims.iter().product();
            let mut strides = vec![1i64; dims.len()];
            for i in (0..dims.len().saturating_sub(1)).rev() {
                strides[i] = strides[i + 1] * dims[i + 1];
            }
            let off = if stride0 == 1 {
                ic.clone()
            } else {
                let cst = format!("%cs{idx}");
                let o = format!("%off{idx}");
                self.body += &format!("  {cst} = arith.constant {stride0} : index\n");
                self.body += &format!("  {o} = arith.muli {ic}, {cst} : index\n");
                o
            };
            let sizes_s = join_i64(&dims);
            let strides_s = join_i64(&strides);
            let dimx: String = dims.iter().map(|d| format!("{d}x")).collect();
            // A sub-view of shared storage stays in its space: dropping the `, 3` here
            // would make the row a generic pointer and the PTX would address `.shared`
            // data with global loads.
            let space_sfx = if base_memty.ends_with(", 3>") {
                ", 3"
            } else {
                ""
            };
            let result_ty =
                format!("memref<{dimx}{et}, strided<[{strides_s}], offset: ?>{space_sfx}>");
            let n = format!("%v{idx}");
            self.body += &format!(
                "  {n} = memref.reinterpret_cast {base} to offset: [{off}], sizes: [{sizes_s}], strides: [{strides_s}] : {base_memty} to {result_ty}\n"
            );
            self.names[idx] = n;
            self.mem_of[idx] = Some(result_ty);
        }
        Ok(())
    }

    /// The rank-1 row of a rank-2 dynamically shaped tensor. Its extent is not a number to
    /// compute with, so it is read off the base with `memref.dim` and the flat offset is
    /// computed against that value. The stride is 1: a row of a row-major rank-2 is contiguous
    /// whatever its length.
    fn dynamic_row(
        &mut self,
        idx: usize,
        base: &str,
        base_memty: &str,
        ic: &str,
        et: &str,
        shape: &[String],
    ) -> Lowered<()> {
        // Sizes: a literal stays one; a `?` is `memref.dim` of the base at that position, which
        // is one deeper than in the row (the outermost dimension is the one being indexed).
        let mut sizes: Vec<String> = Vec::with_capacity(shape.len());
        for (k, d) in shape.iter().enumerate() {
            if d == DYN_DIM {
                let c = format!("%rdc{idx}_{k}");
                let s = format!("%rds{idx}_{k}");
                self.body += &format!("  {c} = arith.constant {} : index\n", k + 1);
                self.body += &format!("  {s} = memref.dim {base}, {c} : {base_memty}\n");
                sizes.push(s);
            } else {
                sizes.push(d.clone());
            }
        }
        // Strides, row-major from the back: the last is 1, each other is the next stride
        // times the next size. A product of literals stays a literal.
        let r = shape.len();
        let mut strides: Vec<String> = vec!["1".to_string(); r];
        for k in (0..r.saturating_sub(1)).rev() {
            let (a, b) = (strides[k + 1].clone(), sizes[k + 1].clone());
            strides[k] = self.mul_extent(&a, &b, &format!("%rdt{idx}_{k}"));
        }
        // Offset: the index times the row's element count.
        let count = self.mul_extent(&strides[0], &sizes[0], &format!("%rdn{idx}"));
        let off = self.mul_extent(ic, &count, &format!("%rdo{idx}"));
        let as_ty = |v: &String| {
            if v.starts_with('%') {
                "?".to_string()
            } else {
                v.clone()
            }
        };
        let dims: String = sizes.iter().map(|s| format!("{}x", as_ty(s))).collect();
        let stride_tys: Vec<String> = strides.iter().map(as_ty).collect();
        let result_ty = format!(
            "memref<{dims}{et}, strided<[{}], offset: ?>>",
            stride_tys.join(", ")
        );
        let n = format!("%v{idx}");
        self.body += &format!(
            "  {n} = memref.reinterpret_cast {base} to offset: [{off}], sizes: [{}], strides: [{}] : {base_memty} to {result_ty}\n",
            sizes.join(", "),
            strides.join(", ")
        );
        self.names[idx] = n;
        self.mem_of[idx] = Some(result_ty);
        Ok(())
    }

    /// Index a strided row along its outermost dimension. The base type spells its sizes and
    /// strides (`memref<AxBx..xet, strided<[s0, s1, ..], offset: ?>>`), so the result's are
    /// those with the first dropped, a literal staying a literal and a `?` taking the value
    /// `extract_strided_metadata` hands back; the offset is the base's plus `index * s0`.
    fn strided_row(&mut self, idx: usize, base: &str, base_memty: &str, ic: &str) -> Lowered<()> {
        let (dims_x, et) = memref_lead_dims_and_elem(base_memty).ok_or(crate::emitter_gap!())?;
        let dims: Vec<&str> = dims_x.split('x').filter(|d| !d.is_empty()).collect();
        let strides_txt = base_memty
            .split("strided<[")
            .nth(1)
            .and_then(|s| s.split(']').next())
            .ok_or(crate::emitter_gap!())?;
        let strides: Vec<&str> = strides_txt.split(',').map(|s| s.trim()).collect();
        let rank = dims.len();
        if rank < 2 || strides.len() != rank {
            return Err(crate::emitter_gap!());
        }
        let space_sfx = if base_memty.ends_with(", 3>") {
            ", 3"
        } else {
            ""
        };
        let buf = format!("%srb{idx}");
        let off = format!("%sro{idx}");
        let sizes: Vec<String> = (0..rank).map(|k| format!("%srs{idx}_{k}")).collect();
        let strs: Vec<String> = (0..rank).map(|k| format!("%srt{idx}_{k}")).collect();
        let results = [vec![buf.clone(), off.clone()], sizes.clone(), strs.clone()]
            .concat()
            .join(", ");
        let index_tys = vec!["index"; 1 + 2 * rank].join(", ");
        self.body += &format!(
            "  {results} = memref.extract_strided_metadata {base} : {base_memty} -> memref<{et}{space_sfx}>, {index_tys}\n"
        );
        let step = format!("%srm{idx}");
        let new_off = format!("%srn{idx}");
        self.body += &format!("  {step} = arith.muli {ic}, {} : index\n", strs[0]);
        self.body += &format!("  {new_off} = arith.addi {off}, {step} : index\n");
        let pick = |txt: &str, ssa: &String| {
            if txt == "?" {
                ssa.clone()
            } else {
                txt.to_string()
            }
        };
        let size_ops: Vec<String> = (1..rank).map(|k| pick(dims[k], &sizes[k])).collect();
        let stride_ops: Vec<String> = (1..rank).map(|k| pick(strides[k], &strs[k])).collect();
        let result_ty = format!(
            "memref<{}{et}, strided<[{}], offset: ?>{space_sfx}>",
            dims[1..]
                .iter()
                .map(|d| format!("{d}x"))
                .collect::<String>(),
            strides[1..].join(", ")
        );
        let n = format!("%v{idx}");
        self.body += &format!(
            "  {n} = memref.reinterpret_cast {buf} to offset: [{new_off}], sizes: [{}], strides: [{}] : memref<{et}{space_sfx}> to {result_ty}\n",
            size_ops.join(", "),
            stride_ops.join(", ")
        );
        self.names[idx] = n;
        self.mem_of[idx] = Some(result_ty);
        Ok(())
    }

    /// The product of two extents, each a literal or an `index` SSA name: a literal when both
    /// are, otherwise an `arith.muli` with any literal operand materialized first.
    fn mul_extent(&mut self, a: &str, b: &str, name: &str) -> String {
        if let (Ok(x), Ok(y)) = (a.parse::<i64>(), b.parse::<i64>()) {
            return (x * y).to_string();
        }
        // A unit factor contributes nothing; a rank-1 row's element count is its width.
        if a == "1" {
            return b.to_string();
        }
        if b == "1" {
            return a.to_string();
        }
        let operand = |v: &str, tag: &str, body: &mut String| -> String {
            if v.starts_with('%') {
                v.to_string()
            } else {
                let c = format!("{name}{tag}");
                *body += &format!("  {c} = arith.constant {v} : index\n");
                c
            }
        };
        let a = operand(a, "a", &mut self.body);
        let b = operand(b, "b", &mut self.body);
        self.body += &format!("  {name} = arith.muli {a}, {b} : index\n");
        name.to_string()
    }

    // `matmul_into(&mut dst, &a, &b)` (no result): `linalg.fill` + `linalg.matmul` on the
    // three whole-tensor memrefs, the same pair the AST path builds -- and the exact shape
    // `kernelKindOf` classifies, so a spawn whose whole job is this op still routes to
    // cuBLAS. The destination register rides the imm (see `Opcode::MatmulInto`).
    // `a @ b`: a matmul producing a fresh tensor, as against `MatmulInto`'s write into a buffer
    // the caller owns. `type_idx` is the result's own GID -- the flattener sizes it `[m, n]` from
    // the two operands -- so the destination is allocated here and then filled by the same pair of
    // ops `MatmulInto` emits.
    pub(crate) fn op_matmul(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let a = self
            .names
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        let ma = self
            .mem_of
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone()
            .ok_or(crate::emitter_gap!())?;
        let b = self
            .names
            .get(ins.operand2.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        let mb = self
            .mem_of
            .get(ins.operand2.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone()
            .ok_or(crate::emitter_gap!())?;
        let gid = *self
            .types
            .get(ins.type_idx.0 as usize)
            .ok_or(crate::emitter_gap!())?;
        let (elem, shape) = self.ctx.tensors.get(&gid).ok_or(crate::emitter_gap!())?;
        let (elem, shape) = (elem.clone(), shape.clone());
        let md = tensor_memref_ty(&elem, &shape).ok_or(crate::emitter_gap!())?;
        // Floats only, on the same terms as `MatmulInto`: linalg's integer semantics are not
        // improvised here.
        let et = mlir_scalar(&elem).ok_or(crate::emitter_gap!())?;
        if !matches!(et, "f32" | "f64" | "f16" | "bf16") {
            return Err(crate::emitter_gap!());
        }
        // A `?` extent of the result is read off the operand it comes from: the rows of the
        // left operand, the columns of the right.
        let mut sizes: Vec<String> = Vec::new();
        for (k, d) in shape.iter().enumerate() {
            if d == DYN_DIM {
                let (src, sm) = if k == 0 { (&a, &ma) } else { (&b, &mb) };
                let c = format!("%mmc{idx}_{k}");
                let sz = format!("%mms{idx}_{k}");
                self.body += &format!("  {c} = arith.constant {k} : index\n");
                self.body += &format!("  {sz} = memref.dim {src}, {c} : {sm}\n");
                sizes.push(sz);
            }
        }
        // A returned product accumulates straight into the caller's buffer -- the biggest of the
        // buffers this saves, since a matmul result is the whole output matrix.
        let n = match self.nrvo_slot(idx, &md) {
            Some(slot) => slot,
            None => {
                let n = format!("%v{idx}");
                self.body += &format!("  {n} = memref.alloc({}) : {md}\n", sizes.join(", "));
                n
            }
        };
        self.body += &format!("  %mz{idx} = arith.constant 0.0 : {et}\n");
        self.body += &format!("  linalg.fill ins(%mz{idx} : {et}) outs({n} : {md})\n");
        self.body += &format!("  linalg.matmul ins({a}, {b} : {ma}, {mb}) outs({n} : {md})\n");
        self.names[idx] = n;
        self.mem_of[idx] = Some(md);
        Ok(())
    }

    pub(crate) fn op_matmul_into(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let a = self
            .names
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        let ma = self
            .mem_of
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone()
            .ok_or(crate::emitter_gap!())?;
        let b = self
            .names
            .get(ins.operand2.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        let mb = self
            .mem_of
            .get(ins.operand2.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone()
            .ok_or(crate::emitter_gap!())?;
        let dst = self
            .names
            .get(ins.imm as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        let md = self
            .mem_of
            .get(ins.imm as usize)
            .ok_or(crate::emitter_gap!())?
            .clone()
            .ok_or(crate::emitter_gap!())?;
        // Floats only -- an int matmul declines the program to the AST path rather than
        // improvising linalg's integer semantics here. The half-precision pair is in (Vx#320):
        // the routed backend runs them through cublasGemmEx with f32 accumulation, and the host
        // fallback's linalg lowers them like any float -- restricting to f32 here silently
        // evicted every f16 program from the flat path, prover and all.
        let et = memref_element(&md).ok_or(crate::emitter_gap!())?;
        if et != "f32" && et != "f64" && et != "f16" && et != "bf16" {
            return Err(crate::emitter_gap!());
        }
        self.body += &format!("  %mz{idx} = arith.constant 0.0 : {et}\n");
        self.body += &format!("  linalg.fill ins(%mz{idx} : {et}) outs({dst} : {md})\n");
        self.body += &format!("  linalg.matmul ins({a}, {b} : {ma}, {mb}) outs({dst} : {md})\n");
        Ok(())
    }

    // Store into a tensor place (no result). A scalar-element place (an `imm = 1`
    // `TensorIndex`) → `memref.store`; a row/sub-view place (an `imm = 0` `TensorIndex`, a row
    // memref in `mem_of`) takes an elementwise vector value → `vector.store`.
    pub(crate) fn op_tensor_store(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        if let Some((base, ic, memty)) = self
            .place_of
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone()
        {
            let vreg = ins.operand2.0;
            let mut val = self
                .names
                .get(vreg as usize)
                .ok_or(crate::emitter_gap!())?
                .clone();
            // Coerce the stored scalar to the tensor's element type when they differ — a
            // default-`f32` float literal `1.0` stored into a `bf16` tensor becomes
            // `arith.truncf`, `f32 -> f64` becomes `arith.extf`, etc. The AST path does the
            // same via `coerce_type` before its `memref.store`; without it the store is
            // ill-typed (`f32` value into a `memref<..xbf16>`).
            if let (Some(src_e), Some(tgt_s)) = (self.elem_at(vreg), memref_elem(&memty)) {
                if let Some(tgt_e) = elem_from_mlir_scalar(tgt_s) {
                    match cast_op(&src_e, &tgt_e) {
                        Some("") => {} // same MLIR type: no conversion
                        Some(op) => {
                            let c = format!("%tsc{idx}");
                            self.body += &format!(
                                "  {c} = {op} {val} : {} to {tgt_s}\n",
                                mlir_scalar(&src_e).ok_or(crate::emitter_gap!())?
                            );
                            val = c;
                        }
                        None => return Err(crate::emitter_gap!()), // unmodelled conversion -> decline (AST oracle)
                    }
                }
            }
            self.body += &format!("  memref.store {val}, {base}[{ic}] : {memty}\n");
        } else if let Some(rowty) = self
            .mem_of
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone()
        {
            let dst = self
                .names
                .get(ins.operand1.0 as usize)
                .ok_or(crate::emitter_gap!())?
                .clone();
            // A rank-0 tensor (`memref<el>`, no dims): the stored value is the scalar itself,
            // written with an empty index list. Elements are identical by the checker's
            // scalar-into-tensor rule (Vx#396); anything else declines.
            if memref_lead_dims_and_elem(&rowty).is_some_and(|(d, _)| d.is_empty()) {
                let vreg = ins.operand2.0;
                let val = self
                    .names
                    .get(vreg as usize)
                    .ok_or(crate::emitter_gap!())?
                    .clone();
                let same_elem = match (self.elem_at(vreg), memref_elem(&rowty)) {
                    (Some(src_e), Some(tgt_s)) => mlir_scalar(&src_e) == Some(tgt_s),
                    _ => false,
                };
                if !same_elem {
                    return Err(crate::emitter_gap!());
                }
                self.body += &format!(
                    "  memref.store {val}, {dst}[] : {rowty}
"
                );
                return Ok(());
            }
            let vecname = self
                .names
                .get(ins.operand2.0 as usize)
                .ok_or(crate::emitter_gap!())?
                .clone();
            let vec_rec = self
                .vec_of
                .get(ins.operand2.0 as usize)
                .ok_or(crate::emitter_gap!())?
                .clone();
            let c0 = format!("%sc{idx}");
            if let Some(vecty) = vec_rec {
                let al = vector_align_attr(&vecty);
                self.body += &format!("  {c0} = arith.constant 0 : index\n");
                self.body +=
                    &format!("  vector.store {vecname}, {dst}[{c0}]{al} : {rowty}, {vecty}\n");
            } else if let Some(srcty) = self
                .mem_of
                .get(ins.operand2.0 as usize)
                .ok_or(crate::emitter_gap!())?
                .clone()
            {
                // The stored value is a rank-1 tensor, not a vector -- `m[0] = [10.0, ..]`, a row
                // written from an array literal's buffer. Read the whole source row as a vector
                // and store it; the checker already matched the two lengths.
                let d = memref_lead_dim(&srcty).ok_or(crate::emitter_gap!())?;
                let et = memref_elem(&srcty).ok_or(crate::emitter_gap!())?;
                let vecty = format!("vector<{d}x{et}>");
                let al = vector_align_attr(&vecty);
                let v = format!("%svl{idx}");
                self.body += &format!("  {c0} = arith.constant 0 : index\n");
                self.body +=
                    &format!("  {v} = vector.load {vecname}[{c0}]{al} : {srcty}, {vecty}\n");
                self.body += &format!("  vector.store {v}, {dst}[{c0}]{al} : {rowty}, {vecty}\n");
            } else {
                return Err(crate::emitter_gap!());
            }
        } else {
            return Err(crate::emitter_gap!());
        }
        Ok(())
    }
}

/// What a view-producing op reinterprets: the result tensor, and the source's name, memref type
/// and static sizes.
struct ViewSource {
    elem: ElementType,
    shape: Vec<String>,
    src: String,
    src_mem: String,
    src_sizes: Vec<i64>,
}

/// Row-major strides for static sizes: the last axis is unit.
fn contiguous_strides(sizes: &[i64]) -> Vec<i64> {
    let mut strides = vec![1i64; sizes.len()];
    for i in (0..sizes.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * sizes[i + 1];
    }
    strides
}

fn i64_list(v: &[i64]) -> String {
    v.iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}
