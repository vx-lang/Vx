//! Tensors: allocation, indexing, stores, and the reduction/matmul/attention kernels.
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
    // Allocate a tensor buffer (`Tensor<T>([..])`): a static `memref` of the shape recovered
    // from the side table by GID. Its register is tracked in `mem_of` for later index/store.
    pub(crate) fn op_tensor_alloc(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let gid = *self
            .types
            .get(ins.type_idx.0 as usize)
            .ok_or(crate::emitter_gap!())?;
        let (elem, shape) = self.ctx.tensors.get(&gid).ok_or(crate::emitter_gap!())?;
        let memty = tensor_memref_ty(elem, shape).ok_or(crate::emitter_gap!())?;
        let n = format!("%v{idx}");
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
            self.body += &format!("  {n} = memref.alloc() : {memty}\n");
            self.names[idx] = n;
            self.mem_of[idx] = Some(memty);
        }
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
            if base_memty.contains("strided") {
                return Err(crate::emitter_gap!()); // a sub-view of an already-strided row is deferred
            }
            let (elem, shape) = self
                .ctx
                .tensors
                .get(&result_gid)
                .ok_or(crate::emitter_gap!())?;
            let et = mlir_scalar(elem).ok_or(crate::emitter_gap!())?;
            // A row of a dynamically shaped tensor: its extent is not a number here, so it is
            // read off the base with `memref.dim` and the offset is computed against that.
            // Only the rank-1 row of a rank-2 base, which is the one shape a `[?, ?]` tensor has
            // (Vx#404); anything deeper still needs a stride computation there is nothing to
            // compute with.
            if shape.iter().any(|d| d == DYN_DIM) {
                let et = et.to_string();
                if shape.len() != 1
                    || !base_memty.starts_with("memref<?x?x")
                    || base_memty.ends_with(", 3>")
                {
                    return Err(Decline::TypeNotModelled {
                        what: "a row of a tensor whose extents are not known at compile time",
                    });
                }
                return self.dynamic_row(idx, &base, &base_memty, &ic, &et);
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
    ) -> Lowered<()> {
        let one = format!("%rdk{idx}");
        let width = format!("%rdw{idx}");
        let off = format!("%rdo{idx}");
        let n = format!("%v{idx}");
        self.body += &format!("  {one} = arith.constant 1 : index\n");
        self.body += &format!("  {width} = memref.dim {base}, {one} : {base_memty}\n");
        self.body += &format!("  {off} = arith.muli {ic}, {width} : index\n");
        let result_ty = format!("memref<?x{et}, strided<[1], offset: ?>>");
        self.body += &format!(
            "  {n} = memref.reinterpret_cast {base} to offset: [{off}], sizes: [{width}], strides: [1] : {base_memty} to {result_ty}\n"
        );
        self.names[idx] = n;
        self.mem_of[idx] = Some(result_ty);
        Ok(())
    }

    // Reduce a rank-1 float slice to a scalar. `operand1` (and `operand2` for `dot`) are the
    // slices; `imm` the kind (0 = dot, 1 = sum, 2 = max, 3 = min). Each slice is `vector.load`ed
    // to a `vector<Nxf32>`; `dot` fuses the two with `arith.mulf`; then `vector.reduction`.
    // Float only, matching the AST oracle (`vector<Nxf32>` → `f32`).
    pub(crate) fn op_reduce(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let e = self.ty_at(ins.type_idx.0).ok_or(crate::emitter_gap!())?;
        if !e.is_float() {
            return Err(crate::emitter_gap!()); // the AST lowers only f32 reductions
        }
        let et = mlir_scalar(&e).ok_or(crate::emitter_gap!())?;
        // One load per operand, widened when the ROW is half-precision (Vx#320):
        // f16/bf16 rows come up through `arith.extf` and the multiply, reduction and
        // result are f32 -- a reduction's precision is its accumulator's, exactly the
        // contract the checker types (`dot` over any float slices -> f32).
        let c0 = format!("%rc{idx}");
        self.body += &format!("  {c0} = arith.constant 0 : index\n");
        let names = &self.names;
        let mem_of = &self.mem_of;
        let load_wide = |reg: u32, tag: &str, body: &mut String| -> Option<String> {
            let s = names.get(reg as usize)?.clone();
            let m = mem_of.get(reg as usize)?.clone()?;
            let d = memref_lead_dim(&m)?;
            let row_elem = m.rsplit('x').next()?.trim_end_matches('>');
            let row_elem = row_elem.split(',').next()?.trim();
            let rvec = format!("vector<{d}x{row_elem}>");
            let al = vector_align_attr(&rvec);
            let v = format!("%{tag}{idx}");
            body.push_str(&format!(
                "  {v} = vector.load {s}[{c0}]{al} : {m}, {rvec}\n"
            ));
            if row_elem == et {
                return Some(v);
            }
            if row_elem != "f16" && row_elem != "bf16" {
                return None;
            }
            let w = format!("%{tag}w{idx}");
            body.push_str(&format!(
                "  {w} = arith.extf {v} : {rvec} to vector<{d}x{et}>\n"
            ));
            Some(w)
        };
        let v0 = load_wide(ins.operand1.0, "vl", &mut self.body).ok_or(crate::emitter_gap!())?;
        let m0 = self
            .mem_of
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone()
            .ok_or(crate::emitter_gap!())?;
        let d = memref_lead_dim(&m0).ok_or(crate::emitter_gap!())?;
        let vecty = format!("vector<{d}x{et}>");
        let (reduce_in, kind) = match ins.imm {
            0 => {
                let v1 =
                    load_wide(ins.operand2.0, "vr", &mut self.body).ok_or(crate::emitter_gap!())?;
                let prod = format!("%vp{idx}");
                self.body += &format!("  {prod} = arith.mulf {v0}, {v1} : {vecty}\n");
                (prod, "add")
            }
            1 => (v0, "add"),
            2 => (v0, "maximumf"),
            3 => (v0, "minimumf"),
            _ => return Err(crate::emitter_gap!()),
        };
        let n = format!("%v{idx}");
        self.body +=
            &format!("  {n} = vector.reduction <{kind}>, {reduce_in} : {vecty} into {et}\n");
        self.names[idx] = n;
        self.etypes[idx] = Some(e);
        Ok(())
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
        let md = tensor_memref_ty(elem, shape).ok_or(crate::emitter_gap!())?;
        // A dynamic extent would need its own `memref.dim` operands on the allocation, which the
        // flattener already declines rather than emitting an allocation with none.
        if md.contains('?') {
            return Err(Decline::TypeNotModelled {
                what: "a matmul result whose shape is not static",
            });
        }
        // Floats only, on the same terms as `MatmulInto`: linalg's integer semantics are not
        // improvised here.
        let et = mlir_scalar(elem).ok_or(crate::emitter_gap!())?;
        if !matches!(et, "f32" | "f64" | "f16" | "bf16") {
            return Err(crate::emitter_gap!());
        }
        let n = format!("%v{idx}");
        self.body += &format!("  {n} = memref.alloc() : {md}\n");
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
        // evicted every f16 attention program from the flat path, prover and all.
        let et = memref_element(&md).ok_or(crate::emitter_gap!())?;
        if et != "f32" && et != "f64" && et != "f16" && et != "bf16" {
            return Err(crate::emitter_gap!());
        }
        self.body += &format!("  %mz{idx} = arith.constant 0.0 : {et}\n");
        self.body += &format!("  linalg.fill ins(%mz{idx} : {et}) outs({dst} : {md})\n");
        self.body += &format!("  linalg.matmul ins({a}, {b} : {ma}, {mb}) outs({dst} : {md})\n");
        Ok(())
    }

    // `flash_attention_into(&mut o, &q, &k, &v, scale)` (no result): a serial
    // `o = softmax(q @ k^T * scale) @ v` nest, preceded by a `vx.attention_note` naming
    // which memref plays which role. The nest is the correctness contract — a runtime
    // that cannot (or will not) route runs it as written; the note is what
    // `kernelKindOf` classifies as `kind=attention` so a runtime that CAN route hands
    // the region to a fused vendor kernel instead. o, v and scale ride the imm (see
    // `Opcode::FlashAttnInto`). f16 storage with f32 arithmetic throughout, the same
    // split the widening contracts (Vx#320) give every half slice.
    pub(crate) fn op_flash_attn_into(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let o_reg = (ins.imm & 0xffff) as usize;
        let v_reg = ((ins.imm >> 16) & 0xffff) as usize;
        let s_reg = ((ins.imm >> 32) & 0xffff) as usize;
        let q = self
            .names
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        let mq = self
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
        let mk = self
            .mem_of
            .get(ins.operand2.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone()
            .ok_or(crate::emitter_gap!())?;
        let o = self.names.get(o_reg).ok_or(crate::emitter_gap!())?.clone();
        let mo = self
            .mem_of
            .get(o_reg)
            .ok_or(crate::emitter_gap!())?
            .clone()
            .ok_or(crate::emitter_gap!())?;
        let v = self.names.get(v_reg).ok_or(crate::emitter_gap!())?.clone();
        let mv = self
            .mem_of
            .get(v_reg)
            .ok_or(crate::emitter_gap!())?
            .clone()
            .ok_or(crate::emitter_gap!())?;
        let sc = self.names.get(s_reg).ok_or(crate::emitter_gap!())?.clone();
        if !matches!(self.elem_at(s_reg as u32), Some(ElementType::F32)) {
            return Err(crate::emitter_gap!());
        }
        // "memref<AxBxf16>" -> (A, B, "f16"); anything shaped differently (a strided
        // layout, a surprise rank) declines rather than guesses.
        let dims = |m: &str| -> Option<(i64, i64, String)> {
            let inner = m.strip_prefix("memref<")?.strip_suffix('>')?;
            let mut it = inner.split('x');
            let a = it.next()?.parse::<i64>().ok()?;
            let b = it.next()?.parse::<i64>().ok()?;
            let e = it.next()?.to_string();
            if it.next().is_some() {
                return None;
            }
            Some((a, b, e))
        };
        let (sq, hd, eo) = dims(&mo).ok_or(crate::emitter_gap!())?;
        let (sq2, hd2, eq) = dims(&mq).ok_or(crate::emitter_gap!())?;
        let (sk, hd3, ek) = dims(&mk).ok_or(crate::emitter_gap!())?;
        let (sk2, hd4, ev) = dims(&mv).ok_or(crate::emitter_gap!())?;
        if [&eo, &eq, &ek, &ev].iter().any(|e| e.as_str() != "f16") {
            return Err(crate::emitter_gap!());
        }
        if sq != sq2 || sk != sk2 || hd != hd2 || hd != hd3 || hd != hd4 {
            return Err(crate::emitter_gap!());
        }
        self.body += &format!(
            "  \"vx.attention_note\"({o}, {q}, {k}, {v}, {sc}) : ({mo}, {mq}, {mk}, {mv}, f32) -> ()\n"
        );
        self.body += &format!("  %fasb{idx} = memref.alloca() : memref<{sk}xf32>\n");
        self.body += &format!("  %fac0{idx} = arith.constant 0 : index\n");
        self.body += &format!("  %fac1{idx} = arith.constant 1 : index\n");
        self.body += &format!("  %facq{idx} = arith.constant {sq} : index\n");
        self.body += &format!("  %fack{idx} = arith.constant {sk} : index\n");
        self.body += &format!("  %facd{idx} = arith.constant {hd} : index\n");
        self.body += &format!("  %faz{idx} = arith.constant 0.000000e+00 : f32\n");
        // -inf: the identity of max over scores that may all be negative.
        self.body += &format!("  %fan{idx} = arith.constant 0xFF800000 : f32\n");
        self.body +=
            &format!("  scf.for %fai{idx} = %fac0{idx} to %facq{idx} step %fac1{idx} {{\n");
        // Pass 1: this row's scores into the scratch buffer, tracking their max.
        self.body += &format!(
            "    %fam{idx} = scf.for %faj{idx} = %fac0{idx} to %fack{idx} step %fac1{idx} iter_args(%famx{idx} = %fan{idx}) -> (f32) {{\n"
        );
        self.body += &format!(
            "      %fad{idx} = scf.for %fae{idx} = %fac0{idx} to %facd{idx} step %fac1{idx} iter_args(%faa{idx} = %faz{idx}) -> (f32) {{\n"
        );
        self.body +=
            &format!("        %faqh{idx} = memref.load {q}[%fai{idx}, %fae{idx}] : {mq}\n");
        self.body += &format!("        %faqf{idx} = arith.extf %faqh{idx} : f16 to f32\n");
        self.body +=
            &format!("        %fakh{idx} = memref.load {k}[%faj{idx}, %fae{idx}] : {mk}\n");
        self.body += &format!("        %fakf{idx} = arith.extf %fakh{idx} : f16 to f32\n");
        self.body += &format!("        %fap{idx} = arith.mulf %faqf{idx}, %fakf{idx} : f32\n");
        self.body += &format!("        %fapa{idx} = arith.addf %faa{idx}, %fap{idx} : f32\n");
        self.body += &format!("        scf.yield %fapa{idx} : f32\n");
        self.body += "      }\n";
        self.body += &format!("      %fas{idx} = arith.mulf %fad{idx}, {sc} : f32\n");
        self.body +=
            &format!("      memref.store %fas{idx}, %fasb{idx}[%faj{idx}] : memref<{sk}xf32>\n");
        self.body += &format!("      %fam2{idx} = arith.maximumf %famx{idx}, %fas{idx} : f32\n");
        self.body += &format!("      scf.yield %fam2{idx} : f32\n");
        self.body += "    }\n";
        // Pass 2: exponentiate shifted scores in place, summing them.
        self.body += &format!(
            "    %fal{idx} = scf.for %fajj{idx} = %fac0{idx} to %fack{idx} step %fac1{idx} iter_args(%fall{idx} = %faz{idx}) -> (f32) {{\n"
        );
        self.body +=
            &format!("      %fasl{idx} = memref.load %fasb{idx}[%fajj{idx}] : memref<{sk}xf32>\n");
        self.body += &format!("      %fash{idx} = arith.subf %fasl{idx}, %fam{idx} : f32\n");
        // `math.exp`, not a `func.call` into the stdlib: math is the portable spelling
        // (libm on the host pipeline, the NVVM intrinsic on device), and a func.call to
        // a host symbol inside a kernel cannot be loaded (see useDeviceMathIn).
        self.body += &format!("      %faex{idx} = math.exp %fash{idx} : f32\n");
        self.body +=
            &format!("      memref.store %faex{idx}, %fasb{idx}[%fajj{idx}] : memref<{sk}xf32>\n");
        self.body += &format!("      %fal2{idx} = arith.addf %fall{idx}, %faex{idx} : f32\n");
        self.body += &format!("      scf.yield %fal2{idx} : f32\n");
        self.body += "    }\n";
        // Pass 3: o[i][d] = sum_j p[j] * v[j][d] / l, narrowed once on store.
        self.body +=
            &format!("    scf.for %fadd{idx} = %fac0{idx} to %facd{idx} step %fac1{idx} {{\n");
        self.body += &format!(
            "      %faoc{idx} = scf.for %fajk{idx} = %fac0{idx} to %fack{idx} step %fac1{idx} iter_args(%faoa{idx} = %faz{idx}) -> (f32) {{\n"
        );
        self.body += &format!(
            "        %fapl{idx} = memref.load %fasb{idx}[%fajk{idx}] : memref<{sk}xf32>\n"
        );
        self.body +=
            &format!("        %favh{idx} = memref.load {v}[%fajk{idx}, %fadd{idx}] : {mv}\n");
        self.body += &format!("        %favf{idx} = arith.extf %favh{idx} : f16 to f32\n");
        self.body += &format!("        %fapv{idx} = arith.mulf %fapl{idx}, %favf{idx} : f32\n");
        self.body += &format!("        %fao2{idx} = arith.addf %faoa{idx}, %fapv{idx} : f32\n");
        self.body += &format!("        scf.yield %fao2{idx} : f32\n");
        self.body += "      }\n";
        self.body += &format!("      %faon{idx} = arith.divf %faoc{idx}, %fal{idx} : f32\n");
        self.body += &format!("      %faoh{idx} = arith.truncf %faon{idx} : f32 to f16\n");
        self.body += &format!("      memref.store %faoh{idx}, {o}[%fai{idx}, %fadd{idx}] : {mo}\n");
        self.body += "    }\n";
        self.body += "  }\n";
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
