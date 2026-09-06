//! Constants and the scalar operators: arithmetic, comparison, negation and casts.
//!
//! One `impl` block on the per-function emitter; the dispatch that reaches these
//! lives in `step`, next to the struct.

use super::super::*;

impl FnEmit<'_> {
    pub(crate) fn op_const(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let e = self.ty_at(ins.type_idx.0).ok_or(crate::emitter_gap!())?;
        let mt = mlir_scalar(&e).ok_or(crate::emitter_gap!())?;
        let lit = if e.is_float() {
            mlir_float_literal(f64::from_bits(ins.imm))
        } else {
            (ins.imm as i64).to_string()
        };
        let n = format!("%v{idx}");
        self.body += &format!("  {n} = arith.constant {lit} : {mt}\n");
        self.names[idx] = n;
        self.etypes[idx] = Some(e);
        Ok(())
    }

    // Arithmetic: scalar (a scalar-GID result) or *elementwise* over a rank-1 float slice (a
    // tensor-GID result). The scalar form is `arith.{addi,mulf,…}`; the elementwise form
    // coerces each operand to a `vector<Nxf32>` (`vector.load`/`broadcast`), applies
    // `arith.{addf,subf,mulf,divf}`, and yields a vector that a row `TensorStore` writes back.
    pub(crate) fn op_binary(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let result_gid = *self
            .types
            .get(ins.type_idx.0 as usize)
            .ok_or(crate::emitter_gap!())?;
        if let Some(e) = elem_of_gid(result_gid) {
            let mt = mlir_scalar(&e).ok_or(crate::emitter_gap!())?;
            let op = arith_op(ins.opcode, &e).ok_or(crate::emitter_gap!())?;
            let a = self
                .names
                .get(ins.operand1.0 as usize)
                .ok_or(crate::emitter_gap!())?;
            let b = self
                .names
                .get(ins.operand2.0 as usize)
                .ok_or(crate::emitter_gap!())?;
            let n = format!("%v{idx}");
            // The latch increment of a stridable loop -- the marker the device clone
            // widens to a stride (#251 flat; Vx#379 block/thread). Only an `Add` can
            // carry it: `lower_for` is the sole tagger.
            let attr = if ins.opcode == Opcode::Add {
                match ins.imm {
                    crate::bytecode::IMM_PARALLEL_STEP => " {vx.parallel_step}",
                    crate::bytecode::IMM_BLOCK_STEP => " {vx.parallel_bstep}",
                    crate::bytecode::IMM_THREAD_STEP => " {vx.parallel_tstep}",
                    _ => "",
                }
            } else {
                ""
            };
            self.body += &format!("  {n} = {op} {a}, {b}{attr} : {mt}\n");
            self.names[idx] = n;
            self.etypes[idx] = Some(e);
        } else {
            let (elem, shape) = self
                .ctx
                .tensors
                .get(&result_gid)
                .ok_or(crate::emitter_gap!())?;
            if !elem.is_float() {
                return Err(Decline::TypeNotModelled {
                    what: "an elementwise op on non-float elements",
                });
            }
            let et = mlir_scalar(elem).ok_or(crate::emitter_gap!())?;
            // Rank 1 takes the vector path below: the operands are read with a single-index
            // `vector.load`, which a rank-2 memref rejects ("requires 2 indices"). Anything
            // deeper is a named linalg op over the operand memrefs.
            if shape.len() != 1 {
                let (elem, shape) = (elem.clone(), shape.clone());
                return self.elementwise_linalg(idx, ins, &elem, &shape);
            }
            let d: i64 = shape[0].parse::<i64>().map_err(|_| crate::emitter_gap!())?;
            let vecty = format!("vector<{d}x{et}>");
            let va = coerce_vector(
                &mut self.body,
                &format!("{idx}a"),
                ins.operand1.0,
                &vecty,
                et,
                &self.names,
                &self.mem_of,
                &self.vec_of,
                &self.etypes,
            )?;
            let vb = coerce_vector(
                &mut self.body,
                &format!("{idx}b"),
                ins.operand2.0,
                &vecty,
                et,
                &self.names,
                &self.mem_of,
                &self.vec_of,
                &self.etypes,
            )?;
            let op = match ins.opcode {
                Opcode::Add => "arith.addf",
                Opcode::Sub => "arith.subf",
                Opcode::Mul => "arith.mulf",
                Opcode::Div => "arith.divf",
                _ => return Err(crate::emitter_gap!()),
            };
            let n = format!("%v{idx}");
            self.body += &format!("  {n} = {op} {va}, {vb} : {vecty}\n");
            self.names[idx] = n;
            self.vec_of[idx] = Some(vecty);
        }
        Ok(())
    }

    /// Elementwise arithmetic over a tensor of rank 2 or deeper: `linalg.{add,sub,mul,div}`
    /// over the two operand memrefs into a fresh buffer of the result shape, which is what the
    /// oracle emits and what `convert-linalg-to-loops` (or the vectorize pipeline's affine
    /// route) already lowers. A `?` in the result shape is read off the first operand.
    fn elementwise_linalg(
        &mut self,
        idx: usize,
        ins: &HirInstruction,
        elem: &ElementType,
        shape: &[String],
    ) -> Lowered<()> {
        let memty = tensor_memref_ty(elem, shape).ok_or(crate::emitter_gap!())?;
        let (ia, ib) = (ins.operand1.0 as usize, ins.operand2.0 as usize);
        let a = self.names.get(ia).ok_or(crate::emitter_gap!())?.clone();
        let b = self.names.get(ib).ok_or(crate::emitter_gap!())?.clone();
        let ma = self
            .mem_of
            .get(ia)
            .cloned()
            .flatten()
            .ok_or(crate::emitter_gap!())?;
        let mb = self
            .mem_of
            .get(ib)
            .cloned()
            .flatten()
            .ok_or(crate::emitter_gap!())?;
        let op = match ins.opcode {
            Opcode::Add => "linalg.add",
            Opcode::Sub => "linalg.sub",
            Opcode::Mul => "linalg.mul",
            Opcode::Div => "linalg.div",
            _ => return Err(crate::emitter_gap!()),
        };
        let mut sizes: Vec<String> = Vec::new();
        for (k, d) in shape.iter().enumerate() {
            if d == DYN_DIM {
                let c = format!("%ewc{idx}_{k}");
                let s = format!("%ews{idx}_{k}");
                self.body += &format!("  {c} = arith.constant {k} : index\n");
                self.body += &format!("  {s} = memref.dim {a}, {c} : {ma}\n");
                sizes.push(s);
            }
        }
        let n = format!("%v{idx}");
        self.body += &format!("  {n} = memref.alloc({}) : {memty}\n", sizes.join(", "));
        self.body += &format!("  {op} ins({a}, {b} : {ma}, {mb}) outs({n} : {memty})\n");
        self.names[idx] = n;
        self.mem_of[idx] = Some(memty);
        Ok(())
    }

    // Scalar comparison → `i1`; the relation is in `imm`, the operand type comes from the
    // first operand's tracked type (this instruction's own type is `bool`, the result).
    pub(crate) fn op_cmp(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let e = self.elem_at(ins.operand1.0).ok_or(crate::emitter_gap!())?;
        let mt = mlir_scalar(&e).ok_or(crate::emitter_gap!())?;
        let (op, pred) = cmp_op(ins.imm, &e).ok_or(crate::emitter_gap!())?;
        let a = self
            .names
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?;
        let b = self
            .names
            .get(ins.operand2.0 as usize)
            .ok_or(crate::emitter_gap!())?;
        let n = format!("%v{idx}");
        self.body += &format!("  {n} = {op} {pred}, {a}, {b} : {mt}\n");
        self.names[idx] = n;
        self.etypes[idx] = Some(ElementType::Bool);
        Ok(())
    }

    // Arithmetic negation `-x` (#214). `type_idx` is the result (= operand) scalar type. Float
    // → `arith.negf`; integers have no `negi`, so `0 - x` via `arith.subi`.
    pub(crate) fn op_neg(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let e = elem_of_gid(
            *self
                .types
                .get(ins.type_idx.0 as usize)
                .ok_or(crate::emitter_gap!())?,
        )
        .ok_or(crate::emitter_gap!())?;
        let mt = mlir_scalar(&e).ok_or(crate::emitter_gap!())?;
        let a = self
            .names
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        let n = format!("%v{idx}");
        if e.is_float() {
            self.body += &format!("  {n} = arith.negf {a} : {mt}\n");
        } else {
            let z = format!("%z{idx}");
            self.body += &format!("  {z} = arith.constant 0 : {mt}\n");
            self.body += &format!("  {n} = arith.subi {z}, {a} : {mt}\n");
        }
        self.names[idx] = n;
        self.etypes[idx] = Some(e);
        Ok(())
    }

    // Logical / bitwise not `!x` (#214): `x ^ all-ones` (`1` for a bool `i1`, `-1` for ints).
    pub(crate) fn op_not(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let e = elem_of_gid(
            *self
                .types
                .get(ins.type_idx.0 as usize)
                .ok_or(crate::emitter_gap!())?,
        )
        .ok_or(crate::emitter_gap!())?;
        let mt = mlir_scalar(&e).ok_or(crate::emitter_gap!())?;
        let a = self
            .names
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        let ones_val = if matches!(e, ElementType::Bool) {
            "1"
        } else {
            "-1"
        };
        let ones = format!("%ones{idx}");
        let n = format!("%v{idx}");
        self.body += &format!("  {ones} = arith.constant {ones_val} : {mt}\n");
        self.body += &format!("  {n} = arith.xori {a}, {ones} : {mt}\n");
        self.names[idx] = n;
        self.etypes[idx] = Some(e);
        Ok(())
    }

    // Scalar `as` cast (#214): `type_idx` is the *target* type, `operand1` the source value
    // (whose type comes from its tracked `etypes`). The right `arith` conversion is chosen by
    // the source/target kinds + widths; a same-type cast is a no-op that just aliases.
    pub(crate) fn op_cast(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let gid = *self
            .types
            .get(ins.type_idx.0 as usize)
            .ok_or(crate::emitter_gap!())?;
        // A slice `as`: the source is a vector an elementwise op produced, so the cast is
        // `arith.truncf`/`arith.extf` over the lanes. This is the spelling that narrows a
        // widened slice back into half storage, which the checker requires be written.
        if let Some(src_vec) = self
            .vec_of
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone()
        {
            // `as f16` names an element type, so the target GID is the scalar; the lane
            // count comes from the operand it is applied to.
            let elem = elem_of_gid(gid).ok_or(crate::emitter_gap!())?;
            if !elem.is_float() {
                return Err(Decline::TypeNotModelled {
                    what: "a slice cast to a non-float element type",
                });
            }
            let lanes = src_vec
                .strip_prefix("vector<")
                .ok_or(crate::emitter_gap!())?
                .split('x')
                .next()
                .ok_or(crate::emitter_gap!())?
                .to_string();
            let src_elem = src_vec
                .rsplit('x')
                .next()
                .ok_or(crate::emitter_gap!())?
                .trim_end_matches('>')
                .to_string();
            let et = mlir_scalar(&elem).ok_or(crate::emitter_gap!())?;
            let dst_vec = format!("vector<{lanes}x{et}>");
            let a = self
                .names
                .get(ins.operand1.0 as usize)
                .ok_or(crate::emitter_gap!())?
                .clone();
            if src_elem == et {
                self.names[idx] = a;
                self.vec_of[idx] = Some(dst_vec);
                return Ok(());
            }
            let src_ty =
                crate::mlir_ty::float_elem_of_mlir(&src_elem).ok_or(crate::emitter_gap!())?;
            let src_bits = src_ty.bits().ok_or(crate::emitter_gap!())?;
            let dst_bits = elem.bits().ok_or(crate::emitter_gap!())?;
            // f16 and bf16 are both 16 bits and neither `truncf` nor `extf` converts between
            // them, so that pair declines rather than emitting a conversion that lies.
            let conv = match dst_bits.cmp(&src_bits) {
                std::cmp::Ordering::Less => "arith.truncf",
                std::cmp::Ordering::Greater => "arith.extf",
                std::cmp::Ordering::Equal => {
                    return Err(Decline::TypeNotModelled {
                        what: "a slice cast between two equal-width float types",
                    })
                }
            };
            let n = format!("%v{idx}");
            self.body += &format!("  {n} = {conv} {a} : {src_vec} to {dst_vec}\n");
            self.names[idx] = n;
            self.vec_of[idx] = Some(dst_vec);
            return Ok(());
        }
        // A tensor target: the two memref types differ only in which extents are known, so this
        // is `memref.cast` rather than any arithmetic conversion.
        if let Some((elem, shape)) = self.ctx.tensors.get(&gid).cloned() {
            return self.cast_memref(idx, ins, &elem, &shape);
        }
        let src = self.elem_at(ins.operand1.0).ok_or(crate::emitter_gap!())?;
        // An integer cast to a pointer (`0 as *mut T`): `llvm.inttoptr`, same as the AST path.
        // The flattener admits only integer sources, so `src` has an integer spelling here.
        if gid == ptr_gid() {
            let a = self
                .names
                .get(ins.operand1.0 as usize)
                .ok_or(crate::emitter_gap!())?
                .clone();
            let n = format!("%v{idx}");
            self.body += &format!(
                "  {n} = llvm.inttoptr {a} : {} to !llvm.ptr\n",
                mlir_scalar(&src).ok_or(crate::emitter_gap!())?
            );
            self.names[idx] = n;
            self.ptr_of[idx] = true;
            return Ok(());
        }
        let tgt = elem_of_gid(gid).ok_or(crate::emitter_gap!())?;
        let a = self
            .names
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        let op = cast_op(&src, &tgt).ok_or(crate::emitter_gap!())?;
        if op.is_empty() {
            self.names[idx] = a; // reinterpret (e.g. i32 as u32) -> alias
        } else {
            let n = format!("%v{idx}");
            self.body += &format!(
                "  {n} = {op} {a} : {} to {}\n",
                mlir_scalar(&src).ok_or(crate::emitter_gap!())?,
                mlir_scalar(&tgt).ok_or(crate::emitter_gap!())?
            );
            self.names[idx] = n;
        }
        self.etypes[idx] = Some(tgt);
        Ok(())
    }

    /// Forget a tensor's extents: `memref.cast %a : memref<2x3xf32> to memref<?x?xf32>`. Emitted
    /// where a shaped value reaches a position with `?` extents, which in MLIR is a different type
    /// rather than a subtype. Identical types alias instead of emitting a no-op cast.
    fn cast_memref(
        &mut self,
        idx: usize,
        ins: &HirInstruction,
        elem: &ElementType,
        shape: &[String],
    ) -> Lowered<()> {
        let a = self
            .names
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        let src_ty = self
            .mem_of
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone()
            .ok_or(crate::emitter_gap!())?;
        let dst_ty = tensor_memref_ty(elem, shape).ok_or(crate::emitter_gap!())?;
        if src_ty == dst_ty {
            self.names[idx] = a;
        } else {
            let n = format!("%v{idx}");
            self.body += &format!("  {n} = memref.cast {a} : {src_ty} to {dst_ty}\n");
            self.names[idx] = n;
        }
        self.mem_of[idx] = Some(dst_ty);
        Ok(())
    }
}
