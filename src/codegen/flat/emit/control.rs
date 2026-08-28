//! Block labels, branches, and returning from the function.
//!
//! One `impl` block on the per-function emitter; the dispatch that reaches these
//! lives in `step`, next to the struct.

use super::super::*;

impl FnEmit<'_> {
    // Block markers → MLIR blocks. Block 0 is the func's entry block (implicit; it carries the
    // params), so it gets no label; every other id opens `^bbN:`.
    pub(crate) fn op_block_start(&mut self, _idx: usize, ins: &HirInstruction) -> Lowered<()> {
        if ins.imm != 0 {
            self.body += &format!("^bb{}:\n", ins.imm);
        }
        self.terminated = false;
        Ok(())
    }

    pub(crate) fn op_br(&mut self, _idx: usize, ins: &HirInstruction) -> Lowered<()> {
        self.body += &format!("  cf.br ^bb{}\n", ins.imm);
        self.terminated = true;
        Ok(())
    }

    // `imm` packs the two targets as `then | (else << 32)` (see `flatten::pack_targets`).
    pub(crate) fn op_cond_br(&mut self, _idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let cond = self
            .names
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?;
        let then_b = ins.imm & 0xffff_ffff;
        let else_b = ins.imm >> 32;
        self.body += &format!("  cf.cond_br {cond}, ^bb{then_b}, ^bb{else_b}\n");
        self.terminated = true;
        Ok(())
    }

    pub(crate) fn op_ret(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let gid = *self
            .types
            .get(ins.type_idx.0 as usize)
            .ok_or(crate::emitter_gap!())?;
        let a = self
            .names
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        if let Some(e) = elem_of_gid(gid) {
            // Coerce the returned scalar to the function's declared return type if they differ
            // (e.g. `return 10` — a default-`i32` literal — from an `-> i64` function ->
            // `arith.extsi`), mirroring the AST's `coerce_type` at return. Otherwise the
            // `func.return` type contradicts the signature.
            let target = self.ret_elem.clone().unwrap_or_else(|| e.clone());
            if mlir_scalar(&e).ok_or(crate::emitter_gap!())?
                != mlir_scalar(&target).ok_or(crate::emitter_gap!())?
            {
                match cast_op(&e, &target).ok_or(crate::emitter_gap!())? {
                    "" => {
                        self.body += &format!(
                            "  func.return {a} : {}\n",
                            mlir_scalar(&target).ok_or(crate::emitter_gap!())?
                        )
                    }
                    op => {
                        let c = format!("%rc{idx}");
                        self.body += &format!(
                            "  {c} = {op} {a} : {} to {}\n",
                            mlir_scalar(&e).ok_or(crate::emitter_gap!())?,
                            mlir_scalar(&target).ok_or(crate::emitter_gap!())?
                        );
                        self.body += &format!(
                            "  func.return {c} : {}\n",
                            mlir_scalar(&target).ok_or(crate::emitter_gap!())?
                        );
                    }
                }
            } else {
                self.body += &format!(
                    "  func.return {a} : {}\n",
                    mlir_scalar(&e).ok_or(crate::emitter_gap!())?
                );
            }
        } else if gid == ptr_gid() {
            self.body += &format!("  func.return {a} : !llvm.ptr\n"); // a pointer return (#235)
        } else if let Some((elem, shape)) = self.ctx.tensors.get(&gid) {
            if let Some(vecty) = self.vec_of.get(ins.operand1.0 as usize).cloned().flatten() {
                // An elementwise result is a vector register; the signature promises a buffer.
                // Materialize it: alloc, store the vector, return the alloc.
                let memty = tensor_memref_ty(elem, shape).ok_or(crate::emitter_gap!())?;
                let al = vector_align_attr(&vecty);
                let rt = format!("%rt{idx}");
                let rc = format!("%rc{idx}");
                self.body += &format!("  {rt} = memref.alloc() : {memty}\n");
                self.body += &format!("  {rc} = arith.constant 0 : index\n");
                self.body += &format!("  vector.store {a}, {rt}[{rc}]{al} : {memty}, {vecty}\n");
                self.body += &format!("  func.return {rt} : {memty}\n");
            } else {
                // A memref value: its tracked type is authoritative (a strided row differs from
                // the plain spelling); fall back to the GID's shape.
                let memty = match self.mem_of.get(ins.operand1.0 as usize).cloned().flatten() {
                    Some(m) => m,
                    None => tensor_memref_ty(elem, shape).ok_or(crate::emitter_gap!())?,
                };
                self.body += &format!("  func.return {a} : {memty}\n");
            }
        } else if let Some(agg) = self.ctx.aggs.get(&gid) {
            // A struct return (#215). The operand is either a slot pointer (a constructed
            // struct) -> load the value; or already a struct value (a returned call result) ->
            // return it directly.
            if self
                .agg_of
                .get(ins.operand1.0 as usize)
                .copied()
                .flatten()
                .is_some()
            {
                let rv = format!("%rv{idx}");
                self.body += &format!("  {rv} = llvm.load {a} : !llvm.ptr -> {}\n", agg.struct_ty);
                self.body += &format!("  func.return {rv} : {}\n", agg.struct_ty);
            } else {
                self.body += &format!("  func.return {a} : {}\n", agg.struct_ty);
            }
        } else {
            return Err(crate::emitter_gap!());
        }
        self.terminated = true;
        Ok(())
    }
}
