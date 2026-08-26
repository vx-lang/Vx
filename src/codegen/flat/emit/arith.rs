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
            // Rank 1 only. The operands are read with a single-index `vector.load`, which a
            // rank-2 memref rejects ("requires 2 indices"), and flattening the shape to one
            // vector would address it as if it were contiguous rank-1 storage.
            if shape.len() != 1 {
                return Err(Decline::TypeNotModelled {
                    what: "an elementwise op on a tensor that is not rank 1",
                });
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
        let src = self.elem_at(ins.operand1.0).ok_or(crate::emitter_gap!())?;
        let tgt = elem_of_gid(
            *self
                .types
                .get(ins.type_idx.0 as usize)
                .ok_or(crate::emitter_gap!())?,
        )
        .ok_or(crate::emitter_gap!())?;
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
}
