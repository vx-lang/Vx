//! Parameters, stack slots, and the loads and stores that reach them.
//!
//! One `impl` block on the per-function emitter; the dispatch that reaches these
//! lives in `step`, next to the struct.

use super::super::*;

impl FnEmit<'_> {
    // Parameter materialization: the register *is* the block argument, no op emitted. A
    // scalar param records its element type; a tensor param records its memref type (from the
    // side table) so later index/store ops address it.
    pub(crate) fn op_load(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        self.names[idx] = format!("%arg{}", ins.imm);
        let gid = *self
            .types
            .get(ins.type_idx.0 as usize)
            .ok_or(crate::emitter_gap!())?;
        if let Some(e) = elem_of_gid(gid) {
            self.etypes[idx] = Some(e);
        } else if gid == ptr_gid() {
            self.ptr_of[idx] = true; // a `!llvm.ptr` parameter (#235)
                                     // A pointer *to a modelled aggregate* (`self : &mut Vec<i32>`): also track its
                                     // pointee layout GID, so a `FieldLoad`/`FieldStore` through `self` GEPs the field
                                     // exactly as through an aggregate slot (#242).
            if let Some(agg_gid) = self
                .func
                .params
                .get(ins.imm as usize)
                .and_then(|(_, ty)| pointee_agg_gid(ty, self.ctx))
            {
                self.agg_of[idx] = Some(agg_gid);
            }
        } else if let Some((elem, shape)) = self.ctx.tensors.get(&gid) {
            self.mem_of[idx] = tensor_memref_ty(elem, shape);
        } else if self.ctx.aggs.contains_key(&gid) {
            // A by-value aggregate parameter (`self : Option<i32>` in `Option::unwrap`, a
            // by-value struct arg): an `!llvm.struct` value, tracked so it can be spilled to a
            // slot / passed on by value. (#242)
            self.agg_val_of[idx] = Some(gid);
        } else if let Some(Type::Simd(e, lanes)) = self
            .func
            .params
            .get(ins.imm as usize)
            .map(|(_, ty)| peel_wrappers(ty))
        {
            // A `<N x T>` parameter: a `vector<NxT>` block argument. The spelling comes off the
            // declared type rather than a GID side table -- a vector carries no shape beyond its
            // element and lane count, both of which the declaration already states.
            self.vec_of[idx] = mlir_vector(e, *lanes);
        }
        Ok(())
    }

    // A named local's stack slot. A scalar slot is a rank-0 memref (matching the AST codegen's
    // scalar locals); an aggregate (struct) slot is an `llvm.alloca` of the `!llvm.struct`
    // type, its pointer tracked in `agg_of` so field ops can address it.
    pub(crate) fn op_alloca(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let gid = *self
            .types
            .get(ins.type_idx.0 as usize)
            .ok_or(crate::emitter_gap!())?;
        if let Some(e) = elem_of_gid(gid) {
            let mt = mlir_scalar(&e).ok_or(crate::emitter_gap!())?;
            let n = format!("%v{idx}");
            if ins.imm == 1 {
                // An address-taken scalar (`&x`): an `llvm.alloca` of the element type, yielding
                // a real `!llvm.ptr` the borrow can hand out. Its `Store`/`SlotLoad` go through
                // `sslot_of` (llvm.store/load); `ptr_of` marks it so `&x` types as a pointer arg
                // / return. Do *not* set `etypes` — the slot register is a pointer, not a scalar
                // value (that would mistype `&x` as its element at a call site). (#230)
                let cnt = format!("%n{idx}");
                self.emit_slot(&format!("  {cnt} = llvm.mlir.constant(1 : i32) : i32\n"));
                self.emit_slot(&format!(
                    "  {n} = llvm.alloca {cnt} x {mt} : (i32) -> !llvm.ptr\n"
                ));
                self.names[idx] = n;
                self.sslot_of[idx] = Some(e);
                self.ptr_of[idx] = true;
            } else {
                self.emit_slot(&format!("  {n} = memref.alloca() : memref<{mt}>\n"));
                self.names[idx] = n;
                self.etypes[idx] = Some(e);
            }
        } else if gid == ptr_gid() {
            // A pointer local (memory mode): an `llvm.alloca` of one `!llvm.ptr` cell, tracked
            // in `pslot_of` so its `Store`/`SlotLoad` use `llvm.store`/`llvm.load`. (#235)
            let cnt = format!("%n{idx}");
            let n = format!("%v{idx}");
            self.emit_slot(&format!("  {cnt} = llvm.mlir.constant(1 : i32) : i32\n"));
            self.emit_slot(&format!(
                "  {n} = llvm.alloca {cnt} x !llvm.ptr : (i32) -> !llvm.ptr\n"
            ));
            self.names[idx] = n;
            self.pslot_of[idx] = true;
        } else {
            let agg = self.ctx.aggs.get(&gid).ok_or(Decline::TypeNotModelled {
                what: "an aggregate slot with no struct type",
            })?;
            let cnt = format!("%n{idx}");
            let n = format!("%v{idx}");
            self.emit_slot(&format!("  {cnt} = llvm.mlir.constant(1 : i32) : i32\n"));
            self.emit_slot(&format!(
                "  {n} = llvm.alloca {cnt} x {} : (i32) -> !llvm.ptr\n",
                agg.struct_ty
            ));
            self.names[idx] = n;
            self.agg_of[idx] = Some(gid);
        }
        Ok(())
    }

    // Store a value into a slot (no result). A scalar slot is a rank-0 `memref`; an aggregate
    // slot (a struct value spilled from a struct-returning call) is an `llvm.store` (#215).
    pub(crate) fn op_store(&mut self, _idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let slot = self
            .names
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        let val = self
            .names
            .get(ins.operand2.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        // The induction-variable init of a stridable loop carries its tag into the MLIR
        // text as a discardable attribute -- inert on the host path, the marker the
        // device clone offsets by an id (#251 flat grid-stride; Vx#379 block/thread).
        let attr = match ins.imm {
            crate::bytecode::IMM_PARALLEL_INIT => " {vx.parallel_init}",
            crate::bytecode::IMM_BLOCK_INIT => " {vx.parallel_binit}",
            crate::bytecode::IMM_THREAD_INIT => " {vx.parallel_tinit}",
            _ => "",
        };
        if let Some(&Some(agg_gid)) = self.agg_of.get(ins.operand1.0 as usize) {
            let agg = self.ctx.aggs.get(&agg_gid).ok_or(crate::emitter_gap!())?;
            self.body += &format!(
                "  llvm.store {val}, {slot} : {}, !llvm.ptr\n",
                agg.struct_ty
            );
        } else if *self
            .pslot_of
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
        {
            // A pointer local: store the `!llvm.ptr` value into its `llvm.alloca` cell. (#235)
            self.body += &format!("  llvm.store {val}, {slot} : !llvm.ptr, !llvm.ptr\n");
        } else if let Some(e) = self
            .sslot_of
            .get(ins.operand1.0 as usize)
            .cloned()
            .flatten()
        {
            // An address-taken scalar slot (an `llvm.alloca` of the element): `llvm.store`. (#230)
            let mt = mlir_scalar(&e).ok_or(crate::emitter_gap!())?;
            self.body += &format!("  llvm.store {val}, {slot}{attr} : {mt}, !llvm.ptr\n");
        } else {
            let e = self.elem_at(ins.operand1.0).ok_or(crate::emitter_gap!())?;
            let mt = mlir_scalar(&e).ok_or(crate::emitter_gap!())?;
            self.body += &format!("  memref.store {val}, {slot}[]{attr} : memref<{mt}>\n");
        }
        Ok(())
    }

    // Load a value back from a slot; the result type is the slot's element (this
    // instruction's own `type_idx`).
    pub(crate) fn op_slot_load(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        // A pointer slot loads back an `!llvm.ptr` value (`llvm.load`); an aggregate slot loads
        // the whole `!llvm.struct` value (`llvm.load`, #242 Vec<Vec<T>>); a scalar slot loads
        // its element from the rank-0 memref (`memref.load`). (#235)
        if let Some(&Some(agg_gid)) = self.agg_of.get(ins.operand1.0 as usize) {
            let agg = self.ctx.aggs.get(&agg_gid).ok_or(crate::emitter_gap!())?;
            let slot = self
                .names
                .get(ins.operand1.0 as usize)
                .ok_or(crate::emitter_gap!())?;
            let n = format!("%v{idx}");
            self.body += &format!(
                "  {n} = llvm.load {slot} : !llvm.ptr -> {}\n",
                agg.struct_ty
            );
            self.names[idx] = n;
            self.agg_val_of[idx] = Some(agg_gid);
        } else if *self
            .pslot_of
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
        {
            let slot = self
                .names
                .get(ins.operand1.0 as usize)
                .ok_or(crate::emitter_gap!())?;
            let n = format!("%v{idx}");
            self.body += &format!("  {n} = llvm.load {slot} : !llvm.ptr -> !llvm.ptr\n");
            self.names[idx] = n;
            self.ptr_of[idx] = true;
        } else if let Some(e) = self
            .sslot_of
            .get(ins.operand1.0 as usize)
            .cloned()
            .flatten()
        {
            // An address-taken scalar slot: `llvm.load` the element back from the `!llvm.ptr`. (#230)
            let mt = mlir_scalar(&e).ok_or(crate::emitter_gap!())?;
            let slot = self
                .names
                .get(ins.operand1.0 as usize)
                .ok_or(crate::emitter_gap!())?;
            let n = format!("%v{idx}");
            self.body += &format!("  {n} = llvm.load {slot} : !llvm.ptr -> {mt}\n");
            self.names[idx] = n;
            self.etypes[idx] = Some(e);
        } else {
            let e = self.ty_at(ins.type_idx.0).ok_or(crate::emitter_gap!())?;
            let mt = mlir_scalar(&e).ok_or(crate::emitter_gap!())?;
            let slot = self
                .names
                .get(ins.operand1.0 as usize)
                .ok_or(crate::emitter_gap!())?;
            let n = format!("%v{idx}");
            self.body += &format!("  {n} = memref.load {slot}[] : memref<{mt}>\n");
            self.names[idx] = n;
            self.etypes[idx] = Some(e);
        }
        Ok(())
    }
}
