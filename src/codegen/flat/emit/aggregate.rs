//! Aggregates: struct field access, and indexing and storing through raw pointers.
//!
//! One `impl` block on the per-function emitter; the dispatch that reaches these
//! lives in `step`, next to the struct.

use super::super::*;

impl FnEmit<'_> {
    // Store a scalar into a struct field (no result). `operand1` is the struct slot pointer,
    // `operand2` the value, `imm` the field's byte offset. GEP to the field, then `llvm.store`;
    // the field index comes from matching the offset against the layout, the value type from
    // the stored register's tracked type.
    pub(crate) fn op_field_store(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let gid = (*self
            .agg_of
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?)
        .ok_or(crate::emitter_gap!())?;
        let agg = self.ctx.aggs.get(&gid).ok_or(crate::emitter_gap!())?;
        let field_idx = agg
            .offsets
            .iter()
            .position(|&o| o == ins.imm)
            .ok_or(crate::emitter_gap!())?;
        // The field's declared MLIR type (a scalar element or `!llvm.ptr`), so a pointer field
        // (`Vec`'s `data`) stores an `!llvm.ptr` value and a scalar field its element (#242).
        let fty = agg
            .field_tys
            .get(field_idx)
            .ok_or(crate::emitter_gap!())?
            .clone();
        let slot = self
            .names
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?;
        let val = self
            .names
            .get(ins.operand2.0 as usize)
            .ok_or(crate::emitter_gap!())?;
        let p = format!("%p{idx}");
        self.body += &format!(
            "  {p} = llvm.getelementptr {slot}[0, {field_idx}] : (!llvm.ptr) -> !llvm.ptr, {}\n",
            agg.struct_ty
        );
        // A place-write store carries alias-scope metadata (M2b-2): it belongs to its own scope
        // and does not alias its disjoint siblings' scopes. Direct field stores are unscoped.
        let attrs = self
            .alias_scope_of
            .get(&idx)
            .map(|(own, sibs)| alias_store_attrs(*own, sibs))
            .unwrap_or_default();
        self.body += &format!("  llvm.store {val}, {p}{attrs} : {fty}, !llvm.ptr\n");
        Ok(())
    }

    // Load a scalar struct field. `operand1` is the struct slot, `imm` the field's byte offset,
    // and this instruction's own `type_idx` the field's scalar type. GEP to the field, then
    // `llvm.load`.
    pub(crate) fn op_field_load(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let gid = (*self
            .agg_of
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?)
        .ok_or(crate::emitter_gap!())?;
        let agg = self.ctx.aggs.get(&gid).ok_or(crate::emitter_gap!())?;
        let field_idx = agg
            .offsets
            .iter()
            .position(|&o| o == ins.imm)
            .ok_or(crate::emitter_gap!())?;
        // The field's declared MLIR type drives the load: a scalar field yields its element
        // (tracked in `etypes`), a pointer field (`Vec`'s `data`) an `!llvm.ptr` value
        // (tracked in `ptr_of`) — the type is taken from the layout, not the read register's
        // `type_idx`, so a pointer field (whose `type_idx` is `ptr_gid`) resolves too (#242).
        let fty = agg
            .field_tys
            .get(field_idx)
            .ok_or(crate::emitter_gap!())?
            .clone();
        // A pointer field pointing to a modelled aggregate (`VecIter`'s `vec : *const Vec<T>`)
        // tags its loaded value with the pointee GID, so a chained field access through it
        // (`(*self.vec).len`) GEPs the pointee. (#242)
        let pointee = agg.field_pointee.get(field_idx).copied().flatten();
        let slot = self
            .names
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?;
        let p = format!("%p{idx}");
        let n = format!("%v{idx}");
        self.body += &format!(
            "  {p} = llvm.getelementptr {slot}[0, {field_idx}] : (!llvm.ptr) -> !llvm.ptr, {}\n",
            agg.struct_ty
        );
        self.body += &format!("  {n} = llvm.load {p} : !llvm.ptr -> {fty}\n");
        self.names[idx] = n;
        if fty == "!llvm.ptr" {
            self.ptr_of[idx] = true;
            self.agg_of[idx] = pointee;
        } else if let Some(nested_gid) = agg.field_agg.get(field_idx).copied().flatten() {
            // A by-value nested-aggregate field load yields the whole `!llvm.struct` value,
            // tracked as an aggregate value so it can be re-stored / passed by value. (#242)
            self.agg_val_of[idx] = Some(nested_gid);
        } else {
            self.etypes[idx] = elem_from_mlir_scalar(&fty);
        }
        Ok(())
    }

    // The address of a by-value nested-aggregate field (`&outer.inner`): GEP to the field and
    // yield the pointer, tracked as an aggregate slot (its layout GID from `type_idx`), so a
    // chained field access or a method receiver addresses through it. (#242)
    pub(crate) fn op_field_addr(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let parent_gid = (*self
            .agg_of
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?)
        .ok_or(crate::emitter_gap!())?;
        let agg = self
            .ctx
            .aggs
            .get(&parent_gid)
            .ok_or(crate::emitter_gap!())?;
        let field_idx = agg
            .offsets
            .iter()
            .position(|&o| o == ins.imm)
            .ok_or(crate::emitter_gap!())?;
        let field_gid = *self
            .types
            .get(ins.type_idx.0 as usize)
            .ok_or(crate::emitter_gap!())?;
        let slot = self
            .names
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?;
        let n = format!("%v{idx}");
        self.body += &format!(
            "  {n} = llvm.getelementptr {slot}[0, {field_idx}] : (!llvm.ptr) -> !llvm.ptr, {}\n",
            agg.struct_ty
        );
        self.names[idx] = n;
        // A nested-aggregate field's address is tracked as an aggregate slot, so a chained field
        // access GEPs through it (#242); a *scalar* field's address (`&param.scalar`, #275 M3b
        // reference returns) is a plain element pointer — the result GID is the field's scalar GID,
        // which is not in `aggs`, so track it as a pointer instead.
        if self.ctx.aggs.contains_key(&field_gid) {
            self.agg_of[idx] = Some(field_gid);
        } else {
            self.ptr_of[idx] = true;
        }
        Ok(())
    }

    // Index a raw pointer `p[i]` (`p : *mut T`): GEP the element, then either load it (a value
    // read) or hand back the element pointer as a store place. `operand1` is the base pointer,
    // `operand2` the (scalar) index, `type_idx` the pointee element type. The GEP's base
    // element type sets the stride, so `p[i]` addresses `base + i * sizeof(T)`. (#242)
    pub(crate) fn op_ptr_index(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let base = self
            .names
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        // The pointee element is a scalar (`*mut i32`) or a by-value aggregate
        // (`*mut Vec<i32>`, #242 Vec<Vec<T>>). The GEP's base element type (`et`) sets the
        // stride either way; a struct element loads/stores the whole `!llvm.struct`.
        let gid = *self
            .types
            .get(ins.type_idx.0 as usize)
            .ok_or(crate::emitter_gap!())?;
        // A pointer whose pointee is itself a pointer (`&&T`): the element is a bare `!llvm.ptr`
        // — the deref loads/stores an 8-byte pointer and the loaded value is itself a pointer
        // (`ptr_of`), so a further deref (`**rr`) chains. (#278)
        let elem_is_ptr = gid == ptr_gid();
        let (et, scalar_e, agg_gid) = if elem_is_ptr {
            ("!llvm.ptr".to_string(), None, None)
        } else if let Some(e) = elem_of_gid(gid) {
            (
                mlir_scalar(&e).ok_or(crate::emitter_gap!())?.to_string(),
                Some(e),
                None,
            )
        } else if let Some(agg) = self.ctx.aggs.get(&gid) {
            (agg.struct_ty.clone(), None, Some(gid))
        } else {
            return Err(crate::emitter_gap!());
        };
        let imt = mlir_scalar(&self.elem_at(ins.operand2.0).ok_or(crate::emitter_gap!())?)
            .ok_or(crate::emitter_gap!())?;
        let iname = self
            .names
            .get(ins.operand2.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        let p = format!("%pg{idx}");
        self.body += &format!(
            "  {p} = llvm.getelementptr {base}[{iname}] : (!llvm.ptr, {imt}) -> !llvm.ptr, {et}\n"
        );
        if ins.imm == 1 {
            // An element place: the following `PtrStore` writes through it.
            self.names[idx] = p;
            self.ptr_of[idx] = true;
            self.pptr_elem[idx] = Some(et);
        } else {
            let n = format!("%v{idx}");
            self.body += &format!("  {n} = llvm.load {p} : !llvm.ptr -> {et}\n");
            self.names[idx] = n;
            self.etypes[idx] = scalar_e;
            self.agg_val_of[idx] = agg_gid;
            // The loaded value is a pointer (`*rr : &i32`): mark it so an outer deref treats it
            // as a `!llvm.ptr` base rather than a scalar. (#278)
            if elem_is_ptr {
                self.ptr_of[idx] = true;
            }
        }
        Ok(())
    }

    // Store into a raw-pointer place (no result): `operand1` is the `PtrIndex` place (the GEP'd
    // element pointer), `operand2` the value, and the pointee element type comes from the
    // place. (#242)
    pub(crate) fn op_ptr_store(&mut self, _idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let place = self
            .names
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        let et = self
            .pptr_elem
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone()
            .ok_or(crate::emitter_gap!())?;
        let val = self
            .names
            .get(ins.operand2.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        self.body += &format!("  llvm.store {val}, {place} : {et}, !llvm.ptr\n");
        Ok(())
    }
}
