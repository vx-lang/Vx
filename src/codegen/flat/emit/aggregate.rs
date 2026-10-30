//! Aggregates: struct field access, and indexing and storing through raw pointers.
//!
//! One `impl` block on the per-function emitter; the dispatch that reaches these
//! lives in `step`, next to the struct.

use super::super::*;

/// Is this field a plain scalar, rather than a pointer or a memref descriptor? Only a
/// scalar slot may be written and read at a type other than the one it was declared with.
fn is_scalar_field(fty: &str) -> bool {
    !fty.starts_with("!llvm.") && !fty.starts_with("memref<")
}

impl FnEmit<'_> {
    // Store a field by explicit index.
    // The stored register supplies the active value type, including enum payload values.
    pub(crate) fn op_field_store(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let gid = (*self
            .agg_of
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?)
        .ok_or(crate::emitter_gap!())?;
        let agg = self.ctx.aggs.get(&gid).ok_or(crate::emitter_gap!())?;
        let field_idx = agg.field_index(ins.imm)?;
        // Tensor descriptors use the storage field's declared type.
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
            .ok_or(crate::emitter_gap!())?
            .clone();
        let p = format!("%p{idx}");
        self.body += &format!(
            "  {p} = llvm.getelementptr {slot}[0, {field_idx}] : (!llvm.ptr) -> !llvm.ptr, {}\n",
            agg.struct_ty
        );
        // A tensor value goes in as its descriptor. A rank the field does not have (a rank-0
        // `Tensor<f32>()` for a `[?, ?]` field, which the checker admits) has no descriptor the
        // field can hold, and no pass folds the cast, so the function declines.
        let val = match self.mem_of.get(ins.operand2.0 as usize).cloned().flatten() {
            Some(memty) => {
                if memref_rank(&memty) != descriptor_rank(&fty) {
                    return Err(crate::emitter_gap!());
                }
                // A row is a strided view whose descriptor carries an offset into its tensor. The
                // field is read back as a plain memref, whose type says the offset is 0, so the
                // offset would be lost and the read would land in row 0. Move the data pointer
                // to the row first.
                let (val, memty) = if memty.contains("strided<") {
                    let (dims_x, et) =
                        memref_lead_dims_and_elem(&memty).ok_or(crate::emitter_gap!())?;
                    let plain = format!("memref<{dims_x}{et}>");
                    let v = self.cast_memref_value(&format!("fv{idx}"), &val, &memty, &plain)?;
                    (v, plain)
                } else {
                    (val, memty)
                };
                let d = format!("%fd{idx}");
                self.body += &format!(
                    "  {d} = builtin.unrealized_conversion_cast {val} : {memty} to {fty}\n"
                );
                d
            }
            None => val,
        }; // A place-write store carries alias-scope metadata (M2b-2): it belongs to its own scope
           // and does not alias its disjoint siblings' scopes. Direct field stores are unscoped.
        let attrs = self
            .alias_scope_of
            .get(&idx)
            .map(|(own, sibs)| alias_store_attrs(*own, sibs))
            .unwrap_or_default();
        // Enum variants share storage but can store different types. Use the value's own type.
        let value_agg = self
            .agg_val_of
            .get(ins.operand2.0 as usize)
            .copied()
            .flatten()
            .and_then(|g| self.ctx.aggs.get(&g))
            .map(|a| a.struct_ty.clone());
        // A tensor goes in as its descriptor, which is a struct too, so only a value that is
        // not a tensor takes its own scalar type in a struct-shaped slot.
        let is_tensor = self
            .mem_of
            .get(ins.operand2.0 as usize)
            .cloned()
            .flatten()
            .is_some();
        let store_ty = if self.is_pointer_value(ins.operand2.0 as usize)? {
            "!llvm.ptr".to_string()
        } else {
            match self.etypes.get(ins.operand2.0 as usize).cloned().flatten() {
                Some(e) if !is_tensor => crate::mlir_ty::mlir_scalar(&e)
                    .ok_or(crate::emitter_gap!())?
                    .to_string(),
                _ => value_agg.unwrap_or_else(|| fty.clone()),
            }
        };
        self.body += &format!("  llvm.store {val}, {p}{attrs} : {store_ty}, !llvm.ptr\n");
        Ok(())
    }

    // Load a field by index, or an enum payload by variant and payload position.
    pub(crate) fn op_field_load(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let gid = (*self
            .agg_of
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?)
        .ok_or(crate::emitter_gap!())?;
        let agg = self.ctx.aggs.get(&gid).ok_or(crate::emitter_gap!())?;
        let payload_index = (ins.opcode == Opcode::EnumPayloadLoad)
            .then(|| crate::bytecode::EnumPayloadIndex::decode(ins.imm));
        let field_idx = match payload_index {
            Some(index) => (index.field as usize)
                .checked_add(1)
                .ok_or(crate::emitter_gap!())?,
            None => agg.field_index(ins.imm)?,
        };
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
        let payload = match payload_index {
            None => None,
            Some(index) => Some(
                agg.variant_payloads
                    .get(index.variant as usize)
                    .and_then(|payloads| payloads.get(index.field as usize))
                    .ok_or(crate::emitter_gap!())?,
            ),
        };
        let pointee = match payload {
            Some(payload) => payload.pointer_target_id,
            None => agg.field_pointee.get(field_idx).copied().flatten(),
        };
        let nested = match payload {
            Some(payload) => payload.aggregate_id,
            None => agg.field_agg.get(field_idx).copied().flatten(),
        };
        let pointer = match payload {
            Some(payload) => matches!(
                payload.ty,
                Type::Pointer(..) | Type::Borrow { .. } | Type::Ref(..)
            ),
            None => fty == "!llvm.ptr",
        };
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
        // Read back at the type this instruction says it produces, for the same reason the
        // store writes at the value's: an enum payload slot holds whichever variant was
        // built, not the one the layout is named after.
        let result_agg = nested
            .or_else(|| self.types.get(ins.type_idx.0 as usize).copied())
            .and_then(|g| self.ctx.aggs.get(&g))
            .map(|a| a.struct_ty.clone());
        let result_is_tensor = self
            .types
            .get(ins.type_idx.0 as usize)
            .is_some_and(|g| self.ctx.tensors.contains_key(g));
        let load_ty = if pointer {
            "!llvm.ptr".to_string()
        } else {
            match self
                .types
                .get(ins.type_idx.0 as usize)
                .copied()
                .and_then(elem_of_gid)
            {
                Some(e)
                    if payload.is_some()
                        || is_scalar_field(&fty)
                        || (fty.starts_with("!llvm.struct") && !result_is_tensor) =>
                {
                    crate::mlir_ty::mlir_scalar(&e).unwrap_or(&fty).to_string()
                }
                _ => result_agg.unwrap_or_else(|| fty.clone()),
            }
        };
        self.body += &format!("  {n} = llvm.load {p} : !llvm.ptr -> {load_ty}\n");
        self.names[idx] = n;
        // A tensor field comes out as its descriptor, cast back to the memref the result names.
        let gid_res = self.types.get(ins.type_idx.0 as usize).copied();
        if let Some((elem, shape)) = gid_res.and_then(|g| self.ctx.tensors.get(&g).cloned()) {
            let memty = tensor_memref_ty(&elem, &shape).ok_or(crate::emitter_gap!())?;
            let d = self.names[idx].clone();
            let m = format!("%fm{idx}");
            self.body +=
                &format!("  {m} = builtin.unrealized_conversion_cast {d} : {fty} to {memty}\n");
            self.names[idx] = m;
            self.mem_of[idx] = Some(memty);
            return Ok(());
        }
        if pointer {
            self.ptr_of[idx] = true;
            self.agg_of[idx] = pointee;
        } else if let Some(nested_gid) = nested {
            // A by-value nested-aggregate field load yields the whole `!llvm.struct` value,
            // tracked as an aggregate value so it can be re-stored / passed by value. (#242)
            self.agg_val_of[idx] = Some(nested_gid);
        } else if let Some(g) =
            gid_res.filter(|g| fty.starts_with("!llvm.struct") && self.ctx.aggs.contains_key(g))
        {
            // The result type can identify an aggregate when field metadata is unavailable.
            self.agg_val_of[idx] = Some(g);
        } else {
            // The instruction's own type keeps the field's signedness, which its MLIR spelling
            // does not (`i64` is both `i64` and `u64`): a comparison reads it from here.
            self.etypes[idx] = gid_res
                .and_then(elem_of_gid)
                .or_else(|| elem_from_mlir_scalar(&fty));
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
        let field_idx = agg.field_index(ins.imm)?;
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

/// The rank a memref type spells: its `x`-separated extents before the element.
pub(super) fn memref_rank(memty: &str) -> Option<usize> {
    let inner = memty.strip_prefix("memref<")?;
    let inner = inner.split(',').next()?;
    Some(inner.matches('x').count())
}

/// The rank a descriptor struct type spells: `array<N x i64>` for rank N, none for rank 0.
fn descriptor_rank(fty: &str) -> Option<usize> {
    if !fty.starts_with("!llvm.struct<(ptr, ptr, i64") {
        return None;
    }
    match fty.split_once("array<") {
        Some((_, rest)) => rest.split_once(' ')?.0.parse().ok(),
        None => Some(0),
    }
}
