//! Calls: argument collection, direct calls, function constants and indirect calls.
//!
//! One `impl` block on the per-function emitter; the dispatch that reaches these
//! lives in `step`, next to the struct.

use super::super::*;

impl FnEmit<'_> {
    // One argument of the following `Call`: record its value register (no op emitted).
    pub(crate) fn op_arg(&mut self, _idx: usize, ins: &HirInstruction) -> Lowered<()> {
        self.pending_args.push(ins.operand1.0);
        Ok(())
    }

    // A fixed-arity call. `type_idx` is the callee's GID (resolved to name + return type via
    // `ctx.callees`); `imm` is the arg count, taken from the tail of `pending_args`. Emit
    // `%r = func.call @name(%a, %b) : (Ta, Tb) -> Tret`.
    pub(crate) fn op_call(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let gid = *self
            .types
            .get(ins.type_idx.0 as usize)
            .ok_or(crate::emitter_gap!())?;
        let callee = self.ctx.callees.get(&gid).ok_or(crate::emitter_gap!())?;
        // Return type: a scalar, an `!llvm.struct` by value for a struct-returning callee
        // (#215), a pointer, or `()` for a void callee (a `&mut` mutator called in statement
        // position, #230).
        let rt = if let Some(e) = &callee.ret {
            mlir_scalar(e).ok_or(crate::emitter_gap!())?.to_string()
        } else if let Some(agg_gid) = callee.ret_agg {
            self.ctx
                .aggs
                .get(&agg_gid)
                .ok_or(crate::emitter_gap!())?
                .struct_ty
                .clone()
        } else if callee.ret_ptr {
            "!llvm.ptr".to_string() // an FFI pointer-returning callee (#235)
        } else if callee.ret_void {
            "()".to_string()
        } else {
            return Err(Decline::TypeNotModelled {
                what: "a callee whose return type has no MLIR spelling",
            });
        };
        let n = ins.imm as usize;
        if self.pending_args.len() < n {
            return Err(crate::emitter_gap!());
        }
        let args = self.pending_args.split_off(self.pending_args.len() - n);
        let mut arg_names = Vec::with_capacity(n);
        let mut arg_types: Vec<String> = Vec::with_capacity(n);
        for a in &args {
            arg_names.push(
                self.names
                    .get(*a as usize)
                    .ok_or(crate::emitter_gap!())?
                    .clone(),
            );
            // A scalar arg is its element type; a pointer arg (a string value / FFI pointer, or
            // an aggregate *slot* passed by reference — `&v` / a `self` pointer, #242) is
            // `!llvm.ptr`; a tensor arg is its memref type.
            let at = if let Some(e) = self.elem_at(*a) {
                mlir_scalar(&e).ok_or(crate::emitter_gap!())?.to_string()
            } else if let Some(agg_gid) = self.agg_val_of.get(*a as usize).copied().flatten() {
                // A by-value aggregate argument (`push(&outer, a)` passing `a : Vec<i32>` by
                // value into a `Vec<Vec<T>>::push`) — an `!llvm.struct` value (#242).
                self.ctx
                    .aggs
                    .get(&agg_gid)
                    .ok_or(crate::emitter_gap!())?
                    .struct_ty
                    .clone()
            } else if *self.ptr_of.get(*a as usize).ok_or(crate::emitter_gap!())?
                || self.agg_of.get(*a as usize).copied().flatten().is_some()
            {
                "!llvm.ptr".to_string()
            } else {
                self.mem_of
                    .get(*a as usize)
                    .ok_or(crate::emitter_gap!())?
                    .clone()
                    .ok_or(crate::emitter_gap!())?
            };
            arg_types.push(at);
        }
        if callee.ret_void {
            // A void call binds no result register (MLIR forbids `%v = func.call ... -> ()`);
            // the call is a pure effect (mutation through a `&mut` arg). The private extern decl
            // records an empty return (no `->`) so a void `extern` declares as `(args)`. (#230)
            self.body += &format!(
                "  func.call {}({}) : ({}) -> ()\n",
                sym_ref(&callee.name),
                arg_names.join(", "),
                arg_types.join(", "),
            );
            self.calls
                .push((callee.name.clone(), arg_types.clone(), String::new()));
        } else {
            let nm = format!("%v{idx}");
            self.body += &format!(
                "  {nm} = func.call {}({}) : ({}) -> {rt}\n",
                sym_ref(&callee.name),
                arg_names.join(", "),
                arg_types.join(", "),
            );
            // Record the callee's signature so the module emitter can declare it if it is a
            // called-but-undefined symbol (an `extern`): the private decl's signature is taken
            // from the emitted call, so they match by construction.
            self.calls
                .push((callee.name.clone(), arg_types.clone(), rt.clone()));
            self.names[idx] = nm;
            if let Some(e) = &callee.ret {
                self.etypes[idx] = Some(e.clone());
            } else if callee.ret_ptr {
                self.ptr_of[idx] = true; // the call result is a pointer value (#235)
            } else if let Some(agg_gid) = callee.ret_agg {
                // A struct-returning call result is a struct *value*; tracked so it can be
                // spilled to a slot (`Store`), returned (`Ret`), or passed by value to another
                // call (#242).
                self.agg_val_of[idx] = Some(agg_gid);
            }
        }
        Ok(())
    }

    // Materialize a function pointer for a named function: `type_idx` is the target's GID
    // (name via `ctx.callees`, signature via `ctx.func_sigs`). Emit `func.constant @name : sig`
    // then cast the `FunctionType` value to an opaque `!llvm.ptr` (the ABI of a fn pointer),
    // tracked in `ptr_of`. (#242)
    pub(crate) fn op_func_const(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let gid = *self
            .types
            .get(ins.type_idx.0 as usize)
            .ok_or(crate::emitter_gap!())?;
        let callee = self.ctx.callees.get(&gid).ok_or(crate::emitter_gap!())?;
        let (params, ret) = self.ctx.func_sigs.get(&gid).ok_or(crate::emitter_gap!())?;
        let fnty = format!("({}) -> {}", params.join(", "), ret);
        let fc = format!("%fc{idx}");
        let nm = format!("%v{idx}");
        self.body += &format!(
            "  {fc} = func.constant {} : {fnty}\n",
            sym_ref(&callee.name)
        );
        self.body +=
            &format!("  {nm} = builtin.unrealized_conversion_cast {fc} : {fnty} to !llvm.ptr\n");
        self.names[idx] = nm;
        self.ptr_of[idx] = true;
        Ok(())
    }

    // A differentiated call (`grad`/`vjp`/`jvp`). `type_idx` is the TARGET's GID, not a callee's:
    // the target is materialized as a function constant and handed to Enzyme's wrapper as its
    // first argument, which is the shape Enzyme recognizes. `imm` packs the argument count with
    // the mode. The wrapper itself is never defined, so it declares like any other extern.
    pub(crate) fn op_autodiff(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let gid = *self
            .types
            .get(ins.type_idx.0 as usize)
            .ok_or(crate::emitter_gap!())?;
        let target = self.ctx.callees.get(&gid).ok_or(crate::emitter_gap!())?;
        let (params, ret) = self.ctx.func_sigs.get(&gid).ok_or(crate::emitter_gap!())?;
        let ret_elem = target.ret.clone().ok_or(Decline::TypeNotModelled {
            what: "an autodiff target whose return type has no MLIR spelling",
        })?;
        let fnty = format!("({}) -> {}", params.join(", "), ret);
        let forward =
            (ins.imm >> crate::bytecode::AUTODIFF_MODE_SHIFT) == crate::bytecode::AUTODIFF_FORWARD;
        let n = (ins.imm & 0xffff_ffff) as usize;
        if self.pending_args.len() < n {
            return Err(crate::emitter_gap!());
        }
        let args = self.pending_args.split_off(self.pending_args.len() - n);
        // The function constant leads, so the wrapper's signature starts with the target's type.
        let mut arg_names = vec![format!("%fc{idx}")];
        let mut arg_types = vec![fnty.clone()];
        for a in &args {
            arg_names.push(
                self.names
                    .get(*a as usize)
                    .ok_or(crate::emitter_gap!())?
                    .clone(),
            );
            let e = self.elem_at(*a).ok_or(Decline::TypeNotModelled {
                what: "an autodiff argument that is not a scalar",
            })?;
            arg_types.push(mlir_scalar(&e).ok_or(crate::emitter_gap!())?.to_string());
        }
        let wrapper = crate::codegen::enzyme_wrapper_name(forward, &target.name);
        let nm = format!("%v{idx}");
        self.body += &format!(
            "  %fc{idx} = func.constant {} : {fnty}\n",
            sym_ref(&target.name)
        );
        self.body += &format!(
            "  {nm} = func.call {}({}) : ({}) -> {ret}\n",
            sym_ref(&wrapper),
            arg_names.join(", "),
            arg_types.join(", "),
        );
        self.calls.push((wrapper, arg_types, ret.clone()));
        self.names[idx] = nm;
        self.etypes[idx] = Some(ret_elem);
        Ok(())
    }

    // An indirect call through a function pointer. `operand1` is the callee `!llvm.ptr`, `imm`
    // the arg count (the tail of `pending_args`, like `Call`), and this instruction's `type_idx`
    // the scalar return type. Reconstruct the function type `(arg types)->ret` from the actual
    // args, cast the pointer to it, and `func.call_indirect`. (#242)
    pub(crate) fn op_call_indirect(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let ret_elem = self.ty_at(ins.type_idx.0).ok_or(crate::emitter_gap!())?;
        let rt = mlir_scalar(&ret_elem)
            .ok_or(crate::emitter_gap!())?
            .to_string();
        let n = ins.imm as usize;
        if self.pending_args.len() < n {
            return Err(crate::emitter_gap!());
        }
        let args = self.pending_args.split_off(self.pending_args.len() - n);
        let mut arg_names = Vec::with_capacity(n);
        let mut arg_types: Vec<String> = Vec::with_capacity(n);
        for a in &args {
            arg_names.push(
                self.names
                    .get(*a as usize)
                    .ok_or(crate::emitter_gap!())?
                    .clone(),
            );
            let at = if let Some(e) = self.elem_at(*a) {
                mlir_scalar(&e).ok_or(crate::emitter_gap!())?.to_string()
            } else if let Some(agg_gid) = self.agg_val_of.get(*a as usize).copied().flatten() {
                self.ctx
                    .aggs
                    .get(&agg_gid)
                    .ok_or(crate::emitter_gap!())?
                    .struct_ty
                    .clone()
            } else if *self.ptr_of.get(*a as usize).ok_or(crate::emitter_gap!())?
                || self.agg_of.get(*a as usize).copied().flatten().is_some()
            {
                "!llvm.ptr".to_string()
            } else {
                self.mem_of
                    .get(*a as usize)
                    .ok_or(crate::emitter_gap!())?
                    .clone()
                    .ok_or(crate::emitter_gap!())?
            };
            arg_types.push(at);
        }
        let fnty = format!("({}) -> {rt}", arg_types.join(", "));
        let fnptr = self
            .names
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        let fc = format!("%fc{idx}");
        let nm = format!("%v{idx}");
        self.body +=
            &format!("  {fc} = builtin.unrealized_conversion_cast {fnptr} : !llvm.ptr to {fnty}\n");
        self.body += &format!(
            "  {nm} = func.call_indirect {fc}({}) : {fnty}\n",
            arg_names.join(", ")
        );
        self.names[idx] = nm;
        self.etypes[idx] = Some(ret_elem);
        Ok(())
    }
}
