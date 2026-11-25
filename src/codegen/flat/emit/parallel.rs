//! The parallel constructs: memory transfers, spawn regions and barriers.
//!
//! One `impl` block on the per-function emitter; the dispatch that reaches these
//! lives in `step`, next to the struct.

use super::super::*;

impl FnEmit<'_> {
    // Move a tensor to a memory space: `operand1` is the source, the low half of `imm` the target
    // space's dispatch id and the high half the source's topology, `type_idx` the result tensor
    // (same element + shape). Emits `vx.transfer`
    // (generic form) with `target_topology`; the vx→standard lowering turns it into an
    // alloc + `memref.copy` (it ignores the source's layout suffix, so the result is a plain
    // `memref<NxT>`). When the target space declares a sub-space descriptor, re-attach the
    // scheduling attrs (`space`/`within`/`granule`/`capacity`/`scope` + a bump-allocated
    // `offset`/`slots`) the AST path emits — a device backend needs them to place the tile
    // into VMEM/TMEM, and they are dropped otherwise.
    pub(crate) fn op_transfer(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let src = self
            .names
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        // A scalar operand: the op restates its type, and lowering folds it to the operand.
        // A tensor: the result is the transferred tile's memref.
        let (srcty, dstty, elem, shape) = if let Some(e) = self.elem_at(ins.operand1.0) {
            let t = mlir_scalar(&e).ok_or(crate::emitter_gap!())?.to_string();
            self.etypes[idx] = Some(e);
            (t.clone(), t, None, None)
        } else {
            let result_gid = *self
                .types
                .get(ins.type_idx.0 as usize)
                .ok_or(crate::emitter_gap!())?;
            let (elem, shape) = self
                .ctx
                .tensors
                .get(&result_gid)
                .ok_or(crate::emitter_gap!())?;
            let dstty = tensor_memref_ty(elem, shape).ok_or(crate::emitter_gap!())?;
            let srcty = self
                .mem_of
                .get(ins.operand1.0 as usize)
                .ok_or(crate::emitter_gap!())?
                .clone()
                .ok_or(crate::emitter_gap!())?;
            (srcty, dstty, Some(elem.clone()), Some(shape.clone()))
        };
        let n = format!("%v{idx}");
        let space_id = ins.imm & 0xffff_ffff;
        let source = (ins.imm & !crate::bytecode::TRANSFER_SHARED) >> 32;
        // A space declaring `node: N` is dispatched by its NUMA band id rather than by the hash
        // of its name. The hash says which space; the backend has to know which NODE, and cannot
        // recover a node number from a hash. Copied out rather than held, so the descriptor's
        // borrow ends before the bump allocation below takes `&mut self`.
        let numa_target = self
            .ctx
            .subspaces
            .get(&space_id)
            .and_then(|d| d.numa_node)
            .and_then(crate::arch::numa_dispatch_id);
        let target = numa_target.map(|i| i as u64).unwrap_or(space_id);
        let mut attrs = format!("target_topology = {} : i32", target);
        if source != 0 {
            attrs += &format!(", source_topology = {source} : i32");
        }
        if ins.imm & crate::bytecode::TRANSFER_SHARED != 0 {
            attrs += ", shared";
        }
        if let Some(desc) = self.ctx.subspaces.get(&space_id) {
            // Descriptor attrs, in the AST path's emission order (MLIR sorts on print, so the
            // final parsed form is byte-identical regardless of the order emitted here).
            attrs += &format!(", space = \"{}\"", desc.name);
            if let Some(w) = &desc.within {
                attrs += &format!(", within = \"{w}\"");
            }
            if let Some(g) = desc.granule {
                attrs += &format!(", granule = {g} : i64");
            }
            if let Some(c) = desc.capacity {
                attrs += &format!(", capacity = {c} : i64");
            }
            if let Some(s) = &desc.scope {
                attrs += &format!(", scope = \"{s}\"");
            }
            if let Some(m) = &desc.managed {
                attrs += &format!(", managed = \"{m}\"");
            }
            // SS2 bump allocation: a statically-shaped tile into a granule'd space claims the
            // next granule-rounded `offset`; `slots` is the granule count it occupies.
            let tile_bytes = match (&elem, &shape) {
                (Some(e), Some(s)) => static_tile_bytes(e, s),
                _ => None,
            };
            if let (Some(granule), Some(bytes)) = (desc.granule, tile_bytes) {
                if granule > 0 {
                    let rounded = bytes.div_ceil(granule) * granule;
                    let offset = *self.subspace_offsets.entry(space_id).or_insert(0);
                    self.subspace_offsets.insert(space_id, offset + rounded);
                    attrs += &format!(", offset = {offset} : i64");
                    attrs += &format!(", slots = {} : i64", rounded / granule);
                }
            }
        } else if let Some(name) = crate::arch::builtin_memory_space_name(space_id) {
            // A built-in space declares nothing, but its name still goes on the op.
            attrs += &format!(", space = \"{name}\"");
        }
        // The count of the passes of a nest of blocks, which `--emit-ptx` names by its length.
        let user_lowered = ins.operand2.0 & crate::bytecode::TRANSFER_USER_LOWERED;
        let is_async = ins.operand2.0 & crate::bytecode::TRANSFER_ASYNC;
        let read_only_source = ins.operand2.0 & crate::bytecode::TRANSFER_READ_ONLY_SOURCE;
        let borrowed_for_call = ins.operand2.0 & crate::bytecode::TRANSFER_SOURCE_BORROWED_FOR_CALL;
        let arrivals = ins.operand2.0
            & !(crate::bytecode::TRANSFER_USER_LOWERED
                | crate::bytecode::TRANSFER_ASYNC
                | crate::bytecode::TRANSFER_READ_ONLY_SOURCE
                | crate::bytecode::TRANSFER_SOURCE_BORROWED_FOR_CALL);
        if let Some(i) = (arrivals as usize).checked_sub(1) {
            let length = self.strings.get(i).ok_or(crate::emitter_gap!())?;
            attrs += &format!(", vx.arrivals = \"{length}\"");
        }
        // The body of a user lowering copies the tile, so the backend only places it.
        if user_lowered != 0 {
            attrs += ", user_lowered";
        }
        // A `transfer_async`: a device copies in the background until a use of the tensor.
        if is_async != 0 {
            attrs += ", async";
        }
        if read_only_source != 0 {
            attrs += ", read_only_source";
        }
        if borrowed_for_call != 0 {
            attrs += ", source_borrowed_for_call";
        }
        self.body +=
            &format!("  {n} = \"vx.transfer\"({src}) {{{attrs}}} : ({srcty}) -> {dstty}\n");
        self.names[idx] = n;
        if elem.is_some() {
            self.mem_of[idx] = Some(dstty);
        }
        Ok(())
    }

    // Open a `vx.spawn` region (generic form). The op is inline in the enclosing block, which
    // continues after it; the instructions up to the matching `SpawnEnd` form the region body.
    // The ops immediately after `Spawn` (a control-flow body's setup, before its first explicit
    // block) go in the region's entry block, so open a label for it. The low half of `imm` is the
    // topology dispatch id (the AST path's `topology` attribute), the high half its name's index.
    pub(crate) fn op_spawn(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        self.spawn_stack
            .push((ins.imm as u32 as i64, self.body.len(), ins.imm >> 32));
        self.region_slots.push(None);
        self.body += &format!("  \"vx.spawn\"() ({{\n^bbspawn{idx}:\n");
        self.terminated = false;
        Ok(())
    }

    // Close the `vx.spawn` region: terminate its last block with `vx.yield` (unless a body
    // terminator already ended it), stamp the `topology` attribute, and resume emitting into
    // the enclosing block (which the spawn op did not terminate).
    /// The MLIR type of register `r`: a pointer for a slot, or the type of the value.
    pub(crate) fn value_mlir_ty(&self, r: usize) -> Lowered<String> {
        if let Some(Some(m)) = self.mem_of.get(r) {
            return Ok(m.clone());
        }
        let in_slot = self.agg_of.get(r).copied().flatten().is_some();
        if in_slot || self.ptr_of.get(r).copied().unwrap_or(false) {
            return Ok("!llvm.ptr".to_string());
        }
        let by_value = self.agg_val_of.get(r).copied().flatten();
        if let Some(a) = by_value.and_then(|g| self.ctx.aggs.get(&g)) {
            return Ok(a.struct_ty.clone());
        }
        let elem = self.elem_at(r as u32).ok_or(crate::emitter_gap!())?;
        Ok(mlir_scalar(&elem).ok_or(crate::emitter_gap!())?.to_string())
    }

    /// Yields the value of register `r` from the region, when the block is not terminated, and
    /// gives its MLIR type. A struct in a slot is yielded as its value, and the struct's id
    /// comes back too.
    pub(crate) fn yield_value(
        &mut self,
        idx: usize,
        r: usize,
    ) -> Lowered<(String, Option<TypeId>)> {
        let mut v = self.names.get(r).ok_or(crate::emitter_gap!())?.clone();
        let struct_of = |gid: Option<TypeId>| {
            gid.and_then(|g| self.ctx.aggs.get(&g))
                .map(|a| (gid.unwrap(), a.struct_ty.clone()))
        };
        let in_slot = struct_of(self.agg_of.get(r).copied().flatten());
        let by_value = struct_of(self.agg_val_of.get(r).copied().flatten());
        let (ty, gid) = if let Some(Some(m)) = self.mem_of.get(r) {
            (m.clone(), None)
        } else if let Some((gid, sty)) = in_slot.or(by_value.clone()) {
            if by_value.is_none() && !self.terminated {
                let loaded = format!("%v{idx}_value");
                self.body += &format!("  {loaded} = llvm.load {v} : !llvm.ptr -> {sty}\n");
                v = loaded;
            }
            (sty, Some(gid))
        } else {
            let elem = self.elem_at(r as u32).ok_or(crate::emitter_gap!())?;
            (
                mlir_scalar(&elem).ok_or(crate::emitter_gap!())?.to_string(),
                None,
            )
        };
        if !self.terminated {
            self.body += &format!("  \"vx.yield\"({v}) : ({ty}) -> ()\n");
        }
        Ok((ty, gid))
    }

    pub(crate) fn op_spawn_end(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let (topo, open_at, name) = self.spawn_stack.pop().ok_or(crate::emitter_gap!())?;
        let across = matches!(self.region_slots.pop(), Some(Some(_)));
        // The topology's name as the program wrote it, from the function's string table, and the
        // line of the `spawn` from the string after it. A task has no name: its high half is the
        // line of its `spawn`.
        let file = self.file.map_or(String::new(), |f| {
            format!(
                ", vx.file = \"{}\"",
                f.replace('\\', "\\\\").replace('"', "\\\"")
            )
        });
        let name = match name.checked_sub(1) {
            _ if topo == crate::arch::TASK_DISPATCH_ID as i64 => {
                format!(", vx.line = {name} : i64{file}")
            }
            Some(i) => {
                let s = self.strings.get(i as usize).ok_or(crate::emitter_gap!())?;
                let line: u64 = self
                    .strings
                    .get(i as usize + 1)
                    .and_then(|l| l.parse().ok())
                    .ok_or(crate::emitter_gap!())?;
                format!(
                    ", topology_name = \"{}\", vx.line = {line} : i64{file}",
                    s.replace('\\', "\\\\").replace('"', "\\\"")
                )
            }
            None => String::new(),
        };
        // A region with a value yields `operand1`, and the op that opened at `open_at` gets
        // the result name now that the value's type is known. A scalar, a tensor, or a struct
        // (a task can return one). A struct in a slot is yielded as its value.
        let mut result_ty: Option<String> = None;
        if self.types.get(ins.type_idx.0 as usize).is_some() {
            let r = ins.operand1.0 as usize;
            let (ty, gid) = self.yield_value(idx, r)?;
            if gid.is_some() {
                self.agg_val_of[idx] = gid;
            }
            let n = format!("%v{idx}");
            self.body.insert_str(open_at + 2, &format!("{n} = "));
            self.names[idx] = n;
            if self.mem_of.get(r).is_some_and(|m| m.is_some()) {
                self.mem_of[idx] = Some(ty.clone());
            } else if self.agg_val_of[idx].is_none() {
                self.etypes[idx] = self.elem_at(ins.operand1.0);
            }
            result_ty = Some(ty);
        } else if !self.terminated {
            self.body += "  \"vx.yield\"() : () -> ()\n";
        }
        let result_ty = result_ty.unwrap_or_default();
        // The topology gives its `arch:` when it has one, so the device pipeline can
        // gate on the architecture. Discardable attribute on the generic form -- no
        // dialect change involved.
        //
        // A nonzero `imm` is the trip count the checker's recorded split gives the region's
        // outermost loop: `vx_parallel_trip` rides the same attribute dict, telling the
        // device pipeline the loop is safe to grid-stride and how wide the work is (#251).
        // The `SPAWN_TWO_LEVEL` bit (Vx#379) marks the trip as a BLOCK count instead, and
        // `vx_parallel_two_level` rides along so the launch is sized as blocks x threads.
        let two_level = ins.imm & crate::hir::flatten::SPAWN_TWO_LEVEL != 0;
        // The `SPAWN_COOP` bit (Vx#379 stage C): the barriers are INSIDE the thread
        // loop, so the region has no serial schedule and the host must refuse it --
        // `vx_parallel_coop` travels to the launch payload for exactly that refusal.
        let coop = ins.imm & crate::hir::flatten::SPAWN_COOP != 0;
        let trip_count = ins.imm & 0xffff_ffff;
        // A trip of 0 says that the trip is known only at run time.
        let run_time_trip = ins.imm & crate::hir::flatten::SPAWN_RUN_TIME_TRIP != 0;
        // Bits 32..47: the widest thread-loop trip, i.e. the block shape the launch
        // should use (Vx#379). Zero (a pre-two-level stream) falls back to 128.
        let threads = (ins.imm >> 32) & 0xffff;
        let across = if across { ", vx.across" } else { "" };
        let trip = if trip_count > 0 || run_time_trip {
            format!(
                ", vx_parallel_trip = {trip_count} : i64{}",
                if two_level {
                    format!(
                        ", vx_parallel_two_level, vx_parallel_threads = {} : i64{}",
                        if threads > 0 { threads } else { 128 },
                        if coop { ", vx_parallel_coop" } else { "" }
                    )
                } else {
                    String::new()
                }
            )
        } else {
            String::new()
        };

        let mut attrs = String::new();
        if let Some(declared) = self.ctx.topo_arch(topo) {
            if !declared.arch.is_empty() {
                attrs.push_str(&format!("arch = \"{}\", ", declared.arch));
            }
            if let Some(offload) = declared.offload {
                attrs.push_str(&format!("offload = {}, ", offload));
            }
            if let Some(chip) = &declared.chip {
                attrs.push_str(&format!("chip = \"{}\", ", chip));
            }
            if let Some(bytes) = declared.shared_bytes {
                attrs.push_str(&format!("shared_bytes = {bytes} : i64, "));
            }
        }

        self.body += &format!(
            "  }}) {{{attrs}topology = {topo} : i32{name}{trip}{across}}} : () -> ({result_ty})\n"
        );
        self.terminated = false;
        Ok(())
    }

    // Print a string literal (no result): take the address of the module-level global emitted
    // for this string (`@".str.<n>"`, `n = str_base + imm`) and call the `@print_str` runtime
    // helper. `emit_module_mlir` emits the global's bytes and the helper's `private` decl.
    // Conditional abort -- `assert`, and `abort()` with a constant-false condition.
    // `cf.assert` rather than a hand-written branch onto a call, because MLIR lowers it
    // for whichever target the code reaches: `puts` + `abort` + `unreachable` on the
    // host, `__assertfail` inside a kernel. It is also an ORDINARY op, not a
    // terminator, so it needs no block splitting here.
    // Block-level sync (Vx#379). A registered op with no results and no Pure trait, so
    // the greedy folder cannot erase it before the device clone rewrites it to
    // `gpu.barrier`; the host lowering erases it instead (a serial loop IS the barrier).
    // `subgroup`: only the lanes of one subgroup wait, as at the end of a lane loop in a
    // subgroup loop.
    pub(crate) fn op_barrier(&mut self, _idx: usize, ins: &HirInstruction) -> Lowered<()> {
        if ins.imm == crate::bytecode::BARRIER_GPU {
            self.body += "  gpu.barrier\n";
        } else if ins.imm == crate::bytecode::BARRIER_SUBGROUP {
            self.body += "  \"vx.barrier\"() {subgroup} : () -> ()\n";
        } else {
            self.body += "  \"vx.barrier\"() : () -> ()\n";
        }
        Ok(())
    }

    /// A call to a lane routine: its op, with the `Arg`s before as its operands. The lowering of
    /// each copy of the kernel expands the op, on the GPU and into loops on the CPU.
    pub(crate) fn op_lane_call(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        use crate::hir::check::lanes::{LaneArg, LaneAttr, LANE_ROUTINES};
        let call = crate::bytecode::LaneCallImm::decode(ins.imm);
        let routine = LANE_ROUTINES
            .get(call.routine)
            .ok_or(crate::emitter_gap!())?;
        let kinds = routine
            .operand_kinds(call.operands)
            .ok_or(crate::emitter_gap!())?;
        if self.pending_args.len() < call.operands {
            return Err(crate::emitter_gap!());
        }
        let regs = self
            .pending_args
            .split_off(self.pending_args.len() - call.operands);
        let mut names = Vec::new();
        let mut types = Vec::new();
        for (r, kind) in regs.into_iter().zip(kinds) {
            let r = r as usize;
            names.push(self.names.get(r).ok_or(crate::emitter_gap!())?.clone());
            types.push(match kind {
                kind if kind.is_tensor() => self
                    .mem_of
                    .get(r)
                    .cloned()
                    .flatten()
                    .ok_or(crate::emitter_gap!())?,
                LaneArg::I32 => "i32".to_string(),
                _ => {
                    let e = self
                        .etypes
                        .get(r)
                        .cloned()
                        .flatten()
                        .ok_or(crate::emitter_gap!())?;
                    mlir_scalar(&e).ok_or(crate::emitter_gap!())?.to_string()
                }
            });
        }
        let attrs: Vec<String> = routine
            .op_attributes(&call.site)
            .into_iter()
            .map(|(name, value)| match value {
                LaneAttr::Unit => name.to_string(),
                LaneAttr::Str(s) => format!("{name} = \"{s}\""),
                LaneAttr::I64(n) => format!("{name} = {n} : i64"),
            })
            .collect();
        let attrs = if attrs.is_empty() {
            String::new()
        } else {
            format!(" {{{}}}", attrs.join(", "))
        };
        let op = format!(
            "\"{}\"({}){attrs} : ({})",
            routine.mlir,
            names.join(", "),
            types.join(", ")
        );
        match routine.result {
            None => self.body += &format!("  {op} -> ()\n"),
            Some(value) => {
                let elem = value.element();
                let v = format!("%v{idx}");
                let ty = mlir_scalar(&elem).ok_or(crate::emitter_gap!())?;
                self.body += &format!("  {v} = {op} -> {ty}\n");
                self.names[idx] = v;
                self.etypes[idx] = Some(elem);
            }
        }
        Ok(())
    }

    pub(crate) fn op_launch_dim(&mut self, _idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let v = self
            .names
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        self.body += &format!(
            "  \"vx.launch_dim\"({v}) {{axis = {} : i32}} : (i32) -> ()\n",
            ins.imm
        );
        Ok(())
    }

    // The GPU's own index ops. The CPU copy of the kernel reads an index as 0 and a count as 1,
    // so a step of `min(lanes, block size)` is 1 there.
    pub(crate) fn op_across_id(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let arg = ins.imm >> 2;
        let axis = ["x", "y", "z"];
        let (op, axis) = match ins.imm & 3 {
            crate::bytecode::ACROSS_LANE => ("gpu.thread_id", "x"),
            crate::bytecode::ACROSS_STRIDE => ("gpu.block_dim", "x"),
            crate::bytecode::ACROSS_BLOCK => (
                "gpu.block_id",
                *axis.get(arg as usize).ok_or(crate::emitter_gap!())?,
            ),
            _ => (
                "gpu.grid_dim",
                *axis.get(arg as usize).ok_or(crate::emitter_gap!())?,
            ),
        };
        let n = format!("%v{idx}");
        self.body += &format!("  {n}_index = {op} {axis}\n");
        if ins.imm & 3 == crate::bytecode::ACROSS_STRIDE {
            self.body += &format!("  {n}_dim = arith.index_cast {n}_index : index to i32\n");
            self.body += &format!("  {n}_lanes = arith.constant {arg} : i32\n");
            self.body += &format!("  {n} = arith.minsi {n}_lanes, {n}_dim : i32\n");
        } else {
            self.body += &format!("  {n} = arith.index_cast {n}_index : index to i32\n");
        }
        self.names[idx] = n;
        self.etypes[idx] = Some(ElementType::I32);
        Ok(())
    }

    pub(crate) fn op_across_region(&mut self, _idx: usize, _ins: &HirInstruction) -> Lowered<()> {
        let at = self.region_slots.last_mut().ok_or(crate::emitter_gap!())?;
        *at = Some(self.body.len());
        Ok(())
    }
}
