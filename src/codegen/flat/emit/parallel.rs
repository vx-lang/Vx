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
        let source = ins.imm >> 32;
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
    pub(crate) fn op_spawn_end(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let (topo, open_at, name) = self.spawn_stack.pop().ok_or(crate::emitter_gap!())?;
        let across = matches!(self.region_slots.pop(), Some(Some(_)));
        // The topology's name as the program wrote it, from the function's string table.
        let name = match name.checked_sub(1) {
            Some(i) => {
                let s = self.strings.get(i as usize).ok_or(crate::emitter_gap!())?;
                format!(
                    ", topology_name = \"{}\"",
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
            let mut v = self.names.get(r).ok_or(crate::emitter_gap!())?.clone();
            let struct_of = |gid: Option<TypeId>| {
                gid.and_then(|g| self.ctx.aggs.get(&g))
                    .map(|a| (gid.unwrap(), a.struct_ty.clone()))
            };
            let in_slot = struct_of(self.agg_of.get(r).copied().flatten());
            let by_value = struct_of(self.agg_val_of.get(r).copied().flatten());
            let ty = if let Some(Some(m)) = self.mem_of.get(r) {
                m.clone()
            } else if let Some((gid, sty)) = in_slot.or(by_value.clone()) {
                if by_value.is_none() && !self.terminated {
                    let loaded = format!("%v{idx}_value");
                    self.body += &format!("  {loaded} = llvm.load {v} : !llvm.ptr -> {sty}\n");
                    v = loaded;
                }
                self.agg_val_of[idx] = Some(gid);
                sty
            } else {
                mlir_scalar(&self.elem_at(ins.operand1.0).ok_or(crate::emitter_gap!())?)
                    .ok_or(crate::emitter_gap!())?
                    .to_string()
            };
            if !self.terminated {
                self.body += &format!("  \"vx.yield\"({v}) : ({ty}) -> ()\n");
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
        // A topology that declared an `arch:` sends it along, so the device pipeline can
        // gate on the declaration instead of the dispatch-id band (Vx#352). Discardable
        // attribute on the generic form -- no dialect change involved.
        //
        // A nonzero `imm` is the trip count `parallel_outer_for` proved for the region's
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
        // Bits 32..47: the widest thread-loop trip, i.e. the block shape the launch
        // should use (Vx#379). Zero (a pre-two-level stream) falls back to 128.
        let threads = (ins.imm >> 32) & 0xffff;
        let across = if across { ", vx.across" } else { "" };
        let trip = if trip_count > 0 {
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
        if let Some(declared) = self.ctx.topo_archs.get(&topo) {
            let arch = &declared.arch;
            let chip = match &declared.chip {
                Some(chip) => format!(", chip = \"{chip}\""),
                None => String::new(),
            };
            self.body += &format!(
                "  }}) {{arch = \"{arch}\"{chip}, topology = {topo} : i32{name}{trip}{across}}} : () -> ({result_ty})\n"
            );
        } else {
            self.body += &format!(
                "  }}) {{topology = {topo} : i32{name}{trip}{across}}} : () -> ({result_ty})\n"
            );
        }
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
        if ins.imm == crate::bytecode::BARRIER_SUBGROUP {
            self.body += "  \"vx.barrier\"() {subgroup} : () -> ()\n";
        } else {
            self.body += "  \"vx.barrier\"() : () -> ()\n";
        }
        Ok(())
    }

    /// `vx.lane_reduce`: the lowering of each copy of the kernel expands it, into shuffles on the
    /// GPU and into loops on the CPU.
    pub(crate) fn op_lane_reduce(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
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
        let argmax = ins.imm & crate::bytecode::LANE_REDUCE_ARGMAX != 0;
        // The count: `operand2` for a sum or a larger value, the `Arg` before for the argmax.
        let count = if argmax {
            self.pending_args.pop().ok_or(crate::emitter_gap!())?
        } else {
            ins.operand2.0
        };
        let n = self
            .names
            .get(count as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        let kind = if ins.imm & crate::bytecode::LANE_REDUCE_MAX != 0 {
            "max"
        } else {
            "sum"
        };
        let mut attrs = if argmax {
            String::new()
        } else {
            format!("kind = \"{kind}\"")
        };
        let mut add = |a: &str| {
            if !attrs.is_empty() {
                attrs += ", ";
            }
            attrs += a;
        };
        if ins.imm & crate::bytecode::LANE_REDUCE_ORDERED != 0 {
            add("ordered");
        }
        if ins.imm & crate::bytecode::LANE_REDUCE_SUBGROUP != 0 {
            add("subgroup");
        }
        let v = format!("%v{idx}");
        if argmax {
            let i = self
                .names
                .get(ins.operand2.0 as usize)
                .ok_or(crate::emitter_gap!())?
                .clone();
            let imemty = self
                .mem_of
                .get(ins.operand2.0 as usize)
                .ok_or(crate::emitter_gap!())?
                .clone()
                .ok_or(crate::emitter_gap!())?;
            self.body += &format!(
                "  {v} = \"vx.lane_argmax\"({t}, {i}, {n}) {{{attrs}}} : ({memty}, {imemty}, i32) \
                 -> i32\n"
            );
            self.names[idx] = v;
            self.etypes[idx] = Some(ElementType::I32);
            return Ok(());
        }
        self.body +=
            &format!("  {v} = \"vx.lane_reduce\"({t}, {n}) {{{attrs}}} : ({memty}, i32) -> f32\n");
        self.names[idx] = v;
        self.etypes[idx] = Some(ElementType::F32);
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
