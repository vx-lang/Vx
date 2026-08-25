//! The parallel constructs: memory transfers, spawn regions and barriers.
//!
//! One `impl` block on the per-function emitter; the dispatch that reaches these
//! lives in `step`, next to the struct.

use super::super::*;

impl FnEmit<'_> {
    // Move a tensor to a memory space: `operand1` is the source, `imm` the target space's
    // dispatch id, `type_idx` the result tensor (same element + shape). Emits `vx.transfer`
    // (generic form) with `target_topology`; the vx→standard lowering turns it into an
    // alloc + `memref.copy` (it ignores the source's layout suffix, so the result is a plain
    // `memref<NxT>`). When the target space declares a sub-space descriptor, re-attach the
    // scheduling attrs (`space`/`within`/`granule`/`capacity`/`scope` + a bump-allocated
    // `offset`/`slots`) the AST path emits — a device backend needs them to place the tile
    // into VMEM/TMEM, and they are dropped otherwise (B1/P0-1).
    pub(crate) fn op_transfer(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
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
        let src = self
            .names
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        let srcty = self
            .mem_of
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone()
            .ok_or(crate::emitter_gap!())?;
        let n = format!("%v{idx}");
        let mut attrs = format!("target_topology = {} : i32", ins.imm);
        if let Some(desc) = self.ctx.subspaces.get(&ins.imm) {
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
            let tile_bytes = static_tile_bytes(elem, shape);
            if let (Some(granule), Some(bytes)) = (desc.granule, tile_bytes) {
                if granule > 0 {
                    let rounded = bytes.div_ceil(granule) * granule;
                    let offset = *self.subspace_offsets.entry(ins.imm).or_insert(0);
                    self.subspace_offsets.insert(ins.imm, offset + rounded);
                    attrs += &format!(", offset = {offset} : i64");
                    attrs += &format!(", slots = {} : i64", rounded / granule);
                }
            }
        }
        self.body +=
            &format!("  {n} = \"vx.transfer\"({src}) {{{attrs}}} : ({srcty}) -> {dstty}\n");
        self.names[idx] = n;
        self.mem_of[idx] = Some(dstty);
        Ok(())
    }

    // Open a `vx.spawn` region (generic form). The op is inline in the enclosing block, which
    // continues after it; the instructions up to the matching `SpawnEnd` form the region body.
    // The ops immediately after `Spawn` (a control-flow body's setup, before its first explicit
    // block) go in the region's entry block, so open a label for it. `imm` is the topology
    // dispatch id (the same value the AST path emits as `vx.spawn`'s `topology` attribute).
    pub(crate) fn op_spawn(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        if self.spawn_topology.is_some() {
            return Err(crate::emitter_gap!()); // nested spawn is not modelled
        }
        self.spawn_topology = Some(ins.imm as i64);
        self.body += &format!("  \"vx.spawn\"() ({{\n^bbspawn{idx}:\n");
        self.terminated = false;
        Ok(())
    }

    // Close the `vx.spawn` region: terminate its last block with `vx.yield` (unless a body
    // terminator already ended it), stamp the `topology` attribute, and resume emitting into
    // the enclosing block (which the spawn op did not terminate).
    pub(crate) fn op_spawn_end(&mut self, _idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let topo = self.spawn_topology.take().ok_or(crate::emitter_gap!())?;
        if !self.terminated {
            self.body += "  \"vx.yield\"() : () -> ()\n";
        }
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
        if let Some(arch) = self.ctx.topo_archs.get(&topo) {
            self.body +=
                &format!("  }}) {{arch = \"{arch}\", topology = {topo} : i32{trip}}} : () -> ()\n");
        } else {
            self.body += &format!("  }}) {{topology = {topo} : i32{trip}}} : () -> ()\n");
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
    pub(crate) fn op_barrier(&mut self, _idx: usize, _ins: &HirInstruction) -> Lowered<()> {
        self.body += "  \"vx.barrier\"() : () -> ()\n";
        Ok(())
    }
}
