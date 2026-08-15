use super::*;
use crate::syntax;
use crate::syntax::*;
use melior::ir::{
    attribute::{FlatSymbolRefAttribute, IntegerAttribute, StringAttribute},
    operation::OperationBuilder,
    Identifier, Type, Value,
};

impl<'c> LowerToMelior<'c> for syntax::SpawnOnExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let location = gen.loc();
        let region = melior::ir::Region::new();
        let mut body_block = region.append_block(melior::ir::Block::new(&[]));
        let prev_in_spawn = gen.in_spawn;
        gen.in_spawn = true;

        // Whether anything is waiting on a value from this region. A spawn in
        // statement position is asked for `none`, and its tail expression is
        // then the last thing the region does rather than something it returns
        // -- which is what a trailing `if` is. Reading it as a result gave the
        // region a `vx.yield` of the condition against a declared result type of
        // `none`, a mismatch nothing looked at because the result went unused.
        let wants_value = gen.expected_type != Some(gen.none_ty);

        // Transport host-proven `assert` facts into the device kernel as
        // `llvm.intr.assume` certificates before the body is lowered, so they dominate
        // the guard they let the device backend fold. Host targets get nothing (the
        // relation never crossed a launch boundary). Off unless `--emit-seam-certs`.
        let is_device = !matches!(
            self.top,
            Topology::CPU | Topology::CpuAvx512 | Topology::CpuNeon | Topology::Current
        );
        if gen.emit_seam_certs && is_device {
            body_block =
                super::seam_cert::emit_seam_certificates(gen, &self.stmts, &self.ret, body_block)?;
        }

        for stmt in &self.stmts {
            if let Some(b) = gen.generate_statement(stmt, body_block)? {
                body_block = b;
            }
        }

        let mut result_types = vec![];
        let mut ret_val = None;

        // The region's tail expression -- its value is the region's value. The
        // block it hands back is where the terminator has to go: an `if` (or a
        // `match`, or anything else that branches) leaves the insertion point in
        // a fresh merge block, and appending `vx.yield` to the block we started
        // in would put it after that block's `cf.cond_br`. That is invalid IR
        // rather than a wrong answer, and it is why a spawn whose last statement
        // is an `if` did not compile: the parser reads a trailing `if` as the
        // tail expression whether or not a semicolon follows it, so
        // `spawn on(...) { ...; if c { o[0][0] = 1.0; } }` came through here.
        if let Some(r) = &self.ret {
            let (val, ty, tail_block) = gen.generate_expr(r, body_block)?;
            body_block = tail_block;
            if wants_value {
                result_types.push(ty);
                ret_val = Some(val);
            }
        }

        let mut needs_yield = true;
        if let Some(syntax::Statement::Return(_)) = self.stmts.last() {
            needs_yield = false;
        }

        if needs_yield {
            let mut yield_builder = OperationBuilder::new("vx.yield", location);
            if let Some(v) = ret_val {
                yield_builder = yield_builder.add_operands(&[v]);
            }
            let yield_op = yield_builder
                .build()
                .expect("Failed to build vx.yield operation");
            body_block.append_operation(yield_op);
        }

        gen.in_spawn = prev_in_spawn;

        let topology_id = topology_to_i32(&self.top);
        let top_attr = IntegerAttribute::new(gen.i32_ty, topology_id as i64).into();

        // The topology's declared spelling, alongside its id.
        //
        // The id is a hash for a name declared in a machine file --
        // `1000 + fnv32(name) % 1000` -- so it is one-way and only 1000 wide. A plugin holding
        // `topo=1113` cannot recover `DecodeWorker`, so it cannot look the worker up in a fleet
        // manifest and discover where it lives, which is the whole of resolving a placement to a
        // machine (#348). Two names can also collide onto one id, and nothing would notice.
        //
        // Carrying the name makes it the identity and the id an optimisation. It costs a few bytes
        // per kernel and lets the compiler stay out of the business of knowing endpoints: the
        // program names a role, the machine file says what the role is, and the plugin maps the
        // name to an address.
        let name_attr =
            melior::ir::attribute::StringAttribute::new(gen.context, &self.top.display_name())
                .into();

        let mut spawn_builder = OperationBuilder::new("vx.spawn", location)
            .add_attributes(&[
                (Identifier::new(gen.context, "topology"), top_attr),
                (Identifier::new(gen.context, "topology_name"), name_attr),
            ])
            .add_regions([region]);

        // Topology → plugin selection: if a hardware plugin claims this topology, record its
        // identity on the op so the emitted IR reflects which backend owns the region (a
        // later lowering / the runtime dispatcher can route on it). This is the point where
        // the `VxHardwarePlugin` trait is actually consulted during compilation.
        if let Some(plugin) = crate::plugin::plugin_for(topology_id as u32) {
            let plugin_attr =
                melior::ir::attribute::StringAttribute::new(gen.context, &plugin.plugin_name())
                    .into();
            spawn_builder = spawn_builder
                .add_attributes(&[(Identifier::new(gen.context, "plugin"), plugin_attr)]);
        }

        if !result_types.is_empty() {
            spawn_builder = spawn_builder.add_results(&result_types);
        }

        let spawn_op = spawn_builder.build()?;
        let spawn_ref = block.append_operation(spawn_op);

        if !needs_yield {
            let mut ret_builder = OperationBuilder::new("func.return", location);
            if !result_types.is_empty() {
                ret_builder = ret_builder.add_operands(&[spawn_ref.result(0)?.into()]);
            }
            let ret_op = ret_builder.build()?;
            block.append_operation(ret_op);
        }

        if !result_types.is_empty() {
            Ok((spawn_ref.result(0)?.into(), result_types[0], block))
        } else {
            let _none_ty = gen.none_ty;
            let dummy_op = OperationBuilder::new("arith.constant", location)
                .add_attributes(&[(
                    Identifier::new(gen.context, "value"),
                    IntegerAttribute::new(Type::index(gen.context), 0).into(),
                )])
                .add_results(&[Type::index(gen.context)])
                .build()?;
            let dummy_ref = block.append_operation(dummy_op);
            Ok((dummy_ref.result(0)?.into(), Type::index(gen.context), block))
        }
    }
}

impl<'c> LowerToMelior<'c> for syntax::TransferExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let (src_val, src_ty, block) = gen.generate_expr(&self.expr, block)?;
        let location = gen.loc();

        // Static byte size of the transferred tile, for the SS2 sub-space bump allocator.
        let tile_bytes = gen.infer_ast_type(&self.expr).and_then(|ty| {
            let inner = match ty {
                syntax::Type::Pinned(b, _) | syntax::Type::Ref(b, _) => *b,
                other => other,
            };
            match inner {
                syntax::Type::Tensor(e, d, _) => crate::hir::memory::static_tensor_bytes(&e, &d),
                _ => None,
            }
        });

        // Map memory space to its canonical topology's dispatch id (single source of truth
        // in `arch`, adjacent to `topology_dispatch_id` so the two mappings stay in sync).
        let target_topology_id = crate::arch::memory_space_dispatch_id(&self.space);

        let top_attr = IntegerAttribute::new(gen.i32_ty, target_topology_id as i64).into();

        // Where the data is *now*, which is a different question from where it is going and one
        // only this stage can answer cheaply. A placed tensor's type is `Pinned(_, topology)`, so
        // the checker has already decided this; the backend would otherwise have to re-derive it
        // by walking use-def edges back to the transfer that did the placing, and that walk ends
        // at the first variable the value was bound to -- every `let` is an `alloca` and a store,
        // and the read is a load whose operand is the slot. The way home was compiled into a
        // `memcpy` from device memory for exactly that reason (#321, #348).
        let source_topology_id = match gen.infer_ast_type(&self.expr) {
            Some(syntax::Type::Pinned(_, top)) => crate::arch::topology_dispatch_id(&top),
            _ => 0,
        };

        let mut target_ty = src_ty;
        let src_ty_str = src_ty.to_string();
        if src_ty_str.starts_with("memref<") && src_ty_str.ends_with(">") {
            let inner_str = if src_ty_str.contains(", ") {
                src_ty_str[7..src_ty_str.rfind(", ").unwrap()].to_string()
            } else {
                src_ty_str[7..src_ty_str.len() - 1].to_string()
            };
            let target_ty_str = format!("memref<{}>", inner_str);
            target_ty = Type::parse(gen.context, &target_ty_str).unwrap_or(src_ty);
        }

        let mut transfer_builder = OperationBuilder::new("vx.transfer", location)
            .add_operands(&[src_val])
            .add_attributes(&[(Identifier::new(gen.context, "target_topology"), top_attr)]);

        if source_topology_id != 0 {
            let src_attr = IntegerAttribute::new(gen.i32_ty, source_topology_id as i64).into();
            transfer_builder = transfer_builder
                .add_attributes(&[(Identifier::new(gen.context, "source_topology"), src_attr)]);
        }

        // The bandwidth-derived roofline cost (set by sema when the memory hierarchy declares
        // bandwidths). Emitting it makes the paper's data-movement cost visible in the IR.
        // i64, not i32: a picosecond-resolution time cost overflows i32 at 4.3 ms, which any
        // multi-gigabyte host transfer exceeds.
        if let Some(cost) = self.cost {
            let cost_attr = IntegerAttribute::new(gen.i64_ty, cost as i64).into();
            transfer_builder = transfer_builder
                .add_attributes(&[(Identifier::new(gen.context, "cost"), cost_attr)]);
        }

        // Sub-space descriptor as IR metadata (SS1, subspace_scheduling.md): the target space name
        // plus, when declared, its containment/granule/capacity/scope. A later vx.* pass reads this
        // to schedule tiles into sub-spaces (TMEM/SMEM); it does not affect lowering.
        transfer_builder = transfer_builder.add_attributes(&[(
            Identifier::new(gen.context, "space"),
            StringAttribute::new(gen.context, &self.space.name()).into(),
        )]);
        let mut granule_bytes: Option<u64> = None;
        if let Some(decl) = gen.memories.get(&self.space) {
            if let Some(parent) = &decl.parent {
                transfer_builder = transfer_builder.add_attributes(&[(
                    Identifier::new(gen.context, "within"),
                    StringAttribute::new(gen.context, &parent.name()).into(),
                )]);
            }
            if let Some(g) = &decl.granule {
                granule_bytes = Some(g.0);
                transfer_builder = transfer_builder.add_attributes(&[(
                    Identifier::new(gen.context, "granule"),
                    IntegerAttribute::new(gen.i64_ty, g.0 as i64).into(),
                )]);
            }
            if let Some(c) = &decl.capacity {
                transfer_builder = transfer_builder.add_attributes(&[(
                    Identifier::new(gen.context, "capacity"),
                    IntegerAttribute::new(gen.i64_ty, c.0 as i64).into(),
                )]);
            }
            if let Some(scope) = &decl.scope {
                let scope_str = match scope {
                    syntax::Scope::Device => "device",
                    syntax::Scope::Sm => "sm",
                    syntax::Scope::Cta => "cta",
                    syntax::Scope::Thread => "thread",
                };
                transfer_builder = transfer_builder.add_attributes(&[(
                    Identifier::new(gen.context, "scope"),
                    StringAttribute::new(gen.context, scope_str).into(),
                )]);
            }

            // Whether the host can reach this space without being told to move
            // anything. `explicit` means it cannot: the bytes are somewhere only
            // an explicit transfer reaches, which on a discrete GPU is device
            // memory the CPU cannot load from at all.
            //
            // Every other field here is descriptive, and this one is not -- it
            // decides whether a kernel that fails to route may fall back to the
            // host. Without it in the IR, `vx.spawn` lowering could see that a
            // region was placed on a device and that its kernel was
            // unclassifiable, and still not know that running it here would
            // dereference device memory. That combination segmentation-faults
            // inside the outlined kernel on a real GPU (#251, #348), and the
            // fact needed to reject it at compile time was being dropped one
            // stage before the place that needed it.
            let managed_str = match decl.managed {
                syntax::Management::Explicit => "explicit",
                syntax::Management::Cached => "cached",
            };
            transfer_builder = transfer_builder.add_attributes(&[(
                Identifier::new(gen.context, "managed"),
                StringAttribute::new(gen.context, managed_str).into(),
            )]);
        }

        // SS2 — schedule the tile into the sub-space: a granule-rounded bump allocation. When the
        // target sub-space declares a `granule` and the tile size is statically known, assign the
        // next free `offset` (bytes) and `slots` (granule count) within the space and advance the
        // per-function cursor. Later passes / a device backend read these to place the tile.
        if let (Some(granule), Some(bytes)) = (granule_bytes, tile_bytes) {
            if granule > 0 {
                let rounded = bytes.div_ceil(granule) * granule;
                let offset = *gen.subspace_offsets.entry(self.space.clone()).or_insert(0);
                gen.subspace_offsets
                    .insert(self.space.clone(), offset + rounded);
                transfer_builder = transfer_builder.add_attributes(&[
                    (
                        Identifier::new(gen.context, "offset"),
                        IntegerAttribute::new(gen.i64_ty, offset as i64).into(),
                    ),
                    (
                        Identifier::new(gen.context, "slots"),
                        IntegerAttribute::new(gen.i64_ty, (rounded / granule) as i64).into(),
                    ),
                ]);
            }
        }

        let transfer_op = transfer_builder
            .add_results(&[target_ty])
            .build()
            .expect("Failed to build vx.transfer operation");

        use melior::ir::operation::OperationLike;
        let result_val = transfer_op.result(0)?.into();
        block.append_operation(transfer_op);

        Ok((result_val, target_ty, block))
    }
}

impl<'c> LowerToMelior<'c> for GradExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let GradExpr {
            target_fn,
            args,
            span: _,
        } = self;
        let (ret_ty, orig_arg_types) = gen
            .functions
            .get(target_fn)
            .cloned()
            .expect("Function not found");

        let fn_ty = melior::ir::r#type::FunctionType::new(gen.context, &orig_arg_types, &[ret_ty]);
        let const_op = OperationBuilder::new("func.constant", gen.loc())
            .add_attributes(&[(
                Identifier::new(gen.context, "value"),
                FlatSymbolRefAttribute::new(gen.context, target_fn).into(),
            )])
            .add_results(&[fn_ty.into()])
            .build()?;
        let const_ref = block.append_operation(const_op);
        let target_fn_val = const_ref.result(0)?.into();

        let mut arg_vals = vec![target_fn_val];
        let mut enzyme_arg_types = vec![fn_ty.into()];

        let mut current_b = block;
        for arg in args {
            let (v, ty, new_b) = gen.generate_expr(arg, current_b)?;
            arg_vals.push(v);
            enzyme_arg_types.push(ty);
            current_b = new_b;
        }

        let enzyme_name = emit_enzyme_decl(gen, "grad", target_fn, &enzyme_arg_types, ret_ty);

        let name_attr = FlatSymbolRefAttribute::new(gen.context, &enzyme_name);
        let call_op = OperationBuilder::new("func.call", gen.loc())
            .add_operands(&arg_vals)
            .add_results(&[ret_ty])
            .add_attributes(&[(Identifier::new(gen.context, "callee"), name_attr.into())])
            .build()?;

        let call_ref = current_b.append_operation(call_op);
        Ok((call_ref.result(0)?.into(), ret_ty, current_b))
    }
}

impl<'c> LowerToMelior<'c> for VjpExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let VjpExpr {
            target_fn,
            args,
            cotangent,
            span: _,
        } = self;
        let (ret_ty, orig_arg_types) = gen
            .functions
            .get(target_fn)
            .cloned()
            .expect("Function not found");

        let fn_ty = melior::ir::r#type::FunctionType::new(gen.context, &orig_arg_types, &[ret_ty]);
        let const_op = OperationBuilder::new("func.constant", gen.loc())
            .add_attributes(&[(
                Identifier::new(gen.context, "value"),
                FlatSymbolRefAttribute::new(gen.context, target_fn).into(),
            )])
            .add_results(&[fn_ty.into()])
            .build()?;
        let const_ref = block.append_operation(const_op);
        let target_fn_val = const_ref.result(0)?.into();

        let mut arg_vals = vec![target_fn_val];
        let mut enzyme_arg_types = vec![fn_ty.into()];

        let mut current_b = block;
        for arg in args {
            let (v, ty, new_b) = gen.generate_expr(arg, current_b)?;
            arg_vals.push(v);
            enzyme_arg_types.push(ty);
            current_b = new_b;
        }

        // For a scalar VJP in Enzyme, we just compute the gradient (implicitly seed=1.0)
        // and then multiply by the cotangent seed.
        let enzyme_name = emit_enzyme_decl(gen, "grad", target_fn, &enzyme_arg_types, ret_ty);

        let name_attr = FlatSymbolRefAttribute::new(gen.context, &enzyme_name);
        let call_op = OperationBuilder::new("func.call", gen.loc())
            .add_operands(&arg_vals)
            .add_results(&[ret_ty])
            .add_attributes(&[(Identifier::new(gen.context, "callee"), name_attr.into())])
            .build()?;

        let call_ref = current_b.append_operation(call_op);
        let grad_val = call_ref.result(0)?.into();

        let (c_val, _, current_b) = gen.generate_expr(cotangent, current_b)?;

        // Multiply grad by cotangent
        let is_float = ret_ty.to_string().contains("f32")
            || ret_ty.to_string().contains("f64")
            || ret_ty.to_string().contains("f16")
            || ret_ty.to_string().contains("bf16");
        let op_name = if is_float { "arith.mulf" } else { "arith.muli" };
        let mul_op = OperationBuilder::new(op_name, gen.loc())
            .add_operands(&[grad_val, c_val])
            .add_results(&[ret_ty])
            .build()?;
        let mul_ref = current_b.append_operation(mul_op);

        Ok((mul_ref.result(0)?.into(), ret_ty, current_b))
    }
}

impl<'c> LowerToMelior<'c> for JvpExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let JvpExpr {
            target_fn,
            args,
            tangent,
            span: _,
        } = self;
        let (ret_ty, orig_arg_types) = gen
            .functions
            .get(target_fn)
            .cloned()
            .expect("Function not found");

        let fn_ty = melior::ir::r#type::FunctionType::new(gen.context, &orig_arg_types, &[ret_ty]);
        let const_op = OperationBuilder::new("func.constant", gen.loc())
            .add_attributes(&[(
                Identifier::new(gen.context, "value"),
                FlatSymbolRefAttribute::new(gen.context, target_fn).into(),
            )])
            .add_results(&[fn_ty.into()])
            .build()?;
        let const_ref = block.append_operation(const_op);
        let target_fn_val = const_ref.result(0)?.into();

        let mut arg_vals = vec![target_fn_val];
        let mut enzyme_arg_types = vec![fn_ty.into()];

        let mut current_b = block;
        for arg in args {
            let (v, ty, new_b) = gen.generate_expr(arg, current_b)?;
            arg_vals.push(v);
            enzyme_arg_types.push(ty);
            current_b = new_b;
        }

        let (t_val, t_ty, current_b) = gen.generate_expr(tangent, current_b)?;
        arg_vals.push(t_val);
        enzyme_arg_types.push(t_ty);

        // Enzyme intercepts `__enzyme_fwddiff` for forward mode.
        let enzyme_name = emit_enzyme_decl(gen, "fwddiff", target_fn, &enzyme_arg_types, ret_ty);

        let name_attr = FlatSymbolRefAttribute::new(gen.context, &enzyme_name);
        let call_op = OperationBuilder::new("func.call", gen.loc())
            .add_operands(&arg_vals)
            .add_results(&[ret_ty])
            .add_attributes(&[(Identifier::new(gen.context, "callee"), name_attr.into())])
            .build()?;

        let call_ref = current_b.append_operation(call_op);
        Ok((call_ref.result(0)?.into(), ret_ty, current_b))
    }
}
