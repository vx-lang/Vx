use super::*;
use crate::syntax;
use crate::syntax::*;
use melior::ir::{
    attribute::{FlatSymbolRefAttribute, IntegerAttribute},
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
        for stmt in &self.stmts {
            if let Some(b) = gen.generate_statement(stmt, body_block)? {
                body_block = b;
            }
        }

        let mut result_types = vec![];
        let mut ret_val = None;

        if let Some(r) = &self.ret {
            let (val, ty, _block) = gen.generate_expr(r, body_block)?;
            result_types.push(ty);
            ret_val = Some(val);
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

        let mut spawn_builder = OperationBuilder::new("vx.spawn", location)
            .add_attributes(&[(Identifier::new(gen.context, "topology"), top_attr)])
            .add_regions([region]);

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

        // Map memory space to topology target.
        let target_topology_id = match self.space {
            syntax::MemorySpace::CPUDRAM => 0,
            syntax::MemorySpace::NPUHBM => 100,
            syntax::MemorySpace::LocalSRAM => 200,
            syntax::MemorySpace::NicRam | syntax::MemorySpace::RemoteHbm => 300,
        };

        let top_attr = IntegerAttribute::new(gen.i32_ty, target_topology_id as i64).into();

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

        let transfer_op = OperationBuilder::new("vx.transfer", location)
            .add_operands(&[src_val])
            .add_attributes(&[(Identifier::new(gen.context, "target_topology"), top_attr)])
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
