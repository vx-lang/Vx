use super::*;
use crate::syntax;
use crate::syntax::*;
use melior::ir::{
    attribute::{FloatAttribute, IntegerAttribute},
    operation::OperationBuilder,
    Identifier, Type, Value,
};

impl<'c> LowerToMelior<'c> for IfExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let IfExpr {
            is_comptime,
            cond,
            then_block,
            else_block: else_block_opt,
            span: _,
        } = self;

        if *is_comptime {
            let mut last_val = None;
            let target_block = if !then_block.is_empty() {
                Some(then_block)
            } else {
                else_block_opt.as_ref()
            };

            if let Some(tb) = target_block {
                for (i, stmt) in tb.iter().enumerate() {
                    let is_last = i == tb.len() - 1;
                    if is_last {
                        if let syntax::Statement::ExprStmt(syntax::stmt::ExprStmtStmt {
                            expr,
                            has_semi,
                            ..
                        }) = stmt
                        {
                            let (val, ty, _block) = gen.generate_expr(expr, block)?;
                            if !has_semi {
                                last_val = Some((val, ty));
                            }
                        } else {
                            gen.generate_statement(stmt, block)?;
                        }
                    } else {
                        gen.generate_statement(stmt, block)?;
                    }
                }
            }

            if let Some((val, ty)) = last_val {
                return Ok((val, ty, block));
            }

            let ret_ty = gen.expected_type.unwrap_or(gen.f32_ty);
            let dummy_op = if ret_ty.to_string() == "f32" {
                OperationBuilder::new("arith.constant", gen.loc())
                    .add_attributes(&[(
                        Identifier::new(gen.context, "value"),
                        FloatAttribute::new(gen.context, ret_ty, 0.0).into(),
                    )])
                    .add_results(&[ret_ty])
                    .build()
                    .unwrap()
            } else if ret_ty.to_string() == "i1" {
                OperationBuilder::new("arith.constant", gen.loc())
                    .add_attributes(&[(
                        Identifier::new(gen.context, "value"),
                        IntegerAttribute::new(ret_ty, 0).into(),
                    )])
                    .add_results(&[ret_ty])
                    .build()
                    .unwrap()
            } else {
                OperationBuilder::new("arith.constant", gen.loc())
                    .add_attributes(&[(
                        Identifier::new(gen.context, "value"),
                        IntegerAttribute::new(ret_ty, 0).into(),
                    )])
                    .add_results(&[ret_ty])
                    .build()
                    .unwrap()
            };
            return Ok((
                block.append_operation(dummy_op).result(0).unwrap().into(),
                ret_ty,
                block,
            ));
        }

        let (cond_val, _, block) = gen.generate_expr(cond, block)?;

        let ret_ty = gen.expected_type.unwrap_or(gen.f32_ty);
        let parent_region = block.parent_region().unwrap();
        let mut then_b = parent_region.append_block(melior::ir::Block::new(&[]));
        let mut else_b = parent_region.append_block(melior::ir::Block::new(&[]));

        let has_ret = ret_ty.to_string() != "none" && ret_ty.to_string() != "void";
        let merge_b = if has_ret {
            parent_region.append_block(melior::ir::Block::new(&[(ret_ty, gen.loc())]))
        } else {
            parent_region.append_block(melior::ir::Block::new(&[]))
        };

        block.append_operation(
            OperationBuilder::new("cf.cond_br", gen.loc())
                .add_operands(&[cond_val])
                .add_successors(&[&*then_b, &*else_b])
                .add_attributes(&[(
                    Identifier::new(gen.context, "operandSegmentSizes"),
                    melior::ir::attribute::DenseI32ArrayAttribute::new(gen.context, &[1, 0, 0])
                        .into(),
                )])
                .build()
                .unwrap(),
        );

        let mut then_terminated = false;
        let mut then_val = None;
        for (i, stmt) in then_block.iter().enumerate() {
            let is_last = i == then_block.len() - 1;
            if is_last && has_ret {
                if let syntax::Statement::ExprStmt(syntax::stmt::ExprStmtStmt {
                    expr,
                    has_semi: false,
                    ..
                }) = stmt
                {
                    let (val, _, b) = gen.generate_expr(expr, then_b)?;
                    then_b = b;
                    then_val = Some(val);
                } else {
                    if let Some(b) = gen.generate_statement(stmt, then_b)? {
                        then_b = b;
                    } else {
                        then_terminated = true;
                        break;
                    }
                }
            } else {
                if let Some(b) = gen.generate_statement(stmt, then_b)? {
                    then_b = b;
                } else {
                    then_terminated = true;
                    break;
                }
            }
        }

        if !then_terminated {
            let mut yield_operands = vec![];
            if has_ret {
                if let Some(val) = then_val {
                    yield_operands.push(val);
                } else {
                    let dummy_op = OperationBuilder::new("arith.constant", gen.loc())
                        .add_attributes(&[(
                            Identifier::new(gen.context, "value"),
                            melior::ir::attribute::IntegerAttribute::new(ret_ty, 0).into(),
                        )])
                        .build()
                        .unwrap();
                    let val = then_b.append_operation(dummy_op).result(0).unwrap().into();
                    yield_operands.push(val);
                }
            }
            then_b.append_operation(
                OperationBuilder::new("cf.br", gen.loc())
                    .add_operands(&yield_operands)
                    .add_successors(&[&*merge_b])
                    .build()
                    .unwrap(),
            );
        }

        let mut else_terminated = false;
        let mut else_val = None;
        if let Some(else_block) = else_block_opt {
            for (i, stmt) in else_block.iter().enumerate() {
                let is_last = i == else_block.len() - 1;
                if is_last && has_ret {
                    if let syntax::Statement::ExprStmt(syntax::stmt::ExprStmtStmt {
                        expr,
                        has_semi: false,
                        ..
                    }) = stmt
                    {
                        let (val, _, b) = gen.generate_expr(expr, else_b)?;
                        else_b = b;
                        else_val = Some(val);
                    } else {
                        if let Some(b) = gen.generate_statement(stmt, else_b)? {
                            else_b = b;
                        } else {
                            else_terminated = true;
                            break;
                        }
                    }
                } else {
                    if let Some(b) = gen.generate_statement(stmt, else_b)? {
                        else_b = b;
                    } else {
                        else_terminated = true;
                        break;
                    }
                }
            }
        }
        if !else_terminated {
            let mut yield_operands = vec![];
            if has_ret {
                if let Some(val) = else_val {
                    yield_operands.push(val);
                } else {
                    let dummy_op = OperationBuilder::new("arith.constant", gen.loc())
                        .add_attributes(&[(
                            Identifier::new(gen.context, "value"),
                            melior::ir::attribute::IntegerAttribute::new(ret_ty, 0).into(),
                        )])
                        .build()
                        .unwrap();
                    let val = else_b.append_operation(dummy_op).result(0).unwrap().into();
                    yield_operands.push(val);
                }
            }
            else_b.append_operation(
                OperationBuilder::new("cf.br", gen.loc())
                    .add_operands(&yield_operands)
                    .add_successors(&[&*merge_b])
                    .build()
                    .unwrap(),
            );
        }

        if has_ret {
            let res = merge_b.argument(0).unwrap().into();
            Ok((res, ret_ty, merge_b))
        } else {
            Ok((cond_val, ret_ty, merge_b))
        }
    }
}

impl<'c> LowerToMelior<'c> for ForLoopStmt {
    type Output = Result<Option<melior::ir::BlockRef<'c, 'c>>, LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let ForLoopStmt {
            iter,
            iterable,
            invariants: _,
            body,
            span: _,
        } = self;

        if let Expr::Range(syntax::expr::RangeExpr {
            start,
            end,
            span: _,
        }) = &**iterable
        {
            let (start_val, start_ty, block) = gen.generate_expr(start, block)?;
            let (end_val, end_ty, block) = gen.generate_expr(end, block)?;

            let ty_index = gen.index_ty;

            let start_idx = if start_ty == ty_index {
                start_val
            } else {
                let cast_start_op = OperationBuilder::new("arith.index_cast", gen.loc())
                    .add_operands(&[start_val])
                    .add_results(&[ty_index])
                    .build()
                    .unwrap();
                block
                    .append_operation(cast_start_op)
                    .result(0)
                    .unwrap()
                    .into()
            };

            let end_idx = if end_ty == ty_index {
                end_val
            } else {
                let cast_end_op = OperationBuilder::new("arith.index_cast", gen.loc())
                    .add_operands(&[end_val])
                    .add_results(&[ty_index])
                    .build()
                    .unwrap();
                block
                    .append_operation(cast_end_op)
                    .result(0)
                    .unwrap()
                    .into()
            };

            let step_op = block.append_operation(
                OperationBuilder::new("arith.constant", gen.loc())
                    .add_results(&[ty_index])
                    .add_attributes(&[(
                        Identifier::new(gen.context, "value"),
                        IntegerAttribute::new(ty_index, 1).into(),
                    )])
                    .build()
                    .unwrap(),
            );
            let step_idx = step_op.result(0).unwrap().into();

            let parent_region = block.parent_region().unwrap();
            let cond_block =
                parent_region.append_block(melior::ir::Block::new(&[(ty_index, gen.loc())]));
            let mut body_block = parent_region.append_block(melior::ir::Block::new(&[]));
            let merge_block = parent_region.append_block(melior::ir::Block::new(&[]));

            block.append_operation(
                OperationBuilder::new("cf.br", gen.loc())
                    .add_operands(&[start_idx])
                    .add_successors(&[&*cond_block])
                    .build()
                    .unwrap(),
            );

            let current_idx = cond_block.argument(0).unwrap().into();

            let cmp_op = cond_block.append_operation(
                OperationBuilder::new("arith.cmpi", gen.loc())
                    .add_operands(&[current_idx, end_idx])
                    .add_results(&[gen.i1_ty])
                    .add_attributes(&[(
                        Identifier::new(gen.context, "predicate"),
                        IntegerAttribute::new(gen.i64_ty, 2).into(), // slt
                    )])
                    .build()
                    .unwrap(),
            );
            let cond_val = cmp_op.result(0).unwrap().into();

            cond_block.append_operation(
                OperationBuilder::new("cf.cond_br", gen.loc())
                    .add_operands(&[cond_val])
                    .add_successors(&[&*body_block, &*merge_block])
                    .add_attributes(&[(
                        Identifier::new(gen.context, "operandSegmentSizes"),
                        melior::ir::attribute::DenseI32ArrayAttribute::new(gen.context, &[1, 0, 0])
                            .into(),
                    )])
                    .build()
                    .unwrap(),
            );

            gen.env.insert(iter.clone().into(), (current_idx, ty_index));
            // `continue` must still advance the loop index, so it targets a
            // dedicated latch block rather than branching to the condition
            // block directly (which would skip the increment and, since the
            // condition block takes the index as an argument, emit invalid IR).
            let latch_block = parent_region.append_block(melior::ir::Block::new(&[]));
            gen.break_blocks.push(&*merge_block as *const _);
            gen.continue_blocks.push(&*latch_block as *const _);

            let mut body_terminated = false;
            for stmt in body {
                if let Some(b) = gen.generate_statement(stmt, body_block)? {
                    body_block = b;
                } else {
                    body_terminated = true;
                    break;
                }
            }

            gen.break_blocks.pop();
            gen.continue_blocks.pop();

            if !body_terminated {
                body_block.append_operation(
                    OperationBuilder::new("cf.br", gen.loc())
                        .add_successors(&[&*latch_block])
                        .build()
                        .unwrap(),
                );
            }

            // Latch: increment the index and branch back to the condition.
            // Both normal fall-through and `continue` route through here.
            let next_idx_op = latch_block.append_operation(
                OperationBuilder::new("arith.addi", gen.loc())
                    .add_operands(&[current_idx, step_idx])
                    .add_results(&[ty_index])
                    .build()
                    .unwrap(),
            );
            let next_idx = next_idx_op.result(0).unwrap().into();
            latch_block.append_operation(
                OperationBuilder::new("cf.br", gen.loc())
                    .add_operands(&[next_idx])
                    .add_successors(&[&*cond_block])
                    .build()
                    .unwrap(),
            );

            return Ok(Some(merge_block));
        }

        let (iter_val, iter_ty, block) = gen.generate_expr(iterable, block)?;
        let iter_ty_str = iter_ty.to_string();

        if iter_ty_str.starts_with("tensor<") {
            let ty_index = gen.index_ty;
            let start_idx = block
                .append_operation(
                    OperationBuilder::new("arith.constant", gen.loc())
                        .add_results(&[ty_index])
                        .add_attributes(&[(
                            Identifier::new(gen.context, "value"),
                            IntegerAttribute::new(ty_index, 0).into(),
                        )])
                        .build()
                        .unwrap(),
                )
                .result(0)
                .unwrap()
                .into();

            let len_str = iter_ty_str
                .split('x')
                .next()
                .unwrap()
                .split('<')
                .nth(1)
                .unwrap();
            let len: i64 = len_str.parse().unwrap();

            let end_idx = block
                .append_operation(
                    OperationBuilder::new("arith.constant", gen.loc())
                        .add_results(&[ty_index])
                        .add_attributes(&[(
                            Identifier::new(gen.context, "value"),
                            IntegerAttribute::new(ty_index, len).into(),
                        )])
                        .build()
                        .unwrap(),
                )
                .result(0)
                .unwrap()
                .into();

            let step_idx = block
                .append_operation(
                    OperationBuilder::new("arith.constant", gen.loc())
                        .add_results(&[ty_index])
                        .add_attributes(&[(
                            Identifier::new(gen.context, "value"),
                            IntegerAttribute::new(ty_index, 1).into(),
                        )])
                        .build()
                        .unwrap(),
                )
                .result(0)
                .unwrap()
                .into();

            let parent_region = block.parent_region().unwrap();
            let cond_block =
                parent_region.append_block(melior::ir::Block::new(&[(ty_index, gen.loc())]));
            let mut body_block = parent_region.append_block(melior::ir::Block::new(&[]));
            let merge_block = parent_region.append_block(melior::ir::Block::new(&[]));

            block.append_operation(
                OperationBuilder::new("cf.br", gen.loc())
                    .add_operands(&[start_idx])
                    .add_successors(&[&*cond_block])
                    .build()
                    .unwrap(),
            );

            let current_idx = cond_block.argument(0).unwrap().into();

            let cmp_op = cond_block.append_operation(
                OperationBuilder::new("arith.cmpi", gen.loc())
                    .add_operands(&[current_idx, end_idx])
                    .add_results(&[gen.i1_ty])
                    .add_attributes(&[(
                        Identifier::new(gen.context, "predicate"),
                        IntegerAttribute::new(gen.i64_ty, 2).into(), // slt
                    )])
                    .build()
                    .unwrap(),
            );
            let cond_val = cmp_op.result(0).unwrap().into();

            cond_block.append_operation(
                OperationBuilder::new("cf.cond_br", gen.loc())
                    .add_operands(&[cond_val])
                    .add_successors(&[&*body_block, &*merge_block])
                    .add_attributes(&[(
                        Identifier::new(gen.context, "operandSegmentSizes"),
                        melior::ir::attribute::DenseI32ArrayAttribute::new(gen.context, &[1, 0, 0])
                            .into(),
                    )])
                    .build()
                    .unwrap(),
            );

            let el_ty_str = iter_ty_str.split('x').nth(1).unwrap().trim_end_matches('>');
            let el_ty = Type::parse(gen.context, el_ty_str).unwrap();

            let extract_op = OperationBuilder::new("tensor.extract", gen.loc())
                .add_operands(&[iter_val, current_idx])
                .add_results(&[el_ty])
                .build()
                .unwrap();

            let el_val = body_block
                .append_operation(extract_op)
                .result(0)
                .unwrap()
                .into();

            gen.env.insert(iter.clone().into(), (el_val, el_ty));
            // `continue` must still advance the loop index, so it targets a
            // dedicated latch block rather than branching to the condition
            // block directly (which would skip the increment and, since the
            // condition block takes the index as an argument, emit invalid IR).
            let latch_block = parent_region.append_block(melior::ir::Block::new(&[]));
            gen.break_blocks.push(&*merge_block as *const _);
            gen.continue_blocks.push(&*latch_block as *const _);

            let mut body_terminated = false;
            for stmt in body {
                if let Some(b) = gen.generate_statement(stmt, body_block)? {
                    body_block = b;
                } else {
                    body_terminated = true;
                    break;
                }
            }

            gen.break_blocks.pop();
            gen.continue_blocks.pop();

            if !body_terminated {
                body_block.append_operation(
                    OperationBuilder::new("cf.br", gen.loc())
                        .add_successors(&[&*latch_block])
                        .build()
                        .unwrap(),
                );
            }

            // Latch: increment the index and branch back to the condition.
            // Both normal fall-through and `continue` route through here.
            let next_idx_op = latch_block.append_operation(
                OperationBuilder::new("arith.addi", gen.loc())
                    .add_operands(&[current_idx, step_idx])
                    .add_results(&[ty_index])
                    .build()
                    .unwrap(),
            );
            let next_idx = next_idx_op.result(0).unwrap().into();
            latch_block.append_operation(
                OperationBuilder::new("cf.br", gen.loc())
                    .add_operands(&[next_idx])
                    .add_successors(&[&*cond_block])
                    .build()
                    .unwrap(),
            );

            return Ok(Some(merge_block));
        }

        // Generic Iterator loop via cf
        let ptr_ty = gen.ptr_ty;
        let mut actual_next_name = "next".to_string();
        for (name, (_, _args)) in &gen.functions {
            if name.contains("_next_") || name.ends_with("_next") {
                actual_next_name = name.to_string();
                break;
            }
        }

        let i32_ty = gen.i32_ty;
        let c1_op_alloc = block.append_operation(
            OperationBuilder::new("llvm.mlir.constant", gen.loc())
                .add_results(&[i32_ty])
                .add_attributes(&[(
                    Identifier::new(gen.context, "value"),
                    IntegerAttribute::new(i32_ty, 1).into(),
                )])
                .build()
                .unwrap(),
        );
        let c1_val_alloc = c1_op_alloc.result(0).unwrap().into();

        let alloca_op = block.append_operation(
            OperationBuilder::new("llvm.alloca", gen.loc())
                .add_operands(&[c1_val_alloc])
                .add_results(&[ptr_ty])
                .add_attributes(&[(
                    Identifier::new(gen.context, "elem_type"),
                    melior::ir::attribute::TypeAttribute::new(iter_ty).into(),
                )])
                .build()
                .unwrap(),
        );
        let ptr_val = alloca_op.result(0).unwrap().into();

        block.append_operation(
            OperationBuilder::new("llvm.store", gen.loc())
                .add_operands(&[iter_val, ptr_val])
                .build()
                .unwrap(),
        );

        let tmp_iter_name = format!("__iter_ptr_{}", gen.string_counter);
        gen.string_counter += 1;
        gen.env
            .insert(tmp_iter_name.clone().into(), (ptr_val, iter_ty));
        gen.allocs.insert(tmp_iter_name.clone());

        let parent_region = block.parent_region().unwrap();
        let cond_block = parent_region.append_block(melior::ir::Block::new(&[]));
        let mut body_block = parent_region.append_block(melior::ir::Block::new(&[]));
        let merge_block = parent_region.append_block(melior::ir::Block::new(&[]));

        block.append_operation(
            OperationBuilder::new("cf.br", gen.loc())
                .add_successors(&[&*cond_block])
                .build()
                .unwrap(),
        );

        let next_call = Expr::FunctionCall(FunctionCallExpr {
            name: actual_next_name.into(),
            type_args: None,
            args: vec![Expr::Borrow(BorrowExpr {
                expr: Box::new(Expr::Identifier(IdentifierExpr {
                    name: tmp_iter_name.clone().into(),
                    span: Span::default(),
                })),
                is_mut: true,
                span: Span::default(),
            })],
            span: Span::default(),
        });

        let (opt_val, opt_ty, cond_block_end) = gen.generate_expr(&next_call, cond_block)?;

        let extract_tag_op = OperationBuilder::new("llvm.extractvalue", gen.loc())
            .add_operands(&[opt_val])
            .add_results(&[i32_ty])
            .add_attributes(&[(
                Identifier::new(gen.context, "position"),
                melior::ir::attribute::DenseI64ArrayAttribute::new(gen.context, &[0]).into(),
            )])
            .build()
            .unwrap();
        let tag_val = cond_block_end
            .append_operation(extract_tag_op)
            .result(0)
            .unwrap()
            .into();

        let c1_op = cond_block_end.append_operation(
            OperationBuilder::new("arith.constant", gen.loc())
                .add_results(&[i32_ty])
                .add_attributes(&[(
                    Identifier::new(gen.context, "value"),
                    IntegerAttribute::new(i32_ty, 0).into(),
                )])
                .build()
                .unwrap(),
        );
        let c1_val = c1_op.result(0).unwrap().into();

        let cmpi_op = cond_block_end.append_operation(
            OperationBuilder::new("arith.cmpi", gen.loc())
                .add_operands(&[tag_val, c1_val])
                .add_results(&[gen.i1_ty])
                .add_attributes(&[(
                    Identifier::new(gen.context, "predicate"),
                    IntegerAttribute::new(gen.i64_ty, 0).into(), // eq
                )])
                .build()
                .unwrap(),
        );
        let cond_val = cmpi_op.result(0).unwrap().into();

        cond_block_end.append_operation(
            OperationBuilder::new("cf.cond_br", gen.loc())
                .add_operands(&[cond_val])
                .add_successors(&[&*body_block, &*merge_block])
                .add_attributes(&[(
                    Identifier::new(gen.context, "operandSegmentSizes"),
                    melior::ir::attribute::DenseI32ArrayAttribute::new(gen.context, &[1, 0, 0])
                        .into(),
                )])
                .build()
                .unwrap(),
        );

        let opt_ty_str = opt_ty.to_string();
        let payload_ty_str = if opt_ty_str.contains("(i32, ") {
            let start = opt_ty_str.find("(i32, ").unwrap() + 6;
            let end = opt_ty_str.rfind(')').unwrap();
            opt_ty_str[start..end].to_string()
        } else {
            "i32".to_string()
        };
        let payload_ty = Type::parse(gen.context, &payload_ty_str).unwrap();

        let extract_payload_op = OperationBuilder::new("llvm.extractvalue", gen.loc())
            .add_operands(&[opt_val])
            .add_results(&[payload_ty])
            .add_attributes(&[(
                Identifier::new(gen.context, "position"),
                melior::ir::attribute::DenseI64ArrayAttribute::new(gen.context, &[1]).into(),
            )])
            .build()
            .unwrap();
        let payload_val = body_block
            .append_operation(extract_payload_op)
            .result(0)
            .unwrap()
            .into();

        gen.env
            .insert(iter.clone().into(), (payload_val, payload_ty));
        gen.break_blocks.push(&*merge_block as *const _);
        gen.continue_blocks.push(&*cond_block as *const _);

        let mut body_terminated = false;
        for stmt in body {
            if let Some(b) = gen.generate_statement(stmt, body_block)? {
                body_block = b;
            } else {
                body_terminated = true;
                break;
            }
        }

        gen.break_blocks.pop();
        gen.continue_blocks.pop();

        if !body_terminated {
            body_block.append_operation(
                OperationBuilder::new("cf.br", gen.loc())
                    .add_successors(&[&*cond_block])
                    .build()
                    .unwrap(),
            );
        }

        Ok(Some(merge_block))
    }
}

impl<'c> LowerToMelior<'c> for LoopStmt {
    type Output = Result<Option<melior::ir::BlockRef<'c, 'c>>, LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let LoopStmt {
            invariants: _,
            body,
            span: _,
        } = self;

        let parent_region = block.parent_region().unwrap();
        let mut body_block = parent_region.append_block(melior::ir::Block::new(&[]));
        let loop_entry_block_ref = body_block; // separate variable
        let merge_block = parent_region.append_block(melior::ir::Block::new(&[]));

        let loop_entry_block = &*loop_entry_block_ref as *const melior::ir::Block<'c>;

        block.append_operation(
            OperationBuilder::new("cf.br", gen.loc())
                .add_successors(&[&*body_block])
                .build()
                .unwrap(),
        );

        gen.break_blocks.push(&*merge_block as *const _);
        gen.continue_blocks.push(loop_entry_block);

        let mut body_terminated = false;
        for stmt in body {
            if let Some(b) = gen.generate_statement(stmt, body_block)? {
                body_block = b;
            } else {
                body_terminated = true;
                break;
            }
        }

        gen.break_blocks.pop();
        gen.continue_blocks.pop();

        if !body_terminated {
            let entry_b: &melior::ir::Block<'c> = unsafe { &*loop_entry_block };
            body_block.append_operation(
                OperationBuilder::new("cf.br", gen.loc())
                    .add_successors(&[entry_b])
                    .build()
                    .unwrap(),
            );
        }

        Ok(Some(merge_block))
    }
}

impl<'c> LowerToMelior<'c> for BreakStmt {
    type Output = Result<Option<melior::ir::BlockRef<'c, 'c>>, LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        if let Some(&break_ptr) = gen.break_blocks.last() {
            let break_block: &melior::ir::Block<'c> = unsafe { &*break_ptr };
            block.append_operation(
                OperationBuilder::new("cf.br", gen.loc())
                    .add_successors(&[break_block])
                    .build()
                    .unwrap(),
            );
        } else {
            panic!("break outside of a loop");
        }
        Ok(None)
    }
}

impl<'c> LowerToMelior<'c> for ContinueStmt {
    type Output = Result<Option<melior::ir::BlockRef<'c, 'c>>, LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        if let Some(&continue_ptr) = gen.continue_blocks.last() {
            let continue_block: &melior::ir::Block<'c> = unsafe { &*continue_ptr };
            block.append_operation(
                OperationBuilder::new("cf.br", gen.loc())
                    .add_successors(&[continue_block])
                    .build()
                    .unwrap(),
            );
        } else {
            panic!("continue outside of a loop");
        }
        Ok(None)
    }
}
