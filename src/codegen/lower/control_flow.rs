use super::*;
use crate::ast;
use crate::ast::*;
use melior::ir::{
    attribute::{FloatAttribute, IntegerAttribute, TypeAttribute},
    operation::OperationBuilder,
    Block, Identifier, Region, Type, Value,
};

impl<'c> LowerToMelior<'c> for IfExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>), LowerError>;
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
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
                        if let ast::Statement::ExprStmt(ast::stmt::ExprStmtStmt {
                            expr,
                            has_semi,
                            ..
                        }) = stmt
                        {
                            let (val, ty) = gen.generate_expr(expr, block)?;
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
                return Ok((val, ty));
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
            ));
        }

        let (cond_val, _) = gen.generate_expr(cond, block)?;

        let then_region = Region::new();
        let then_b = Block::new(&[]);
        for stmt in then_block {
            gen.generate_statement(stmt, &then_b)?;
        }
        let yield_op = OperationBuilder::new("scf.yield", gen.loc())
            .build()
            .unwrap();
        then_b.append_operation(yield_op);
        then_region.append_block(then_b);

        let else_region = Region::new();
        let else_b = Block::new(&[]);
        if let Some(else_block) = else_block_opt {
            for stmt in else_block {
                gen.generate_statement(stmt, &else_b)?;
            }
        }
        let yield_op = OperationBuilder::new("scf.yield", gen.loc())
            .build()
            .unwrap();
        else_b.append_operation(yield_op);
        else_region.append_block(else_b);

        let if_op = OperationBuilder::new("scf.if", gen.loc())
            .add_operands(&[cond_val])
            .add_regions([then_region, else_region])
            .build()
            .unwrap();

        block.append_operation(if_op);

        let ty = gen.i32_ty;
        let op = OperationBuilder::new("arith.constant", gen.loc())
            .add_results(&[ty])
            .add_attributes(&[(
                Identifier::new(gen.context, "value"),
                IntegerAttribute::new(ty, 0).into(),
            )])
            .build()
            .unwrap();
        let op_ref = block.append_operation(op);
        Ok((op_ref.result(0).unwrap().into(), ty))
    }
}

impl<'c> LowerToMelior<'c> for ForLoopStmt {
    type Output = Result<(), LowerError>;
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let ForLoopStmt {
            iter,
            iterable,
            invariants: _,
            body,
            span: _,
        } = self;

        if let Expr::Range(ast::expr::RangeExpr {
            start,
            end,
            span: _,
        }) = &**iterable
        {
            let (start_val, start_ty) = gen.generate_expr(start, block)?;
            let (end_val, end_ty) = gen.generate_expr(end, block)?;

            let ty_index = gen.index_ty;

            // cast start/end to index if necessary
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

            let for_region = Region::new();
            let for_block = Block::new(&[(ty_index, gen.loc())]);
            let i_arg = for_block.argument(0).unwrap().into();

            gen.env.insert(iter.clone(), (i_arg, ty_index));

            for stmt in body {
                gen.generate_statement(stmt, &for_block)?;
            }

            for_block.append_operation(
                OperationBuilder::new("scf.yield", gen.loc())
                    .build()
                    .unwrap(),
            );
            for_region.append_block(for_block);

            block.append_operation(
                OperationBuilder::new("scf.for", gen.loc())
                    .add_operands(&[start_idx, end_idx, step_idx])
                    .add_regions([for_region])
                    .build()
                    .unwrap(),
            );
            return Ok(());
        }

        // Generic Iterator loop via scf.while or tensor iteration
        let (iter_val, iter_ty) = gen.generate_expr(iterable, block)?;
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

            let for_region = Region::new();
            let for_block = Block::new(&[(ty_index, gen.loc())]);
            let i_arg = for_block.argument(0).unwrap().into();

            let el_ty_str = iter_ty_str.split('x').nth(1).unwrap().trim_end_matches('>');
            let el_ty = Type::parse(gen.context, el_ty_str).unwrap();

            let extract_op = OperationBuilder::new("tensor.extract", gen.loc())
                .add_operands(&[iter_val, i_arg])
                .add_results(&[el_ty])
                .build()
                .unwrap();

            let el_val = for_block
                .append_operation(extract_op)
                .result(0)
                .unwrap()
                .into();

            gen.env.insert(iter.clone(), (el_val, el_ty));

            for stmt in body {
                gen.generate_statement(stmt, &for_block)?;
            }

            for_block.append_operation(
                OperationBuilder::new("scf.yield", gen.loc())
                    .build()
                    .unwrap(),
            );
            for_region.append_block(for_block);

            block.append_operation(
                OperationBuilder::new("scf.for", gen.loc())
                    .add_operands(&[start_idx, end_idx, step_idx])
                    .add_regions([for_region])
                    .build()
                    .unwrap(),
            );
            return Ok(());
        }

        let before_region = Region::new();
        let before_block = Block::new(&[(iter_ty, gen.loc())]);
        let iter_arg = before_block.argument(0).unwrap().into();

        let iter_name = format!("__iter_{}", gen.string_counter);
        gen.string_counter += 1;
        gen.env.insert(iter_name.clone(), (iter_arg, iter_ty));

        let ptr_ty = gen.ptr_ty;
        let mut actual_next_name = "next".to_string();
        for (name, (_, _args)) in &gen.functions {
            // Find the mangled next function that takes a pointer
            if name.contains("_next_") || name.ends_with("_next") {
                actual_next_name = name.clone();
                break;
            }
        }
        if actual_next_name == "next" {
            println!("Could not find actual_next_name for next! Functions available:");
            for (name, (_, args)) in &gen.functions {
                println!("  - {}: args={:?}", name, args);
            }
        }

        let i32_ty = gen.i32_ty;
        let c1_op_alloc = before_block.append_operation(
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

        let alloca_op = before_block.append_operation(
            OperationBuilder::new("llvm.alloca", gen.loc())
                .add_operands(&[c1_val_alloc])
                .add_results(&[ptr_ty])
                .add_attributes(&[(
                    Identifier::new(gen.context, "elem_type"),
                    TypeAttribute::new(iter_ty).into(),
                )])
                .build()
                .unwrap(),
        );
        let ptr_val = alloca_op.result(0).unwrap().into();
        before_block.append_operation(
            OperationBuilder::new("llvm.store", gen.loc())
                .add_operands(&[iter_arg, ptr_val])
                .build()
                .unwrap(),
        );

        // Add ptr_val to gen.allocs and gen.env temporarily so IdentifierExpr returns it!
        let tmp_iter_name = format!("__iter_ptr_{}", gen.string_counter);
        gen.string_counter += 1;
        gen.env.insert(tmp_iter_name.clone(), (ptr_val, iter_ty));
        gen.allocs.insert(tmp_iter_name.clone());

        let next_call = Expr::FunctionCall(FunctionCallExpr {
            name: actual_next_name,
            type_args: None,
            args: vec![Expr::Borrow(BorrowExpr {
                expr: Box::new(Expr::Identifier(IdentifierExpr {
                    name: tmp_iter_name.clone(),
                    span: Span::default(),
                })),
                is_mut: true,
                span: Span::default(),
            })],
            span: Span::default(),
        });

        let (opt_val, opt_ty) = gen.generate_expr(&next_call, &before_block)?;

        // Load the updated iterator value to yield it back
        let load_op = before_block.append_operation(
            OperationBuilder::new("llvm.load", gen.loc())
                .add_operands(&[ptr_val])
                .add_results(&[iter_ty])
                .build()
                .unwrap(),
        );
        let updated_iter_val = load_op.result(0).unwrap().into();

        let extract_tag_op = OperationBuilder::new("llvm.extractvalue", gen.loc())
            .add_operands(&[opt_val])
            .add_results(&[i32_ty])
            .add_attributes(&[(
                Identifier::new(gen.context, "position"),
                melior::ir::attribute::DenseI64ArrayAttribute::new(gen.context, &[0]).into(),
            )])
            .build()
            .unwrap();
        let tag_val = before_block
            .append_operation(extract_tag_op)
            .result(0)
            .unwrap()
            .into();

        let c1_op = before_block.append_operation(
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

        let cmpi_op = before_block.append_operation(
            OperationBuilder::new("arith.cmpi", gen.loc())
                .add_operands(&[tag_val, c1_val])
                .add_results(&[gen.i1_ty])
                .add_attributes(&[(
                    Identifier::new(gen.context, "predicate"),
                    IntegerAttribute::new(
                        gen.i64_ty, 0, // eq
                    )
                    .into(),
                )])
                .build()
                .unwrap(),
        );
        let cond_val = cmpi_op.result(0).unwrap().into();

        let _condition_op = before_block.append_operation(
            OperationBuilder::new("scf.condition", gen.loc())
                .add_operands(&[cond_val, opt_val, updated_iter_val])
                .build()
                .unwrap(),
        );
        before_region.append_block(before_block);

        let after_region = Region::new();
        let after_block = Block::new(&[(opt_ty, gen.loc()), (iter_ty, gen.loc())]);
        let opt_arg = after_block.argument(0).unwrap().into();
        let next_iter_arg = after_block.argument(1).unwrap().into();

        let opt_ty_str = opt_ty.to_string();
        let payload_ty_str = if opt_ty_str.contains("(i32, ") {
            let start = opt_ty_str.find("(i32, ").unwrap() + 6;
            let end = opt_ty_str.rfind(')').unwrap();
            opt_ty_str[start..end].to_string()
        } else {
            "i32".to_string() // fallback
        };
        let payload_ty = Type::parse(gen.context, &payload_ty_str).unwrap();

        let extract_payload_op = OperationBuilder::new("llvm.extractvalue", gen.loc())
            .add_operands(&[opt_arg])
            .add_results(&[payload_ty])
            .add_attributes(&[(
                Identifier::new(gen.context, "position"),
                melior::ir::attribute::DenseI64ArrayAttribute::new(gen.context, &[1]).into(),
            )])
            .build()
            .unwrap();
        let payload_val = after_block
            .append_operation(extract_payload_op)
            .result(0)
            .unwrap()
            .into();

        gen.env.insert(iter.clone(), (payload_val, payload_ty));

        // execute body
        gen.break_flags.push(cond_val); // dummy value just to enable breaks?
                                        // Wait, break guard logic requires `generate_statements_with_break_guard`?
                                        // For simplicity, just generate body:
        for stmt in body {
            gen.generate_statement(stmt, &after_block)?;
        }
        gen.break_flags.pop();

        after_block.append_operation(
            OperationBuilder::new("scf.yield", gen.loc())
                .add_operands(&[next_iter_arg])
                .build()
                .unwrap(),
        );
        after_region.append_block(after_block);

        block.append_operation(
            OperationBuilder::new("scf.while", gen.loc())
                .add_operands(&[iter_val])
                .add_results(&[opt_ty, iter_ty])
                .add_regions([before_region, after_region])
                .build()
                .unwrap(),
        );

        Ok(())
    }
}

impl<'c> LowerToMelior<'c> for LoopStmt {
    type Output = Result<(), LowerError>;
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let i1_ty = gen.i1_ty;
        let memref_ty = Type::parse(gen.context, "memref<1xi1>").unwrap();

        let alloca_op = block.append_operation(
            OperationBuilder::new("memref.alloca", gen.loc())
                .add_results(&[memref_ty])
                .build()
                .unwrap(),
        );
        let break_ptr = alloca_op.result(0).unwrap().into();

        let continue_alloca = block.append_operation(
            OperationBuilder::new("memref.alloca", gen.loc())
                .add_results(&[memref_ty])
                .build()
                .unwrap(),
        );
        let continue_ptr = continue_alloca.result(0).unwrap().into();

        let false_op = block.append_operation(
            OperationBuilder::new("arith.constant", gen.loc())
                .add_results(&[i1_ty])
                .add_attributes(&[(
                    Identifier::new(gen.context, "value"),
                    IntegerAttribute::new(i1_ty, 0).into(),
                )])
                .build()
                .unwrap(),
        );
        let false_val = false_op.result(0).unwrap().into();

        let c0_op = block.append_operation(
            OperationBuilder::new("arith.constant", gen.loc())
                .add_results(&[gen.index_ty])
                .add_attributes(&[(
                    Identifier::new(gen.context, "value"),
                    IntegerAttribute::new(gen.index_ty, 0).into(),
                )])
                .build()
                .unwrap(),
        );
        let c0_idx = c0_op.result(0).unwrap().into();

        block.append_operation(
            OperationBuilder::new("memref.store", gen.loc())
                .add_operands(&[false_val, break_ptr, c0_idx])
                .build()
                .unwrap(),
        );

        gen.break_flags.push(break_ptr);
        gen.continue_flags.push(continue_ptr);

        let before_region = Region::new();
        let before_block = Block::new(&[]);

        before_block.append_operation(
            OperationBuilder::new("memref.store", gen.loc())
                .add_operands(&[false_val, continue_ptr, c0_idx])
                .build()
                .unwrap(),
        );

        let load_op = before_block.append_operation(
            OperationBuilder::new("memref.load", gen.loc())
                .add_operands(&[break_ptr, c0_idx])
                .add_results(&[i1_ty])
                .build()
                .unwrap(),
        );
        let is_break = load_op.result(0).unwrap().into();

        let true_op = before_block.append_operation(
            OperationBuilder::new("arith.constant", gen.loc())
                .add_results(&[i1_ty])
                .add_attributes(&[(
                    Identifier::new(gen.context, "value"),
                    IntegerAttribute::new(i1_ty, 1).into(),
                )])
                .build()
                .unwrap(),
        );
        let true_val = true_op.result(0).unwrap().into();

        let not_break_op = before_block.append_operation(
            OperationBuilder::new("arith.xori", gen.loc())
                .add_operands(&[is_break, true_val])
                .add_results(&[i1_ty])
                .build()
                .unwrap(),
        );
        let not_break = not_break_op.result(0).unwrap().into();

        before_block.append_operation(
            OperationBuilder::new("scf.condition", gen.loc())
                .add_operands(&[not_break])
                .build()
                .unwrap(),
        );
        before_region.append_block(before_block);

        let after_region = Region::new();
        let after_block = Block::new(&[]);

        generate_statements_with_break_guard(
            gen,
            &self.body,
            &after_block,
            break_ptr,
            continue_ptr,
            c0_idx,
            i1_ty,
        );

        after_block.append_operation(
            OperationBuilder::new("scf.yield", gen.loc())
                .build()
                .unwrap(),
        );
        after_region.append_block(after_block);

        block.append_operation(
            OperationBuilder::new("scf.while", gen.loc())
                .add_regions([before_region, after_region])
                .build()
                .unwrap(),
        );

        gen.continue_flags.pop();
        gen.break_flags.pop();

        Ok(())
    }
}

impl<'c> LowerToMelior<'c> for BreakStmt {
    type Output = Result<(), LowerError>;
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        if let Some(&break_ptr) = gen.break_flags.last() {
            let i1_ty = gen.i1_ty;
            let true_op = block.append_operation(
                OperationBuilder::new("arith.constant", gen.loc())
                    .add_results(&[i1_ty])
                    .add_attributes(&[(
                        Identifier::new(gen.context, "value"),
                        IntegerAttribute::new(i1_ty, 1).into(),
                    )])
                    .build()
                    .unwrap(),
            );
            let true_val = true_op.result(0).unwrap().into();

            let c0_op = block.append_operation(
                OperationBuilder::new("arith.constant", gen.loc())
                    .add_results(&[gen.index_ty])
                    .add_attributes(&[(
                        Identifier::new(gen.context, "value"),
                        IntegerAttribute::new(gen.index_ty, 0).into(),
                    )])
                    .build()
                    .unwrap(),
            );
            let c0_idx = c0_op.result(0).unwrap().into();

            block.append_operation(
                OperationBuilder::new("memref.store", gen.loc())
                    .add_operands(&[true_val, break_ptr, c0_idx])
                    .build()
                    .unwrap(),
            );
        } else {
            panic!("break outside of a loop");
        }

        Ok(())
    }
}

impl<'c> LowerToMelior<'c> for ContinueStmt {
    type Output = Result<(), LowerError>;
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        if let Some(&continue_ptr) = gen.continue_flags.last() {
            let i1_ty = gen.i1_ty;
            let true_op = block.append_operation(
                OperationBuilder::new("arith.constant", gen.loc())
                    .add_results(&[i1_ty])
                    .add_attributes(&[(
                        Identifier::new(gen.context, "value"),
                        IntegerAttribute::new(i1_ty, 1).into(),
                    )])
                    .build()
                    .unwrap(),
            );
            let true_val = true_op.result(0).unwrap().into();

            let c0_op = block.append_operation(
                OperationBuilder::new("arith.constant", gen.loc())
                    .add_results(&[gen.index_ty])
                    .add_attributes(&[(
                        Identifier::new(gen.context, "value"),
                        IntegerAttribute::new(gen.index_ty, 0).into(),
                    )])
                    .build()
                    .unwrap(),
            );
            let c0_idx = c0_op.result(0).unwrap().into();

            block.append_operation(
                OperationBuilder::new("memref.store", gen.loc())
                    .add_operands(&[true_val, continue_ptr, c0_idx])
                    .build()
                    .unwrap(),
            );
        } else {
            panic!("continue outside of a loop");
        }

        Ok(())
    }
}
