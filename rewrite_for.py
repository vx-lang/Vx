import re

with open("src/codegen/lower/control_flow.rs", "r") as f:
    content = f.read()

for_pattern = r"impl\s*<\s*'c\s*>\s*LowerToMelior\s*<\s*'c\s*>\s*for\s*ForLoopStmt.*?\{.*?\n        \}\n    \}\n\}"
for_replacement = """impl<'c> LowerToMelior<'c> for ForLoopStmt {
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

        if let Expr::Range(ast::expr::RangeExpr {
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
            let cond_block = parent_region.append_block(melior::ir::Block::new(&[(ty_index, gen.loc())]));
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
                    .build()
                    .unwrap(),
            );

            gen.env.insert(iter.clone().into(), (current_idx, ty_index));
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
                let next_idx_op = body_block.append_operation(
                    OperationBuilder::new("arith.addi", gen.loc())
                        .add_operands(&[current_idx, step_idx])
                        .add_results(&[ty_index])
                        .build()
                        .unwrap(),
                );
                let next_idx = next_idx_op.result(0).unwrap().into();

                body_block.append_operation(
                    OperationBuilder::new("cf.br", gen.loc())
                        .add_operands(&[next_idx])
                        .add_successors(&[&*cond_block])
                        .build()
                        .unwrap(),
                );
            }

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
            let cond_block = parent_region.append_block(melior::ir::Block::new(&[(ty_index, gen.loc())]));
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
                let next_idx_op = body_block.append_operation(
                    OperationBuilder::new("arith.addi", gen.loc())
                        .add_operands(&[current_idx, step_idx])
                        .add_results(&[ty_index])
                        .build()
                        .unwrap(),
                );
                let next_idx = next_idx_op.result(0).unwrap().into();

                body_block.append_operation(
                    OperationBuilder::new("cf.br", gen.loc())
                        .add_operands(&[next_idx])
                        .add_successors(&[&*cond_block])
                        .build()
                        .unwrap(),
                );
            }

            return Ok(Some(merge_block));
        }

        // Generic Iterator loop
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

        gen.env.insert(iter.clone().into(), (payload_val, payload_ty));
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
}"""

content = re.sub(for_pattern, for_replacement, content, flags=re.DOTALL)

with open("src/codegen/lower/control_flow.rs", "w") as f:
    f.write(content)
