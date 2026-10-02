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
            let depth = gen.open_block();
            let mut last_val = None;
            let target_block = if !then_block.is_empty() {
                Some(then_block)
            } else {
                else_block_opt.as_ref()
            };

            // The surviving branch is spliced straight into the enclosing block,
            // with no branch of its own -- which means every statement in it can
            // move the insertion point, and the branch as a whole can close the
            // block outright. A `for` opens three blocks and lands in a fourth;
            // an `if` lands in its merge block; a `return` ends the block for
            // good. Lowering the rest of the function into the block we started
            // in appends past a terminator, so `if comptime <taken> { for ... }`
            // did not compile at all.
            let mut cur = block;
            let mut terminated = false;
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
                            let (val, ty, tail_block) = gen.generate_expr(expr, cur)?;
                            cur = tail_block;
                            if !has_semi {
                                last_val = Some((val, ty));
                            }
                            continue;
                        }
                    }
                    match gen.generate_statement(stmt, cur)? {
                        Some(b) => cur = b,
                        None => {
                            terminated = true;
                            break;
                        }
                    }
                }
            }

            if terminated {
                cur = super::dead_continuation(cur);
            }
            gen.close_blocks_to(depth);

            if let Some((val, ty)) = last_val {
                return Ok((val, ty, cur));
            }

            let ret_ty = gen.expected_type.unwrap_or(gen.f32_ty);
            let dummy_op = if ret_ty.to_string() == "f32" {
                OperationBuilder::new("arith.constant", gen.loc())
                    .add_attributes(&[(
                        Identifier::new(gen.context, "value"),
                        FloatAttribute::new(gen.context, ret_ty, 0.0).into(),
                    )])
                    .add_results(&[ret_ty])
                    .build()?
            } else if ret_ty.to_string() == "i1" {
                OperationBuilder::new("arith.constant", gen.loc())
                    .add_attributes(&[(
                        Identifier::new(gen.context, "value"),
                        IntegerAttribute::new(ret_ty, 0).into(),
                    )])
                    .add_results(&[ret_ty])
                    .build()?
            } else {
                OperationBuilder::new("arith.constant", gen.loc())
                    .add_attributes(&[(
                        Identifier::new(gen.context, "value"),
                        IntegerAttribute::new(ret_ty, 0).into(),
                    )])
                    .add_results(&[ret_ty])
                    .build()?
            };
            return Ok((
                cur.append_operation(dummy_op).result(0)?.into(),
                ret_ty,
                cur,
            ));
        }

        let (cond_val, _, block) = gen.generate_expr(cond, block)?;

        let expected = gen.expected_type;
        let has_ret = expected.is_none_or(|t| {
            let t = t.to_string();
            t != "none" && t != "void"
        });
        let parent_region = block.parent_region().unwrap();
        let then_b = parent_region.append_block(melior::ir::Block::new(&[]));
        let else_b = parent_region.append_block(melior::ir::Block::new(&[]));
        // The merge block's argument is added once both branches are lowered, because with no
        // expected type the value's type is whatever the branches produce.
        let merge_b = parent_region.append_block(melior::ir::Block::new(&[]));

        block.append_operation(
            OperationBuilder::new("cf.cond_br", gen.loc())
                .add_operands(&[cond_val])
                .add_successors(&[&*then_b, &*else_b])
                .add_attributes(&[(
                    Identifier::new(gen.context, "operandSegmentSizes"),
                    melior::ir::attribute::DenseI32ArrayAttribute::new(gen.context, &[1, 0, 0])
                        .into(),
                )])
                .build()?,
        );

        let empty = Vec::new();
        let (then_end, then_val) = lower_branch(gen, then_block, then_b, has_ret)?;
        let (else_end, else_val) = lower_branch(
            gen,
            else_block_opt.as_ref().unwrap_or(&empty),
            else_b,
            has_ret,
        )?;

        // Without an expected type, the branches decide. A loop variable or an extent is an
        // `index` in MLIR but an `i32` in Vx, so an `index` value is given the Vx type, and a
        // branch that yields an `i32` and one that yields an `index` agree.
        let ret_ty = match expected {
            Some(t) => t,
            None => match then_val.or(else_val).map(|(_, t)| t) {
                Some(t) if t == gen.index_ty => gen.i32_ty,
                Some(t) => t,
                None => gen.f32_ty,
            },
        };
        let merge_arg = has_ret.then(|| merge_b.add_argument(ret_ty, gen.loc()));

        for (end, val) in [(then_end, then_val), (else_end, else_val)] {
            let Some(end) = end else { continue };
            let mut yield_operands = vec![];
            if has_ret {
                let v = match val {
                    Some((v, t)) => gen.coerce_type(&end, v, t, ret_ty)?,
                    None => {
                        let dummy_op = OperationBuilder::new("arith.constant", gen.loc())
                            .add_attributes(&[(
                                Identifier::new(gen.context, "value"),
                                melior::ir::attribute::IntegerAttribute::new(ret_ty, 0).into(),
                            )])
                            .build()?;
                        end.append_operation(dummy_op).result(0)?.into()
                    }
                };
                yield_operands.push(v);
            }
            end.append_operation(
                OperationBuilder::new("cf.br", gen.loc())
                    .add_operands(&yield_operands)
                    .add_successors(&[&*merge_b])
                    .build()?,
            );
        }

        // Both arms ended in a terminator, so nothing branches to the merge block -- but it still
        // needs a terminator of its own, because a block without one fails the MLIR verifier. That
        // is what a body ending in `if c { return a; } else { return b; }` produced: an empty block
        // the verifier rejected, after the frontend had accepted the program.
        if then_end.is_none() && else_end.is_none() {
            merge_b.append_operation(OperationBuilder::new("llvm.unreachable", gen.loc()).build()?);
        }

        match merge_arg {
            Some(res) => Ok((res, ret_ty, merge_b)),
            None => Ok((cond_val, ret_ty, merge_b)),
        }
    }
}

/// Lower one branch of an `if` into `b`. Returns the block the branch ends in, or `None` when it
/// ended in a terminator of its own (a `return`), and the value of its last expression when the
/// `if` produces one.
#[allow(clippy::type_complexity)]
fn lower_branch<'c>(
    gen: &mut MeliorGenerator<'c>,
    stmts: &[syntax::Statement],
    b: melior::ir::BlockRef<'c, 'c>,
    has_ret: bool,
) -> Result<
    (
        Option<melior::ir::BlockRef<'c, 'c>>,
        Option<(Value<'c, 'c>, Type<'c>)>,
    ),
    LowerError,
> {
    let depth = gen.open_block();
    let lowered = lower_branch_statements(gen, stmts, b, has_ret);
    gen.close_blocks_to(depth);
    lowered
}

#[allow(clippy::type_complexity)]
fn lower_branch_statements<'c>(
    gen: &mut MeliorGenerator<'c>,
    stmts: &[syntax::Statement],
    mut b: melior::ir::BlockRef<'c, 'c>,
    has_ret: bool,
) -> Result<
    (
        Option<melior::ir::BlockRef<'c, 'c>>,
        Option<(Value<'c, 'c>, Type<'c>)>,
    ),
    LowerError,
> {
    let mut val = None;
    for (i, stmt) in stmts.iter().enumerate() {
        if i == stmts.len() - 1 && has_ret {
            if let syntax::Statement::ExprStmt(syntax::stmt::ExprStmtStmt {
                expr,
                has_semi: false,
                ..
            }) = stmt
            {
                let (v, t, next) = gen.generate_expr(expr, b)?;
                b = next;
                val = Some((v, t));
                continue;
            }
        }
        match gen.generate_statement(stmt, b)? {
            Some(next) => b = next,
            None => return Ok((None, None)),
        }
    }
    Ok((Some(b), val))
}

impl<'c> LowerToMelior<'c> for ForLoopStmt {
    type Output = Result<Option<melior::ir::BlockRef<'c, 'c>>, LowerError>;
    /// The loop variable is a value, not a stack slot, so a `let mut` of the same name earlier
    /// in the function must not make the body read it through that slot. What the name meant
    /// before the loop is put back after it.
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let depth = gen.open_block();
        gen.note_shadow(self.iter.as_str());
        gen.allocs.remove(self.iter.as_str());
        let result = lower_for_loop(self, gen, block);
        gen.close_blocks_to(depth);
        result
    }
}

fn lower_for_loop<'c>(
    for_loop: &ForLoopStmt,
    gen: &mut MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
) -> Result<Option<melior::ir::BlockRef<'c, 'c>>, LowerError> {
    {
        let ForLoopStmt {
            iter,
            iterable,
            invariants: _,
            body,
            span: _,
            next_fn,
        } = for_loop;

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
                    .build()?;
                block.append_operation(cast_start_op).result(0)?.into()
            };

            let end_idx = if end_ty == ty_index {
                end_val
            } else {
                let cast_end_op = OperationBuilder::new("arith.index_cast", gen.loc())
                    .add_operands(&[end_val])
                    .add_results(&[ty_index])
                    .build()?;
                block.append_operation(cast_end_op).result(0)?.into()
            };

            let step_op = block.append_operation(
                OperationBuilder::new("arith.constant", gen.loc())
                    .add_results(&[ty_index])
                    .add_attributes(&[(
                        Identifier::new(gen.context, "value"),
                        IntegerAttribute::new(ty_index, 1).into(),
                    )])
                    .build()?,
            );
            let step_idx = step_op.result(0)?.into();

            let parent_region = block.parent_region().unwrap();
            let cond_block =
                parent_region.append_block(melior::ir::Block::new(&[(ty_index, gen.loc())]));
            let mut body_block = parent_region.append_block(melior::ir::Block::new(&[]));
            let merge_block = parent_region.append_block(melior::ir::Block::new(&[]));

            block.append_operation(
                OperationBuilder::new("cf.br", gen.loc())
                    .add_operands(&[start_idx])
                    .add_successors(&[&*cond_block])
                    .build()?,
            );

            let current_idx = cond_block
                .argument(0)
                .map_err(|_| {
                    crate::codegen::lower::LowerError::from(format!("Missing argument {} block", 0))
                })?
                .into();

            let cmp_op = cond_block.append_operation(
                OperationBuilder::new("arith.cmpi", gen.loc())
                    .add_operands(&[current_idx, end_idx])
                    .add_results(&[gen.i1_ty])
                    .add_attributes(&[(
                        Identifier::new(gen.context, "predicate"),
                        IntegerAttribute::new(
                            gen.i64_ty,
                            melior::dialect::arith::CmpiPredicate::Slt as i64,
                        )
                        .into(),
                    )])
                    .build()?,
            );
            let cond_val = cmp_op.result(0)?.into();

            cond_block.append_operation(
                OperationBuilder::new("cf.cond_br", gen.loc())
                    .add_operands(&[cond_val])
                    .add_successors(&[&*body_block, &*merge_block])
                    .add_attributes(&[(
                        Identifier::new(gen.context, "operandSegmentSizes"),
                        melior::ir::attribute::DenseI32ArrayAttribute::new(gen.context, &[1, 0, 0])
                            .into(),
                    )])
                    .build()?,
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
                        .build()?,
                );
            }

            // Latch: increment the index and branch back to the condition.
            // Both normal fall-through and `continue` route through here.
            let next_idx_op = latch_block.append_operation(
                OperationBuilder::new("arith.addi", gen.loc())
                    .add_operands(&[current_idx, step_idx])
                    .add_results(&[ty_index])
                    .build()?,
            );
            let next_idx = next_idx_op.result(0)?.into();
            latch_block.append_operation(
                OperationBuilder::new("cf.br", gen.loc())
                    .add_operands(&[next_idx])
                    .add_successors(&[&*cond_block])
                    .build()?,
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
                        .build()?,
                )
                .result(0)?
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
                        .build()?,
                )
                .result(0)?
                .into();

            let step_idx = block
                .append_operation(
                    OperationBuilder::new("arith.constant", gen.loc())
                        .add_results(&[ty_index])
                        .add_attributes(&[(
                            Identifier::new(gen.context, "value"),
                            IntegerAttribute::new(ty_index, 1).into(),
                        )])
                        .build()?,
                )
                .result(0)?
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
                    .build()?,
            );

            let current_idx = cond_block
                .argument(0)
                .map_err(|_| {
                    crate::codegen::lower::LowerError::from(format!("Missing argument {} block", 0))
                })?
                .into();

            let cmp_op = cond_block.append_operation(
                OperationBuilder::new("arith.cmpi", gen.loc())
                    .add_operands(&[current_idx, end_idx])
                    .add_results(&[gen.i1_ty])
                    .add_attributes(&[(
                        Identifier::new(gen.context, "predicate"),
                        IntegerAttribute::new(
                            gen.i64_ty,
                            melior::dialect::arith::CmpiPredicate::Slt as i64,
                        )
                        .into(),
                    )])
                    .build()?,
            );
            let cond_val = cmp_op.result(0)?.into();

            cond_block.append_operation(
                OperationBuilder::new("cf.cond_br", gen.loc())
                    .add_operands(&[cond_val])
                    .add_successors(&[&*body_block, &*merge_block])
                    .add_attributes(&[(
                        Identifier::new(gen.context, "operandSegmentSizes"),
                        melior::ir::attribute::DenseI32ArrayAttribute::new(gen.context, &[1, 0, 0])
                            .into(),
                    )])
                    .build()?,
            );

            let el_ty_str = iter_ty_str.split('x').nth(1).unwrap().trim_end_matches('>');
            let el_ty = Type::parse(gen.context, el_ty_str).ok_or_else(|| {
                crate::codegen::lower::LowerError::ParseType("Type::parse failed".to_string())
            })?;

            let extract_op = OperationBuilder::new("tensor.extract", gen.loc())
                .add_operands(&[iter_val, current_idx])
                .add_results(&[el_ty])
                .build()?;

            let el_val = body_block.append_operation(extract_op).result(0)?.into();

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
                        .build()?,
                );
            }

            // Latch: increment the index and branch back to the condition.
            // Both normal fall-through and `continue` route through here.
            let next_idx_op = latch_block.append_operation(
                OperationBuilder::new("arith.addi", gen.loc())
                    .add_operands(&[current_idx, step_idx])
                    .add_results(&[ty_index])
                    .build()?,
            );
            let next_idx = next_idx_op.result(0)?.into();
            latch_block.append_operation(
                OperationBuilder::new("cf.br", gen.loc())
                    .add_operands(&[next_idx])
                    .add_successors(&[&*cond_block])
                    .build()?,
            );

            return Ok(Some(merge_block));
        }

        // Generic Iterator loop via cf
        let ptr_ty = gen.ptr_ty;
        // Resolve the monomorphized `next` for *this* iterator. The mangler spells it with `$`
        // (`VecIter$i32$next$i32`), which the old `_next_`/`_next` patterns never matched (hence the
        // `Function next not found` panic). Prefer a `next` whose name shares the iterator's struct-name
        // prefix (from its MLIR type `!llvm.struct<"VecIter_i32", …>` -> `VecIter`), so the right
        // iterator's `next` is chosen when several are in scope. (#242)
        let iter_base: String = iter_ty_str
            .find('"')
            .and_then(|q| {
                iter_ty_str[q + 1..]
                    .find('"')
                    .map(|e| &iter_ty_str[q + 1..q + 1 + e])
            })
            .unwrap_or("")
            .chars()
            .take_while(|c| c.is_alphabetic())
            .collect();
        let is_next = |name: &str| {
            name.contains("_next_")
                || name.ends_with("_next")
                || name.contains("$next$")
                || name.ends_with("$next")
        };
        let mut actual_next_name = "next".to_string();
        if let Some(n) = next_fn
            .as_ref()
            .filter(|n| gen.functions.contains_key(n.as_ref()))
        {
            actual_next_name = n.to_string();
        }
        for (name, (_, _args)) in gen.functions.iter().filter(|_| actual_next_name == "next") {
            if is_next(name) && (iter_base.is_empty() || name.starts_with(&iter_base)) {
                actual_next_name = name.to_string();
                break;
            }
        }
        // Fallback: any `next` (a single-iterator program, or an unnamed iterator type).
        if actual_next_name == "next" {
            for (name, (_, _args)) in &gen.functions {
                if is_next(name) {
                    actual_next_name = name.to_string();
                    break;
                }
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
                .build()?,
        );
        let c1_val_alloc = c1_op_alloc.result(0)?.into();

        let alloca_op = block.append_operation(
            OperationBuilder::new("llvm.alloca", gen.loc())
                .add_operands(&[c1_val_alloc])
                .add_results(&[ptr_ty])
                .add_attributes(&[(
                    Identifier::new(gen.context, "elem_type"),
                    melior::ir::attribute::TypeAttribute::new(iter_ty).into(),
                )])
                .build()?,
        );
        let ptr_val = alloca_op.result(0)?.into();

        block.append_operation(
            OperationBuilder::new("llvm.store", gen.loc())
                .add_operands(&[iter_val, ptr_val])
                .build()?,
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
                .build()?,
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
            .build()?;
        let tag_val = cond_block_end
            .append_operation(extract_tag_op)
            .result(0)?
            .into();

        let c1_op = cond_block_end.append_operation(
            OperationBuilder::new("arith.constant", gen.loc())
                .add_results(&[i32_ty])
                .add_attributes(&[(
                    Identifier::new(gen.context, "value"),
                    IntegerAttribute::new(i32_ty, 0).into(),
                )])
                .build()?,
        );
        let c1_val = c1_op.result(0)?.into();

        let cmpi_op = cond_block_end.append_operation(
            OperationBuilder::new("arith.cmpi", gen.loc())
                .add_operands(&[tag_val, c1_val])
                .add_results(&[gen.i1_ty])
                .add_attributes(&[(
                    Identifier::new(gen.context, "predicate"),
                    IntegerAttribute::new(
                        gen.i64_ty,
                        melior::dialect::arith::CmpiPredicate::Eq as i64,
                    )
                    .into(),
                )])
                .build()?,
        );
        let cond_val = cmpi_op.result(0)?.into();

        cond_block_end.append_operation(
            OperationBuilder::new("cf.cond_br", gen.loc())
                .add_operands(&[cond_val])
                .add_successors(&[&*body_block, &*merge_block])
                .add_attributes(&[(
                    Identifier::new(gen.context, "operandSegmentSizes"),
                    melior::ir::attribute::DenseI32ArrayAttribute::new(gen.context, &[1, 0, 0])
                        .into(),
                )])
                .build()?,
        );

        let opt_ty_str = opt_ty.to_string();
        let mut payload_ty_str = if opt_ty_str.contains("(i32, ") {
            let start = opt_ty_str.find("(i32, ").unwrap() + 6;
            let end = opt_ty_str.rfind(')').unwrap();
            opt_ty_str[start..end].to_string()
        } else {
            "i32".to_string()
        };
        // A struct payload prints inside the option's type without its dialect prefix, and
        // parses only with it, as `match` lowering also finds.
        if ["struct", "ptr", "array"]
            .iter()
            .any(|p| payload_ty_str.starts_with(p))
        {
            payload_ty_str = format!("!llvm.{payload_ty_str}");
        }
        let payload_ty = Type::parse(gen.context, &payload_ty_str).ok_or_else(|| {
            crate::codegen::lower::LowerError::ParseType("Type::parse failed".to_string())
        })?;

        let extract_payload_op = OperationBuilder::new("llvm.extractvalue", gen.loc())
            .add_operands(&[opt_val])
            .add_results(&[payload_ty])
            .add_attributes(&[(
                Identifier::new(gen.context, "position"),
                melior::ir::attribute::DenseI64ArrayAttribute::new(gen.context, &[1]).into(),
            )])
            .build()?;
        let payload_val = body_block
            .append_operation(extract_payload_op)
            .result(0)?
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
                    .build()?,
            );
        }

        Ok(Some(merge_block))
    }
}

impl<'c> LowerToMelior<'c> for LoopStmt {
    type Output = Result<Option<melior::ir::BlockRef<'c, 'c>>, LowerError>;
    /// A block: a `let` in it hides an outer variable of the same name only until it ends.
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let depth = gen.open_block();
        let lowered = lower_loop_body(self, gen, block);
        gen.close_blocks_to(depth);
        lowered
    }
}

fn lower_loop_body<'c>(
    this: &LoopStmt,
    gen: &mut MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
) -> Result<Option<melior::ir::BlockRef<'c, 'c>>, LowerError> {
    let LoopStmt {
        invariants: _,
        body,
        span: _,
    } = this;

    let parent_region = block.parent_region().unwrap();
    let mut body_block = parent_region.append_block(melior::ir::Block::new(&[]));
    let loop_entry_block_ref = body_block; // separate variable
    let merge_block = parent_region.append_block(melior::ir::Block::new(&[]));

    let loop_entry_block = &*loop_entry_block_ref as *const melior::ir::Block<'c>;

    block.append_operation(
        OperationBuilder::new("cf.br", gen.loc())
            .add_successors(&[&*body_block])
            .build()?,
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
                .build()?,
        );
    }

    Ok(Some(merge_block))
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
                    .build()?,
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
                    .build()?,
            );
        } else {
            panic!("continue outside of a loop");
        }
        Ok(None)
    }
}
