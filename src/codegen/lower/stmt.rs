use super::*;
use crate::ast;
use crate::ast::*;
use melior::ir::{
    attribute::{DenseI32ArrayAttribute, IntegerAttribute, TypeAttribute},
    operation::OperationBuilder,
    Identifier, Type,
};

impl<'c> LowerToMelior<'c> for ReturnStmt {
    type Output = Result<(), LowerError>;
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let ReturnStmt { expr, span: _ } = self;
        gen.expected_type = gen.current_return_type;
        let (mut val, expr_ty) = gen.generate_expr(expr, block)?;
        gen.expected_type = None;
        if let Some(ret_ty) = gen.current_return_type {
            if expr_ty != ret_ty {
                if gen.is_memref(&expr_ty) && gen.is_memref(&ret_ty) {
                    let mut cast_op_name = "memref.cast";
                    let expr_parts = expr_ty.to_string();
                    let ret_parts = ret_ty.to_string();
                    let _expr_has_space =
                        expr_parts.matches(',').count() > 0 && !expr_parts.contains("strided");
                    let _ret_has_space =
                        ret_parts.matches(',').count() > 0 && !ret_parts.contains("strided");
                    if expr_parts.matches(',').count() != ret_parts.matches(',').count() {
                        cast_op_name = "memref.memory_space_cast";
                    }

                    let cast_op = OperationBuilder::new(cast_op_name, gen.loc())
                        .add_operands(&[val])
                        .add_results(&[ret_ty])
                        .build()
                        .unwrap();
                    val = block.append_operation(cast_op).result(0).unwrap().into();
                } else if ret_ty.to_string() == "i32" && gen.is_memref(&expr_ty) {
                    let zero_op = OperationBuilder::new("arith.constant", gen.loc())
                        .add_results(&[ret_ty])
                        .add_attributes(&[(
                            Identifier::new(gen.context, "value"),
                            IntegerAttribute::new(ret_ty, 0).into(),
                        )])
                        .build()
                        .unwrap();
                    val = block.append_operation(zero_op).result(0).unwrap().into();
                } else {
                    val = gen.coerce_type(block, val, expr_ty, ret_ty);
                }
            }
        }
        let op_name = if gen.in_spawn {
            "vx.return"
        } else {
            "func.return"
        };
        let ret_op = OperationBuilder::new(op_name, gen.loc())
            .add_operands(&[val])
            .build()
            .unwrap();
        block.append_operation(ret_op);
        gen.has_returned = true;

        Ok(())
    }
}

impl<'c> LowerToMelior<'c> for LetDeclStmt {
    type Output = Result<(), LowerError>;
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let LetDeclStmt {
            name,
            is_mut,
            ty_ann,
            expr,
            span: _,
        } = self;
        let prev_expected = gen.expected_type;
        if let Some(ann) = ty_ann {
            gen.expected_type = Some(gen.lower_type(ann));
        }
        let (val, ty) = gen.generate_expr(expr, block)?;
        if let Expr::Closure(c) = expr {
            let func_args = c.params.iter().map(|(_, t)| t.clone()).collect();
            let ret_ty = c
                .ret_ty
                .clone()
                .unwrap_or(ast::Type::Scalar(ast::ElementType::I32));
            gen.ast_env.insert(
                name.clone(),
                ast::Type::Closure(func_args, Box::new(ret_ty)),
            );
        }
        gen.expected_type = prev_expected;
        if *is_mut {
            let ty_str = ty.to_string();
            if ty_str.contains("!llvm.struct") || ty_str.contains("!llvm.ptr") {
                let ptr_ty = gen.ptr_ty;
                let i32_ty = gen.i32_ty;
                let one_attr = IntegerAttribute::new(i32_ty, 1).into();
                let const_op = OperationBuilder::new("llvm.mlir.constant", gen.loc())
                    .add_results(&[i32_ty])
                    .add_attributes(&[(Identifier::new(gen.context, "value"), one_attr)])
                    .build()
                    .unwrap();
                let one_val = block.append_operation(const_op).result(0).unwrap().into();

                let alloca_op = OperationBuilder::new("llvm.alloca", gen.loc())
                    .add_operands(&[one_val])
                    .add_results(&[ptr_ty])
                    .add_attributes(&[(
                        Identifier::new(gen.context, "elem_type"),
                        TypeAttribute::new(ty).into(),
                    )])
                    .build()
                    .unwrap();
                let alloca_ref = block.append_operation(alloca_op);
                let alloca_val = alloca_ref.result(0).unwrap().into();

                let store_op = OperationBuilder::new("llvm.store", gen.loc())
                    .add_operands(&[val, alloca_val])
                    .build()
                    .unwrap();
                block.append_operation(store_op);

                gen.env.insert(name.clone(), (alloca_val, ty));
                gen.allocs.insert(name.clone());
            } else {
                let memref_ty = format!("memref<{}>", ty);
                let parsed_memref_ty = Type::parse(gen.context, &memref_ty).unwrap();
                let alloca_op = OperationBuilder::new("memref.alloca", gen.loc())
                    .add_results(&[parsed_memref_ty])
                    .build()
                    .unwrap();
                let alloca_ref = block.append_operation(alloca_op);
                let alloca_val = alloca_ref.result(0).unwrap().into();

                let store_op = OperationBuilder::new("memref.store", gen.loc())
                    .add_operands(&[val, alloca_val])
                    .build()
                    .unwrap();
                block.append_operation(store_op);
                gen.env.insert(name.clone(), (alloca_val, parsed_memref_ty));
                gen.allocs.insert(name.clone());
            }
        } else {
            let ast_ty = ty_ann.clone().or_else(|| gen.infer_ast_type(expr));
            if let Some(t) = ast_ty {
                gen.ast_env.insert(name.clone(), t);
            }
            gen.env.insert(name.clone(), (val, ty));
        }

        Ok(())
    }
}

impl<'c> LowerToMelior<'c> for AssignStmt {
    type Output = Result<(), LowerError>;
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let AssignStmt { lhs, rhs, span: _ } = self;

        let mut expected_ty = None;
        if let Expr::Identifier(IdentifierExpr { name, span: _ }) = lhs {
            if let Some((_, mem_ty)) = gen.env.get(name) {
                let mem_ty_str = mem_ty.to_string();
                if mem_ty_str.starts_with("memref<") {
                    let inner_ty_str = &mem_ty_str[7..mem_ty_str.len() - 1];
                    expected_ty = Some(Type::parse(gen.context, inner_ty_str).unwrap());
                } else {
                    expected_ty = Some(*mem_ty);
                }
            }
        }

        let prev_expected = gen.expected_type;
        if expected_ty.is_some() {
            gen.expected_type = expected_ty;
        }
        let (rhs_val, rhs_ty) = gen.generate_expr(rhs, block)?;
        gen.expected_type = prev_expected;

        if let Expr::Identifier(IdentifierExpr { name, span: _ }) = lhs {
            if let Some((mem_val, mem_ty)) = gen.env.get(name).cloned() {
                let mem_ty_str = mem_ty.to_string();
                if mem_ty_str.starts_with("memref<") {
                    let mut store_val = rhs_val;
                    let inner_ty_str = &mem_ty_str[7..mem_ty_str.len() - 1];
                    let inner_ty = Type::parse(gen.context, inner_ty_str).unwrap();
                    if rhs_ty != inner_ty
                        && ((rhs_ty.to_string() == "i32" && inner_ty_str == "index")
                            || (rhs_ty.to_string() == "index" && inner_ty_str == "i32"))
                    {
                        let cast_op = OperationBuilder::new("arith.index_cast", gen.loc())
                            .add_operands(&[rhs_val])
                            .add_results(&[inner_ty])
                            .build()
                            .unwrap();

                        store_val = block.append_operation(cast_op).result(0).unwrap().into();
                    }

                    let store_op = OperationBuilder::new("memref.store", gen.loc())
                        .add_operands(&[store_val, mem_val])
                        .build()
                        .unwrap();
                    block.append_operation(store_op);
                } else if gen.allocs.contains(name) {
                    let store_op = OperationBuilder::new("llvm.store", gen.loc())
                        .add_operands(&[rhs_val, mem_val])
                        .build()
                        .unwrap();
                    block.append_operation(store_op);
                } else {
                    gen.env.insert(name.clone(), (rhs_val, rhs_ty));
                }
            }
        } else if let Expr::IndexAccess(ast::IndexAccessExpr {
            base,
            index: _,
            span: _,
        }) = lhs
        {
            if let Some((base_val, base_ty, indices)) = gen.flatten_indices(
                &ast::Expr::IndexAccess(ast::IndexAccessExpr {
                    base: base.clone(),
                    index: match lhs {
                        Expr::IndexAccess(i) => i.index.clone(),
                        _ => unreachable!(),
                    },
                    span: Span::default(),
                }),
                block,
            ) {
                let base_ty_str = base_ty.to_string();
                if base_ty_str.starts_with("!llvm.ptr") {
                    let i64_ty = gen.i64_ty;
                    let cast_op = OperationBuilder::new("arith.index_cast", gen.loc())
                        .add_operands(&[indices[0]])
                        .add_results(&[i64_ty])
                        .build()
                        .unwrap();
                    let idx_i64 = block.append_operation(cast_op).result(0).unwrap().into();

                    let gep_op = OperationBuilder::new("llvm.getelementptr", gen.loc())
                        .add_attributes(&[
                            (
                                Identifier::new(gen.context, "rawConstantIndices"),
                                DenseI32ArrayAttribute::new(gen.context, &[-2147483648]).into(),
                            ),
                            (
                                Identifier::new(gen.context, "elem_type"),
                                TypeAttribute::new(rhs_ty).into(),
                            ),
                        ])
                        .add_operands(&[base_val, idx_i64])
                        .add_results(&[base_ty])
                        .build()
                        .unwrap();

                    let gep_ref = block.append_operation(gep_op);
                    let ptr_val = gep_ref.result(0).unwrap().into();

                    let store_op = OperationBuilder::new("llvm.store", gen.loc())
                        .add_operands(&[rhs_val, ptr_val])
                        .build()
                        .unwrap();

                    block.append_operation(store_op);
                } else {
                    let mut inner_ty_str = String::new();
                    if let Some(start) = base_ty_str.find('<') {
                        let inner = &base_ty_str[start + 1..base_ty_str.len() - 1];
                        let parts: Vec<&str> = inner.split(',').collect();
                        let shape_type = parts[0].trim();
                        if let Some(last_x) = shape_type.rfind('x') {
                            inner_ty_str = shape_type[last_x + 1..].to_string();
                        } else {
                            inner_ty_str = shape_type.to_string();
                        }
                    }

                    let mut store_val = rhs_val;
                    if !inner_ty_str.is_empty() {
                        let inner_ty = melior::ir::Type::parse(gen.context, &inner_ty_str).unwrap();
                        store_val = gen.coerce_type(block, store_val, rhs_ty, inner_ty);
                    }

                    let mut store_builder = OperationBuilder::new("memref.store", gen.loc())
                        .add_operands(&[store_val, base_val]);

                    for idx in indices {
                        store_builder = store_builder.add_operands(&[idx]);
                    }

                    let store_op = store_builder.build().unwrap();
                    block.append_operation(store_op);
                }
            }
        } else if let Expr::MemberAccess(MemberAccessExpr {
            base,
            member,
            struct_name,
            span: _,
        }) = lhs
        {
            if let Expr::Identifier(IdentifierExpr {
                name: base_name,
                span: _,
            }) = &**base
            {
                let (base_val, base_ty) = gen.generate_expr(base, block)?;
                let base_ty_str = base_ty.to_string();

                let mut struct_name_opt = struct_name.clone();
                if struct_name_opt.is_none() {
                    if let Some(start_idx) = base_ty_str.find('"') {
                        if let Some(end_idx) = base_ty_str[start_idx + 1..].find('"') {
                            struct_name_opt = Some(
                                base_ty_str[start_idx + 1..start_idx + 1 + end_idx].to_string(),
                            );
                        }
                    }
                }

                let is_ptr = base_ty_str.starts_with("!llvm.ptr");

                if let Some(resolved_struct_name) = struct_name_opt {
                    if let Some(struct_decl) = gen.structs.get(&resolved_struct_name).cloned() {
                        if let Some(field_idx) =
                            struct_decl.fields.iter().position(|(n, _)| n == member)
                        {
                            let field_ty = gen.lower_type(&struct_decl.fields[field_idx].1);
                            let mut field_val = rhs_val;

                            if rhs_ty != field_ty
                                && ((rhs_ty.to_string() == "index"
                                    && field_ty.to_string() == "i32")
                                    || (rhs_ty.to_string() == "i32"
                                        && field_ty.to_string() == "index"))
                            {
                                let cast_op = OperationBuilder::new("arith.index_cast", gen.loc())
                                    .add_operands(&[rhs_val])
                                    .add_results(&[field_ty])
                                    .build()
                                    .unwrap();
                                field_val =
                                    block.append_operation(cast_op).result(0).unwrap().into();
                            }

                            if is_ptr {
                                let ptr_ty = gen.ptr_ty;
                                let mut field_types = Vec::new();
                                for (_, ty) in &struct_decl.fields {
                                    let mut lowered = gen.lower_type_str(ty);
                                    if lowered.starts_with("memref<") {
                                        lowered = "!llvm.ptr".to_string();
                                    }
                                    field_types.push(lowered);
                                }
                                let struct_llvm_ty_str = format!(
                                    "!llvm.struct<\"{}\", ({})>",
                                    resolved_struct_name,
                                    field_types.join(", ")
                                );
                                let struct_llvm_ty =
                                    Type::parse(gen.context, &struct_llvm_ty_str).unwrap();

                                let gep_op = OperationBuilder::new("llvm.getelementptr", gen.loc())
                                    .add_attributes(&[
                                        (
                                            Identifier::new(gen.context, "rawConstantIndices"),
                                            DenseI32ArrayAttribute::new(
                                                gen.context,
                                                &[0, field_idx as i32],
                                            )
                                            .into(),
                                        ),
                                        (
                                            Identifier::new(gen.context, "elem_type"),
                                            TypeAttribute::new(struct_llvm_ty).into(),
                                        ),
                                    ])
                                    .add_operands(&[base_val])
                                    .add_results(&[ptr_ty])
                                    .build()
                                    .unwrap();

                                let gep_ref = block.append_operation(gep_op);
                                let ptr_val = gep_ref.result(0).unwrap().into();

                                let store_op = OperationBuilder::new("llvm.store", gen.loc())
                                    .add_operands(&[field_val, ptr_val])
                                    .build()
                                    .unwrap();
                                block.append_operation(store_op);
                            } else {
                                let pos_attr = melior::ir::attribute::DenseI64ArrayAttribute::new(
                                    gen.context,
                                    &[field_idx as i64],
                                );
                                let insert_op =
                                    OperationBuilder::new("llvm.insertvalue", gen.loc())
                                        .add_operands(&[base_val, field_val])
                                        .add_attributes(&[(
                                            Identifier::new(gen.context, "position"),
                                            pos_attr.into(),
                                        )])
                                        .add_results(&[base_ty])
                                        .build()
                                        .unwrap();

                                let new_struct_val =
                                    block.append_operation(insert_op).result(0).unwrap().into();

                                if let Some((mem_val, mem_ty)) = gen.env.get(base_name).cloned() {
                                    let mem_ty_str = mem_ty.to_string();
                                    if mem_ty_str.starts_with("memref<") {
                                        let store_op =
                                            OperationBuilder::new("memref.store", gen.loc())
                                                .add_operands(&[new_struct_val, mem_val])
                                                .build()
                                                .unwrap();
                                        block.append_operation(store_op);
                                    } else {
                                        gen.env
                                            .insert(base_name.clone(), (new_struct_val, base_ty));
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                panic!("Complex struct assignment lhs not supported");
            }
        }

        Ok(())
    }
}

impl<'c> LowerToMelior<'c> for CompoundAssignStmt {
    type Output = Result<(), LowerError>;
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let CompoundAssignStmt {
            lhs,
            op,
            rhs,
            span: _,
        } = self;
        let (rhs_val, rhs_ty) = gen.generate_expr(rhs, block)?;
        let (lhs_val, ty) = gen.generate_expr(lhs, block)?;

        let mut actual_rhs = rhs_val;
        if rhs_ty != ty
            && ((rhs_ty.to_string() == "index" && ty.to_string() == "i32")
                || (rhs_ty.to_string() == "i32" && ty.to_string() == "index"))
        {
            let cast_op = OperationBuilder::new("arith.index_cast", gen.loc())
                .add_operands(&[actual_rhs])
                .add_results(&[ty])
                .build()
                .unwrap();
            actual_rhs = block.append_operation(cast_op).result(0).unwrap().into();
        }

        let is_float = ty.to_string().contains("f32")
            || ty.to_string().contains("f64")
            || ty.to_string().contains("f16")
            || ty.to_string().contains("bf16");
        let bin_op = OperationBuilder::new(op.get_op_name(is_float), gen.loc())
            .add_operands(&[lhs_val, actual_rhs])
            .add_results(&[ty])
            .build()
            .unwrap();
        let bin_ref = block.append_operation(bin_op);
        let result_val = bin_ref.result(0).unwrap().into();

        if let Expr::Identifier(IdentifierExpr { name, span: _ }) = lhs {
            if let Some((mem_val, mem_ty)) = gen.env.get(name).cloned() {
                let mem_ty_str = mem_ty.to_string();
                if mem_ty_str.starts_with("memref<") {
                    let store_op = OperationBuilder::new("memref.store", gen.loc())
                        .add_operands(&[result_val, mem_val])
                        .build()
                        .unwrap();
                    block.append_operation(store_op);
                } else {
                    gen.env.insert(name.clone(), (result_val, ty));
                }
            }
        } else if let Expr::IndexAccess(ast::IndexAccessExpr {
            base,
            index: _,
            span: _,
        }) = lhs
        {
            if let Some((mem_val, mem_ty, indices)) = gen.flatten_indices(
                &ast::Expr::IndexAccess(ast::IndexAccessExpr {
                    base: base.clone(),
                    index: match lhs {
                        Expr::IndexAccess(i) => i.index.clone(),
                        _ => unreachable!(),
                    },
                    span: match lhs {
                        Expr::IndexAccess(i) => i.span,
                        _ => unreachable!(),
                    },
                }),
                block,
            ) {
                let mem_ty_str = mem_ty.to_string();
                if mem_ty_str.starts_with("memref<") {
                    let mut operands = vec![result_val, mem_val];
                    operands.extend(indices);
                    let store_op = OperationBuilder::new("memref.store", gen.loc())
                        .add_operands(&operands)
                        .build()
                        .unwrap();
                    block.append_operation(store_op);
                }
            }
        }

        Ok(())
    }
}

impl<'c> LowerToMelior<'c> for ExprStmtStmt {
    type Output = Result<(), LowerError>;
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let ExprStmtStmt {
            expr,
            has_semi: _,
            span: _,
        } = self;
        gen.generate_expr(expr, block)?;

        Ok(())
    }
}
