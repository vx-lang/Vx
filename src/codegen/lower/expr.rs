use super::*;
use crate::ast;
use crate::ast::*;
use melior::ir::{
    attribute::{
        DenseI32ArrayAttribute, FlatSymbolRefAttribute, FloatAttribute, IntegerAttribute,
        StringAttribute, TypeAttribute,
    },
    operation::OperationBuilder,
    Identifier, Region, Type, Value,
};

impl<'c> LowerToMelior<'c> for IdentifierExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>), LowerError>;
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let IdentifierExpr { name, span: _ } = self;
        if name == "true" || name == "false" {
            let i1_ty = gen.i1_ty;
            let val = if name == "true" { 1 } else { 0 };
            let const_op = OperationBuilder::new("arith.constant", gen.loc())
                .add_results(&[i1_ty])
                .add_attributes(&[(
                    Identifier::new(gen.context, "value"),
                    IntegerAttribute::new(i1_ty, val).into(),
                )])
                .build()
                .unwrap();
            let const_ref = block.append_operation(const_op);
            return Ok((const_ref.result(0).unwrap().into(), i1_ty));
        }
        if let Some((val, ty)) = gen.env.get(name) {
            let ty_str = ty.to_string();
            if gen.allocs.contains(name) {
                if ty_str.starts_with("memref<") {
                    let inner_ty_str = &ty_str[7..ty_str.len() - 1];
                    let inner_ty = Type::parse(gen.context, inner_ty_str).unwrap_or_else(|| {
                        panic!("failed to parse {:?} for variable {:?}", inner_ty_str, name)
                    });
                    let load_op = OperationBuilder::new("memref.load", gen.loc())
                        .add_operands(&[*val])
                        .add_results(&[inner_ty])
                        .build()
                        .unwrap();
                    let load_ref = block.append_operation(load_op);
                    return Ok((load_ref.result(0).unwrap().into(), inner_ty));
                } else if ty_str.starts_with("!llvm.ptr")
                    || ty_str.starts_with("!llvm.struct")
                    || ty_str.starts_with("i")
                    || ty_str.starts_with("u")
                    || ty_str.starts_with("f")
                {
                    if gen.is_lvalue_context {
                        let ptr_ty = gen.ptr_ty;
                        return Ok((*val, ptr_ty));
                    }
                    let elem_ty = *ty;
                    let load_op = OperationBuilder::new("llvm.load", gen.loc())
                        .add_operands(&[*val])
                        .add_results(&[elem_ty])
                        .build()
                        .unwrap();
                    let load_ref = block.append_operation(load_op);
                    return Ok((load_ref.result(0).unwrap().into(), elem_ty));
                }
            }
            if ty_str.starts_with("memref<memref<") {
                let inner_ty_str = &ty_str[7..ty_str.len() - 1];
                let inner_ty = Type::parse(gen.context, inner_ty_str).unwrap();
                let load_op = OperationBuilder::new("memref.load", gen.loc())
                    .add_operands(&[*val])
                    .add_results(&[inner_ty])
                    .build()
                    .unwrap();
                let load_ref = block.append_operation(load_op);
                Ok((load_ref.result(0).unwrap().into(), inner_ty))
            } else if ty_str.starts_with("memref<") && !ty_str.contains("x") {
                let inner_ty_str = &ty_str[7..ty_str.len() - 1];
                let inner_ty = Type::parse(gen.context, inner_ty_str).unwrap();
                let load_op = OperationBuilder::new("memref.load", gen.loc())
                    .add_operands(&[*val])
                    .add_results(&[inner_ty])
                    .build()
                    .unwrap();
                let load_ref = block.append_operation(load_op);
                Ok((load_ref.result(0).unwrap().into(), inner_ty))
            } else {
                Ok((*val, *ty))
            }
        } else if gen.functions.contains_key(name) {
            let (ret_ty, arg_tys) = gen.functions.get(name).unwrap();
            let func_ty = melior::ir::r#type::FunctionType::new(gen.context, arg_tys, &[*ret_ty]);
            let const_op = OperationBuilder::new("func.constant", gen.loc())
                .add_attributes(&[(
                    Identifier::new(gen.context, "value"),
                    FlatSymbolRefAttribute::new(gen.context, name).into(),
                )])
                .add_results(&[func_ty.into()])
                .build()
                .unwrap();
            let const_ref = block.append_operation(const_op);

            let ptr_ty = gen.ptr_ty;
            let cast_op = OperationBuilder::new("builtin.unrealized_conversion_cast", gen.loc())
                .add_operands(&[const_ref.result(0).unwrap().into()])
                .add_results(&[ptr_ty])
                .build()
                .unwrap();
            let cast_ref = block.append_operation(cast_op);

            Ok((cast_ref.result(0).unwrap().into(), ptr_ty))
        } else {
            panic!("Undefined variable: {}", name);
        }
    }
}

impl<'c> LowerToMelior<'c> for BorrowExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>), LowerError>;
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let BorrowExpr { expr, .. } = self;
        if let Expr::Identifier(id) = &**expr {
            if gen.allocs.contains(&id.name) {
                if let Some((val, val_ty)) = gen.env.get(&id.name) {
                    let ptr_ty = gen.ptr_ty;
                    if gen.is_memref(val_ty) {
                        return Ok((*val, *val_ty));
                    } else if *val_ty == ptr_ty {
                        return Ok((*val, ptr_ty));
                    } else {
                        // Cast from val to ptr_ty if necessary? No, just return val_ty.
                        return Ok((*val, *val_ty));
                    }
                }
            }
        }
        let prev_lvalue = gen.is_lvalue_context;
        gen.is_lvalue_context = true;
        let (val, ty) = gen.generate_expr(expr, block)?;
        gen.is_lvalue_context = prev_lvalue;
        let ptr_ty = gen.ptr_ty;
        if ty == ptr_ty {
            return Ok((val, ptr_ty));
        }
        if ty.to_string().starts_with("memref<memref<") {
            return Ok((val, ty));
        }
        if gen.is_memref(&ty) {
            // Allocate a pointer to the memref
            let alloca_op = block.append_operation(
                OperationBuilder::new("memref.alloca", gen.loc())
                    .add_results(&[Type::parse(gen.context, &format!("memref<{}>", ty)).unwrap()])
                    .build()
                    .unwrap(),
            );
            let ptr = alloca_op.result(0).unwrap().into();

            block.append_operation(
                OperationBuilder::new("memref.store", gen.loc())
                    .add_operands(&[val, ptr])
                    .build()
                    .unwrap(),
            );
            return Ok((
                ptr,
                Type::parse(gen.context, &format!("memref<{}>", ty)).unwrap(),
            ));
        }

        let i32_ty = gen.i32_ty;
        let c1_op = block.append_operation(
            OperationBuilder::new("llvm.mlir.constant", gen.loc())
                .add_results(&[i32_ty])
                .add_attributes(&[(
                    Identifier::new(gen.context, "value"),
                    IntegerAttribute::new(i32_ty, 1).into(),
                )])
                .build()
                .unwrap(),
        );
        let c1 = c1_op.result(0).unwrap().into();

        let alloca_op = block.append_operation(
            OperationBuilder::new("llvm.alloca", gen.loc())
                .add_operands(&[c1])
                .add_results(&[ptr_ty])
                .add_attributes(&[(
                    Identifier::new(gen.context, "elem_type"),
                    TypeAttribute::new(ty).into(),
                )])
                .build()
                .unwrap(),
        );
        let ptr = alloca_op.result(0).unwrap().into();

        block.append_operation(
            OperationBuilder::new("llvm.store", gen.loc())
                .add_operands(&[val, ptr])
                .build()
                .unwrap(),
        );

        Ok((ptr, ptr_ty))
    }
}

impl<'c> LowerToMelior<'c> for StringLiteralExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>), LowerError>;
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let str_val = format!("{}\0", self.value);
        let str_name = format!(".str.{}", gen.string_counter);
        gen.string_counter += 1;

        let module_body = gen.module.body();
        let array_ty =
            Type::parse(gen.context, &format!("!llvm.array<{} x i8>", str_val.len())).unwrap();
        let ptr_ty = gen.ptr_ty;

        let global_op = OperationBuilder::new("llvm.mlir.global", gen.loc())
            .add_attributes(&[
                (
                    Identifier::new(gen.context, "sym_name"),
                    StringAttribute::new(gen.context, &str_name).into(),
                ),
                (
                    Identifier::new(gen.context, "global_type"),
                    TypeAttribute::new(array_ty).into(),
                ),
                (
                    Identifier::new(gen.context, "constant"),
                    Attribute::parse(gen.context, "unit").unwrap(),
                ),
                (
                    Identifier::new(gen.context, "linkage"),
                    Attribute::parse(gen.context, "#llvm.linkage<internal>").unwrap(),
                ),
                (
                    Identifier::new(gen.context, "value"),
                    StringAttribute::new(gen.context, &str_val).into(),
                ),
            ])
            .add_regions([Region::new()])
            .build()
            .unwrap();
        module_body.append_operation(global_op);

        let addressof_op = OperationBuilder::new("llvm.mlir.addressof", gen.loc())
            .add_attributes(&[(
                Identifier::new(gen.context, "global_name"),
                FlatSymbolRefAttribute::new(gen.context, &str_name).into(),
            )])
            .add_results(&[ptr_ty])
            .build()
            .unwrap();
        let addressof_ref = block.append_operation(addressof_op);

        Ok((addressof_ref.result(0).unwrap().into(), ptr_ty))
    }
}

impl<'c> LowerToMelior<'c> for ComptimeBlockExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>), LowerError>;
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        for stmt in &self.stmts {
            gen.generate_statement(stmt, block)?;
        }
        if let Some(ret_expr) = &self.ret {
            gen.generate_expr(ret_expr, block)
        } else {
            let none_ty = gen.none_ty;
            let dummy_val = OperationBuilder::new("arith.constant", gen.loc())
                .add_attributes(&[(
                    Identifier::new(gen.context, "value"),
                    IntegerAttribute::new(Type::index(gen.context), 0).into(),
                )])
                .add_results(&[Type::index(gen.context)])
                .build()
                .unwrap();
            let dummy_ref = block.append_operation(dummy_val);
            Ok((dummy_ref.result(0).unwrap().into(), none_ty))
        }
    }
}

impl<'c> LowerToMelior<'c> for DereferenceExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>), LowerError>;
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let (ptr_val, ptr_ty) = gen.generate_expr(&self.expr, block)?;
        let ptr_ty_str = ptr_ty.to_string();

        let inner_ty = if let Some(t) = &self.ty {
            gen.lower_type(t)
        } else {
            let inner_ty_str = if ptr_ty_str.starts_with("!llvm.ptr<") {
                ptr_ty_str[10..ptr_ty_str.len() - 1].to_string()
            } else {
                "f32".to_string()
            };
            Type::parse(gen.context, &inner_ty_str).unwrap()
        };

        if gen.is_lvalue_context {
            return Ok((ptr_val, ptr_ty));
        }

        let load_op = OperationBuilder::new("llvm.load", gen.loc())
            .add_operands(&[ptr_val])
            .add_results(&[inner_ty])
            .build()
            .unwrap();
        let load_ref = block.append_operation(load_op);
        Ok((load_ref.result(0).unwrap().into(), inner_ty))
    }
}

impl<'c> LowerToMelior<'c> for ast::IndexAccessExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>), LowerError>;
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let (base_val, base_ty, indices) = gen
            .flatten_indices(&ast::Expr::IndexAccess(self.clone()), block)
            .expect("Failed to flatten indices for IndexAccess");

        let base_ty_str = base_ty.to_string();
        let is_ptr = base_ty_str.starts_with("!llvm.ptr");

        if is_ptr {
            let inner_ty_str = if base_ty_str.contains("<") {
                base_ty_str[base_ty_str.find('<').unwrap() + 1..base_ty_str.len() - 1].to_string()
            } else if let Some(expected) = gen.expected_type {
                expected.to_string()
            } else {
                "f32".to_string()
            };
            let inner_ty = Type::parse(gen.context, &inner_ty_str)
                .unwrap_or_else(|| panic!("Failed to parse inner ptr type: {}", inner_ty_str));

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
                        TypeAttribute::new(inner_ty).into(),
                    ),
                ])
                .add_operands(&[base_val, idx_i64])
                .add_results(&[base_ty])
                .build()
                .unwrap();

            let gep_ref = block.append_operation(gep_op);
            let ptr_val = gep_ref.result(0).unwrap().into();

            if gen.is_lvalue_context {
                let ptr_ty = gen.ptr_ty;
                return Ok((ptr_val, ptr_ty));
            }

            let load_op = OperationBuilder::new("llvm.load", gen.loc())
                .add_operands(&[ptr_val])
                .add_results(&[inner_ty])
                .build()
                .unwrap();

            let load_ref = block.append_operation(load_op);
            Ok((load_ref.result(0).unwrap().into(), inner_ty))
        } else {
            let inner_ty_str = if base_ty_str.starts_with("memref<") {
                let inner = base_ty_str.replace("memref<", "").replace('>', "");
                let parts: Vec<&str> = inner.split('x').collect();
                let last_part = parts.last().unwrap();
                last_part.split(',').next().unwrap().trim().to_string()
            } else {
                "f32".to_string()
            };

            let inner_ty = Type::parse(gen.context, &inner_ty_str).unwrap();
            let mut load_builder =
                OperationBuilder::new("memref.load", gen.loc()).add_operands(&[base_val]);

            for idx in indices {
                load_builder = load_builder.add_operands(&[idx]);
            }

            let load_op = load_builder.add_results(&[inner_ty]).build().unwrap();

            let load_ref = block.append_operation(load_op);
            Ok((load_ref.result(0).unwrap().into(), inner_ty))
        }
    }
}

impl<'c> LowerToMelior<'c> for BinaryOpExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>), LowerError>;
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let BinaryOpExpr {
            lhs,
            op,
            rhs,
            span: _,
        } = self;
        let (mut lhs_val, lhs_ty) = gen.generate_expr(lhs, block)?;
        let prev_expected = gen.expected_type;
        gen.expected_type = Some(lhs_ty);
        let (mut rhs_val, mut rhs_ty) = gen.generate_expr(rhs, block)?;
        gen.expected_type = prev_expected;

        let mut final_ty = lhs_ty;
        let _lhs_ty_str = lhs_ty.to_string();
        let _rhs_ty_str = rhs_ty.to_string();

        if lhs_ty != rhs_ty && op != &BinaryOp::MatMul {
            // Priority coercion: f64 > f32 > i64 > i32
            // To simplify, we'll cast rhs to lhs for now.
            rhs_val = gen.coerce_type(block, rhs_val, rhs_ty, lhs_ty);
            rhs_ty = lhs_ty;
            final_ty = lhs_ty;
        }

        let is_memref = gen.is_memref(&lhs_ty) && gen.is_memref(&rhs_ty);

        let mut is_matmul = false;
        let mut lhs_parts = Vec::new();
        let mut rhs_parts = Vec::new();
        let lhs_ty_str = lhs_ty.to_string();
        let rhs_ty_str = rhs_ty.to_string();

        let lhs_inner = lhs_ty_str.replace("memref<", "").replace('>', "");
        let rhs_inner = rhs_ty_str.replace("memref<", "").replace('>', "");

        if is_memref {
            lhs_parts = lhs_inner.split('x').collect();
            rhs_parts = rhs_inner.split('x').collect();
            is_matmul = if op == &BinaryOp::MatMul
                && is_memref
                && lhs_parts.len() == 3
                && rhs_parts.len() == 3
            {
                let lhs_dim1 = lhs_parts[1];
                let rhs_dim0 = rhs_parts[0];
                lhs_dim1 == rhs_dim0 || lhs_dim1 == "?" || rhs_dim0 == "?"
            } else {
                false
            };
            println!(
                "DEBUG is_matmul: op={:?}, is_memref={}, lhs={}, rhs={}, is_matmul={}",
                op, is_memref, lhs_ty_str, rhs_ty_str, is_matmul
            );
        }

        if is_matmul {
            let m_str = lhs_parts[0];
            let n_str = rhs_parts[1];
            let el_ty_str = lhs_parts[2].split(',').next().unwrap().trim();

            let out_ty_str = format!("memref<{}x{}x{}>", m_str, n_str, el_ty_str);
            let out_ty = Type::parse(gen.context, &out_ty_str).unwrap();

            // Determine dynamic dimensions for alloc
            let mut alloc_operands = Vec::new();
            let index_ty = gen.index_ty;

            if m_str == "?" {
                let m_idx_attr = IntegerAttribute::new(Type::index(gen.context), 0).into();
                let cst_op = OperationBuilder::new("arith.constant", gen.loc())
                    .add_results(&[index_ty])
                    .add_attributes(&[(Identifier::new(gen.context, "value"), m_idx_attr)])
                    .build()
                    .unwrap();
                let idx_val = block.append_operation(cst_op).result(0).unwrap().into();

                let dim_m_op = OperationBuilder::new("memref.dim", gen.loc())
                    .add_operands(&[lhs_val, idx_val])
                    .add_results(&[index_ty])
                    .build()
                    .unwrap();
                alloc_operands.push(block.append_operation(dim_m_op).result(0).unwrap().into());
            }

            if n_str == "?" {
                let n_idx_attr = IntegerAttribute::new(Type::index(gen.context), 1).into();
                let cst_op = OperationBuilder::new("arith.constant", gen.loc())
                    .add_results(&[index_ty])
                    .add_attributes(&[(Identifier::new(gen.context, "value"), n_idx_attr)])
                    .build()
                    .unwrap();
                let idx_val = block.append_operation(cst_op).result(0).unwrap().into();

                let dim_n_op = OperationBuilder::new("memref.dim", gen.loc())
                    .add_operands(&[rhs_val, idx_val])
                    .add_results(&[index_ty])
                    .build()
                    .unwrap();
                alloc_operands.push(block.append_operation(dim_n_op).result(0).unwrap().into());
            }

            // Alloc output buffer
            let alloc_op = OperationBuilder::new("memref.alloc", gen.loc())
                .add_operands(&alloc_operands)
                .add_attributes(&[(
                    Identifier::new(gen.context, "operandSegmentSizes"),
                    DenseI32ArrayAttribute::new(gen.context, &[alloc_operands.len() as i32, 0])
                        .into(),
                )])
                .add_results(&[out_ty])
                .build()
                .unwrap();
            let out_val = block.append_operation(alloc_op).result(0).unwrap().into();

            // Zero initialize the output buffer since matmul accumulates!
            let zero_attr = if el_ty_str.starts_with('i') {
                IntegerAttribute::new(Type::parse(gen.context, el_ty_str).unwrap(), 0).into()
            } else {
                FloatAttribute::new(
                    gen.context,
                    Type::parse(gen.context, el_ty_str).unwrap(),
                    0.0,
                )
                .into()
            };

            let zero_op = OperationBuilder::new("arith.constant", gen.loc())
                .add_results(&[Type::parse(gen.context, el_ty_str).unwrap()])
                .add_attributes(&[(Identifier::new(gen.context, "value"), zero_attr)])
                .build()
                .unwrap();
            let zero_val = block.append_operation(zero_op).result(0).unwrap().into();

            let region_fill = Region::new();
            let block_fill = melior::ir::Block::new(&[
                (Type::parse(gen.context, el_ty_str).unwrap(), gen.loc()),
                (Type::parse(gen.context, el_ty_str).unwrap(), gen.loc()),
            ]);
            let yield_fill = OperationBuilder::new("linalg.yield", gen.loc())
                .add_operands(&[block_fill.argument(0).unwrap().into()])
                .build()
                .unwrap();
            block_fill.append_operation(yield_fill);
            region_fill.append_block(block_fill);

            let linalg_fill = OperationBuilder::new("linalg.fill", gen.loc())
                .add_operands(&[zero_val, out_val])
                .add_attributes(&[(
                    Identifier::new(gen.context, "operandSegmentSizes"),
                    DenseI32ArrayAttribute::new(gen.context, &[1, 1]).into(),
                )])
                .add_regions([region_fill])
                .build()
                .unwrap();
            block.append_operation(linalg_fill);

            // Execute linalg.matmul
            let region_matmul = Region::new();
            let block_matmul = melior::ir::Block::new(&[
                (Type::parse(gen.context, el_ty_str).unwrap(), gen.loc()),
                (Type::parse(gen.context, el_ty_str).unwrap(), gen.loc()),
                (Type::parse(gen.context, el_ty_str).unwrap(), gen.loc()),
            ]);

            let is_float = el_ty_str.contains("f32")
                || el_ty_str.contains("f64")
                || el_ty_str.contains("f16")
                || el_ty_str.contains("bf16");
            let mul_op_name = if is_float { "arith.mulf" } else { "arith.muli" };
            let add_op_name = if is_float { "arith.addf" } else { "arith.addi" };

            let mul_op = OperationBuilder::new(mul_op_name, gen.loc())
                .add_operands(&[
                    block_matmul.argument(0).unwrap().into(),
                    block_matmul.argument(1).unwrap().into(),
                ])
                .add_results(&[Type::parse(gen.context, el_ty_str).unwrap()])
                .build()
                .unwrap();
            let mul_val = block_matmul
                .append_operation(mul_op)
                .result(0)
                .unwrap()
                .into();

            let add_op = OperationBuilder::new(add_op_name, gen.loc())
                .add_operands(&[block_matmul.argument(2).unwrap().into(), mul_val])
                .add_results(&[Type::parse(gen.context, el_ty_str).unwrap()])
                .build()
                .unwrap();
            let add_val = block_matmul
                .append_operation(add_op)
                .result(0)
                .unwrap()
                .into();

            let yield_matmul = OperationBuilder::new("linalg.yield", gen.loc())
                .add_operands(&[add_val])
                .build()
                .unwrap();
            block_matmul.append_operation(yield_matmul);
            region_matmul.append_block(block_matmul);

            let matmul_op = OperationBuilder::new("linalg.matmul", gen.loc())
                .add_operands(&[lhs_val, rhs_val, out_val])
                .add_attributes(&[(
                    Identifier::new(gen.context, "operandSegmentSizes"),
                    DenseI32ArrayAttribute::new(gen.context, &[2, 1]).into(),
                )])
                .add_regions([region_matmul])
                .build()
                .unwrap();
            block.append_operation(matmul_op);

            return Ok((out_val, out_ty));
        } else if is_memref {
            // Element-wise Linalg Lowering (Add, Sub, Mul, Div)
            let out_ty = Type::parse(gen.context, &lhs_ty_str).unwrap();

            let mut alloc_operands = Vec::new();
            let index_ty = gen.index_ty;

            let rank = lhs_parts.len() - 1; // Last part is element type
            let el_ty_str = lhs_parts.last().unwrap();

            for (i, dim_str) in lhs_parts.iter().take(rank).enumerate() {
                if *dim_str == "?" {
                    let idx_attr = IntegerAttribute::new(Type::index(gen.context), i as i64).into();
                    let cst_op = OperationBuilder::new("arith.constant", gen.loc())
                        .add_results(&[index_ty])
                        .add_attributes(&[(Identifier::new(gen.context, "value"), idx_attr)])
                        .build()
                        .unwrap();
                    let idx_val = block.append_operation(cst_op).result(0).unwrap().into();

                    let dim_op = OperationBuilder::new("memref.dim", gen.loc())
                        .add_operands(&[lhs_val, idx_val])
                        .add_results(&[index_ty])
                        .build()
                        .unwrap();
                    alloc_operands.push(block.append_operation(dim_op).result(0).unwrap().into());
                }
            }

            // Alloc output buffer
            let alloc_op = OperationBuilder::new("memref.alloc", gen.loc())
                .add_operands(&alloc_operands)
                .add_attributes(&[(
                    Identifier::new(gen.context, "operandSegmentSizes"),
                    DenseI32ArrayAttribute::new(gen.context, &[alloc_operands.len() as i32, 0])
                        .into(),
                )])
                .add_results(&[out_ty])
                .build()
                .unwrap();
            let out_val = block.append_operation(alloc_op).result(0).unwrap().into();

            let op_name = match op {
                BinaryOp::Add => "linalg.add",
                BinaryOp::Sub => "linalg.sub",
                BinaryOp::Mul => "linalg.mul",
                BinaryOp::MatMul => panic!("MatMul must have been handled by is_matmul branch"),
                BinaryOp::Div => "linalg.div",
            };

            let is_float = el_ty_str.contains("f32")
                || el_ty_str.contains("f64")
                || el_ty_str.contains("f16")
                || el_ty_str.contains("bf16");
            let arith_op_name = op.get_op_name(is_float);

            let region = Region::new();
            let block_inner = melior::ir::Block::new(&[
                (Type::parse(gen.context, el_ty_str).unwrap(), gen.loc()),
                (Type::parse(gen.context, el_ty_str).unwrap(), gen.loc()),
                (Type::parse(gen.context, el_ty_str).unwrap(), gen.loc()),
            ]);

            let arith_op = OperationBuilder::new(arith_op_name, gen.loc())
                .add_operands(&[
                    block_inner.argument(0).unwrap().into(),
                    block_inner.argument(1).unwrap().into(),
                ])
                .add_results(&[Type::parse(gen.context, el_ty_str).unwrap()])
                .build()
                .unwrap();

            let arith_val = block_inner
                .append_operation(arith_op)
                .result(0)
                .unwrap()
                .into();

            let yield_op = OperationBuilder::new("linalg.yield", gen.loc())
                .add_operands(&[arith_val])
                .build()
                .unwrap();

            block_inner.append_operation(yield_op);
            region.append_block(block_inner);

            let linalg_op = OperationBuilder::new(op_name, gen.loc())
                .add_operands(&[lhs_val, rhs_val, out_val])
                .add_attributes(&[(
                    Identifier::new(gen.context, "operandSegmentSizes"),
                    DenseI32ArrayAttribute::new(gen.context, &[2, 1]).into(),
                )])
                .add_regions([region])
                .build()
                .unwrap();
            block.append_operation(linalg_op);

            return Ok((out_val, out_ty));
        }

        if lhs_ty != rhs_ty
            && ((lhs_ty_str == "index" && rhs_ty_str == "i32")
                || (lhs_ty_str == "i32" && rhs_ty_str == "index"))
        {
            if lhs_ty_str == "index" && rhs_ty_str == "i32" {
                let cast_op = OperationBuilder::new("arith.index_cast", gen.loc())
                    .add_operands(&[rhs_val])
                    .add_results(&[lhs_ty])
                    .build()
                    .unwrap();
                rhs_val = block.append_operation(cast_op).result(0).unwrap().into();
            } else {
                let cast_op = OperationBuilder::new("arith.index_cast", gen.loc())
                    .add_operands(&[lhs_val])
                    .add_results(&[rhs_ty])
                    .build()
                    .unwrap();
                lhs_val = block.append_operation(cast_op).result(0).unwrap().into();
                final_ty = rhs_ty;
            }
        }

        let is_float = final_ty.to_string().contains("f32")
            || final_ty.to_string().contains("f64")
            || final_ty.to_string().contains("f16")
            || final_ty.to_string().contains("bf16");

        let mut builder = OperationBuilder::new(op.get_op_name(is_float), gen.loc());
        builder = builder.add_operands(&[lhs_val, rhs_val]);

        let ret_ty = if let Some(pred_val) = op.get_predicate(is_float) {
            let i1_ty = gen.i1_ty;
            let i64_ty = gen.i64_ty;
            builder = builder.add_results(&[i1_ty]).add_attributes(&[(
                Identifier::new(gen.context, "predicate"),
                IntegerAttribute::new(i64_ty, pred_val).into(),
            )]);
            i1_ty
        } else {
            builder = builder.add_results(&[final_ty]);
            final_ty
        };

        let bin_op = builder.build().unwrap();
        let bin_ref = block.append_operation(bin_op);
        Ok((bin_ref.result(0).unwrap().into(), ret_ty))
    }
}

impl<'c> LowerToMelior<'c> for RelationalOpExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>), LowerError>;
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let RelationalOpExpr {
            lhs,
            op,
            rhs,
            span: _,
        } = self;
        let (lhs_val, lhs_ty) = gen.generate_expr(lhs, block)?;
        let prev_expected = gen.expected_type;
        gen.expected_type = Some(lhs_ty);
        let (mut rhs_val, rhs_ty) = gen.generate_expr(rhs, block)?;
        gen.expected_type = prev_expected;

        let _lhs_ty_str = lhs_ty.to_string();
        let _rhs_ty_str = rhs_ty.to_string();

        let mut final_ty = lhs_ty;

        if lhs_ty != rhs_ty {
            // Prioritize standard coercion depending on which type is more generic (e.g. f64 > f32 > i64 > i32)
            // For simplicity, just cast rhs to lhs for now.
            rhs_val = gen.coerce_type(block, rhs_val, rhs_ty, lhs_ty);
            _ = lhs_ty;
            final_ty = lhs_ty;
        }

        let is_float = final_ty.to_string().contains("f32")
            || final_ty.to_string().contains("f64")
            || final_ty.to_string().contains("f16")
            || final_ty.to_string().contains("bf16");

        let mut builder = OperationBuilder::new(op.get_op_name(is_float), gen.loc());
        builder = builder.add_operands(&[lhs_val, rhs_val]);

        let ret_ty = if let Some(pred_val) = op.get_predicate(is_float) {
            let i1_ty = gen.i1_ty;
            let i64_ty = gen.i64_ty;
            builder = builder.add_results(&[i1_ty]).add_attributes(&[(
                Identifier::new(gen.context, "predicate"),
                IntegerAttribute::new(i64_ty, pred_val).into(),
            )]);
            i1_ty
        } else {
            builder = builder.add_results(&[final_ty]);
            final_ty
        };

        let bin_op = builder.build().unwrap();
        let bin_ref = block.append_operation(bin_op);
        Ok((bin_ref.result(0).unwrap().into(), ret_ty))
    }
}

impl<'c> LowerToMelior<'c> for LogicalOpExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>), LowerError>;
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let LogicalOpExpr {
            lhs,
            op,
            rhs,
            span: _,
        } = self;
        let (lhs_val, _lhs_ty) = gen.generate_expr(lhs, block)?;
        let (rhs_val, _rhs_ty) = gen.generate_expr(rhs, block)?;

        let final_ty = gen.i1_ty;

        let builder = OperationBuilder::new(op.get_op_name(false), gen.loc())
            .add_operands(&[lhs_val, rhs_val])
            .add_results(&[final_ty]);

        let bin_op = builder.build().unwrap();
        let bin_ref = block.append_operation(bin_op);
        Ok((bin_ref.result(0).unwrap().into(), final_ty))
    }
}

impl<'c> LowerToMelior<'c> for ast::UnaryOpExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>), LowerError>;
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let ast::UnaryOpExpr { op, expr, span: _ } = self;
        let (val, ty) = gen.generate_expr(expr, block)?;
        match op {
            ast::UnaryOp::Not => {
                let true_val_op = OperationBuilder::new("arith.constant", gen.loc())
                    .add_results(&[ty])
                    .add_attributes(&[(
                        Identifier::new(gen.context, "value"),
                        IntegerAttribute::new(ty, 1).into(),
                    )])
                    .build()
                    .unwrap();
                let true_val_ref = block.append_operation(true_val_op);

                let not_op = OperationBuilder::new("arith.xori", gen.loc())
                    .add_operands(&[val, true_val_ref.result(0).unwrap().into()])
                    .add_results(&[ty])
                    .build()
                    .unwrap();

                let not_ref = block.append_operation(not_op);
                Ok((not_ref.result(0).unwrap().into(), ty))
            }
            ast::UnaryOp::Neg => {
                let is_float = ty.to_string().contains("f32")
                    || ty.to_string().contains("f64")
                    || ty.to_string().contains("f16")
                    || ty.to_string().contains("bf16");
                if is_float {
                    let neg_op = OperationBuilder::new("arith.negf", gen.loc())
                        .add_operands(&[val])
                        .add_results(&[ty])
                        .build()
                        .unwrap();
                    let neg_ref = block.append_operation(neg_op);
                    Ok((neg_ref.result(0).unwrap().into(), ty))
                } else {
                    let zero_op = OperationBuilder::new("arith.constant", gen.loc())
                        .add_results(&[ty])
                        .add_attributes(&[(
                            Identifier::new(gen.context, "value"),
                            IntegerAttribute::new(ty, 0).into(),
                        )])
                        .build()
                        .unwrap();
                    let zero_ref = block.append_operation(zero_op);

                    let neg_op = OperationBuilder::new("arith.subi", gen.loc())
                        .add_operands(&[zero_ref.result(0).unwrap().into(), val])
                        .add_results(&[ty])
                        .build()
                        .unwrap();
                    let neg_ref = block.append_operation(neg_op);
                    Ok((neg_ref.result(0).unwrap().into(), ty))
                }
            }
        }
    }
}

impl<'c> LowerToMelior<'c> for StructInitExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>), LowerError>;
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let StructInitExpr {
            name,
            fields,
            span: _,
        } = self;
        let base_name = name.split('<').next().unwrap_or(name).to_string();
        let struct_decl = gen
            .structs
            .get(&base_name)
            .unwrap_or_else(|| {
                panic!(
                    "Struct {} not found in gen.structs! Available: {:?}",
                    base_name,
                    gen.structs.keys().collect::<Vec<_>>()
                )
            })
            .clone();

        let mut mapping = std::collections::HashMap::new();
        let struct_ty = if name.contains('<') && name.ends_with('>') {
            let inner_ty_str = &name[name.find('<').unwrap() + 1..name.len() - 1];
            let mut inner_tys = Vec::new();
            for ty_arg_raw in inner_ty_str.split(',') {
                let ty_arg = ty_arg_raw.trim();
                let inner_ty = if ty_arg == "i32" {
                    ast::Type::Scalar(ast::ElementType::I32)
                } else if ty_arg == "f32" {
                    ast::Type::Scalar(ast::ElementType::F32)
                } else if ty_arg == "i64" {
                    ast::Type::Scalar(ast::ElementType::I64)
                } else if ty_arg.chars().all(|c| c.is_ascii_digit()) {
                    ast::Type::Const(Box::new(ast::Expr::Number(ast::expr::NumberExpr::new(
                        ty_arg.to_string(),
                        None,
                        ast::Span::default(),
                    ))))
                } else {
                    ast::Type::Struct(ty_arg.to_string(), None)
                };
                inner_tys.push(inner_ty);
            }
            for (i, param) in struct_decl.generics.iter().enumerate() {
                if i < inner_tys.len() {
                    mapping.insert(param.name().to_string(), inner_tys[i].clone());
                }
            }

            gen.lower_type(&ast::Type::GenericInstance(
                Box::new(ast::Type::Struct(base_name.clone(), None)),
                inner_tys,
            ))
        } else {
            gen.lower_type(&ast::Type::Struct(name.clone(), None))
        };

        let undef_op = OperationBuilder::new("llvm.mlir.undef", gen.loc())
            .add_results(&[struct_ty])
            .build()
            .unwrap();
        let mut current_struct = block.append_operation(undef_op).result(0).unwrap().into();

        for (field_name, f_expr) in fields {
            let field_idx = struct_decl
                .fields
                .iter()
                .position(|(n, _)| n == field_name)
                .unwrap();
            let sub_ty = struct_decl.fields[field_idx].1.substitute(&mapping);

            let field_ty = gen.lower_type(&sub_ty);
            let prev_expected = gen.expected_type;
            gen.expected_type = Some(field_ty);
            let (mut field_val, expr_ty) = gen.generate_expr(f_expr, block)?;
            gen.expected_type = prev_expected;

            if expr_ty != field_ty
                && ((expr_ty.to_string() == "index" && field_ty.to_string() == "i32")
                    || (expr_ty.to_string() == "i32" && field_ty.to_string() == "index"))
            {
                let cast_op = OperationBuilder::new("arith.index_cast", gen.loc())
                    .add_operands(&[field_val])
                    .add_results(&[field_ty])
                    .build()
                    .unwrap();
                field_val = block.append_operation(cast_op).result(0).unwrap().into();
            }

            let pos_attr = melior::ir::attribute::DenseI64ArrayAttribute::new(
                gen.context,
                &[field_idx as i64],
            );

            let insert_op = OperationBuilder::new("llvm.insertvalue", gen.loc())
                .add_operands(&[current_struct, field_val])
                .add_attributes(&[(Identifier::new(gen.context, "position"), pos_attr.into())])
                .add_results(&[struct_ty])
                .build()
                .unwrap();
            current_struct = block.append_operation(insert_op).result(0).unwrap().into();
        }
        Ok((current_struct, struct_ty))
    }
}

impl<'c> LowerToMelior<'c> for UnsafeBlockExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>), LowerError>;
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        for stmt in &self.stmts {
            gen.generate_statement(stmt, block)?;
        }
        if let Some(ret_expr) = &self.ret {
            gen.generate_expr(ret_expr, block)
        } else {
            // Return an i32 0 or something empty if no return type is expected.
            let i32_ty = gen.i32_ty;
            let zero_attr = IntegerAttribute::new(i32_ty, 0).into();
            let zero_op = OperationBuilder::new("arith.constant", gen.loc())
                .add_results(&[i32_ty])
                .add_attributes(&[(Identifier::new(gen.context, "value"), zero_attr)])
                .build()
                .unwrap();
            let zero_val = block.append_operation(zero_op).result(0).unwrap().into();
            Ok((zero_val, i32_ty))
        }
    }
}

impl<'c> LowerToMelior<'c> for MemberAccessExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>), LowerError>;
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let MemberAccessExpr {
            base,
            member,
            struct_name,
            span: _,
        } = self;
        let (base_val, base_ty) = gen.generate_expr(base, block)?;
        let base_ty_str = base_ty.to_string();

        let mut struct_name_opt = struct_name.clone();
        if struct_name_opt.is_none() {
            if let Some(start_idx) = base_ty_str.find('\"') {
                if let Some(end_idx) = base_ty_str[start_idx + 1..].find('"') {
                    struct_name_opt =
                        Some(base_ty_str[start_idx + 1..start_idx + 1 + end_idx].to_string());
                }
            }
        }

        let is_ptr = base_ty_str.starts_with("!llvm.ptr");

        if let Some(ref resolved_struct_name) = struct_name_opt {
            let base_name = resolved_struct_name
                .split('<')
                .next()
                .unwrap_or(resolved_struct_name.as_str())
                .to_string();
            let mut mapping = std::collections::HashMap::new();
            if resolved_struct_name.contains('<') && resolved_struct_name.ends_with('>') {
                let inner_ty_str = &resolved_struct_name
                    [resolved_struct_name.find('<').unwrap() + 1..resolved_struct_name.len() - 1];
                let mut inner_tys = Vec::new();
                for ty_arg_raw in inner_ty_str.split(',') {
                    let ty_arg = ty_arg_raw.trim();
                    let inner_ty = if ty_arg == "i32" {
                        ast::Type::Scalar(ast::ElementType::I32)
                    } else if ty_arg == "f32" {
                        ast::Type::Scalar(ast::ElementType::F32)
                    } else if ty_arg == "i64" {
                        ast::Type::Scalar(ast::ElementType::I64)
                    } else if ty_arg.chars().all(|c| c.is_ascii_digit()) {
                        ast::Type::Const(Box::new(ast::Expr::Number(ast::expr::NumberExpr::new(
                            ty_arg.to_string(),
                            None,
                            ast::Span::default(),
                        ))))
                    } else {
                        ast::Type::Struct(ty_arg.to_string(), None)
                    };
                    inner_tys.push(inner_ty);
                }
                if let Some(struct_decl) = gen.structs.get(&base_name) {
                    for (i, param) in struct_decl.generics.iter().enumerate() {
                        if i < inner_tys.len() {
                            mapping.insert(param.name().to_string(), inner_tys[i].clone());
                        }
                    }
                }
            }

            if let Some(struct_decl) = gen.structs.get(&base_name).cloned() {
                if let Some(field_idx) = struct_decl.fields.iter().position(|(n, _)| n == member) {
                    let sub_ty = struct_decl.fields[field_idx].1.substitute(&mapping);
                    let field_ty = gen.lower_type(&sub_ty);

                    if is_ptr {
                        let ptr_ty = gen.ptr_ty;
                        let mut field_types = Vec::new();
                        for (_, ty) in &struct_decl.fields {
                            let sub_ty2 = ty.substitute(&mapping);
                            let mut lowered = gen.lower_type_str(&sub_ty2);
                            if lowered.starts_with("memref<") {
                                lowered = "!llvm.ptr".to_string();
                            }
                            field_types.push(lowered);
                        }
                        let struct_llvm_ty_str = format!(
                            "!llvm.struct<\"{}\", ({})>",
                            base_name,
                            field_types.join(", ")
                        );
                        let struct_llvm_ty = Type::parse(gen.context, &struct_llvm_ty_str).unwrap();

                        let gep_op = OperationBuilder::new("llvm.getelementptr", gen.loc())
                            .add_operands(&[base_val])
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
                            .add_results(&[ptr_ty])
                            .build()
                            .unwrap();
                        let gep_ref = block.append_operation(gep_op);
                        let field_ptr = gep_ref.result(0).unwrap().into();

                        if gen.is_lvalue_context {
                            return Ok((field_ptr, ptr_ty));
                        }

                        let load_op = OperationBuilder::new("llvm.load", gen.loc())
                            .add_operands(&[field_ptr])
                            .add_results(&[field_ty])
                            .build()
                            .unwrap();
                        let load_ref = block.append_operation(load_op);
                        return Ok((load_ref.result(0).unwrap().into(), field_ty));
                    } else {
                        let pos_attr = melior::ir::attribute::DenseI64ArrayAttribute::new(
                            gen.context,
                            &[field_idx as i64],
                        );

                        let ext_op = OperationBuilder::new("llvm.extractvalue", gen.loc())
                            .add_operands(&[base_val])
                            .add_attributes(&[(
                                Identifier::new(gen.context, "position"),
                                pos_attr.into(),
                            )])
                            .add_results(&[field_ty])
                            .build()
                            .unwrap();
                        let ext_ref = block.append_operation(ext_op);
                        return Ok((ext_ref.result(0).unwrap().into(), field_ty));
                    }
                }
            }
        }
        println!(
            "DEBUG PANIC: member={}, struct_name_opt={:?}, base_ty_str={}, base_val={:?}",
            member, struct_name_opt, base_ty_str, base_val
        );
        if let Some(name) = &struct_name_opt {
            let base_name = name.split('<').next().unwrap_or(name);
            println!(
                "DEBUG PANIC: gen.structs.get({:?}) = {:?}",
                base_name,
                gen.structs.get(base_name)
            );
        }
        panic!("Cannot resolve member access {}", member);
    }
}

impl<'c> LowerToMelior<'c> for FunctionCallExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>), LowerError>;
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let FunctionCallExpr {
            name,
            args,
            type_args,
            span: _,
        } = self;
        if name == "Verified" {
            return gen.generate_expr(&args[0], block);
        }
        if name == "Tensor" {
            let mlir_ty_str = if let Some(tys) = type_args {
                if !tys.is_empty() {
                    gen.lower_type_str(&tys[0])
                } else {
                    panic!("Tensor initialization requires an explicit generic type argument");
                }
            } else {
                panic!("Tensor initialization requires an explicit generic type argument");
            };
            let mut dynamic_sizes = Vec::new();
            let mut dims_count = 2; // Default fallback

            if args.len() == 1 {
                if let Expr::Array(arr) = &args[0] {
                    dims_count = arr.elements.len();
                    for el in &arr.elements {
                        let (mut val, ty) = gen.generate_expr(el, block)?;
                        if ty.to_string() != "index" {
                            let cast_op = OperationBuilder::new("arith.index_cast", gen.loc())
                                .add_operands(&[val])
                                .add_results(&[Type::index(gen.context)])
                                .build()
                                .unwrap();
                            val = block.append_operation(cast_op).result(0).unwrap().into();
                        }
                        dynamic_sizes.push(val);
                    }
                }
            } else if !args.is_empty() {
                dims_count = args.len();
                for el in args {
                    let (mut val, ty) = gen.generate_expr(el, block)?;
                    if ty.to_string() != "index" {
                        let cast_op = OperationBuilder::new("arith.index_cast", gen.loc())
                            .add_operands(&[val])
                            .add_results(&[Type::index(gen.context)])
                            .build()
                            .unwrap();
                        val = block.append_operation(cast_op).result(0).unwrap().into();
                    }
                    dynamic_sizes.push(val);
                }
            }

            let mut shape_str = String::new();
            for _ in 0..dims_count {
                shape_str.push_str("?x");
            }
            let tensor_ty_str = format!("memref<{}{}>", shape_str, mlir_ty_str);
            let tensor_ty = Type::parse(gen.context, &tensor_ty_str).unwrap();

            let alloc_op = OperationBuilder::new("memref.alloc", gen.loc())
                .add_operands(&dynamic_sizes)
                .add_attributes(&[(
                    Identifier::new(gen.context, "operandSegmentSizes"),
                    DenseI32ArrayAttribute::new(gen.context, &[dynamic_sizes.len() as i32, 0])
                        .into(),
                )])
                .add_results(&[tensor_ty])
                .build()
                .unwrap();
            let alloc_ref = block.append_operation(alloc_op);
            return Ok((alloc_ref.result(0).unwrap().into(), tensor_ty));
        }

        if name == "reshape" || name == "transpose" {
            let (arg_val, expr_ty) = gen.generate_expr(&args[0], block)?;
            let expr_ty_str = expr_ty.to_string();

            // Extract element type
            let el_ty_str =
                extract_mlir_element_type(&expr_ty_str).unwrap_or_else(|e| panic!("{}", e));

            let mut shape_str = String::new();
            if let Expr::Array(arr) = &args[1] {
                for el in &arr.elements {
                    if let Expr::Number(num) = el {
                        shape_str.push_str(&num.value);
                        shape_str.push('x');
                    }
                }
            } else {
                shape_str.push_str("*x"); // unranked fallback
            }

            let target_ty_str = if shape_str == "*x" {
                format!("memref<*x{}>", el_ty_str)
            } else {
                format!("memref<{}{}>", shape_str, el_ty_str)
            };

            let unranked_ty_str = format!("memref<*x{}>", el_ty_str);
            let unranked_ty = Type::parse(gen.context, &unranked_ty_str).unwrap();

            // Cast to unranked first
            let cast1_op = OperationBuilder::new("memref.cast", gen.loc())
                .add_operands(&[arg_val])
                .add_results(&[unranked_ty])
                .build()
                .unwrap();
            let cast1_ref = block.append_operation(cast1_op);
            let unranked_val = cast1_ref.result(0).unwrap().into();

            let target_ty = Type::parse(gen.context, &target_ty_str).unwrap();

            // Cast to targeted shape
            let cast2_op = OperationBuilder::new("memref.cast", gen.loc())
                .add_operands(&[unranked_val])
                .add_results(&[target_ty])
                .build()
                .unwrap();

            let cast2_ref = block.append_operation(cast2_op);
            return Ok((cast2_ref.result(0).unwrap().into(), target_ty));
        }

        if name == "with_memory" {
            // For now, with_memory is a no-op in lowering, just returns the tensor
            let (arg_val, expr_ty) = gen.generate_expr(&args[0], block)?;
            return Ok((arg_val, expr_ty));
        }

        if name == "map" {
            return lower_map_call(gen, block, args);
        }

        if name == "print" {
            return lower_print_call(gen, block, args);
        }

        if name == "printf" || name == "vx_internal_printf" {
            let mut arg_vals = Vec::new();
            for arg in args {
                let (arg_val, _arg_ty) = gen.generate_expr(arg, block)?;
                arg_vals.push(arg_val);
            }

            let printf_func_ty = TypeAttribute::new(
                Type::parse(gen.context, "!llvm.func<i32 (!llvm.ptr, ...)>").unwrap(),
            );

            // Iterate over operations to check if printf is already declared
            // We can't iterate easily over body in melior, but we can just use a separate set for llvm_funcs.
            // Wait, we can just check a boolean in gen or insert "llvm_printf" into gen.functions.
            // Let's just forcefully insert it if we haven't seen it in gen.functions, BUT wait,
            // gen.functions has "printf" from the AST extern. So let's use a unique key for the llvm decl.
            if !gen.functions.contains_key("llvm_printf_decl") {
                let printf_decl = OperationBuilder::new("llvm.func", gen.loc())
                    .add_attributes(&[
                        (
                            Identifier::new(gen.context, "sym_name"),
                            StringAttribute::new(gen.context, "printf").into(),
                        ),
                        (
                            Identifier::new(gen.context, "linkage_name"),
                            StringAttribute::new(gen.context, "printf").into(),
                        ),
                        (
                            Identifier::new(gen.context, "function_type"),
                            printf_func_ty.into(),
                        ),
                    ])
                    .add_regions([melior::ir::Region::new()])
                    .build()
                    .unwrap();

                gen.module.body().append_operation(printf_decl);
                gen.functions.insert(
                    "llvm_printf_decl".to_string(),
                    (gen.i32_ty, vec![gen.ptr_ty]),
                );
            }

            let call_op = block.append_operation(
                OperationBuilder::new("llvm.call", gen.loc())
                    .add_operands(&arg_vals)
                    .add_attributes(&[
                        (
                            Identifier::new(gen.context, "callee"),
                            FlatSymbolRefAttribute::new(gen.context, "printf").into(),
                        ),
                        (
                            Identifier::new(gen.context, "var_callee_type"),
                            printf_func_ty.into(),
                        ),
                        (
                            Identifier::new(gen.context, "operandSegmentSizes"),
                            DenseI32ArrayAttribute::new(gen.context, &[arg_vals.len() as i32, 0])
                                .into(),
                        ),
                        (
                            Identifier::new(gen.context, "op_bundle_sizes"),
                            DenseI32ArrayAttribute::new(gen.context, &[]).into(),
                        ),
                    ])
                    .add_results(&[gen.i32_ty])
                    .build()
                    .unwrap(),
            );

            return Ok((call_op.result(0).unwrap().into(), gen.i32_ty));
        }

        if let Some((ret_ty, arg_tys)) = gen.functions.get(name).cloned() {
            let mut arg_vals = Vec::new();
            for (i, arg) in args.iter().enumerate() {
                let (mut arg_val, expr_ty) = gen.generate_expr(arg, block)?;
                let field_ty = arg_tys[i];
                if expr_ty != field_ty {
                    if gen.is_memref(&expr_ty) && gen.is_memref(&field_ty) {
                        let cast_op = OperationBuilder::new("memref.cast", gen.loc())
                            .add_operands(&[arg_val])
                            .add_results(&[field_ty])
                            .build()
                            .unwrap();
                        arg_val = block.append_operation(cast_op).result(0).unwrap().into();
                    } else {
                        arg_val = gen.coerce_type(block, arg_val, expr_ty, field_ty);
                    }
                }
                arg_vals.push(arg_val);
            }

            let name_attr = FlatSymbolRefAttribute::new(gen.context, name);
            let mut builder = OperationBuilder::new("func.call", gen.loc())
                .add_operands(&arg_vals)
                .add_attributes(&[(Identifier::new(gen.context, "callee"), name_attr.into())]);

            if ret_ty.to_string() != "none" {
                builder = builder.add_results(&[ret_ty]);
                let call_op = builder.build().unwrap();
                let call_ref = block.append_operation(call_op);
                Ok((call_ref.result(0).unwrap().into(), ret_ty))
            } else {
                let call_op = builder.build().unwrap();
                block.append_operation(call_op);
                let none_ty = gen.none_ty;
                // this value shouldn't be used
                let dummy_op = OperationBuilder::new("llvm.mlir.constant", gen.loc())
                    .add_results(&[gen.i32_ty])
                    .add_attributes(&[(
                        Identifier::new(gen.context, "value"),
                        IntegerAttribute::new(gen.i32_ty, 0).into(),
                    )])
                    .build()
                    .unwrap();
                Ok((
                    block.append_operation(dummy_op).result(0).unwrap().into(),
                    none_ty,
                ))
            }
        } else if let Some((ptr_val, func_ty)) = gen.env.get(name).cloned() {
            let mut actual_func_ty = func_ty;
            let is_closure = func_ty.to_string() == "!llvm.struct<(ptr, ptr)>";
            if func_ty.to_string() == "!llvm.ptr" {
                if let Some(ast::Type::Function(func_args, ret)) = gen.ast_env.get(name) {
                    println!("Lowering function pointer ret type for name={}", name);
                    let r = gen.lower_type(ret.as_ref());
                    let a: Vec<_> = func_args.iter().map(|t| gen.lower_type(t)).collect();
                    actual_func_ty =
                        melior::ir::r#type::FunctionType::new(gen.context, &a, &[r]).into();
                } else {
                    panic!("Missing signature for function pointer '{}'", name);
                }
            } else if is_closure {
                if let Some(ast::Type::Closure(func_args, ret)) = gen.ast_env.get(name) {
                    let r = gen.lower_type(ret.as_ref());
                    let mut a: Vec<_> = vec![gen.ptr_ty];
                    a.extend(func_args.iter().map(|t| gen.lower_type(t)));
                    actual_func_ty =
                        melior::ir::r#type::FunctionType::new(gen.context, &a, &[r]).into();
                } else {
                    panic!("Missing signature for closure '{}'", name);
                }
            }
            if let Ok(mlir_func_ty) = melior::ir::r#type::FunctionType::try_from(actual_func_ty) {
                let ret_ty = mlir_func_ty.result(0).unwrap();
                let mut arg_vals = Vec::new();

                let mut arg_offset = 0;
                let mut env_ptr = None;
                let mut actual_ptr_val = ptr_val;

                if is_closure {
                    arg_offset = 1;
                    // Extract env_ptr
                    let extract_env_op = OperationBuilder::new("llvm.extractvalue", gen.loc())
                        .add_operands(&[ptr_val])
                        .add_attributes(&[(
                            Identifier::new(gen.context, "position"),
                            melior::ir::attribute::DenseI64ArrayAttribute::new(gen.context, &[0])
                                .into(),
                        )])
                        .add_results(&[gen.ptr_ty])
                        .build()
                        .unwrap();
                    let extract_env_ref = block.append_operation(extract_env_op);
                    env_ptr = Some(extract_env_ref.result(0).unwrap().into());

                    // Extract func_ptr
                    let extract_func_op = OperationBuilder::new("llvm.extractvalue", gen.loc())
                        .add_operands(&[ptr_val])
                        .add_attributes(&[(
                            Identifier::new(gen.context, "position"),
                            melior::ir::attribute::DenseI64ArrayAttribute::new(gen.context, &[1])
                                .into(),
                        )])
                        .add_results(&[gen.ptr_ty])
                        .build()
                        .unwrap();
                    let extract_func_ref = block.append_operation(extract_func_op);
                    actual_ptr_val = extract_func_ref.result(0).unwrap().into();
                }

                for (i, arg) in args.iter().enumerate() {
                    let (mut arg_val, expr_ty) = gen.generate_expr(arg, block)?;
                    let field_ty = mlir_func_ty.input(i + arg_offset).unwrap();
                    if expr_ty != field_ty {
                        arg_val = gen.coerce_type(block, arg_val, expr_ty, field_ty);
                    }
                    arg_vals.push(arg_val);
                }

                if func_ty.to_string() == "!llvm.ptr" || is_closure {
                    let cast_op =
                        OperationBuilder::new("builtin.unrealized_conversion_cast", gen.loc())
                            .add_operands(&[actual_ptr_val])
                            .add_results(&[actual_func_ty])
                            .build()
                            .unwrap();
                    let cast_ref = block.append_operation(cast_op);
                    actual_ptr_val = cast_ref.result(0).unwrap().into();
                }

                let mut builder = OperationBuilder::new("func.call_indirect", gen.loc())
                    .add_operands(&[actual_ptr_val]);

                if let Some(ep) = env_ptr {
                    builder = builder.add_operands(&[ep]);
                }

                for a in &arg_vals {
                    builder = builder.add_operands(&[*a]);
                }

                if ret_ty.to_string() != "none" {
                    builder = builder.add_results(&[ret_ty]);
                    let call_op = builder.build().unwrap();
                    let call_ref = block.append_operation(call_op);
                    Ok((call_ref.result(0).unwrap().into(), ret_ty))
                } else {
                    let call_op = builder.build().unwrap();
                    block.append_operation(call_op);
                    let none_ty = gen.none_ty;
                    let dummy_op = OperationBuilder::new("llvm.mlir.constant", gen.loc())
                        .add_results(&[gen.i32_ty])
                        .add_attributes(&[(
                            Identifier::new(gen.context, "value"),
                            IntegerAttribute::new(gen.i32_ty, 0).into(),
                        )])
                        .build()
                        .unwrap();
                    Ok((
                        block.append_operation(dummy_op).result(0).unwrap().into(),
                        none_ty,
                    ))
                }
            } else {
                panic!(
                    "Function pointer {} missing type info. func_ty={}",
                    name, func_ty
                );
            }
        } else {
            panic!("Function {} not found", name);
        }
    }
}

impl<'c> LowerToMelior<'c> for IndirectCallExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>), LowerError>;
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let IndirectCallExpr {
            callee,
            args,
            target_func_ty,
            span: _,
        } = self;

        let (callee_val, callee_ty) = gen.generate_expr(callee, block)?;

        if callee_ty.to_string() == "!llvm.struct<(ptr, ptr)>" {
            // Extract env_ptr
            let extract_env_op = OperationBuilder::new("llvm.extractvalue", gen.loc())
                .add_operands(&[callee_val])
                .add_attributes(&[(
                    Identifier::new(gen.context, "position"),
                    melior::ir::attribute::DenseI64ArrayAttribute::new(gen.context, &[0]).into(),
                )])
                .add_results(&[gen.ptr_ty])
                .build()
                .unwrap();
            let extract_env_ref = block.append_operation(extract_env_op);
            let env_ptr = extract_env_ref.result(0).unwrap().into();

            // Extract func_ptr
            let extract_func_op = OperationBuilder::new("llvm.extractvalue", gen.loc())
                .add_operands(&[callee_val])
                .add_attributes(&[(
                    Identifier::new(gen.context, "position"),
                    melior::ir::attribute::DenseI64ArrayAttribute::new(gen.context, &[1]).into(),
                )])
                .add_results(&[gen.ptr_ty])
                .build()
                .unwrap();
            let extract_func_ref = block.append_operation(extract_func_op);
            let mut actual_ptr_val = extract_func_ref.result(0).unwrap().into();

            let mut arg_vals = Vec::new();
            for arg in args {
                let (arg_val, _) = gen.generate_expr(arg, block)?;
                arg_vals.push(arg_val);
            }

            // In our current implementation we assume all functions taking closure
            // arguments are dynamically typed correctly via Sema. We don't have
            // the exact MLIR FunctionType statically available here without
            // looking it up, but `func.call_indirect` requires the operand to
            // match the callee type exactly. Since we don't have the explicit
            // func signature here, we construct it from the generated arg types!
            // To figure out the return type, we use `target_func_ty` from Sema.
            let target_func_ty = target_func_ty
                .as_ref()
                .expect("IndirectCallExpr MLIR lowering needs explicit target_func_ty from Sema");
            let ast::Type::Closure(func_args, ret) = target_func_ty else {
                panic!("Expected Type::Closure for indirect call fat pointer target");
            };

            let r = gen.lower_type(ret);
            let mut a: Vec<_> = vec![gen.ptr_ty];
            a.extend(func_args.iter().map(|t| gen.lower_type(t)));
            let actual_mlir_func_ty = melior::ir::r#type::FunctionType::new(gen.context, &a, &[r]);
            let actual_func_ty: melior::ir::Type = actual_mlir_func_ty.into();

            let ret_ty = actual_mlir_func_ty.result(0).unwrap();

            // Cast the raw func ptr to the actual function signature
            let cast_op = OperationBuilder::new("builtin.unrealized_conversion_cast", gen.loc())
                .add_operands(&[actual_ptr_val])
                .add_results(&[actual_func_ty])
                .build()
                .unwrap();
            let cast_ref = block.append_operation(cast_op);
            actual_ptr_val = cast_ref.result(0).unwrap().into();

            let mut builder = OperationBuilder::new("func.call_indirect", gen.loc())
                .add_operands(&[actual_ptr_val, env_ptr]);

            for a_val in &arg_vals {
                builder = builder.add_operands(&[*a_val]);
            }

            if ret_ty.to_string() != "none" {
                builder = builder.add_results(&[ret_ty]);
                let call_op = builder.build().unwrap();
                let call_ref = block.append_operation(call_op);
                Ok((call_ref.result(0).unwrap().into(), ret_ty))
            } else {
                let call_op = builder.build().unwrap();
                block.append_operation(call_op);
                let none_ty = gen.none_ty;
                let dummy_op = OperationBuilder::new("llvm.mlir.constant", gen.loc())
                    .add_results(&[gen.i32_ty])
                    .add_attributes(&[(
                        Identifier::new(gen.context, "value"),
                        IntegerAttribute::new(gen.i32_ty, 0).into(),
                    )])
                    .build()
                    .unwrap();
                Ok((
                    block.append_operation(dummy_op).result(0).unwrap().into(),
                    none_ty,
                ))
            }
        } else {
            panic!("Unsupported callee type for indirect call: {}", callee_ty);
        }
    }
}

impl<'c> LowerToMelior<'c> for MethodCallExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>), LowerError>;
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let MethodCallExpr {
            base,
            method_name,
            args,
            type_args: _,
            span,
        } = self;
        let mut new_args = vec![*base.clone()];
        new_args.extend(args.clone());
        gen.generate_expr(
            &Expr::FunctionCall(FunctionCallExpr {
                name: method_name.clone(),
                type_args: None,
                args: new_args,
                span: *span,
            }),
            block,
        )
    }
}

impl<'c> LowerToMelior<'c> for InlineMlirExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>), LowerError>;

    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let mut mlir_args = Vec::new();
        let mut input_types_str = Vec::new();

        for (name, expr, ty_str) in &self.inputs {
            let (mut val, val_ty) = gen.generate_expr(expr, block)?;
            if val_ty.to_string().starts_with("memref<memref<") && ty_str.starts_with("memref<") {
                let load_op = OperationBuilder::new("memref.load", gen.loc())
                    .add_operands(&[val])
                    .add_results(&[Type::parse(gen.context, ty_str).unwrap()])
                    .build()
                    .unwrap();
                val = block.append_operation(load_op).result(0).unwrap().into();
            }
            mlir_args.push(val);
            input_types_str.push(format!("{}: {}", name, ty_str));
        }

        let args_str = input_types_str.join(", ");

        let mut ret_str = String::new();
        let mut call_ret_tys = Vec::new();
        if let Some(ret_ty) = &self.returns {
            let mlir_ret_ty = gen.lower_type(ret_ty);
            if mlir_ret_ty.to_string() != "void" {
                ret_str = format!("-> {}", mlir_ret_ty);
            }
            call_ret_tys.push(mlir_ret_ty);
        }

        let unique_id = format!(
            "vx_macro_mlir_L{}_C{}_{}",
            self.span.line, self.span.column, gen.mlir_block_counter
        );
        gen.mlir_block_counter += 1;

        let block_str = self.block_str.replace("macro.yield", "return");

        let mlir_source = format!(
            "module {{\n    func.func private @{}({}) {} {{\n{}\n    }}\n}}",
            unique_id, args_str, ret_str, block_str
        );

        gen.context.set_allow_unregistered_dialects(true);
        let parsed_module = match melior::ir::Module::parse(gen.context, &mlir_source) {
            Some(m) => m,
            None => {
                panic!(
                    "Failed to parse mlir! block at {:?}. Source:\n{}",
                    self.span, mlir_source
                );
            }
        };

        use melior::ir::BlockLike;
        let func_op = parsed_module.body().first_operation().unwrap();
        let cloned_func = (*func_op).clone();

        gen.module.body().append_operation(cloned_func);

        let call_op = OperationBuilder::new("func.call", gen.loc())
            .add_attributes(&[(
                Identifier::new(gen.context, "callee"),
                FlatSymbolRefAttribute::new(gen.context, &unique_id).into(),
            )])
            .add_operands(&mlir_args)
            .add_results(&call_ret_tys)
            .build()
            .unwrap();

        let op = block.append_operation(call_op);

        if let Some(ret_ty) = &self.returns {
            Ok((op.result(0).unwrap().into(), gen.lower_type(ret_ty)))
        } else {
            let dummy_val = OperationBuilder::new("arith.constant", gen.loc())
                .add_attributes(&[(
                    Identifier::new(gen.context, "value"),
                    IntegerAttribute::new(Type::index(gen.context), 0).into(),
                )])
                .add_results(&[Type::index(gen.context)])
                .build()
                .unwrap();
            Ok((
                block.append_operation(dummy_val).result(0).unwrap().into(),
                Type::index(gen.context),
            ))
        }
    }
}

impl<'c> LowerToMelior<'c> for ArrayExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>), LowerError>;
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let ArrayExpr { elements, span: _ } = self;
        if elements.is_empty() {
            panic!("Empty arrays not supported yet");
        }
        let mut vals = Vec::new();
        let mut el_ty = None;
        for el in elements {
            let (v, t) = gen.generate_expr(el, block)?;
            vals.push(v);
            if el_ty.is_none() {
                el_ty = Some(t);
            }
        }
        let el_ty = el_ty.unwrap();
        let num_elements = elements.len();

        let tensor_ty =
            Type::parse(gen.context, &format!("tensor<{}x{}>", num_elements, el_ty)).unwrap();

        let op = OperationBuilder::new("tensor.from_elements", gen.loc())
            .add_operands(&vals)
            .add_results(&[tensor_ty])
            .build()
            .unwrap();

        let val = block.append_operation(op).result(0).unwrap().into();
        Ok((val, tensor_ty))
    }
}

impl<'c> LowerToMelior<'c> for MemorySpaceExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>), LowerError>;
    fn lower(
        &self,
        _gen: &mut MeliorGenerator<'c>,
        _block: &melior::ir::Block<'c>,
    ) -> Self::Output {
        panic!("Should not be evaluated directly")
    }
}

impl<'c> LowerToMelior<'c> for TopologyExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>), LowerError>;
    fn lower(
        &self,
        _gen: &mut MeliorGenerator<'c>,
        _block: &melior::ir::Block<'c>,
    ) -> Self::Output {
        panic!("Should not be evaluated directly")
    }
}

impl<'c> LowerToMelior<'c> for NumberExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>), LowerError>;
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let NumberExpr {
            value: val_str,
            ty: ast_ty_opt,
            span: _,
        } = self;
        let ty = if let Some(ast_ty) = ast_ty_opt {
            gen.lower_type(&ast::Type::Scalar(ast_ty.clone()))
        } else if val_str.contains('.') {
            gen.f32_ty
        } else {
            gen.i32_ty
        };
        let ty_str = ty.to_string();
        if ty_str.contains("f32")
            || ty_str.contains("f64")
            || ty_str.contains("f16")
            || ty_str.contains("bf16")
        {
            let op = OperationBuilder::new("arith.constant", gen.loc())
                .add_results(&[ty])
                .add_attributes(&[(
                    Identifier::new(gen.context, "value"),
                    FloatAttribute::new(gen.context, ty, val_str.parse::<f64>().unwrap()).into(),
                )])
                .build()
                .unwrap();
            let op_ref = block.append_operation(op);
            Ok((op_ref.result(0).unwrap().into(), ty))
        } else {
            let op = OperationBuilder::new("arith.constant", gen.loc())
                .add_results(&[ty])
                .add_attributes(&[(
                    Identifier::new(gen.context, "value"),
                    IntegerAttribute::new(ty, val_str.parse::<i64>().unwrap()).into(),
                )])
                .build()
                .unwrap();
            let op_ref = block.append_operation(op);
            Ok((op_ref.result(0).unwrap().into(), ty))
        }
    }
}

impl<'c> LowerToMelior<'c> for MatchExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>), LowerError>;
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let (match_val, match_ty) = gen.generate_expr(&self.expr, block)?;

        generate_match_chain(gen, &self.arms, match_val, match_ty, block)?;

        // Return dummy value for now like IfExpr
        let ty = melior::ir::r#type::IntegerType::new(gen.context, 32).into();
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

impl<'c> LowerToMelior<'c> for EnumVariantExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>), LowerError>;
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let EnumVariantExpr {
            enum_name,
            variant_name,
            payload,
            span: _,
        } = self;

        let mut tag_val = 0;
        let mut enum_ty_str = "i32".to_string();
        let mut has_payload = false;

        let actual_enum_name = if let Some(idx) = enum_name.find('<') {
            &enum_name[0..idx]
        } else {
            enum_name
        };

        if let Some(enum_def) = gen.enums.get(actual_enum_name) {
            for (i, v) in enum_def.iter().enumerate() {
                if v.0 == *variant_name {
                    tag_val = i as i64;
                    break;
                }
            }
            if enum_name.starts_with("Option<") {
                if let Some(idx) = enum_name.find('<') {
                    if let Some(end_idx) = enum_name.find('>') {
                        let base = &enum_name[..idx];
                        let ty_arg = &enum_name[idx + 1..end_idx];
                        let parsed_ty = match ty_arg {
                            "i32" => ast::Type::Scalar(ast::ElementType::I32),
                            "f32" => ast::Type::Scalar(ast::ElementType::F32),
                            "f64" => ast::Type::Scalar(ast::ElementType::F64),
                            "i64" => ast::Type::Scalar(ast::ElementType::I64),
                            "Bool" => ast::Type::Scalar(ast::ElementType::Bool),
                            _ => ast::Type::Struct(ty_arg.to_string(), None),
                        };
                        let t = ast::Type::GenericInstance(
                            Box::new(ast::Type::Struct(base.to_string(), None)),
                            vec![parsed_ty],
                        );
                        enum_ty_str = gen.lower_type_str(&t);
                    }
                }
                has_payload = true;
            }
        }

        let i32_ty = gen.i32_ty;
        let tag_op = OperationBuilder::new("arith.constant", gen.loc())
            .add_results(&[i32_ty])
            .add_attributes(&[(
                Identifier::new(gen.context, "value"),
                IntegerAttribute::new(i32_ty, tag_val).into(),
            )])
            .build()
            .unwrap();
        let tag_val = block.append_operation(tag_op).result(0).unwrap().into();

        println!(
            "EnumVariantExpr: enum_name={}, variant={}, has_payload={}",
            enum_name, variant_name, has_payload
        );

        if !has_payload {
            return Ok((tag_val, i32_ty));
        }

        // We have an Option<T> struct
        let struct_ty = Type::parse(gen.context, &enum_ty_str).unwrap();
        let undef_op = OperationBuilder::new("llvm.mlir.undef", gen.loc())
            .add_results(&[struct_ty])
            .build()
            .unwrap();
        let undef_val = block.append_operation(undef_op).result(0).unwrap().into();

        let insert_tag_op = OperationBuilder::new("llvm.insertvalue", gen.loc())
            .add_operands(&[undef_val, tag_val])
            .add_results(&[struct_ty])
            .add_attributes(&[(
                Identifier::new(gen.context, "position"),
                melior::ir::attribute::DenseI64ArrayAttribute::new(gen.context, &[0]).into(),
            )])
            .build()
            .unwrap();
        let mut struct_val = block
            .append_operation(insert_tag_op)
            .result(0)
            .unwrap()
            .into();

        if let Some(payload_exprs) = payload {
            if !payload_exprs.is_empty() {
                let (payload_val, _payload_ty) = gen.generate_expr(&payload_exprs[0], block)?;
                let insert_payload_op = OperationBuilder::new("llvm.insertvalue", gen.loc())
                    .add_operands(&[struct_val, payload_val])
                    .add_results(&[struct_ty])
                    .add_attributes(&[(
                        Identifier::new(gen.context, "position"),
                        melior::ir::attribute::DenseI64ArrayAttribute::new(gen.context, &[1])
                            .into(),
                    )])
                    .build()
                    .unwrap();
                struct_val = block
                    .append_operation(insert_payload_op)
                    .result(0)
                    .unwrap()
                    .into();
            }
        }

        Ok((struct_val, struct_ty))
    }
}

impl<'c> LowerToMelior<'c> for VecMacroExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>), LowerError>;
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let mut el_ty = ast::ElementType::F32;
        if !self.elements.is_empty() {
            if let Some(ast::Type::Scalar(t)) = gen.infer_ast_type(&self.elements[0]) {
                el_ty = t;
            } else if let Some(ast::Type::Struct(s, _)) = gen.infer_ast_type(&self.elements[0]) {
                if s == "String" {
                    // String is equivalent to pointer, but generic instantiation requires element type
                }
            }
        }

        let type_suffix = match el_ty {
            ast::ElementType::I32 => "i32",
            ast::ElementType::F32 => "f32",
            ast::ElementType::I64 => "i64",
            ast::ElementType::F64 => "f64",
            ast::ElementType::Bool => "Bool",
            _ => {
                if let Some(ast::Type::Struct(s, _)) =
                    gen.infer_ast_type(self.elements.first().unwrap_or(&Expr::Number(NumberExpr {
                        value: "0".to_string(),
                        ty: None,
                        span: Span::default(),
                    })))
                {
                    if s == "String" {
                        "String"
                    } else {
                        "f32"
                    }
                } else {
                    "f32"
                }
            }
        };

        let new_call = Expr::FunctionCall(FunctionCallExpr {
            name: format!("Vec_{}::new", type_suffix),
            args: vec![],
            type_args: None,
            span: self.span,
        });

        let (vec_val, vec_ty) = gen.generate_expr(&new_call, block)?;

        let i32_ty = gen.i32_ty;
        let c1_op = block.append_operation(
            OperationBuilder::new("llvm.mlir.constant", gen.loc())
                .add_results(&[i32_ty])
                .add_attributes(&[(
                    Identifier::new(gen.context, "value"),
                    IntegerAttribute::new(i32_ty, 1).into(),
                )])
                .build()
                .unwrap(),
        );
        let c1 = c1_op.result(0).unwrap().into();

        let ptr_ty = gen.ptr_ty;
        let alloca_op = block.append_operation(
            OperationBuilder::new("llvm.alloca", gen.loc())
                .add_operands(&[c1])
                .add_results(&[ptr_ty])
                .add_attributes(&[(
                    Identifier::new(gen.context, "elem_type"),
                    TypeAttribute::new(vec_ty).into(),
                )])
                .build()
                .unwrap(),
        );
        let ptr_val = alloca_op.result(0).unwrap().into();

        block.append_operation(
            OperationBuilder::new("llvm.store", gen.loc())
                .add_operands(&[vec_val, ptr_val])
                .build()
                .unwrap(),
        );

        let tmp_vec_name = format!("__vec_ptr_{}", gen.string_counter);
        gen.string_counter += 1;
        gen.env.insert(tmp_vec_name.clone(), (ptr_val, vec_ty));
        gen.allocs.insert(tmp_vec_name.clone());

        for el in &self.elements {
            let push_call = Expr::FunctionCall(FunctionCallExpr {
                name: format!("Vec_{}::push", type_suffix),
                args: vec![
                    Expr::Borrow(BorrowExpr {
                        expr: Box::new(Expr::Identifier(IdentifierExpr {
                            name: tmp_vec_name.clone(),
                            span: Span::default(),
                        })),
                        is_mut: true,
                        span: Span::default(),
                    }),
                    el.clone(),
                ],
                type_args: None,
                span: self.span,
            });
            gen.generate_expr(&push_call, block)?;
        }

        let load_op = block.append_operation(
            OperationBuilder::new("llvm.load", gen.loc())
                .add_operands(&[ptr_val])
                .add_results(&[vec_ty])
                .build()
                .unwrap(),
        );
        Ok((load_op.result(0).unwrap().into(), vec_ty))
    }
}

impl<'c> LowerToMelior<'c> for ClosureExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>), LowerError>;

    fn lower(
        &self,
        _gen: &mut MeliorGenerator<'c>,
        _block: &melior::ir::Block<'c>,
    ) -> Self::Output {
        unreachable!("ClosureExpr should be transformed to StructInitExpr by Sema")
    }
}

impl<'c> LowerToMelior<'c> for ast::expr::PrintExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>), LowerError>;
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        for arg in &self.args {
            let (arg_val, arg_ty) = gen.generate_expr(arg, block)?;

            let func_name = match arg_ty.to_string().as_str() {
                "i32" => "print_i32",
                "f32" => "print_f32",
                "f64" => "print_f64",
                "!llvm.ptr" | "!llvm.ptr<i8>" => "print_str",
                _ => {
                    // Fallback or warning
                    println!("Warning: unsupported print arg type {}", arg_ty);
                    "print_i32"
                }
            };

            // Declare if not exists
            if !gen.functions.contains_key(func_name) {
                let func_ty = if func_name == "print_str" {
                    Type::parse(gen.context, "(!llvm.ptr) -> i32").unwrap()
                } else {
                    Type::parse(gen.context, &format!("({}) -> i32", arg_ty)).unwrap()
                };

                let func_decl = OperationBuilder::new("func.func", gen.loc())
                    .add_attributes(&[
                        (
                            Identifier::new(gen.context, "sym_name"),
                            StringAttribute::new(gen.context, func_name).into(),
                        ),
                        (
                            Identifier::new(gen.context, "function_type"),
                            TypeAttribute::new(func_ty).into(),
                        ),
                        (
                            Identifier::new(gen.context, "sym_visibility"),
                            StringAttribute::new(gen.context, "private").into(),
                        ),
                    ])
                    .add_regions([melior::ir::Region::new()])
                    .build()
                    .unwrap();

                gen.module.body().append_operation(func_decl);
                gen.functions.insert(
                    func_name.to_string(),
                    (
                        gen.i32_ty,
                        vec![if func_name == "print_str" {
                            gen.ptr_ty
                        } else {
                            arg_ty
                        }],
                    ),
                );
            }

            let name_attr = FlatSymbolRefAttribute::new(gen.context, func_name);
            let call_op = OperationBuilder::new("func.call", gen.loc())
                .add_operands(&[arg_val])
                .add_results(&[gen.i32_ty])
                .add_attributes(&[(Identifier::new(gen.context, "callee"), name_attr.into())])
                .build()
                .unwrap();

            block.append_operation(call_op);
        }

        let dummy_op = OperationBuilder::new("llvm.mlir.constant", gen.loc())
            .add_results(&[gen.i32_ty])
            .add_attributes(&[(
                Identifier::new(gen.context, "value"),
                IntegerAttribute::new(gen.i32_ty, 0).into(),
            )])
            .build()
            .unwrap();

        Ok((
            block.append_operation(dummy_op).result(0).unwrap().into(),
            gen.i32_ty,
        ))
    }
}

impl<'c> LowerToMelior<'c> for ast::expr::PrintlnExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>), LowerError>;
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        // First, reuse PrintExpr logic for arguments
        if !self.args.is_empty() {
            let print_expr = ast::expr::PrintExpr {
                args: self.args.clone(),
                span: self.span,
            };
            print_expr.lower(gen, block)?;
        }

        // Then emit a call to println()
        if !gen.functions.contains_key("println") {
            let func_ty = Type::parse(gen.context, "() -> i32").unwrap();

            let func_decl = OperationBuilder::new("func.func", gen.loc())
                .add_attributes(&[
                    (
                        Identifier::new(gen.context, "sym_name"),
                        StringAttribute::new(gen.context, "println").into(),
                    ),
                    (
                        Identifier::new(gen.context, "function_type"),
                        TypeAttribute::new(func_ty).into(),
                    ),
                    (
                        Identifier::new(gen.context, "sym_visibility"),
                        StringAttribute::new(gen.context, "private").into(),
                    ),
                ])
                .add_regions([melior::ir::Region::new()])
                .build()
                .unwrap();

            gen.module.body().append_operation(func_decl);
            gen.functions
                .insert("println".to_string(), (gen.i32_ty, vec![]));
        }

        let name_attr = FlatSymbolRefAttribute::new(gen.context, "println");
        let call_op = OperationBuilder::new("func.call", gen.loc())
            .add_results(&[gen.i32_ty])
            .add_attributes(&[(Identifier::new(gen.context, "callee"), name_attr.into())])
            .build()
            .unwrap();

        Ok((
            block.append_operation(call_op).result(0).unwrap().into(),
            gen.i32_ty,
        ))
    }
}

impl<'c> LowerToMelior<'c> for ast::expr::SizeOfExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>), LowerError>;

    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let size: i64 = match &self.target_ty {
            ast::Type::Scalar(ast::ElementType::F32)
            | ast::Type::Scalar(ast::ElementType::I32)
            | ast::Type::Scalar(ast::ElementType::U32) => 4,
            ast::Type::Scalar(ast::ElementType::F64)
            | ast::Type::Scalar(ast::ElementType::I64)
            | ast::Type::Scalar(ast::ElementType::U64) => 8,
            ast::Type::Scalar(ast::ElementType::I8)
            | ast::Type::Scalar(ast::ElementType::U8)
            | ast::Type::Scalar(ast::ElementType::Bool) => 1,
            ast::Type::Scalar(ast::ElementType::I16)
            | ast::Type::Scalar(ast::ElementType::U16)
            | ast::Type::Scalar(ast::ElementType::BF16)
            | ast::Type::Scalar(ast::ElementType::F16) => 2,
            ast::Type::Pointer(..) | ast::Type::Borrow(..) | ast::Type::Ref(..) => 8,
            _ => 8,
        };

        let size_ty = gen.i64_ty;
        let const_op = OperationBuilder::new("arith.constant", gen.loc())
            .add_attributes(&[(
                Identifier::new(gen.context, "value"),
                IntegerAttribute::new(size_ty, size).into(),
            )])
            .add_results(&[size_ty])
            .build()
            .unwrap();

        let const_op = block.append_operation(const_op);
        Ok((const_op.result(0).unwrap().into(), size_ty))
    }
}

impl<'c> LowerToMelior<'c> for ast::expr::AsCastExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>), LowerError>;

    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let (source_val, _source_ty) = gen.generate_expr(&self.expr, block)?;
        if let ast::Type::Closure(_, _) = &self.target_ty {
            let closure_struct_name = match self.source_ty.as_ref() {
                Some(ast::Type::Struct(name, _)) => name.clone(),
                Some(ast::Type::Borrow(inner, _, _, _)) => {
                    if let ast::Type::Struct(name, _) = &**inner {
                        name.clone()
                    } else {
                        panic!("Expected Closure_N struct, got {:?}", inner);
                    }
                }
                Some(t) => panic!("Expected Closure_N struct, got {:?}", t),
                None => panic!("Missing source_ty for AsCast expression"),
            };

            let call_fn_name = format!("{}_call", closure_struct_name);
            let (ret_ty, orig_arg_types) = gen
                .functions
                .get(&call_fn_name)
                .cloned()
                .expect("Closure call function not found");

            let fn_ty =
                melior::ir::r#type::FunctionType::new(gen.context, &orig_arg_types, &[ret_ty]);
            let const_op = OperationBuilder::new("func.constant", gen.loc())
                .add_attributes(&[(
                    Identifier::new(gen.context, "value"),
                    FlatSymbolRefAttribute::new(gen.context, &call_fn_name).into(),
                )])
                .add_results(&[fn_ty.into()])
                .build()
                .unwrap();
            let const_ref = block.append_operation(const_op);
            let mut fn_ptr_val: melior::ir::Value = const_ref.result(0).unwrap().into();

            let ptr_ty = gen.ptr_ty;
            let bitcast_op = OperationBuilder::new("builtin.unrealized_conversion_cast", gen.loc())
                .add_operands(&[fn_ptr_val])
                .add_results(&[ptr_ty])
                .build()
                .unwrap();
            let bitcast_ref = block.append_operation(bitcast_op);
            fn_ptr_val = bitcast_ref.result(0).unwrap().into();

            let fat_ptr_ty = Type::parse(gen.context, "!llvm.struct<(ptr, ptr)>").unwrap();
            let undef_op = OperationBuilder::new("llvm.mlir.undef", gen.loc())
                .add_results(&[fat_ptr_ty])
                .build()
                .unwrap();
            let undef_ref = block.append_operation(undef_op);
            let mut fat_ptr_val: melior::ir::Value = undef_ref.result(0).unwrap().into();

            let insert_fn_op = OperationBuilder::new("llvm.insertvalue", gen.loc())
                .add_operands(&[fat_ptr_val, fn_ptr_val])
                .add_attributes(&[(
                    Identifier::new(gen.context, "position"),
                    melior::ir::attribute::DenseI64ArrayAttribute::new(gen.context, &[0]).into(),
                )])
                .add_results(&[fat_ptr_ty])
                .build()
                .unwrap();
            let insert_fn_ref = block.append_operation(insert_fn_op);
            fat_ptr_val = insert_fn_ref.result(0).unwrap().into();

            let mut env_ptr_val = source_val;
            let ptr_bitcast_op =
                OperationBuilder::new("builtin.unrealized_conversion_cast", gen.loc())
                    .add_operands(&[env_ptr_val])
                    .add_results(&[ptr_ty])
                    .build()
                    .unwrap();
            let ptr_bitcast_ref = block.append_operation(ptr_bitcast_op);
            env_ptr_val = ptr_bitcast_ref.result(0).unwrap().into();

            let insert_env_op = OperationBuilder::new("llvm.insertvalue", gen.loc())
                .add_operands(&[fat_ptr_val, env_ptr_val])
                .add_attributes(&[(
                    Identifier::new(gen.context, "position"),
                    melior::ir::attribute::DenseI64ArrayAttribute::new(gen.context, &[1]).into(),
                )])
                .add_results(&[fat_ptr_ty])
                .build()
                .unwrap();
            let insert_env_ref = block.append_operation(insert_env_op);
            fat_ptr_val = insert_env_ref.result(0).unwrap().into();

            return Ok((fat_ptr_val, fat_ptr_ty));
        } else if let ast::Type::Scalar(_) = &self.target_ty {
            let target_ty_mlir = gen.lower_type(&self.target_ty);
            let coerced_val = gen.coerce_type(block, source_val, _source_ty, target_ty_mlir);
            return Ok((coerced_val, target_ty_mlir));
        } else if let ast::Type::Pointer(..) = &self.target_ty {
            if let Some(ast::Type::Scalar(_)) = self.source_ty.as_ref() {
                let ptr_ty = gen.ptr_ty;
                let cast_op = OperationBuilder::new("llvm.inttoptr", gen.loc())
                    .add_operands(&[source_val])
                    .add_results(&[ptr_ty])
                    .build()
                    .unwrap();
                let cast_ref = block.append_operation(cast_op);
                return Ok((cast_ref.result(0).unwrap().into(), ptr_ty));
            }
        }

        panic!("Unsupported cast operation in codegen");
    }
}
