use super::*;

pub trait LowerToMelior<'c> {
    type Output;
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output;
}

pub trait MeliorOpInfo {
    fn get_op_name(&self, is_float: bool) -> &'static str;
    fn get_predicate(&self, is_float: bool) -> Option<i64>;
}

impl MeliorOpInfo for BinaryOp {
    fn get_op_name(&self, is_float: bool) -> &'static str {
        match self {
            BinaryOp::Add => {
                if is_float {
                    "arith.addf"
                } else {
                    "arith.addi"
                }
            }
            BinaryOp::Sub => {
                if is_float {
                    "arith.subf"
                } else {
                    "arith.subi"
                }
            }
            BinaryOp::Mul | BinaryOp::MatMul => {
                if is_float {
                    "arith.mulf"
                } else {
                    "arith.muli"
                }
            }
            BinaryOp::Div => {
                if is_float {
                    "arith.divf"
                } else {
                    "arith.divsi"
                }
            }
        }
    }

    fn get_predicate(&self, _is_float: bool) -> Option<i64> {
        None
    }
}

impl MeliorOpInfo for RelationalOp {
    fn get_op_name(&self, is_float: bool) -> &'static str {
        if is_float {
            "arith.cmpf"
        } else {
            "arith.cmpi"
        }
    }

    fn get_predicate(&self, is_float: bool) -> Option<i64> {
        Some(if is_float {
            match self {
                RelationalOp::Eq => 1,    // oeq
                RelationalOp::Gt => 2,    // ogt
                RelationalOp::Ge => 3,    // oge
                RelationalOp::Lt => 4,    // olt
                RelationalOp::Le => 5,    // ole
                RelationalOp::NotEq => 6, // one
            }
        } else {
            match self {
                RelationalOp::Eq => 0,    // eq
                RelationalOp::NotEq => 1, // ne
                RelationalOp::Lt => 2,    // slt
                RelationalOp::Le => 3,    // sle
                RelationalOp::Gt => 4,    // sgt
                RelationalOp::Ge => 5,    // sge
            }
        })
    }
}

impl MeliorOpInfo for LogicalOp {
    fn get_op_name(&self, _is_float: bool) -> &'static str {
        match self {
            LogicalOp::And => "arith.andi",
            LogicalOp::Or => "arith.ori",
        }
    }

    fn get_predicate(&self, _is_float: bool) -> Option<i64> {
        None
    }
}

impl<'c> LowerToMelior<'c> for IdentifierExpr {
    type Output = (Value<'c, 'c>, Type<'c>);
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let IdentifierExpr { name, span: _ } = self;
        if name == "true" || name == "false" {
            let i1_ty = Type::parse(gen.context, "i1").unwrap();
            let val = if name == "true" { 1 } else { 0 };
            let const_op = melior::ir::operation::OperationBuilder::new(
                "arith.constant",
                Location::unknown(gen.context),
            )
            .add_results(&[i1_ty])
            .add_attributes(&[(
                melior::ir::Identifier::new(gen.context, "value"),
                melior::ir::attribute::IntegerAttribute::new(i1_ty, val).into(),
            )])
            .build()
            .unwrap();
            let const_ref = block.append_operation(const_op);
            return (const_ref.result(0).unwrap().into(), i1_ty);
        }
        if let Some((val, ty)) = gen.env.get(name) {
            let ty_str = ty.to_string();
            if ty_str.starts_with("memref<memref<") {
                let inner_ty_str = &ty_str[7..ty_str.len() - 1];
                let inner_ty = Type::parse(gen.context, inner_ty_str).unwrap();
                let load_op = melior::ir::operation::OperationBuilder::new(
                    "memref.load",
                    Location::unknown(gen.context),
                )
                .add_operands(&[*val])
                .add_results(&[inner_ty])
                .build()
                .unwrap();
                let load_ref = block.append_operation(load_op);
                (load_ref.result(0).unwrap().into(), inner_ty)
            } else if ty_str.starts_with("memref<") && !ty_str.contains("x") {
                let inner_ty_str = &ty_str[7..ty_str.len() - 1];
                let inner_ty = Type::parse(gen.context, inner_ty_str).unwrap();
                let load_op = melior::ir::operation::OperationBuilder::new(
                    "memref.load",
                    Location::unknown(gen.context),
                )
                .add_operands(&[*val])
                .add_results(&[inner_ty])
                .build()
                .unwrap();
                let load_ref = block.append_operation(load_op);
                (load_ref.result(0).unwrap().into(), inner_ty)
            } else {
                (*val, *ty)
            }
        } else {
            panic!("Undefined variable: {}", name);
        }
    }
}

impl<'c> LowerToMelior<'c> for BorrowExpr {
    type Output = (Value<'c, 'c>, Type<'c>);
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let BorrowExpr { expr, .. } = self;
        let (val, ty) = gen.generate_expr(expr, block);
        let ptr_ty = Type::parse(gen.context, "!llvm.ptr").unwrap();
        let i64_ty = Type::parse(gen.context, "i64").unwrap();
        let c1_op = block.append_operation(
            melior::ir::operation::OperationBuilder::new(
                "arith.constant",
                Location::unknown(gen.context),
            )
            .add_results(&[i64_ty])
            .add_attributes(&[(
                melior::ir::Identifier::new(gen.context, "value"),
                melior::ir::attribute::IntegerAttribute::new(i64_ty, 1).into(),
            )])
            .build()
            .unwrap(),
        );
        let c1 = c1_op.result(0).unwrap().into();

        let alloca_op = block.append_operation(
            melior::ir::operation::OperationBuilder::new(
                "llvm.alloca",
                Location::unknown(gen.context),
            )
            .add_operands(&[c1])
            .add_results(&[ptr_ty])
            .add_attributes(&[(
                melior::ir::Identifier::new(gen.context, "elem_type"),
                melior::ir::attribute::TypeAttribute::new(ty).into(),
            )])
            .build()
            .unwrap(),
        );
        let ptr = alloca_op.result(0).unwrap().into();

        block.append_operation(
            melior::ir::operation::OperationBuilder::new(
                "llvm.store",
                Location::unknown(gen.context),
            )
            .add_operands(&[val, ptr])
            .build()
            .unwrap(),
        );

        (ptr, ptr_ty)
    }
}

impl<'c> LowerToMelior<'c> for StringLiteralExpr {
    type Output = (Value<'c, 'c>, Type<'c>);
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let str_val = format!("{}\0", self.value);
        let str_name = format!(".str.{}", gen.string_counter);
        gen.string_counter += 1;

        let module_body = gen.module.body();
        let array_ty =
            Type::parse(gen.context, &format!("!llvm.array<{} x i8>", str_val.len())).unwrap();
        let ptr_ty = Type::parse(gen.context, "!llvm.ptr").unwrap();

        let global_op = melior::ir::operation::OperationBuilder::new(
            "llvm.mlir.global",
            Location::unknown(gen.context),
        )
        .add_attributes(&[
            (
                melior::ir::Identifier::new(gen.context, "sym_name"),
                melior::ir::attribute::StringAttribute::new(gen.context, &str_name).into(),
            ),
            (
                melior::ir::Identifier::new(gen.context, "global_type"),
                melior::ir::attribute::TypeAttribute::new(array_ty).into(),
            ),
            (
                melior::ir::Identifier::new(gen.context, "constant"),
                melior::ir::Attribute::parse(gen.context, "unit").unwrap(),
            ),
            (
                melior::ir::Identifier::new(gen.context, "linkage"),
                melior::ir::Attribute::parse(gen.context, "#llvm.linkage<internal>").unwrap(),
            ),
            (
                melior::ir::Identifier::new(gen.context, "value"),
                melior::ir::attribute::StringAttribute::new(gen.context, &str_val).into(),
            ),
        ])
        .add_regions([Region::new()])
        .build()
        .unwrap();
        module_body.append_operation(global_op);

        let addressof_op = melior::ir::operation::OperationBuilder::new(
            "llvm.mlir.addressof",
            Location::unknown(gen.context),
        )
        .add_attributes(&[(
            melior::ir::Identifier::new(gen.context, "global_name"),
            melior::ir::attribute::FlatSymbolRefAttribute::new(gen.context, &str_name).into(),
        )])
        .add_results(&[ptr_ty])
        .build()
        .unwrap();
        let addressof_ref = block.append_operation(addressof_op);

        (addressof_ref.result(0).unwrap().into(), ptr_ty)
    }
}

impl<'c> LowerToMelior<'c> for ComptimeBlockExpr {
    type Output = (Value<'c, 'c>, Type<'c>);
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        for stmt in &self.stmts {
            gen.generate_statement(stmt, block);
        }
        if let Some(ret_expr) = &self.ret {
            gen.generate_expr(ret_expr, block)
        } else {
            let none_ty = Type::parse(gen.context, "none").unwrap();
            let dummy_val = melior::ir::operation::OperationBuilder::new(
                "arith.constant",
                Location::unknown(gen.context),
            )
            .add_attributes(&[(
                melior::ir::Identifier::new(gen.context, "value"),
                melior::ir::attribute::IntegerAttribute::new(Type::index(gen.context), 0).into(),
            )])
            .add_results(&[Type::index(gen.context)])
            .build()
            .unwrap();
            let dummy_ref = block.append_operation(dummy_val);
            (dummy_ref.result(0).unwrap().into(), none_ty)
        }
    }
}

impl<'c> LowerToMelior<'c> for DereferenceExpr {
    type Output = (Value<'c, 'c>, Type<'c>);
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let (ptr_val, ptr_ty) = gen.generate_expr(&self.expr, block);
        let ptr_ty_str = ptr_ty.to_string();

        let inner_ty_str = if ptr_ty_str.starts_with("!llvm.ptr<") {
            ptr_ty_str[10..ptr_ty_str.len() - 1].to_string()
        } else {
            "f32".to_string()
        };
        let inner_ty = Type::parse(gen.context, &inner_ty_str).unwrap();

        let load_op = melior::ir::operation::OperationBuilder::new(
            "llvm.load",
            Location::unknown(gen.context),
        )
        .add_operands(&[ptr_val])
        .add_results(&[inner_ty])
        .build()
        .unwrap();
        let load_ref = block.append_operation(load_op);
        (load_ref.result(0).unwrap().into(), inner_ty)
    }
}

impl<'c> LowerToMelior<'c> for crate::ast::IndexAccessExpr {
    type Output = (Value<'c, 'c>, Type<'c>);
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let (base_val, base_ty, indices) = gen
            .flatten_indices(&crate::ast::Expr::IndexAccess(self.clone()), block)
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

            let i64_ty = Type::parse(gen.context, "i64").unwrap();
            let cast_op = melior::ir::operation::OperationBuilder::new(
                "arith.index_cast",
                Location::unknown(gen.context),
            )
            .add_operands(&[indices[0]])
            .add_results(&[i64_ty])
            .build()
            .unwrap();
            let idx_i64 = block.append_operation(cast_op).result(0).unwrap().into();

            let gep_op = melior::ir::operation::OperationBuilder::new(
                "llvm.getelementptr",
                Location::unknown(gen.context),
            )
            .add_attributes(&[
                (
                    melior::ir::Identifier::new(gen.context, "rawConstantIndices"),
                    melior::ir::attribute::DenseI32ArrayAttribute::new(gen.context, &[-2147483648])
                        .into(),
                ),
                (
                    melior::ir::Identifier::new(gen.context, "elem_type"),
                    melior::ir::attribute::TypeAttribute::new(inner_ty).into(),
                ),
            ])
            .add_operands(&[base_val, idx_i64])
            .add_results(&[base_ty])
            .build()
            .unwrap();

            let gep_ref = block.append_operation(gep_op);
            let ptr_val = gep_ref.result(0).unwrap().into();

            let load_op = melior::ir::operation::OperationBuilder::new(
                "llvm.load",
                Location::unknown(gen.context),
            )
            .add_operands(&[ptr_val])
            .add_results(&[inner_ty])
            .build()
            .unwrap();

            let load_ref = block.append_operation(load_op);
            (load_ref.result(0).unwrap().into(), inner_ty)
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
            let mut load_builder = melior::ir::operation::OperationBuilder::new(
                "memref.load",
                Location::unknown(gen.context),
            )
            .add_operands(&[base_val]);

            for idx in indices {
                load_builder = load_builder.add_operands(&[idx]);
            }

            let load_op = load_builder.add_results(&[inner_ty]).build().unwrap();

            let load_ref = block.append_operation(load_op);
            (load_ref.result(0).unwrap().into(), inner_ty)
        }
    }
}

impl<'c> LowerToMelior<'c> for BinaryOpExpr {
    type Output = (Value<'c, 'c>, Type<'c>);
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let BinaryOpExpr {
            lhs,
            op,
            rhs,
            span: _,
        } = self;
        let (mut lhs_val, lhs_ty) = gen.generate_expr(lhs, block);
        let prev_expected = gen.expected_type;
        gen.expected_type = Some(lhs_ty);
        let (mut rhs_val, mut rhs_ty) = gen.generate_expr(rhs, block);
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

        let is_memref =
            lhs_ty.to_string().starts_with("memref<") && rhs_ty.to_string().starts_with("memref<");

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
            let index_ty = Type::parse(gen.context, "index").unwrap();

            if m_str == "?" {
                let m_idx_attr =
                    melior::ir::attribute::IntegerAttribute::new(Type::index(gen.context), 0)
                        .into();
                let cst_op = melior::ir::operation::OperationBuilder::new(
                    "arith.constant",
                    Location::unknown(gen.context),
                )
                .add_results(&[index_ty])
                .add_attributes(&[(
                    melior::ir::Identifier::new(gen.context, "value"),
                    m_idx_attr,
                )])
                .build()
                .unwrap();
                let idx_val = block.append_operation(cst_op).result(0).unwrap().into();

                let dim_m_op = melior::ir::operation::OperationBuilder::new(
                    "memref.dim",
                    Location::unknown(gen.context),
                )
                .add_operands(&[lhs_val, idx_val])
                .add_results(&[index_ty])
                .build()
                .unwrap();
                alloc_operands.push(block.append_operation(dim_m_op).result(0).unwrap().into());
            }

            if n_str == "?" {
                let n_idx_attr =
                    melior::ir::attribute::IntegerAttribute::new(Type::index(gen.context), 1)
                        .into();
                let cst_op = melior::ir::operation::OperationBuilder::new(
                    "arith.constant",
                    Location::unknown(gen.context),
                )
                .add_results(&[index_ty])
                .add_attributes(&[(
                    melior::ir::Identifier::new(gen.context, "value"),
                    n_idx_attr,
                )])
                .build()
                .unwrap();
                let idx_val = block.append_operation(cst_op).result(0).unwrap().into();

                let dim_n_op = melior::ir::operation::OperationBuilder::new(
                    "memref.dim",
                    Location::unknown(gen.context),
                )
                .add_operands(&[rhs_val, idx_val])
                .add_results(&[index_ty])
                .build()
                .unwrap();
                alloc_operands.push(block.append_operation(dim_n_op).result(0).unwrap().into());
            }

            // Alloc output buffer
            let alloc_op = melior::ir::operation::OperationBuilder::new(
                "memref.alloc",
                Location::unknown(gen.context),
            )
            .add_operands(&alloc_operands)
            .add_attributes(&[(
                melior::ir::Identifier::new(gen.context, "operandSegmentSizes"),
                melior::ir::attribute::DenseI32ArrayAttribute::new(
                    gen.context,
                    &[alloc_operands.len() as i32, 0],
                )
                .into(),
            )])
            .add_results(&[out_ty])
            .build()
            .unwrap();
            let out_val = block.append_operation(alloc_op).result(0).unwrap().into();

            // Zero initialize the output buffer since matmul accumulates!
            let zero_attr = if el_ty_str.starts_with('i') {
                melior::ir::attribute::IntegerAttribute::new(
                    Type::parse(gen.context, el_ty_str).unwrap(),
                    0,
                )
                .into()
            } else {
                melior::ir::attribute::FloatAttribute::new(
                    gen.context,
                    Type::parse(gen.context, el_ty_str).unwrap(),
                    0.0,
                )
                .into()
            };

            let zero_op = melior::ir::operation::OperationBuilder::new(
                "arith.constant",
                Location::unknown(gen.context),
            )
            .add_results(&[Type::parse(gen.context, el_ty_str).unwrap()])
            .add_attributes(&[(melior::ir::Identifier::new(gen.context, "value"), zero_attr)])
            .build()
            .unwrap();
            let zero_val = block.append_operation(zero_op).result(0).unwrap().into();

            let region_fill = Region::new();
            let block_fill = melior::ir::Block::new(&[
                (
                    Type::parse(gen.context, el_ty_str).unwrap(),
                    Location::unknown(gen.context),
                ),
                (
                    Type::parse(gen.context, el_ty_str).unwrap(),
                    Location::unknown(gen.context),
                ),
            ]);
            let yield_fill = melior::ir::operation::OperationBuilder::new(
                "linalg.yield",
                Location::unknown(gen.context),
            )
            .add_operands(&[block_fill.argument(0).unwrap().into()])
            .build()
            .unwrap();
            block_fill.append_operation(yield_fill);
            region_fill.append_block(block_fill);

            let linalg_fill = melior::ir::operation::OperationBuilder::new(
                "linalg.fill",
                Location::unknown(gen.context),
            )
            .add_operands(&[zero_val, out_val])
            .add_attributes(&[(
                melior::ir::Identifier::new(gen.context, "operandSegmentSizes"),
                melior::ir::attribute::DenseI32ArrayAttribute::new(gen.context, &[1, 1]).into(),
            )])
            .add_regions([region_fill])
            .build()
            .unwrap();
            block.append_operation(linalg_fill);

            // Execute linalg.matmul
            let region_matmul = Region::new();
            let block_matmul = melior::ir::Block::new(&[
                (
                    Type::parse(gen.context, el_ty_str).unwrap(),
                    Location::unknown(gen.context),
                ),
                (
                    Type::parse(gen.context, el_ty_str).unwrap(),
                    Location::unknown(gen.context),
                ),
                (
                    Type::parse(gen.context, el_ty_str).unwrap(),
                    Location::unknown(gen.context),
                ),
            ]);

            let is_float = el_ty_str.contains("f32")
                || el_ty_str.contains("f64")
                || el_ty_str.contains("f16")
                || el_ty_str.contains("bf16");
            let mul_op_name = if is_float { "arith.mulf" } else { "arith.muli" };
            let add_op_name = if is_float { "arith.addf" } else { "arith.addi" };

            let mul_op = melior::ir::operation::OperationBuilder::new(
                mul_op_name,
                Location::unknown(gen.context),
            )
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

            let add_op = melior::ir::operation::OperationBuilder::new(
                add_op_name,
                Location::unknown(gen.context),
            )
            .add_operands(&[block_matmul.argument(2).unwrap().into(), mul_val])
            .add_results(&[Type::parse(gen.context, el_ty_str).unwrap()])
            .build()
            .unwrap();
            let add_val = block_matmul
                .append_operation(add_op)
                .result(0)
                .unwrap()
                .into();

            let yield_matmul = melior::ir::operation::OperationBuilder::new(
                "linalg.yield",
                Location::unknown(gen.context),
            )
            .add_operands(&[add_val])
            .build()
            .unwrap();
            block_matmul.append_operation(yield_matmul);
            region_matmul.append_block(block_matmul);

            let matmul_op = melior::ir::operation::OperationBuilder::new(
                "linalg.matmul",
                Location::unknown(gen.context),
            )
            .add_operands(&[lhs_val, rhs_val, out_val])
            .add_attributes(&[(
                melior::ir::Identifier::new(gen.context, "operandSegmentSizes"),
                melior::ir::attribute::DenseI32ArrayAttribute::new(gen.context, &[2, 1]).into(),
            )])
            .add_regions([region_matmul])
            .build()
            .unwrap();
            block.append_operation(matmul_op);

            return (out_val, out_ty);
        } else if is_memref {
            // Element-wise Linalg Lowering (Add, Sub, Mul, Div)
            let out_ty = Type::parse(gen.context, &lhs_ty_str).unwrap();

            let mut alloc_operands = Vec::new();
            let index_ty = Type::parse(gen.context, "index").unwrap();

            let rank = lhs_parts.len() - 1; // Last part is element type
            let el_ty_str = lhs_parts.last().unwrap();

            for (i, dim_str) in lhs_parts.iter().take(rank).enumerate() {
                if *dim_str == "?" {
                    let idx_attr = melior::ir::attribute::IntegerAttribute::new(
                        Type::index(gen.context),
                        i as i64,
                    )
                    .into();
                    let cst_op = melior::ir::operation::OperationBuilder::new(
                        "arith.constant",
                        Location::unknown(gen.context),
                    )
                    .add_results(&[index_ty])
                    .add_attributes(&[(
                        melior::ir::Identifier::new(gen.context, "value"),
                        idx_attr,
                    )])
                    .build()
                    .unwrap();
                    let idx_val = block.append_operation(cst_op).result(0).unwrap().into();

                    let dim_op = melior::ir::operation::OperationBuilder::new(
                        "memref.dim",
                        Location::unknown(gen.context),
                    )
                    .add_operands(&[lhs_val, idx_val])
                    .add_results(&[index_ty])
                    .build()
                    .unwrap();
                    alloc_operands.push(block.append_operation(dim_op).result(0).unwrap().into());
                }
            }

            // Alloc output buffer
            let alloc_op = melior::ir::operation::OperationBuilder::new(
                "memref.alloc",
                Location::unknown(gen.context),
            )
            .add_operands(&alloc_operands)
            .add_attributes(&[(
                melior::ir::Identifier::new(gen.context, "operandSegmentSizes"),
                melior::ir::attribute::DenseI32ArrayAttribute::new(
                    gen.context,
                    &[alloc_operands.len() as i32, 0],
                )
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
                (
                    Type::parse(gen.context, el_ty_str).unwrap(),
                    Location::unknown(gen.context),
                ),
                (
                    Type::parse(gen.context, el_ty_str).unwrap(),
                    Location::unknown(gen.context),
                ),
                (
                    Type::parse(gen.context, el_ty_str).unwrap(),
                    Location::unknown(gen.context),
                ),
            ]);

            let arith_op = melior::ir::operation::OperationBuilder::new(
                arith_op_name,
                Location::unknown(gen.context),
            )
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

            let yield_op = melior::ir::operation::OperationBuilder::new(
                "linalg.yield",
                Location::unknown(gen.context),
            )
            .add_operands(&[arith_val])
            .build()
            .unwrap();

            block_inner.append_operation(yield_op);
            region.append_block(block_inner);

            let linalg_op = melior::ir::operation::OperationBuilder::new(
                op_name,
                Location::unknown(gen.context),
            )
            .add_operands(&[lhs_val, rhs_val, out_val])
            .add_attributes(&[(
                melior::ir::Identifier::new(gen.context, "operandSegmentSizes"),
                melior::ir::attribute::DenseI32ArrayAttribute::new(gen.context, &[2, 1]).into(),
            )])
            .add_regions([region])
            .build()
            .unwrap();
            block.append_operation(linalg_op);

            return (out_val, out_ty);
        }

        if lhs_ty != rhs_ty
            && ((lhs_ty_str == "index" && rhs_ty_str == "i32")
                || (lhs_ty_str == "i32" && rhs_ty_str == "index"))
        {
            if lhs_ty_str == "index" && rhs_ty_str == "i32" {
                let cast_op = melior::ir::operation::OperationBuilder::new(
                    "arith.index_cast",
                    Location::unknown(gen.context),
                )
                .add_operands(&[rhs_val])
                .add_results(&[lhs_ty])
                .build()
                .unwrap();
                rhs_val = block.append_operation(cast_op).result(0).unwrap().into();
            } else {
                let cast_op = melior::ir::operation::OperationBuilder::new(
                    "arith.index_cast",
                    Location::unknown(gen.context),
                )
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

        let mut builder = melior::ir::operation::OperationBuilder::new(
            op.get_op_name(is_float),
            Location::unknown(gen.context),
        );
        builder = builder.add_operands(&[lhs_val, rhs_val]);

        let ret_ty = if let Some(pred_val) = op.get_predicate(is_float) {
            let i1_ty = Type::parse(gen.context, "i1").unwrap();
            let i64_ty = Type::parse(gen.context, "i64").unwrap();
            builder = builder.add_results(&[i1_ty]).add_attributes(&[(
                melior::ir::Identifier::new(gen.context, "predicate"),
                melior::ir::attribute::IntegerAttribute::new(i64_ty, pred_val).into(),
            )]);
            i1_ty
        } else {
            builder = builder.add_results(&[final_ty]);
            final_ty
        };

        let bin_op = builder.build().unwrap();
        let bin_ref = block.append_operation(bin_op);
        (bin_ref.result(0).unwrap().into(), ret_ty)
    }
}

impl<'c> LowerToMelior<'c> for RelationalOpExpr {
    type Output = (Value<'c, 'c>, Type<'c>);
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let RelationalOpExpr {
            lhs,
            op,
            rhs,
            span: _,
        } = self;
        let (lhs_val, lhs_ty) = gen.generate_expr(lhs, block);
        let prev_expected = gen.expected_type;
        gen.expected_type = Some(lhs_ty);
        let (mut rhs_val, rhs_ty) = gen.generate_expr(rhs, block);
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

        let mut builder = melior::ir::operation::OperationBuilder::new(
            op.get_op_name(is_float),
            Location::unknown(gen.context),
        );
        builder = builder.add_operands(&[lhs_val, rhs_val]);

        let ret_ty = if let Some(pred_val) = op.get_predicate(is_float) {
            let i1_ty = Type::parse(gen.context, "i1").unwrap();
            let i64_ty = Type::parse(gen.context, "i64").unwrap();
            builder = builder.add_results(&[i1_ty]).add_attributes(&[(
                melior::ir::Identifier::new(gen.context, "predicate"),
                melior::ir::attribute::IntegerAttribute::new(i64_ty, pred_val).into(),
            )]);
            i1_ty
        } else {
            builder = builder.add_results(&[final_ty]);
            final_ty
        };

        let bin_op = builder.build().unwrap();
        let bin_ref = block.append_operation(bin_op);
        (bin_ref.result(0).unwrap().into(), ret_ty)
    }
}

impl<'c> LowerToMelior<'c> for LogicalOpExpr {
    type Output = (Value<'c, 'c>, Type<'c>);
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let LogicalOpExpr {
            lhs,
            op,
            rhs,
            span: _,
        } = self;
        let (lhs_val, _lhs_ty) = gen.generate_expr(lhs, block);
        let (rhs_val, _rhs_ty) = gen.generate_expr(rhs, block);

        let final_ty = Type::parse(gen.context, "i1").unwrap();

        let builder = melior::ir::operation::OperationBuilder::new(
            op.get_op_name(false),
            Location::unknown(gen.context),
        )
        .add_operands(&[lhs_val, rhs_val])
        .add_results(&[final_ty]);

        let bin_op = builder.build().unwrap();
        let bin_ref = block.append_operation(bin_op);
        (bin_ref.result(0).unwrap().into(), final_ty)
    }
}

impl<'c> LowerToMelior<'c> for crate::ast::UnaryOpExpr {
    type Output = (Value<'c, 'c>, Type<'c>);
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let crate::ast::UnaryOpExpr { op, expr, span: _ } = self;
        let (val, ty) = gen.generate_expr(expr, block);
        match op {
            crate::ast::UnaryOp::Not => {
                let true_val_op = melior::ir::operation::OperationBuilder::new(
                    "arith.constant",
                    Location::unknown(gen.context),
                )
                .add_results(&[ty])
                .add_attributes(&[(
                    melior::ir::Identifier::new(gen.context, "value"),
                    melior::ir::attribute::IntegerAttribute::new(ty, 1).into(),
                )])
                .build()
                .unwrap();
                let true_val_ref = block.append_operation(true_val_op);

                let not_op = melior::ir::operation::OperationBuilder::new(
                    "arith.xori",
                    Location::unknown(gen.context),
                )
                .add_operands(&[val, true_val_ref.result(0).unwrap().into()])
                .add_results(&[ty])
                .build()
                .unwrap();

                let not_ref = block.append_operation(not_op);
                (not_ref.result(0).unwrap().into(), ty)
            }
        }
    }
}

impl<'c> LowerToMelior<'c> for StructInitExpr {
    type Output = (Value<'c, 'c>, Type<'c>);
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let StructInitExpr {
            name,
            fields,
            span: _,
        } = self;
        let struct_decl = gen.structs.get(name).unwrap().clone();
        let struct_ty = gen.lower_type(&crate::ast::Type::Struct(name.clone(), None));

        let undef_op = melior::ir::operation::OperationBuilder::new(
            "llvm.mlir.undef",
            Location::unknown(gen.context),
        )
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
            let field_ty = gen.lower_type(&struct_decl.fields[field_idx].1);
            let prev_expected = gen.expected_type;
            gen.expected_type = Some(field_ty);
            let (mut field_val, expr_ty) = gen.generate_expr(f_expr, block);
            gen.expected_type = prev_expected;
            println!(
                "Struct {} field {} has type {:?}, expr has type {:?}",
                name, field_name, struct_decl.fields[field_idx].1, expr_ty
            );

            if expr_ty != field_ty
                && ((expr_ty.to_string() == "index" && field_ty.to_string() == "i32")
                    || (expr_ty.to_string() == "i32" && field_ty.to_string() == "index"))
            {
                let cast_op = melior::ir::operation::OperationBuilder::new(
                    "arith.index_cast",
                    Location::unknown(gen.context),
                )
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

            let insert_op = melior::ir::operation::OperationBuilder::new(
                "llvm.insertvalue",
                Location::unknown(gen.context),
            )
            .add_operands(&[current_struct, field_val])
            .add_attributes(&[(
                melior::ir::Identifier::new(gen.context, "position"),
                pos_attr.into(),
            )])
            .add_results(&[struct_ty])
            .build()
            .unwrap();
            current_struct = block.append_operation(insert_op).result(0).unwrap().into();
        }
        (current_struct, struct_ty)
    }
}

impl<'c> LowerToMelior<'c> for UnsafeBlockExpr {
    type Output = (Value<'c, 'c>, Type<'c>);
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        for stmt in &self.stmts {
            gen.generate_statement(stmt, block);
        }
        if let Some(ret_expr) = &self.ret {
            gen.generate_expr(ret_expr, block)
        } else {
            // Return an i32 0 or something empty if no return type is expected.
            let i32_ty = Type::parse(gen.context, "i32").unwrap();
            let zero_attr = melior::ir::attribute::IntegerAttribute::new(i32_ty, 0).into();
            let zero_op = melior::ir::operation::OperationBuilder::new(
                "arith.constant",
                Location::unknown(gen.context),
            )
            .add_results(&[i32_ty])
            .add_attributes(&[(melior::ir::Identifier::new(gen.context, "value"), zero_attr)])
            .build()
            .unwrap();
            let zero_val = block.append_operation(zero_op).result(0).unwrap().into();
            (zero_val, i32_ty)
        }
    }
}

fn topology_to_i32(top: &crate::ast::Topology) -> i32 {
    use crate::ast::Topology::*;
    match top {
        Host => 0,
        NPU(expr) => {
            if let crate::ast::Expr::Number(n) = &**expr {
                100 + n.value.parse::<i32>().unwrap_or(0)
            } else {
                100
            }
        }
        AccCore(expr) => {
            if let crate::ast::Expr::Number(n) = &**expr {
                200 + n.value.parse::<i32>().unwrap_or(0)
            } else {
                200
            }
        }
        AMX => 300,
        ANE => 400,
        GPU => 500,
        Slice(_, _, _) => 900,
    }
}

impl<'c> LowerToMelior<'c> for crate::ast::SpawnOnExpr {
    type Output = (Value<'c, 'c>, Type<'c>);
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let location = melior::ir::Location::unknown(gen.context);
        let region = melior::ir::Region::new();
        let body_block = melior::ir::Block::new(&[]);
        let prev_in_spawn = gen.in_spawn;
        gen.in_spawn = true;
        for stmt in &self.stmts {
            gen.generate_statement(stmt, &body_block);
        }

        let mut result_types = vec![];
        let mut ret_val = None;

        if let Some(r) = &self.ret {
            let (val, ty) = gen.generate_expr(r, &body_block);
            result_types.push(ty);
            ret_val = Some(val);
        }

        let mut needs_yield = true;
        if let Some(crate::ast::Statement::Return(_)) = self.stmts.last() {
            needs_yield = false;
        }

        if needs_yield {
            let mut yield_builder =
                melior::ir::operation::OperationBuilder::new("vx.yield", location);
            if let Some(v) = ret_val {
                yield_builder = yield_builder.add_operands(&[v]);
            }
            let yield_op = yield_builder
                .build()
                .expect("Failed to build vx.yield operation");
            body_block.append_operation(yield_op);
        }

        gen.in_spawn = prev_in_spawn;
        region.append_block(body_block);

        let topology_id = topology_to_i32(&self.top);
        let top_attr = melior::ir::attribute::IntegerAttribute::new(
            Type::parse(gen.context, "i32").unwrap(),
            topology_id as i64,
        )
        .into();

        let mut spawn_builder = melior::ir::operation::OperationBuilder::new("vx.spawn", location)
            .add_attributes(&[(
                melior::ir::Identifier::new(gen.context, "topology"),
                top_attr,
            )])
            .add_regions([region]);

        if !result_types.is_empty() {
            spawn_builder = spawn_builder.add_results(&result_types);
        }

        let spawn_op = spawn_builder.build().unwrap();
        let spawn_ref = block.append_operation(spawn_op);

        if !needs_yield {
            let mut ret_builder =
                melior::ir::operation::OperationBuilder::new("func.return", location);
            if !result_types.is_empty() {
                ret_builder = ret_builder.add_operands(&[spawn_ref.result(0).unwrap().into()]);
            }
            let ret_op = ret_builder.build().unwrap();
            block.append_operation(ret_op);
        }

        if !result_types.is_empty() {
            (spawn_ref.result(0).unwrap().into(), result_types[0])
        } else {
            let _none_ty =
                Type::parse(gen.context, "none").unwrap_or_else(|| Type::index(gen.context));
            let dummy_op = melior::ir::operation::OperationBuilder::new("arith.constant", location)
                .add_attributes(&[(
                    melior::ir::Identifier::new(gen.context, "value"),
                    melior::ir::attribute::IntegerAttribute::new(Type::index(gen.context), 0)
                        .into(),
                )])
                .add_results(&[Type::index(gen.context)])
                .build()
                .unwrap();
            let dummy_ref = block.append_operation(dummy_op);
            (
                dummy_ref.result(0).unwrap().into(),
                Type::index(gen.context),
            )
        }
    }
}

impl<'c> LowerToMelior<'c> for crate::ast::TransferExpr {
    type Output = (Value<'c, 'c>, Type<'c>);
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let (src_val, src_ty) = gen.generate_expr(&self.expr, block);
        let location = melior::ir::Location::unknown(gen.context);

        // Map memory space to topology target.
        let target_topology_id = match self.space {
            crate::ast::MemorySpace::HostDRAM => 0,
            crate::ast::MemorySpace::NPUHBM => 100,
            crate::ast::MemorySpace::LocalSRAM => 200,
        };

        let top_attr = melior::ir::attribute::IntegerAttribute::new(
            melior::ir::Type::parse(gen.context, "i32").unwrap(),
            target_topology_id as i64,
        )
        .into();

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

        let transfer_op = melior::ir::operation::OperationBuilder::new("vx.transfer", location)
            .add_operands(&[src_val])
            .add_attributes(&[(
                melior::ir::Identifier::new(gen.context, "target_topology"),
                top_attr,
            )])
            .add_results(&[target_ty])
            .build()
            .expect("Failed to build vx.transfer operation");

        use melior::ir::operation::OperationLike;
        let result_val = transfer_op.result(0).unwrap().into();
        block.append_operation(transfer_op);

        (result_val, target_ty)
    }
}

impl<'c> LowerToMelior<'c> for MemberAccessExpr {
    type Output = (Value<'c, 'c>, Type<'c>);
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let MemberAccessExpr {
            base,
            member,
            struct_name,
            span: _,
        } = self;
        let (base_val, base_ty) = gen.generate_expr(base, block);
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

        if let Some(resolved_struct_name) = struct_name_opt {
            if let Some(struct_decl) = gen.structs.get(&resolved_struct_name).cloned() {
                if let Some(field_idx) = struct_decl.fields.iter().position(|(n, _)| n == member) {
                    let field_ty = gen.lower_type(&struct_decl.fields[field_idx].1);

                    if is_ptr {
                        let ptr_ty = Type::parse(gen.context, "!llvm.ptr").unwrap();
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
                        let struct_llvm_ty = Type::parse(gen.context, &struct_llvm_ty_str).unwrap();

                        let gep_op = melior::ir::operation::OperationBuilder::new(
                            "llvm.getelementptr",
                            Location::unknown(gen.context),
                        )
                        .add_operands(&[base_val])
                        .add_attributes(&[
                            (
                                melior::ir::Identifier::new(gen.context, "rawConstantIndices"),
                                melior::ir::attribute::DenseI32ArrayAttribute::new(
                                    gen.context,
                                    &[0, field_idx as i32],
                                )
                                .into(),
                            ),
                            (
                                melior::ir::Identifier::new(gen.context, "elem_type"),
                                melior::ir::attribute::TypeAttribute::new(struct_llvm_ty).into(),
                            ),
                        ])
                        .add_results(&[ptr_ty])
                        .build()
                        .unwrap();
                        let gep_ref = block.append_operation(gep_op);
                        let field_ptr = gep_ref.result(0).unwrap().into();

                        let load_op = melior::ir::operation::OperationBuilder::new(
                            "llvm.load",
                            Location::unknown(gen.context),
                        )
                        .add_operands(&[field_ptr])
                        .add_results(&[field_ty])
                        .build()
                        .unwrap();
                        let load_ref = block.append_operation(load_op);
                        return (load_ref.result(0).unwrap().into(), field_ty);
                    } else {
                        let pos_attr = melior::ir::attribute::DenseI64ArrayAttribute::new(
                            gen.context,
                            &[field_idx as i64],
                        );

                        let ext_op = melior::ir::operation::OperationBuilder::new(
                            "llvm.extractvalue",
                            Location::unknown(gen.context),
                        )
                        .add_operands(&[base_val])
                        .add_attributes(&[(
                            melior::ir::Identifier::new(gen.context, "position"),
                            pos_attr.into(),
                        )])
                        .add_results(&[field_ty])
                        .build()
                        .unwrap();
                        let ext_ref = block.append_operation(ext_op);
                        return (ext_ref.result(0).unwrap().into(), field_ty);
                    }
                }
            }
        }
        panic!("Cannot resolve member access {}", member);
    }
}

impl<'c> LowerToMelior<'c> for FunctionCallExpr {
    type Output = (Value<'c, 'c>, Type<'c>);
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let FunctionCallExpr {
            name,
            args,
            span: _,
        } = self;
        if name == "Verified" {
            return gen.generate_expr(&args[0], block);
        }
        if (name.starts_with("Tensor_") && !name.contains("__") && !name.contains("_dim"))
            || name == "Tensor"
        {
            let el_ty_str = if name.starts_with("Tensor_") {
                name.strip_prefix("Tensor_").unwrap()
            } else {
                "f32"
            };
            let mlir_ty_str = match el_ty_str {
                "f16" => "f16",
                "f32" => "f32",
                "f64" => "f64",
                "bf16" => "bf16",
                "i32" => "i32",
                "i64" => "i64",
                "Bool" => "i1",
                _ => "f32", // Default fallback
            };
            let mut dynamic_sizes = Vec::new();
            let mut dims_count = 2; // Default fallback

            if args.len() == 1 {
                if let Expr::Array(arr) = &args[0] {
                    dims_count = arr.elements.len();
                    for el in &arr.elements {
                        let (mut val, ty) = gen.generate_expr(el, block);
                        if ty.to_string() != "index" {
                            let cast_op = melior::ir::operation::OperationBuilder::new(
                                "arith.index_cast",
                                Location::unknown(gen.context),
                            )
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
                    let (mut val, ty) = gen.generate_expr(el, block);
                    if ty.to_string() != "index" {
                        let cast_op = melior::ir::operation::OperationBuilder::new(
                            "arith.index_cast",
                            Location::unknown(gen.context),
                        )
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

            let alloc_op = melior::ir::operation::OperationBuilder::new(
                "memref.alloc",
                Location::unknown(gen.context),
            )
            .add_operands(&dynamic_sizes)
            .add_attributes(&[(
                melior::ir::Identifier::new(gen.context, "operandSegmentSizes"),
                melior::ir::attribute::DenseI32ArrayAttribute::new(
                    gen.context,
                    &[dynamic_sizes.len() as i32, 0],
                )
                .into(),
            )])
            .add_results(&[tensor_ty])
            .build()
            .unwrap();
            let alloc_ref = block.append_operation(alloc_op);
            return (alloc_ref.result(0).unwrap().into(), tensor_ty);
        }

        if name == "reshape" || name == "transpose" {
            let (arg_val, expr_ty) = gen.generate_expr(&args[0], block);
            let expr_ty_str = expr_ty.to_string();

            // Extract element type
            let el_ty_str = if expr_ty_str.starts_with("memref<") {
                if expr_ty_str.contains("f16") {
                    "f16"
                } else if expr_ty_str.contains("f32") {
                    "f32"
                } else if expr_ty_str.contains("f64") {
                    "f64"
                } else if expr_ty_str.contains("bf16") {
                    "bf16"
                } else if expr_ty_str.contains("i32") {
                    "i32"
                } else if expr_ty_str.contains("i64") {
                    "i64"
                } else if expr_ty_str.contains("i1") {
                    "i1"
                } else {
                    "f32"
                }
            } else {
                "f32"
            };

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
            let cast1_op = melior::ir::operation::OperationBuilder::new(
                "memref.cast",
                Location::unknown(gen.context),
            )
            .add_operands(&[arg_val])
            .add_results(&[unranked_ty])
            .build()
            .unwrap();
            let cast1_ref = block.append_operation(cast1_op);
            let unranked_val = cast1_ref.result(0).unwrap().into();

            let target_ty = Type::parse(gen.context, &target_ty_str).unwrap();

            // Cast to targeted shape
            let cast2_op = melior::ir::operation::OperationBuilder::new(
                "memref.cast",
                Location::unknown(gen.context),
            )
            .add_operands(&[unranked_val])
            .add_results(&[target_ty])
            .build()
            .unwrap();

            let cast2_ref = block.append_operation(cast2_op);
            return (cast2_ref.result(0).unwrap().into(), target_ty);
        }

        if name == "with_memory" {
            // For now, with_memory is a no-op in lowering, just returns the tensor
            let (arg_val, expr_ty) = gen.generate_expr(&args[0], block);
            return (arg_val, expr_ty);
        }

        if name == "print" {
            let mut print_arg = &args[0];
            if let Expr::Borrow(borrow) = print_arg {
                print_arg = &borrow.expr;
            }

            let (mut arg_val, arg_ty) = gen.generate_expr(print_arg, block);

            let el_ty_str = if arg_ty.to_string().contains("f64") {
                "f64"
            } else if arg_ty.to_string().contains("i64") {
                "i64"
            } else if arg_ty.to_string().contains("i32") {
                "i32"
            } else if arg_ty.to_string().contains("bf16") {
                "bf16"
            } else {
                "f32"
            };

            let print_fn_name = match el_ty_str {
                "f64" => "printMemrefF64",
                "i64" => "printMemrefI64",
                "i32" => "printMemrefI32",
                "bf16" => "printMemrefBF16",
                _ => "printMemrefF32",
            };

            if arg_ty.to_string().contains(", ") {
                let stripped_ty =
                    Type::parse(gen.context, &format!("memref<?x?x{}>", el_ty_str)).unwrap();
                let mcast_op = block.append_operation(
                    melior::ir::operation::OperationBuilder::new(
                        "memref.memory_space_cast",
                        Location::unknown(gen.context),
                    )
                    .add_operands(&[arg_val])
                    .add_results(&[stripped_ty])
                    .build()
                    .unwrap(),
                );
                arg_val = mcast_op.result(0).unwrap().into();
            }

            let unranked_memref_ty =
                Type::parse(gen.context, &format!("memref<*x{}>", el_ty_str)).unwrap();
            let cast_op = block.append_operation(
                melior::ir::operation::OperationBuilder::new(
                    "memref.cast",
                    Location::unknown(gen.context),
                )
                .add_operands(&[arg_val])
                .add_results(&[unranked_memref_ty])
                .build()
                .unwrap(),
            );
            let cast_val: Value = cast_op.result(0).unwrap().into();

            block.append_operation(
                melior::ir::operation::OperationBuilder::new(
                    "func.call",
                    Location::unknown(gen.context),
                )
                .add_operands(&[cast_val])
                .add_attributes(&[(
                    melior::ir::Identifier::new(gen.context, "callee"),
                    melior::ir::attribute::FlatSymbolRefAttribute::new(gen.context, print_fn_name)
                        .into(),
                )])
                .build()
                .unwrap(),
            );

            return (
                cast_val, // Dummy return value, caller ignores it
                Type::parse(gen.context, "none").unwrap(),
            );
        }

        if let Some((ret_ty, arg_tys)) = gen.functions.get(name).cloned() {
            let mut arg_vals = Vec::new();
            for (i, arg) in args.iter().enumerate() {
                let (mut arg_val, expr_ty) = gen.generate_expr(arg, block);
                let field_ty = arg_tys[i];
                if expr_ty != field_ty {
                    if expr_ty.to_string().starts_with("memref<")
                        && field_ty.to_string().starts_with("memref<")
                    {
                        let cast_op = melior::ir::operation::OperationBuilder::new(
                            "memref.cast",
                            Location::unknown(gen.context),
                        )
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

            let name_attr = melior::ir::attribute::FlatSymbolRefAttribute::new(gen.context, name);
            let mut builder = melior::ir::operation::OperationBuilder::new(
                "func.call",
                Location::unknown(gen.context),
            )
            .add_operands(&arg_vals)
            .add_attributes(&[(
                melior::ir::Identifier::new(gen.context, "callee"),
                name_attr.into(),
            )]);

            if ret_ty.to_string() != "none" {
                builder = builder.add_results(&[ret_ty]);
                let call_op = builder.build().unwrap();
                let call_ref = block.append_operation(call_op);
                (call_ref.result(0).unwrap().into(), ret_ty)
            } else {
                let call_op = builder.build().unwrap();
                block.append_operation(call_op);
                let none_ty = Type::parse(gen.context, "none").unwrap();
                // this value shouldn't be used
                let dummy_op = melior::ir::operation::OperationBuilder::new(
                    "arith.constant",
                    Location::unknown(gen.context),
                )
                .add_results(&[Type::parse(gen.context, "i32").unwrap()])
                .add_attributes(&[(
                    melior::ir::Identifier::new(gen.context, "value"),
                    melior::ir::attribute::IntegerAttribute::new(
                        Type::parse(gen.context, "i32").unwrap(),
                        0,
                    )
                    .into(),
                )])
                .build()
                .unwrap();
                (
                    block.append_operation(dummy_op).result(0).unwrap().into(),
                    none_ty,
                )
            }
        } else {
            panic!("Function {} not found", name);
        }
    }
}

impl<'c> LowerToMelior<'c> for MethodCallExpr {
    type Output = (Value<'c, 'c>, Type<'c>);
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let MethodCallExpr {
            base,
            method_name,
            args,
            span,
        } = self;
        let mut new_args = vec![*base.clone()];
        new_args.extend(args.clone());
        gen.generate_expr(
            &Expr::FunctionCall(FunctionCallExpr {
                name: method_name.clone(),
                args: new_args,
                span: span.clone(),
            }),
            block,
        )
    }
}

impl<'c> LowerToMelior<'c> for ArrayExpr {
    type Output = (Value<'c, 'c>, Type<'c>);
    fn lower(
        &self,
        _gen: &mut MeliorGenerator<'c>,
        _block: &melior::ir::Block<'c>,
    ) -> Self::Output {
        panic!("Should not be evaluated directly")
    }
}

impl<'c> LowerToMelior<'c> for MemorySpaceExpr {
    type Output = (Value<'c, 'c>, Type<'c>);
    fn lower(
        &self,
        _gen: &mut MeliorGenerator<'c>,
        _block: &melior::ir::Block<'c>,
    ) -> Self::Output {
        panic!("Should not be evaluated directly")
    }
}

impl<'c> LowerToMelior<'c> for TopologyExpr {
    type Output = (Value<'c, 'c>, Type<'c>);
    fn lower(
        &self,
        _gen: &mut MeliorGenerator<'c>,
        _block: &melior::ir::Block<'c>,
    ) -> Self::Output {
        panic!("Should not be evaluated directly")
    }
}

impl<'c> LowerToMelior<'c> for IfExpr {
    type Output = (Value<'c, 'c>, Type<'c>);
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let IfExpr {
            cond,
            then_block,
            else_block: else_block_opt,
            span: _,
        } = self;
        let (cond_val, _) = gen.generate_expr(cond, block);

        let then_region = Region::new();
        let then_b = Block::new(&[]);
        for stmt in then_block {
            gen.generate_statement(stmt, &then_b);
        }
        let yield_op = melior::ir::operation::OperationBuilder::new(
            "scf.yield",
            Location::unknown(gen.context),
        )
        .build()
        .unwrap();
        then_b.append_operation(yield_op);
        then_region.append_block(then_b);

        let else_region = Region::new();
        let else_b = Block::new(&[]);
        if let Some(else_block) = else_block_opt {
            for stmt in else_block {
                gen.generate_statement(stmt, &else_b);
            }
        }
        let yield_op = melior::ir::operation::OperationBuilder::new(
            "scf.yield",
            Location::unknown(gen.context),
        )
        .build()
        .unwrap();
        else_b.append_operation(yield_op);
        else_region.append_block(else_b);

        let if_op =
            melior::ir::operation::OperationBuilder::new("scf.if", Location::unknown(gen.context))
                .add_operands(&[cond_val])
                .add_regions([then_region, else_region])
                .build()
                .unwrap();

        block.append_operation(if_op);

        let ty = Type::parse(gen.context, "i32").unwrap();
        let op = melior::ir::operation::OperationBuilder::new(
            "arith.constant",
            Location::unknown(gen.context),
        )
        .add_results(&[ty])
        .add_attributes(&[(
            melior::ir::Identifier::new(gen.context, "value"),
            melior::ir::attribute::IntegerAttribute::new(ty, 0).into(),
        )])
        .build()
        .unwrap();
        let op_ref = block.append_operation(op);
        (op_ref.result(0).unwrap().into(), ty)
    }
}

impl<'c> LowerToMelior<'c> for NumberExpr {
    type Output = (Value<'c, 'c>, Type<'c>);
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let NumberExpr {
            value: val_str,
            ty: ast_ty_opt,
            span: _,
        } = self;
        let ty = if let Some(ast_ty) = ast_ty_opt {
            gen.lower_type(&crate::ast::Type::Scalar(ast_ty.clone()))
        } else if val_str.contains('.') {
            Type::parse(gen.context, "f32").unwrap()
        } else {
            Type::parse(gen.context, "i32").unwrap()
        };
        let ty_str = ty.to_string();
        if ty_str.contains("f32")
            || ty_str.contains("f64")
            || ty_str.contains("f16")
            || ty_str.contains("bf16")
        {
            let op = melior::ir::operation::OperationBuilder::new(
                "arith.constant",
                Location::unknown(gen.context),
            )
            .add_results(&[ty])
            .add_attributes(&[(
                melior::ir::Identifier::new(gen.context, "value"),
                melior::ir::attribute::FloatAttribute::new(
                    gen.context,
                    ty,
                    val_str.parse::<f64>().unwrap(),
                )
                .into(),
            )])
            .build()
            .unwrap();
            let op_ref = block.append_operation(op);
            (op_ref.result(0).unwrap().into(), ty)
        } else {
            let op = melior::ir::operation::OperationBuilder::new(
                "arith.constant",
                Location::unknown(gen.context),
            )
            .add_results(&[ty])
            .add_attributes(&[(
                melior::ir::Identifier::new(gen.context, "value"),
                melior::ir::attribute::IntegerAttribute::new(ty, val_str.parse::<i64>().unwrap())
                    .into(),
            )])
            .build()
            .unwrap();
            let op_ref = block.append_operation(op);
            (op_ref.result(0).unwrap().into(), ty)
        }
    }
}
impl<'c> LowerToMelior<'c> for ReturnStmt {
    type Output = ();
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let ReturnStmt { expr, span: _ } = self;
        let (mut val, expr_ty) = gen.generate_expr(expr, block);
        if let Some(ret_ty) = gen.current_return_type {
            if expr_ty != ret_ty {
                if expr_ty.to_string().starts_with("memref<")
                    && ret_ty.to_string().starts_with("memref<")
                {
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

                    let cast_op = melior::ir::operation::OperationBuilder::new(
                        cast_op_name,
                        Location::unknown(gen.context),
                    )
                    .add_operands(&[val])
                    .add_results(&[ret_ty])
                    .build()
                    .unwrap();
                    val = block.append_operation(cast_op).result(0).unwrap().into();
                } else if ret_ty.to_string() == "i32" && expr_ty.to_string().starts_with("memref<")
                {
                    let zero_op = melior::ir::operation::OperationBuilder::new(
                        "arith.constant",
                        Location::unknown(gen.context),
                    )
                    .add_results(&[ret_ty])
                    .add_attributes(&[(
                        melior::ir::Identifier::new(gen.context, "value"),
                        melior::ir::attribute::IntegerAttribute::new(ret_ty, 0).into(),
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
        let ret_op =
            melior::ir::operation::OperationBuilder::new(op_name, Location::unknown(gen.context))
                .add_operands(&[val])
                .build()
                .unwrap();
        block.append_operation(ret_op);
    }
}

impl<'c> LowerToMelior<'c> for LetDeclStmt {
    type Output = ();
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
        let (val, ty) = gen.generate_expr(expr, block);
        gen.expected_type = prev_expected;
        if *is_mut {
            let memref_ty = format!("memref<{}>", ty);
            let parsed_memref_ty = Type::parse(gen.context, &memref_ty).unwrap();
            let alloca_op = melior::ir::operation::OperationBuilder::new(
                "memref.alloca",
                Location::unknown(gen.context),
            )
            .add_results(&[parsed_memref_ty])
            .build()
            .unwrap();
            let alloca_ref = block.append_operation(alloca_op);
            let alloca_val = alloca_ref.result(0).unwrap().into();

            let store_op = melior::ir::operation::OperationBuilder::new(
                "memref.store",
                Location::unknown(gen.context),
            )
            .add_operands(&[val, alloca_val])
            .build()
            .unwrap();
            block.append_operation(store_op);

            gen.env.insert(name.clone(), (alloca_val, parsed_memref_ty));
        } else {
            gen.env.insert(name.clone(), (val, ty));
        }
    }
}
impl<'c> LowerToMelior<'c> for AssignStmt {
    type Output = ();
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
        let (rhs_val, rhs_ty) = gen.generate_expr(rhs, block);
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
                        let cast_op = melior::ir::operation::OperationBuilder::new(
                            "arith.index_cast",
                            Location::unknown(gen.context),
                        )
                        .add_operands(&[rhs_val])
                        .add_results(&[inner_ty])
                        .build()
                        .unwrap();

                        store_val = block.append_operation(cast_op).result(0).unwrap().into();
                    }

                    let store_op = melior::ir::operation::OperationBuilder::new(
                        "memref.store",
                        Location::unknown(gen.context),
                    )
                    .add_operands(&[store_val, mem_val])
                    .build()
                    .unwrap();
                    block.append_operation(store_op);
                } else {
                    gen.env.insert(name.clone(), (rhs_val, rhs_ty));
                }
            }
        } else if let Expr::IndexAccess(crate::ast::IndexAccessExpr {
            base,
            index: _,
            span: _,
        }) = lhs
        {
            if let Some((base_val, base_ty, indices)) = gen.flatten_indices(
                &crate::ast::Expr::IndexAccess(crate::ast::IndexAccessExpr {
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
                    let i64_ty = Type::parse(gen.context, "i64").unwrap();
                    let cast_op = melior::ir::operation::OperationBuilder::new(
                        "arith.index_cast",
                        Location::unknown(gen.context),
                    )
                    .add_operands(&[indices[0]])
                    .add_results(&[i64_ty])
                    .build()
                    .unwrap();
                    let idx_i64 = block.append_operation(cast_op).result(0).unwrap().into();

                    let gep_op = melior::ir::operation::OperationBuilder::new(
                        "llvm.getelementptr",
                        Location::unknown(gen.context),
                    )
                    .add_attributes(&[
                        (
                            melior::ir::Identifier::new(gen.context, "rawConstantIndices"),
                            melior::ir::attribute::DenseI32ArrayAttribute::new(
                                gen.context,
                                &[-2147483648],
                            )
                            .into(),
                        ),
                        (
                            melior::ir::Identifier::new(gen.context, "elem_type"),
                            melior::ir::attribute::TypeAttribute::new(rhs_ty).into(),
                        ),
                    ])
                    .add_operands(&[base_val, idx_i64])
                    .add_results(&[base_ty])
                    .build()
                    .unwrap();

                    let gep_ref = block.append_operation(gep_op);
                    let ptr_val = gep_ref.result(0).unwrap().into();

                    let store_op = melior::ir::operation::OperationBuilder::new(
                        "llvm.store",
                        Location::unknown(gen.context),
                    )
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

                    let mut store_builder = melior::ir::operation::OperationBuilder::new(
                        "memref.store",
                        Location::unknown(gen.context),
                    )
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
            struct_name: _,
            span: _,
        }) = lhs
        {
            if let Expr::Identifier(IdentifierExpr {
                name: base_name,
                span: _,
            }) = &**base
            {
                let (base_val, base_ty) = gen.generate_expr(base, block);
                let base_ty_str = base_ty.to_string();

                let mut struct_name_opt = None;
                if let Some(start_idx) = base_ty_str.find('"') {
                    if let Some(end_idx) = base_ty_str[start_idx + 1..].find('"') {
                        struct_name_opt =
                            Some(base_ty_str[start_idx + 1..start_idx + 1 + end_idx].to_string());
                    }
                }

                if let Some(struct_name) = struct_name_opt {
                    if let Some(struct_decl) = gen.structs.get(&struct_name).cloned() {
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
                                let cast_op = melior::ir::operation::OperationBuilder::new(
                                    "arith.index_cast",
                                    Location::unknown(gen.context),
                                )
                                .add_operands(&[rhs_val])
                                .add_results(&[field_ty])
                                .build()
                                .unwrap();
                                field_val =
                                    block.append_operation(cast_op).result(0).unwrap().into();
                            }

                            let pos_attr = melior::ir::attribute::DenseI64ArrayAttribute::new(
                                gen.context,
                                &[field_idx as i64],
                            );
                            let insert_op = melior::ir::operation::OperationBuilder::new(
                                "llvm.insertvalue",
                                Location::unknown(gen.context),
                            )
                            .add_operands(&[base_val, field_val])
                            .add_attributes(&[(
                                melior::ir::Identifier::new(gen.context, "position"),
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
                                    let store_op = melior::ir::operation::OperationBuilder::new(
                                        "memref.store",
                                        Location::unknown(gen.context),
                                    )
                                    .add_operands(&[new_struct_val, mem_val])
                                    .build()
                                    .unwrap();
                                    block.append_operation(store_op);
                                } else {
                                    gen.env.insert(base_name.clone(), (new_struct_val, base_ty));
                                }
                            }
                        }
                    }
                }
            } else {
                panic!("Complex struct assignment lhs not supported");
            }
        }
    }
}

impl<'c> LowerToMelior<'c> for CompoundAssignStmt {
    type Output = ();
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let CompoundAssignStmt {
            lhs,
            op,
            rhs,
            span: _,
        } = self;
        let (rhs_val, rhs_ty) = gen.generate_expr(rhs, block);
        let (lhs_val, ty) = gen.generate_expr(lhs, block);

        let mut actual_rhs = rhs_val;
        if rhs_ty != ty
            && ((rhs_ty.to_string() == "index" && ty.to_string() == "i32")
                || (rhs_ty.to_string() == "i32" && ty.to_string() == "index"))
        {
            let cast_op = melior::ir::operation::OperationBuilder::new(
                "arith.index_cast",
                Location::unknown(gen.context),
            )
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
        let bin_op = melior::ir::operation::OperationBuilder::new(
            op.get_op_name(is_float),
            Location::unknown(gen.context),
        )
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
                    let store_op = melior::ir::operation::OperationBuilder::new(
                        "memref.store",
                        Location::unknown(gen.context),
                    )
                    .add_operands(&[result_val, mem_val])
                    .build()
                    .unwrap();
                    block.append_operation(store_op);
                } else {
                    gen.env.insert(name.clone(), (result_val, ty));
                }
            }
        } else if let Expr::IndexAccess(crate::ast::IndexAccessExpr {
            base,
            index: _,
            span: _,
        }) = lhs
        {
            if let Some((mem_val, mem_ty, indices)) = gen.flatten_indices(
                &crate::ast::Expr::IndexAccess(crate::ast::IndexAccessExpr {
                    base: base.clone(),
                    index: match lhs {
                        Expr::IndexAccess(i) => i.index.clone(),
                        _ => unreachable!(),
                    },
                    span: match lhs {
                        Expr::IndexAccess(i) => i.span.clone(),
                        _ => unreachable!(),
                    },
                }),
                block,
            ) {
                let mem_ty_str = mem_ty.to_string();
                if mem_ty_str.starts_with("memref<") {
                    let mut operands = vec![result_val, mem_val];
                    operands.extend(indices);
                    let store_op = melior::ir::operation::OperationBuilder::new(
                        "memref.store",
                        Location::unknown(gen.context),
                    )
                    .add_operands(&operands)
                    .build()
                    .unwrap();
                    block.append_operation(store_op);
                }
            }
        }
    }
}

impl<'c> LowerToMelior<'c> for ExprStmtStmt {
    type Output = ();
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let ExprStmtStmt {
            expr,
            has_semi: _,
            span: _,
        } = self;
        gen.generate_expr(expr, block);
    }
}

impl<'c> LowerToMelior<'c> for ForLoopStmt {
    type Output = ();
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let ForLoopStmt {
            iter,
            start,
            end,
            body,
            span: _,
        } = self;
        let (start_val, start_ty) = gen.generate_expr(start, block);
        let (end_val, end_ty) = gen.generate_expr(end, block);

        let ty_index = Type::parse(gen.context, "index").unwrap();

        // cast start/end to index if necessary
        let start_idx = if start_ty == ty_index {
            start_val
        } else {
            let cast_start_op = melior::ir::operation::OperationBuilder::new(
                "arith.index_cast",
                Location::unknown(gen.context),
            )
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
            let cast_end_op = melior::ir::operation::OperationBuilder::new(
                "arith.index_cast",
                Location::unknown(gen.context),
            )
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

        let body_region = Region::new();
        let body_block = Block::new(&[(ty_index, Location::unknown(gen.context))]);

        let iter_val = body_block.argument(0).unwrap().into();
        let prev_env_val = gen.env.get(iter).cloned();
        gen.env.insert(iter.clone(), (iter_val, ty_index));

        for stmt in body {
            gen.generate_statement(stmt, &body_block);
        }

        if let Some(prev) = prev_env_val {
            gen.env.insert(iter.clone(), prev);
        } else {
            gen.env.remove(iter);
        }

        let yield_op = melior::ir::operation::OperationBuilder::new(
            "scf.yield",
            Location::unknown(gen.context),
        )
        .build()
        .unwrap();
        body_block.append_operation(yield_op);
        body_region.append_block(body_block);

        let step_val = melior::ir::operation::OperationBuilder::new(
            "arith.constant",
            Location::unknown(gen.context),
        )
        .add_attributes(&[(
            melior::ir::Identifier::new(gen.context, "value"),
            melior::ir::attribute::IntegerAttribute::new(
                Type::parse(gen.context, "index").unwrap(),
                1,
            )
            .into(),
        )])
        .add_results(&[ty_index])
        .build()
        .unwrap();
        let step_idx = block.append_operation(step_val).result(0).unwrap().into();

        let for_op =
            melior::ir::operation::OperationBuilder::new("scf.for", Location::unknown(gen.context))
                .add_operands(&[start_idx, end_idx, step_idx])
                .add_regions([body_region])
                .build()
                .unwrap();
        block.append_operation(for_op);
    }
}

fn emit_enzyme_decl<'c>(
    gen: &mut MeliorGenerator<'c>,
    base_name: &str,
    target_fn: &str,
    arg_tys: &[Type<'c>],
    ret_ty: Type<'c>,
) -> String {
    let prefix = match base_name {
        "fwddiff" => "__enzyme_fwddiff",
        _ => "__enzyme_autodiff",
    };
    let suffix = match base_name {
        "fwddiff" => "jvp",
        "grad" => "grad",
        "vjp" => "vjp",
        _ => base_name,
    };
    let enzyme_name = format!("{}_{}_{}", prefix, suffix, target_fn);
    if !gen.functions.contains_key(&enzyme_name) && gen.enzyme_decls.insert(enzyme_name.clone()) {
        let func_type = melior::ir::r#type::FunctionType::new(gen.context, arg_tys, &[ret_ty]);
        let _name_attr = melior::ir::attribute::StringAttribute::new(gen.context, &enzyme_name);
        let _type_attr = melior::ir::attribute::TypeAttribute::new(func_type.into());

        let region = melior::ir::Region::new();
        let func_op = melior::ir::operation::OperationBuilder::new(
            "func.func",
            Location::unknown(gen.context),
        )
        .add_attributes(&[
            (
                melior::ir::Identifier::new(gen.context, "sym_name"),
                melior::ir::attribute::StringAttribute::new(gen.context, &enzyme_name).into(),
            ),
            (
                melior::ir::Identifier::new(gen.context, "function_type"),
                melior::ir::attribute::TypeAttribute::new(func_type.into()).into(),
            ),
            (
                melior::ir::Identifier::new(gen.context, "sym_visibility"),
                melior::ir::attribute::StringAttribute::new(gen.context, "private").into(),
            ),
        ])
        .add_regions([region])
        .build()
        .unwrap();

        gen.module.body().append_operation(func_op);
    }
    enzyme_name
}

impl<'c> LowerToMelior<'c> for GradExpr {
    type Output = (Value<'c, 'c>, Type<'c>);
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
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
        let const_op = melior::ir::operation::OperationBuilder::new(
            "func.constant",
            Location::unknown(gen.context),
        )
        .add_attributes(&[(
            melior::ir::Identifier::new(gen.context, "value"),
            melior::ir::attribute::FlatSymbolRefAttribute::new(gen.context, target_fn).into(),
        )])
        .add_results(&[fn_ty.into()])
        .build()
        .unwrap();
        let const_ref = block.append_operation(const_op);
        let target_fn_val = const_ref.result(0).unwrap().into();

        let mut arg_vals = vec![target_fn_val];
        let mut enzyme_arg_types = vec![fn_ty.into()];

        for arg in args {
            let (v, ty) = gen.generate_expr(arg, block);
            arg_vals.push(v);
            enzyme_arg_types.push(ty);
        }

        let enzyme_name = emit_enzyme_decl(gen, "grad", target_fn, &enzyme_arg_types, ret_ty);

        let name_attr =
            melior::ir::attribute::FlatSymbolRefAttribute::new(gen.context, &enzyme_name);
        let call_op = melior::ir::operation::OperationBuilder::new(
            "func.call",
            Location::unknown(gen.context),
        )
        .add_operands(&arg_vals)
        .add_results(&[ret_ty])
        .add_attributes(&[(
            melior::ir::Identifier::new(gen.context, "callee"),
            name_attr.into(),
        )])
        .build()
        .unwrap();

        let call_ref = block.append_operation(call_op);
        (call_ref.result(0).unwrap().into(), ret_ty)
    }
}

impl<'c> LowerToMelior<'c> for VjpExpr {
    type Output = (Value<'c, 'c>, Type<'c>);
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
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
        let const_op = melior::ir::operation::OperationBuilder::new(
            "func.constant",
            Location::unknown(gen.context),
        )
        .add_attributes(&[(
            melior::ir::Identifier::new(gen.context, "value"),
            melior::ir::attribute::FlatSymbolRefAttribute::new(gen.context, target_fn).into(),
        )])
        .add_results(&[fn_ty.into()])
        .build()
        .unwrap();
        let const_ref = block.append_operation(const_op);
        let target_fn_val = const_ref.result(0).unwrap().into();

        let mut arg_vals = vec![target_fn_val];
        let mut enzyme_arg_types = vec![fn_ty.into()];

        for arg in args {
            let (v, ty) = gen.generate_expr(arg, block);
            arg_vals.push(v);
            enzyme_arg_types.push(ty);
        }

        // For a scalar VJP in Enzyme, we just compute the gradient (implicitly seed=1.0)
        // and then multiply by the cotangent seed.
        let enzyme_name = emit_enzyme_decl(gen, "grad", target_fn, &enzyme_arg_types, ret_ty);

        let name_attr =
            melior::ir::attribute::FlatSymbolRefAttribute::new(gen.context, &enzyme_name);
        let call_op = melior::ir::operation::OperationBuilder::new(
            "func.call",
            Location::unknown(gen.context),
        )
        .add_operands(&arg_vals)
        .add_results(&[ret_ty])
        .add_attributes(&[(
            melior::ir::Identifier::new(gen.context, "callee"),
            name_attr.into(),
        )])
        .build()
        .unwrap();

        let call_ref = block.append_operation(call_op);
        let grad_val = call_ref.result(0).unwrap().into();

        let (c_val, _) = gen.generate_expr(cotangent, block);

        // Multiply grad by cotangent
        let is_float = ret_ty.to_string().contains("f32")
            || ret_ty.to_string().contains("f64")
            || ret_ty.to_string().contains("f16")
            || ret_ty.to_string().contains("bf16");
        let op_name = if is_float { "arith.mulf" } else { "arith.muli" };
        let mul_op =
            melior::ir::operation::OperationBuilder::new(op_name, Location::unknown(gen.context))
                .add_operands(&[grad_val, c_val])
                .add_results(&[ret_ty])
                .build()
                .unwrap();
        let mul_ref = block.append_operation(mul_op);

        (mul_ref.result(0).unwrap().into(), ret_ty)
    }
}

impl<'c> LowerToMelior<'c> for JvpExpr {
    type Output = (Value<'c, 'c>, Type<'c>);
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
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
        let const_op = melior::ir::operation::OperationBuilder::new(
            "func.constant",
            Location::unknown(gen.context),
        )
        .add_attributes(&[(
            melior::ir::Identifier::new(gen.context, "value"),
            melior::ir::attribute::FlatSymbolRefAttribute::new(gen.context, target_fn).into(),
        )])
        .add_results(&[fn_ty.into()])
        .build()
        .unwrap();
        let const_ref = block.append_operation(const_op);
        let target_fn_val = const_ref.result(0).unwrap().into();

        let mut arg_vals = vec![target_fn_val];
        let mut enzyme_arg_types = vec![fn_ty.into()];

        for arg in args {
            let (v, ty) = gen.generate_expr(arg, block);
            arg_vals.push(v);
            enzyme_arg_types.push(ty);
        }

        let (t_val, t_ty) = gen.generate_expr(tangent, block);
        arg_vals.push(t_val);
        enzyme_arg_types.push(t_ty);

        // Enzyme intercepts `__enzyme_fwddiff` for forward mode.
        let enzyme_name = emit_enzyme_decl(gen, "fwddiff", target_fn, &enzyme_arg_types, ret_ty);

        let name_attr =
            melior::ir::attribute::FlatSymbolRefAttribute::new(gen.context, &enzyme_name);
        let call_op = melior::ir::operation::OperationBuilder::new(
            "func.call",
            Location::unknown(gen.context),
        )
        .add_operands(&arg_vals)
        .add_results(&[ret_ty])
        .add_attributes(&[(
            melior::ir::Identifier::new(gen.context, "callee"),
            name_attr.into(),
        )])
        .build()
        .unwrap();

        let call_ref = block.append_operation(call_op);
        (call_ref.result(0).unwrap().into(), ret_ty)
    }
}

use crate::codegen::break_utils::*;

impl<'c> LowerToMelior<'c> for LoopStmt {
    type Output = ();
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let i1_ty = Type::parse(gen.context, "i1").unwrap();
        let memref_ty = Type::parse(gen.context, "memref<1xi1>").unwrap();

        let alloca_op = block.append_operation(
            melior::ir::operation::OperationBuilder::new(
                "memref.alloca",
                Location::unknown(gen.context),
            )
            .add_results(&[memref_ty])
            .build()
            .unwrap(),
        );
        let break_ptr = alloca_op.result(0).unwrap().into();

        let false_op = block.append_operation(
            melior::ir::operation::OperationBuilder::new(
                "arith.constant",
                Location::unknown(gen.context),
            )
            .add_results(&[i1_ty])
            .add_attributes(&[(
                melior::ir::Identifier::new(gen.context, "value"),
                melior::ir::attribute::IntegerAttribute::new(i1_ty, 0).into(),
            )])
            .build()
            .unwrap(),
        );
        let false_val = false_op.result(0).unwrap().into();

        let c0_op = block.append_operation(
            melior::ir::operation::OperationBuilder::new(
                "arith.constant",
                Location::unknown(gen.context),
            )
            .add_results(&[Type::parse(gen.context, "index").unwrap()])
            .add_attributes(&[(
                melior::ir::Identifier::new(gen.context, "value"),
                melior::ir::attribute::IntegerAttribute::new(
                    Type::parse(gen.context, "index").unwrap(),
                    0,
                )
                .into(),
            )])
            .build()
            .unwrap(),
        );
        let c0_idx = c0_op.result(0).unwrap().into();

        block.append_operation(
            melior::ir::operation::OperationBuilder::new(
                "memref.store",
                Location::unknown(gen.context),
            )
            .add_operands(&[false_val, break_ptr, c0_idx])
            .build()
            .unwrap(),
        );

        gen.break_flags.push(break_ptr);

        let before_region = Region::new();
        let before_block = Block::new(&[]);

        let load_op = before_block.append_operation(
            melior::ir::operation::OperationBuilder::new(
                "memref.load",
                Location::unknown(gen.context),
            )
            .add_operands(&[break_ptr, c0_idx])
            .add_results(&[i1_ty])
            .build()
            .unwrap(),
        );
        let is_break = load_op.result(0).unwrap().into();

        let true_op = before_block.append_operation(
            melior::ir::operation::OperationBuilder::new(
                "arith.constant",
                Location::unknown(gen.context),
            )
            .add_results(&[i1_ty])
            .add_attributes(&[(
                melior::ir::Identifier::new(gen.context, "value"),
                melior::ir::attribute::IntegerAttribute::new(i1_ty, 1).into(),
            )])
            .build()
            .unwrap(),
        );
        let true_val = true_op.result(0).unwrap().into();

        let not_break_op = before_block.append_operation(
            melior::ir::operation::OperationBuilder::new(
                "arith.xori",
                Location::unknown(gen.context),
            )
            .add_operands(&[is_break, true_val])
            .add_results(&[i1_ty])
            .build()
            .unwrap(),
        );
        let not_break = not_break_op.result(0).unwrap().into();

        before_block.append_operation(
            melior::ir::operation::OperationBuilder::new(
                "scf.condition",
                Location::unknown(gen.context),
            )
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
            c0_idx,
            i1_ty,
        );

        after_block.append_operation(
            melior::ir::operation::OperationBuilder::new(
                "scf.yield",
                Location::unknown(gen.context),
            )
            .build()
            .unwrap(),
        );
        after_region.append_block(after_block);

        block.append_operation(
            melior::ir::operation::OperationBuilder::new(
                "scf.while",
                Location::unknown(gen.context),
            )
            .add_regions([before_region, after_region])
            .build()
            .unwrap(),
        );

        gen.break_flags.pop();
    }
}

impl<'c> LowerToMelior<'c> for BreakStmt {
    type Output = ();
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        if let Some(&break_ptr) = gen.break_flags.last() {
            let i1_ty = Type::parse(gen.context, "i1").unwrap();
            let true_op = block.append_operation(
                melior::ir::operation::OperationBuilder::new(
                    "arith.constant",
                    Location::unknown(gen.context),
                )
                .add_results(&[i1_ty])
                .add_attributes(&[(
                    melior::ir::Identifier::new(gen.context, "value"),
                    melior::ir::attribute::IntegerAttribute::new(i1_ty, 1).into(),
                )])
                .build()
                .unwrap(),
            );
            let true_val = true_op.result(0).unwrap().into();

            let c0_op = block.append_operation(
                melior::ir::operation::OperationBuilder::new(
                    "arith.constant",
                    Location::unknown(gen.context),
                )
                .add_results(&[Type::parse(gen.context, "index").unwrap()])
                .add_attributes(&[(
                    melior::ir::Identifier::new(gen.context, "value"),
                    melior::ir::attribute::IntegerAttribute::new(
                        Type::parse(gen.context, "index").unwrap(),
                        0,
                    )
                    .into(),
                )])
                .build()
                .unwrap(),
            );
            let c0_idx = c0_op.result(0).unwrap().into();

            block.append_operation(
                melior::ir::operation::OperationBuilder::new(
                    "memref.store",
                    Location::unknown(gen.context),
                )
                .add_operands(&[true_val, break_ptr, c0_idx])
                .build()
                .unwrap(),
            );
        } else {
            panic!("break outside of a loop");
        }
    }
}

pub fn generate_match_chain<'c>(
    gen: &mut MeliorGenerator<'c>,
    arms: &[MatchArm],
    match_val: melior::ir::Value<'c, 'c>,
    _match_ty: melior::ir::Type<'c>,
    block: &melior::ir::Block<'c>,
) {
    if arms.is_empty() {
        return;
    }

    let arm = &arms[0];

    if let Pattern::Wildcard = arm.pattern {
        // Wildcard matches unconditionally.
        for stmt in &arm.body {
            gen.generate_statement(stmt, block);
        }
        return;
    }

    // Evaluate condition
    let cond_val = match &arm.pattern {
        Pattern::EnumVariant(_, variant_name, _) => {
            // For now, if Enums are represented as i32 tags, we check equality.
            // We need to look up the variant's tag value.
            // Let's assume `match_val` is an `i32` for simplicity, or we do a generic equality check.

            // We'll just generate an arith.cmpi!
            let i32_ty = melior::ir::r#type::IntegerType::new(gen.context, 32).into();

            // Find variant tag
            let mut tag_val = 0;
            // Hack: just parse the variant name if it's a number, or assume 0.
            // Real enums should look up the tag in `gen.enums`.
            for enum_def in gen.enums.values() {
                for (i, v) in enum_def.iter().enumerate() {
                    if v.0 == *variant_name {
                        tag_val = i as i64;
                        break;
                    }
                }
            }

            let tag_op = block.append_operation(
                melior::ir::operation::OperationBuilder::new(
                    "arith.constant",
                    melior::ir::Location::unknown(gen.context),
                )
                .add_results(&[i32_ty])
                .add_attributes(&[(
                    melior::ir::Identifier::new(gen.context, "value"),
                    melior::ir::attribute::IntegerAttribute::new(i32_ty, tag_val).into(),
                )])
                .build()
                .unwrap(),
            );
            let tag = tag_op.result(0).unwrap().into();

            let cmp_op = block.append_operation(
                melior::ir::operation::OperationBuilder::new(
                    "arith.cmpi",
                    melior::ir::Location::unknown(gen.context),
                )
                .add_operands(&[match_val, tag])
                .add_results(&[melior::ir::r#type::IntegerType::new(gen.context, 1).into()])
                .add_attributes(&[(
                    melior::ir::Identifier::new(gen.context, "predicate"),
                    melior::ir::attribute::IntegerAttribute::new(
                        melior::ir::r#type::IntegerType::new(gen.context, 64).into(),
                        0,
                    )
                    .into(),
                )]) // 0 = eq
                .build()
                .unwrap(),
            );
            cmp_op.result(0).unwrap().into()
        }
        _ => panic!("Unsupported pattern in codegen"),
    };

    let then_region = melior::ir::Region::new();
    let then_block = melior::ir::Block::new(&[]);
    for stmt in &arm.body {
        gen.generate_statement(stmt, &then_block);
    }
    then_block.append_operation(
        melior::ir::operation::OperationBuilder::new(
            "scf.yield",
            melior::ir::Location::unknown(gen.context),
        )
        .build()
        .unwrap(),
    );
    then_region.append_block(then_block);

    let else_region = melior::ir::Region::new();
    let else_block = melior::ir::Block::new(&[]);

    // Recursively generate the rest of the arms inside the else block
    generate_match_chain(gen, &arms[1..], match_val, _match_ty, &else_block);

    else_block.append_operation(
        melior::ir::operation::OperationBuilder::new(
            "scf.yield",
            melior::ir::Location::unknown(gen.context),
        )
        .build()
        .unwrap(),
    );
    else_region.append_block(else_block);

    block.append_operation(
        melior::ir::operation::OperationBuilder::new(
            "scf.if",
            melior::ir::Location::unknown(gen.context),
        )
        .add_operands(&[cond_val])
        .add_regions([then_region, else_region])
        .build()
        .unwrap(),
    );
}

impl<'c> LowerToMelior<'c> for MatchExpr {
    type Output = (melior::ir::Value<'c, 'c>, melior::ir::Type<'c>);
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let (match_val, match_ty) = gen.generate_expr(&self.expr, block);

        generate_match_chain(gen, &self.arms, match_val, match_ty, block);

        // Return dummy value for now like IfExpr
        let ty = melior::ir::r#type::IntegerType::new(gen.context, 32).into();
        let op = melior::ir::operation::OperationBuilder::new(
            "arith.constant",
            melior::ir::Location::unknown(gen.context),
        )
        .add_results(&[ty])
        .add_attributes(&[(
            melior::ir::Identifier::new(gen.context, "value"),
            melior::ir::attribute::IntegerAttribute::new(ty, 0).into(),
        )])
        .build()
        .unwrap();
        let op_ref = block.append_operation(op);
        (op_ref.result(0).unwrap().into(), ty)
    }
}

fn generate_statements_with_break_guard<'c>(
    gen: &mut MeliorGenerator<'c>,
    stmts: &[Statement],
    block: &melior::ir::Block<'c>,
    break_ptr: Value<'c, 'c>,
    c0_idx: Value<'c, 'c>,
    i1_ty: Type<'c>,
) {
    if stmts.is_empty() {
        return;
    }

    gen.generate_statement(&stmts[0], block);

    if stmts.len() > 1 {
        if contains_break(&stmts[0]) {
            let load_op = block.append_operation(
                melior::ir::operation::OperationBuilder::new(
                    "memref.load",
                    Location::unknown(gen.context),
                )
                .add_operands(&[break_ptr, c0_idx])
                .add_results(&[i1_ty])
                .build()
                .unwrap(),
            );
            let is_break = load_op.result(0).unwrap().into();

            let true_op = block.append_operation(
                melior::ir::operation::OperationBuilder::new(
                    "arith.constant",
                    Location::unknown(gen.context),
                )
                .add_results(&[i1_ty])
                .add_attributes(&[(
                    melior::ir::Identifier::new(gen.context, "value"),
                    melior::ir::attribute::IntegerAttribute::new(i1_ty, 1).into(),
                )])
                .build()
                .unwrap(),
            );
            let true_val = true_op.result(0).unwrap().into();

            let not_break_op = block.append_operation(
                melior::ir::operation::OperationBuilder::new(
                    "arith.xori",
                    Location::unknown(gen.context),
                )
                .add_operands(&[is_break, true_val])
                .add_results(&[i1_ty])
                .build()
                .unwrap(),
            );
            let not_break = not_break_op.result(0).unwrap().into();

            let if_region = Region::new();
            let if_block = Block::new(&[]);

            generate_statements_with_break_guard(
                gen,
                &stmts[1..],
                &if_block,
                break_ptr,
                c0_idx,
                i1_ty,
            );

            if_block.append_operation(
                melior::ir::operation::OperationBuilder::new(
                    "scf.yield",
                    Location::unknown(gen.context),
                )
                .build()
                .unwrap(),
            );
            if_region.append_block(if_block);

            let else_region = melior::ir::Region::new();
            let else_block = melior::ir::Block::new(&[]);
            let yield_op = melior::ir::operation::OperationBuilder::new(
                "scf.yield",
                Location::unknown(gen.context),
            )
            .build()
            .unwrap();
            else_block.append_operation(yield_op);
            else_region.append_block(else_block);

            block.append_operation(
                melior::ir::operation::OperationBuilder::new(
                    "scf.if",
                    Location::unknown(gen.context),
                )
                .add_operands(&[not_break])
                .add_regions([if_region, else_region])
                .build()
                .unwrap(),
            );
        } else {
            generate_statements_with_break_guard(gen, &stmts[1..], block, break_ptr, c0_idx, i1_ty);
        }
    }
}

impl<'c> LowerToMelior<'c> for EnumVariantExpr {
    type Output = (Value<'c, 'c>, Type<'c>);
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let EnumVariantExpr {
            enum_name,
            variant_name,
            payload: _,
            span: _,
        } = self;

        let mut tag_val = 0;
        if let Some(enum_def) = gen.enums.get(enum_name) {
            for (i, v) in enum_def.iter().enumerate() {
                if v.0 == *variant_name {
                    tag_val = i as i64;
                    break;
                }
            }
        }

        let i32_ty = Type::parse(gen.context, "i32").unwrap();
        let const_op = melior::ir::operation::OperationBuilder::new(
            "arith.constant",
            Location::unknown(gen.context),
        )
        .add_results(&[i32_ty])
        .add_attributes(&[(
            melior::ir::Identifier::new(gen.context, "value"),
            melior::ir::attribute::IntegerAttribute::new(i32_ty, tag_val).into(),
        )])
        .build()
        .unwrap();
        let const_ref = block.append_operation(const_op);

        (const_ref.result(0).unwrap().into(), i32_ty)
    }
}
