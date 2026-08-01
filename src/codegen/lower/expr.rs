use super::*;
use crate::syntax;
use crate::syntax::*;
use melior::ir::{
    attribute::{
        DenseI32ArrayAttribute, DenseI64ArrayAttribute, FlatSymbolRefAttribute, FloatAttribute,
        IntegerAttribute, StringAttribute, TypeAttribute,
    },
    operation::OperationBuilder,
    Attribute, Identifier, Region, Type, Value,
};

impl<'c> LowerToMelior<'c> for IdentifierExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let IdentifierExpr { name, span: _ } = self;
        if name.as_ref() == "true" || **name == *"false" {
            let i1_ty = gen.i1_ty;
            let val = if name.as_ref() == "true" { 1 } else { 0 };
            let const_op = OperationBuilder::new("arith.constant", gen.loc())
                .add_results(&[i1_ty])
                .add_attributes(&[(
                    Identifier::new(gen.context, "value"),
                    IntegerAttribute::new(i1_ty, val).into(),
                )])
                .build()?;
            let const_ref = block.append_operation(const_op);
            return Ok((const_ref.result(0)?.into(), i1_ty, block));
        }
        if let Some((val, ty)) = gen.env.get(name) {
            let ty_str = ty.to_string();
            if gen.allocs.contains(name.as_ref()) {
                if ty_str.starts_with("memref<") {
                    let inner_ty_str = &ty_str[7..ty_str.len() - 1];
                    let inner_ty = Type::parse(gen.context, inner_ty_str).unwrap_or_else(|| {
                        panic!("failed to parse {:?} for variable {:?}", inner_ty_str, name)
                    });
                    let load_op = OperationBuilder::new("memref.load", gen.loc())
                        .add_operands(&[*val])
                        .add_results(&[inner_ty])
                        .build()?;
                    let load_ref = block.append_operation(load_op);
                    return Ok((load_ref.result(0)?.into(), inner_ty, block));
                } else if ty_str.starts_with("!llvm.ptr")
                    || ty_str.starts_with("!llvm.struct")
                    || ty_str.starts_with("i")
                    || ty_str.starts_with("u")
                    || ty_str.starts_with("f")
                {
                    if gen.is_lvalue_context {
                        let ptr_ty = gen.ptr_ty;
                        return Ok((*val, ptr_ty, block));
                    }
                    let elem_ty = *ty;
                    let load_op = OperationBuilder::new("llvm.load", gen.loc())
                        .add_operands(&[*val])
                        .add_results(&[elem_ty])
                        .build()?;
                    let load_ref = block.append_operation(load_op);
                    return Ok((load_ref.result(0)?.into(), elem_ty, block));
                }
            }
            if ty_str.starts_with("memref<memref<") {
                let inner_ty_str = &ty_str[7..ty_str.len() - 1];
                let inner_ty = Type::parse(gen.context, inner_ty_str).ok_or_else(|| {
                    crate::codegen::lower::LowerError::ParseType("Type::parse failed".to_string())
                })?;
                let load_op = OperationBuilder::new("memref.load", gen.loc())
                    .add_operands(&[*val])
                    .add_results(&[inner_ty])
                    .build()?;
                let load_ref = block.append_operation(load_op);
                Ok((load_ref.result(0)?.into(), inner_ty, block))
            } else if ty_str.starts_with("memref<") && !ty_str.contains("x") {
                let inner_ty_str = &ty_str[7..ty_str.len() - 1];
                let inner_ty = Type::parse(gen.context, inner_ty_str).ok_or_else(|| {
                    crate::codegen::lower::LowerError::ParseType("Type::parse failed".to_string())
                })?;
                let load_op = OperationBuilder::new("memref.load", gen.loc())
                    .add_operands(&[*val])
                    .add_results(&[inner_ty])
                    .build()?;
                let load_ref = block.append_operation(load_op);
                Ok((load_ref.result(0)?.into(), inner_ty, block))
            } else {
                Ok((*val, *ty, block))
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
                .build()?;
            let const_ref = block.append_operation(const_op);

            let ptr_ty = gen.ptr_ty;
            let cast_op = OperationBuilder::new("builtin.unrealized_conversion_cast", gen.loc())
                .add_operands(&[const_ref.result(0)?.into()])
                .add_results(&[ptr_ty])
                .build()?;
            let cast_ref = block.append_operation(cast_op);

            Ok((cast_ref.result(0)?.into(), ptr_ty, block))
        } else {
            panic!("Undefined variable: {}", name);
        }
    }
}

impl<'c> LowerToMelior<'c> for BorrowExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let BorrowExpr { expr, .. } = self;
        if let Expr::Identifier(id) = &**expr {
            if gen.allocs.contains(&*id.name) {
                if let Some((val, val_ty)) = gen.env.get(&*id.name) {
                    let ptr_ty = gen.ptr_ty;
                    if gen.is_memref(val_ty) {
                        return Ok((*val, *val_ty, block));
                    } else if *val_ty == ptr_ty {
                        return Ok((*val, ptr_ty, block));
                    } else {
                        // Cast from val to ptr_ty if necessary? No, just return val_ty.
                        return Ok((*val, *val_ty, block));
                    }
                }
            } else if let Some((val, ptr_ty)) = gen
                .env
                .get(&*id.name)
                .filter(|(_, t)| *t == gen.ptr_ty)
                .map(|(v, t)| (*v, *t))
            {
                // `&r` where `r : &T` is a non-`mut` reference local: `let r = &x` bound it to a
                // bare SSA pointer with no backing storage, so there is no address to hand back.
                // Materialize a slot, store the pointer into it, and return the slot — otherwise
                // `&r` would alias `r`'s pointee (returning `r`'s value verbatim) and `**rr` reads
                // garbage / segfaults (#278). Only bare pointers take this path; a memref descriptor
                // or a non-pointer rvalue falls through to the general materialization below.
                let i32_ty = gen.i32_ty;
                let c1 = block
                    .append_operation(
                        OperationBuilder::new("llvm.mlir.constant", gen.loc())
                            .add_results(&[i32_ty])
                            .add_attributes(&[(
                                Identifier::new(gen.context, "value"),
                                IntegerAttribute::new(i32_ty, 1).into(),
                            )])
                            .build()?,
                    )
                    .result(0)?
                    .into();
                let slot = block
                    .append_operation(
                        OperationBuilder::new("llvm.alloca", gen.loc())
                            .add_operands(&[c1])
                            .add_results(&[ptr_ty])
                            .add_attributes(&[(
                                Identifier::new(gen.context, "elem_type"),
                                TypeAttribute::new(ptr_ty).into(),
                            )])
                            .build()?,
                    )
                    .result(0)?
                    .into();
                block.append_operation(
                    OperationBuilder::new("llvm.store", gen.loc())
                        .add_operands(&[val, slot])
                        .build()?,
                );
                return Ok((slot, ptr_ty, block));
            }
        }
        let prev_lvalue = gen.is_lvalue_context;
        gen.is_lvalue_context = true;
        let (val, ty, block) = gen.generate_expr(expr, block)?;
        gen.is_lvalue_context = prev_lvalue;
        let ptr_ty = gen.ptr_ty;
        if ty == ptr_ty {
            return Ok((val, ptr_ty, block));
        }
        if ty.to_string().starts_with("memref<memref<") {
            return Ok((val, ty, block));
        }
        if gen.is_memref(&ty) {
            // Allocate a pointer to the memref
            let alloca_op = block.append_operation(
                OperationBuilder::new("memref.alloca", gen.loc())
                    .add_results(&[Type::parse(gen.context, &format!("memref<{}>", ty)).unwrap()])
                    .build()?,
            );
            let ptr = alloca_op.result(0)?.into();

            block.append_operation(
                OperationBuilder::new("memref.store", gen.loc())
                    .add_operands(&[val, ptr])
                    .build()?,
            );
            return Ok((
                ptr,
                Type::parse(gen.context, &format!("memref<{}>", ty)).unwrap(),
                block,
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
                .build()?,
        );
        let c1 = c1_op.result(0)?.into();

        let alloca_op = block.append_operation(
            OperationBuilder::new("llvm.alloca", gen.loc())
                .add_operands(&[c1])
                .add_results(&[ptr_ty])
                .add_attributes(&[(
                    Identifier::new(gen.context, "elem_type"),
                    TypeAttribute::new(ty).into(),
                )])
                .build()?,
        );
        let ptr = alloca_op.result(0)?.into();

        block.append_operation(
            OperationBuilder::new("llvm.store", gen.loc())
                .add_operands(&[val, ptr])
                .build()?,
        );

        Ok((ptr, ptr_ty, block))
    }
}

impl<'c> LowerToMelior<'c> for StringLiteralExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
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
            .build()?;
        module_body.append_operation(global_op);

        let addressof_op = OperationBuilder::new("llvm.mlir.addressof", gen.loc())
            .add_attributes(&[(
                Identifier::new(gen.context, "global_name"),
                FlatSymbolRefAttribute::new(gen.context, &str_name).into(),
            )])
            .add_results(&[ptr_ty])
            .build()?;
        let addressof_ref = block.append_operation(addressof_op);

        Ok((addressof_ref.result(0)?.into(), ptr_ty, block))
    }
}

impl<'c> LowerToMelior<'c> for ComptimeBlockExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
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
                .build()?;
            let dummy_ref = block.append_operation(dummy_val);
            Ok((dummy_ref.result(0)?.into(), none_ty, block))
        }
    }
}

/// The pointee AST type of a pointer/borrow/ref (`*mut i32` -> `i32`), for recovering a dereference's
/// loaded type. `None` for a non-pointer. (#242)
fn deref_pointee_type(ty: &syntax::Type) -> Option<syntax::Type> {
    match ty {
        syntax::Type::Pointer(inner, ..)
        | syntax::Type::Borrow { inner, .. }
        | syntax::Type::Ref(inner, ..) => Some((**inner).clone()),
        _ => None,
    }
}

impl<'c> LowerToMelior<'c> for DereferenceExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let (ptr_val, ptr_ty, block) = gen.generate_expr(&self.expr, block)?;
        let ptr_ty_str = ptr_ty.to_string();

        // The pointee (loaded) type: the checker-set `self.ty` when present; else the pointee recovered
        // from the pointer expression's AST type (`p : *mut i32` -> `i32`) — MLIR opaque pointers carry
        // no pointee, so the old `!llvm.ptr<…>` string-parse never fired and fell through to a wrong
        // `f32` default (a bare `*p` on an i32 pointer returned garbage). Only if inference fails too do
        // we fall back to the string-parse/`f32` path. (#242)
        let inner_ty = if let Some(t) = &self.ty {
            gen.lower_type(t)?
        } else if let Some(inner) = gen
            .infer_ast_type(&self.expr)
            .and_then(|t| deref_pointee_type(&t))
        {
            gen.lower_type(&inner)?
        } else {
            let inner_ty_str = if ptr_ty_str.starts_with("!llvm.ptr<") {
                ptr_ty_str[10..ptr_ty_str.len() - 1].to_string()
            } else {
                "f32".to_string()
            };
            Type::parse(gen.context, &inner_ty_str).ok_or_else(|| {
                crate::codegen::lower::LowerError::ParseType("Type::parse failed".to_string())
            })?
        };

        if gen.is_lvalue_context {
            return Ok((ptr_val, ptr_ty, block));
        }

        let load_op = OperationBuilder::new("llvm.load", gen.loc())
            .add_operands(&[ptr_val])
            .add_results(&[inner_ty])
            .build()?;
        let load_ref = block.append_operation(load_op);
        Ok((load_ref.result(0)?.into(), inner_ty, block))
    }
}

impl<'c> LowerToMelior<'c> for syntax::IndexAccessExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let (base_val, base_ty, indices, block) = gen
            .flatten_indices(&syntax::Expr::IndexAccess(self.clone()), block)
            .expect("Failed to flatten indices for IndexAccess");

        let base_ty_str = base_ty.to_string();
        let is_ptr = base_ty_str.starts_with("!llvm.ptr");

        if is_ptr {
            let mut inferred_el_ty_str = None;
            if let Some(syntax::Type::Pointer(inner, _, _)) = gen.infer_ast_type(self.base.as_ref())
            {
                inferred_el_ty_str = Some(gen.lower_type_str(&inner)?);
            }

            let inner_ty_str = if base_ty_str.contains("<") {
                base_ty_str[base_ty_str.find('<').unwrap() + 1..base_ty_str.len() - 1].to_string()
            } else if let Some(el_str) = inferred_el_ty_str {
                el_str
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
                .build()?;
            let idx_i64 = block.append_operation(cast_op).result(0)?.into();

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
                .build()?;

            let gep_ref = block.append_operation(gep_op);
            let ptr_val = gep_ref.result(0)?.into();

            if gen.is_lvalue_context {
                let ptr_ty = gen.ptr_ty;
                return Ok((ptr_val, ptr_ty, block));
            }

            let load_op = OperationBuilder::new("llvm.load", gen.loc())
                .add_operands(&[ptr_val])
                .add_results(&[inner_ty])
                .build()?;

            let load_ref = block.append_operation(load_op);
            Ok((load_ref.result(0)?.into(), inner_ty, block))
        } else {
            let inner_ty_str = if base_ty_str.starts_with("memref<") {
                let inner = base_ty_str.replace("memref<", "").replace('>', "");
                let parts: Vec<&str> = inner.split('x').collect();
                let last_part = parts.last().unwrap();
                last_part.split(',').next().unwrap().trim().to_string()
            } else {
                "f32".to_string()
            };

            // Slice indexing (S1): fewer indices than the tensor's rank -> a rank-reduced view
            // of the remaining dimensions (e.g. `q[i]` on Tensor<f32,[N,D]> is row i, a
            // Tensor<f32,[D]>). The memref lowers to `?x?` but the Vx type carries the static
            // shape, so we emit a `memref.reinterpret_cast` with a static row size at the flat
            // offset `sum_m idx_m * stride_m`. This is what makes `dot(q[i], k[j])` (S2) possible.
            let base_dims: Vec<i64> = match gen.infer_ast_type(self.base.as_ref()) {
                Some(syntax::Type::Tensor(_, dims, _)) => dims
                    .iter()
                    .map(|d| match d {
                        syntax::Expr::Number(n) => n.value.as_ref().parse::<i64>().ok(),
                        _ => None,
                    })
                    .collect::<Option<Vec<i64>>>()
                    .unwrap_or_default(),
                _ => Vec::new(),
            };
            let rank = base_dims.len();
            if rank >= 2 && indices.len() < rank {
                let index_ty = Type::index(gen.context);
                // Flat offset = sum over each leading index of idx_m * prod(dims[m+1..]).
                let mut offset_val: Option<Value<'c, 'c>> = None;
                for (m, idx) in indices.iter().enumerate() {
                    let stride_elems: i64 = base_dims[m + 1..].iter().product();
                    let term = if stride_elems == 1 {
                        *idx
                    } else {
                        let c_op = OperationBuilder::new("arith.constant", gen.loc())
                            .add_attributes(&[(
                                Identifier::new(gen.context, "value"),
                                IntegerAttribute::new(index_ty, stride_elems).into(),
                            )])
                            .add_results(&[index_ty])
                            .build()?;
                        let c = block.append_operation(c_op).result(0)?.into();
                        let mul_op = OperationBuilder::new("arith.muli", gen.loc())
                            .add_operands(&[*idx, c])
                            .add_results(&[index_ty])
                            .build()?;
                        block.append_operation(mul_op).result(0)?.into()
                    };
                    offset_val = Some(match offset_val {
                        None => term,
                        Some(acc) => {
                            let add_op = OperationBuilder::new("arith.addi", gen.loc())
                                .add_operands(&[acc, term])
                                .add_results(&[index_ty])
                                .build()?;
                            block.append_operation(add_op).result(0)?.into()
                        }
                    });
                }
                let offset_val = offset_val.expect("partial index has >= 1 index");

                let result_dims: Vec<i64> = base_dims[indices.len()..].to_vec();
                // Row-major contiguous strides for the remaining dims.
                let mut result_strides: Vec<i64> = vec![1; result_dims.len()];
                for i in (0..result_dims.len().saturating_sub(1)).rev() {
                    result_strides[i] = result_strides[i + 1] * result_dims[i + 1];
                }
                let dims_str = result_dims
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join("x");
                let strides_str = result_strides
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                let result_ty_str = format!(
                    "memref<{}x{}, strided<[{}], offset: ?>>",
                    dims_str, inner_ty_str, strides_str
                );
                let result_ty = Type::parse(gen.context, &result_ty_str).ok_or_else(|| {
                    crate::codegen::lower::LowerError::ParseType(result_ty_str.clone())
                })?;
                let dyn_offset = i64::MIN; // ShapedType::kDynamic marker
                let reinterp = OperationBuilder::new("memref.reinterpret_cast", gen.loc())
                    .add_operands(&[base_val, offset_val])
                    .add_attributes(&[
                        (
                            Identifier::new(gen.context, "operandSegmentSizes"),
                            DenseI32ArrayAttribute::new(gen.context, &[1, 1, 0, 0]).into(),
                        ),
                        (
                            Identifier::new(gen.context, "static_offsets"),
                            DenseI64ArrayAttribute::new(gen.context, &[dyn_offset]).into(),
                        ),
                        (
                            Identifier::new(gen.context, "static_sizes"),
                            DenseI64ArrayAttribute::new(gen.context, &result_dims).into(),
                        ),
                        (
                            Identifier::new(gen.context, "static_strides"),
                            DenseI64ArrayAttribute::new(gen.context, &result_strides).into(),
                        ),
                    ])
                    .add_results(&[result_ty])
                    .build()?;
                let r = block.append_operation(reinterp).result(0)?.into();
                return Ok((r, result_ty, block));
            }

            let inner_ty = Type::parse(gen.context, &inner_ty_str).ok_or_else(|| {
                crate::codegen::lower::LowerError::ParseType("Type::parse failed".to_string())
            })?;
            let mut load_builder =
                OperationBuilder::new("memref.load", gen.loc()).add_operands(&[base_val]);

            for idx in indices {
                load_builder = load_builder.add_operands(&[idx]);
            }

            let load_op = load_builder.add_results(&[inner_ty]).build()?;

            let load_ref = block.append_operation(load_op);
            Ok((load_ref.result(0)?.into(), inner_ty, block))
        }
    }
}

impl<'c> LowerToMelior<'c> for BinaryOpExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let BinaryOpExpr {
            lhs,
            op,
            rhs,
            span: _,
        } = self;
        let (mut lhs_val, lhs_ty, block) = gen.generate_expr(lhs, block)?;
        let prev_expected = gen.expected_type;
        gen.expected_type = Some(lhs_ty);
        let (mut rhs_val, mut rhs_ty, block) = gen.generate_expr(rhs, block)?;
        gen.expected_type = prev_expected;

        // Slice elementwise (S3): if either operand is a rank-1 f32 slice/vector, lower to
        // `vector.load`/`vector.broadcast` + `arith.{mulf,addf,subf,divf}` -> `vector<Dxf32>`
        // (SIMD). The result flows to a `vector.store` at the assignment site (see stmt.rs).
        if matches!(
            op,
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div
        ) {
            let lhs_ty_s = lhs_ty.to_string();
            let rhs_ty_s = rhs_ty.to_string();
            let is_slice = is_slice_operand(&lhs_ty_s) || is_slice_operand(&rhs_ty_s);
            if let Some(d) = is_slice
                .then(|| slice_vec_len(&lhs_ty_s).or_else(|| slice_vec_len(&rhs_ty_s)))
                .flatten()
            {
                let va = to_vector(gen, lhs_val, &lhs_ty_s, d, block)?;
                let vb = to_vector(gen, rhs_val, &rhs_ty_s, d, block)?;
                let vec_ty = Type::parse(gen.context, &format!("vector<{}xf32>", d))
                    .ok_or_else(|| LowerError::ParseType(format!("vector<{}xf32>", d)))?;
                let op_name = match op {
                    BinaryOp::Add => "arith.addf",
                    BinaryOp::Sub => "arith.subf",
                    BinaryOp::Mul => "arith.mulf",
                    BinaryOp::Div => "arith.divf",
                    BinaryOp::MatMul => unreachable!(),
                };
                let arith_op = OperationBuilder::new(op_name, gen.loc())
                    .add_operands(&[va, vb])
                    .add_results(&[vec_ty])
                    .build()?;
                let res: Value = block.append_operation(arith_op).result(0)?.into();
                return Ok((res, vec_ty, block));
            }
        }

        let mut final_ty = lhs_ty;
        let _lhs_ty_str = lhs_ty.to_string();
        let _rhs_ty_str = rhs_ty.to_string();

        if lhs_ty != rhs_ty && op != &BinaryOp::MatMul {
            // Priority coercion: f64 > f32 > i64 > i32
            // To simplify, we'll cast rhs to lhs for now.
            rhs_val = gen.coerce_type(&block, rhs_val, rhs_ty, lhs_ty)?;
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
            let out_ty = Type::parse(gen.context, &out_ty_str).ok_or_else(|| {
                crate::codegen::lower::LowerError::ParseType("Type::parse failed".to_string())
            })?;

            // Determine dynamic dimensions for alloc
            let mut alloc_operands = Vec::new();
            let index_ty = gen.index_ty;

            if m_str == "?" {
                let m_idx_attr = IntegerAttribute::new(Type::index(gen.context), 0).into();
                let cst_op = OperationBuilder::new("arith.constant", gen.loc())
                    .add_results(&[index_ty])
                    .add_attributes(&[(Identifier::new(gen.context, "value"), m_idx_attr)])
                    .build()?;
                let idx_val = block.append_operation(cst_op).result(0)?.into();

                let dim_m_op = OperationBuilder::new("memref.dim", gen.loc())
                    .add_operands(&[lhs_val, idx_val])
                    .add_results(&[index_ty])
                    .build()?;
                alloc_operands.push(block.append_operation(dim_m_op).result(0)?.into());
            }

            if n_str == "?" {
                let n_idx_attr = IntegerAttribute::new(Type::index(gen.context), 1).into();
                let cst_op = OperationBuilder::new("arith.constant", gen.loc())
                    .add_results(&[index_ty])
                    .add_attributes(&[(Identifier::new(gen.context, "value"), n_idx_attr)])
                    .build()?;
                let idx_val = block.append_operation(cst_op).result(0)?.into();

                let dim_n_op = OperationBuilder::new("memref.dim", gen.loc())
                    .add_operands(&[rhs_val, idx_val])
                    .add_results(&[index_ty])
                    .build()?;
                alloc_operands.push(block.append_operation(dim_n_op).result(0)?.into());
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
                .build()?;
            let out_val = block.append_operation(alloc_op).result(0)?.into();

            // Zero initialize the output buffer since matmul accumulates!
            let zero_attr = if el_ty_str.starts_with('i') {
                IntegerAttribute::new(
                    Type::parse(gen.context, el_ty_str).ok_or_else(|| {
                        crate::codegen::lower::LowerError::ParseType(
                            "Type::parse failed".to_string(),
                        )
                    })?,
                    0,
                )
                .into()
            } else {
                FloatAttribute::new(
                    gen.context,
                    Type::parse(gen.context, el_ty_str).ok_or_else(|| {
                        crate::codegen::lower::LowerError::ParseType(
                            "Type::parse failed".to_string(),
                        )
                    })?,
                    0.0,
                )
                .into()
            };

            let zero_op = OperationBuilder::new("arith.constant", gen.loc())
                .add_results(&[Type::parse(gen.context, el_ty_str).ok_or_else(|| {
                    crate::codegen::lower::LowerError::ParseType("Type::parse failed".to_string())
                })?])
                .add_attributes(&[(Identifier::new(gen.context, "value"), zero_attr)])
                .build()?;
            let zero_val = block.append_operation(zero_op).result(0)?.into();

            let region_fill = Region::new();
            let block_fill = melior::ir::Block::new(&[
                (
                    Type::parse(gen.context, el_ty_str).ok_or_else(|| {
                        crate::codegen::lower::LowerError::ParseType(
                            "Type::parse failed".to_string(),
                        )
                    })?,
                    gen.loc(),
                ),
                (
                    Type::parse(gen.context, el_ty_str).ok_or_else(|| {
                        crate::codegen::lower::LowerError::ParseType(
                            "Type::parse failed".to_string(),
                        )
                    })?,
                    gen.loc(),
                ),
            ]);
            let yield_fill = OperationBuilder::new("linalg.yield", gen.loc())
                .add_operands(&[block_fill
                    .argument(0)
                    .map_err(|_| {
                        crate::codegen::lower::LowerError::from(format!(
                            "Missing argument {} block",
                            0
                        ))
                    })?
                    .into()])
                .build()?;
            block_fill.append_operation(yield_fill);
            region_fill.append_block(block_fill);

            let linalg_fill = OperationBuilder::new("linalg.fill", gen.loc())
                .add_operands(&[zero_val, out_val])
                .add_attributes(&[(
                    Identifier::new(gen.context, "operandSegmentSizes"),
                    DenseI32ArrayAttribute::new(gen.context, &[1, 1]).into(),
                )])
                .add_regions([region_fill])
                .build()?;
            block.append_operation(linalg_fill);

            // Execute linalg.matmul
            let region_matmul = Region::new();
            let block_matmul = melior::ir::Block::new(&[
                (
                    Type::parse(gen.context, el_ty_str).ok_or_else(|| {
                        crate::codegen::lower::LowerError::ParseType(
                            "Type::parse failed".to_string(),
                        )
                    })?,
                    gen.loc(),
                ),
                (
                    Type::parse(gen.context, el_ty_str).ok_or_else(|| {
                        crate::codegen::lower::LowerError::ParseType(
                            "Type::parse failed".to_string(),
                        )
                    })?,
                    gen.loc(),
                ),
                (
                    Type::parse(gen.context, el_ty_str).ok_or_else(|| {
                        crate::codegen::lower::LowerError::ParseType(
                            "Type::parse failed".to_string(),
                        )
                    })?,
                    gen.loc(),
                ),
            ]);

            let is_float = el_ty_str.contains("f32")
                || el_ty_str.contains("f64")
                || el_ty_str.contains("f16")
                || el_ty_str.contains("bf16");
            let mul_op_name = if is_float { "arith.mulf" } else { "arith.muli" };
            let add_op_name = if is_float { "arith.addf" } else { "arith.addi" };

            let mul_op = OperationBuilder::new(mul_op_name, gen.loc())
                .add_operands(&[
                    block_matmul
                        .argument(0)
                        .map_err(|_| {
                            crate::codegen::lower::LowerError::from(format!(
                                "Missing argument {} block",
                                0
                            ))
                        })?
                        .into(),
                    block_matmul
                        .argument(1)
                        .map_err(|_| {
                            crate::codegen::lower::LowerError::from(format!(
                                "Missing argument {} block",
                                1
                            ))
                        })?
                        .into(),
                ])
                .add_results(&[Type::parse(gen.context, el_ty_str).ok_or_else(|| {
                    crate::codegen::lower::LowerError::ParseType("Type::parse failed".to_string())
                })?])
                .build()?;
            let mul_val = block_matmul.append_operation(mul_op).result(0)?.into();

            let add_op = OperationBuilder::new(add_op_name, gen.loc())
                .add_operands(&[
                    block_matmul
                        .argument(2)
                        .map_err(|_| {
                            crate::codegen::lower::LowerError::from(format!(
                                "Missing argument {} block",
                                2
                            ))
                        })?
                        .into(),
                    mul_val,
                ])
                .add_results(&[Type::parse(gen.context, el_ty_str).ok_or_else(|| {
                    crate::codegen::lower::LowerError::ParseType("Type::parse failed".to_string())
                })?])
                .build()?;
            let add_val = block_matmul.append_operation(add_op).result(0)?.into();

            let yield_matmul = OperationBuilder::new("linalg.yield", gen.loc())
                .add_operands(&[add_val])
                .build()?;
            block_matmul.append_operation(yield_matmul);
            region_matmul.append_block(block_matmul);

            let matmul_op = OperationBuilder::new("linalg.matmul", gen.loc())
                .add_operands(&[lhs_val, rhs_val, out_val])
                .add_attributes(&[(
                    Identifier::new(gen.context, "operandSegmentSizes"),
                    DenseI32ArrayAttribute::new(gen.context, &[2, 1]).into(),
                )])
                .add_regions([region_matmul])
                .build()?;
            block.append_operation(matmul_op);

            return Ok((out_val, out_ty, block));
        } else if is_memref {
            // Element-wise Linalg Lowering (Add, Sub, Mul, Div)
            let out_ty = Type::parse(gen.context, &lhs_ty_str).ok_or_else(|| {
                crate::codegen::lower::LowerError::ParseType("Type::parse failed".to_string())
            })?;

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
                        .build()?;
                    let idx_val = block.append_operation(cst_op).result(0)?.into();

                    let dim_op = OperationBuilder::new("memref.dim", gen.loc())
                        .add_operands(&[lhs_val, idx_val])
                        .add_results(&[index_ty])
                        .build()?;
                    alloc_operands.push(block.append_operation(dim_op).result(0)?.into());
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
                .build()?;
            let out_val = block.append_operation(alloc_op).result(0)?.into();

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
                    Type::parse(gen.context, el_ty_str).ok_or_else(|| {
                        crate::codegen::lower::LowerError::ParseType(
                            "Type::parse failed".to_string(),
                        )
                    })?,
                    gen.loc(),
                ),
                (
                    Type::parse(gen.context, el_ty_str).ok_or_else(|| {
                        crate::codegen::lower::LowerError::ParseType(
                            "Type::parse failed".to_string(),
                        )
                    })?,
                    gen.loc(),
                ),
                (
                    Type::parse(gen.context, el_ty_str).ok_or_else(|| {
                        crate::codegen::lower::LowerError::ParseType(
                            "Type::parse failed".to_string(),
                        )
                    })?,
                    gen.loc(),
                ),
            ]);

            let arith_op = OperationBuilder::new(arith_op_name, gen.loc())
                .add_operands(&[
                    block_inner
                        .argument(0)
                        .map_err(|_| {
                            crate::codegen::lower::LowerError::from(format!(
                                "Missing argument {} block",
                                0
                            ))
                        })?
                        .into(),
                    block_inner
                        .argument(1)
                        .map_err(|_| {
                            crate::codegen::lower::LowerError::from(format!(
                                "Missing argument {} block",
                                1
                            ))
                        })?
                        .into(),
                ])
                .add_results(&[Type::parse(gen.context, el_ty_str).ok_or_else(|| {
                    crate::codegen::lower::LowerError::ParseType("Type::parse failed".to_string())
                })?])
                .build()?;

            let arith_val = block_inner.append_operation(arith_op).result(0)?.into();

            let yield_op = OperationBuilder::new("linalg.yield", gen.loc())
                .add_operands(&[arith_val])
                .build()?;

            block_inner.append_operation(yield_op);
            region.append_block(block_inner);

            let linalg_op = OperationBuilder::new(op_name, gen.loc())
                .add_operands(&[lhs_val, rhs_val, out_val])
                .add_attributes(&[(
                    Identifier::new(gen.context, "operandSegmentSizes"),
                    DenseI32ArrayAttribute::new(gen.context, &[2, 1]).into(),
                )])
                .add_regions([region])
                .build()?;
            block.append_operation(linalg_op);

            return Ok((out_val, out_ty, block));
        }

        if lhs_ty != rhs_ty
            && ((lhs_ty_str == "index" && rhs_ty_str == "i32")
                || (lhs_ty_str == "i32" && rhs_ty_str == "index"))
        {
            if lhs_ty_str == "index" && rhs_ty_str == "i32" {
                let cast_op = OperationBuilder::new("arith.index_cast", gen.loc())
                    .add_operands(&[rhs_val])
                    .add_results(&[lhs_ty])
                    .build()?;
                rhs_val = block.append_operation(cast_op).result(0)?.into();
            } else {
                let cast_op = OperationBuilder::new("arith.index_cast", gen.loc())
                    .add_operands(&[lhs_val])
                    .add_results(&[rhs_ty])
                    .build()?;
                lhs_val = block.append_operation(cast_op).result(0)?.into();
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

        let bin_op = builder.build()?;
        let bin_ref = block.append_operation(bin_op);
        Ok((bin_ref.result(0)?.into(), ret_ty, block))
    }
}

impl<'c> LowerToMelior<'c> for RelationalOpExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let RelationalOpExpr {
            lhs,
            op,
            rhs,
            span: _,
        } = self;
        let (lhs_val, lhs_ty, block) = gen.generate_expr(lhs, block)?;
        let prev_expected = gen.expected_type;
        gen.expected_type = Some(lhs_ty);
        let (mut rhs_val, rhs_ty, block) = gen.generate_expr(rhs, block)?;
        gen.expected_type = prev_expected;

        let _lhs_ty_str = lhs_ty.to_string();
        let _rhs_ty_str = rhs_ty.to_string();

        let mut final_ty = lhs_ty;

        if lhs_ty != rhs_ty {
            // Prioritize standard coercion depending on which type is more generic (e.g. f64 > f32 > i64 > i32)
            // For simplicity, just cast rhs to lhs for now.
            rhs_val = gen.coerce_type(&block, rhs_val, rhs_ty, lhs_ty)?;
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

        let bin_op = builder.build()?;
        let bin_ref = block.append_operation(bin_op);
        Ok((bin_ref.result(0)?.into(), ret_ty, block))
    }
}

impl<'c> LowerToMelior<'c> for LogicalOpExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let LogicalOpExpr {
            lhs,
            op,
            rhs,
            span: _,
        } = self;
        let (lhs_val, _lhs_ty, block) = gen.generate_expr(lhs, block)?;
        let (rhs_val, _rhs_ty, block) = gen.generate_expr(rhs, block)?;

        let final_ty = gen.i1_ty;

        let builder = OperationBuilder::new(op.get_op_name(false), gen.loc())
            .add_operands(&[lhs_val, rhs_val])
            .add_results(&[final_ty]);

        let bin_op = builder.build()?;
        let bin_ref = block.append_operation(bin_op);
        Ok((bin_ref.result(0)?.into(), final_ty, block))
    }
}

impl<'c> LowerToMelior<'c> for syntax::UnaryOpExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let syntax::UnaryOpExpr { op, expr, span: _ } = self;
        let (val, ty, block) = gen.generate_expr(expr, block)?;
        match op {
            syntax::UnaryOp::Not => {
                let true_val_op = OperationBuilder::new("arith.constant", gen.loc())
                    .add_results(&[ty])
                    .add_attributes(&[(
                        Identifier::new(gen.context, "value"),
                        IntegerAttribute::new(ty, 1).into(),
                    )])
                    .build()?;
                let true_val_ref = block.append_operation(true_val_op);

                let not_op = OperationBuilder::new("arith.xori", gen.loc())
                    .add_operands(&[val, true_val_ref.result(0)?.into()])
                    .add_results(&[ty])
                    .build()?;

                let not_ref = block.append_operation(not_op);
                Ok((not_ref.result(0)?.into(), ty, block))
            }
            syntax::UnaryOp::Neg => {
                let is_float = ty.to_string().contains("f32")
                    || ty.to_string().contains("f64")
                    || ty.to_string().contains("f16")
                    || ty.to_string().contains("bf16");
                if is_float {
                    let neg_op = OperationBuilder::new("arith.negf", gen.loc())
                        .add_operands(&[val])
                        .add_results(&[ty])
                        .build()?;
                    let neg_ref = block.append_operation(neg_op);
                    Ok((neg_ref.result(0)?.into(), ty, block))
                } else {
                    let zero_op = OperationBuilder::new("arith.constant", gen.loc())
                        .add_results(&[ty])
                        .add_attributes(&[(
                            Identifier::new(gen.context, "value"),
                            IntegerAttribute::new(ty, 0).into(),
                        )])
                        .build()?;
                    let zero_ref = block.append_operation(zero_op);

                    let neg_op = OperationBuilder::new("arith.subi", gen.loc())
                        .add_operands(&[zero_ref.result(0)?.into(), val])
                        .add_results(&[ty])
                        .build()?;
                    let neg_ref = block.append_operation(neg_op);
                    Ok((neg_ref.result(0)?.into(), ty, block))
                }
            }
        }
    }
}

impl<'c> LowerToMelior<'c> for StructInitExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let StructInitExpr {
            name,
            fields,
            type_id: _,
            span: _,
        } = self;
        let base_name = name.split('<').next().unwrap_or(name).to_string();
        let struct_decl = gen
            .structs
            .get(base_name.as_str())
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
                    syntax::Type::Scalar(syntax::ElementType::I32)
                } else if ty_arg == "f32" {
                    syntax::Type::Scalar(syntax::ElementType::F32)
                } else if ty_arg == "i64" {
                    syntax::Type::Scalar(syntax::ElementType::I64)
                } else if ty_arg.chars().all(|c| c.is_ascii_digit()) {
                    syntax::Type::Const(Box::new(syntax::Expr::Number(
                        syntax::expr::NumberExpr::new(
                            ty_arg.to_string(),
                            None,
                            syntax::Span::default(),
                        ),
                    )))
                } else {
                    syntax::Type::Struct(ty_arg.to_string().into(), None)
                };
                inner_tys.push(inner_ty);
            }
            for (i, param) in struct_decl.generics.iter().enumerate() {
                if i < inner_tys.len() {
                    mapping.insert(param.name().into(), inner_tys[i].clone());
                }
            }

            gen.lower_type(&syntax::Type::GenericInstance(
                Box::new(syntax::Type::Struct(base_name.clone().into(), None)),
                inner_tys,
            ))?
        } else {
            gen.lower_type(&syntax::Type::Struct(name.clone(), None))?
        };

        let undef_op = OperationBuilder::new("llvm.mlir.undef", gen.loc())
            .add_results(&[struct_ty])
            .build()?;
        let mut current_struct = block.append_operation(undef_op).result(0)?.into();

        for (field_name, f_expr) in fields {
            let field_idx = struct_decl
                .fields
                .iter()
                .position(|(n, _)| n == field_name)
                .unwrap();
            let sub_ty = struct_decl.fields[field_idx].1.substitute(&mapping);

            let field_ty = gen.lower_type(&sub_ty)?;
            let prev_expected = gen.expected_type;
            gen.expected_type = Some(field_ty);
            let (mut field_val, expr_ty, block) = gen.generate_expr(f_expr, block)?;
            gen.expected_type = prev_expected;

            if expr_ty != field_ty
                && ((expr_ty.to_string() == "index" && field_ty.to_string() == "i32")
                    || (expr_ty.to_string() == "i32" && field_ty.to_string() == "index"))
            {
                let cast_op = OperationBuilder::new("arith.index_cast", gen.loc())
                    .add_operands(&[field_val])
                    .add_results(&[field_ty])
                    .build()?;
                field_val = block.append_operation(cast_op).result(0)?.into();
            }

            let pos_attr = melior::ir::attribute::DenseI64ArrayAttribute::new(
                gen.context,
                &[field_idx as i64],
            );

            let insert_op = OperationBuilder::new("llvm.insertvalue", gen.loc())
                .add_operands(&[current_struct, field_val])
                .add_attributes(&[(Identifier::new(gen.context, "position"), pos_attr.into())])
                .add_results(&[struct_ty])
                .build()?;
            current_struct = block.append_operation(insert_op).result(0)?.into();
        }
        Ok((current_struct, struct_ty, block))
    }
}

impl<'c> LowerToMelior<'c> for UnsafeBlockExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let mut current_block = block;
        for stmt in &self.stmts {
            if let Some(b) = gen.generate_statement(stmt, current_block)? {
                current_block = b;
            } else {
                // Block was terminated (e.g., by break/continue/return)
                break;
            }
        }
        if let Some(ret_expr) = &self.ret {
            gen.generate_expr(ret_expr, current_block)
        } else {
            // Return an i32 0 or something empty if no return type is expected.
            let i32_ty = gen.i32_ty;
            let zero_attr = IntegerAttribute::new(i32_ty, 0).into();
            let zero_op = OperationBuilder::new("arith.constant", gen.loc())
                .add_results(&[i32_ty])
                .add_attributes(&[(Identifier::new(gen.context, "value"), zero_attr)])
                .build()?;
            let zero_val = current_block.append_operation(zero_op).result(0)?.into();
            Ok((zero_val, i32_ty, current_block))
        }
    }
}

impl<'c> LowerToMelior<'c> for MemberAccessExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let MemberAccessExpr {
            base,
            member,
            struct_name,
            span: _,
        } = self;
        let (base_val, base_ty, block) = gen.generate_expr(base, block)?;
        let base_ty_str = base_ty.to_string();

        let mut struct_name_opt = struct_name.clone();
        if struct_name_opt.is_none() {
            if let Some(start_idx) = base_ty_str.find('\"') {
                if let Some(end_idx) = base_ty_str[start_idx + 1..].find('"') {
                    struct_name_opt = Some(
                        base_ty_str[start_idx + 1..start_idx + 1 + end_idx]
                            .to_string()
                            .into(),
                    );
                }
            }
        }

        let is_ptr = base_ty_str.starts_with("!llvm.ptr");

        if let Some(ref resolved_struct_name) = struct_name_opt {
            let base_name = resolved_struct_name
                .split('<')
                .next()
                .unwrap_or(resolved_struct_name.as_ref())
                .to_string();
            let mut mapping = std::collections::HashMap::new();
            if resolved_struct_name.contains('<') && resolved_struct_name.ends_with('>') {
                let inner_ty_str = &resolved_struct_name
                    [resolved_struct_name.find('<').unwrap() + 1..resolved_struct_name.len() - 1];
                let mut inner_tys = Vec::new();
                for ty_arg_raw in inner_ty_str.split(',') {
                    let ty_arg = ty_arg_raw.trim();
                    let inner_ty = if ty_arg == "i32" {
                        syntax::Type::Scalar(syntax::ElementType::I32)
                    } else if ty_arg == "f32" {
                        syntax::Type::Scalar(syntax::ElementType::F32)
                    } else if ty_arg == "i64" {
                        syntax::Type::Scalar(syntax::ElementType::I64)
                    } else if ty_arg.chars().all(|c| c.is_ascii_digit()) {
                        syntax::Type::Const(Box::new(syntax::Expr::Number(
                            syntax::expr::NumberExpr::new(
                                ty_arg.to_string(),
                                None,
                                syntax::Span::default(),
                            ),
                        )))
                    } else {
                        syntax::Type::Struct(ty_arg.to_string().into(), None)
                    };
                    inner_tys.push(inner_ty);
                }
                if let Some(struct_decl) = gen.structs.get(base_name.as_str()) {
                    for (i, param) in struct_decl.generics.iter().enumerate() {
                        if i < inner_tys.len() {
                            mapping.insert(param.name().into(), inner_tys[i].clone());
                        }
                    }
                }
            }

            if let Some(struct_decl) = gen.structs.get(base_name.as_str()).cloned() {
                if let Some(field_idx) = struct_decl.fields.iter().position(|(n, _)| n == member) {
                    let sub_ty = struct_decl.fields[field_idx].1.substitute(&mapping);
                    let field_ty = gen.lower_type(&sub_ty)?;

                    if is_ptr {
                        let ptr_ty = gen.ptr_ty;
                        let mut field_types = Vec::new();
                        for (_, ty) in &struct_decl.fields {
                            let sub_ty2 = ty.substitute(&mapping);
                            let mut lowered = gen.lower_type_str(&sub_ty2)?;
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
                        let struct_llvm_ty = Type::parse(gen.context, &struct_llvm_ty_str)
                            .ok_or_else(|| {
                                crate::codegen::lower::LowerError::ParseType(
                                    "Type::parse failed".to_string(),
                                )
                            })?;

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
                        let field_ptr = gep_ref.result(0)?.into();

                        if gen.is_lvalue_context {
                            return Ok((field_ptr, ptr_ty, block));
                        }

                        let load_op = OperationBuilder::new("llvm.load", gen.loc())
                            .add_operands(&[field_ptr])
                            .add_results(&[field_ty])
                            .build()
                            .unwrap();
                        let load_ref = block.append_operation(load_op);
                        return Ok((load_ref.result(0)?.into(), field_ty, block));
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
                        return Ok((ext_ref.result(0)?.into(), field_ty, block));
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

/// Lower a slice reduction (`dot`/`sum`/`max`/`min`, S2) to the `vector` dialect.
///
/// Each rank-1 slice operand (a strided `memref<Dxf32, ...>` from S1, or a contiguous 1-D
/// tensor) is `vector.load`ed into a `vector<Dxf32>`; `dot` first `arith.mulf`s the two
/// vectors, then all reduce via `vector.reduction` (`add` for dot/sum, `maximumf`/`minimumf`
/// for max/min). The pipeline's convert-vector-to-llvm turns these into `@llvm.vector.reduce.*`.
fn lower_slice_reduction<'c>(
    gen: &mut MeliorGenerator<'c>,
    op: &str,
    args: &[Expr],
    block: melior::ir::BlockRef<'c, 'c>,
) -> Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError> {
    let index_ty = Type::index(gen.context);
    let f32_ty =
        Type::parse(gen.context, "f32").ok_or_else(|| LowerError::ParseType("f32".to_string()))?;

    // A constant `0 : index` to address the first (only) memref dimension for vector.load.
    let c0_op = OperationBuilder::new("arith.constant", gen.loc())
        .add_attributes(&[(
            Identifier::new(gen.context, "value"),
            IntegerAttribute::new(index_ty, 0).into(),
        )])
        .add_results(&[index_ty])
        .build()?;
    let c0: Value = block.append_operation(c0_op).result(0)?.into();

    // Load each slice operand into a vector<Dxf32>. `len` is the static row length D,
    // parsed from the operand's memref type (`memref<Dx...`), shared across all operands.
    let mut vecs: Vec<Value> = Vec::new();
    let mut len: Option<i64> = None;
    let mut cur = block;
    for arg in args {
        let (base_val, base_ty, nb) = gen.generate_expr(arg, cur)?;
        cur = nb;
        let ty_str = base_ty.to_string();
        let d: i64 = ty_str
            .strip_prefix("memref<")
            .and_then(|s| s.split('x').next())
            .and_then(|s| s.parse::<i64>().ok())
            .ok_or_else(|| LowerError::ParseType(format!("slice length from {}", ty_str)))?;
        len = Some(d);
        let vec_ty_str = format!("vector<{}xf32>", d);
        let vec_ty = Type::parse(gen.context, &vec_ty_str)
            .ok_or_else(|| LowerError::ParseType(vec_ty_str.clone()))?;
        let load_op = OperationBuilder::new("vector.load", gen.loc())
            .add_operands(&[base_val, c0])
            .add_results(&[vec_ty])
            .build()?;
        vecs.push(cur.append_operation(load_op).result(0)?.into());
    }

    let d = len.ok_or_else(|| LowerError::ParseType("empty slice reduction".to_string()))?;
    let vec_ty = Type::parse(gen.context, &format!("vector<{}xf32>", d))
        .ok_or_else(|| LowerError::ParseType(format!("vector<{}xf32>", d)))?;

    // For `dot`, fuse the two operands with an elementwise multiply before reducing.
    let (reduce_in, kind) = if op == "dot" {
        let mul_op = OperationBuilder::new("arith.mulf", gen.loc())
            .add_operands(&[vecs[0], vecs[1]])
            .add_results(&[vec_ty])
            .build()?;
        let prod: Value = cur.append_operation(mul_op).result(0)?.into();
        (prod, "add")
    } else {
        let kind = match op {
            "sum" => "add",
            "max" => "maximumf",
            "min" => "minimumf",
            _ => unreachable!("slice reduction op {}", op),
        };
        (vecs[0], kind)
    };

    let kind_attr = Attribute::parse(gen.context, &format!("#vector.kind<{}>", kind))
        .ok_or_else(|| LowerError::ParseType(format!("#vector.kind<{}>", kind)))?;
    let red_op = OperationBuilder::new("vector.reduction", gen.loc())
        .add_operands(&[reduce_in])
        .add_attributes(&[(Identifier::new(gen.context, "kind"), kind_attr)])
        .add_results(&[f32_ty])
        .build()?;
    let result: Value = cur.append_operation(red_op).result(0)?.into();
    Ok((result, f32_ty, cur))
}

/// Lower a tensor initializer list (`Tensor<T>([[..],[..]])`): allocate a buffer of the shape
/// inferred from the nesting and store each constant element in row-major order. The buffer keeps
/// the dynamic `memref<?x..xT>` type (with constant sizes) that the rest of codegen expects.
fn lower_tensor_initializer<'c>(
    gen: &mut MeliorGenerator<'c>,
    elem_ty_str: &str,
    shape: &[usize],
    arr: &ArrayExpr,
    block: melior::ir::BlockRef<'c, 'c>,
) -> Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError> {
    let index_ty = Type::index(gen.context);
    let elem_ty = Type::parse(gen.context, elem_ty_str)
        .ok_or_else(|| LowerError::ParseType(elem_ty_str.to_string()))?;
    let rank = shape.len();

    let a_const = |gen: &mut MeliorGenerator<'c>, b: melior::ir::BlockRef<'c, 'c>, n: usize| {
        let op = OperationBuilder::new("arith.constant", gen.loc())
            .add_attributes(&[(
                Identifier::new(gen.context, "value"),
                IntegerAttribute::new(index_ty, n as i64).into(),
            )])
            .add_results(&[index_ty])
            .build()?;
        Ok::<Value, LowerError>(b.append_operation(op).result(0)?.into())
    };

    // A dynamic `memref<?x..xT>` sized by constants -- matches how plain `Tensor<T>([n, m])`
    // buffers are typed, so transfers / slices downstream are unaffected.
    let ty_str = format!("memref<{}{}>", "?x".repeat(rank), elem_ty_str);
    let tensor_ty =
        Type::parse(gen.context, &ty_str).ok_or_else(|| LowerError::ParseType(ty_str.clone()))?;
    let mut size_vals = Vec::with_capacity(rank);
    for &d in shape {
        size_vals.push(a_const(gen, block, d)?);
    }
    let alloc_op = OperationBuilder::new("memref.alloc", gen.loc())
        .add_operands(&size_vals)
        .add_attributes(&[(
            Identifier::new(gen.context, "operandSegmentSizes"),
            DenseI32ArrayAttribute::new(gen.context, &[size_vals.len() as i32, 0]).into(),
        )])
        .add_results(&[tensor_ty])
        .build()?;
    let buf: Value = block.append_operation(alloc_op).result(0)?.into();

    let values = arr.initializer_values();
    let mut cur = block;
    for (flat, val_expr) in values.iter().enumerate() {
        let (v, vty, nb) = gen.generate_expr(val_expr, cur)?;
        cur = nb;
        let v = gen.coerce_type(&cur, v, vty, elem_ty)?;
        // Row-major flat index -> per-dimension coordinates.
        let mut rem = flat;
        let mut coords = vec![0usize; rank];
        for dim in (0..rank).rev() {
            coords[dim] = rem % shape[dim];
            rem /= shape[dim];
        }
        let mut store = OperationBuilder::new("memref.store", gen.loc()).add_operands(&[v, buf]);
        let idx_vals: Vec<Value> = coords
            .iter()
            .map(|&c| a_const(gen, cur, c))
            .collect::<Result<_, _>>()?;
        for iv in &idx_vals {
            store = store.add_operands(std::slice::from_ref(iv));
        }
        cur.append_operation(store.build()?);
    }
    Ok((buf, tensor_ty, cur))
}

/// Static length D of a rank-1 f32 slice value, from its melior type string. Matches a
/// `vector<Dxf32>` or a rank-1 `memref<Dxf32, ...>` (the S1 row view); returns `None` for
/// scalars, dynamic dims, and higher-rank memrefs (which are not slice operands).
fn slice_vec_len(ty_str: &str) -> Option<i64> {
    let inner = ty_str
        .strip_prefix("vector<")
        .or_else(|| ty_str.strip_prefix("memref<"))?;
    // Shape is the text before any `, strided<...>` layout, e.g. `4xf32`; require exactly one
    // `x` (rank 1) and an f32 element (the slice ops emit vector<Dxf32>).
    let shape = inner.split(',').next()?.trim_end_matches('>');
    let parts: Vec<&str> = shape.split('x').collect();
    if parts.len() == 2 && parts[1] == "f32" {
        return parts[0].parse().ok();
    }
    None
}

/// Whether an operand should drive the slice-elementwise (S3) vector path: a `vector<...>`
/// value or a strided slice view (`memref<..., strided<...>>`, as produced by S1). A plain
/// contiguous `memref<Nxf32>` (a whole 1-D tensor) is deliberately excluded -- those keep the
/// loop-based lowering that the optimization-pass tests exercise (scf-to-cf, loop unroll).
fn is_slice_operand(ty_str: &str) -> bool {
    ty_str.starts_with("vector<") || (ty_str.starts_with("memref<") && ty_str.contains("strided"))
}

/// Coerce a slice-elementwise operand to `vector<Dxf32>`: a vector passes through, a rank-1
/// memref is `vector.load`ed, and a scalar is `vector.broadcast`ed to the slice width.
fn to_vector<'c>(
    gen: &mut MeliorGenerator<'c>,
    val: Value<'c, 'c>,
    ty_str: &str,
    d: i64,
    block: melior::ir::BlockRef<'c, 'c>,
) -> Result<Value<'c, 'c>, LowerError> {
    let vec_ty = Type::parse(gen.context, &format!("vector<{}xf32>", d))
        .ok_or_else(|| LowerError::ParseType(format!("vector<{}xf32>", d)))?;
    if ty_str.starts_with("vector<") {
        return Ok(val);
    }
    if ty_str.starts_with("memref<") {
        let index_ty = Type::index(gen.context);
        let c0_op = OperationBuilder::new("arith.constant", gen.loc())
            .add_attributes(&[(
                Identifier::new(gen.context, "value"),
                IntegerAttribute::new(index_ty, 0).into(),
            )])
            .add_results(&[index_ty])
            .build()?;
        let c0: Value = block.append_operation(c0_op).result(0)?.into();
        let load_op = OperationBuilder::new("vector.load", gen.loc())
            .add_operands(&[val, c0])
            .add_results(&[vec_ty])
            .build()?;
        return Ok(block.append_operation(load_op).result(0)?.into());
    }
    // Scalar: broadcast to the slice width.
    let bcast_op = OperationBuilder::new("vector.broadcast", gen.loc())
        .add_operands(&[val])
        .add_results(&[vec_ty])
        .build()?;
    Ok(block.append_operation(bcast_op).result(0)?.into())
}

impl<'c> LowerToMelior<'c> for FunctionCallExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let FunctionCallExpr {
            name,
            args,
            type_args,
            span: _,
        } = self;
        if name.as_ref() == "Verified" {
            return gen.generate_expr(&args[0], block);
        }
        // Slice reductions (S2): dot/sum/max/min over rank-1 f32 slices lower to
        // `vector.load` + (`arith.mulf` for dot) + `vector.reduction`, which the pipeline's
        // convert-vector-to-llvm turns into real SIMD (`@llvm.vector.reduce.*`). See
        // docs/discussions/implementation_plans/slice_operators.md.
        if matches!(name.as_ref(), "dot" | "sum" | "max" | "min") {
            return lower_slice_reduction(gen, name.as_ref(), args, block);
        }
        if name.as_ref() == "Tensor" {
            let mlir_ty_str = if let Some(tys) = type_args {
                if !tys.is_empty() {
                    gen.lower_type_str(&tys[0])?
                } else {
                    panic!("Tensor initialization requires an explicit generic type argument");
                }
            } else {
                panic!("Tensor initialization requires an explicit generic type argument");
            };
            // Initializer list: `Tensor<T>([[..],[..]])` allocates a shaped buffer and stores the
            // constant values in place, so no fill loop is needed.
            if let Some(Expr::Array(arr)) = args.first() {
                if let Some(shape) = arr.initializer_shape() {
                    return lower_tensor_initializer(gen, &mlir_ty_str, &shape, arr, block);
                }
            }
            let mut dynamic_sizes = Vec::new();
            // Number of dynamic dimensions; the type below uses one `?` per dim,
            // so this must equal the number of size operands collected into
            // `dynamic_sizes`. With no shape args (e.g. `Tensor<f32>()`) it stays
            // 0, yielding a valid rank-0 `memref<f32>` rather than a `?x?` memref
            // with no sizes (invalid IR). See GitHub #147.
            let mut dims_count = 0;

            let mut current_b = block;
            if args.len() == 1 {
                if let Expr::Array(arr) = &args[0] {
                    dims_count = arr.elements.len();
                    for el in &arr.elements {
                        let (mut val, ty, new_b) = gen.generate_expr(el, current_b)?;
                        current_b = new_b;
                        if ty.to_string() != "index" {
                            let cast_op = OperationBuilder::new("arith.index_cast", gen.loc())
                                .add_operands(&[val])
                                .add_results(&[Type::index(gen.context)])
                                .build()
                                .unwrap();
                            val = current_b.append_operation(cast_op).result(0)?.into();
                        }
                        dynamic_sizes.push(val);
                    }
                }
            } else if !args.is_empty() {
                dims_count = args.len();
                for el in args {
                    let (mut val, ty, new_b) = gen.generate_expr(el, current_b)?;
                    current_b = new_b;
                    if ty.to_string() != "index" {
                        let cast_op = OperationBuilder::new("arith.index_cast", gen.loc())
                            .add_operands(&[val])
                            .add_results(&[Type::index(gen.context)])
                            .build()
                            .unwrap();
                        val = current_b.append_operation(cast_op).result(0)?.into();
                    }
                    dynamic_sizes.push(val);
                }
            }

            let mut shape_str = String::new();
            for _ in 0..dims_count {
                shape_str.push_str("?x");
            }
            let tensor_ty_str = format!("memref<{}{}>", shape_str, mlir_ty_str);
            let tensor_ty = Type::parse(gen.context, &tensor_ty_str).ok_or_else(|| {
                crate::codegen::lower::LowerError::ParseType("Type::parse failed".to_string())
            })?;

            let alloc_op = OperationBuilder::new("memref.alloc", gen.loc())
                .add_operands(&dynamic_sizes)
                .add_attributes(&[(
                    Identifier::new(gen.context, "operandSegmentSizes"),
                    DenseI32ArrayAttribute::new(gen.context, &[dynamic_sizes.len() as i32, 0])
                        .into(),
                )])
                .add_results(&[tensor_ty])
                .build()?;
            let alloc_ref = block.append_operation(alloc_op);
            return Ok((alloc_ref.result(0)?.into(), tensor_ty, block));
        }

        if name.as_ref() == "reshape" || **name == *"transpose" {
            // `memref.cast` cannot reshape or permute data (it only changes
            // static/dynamic/ranked info), so model these as buffer views:
            //   - reshape: reinterpret the contiguous source as the target shape
            //     (a row-major view; also handles a smaller "trim" target).
            //   - transpose: a permuted-stride view, materialized into a fresh
            //     contiguous buffer so downstream ops see row-major data.
            // See GitHub #148.
            let is_transpose = **name == *"transpose";
            let (arg_val, expr_ty, block) = gen.generate_expr(&args[0], block)?;
            let expr_ty_str = expr_ty.to_string();
            let el_ty_str =
                extract_mlir_element_type(&expr_ty_str).unwrap_or_else(|e| panic!("{}", e));

            // Static dimensions of the source, parsed from "memref<AxBx...xT>".
            let src_dims: Vec<i64> = expr_ty_str
                .trim_start_matches("memref<")
                .split('x')
                .map_while(|t| t.parse::<i64>().ok())
                .collect();

            // Row-major (contiguous) strides for a shape.
            let contiguous = |dims: &[i64]| -> Vec<i64> {
                let mut s = vec![1i64; dims.len()];
                for i in (0..dims.len().saturating_sub(1)).rev() {
                    s[i] = s[i + 1] * dims[i + 1];
                }
                s
            };

            // The second argument is an integer array: the target shape for
            // `reshape`, or the dimension permutation for `transpose`.
            let idx_vals: Vec<i64> = match &args[1] {
                Expr::Array(arr) => arr
                    .elements
                    .iter()
                    .filter_map(|el| match el {
                        Expr::Number(num) => num.value.parse::<i64>().ok(),
                        _ => None,
                    })
                    .collect(),
                _ => Vec::new(),
            };

            let dims_str = |dims: &[i64]| -> String {
                let mut s = String::new();
                for d in dims {
                    s.push_str(&d.to_string());
                    s.push('x');
                }
                s.push_str(el_ty_str);
                s
            };
            let i64_list = |v: &[i64]| -> String {
                v.iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            let reinterpret = |src: Value<'c, 'c>,
                               sizes: &[i64],
                               strides: &[i64],
                               result_ty: Type<'c>,
                               blk: melior::ir::BlockRef<'c, 'c>|
             -> Value<'c, 'c> {
                let op = OperationBuilder::new("memref.reinterpret_cast", gen.loc())
                    .add_operands(&[src])
                    .add_attributes(&[
                        (
                            Identifier::new(gen.context, "operandSegmentSizes"),
                            DenseI32ArrayAttribute::new(gen.context, &[1, 0, 0, 0]).into(),
                        ),
                        (
                            Identifier::new(gen.context, "static_offsets"),
                            DenseI64ArrayAttribute::new(gen.context, &[0]).into(),
                        ),
                        (
                            Identifier::new(gen.context, "static_sizes"),
                            DenseI64ArrayAttribute::new(gen.context, sizes).into(),
                        ),
                        (
                            Identifier::new(gen.context, "static_strides"),
                            DenseI64ArrayAttribute::new(gen.context, strides).into(),
                        ),
                    ])
                    .add_results(&[result_ty])
                    .build()
                    .expect("Failed to build reinterpret_cast");
                blk.append_operation(op)
                    .result(0)
                    .expect("Failed to get result")
                    .into()
            };

            if is_transpose {
                let src_strides = contiguous(&src_dims);
                let view_dims: Vec<i64> = idx_vals.iter().map(|&p| src_dims[p as usize]).collect();
                let view_strides: Vec<i64> =
                    idx_vals.iter().map(|&p| src_strides[p as usize]).collect();

                let view_ty = Type::parse(
                    gen.context,
                    &format!(
                        "memref<{}, strided<[{}], offset: 0>>",
                        dims_str(&view_dims),
                        i64_list(&view_strides)
                    ),
                )
                .unwrap();
                let view_val = reinterpret(arg_val, &view_dims, &view_strides, view_ty, block);

                let dst_ty =
                    Type::parse(gen.context, &format!("memref<{}>", dims_str(&view_dims))).unwrap();
                let alloc = OperationBuilder::new("memref.alloc", gen.loc())
                    .add_attributes(&[(
                        Identifier::new(gen.context, "operandSegmentSizes"),
                        DenseI32ArrayAttribute::new(gen.context, &[0, 0]).into(),
                    )])
                    .add_results(&[dst_ty])
                    .build()?;
                let dst_val = block.append_operation(alloc).result(0)?.into();

                let copy = OperationBuilder::new("memref.copy", gen.loc())
                    .add_operands(&[view_val, dst_val])
                    .build()?;
                block.append_operation(copy);

                return Ok((dst_val, dst_ty, block));
            }

            // reshape
            let tgt_dims = idx_vals;
            let tgt_strides = contiguous(&tgt_dims);
            let tgt_ty =
                Type::parse(gen.context, &format!("memref<{}>", dims_str(&tgt_dims))).unwrap();
            let out_val = reinterpret(arg_val, &tgt_dims, &tgt_strides, tgt_ty, block);
            return Ok((out_val, tgt_ty, block));
        }

        if name.as_ref() == "with_memory" {
            // For now, with_memory is a no-op in lowering, just returns the tensor
            let (arg_val, expr_ty, block) = gen.generate_expr(&args[0], block)?;
            return Ok((arg_val, expr_ty, block));
        }

        if name.as_ref() == "map" {
            return lower_map_call(gen, block, args);
        }

        if name.as_ref() == "print" {
            return lower_print_call(gen, block, args);
        }

        if name.as_ref() == "printf" || **name == *"vx_internal_printf" {
            let mut arg_vals = Vec::new();
            let mut current_b = block;
            for arg in args {
                let (arg_val, _arg_ty, new_b) = gen.generate_expr(arg, current_b)?;
                arg_vals.push(arg_val);
                current_b = new_b;
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
                    .build()?;

                gen.module.body().append_operation(printf_decl);
                gen.functions.insert(
                    "llvm_printf_decl".to_string().into(),
                    (gen.i32_ty, vec![gen.ptr_ty]),
                );
            }

            let call_op = current_b.append_operation(
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
                    .build()?,
            );

            return Ok((call_op.result(0)?.into(), gen.i32_ty, current_b));
        }

        if let Some((ret_ty, arg_tys)) = gen.functions.get(name).cloned() {
            let mut arg_vals = Vec::new();
            let mut current_b = block;
            for (i, arg) in args.iter().enumerate() {
                let field_ty = arg_tys[i];
                let prev_expected = gen.expected_type;
                gen.expected_type = Some(field_ty);
                let (mut arg_val, expr_ty, new_b) = gen.generate_expr(arg, current_b)?;
                current_b = new_b;
                gen.expected_type = prev_expected;
                if expr_ty != field_ty {
                    if gen.is_memref(&expr_ty) && gen.is_memref(&field_ty) {
                        let cast_op = OperationBuilder::new("memref.cast", gen.loc())
                            .add_operands(&[arg_val])
                            .add_results(&[field_ty])
                            .build()
                            .unwrap();
                        arg_val = current_b.append_operation(cast_op).result(0)?.into();
                    } else if let Some(adapted) =
                        gen.adapt_closure_to_nominal(&current_b, arg_val, expr_ty, field_ty)?
                    {
                        // A closure literal (`Closure_N` env) passed where a nominal `ClosureK`
                        // struct is expected (e.g. `.map(|x| ..)`): build the `{env, func}` value.
                        arg_val = adapted;
                    } else {
                        arg_val = gen.coerce_type(&current_b, arg_val, expr_ty, field_ty)?;
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
                let call_op = builder.build()?;
                let call_ref = current_b.append_operation(call_op);
                Ok((call_ref.result(0)?.into(), ret_ty, current_b))
            } else {
                let call_op = builder.build()?;
                current_b.append_operation(call_op);
                let none_ty = gen.none_ty;
                // this value shouldn't be used
                let dummy_op = OperationBuilder::new("llvm.mlir.constant", gen.loc())
                    .add_results(&[gen.i32_ty])
                    .add_attributes(&[(
                        Identifier::new(gen.context, "value"),
                        IntegerAttribute::new(gen.i32_ty, 0).into(),
                    )])
                    .build()?;
                Ok((
                    current_b.append_operation(dummy_op).result(0)?.into(),
                    none_ty,
                    current_b,
                ))
            }
        } else if let Some((ptr_val, func_ty)) = gen.env.get(name).cloned() {
            let mut actual_func_ty = func_ty;
            let is_closure = func_ty.to_string() == "!llvm.struct<(ptr, ptr)>";
            if func_ty.to_string() == "!llvm.ptr" {
                if let Some(syntax::Type::Function(func_args, ret)) = gen.ast_env.get(name) {
                    let r = gen.lower_type(ret.as_ref())?;
                    let mut a: Vec<_> = Vec::new();
                    for t in func_args.iter() {
                        a.push(gen.lower_type(t)?);
                    }
                    actual_func_ty =
                        melior::ir::r#type::FunctionType::new(gen.context, &a, &[r]).into();
                } else {
                    panic!("Missing signature for function pointer '{}'", name);
                }
            } else if is_closure {
                if let Some(syntax::Type::Closure(func_args, ret)) = gen.ast_env.get(name) {
                    let r = gen.lower_type(ret.as_ref())?;
                    let mut a: Vec<_> = vec![gen.ptr_ty];
                    for t in func_args.iter() {
                        a.push(gen.lower_type(t)?);
                    }
                    actual_func_ty =
                        melior::ir::r#type::FunctionType::new(gen.context, &a, &[r]).into();
                } else {
                    panic!("Missing signature for closure '{}'", name);
                }
            }
            if let Ok(mlir_func_ty) = melior::ir::r#type::FunctionType::try_from(actual_func_ty) {
                let ret_ty = mlir_func_ty.result(0)?;
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
                        .build()?;
                    let extract_env_ref = block.append_operation(extract_env_op);
                    env_ptr = Some(extract_env_ref.result(0)?.into());

                    // Extract func_ptr
                    let extract_func_op = OperationBuilder::new("llvm.extractvalue", gen.loc())
                        .add_operands(&[ptr_val])
                        .add_attributes(&[(
                            Identifier::new(gen.context, "position"),
                            melior::ir::attribute::DenseI64ArrayAttribute::new(gen.context, &[1])
                                .into(),
                        )])
                        .add_results(&[gen.ptr_ty])
                        .build()?;
                    let extract_func_ref = block.append_operation(extract_func_op);
                    actual_ptr_val = extract_func_ref.result(0)?.into();
                }

                let mut current_b = block;
                for (i, arg) in args.iter().enumerate() {
                    let (mut arg_val, expr_ty, new_b) = gen.generate_expr(arg, current_b)?;
                    current_b = new_b;
                    let field_ty = mlir_func_ty.input(i + arg_offset).unwrap();
                    if expr_ty != field_ty {
                        arg_val = gen.coerce_type(&current_b, arg_val, expr_ty, field_ty)?;
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
                    let cast_ref = current_b.append_operation(cast_op);
                    actual_ptr_val = cast_ref.result(0)?.into();
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
                    let call_op = builder.build()?;
                    let call_ref = current_b.append_operation(call_op);
                    Ok((call_ref.result(0)?.into(), ret_ty, current_b))
                } else {
                    let call_op = builder.build()?;
                    current_b.append_operation(call_op);
                    let none_ty = gen.none_ty;
                    let dummy_op = OperationBuilder::new("llvm.mlir.constant", gen.loc())
                        .add_results(&[gen.i32_ty])
                        .add_attributes(&[(
                            Identifier::new(gen.context, "value"),
                            IntegerAttribute::new(gen.i32_ty, 0).into(),
                        )])
                        .build()?;
                    Ok((
                        current_b.append_operation(dummy_op).result(0)?.into(),
                        none_ty,
                        current_b,
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
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let IndirectCallExpr {
            callee,
            args,
            target_func_ty,
            span: _,
        } = self;

        let (callee_val, callee_ty, block) = gen.generate_expr(callee, block)?;

        if callee_ty.to_string() == "!llvm.struct<(ptr, ptr)>" {
            // Extract env_ptr
            let extract_env_op = OperationBuilder::new("llvm.extractvalue", gen.loc())
                .add_operands(&[callee_val])
                .add_attributes(&[(
                    Identifier::new(gen.context, "position"),
                    melior::ir::attribute::DenseI64ArrayAttribute::new(gen.context, &[0]).into(),
                )])
                .add_results(&[gen.ptr_ty])
                .build()?;
            let extract_env_ref = block.append_operation(extract_env_op);
            let env_ptr = extract_env_ref.result(0)?.into();

            // Extract func_ptr
            let extract_func_op = OperationBuilder::new("llvm.extractvalue", gen.loc())
                .add_operands(&[callee_val])
                .add_attributes(&[(
                    Identifier::new(gen.context, "position"),
                    melior::ir::attribute::DenseI64ArrayAttribute::new(gen.context, &[1]).into(),
                )])
                .add_results(&[gen.ptr_ty])
                .build()?;
            let extract_func_ref = block.append_operation(extract_func_op);
            let mut actual_ptr_val = extract_func_ref.result(0)?.into();

            let mut arg_vals = Vec::new();
            let mut current_b = block;
            for arg in args {
                let (arg_val, _, new_b) = gen.generate_expr(arg, current_b)?;
                arg_vals.push(arg_val);
                current_b = new_b;
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
            let syntax::Type::Closure(func_args, ret) = target_func_ty else {
                panic!("Expected Type::Closure for indirect call fat pointer target");
            };

            let r = gen.lower_type(ret)?;
            let mut a: Vec<_> = vec![gen.ptr_ty];
            for t in func_args.iter() {
                a.push(gen.lower_type(t)?);
            }
            let actual_mlir_func_ty = melior::ir::r#type::FunctionType::new(gen.context, &a, &[r]);
            let actual_func_ty: melior::ir::Type = actual_mlir_func_ty.into();

            let ret_ty = actual_mlir_func_ty.result(0)?;

            // Cast the raw func ptr to the actual function signature
            let cast_op = OperationBuilder::new("builtin.unrealized_conversion_cast", gen.loc())
                .add_operands(&[actual_ptr_val])
                .add_results(&[actual_func_ty])
                .build()?;
            let cast_ref = current_b.append_operation(cast_op);
            actual_ptr_val = cast_ref.result(0)?.into();

            let mut builder = OperationBuilder::new("func.call_indirect", gen.loc())
                .add_operands(&[actual_ptr_val, env_ptr]);

            for a_val in &arg_vals {
                builder = builder.add_operands(&[*a_val]);
            }

            if ret_ty.to_string() != "none" {
                builder = builder.add_results(&[ret_ty]);
                let call_op = builder.build()?;
                let call_ref = current_b.append_operation(call_op);
                Ok((call_ref.result(0)?.into(), ret_ty, current_b))
            } else {
                let call_op = builder.build()?;
                current_b.append_operation(call_op);
                let none_ty = gen.none_ty;
                let dummy_op = OperationBuilder::new("llvm.mlir.constant", gen.loc())
                    .add_results(&[gen.i32_ty])
                    .add_attributes(&[(
                        Identifier::new(gen.context, "value"),
                        IntegerAttribute::new(gen.i32_ty, 0).into(),
                    )])
                    .build()?;
                Ok((
                    current_b.append_operation(dummy_op).result(0)?.into(),
                    none_ty,
                    current_b,
                ))
            }
        } else {
            panic!("Unsupported callee type for indirect call: {}", callee_ty);
        }
    }
}

impl<'c> LowerToMelior<'c> for MethodCallExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
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
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;

    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let mut mlir_args = Vec::new();
        let mut input_types_str = Vec::new();

        for (name, expr, ty_str) in &self.inputs {
            let (mut val, val_ty, block) = gen.generate_expr(expr, block)?;
            if val_ty.to_string().starts_with("memref<memref<") && ty_str.starts_with("memref<") {
                let load_op = OperationBuilder::new("memref.load", gen.loc())
                    .add_operands(&[val])
                    .add_results(&[Type::parse(gen.context, ty_str).ok_or_else(|| {
                        crate::codegen::lower::LowerError::ParseType(
                            "Type::parse failed".to_string(),
                        )
                    })?])
                    .build()?;
                val = block.append_operation(load_op).result(0)?.into();
            }
            mlir_args.push(val);
            input_types_str.push(format!("{}: {}", name, ty_str));
        }

        let args_str = input_types_str.join(", ");

        let mut ret_str = String::new();
        let mut call_ret_tys = Vec::new();
        if let Some(ret_ty) = &self.returns {
            let mlir_ret_ty = gen.lower_type(ret_ty)?;
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
            .build()?;

        let op = block.append_operation(call_op);

        if let Some(ret_ty) = &self.returns {
            Ok((op.result(0)?.into(), gen.lower_type(ret_ty)?, block))
        } else {
            let dummy_val = OperationBuilder::new("arith.constant", gen.loc())
                .add_attributes(&[(
                    Identifier::new(gen.context, "value"),
                    IntegerAttribute::new(Type::index(gen.context), 0).into(),
                )])
                .add_results(&[Type::index(gen.context)])
                .build()?;
            Ok((
                block.append_operation(dummy_val).result(0)?.into(),
                Type::index(gen.context),
                block,
            ))
        }
    }
}

impl<'c> LowerToMelior<'c> for ArrayExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let ArrayExpr { elements, span: _ } = self;
        if elements.is_empty() {
            panic!("Empty arrays not supported yet");
        }
        let mut vals = Vec::new();
        let mut el_ty = None;
        let mut current_b = block;
        for el in elements {
            let (v, t, new_b) = gen.generate_expr(el, current_b)?;
            vals.push(v);
            current_b = new_b;
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
            .build()?;

        let val = current_b.append_operation(op).result(0)?.into();
        Ok((val, tensor_ty, current_b))
    }
}

impl<'c> LowerToMelior<'c> for MemorySpaceExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        _gen: &mut MeliorGenerator<'c>,
        _block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        panic!("Should not be evaluated directly")
    }
}

impl<'c> LowerToMelior<'c> for TopologyExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        // A topology used as a value lowers to its stable runtime dispatch id
        // (an i32 discriminant, `arch::topology_dispatch_id`). This makes
        // `Topology::GPU` storable (e.g. in `Vec<Topology>`) and comparable at
        // runtime. Comptime placement comparisons are folded before codegen, so
        // a topology reaching here is genuinely being used as a value.
        let i32_ty = gen.i32_ty;
        let id = crate::arch::topology_dispatch_id(&self.top) as i64;
        let op = OperationBuilder::new("arith.constant", gen.loc())
            .add_results(&[i32_ty])
            .add_attributes(&[(
                Identifier::new(gen.context, "value"),
                IntegerAttribute::new(i32_ty, id).into(),
            )])
            .build()?;
        let val = block.append_operation(op).result(0)?.into();
        Ok((val, i32_ty, block))
    }
}

impl<'c> LowerToMelior<'c> for NumberExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let NumberExpr {
            value: val_str,
            ty: ast_ty_opt,
            span: _,
        } = self;
        let ty = if let Some(ast_ty) = ast_ty_opt {
            gen.lower_type(&syntax::Type::Scalar(ast_ty.clone()))
                .unwrap()
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
                .build()?;
            let op_ref = block.append_operation(op);
            Ok((op_ref.result(0)?.into(), ty, block))
        } else {
            let op = OperationBuilder::new("arith.constant", gen.loc())
                .add_results(&[ty])
                .add_attributes(&[(
                    Identifier::new(gen.context, "value"),
                    IntegerAttribute::new(ty, val_str.parse::<i64>().unwrap()).into(),
                )])
                .build()?;
            let op_ref = block.append_operation(op);
            Ok((op_ref.result(0)?.into(), ty, block))
        }
    }
}

impl<'c> LowerToMelior<'c> for MatchExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let (match_val, match_ty, block) = gen.generate_expr(&self.expr, block)?;

        let parent_region = block.parent_region().unwrap();
        let merge_block = parent_region.append_block(melior::ir::Block::new(&[]));
        generate_match_chain(gen, &self.arms, match_val, match_ty, block, merge_block)?;
        let block = merge_block;

        // Return dummy value for now like IfExpr
        let ty = melior::ir::r#type::IntegerType::new(gen.context, 32).into();
        let op = OperationBuilder::new("arith.constant", gen.loc())
            .add_results(&[ty])
            .add_attributes(&[(
                Identifier::new(gen.context, "value"),
                IntegerAttribute::new(ty, 0).into(),
            )])
            .build()?;
        let op_ref = block.append_operation(op);
        Ok((op_ref.result(0)?.into(), ty, block))
    }
}

impl<'c> LowerToMelior<'c> for EnumVariantExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
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
                if *v.0 == **variant_name {
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
                            "i32" => syntax::Type::Scalar(syntax::ElementType::I32),
                            "f32" => syntax::Type::Scalar(syntax::ElementType::F32),
                            "f64" => syntax::Type::Scalar(syntax::ElementType::F64),
                            "i64" => syntax::Type::Scalar(syntax::ElementType::I64),
                            "Bool" => syntax::Type::Scalar(syntax::ElementType::Bool),
                            _ => syntax::Type::Struct(ty_arg.to_string().into(), None),
                        };
                        let t = syntax::Type::GenericInstance(
                            Box::new(syntax::Type::Struct(base.to_string().into(), None)),
                            vec![parsed_ty],
                        );
                        enum_ty_str = gen.lower_type_str(&t)?;
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
            .build()?;
        let tag_val = block.append_operation(tag_op).result(0)?.into();

        if !has_payload {
            return Ok((tag_val, i32_ty, block));
        }

        // We have an Option<T> struct
        let struct_ty = Type::parse(gen.context, &enum_ty_str).ok_or_else(|| {
            crate::codegen::lower::LowerError::ParseType("Type::parse failed".to_string())
        })?;
        let undef_op = OperationBuilder::new("llvm.mlir.undef", gen.loc())
            .add_results(&[struct_ty])
            .build()?;
        let undef_val = block.append_operation(undef_op).result(0)?.into();

        let insert_tag_op = OperationBuilder::new("llvm.insertvalue", gen.loc())
            .add_operands(&[undef_val, tag_val])
            .add_results(&[struct_ty])
            .add_attributes(&[(
                Identifier::new(gen.context, "position"),
                melior::ir::attribute::DenseI64ArrayAttribute::new(gen.context, &[0]).into(),
            )])
            .build()?;
        let mut struct_val = block.append_operation(insert_tag_op).result(0)?.into();

        if let Some(payload_exprs) = payload {
            if !payload_exprs.is_empty() {
                let (payload_val, _payload_ty, block) =
                    gen.generate_expr(&payload_exprs[0], block)?;
                let insert_payload_op = OperationBuilder::new("llvm.insertvalue", gen.loc())
                    .add_operands(&[struct_val, payload_val])
                    .add_results(&[struct_ty])
                    .add_attributes(&[(
                        Identifier::new(gen.context, "position"),
                        melior::ir::attribute::DenseI64ArrayAttribute::new(gen.context, &[1])
                            .into(),
                    )])
                    .build()?;
                struct_val = block.append_operation(insert_payload_op).result(0)?.into();
            }
        }

        Ok((struct_val, struct_ty, block))
    }
}

impl<'c> LowerToMelior<'c> for VecMacroExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let mut el_ty = syntax::ElementType::F32;
        if !self.elements.is_empty() {
            if let Some(syntax::Type::Scalar(t)) = gen.infer_ast_type(&self.elements[0]) {
                el_ty = t;
            } else if let Some(syntax::Type::Struct(s, _)) = gen.infer_ast_type(&self.elements[0]) {
                if s == "String".into() {
                    // String is equivalent to pointer, but generic instantiation requires element type
                }
            }
        }

        let type_suffix = match el_ty {
            syntax::ElementType::I32 => "i32",
            syntax::ElementType::F32 => "f32",
            syntax::ElementType::I64 => "i64",
            syntax::ElementType::F64 => "f64",
            syntax::ElementType::Bool => "Bool",
            _ => {
                if let Some(syntax::Type::Struct(s, _)) =
                    gen.infer_ast_type(self.elements.first().unwrap_or(&Expr::Number(NumberExpr {
                        value: "0".into(),
                        ty: None,
                        span: Span::default(),
                    })))
                {
                    if s == "String".into() {
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
            name: format!("Vec_{}::new", type_suffix).into(),
            args: vec![],
            type_args: None,
            span: self.span,
        });

        let (vec_val, vec_ty, block) = gen.generate_expr(&new_call, block)?;

        let i32_ty = gen.i32_ty;
        let c1_op = block.append_operation(
            OperationBuilder::new("llvm.mlir.constant", gen.loc())
                .add_results(&[i32_ty])
                .add_attributes(&[(
                    Identifier::new(gen.context, "value"),
                    IntegerAttribute::new(i32_ty, 1).into(),
                )])
                .build()?,
        );
        let c1 = c1_op.result(0)?.into();

        let ptr_ty = gen.ptr_ty;
        let alloca_op = block.append_operation(
            OperationBuilder::new("llvm.alloca", gen.loc())
                .add_operands(&[c1])
                .add_results(&[ptr_ty])
                .add_attributes(&[(
                    Identifier::new(gen.context, "elem_type"),
                    TypeAttribute::new(vec_ty).into(),
                )])
                .build()?,
        );
        let ptr_val = alloca_op.result(0)?.into();

        block.append_operation(
            OperationBuilder::new("llvm.store", gen.loc())
                .add_operands(&[vec_val, ptr_val])
                .build()?,
        );

        let tmp_vec_name = format!("__vec_ptr_{}", gen.string_counter);
        gen.string_counter += 1;
        gen.env
            .insert(tmp_vec_name.clone().into(), (ptr_val, vec_ty));
        gen.allocs.insert(tmp_vec_name.clone());

        for el in &self.elements {
            let push_call = Expr::FunctionCall(FunctionCallExpr {
                name: format!("Vec_{}::push", type_suffix).into(),
                args: vec![
                    Expr::Borrow(BorrowExpr {
                        expr: Box::new(Expr::Identifier(IdentifierExpr {
                            name: tmp_vec_name.clone().into(),
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
                .build()?,
        );
        Ok((load_op.result(0)?.into(), vec_ty, block))
    }
}

impl<'c> LowerToMelior<'c> for ClosureExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;

    fn lower(
        &self,
        _gen: &mut MeliorGenerator<'c>,
        _block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        unreachable!("ClosureExpr should be transformed to StructInitExpr by Sema")
    }
}

impl<'c> LowerToMelior<'c> for syntax::expr::PrintExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        for arg in &self.args {
            let (arg_val, arg_ty, block) = gen.generate_expr(arg, block)?;

            let func_name = match arg_ty.to_string().as_ref() {
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
                    .build()?;

                gen.module.body().append_operation(func_decl);
                gen.functions.insert(
                    func_name.to_string().into(),
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
                .build()?;

            block.append_operation(call_op);
        }

        let dummy_op = OperationBuilder::new("llvm.mlir.constant", gen.loc())
            .add_results(&[gen.i32_ty])
            .add_attributes(&[(
                Identifier::new(gen.context, "value"),
                IntegerAttribute::new(gen.i32_ty, 0).into(),
            )])
            .build()?;

        Ok((
            block.append_operation(dummy_op).result(0)?.into(),
            gen.i32_ty,
            block,
        ))
    }
}

impl<'c> LowerToMelior<'c> for syntax::expr::PrintlnExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        // First, reuse PrintExpr logic for arguments
        if !self.args.is_empty() {
            let print_expr = syntax::expr::PrintExpr {
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
                .build()?;

            gen.module.body().append_operation(func_decl);
            gen.functions
                .insert("println".to_string().into(), (gen.i32_ty, vec![]));
        }

        let name_attr = FlatSymbolRefAttribute::new(gen.context, "println");
        let call_op = OperationBuilder::new("func.call", gen.loc())
            .add_results(&[gen.i32_ty])
            .add_attributes(&[(Identifier::new(gen.context, "callee"), name_attr.into())])
            .build()?;

        Ok((
            block.append_operation(call_op).result(0)?.into(),
            gen.i32_ty,
            block,
        ))
    }
}

impl<'c> LowerToMelior<'c> for syntax::expr::SizeOfExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;

    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let size: i64 = match &self.target_ty {
            // Scalar byte size from the single width source (`scalar_size_align` -> `ElementType::bits`),
            // rather than a fourth hand-maintained per-type table; also sizes sub-byte scalars (`i4` = 1)
            // that the old match dropped to the `_ => 8` fallback. (P1-4a)
            syntax::Type::Scalar(e) => crate::layout::scalar_size_align(e)
                .map(|(s, _)| s as i64)
                .unwrap_or(8),
            syntax::Type::Pointer(..) | syntax::Type::Borrow { .. } | syntax::Type::Ref(..) => 8,
            // A nominal struct (or a monomorphized generic instance of one) gets its *real* layout
            // size, not the old hardcoded `8` — otherwise a `Vec<struct>`'s element buffer sized by
            // `sizeof<T>()` under-allocates and its element stores run out of bounds (a `Vec<Vec<T>>`
            // UB). Falls back to `8` for a type the layout pass doesn't model (tensor, closure). (#242)
            ty @ (syntax::Type::Struct(..) | syntax::Type::GenericInstance(..)) => gen
                .type_size_align(ty, 0)
                .map(|(s, _)| s as i64)
                .unwrap_or(8),
            _ => 8,
        };

        let size_ty = gen.i64_ty;
        let const_op = OperationBuilder::new("arith.constant", gen.loc())
            .add_attributes(&[(
                Identifier::new(gen.context, "value"),
                IntegerAttribute::new(size_ty, size).into(),
            )])
            .add_results(&[size_ty])
            .build()?;

        let const_op = block.append_operation(const_op);
        Ok((const_op.result(0)?.into(), size_ty, block))
    }
}

impl<'c> LowerToMelior<'c> for syntax::expr::AsCastExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>;

    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let (source_val, _source_ty, block) = gen.generate_expr(&self.expr, block)?;
        if let syntax::Type::Closure(_, _) = &self.target_ty {
            let closure_struct_name = match self.source_ty.as_ref() {
                Some(syntax::Type::Struct(name, _)) => name.clone(),
                Some(syntax::Type::Borrow { inner, .. }) => {
                    if let syntax::Type::Struct(name, _) = &**inner {
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
                .get(&*call_fn_name)
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
                .build()?;
            let const_ref = block.append_operation(const_op);
            let mut fn_ptr_val: melior::ir::Value = const_ref.result(0)?.into();

            let ptr_ty = gen.ptr_ty;
            let bitcast_op = OperationBuilder::new("builtin.unrealized_conversion_cast", gen.loc())
                .add_operands(&[fn_ptr_val])
                .add_results(&[ptr_ty])
                .build()?;
            let bitcast_ref = block.append_operation(bitcast_op);
            fn_ptr_val = bitcast_ref.result(0)?.into();

            let fat_ptr_ty = Type::parse(gen.context, "!llvm.struct<(ptr, ptr)>").unwrap();
            let undef_op = OperationBuilder::new("llvm.mlir.undef", gen.loc())
                .add_results(&[fat_ptr_ty])
                .build()?;
            let undef_ref = block.append_operation(undef_op);
            let mut fat_ptr_val: melior::ir::Value = undef_ref.result(0)?.into();

            let insert_fn_op = OperationBuilder::new("llvm.insertvalue", gen.loc())
                .add_operands(&[fat_ptr_val, fn_ptr_val])
                .add_attributes(&[(
                    Identifier::new(gen.context, "position"),
                    melior::ir::attribute::DenseI64ArrayAttribute::new(gen.context, &[0]).into(),
                )])
                .add_results(&[fat_ptr_ty])
                .build()?;
            let insert_fn_ref = block.append_operation(insert_fn_op);
            fat_ptr_val = insert_fn_ref.result(0)?.into();

            let mut env_ptr_val = source_val;
            let ptr_bitcast_op =
                OperationBuilder::new("builtin.unrealized_conversion_cast", gen.loc())
                    .add_operands(&[env_ptr_val])
                    .add_results(&[ptr_ty])
                    .build()?;
            let ptr_bitcast_ref = block.append_operation(ptr_bitcast_op);
            env_ptr_val = ptr_bitcast_ref.result(0)?.into();

            let insert_env_op = OperationBuilder::new("llvm.insertvalue", gen.loc())
                .add_operands(&[fat_ptr_val, env_ptr_val])
                .add_attributes(&[(
                    Identifier::new(gen.context, "position"),
                    melior::ir::attribute::DenseI64ArrayAttribute::new(gen.context, &[1]).into(),
                )])
                .add_results(&[fat_ptr_ty])
                .build()?;
            let insert_env_ref = block.append_operation(insert_env_op);
            fat_ptr_val = insert_env_ref.result(0)?.into();

            return Ok((fat_ptr_val, fat_ptr_ty, block));
        } else if let syntax::Type::Scalar(_) = &self.target_ty {
            let target_ty_mlir = gen.lower_type(&self.target_ty)?;
            let coerced_val = gen.coerce_type(&block, source_val, _source_ty, target_ty_mlir)?;
            return Ok((coerced_val, target_ty_mlir, block));
        } else if let syntax::Type::Pointer(..) = &self.target_ty {
            if let Some(syntax::Type::Scalar(_)) = self.source_ty.as_ref() {
                let ptr_ty = gen.ptr_ty;
                let cast_op = OperationBuilder::new("llvm.inttoptr", gen.loc())
                    .add_operands(&[source_val])
                    .add_results(&[ptr_ty])
                    .build()?;
                let cast_ref = block.append_operation(cast_op);
                return Ok((cast_ref.result(0)?.into(), ptr_ty, block));
            }
        }

        panic!("Unsupported cast operation in codegen");
    }
}
