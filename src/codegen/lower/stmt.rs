use super::*;
use crate::syntax;
use crate::syntax::*;
use melior::ir::{
    attribute::{DenseI32ArrayAttribute, IntegerAttribute, TypeAttribute},
    operation::OperationBuilder,
    Identifier, Type,
};

impl<'c> LowerToMelior<'c> for ReturnStmt {
    type Output = Result<Option<melior::ir::BlockRef<'c, 'c>>, LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let ReturnStmt { expr, span: _ } = self;
        gen.expected_type = gen.current_return_type;
        let (mut val, expr_ty, block) = gen.generate_expr(expr, block)?;
        gen.expected_type = None;
        if let Some(ret_ty) = gen.current_return_type {
            if expr_ty != ret_ty {
                if gen.is_memref(&expr_ty) && gen.is_memref(&ret_ty) {
                    let expr_parts = expr_ty.to_string();
                    let ret_parts = ret_ty.to_string();
                    let space_differs =
                        expr_parts.matches(',').count() != ret_parts.matches(',').count();
                    // `memref.memory_space_cast` changes the space and nothing else, and
                    // `memref.cast` cannot change the space at all. A return whose type differs in
                    // both -- a placed tensor with a written shape, returned from a body that
                    // allocated it dynamically -- needs one of each, shape first (Vx#429).
                    if space_differs {
                        if let Some(no_space) =
                            crate::codegen::generator::strip_memref_space(&ret_parts)
                                .filter(|s| *s != expr_parts)
                        {
                            let shaped = Type::parse(gen.context, &no_space)
                                .ok_or_else(|| LowerError::ParseType(no_space.clone()))?;
                            let shape_op = OperationBuilder::new("memref.cast", gen.loc())
                                .add_operands(&[val])
                                .add_results(&[shaped])
                                .build()?;
                            val = block.append_operation(shape_op).result(0)?.into();
                        }
                    }
                    let cast_op_name = if space_differs {
                        "memref.memory_space_cast"
                    } else {
                        "memref.cast"
                    };
                    let cast_op = OperationBuilder::new(cast_op_name, gen.loc())
                        .add_operands(&[val])
                        .add_results(&[ret_ty])
                        .build()?;
                    val = block.append_operation(cast_op).result(0)?.into();
                } else if ret_ty.to_string() == "i32" && gen.is_memref(&expr_ty) {
                    let zero_op = OperationBuilder::new("arith.constant", gen.loc())
                        .add_results(&[ret_ty])
                        .add_attributes(&[(
                            Identifier::new(gen.context, "value"),
                            IntegerAttribute::new(ret_ty, 0).into(),
                        )])
                        .build()?;
                    val = block.append_operation(zero_op).result(0)?.into();
                } else {
                    val = gen.coerce_type(&block, val, expr_ty, ret_ty)?;
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
            .build()?;
        block.append_operation(ret_op);
        gen.has_returned = true;
        Ok(None)
    }
}

impl<'c> LowerToMelior<'c> for LetDeclStmt {
    type Output = Result<Option<melior::ir::BlockRef<'c, 'c>>, LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let LetDeclStmt {
            name,
            is_mut,
            ty_ann,
            expr,
            span: _,
        } = self;
        let prev_expected = gen.expected_type;
        if let Some(ann) = ty_ann {
            gen.expected_type = gen.lower_type(ann).ok();
        }
        let (val, ty, block) = gen.generate_expr(expr, block)?;
        if let Expr::Closure(c) = expr {
            let func_args = c.params.iter().map(|(_, t)| t.clone()).collect();
            let ret_ty = c
                .ret_ty
                .clone()
                .unwrap_or(syntax::Type::Scalar(syntax::ElementType::I32));
            gen.ast_env.insert(
                name.clone(),
                syntax::Type::Closure(func_args, Box::new(ret_ty)),
            );
        }
        gen.expected_type = prev_expected;

        let ast_ty = ty_ann.clone().or_else(|| gen.infer_ast_type(expr));
        if let Some(ref t) = ast_ty {
            gen.ast_env.insert(name.to_string().into(), t.clone());
        }
        if syntax::is_tensor_construction(expr) {
            gen.owned_tensors.insert(name.to_string().into());
        }

        if *is_mut {
            let ty_str = ty.to_string();
            if ty_str.contains("!llvm.struct") || ty_str.contains("!llvm.ptr") {
                let ptr_ty = gen.ptr_ty;
                let i32_ty = gen.i32_ty;
                let one_attr = IntegerAttribute::new(i32_ty, 1).into();
                let const_op = OperationBuilder::new("llvm.mlir.constant", gen.loc())
                    .add_results(&[i32_ty])
                    .add_attributes(&[(Identifier::new(gen.context, "value"), one_attr)])
                    .build()?;
                let one_val = block.append_operation(const_op).result(0)?.into();

                let alloca_op = OperationBuilder::new("llvm.alloca", gen.loc())
                    .add_operands(&[one_val])
                    .add_results(&[ptr_ty])
                    .add_attributes(&[(
                        Identifier::new(gen.context, "elem_type"),
                        TypeAttribute::new(ty).into(),
                    )])
                    .build()?;
                let alloca_ref = block.append_operation(alloca_op);
                let alloca_val = alloca_ref.result(0)?.into();

                let store_op = OperationBuilder::new("llvm.store", gen.loc())
                    .add_operands(&[val, alloca_val])
                    .build()?;
                block.append_operation(store_op);

                gen.env.insert(name.to_string().into(), (alloca_val, ty));
                gen.allocs.insert(name.to_string());
            } else {
                let memref_ty = format!("memref<{}>", ty);
                let parsed_memref_ty = Type::parse(gen.context, &memref_ty).ok_or_else(|| {
                    crate::codegen::lower::LowerError::ParseType("Type::parse failed".to_string())
                })?;
                let alloca_op = OperationBuilder::new("memref.alloca", gen.loc())
                    .add_results(&[parsed_memref_ty])
                    .build()?;
                let alloca_ref = block.append_operation(alloca_op);
                let alloca_val = alloca_ref.result(0)?.into();

                let store_op = OperationBuilder::new("memref.store", gen.loc())
                    .add_operands(&[val, alloca_val])
                    .build()?;
                block.append_operation(store_op);
                gen.env
                    .insert(name.to_string().into(), (alloca_val, parsed_memref_ty));
                gen.allocs.insert(name.to_string());
            }
        } else {
            gen.env.insert(name.to_string().into(), (val, ty));
        }
        Ok(Some(block))
    }
}

/// The nominal struct/enum name a type ultimately names, peeling reference/pointer wrappers and
/// generic instantiation. `&Vec<i32>` -> `Vec`. Used to recover a struct name for field assignment
/// when the base lowers to a bare `!llvm.ptr` (which carries no struct identity) and sema did not
/// stamp `struct_name` on the assignment target (#205).
fn nominal_struct_name(ty: &syntax::Type) -> Option<crate::symbol::Symbol> {
    use syntax::Type;
    match ty {
        Type::Struct(n, _) | Type::Enum(n, _) => Some(n.clone()),
        Type::GenericInstance(base, _) => nominal_struct_name(base),
        Type::Ref(inner, _)
        | Type::Pointer(inner, _, _)
        | Type::Verified(inner)
        | Type::Pinned(inner, _) => nominal_struct_name(inner),
        Type::Borrow { inner, .. } => nominal_struct_name(inner),
        _ => None,
    }
}

/// Substitution `struct generic param -> concrete type arg`, parsed from a monomorphized struct name
/// like `Vec<i32>` and the struct's declared generics. Mirrors the read-path logic so a generic
/// field type (`data: *mut T`) lowers correctly on the *assignment* path (#205), instead of a raw
/// `T` reaching codegen.
fn generic_arg_mapping(
    resolved_struct_name: &str,
    generics: &[syntax::GenericParam],
) -> std::collections::HashMap<crate::symbol::Symbol, syntax::Type> {
    use syntax::Type;
    let mut mapping = std::collections::HashMap::new();
    let (Some(lt), true) = (
        resolved_struct_name.find('<'),
        resolved_struct_name.ends_with('>'),
    ) else {
        return mapping;
    };
    let inner = &resolved_struct_name[lt + 1..resolved_struct_name.len() - 1];
    let inner_tys: Vec<Type> = inner
        .split(',')
        .map(|raw| {
            let a = raw.trim();
            match a {
                "i32" => Type::Scalar(syntax::ElementType::I32),
                "f32" => Type::Scalar(syntax::ElementType::F32),
                "i64" => Type::Scalar(syntax::ElementType::I64),
                _ if !a.is_empty() && a.chars().all(|c| c.is_ascii_digit()) => {
                    Type::Const(Box::new(syntax::Expr::Number(
                        syntax::expr::NumberExpr::new(a.to_string(), None, syntax::Span::default()),
                    )))
                }
                _ => Type::Struct(a.to_string().into(), None),
            }
        })
        .collect();
    for (i, param) in generics.iter().enumerate() {
        if let Some(ty) = inner_tys.get(i) {
            mapping.insert(param.name().into(), ty.clone());
        }
    }
    mapping
}

/// Do `dst = a @ b`'s three tensors have statically agreeing rank-2 shapes?
///
/// Only then is the destination known to be the size of the product, which is what makes
/// filling it in place safe. A dynamic operand has no product shape for the checker to have
/// compared `dst` against, and keeps the allocate-and-publish form.
fn matmul_assign_shapes_agree(gen: &MeliorGenerator<'_>, dst: &Expr, a: &Expr, b: &Expr) -> bool {
    // Filling `dst` in place zeroes it before the multiply reads its inputs, so an operand
    // naming the same buffer would read zeros. Each of the three has to be a local declared a
    // tensor -- a name bound to a borrow denotes whatever it points at, and following that is
    // the analysis this check exists to avoid.
    let root = |e: &Expr| -> Option<crate::symbol::Symbol> {
        let r = syntax::matmul_operand_root(e)?;
        let named = Expr::Identifier(IdentifierExpr {
            name: r.clone(),
            span: Span::default(),
        });
        // A name declared a borrow or a pointer denotes whatever it points at, which is the
        // alias the destination's own name does not reveal.
        match gen.infer_ast_type(&named) {
            Some(syntax::Type::Borrow { .. } | syntax::Type::Pointer(..)) => None,
            _ => Some(r.clone()),
        }
    };
    let (Some(d), Some(l), Some(r)) = (root(dst), root(a), root(b)) else {
        return false;
    };
    if !gen.owned_tensors.contains(&d) || d == l || d == r {
        return false;
    }

    let dims = |e: &Expr| -> Option<Vec<i64>> {
        let e = match e {
            Expr::Borrow(borrow) => borrow.expr.as_ref(),
            other => other,
        };
        match gen.infer_ast_type(e)? {
            syntax::Type::Tensor(_, dims, _) => dims
                .iter()
                .map(|d| match d {
                    Expr::Number(n) => n.value.as_ref().parse::<i64>().ok(),
                    _ => None,
                })
                .collect(),
            _ => None,
        }
    };
    let (Some(d), Some(l), Some(r)) = (dims(dst), dims(a), dims(b)) else {
        return false;
    };
    d.len() == 2 && l.len() == 2 && r.len() == 2 && l[1] == r[0] && d[0] == l[0] && d[1] == r[1]
}

impl<'c> LowerToMelior<'c> for AssignStmt {
    type Output = Result<Option<melior::ir::BlockRef<'c, 'c>>, LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let AssignStmt { lhs, rhs, span: _ } = self;

        // Slice/row initializer: `o[i] = [a, b, c, d]`. Lower the LHS partial index to its S1 row
        // view and store each element. This is cold setup code, so scalar stores are fine.
        if let (Expr::IndexAccess(_), Expr::Array(arr)) = (lhs, rhs) {
            let (row, row_ty, mut b) = gen.generate_expr(lhs, block)?;
            let row_ty_str = row_ty.to_string();
            if row_ty_str.starts_with("memref<") {
                let elem_ty_str = row_ty_str
                    .strip_prefix("memref<")
                    .and_then(|s| s.split(',').next())
                    .and_then(|s| s.rsplit('x').next())
                    .unwrap_or("f32");
                let elem_ty = Type::parse(gen.context, elem_ty_str).ok_or_else(|| {
                    crate::codegen::lower::LowerError::ParseType(elem_ty_str.to_string())
                })?;
                let index_ty = Type::index(gen.context);
                for (k, el) in arr.elements.iter().enumerate() {
                    let (v, vty, nb) = gen.generate_expr(el, b)?;
                    b = nb;
                    let v = gen.coerce_type(&b, v, vty, elem_ty)?;
                    let idx_op = OperationBuilder::new("arith.constant", gen.loc())
                        .add_attributes(&[(
                            Identifier::new(gen.context, "value"),
                            IntegerAttribute::new(index_ty, k as i64).into(),
                        )])
                        .add_results(&[index_ty])
                        .build()?;
                    let idx = b.append_operation(idx_op).result(0)?.into();
                    let store = OperationBuilder::new("memref.store", gen.loc())
                        .add_operands(&[v, row, idx])
                        .build()?;
                    b.append_operation(store);
                }
                return Ok(Some(b));
            }
        }

        // `c = a @ b` where `c` already names a buffer: fill it, rather than allocating a second
        // one and rebinding `c` to it. The checker has compared `c`'s declared shape against the
        // product, so the destination is the right size (Vx#391).
        if let Expr::BinaryOp(BinaryOpExpr {
            lhs: ml,
            op: BinaryOp::MatMul,
            rhs: mr,
            ..
        }) = rhs
        {
            if matmul_assign_shapes_agree(gen, lhs, ml, mr) {
                let (dst_val, mut dst_ty, block) = gen.generate_expr(lhs, block)?;
                let (a_val, mut a_ty, block) = gen.generate_expr(ml, block)?;
                let (b_val, mut b_ty, block) = gen.generate_expr(mr, block)?;
                let dst_val = super::expr::load_tensor_slot(gen, &block, dst_val, &mut dst_ty)?;
                let a_val = super::expr::load_tensor_slot(gen, &block, a_val, &mut a_ty)?;
                let b_val = super::expr::load_tensor_slot(gen, &block, b_val, &mut b_ty)?;
                let dst_ty_str = dst_ty.to_string();
                let el_ty_str = dst_ty_str
                    .rsplit('x')
                    .next()
                    .and_then(|t| t.strip_suffix('>'))
                    .ok_or_else(|| LowerError::from(format!("not a memref: {dst_ty_str}")))?
                    .to_string();
                super::expr::emit_matmul_into(gen, &block, a_val, b_val, dst_val, &el_ty_str)?;
                return Ok(Some(block));
            }
        }

        let mut expected_ty = None;
        if let Expr::Identifier(IdentifierExpr { name, span: _ }) = lhs {
            if let Some((_, mem_ty)) = gen.env.get(name) {
                let mem_ty_str = mem_ty.to_string();
                if mem_ty_str.starts_with("memref<") {
                    let inner_ty_str = &mem_ty_str[7..mem_ty_str.len() - 1];
                    expected_ty =
                        Some(Type::parse(gen.context, inner_ty_str).ok_or_else(|| {
                            crate::codegen::lower::LowerError::ParseType(
                                "Type::parse failed".to_string(),
                            )
                        })?);
                } else {
                    expected_ty = Some(*mem_ty);
                }
            }
        }

        if expected_ty.is_none() {
            if let Some(ast_ty) = gen.infer_ast_type(lhs) {
                expected_ty = gen.lower_type(&ast_ty).ok();
            }
        }

        let prev_expected = gen.expected_type;
        if expected_ty.is_some() {
            gen.expected_type = expected_ty;
        }
        let (rhs_val, rhs_ty, block) = gen.generate_expr(rhs, block)?;
        gen.expected_type = prev_expected;

        // Slice store (S3): `o[i] = <slice>` where the RHS is a vector<Dxf32> (from a slice
        // elementwise op). Lower the LHS partial index to its S1 row view (a reinterpret_cast'd
        // rank-1 memref aliasing `o`) and `vector.store` the result back into it.
        if rhs_ty.to_string().starts_with("vector<") {
            let (dst, _dst_ty, block) = gen.generate_expr(lhs, block)?;
            let index_ty = Type::index(gen.context);
            let c0_op = OperationBuilder::new("arith.constant", gen.loc())
                .add_attributes(&[(
                    Identifier::new(gen.context, "value"),
                    IntegerAttribute::new(index_ty, 0).into(),
                )])
                .add_results(&[index_ty])
                .build()?;
            let c0 = block.append_operation(c0_op).result(0)?.into();
            let store_op = OperationBuilder::new("vector.store", gen.loc())
                .add_operands(&[rhs_val, dst, c0])
                .build()?;
            block.append_operation(store_op);
            return Ok(Some(block));
        }

        if let Expr::Identifier(IdentifierExpr { name, span: _ }) = lhs {
            if let Some((mem_val, mem_ty)) = gen.env.get(name).cloned() {
                let mem_ty_str = mem_ty.to_string();
                if mem_ty_str.starts_with("memref<") {
                    let mut store_val = rhs_val;
                    let inner_ty_str = &mem_ty_str[7..mem_ty_str.len() - 1];
                    let inner_ty = Type::parse(gen.context, inner_ty_str).ok_or_else(|| {
                        crate::codegen::lower::LowerError::ParseType(
                            "Type::parse failed".to_string(),
                        )
                    })?;
                    // Coerce the value to the slot's element type. An annotated `let` now sizes its
                    // slot from the annotation and types its literal to match (#240), so a later
                    // assignment of a default-`i32` literal to a wider slot (`let mut r: i64; r = 5`)
                    // must widen; `coerce_type` also still handles the `index`↔`i32` loop case.
                    if rhs_ty != inner_ty {
                        store_val = gen.coerce_type(&block, rhs_val, rhs_ty, inner_ty)?;
                    }

                    let store_op = OperationBuilder::new("memref.store", gen.loc())
                        .add_operands(&[store_val, mem_val])
                        .build()?;
                    block.append_operation(store_op);
                } else if gen.allocs.contains(name.as_ref()) {
                    let store_op = OperationBuilder::new("llvm.store", gen.loc())
                        .add_operands(&[rhs_val, mem_val])
                        .build()?;
                    block.append_operation(store_op);
                } else {
                    gen.env.insert(name.to_string().into(), (rhs_val, rhs_ty));
                }
            }
        } else if let Expr::IndexAccess(syntax::IndexAccessExpr {
            base,
            index: _,
            span: _,
        }) = lhs
        {
            if let Some((base_val, base_ty, indices, new_b)) = gen.flatten_indices(
                &syntax::Expr::IndexAccess(syntax::IndexAccessExpr {
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
                        .build()?;
                    let idx_i64 = new_b.append_operation(cast_op).result(0)?.into();

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
                        .build()?;

                    let gep_ref = new_b.append_operation(gep_op);
                    let ptr_val = gep_ref.result(0)?.into();

                    let store_op = OperationBuilder::new("llvm.store", gen.loc())
                        .add_operands(&[rhs_val, ptr_val])
                        .build()?;

                    new_b.append_operation(store_op);
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
                        let inner_ty = melior::ir::Type::parse(gen.context, &inner_ty_str)
                            .ok_or_else(|| {
                                crate::codegen::lower::LowerError::ParseType(
                                    "Type::parse failed".to_string(),
                                )
                            })?;
                        store_val = gen.coerce_type(&new_b, store_val, rhs_ty, inner_ty)?;
                    }

                    let mut store_builder = OperationBuilder::new("memref.store", gen.loc())
                        .add_operands(&[store_val, base_val]);

                    for idx in indices {
                        store_builder = store_builder.add_operands(&[idx]);
                    }

                    let store_op = store_builder.build()?;
                    new_b.append_operation(store_op);
                }
                return Ok(Some(new_b));
            }
        } else if let Expr::Dereference(d) = lhs {
            // `*p = val`: store the RHS *through* the pointer. Without this case a dereference
            // assignment fell through to the no-op tail below and was **silently dropped** (`Box`'s
            // `*p = val` never wrote, so a read-back saw uninitialized memory). The store type comes
            // from the RHS value, so there's no pointee-type guesswork. (#242)
            let (ptr_val, _ptr_ty, new_b) = gen.generate_expr(d.expr.as_ref(), block)?;
            let store_op = OperationBuilder::new("llvm.store", gen.loc())
                .add_operands(&[rhs_val, ptr_val])
                .build()?;
            new_b.append_operation(store_op);
            return Ok(Some(new_b));
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
                let (base_val, base_ty, new_b) = gen.generate_expr(base, block)?;
                let base_ty_str = base_ty.to_string();

                let mut struct_name_opt = struct_name.clone();
                if struct_name_opt.is_none() {
                    if let Some(start_idx) = base_ty_str.find('"') {
                        if let Some(end_idx) = base_ty_str[start_idx + 1..].find('"') {
                            struct_name_opt = Some(
                                base_ty_str[start_idx + 1..start_idx + 1 + end_idx]
                                    .to_string()
                                    .into(),
                            );
                        }
                    }
                }
                // #205: a `&Struct` base lowers to a bare `!llvm.ptr` with no embedded struct name,
                // and sema does not always stamp `struct_name` on an assignment target. Without the
                // name we cannot find the field to store to, and the write is silently dropped (the
                // bug that broke `Vec::push`'s `self.len = self.len + 1`). Recover it from the base's
                // AST type.
                if struct_name_opt.is_none() {
                    if let Some(ast_ty) = gen.ast_env.get(base_name) {
                        struct_name_opt = nominal_struct_name(ast_ty);
                    }
                }

                let is_ptr = base_ty_str.starts_with("!llvm.ptr");

                if let Some(resolved_struct_name) = struct_name_opt {
                    // A monomorphized generic carries its instantiation in the name ("Vec<i32>"),
                    // but `gen.structs` and the emitted LLVM struct type are keyed by the *base*
                    // name ("Vec"). Strip the args before looking the declaration up — without this
                    // the lookup missed and the field store was silently dropped (#205, e.g.
                    // `Vec::push`'s `self.len = self.len + 1`). The field order (hence offset) is the
                    // same for every instantiation, so the generic declaration is the right one.
                    let base_struct: crate::symbol::Symbol = resolved_struct_name
                        .split('<')
                        .next()
                        .unwrap_or(resolved_struct_name.as_ref())
                        .into();
                    if let Some(struct_decl) = gen.structs.get(&base_struct).cloned() {
                        if let Some(field_idx) =
                            struct_decl.fields.iter().position(|(n, _)| n == member)
                        {
                            // Substitute the struct's generic params (T -> i32, …) so a generic field
                            // type like `data: *mut T` lowers instead of a raw `T` reaching codegen.
                            let mapping =
                                generic_arg_mapping(&resolved_struct_name, &struct_decl.generics);
                            let field_ty = gen.lower_type(
                                &struct_decl.fields[field_idx].1.substitute(&mapping),
                            )?;
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
                                field_val = new_b.append_operation(cast_op).result(0)?.into();
                            }

                            if is_ptr {
                                let ptr_ty = gen.ptr_ty;
                                let mut field_types = Vec::new();
                                for (_, ty) in &struct_decl.fields {
                                    let mut lowered =
                                        gen.lower_type_str(&ty.substitute(&mapping))?;
                                    if lowered.starts_with("memref<") {
                                        lowered = "!llvm.ptr".to_string();
                                    }
                                    field_types.push(lowered);
                                }
                                let struct_llvm_ty_str = format!(
                                    "!llvm.struct<\"{}\", ({})>",
                                    base_struct,
                                    field_types.join(", ")
                                );
                                let struct_llvm_ty = Type::parse(gen.context, &struct_llvm_ty_str)
                                    .ok_or_else(|| {
                                        crate::codegen::lower::LowerError::ParseType(
                                            "Type::parse failed".to_string(),
                                        )
                                    })?;

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

                                let gep_ref = new_b.append_operation(gep_op);
                                let ptr_val = gep_ref.result(0)?.into();

                                let store_op = OperationBuilder::new("llvm.store", gen.loc())
                                    .add_operands(&[field_val, ptr_val])
                                    .build()
                                    .unwrap();
                                new_b.append_operation(store_op);
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
                                    new_b.append_operation(insert_op).result(0)?.into();

                                if let Some((mem_val, mem_ty)) = gen.env.get(base_name).cloned() {
                                    let mem_ty_str = mem_ty.to_string();
                                    if mem_ty_str.starts_with("memref<") {
                                        let store_op =
                                            OperationBuilder::new("memref.store", gen.loc())
                                                .add_operands(&[new_struct_val, mem_val])
                                                .build()
                                                .unwrap();
                                        new_b.append_operation(store_op);
                                    } else {
                                        gen.env.insert(
                                            base_name.to_string().into(),
                                            (new_struct_val, base_ty),
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
                return Ok(Some(new_b));
            } else {
                panic!("Complex struct assignment lhs not supported");
            }
        }
        Ok(Some(block))
    }
}

impl<'c> LowerToMelior<'c> for CompoundAssignStmt {
    type Output = Result<Option<melior::ir::BlockRef<'c, 'c>>, LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let CompoundAssignStmt {
            lhs,
            op,
            rhs,
            span: _,
        } = self;
        let (lhs_val, ty, block) = gen.generate_expr(lhs, block)?;
        let prev_expected = gen.expected_type;
        gen.expected_type = Some(ty);
        let (rhs_val, rhs_ty, block) = gen.generate_expr(rhs, block)?;
        gen.expected_type = prev_expected;

        // Coerce the operand to the accumulator's type before the arithmetic. Now that an
        // annotated `let` types its literal at the annotation (#240), a wider accumulator
        // (`let mut i: i64`) legitimately combines with a default-`i32` literal (`i += 5`); the
        // general `coerce_type` widens/narrows it (and still handles the `index`↔`i32` loop case).
        let mut actual_rhs = rhs_val;
        if rhs_ty != ty {
            actual_rhs = gen.coerce_type(&block, rhs_val, rhs_ty, ty)?;
        }

        let is_float = ty.to_string().contains("f32")
            || ty.to_string().contains("f64")
            || ty.to_string().contains("f16")
            || ty.to_string().contains("bf16");
        let bin_op = OperationBuilder::new(op.get_op_name(is_float), gen.loc())
            .add_operands(&[lhs_val, actual_rhs])
            .add_results(&[ty])
            .build()?;
        let bin_ref = block.append_operation(bin_op);
        let result_val = bin_ref.result(0)?.into();

        if let Expr::Identifier(IdentifierExpr { name, span: _ }) = lhs {
            if let Some((mem_val, mem_ty)) = gen.env.get(name).cloned() {
                let mem_ty_str = mem_ty.to_string();
                if mem_ty_str.starts_with("memref<") {
                    let store_op = OperationBuilder::new("memref.store", gen.loc())
                        .add_operands(&[result_val, mem_val])
                        .build()?;
                    block.append_operation(store_op);
                } else {
                    gen.env.insert(name.to_string().into(), (result_val, ty));
                }
            }
        } else if let Expr::IndexAccess(syntax::IndexAccessExpr {
            base,
            index: _,
            span: _,
        }) = lhs
        {
            if let Some((mem_val, mem_ty, indices, new_b)) = gen.flatten_indices(
                &syntax::Expr::IndexAccess(syntax::IndexAccessExpr {
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
                        .build()?;
                    new_b.append_operation(store_op);
                }
                return Ok(Some(new_b));
            }
        }
        Ok(Some(block))
    }
}

impl<'c> LowerToMelior<'c> for ExprStmtStmt {
    type Output = Result<Option<melior::ir::BlockRef<'c, 'c>>, LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let ExprStmtStmt {
            expr,
            has_semi: _,
            span: _,
        } = self;
        let prev = gen.expected_type;
        gen.expected_type = Some(gen.none_ty);
        let (_, _, updated_block) = gen.generate_expr(expr, block)?;
        gen.expected_type = prev;
        Ok(Some(updated_block))
    }
}
