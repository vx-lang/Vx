//===- lower.rs - Vx Compiler -------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// Lowering logic translating Vx AST into MLIR operations using the Melior API.
//
//===----------------------------------------------------------------------===//

use super::*;
use melior::ir::{
    attribute::{
        ArrayAttribute, Attribute, DenseI32ArrayAttribute, FlatSymbolRefAttribute, FloatAttribute,
        IntegerAttribute, StringAttribute, TypeAttribute,
    },
    operation::OperationBuilder,
    Identifier,
};

use crate::ast;
use crate::codegen;
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
            let const_op = OperationBuilder::new("arith.constant", Location::unknown(gen.context))
                .add_results(&[i1_ty])
                .add_attributes(&[(
                    Identifier::new(gen.context, "value"),
                    IntegerAttribute::new(i1_ty, val).into(),
                )])
                .build()
                .unwrap();
            let const_ref = block.append_operation(const_op);
            return (const_ref.result(0).unwrap().into(), i1_ty);
        }
        if let Some((val, ty)) = gen.env.get(name) {
            let ty_str = ty.to_string();
            if gen.allocs.contains(name) {
                if ty_str.starts_with("memref<") {
                    let inner_ty_str = &ty_str[7..ty_str.len() - 1];
                    let inner_ty = Type::parse(gen.context, inner_ty_str).unwrap_or_else(|| {
                        panic!("failed to parse {:?} for variable {:?}", inner_ty_str, name)
                    });
                    let load_op =
                        OperationBuilder::new("memref.load", Location::unknown(gen.context))
                            .add_operands(&[*val])
                            .add_results(&[inner_ty])
                            .build()
                            .unwrap();
                    let load_ref = block.append_operation(load_op);
                    return (load_ref.result(0).unwrap().into(), inner_ty);
                } else if ty_str.starts_with("!llvm.ptr")
                    || ty_str.starts_with("!llvm.struct")
                    || ty_str.starts_with("i")
                    || ty_str.starts_with("u")
                    || ty_str.starts_with("f")
                {
                    if gen.is_lvalue_context {
                        let ptr_ty = Type::parse(gen.context, "!llvm.ptr").unwrap();
                        return (*val, ptr_ty);
                    }
                    let elem_ty = *ty;
                    let load_op =
                        OperationBuilder::new("llvm.load", Location::unknown(gen.context))
                            .add_operands(&[*val])
                            .add_results(&[elem_ty])
                            .build()
                            .unwrap();
                    let load_ref = block.append_operation(load_op);
                    return (load_ref.result(0).unwrap().into(), elem_ty);
                }
            }
            if ty_str.starts_with("memref<memref<") {
                let inner_ty_str = &ty_str[7..ty_str.len() - 1];
                let inner_ty = Type::parse(gen.context, inner_ty_str).unwrap();
                let load_op = OperationBuilder::new("memref.load", Location::unknown(gen.context))
                    .add_operands(&[*val])
                    .add_results(&[inner_ty])
                    .build()
                    .unwrap();
                let load_ref = block.append_operation(load_op);
                (load_ref.result(0).unwrap().into(), inner_ty)
            } else if ty_str.starts_with("memref<") && !ty_str.contains("x") {
                let inner_ty_str = &ty_str[7..ty_str.len() - 1];
                let inner_ty = Type::parse(gen.context, inner_ty_str).unwrap();
                let load_op = OperationBuilder::new("memref.load", Location::unknown(gen.context))
                    .add_operands(&[*val])
                    .add_results(&[inner_ty])
                    .build()
                    .unwrap();
                let load_ref = block.append_operation(load_op);
                (load_ref.result(0).unwrap().into(), inner_ty)
            } else {
                (*val, *ty)
            }
        } else if gen.functions.contains_key(name) {
            let (ret_ty, arg_tys) = gen.functions.get(name).unwrap();
            let func_ty = melior::ir::r#type::FunctionType::new(gen.context, arg_tys, &[*ret_ty]);
            let const_op = OperationBuilder::new("func.constant", Location::unknown(gen.context))
                .add_attributes(&[(
                    Identifier::new(gen.context, "value"),
                    FlatSymbolRefAttribute::new(gen.context, name).into(),
                )])
                .add_results(&[func_ty.into()])
                .build()
                .unwrap();
            let const_ref = block.append_operation(const_op);

            let ptr_ty = Type::parse(gen.context, "!llvm.ptr").unwrap();
            let cast_op = OperationBuilder::new(
                "builtin.unrealized_conversion_cast",
                Location::unknown(gen.context),
            )
            .add_operands(&[const_ref.result(0).unwrap().into()])
            .add_results(&[ptr_ty])
            .build()
            .unwrap();
            let cast_ref = block.append_operation(cast_op);

            (cast_ref.result(0).unwrap().into(), ptr_ty)
        } else {
            panic!("Undefined variable: {}", name);
        }
    }
}

impl<'c> LowerToMelior<'c> for BorrowExpr {
    type Output = (Value<'c, 'c>, Type<'c>);
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let BorrowExpr { expr, .. } = self;
        if let Expr::Identifier(id) = &**expr {
            if gen.allocs.contains(&id.name) {
                if let Some((val, val_ty)) = gen.env.get(&id.name) {
                    let ptr_ty = Type::parse(gen.context, "!llvm.ptr").unwrap();
                    if val_ty.to_string().starts_with("memref<") {
                        return (*val, *val_ty);
                    } else if *val_ty == ptr_ty {
                        return (*val, ptr_ty);
                    } else {
                        // Cast from val to ptr_ty if necessary? No, just return val_ty.
                        return (*val, *val_ty);
                    }
                }
            }
        }
        let prev_lvalue = gen.is_lvalue_context;
        gen.is_lvalue_context = true;
        let (val, ty) = gen.generate_expr(expr, block);
        gen.is_lvalue_context = prev_lvalue;
        let ptr_ty = Type::parse(gen.context, "!llvm.ptr").unwrap();
        if ty == ptr_ty {
            return (val, ptr_ty);
        }
        if ty.to_string().starts_with("memref<memref<") {
            return (val, ty);
        }
        if ty.to_string().starts_with("memref<") {
            // Allocate a pointer to the memref
            let alloca_op = block.append_operation(
                OperationBuilder::new("memref.alloca", Location::unknown(gen.context))
                    .add_results(&[Type::parse(gen.context, &format!("memref<{}>", ty)).unwrap()])
                    .build()
                    .unwrap(),
            );
            let ptr = alloca_op.result(0).unwrap().into();

            block.append_operation(
                OperationBuilder::new("memref.store", Location::unknown(gen.context))
                    .add_operands(&[val, ptr])
                    .build()
                    .unwrap(),
            );
            return (
                ptr,
                Type::parse(gen.context, &format!("memref<{}>", ty)).unwrap(),
            );
        }

        let i32_ty = Type::parse(gen.context, "i32").unwrap();
        let c1_op = block.append_operation(
            OperationBuilder::new("llvm.mlir.constant", Location::unknown(gen.context))
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
            OperationBuilder::new("llvm.alloca", Location::unknown(gen.context))
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
            OperationBuilder::new("llvm.store", Location::unknown(gen.context))
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

        let global_op = OperationBuilder::new("llvm.mlir.global", Location::unknown(gen.context))
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

        let addressof_op =
            OperationBuilder::new("llvm.mlir.addressof", Location::unknown(gen.context))
                .add_attributes(&[(
                    Identifier::new(gen.context, "global_name"),
                    FlatSymbolRefAttribute::new(gen.context, &str_name).into(),
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
            let dummy_val = OperationBuilder::new("arith.constant", Location::unknown(gen.context))
                .add_attributes(&[(
                    Identifier::new(gen.context, "value"),
                    IntegerAttribute::new(Type::index(gen.context), 0).into(),
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
            return (ptr_val, ptr_ty);
        }

        let load_op = OperationBuilder::new("llvm.load", Location::unknown(gen.context))
            .add_operands(&[ptr_val])
            .add_results(&[inner_ty])
            .build()
            .unwrap();
        let load_ref = block.append_operation(load_op);
        (load_ref.result(0).unwrap().into(), inner_ty)
    }
}

impl<'c> LowerToMelior<'c> for ast::IndexAccessExpr {
    type Output = (Value<'c, 'c>, Type<'c>);
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

            let i64_ty = Type::parse(gen.context, "i64").unwrap();
            let cast_op = OperationBuilder::new("arith.index_cast", Location::unknown(gen.context))
                .add_operands(&[indices[0]])
                .add_results(&[i64_ty])
                .build()
                .unwrap();
            let idx_i64 = block.append_operation(cast_op).result(0).unwrap().into();

            let gep_op =
                OperationBuilder::new("llvm.getelementptr", Location::unknown(gen.context))
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
                let ptr_ty = Type::parse(gen.context, "!llvm.ptr").unwrap();
                return (ptr_val, ptr_ty);
            }

            let load_op = OperationBuilder::new("llvm.load", Location::unknown(gen.context))
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
            let mut load_builder =
                OperationBuilder::new("memref.load", Location::unknown(gen.context))
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
                let m_idx_attr = IntegerAttribute::new(Type::index(gen.context), 0).into();
                let cst_op =
                    OperationBuilder::new("arith.constant", Location::unknown(gen.context))
                        .add_results(&[index_ty])
                        .add_attributes(&[(Identifier::new(gen.context, "value"), m_idx_attr)])
                        .build()
                        .unwrap();
                let idx_val = block.append_operation(cst_op).result(0).unwrap().into();

                let dim_m_op = OperationBuilder::new("memref.dim", Location::unknown(gen.context))
                    .add_operands(&[lhs_val, idx_val])
                    .add_results(&[index_ty])
                    .build()
                    .unwrap();
                alloc_operands.push(block.append_operation(dim_m_op).result(0).unwrap().into());
            }

            if n_str == "?" {
                let n_idx_attr = IntegerAttribute::new(Type::index(gen.context), 1).into();
                let cst_op =
                    OperationBuilder::new("arith.constant", Location::unknown(gen.context))
                        .add_results(&[index_ty])
                        .add_attributes(&[(Identifier::new(gen.context, "value"), n_idx_attr)])
                        .build()
                        .unwrap();
                let idx_val = block.append_operation(cst_op).result(0).unwrap().into();

                let dim_n_op = OperationBuilder::new("memref.dim", Location::unknown(gen.context))
                    .add_operands(&[rhs_val, idx_val])
                    .add_results(&[index_ty])
                    .build()
                    .unwrap();
                alloc_operands.push(block.append_operation(dim_n_op).result(0).unwrap().into());
            }

            // Alloc output buffer
            let alloc_op = OperationBuilder::new("memref.alloc", Location::unknown(gen.context))
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

            let zero_op = OperationBuilder::new("arith.constant", Location::unknown(gen.context))
                .add_results(&[Type::parse(gen.context, el_ty_str).unwrap()])
                .add_attributes(&[(Identifier::new(gen.context, "value"), zero_attr)])
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
            let yield_fill = OperationBuilder::new("linalg.yield", Location::unknown(gen.context))
                .add_operands(&[block_fill.argument(0).unwrap().into()])
                .build()
                .unwrap();
            block_fill.append_operation(yield_fill);
            region_fill.append_block(block_fill);

            let linalg_fill = OperationBuilder::new("linalg.fill", Location::unknown(gen.context))
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

            let mul_op = OperationBuilder::new(mul_op_name, Location::unknown(gen.context))
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

            let add_op = OperationBuilder::new(add_op_name, Location::unknown(gen.context))
                .add_operands(&[block_matmul.argument(2).unwrap().into(), mul_val])
                .add_results(&[Type::parse(gen.context, el_ty_str).unwrap()])
                .build()
                .unwrap();
            let add_val = block_matmul
                .append_operation(add_op)
                .result(0)
                .unwrap()
                .into();

            let yield_matmul =
                OperationBuilder::new("linalg.yield", Location::unknown(gen.context))
                    .add_operands(&[add_val])
                    .build()
                    .unwrap();
            block_matmul.append_operation(yield_matmul);
            region_matmul.append_block(block_matmul);

            let matmul_op = OperationBuilder::new("linalg.matmul", Location::unknown(gen.context))
                .add_operands(&[lhs_val, rhs_val, out_val])
                .add_attributes(&[(
                    Identifier::new(gen.context, "operandSegmentSizes"),
                    DenseI32ArrayAttribute::new(gen.context, &[2, 1]).into(),
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
                    let idx_attr = IntegerAttribute::new(Type::index(gen.context), i as i64).into();
                    let cst_op =
                        OperationBuilder::new("arith.constant", Location::unknown(gen.context))
                            .add_results(&[index_ty])
                            .add_attributes(&[(Identifier::new(gen.context, "value"), idx_attr)])
                            .build()
                            .unwrap();
                    let idx_val = block.append_operation(cst_op).result(0).unwrap().into();

                    let dim_op =
                        OperationBuilder::new("memref.dim", Location::unknown(gen.context))
                            .add_operands(&[lhs_val, idx_val])
                            .add_results(&[index_ty])
                            .build()
                            .unwrap();
                    alloc_operands.push(block.append_operation(dim_op).result(0).unwrap().into());
                }
            }

            // Alloc output buffer
            let alloc_op = OperationBuilder::new("memref.alloc", Location::unknown(gen.context))
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

            let arith_op = OperationBuilder::new(arith_op_name, Location::unknown(gen.context))
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

            let yield_op = OperationBuilder::new("linalg.yield", Location::unknown(gen.context))
                .add_operands(&[arith_val])
                .build()
                .unwrap();

            block_inner.append_operation(yield_op);
            region.append_block(block_inner);

            let linalg_op = OperationBuilder::new(op_name, Location::unknown(gen.context))
                .add_operands(&[lhs_val, rhs_val, out_val])
                .add_attributes(&[(
                    Identifier::new(gen.context, "operandSegmentSizes"),
                    DenseI32ArrayAttribute::new(gen.context, &[2, 1]).into(),
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
                let cast_op =
                    OperationBuilder::new("arith.index_cast", Location::unknown(gen.context))
                        .add_operands(&[rhs_val])
                        .add_results(&[lhs_ty])
                        .build()
                        .unwrap();
                rhs_val = block.append_operation(cast_op).result(0).unwrap().into();
            } else {
                let cast_op =
                    OperationBuilder::new("arith.index_cast", Location::unknown(gen.context))
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

        let mut builder =
            OperationBuilder::new(op.get_op_name(is_float), Location::unknown(gen.context));
        builder = builder.add_operands(&[lhs_val, rhs_val]);

        let ret_ty = if let Some(pred_val) = op.get_predicate(is_float) {
            let i1_ty = Type::parse(gen.context, "i1").unwrap();
            let i64_ty = Type::parse(gen.context, "i64").unwrap();
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

        let mut builder =
            OperationBuilder::new(op.get_op_name(is_float), Location::unknown(gen.context));
        builder = builder.add_operands(&[lhs_val, rhs_val]);

        let ret_ty = if let Some(pred_val) = op.get_predicate(is_float) {
            let i1_ty = Type::parse(gen.context, "i1").unwrap();
            let i64_ty = Type::parse(gen.context, "i64").unwrap();
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

        let builder = OperationBuilder::new(op.get_op_name(false), Location::unknown(gen.context))
            .add_operands(&[lhs_val, rhs_val])
            .add_results(&[final_ty]);

        let bin_op = builder.build().unwrap();
        let bin_ref = block.append_operation(bin_op);
        (bin_ref.result(0).unwrap().into(), final_ty)
    }
}

impl<'c> LowerToMelior<'c> for ast::UnaryOpExpr {
    type Output = (Value<'c, 'c>, Type<'c>);
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let ast::UnaryOpExpr { op, expr, span: _ } = self;
        let (val, ty) = gen.generate_expr(expr, block);
        match op {
            ast::UnaryOp::Not => {
                let true_val_op =
                    OperationBuilder::new("arith.constant", Location::unknown(gen.context))
                        .add_results(&[ty])
                        .add_attributes(&[(
                            Identifier::new(gen.context, "value"),
                            IntegerAttribute::new(ty, 1).into(),
                        )])
                        .build()
                        .unwrap();
                let true_val_ref = block.append_operation(true_val_op);

                let not_op = OperationBuilder::new("arith.xori", Location::unknown(gen.context))
                    .add_operands(&[val, true_val_ref.result(0).unwrap().into()])
                    .add_results(&[ty])
                    .build()
                    .unwrap();

                let not_ref = block.append_operation(not_op);
                (not_ref.result(0).unwrap().into(), ty)
            }
            ast::UnaryOp::Neg => {
                let is_float = ty.to_string().contains("f32")
                    || ty.to_string().contains("f64")
                    || ty.to_string().contains("f16")
                    || ty.to_string().contains("bf16");
                if is_float {
                    let neg_op =
                        OperationBuilder::new("arith.negf", Location::unknown(gen.context))
                            .add_operands(&[val])
                            .add_results(&[ty])
                            .build()
                            .unwrap();
                    let neg_ref = block.append_operation(neg_op);
                    (neg_ref.result(0).unwrap().into(), ty)
                } else {
                    let zero_op =
                        OperationBuilder::new("arith.constant", Location::unknown(gen.context))
                            .add_results(&[ty])
                            .add_attributes(&[(
                                Identifier::new(gen.context, "value"),
                                IntegerAttribute::new(ty, 0).into(),
                            )])
                            .build()
                            .unwrap();
                    let zero_ref = block.append_operation(zero_op);

                    let neg_op =
                        OperationBuilder::new("arith.subi", Location::unknown(gen.context))
                            .add_operands(&[zero_ref.result(0).unwrap().into(), val])
                            .add_results(&[ty])
                            .build()
                            .unwrap();
                    let neg_ref = block.append_operation(neg_op);
                    (neg_ref.result(0).unwrap().into(), ty)
                }
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

        let undef_op = OperationBuilder::new("llvm.mlir.undef", Location::unknown(gen.context))
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
            let (mut field_val, expr_ty) = gen.generate_expr(f_expr, block);
            gen.expected_type = prev_expected;

            if expr_ty != field_ty
                && ((expr_ty.to_string() == "index" && field_ty.to_string() == "i32")
                    || (expr_ty.to_string() == "i32" && field_ty.to_string() == "index"))
            {
                let cast_op =
                    OperationBuilder::new("arith.index_cast", Location::unknown(gen.context))
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

            let insert_op =
                OperationBuilder::new("llvm.insertvalue", Location::unknown(gen.context))
                    .add_operands(&[current_struct, field_val])
                    .add_attributes(&[(Identifier::new(gen.context, "position"), pos_attr.into())])
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
            let zero_attr = IntegerAttribute::new(i32_ty, 0).into();
            let zero_op = OperationBuilder::new("arith.constant", Location::unknown(gen.context))
                .add_results(&[i32_ty])
                .add_attributes(&[(Identifier::new(gen.context, "value"), zero_attr)])
                .build()
                .unwrap();
            let zero_val = block.append_operation(zero_op).result(0).unwrap().into();
            (zero_val, i32_ty)
        }
    }
}

fn topology_to_i32(top: &ast::Topology) -> i32 {
    use ast::Topology::*;
    match top {
        CPU => 0,
        NPU(expr) => {
            if let ast::Expr::Number(n) = &**expr {
                100 + n.value.parse::<i32>().unwrap_or(0)
            } else {
                100
            }
        }
        AccCore(expr) => {
            if let ast::Expr::Number(n) = &**expr {
                200 + n.value.parse::<i32>().unwrap_or(0)
            } else {
                200
            }
        }
        AMX => 300,
        ANE => 400,
        GPU => 500,
        CPU_AVX512 => 600,
        CPU_Neon => 700,
        Slice(_, _, _) => 900,
        Current => 0,
    }
}

impl<'c> LowerToMelior<'c> for ast::SpawnOnExpr {
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
        if let Some(ast::Statement::Return(_)) = self.stmts.last() {
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
        region.append_block(body_block);

        let topology_id = topology_to_i32(&self.top);
        let top_attr =
            IntegerAttribute::new(Type::parse(gen.context, "i32").unwrap(), topology_id as i64)
                .into();

        let mut spawn_builder = OperationBuilder::new("vx.spawn", location)
            .add_attributes(&[(Identifier::new(gen.context, "topology"), top_attr)])
            .add_regions([region]);

        if !result_types.is_empty() {
            spawn_builder = spawn_builder.add_results(&result_types);
        }

        let spawn_op = spawn_builder.build().unwrap();
        let spawn_ref = block.append_operation(spawn_op);

        if !needs_yield {
            let mut ret_builder = OperationBuilder::new("func.return", location);
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
            let dummy_op = OperationBuilder::new("arith.constant", location)
                .add_attributes(&[(
                    Identifier::new(gen.context, "value"),
                    IntegerAttribute::new(Type::index(gen.context), 0).into(),
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

impl<'c> LowerToMelior<'c> for ast::TransferExpr {
    type Output = (Value<'c, 'c>, Type<'c>);
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let (src_val, src_ty) = gen.generate_expr(&self.expr, block);
        let location = melior::ir::Location::unknown(gen.context);

        // Map memory space to topology target.
        let target_topology_id = match self.space {
            ast::MemorySpace::CPUDRAM => 0,
            ast::MemorySpace::NPUHBM => 100,
            ast::MemorySpace::LocalSRAM => 200,
        };

        let top_attr = IntegerAttribute::new(
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

        let transfer_op = OperationBuilder::new("vx.transfer", location)
            .add_operands(&[src_val])
            .add_attributes(&[(Identifier::new(gen.context, "target_topology"), top_attr)])
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
                        let ptr_ty = Type::parse(gen.context, "!llvm.ptr").unwrap();
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

                        let gep_op = OperationBuilder::new(
                            "llvm.getelementptr",
                            Location::unknown(gen.context),
                        )
                        .add_operands(&[base_val])
                        .add_attributes(&[
                            (
                                Identifier::new(gen.context, "rawConstantIndices"),
                                DenseI32ArrayAttribute::new(gen.context, &[0, field_idx as i32])
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
                            return (field_ptr, ptr_ty);
                        }

                        let load_op =
                            OperationBuilder::new("llvm.load", Location::unknown(gen.context))
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

                        let ext_op = OperationBuilder::new(
                            "llvm.extractvalue",
                            Location::unknown(gen.context),
                        )
                        .add_operands(&[base_val])
                        .add_attributes(&[(
                            Identifier::new(gen.context, "position"),
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

fn map_frontend_type_to_mlir(el_ty_str: &str) -> Result<&'static str, String> {
    match el_ty_str {
        "f16" => Ok("f16"),
        "f32" => Ok("f32"),
        "f64" => Ok("f64"),
        "bf16" => Ok("bf16"),
        "i32" => Ok("i32"),
        "i64" => Ok("i64"),
        "Bool" | "i1" => Ok("i1"),
        _ => Err(format!(
            "Unsupported frontend element type for MLIR lowering: {}",
            el_ty_str
        )),
    }
}

fn extract_mlir_element_type(ty_str: &str) -> Result<&'static str, String> {
    let mut inner = ty_str;
    if inner.starts_with("memref<") && inner.ends_with('>') {
        inner = &inner[7..inner.len() - 1];
    } else if inner.starts_with("tensor<") && inner.ends_with('>') {
        inner = &inner[7..inner.len() - 1];
    } else {
        return Err(format!("Unsupported MLIR element type in: {}", ty_str));
    }

    if let Some(idx) = inner.find(',') {
        inner = &inner[..idx];
    }

    if let Some(idx) = inner.rfind('x') {
        inner = &inner[idx + 1..];
    }

    inner = inner.trim();

    match inner {
        "bf16" => Ok("bf16"),
        "f16" => Ok("f16"),
        "f32" => Ok("f32"),
        "f64" => Ok("f64"),
        "i32" => Ok("i32"),
        "i64" => Ok("i64"),
        "i1" => Ok("i1"),
        _ => Err(format!("Unsupported MLIR element type in: {}", ty_str)),
    }
}

impl<'c> LowerToMelior<'c> for FunctionCallExpr {
    type Output = Result<(Value<'c, 'c>, Type<'c>), LowerError>;
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let FunctionCallExpr {
            name,
            args,
            span: _,
        } = self;
        if name == "Verified" {
            return Ok(gen.generate_expr(&args[0], block));
        }
        if (name.starts_with("Tensor<")
            && name.ends_with(">")
            && !name.contains("__")
            && !name.contains("_dim"))
            || name == "Tensor"
        {
            let el_ty_str = if name.starts_with("Tensor<") {
                name.strip_prefix("Tensor<")
                    .unwrap()
                    .strip_suffix(">")
                    .unwrap()
            } else {
                "f32"
            };
            let mlir_ty_str =
                map_frontend_type_to_mlir(el_ty_str).unwrap_or_else(|e| panic!("{}", e));
            let mut dynamic_sizes = Vec::new();
            let mut dims_count = 2; // Default fallback

            if args.len() == 1 {
                if let Expr::Array(arr) = &args[0] {
                    dims_count = arr.elements.len();
                    for el in &arr.elements {
                        let (mut val, ty) = gen.generate_expr(el, block);
                        if ty.to_string() != "index" {
                            let cast_op = OperationBuilder::new(
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
                        let cast_op = OperationBuilder::new(
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

            let alloc_op = OperationBuilder::new("memref.alloc", Location::unknown(gen.context))
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
            let (arg_val, expr_ty) = gen.generate_expr(&args[0], block);
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
            let cast1_op = OperationBuilder::new("memref.cast", Location::unknown(gen.context))
                .add_operands(&[arg_val])
                .add_results(&[unranked_ty])
                .build()
                .unwrap();
            let cast1_ref = block.append_operation(cast1_op);
            let unranked_val = cast1_ref.result(0).unwrap().into();

            let target_ty = Type::parse(gen.context, &target_ty_str).unwrap();

            // Cast to targeted shape
            let cast2_op = OperationBuilder::new("memref.cast", Location::unknown(gen.context))
                .add_operands(&[unranked_val])
                .add_results(&[target_ty])
                .build()
                .unwrap();

            let cast2_ref = block.append_operation(cast2_op);
            return Ok((cast2_ref.result(0).unwrap().into(), target_ty));
        }

        if name == "with_memory" {
            // For now, with_memory is a no-op in lowering, just returns the tensor
            let (arg_val, expr_ty) = gen.generate_expr(&args[0], block);
            return Ok((arg_val, expr_ty));
        }

        if name == "map" {
            return Ok(lower_map_call(gen, block, args));
        }

        if name == "print" {
            return lower_print_call(gen, block, args);
        }

        if name == "printf" || name == "vx_internal_printf" {
            let mut arg_vals = Vec::new();
            for arg in args {
                let (arg_val, _arg_ty) = gen.generate_expr(arg, block);
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
                let printf_decl =
                    OperationBuilder::new("llvm.func", Location::unknown(gen.context))
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
                    (
                        Type::parse(gen.context, "i32").unwrap(),
                        vec![Type::parse(gen.context, "!llvm.ptr").unwrap()],
                    ),
                );
            }

            let call_op = block.append_operation(
                OperationBuilder::new("llvm.call", Location::unknown(gen.context))
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
                    .add_results(&[Type::parse(gen.context, "i32").unwrap()])
                    .build()
                    .unwrap(),
            );

            return Ok((
                call_op.result(0).unwrap().into(),
                Type::parse(gen.context, "i32").unwrap(),
            ));
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
                        let cast_op =
                            OperationBuilder::new("memref.cast", Location::unknown(gen.context))
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
            let mut builder = OperationBuilder::new("func.call", Location::unknown(gen.context))
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
                let none_ty = Type::parse(gen.context, "none").unwrap();
                // this value shouldn't be used
                let dummy_op =
                    OperationBuilder::new("llvm.mlir.constant", Location::unknown(gen.context))
                        .add_results(&[Type::parse(gen.context, "i32").unwrap()])
                        .add_attributes(&[(
                            Identifier::new(gen.context, "value"),
                            IntegerAttribute::new(Type::parse(gen.context, "i32").unwrap(), 0)
                                .into(),
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
                    let r = gen.lower_type(ret);
                    let a: Vec<_> = func_args.iter().map(|t| gen.lower_type(t)).collect();
                    actual_func_ty =
                        melior::ir::r#type::FunctionType::new(gen.context, &a, &[r]).into();
                } else {
                    panic!("Missing signature for function pointer '{}'", name);
                }
            } else if is_closure {
                if let Some(ast::Type::Closure(func_args, ret)) = gen.ast_env.get(name) {
                    let r = gen.lower_type(ret);
                    let mut a: Vec<_> = vec![Type::parse(gen.context, "!llvm.ptr").unwrap()];
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
                    let extract_env_op =
                        OperationBuilder::new("llvm.extractvalue", Location::unknown(gen.context))
                            .add_operands(&[ptr_val])
                            .add_attributes(&[(
                                Identifier::new(gen.context, "position"),
                                melior::ir::attribute::DenseI64ArrayAttribute::new(
                                    gen.context,
                                    &[0],
                                )
                                .into(),
                            )])
                            .add_results(&[Type::parse(gen.context, "!llvm.ptr").unwrap()])
                            .build()
                            .unwrap();
                    let extract_env_ref = block.append_operation(extract_env_op);
                    env_ptr = Some(extract_env_ref.result(0).unwrap().into());

                    // Extract func_ptr
                    let extract_func_op =
                        OperationBuilder::new("llvm.extractvalue", Location::unknown(gen.context))
                            .add_operands(&[ptr_val])
                            .add_attributes(&[(
                                Identifier::new(gen.context, "position"),
                                melior::ir::attribute::DenseI64ArrayAttribute::new(
                                    gen.context,
                                    &[1],
                                )
                                .into(),
                            )])
                            .add_results(&[Type::parse(gen.context, "!llvm.ptr").unwrap()])
                            .build()
                            .unwrap();
                    let extract_func_ref = block.append_operation(extract_func_op);
                    actual_ptr_val = extract_func_ref.result(0).unwrap().into();
                }

                for (i, arg) in args.iter().enumerate() {
                    let (mut arg_val, expr_ty) = gen.generate_expr(arg, block);
                    let field_ty = mlir_func_ty.input(i + arg_offset).unwrap();
                    if expr_ty != field_ty {
                        arg_val = gen.coerce_type(block, arg_val, expr_ty, field_ty);
                    }
                    arg_vals.push(arg_val);
                }

                if func_ty.to_string() == "!llvm.ptr" || is_closure {
                    let cast_op = OperationBuilder::new(
                        "builtin.unrealized_conversion_cast",
                        Location::unknown(gen.context),
                    )
                    .add_operands(&[actual_ptr_val])
                    .add_results(&[actual_func_ty])
                    .build()
                    .unwrap();
                    let cast_ref = block.append_operation(cast_op);
                    actual_ptr_val = cast_ref.result(0).unwrap().into();
                }

                let mut builder =
                    OperationBuilder::new("func.call_indirect", Location::unknown(gen.context))
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
                    let none_ty = Type::parse(gen.context, "none").unwrap();
                    let dummy_op =
                        OperationBuilder::new("llvm.mlir.constant", Location::unknown(gen.context))
                            .add_results(&[Type::parse(gen.context, "i32").unwrap()])
                            .add_attributes(&[(
                                Identifier::new(gen.context, "value"),
                                IntegerAttribute::new(Type::parse(gen.context, "i32").unwrap(), 0)
                                    .into(),
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
    type Output = (Value<'c, 'c>, Type<'c>);
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let IndirectCallExpr {
            callee,
            args,
            target_func_ty,
            span: _,
        } = self;

        let (callee_val, callee_ty) = gen.generate_expr(callee, block);

        if callee_ty.to_string() == "!llvm.struct<(ptr, ptr)>" {
            // Extract env_ptr
            let extract_env_op =
                OperationBuilder::new("llvm.extractvalue", Location::unknown(gen.context))
                    .add_operands(&[callee_val])
                    .add_attributes(&[(
                        Identifier::new(gen.context, "position"),
                        melior::ir::attribute::DenseI64ArrayAttribute::new(gen.context, &[0])
                            .into(),
                    )])
                    .add_results(&[Type::parse(gen.context, "!llvm.ptr").unwrap()])
                    .build()
                    .unwrap();
            let extract_env_ref = block.append_operation(extract_env_op);
            let env_ptr = extract_env_ref.result(0).unwrap().into();

            // Extract func_ptr
            let extract_func_op =
                OperationBuilder::new("llvm.extractvalue", Location::unknown(gen.context))
                    .add_operands(&[callee_val])
                    .add_attributes(&[(
                        Identifier::new(gen.context, "position"),
                        melior::ir::attribute::DenseI64ArrayAttribute::new(gen.context, &[1])
                            .into(),
                    )])
                    .add_results(&[Type::parse(gen.context, "!llvm.ptr").unwrap()])
                    .build()
                    .unwrap();
            let extract_func_ref = block.append_operation(extract_func_op);
            let mut actual_ptr_val = extract_func_ref.result(0).unwrap().into();

            let mut arg_vals = Vec::new();
            for arg in args {
                let (arg_val, _) = gen.generate_expr(arg, block);
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
            let mut a: Vec<_> = vec![Type::parse(gen.context, "!llvm.ptr").unwrap()];
            a.extend(func_args.iter().map(|t| gen.lower_type(t)));
            let actual_mlir_func_ty = melior::ir::r#type::FunctionType::new(gen.context, &a, &[r]);
            let actual_func_ty: melior::ir::Type = actual_mlir_func_ty.into();

            let ret_ty = actual_mlir_func_ty.result(0).unwrap();

            // Cast the raw func ptr to the actual function signature
            let cast_op = OperationBuilder::new(
                "builtin.unrealized_conversion_cast",
                Location::unknown(gen.context),
            )
            .add_operands(&[actual_ptr_val])
            .add_results(&[actual_func_ty])
            .build()
            .unwrap();
            let cast_ref = block.append_operation(cast_op);
            actual_ptr_val = cast_ref.result(0).unwrap().into();

            let mut builder =
                OperationBuilder::new("func.call_indirect", Location::unknown(gen.context))
                    .add_operands(&[actual_ptr_val, env_ptr]);

            for a_val in &arg_vals {
                builder = builder.add_operands(&[*a_val]);
            }

            if ret_ty.to_string() != "none" {
                builder = builder.add_results(&[ret_ty]);
                let call_op = builder.build().unwrap();
                let call_ref = block.append_operation(call_op);
                (call_ref.result(0).unwrap().into(), ret_ty)
            } else {
                let call_op = builder.build().unwrap();
                block.append_operation(call_op);
                let none_ty = Type::parse(gen.context, "none").unwrap();
                let dummy_op =
                    OperationBuilder::new("llvm.mlir.constant", Location::unknown(gen.context))
                        .add_results(&[Type::parse(gen.context, "i32").unwrap()])
                        .add_attributes(&[(
                            Identifier::new(gen.context, "value"),
                            IntegerAttribute::new(Type::parse(gen.context, "i32").unwrap(), 0)
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
            panic!("Unsupported callee type for indirect call: {}", callee_ty);
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

impl<'c> LowerToMelior<'c> for InlineMlirExpr {
    type Output = (Value<'c, 'c>, Type<'c>);

    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let mut mlir_args = Vec::new();
        let mut input_types_str = Vec::new();

        for (name, expr, ty_str) in &self.inputs {
            let (mut val, val_ty) = gen.generate_expr(expr, block);
            if val_ty.to_string().starts_with("memref<memref<") && ty_str.starts_with("memref<") {
                let load_op = OperationBuilder::new("memref.load", Location::unknown(gen.context))
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

        let call_op =
            OperationBuilder::new("func.call", melior::ir::Location::unknown(gen.context))
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
            (op.result(0).unwrap().into(), gen.lower_type(ret_ty))
        } else {
            let dummy_val =
                OperationBuilder::new("arith.constant", melior::ir::Location::unknown(gen.context))
                    .add_attributes(&[(
                        Identifier::new(gen.context, "value"),
                        IntegerAttribute::new(Type::index(gen.context), 0).into(),
                    )])
                    .add_results(&[Type::index(gen.context)])
                    .build()
                    .unwrap();
            (
                block.append_operation(dummy_val).result(0).unwrap().into(),
                Type::index(gen.context),
            )
        }
    }
}

impl<'c> LowerToMelior<'c> for ArrayExpr {
    type Output = (Value<'c, 'c>, Type<'c>);
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let ArrayExpr { elements, span: _ } = self;
        if elements.is_empty() {
            panic!("Empty arrays not supported yet");
        }
        let mut vals = Vec::new();
        let mut el_ty = None;
        for el in elements {
            let (v, t) = gen.generate_expr(el, block);
            vals.push(v);
            if el_ty.is_none() {
                el_ty = Some(t);
            }
        }
        let el_ty = el_ty.unwrap();
        let num_elements = elements.len();

        let tensor_ty =
            Type::parse(gen.context, &format!("tensor<{}x{}>", num_elements, el_ty)).unwrap();

        let op = OperationBuilder::new("tensor.from_elements", Location::unknown(gen.context))
            .add_operands(&vals)
            .add_results(&[tensor_ty])
            .build()
            .unwrap();

        let val = block.append_operation(op).result(0).unwrap().into();
        (val, tensor_ty)
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
                            let (val, ty) = gen.generate_expr(expr, block);
                            if !has_semi {
                                last_val = Some((val, ty));
                            }
                        } else {
                            gen.generate_statement(stmt, block);
                        }
                    } else {
                        gen.generate_statement(stmt, block);
                    }
                }
            }

            if let Some((val, ty)) = last_val {
                return (val, ty);
            }

            let ret_ty = gen
                .expected_type
                .unwrap_or_else(|| Type::parse(gen.context, "f32").unwrap());
            let dummy_op = if ret_ty.to_string() == "f32" {
                OperationBuilder::new("arith.constant", Location::unknown(gen.context))
                    .add_attributes(&[(
                        Identifier::new(gen.context, "value"),
                        FloatAttribute::new(gen.context, ret_ty, 0.0).into(),
                    )])
                    .add_results(&[ret_ty])
                    .build()
                    .unwrap()
            } else if ret_ty.to_string() == "i1" {
                OperationBuilder::new("arith.constant", Location::unknown(gen.context))
                    .add_attributes(&[(
                        Identifier::new(gen.context, "value"),
                        IntegerAttribute::new(ret_ty, 0).into(),
                    )])
                    .add_results(&[ret_ty])
                    .build()
                    .unwrap()
            } else {
                OperationBuilder::new("arith.constant", Location::unknown(gen.context))
                    .add_attributes(&[(
                        Identifier::new(gen.context, "value"),
                        IntegerAttribute::new(ret_ty, 0).into(),
                    )])
                    .add_results(&[ret_ty])
                    .build()
                    .unwrap()
            };
            return (
                block.append_operation(dummy_op).result(0).unwrap().into(),
                ret_ty,
            );
        }

        let (cond_val, _) = gen.generate_expr(cond, block);

        let then_region = Region::new();
        let then_b = Block::new(&[]);
        for stmt in then_block {
            gen.generate_statement(stmt, &then_b);
        }
        let yield_op = OperationBuilder::new("scf.yield", Location::unknown(gen.context))
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
        let yield_op = OperationBuilder::new("scf.yield", Location::unknown(gen.context))
            .build()
            .unwrap();
        else_b.append_operation(yield_op);
        else_region.append_block(else_b);

        let if_op = OperationBuilder::new("scf.if", Location::unknown(gen.context))
            .add_operands(&[cond_val])
            .add_regions([then_region, else_region])
            .build()
            .unwrap();

        block.append_operation(if_op);

        let ty = Type::parse(gen.context, "i32").unwrap();
        let op = OperationBuilder::new("arith.constant", Location::unknown(gen.context))
            .add_results(&[ty])
            .add_attributes(&[(
                Identifier::new(gen.context, "value"),
                IntegerAttribute::new(ty, 0).into(),
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
            gen.lower_type(&ast::Type::Scalar(ast_ty.clone()))
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
            let op = OperationBuilder::new("arith.constant", Location::unknown(gen.context))
                .add_results(&[ty])
                .add_attributes(&[(
                    Identifier::new(gen.context, "value"),
                    FloatAttribute::new(gen.context, ty, val_str.parse::<f64>().unwrap()).into(),
                )])
                .build()
                .unwrap();
            let op_ref = block.append_operation(op);
            (op_ref.result(0).unwrap().into(), ty)
        } else {
            let op = OperationBuilder::new("arith.constant", Location::unknown(gen.context))
                .add_results(&[ty])
                .add_attributes(&[(
                    Identifier::new(gen.context, "value"),
                    IntegerAttribute::new(ty, val_str.parse::<i64>().unwrap()).into(),
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
        gen.expected_type = gen.current_return_type;
        let (mut val, expr_ty) = gen.generate_expr(expr, block);
        gen.expected_type = None;
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

                    let cast_op =
                        OperationBuilder::new(cast_op_name, Location::unknown(gen.context))
                            .add_operands(&[val])
                            .add_results(&[ret_ty])
                            .build()
                            .unwrap();
                    val = block.append_operation(cast_op).result(0).unwrap().into();
                } else if ret_ty.to_string() == "i32" && expr_ty.to_string().starts_with("memref<")
                {
                    let zero_op =
                        OperationBuilder::new("arith.constant", Location::unknown(gen.context))
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
        let ret_op = OperationBuilder::new(op_name, Location::unknown(gen.context))
            .add_operands(&[val])
            .build()
            .unwrap();
        block.append_operation(ret_op);
        gen.has_returned = true;
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
                let ptr_ty = Type::parse(gen.context, "!llvm.ptr").unwrap();
                let i32_ty = Type::parse(gen.context, "i32").unwrap();
                let one_attr = IntegerAttribute::new(i32_ty, 1).into();
                let const_op =
                    OperationBuilder::new("llvm.mlir.constant", Location::unknown(gen.context))
                        .add_results(&[i32_ty])
                        .add_attributes(&[(Identifier::new(gen.context, "value"), one_attr)])
                        .build()
                        .unwrap();
                let one_val = block.append_operation(const_op).result(0).unwrap().into();

                let alloca_op =
                    OperationBuilder::new("llvm.alloca", Location::unknown(gen.context))
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

                let store_op = OperationBuilder::new("llvm.store", Location::unknown(gen.context))
                    .add_operands(&[val, alloca_val])
                    .build()
                    .unwrap();
                block.append_operation(store_op);

                gen.env.insert(name.clone(), (alloca_val, ty));
                gen.allocs.insert(name.clone());
            } else {
                let memref_ty = format!("memref<{}>", ty);
                let parsed_memref_ty = Type::parse(gen.context, &memref_ty).unwrap();
                let alloca_op =
                    OperationBuilder::new("memref.alloca", Location::unknown(gen.context))
                        .add_results(&[parsed_memref_ty])
                        .build()
                        .unwrap();
                let alloca_ref = block.append_operation(alloca_op);
                let alloca_val = alloca_ref.result(0).unwrap().into();

                let store_op =
                    OperationBuilder::new("memref.store", Location::unknown(gen.context))
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
                        let cast_op = OperationBuilder::new(
                            "arith.index_cast",
                            Location::unknown(gen.context),
                        )
                        .add_operands(&[rhs_val])
                        .add_results(&[inner_ty])
                        .build()
                        .unwrap();

                        store_val = block.append_operation(cast_op).result(0).unwrap().into();
                    }

                    let store_op =
                        OperationBuilder::new("memref.store", Location::unknown(gen.context))
                            .add_operands(&[store_val, mem_val])
                            .build()
                            .unwrap();
                    block.append_operation(store_op);
                } else if gen.allocs.contains(name) {
                    let store_op =
                        OperationBuilder::new("llvm.store", Location::unknown(gen.context))
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
                    let i64_ty = Type::parse(gen.context, "i64").unwrap();
                    let cast_op =
                        OperationBuilder::new("arith.index_cast", Location::unknown(gen.context))
                            .add_operands(&[indices[0]])
                            .add_results(&[i64_ty])
                            .build()
                            .unwrap();
                    let idx_i64 = block.append_operation(cast_op).result(0).unwrap().into();

                    let gep_op =
                        OperationBuilder::new("llvm.getelementptr", Location::unknown(gen.context))
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

                    let store_op =
                        OperationBuilder::new("llvm.store", Location::unknown(gen.context))
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

                    let mut store_builder =
                        OperationBuilder::new("memref.store", Location::unknown(gen.context))
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
                let (base_val, base_ty) = gen.generate_expr(base, block);
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
                                let cast_op = OperationBuilder::new(
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
                                let struct_llvm_ty =
                                    Type::parse(gen.context, &struct_llvm_ty_str).unwrap();

                                let gep_op = OperationBuilder::new(
                                    "llvm.getelementptr",
                                    Location::unknown(gen.context),
                                )
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

                                let store_op = OperationBuilder::new(
                                    "llvm.store",
                                    Location::unknown(gen.context),
                                )
                                .add_operands(&[field_val, ptr_val])
                                .build()
                                .unwrap();
                                block.append_operation(store_op);
                            } else {
                                let pos_attr = melior::ir::attribute::DenseI64ArrayAttribute::new(
                                    gen.context,
                                    &[field_idx as i64],
                                );
                                let insert_op = OperationBuilder::new(
                                    "llvm.insertvalue",
                                    Location::unknown(gen.context),
                                )
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
                                        let store_op = OperationBuilder::new(
                                            "memref.store",
                                            Location::unknown(gen.context),
                                        )
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
            let cast_op = OperationBuilder::new("arith.index_cast", Location::unknown(gen.context))
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
        let bin_op =
            OperationBuilder::new(op.get_op_name(is_float), Location::unknown(gen.context))
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
                    let store_op =
                        OperationBuilder::new("memref.store", Location::unknown(gen.context))
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
                    let store_op =
                        OperationBuilder::new("memref.store", Location::unknown(gen.context))
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
            let (start_val, start_ty) = gen.generate_expr(start, block);
            let (end_val, end_ty) = gen.generate_expr(end, block);

            let ty_index = Type::parse(gen.context, "index").unwrap();

            // cast start/end to index if necessary
            let start_idx = if start_ty == ty_index {
                start_val
            } else {
                let cast_start_op =
                    OperationBuilder::new("arith.index_cast", Location::unknown(gen.context))
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
                let cast_end_op =
                    OperationBuilder::new("arith.index_cast", Location::unknown(gen.context))
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
                OperationBuilder::new("arith.constant", Location::unknown(gen.context))
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
            let for_block = Block::new(&[(ty_index, Location::unknown(gen.context))]);
            let i_arg = for_block.argument(0).unwrap().into();

            gen.env.insert(iter.clone(), (i_arg, ty_index));

            for stmt in body {
                gen.generate_statement(stmt, &for_block);
            }

            for_block.append_operation(
                OperationBuilder::new("scf.yield", Location::unknown(gen.context))
                    .build()
                    .unwrap(),
            );
            for_region.append_block(for_block);

            block.append_operation(
                OperationBuilder::new("scf.for", Location::unknown(gen.context))
                    .add_operands(&[start_idx, end_idx, step_idx])
                    .add_regions([for_region])
                    .build()
                    .unwrap(),
            );
            return;
        }

        // Generic Iterator loop via scf.while or tensor iteration
        let (iter_val, iter_ty) = gen.generate_expr(iterable, block);
        let iter_ty_str = iter_ty.to_string();

        if iter_ty_str.starts_with("tensor<") {
            let ty_index = Type::parse(gen.context, "index").unwrap();
            let start_idx = block
                .append_operation(
                    OperationBuilder::new("arith.constant", Location::unknown(gen.context))
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
                    OperationBuilder::new("arith.constant", Location::unknown(gen.context))
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
                    OperationBuilder::new("arith.constant", Location::unknown(gen.context))
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
            let for_block = Block::new(&[(ty_index, Location::unknown(gen.context))]);
            let i_arg = for_block.argument(0).unwrap().into();

            let el_ty_str = iter_ty_str.split('x').nth(1).unwrap().trim_end_matches('>');
            let el_ty = Type::parse(gen.context, el_ty_str).unwrap();

            let extract_op =
                OperationBuilder::new("tensor.extract", Location::unknown(gen.context))
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
                gen.generate_statement(stmt, &for_block);
            }

            for_block.append_operation(
                OperationBuilder::new("scf.yield", Location::unknown(gen.context))
                    .build()
                    .unwrap(),
            );
            for_region.append_block(for_block);

            block.append_operation(
                OperationBuilder::new("scf.for", Location::unknown(gen.context))
                    .add_operands(&[start_idx, end_idx, step_idx])
                    .add_regions([for_region])
                    .build()
                    .unwrap(),
            );
            return;
        }

        let before_region = Region::new();
        let before_block = Block::new(&[(iter_ty, Location::unknown(gen.context))]);
        let iter_arg = before_block.argument(0).unwrap().into();

        let iter_name = format!("__iter_{}", gen.string_counter);
        gen.string_counter += 1;
        gen.env.insert(iter_name.clone(), (iter_arg, iter_ty));

        let ptr_ty = Type::parse(gen.context, "!llvm.ptr").unwrap();
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

        let i32_ty = Type::parse(gen.context, "i32").unwrap();
        let c1_op_alloc = before_block.append_operation(
            OperationBuilder::new("llvm.mlir.constant", Location::unknown(gen.context))
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
            OperationBuilder::new("llvm.alloca", Location::unknown(gen.context))
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
            OperationBuilder::new("llvm.store", Location::unknown(gen.context))
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

        let (opt_val, opt_ty) = gen.generate_expr(&next_call, &before_block);

        // Load the updated iterator value to yield it back
        let load_op = before_block.append_operation(
            OperationBuilder::new("llvm.load", Location::unknown(gen.context))
                .add_operands(&[ptr_val])
                .add_results(&[iter_ty])
                .build()
                .unwrap(),
        );
        let updated_iter_val = load_op.result(0).unwrap().into();

        let extract_tag_op =
            OperationBuilder::new("llvm.extractvalue", Location::unknown(gen.context))
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
            OperationBuilder::new("arith.constant", Location::unknown(gen.context))
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
            OperationBuilder::new("arith.cmpi", Location::unknown(gen.context))
                .add_operands(&[tag_val, c1_val])
                .add_results(&[Type::parse(gen.context, "i1").unwrap()])
                .add_attributes(&[(
                    Identifier::new(gen.context, "predicate"),
                    IntegerAttribute::new(
                        Type::parse(gen.context, "i64").unwrap(),
                        0, // eq
                    )
                    .into(),
                )])
                .build()
                .unwrap(),
        );
        let cond_val = cmpi_op.result(0).unwrap().into();

        let _condition_op = before_block.append_operation(
            OperationBuilder::new("scf.condition", Location::unknown(gen.context))
                .add_operands(&[cond_val, opt_val, updated_iter_val])
                .build()
                .unwrap(),
        );
        before_region.append_block(before_block);

        let after_region = Region::new();
        let after_block = Block::new(&[
            (opt_ty, Location::unknown(gen.context)),
            (iter_ty, Location::unknown(gen.context)),
        ]);
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

        let extract_payload_op =
            OperationBuilder::new("llvm.extractvalue", Location::unknown(gen.context))
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
            gen.generate_statement(stmt, &after_block);
        }
        gen.break_flags.pop();

        after_block.append_operation(
            OperationBuilder::new("scf.yield", Location::unknown(gen.context))
                .add_operands(&[next_iter_arg])
                .build()
                .unwrap(),
        );
        after_region.append_block(after_block);

        block.append_operation(
            OperationBuilder::new("scf.while", Location::unknown(gen.context))
                .add_operands(&[iter_val])
                .add_results(&[opt_ty, iter_ty])
                .add_regions([before_region, after_region])
                .build()
                .unwrap(),
        );
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
        let _name_attr = StringAttribute::new(gen.context, &enzyme_name);
        let _type_attr = TypeAttribute::new(func_type.into());

        let region = melior::ir::Region::new();
        let func_op = OperationBuilder::new("func.func", Location::unknown(gen.context))
            .add_attributes(&[
                (
                    Identifier::new(gen.context, "sym_name"),
                    StringAttribute::new(gen.context, &enzyme_name).into(),
                ),
                (
                    Identifier::new(gen.context, "function_type"),
                    TypeAttribute::new(func_type.into()).into(),
                ),
                (
                    Identifier::new(gen.context, "sym_visibility"),
                    StringAttribute::new(gen.context, "private").into(),
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
        let const_op = OperationBuilder::new("func.constant", Location::unknown(gen.context))
            .add_attributes(&[(
                Identifier::new(gen.context, "value"),
                FlatSymbolRefAttribute::new(gen.context, target_fn).into(),
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

        let name_attr = FlatSymbolRefAttribute::new(gen.context, &enzyme_name);
        let call_op = OperationBuilder::new("func.call", Location::unknown(gen.context))
            .add_operands(&arg_vals)
            .add_results(&[ret_ty])
            .add_attributes(&[(Identifier::new(gen.context, "callee"), name_attr.into())])
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
        let const_op = OperationBuilder::new("func.constant", Location::unknown(gen.context))
            .add_attributes(&[(
                Identifier::new(gen.context, "value"),
                FlatSymbolRefAttribute::new(gen.context, target_fn).into(),
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

        let name_attr = FlatSymbolRefAttribute::new(gen.context, &enzyme_name);
        let call_op = OperationBuilder::new("func.call", Location::unknown(gen.context))
            .add_operands(&arg_vals)
            .add_results(&[ret_ty])
            .add_attributes(&[(Identifier::new(gen.context, "callee"), name_attr.into())])
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
        let mul_op = OperationBuilder::new(op_name, Location::unknown(gen.context))
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
        let const_op = OperationBuilder::new("func.constant", Location::unknown(gen.context))
            .add_attributes(&[(
                Identifier::new(gen.context, "value"),
                FlatSymbolRefAttribute::new(gen.context, target_fn).into(),
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

        let name_attr = FlatSymbolRefAttribute::new(gen.context, &enzyme_name);
        let call_op = OperationBuilder::new("func.call", Location::unknown(gen.context))
            .add_operands(&arg_vals)
            .add_results(&[ret_ty])
            .add_attributes(&[(Identifier::new(gen.context, "callee"), name_attr.into())])
            .build()
            .unwrap();

        let call_ref = block.append_operation(call_op);
        (call_ref.result(0).unwrap().into(), ret_ty)
    }
}

use codegen::break_utils::*;

impl<'c> LowerToMelior<'c> for LoopStmt {
    type Output = ();
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let i1_ty = Type::parse(gen.context, "i1").unwrap();
        let memref_ty = Type::parse(gen.context, "memref<1xi1>").unwrap();

        let alloca_op = block.append_operation(
            OperationBuilder::new("memref.alloca", Location::unknown(gen.context))
                .add_results(&[memref_ty])
                .build()
                .unwrap(),
        );
        let break_ptr = alloca_op.result(0).unwrap().into();

        let continue_alloca = block.append_operation(
            OperationBuilder::new("memref.alloca", Location::unknown(gen.context))
                .add_results(&[memref_ty])
                .build()
                .unwrap(),
        );
        let continue_ptr = continue_alloca.result(0).unwrap().into();

        let false_op = block.append_operation(
            OperationBuilder::new("arith.constant", Location::unknown(gen.context))
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
            OperationBuilder::new("arith.constant", Location::unknown(gen.context))
                .add_results(&[Type::parse(gen.context, "index").unwrap()])
                .add_attributes(&[(
                    Identifier::new(gen.context, "value"),
                    IntegerAttribute::new(Type::parse(gen.context, "index").unwrap(), 0).into(),
                )])
                .build()
                .unwrap(),
        );
        let c0_idx = c0_op.result(0).unwrap().into();

        block.append_operation(
            OperationBuilder::new("memref.store", Location::unknown(gen.context))
                .add_operands(&[false_val, break_ptr, c0_idx])
                .build()
                .unwrap(),
        );

        gen.break_flags.push(break_ptr);
        gen.continue_flags.push(continue_ptr);

        let before_region = Region::new();
        let before_block = Block::new(&[]);

        before_block.append_operation(
            OperationBuilder::new("memref.store", Location::unknown(gen.context))
                .add_operands(&[false_val, continue_ptr, c0_idx])
                .build()
                .unwrap(),
        );

        let load_op = before_block.append_operation(
            OperationBuilder::new("memref.load", Location::unknown(gen.context))
                .add_operands(&[break_ptr, c0_idx])
                .add_results(&[i1_ty])
                .build()
                .unwrap(),
        );
        let is_break = load_op.result(0).unwrap().into();

        let true_op = before_block.append_operation(
            OperationBuilder::new("arith.constant", Location::unknown(gen.context))
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
            OperationBuilder::new("arith.xori", Location::unknown(gen.context))
                .add_operands(&[is_break, true_val])
                .add_results(&[i1_ty])
                .build()
                .unwrap(),
        );
        let not_break = not_break_op.result(0).unwrap().into();

        before_block.append_operation(
            OperationBuilder::new("scf.condition", Location::unknown(gen.context))
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
            OperationBuilder::new("scf.yield", Location::unknown(gen.context))
                .build()
                .unwrap(),
        );
        after_region.append_block(after_block);

        block.append_operation(
            OperationBuilder::new("scf.while", Location::unknown(gen.context))
                .add_regions([before_region, after_region])
                .build()
                .unwrap(),
        );

        gen.continue_flags.pop();
        gen.break_flags.pop();
    }
}

impl<'c> LowerToMelior<'c> for BreakStmt {
    type Output = ();
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        if let Some(&break_ptr) = gen.break_flags.last() {
            let i1_ty = Type::parse(gen.context, "i1").unwrap();
            let true_op = block.append_operation(
                OperationBuilder::new("arith.constant", Location::unknown(gen.context))
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
                OperationBuilder::new("arith.constant", Location::unknown(gen.context))
                    .add_results(&[Type::parse(gen.context, "index").unwrap()])
                    .add_attributes(&[(
                        Identifier::new(gen.context, "value"),
                        IntegerAttribute::new(Type::parse(gen.context, "index").unwrap(), 0).into(),
                    )])
                    .build()
                    .unwrap(),
            );
            let c0_idx = c0_op.result(0).unwrap().into();

            block.append_operation(
                OperationBuilder::new("memref.store", Location::unknown(gen.context))
                    .add_operands(&[true_val, break_ptr, c0_idx])
                    .build()
                    .unwrap(),
            );
        } else {
            panic!("break outside of a loop");
        }
    }
}

impl<'c> LowerToMelior<'c> for ContinueStmt {
    type Output = ();
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        if let Some(&continue_ptr) = gen.continue_flags.last() {
            let i1_ty = Type::parse(gen.context, "i1").unwrap();
            let true_op = block.append_operation(
                OperationBuilder::new("arith.constant", Location::unknown(gen.context))
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
                OperationBuilder::new("arith.constant", Location::unknown(gen.context))
                    .add_results(&[Type::parse(gen.context, "index").unwrap()])
                    .add_attributes(&[(
                        Identifier::new(gen.context, "value"),
                        IntegerAttribute::new(Type::parse(gen.context, "index").unwrap(), 0).into(),
                    )])
                    .build()
                    .unwrap(),
            );
            let c0_idx = c0_op.result(0).unwrap().into();

            block.append_operation(
                OperationBuilder::new("memref.store", Location::unknown(gen.context))
                    .add_operands(&[true_val, continue_ptr, c0_idx])
                    .build()
                    .unwrap(),
            );
        } else {
            panic!("continue outside of a loop");
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
                OperationBuilder::new("arith.constant", melior::ir::Location::unknown(gen.context))
                    .add_results(&[i32_ty])
                    .add_attributes(&[(
                        Identifier::new(gen.context, "value"),
                        IntegerAttribute::new(i32_ty, tag_val).into(),
                    )])
                    .build()
                    .unwrap(),
            );
            let tag = tag_op.result(0).unwrap().into();

            let actual_tag = if _match_ty.to_string() == "i32" {
                match_val
            } else {
                let extract_tag_op = block.append_operation(
                    OperationBuilder::new(
                        "llvm.extractvalue",
                        melior::ir::Location::unknown(gen.context),
                    )
                    .add_operands(&[match_val])
                    .add_results(&[i32_ty])
                    .add_attributes(&[(
                        Identifier::new(gen.context, "position"),
                        melior::ir::attribute::DenseI64ArrayAttribute::new(gen.context, &[0])
                            .into(),
                    )])
                    .build()
                    .unwrap(),
                );
                extract_tag_op.result(0).unwrap().into()
            };

            let cmp_op = block.append_operation(
                OperationBuilder::new("arith.cmpi", melior::ir::Location::unknown(gen.context))
                    .add_operands(&[actual_tag, tag])
                    .add_results(&[melior::ir::r#type::IntegerType::new(gen.context, 1).into()])
                    .add_attributes(&[(
                        Identifier::new(gen.context, "predicate"),
                        IntegerAttribute::new(
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

    if let Pattern::EnumVariant(_, _, Some(payloads)) = &arm.pattern {
        if payloads.len() == 1 {
            if let Pattern::Identifier(name) = &payloads[0] {
                let opt_ty_str = _match_ty.to_string();
                let mut payload_ty_str = if opt_ty_str.contains("(i32, ") {
                    let start = opt_ty_str.find("(i32, ").unwrap() + 6;
                    let end = opt_ty_str.rfind(')').unwrap();
                    opt_ty_str[start..end].to_string()
                } else {
                    "i32".to_string() // fallback
                };
                if payload_ty_str.starts_with("struct<")
                    || payload_ty_str.starts_with("ptr")
                    || payload_ty_str.starts_with("func")
                    || payload_ty_str.starts_with("array")
                {
                    payload_ty_str = format!("!llvm.{}", payload_ty_str);
                }
                let payload_ty = melior::ir::Type::parse(gen.context, &payload_ty_str)
                    .unwrap_or_else(|| {
                        panic!(
                            "Failed to parse payload_ty_str: {:?} from opt_ty_str: {:?}",
                            payload_ty_str, opt_ty_str
                        );
                    });
                let extract_payload_op = OperationBuilder::new(
                    "llvm.extractvalue",
                    melior::ir::Location::unknown(gen.context),
                )
                .add_operands(&[match_val])
                .add_results(&[payload_ty])
                .add_attributes(&[(
                    Identifier::new(gen.context, "position"),
                    melior::ir::attribute::DenseI64ArrayAttribute::new(gen.context, &[1]).into(),
                )])
                .build()
                .unwrap();
                let payload_val = then_block
                    .append_operation(extract_payload_op)
                    .result(0)
                    .unwrap()
                    .into();
                gen.env.insert(name.clone(), (payload_val, payload_ty));
            }
        }
    }

    for stmt in &arm.body {
        gen.generate_statement(stmt, &then_block);
    }
    then_block.append_operation(
        OperationBuilder::new("scf.yield", melior::ir::Location::unknown(gen.context))
            .build()
            .unwrap(),
    );
    then_region.append_block(then_block);

    let else_region = melior::ir::Region::new();
    let else_block = melior::ir::Block::new(&[]);

    // Recursively generate the rest of the arms inside the else block
    generate_match_chain(gen, &arms[1..], match_val, _match_ty, &else_block);

    else_block.append_operation(
        OperationBuilder::new("scf.yield", melior::ir::Location::unknown(gen.context))
            .build()
            .unwrap(),
    );
    else_region.append_block(else_block);

    block.append_operation(
        OperationBuilder::new("scf.if", melior::ir::Location::unknown(gen.context))
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
        let op =
            OperationBuilder::new("arith.constant", melior::ir::Location::unknown(gen.context))
                .add_results(&[ty])
                .add_attributes(&[(
                    Identifier::new(gen.context, "value"),
                    IntegerAttribute::new(ty, 0).into(),
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
    continue_ptr: Value<'c, 'c>,
    c0_idx: Value<'c, 'c>,
    i1_ty: Type<'c>,
) {
    if stmts.is_empty() {
        return;
    }

    gen.generate_statement(&stmts[0], block);

    if stmts.len() > 1 {
        if contains_break(&stmts[0]) {
            let load_break = block
                .append_operation(
                    OperationBuilder::new("memref.load", Location::unknown(gen.context))
                        .add_operands(&[break_ptr, c0_idx])
                        .add_results(&[i1_ty])
                        .build()
                        .unwrap(),
                )
                .result(0)
                .unwrap()
                .into();

            let load_cont = block
                .append_operation(
                    OperationBuilder::new("memref.load", Location::unknown(gen.context))
                        .add_operands(&[continue_ptr, c0_idx])
                        .add_results(&[i1_ty])
                        .build()
                        .unwrap(),
                )
                .result(0)
                .unwrap()
                .into();

            let is_break_or_cont = block
                .append_operation(
                    OperationBuilder::new("arith.ori", Location::unknown(gen.context))
                        .add_operands(&[load_break, load_cont])
                        .add_results(&[i1_ty])
                        .build()
                        .unwrap(),
                )
                .result(0)
                .unwrap()
                .into();

            let true_val = block
                .append_operation(
                    OperationBuilder::new("arith.constant", Location::unknown(gen.context))
                        .add_results(&[i1_ty])
                        .add_attributes(&[(
                            Identifier::new(gen.context, "value"),
                            IntegerAttribute::new(i1_ty, 1).into(),
                        )])
                        .build()
                        .unwrap(),
                )
                .result(0)
                .unwrap()
                .into();

            let not_break = block
                .append_operation(
                    OperationBuilder::new("arith.xori", Location::unknown(gen.context))
                        .add_operands(&[is_break_or_cont, true_val])
                        .add_results(&[i1_ty])
                        .build()
                        .unwrap(),
                )
                .result(0)
                .unwrap()
                .into();

            let if_region = Region::new();
            let if_block = Block::new(&[]);

            generate_statements_with_break_guard(
                gen,
                &stmts[1..],
                &if_block,
                break_ptr,
                continue_ptr,
                c0_idx,
                i1_ty,
            );

            if_block.append_operation(
                OperationBuilder::new("scf.yield", Location::unknown(gen.context))
                    .build()
                    .unwrap(),
            );
            if_region.append_block(if_block);

            let else_region = melior::ir::Region::new();
            let else_block = melior::ir::Block::new(&[]);
            let yield_op = OperationBuilder::new("scf.yield", Location::unknown(gen.context))
                .build()
                .unwrap();
            else_block.append_operation(yield_op);
            else_region.append_block(else_block);

            block.append_operation(
                OperationBuilder::new("scf.if", Location::unknown(gen.context))
                    .add_operands(&[not_break])
                    .add_regions([if_region, else_region])
                    .build()
                    .unwrap(),
            );
        } else {
            generate_statements_with_break_guard(
                gen,
                &stmts[1..],
                block,
                break_ptr,
                continue_ptr,
                c0_idx,
                i1_ty,
            );
        }
    }
}

impl<'c> LowerToMelior<'c> for EnumVariantExpr {
    type Output = (Value<'c, 'c>, Type<'c>);
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

        let i32_ty = Type::parse(gen.context, "i32").unwrap();
        let tag_op = OperationBuilder::new("arith.constant", Location::unknown(gen.context))
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
            return (tag_val, i32_ty);
        }

        // We have an Option<T> struct
        let struct_ty = Type::parse(gen.context, &enum_ty_str).unwrap();
        let undef_op = OperationBuilder::new("llvm.mlir.undef", Location::unknown(gen.context))
            .add_results(&[struct_ty])
            .build()
            .unwrap();
        let undef_val = block.append_operation(undef_op).result(0).unwrap().into();

        let insert_tag_op =
            OperationBuilder::new("llvm.insertvalue", Location::unknown(gen.context))
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
                let (payload_val, _payload_ty) = gen.generate_expr(&payload_exprs[0], block);
                let insert_payload_op =
                    OperationBuilder::new("llvm.insertvalue", Location::unknown(gen.context))
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

        (struct_val, struct_ty)
    }
}

impl<'c> LowerToMelior<'c> for VecMacroExpr {
    type Output = (Value<'c, 'c>, Type<'c>);
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
            span: self.span.clone(),
        });

        let (vec_val, vec_ty) = gen.generate_expr(&new_call, block);

        let i32_ty = Type::parse(gen.context, "i32").unwrap();
        let c1_op = block.append_operation(
            OperationBuilder::new("llvm.mlir.constant", Location::unknown(gen.context))
                .add_results(&[i32_ty])
                .add_attributes(&[(
                    Identifier::new(gen.context, "value"),
                    IntegerAttribute::new(i32_ty, 1).into(),
                )])
                .build()
                .unwrap(),
        );
        let c1 = c1_op.result(0).unwrap().into();

        let ptr_ty = Type::parse(gen.context, "!llvm.ptr").unwrap();
        let alloca_op = block.append_operation(
            OperationBuilder::new("llvm.alloca", Location::unknown(gen.context))
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
            OperationBuilder::new("llvm.store", Location::unknown(gen.context))
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
                span: self.span.clone(),
            });
            gen.generate_expr(&push_call, block);
        }

        let load_op = block.append_operation(
            OperationBuilder::new("llvm.load", Location::unknown(gen.context))
                .add_operands(&[ptr_val])
                .add_results(&[vec_ty])
                .build()
                .unwrap(),
        );
        (load_op.result(0).unwrap().into(), vec_ty)
    }
}

impl<'c> LowerToMelior<'c> for ClosureExpr {
    type Output = (Value<'c, 'c>, Type<'c>);

    fn lower(
        &self,
        _gen: &mut MeliorGenerator<'c>,
        _block: &melior::ir::Block<'c>,
    ) -> Self::Output {
        unreachable!("ClosureExpr should be transformed to StructInitExpr by Sema")
    }
}

impl<'c> LowerToMelior<'c> for ast::expr::PrintExpr {
    type Output = (Value<'c, 'c>, Type<'c>);
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        for arg in &self.args {
            let (arg_val, arg_ty) = gen.generate_expr(arg, block);

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

                let func_decl = OperationBuilder::new("func.func", Location::unknown(gen.context))
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
                        Type::parse(gen.context, "i32").unwrap(),
                        vec![if func_name == "print_str" {
                            Type::parse(gen.context, "!llvm.ptr").unwrap()
                        } else {
                            arg_ty
                        }],
                    ),
                );
            }

            let name_attr = FlatSymbolRefAttribute::new(gen.context, func_name);
            let call_op = OperationBuilder::new("func.call", Location::unknown(gen.context))
                .add_operands(&[arg_val])
                .add_results(&[Type::parse(gen.context, "i32").unwrap()])
                .add_attributes(&[(Identifier::new(gen.context, "callee"), name_attr.into())])
                .build()
                .unwrap();

            block.append_operation(call_op);
        }

        let dummy_op = OperationBuilder::new("llvm.mlir.constant", Location::unknown(gen.context))
            .add_results(&[Type::parse(gen.context, "i32").unwrap()])
            .add_attributes(&[(
                Identifier::new(gen.context, "value"),
                IntegerAttribute::new(Type::parse(gen.context, "i32").unwrap(), 0).into(),
            )])
            .build()
            .unwrap();

        (
            block.append_operation(dummy_op).result(0).unwrap().into(),
            Type::parse(gen.context, "i32").unwrap(),
        )
    }
}

impl<'c> LowerToMelior<'c> for ast::expr::PrintlnExpr {
    type Output = (Value<'c, 'c>, Type<'c>);
    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        // First, reuse PrintExpr logic for arguments
        if !self.args.is_empty() {
            let print_expr = ast::expr::PrintExpr {
                args: self.args.clone(),
                span: self.span.clone(),
            };
            print_expr.lower(gen, block);
        }

        // Then emit a call to println()
        if !gen.functions.contains_key("println") {
            let func_ty = Type::parse(gen.context, "() -> i32").unwrap();

            let func_decl = OperationBuilder::new("func.func", Location::unknown(gen.context))
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
            gen.functions.insert(
                "println".to_string(),
                (Type::parse(gen.context, "i32").unwrap(), vec![]),
            );
        }

        let name_attr = FlatSymbolRefAttribute::new(gen.context, "println");
        let call_op = OperationBuilder::new("func.call", Location::unknown(gen.context))
            .add_results(&[Type::parse(gen.context, "i32").unwrap()])
            .add_attributes(&[(Identifier::new(gen.context, "callee"), name_attr.into())])
            .build()
            .unwrap();

        (
            block.append_operation(call_op).result(0).unwrap().into(),
            Type::parse(gen.context, "i32").unwrap(),
        )
    }
}

impl<'c> LowerToMelior<'c> for ast::expr::AsCastExpr {
    type Output = (Value<'c, 'c>, Type<'c>);

    fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {
        let (source_val, _source_ty) = gen.generate_expr(&self.expr, block);

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
            let const_op = OperationBuilder::new("func.constant", Location::unknown(gen.context))
                .add_attributes(&[(
                    Identifier::new(gen.context, "value"),
                    FlatSymbolRefAttribute::new(gen.context, &call_fn_name).into(),
                )])
                .add_results(&[fn_ty.into()])
                .build()
                .unwrap();
            let const_ref = block.append_operation(const_op);
            let mut fn_ptr_val: melior::ir::Value = const_ref.result(0).unwrap().into();

            let ptr_ty = Type::parse(gen.context, "!llvm.ptr").unwrap();
            let bitcast_op = OperationBuilder::new(
                "builtin.unrealized_conversion_cast",
                Location::unknown(gen.context),
            )
            .add_operands(&[fn_ptr_val])
            .add_results(&[ptr_ty])
            .build()
            .unwrap();
            let bitcast_ref = block.append_operation(bitcast_op);
            fn_ptr_val = bitcast_ref.result(0).unwrap().into();

            let fat_ptr_ty = Type::parse(gen.context, "!llvm.struct<(ptr, ptr)>").unwrap();
            let undef_op = OperationBuilder::new("llvm.mlir.undef", Location::unknown(gen.context))
                .add_results(&[fat_ptr_ty])
                .build()
                .unwrap();
            let undef_ref = block.append_operation(undef_op);
            let mut fat_ptr_val: melior::ir::Value = undef_ref.result(0).unwrap().into();

            let insert_fn_op =
                OperationBuilder::new("llvm.insertvalue", Location::unknown(gen.context))
                    .add_operands(&[fat_ptr_val, fn_ptr_val])
                    .add_attributes(&[(
                        Identifier::new(gen.context, "position"),
                        melior::ir::attribute::DenseI64ArrayAttribute::new(gen.context, &[0])
                            .into(),
                    )])
                    .add_results(&[fat_ptr_ty])
                    .build()
                    .unwrap();
            let insert_fn_ref = block.append_operation(insert_fn_op);
            fat_ptr_val = insert_fn_ref.result(0).unwrap().into();

            let mut env_ptr_val = source_val;
            let ptr_bitcast_op = OperationBuilder::new(
                "builtin.unrealized_conversion_cast",
                Location::unknown(gen.context),
            )
            .add_operands(&[env_ptr_val])
            .add_results(&[ptr_ty])
            .build()
            .unwrap();
            let ptr_bitcast_ref = block.append_operation(ptr_bitcast_op);
            env_ptr_val = ptr_bitcast_ref.result(0).unwrap().into();

            let insert_env_op =
                OperationBuilder::new("llvm.insertvalue", Location::unknown(gen.context))
                    .add_operands(&[fat_ptr_val, env_ptr_val])
                    .add_attributes(&[(
                        Identifier::new(gen.context, "position"),
                        melior::ir::attribute::DenseI64ArrayAttribute::new(gen.context, &[1])
                            .into(),
                    )])
                    .add_results(&[fat_ptr_ty])
                    .build()
                    .unwrap();
            let insert_env_ref = block.append_operation(insert_env_op);
            fat_ptr_val = insert_env_ref.result(0).unwrap().into();

            return (fat_ptr_val, fat_ptr_ty);
        } else if let ast::Type::Scalar(_) = &self.target_ty {
            let target_ty_mlir = gen.lower_type(&self.target_ty);
            let coerced_val = gen.coerce_type(block, source_val, _source_ty, target_ty_mlir);
            return (coerced_val, target_ty_mlir);
        }

        panic!("Unsupported cast operation in codegen");
    }
}
fn lower_map_call<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: &melior::ir::Block<'c>,
    args: &[Expr],
) -> (Value<'c, 'c>, Type<'c>) {
    let (tensor_val, tensor_ty) = gen.generate_expr(&args[0], block);
    // args[1] is the closure
    // We need to fetch the closure's function and invoke it inside a linalg.generic.
    let tensor_ty_str = tensor_ty.to_string();
    if tensor_ty_str.starts_with("memref<") {
        let parts: Vec<&str> = tensor_ty_str
            .trim_start_matches("memref<")
            .trim_end_matches('>')
            .split('x')
            .collect();
        let el_ty_str = parts.last().unwrap_or(&"f32").trim();
        let rank = parts.len() - 1;

        // 1. Allocate output memref
        let mut alloc_operands = Vec::new();
        let index_ty = Type::parse(gen.context, "index").unwrap();
        for (i, dim_str) in parts.iter().take(rank).enumerate() {
            if *dim_str == "?" {
                let idx_attr = IntegerAttribute::new(Type::index(gen.context), i as i64).into();
                let cst_op =
                    OperationBuilder::new("arith.constant", Location::unknown(gen.context))
                        .add_results(&[index_ty])
                        .add_attributes(&[(Identifier::new(gen.context, "value"), idx_attr)])
                        .build()
                        .unwrap();
                let idx_val = block.append_operation(cst_op).result(0).unwrap().into();

                let dim_op = OperationBuilder::new("memref.dim", Location::unknown(gen.context))
                    .add_operands(&[tensor_val, idx_val])
                    .add_results(&[index_ty])
                    .build()
                    .unwrap();
                alloc_operands.push(block.append_operation(dim_op).result(0).unwrap().into());
            }
        }

        let alloc_op = OperationBuilder::new("memref.alloc", Location::unknown(gen.context))
            .add_operands(&alloc_operands)
            .add_attributes(&[(
                Identifier::new(gen.context, "operandSegmentSizes"),
                DenseI32ArrayAttribute::new(gen.context, &[alloc_operands.len() as i32, 0]).into(),
            )])
            .add_results(&[tensor_ty])
            .build()
            .unwrap();
        let out_val = block.append_operation(alloc_op).result(0).unwrap().into();

        // 2. Generate linalg.generic
        let region = Region::new();
        let block_generic = melior::ir::Block::new(&[
            (
                Type::parse(gen.context, el_ty_str).unwrap(),
                Location::unknown(gen.context),
            ),
            (
                Type::parse(gen.context, el_ty_str).unwrap(),
                Location::unknown(gen.context),
            ),
        ]);

        // We need to call the closure!
        // args[1] is the closure expression (StructInitExpr for Closure_N).
        let (closure_val, closure_ty) = gen.generate_expr(&args[1], block);

        // Allocate it on stack to get a pointer
        let ptr_ty = Type::parse(gen.context, "!llvm.ptr").unwrap();
        let i32_ty = Type::parse(gen.context, "i32").unwrap();
        let c1_op = OperationBuilder::new("arith.constant", Location::unknown(gen.context))
            .add_results(&[i32_ty])
            .add_attributes(&[(
                Identifier::new(gen.context, "value"),
                IntegerAttribute::new(i32_ty, 1).into(),
            )])
            .build()
            .unwrap();
        let c1 = block.append_operation(c1_op).result(0).unwrap().into();

        let alloca_op = OperationBuilder::new("llvm.alloca", Location::unknown(gen.context))
            .add_operands(&[c1])
            .add_results(&[ptr_ty])
            .add_attributes(&[(
                Identifier::new(gen.context, "elem_type"),
                TypeAttribute::new(closure_ty).into(),
            )])
            .build()
            .unwrap();
        let alloca_ptr = block.append_operation(alloca_op).result(0).unwrap().into();

        let store_op = OperationBuilder::new("llvm.store", Location::unknown(gen.context))
            .add_operands(&[closure_val, alloca_ptr])
            .build()
            .unwrap();
        block.append_operation(store_op);

        // Extract invoke method name
        let ty_str = closure_ty.to_string();
        let struct_name = ty_str
            .strip_prefix("!llvm.struct<\"")
            .unwrap_or(&ty_str)
            .split("\"")
            .next()
            .unwrap();
        let invoke_method = format!("{}_call", struct_name);

        let call_op = OperationBuilder::new("func.call", Location::unknown(gen.context))
            .add_operands(&[alloca_ptr, block_generic.argument(0).unwrap().into()])
            .add_attributes(&[(
                Identifier::new(gen.context, "callee"),
                FlatSymbolRefAttribute::new(gen.context, &invoke_method).into(),
            )])
            .add_results(&[Type::parse(gen.context, el_ty_str).unwrap()])
            .build()
            .unwrap();

        let call_res = block_generic
            .append_operation(call_op)
            .result(0)
            .unwrap()
            .into();

        let yield_op = OperationBuilder::new("linalg.yield", Location::unknown(gen.context))
            .add_operands(&[call_res])
            .build()
            .unwrap();
        block_generic.append_operation(yield_op);
        region.append_block(block_generic);

        // Affine maps: both input and output are identity maps
        let affine_map = format!(
            "affine_map<(d0{}) -> (d0{})>",
            (1..rank).map(|i| format!(", d{}", i)).collect::<String>(),
            (1..rank).map(|i| format!(", d{}", i)).collect::<String>()
        );
        // For rank 0 it's affine_map<() -> ()>
        let affine_map_attr = if rank == 0 {
            Attribute::parse(gen.context, "affine_map<() -> ()>").unwrap()
        } else {
            Attribute::parse(gen.context, &affine_map).unwrap()
        };

        let linalg_generic =
            OperationBuilder::new("linalg.generic", Location::unknown(gen.context))
                .add_operands(&[tensor_val, out_val])
                .add_attributes(&[
                    (
                        Identifier::new(gen.context, "operandSegmentSizes"),
                        DenseI32ArrayAttribute::new(gen.context, &[1, 1]).into(),
                    ),
                    (
                        Identifier::new(gen.context, "indexing_maps"),
                        ArrayAttribute::new(gen.context, &[affine_map_attr, affine_map_attr])
                            .into(),
                    ),
                    (
                        Identifier::new(gen.context, "iterator_types"),
                        ArrayAttribute::new(
                            gen.context,
                            &vec![
                                Attribute::parse(gen.context, "#linalg.iterator_type<parallel>")
                                    .unwrap();
                                rank
                            ],
                        )
                        .into(),
                    ),
                ])
                .add_regions([region])
                .build()
                .unwrap();
        block.append_operation(linalg_generic);

        return (out_val, tensor_ty);
    }
    panic!("map called on unsupported tensor type: {}", tensor_ty_str);
}

#[derive(Debug)]
pub enum LowerError {
    UnsupportedElementType(String),
    Melior(melior::Error),
    ParseType(String),
}

impl From<melior::Error> for LowerError {
    fn from(e: melior::Error) -> Self {
        LowerError::Melior(e)
    }
}

impl From<String> for LowerError {
    fn from(s: String) -> Self {
        LowerError::ParseType(s)
    }
}

fn lower_print_call<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: &melior::ir::Block<'c>,
    args: &[Expr],
) -> Result<(Value<'c, 'c>, Type<'c>), LowerError> {
    let mut print_arg = &args[0];
    if let Expr::Borrow(borrow) = print_arg {
        print_arg = &borrow.expr;
    }

    let (mut arg_val, arg_ty) = gen.generate_expr(print_arg, block);

    let el_ty_str = extract_mlir_element_type(&arg_ty.to_string())?;

    let print_fn_name = match el_ty_str {
        "f32" => "printMemrefF32",
        "f64" => "printMemrefF64",
        "i32" => "printMemrefI32",
        "i64" => "printMemrefI64",
        "bf16" => "printMemrefBF16",
        _ => return Err(LowerError::UnsupportedElementType(el_ty_str.to_string())),
    };

    if arg_ty.to_string().contains(", ") {
        let stripped_ty = Type::parse(gen.context, &format!("memref<?x?x{}>", el_ty_str))
            .ok_or_else(|| LowerError::ParseType(format!("memref<?x?x{}>", el_ty_str)))?;
        let mcast_op = block.append_operation(
            OperationBuilder::new("memref.memory_space_cast", Location::unknown(gen.context))
                .add_operands(&[arg_val])
                .add_results(&[stripped_ty])
                .build()?,
        );
        arg_val = mcast_op.result(0)?.into();
    }

    let unranked_memref_ty = Type::parse(gen.context, &format!("memref<*x{}>", el_ty_str))
        .ok_or_else(|| LowerError::ParseType(format!("memref<*x{}>", el_ty_str)))?;
    let cast_op = block.append_operation(
        OperationBuilder::new("memref.cast", Location::unknown(gen.context))
            .add_operands(&[arg_val])
            .add_results(&[unranked_memref_ty])
            .build()?,
    );
    let cast_val: Value = cast_op.result(0)?.into();

    block.append_operation(
        OperationBuilder::new("func.call", Location::unknown(gen.context))
            .add_operands(&[cast_val])
            .add_attributes(&[(
                Identifier::new(gen.context, "callee"),
                FlatSymbolRefAttribute::new(gen.context, print_fn_name).into(),
            )])
            .build()?,
    );

    return Ok((
        cast_val, // Dummy return value, caller ignores it
        Type::parse(gen.context, "none")
            .ok_or_else(|| LowerError::ParseType("none".to_string()))?,
    ));
}
