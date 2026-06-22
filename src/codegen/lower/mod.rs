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
        ArrayAttribute, Attribute, DenseI32ArrayAttribute, FlatSymbolRefAttribute,
        IntegerAttribute, StringAttribute, TypeAttribute,
    },
    operation::OperationBuilder,
    Identifier,
};

use crate::ast;
mod control_flow;
mod expr;
mod stmt;
mod tensors;

pub trait LowerToMelior<'c> {
    type Output;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output;
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

pub(crate) fn topology_to_i32(top: &ast::Topology) -> i32 {
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
        CpuAvx512 => 600,
        CpuNeon => 700,
        Slice(_, _, _) => 900,
        Current => 0,
    }
}

pub(crate) fn extract_mlir_element_type(ty_str: &str) -> Result<&'static str, String> {
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

pub(crate) fn emit_enzyme_decl<'c>(
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
    if !gen.functions.contains_key(&*enzyme_name) && gen.enzyme_decls.insert(enzyme_name.clone()) {
        let func_type = melior::ir::r#type::FunctionType::new(gen.context, arg_tys, &[ret_ty]);
        let _name_attr = StringAttribute::new(gen.context, &enzyme_name);
        let _type_attr = TypeAttribute::new(func_type.into());

        let region = melior::ir::Region::new();
        let func_op = OperationBuilder::new("func.func", gen.loc())
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

pub fn generate_match_chain<'c>(
    gen: &mut MeliorGenerator<'c>,
    arms: &[MatchArm],
    match_val: melior::ir::Value<'c, 'c>,
    _match_ty: melior::ir::Type<'c>,
    mut block: melior::ir::BlockRef<'c, 'c>,
    merge_block: melior::ir::BlockRef<'c, 'c>,
) -> Result<melior::ir::BlockRef<'c, 'c>, LowerError> {
    if arms.is_empty() {
        block.append_operation(
            OperationBuilder::new("cf.br", gen.loc())
                .add_successors(&[&*merge_block])
                .build()
                .unwrap(),
        );
        return Ok(block);
    }

    let arm = &arms[0];

    if let Pattern::Wildcard = arm.pattern {
        let mut then_terminated = false;
        for stmt in &arm.body {
            if let Some(b) = gen.generate_statement(stmt, block)? {
                block = b;
            } else {
                then_terminated = true;
                break;
            }
        }
        if !then_terminated {
            block.append_operation(
                OperationBuilder::new("cf.br", gen.loc())
                    .add_successors(&[&*merge_block])
                    .build()
                    .unwrap(),
            );
        }
        return Ok(block);
    }

    let parent_region = block.parent_region().unwrap();
    let mut then_block = parent_region.append_block(melior::ir::Block::new(&[]));
    let else_block = parent_region.append_block(melior::ir::Block::new(&[]));

    let cond_val = match &arm.pattern {
        Pattern::EnumVariant(_, variant_name, _) => {
            let i32_ty = melior::ir::r#type::IntegerType::new(gen.context, 32).into();
            let mut tag_val = 0;
            for enum_def in gen.enums.values() {
                for (i, v) in enum_def.iter().enumerate() {
                    if *v.0 == **variant_name {
                        tag_val = i as i64;
                        break;
                    }
                }
            }

            let tag_op = block.append_operation(
                OperationBuilder::new("arith.constant", gen.loc())
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
                    OperationBuilder::new("llvm.extractvalue", gen.loc())
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
                OperationBuilder::new("arith.cmpi", gen.loc())
                    .add_operands(&[actual_tag, tag])
                    .add_results(&[melior::ir::r#type::IntegerType::new(gen.context, 1).into()])
                    .add_attributes(&[(
                        Identifier::new(gen.context, "predicate"),
                        IntegerAttribute::new(
                            melior::ir::r#type::IntegerType::new(gen.context, 64).into(),
                            0, // eq
                        )
                        .into(),
                    )])
                    .build()
                    .unwrap(),
            );
            cmp_op.result(0).unwrap().into()
        }
        _ => panic!("Unsupported pattern in codegen"),
    };

    block.append_operation(
        OperationBuilder::new("cf.cond_br", gen.loc())
            .add_operands(&[cond_val])
            .add_successors(&[&*then_block, &*else_block])
            .build()
            .unwrap(),
    );

    if let Pattern::EnumVariant(_, _, Some(payloads)) = &arm.pattern {
        if payloads.len() == 1 {
            if let Pattern::Identifier(name) = &payloads[0] {
                let opt_ty_str = _match_ty.to_string();
                let mut payload_ty_str = if opt_ty_str.contains("(i32, ") {
                    let start = opt_ty_str.find("(i32, ").unwrap() + 6;
                    let end = opt_ty_str.rfind(')').unwrap();
                    opt_ty_str[start..end].to_string()
                } else {
                    "i32".to_string()
                };
                if payload_ty_str.starts_with("struct<")
                    || payload_ty_str.starts_with("ptr")
                    || payload_ty_str.starts_with("func")
                    || payload_ty_str.starts_with("array")
                {
                    payload_ty_str = format!("!llvm.{}", payload_ty_str);
                }
                let payload_ty = melior::ir::Type::parse(gen.context, &payload_ty_str).unwrap();
                let extract_payload_op = OperationBuilder::new("llvm.extractvalue", gen.loc())
                    .add_operands(&[match_val])
                    .add_results(&[payload_ty])
                    .add_attributes(&[(
                        Identifier::new(gen.context, "position"),
                        melior::ir::attribute::DenseI64ArrayAttribute::new(gen.context, &[1])
                            .into(),
                    )])
                    .build()
                    .unwrap();
                let payload_val = then_block
                    .append_operation(extract_payload_op)
                    .result(0)
                    .unwrap()
                    .into();
                gen.env
                    .insert(name.to_string().into(), (payload_val, payload_ty));
            }
        }
    }

    let mut then_terminated = false;
    for stmt in &arm.body {
        if let Some(b) = gen.generate_statement(stmt, then_block)? {
            then_block = b;
        } else {
            then_terminated = true;
            break;
        }
    }
    if !then_terminated {
        then_block.append_operation(
            OperationBuilder::new("cf.br", gen.loc())
                .add_successors(&[&*merge_block])
                .build()
                .unwrap(),
        );
    }

    generate_match_chain(
        gen,
        &arms[1..],
        match_val,
        _match_ty,
        else_block,
        merge_block,
    )?;

    Ok(block)
}

pub(crate) fn lower_map_call<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    args: &[Expr],
) -> Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError> {
    let (tensor_val, tensor_ty, block) = gen.generate_expr(&args[0], block)?;
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
        let index_ty = gen.index_ty;
        for (i, dim_str) in parts.iter().take(rank).enumerate() {
            if *dim_str == "?" {
                let idx_attr = IntegerAttribute::new(Type::index(gen.context), i as i64).into();
                let cst_op = OperationBuilder::new("arith.constant", gen.loc())
                    .add_results(&[index_ty])
                    .add_attributes(&[(Identifier::new(gen.context, "value"), idx_attr)])
                    .build()
                    .unwrap();
                let idx_val = block.append_operation(cst_op).result(0).unwrap().into();

                let dim_op = OperationBuilder::new("memref.dim", gen.loc())
                    .add_operands(&[tensor_val, idx_val])
                    .add_results(&[index_ty])
                    .build()
                    .unwrap();
                alloc_operands.push(block.append_operation(dim_op).result(0).unwrap().into());
            }
        }

        let alloc_op = OperationBuilder::new("memref.alloc", gen.loc())
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
            (Type::parse(gen.context, el_ty_str).unwrap(), gen.loc()),
            (Type::parse(gen.context, el_ty_str).unwrap(), gen.loc()),
        ]);

        // We need to call the closure!
        // args[1] is the closure expression (StructInitExpr for Closure_N).
        let (closure_val, closure_ty, block) = gen.generate_expr(&args[1], block)?;

        // Allocate it on stack to get a pointer
        let ptr_ty = gen.ptr_ty;
        let i32_ty = gen.i32_ty;
        let c1_op = OperationBuilder::new("arith.constant", gen.loc())
            .add_results(&[i32_ty])
            .add_attributes(&[(
                Identifier::new(gen.context, "value"),
                IntegerAttribute::new(i32_ty, 1).into(),
            )])
            .build()
            .unwrap();
        let c1 = block.append_operation(c1_op).result(0).unwrap().into();

        let alloca_op = OperationBuilder::new("llvm.alloca", gen.loc())
            .add_operands(&[c1])
            .add_results(&[ptr_ty])
            .add_attributes(&[(
                Identifier::new(gen.context, "elem_type"),
                TypeAttribute::new(closure_ty).into(),
            )])
            .build()
            .unwrap();
        let alloca_ptr = block.append_operation(alloca_op).result(0).unwrap().into();

        let store_op = OperationBuilder::new("llvm.store", gen.loc())
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

        let call_op = OperationBuilder::new("func.call", gen.loc())
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

        let yield_op = OperationBuilder::new("linalg.yield", gen.loc())
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

        let linalg_generic = OperationBuilder::new("linalg.generic", gen.loc())
            .add_operands(&[tensor_val, out_val])
            .add_attributes(&[
                (
                    Identifier::new(gen.context, "operandSegmentSizes"),
                    DenseI32ArrayAttribute::new(gen.context, &[1, 1]).into(),
                ),
                (
                    Identifier::new(gen.context, "indexing_maps"),
                    ArrayAttribute::new(gen.context, &[affine_map_attr, affine_map_attr]).into(),
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

        return Ok((out_val, tensor_ty, block));
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

pub(crate) fn lower_print_call<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    args: &[Expr],
) -> Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError> {
    let mut print_arg = &args[0];
    if let Expr::Borrow(borrow) = print_arg {
        print_arg = &borrow.expr;
    }

    let (mut arg_val, arg_ty, block) = gen.generate_expr(print_arg, block)?;

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
            OperationBuilder::new("memref.memory_space_cast", gen.loc())
                .add_operands(&[arg_val])
                .add_results(&[stripped_ty])
                .build()?,
        );
        arg_val = mcast_op.result(0)?.into();
    }

    let unranked_memref_ty = Type::parse(gen.context, &format!("memref<*x{}>", el_ty_str))
        .ok_or_else(|| LowerError::ParseType(format!("memref<*x{}>", el_ty_str)))?;
    let cast_op = block.append_operation(
        OperationBuilder::new("memref.cast", gen.loc())
            .add_operands(&[arg_val])
            .add_results(&[unranked_memref_ty])
            .build()?,
    );
    let cast_val: Value = cast_op.result(0)?.into();

    block.append_operation(
        OperationBuilder::new("func.call", gen.loc())
            .add_operands(&[cast_val])
            .add_attributes(&[(
                Identifier::new(gen.context, "callee"),
                FlatSymbolRefAttribute::new(gen.context, print_fn_name).into(),
            )])
            .build()?,
    );

    Ok((
        cast_val, // Dummy return value, caller ignores it
        Some(gen.none_ty).ok_or_else(|| LowerError::ParseType("none".to_string()))?,
        block,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ast::{Expr, NumberExpr, Span, Topology};

    fn make_num_expr(val: &str) -> Box<Expr> {
        Box::new(Expr::Number(NumberExpr::new(
            val.to_string(),
            None,
            Span::default(),
        )))
    }

    #[test]
    fn test_topology_to_i32_all_variants() {
        assert_eq!(topology_to_i32(&Topology::CPU), 0);
        assert_eq!(topology_to_i32(&Topology::NPU(make_num_expr("0"))), 100);
        assert_eq!(topology_to_i32(&Topology::NPU(make_num_expr("3"))), 103);
        assert_eq!(topology_to_i32(&Topology::AccCore(make_num_expr("0"))), 200);
        assert_eq!(topology_to_i32(&Topology::AccCore(make_num_expr("5"))), 205);
        assert_eq!(topology_to_i32(&Topology::AMX), 300);
        assert_eq!(topology_to_i32(&Topology::ANE), 400);
        assert_eq!(topology_to_i32(&Topology::GPU), 500);
        assert_eq!(topology_to_i32(&Topology::CpuAvx512), 600);
        assert_eq!(topology_to_i32(&Topology::CpuNeon), 700);
        assert_eq!(topology_to_i32(&Topology::Current), 0);
    }

    #[test]
    fn test_topology_to_i32_slice() {
        let slice = Topology::Slice(
            Box::new(Topology::NPU(make_num_expr("0"))),
            make_num_expr("0"),
            make_num_expr("4"),
        );
        assert_eq!(topology_to_i32(&slice), 900);
    }

    #[test]
    fn test_topology_to_i32_npu_non_numeric_falls_back() {
        // NPU with a non-numeric expr should fall back to 100
        let ident_expr = Box::new(Expr::Identifier(ast::IdentifierExpr {
            name: "i".into(),
            span: Span::default(),
        }));
        assert_eq!(topology_to_i32(&Topology::NPU(ident_expr)), 100);
    }
}
