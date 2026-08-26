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

use crate::syntax;
mod control_flow;
mod expr;
mod seam_cert;
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

/// A block to keep appending to after a statement closed the current one.
///
/// `return`, `break` and `continue` terminate their block, and MLIR admits no
/// operation after a terminator. Whatever follows them in the source is
/// unreachable rather than illegal, so it gets a block of its own: the verifier
/// checks dominance only for blocks reachable from the entry, and a block with
/// no predecessors is not one. The alternative -- piling the rest of the
/// function on after a `func.return` -- does not compile at all.
pub(crate) fn dead_continuation<'c>(
    block: melior::ir::BlockRef<'c, 'c>,
) -> melior::ir::BlockRef<'c, 'c> {
    block
        .parent_region()
        .expect("a block being lowered into always sits in a region")
        .append_block(melior::ir::Block::new(&[]))
}

/// Runtime dispatch id for a topology. Thin delegate to the single source of truth in
/// `arch`, which co-locates this with the memory-space mapping so the two cannot diverge.
pub(crate) fn topology_to_i32(top: &syntax::Topology) -> i32 {
    crate::arch::topology_dispatch_id(top)
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
    let enzyme_name = crate::codegen::enzyme_wrapper_name(base_name == "fwddiff", target_fn);
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
            .expect("Failed to build enzyme decl");

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
                .build()?,
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
                    .build()?,
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
                    .build()?,
            );
            let tag = tag_op.result(0)?.into();

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
                        .build()?,
                );
                extract_tag_op.result(0)?.into()
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
                    .build()?,
            );
            cmp_op.result(0)?.into()
        }
        // A scalar literal arm (`0 => …`, `1 => …`) over an integer scrutinee: compare the
        // scrutinee against the literal with the same `arith.cmpi eq` shape as the enum-tag path,
        // but against `match_val` directly (no tag extraction). Without this, an integer `match`
        // ICE'd here — `Unsupported pattern in codegen` (#263).
        Pattern::Literal(lit) => {
            let lit_val = match lit {
                Expr::Number(n) => n.value.as_ref().parse::<i64>().unwrap_or(0),
                other => panic!("Unsupported literal pattern in codegen: {:?}", other),
            };
            let const_op = block.append_operation(
                OperationBuilder::new("arith.constant", gen.loc())
                    .add_results(&[_match_ty])
                    .add_attributes(&[(
                        Identifier::new(gen.context, "value"),
                        IntegerAttribute::new(_match_ty, lit_val).into(),
                    )])
                    .build()?,
            );
            let const_val = const_op.result(0)?.into();
            let cmp_op = block.append_operation(
                OperationBuilder::new("arith.cmpi", gen.loc())
                    .add_operands(&[match_val, const_val])
                    .add_results(&[melior::ir::r#type::IntegerType::new(gen.context, 1).into()])
                    .add_attributes(&[(
                        Identifier::new(gen.context, "predicate"),
                        IntegerAttribute::new(
                            melior::ir::r#type::IntegerType::new(gen.context, 64).into(),
                            0, // eq
                        )
                        .into(),
                    )])
                    .build()?,
            );
            cmp_op.result(0)?.into()
        }
        _ => panic!("Unsupported pattern in codegen"),
    };

    block.append_operation(
        OperationBuilder::new("cf.cond_br", gen.loc())
            .add_operands(&[cond_val])
            .add_successors(&[&*then_block, &*else_block])
            .add_attributes(&[(
                Identifier::new(gen.context, "operandSegmentSizes"),
                melior::ir::attribute::DenseI32ArrayAttribute::new(gen.context, &[1, 0, 0]).into(),
            )])
            .build()?,
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
                let payload_ty =
                    melior::ir::Type::parse(gen.context, &payload_ty_str).ok_or_else(|| {
                        crate::codegen::lower::LowerError::ParseType(
                            "Type::parse failed".to_string(),
                        )
                    })?;
                let extract_payload_op = OperationBuilder::new("llvm.extractvalue", gen.loc())
                    .add_operands(&[match_val])
                    .add_results(&[payload_ty])
                    .add_attributes(&[(
                        Identifier::new(gen.context, "position"),
                        melior::ir::attribute::DenseI64ArrayAttribute::new(gen.context, &[1])
                            .into(),
                    )])
                    .build()?;
                let payload_val = then_block
                    .append_operation(extract_payload_op)
                    .result(0)?
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
                .build()?,
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
                    .build()?;
                let idx_val = block.append_operation(cst_op).result(0)?.into();

                let dim_op = OperationBuilder::new("memref.dim", gen.loc())
                    .add_operands(&[tensor_val, idx_val])
                    .add_results(&[index_ty])
                    .build()?;
                alloc_operands.push(block.append_operation(dim_op).result(0)?.into());
            }
        }

        let alloc_op = OperationBuilder::new("memref.alloc", gen.loc())
            .add_operands(&alloc_operands)
            .add_attributes(&[(
                Identifier::new(gen.context, "operandSegmentSizes"),
                DenseI32ArrayAttribute::new(gen.context, &[alloc_operands.len() as i32, 0]).into(),
            )])
            .add_results(&[tensor_ty])
            .build()?;
        let out_val = block.append_operation(alloc_op).result(0)?.into();

        // 2. Generate linalg.generic
        let region = Region::new();
        let block_generic = melior::ir::Block::new(&[
            (
                Type::parse(gen.context, el_ty_str).ok_or_else(|| {
                    crate::codegen::lower::LowerError::ParseType("Type::parse failed".to_string())
                })?,
                gen.loc(),
            ),
            (
                Type::parse(gen.context, el_ty_str).ok_or_else(|| {
                    crate::codegen::lower::LowerError::ParseType("Type::parse failed".to_string())
                })?,
                gen.loc(),
            ),
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
            .build()?;
        let c1 = block.append_operation(c1_op).result(0)?.into();

        let alloca_op = OperationBuilder::new("llvm.alloca", gen.loc())
            .add_operands(&[c1])
            .add_results(&[ptr_ty])
            .add_attributes(&[(
                Identifier::new(gen.context, "elem_type"),
                TypeAttribute::new(closure_ty).into(),
            )])
            .build()?;
        let alloca_ptr = block.append_operation(alloca_op).result(0)?.into();

        let store_op = OperationBuilder::new("llvm.store", gen.loc())
            .add_operands(&[closure_val, alloca_ptr])
            .build()?;
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
            .add_operands(&[
                alloca_ptr,
                block_generic
                    .argument(0)
                    .map_err(|_| {
                        crate::codegen::lower::LowerError::from(format!(
                            "Missing argument {} block",
                            0
                        ))
                    })?
                    .into(),
            ])
            .add_attributes(&[(
                Identifier::new(gen.context, "callee"),
                FlatSymbolRefAttribute::new(gen.context, &invoke_method).into(),
            )])
            .add_results(&[Type::parse(gen.context, el_ty_str).ok_or_else(|| {
                crate::codegen::lower::LowerError::ParseType("Type::parse failed".to_string())
            })?])
            .build()?;

        let call_res = block_generic.append_operation(call_op).result(0)?.into();

        let yield_op = OperationBuilder::new("linalg.yield", gen.loc())
            .add_operands(&[call_res])
            .build()?;
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
            .build()?;
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

    // Scalar fast-path (#185): `printMemref*` only accepts ranked memrefs/tensors,
    // so a bare scalar (`print(x)` where x : f32/f64/i32/i64) must route through the
    // scalar `print_*` runtime helpers instead of the memref path below.
    // Narrow scalars have no print helper of their own, so widen to one that
    // does rather than falling through to the memref path, which rejects a bare
    // scalar type and reports it as an unsupported element type (#320). f16
    // arithmetic, tensors and matmul all execute; only printing was missing.
    let mut arg_ty_str = arg_ty.to_string();
    let widened: Option<(&str, &str)> = match arg_ty_str.as_str() {
        "f16" | "bf16" => Some(("arith.extf", "f32")),
        "i8" | "i16" => Some(("arith.extsi", "i32")),
        _ => None,
    };
    if let Some((op_name, wide_ty_str)) = widened {
        let wide_ty = Type::parse(gen.context, wide_ty_str)
            .ok_or_else(|| LowerError::ParseType(wide_ty_str.to_string()))?;
        let cast_op = OperationBuilder::new(op_name, gen.loc())
            .add_operands(&[arg_val])
            .add_results(&[wide_ty])
            .build()?;
        arg_val = block.append_operation(cast_op).result(0)?.into();
        arg_ty_str = wide_ty_str.to_string();
    }

    let scalar_print_fn = match arg_ty_str.as_str() {
        "i32" => Some("print_i32"),
        "i64" => Some("print_i64"),
        "f32" => Some("print_f32"),
        "f64" => Some("print_f64"),
        _ => None,
    };
    if let Some(fn_name) = scalar_print_fn {
        if !gen.functions.contains_key(fn_name) {
            let func_ty = Type::parse(gen.context, &format!("({}) -> i32", arg_ty_str))
                .ok_or_else(|| LowerError::ParseType(format!("({}) -> i32", arg_ty_str)))?;
            let func_decl = OperationBuilder::new("func.func", gen.loc())
                .add_attributes(&[
                    (
                        Identifier::new(gen.context, "sym_name"),
                        StringAttribute::new(gen.context, fn_name).into(),
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
            // Record the widened parameter type, not the source scalar's, or a
            // later call would be checked against a signature the declaration
            // does not have.
            let recorded_ty = Type::parse(gen.context, &arg_ty_str)
                .ok_or_else(|| LowerError::ParseType(arg_ty_str.clone()))?;
            gen.functions
                .insert(fn_name.to_string().into(), (gen.i32_ty, vec![recorded_ty]));
        }
        let call_op = block.append_operation(
            OperationBuilder::new("func.call", gen.loc())
                .add_operands(&[arg_val])
                .add_results(&[gen.i32_ty])
                .add_attributes(&[(
                    Identifier::new(gen.context, "callee"),
                    FlatSymbolRefAttribute::new(gen.context, fn_name).into(),
                )])
                .build()?,
        );
        return Ok((call_op.result(0)?.into(), gen.none_ty, block));
    }

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

/// The eight `raw::` transfer-lowering primitives (#353 A3), emitted in place.
///
/// These are reachable only from an `impl transfer` body inlined at a transfer
/// site -- the checker refuses `raw::` anywhere else (E6017) and discharges the
/// bounds, barrier-shape, and async obligations before codegen ever runs. The
/// lowering here is deliberately direct: loads and stores against the tile
/// memrefs, `gpu.barrier` for the fence, thread identity for the work split --
/// never a `func.call`, which would silently cost the enclosing kernel its
/// device twin (`isDeviceLowerableDialect` excludes func).
///
/// Tiles are FLAT-indexed at the surface (`raw::load(t, i)` with
/// `i < raw::extent(t)`) while the memrefs are rank-N, so a linear index is
/// delinearized here with the tile's static dims (row-major div/mod chain).
/// Static dims are a checked fact: the A2 prover obligations only close for
/// statically shaped tiles, and the device path refuses dynamic shared tiles.
///
/// `raw::async_copy` lowers to its synchronous fallback (an element load+store)
/// and `raw::async_wait` to nothing: the copy-engine capability is declared and
/// gated (E6020), but the engine itself is not driven until the nvgpu route
/// lands -- the fallback preserves the contract's semantics exactly, one
/// element per call, ordered before the wait.
pub(crate) fn lower_raw_primitive<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: melior::ir::BlockRef<'c, 'c>,
    prim: &str,
    args: &[Expr],
) -> Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError> {
    let index_ty = Type::index(gen.context);
    let i64_ty = gen.i64_ty;

    // The tile's static dims, read off the AST type the checker validated.
    let tile_dims = |gen: &MeliorGenerator<'c>, e: &Expr| -> Result<Vec<u64>, LowerError> {
        let ty = gen
            .infer_ast_type(e)
            .ok_or_else(|| LowerError::from("raw:: tile has no inferable type".to_string()))?;
        let mut inner = &ty;
        loop {
            match inner {
                syntax::Type::Tensor(_, dims, _) => {
                    let mut out = Vec::new();
                    for d in dims {
                        let syntax::Expr::Number(n) = d else {
                            return Err(LowerError::from(
                                "raw:: tile has a non-static dim; the device path \
                                 refuses dynamic shared tiles (#353 A3)"
                                    .to_string(),
                            ));
                        };
                        out.push(n.value.as_ref().parse::<u64>().map_err(|_| {
                            LowerError::from("raw:: tile dim is not an integer".to_string())
                        })?);
                    }
                    return Ok(out);
                }
                syntax::Type::Borrow { inner: b, .. }
                | syntax::Type::Pointer(b, _, _)
                | syntax::Type::Pinned(b, _)
                | syntax::Type::Ref(b, _) => inner = b,
                _ => {
                    return Err(LowerError::from(
                        "raw:: tile argument is not tensor-typed".to_string(),
                    ))
                }
            }
        }
    };
    let elem_mlir = |gen: &MeliorGenerator<'c>, e: &Expr| -> Result<Type<'c>, LowerError> {
        let ty = gen
            .infer_ast_type(e)
            .ok_or_else(|| LowerError::from("raw:: tile has no inferable type".to_string()))?;
        let mut inner = &ty;
        loop {
            match inner {
                syntax::Type::Tensor(el, _, _) => {
                    let s = match el {
                        ElementType::F16 => "f16",
                        ElementType::F32 => "f32",
                        ElementType::F64 => "f64",
                        ElementType::BF16 => "bf16",
                        ElementType::I8 | ElementType::U8 => "i8",
                        ElementType::I16 | ElementType::U16 => "i16",
                        ElementType::I32 | ElementType::U32 => "i32",
                        ElementType::I64 | ElementType::U64 => "i64",
                        other => {
                            return Err(LowerError::from(format!(
                                "raw:: has no lowering for {other:?} tiles"
                            )))
                        }
                    };
                    return Ok(Type::parse(gen.context, s).unwrap());
                }
                syntax::Type::Borrow { inner: b, .. }
                | syntax::Type::Pointer(b, _, _)
                | syntax::Type::Pinned(b, _)
                | syntax::Type::Ref(b, _) => inner = b,
                _ => {
                    return Err(LowerError::from(
                        "raw:: tile argument is not tensor-typed".to_string(),
                    ))
                }
            }
        }
    };
    // A zero-result primitive still returns a value to the expression walk: the
    // same index-0 dummy the block-expression lowering uses.
    let unit = |gen: &MeliorGenerator<'c>,
                b: melior::ir::BlockRef<'c, 'c>|
     -> Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError> {
        let dummy = OperationBuilder::new("arith.constant", gen.loc())
            .add_attributes(&[(
                Identifier::new(gen.context, "value"),
                IntegerAttribute::new(Type::index(gen.context), 0).into(),
            )])
            .add_results(&[Type::index(gen.context)])
            .build()?;
        let r = b.append_operation(dummy);
        Ok((r.result(0)?.into(), gen.none_ty, b))
    };
    let const_index = |gen: &MeliorGenerator<'c>,
                       b: melior::ir::BlockRef<'c, 'c>,
                       v: u64|
     -> Result<Value<'c, 'c>, LowerError> {
        let op = OperationBuilder::new("arith.constant", gen.loc())
            .add_attributes(&[(
                Identifier::new(gen.context, "value"),
                IntegerAttribute::new(index_ty, v as i64).into(),
            )])
            .add_results(&[index_ty])
            .build()?;
        Ok(b.append_operation(op).result(0)?.into())
    };
    // Flat i64 index -> per-dim index values, row-major: idx_k = (i / stride_k) % d_k.
    let delinearize = |gen: &mut MeliorGenerator<'c>,
                       b: melior::ir::BlockRef<'c, 'c>,
                       lin_in: Value<'c, 'c>,
                       lin_ty: Type<'c>,
                       dims: &[u64]|
     -> Result<Vec<Value<'c, 'c>>, LowerError> {
        // The index expression arrives as i64 from raw::extent/lane arithmetic,
        // but a for-loop induction variable is already `index` on this path --
        // an index->index cast is invalid IR, so cast only when needed.
        let lin: Value = if lin_ty == index_ty {
            lin_in
        } else {
            let cast = OperationBuilder::new("arith.index_cast", gen.loc())
                .add_operands(&[lin_in])
                .add_results(&[index_ty])
                .build()?;
            b.append_operation(cast).result(0)?.into()
        };
        if dims.len() <= 1 {
            return Ok(vec![lin]);
        }
        let mut out = Vec::new();
        let mut stride: u64 = dims.iter().product();
        for (k, d) in dims.iter().enumerate() {
            stride /= d;
            let sv = const_index(gen, b, stride)?;
            let div = OperationBuilder::new("arith.divui", gen.loc())
                .add_operands(&[lin, sv])
                .add_results(&[index_ty])
                .build()?;
            let q: Value = b.append_operation(div).result(0)?.into();
            let idx = if k == 0 {
                // The leading digit needs no mod: the checked bound i < extent
                // already caps it below dims[0].
                q
            } else {
                let dv = const_index(gen, b, *d)?;
                let rem = OperationBuilder::new("arith.remui", gen.loc())
                    .add_operands(&[q, dv])
                    .add_results(&[index_ty])
                    .build()?;
                b.append_operation(rem).result(0)?.into()
            };
            out.push(idx);
        }
        Ok(out)
    };

    match prim {
        "extent" => {
            let dims = tile_dims(gen, &args[0])?;
            let n: u64 = dims.iter().product();
            let op = OperationBuilder::new("arith.constant", gen.loc())
                .add_attributes(&[(
                    Identifier::new(gen.context, "value"),
                    IntegerAttribute::new(i64_ty, n as i64).into(),
                )])
                .add_results(&[i64_ty])
                .build()?;
            let r = block.append_operation(op);
            Ok((r.result(0)?.into(), i64_ty, block))
        }
        "lane" | "lanes" => {
            // Thread identity, not a constant: correct under any launch geometry,
            // and the 1x1x1 launches of today read 0 and 1 (#251). The host clone
            // of the kernel body gets the same ops; the vx-to-llvm stage folds
            // them to 0/1 there, because the CPU fallback runs one thread.
            let op_name = if prim == "lane" {
                "gpu.thread_id"
            } else {
                "gpu.block_dim"
            };
            let dim_attr = melior::ir::Attribute::parse(gen.context, "#gpu<dim x>")
                .ok_or_else(|| LowerError::from("cannot parse #gpu<dim x>".to_string()))?;
            let op = OperationBuilder::new(op_name, gen.loc())
                .add_attributes(&[(Identifier::new(gen.context, "dimension"), dim_attr)])
                .add_results(&[index_ty])
                .build()?;
            let idx: Value = block.append_operation(op).result(0)?.into();
            let cast = OperationBuilder::new("arith.index_cast", gen.loc())
                .add_operands(&[idx])
                .add_results(&[i64_ty])
                .build()?;
            let r = block.append_operation(cast);
            Ok((r.result(0)?.into(), i64_ty, block))
        }
        "load" => {
            let (tile, _tty, block) = gen.generate_expr(&args[0], block)?;
            let (lin, lin_ty, block) = gen.generate_expr(&args[1], block)?;
            let dims = tile_dims(gen, &args[0])?;
            let idx = delinearize(gen, block, lin, lin_ty, &dims)?;
            let elem = elem_mlir(gen, &args[0])?;
            let mut operands = vec![tile];
            operands.extend(idx);
            let op = OperationBuilder::new("memref.load", gen.loc())
                .add_operands(&operands)
                .add_results(&[elem])
                .build()?;
            let r = block.append_operation(op);
            Ok((r.result(0)?.into(), elem, block))
        }
        "store" => {
            let (tile, _tty, block) = gen.generate_expr(&args[0], block)?;
            let (lin, lin_ty, block) = gen.generate_expr(&args[1], block)?;
            let (val, _vty, block) = gen.generate_expr(&args[2], block)?;
            let dims = tile_dims(gen, &args[0])?;
            let idx = delinearize(gen, block, lin, lin_ty, &dims)?;
            let mut operands = vec![val, tile];
            operands.extend(idx);
            let op = OperationBuilder::new("memref.store", gen.loc())
                .add_operands(&operands)
                .build()?;
            block.append_operation(op);
            unit(gen, block)
        }
        "barrier" => {
            let op = OperationBuilder::new("gpu.barrier", gen.loc()).build()?;
            block.append_operation(op);
            unit(gen, block)
        }
        "async_copy" => {
            // Synchronous fallback: dst[i] = src[i]. The capability is declared
            // (E6020 checked it); the engine is not driven yet -- when the nvgpu
            // route lands this becomes nvgpu.device_async_copy.
            let (dst, _dty, block) = gen.generate_expr(&args[0], block)?;
            let (src, _sty, block) = gen.generate_expr(&args[1], block)?;
            let (lin, lin_ty, block) = gen.generate_expr(&args[2], block)?;
            let sdims = tile_dims(gen, &args[1])?;
            let ddims = tile_dims(gen, &args[0])?;
            let elem = elem_mlir(gen, &args[1])?;
            let sidx = delinearize(gen, block, lin, lin_ty, &sdims)?;
            let mut load_ops = vec![src];
            load_ops.extend(sidx);
            let load = OperationBuilder::new("memref.load", gen.loc())
                .add_operands(&load_ops)
                .add_results(&[elem])
                .build()?;
            let v: Value = block.append_operation(load).result(0)?.into();
            let didx = delinearize(gen, block, lin, lin_ty, &ddims)?;
            let mut store_ops = vec![v, dst];
            store_ops.extend(didx);
            let store = OperationBuilder::new("memref.store", gen.loc())
                .add_operands(&store_ops)
                .build()?;
            block.append_operation(store);
            unit(gen, block)
        }
        "async_wait" => unit(gen, block),
        other => Err(LowerError::from(format!(
            "raw::{other} has no lowering; the checker should have refused it"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use syntax::{Expr, NumberExpr, Span, Topology};

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
        assert_eq!(topology_to_i32(&Topology::gpu(0)), 500);
        assert_eq!(topology_to_i32(&Topology::CpuAvx512), 600);
        assert_eq!(topology_to_i32(&Topology::CpuNeon), 700);
        assert_eq!(topology_to_i32(&Topology::Current), 0);
    }

    #[test]
    fn test_topology_to_i32_slice() {
        // B4 (#253): a slice's id derives from (base, start, end) in the dedicated 2000..2999
        // band — stable across calls, distinct for a different extent or base, no longer the
        // single constant every slice used to collapse onto.
        let slice = |base: &str, start: &str, end: &str| {
            Topology::Slice(
                Box::new(Topology::NPU(make_num_expr(base))),
                make_num_expr(start),
                make_num_expr(end),
            )
        };
        let nvl144 = topology_to_i32(&slice("0", "0", "144"));
        let nvl72 = topology_to_i32(&slice("0", "0", "72"));
        let other_base = topology_to_i32(&slice("1", "0", "144"));
        assert!((2000..3000).contains(&nvl144), "banded: {nvl144}");
        assert!((2000..3000).contains(&nvl72), "banded: {nvl72}");
        assert_ne!(nvl144, nvl72, "distinct extents get distinct ids");
        assert_ne!(nvl144, other_base, "distinct bases get distinct ids");
        assert_eq!(
            nvl144,
            topology_to_i32(&slice("0", "0", "144")),
            "stable across calls"
        );
    }

    #[test]
    fn test_topology_to_i32_npu_non_numeric_falls_back() {
        // NPU with a non-numeric expr should fall back to 100
        let ident_expr = Box::new(Expr::Identifier(syntax::IdentifierExpr {
            name: "i".into(),
            span: Span::default(),
        }));
        assert_eq!(topology_to_i32(&Topology::NPU(ident_expr)), 100);
    }
}
