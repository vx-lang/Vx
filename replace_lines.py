with open("src/codegen/lower/mod.rs", "r") as f:
    lines = f.readlines()

new_content = """pub fn generate_match_chain<'c>(
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

    generate_match_chain(gen, &arms[1..], match_val, _match_ty, else_block, merge_block)?;

    Ok(block)
}
"""

lines = lines[:234] + [new_content] + lines[544:]

with open("src/codegen/lower/mod.rs", "w") as f:
    f.writelines(lines)
