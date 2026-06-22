import re

def fix_generator():
    with open("src/codegen/generator.rs", "r") as f:
        content = f.read()

    # Revert the accidental Ok(Some(block)) to Ok(())
    content = content.replace("        Ok(Some(block))\n    }\n\n    pub(crate) fn generate_function", "        Ok(())\n    }\n\n    pub(crate) fn generate_function")

    # In generate_function, append block and use mutable block
    content = content.replace("let block = Block::new(&block_args);", "let mut block = region.append_block(Block::new(&block_args));")
    
    # Fix generate_statement call in generate_function loop
    content = re.sub(
        r"self\.generate_statement\(stmt,\s*region,\s*block\)\?;",
        r"if let Some(b) = self.generate_statement(stmt, &region, block)? { block = b; }",
        content
    )

    # In flatten_indices return
    content = content.replace("Some((val, ty, Vec::new()))", "Some((val, ty, Vec::new(), block))")

    # In generate_expr
    content = content.replace("let (val, ty) = self.generate_expr(expr, block).ok()?;", "let (val, ty, block) = self.generate_expr(expr, region, block).ok()?;")

    # In MeliorGenerator struct
    if "continue_blocks: Vec<*const melior::ir::Block<'c>>" not in content:
        content = content.replace("pub break_flags: Vec<melior::ir::Value<'c, 'c>>,", "pub break_flags: Vec<melior::ir::Value<'c, 'c>>,\n    pub break_blocks: Vec<*const melior::ir::Block<'c>>,")
        content = content.replace("pub continue_flags: Vec<melior::ir::Value<'c, 'c>>,", "pub continue_flags: Vec<melior::ir::Value<'c, 'c>>,\n    pub continue_blocks: Vec<*const melior::ir::Block<'c>>,")
    
    # In MeliorGenerator::new
    if "continue_blocks: Vec::new()," not in content:
        content = content.replace("break_flags: Vec::new(),", "break_flags: Vec::new(),\n            break_blocks: Vec::new(),")
        content = content.replace("continue_flags: Vec::new(),", "continue_flags: Vec::new(),\n            continue_blocks: Vec::new(),")

    with open("src/codegen/generator.rs", "w") as f:
        f.write(content)

def fix_control_flow():
    with open("src/codegen/lower/control_flow.rs", "r") as f:
        content = f.read()
    
    content = content.replace("gen.continue_flags", "gen.continue_blocks")
    content = content.replace("gen.break_flags", "gen.break_blocks")
    content = content.replace("melior::ir::type::IntegerType", "melior::ir::r#type::IntegerType")
    
    with open("src/codegen/lower/control_flow.rs", "w") as f:
        f.write(content)

def fix_expr():
    with open("src/codegen/lower/expr.rs", "r") as f:
        content = f.read()

    # Fix return tuples
    content = content.replace("Ok((load_ref.result(0).unwrap().into(), field_ty))", "Ok((load_ref.result(0).unwrap().into(), field_ty, block))")
    content = content.replace("Ok((op.result(0).unwrap().into(), gen.lower_type(ret_ty)))", "Ok((op.result(0).unwrap().into(), gen.lower_type(ret_ty), block))")
    content = content.replace("Ok((\n            block.append_operation(dummy_op).result(0).unwrap().into(),\n            gen.i32_ty,\n        ))", "Ok((\n            block.append_operation(dummy_op).result(0).unwrap().into(),\n            gen.i32_ty,\n            block,\n        ))")
    content = content.replace("Ok((\n                ptr,\n                Type::parse(gen.context, &format!(\"memref<{}>\", ty)).unwrap(),\n            ))", "Ok((\n                ptr,\n                Type::parse(gen.context, &format!(\"memref<{}>\", ty)).unwrap(),\n                block,\n            ))")

    # Fix generate_expr calls inside expr.rs
    content = content.replace("gen.generate_expr(\n                arg,\n                block,", "gen.generate_expr(\n                arg,\n                region,\n                block,")
    content = content.replace("gen.generate_expr(\n                        arg,\n                        block,", "gen.generate_expr(\n                        arg,\n                        region,\n                        block,")
    content = content.replace("gen.generate_expr(arg, block)?;", "gen.generate_expr(arg, region, block)?;")
    
    with open("src/codegen/lower/expr.rs", "w") as f:
        f.write(content)

def fix_stmt():
    with open("src/codegen/lower/stmt.rs", "r") as f:
        content = f.read()
    
    content = content.replace("gen.flatten_indices(\n                    &base,\n                    block,", "gen.flatten_indices(\n                    &base,\n                    region,\n                    block,")
    content = content.replace("gen.flatten_indices(\n                    &base_tensor,\n                    block,", "gen.flatten_indices(\n                    &base_tensor,\n                    region,\n                    block,")

    with open("src/codegen/lower/stmt.rs", "w") as f:
        f.write(content)

def fix_tensors():
    with open("src/codegen/lower/tensors.rs", "r") as f:
        content = f.read()

    content = content.replace("gen.generate_statement(stmt, region, body_block)", "gen.generate_statement(stmt, &region, body_block)")
    content = content.replace("gen.generate_expr(r, region, body_block)", "gen.generate_expr(r, &region, body_block)")
    content = content.replace("region.append_block(body_block);", "/* region.append_block(body_block); handled by BlockRef */")
    content = content.replace("Ok((spawn_ref.result(0).unwrap().into(), result_types[0]))", "Ok((spawn_ref.result(0).unwrap().into(), result_types[0], block))")
    content = content.replace("Ok((\n                dummy_ref.result(0).unwrap().into(),\n                Type::index(gen.context),\n            ))", "Ok((\n                dummy_ref.result(0).unwrap().into(),\n                Type::index(gen.context),\n                block,\n            ))")

    with open("src/codegen/lower/tensors.rs", "w") as f:
        f.write(content)

def fix_mod():
    with open("src/codegen/lower/mod.rs", "r") as f:
        content = f.read()

    content = content.replace("then_region.append_block(then_block);", "/* then_region.append_block(then_block); */")
    content = content.replace("else_region.append_block(else_block);", "/* else_region.append_block(else_block); */")
    content = content.replace("let (closure_val, closure_ty, block) = gen.generate_expr(&args[1], region, block)?;", "let (closure_val, closure_ty, block) = gen.generate_expr(&args[1], region, block)?;")

    # Fix generate_match_chain blocks
    content = re.sub(
        r"let then_block = region.append_block\(melior::ir::Block::new\(&\[\]\)\);",
        r"let mut then_block = region.append_block(melior::ir::Block::new(&[]));",
        content
    )
    content = re.sub(
        r"let else_block = region.append_block\(melior::ir::Block::new\(&\[\]\)\);",
        r"let mut else_block = region.append_block(melior::ir::Block::new(&[]));",
        content
    )

    with open("src/codegen/lower/mod.rs", "w") as f:
        f.write(content)

fix_generator()
fix_control_flow()
fix_expr()
fix_stmt()
fix_tensors()
fix_mod()
