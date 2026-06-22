import re

# Fix control_flow.rs
with open("src/codegen/lower/control_flow.rs", "r") as f:
    content = f.read()

content = content.replace("let LoopStmt { block: loop_body, span: _ } = self;", "let LoopStmt { block: loop_body, .. } = self;")
content = content.replace("let ForLoopStmt { item, iterator, block: body_stmts, span: _ } = self;", "let ForLoopStmt { item, block: body_stmts, .. } = self;")
content = content.replace('Expr::Number(ast::expr::NumberExpr { value: "0".into(), span: "".into() })', 'Expr::Number(ast::expr::NumberExpr { value: "0".into(), ty: None, span: ast::types::Span::new(0, 0) })')
content = content.replace('Expr::Number(ast::expr::NumberExpr { value: "10".into(), span: "".into() })', 'Expr::Number(ast::expr::NumberExpr { value: "10".into(), ty: None, span: ast::types::Span::new(0, 0) })')
content = content.replace('Expr::Number(ast::expr::NumberExpr { value: "1".into(), span: "".into() })', 'Expr::Number(ast::expr::NumberExpr { value: "1".into(), ty: None, span: ast::types::Span::new(0, 0) })')
content = content.replace("IntegerAttribute::new(Type::integer(gen.context, 64), 2)", "IntegerAttribute::new(melior::ir::type::IntegerType::new(gen.context, 64).into(), 2)")
content = content.replace("gen.continue_blocks.push(header_b);", "gen.continue_blocks.push(&*header_b as *const _);")
content = content.replace("gen.break_blocks.push(merge_b);", "gen.break_blocks.push(&*merge_b as *const _);")
content = content.replace("gen.continue_blocks.push(continue_b);", "gen.continue_blocks.push(&*continue_b as *const _);")
content = content.replace("let break_b = *gen.break_blocks.last().unwrap();", "let break_b = unsafe { &**gen.break_blocks.last().unwrap() };")
content = content.replace("let continue_b = *gen.continue_blocks.last().unwrap();", "let continue_b = unsafe { &**gen.continue_blocks.last().unwrap() };")
content = content.replace(".add_successors(&[&break_b])", ".add_successors(&[break_b])")
content = content.replace(".add_successors(&[&continue_b])", ".add_successors(&[continue_b])")

with open("src/codegen/lower/control_flow.rs", "w") as f:
    f.write(content)

# Fix expr.rs
with open("src/codegen/lower/expr.rs", "r") as f:
    content = f.read()

content = content.replace('format!("memref<{}>", ty, block)', 'format!("memref<{}>", ty)')
content = content.replace('Ok((load_ref.result(0).unwrap().into(), inner_ty))', 'Ok((load_ref.result(0).unwrap().into(), inner_ty, block))')
content = content.replace('Ok((cast_ref.result(0).unwrap().into(), ptr_ty))', 'Ok((cast_ref.result(0).unwrap().into(), ptr_ty, block))')
content = content.replace('Ok((addressof_ref.result(0).unwrap().into(), ptr_ty))', 'Ok((addressof_ref.result(0).unwrap().into(), ptr_ty, block))')
content = content.replace('Ok((dummy_ref.result(0).unwrap().into(), none_ty))', 'Ok((dummy_ref.result(0).unwrap().into(), none_ty, block))')
content = content.replace('Ok((bin_ref.result(0).unwrap().into(), ret_ty))', 'Ok((bin_ref.result(0).unwrap().into(), ret_ty, block))')
content = content.replace('Ok((bin_ref.result(0).unwrap().into(), final_ty))', 'Ok((bin_ref.result(0).unwrap().into(), final_ty, block))')
content = content.replace('Ok((not_ref.result(0).unwrap().into(), ty))', 'Ok((not_ref.result(0).unwrap().into(), ty, block))')
content = content.replace('Ok((ext_ref.result(0).unwrap().into(), field_ty))', 'Ok((ext_ref.result(0).unwrap().into(), field_ty, block))')
content = content.replace('Ok((alloc_ref.result(0).unwrap().into(), tensor_ty))', 'Ok((alloc_ref.result(0).unwrap().into(), tensor_ty, block))')
content = content.replace('Ok((cast2_ref.result(0).unwrap().into(), target_ty))', 'Ok((cast2_ref.result(0).unwrap().into(), target_ty, block))')
content = content.replace('Ok((call_op.result(0).unwrap().into(), gen.i32_ty))', 'Ok((call_op.result(0).unwrap().into(), gen.i32_ty, block))')
content = content.replace('Ok((const_op.result(0).unwrap().into(), size_ty))', 'Ok((const_op.result(0).unwrap().into(), size_ty, block))')
content = content.replace('Ok((load_op.result(0).unwrap().into(), vec_ty))', 'Ok((load_op.result(0).unwrap().into(), vec_ty, block))')

# Multiline Ok tuples in expr.rs that I missed
content = re.sub(r'Ok\(\(\s*block\.append_operation\(dummy_op\)\.result\(0\)\.unwrap\(\)\.into\(\),\s*none_ty,\s*\)\)', r'Ok((\n                        block.append_operation(dummy_op).result(0).unwrap().into(),\n                        none_ty,\n                        block,\n                    ))', content)
content = re.sub(r'Ok\(\(\s*block\.append_operation\(dummy_val\)\.result\(0\)\.unwrap\(\)\.into\(\),\s*Type::index\(gen\.context\),\s*\)\)', r'Ok((\n                block.append_operation(dummy_val).result(0).unwrap().into(),\n                Type::index(gen.context),\n                block,\n            ))', content)
content = re.sub(r'Ok\(\(\s*block\.append_operation\(call_op\)\.result\(0\)\.unwrap\(\)\.into\(\),\s*gen\.i32_ty,\s*\)\)', r'Ok((\n            block.append_operation(call_op).result(0).unwrap().into(),\n            gen.i32_ty,\n            block,\n        ))', content)

# if let Some((base_val, base_ty, indices)) = gen.flatten_indices(... in expr.rs
content = re.sub(r'let \(([^,]+),\s*([^,]+),\s*([^)]+)\)\s*=\s*gen\s*\n\s*\.flatten_indices', r'let (\1, \2, \3, mut block) = gen\n            .flatten_indices', content)
content = re.sub(r'if let Some\(b\) = gen\.generate_statement\(stmt, region, block\)\? \{ block = b; \} else \{ return Ok\(None\); \}', r'if let Some(b) = gen.generate_statement(stmt, region, block)? { block = b; } else { unreachable!() }', content)

with open("src/codegen/lower/expr.rs", "w") as f:
    f.write(content)

# Fix stmt.rs
with open("src/codegen/lower/stmt.rs", "r") as f:
    content = f.read()

content = content.replace(', (alloca_val, ty, block))', ', (alloca_val, ty))')
content = content.replace(', (val, ty, block))', ', (val, ty))')
content = content.replace(', (result_val, ty, block))', ', (result_val, ty))')

content = re.sub(r'if let Some\(\(([^,]+),\s*([^,]+),\s*([^)]+)\)\)\s*=\s*gen\.flatten_indices\(\s*&ast::Expr::IndexAccess', r'if let Some((\1, \2, \3, b)) = gen.flatten_indices(\n                &ast::Expr::IndexAccess', content)

with open("src/codegen/lower/stmt.rs", "w") as f:
    f.write(content)

