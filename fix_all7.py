import sys

def replace_in_file(path, replacements):
    with open(path, "r") as f:
        content = f.read()
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
        else:
            print(f"Warning: {old} not found in {path}")
    with open(path, "w") as f:
        f.write(content)

replace_in_file("src/codegen/generator.rs", [
    ("self.generate_statement(stmt, &block)", "self.generate_statement(stmt, region, block)"),
    ("Ok(())", "Ok(Some(block))"),
    ("Some((val, ty, Vec::new()))", "Some((val, ty, Vec::new(), block))"),
    ("let (val, ty) = self.generate_expr(expr, block).ok()?;", "let (val, ty, block) = self.generate_expr(expr, region, block).ok()?;"),
    (
        "pub fn coerce_type(\n        &mut self,\n        block: &melior::ir::Block<'c>,\n        val: Value<'c, 'c>,\n        from_ty: Type<'c>,\n        to_ty: Type<'c>,\n    ) -> Value<'c, 'c> {",
        "pub fn coerce_type<'a>(\n        &mut self,\n        block: &melior::ir::BlockRef<'c, 'a>,\n        val: Value<'c, 'a>,\n        from_ty: Type<'c>,\n        to_ty: Type<'c>,\n    ) -> Value<'c, 'a> {"
    )
])

replace_in_file("src/codegen/lower/control_flow.rs", [
    ("melior::ir::type::IntegerType", "melior::ir::r#type::IntegerType"),
    ("gen.continue_blocks", "gen.continue_flags"),
    ("gen.break_blocks", "gen.break_flags"),
])

replace_in_file("src/codegen/lower/expr.rs", [
    ("Ok((load_ref.result(0).unwrap().into(), field_ty))", "Ok((load_ref.result(0).unwrap().into(), field_ty, block))"),
    ("gen.generate_expr(\n                arg,\n                block,", "gen.generate_expr(\n                arg,\n                region,\n                block,"),
    ("gen.generate_expr(arg, region, block)?;", "gen.generate_expr(arg, region, block)?;\n                //"),
    ("let (arg_val, _arg_ty, block)", "let (arg_val, _arg_ty, block)"), # This was already changed by something? wait, just match the old and new.
    ("Ok((op.result(0).unwrap().into(), gen.lower_type(ret_ty)))", "Ok((op.result(0).unwrap().into(), gen.lower_type(ret_ty), block))"),
    ("Ok((\n            block.append_operation(dummy_op).result(0).unwrap().into(),\n            gen.i32_ty,\n        ))", "Ok((\n            block.append_operation(dummy_op).result(0).unwrap().into(),\n            gen.i32_ty,\n            block,\n        ))"),
    ("Ok((\n                ptr,\n                Type::parse(gen.context, &format!(\"memref<{}>\", ty)).unwrap(),\n            ))", "Ok((\n                ptr,\n                Type::parse(gen.context, &format!(\"memref<{}>\", ty)).unwrap(),\n                block,\n            ))"),
    ("gen.coerce_type(&block, rhs_val, rhs_ty, lhs_ty)", "gen.coerce_type(&block, rhs_val, rhs_ty, lhs_ty)"),
    ("gen.coerce_type(&block, arg_val, expr_ty, field_ty)", "gen.coerce_type(&block, arg_val, expr_ty, field_ty)"),
])

replace_in_file("src/codegen/lower/stmt.rs", [
    ("gen.flatten_indices(\n                    &base,\n                    block,", "gen.flatten_indices(\n                    &base,\n                    region,\n                    block,"),
    ("gen.flatten_indices(\n                    &base_tensor,\n                    block,", "gen.flatten_indices(\n                    &base_tensor,\n                    region,\n                    block,"),
])

replace_in_file("src/codegen/lower/tensors.rs", [
    ("gen.generate_statement(stmt, region, body_block)", "gen.generate_statement(stmt, region, body_block)"),
    ("gen.generate_expr(r, region, body_block)", "gen.generate_expr(r, region, body_block)"),
    ("region.append_block(body_block);", "/* region.append_block(body_block); handled by BlockRef */"),
    ("Ok((spawn_ref.result(0).unwrap().into(), result_types[0]))", "Ok((spawn_ref.result(0).unwrap().into(), result_types[0], block))"),
    ("Ok((\n                dummy_ref.result(0).unwrap().into(),\n                Type::index(gen.context),\n            ))", "Ok((\n                dummy_ref.result(0).unwrap().into(),\n                Type::index(gen.context),\n                block,\n            ))")
])

# For tensors.rs and mod.rs `region.append_block(body_block)` errors
replace_in_file("src/codegen/lower/mod.rs", [
    ("then_region.append_block(then_block);", "/* then_region.append_block(then_block); */"),
    ("else_region.append_block(else_block);", "/* else_region.append_block(else_block); */"),
    ("let (closure_val, closure_ty, block) = gen.generate_expr(&args[1], region, block)?;", "let (closure_val, closure_ty, block) = gen.generate_expr(&args[1], region, block)?;"),
])
