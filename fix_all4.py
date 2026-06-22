import re

files = [
    "src/codegen/lower/expr.rs",
    "src/codegen/lower/control_flow.rs",
    "src/codegen/lower/stmt.rs",
    "src/codegen/lower/tensors.rs",
    "src/codegen/generator.rs",
    "src/codegen/lower/mod.rs"
]

for file in files:
    with open(file, "r") as f:
        content = f.read()

    # Replace Value<'c, 'c> with Value<'c, 'a> in signatures
    content = content.replace("Value<'c, 'c>", "Value<'c, 'a>")
    
    with open(file, "w") as f:
        f.write(content)

# Specific fixes for control_flow.rs
with open("src/codegen/lower/control_flow.rs", "r") as f:
    content = f.read()

content = content.replace("let LoopStmt { block: loop_body, .. } = self;", "let LoopStmt { body: loop_body, .. } = self;")
content = content.replace("let ForLoopStmt { item, block: body_stmts, .. } = self;", "let ForLoopStmt { iter, iterable, body: body_stmts, .. } = self;")
content = content.replace("gen.env.insert(item.name.clone(), (iter_val, gen.index_ty));", "gen.env.insert(iter.clone().into(), (iter_val, gen.index_ty));")
content = content.replace("ast::types::Span::new(0, 0)", "ast::types::Span::default()")
content = content.replace(".add_successors(&[continue_b])", ".add_successors(&[&continue_b])")

with open("src/codegen/lower/control_flow.rs", "w") as f:
    f.write(content)

# Specific fixes for mod.rs (fix missing regions in generate_statement / generate_match_chain)
with open("src/codegen/lower/mod.rs", "r") as f:
    content = f.read()

content = content.replace("if let Some(b) = gen.generate_statement(stmt, region, then_block)? { then_block = b; }", "if let Some(b) = gen.generate_statement(stmt, region, then_block)? { then_block = b; }")
# We need then_block and else_block to be mutable BlockRefs.
content = content.replace("let then_block = melior::ir::Block::new(&[]);", "let then_block = region.append_block(melior::ir::Block::new(&[]));")
content = content.replace("let else_block = melior::ir::Block::new(&[]);", "let else_block = region.append_block(melior::ir::Block::new(&[]));")

with open("src/codegen/lower/mod.rs", "w") as f:
    f.write(content)

# Fix tensors.rs
with open("src/codegen/lower/tensors.rs", "r") as f:
    content = f.read()

content = content.replace("gen.generate_statement(stmt, &body_block)?;", "if let Some(b) = gen.generate_statement(stmt, region, body_block)? { body_block = b; }")
content = content.replace("let (val, ty, block) = gen.generate_expr(r, &body_block)?;", "let (val, ty, mut body_block) = gen.generate_expr(r, region, body_block)?;")
# Make sure body_block is a blockref appended to region
content = content.replace("let body_block = melior::ir::Block::new(&[]);", "let mut body_block = region.append_block(melior::ir::Block::new(&[]));")

with open("src/codegen/lower/tensors.rs", "w") as f:
    f.write(content)

# Fix expr.rs generate_expr missing region
with open("src/codegen/lower/expr.rs", "r") as f:
    content = f.read()

content = content.replace("let (closure_val, closure_ty, block) = gen.generate_expr(&args[1], &region, block)?;", "let (closure_val, closure_ty, block) = gen.generate_expr(&args[1], region, block)?;")
content = content.replace("gen.generate_expr(\n                &Expr::Identifier(IdentifierExpr {\n                    name: param.0.clone(),\n                    span: Span::default(),\n                }),\n                block,\n            )?;", "gen.generate_expr(\n                &Expr::Identifier(IdentifierExpr {\n                    name: param.0.clone(),\n                    span: Span::default(),\n                }),\n                region,\n                block,\n            )?;")
content = content.replace("let mut loop_block = Block::new(&[]);", "let mut loop_block = region.append_block(Block::new(&[]));")

with open("src/codegen/lower/expr.rs", "w") as f:
    f.write(content)

