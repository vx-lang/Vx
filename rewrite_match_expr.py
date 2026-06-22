import re

# Fix MatchExpr call in expr.rs
with open("src/codegen/lower/expr.rs", "r") as f:
    expr_content = f.read()

match_expr_pattern = r"generate_match_chain\(gen, &self\.arms, match_val, match_ty, block\)\?;"
match_expr_replacement = """
        let parent_region = block.parent_region().unwrap();
        let merge_block = parent_region.append_block(melior::ir::Block::new(&[]));
        generate_match_chain(gen, &self.arms, match_val, match_ty, block, merge_block)?;
        let block = merge_block;
"""
expr_content = re.sub(match_expr_pattern, match_expr_replacement, expr_content)

with open("src/codegen/lower/expr.rs", "w") as f:
    f.write(expr_content)

# Delete generate_statements_with_break_guard from mod.rs
with open("src/codegen/lower/mod.rs", "r") as f:
    mod_content = f.read()

# We look for "pub(crate) fn generate_statements_with_break_guard" and delete it to the end.
mod_content = re.sub(r"pub\(crate\) fn generate_statements_with_break_guard.*", "", mod_content, flags=re.DOTALL)

with open("src/codegen/lower/mod.rs", "w") as f:
    f.write(mod_content)
