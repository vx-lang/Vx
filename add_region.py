import os
import re

def fix_file(filepath):
    with open(filepath, "r") as f:
        content = f.read()

    # mod.rs
    content = re.sub(
        r"fn lower\(&self, gen: &mut MeliorGenerator<'c>, (?:mut )?block: melior::ir::BlockRef\<'c, 'c\>\)",
        r"fn lower(&self, gen: &mut MeliorGenerator<'c>, region: &melior::ir::Region<'c>, mut block: melior::ir::BlockRef<'c, 'c>)",
        content
    )

    # expr.rs lower_map_call and lower_print_call
    content = re.sub(
        r"fn lower_map_call\<'c\>\(\s*gen: &mut MeliorGenerator\<'c\>,\s*(?:mut )?block: melior::ir::BlockRef\<'c, 'c\>,\s*args: &\[Expr\],\s*\)",
        r"fn lower_map_call<'c>(\n    gen: &mut MeliorGenerator<'c>,\n    region: &melior::ir::Region<'c>,\n    mut block: melior::ir::BlockRef<'c, 'c>,\n    args: &[Expr],\n)",
        content
    )
    content = re.sub(
        r"fn lower_print_call\<'c\>\(\s*gen: &mut MeliorGenerator\<'c\>,\s*(?:mut )?block: melior::ir::BlockRef\<'c, 'c\>,\s*args: &\[Expr\],\s*\)",
        r"fn lower_print_call<'c>(\n    gen: &mut MeliorGenerator<'c>,\n    region: &melior::ir::Region<'c>,\n    mut block: melior::ir::BlockRef<'c, 'c>,\n    args: &[Expr],\n)",
        content
    )

    # mod.rs generate_match_chain
    content = re.sub(
        r"fn generate_match_chain\<'c\>\(\s*gen: &mut MeliorGenerator\<'c\>,\s*arms: &\[ast::expr::MatchArm\],\s*match_val: Value\<'c, 'c\>,\s*match_ty: Type\<'c\>,\s*(?:mut )?block: melior::ir::BlockRef\<'c, 'c\>,\s*\)",
        r"fn generate_match_chain<'c>(\n    gen: &mut MeliorGenerator<'c>,\n    arms: &[ast::expr::MatchArm],\n    match_val: Value<'c, 'c>,\n    match_ty: Type<'c>,\n    region: &melior::ir::Region<'c>,\n    mut block: melior::ir::BlockRef<'c, 'c>,\n)",
        content
    )
    
    # generate_expr definition in generator.rs
    content = re.sub(
        r"pub\(crate\) fn generate_expr\(\s*&mut self,\s*expr: &Expr,\s*(?:mut )?block: melior::ir::BlockRef\<'c, 'c\>,\s*\)",
        r"pub(crate) fn generate_expr(\n        &mut self,\n        expr: &Expr,\n        region: &melior::ir::Region<'c>,\n        mut block: melior::ir::BlockRef<'c, 'c>,\n    )",
        content
    )

    # generate_statement definition in generator.rs
    content = re.sub(
        r"pub\(crate\) fn generate_statement\(\s*&mut self,\s*stmt: &Statement,\s*(?:mut )?block: melior::ir::BlockRef\<'c, 'c\>,\s*\)",
        r"pub(crate) fn generate_statement(\n        &mut self,\n        stmt: &Statement,\n        region: &melior::ir::Region<'c>,\n        mut block: melior::ir::BlockRef<'c, 'c>,\n    )",
        content
    )
    
    # add region back to `LowerToMelior::lower(s, self, block)`
    content = re.sub(
        r"LowerToMelior::lower\(([^,]+), self, block\)",
        r"LowerToMelior::lower(\1, self, region, block)",
        content
    )

    # add region back to calls
    content = re.sub(r"gen\.generate_expr\(([^,]+), block\)", r"gen.generate_expr(\1, region, block)", content)
    content = re.sub(r"self\.generate_statement\(([^,]+), block\)", r"self.generate_statement(\1, region, block)", content)
    
    content = re.sub(r"lower_map_call\(gen, block, args\)", r"lower_map_call(gen, region, block, args)", content)
    content = re.sub(r"lower_print_call\(gen, block, args\)", r"lower_print_call(gen, region, block, args)", content)
    content = re.sub(r"generate_match_chain\(gen, &self\.arms, match_val, match_ty, block\)", r"generate_match_chain(gen, &self.arms, match_val, match_ty, region, block)", content)
    content = re.sub(r"generate_match_chain\(gen, &arms\[1\.\.\], match_val, match_ty, else_block\)", r"generate_match_chain(gen, &arms[1..], match_val, match_ty, region, else_block)", content)

    # gen.current_region.unwrap().append_block -> region.append_block
    content = re.sub(r"gen\.current_region\.unwrap\(\)\.append_block", r"region.append_block", content)

    with open(filepath, "w") as f:
        f.write(content)

for filename in ["control_flow.rs", "expr.rs", "stmt.rs", "tensors.rs", "mod.rs"]:
    filepath = os.path.join("src/codegen/lower", filename)
    if os.path.exists(filepath):
        fix_file(filepath)
fix_file("src/codegen/generator.rs")

# Fix generator generate_function initial call
def fix_generator_init(filepath):
    with open(filepath, "r") as f:
        content = f.read()
    content = re.sub(
        r"let mut block = region_ref\.first_block\(\)\.unwrap\(\);\n\n        for stmt in &ast_func\.body {\n            if let Some\(b\) = self\.generate_statement\(stmt, block\)\? \{ block = b; \}",
        r"let mut block = region_ref.first_block().unwrap();\n\n        for stmt in &ast_func.body {\n            if let Some(b) = self.generate_statement(stmt, &*region_ref, block)? { block = b; }",
        content
    )
    with open(filepath, "w") as f:
        f.write(content)
fix_generator_init("src/codegen/generator.rs")

