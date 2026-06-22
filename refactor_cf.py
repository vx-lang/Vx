import os
import re

def process_file(filepath):
    with open(filepath, "r") as f:
        content = f.read()

    # Change impl<'c> to impl<'c, 'a>
    content = re.sub(r"impl<'c>\s+LowerToMelior<'c>\s+for", r"impl<'c, 'a> LowerToMelior<'c, 'a> for", content)
    
    # Change block: BlockRef<'c, 'c> to region: &'a melior::ir::Region<'c>, block: melior::ir::BlockRef<'c, 'a>
    content = re.sub(
        r"block:\s*melior::ir::BlockRef<'c,\s*'c>",
        r"region: &'a melior::ir::Region<'c>,\n        block: melior::ir::BlockRef<'c, 'a>",
        content
    )
    
    # Change Option<melior::ir::BlockRef<'c, 'c>> to Option<melior::ir::BlockRef<'c, 'a>>
    content = content.replace("Option<melior::ir::BlockRef<'c, 'c>>", "Option<melior::ir::BlockRef<'c, 'a>>")
    
    # Change Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>
    content = content.replace("melior::ir::BlockRef<'c, 'c>", "melior::ir::BlockRef<'c, 'a>")
    
    with open(filepath, "w") as f:
        f.write(content)

for filename in ["expr.rs", "stmt.rs", "control_flow.rs", "tensors.rs"]:
    filepath = os.path.join("src/codegen/lower", filename)
    process_file(filepath)

# Handle mod.rs specially
with open("src/codegen/lower/mod.rs", "r") as f:
    mod_content = f.read()

# Already changed trait definition manually, but just in case:
mod_content = mod_content.replace(
    "pub trait LowerToMelior<'c, 'a> {\n    type Output;\n    fn lower(\n        &self,\n        gen: &mut MeliorGenerator<'c>,\n        region: &'a melior::ir::Region<'c>,\n        block: &'a melior::ir::Block<'c>,\n    ) -> Self::Output;\n}",
    "pub trait LowerToMelior<'c, 'a> {\n    type Output;\n    fn lower(\n        &self,\n        gen: &mut MeliorGenerator<'c>,\n        region: &'a melior::ir::Region<'c>,\n        block: melior::ir::BlockRef<'c, 'a>,\n    ) -> Self::Output;\n}"
)

# Update flatten_indices
mod_content = mod_content.replace(
    "pub(crate) fn flatten_indices<'c>(\n        &mut self,\n        indices: &[Expr],\n        mut block: melior::ir::BlockRef<'c, 'c>,\n    ) -> Result<(\n        Vec<melior::ir::Value<'c, 'c>>,\n        Vec<melior::ir::Type<'c>>,\n        melior::ir::BlockRef<'c, 'c>,\n    ), LowerError>",
    "pub(crate) fn flatten_indices<'a>(\n        &mut self,\n        indices: &[Expr],\n        region: &'a melior::ir::Region<'c>,\n        mut block: melior::ir::BlockRef<'c, 'a>,\n    ) -> Result<(\n        Vec<melior::ir::Value<'c, 'c>>,\n        Vec<melior::ir::Type<'c>>,\n        melior::ir::BlockRef<'c, 'a>,\n    ), LowerError>"
)

# Update generate_match_chain
mod_content = mod_content.replace(
    "pub fn generate_match_chain<'c>(\n    gen: &mut MeliorGenerator<'c>,\n    arms: &[MatchArm],\n    match_val: melior::ir::Value<'c, 'c>,\n    _match_ty: melior::ir::Type<'c>,\n    mut block: melior::ir::BlockRef<'c, 'c>,\n) -> Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>",
    "pub fn generate_match_chain<'c, 'a>(\n    gen: &mut MeliorGenerator<'c>,\n    arms: &[MatchArm],\n    match_val: melior::ir::Value<'c, 'c>,\n    _match_ty: melior::ir::Type<'c>,\n    region: &'a melior::ir::Region<'c>,\n    mut block: melior::ir::BlockRef<'c, 'a>,\n) -> Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'a>), LowerError>"
)

with open("src/codegen/lower/mod.rs", "w") as f:
    f.write(mod_content)

# Handle generator.rs
with open("src/codegen/generator.rs", "r") as f:
    gen_content = f.read()

gen_content = gen_content.replace(
    "pub fn generate_statement(\n        &mut self,\n        stmt: &Statement,\n        block: melior::ir::BlockRef<'c, 'c>,\n    ) -> Result<Option<melior::ir::BlockRef<'c, 'c>>, LowerError>",
    "pub fn generate_statement<'a>(\n        &mut self,\n        stmt: &Statement,\n        region: &'a melior::ir::Region<'c>,\n        block: melior::ir::BlockRef<'c, 'a>,\n    ) -> Result<Option<melior::ir::BlockRef<'c, 'a>>, LowerError>"
)

gen_content = gen_content.replace(
    "pub fn generate_expr(\n        &mut self,\n        expr: &Expr,\n        block: melior::ir::BlockRef<'c, 'c>,\n    ) -> Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError>",
    "pub fn generate_expr<'a>(\n        &mut self,\n        expr: &Expr,\n        region: &'a melior::ir::Region<'c>,\n        block: melior::ir::BlockRef<'c, 'a>,\n    ) -> Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'a>), LowerError>"
)

# Update lower_function loop
gen_content = gen_content.replace(
    "if let Some(b) = self.generate_statement(stmt, current_block)? {",
    "if let Some(b) = self.generate_statement(stmt, &region, current_block)? {"
)

with open("src/codegen/generator.rs", "w") as f:
    f.write(gen_content)
