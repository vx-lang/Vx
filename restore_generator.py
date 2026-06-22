import re

with open("src/codegen/generator.rs", "r") as f:
    content = f.read()

# Fix generate_statement
content = re.sub(
    r"pub\(crate\) fn generate_statement\(\s*&mut self,\s*stmt: &Statement,\s*block: &melior::ir::Block\<'c\>,\s*\) -> Result<\(\),\s*LowerError>",
    r"pub(crate) fn generate_statement<'a>(\n        &mut self,\n        stmt: &Statement,\n        region: &'a melior::ir::Region<'c>,\n        block: melior::ir::BlockRef<'c, 'a>,\n    ) -> Result<Option<melior::ir::BlockRef<'c, 'a>>, LowerError>",
    content
)

# Fix generate_statement body
content = re.sub(
    r"LowerToMelior::lower\(e, self, block\)",
    r"LowerToMelior::lower(e, self, region, block)",
    content
)

# Fix generate_expr
content = re.sub(
    r"pub\(crate\) fn generate_expr\(\s*&mut self,\s*expr: &Expr,\s*block: &melior::ir::Block\<'c\>,\s*\) -> Result<\(Value\<'c, 'c\>, Type\<'c\>\), LowerError>",
    r"pub(crate) fn generate_expr<'a>(\n        &mut self,\n        expr: &Expr,\n        region: &'a melior::ir::Region<'c>,\n        block: melior::ir::BlockRef<'c, 'a>,\n    ) -> Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'a>), LowerError>",
    content
)

# Fix flatten_indices
content = re.sub(
    r"pub fn flatten_indices\(\s*&mut self,\s*expr: &Expr,\s*block: &melior::ir::Block\<'c\>,\s*\) -> Option<\(Value\<'c, 'c\>, Type\<'c\>, Vec<Value\<'c, 'c\>>\)>",
    r"pub fn flatten_indices<'a>(\n        &mut self,\n        expr: &Expr,\n        region: &'a melior::ir::Region<'c>,\n        mut block: melior::ir::BlockRef<'c, 'a>,\n    ) -> Option<(Value<'c, 'c>, Type<'c>, Vec<Value<'c, 'c>>, melior::ir::BlockRef<'c, 'a>)>",
    content
)

content = content.replace("self.flatten_indices(base, block)?;", "self.flatten_indices(base, region, block)?;")
content = content.replace("self.generate_expr(idx, block).ok()?;", "self.generate_expr(idx, region, block).ok()?;")
content = content.replace("self.generate_expr(&iterable, block).ok()?;", "self.generate_expr(&iterable, region, block).ok()?;")

content = re.sub(r"Some\(\(base_val, base_ty, indices\)\)", r"Some((base_val, base_ty, indices, block))", content)

with open("src/codegen/generator.rs", "w") as f:
    f.write(content)
