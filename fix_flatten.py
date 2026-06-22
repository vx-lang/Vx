import re

with open("src/codegen/generator.rs", "r") as f:
    gen_rs = f.read()

gen_rs = gen_rs.replace(
    "pub fn flatten_indices(\n        &mut self,\n        expr: &Expr,\n        block: &melior::ir::Block<'c>,\n    ) -> Option<(Value<'c, 'c>, Type<'c>, Vec<Value<'c, 'c>>)> {",
    "pub fn flatten_indices<'a>(\n        &mut self,\n        expr: &Expr,\n        region: &'a melior::ir::Region<'c>,\n        mut block: melior::ir::BlockRef<'c, 'a>,\n    ) -> Option<(Value<'c, 'c>, Type<'c>, Vec<Value<'c, 'c>>, melior::ir::BlockRef<'c, 'a>)> {"
)
gen_rs = gen_rs.replace(
    "let (base_val, base_ty, mut indices) = self.flatten_indices(base, block)?;",
    "let (base_val, base_ty, mut indices, mut block) = self.flatten_indices(base, region, block)?;"
)
gen_rs = gen_rs.replace(
    "let (idx_val, _) = self.generate_expr(idx, block).ok()?;",
    "let (idx_val, _, b) = self.generate_expr(idx, region, block).ok()?;\n                block = b;"
)
gen_rs = gen_rs.replace(
    "Some((base_val, base_ty, indices))",
    "Some((base_val, base_ty, indices, block))"
)
gen_rs = gen_rs.replace(
    "let (val, ty) = self.generate_expr(expr, block).ok()?;",
    "let (val, ty, block) = self.generate_expr(expr, region, block).ok()?;"
)
gen_rs = gen_rs.replace(
    "Some((val, ty, Vec::new()))",
    "Some((val, ty, Vec::new(), block))"
)
with open("src/codegen/generator.rs", "w") as f:
    f.write(gen_rs)

files = [
    "src/codegen/lower/stmt.rs",
    "src/codegen/lower/expr.rs",
]

for file in files:
    with open(file, "r") as f:
        content = f.read()

    # Find:
    # if let Some((mem_val, mem_ty, indices)) = gen.flatten_indices(
    #     expr,
    #     &block,
    # )
    content = re.sub(
        r"if let Some\(\(([^,]+),\s*([^,]+),\s*([^)]+)\)\)\s*=\s*gen\.flatten_indices\(\s*([^,]+),\s*&block,\s*\)",
        r"if let Some((\1, \2, \3, b)) = gen.flatten_indices(\4, region, block) {\n                block = b;",
        content
    )
    
    # Also expr.rs has `gen.flatten_indices(base, block)`
    content = re.sub(
        r"let \(([^,]+),\s*([^,]+),\s*([^)]+)\)\s*=\s*gen\.flatten_indices\(\s*([^,]+),\s*&block,\s*\)\?;",
        r"let (\1, \2, \3, mut block) = gen.flatten_indices(\4, region, block)?;",
        content
    )

    with open(file, "w") as f:
        f.write(content)
