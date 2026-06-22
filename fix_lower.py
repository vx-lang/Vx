import os
import re

files = [
    "src/codegen/lower/stmt.rs",
    "src/codegen/lower/expr.rs",
    "src/codegen/lower/tensors.rs",
    "src/codegen/lower/control_flow.rs"
]

for file in files:
    with open(file, "r") as f:
        content = f.read()
    
    # 1. Update Output types
    content = content.replace(
        "type Output = Result<(), LowerError>;",
        "type Output = Result<Option<melior::ir::BlockRef<'c, 'a>>, LowerError>;"
    )
    content = content.replace(
        "type Output = Result<(Value<'c, 'c>, Type<'c>), LowerError>;",
        "type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'a>), LowerError>;"
    )
    
    # 2. Update fn lower signature
    content = content.replace(
        "fn lower(&self, gen: &mut MeliorGenerator<'c>, block: &melior::ir::Block<'c>) -> Self::Output {",
        "fn lower<'a>(&self, gen: &mut MeliorGenerator<'c>, region: &'a melior::ir::Region<'c>, mut block: melior::ir::BlockRef<'c, 'a>) -> Self::Output {"
    )

    # 3. Update return statements
    # Be careful here. This regex will find `Ok((val, ty))` and replace it. But it might match other things.
    content = re.sub(r"Ok\(\(([^,()]+),\s*([^,()]+)\)\)", r"Ok((\1, \2, block))", content)
    
    # We replace Ok(()) with Ok(Some(block)) ONLY if it is returning from lower() for a statement.
    # We can just replace all Ok(()) because inside statements that's what happens. But wait, `Result<(), LowerError>` might be returned internally. Let's see if this compiles.
    content = re.sub(r"Ok\(\(\)\)", r"Ok(Some(block))", content)
    
    # 4. Update recursive calls
    content = re.sub(r"gen\.generate_expr\(([^,]+),\s*block\)", r"gen.generate_expr(\1, region, block)", content)
    content = re.sub(r"gen\.generate_statement\(([^,]+),\s*block\)", r"gen.generate_statement(\1, region, block)", content)

    # 5. Fix Let statements that unpack `gen.generate_expr`
    content = re.sub(r"let \(([^,]+),\s*([^,]+)\)\s*=\s*gen\.generate_expr", r"let (\1, \2, block) = gen.generate_expr", content)
    content = re.sub(r"gen\.generate_statement\(([^,]+),\s*region,\s*block\)\?;", r"if let Some(b) = gen.generate_statement(\1, region, block)? { block = b; } else { return Ok(None); }", content)

    with open(file, "w") as f:
        f.write(content)
