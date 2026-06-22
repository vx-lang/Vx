import os
import re

def fix_file(filepath):
    with open(filepath, "r") as f:
        content = f.read()

    # Fix the double let syntax errors
    # let (tensor_val, tensor_ty, mut block) = let (_, _, b) = gen.generate_expr(&args[0], block)?; block = b;
    # => let (tensor_val, tensor_ty, mut block) = gen.generate_expr(&args[0], block)?;
    content = re.sub(
        r"let \(([^,]+),([^,]+),\s*(?:mut\s*)?block\)\s*=\s*let \(_, _, b\)\s*=\s*gen\.generate_expr\(([^,]+), block\)\?;\s*block = b;",
        r"let (\1,\2, mut block) = gen.generate_expr(\3, block)?;",
        content
    )
    
    # Also fix: let (mut val, val_ty, block) = let (_, _, b) = gen.generate_expr(expr, block)?; block = b;
    content = re.sub(
        r"let \(([^,]+),([^,]+),\s*(?:mut\s*)?block\)\s*=\s*let \(_, _, b\)\s*=\s*gen\.generate_expr\(([^,]+),\s*block\)\?;\s*block = b;",
        r"let (\1,\2, mut block) = gen.generate_expr(\3, block)?;",
        content
    )

    # Some of them don't have mut block: let (v, t, block) = let (_, _, b) = ...
    content = re.sub(
        r"let \(([^,]+),\s*([^,]+),\s*(?:mut\s*)?block\)\s*=\s*let \(_, _, b\)\s*=\s*gen\.generate_expr\((.*?),\s*block\)\?;\s*block\s*=\s*b;",
        r"let (\1, \2, mut block) = gen.generate_expr(\3, block)?;",
        content
    )

    # What if it's 4 items? let (base_val, base_ty, indices, mut block) = let (_, _, b) = gen.flatten_indices(...)
    content = re.sub(
        r"let \(([^,]+),\s*([^,]+),\s*([^,]+),\s*(?:mut\s*)?block\)\s*=\s*let \(_, _, b\)\s*=\s*gen\.flatten_indices\((.*?),\s*block\)\?;\s*block\s*=\s*b;",
        r"let (\1, \2, \3, mut block) = gen.flatten_indices(\4, block)?;",
        content
    )

    with open(filepath, "w") as f:
        f.write(content)

for filename in ["control_flow.rs", "expr.rs", "stmt.rs", "tensors.rs", "mod.rs"]:
    filepath = os.path.join("src/codegen/lower", filename)
    if os.path.exists(filepath):
        fix_file(filepath)

# Let's also fix the Ok(()) issues in stmt.rs
def fix_stmt(filepath):
    with open(filepath, "r") as f:
        content = f.read()
    content = re.sub(r"\n\s*Ok\(\(\)\)", r"\n        Ok(Some(block))", content)
    with open(filepath, "w") as f:
        f.write(content)
fix_stmt("src/codegen/lower/stmt.rs")

