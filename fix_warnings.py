import re
import os

files = [
    "src/codegen/lower/mod.rs",
    "src/codegen/lower/stmt.rs",
    "src/codegen/lower/expr.rs",
    "src/codegen/lower/tensors.rs",
    "src/codegen/lower/control_flow.rs",
    "src/codegen/generator.rs"
]

for f in files:
    with open(f, "r") as file:
        content = file.read()
    
    # warning: variable does not need to be mutable
    # `mut block` -> `block`
    # Replace function parameters `mut block: melior::ir::BlockRef<'c, 'c>`
    content = re.sub(r"mut block: melior::ir::BlockRef\<'c, 'c\>", r"block: melior::ir::BlockRef<'c, 'c>", content)
    
    # Replace tuple destructuring `(..., mut block)` -> `(..., block)`
    content = re.sub(r", mut block\)", r", block)", content)
    content = re.sub(r"let mut block =", r"let block =", content)
    
    with open(f, "w") as file:
        file.write(content)
