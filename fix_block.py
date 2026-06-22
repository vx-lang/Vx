import re

files = [
    "src/codegen/lower/stmt.rs",
    "src/codegen/lower/expr.rs",
    "src/codegen/lower/tensors.rs",
    "src/codegen/lower/control_flow.rs",
    "src/codegen/lower/mod.rs"
]

for file in files:
    with open(file, "r") as f:
        content = f.read()

    # Change generate_statement(..., &block) -> generate_statement(..., region, block)
    content = re.sub(r"gen\.generate_statement\(([^,]+),\s*&block\)", r"gen.generate_statement(\1, region, block)", content)
    # Change generate_expr(..., &block) -> generate_expr(..., region, block)
    content = re.sub(r"gen\.generate_expr\(([^,]+),\s*&block\)", r"gen.generate_expr(\1, region, block)", content)

    # Coerce types passing block -> &block
    content = re.sub(r"gen\.coerce_type\(block,\s*", r"gen.coerce_type(&block, ", content)
    
    # Flatten indices passing block -> &block
    content = re.sub(r"gen\.flatten_indices\((.*),\s*block,\s*", r"gen.flatten_indices(\1, &block, ", content)

    with open(file, "w") as f:
        f.write(content)
