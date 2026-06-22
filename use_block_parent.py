import os
import re

def fix_file(filepath):
    with open(filepath, "r") as f:
        content = f.read()

    # Replace gen.current_region.unwrap().append_block with block.parent().unwrap().append_block
    content = re.sub(r"gen\.current_region\.unwrap\(\)\.append_block", r"block.parent().unwrap().append_block", content)
    
    with open(filepath, "w") as f:
        f.write(content)

for filename in ["control_flow.rs", "expr.rs", "stmt.rs", "tensors.rs", "mod.rs"]:
    filepath = os.path.join("src/codegen/lower", filename)
    if os.path.exists(filepath):
        fix_file(filepath)

# generator.rs fixes
def fix_generator(filepath):
    with open(filepath, "r") as f:
        content = f.read()
    
    content = re.sub(r"pub current_region: Option<melior::ir::RegionRef<'c, 'c>>,\n", r"", content)
    content = re.sub(r"current_region: None,\n", r"", content)
    content = re.sub(r"self\.current_region = Some\(region_ref\);\n", r"", content)
    
    # In generate_function, the initial block is fetched from region_ref
    # Let's make sure it matches what we want
    
    with open(filepath, "w") as f:
        f.write(content)

fix_generator("src/codegen/generator.rs")
