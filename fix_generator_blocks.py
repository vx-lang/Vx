import os
import re

def fix_file(filepath):
    with open(filepath, "r") as f:
        content = f.read()

    # In mod.rs
    if "mod.rs" in filepath:
        content = re.sub(r"fn lower\(&self, gen: &mut MeliorGenerator<'c>, mut block: melior::ir::BlockRef<'c, 'c>\) -> Self::Output;",
                         r"fn lower(&self, gen: &mut MeliorGenerator<'c>, block: melior::ir::BlockRef<'c, 'c>) -> Self::Output;", content)

    # In expr.rs
    if "expr.rs" in filepath:
        content = re.sub(r"_block: &melior::ir::Block<'c>", r"mut _block: melior::ir::BlockRef<'c, 'c>", content)

    with open(filepath, "w") as f:
        f.write(content)

for filename in ["control_flow.rs", "expr.rs", "stmt.rs", "tensors.rs", "mod.rs"]:
    filepath = os.path.join("src/codegen/lower", filename)
    if os.path.exists(filepath):
        fix_file(filepath)

def fix_generator(filepath):
    with open(filepath, "r") as f:
        content = f.read()
    
    # Change mut block: &melior::ir::Block<'c> to mut block: melior::ir::BlockRef<'c, 'c>
    content = re.sub(r"mut block: &melior::ir::Block<'c>", r"mut block: melior::ir::BlockRef<'c, 'c>", content)
    content = re.sub(r"block: &melior::ir::Block<'c>", r"mut block: melior::ir::BlockRef<'c, 'c>", content)
    
    with open(filepath, "w") as f:
        f.write(content)

fix_generator("src/codegen/generator.rs")
