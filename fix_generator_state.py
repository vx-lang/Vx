import re

with open("src/codegen/generator.rs", "r") as f:
    content = f.read()

# Replace break_flags and continue_flags with break_blocks and continue_blocks
content = re.sub(
    r"pub break_flags: Vec<melior::ir::Value<'c, 'c>>,\n    pub continue_flags: Vec<melior::ir::Value<'c, 'c>>,",
    r"pub break_blocks: Vec<*const melior::ir::Block<'c>>,\n    pub continue_blocks: Vec<*const melior::ir::Block<'c>>,",
    content
)

# Update MeliorGenerator::new
content = re.sub(
    r"break_flags: Vec::new\(\),\n            continue_flags: Vec::new\(\),",
    r"break_blocks: Vec::new(),\n            continue_blocks: Vec::new(),",
    content
)

with open("src/codegen/generator.rs", "w") as f:
    f.write(content)
