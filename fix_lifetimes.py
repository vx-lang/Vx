import re

# 1. Update LowerToMelior trait in mod.rs
with open("src/codegen/lower/mod.rs", "r") as f:
    content = f.read()
content = content.replace(
    "type Output<'a> = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'a>), LowerError> where 'c: 'a;",
    "type Output<'a> = Result<(Value<'c, 'a>, Type<'c>, melior::ir::BlockRef<'c, 'a>), LowerError> where 'c: 'a;"
)
with open("src/codegen/lower/mod.rs", "w") as f:
    f.write(content)

# 2. Update type Output<'a> in expr.rs, stmt.rs, control_flow.rs, tensors.rs
for file in ["src/codegen/lower/expr.rs", "src/codegen/lower/stmt.rs", "src/codegen/lower/control_flow.rs", "src/codegen/lower/tensors.rs"]:
    with open(file, "r") as f:
        content = f.read()
    content = content.replace(
        "type Output<'a> = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'a>), LowerError> where 'c: 'a;",
        "type Output<'a> = Result<(Value<'c, 'a>, Type<'c>, melior::ir::BlockRef<'c, 'a>), LowerError> where 'c: 'a;"
    )
    with open(file, "w") as f:
        f.write(content)

# 3. Update generator.rs generate_expr and flatten_indices
with open("src/codegen/generator.rs", "r") as f:
    content = f.read()
content = content.replace(
    "-> Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'a>), LowerError>",
    "-> Result<(Value<'c, 'a>, Type<'c>, melior::ir::BlockRef<'c, 'a>), LowerError>"
)
content = content.replace(
    "-> Option<(Value<'c, 'c>, Type<'c>, Vec<Value<'c, 'c>>, melior::ir::BlockRef<'c, 'a>)>",
    "-> Option<(Value<'c, 'a>, Type<'c>, Vec<Value<'c, 'a>>, melior::ir::BlockRef<'c, 'a>)>"
)
# Also fix index pushing in flatten_indices
content = content.replace("mut indices = Vec::new();", "let mut indices = Vec::new();") # wait, might not be needed
with open("src/codegen/generator.rs", "w") as f:
    f.write(content)

# 4. Update stmt.rs env inserts to transmute
with open("src/codegen/lower/stmt.rs", "r") as f:
    content = f.read()
content = re.sub(
    r"gen\.env\.insert\(([^,]+),\s*\(([^,]+),\s*([^)]+)\)\);",
    r"gen.env.insert(\1, (unsafe { std::mem::transmute(\2) }, \3));",
    content
)
with open("src/codegen/lower/stmt.rs", "w") as f:
    f.write(content)

# 5. Update control_flow.rs env inserts to transmute
with open("src/codegen/lower/control_flow.rs", "r") as f:
    content = f.read()
content = re.sub(
    r"gen\.env\.insert\(([^,]+),\s*\(([^,]+),\s*([^)]+)\)\);",
    r"gen.env.insert(\1, (unsafe { std::mem::transmute(\2) }, \3));",
    content
)
with open("src/codegen/lower/control_flow.rs", "w") as f:
    f.write(content)

