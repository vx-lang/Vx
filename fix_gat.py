import re

files = [
    "src/codegen/lower/stmt.rs",
    "src/codegen/lower/expr.rs",
    "src/codegen/lower/tensors.rs",
    "src/codegen/lower/control_flow.rs",
]

for file in files:
    with open(file, "r") as f:
        content = f.read()

    # 1. Update Output definition to include <'a>
    content = content.replace(
        "type Output = Result<Option<melior::ir::BlockRef<'c, 'a>>, LowerError>;",
        "type Output<'a> = Result<Option<melior::ir::BlockRef<'c, 'a>>, LowerError> where 'c: 'a;"
    )
    content = content.replace(
        "type Output = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'a>), LowerError>;",
        "type Output<'a> = Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'a>), LowerError> where 'c: 'a;"
    )

    with open(file, "w") as f:
        f.write(content)

with open("src/codegen/lower/mod.rs", "r") as f:
    mod_content = f.read()

mod_content = mod_content.replace(
    "type Output;",
    "type Output<'a> where 'c: 'a;"
)
mod_content = mod_content.replace(
    "-> Self::Output;",
    "-> Self::Output<'a>;"
)
mod_content = mod_content.replace(
    "-> Self::Output {",
    "-> Self::Output<'a> {"
)

with open("src/codegen/lower/mod.rs", "w") as f:
    f.write(mod_content)
