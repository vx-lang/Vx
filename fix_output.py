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

    content = content.replace("-> Self::Output {", "-> Self::Output<'a> {")

    with open(file, "w") as f:
        f.write(content)
