import re

with open("src/codegen/lower/control_flow.rs", "r") as f:
    content = f.read()

# I need to insert "let ret_ty = gen.expected_type.unwrap_or(gen.f32_ty);" right before "let dummy_op = if ret_ty.to_string() == "f32" {"
# But wait, we actually need it before "if then_block.is_empty() && else_block.is_none() {"
# Let's just find "if then_block.is_empty() && else_block.is_none() {" and prepend it.

content = content.replace("if then_block.is_empty() && else_block.is_none() {", "let ret_ty = gen.expected_type.unwrap_or(gen.f32_ty);\n        if then_block.is_empty() && else_block.is_none() {")

with open("src/codegen/lower/control_flow.rs", "w") as f:
    f.write(content)
