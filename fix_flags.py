import re

with open("src/codegen/lower/control_flow.rs", "r") as f:
    content = f.read()

content = content.replace("break_flags", "break_blocks")
content = content.replace("continue_flags", "continue_blocks")

with open("src/codegen/lower/control_flow.rs", "w") as f:
    f.write(content)
