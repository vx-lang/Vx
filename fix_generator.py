import re

filepath = "src/codegen/generator.rs"
with open(filepath, "r") as f:
    content = f.read()

# Fix the escaped quotes
content = content.replace(r'\"func.return\"', '"func.return"')
content = content.replace(r'\"llvm.emit_c_interface\"', '"llvm.emit_c_interface"')
content = content.replace(r'\"func.func\"', '"func.func"')
content = content.replace(r'\"Macros should be expanded before codegen\"', '"Macros should be expanded before codegen"')
content = content.replace(r'\"Generic element type should be instantiated before codegen\"', '"Generic element type should be instantiated before codegen"')
content = content.replace(r'\"!llvm.ptr\"', '"!llvm.ptr"')
content = content.replace(r'\"!llvm.struct<\"{}\", (i32, {})>\"', '"!llvm.struct<\"{}\", (i32, {})>"')

# Wait, `!llvm.struct<\"{}\", (i32, {})>` inside format! might be wrong if it replaced quotes!
# Let me just checkout generator.rs and fix it cleanly instead.
