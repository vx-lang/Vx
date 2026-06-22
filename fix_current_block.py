import re

filepath = "src/codegen/generator.rs"
with open(filepath, "r") as f:
    content = f.read()

# Replace block.append_operation with current_block.append_operation
# in the context of generate_function
content = re.sub(r"block\.append_operation\(sig_init_call\);", r"current_block.append_operation(sig_init_call);", content)
content = re.sub(r"self\.generate_statement\(stmt, block\)", r"self.generate_statement(stmt, current_block)", content)
content = re.sub(r"\{ block = b; \}", r"{ current_block = b; }", content)

# c0_op = block.append_operation
content = re.sub(r"let c0_op = block\.append_operation", r"let c0_op = current_block.append_operation", content)
content = re.sub(r"block\.append_operation\(\n\s*melior::ir::operation::OperationBuilder::new\(\"func\.return\", self\.loc\(\)\)", r"current_block.append_operation(\n                melior::ir::operation::OperationBuilder::new(\"func.return\", self.loc())", content)

content = re.sub(r"block\.append_operation\(\n\s*melior::ir::operation::OperationBuilder::new\(\"func\.return\"", r"current_block.append_operation(\n                    melior::ir::operation::OperationBuilder::new(\"func.return\"", content)

with open(filepath, "w") as f:
    f.write(content)
