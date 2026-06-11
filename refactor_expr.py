import re

with open("src/sema/expr.rs", "r") as f:
    content = f.read()

# 1. Change signature
content = content.replace("fn check_functioncall_expr(&mut self, expr: &mut Expr, consume: bool, silent: bool) -> Type {", "fn check_functioncall_expr(&mut self, expr: &mut Expr, consume: bool, silent: bool) -> Result<Type, ()> {")

# 2. Change the caller in check_expr (line ~164)
content = content.replace("Expr::FunctionCall(..) => self.check_functioncall_expr(expr, consume, silent),", "Expr::FunctionCall(..) => self.check_functioncall_expr(expr, consume, silent).unwrap_or(Type::Unknown),")

# Now we need to patch the body of check_functioncall_expr.
# Let's extract the body.
start_idx = content.find("fn check_functioncall_expr(&mut self, expr: &mut Expr, consume: bool, silent: bool) -> Result<Type, ()> {")

# Find the end of the function by counting braces.
idx = start_idx + len("fn check_functioncall_expr(&mut self, expr: &mut Expr, consume: bool, silent: bool) -> Result<Type, ()> {")
brace_count = 1
while idx < len(content) and brace_count > 0:
    if content[idx] == '{':
        brace_count += 1
    elif content[idx] == '}':
        brace_count -= 1
    idx += 1

end_idx = idx

body = content[start_idx:end_idx]

# Replace returns with Ok() or Err() inside the body
lines = body.split("\n")
new_lines = []
for line in lines:
    # If the line returns the dummy f32 error type or Unknown, change to Err(())
    if "Type::Tensor(ElementType::F32, vec![], None)" in line and "return" in line:
        new_lines.append(line.replace("return Type::Tensor(ElementType::F32, vec![], None);", "return Err(());"))
    elif "Type::Unknown" in line and "return" in line:
        new_lines.append(line.replace("return Type::Unknown;", "return Err(());"))
    # Also handle the final return at the end of blocks, e.g. `Type::Unknown` or `Type::Tensor(...)` without return
    elif line.strip() == "Type::Unknown":
        new_lines.append(line.replace("Type::Unknown", "Err(())"))
    elif line.strip() == "Type::Unknown // dummy return":
        new_lines.append(line.replace("Type::Unknown // dummy return", "Err(()) // dummy return"))
    elif line.strip() == "Type::Tensor(ElementType::F32, vec![], None)":
        new_lines.append(line.replace("Type::Tensor(ElementType::F32, vec![], None)", "Err(())"))
    elif "return Type::" in line or "return " in line:
        # Wrap the return value in Ok(...) if it's returning a Type
        # e.g., return Type::Scalar(el_ty); -> return Ok(Type::Scalar(el_ty));
        if "return Err" not in line and "return Ok" not in line and "return " in line:
            m = re.search(r'return\s+(.+);', line)
            if m:
                ret_val = m.group(1)
                new_lines.append(line.replace(f"return {ret_val};", f"return Ok({ret_val});"))
            else:
                new_lines.append(line)
    elif line.strip() == "Type::Tensor(el_ty, dims, Some(Topology::NPU(0)))" or \
         line.strip() == "Type::Struct(base_name, None)" or \
         line.strip() == "Type::Enum(base_name, None)" or \
         line.strip() == "Type::Generic(base_name, None)" or \
         line.strip() == "Type::Matrix":
        # Final expression without return
        new_lines.append(line.replace(line.strip(), f"Ok({line.strip()})"))
    elif line.strip() == "ret_ty":
        new_lines.append(line.replace("ret_ty", "Ok(ret_ty)"))
    elif line.strip() == "self.check_function_call(&callee_type, args, consume, silent, &mut type_args)":
        new_lines.append(line.replace("self.check_function_call(&callee_type, args, consume, silent, &mut type_args)", "Ok(self.check_function_call(&callee_type, args, consume, silent, &mut type_args))"))
    else:
        new_lines.append(line)

new_body = "\n".join(new_lines)
content = content[:start_idx] + new_body + content[end_idx:]

with open("src/sema/expr.rs", "w") as f:
    f.write(content)
