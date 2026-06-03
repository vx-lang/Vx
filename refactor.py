import re

def process():
    boundaries = [
        (129, "Identifier"), (234, "EnumVariant"), (333, "Number"), (338, "Number_2"),
        (343, "StringLiteral"), (348, "Transfer"), (430, "ComptimeBlock"),
        (444, "SpawnOn"), (483, "If"), (521, "FunctionCall"), (972, "Array"),
        (978, "MemberAccess"), (1045, "IndexAccess"), (1062, "MethodCall"),
        (1441, "BinaryOp"), (1478, "RelationalOp"), (1494, "LogicalOp"),
        (1510, "MemorySpace"), (1513, "UnaryOp"), (1524, "Borrow"),
        (1558, "Dereference"), (1582, "UnsafeBlock"), (1598, "StructInit"),
        (1674, "Grad"), (1711, "Vjp"), (1748, "Jvp"), (1785, "Range"),
        (1800, "Match"), (1872, "VecMacro"), (1927, "Closure"), (1980, "MacroCall")
    ]
    
    with open('src/sema/expr.rs', 'r') as f:
        lines = f.readlines()
        
    helper_methods = []
    new_match_lines = ["        match expr {\n"]
    
    for i, (start_line, name) in enumerate(boundaries):
        end_line = boundaries[i+1][0] - 1 if i+1 < len(boundaries) else 1984
        
        # 0-indexed
        start_idx = start_line - 1
        end_idx = end_line - 1
        
        branch_lines = lines[start_idx:end_idx+1]
        
        # Determine actual variant name
        first_line = branch_lines[0].strip()
        m = re.match(r'Expr::([A-Za-z]+)', first_line)
        variant_name = m.group(1) if m else name
        
        if variant_name in ["StringLiteral", "MemorySpace", "MacroCall", "Number"]:
            new_match_lines.extend(branch_lines)
            continue
            
        method_name = f"check_{variant_name.lower()}_expr"
        
        body_text = "".join(branch_lines)
        needs_consume = "consume" in body_text
        
        sig_params = ["&mut self", "expr: &mut Expr", "silent: bool"]
        if needs_consume:
            sig_params.insert(2, "consume: bool")
            
        sig = f"    fn {method_name}({', '.join(sig_params)}) -> Type {{\n"
        helper_methods.append(sig)
        helper_methods.append("        match expr {\n")
        for bl in branch_lines:
            helper_methods.append("    " + bl)
        helper_methods.append("            _ => unreachable!(),\n")
        helper_methods.append("        }\n")
        helper_methods.append("    }\n\n")
        
        call_args = ["expr", "silent"]
        if needs_consume:
            call_args.insert(1, "consume")
            
        # Extract the pattern exactly up to => 
        # Actually, it's safer to just write the general pattern:
        new_match_lines.append(f"            Expr::{variant_name}(..) => self.{method_name}({', '.join(call_args)}),\n")
        
    impl_end = -1
    for i in range(len(lines)-1, -1, -1):
        if lines[i].strip() == "}":
            impl_end = i
            break
            
    # Write new file
    with open('src/sema/expr.rs', 'w') as f:
        f.writelines(lines[:128]) # up to match expr {
        f.writelines(new_match_lines)
        f.writelines(lines[1983:impl_end])
        f.writelines(helper_methods)
        f.writelines(lines[impl_end:])
        
    print("Done refactoring with exact boundaries!")

process()
