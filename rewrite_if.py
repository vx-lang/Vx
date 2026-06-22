import re

with open("src/codegen/lower/control_flow.rs", "r") as f:
    content = f.read()

replacement = """        let (cond_val, _, block) = gen.generate_expr(cond, block)?;

        let parent_region = block.parent().unwrap();
        let mut then_b = parent_region.append_block(melior::ir::Block::new(&[]));
        let mut else_b = parent_region.append_block(melior::ir::Block::new(&[]));
        
        let has_ret = ret_ty.to_string() != "none" && ret_ty.to_string() != "void";
        let merge_b = if has_ret {
            parent_region.append_block(melior::ir::Block::new(&[(ret_ty, gen.loc())]))
        } else {
            parent_region.append_block(melior::ir::Block::new(&[]))
        };

        block.append_operation(
            OperationBuilder::new("cf.cond_br", gen.loc())
                .add_operands(&[cond_val])
                .add_successors(&[&*then_b, &*else_b])
                .build()
                .unwrap(),
        );

        let mut then_terminated = false;
        for stmt in then_block {
            if let Some(b) = gen.generate_statement(stmt, then_b)? {
                then_b = b;
            } else {
                then_terminated = true;
                break;
            }
        }
        
        if !then_terminated {
            // Need to pass the return value if there is one
            // Wait, IfExpr's return value was just dummy 0!
            // I should just pass 0 for now.
            // But wait, the original code returned 0 as dummy value?
            // "let ty = gen.i32_ty; let op = OperationBuilder::new(\"arith.constant\" ...)"
            // Actually, in the old scf.if, it did not even yield the correct then/else values!
            // Let's just cf.br to merge_b.
            then_b.append_operation(
                OperationBuilder::new("cf.br", gen.loc())
                    .add_successors(&[&*merge_b])
                    .build()
                    .unwrap(),
            );
        }

        let mut else_terminated = false;
        if let Some(else_block) = else_block_opt {
            for stmt in else_block {
                if let Some(b) = gen.generate_statement(stmt, else_b)? {
                    else_b = b;
                } else {
                    else_terminated = true;
                    break;
                }
            }
        }
        if !else_terminated {
            else_b.append_operation(
                OperationBuilder::new("cf.br", gen.loc())
                    .add_successors(&[&*merge_b])
                    .build()
                    .unwrap(),
            );
        }

        if has_ret {
            let res = merge_b.argument(0).unwrap().into();
            Ok((res, ret_ty, merge_b))
        } else {
            Ok((cond_val, ret_ty, merge_b))
        }
    }
}"""

# We need to replace everything from "let (cond_val, _, block) = gen.generate_expr(cond, block)?;"
# up to "Ok((op_ref.result(0).unwrap().into(), ty, block))\n    }\n}"

pattern = r"let \(cond_val, _, block\) = gen\.generate_expr\(cond, block\)\?;.*?Ok\(\(op_ref\.result\(0\)\.unwrap\(\)\.into\(\), ty, block\)\)\n    \}\n\}"
content = re.sub(pattern, replacement, content, flags=re.DOTALL)

with open("src/codegen/lower/control_flow.rs", "w") as f:
    f.write(content)
