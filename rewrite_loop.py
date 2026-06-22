import re

with open("src/codegen/lower/control_flow.rs", "r") as f:
    content = f.read()

# Replace LoopStmt::lower body
loop_pattern = r"impl\s*<\s*'c\s*>\s*LowerToMelior\s*<\s*'c\s*>\s*for\s*LoopStmt.*?\{.*?\n        \}\n    \}\n\}"
loop_replacement = """impl<'c> LowerToMelior<'c> for LoopStmt {
    type Output = Result<Option<melior::ir::BlockRef<'c, 'c>>, LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        let LoopStmt { cond, body, span: _ } = self;
        
        let parent_region = block.parent_region().unwrap();
        let cond_block = parent_region.append_block(melior::ir::Block::new(&[]));
        let mut body_block = parent_region.append_block(melior::ir::Block::new(&[]));
        let merge_block = parent_region.append_block(melior::ir::Block::new(&[]));
        
        block.append_operation(
            OperationBuilder::new("cf.br", gen.loc())
                .add_successors(&[&*cond_block])
                .build()
                .unwrap(),
        );
        
        let (cond_val, _, cond_block_end) = gen.generate_expr(cond, cond_block)?;
        cond_block_end.append_operation(
            OperationBuilder::new("cf.cond_br", gen.loc())
                .add_operands(&[cond_val])
                .add_successors(&[&*body_block, &*merge_block])
                .build()
                .unwrap(),
        );

        gen.break_blocks.push(&*merge_block as *const _);
        gen.continue_blocks.push(&*cond_block as *const _);
        
        let mut body_terminated = false;
        for stmt in body {
            if let Some(b) = gen.generate_statement(stmt, body_block)? {
                body_block = b;
            } else {
                body_terminated = true;
                break;
            }
        }
        
        gen.break_blocks.pop();
        gen.continue_blocks.pop();

        if !body_terminated {
            body_block.append_operation(
                OperationBuilder::new("cf.br", gen.loc())
                    .add_successors(&[&*cond_block])
                    .build()
                    .unwrap(),
            );
        }

        Ok(Some(merge_block))
    }
}"""

content = re.sub(loop_pattern, loop_replacement, content, flags=re.DOTALL)

with open("src/codegen/lower/control_flow.rs", "w") as f:
    f.write(content)
