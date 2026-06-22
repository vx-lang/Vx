import re

with open("src/codegen/lower/control_flow.rs", "r") as f:
    content = f.read()

# Replace BreakStmt
break_pattern = r"impl\s*<\s*'c\s*>\s*LowerToMelior\s*<\s*'c\s*>\s*for\s*BreakStmt.*?\{.*?\n        \}\n    \}\n\}"
break_replacement = """impl<'c> LowerToMelior<'c> for BreakStmt {
    type Output = Result<Option<melior::ir::BlockRef<'c, 'c>>, LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        if let Some(&break_ptr) = gen.break_blocks.last() {
            let break_block: &melior::ir::Block<'c> = unsafe { &*break_ptr };
            block.append_operation(
                OperationBuilder::new("cf.br", gen.loc())
                    .add_successors(&[break_block])
                    .build()
                    .unwrap(),
            );
        }
        Ok(None)
    }
}"""
content = re.sub(break_pattern, break_replacement, content, flags=re.DOTALL)

# Replace ContinueStmt
continue_pattern = r"impl\s*<\s*'c\s*>\s*LowerToMelior\s*<\s*'c\s*>\s*for\s*ContinueStmt.*?\{.*?\n        \}\n    \}\n\}"
continue_replacement = """impl<'c> LowerToMelior<'c> for ContinueStmt {
    type Output = Result<Option<melior::ir::BlockRef<'c, 'c>>, LowerError>;
    fn lower(
        &self,
        gen: &mut MeliorGenerator<'c>,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Self::Output {
        if let Some(&continue_ptr) = gen.continue_blocks.last() {
            let continue_block: &melior::ir::Block<'c> = unsafe { &*continue_ptr };
            block.append_operation(
                OperationBuilder::new("cf.br", gen.loc())
                    .add_successors(&[continue_block])
                    .build()
                    .unwrap(),
            );
        }
        Ok(None)
    }
}"""
content = re.sub(continue_pattern, continue_replacement, content, flags=re.DOTALL)

with open("src/codegen/lower/control_flow.rs", "w") as f:
    f.write(content)
