import os

def fix_file(path):
    with open(path, 'r') as f:
        content = f.read()

    new_content = content.replace(
        """.add_operands(&[cond_val])
                .add_successors(&[&*then_block, &*else_block])
                .build()""",
        """.add_operands(&[cond_val])
                .add_successors(&[&*then_block, &*else_block])
                .add_attributes(&[(
                    Identifier::new(gen.context, "operandSegmentSizes"),
                    melior::ir::attribute::DenseI32ArrayAttribute::new(gen.context, &[1, 0, 0]).into(),
                )])
                .build()"""
    ).replace(
        """.add_operands(&[cond_val])
                .add_successors(&[&*then_b, &*else_b])
                .build()""",
        """.add_operands(&[cond_val])
                .add_successors(&[&*then_b, &*else_b])
                .add_attributes(&[(
                    Identifier::new(gen.context, "operandSegmentSizes"),
                    melior::ir::attribute::DenseI32ArrayAttribute::new(gen.context, &[1, 0, 0]).into(),
                )])
                .build()"""
    ).replace(
        """.add_operands(&[cond_val])
                    .add_successors(&[&*body_block, &*merge_block])
                    .build()""",
        """.add_operands(&[cond_val])
                    .add_successors(&[&*body_block, &*merge_block])
                    .add_attributes(&[(
                        Identifier::new(gen.context, "operandSegmentSizes"),
                        melior::ir::attribute::DenseI32ArrayAttribute::new(gen.context, &[1, 0, 0]).into(),
                    )])
                    .build()"""
    ).replace(
        """.add_operands(&[cond_val])
                .add_successors(&[&*body_block, &*merge_block])
                .build()""",
        """.add_operands(&[cond_val])
                .add_successors(&[&*body_block, &*merge_block])
                .add_attributes(&[(
                    Identifier::new(gen.context, "operandSegmentSizes"),
                    melior::ir::attribute::DenseI32ArrayAttribute::new(gen.context, &[1, 0, 0]).into(),
                )])
                .build()"""
    )
    
    with open(path, 'w') as f:
        f.write(new_content)

fix_file('src/codegen/lower/mod.rs')
fix_file('src/codegen/lower/control_flow.rs')
fix_file('src/codegen/lower/expr.rs')
