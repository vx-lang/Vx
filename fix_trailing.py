import re

files = [
    "src/codegen/lower/stmt.rs",
    "src/codegen/lower/expr.rs",
    "src/codegen/lower/tensors.rs",
    "src/codegen/lower/control_flow.rs",
]

for file in files:
    with open(file, "r") as f:
        content = f.read()

    # Fix Ok tuples
    replacements = [
        (", ty))", ", ty, block))"),
        (", elem_ty))", ", elem_ty, block))"),
        (", ptr_ty))", ", ptr_ty, block))"),
        (", ret_ty))", ", ret_ty, block))"),
        (", inner_ty))", ", inner_ty, block))"),
        (", i1_ty))", ", i1_ty, block))"),
        (", final_ty))", ", final_ty, block))"),
        (", none_ty))", ", none_ty, block))"),
        # Multiline Ok returns
        ("ret_ty,\n            ))", "ret_ty,\n                block,\n            ))"),
        ("ty,\n            ))", "ty,\n                block,\n            ))"),
        ("Type::parse(gen.context, &format!(\"memref<{}>\", ty)).unwrap(),\n            ))", "Type::parse(gen.context, &format!(\"memref<{}>\", ty)).unwrap(),\n                block,\n            ))")
    ]
    
    for old, new in replacements:
        content = content.replace(old, new)

    # Fix multiline lower signatures
    bad_sig = """    fn lower<'a>(
        &self,
        _gen: &mut MeliorGenerator<'c>,
        _block: &melior::ir::Block<'c>,
    ) -> Self::Output<'a> {"""
    good_sig = """    fn lower<'a>(
        &self,
        _gen: &mut MeliorGenerator<'c>,
        _region: &'a melior::ir::Region<'c>,
        mut block: melior::ir::BlockRef<'c, 'a>,
    ) -> Self::Output<'a> {"""
    content = content.replace(bad_sig, good_sig)

    bad_sig2 = """    fn lower(
        &self,
        _gen: &mut MeliorGenerator<'c>,
        _block: &melior::ir::Block<'c>,
    ) -> Self::Output<'a> {"""
    content = content.replace(bad_sig2, good_sig)

    with open(file, "w") as f:
        f.write(content)
