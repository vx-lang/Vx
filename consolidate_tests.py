import os
import glob

base_dir = "/Users/adityak/go/Vx/tests/frontend/fail"

groups = {
    "comptime_if_errors.vx": [
        "comptime_active_branch_error.vx",
        "comptime_if_not_bool.vx",
        "comptime_if_pruned_syntax_error.vx",
        "comptime_if_runtime_cond.vx"
    ],
    "const_generic_errors.vx": [
        "const_generic_arg_count.vx",
        "const_generic_impl_mismatch.vx",
        "const_generic_runtime_expr.vx",
        "const_generics_missing_args.vx",
        "const_generics_type_mismatch.vx"
    ],
    "mlir_macro_errors.vx": [
        "mlir_macro_invalid_input_arg.vx",
        "mlir_macro_invalid_syntax.vx",
        "mlir_macro_missing_block.vx",
        "mlir_macro_undefined_clobber.vx",
        "mlir_macro_undefined_input.vx"
    ],
    "topology_errors.vx": [
        "topology_assign_type_mismatch.vx",
        "topology_compare_type_mismatch.vx",
        "topology_invalid_spawn.vx",
        "topology_undefined_variant.vx"
    ]
}

for group_name, files in groups.items():
    combined_content = "// RUN: not vxc %s\n\n"
    for i, file in enumerate(files):
        filepath = os.path.join(base_dir, file)
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found")
            continue
            
        with open(filepath, "r") as f:
            content = f.read()
            
        # Remove the RUN line from individual files
        content = content.replace("// RUN: not vxc %s\n", "")
        
        # Rename 'main' to 'test_X' to avoid duplicate symbol errors if they were valid
        content = content.replace("fn main()", f"fn test_{i}()")
        
        # Also need to rename any duplicate structs, e.g. struct Foo
        content = content.replace("struct Foo", f"struct Foo_{i}")
        content = content.replace("Foo<", f"Foo_{i}<")
        
        combined_content += f"// From {file}\n"
        combined_content += content + "\n\n"
        
        # Delete original
        os.remove(filepath)
        
    group_filepath = os.path.join(base_dir, group_name)
    with open(group_filepath, "w") as f:
        f.write(combined_content)
    print(f"Created {group_filepath}")
    
