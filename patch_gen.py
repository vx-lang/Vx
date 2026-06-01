import sys

with open("src/codegen/generator.rs", "r") as f:
    content = f.read()

# 1. Add allocs to generator struct
content = content.replace(
    "pub continue_flags: Vec<melior::ir::Value<'c, 'c>>,\n}",
    "pub continue_flags: Vec<melior::ir::Value<'c, 'c>>,\n    pub allocs: std::collections::HashSet<String>,\n}"
)

# 2. Add allocs initialization
content = content.replace(
    "continue_flags: Vec::new(),\n        }\n    }",
    "continue_flags: Vec::new(),\n            allocs: std::collections::HashSet::new(),\n        }\n    }"
)

# 3. Add short circuit in coerce_type
content = content.replace(
    """        println!(
            "Warning: Falling back to bitcast from {} to {}\\nBacktrace:\\n{:?}",
            from_str,
            to_str,
            std::backtrace::Backtrace::force_capture()
        );""",
    """        if val.r#type() == target_ty {
            return val;
        }

        println!(
            "Warning: Falling back to bitcast from {} to {}\\nBacktrace:\\n{:?}",
            from_str,
            to_str,
            std::backtrace::Backtrace::force_capture()
        );"""
)

# 4. Add Enum codegen
enum_codegen = """                        format!("!llvm.struct<\\"{}_{}\\", ({})>", name, args_str.join("_"), field_types.join(", "))
                    } else if let Some(enum_def) = self.enums.get(name) {
                        let ty_arg = args.first().unwrap();
                        let mut payload_ty_str = "none".to_string();
                        for (v_name, payload) in enum_def {
                            if v_name == "Some" {
                                if let Some(types) = payload {
                                    if !types.is_empty() {
                                        let mut mapping = std::collections::HashMap::new();
                                        mapping.insert("T".to_string(), ty_arg.clone());
                                        let sub_ty = types[0].substitute(&mapping);
                                        let mut lowered = self.lower_type_str(&sub_ty);
                                        if lowered.starts_with("memref<") {
                                            lowered = "!llvm.ptr".to_string();
                                        }
                                        payload_ty_str = lowered;
                                    }
                                }
                            }
                        }
                        
                        let args_str: Vec<String> = args.iter().map(|a| {
                            let lowered = self.lower_type_str(a);
                            lowered.replace("!", "").replace("<", "_").replace(">", "_").replace(" ", "_").replace(",", "_")
                        }).collect();
                        
                        format!("!llvm.struct<\\"{}_{}\\", (i32, {})>", name, args_str.join("_"), payload_ty_str)
                    } else {
                        panic!("Generic struct/enum {} not found", name);
                    }"""

content = content.replace(
    """                        format!("!llvm.struct<\\"{}_{}\\", ({})>", name, args_str.join("_"), field_types.join(", "))
                    } else {
                        panic!("Generic struct {} not found", name);
                    }""",
    enum_codegen
)

with open("src/codegen/generator.rs", "w") as f:
    f.write(content)
