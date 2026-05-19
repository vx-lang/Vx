use std::collections::HashMap;

use melior::{
    dialect::DialectRegistry,
    ir::{
        attribute::{StringAttribute, TypeAttribute},
        operation::OperationLike,
        Block, BlockLike, Location, Module, Region, RegionLike, Type, Value,
    },
    Context,
};

use crate::ast::*;

pub struct MeliorGenerator<'c> {
    context: &'c Context,
    module: Module<'c>,
    env: HashMap<String, (Value<'c, 'c>, Type<'c>)>,
    structs: HashMap<String, StructDecl>,
    enums: HashMap<String, Vec<String>>,
    functions: HashMap<String, (Type<'c>, Vec<Type<'c>>)>,
    current_el_ty: Type<'c>,
}

impl<'c> MeliorGenerator<'c> {
    pub fn new(context: &'c Context) -> Self {
        let registry = DialectRegistry::new();
        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();

        let location = Location::unknown(context);
        let module = Module::new(location);

        let default_type = Type::parse(context, "f32").unwrap();

        Self {
            context,
            module,
            env: HashMap::new(),
            structs: HashMap::new(),
            enums: HashMap::new(),
            functions: HashMap::new(),
            current_el_ty: default_type,
        }
    }

    pub fn generate(&mut self, program: &Program, modules: &HashMap<String, Program>) -> String {
        for s in &program.structs {
            self.structs.insert(s.name.clone(), s.clone());
        }
        for e in &program.enums {
            self.enums.insert(e.name.clone(), e.variants.clone());
        }
        for ext in &program.externs {
            let ret_ty = self.lower_type(&ext.return_type);
            let mut arg_tys = Vec::new();
            for (_, ty) in &ext.params {
                arg_tys.push(self.lower_type(ty));
            }
            self.functions.insert(ext.name.clone(), (ret_ty, arg_tys));
        }

        let mut operations = Vec::new();

        // Emit module functions
        for module_prog in modules.values() {
            for func in &module_prog.functions {
                operations.push(self.generate_function(func));
            }
        }

        for func in &program.functions {
            operations.push(self.generate_function(func));
        }

        let body = self.module.body();
        for (name, (ret_ty, arg_tys)) in &self.functions {
            // FunctionType::new takes arg_tys and ret_tys
            let func_type =
                melior::ir::r#type::FunctionType::new(self.context, arg_tys, &[*ret_ty]);

            // Define the string attribute for the function name
            let name_attr = melior::ir::attribute::StringAttribute::new(self.context, name);
            let type_attr = melior::ir::attribute::TypeAttribute::new(func_type.into());

            let func_op = melior::ir::operation::OperationBuilder::new(
                "func.func",
                melior::ir::Location::unknown(self.context),
            )
            .add_attributes(&[
                (
                    melior::ir::Identifier::new(self.context, "sym_name"),
                    name_attr.into(),
                ),
                (
                    melior::ir::Identifier::new(self.context, "function_type"),
                    type_attr.into(),
                ),
                (
                    melior::ir::Identifier::new(self.context, "sym_visibility"),
                    melior::ir::attribute::StringAttribute::new(self.context, "private").into(),
                ),
            ])
            .build()
            .unwrap();

            body.append_operation(func_op);
        }

        for op in operations {
            body.append_operation(op);
        }

        let op = self.module.as_operation();
        op.to_string()
    }

    fn generate_function(&mut self, func: &Function) -> melior::ir::Operation<'c> {
        let is_main = func.name == "main";
        let true_ret_ty = self.lower_type(&func.return_type);
        let ret_ty = if is_main {
            Type::parse(self.context, "i32").unwrap()
        } else {
            true_ret_ty
        };

        let mut arg_tys = Vec::new();
        for (_, ty) in &func.params {
            arg_tys.push(self.lower_type(ty));
        }

        let func_type = melior::ir::r#type::FunctionType::new(self.context, &arg_tys, &[ret_ty]);
        let name_attr = melior::ir::attribute::StringAttribute::new(self.context, &func.name);
        let type_attr = melior::ir::attribute::TypeAttribute::new(func_type.into());

        let region = Region::new();

        let mut block_args = Vec::new();
        for ty in &arg_tys {
            block_args.push((*ty, Location::unknown(self.context)));
        }
        let block = Block::new(&block_args);

        // Map arguments into the environment
        for (i, (name, _)) in func.params.iter().enumerate() {
            let arg_val = block.argument(i).unwrap().into();
            self.env.insert(name.clone(), (arg_val, arg_tys[i]));
        }

        // Parse body
        for stmt in &func.body {
            if let Statement::Return(..) = stmt {
                if is_main {
                    // Ignore explicit returns in main for simplicity
                    continue;
                }
            }
            self.generate_statement(stmt, &block);
        }

        if is_main {
            let zero_op = melior::ir::operation::OperationBuilder::new(
                "arith.constant",
                Location::unknown(self.context),
            )
            .add_results(&[Type::parse(self.context, "i32").unwrap()])
            .add_attributes(&[(
                melior::ir::Identifier::new(self.context, "value"),
                melior::ir::attribute::IntegerAttribute::new(
                    Type::parse(self.context, "i32").unwrap(),
                    0,
                )
                .into(),
            )])
            .build()
            .unwrap();
            let zero_val = zero_op.result(0).unwrap().into();
            block.append_operation(zero_op);

            let ret_op = melior::ir::operation::OperationBuilder::new(
                "func.return",
                Location::unknown(self.context),
            )
            .add_operands(&[zero_val])
            .build()
            .unwrap();
            block.append_operation(ret_op);
        }

        region.append_block(block);

        let func_op = melior::ir::operation::OperationBuilder::new(
            "func.func",
            Location::unknown(self.context),
        )
        .add_attributes(&[
            (
                melior::ir::Identifier::new(self.context, "sym_name"),
                name_attr.into(),
            ),
            (
                melior::ir::Identifier::new(self.context, "function_type"),
                type_attr.into(),
            ),
        ])
        .add_regions([region])
        .build()
        .unwrap();

        func_op
    }

    fn generate_statement(&mut self, stmt: &Statement, block: &melior::ir::Block<'c>) {
        match stmt {
            Statement::Return(expr, _) => {
                let (val, _) = self.generate_expr(expr, block);
                let ret_op = melior::ir::operation::OperationBuilder::new(
                    "func.return",
                    Location::unknown(self.context),
                )
                .add_operands(&[val])
                .build()
                .unwrap();
                block.append_operation(ret_op);
            }
            Statement::LetDecl(name, is_mut, ty_ann, expr, _) => {
                let (val, ty) = self.generate_expr(expr, block);
                if *is_mut {
                    // For mutable variables, we should allocate a memref and store the value
                    // But for simple types we can just rely on SSA reassignment or MLIR memref later
                    // For now we just insert the value into env
                    self.env.insert(name.clone(), (val, ty));
                } else {
                    self.env.insert(name.clone(), (val, ty));
                }
            }
            Statement::Assign(lhs, rhs, _) => {
                let (rhs_val, rhs_ty) = self.generate_expr(rhs, block);
                if let Expr::Identifier(name, _) = lhs {
                    // Reassigning in our simple SSA representation
                    self.env.insert(name.clone(), (rhs_val, rhs_ty));
                } else {
                    // TODO: handle array indexing or struct member assign
                }
            }
            Statement::CompoundAssign(lhs, op, rhs, _) => {
                let (rhs_val, _) = self.generate_expr(rhs, block);
                let (lhs_val, ty) = self.generate_expr(lhs, block);

                let is_float = ty.to_string().contains("f32") || ty.to_string().contains("f64");
                let op_name = match op {
                    BinaryOp::Add => {
                        if is_float {
                            "arith.addf"
                        } else {
                            "arith.addi"
                        }
                    }
                    BinaryOp::Sub => {
                        if is_float {
                            "arith.subf"
                        } else {
                            "arith.subi"
                        }
                    }
                    BinaryOp::Mul => {
                        if is_float {
                            "arith.mulf"
                        } else {
                            "arith.muli"
                        }
                    }
                    BinaryOp::Div => {
                        if is_float {
                            "arith.divf"
                        } else {
                            "arith.divsi"
                        }
                    }
                    _ => panic!("Unsupported compound assign op"),
                };

                let bin_op = melior::ir::operation::OperationBuilder::new(
                    op_name,
                    Location::unknown(self.context),
                )
                .add_operands(&[lhs_val, rhs_val])
                .add_results(&[ty])
                .build()
                .unwrap();
                let bin_ref = block.append_operation(bin_op);
                let result_val = bin_ref.result(0).unwrap().into();

                if let Expr::Identifier(name, _) = lhs {
                    self.env.insert(name.clone(), (result_val, ty));
                }
            }
            Statement::ExprStmt(expr, _, _) => {
                self.generate_expr(expr, block);
            }
            _ => {
                // TODO: implement other statements
            }
        }
    }

    fn generate_expr(
        &mut self,
        expr: &Expr,
        block: &melior::ir::Block<'c>,
    ) -> (Value<'c, 'c>, Type<'c>) {
        match expr {
            Expr::Identifier(name, _) => {
                if let Some((val, ty)) = self.env.get(name) {
                    (*val, *ty)
                } else {
                    panic!("Undefined variable: {}", name);
                }
            }
            Expr::BinaryOp(lhs, op, rhs, _) => {
                let (lhs_val, lhs_ty) = self.generate_expr(lhs, block);
                let (rhs_val, rhs_ty) = self.generate_expr(rhs, block);
                // Assume lhs_ty == rhs_ty for simplicity right now
                let is_float =
                    lhs_ty.to_string().contains("f32") || lhs_ty.to_string().contains("f64");

                let op_name = match op {
                    BinaryOp::Add => {
                        if is_float {
                            "arith.addf"
                        } else {
                            "arith.addi"
                        }
                    }
                    BinaryOp::Sub => {
                        if is_float {
                            "arith.subf"
                        } else {
                            "arith.subi"
                        }
                    }
                    BinaryOp::Mul => {
                        if is_float {
                            "arith.mulf"
                        } else {
                            "arith.muli"
                        }
                    }
                    BinaryOp::Div => {
                        if is_float {
                            "arith.divf"
                        } else {
                            "arith.divsi"
                        }
                    }
                    _ => panic!("Unsupported binary op"),
                };

                let bin_op = melior::ir::operation::OperationBuilder::new(
                    op_name,
                    Location::unknown(self.context),
                )
                .add_operands(&[lhs_val, rhs_val])
                .add_results(&[lhs_ty])
                .build()
                .unwrap();
                let bin_ref = block.append_operation(bin_op);
                (bin_ref.result(0).unwrap().into(), lhs_ty)
            }
            Expr::Number(val_str, _, _) => {
                if val_str.contains('.') {
                    let ty = Type::parse(self.context, "f32").unwrap();
                    let op = melior::ir::operation::OperationBuilder::new(
                        "arith.constant",
                        Location::unknown(self.context),
                    )
                    .add_results(&[ty])
                    .add_attributes(&[(
                        melior::ir::Identifier::new(self.context, "value"),
                        melior::ir::attribute::FloatAttribute::new(
                            self.context,
                            ty,
                            val_str.parse::<f64>().unwrap(),
                        )
                        .into(),
                    )])
                    .build()
                    .unwrap();
                    let op_ref = block.append_operation(op);
                    (op_ref.result(0).unwrap().into(), ty)
                } else {
                    let ty = Type::parse(self.context, "i32").unwrap();
                    let op = melior::ir::operation::OperationBuilder::new(
                        "arith.constant",
                        Location::unknown(self.context),
                    )
                    .add_results(&[ty])
                    .add_attributes(&[(
                        melior::ir::Identifier::new(self.context, "value"),
                        melior::ir::attribute::IntegerAttribute::new(
                            ty,
                            val_str.parse::<i64>().unwrap(),
                        )
                        .into(),
                    )])
                    .build()
                    .unwrap();
                    let op_ref = block.append_operation(op);
                    (op_ref.result(0).unwrap().into(), ty)
                }
            }
            _ => {
                // Return dummy for now
                let ty = Type::parse(self.context, "i32").unwrap();
                let op = melior::ir::operation::OperationBuilder::new(
                    "arith.constant",
                    Location::unknown(self.context),
                )
                .add_results(&[ty])
                .add_attributes(&[(
                    melior::ir::Identifier::new(self.context, "value"),
                    melior::ir::attribute::IntegerAttribute::new(ty, 0).into(),
                )])
                .build()
                .unwrap();
                let op_ref = block.append_operation(op);
                (op_ref.result(0).unwrap().into(), ty)
            }
        }
    }

    fn lower_type(&self, ty: &crate::ast::Type) -> Type<'c> {
        let ty_str = match ty {
            crate::ast::Type::Tensor(el_ty, dims, _) => {
                let ty_str = match el_ty {
                    ElementType::F16 => "f16",
                    ElementType::F32 => "f32",
                    ElementType::F64 => "f64",
                    ElementType::BF16 => "bf16",
                    ElementType::I4 | ElementType::U4 => "i4",
                    ElementType::I8 | ElementType::U8 => "i8",
                    ElementType::I16 | ElementType::U16 => "i16",
                    ElementType::I32 | ElementType::U32 => "i32",
                    ElementType::I64 | ElementType::U64 => "i64",
                    ElementType::I128 | ElementType::U128 => "i128",
                    ElementType::Bool => "i1",
                };

                let mut shape_str = String::new();
                if dims.is_empty() {
                    shape_str = "?x?".to_string();
                } else {
                    for (i, dim) in dims.iter().enumerate() {
                        if let crate::ast::Expr::Number(n_str, _, _) = dim {
                            if let Ok(n) = n_str.parse::<f64>() {
                                shape_str.push_str(&format!("{}", n as i64));
                            }
                        } else {
                            shape_str.push('?');
                        }
                        if i < dims.len() - 1 {
                            shape_str.push('x');
                        }
                    }
                }

                if !shape_str.is_empty() && !shape_str.ends_with('x') {
                    shape_str.push('x');
                }

                format!("memref<{}{}>", shape_str, ty_str)
            }
            crate::ast::Type::Scalar(el_ty) => match el_ty {
                ElementType::F16 => "f16",
                ElementType::F32 => "f32",
                ElementType::F64 => "f64",
                ElementType::BF16 => "bf16",
                ElementType::I4 | ElementType::U4 => "i4",
                ElementType::I8 | ElementType::U8 => "i8",
                ElementType::I16 | ElementType::U16 => "i16",
                ElementType::I32 | ElementType::U32 => "i32",
                ElementType::I64 | ElementType::U64 => "i64",
                ElementType::I128 | ElementType::U128 => "i128",
                ElementType::Bool => "i1",
            }
            .to_string(),
            crate::ast::Type::Matrix => "tensor<?x?xf32>".to_string(),
            crate::ast::Type::Ref(inner, _) => return self.lower_type(inner),
            crate::ast::Type::Verified(inner) => return self.lower_type(inner),
            crate::ast::Type::Pinned(inner, top) => {
                let addr_space = match top {
                    Topology::NPU(_) | Topology::Slice(_, _, _) | Topology::ANE => 1,
                    Topology::AccCore(_) => 2,
                    Topology::Host | Topology::AMX | Topology::GPU => 0,
                };
                let inner_ty_str = self.lower_type_str(inner);
                if inner_ty_str.starts_with("memref<")
                    && inner_ty_str.ends_with(">")
                    && addr_space != 0
                {
                    let inner_str = &inner_ty_str[7..inner_ty_str.len() - 1];
                    format!("memref<{}, {}>", inner_str, addr_space)
                } else {
                    inner_ty_str
                }
            }
            crate::ast::Type::Borrow(_, mem, _) | crate::ast::Type::Pointer(_, mem, _) => {
                let addr_space = match mem {
                    Some(MemorySpace::NPUHBM) => 1,
                    Some(MemorySpace::LocalSRAM) => 2,
                    Some(MemorySpace::HostDRAM) | None => 0,
                };
                format!("!llvm.ptr<{}>", addr_space)
            }
            crate::ast::Type::Struct(name, _) => {
                if let Some(decl) = self.structs.get(name).cloned() {
                    let mut field_types = Vec::new();
                    for (_, ty) in &decl.fields {
                        let mut lowered = self.lower_type_str(ty);
                        if lowered.starts_with("memref<") {
                            lowered = "!llvm.ptr".to_string();
                        }
                        field_types.push(lowered);
                    }
                    format!("!llvm.struct<\"{}\", ({})>", name, field_types.join(", "))
                } else {
                    format!("!llvm.struct<\"{}\">", name)
                }
            }
            crate::ast::Type::Generic(_, _) | crate::ast::Type::GenericInstance(_, _) => {
                panic!("Generic types should have been monomorphized before codegen!");
            }
            crate::ast::Type::Simd(el_ty, n) => {
                let ty_str = match el_ty {
                    ElementType::F16 => "f16",
                    ElementType::F32 => "f32",
                    ElementType::F64 => "f64",
                    ElementType::BF16 => "bf16",
                    ElementType::I4 | ElementType::U4 => "i4",
                    ElementType::I8 | ElementType::U8 => "i8",
                    ElementType::I16 | ElementType::U16 => "i16",
                    ElementType::I32 | ElementType::U32 => "i32",
                    ElementType::I64 | ElementType::U64 => "i64",
                    ElementType::I128 | ElementType::U128 => "i128",
                    ElementType::Bool => "i1",
                };
                format!("vector<{}x{}>", n, ty_str)
            }
            crate::ast::Type::Enum(_, _) => "i32".to_string(),
            crate::ast::Type::Module(..) => "none".to_string(),
        };

        Type::parse(self.context, &ty_str)
            .unwrap_or_else(|| panic!("Failed to parse MLIR type: {}", ty_str))
    }

    fn lower_type_str(&self, ty: &crate::ast::Type) -> String {
        let t = self.lower_type(ty);
        t.to_string()
    }
}
