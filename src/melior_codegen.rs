use std::collections::HashMap;

use melior::{
    dialect::DialectRegistry,
    ir::{Block, BlockLike, Location, Module, Region, RegionLike, Type, Value},
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
}

impl<'c> MeliorGenerator<'c> {
    pub fn new(context: &'c Context) -> Self {
        let registry = DialectRegistry::new();
        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();

        let location = Location::unknown(context);
        let module = Module::new(location);

        Self {
            context,
            module,
            env: HashMap::new(),
            structs: HashMap::new(),
            enums: HashMap::new(),
            functions: HashMap::new(),
        }
    }

    pub fn into_module(self) -> Module<'c> {
        self.module
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
                let ret_ty = self.lower_type(&func.return_type);
                let mut arg_tys = Vec::new();
                for (_, ty) in &func.params {
                    arg_tys.push(self.lower_type(ty));
                }
                self.functions.insert(func.name.clone(), (ret_ty, arg_tys));
                operations.push(self.generate_function(func));
            }
        }

        for func in &program.functions {
            let ret_ty = self.lower_type(&func.return_type);
            let mut arg_tys = Vec::new();
            for (_, ty) in &func.params {
                arg_tys.push(self.lower_type(ty));
            }
            self.functions.insert(func.name.clone(), (ret_ty, arg_tys));
            operations.push(self.generate_function(func));
        }

        let body = self.module.body();
        for ext in &program.externs {
            let name = &ext.name;
            let (ret_ty, arg_tys) = self.functions.get(name).unwrap();
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

        for stmt in &func.body {
            self.generate_statement(stmt, &block);
        }

        region.append_block(block);

        let mut func_attributes = vec![
            (
                melior::ir::Identifier::new(self.context, "sym_name"),
                name_attr.into(),
            ),
            (
                melior::ir::Identifier::new(self.context, "function_type"),
                type_attr.into(),
            ),
        ];

        if is_main {
            func_attributes.push((
                melior::ir::Identifier::new(self.context, "llvm.emit_c_interface"),
                melior::ir::attribute::Attribute::parse(self.context, "unit").unwrap(),
            ));
        }

        let func_op = melior::ir::operation::OperationBuilder::new(
            "func.func",
            Location::unknown(self.context),
        )
        .add_attributes(&func_attributes)
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
            Statement::LetDecl(name, is_mut, _ty_ann, expr, _) => {
                let (val, ty) = self.generate_expr(expr, block);
                if *is_mut {
                    let memref_ty = format!("memref<{}>", ty);
                    let parsed_memref_ty = Type::parse(self.context, &memref_ty).unwrap();
                    let alloca_op = melior::ir::operation::OperationBuilder::new(
                        "memref.alloca",
                        Location::unknown(self.context),
                    )
                    .add_results(&[parsed_memref_ty])
                    .build()
                    .unwrap();
                    let alloca_ref = block.append_operation(alloca_op);
                    let alloca_val = alloca_ref.result(0).unwrap().into();

                    let store_op = melior::ir::operation::OperationBuilder::new(
                        "memref.store",
                        Location::unknown(self.context),
                    )
                    .add_operands(&[val, alloca_val])
                    .build()
                    .unwrap();
                    block.append_operation(store_op);

                    self.env
                        .insert(name.clone(), (alloca_val, parsed_memref_ty));
                } else {
                    self.env.insert(name.clone(), (val, ty));
                }
            }
            Statement::Assign(lhs, rhs, _) => {
                let (rhs_val, rhs_ty) = self.generate_expr(rhs, block);
                if let Expr::Identifier(name, _) = lhs {
                    if let Some((mem_val, mem_ty)) = self.env.get(name).cloned() {
                        let mem_ty_str = mem_ty.to_string();
                        if mem_ty_str.starts_with("memref<") {
                            let mut store_val = rhs_val;
                            let inner_ty_str = &mem_ty_str[7..mem_ty_str.len() - 1];
                            let inner_ty = Type::parse(self.context, inner_ty_str).unwrap();
                            if rhs_ty != inner_ty
                                && ((rhs_ty.to_string() == "i32" && inner_ty_str == "index")
                                    || (rhs_ty.to_string() == "index" && inner_ty_str == "i32"))
                            {
                                let cast_op = melior::ir::operation::OperationBuilder::new(
                                    "arith.index_cast",
                                    Location::unknown(self.context),
                                )
                                .add_operands(&[rhs_val])
                                .add_results(&[inner_ty])
                                .build()
                                .unwrap();

                                store_val =
                                    block.append_operation(cast_op).result(0).unwrap().into();
                            }

                            let store_op = melior::ir::operation::OperationBuilder::new(
                                "memref.store",
                                Location::unknown(self.context),
                            )
                            .add_operands(&[store_val, mem_val])
                            .build()
                            .unwrap();
                            block.append_operation(store_op);
                        } else {
                            self.env.insert(name.clone(), (rhs_val, rhs_ty));
                        }
                    }
                } else if let Expr::MemberAccess(base, member, _) = lhs {
                    if let Expr::Identifier(base_name, _) = &**base {
                        let (base_val, base_ty) = self.generate_expr(base, block);
                        let base_ty_str = base_ty.to_string();

                        let mut struct_name_opt = None;
                        if let Some(start_idx) = base_ty_str.find('\"') {
                            if let Some(end_idx) = base_ty_str[start_idx + 1..].find('\"') {
                                struct_name_opt = Some(
                                    base_ty_str[start_idx + 1..start_idx + 1 + end_idx].to_string(),
                                );
                            }
                        }

                        if let Some(struct_name) = struct_name_opt {
                            if let Some(struct_decl) = self.structs.get(&struct_name).cloned() {
                                if let Some(field_idx) =
                                    struct_decl.fields.iter().position(|(n, _)| n == member)
                                {
                                    let field_ty =
                                        self.lower_type(&struct_decl.fields[field_idx].1);
                                    let mut field_val = rhs_val;

                                    if rhs_ty != field_ty
                                        && ((rhs_ty.to_string() == "index"
                                            && field_ty.to_string() == "i32")
                                            || (rhs_ty.to_string() == "i32"
                                                && field_ty.to_string() == "index"))
                                    {
                                        let cast_op = melior::ir::operation::OperationBuilder::new(
                                            "arith.index_cast",
                                            Location::unknown(self.context),
                                        )
                                        .add_operands(&[rhs_val])
                                        .add_results(&[field_ty])
                                        .build()
                                        .unwrap();
                                        field_val = block
                                            .append_operation(cast_op)
                                            .result(0)
                                            .unwrap()
                                            .into();
                                    }

                                    let pos_attr =
                                        melior::ir::attribute::DenseI64ArrayAttribute::new(
                                            self.context,
                                            &[field_idx as i64],
                                        );
                                    let insert_op = melior::ir::operation::OperationBuilder::new(
                                        "llvm.insertvalue",
                                        Location::unknown(self.context),
                                    )
                                    .add_operands(&[base_val, field_val])
                                    .add_attributes(&[(
                                        melior::ir::Identifier::new(self.context, "position"),
                                        pos_attr.into(),
                                    )])
                                    .add_results(&[base_ty])
                                    .build()
                                    .unwrap();

                                    let new_struct_val =
                                        block.append_operation(insert_op).result(0).unwrap().into();

                                    if let Some((mem_val, mem_ty)) =
                                        self.env.get(base_name).cloned()
                                    {
                                        let mem_ty_str = mem_ty.to_string();
                                        if mem_ty_str.starts_with("memref<") {
                                            let store_op =
                                                melior::ir::operation::OperationBuilder::new(
                                                    "memref.store",
                                                    Location::unknown(self.context),
                                                )
                                                .add_operands(&[new_struct_val, mem_val])
                                                .build()
                                                .unwrap();
                                            block.append_operation(store_op);
                                        } else {
                                            self.env.insert(
                                                base_name.clone(),
                                                (new_struct_val, base_ty),
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        panic!("Complex struct assignment lhs not supported");
                    }
                }
            }
            Statement::CompoundAssign(lhs, op, rhs, _) => {
                let (rhs_val, rhs_ty) = self.generate_expr(rhs, block);
                let (lhs_val, ty) = self.generate_expr(lhs, block);

                let mut actual_rhs = rhs_val;
                if rhs_ty != ty
                    && ((rhs_ty.to_string() == "index" && ty.to_string() == "i32")
                        || (rhs_ty.to_string() == "i32" && ty.to_string() == "index"))
                {
                    let cast_op = melior::ir::operation::OperationBuilder::new(
                        "arith.index_cast",
                        Location::unknown(self.context),
                    )
                    .add_operands(&[actual_rhs])
                    .add_results(&[ty])
                    .build()
                    .unwrap();
                    actual_rhs = block.append_operation(cast_op).result(0).unwrap().into();
                }

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
                .add_operands(&[lhs_val, actual_rhs])
                .add_results(&[ty])
                .build()
                .unwrap();
                let bin_ref = block.append_operation(bin_op);
                let result_val = bin_ref.result(0).unwrap().into();

                if let Expr::Identifier(name, _) = lhs {
                    if let Some((mem_val, mem_ty)) = self.env.get(name).cloned() {
                        let mem_ty_str = mem_ty.to_string();
                        if mem_ty_str.starts_with("memref<") {
                            let store_op = melior::ir::operation::OperationBuilder::new(
                                "memref.store",
                                Location::unknown(self.context),
                            )
                            .add_operands(&[result_val, mem_val])
                            .build()
                            .unwrap();
                            block.append_operation(store_op);
                        } else {
                            self.env.insert(name.clone(), (result_val, ty));
                        }
                    }
                }
            }
            Statement::ExprStmt(expr, _, _) => {
                self.generate_expr(expr, block);
            }
            Statement::ForLoop(iter, start, end, body, _) => {
                let (start_val, start_ty) = self.generate_expr(start, block);
                let (end_val, end_ty) = self.generate_expr(end, block);

                let ty_index = Type::parse(self.context, "index").unwrap();

                // step = 1
                let step_op = melior::ir::operation::OperationBuilder::new(
                    "arith.constant",
                    Location::unknown(self.context),
                )
                .add_results(&[ty_index])
                .add_attributes(&[(
                    melior::ir::Identifier::new(self.context, "value"),
                    melior::ir::attribute::IntegerAttribute::new(ty_index, 1).into(),
                )])
                .build()
                .unwrap();
                let step_ref = block.append_operation(step_op);
                let step_val = step_ref.result(0).unwrap().into();

                // cast start/end to index if necessary
                let start_idx = if start_ty == ty_index {
                    start_val
                } else {
                    let cast_start_op = melior::ir::operation::OperationBuilder::new(
                        "arith.index_cast",
                        Location::unknown(self.context),
                    )
                    .add_operands(&[start_val])
                    .add_results(&[ty_index])
                    .build()
                    .unwrap();
                    block
                        .append_operation(cast_start_op)
                        .result(0)
                        .unwrap()
                        .into()
                };

                let end_idx = if end_ty == ty_index {
                    end_val
                } else {
                    let cast_end_op = melior::ir::operation::OperationBuilder::new(
                        "arith.index_cast",
                        Location::unknown(self.context),
                    )
                    .add_operands(&[end_val])
                    .add_results(&[ty_index])
                    .build()
                    .unwrap();
                    block
                        .append_operation(cast_end_op)
                        .result(0)
                        .unwrap()
                        .into()
                };

                let body_region = Region::new();
                let body_block = Block::new(&[(ty_index, Location::unknown(self.context))]);

                let iter_val = body_block.argument(0).unwrap().into();
                let prev_env_val = self.env.get(iter).cloned();
                self.env.insert(iter.clone(), (iter_val, ty_index));

                for stmt in body {
                    self.generate_statement(stmt, &body_block);
                }

                if let Some(prev) = prev_env_val {
                    self.env.insert(iter.clone(), prev);
                } else {
                    self.env.remove(iter);
                }

                let yield_op = melior::ir::operation::OperationBuilder::new(
                    "scf.yield",
                    Location::unknown(self.context),
                )
                .build()
                .unwrap();
                body_block.append_operation(yield_op);
                body_region.append_block(body_block);

                let for_op = melior::ir::operation::OperationBuilder::new(
                    "scf.for",
                    Location::unknown(self.context),
                )
                .add_operands(&[start_idx, end_idx, step_val])
                .add_regions([body_region])
                .build()
                .unwrap();
                block.append_operation(for_op);
            }
            _ => {
                panic!("Not implemented: {:?}", stmt)
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
                    let ty_str = ty.to_string();
                    if ty_str.starts_with("memref<") && !ty_str.contains("x") {
                        let inner_ty_str = &ty_str[7..ty_str.len() - 1];
                        let inner_ty = Type::parse(self.context, inner_ty_str).unwrap();
                        let load_op = melior::ir::operation::OperationBuilder::new(
                            "memref.load",
                            Location::unknown(self.context),
                        )
                        .add_operands(&[*val])
                        .add_results(&[inner_ty])
                        .build()
                        .unwrap();
                        let load_ref = block.append_operation(load_op);
                        (load_ref.result(0).unwrap().into(), inner_ty)
                    } else {
                        (*val, *ty)
                    }
                } else {
                    panic!("Undefined variable: {}", name);
                }
            }
            Expr::BinaryOp(lhs, op, rhs, _) => {
                let (mut lhs_val, lhs_ty) = self.generate_expr(lhs, block);
                let (mut rhs_val, rhs_ty) = self.generate_expr(rhs, block);

                let mut final_ty = lhs_ty;

                if lhs_ty != rhs_ty
                    && ((lhs_ty.to_string() == "index" && rhs_ty.to_string() == "i32")
                        || (lhs_ty.to_string() == "i32" && rhs_ty.to_string() == "index"))
                {
                    if lhs_ty.to_string() == "index" && rhs_ty.to_string() == "i32" {
                        let cast_op = melior::ir::operation::OperationBuilder::new(
                            "arith.index_cast",
                            Location::unknown(self.context),
                        )
                        .add_operands(&[rhs_val])
                        .add_results(&[lhs_ty])
                        .build()
                        .unwrap();
                        rhs_val = block.append_operation(cast_op).result(0).unwrap().into();
                    } else {
                        let cast_op = melior::ir::operation::OperationBuilder::new(
                            "arith.index_cast",
                            Location::unknown(self.context),
                        )
                        .add_operands(&[lhs_val])
                        .add_results(&[rhs_ty])
                        .build()
                        .unwrap();
                        lhs_val = block.append_operation(cast_op).result(0).unwrap().into();
                        final_ty = rhs_ty;
                    }
                }

                let is_float =
                    final_ty.to_string().contains("f32") || final_ty.to_string().contains("f64");

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
                    // Compare operations
                    BinaryOp::Eq => {
                        if is_float {
                            "arith.cmpf"
                        } else {
                            "arith.cmpi"
                        }
                    }
                    BinaryOp::NotEq => {
                        if is_float {
                            "arith.cmpf"
                        } else {
                            "arith.cmpi"
                        }
                    }
                    BinaryOp::Lt => {
                        if is_float {
                            "arith.cmpf"
                        } else {
                            "arith.cmpi"
                        }
                    }
                    BinaryOp::Gt => {
                        if is_float {
                            "arith.cmpf"
                        } else {
                            "arith.cmpi"
                        }
                    }
                    BinaryOp::Le => {
                        if is_float {
                            "arith.cmpf"
                        } else {
                            "arith.cmpi"
                        }
                    }
                    BinaryOp::Ge => {
                        if is_float {
                            "arith.cmpf"
                        } else {
                            "arith.cmpi"
                        }
                    }
                    _ => panic!("Unsupported binary op"),
                };

                let mut builder = melior::ir::operation::OperationBuilder::new(
                    op_name,
                    Location::unknown(self.context),
                );
                builder = builder.add_operands(&[lhs_val, rhs_val]);

                // Comparisons return i1
                let ret_ty = if matches!(
                    op,
                    BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div
                ) {
                    builder = builder.add_results(&[final_ty]);
                    final_ty
                } else {
                    let i1_ty = Type::parse(self.context, "i1").unwrap();
                    builder = builder.add_results(&[i1_ty]);

                    // Add predicate attribute
                    let pred_val: i64 = if is_float {
                        match op {
                            BinaryOp::Eq => 1,    // oeq
                            BinaryOp::Gt => 2,    // ogt
                            BinaryOp::Ge => 3,    // oge
                            BinaryOp::Lt => 4,    // olt
                            BinaryOp::Le => 5,    // ole
                            BinaryOp::NotEq => 6, // one
                            _ => 0,
                        }
                    } else {
                        match op {
                            BinaryOp::Eq => 0,    // eq
                            BinaryOp::NotEq => 1, // ne
                            BinaryOp::Lt => 2,    // slt
                            BinaryOp::Le => 3,    // sle
                            BinaryOp::Gt => 4,    // sgt
                            BinaryOp::Ge => 5,    // sge
                            _ => 0,
                        }
                    };
                    let i64_ty = Type::parse(self.context, "i64").unwrap();
                    builder = builder.add_attributes(&[(
                        melior::ir::Identifier::new(self.context, "predicate"),
                        melior::ir::attribute::IntegerAttribute::new(i64_ty, pred_val).into(),
                    )]);
                    i1_ty
                };

                let bin_op = builder.build().unwrap();
                let bin_ref = block.append_operation(bin_op);
                (bin_ref.result(0).unwrap().into(), ret_ty)
            }
            Expr::StructInit(name, fields, _) => {
                let struct_decl = self.structs.get(name).unwrap().clone();
                let struct_ty = self.lower_type(&crate::ast::Type::Struct(name.clone(), None));

                let undef_op = melior::ir::operation::OperationBuilder::new(
                    "llvm.mlir.undef",
                    Location::unknown(self.context),
                )
                .add_results(&[struct_ty])
                .build()
                .unwrap();
                let mut current_struct = block.append_operation(undef_op).result(0).unwrap().into();

                for (field_name, f_expr) in fields {
                    let field_idx = struct_decl
                        .fields
                        .iter()
                        .position(|(n, _)| n == field_name)
                        .unwrap();
                    let field_ty = self.lower_type(&struct_decl.fields[field_idx].1);
                    let (mut field_val, expr_ty) = self.generate_expr(f_expr, block);

                    if expr_ty != field_ty
                        && ((expr_ty.to_string() == "index" && field_ty.to_string() == "i32")
                            || (expr_ty.to_string() == "i32" && field_ty.to_string() == "index"))
                    {
                        let cast_op = melior::ir::operation::OperationBuilder::new(
                            "arith.index_cast",
                            Location::unknown(self.context),
                        )
                        .add_operands(&[field_val])
                        .add_results(&[field_ty])
                        .build()
                        .unwrap();
                        field_val = block.append_operation(cast_op).result(0).unwrap().into();
                    }

                    let pos_attr = melior::ir::attribute::DenseI64ArrayAttribute::new(
                        self.context,
                        &[field_idx as i64],
                    );

                    let insert_op = melior::ir::operation::OperationBuilder::new(
                        "llvm.insertvalue",
                        Location::unknown(self.context),
                    )
                    .add_operands(&[current_struct, field_val])
                    .add_attributes(&[(
                        melior::ir::Identifier::new(self.context, "position"),
                        pos_attr.into(),
                    )])
                    .add_results(&[struct_ty])
                    .build()
                    .unwrap();
                    current_struct = block.append_operation(insert_op).result(0).unwrap().into();
                }
                (current_struct, struct_ty)
            }
            Expr::MemberAccess(base, member, _) => {
                let (base_val, base_ty) = self.generate_expr(base, block);
                let base_ty_str = base_ty.to_string();

                let mut struct_name_opt = None;
                if let Some(start_idx) = base_ty_str.find('\"') {
                    if let Some(end_idx) = base_ty_str[start_idx + 1..].find('\"') {
                        struct_name_opt =
                            Some(base_ty_str[start_idx + 1..start_idx + 1 + end_idx].to_string());
                    }
                }

                if let Some(struct_name) = struct_name_opt {
                    if let Some(struct_decl) = self.structs.get(&struct_name).cloned() {
                        if let Some(field_idx) =
                            struct_decl.fields.iter().position(|(n, _)| n == member)
                        {
                            let field_ty = self.lower_type(&struct_decl.fields[field_idx].1);
                            let pos_attr = melior::ir::attribute::DenseI64ArrayAttribute::new(
                                self.context,
                                &[field_idx as i64],
                            );

                            let ext_op = melior::ir::operation::OperationBuilder::new(
                                "llvm.extractvalue",
                                Location::unknown(self.context),
                            )
                            .add_operands(&[base_val])
                            .add_attributes(&[(
                                melior::ir::Identifier::new(self.context, "position"),
                                pos_attr.into(),
                            )])
                            .add_results(&[field_ty])
                            .build()
                            .unwrap();
                            let ext_ref = block.append_operation(ext_op);
                            return (ext_ref.result(0).unwrap().into(), field_ty);
                        }
                    }
                }
                panic!("Cannot resolve member access {}", member);
            }
            Expr::FunctionCall(name, args, _) => {
                if let Some((ret_ty, arg_tys)) = self.functions.get(name).cloned() {
                    let mut arg_vals = Vec::new();
                    for (i, arg) in args.iter().enumerate() {
                        let (mut arg_val, expr_ty) = self.generate_expr(arg, block);
                        let field_ty = arg_tys[i];
                        if expr_ty != field_ty
                            && ((expr_ty.to_string() == "index" && field_ty.to_string() == "i32")
                                || (expr_ty.to_string() == "i32"
                                    && field_ty.to_string() == "index"))
                        {
                            let cast_op = melior::ir::operation::OperationBuilder::new(
                                "arith.index_cast",
                                Location::unknown(self.context),
                            )
                            .add_operands(&[arg_val])
                            .add_results(&[field_ty])
                            .build()
                            .unwrap();
                            arg_val = block.append_operation(cast_op).result(0).unwrap().into();
                        }
                        arg_vals.push(arg_val);
                    }

                    let name_attr =
                        melior::ir::attribute::FlatSymbolRefAttribute::new(self.context, name);
                    let mut builder = melior::ir::operation::OperationBuilder::new(
                        "func.call",
                        Location::unknown(self.context),
                    )
                    .add_operands(&arg_vals)
                    .add_attributes(&[(
                        melior::ir::Identifier::new(self.context, "callee"),
                        name_attr.into(),
                    )]);

                    if ret_ty.to_string() != "none" {
                        builder = builder.add_results(&[ret_ty]);
                        let call_op = builder.build().unwrap();
                        let call_ref = block.append_operation(call_op);
                        (call_ref.result(0).unwrap().into(), ret_ty)
                    } else {
                        let call_op = builder.build().unwrap();
                        block.append_operation(call_op);
                        let none_ty = Type::parse(self.context, "none").unwrap();
                        // this value shouldn't be used
                        let dummy_op = melior::ir::operation::OperationBuilder::new(
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
                        (
                            block.append_operation(dummy_op).result(0).unwrap().into(),
                            none_ty,
                        )
                    }
                } else {
                    panic!("Function {} not found", name);
                }
            }
            Expr::MethodCall(base, method_name, args, span) => {
                let mut new_args = vec![*base.clone()];
                new_args.extend(args.clone());
                self.generate_expr(
                    &Expr::FunctionCall(method_name.clone(), new_args, span.clone()),
                    block,
                )
            }
            Expr::Array(..) | Expr::MemorySpace(..) | Expr::Topology(..) => {
                panic!("Should not be evaluated directly")
            }
            Expr::If(cond, then_block, else_block_opt, _) => {
                let (cond_val, _) = self.generate_expr(cond, block);

                let then_region = Region::new();
                let then_b = Block::new(&[]);
                for stmt in then_block {
                    self.generate_statement(stmt, &then_b);
                }
                let yield_op = melior::ir::operation::OperationBuilder::new(
                    "scf.yield",
                    Location::unknown(self.context),
                )
                .build()
                .unwrap();
                then_b.append_operation(yield_op);
                then_region.append_block(then_b);

                let else_region = Region::new();
                let else_b = Block::new(&[]);
                if let Some(else_block) = else_block_opt {
                    for stmt in else_block {
                        self.generate_statement(stmt, &else_b);
                    }
                }
                let yield_op = melior::ir::operation::OperationBuilder::new(
                    "scf.yield",
                    Location::unknown(self.context),
                )
                .build()
                .unwrap();
                else_b.append_operation(yield_op);
                else_region.append_block(else_b);

                let if_op = melior::ir::operation::OperationBuilder::new(
                    "scf.if",
                    Location::unknown(self.context),
                )
                .add_operands(&[cond_val])
                .add_regions([then_region, else_region])
                .build()
                .unwrap();

                block.append_operation(if_op);

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
