use super::*;

pub struct MeliorGenerator<'c> {
    pub(crate) context: &'c Context,
    pub(crate) module: Module<'c>,
    pub(crate) env: HashMap<String, (Value<'c, 'c>, Type<'c>)>,
    pub(crate) structs: HashMap<String, crate::ast::StructDecl>,
    #[allow(clippy::type_complexity)]
    pub(crate) enums: HashMap<String, Vec<(String, Option<Vec<crate::ast::Type>>)>>,
    pub(crate) functions: HashMap<String, (Type<'c>, Vec<Type<'c>>)>,
    pub(crate) enzyme_decls: std::collections::HashSet<String>,
    pub string_counter: usize,
    pub current_return_type: Option<Type<'c>>,
    pub expected_type: Option<Type<'c>>,
    pub in_spawn: bool,
}

impl<'c> MeliorGenerator<'c> {
    pub fn coerce_type(
        &mut self,
        block: &melior::ir::Block<'c>,
        val: Value<'c, 'c>,
        from_ty: Type<'c>,
        to_ty: Type<'c>,
    ) -> Value<'c, 'c> {
        if from_ty == to_ty {
            return val;
        }
        let from_str = from_ty.to_string();
        let to_str = to_ty.to_string();

        if (from_str == "index" && (to_str.starts_with("i") || to_str.starts_with("u")))
            || ((from_str.starts_with("i") || from_str.starts_with("u")) && to_str == "index")
        {
            let cast_op = melior::ir::operation::OperationBuilder::new(
                "arith.index_cast",
                Location::unknown(self.context),
            )
            .add_operands(&[val])
            .add_results(&[to_ty])
            .build()
            .unwrap();
            return block.append_operation(cast_op).result(0).unwrap().into();
        }

        if from_str == "i32" && to_str == "i64" {
            let cast_op = melior::ir::operation::OperationBuilder::new(
                "arith.extsi",
                Location::unknown(self.context),
            )
            .add_operands(&[val])
            .add_results(&[to_ty])
            .build()
            .unwrap();
            return block.append_operation(cast_op).result(0).unwrap().into();
        }
        if from_str == "i64" && to_str == "i32" {
            let cast_op = melior::ir::operation::OperationBuilder::new(
                "arith.trunci",
                Location::unknown(self.context),
            )
            .add_operands(&[val])
            .add_results(&[to_ty])
            .build()
            .unwrap();
            return block.append_operation(cast_op).result(0).unwrap().into();
        }
        if from_str == "f32" && to_str == "f64" {
            let cast_op = melior::ir::operation::OperationBuilder::new(
                "arith.extf",
                Location::unknown(self.context),
            )
            .add_operands(&[val])
            .add_results(&[to_ty])
            .build()
            .unwrap();
            return block.append_operation(cast_op).result(0).unwrap().into();
        }
        if from_str == "f32" && (to_str == "bf16" || to_str == "f16") {
            let cast_op = melior::ir::operation::OperationBuilder::new(
                "arith.truncf",
                Location::unknown(self.context),
            )
            .add_operands(&[val])
            .add_results(&[to_ty])
            .build()
            .unwrap();
            return block.append_operation(cast_op).result(0).unwrap().into();
        }
        if (from_str == "bf16" || from_str == "f16") && to_str == "f32" {
            let cast_op = melior::ir::operation::OperationBuilder::new(
                "arith.extf",
                Location::unknown(self.context),
            )
            .add_operands(&[val])
            .add_results(&[to_ty])
            .build()
            .unwrap();
            return block.append_operation(cast_op).result(0).unwrap().into();
        }
        if (from_str == "f64" || from_str == "f32")
            && (to_str == "f32" || to_str == "f16" || to_str == "bf16")
        {
            let cast_op = melior::ir::operation::OperationBuilder::new(
                "arith.truncf",
                Location::unknown(self.context),
            )
            .add_operands(&[val])
            .add_results(&[to_ty])
            .build()
            .unwrap();
            return block.append_operation(cast_op).result(0).unwrap().into();
        }

        if (from_str == "bf16" || from_str == "f16") && (to_str == "f32" || to_str == "f64") {
            let cast_op = melior::ir::operation::OperationBuilder::new(
                "arith.extf",
                Location::unknown(self.context),
            )
            .add_operands(&[val])
            .add_results(&[to_ty])
            .build()
            .unwrap();
            return block.append_operation(cast_op).result(0).unwrap().into();
        }

        if from_str == "i32" && (to_str == "f32" || to_str == "f64") {
            let cast_op = melior::ir::operation::OperationBuilder::new(
                "arith.sitofp",
                Location::unknown(self.context),
            )
            .add_operands(&[val])
            .add_results(&[to_ty])
            .build()
            .unwrap();
            return block.append_operation(cast_op).result(0).unwrap().into();
        }

        if (from_str == "f32" || from_str == "f64") && to_str == "i32" {
            let cast_op = melior::ir::operation::OperationBuilder::new(
                "arith.fptosi",
                Location::unknown(self.context),
            )
            .add_operands(&[val])
            .add_results(&[to_ty])
            .build()
            .unwrap();
            return block.append_operation(cast_op).result(0).unwrap().into();
        }

        println!(
            "Warning: Falling back to bitcast from {} to {}\nBacktrace:\n{:?}",
            from_str,
            to_str,
            std::backtrace::Backtrace::force_capture()
        );
        // Default to unrealized_conversion_cast if nothing matches but we need a cast
        let cast_op = melior::ir::operation::OperationBuilder::new(
            "builtin.unrealized_conversion_cast",
            Location::unknown(self.context),
        )
        .add_operands(&[val])
        .add_results(&[to_ty])
        .build()
        .unwrap();
        block.append_operation(cast_op).result(0).unwrap().into()
    }

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
            enzyme_decls: std::collections::HashSet::new(),
            string_counter: 0,
            current_return_type: None,
            expected_type: None,
            in_spawn: false,
        }
    }

    pub fn into_module(self) -> Module<'c> {
        self.module
    }

    pub fn generate(&mut self, program: &Program, modules: &HashMap<String, Program>) -> String {
        println!("[CODEGEN] Starting MLIR generation...");
        let location = Location::unknown(self.context);
        self.module = melior::ir::Module::new(location);

        self.generate_module(program, modules);

        println!("[CODEGEN] Finished generating modules.");
        let op = self.module.as_operation();
        let s = format!("{}", op);
        println!("[CODEGEN] Formatted MLIR string.");
        s
    }

    pub(crate) fn generate_module(
        &mut self,
        program: &Program,
        modules: &HashMap<String, Program>,
    ) {
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

        // Declare printMemref functions
        for ty_str in &["f32", "f64", "i32", "i64", "bf16"] {
            let func_name = format!("printMemref{}", ty_str.to_uppercase());
            let unranked_memref_ty =
                Type::parse(self.context, &format!("memref<*x{}>", ty_str)).unwrap();
            let _none_ty = Type::parse(self.context, "none").unwrap();

            let func_ty = melior::ir::attribute::TypeAttribute::new(
                Type::parse(self.context, &format!("({unranked_memref_ty}) -> ()")).unwrap(),
            );

            let decl = melior::ir::operation::OperationBuilder::new(
                "func.func",
                Location::unknown(self.context),
            )
            .add_attributes(&[
                (
                    melior::ir::Identifier::new(self.context, "sym_name"),
                    melior::ir::attribute::StringAttribute::new(self.context, &func_name).into(),
                ),
                (
                    melior::ir::Identifier::new(self.context, "function_type"),
                    func_ty.into(),
                ),
                (
                    melior::ir::Identifier::new(self.context, "sym_visibility"),
                    melior::ir::attribute::StringAttribute::new(self.context, "private").into(),
                ),
            ])
            .add_regions([melior::ir::Region::new()])
            .build()
            .unwrap();

            self.module.body().append_operation(decl);
        }

        for module_prog in modules.values() {
            for ext in &module_prog.externs {
                let ret_ty = self.lower_type(&ext.return_type);
                let mut arg_tys = Vec::new();
                for (_, ty) in &ext.params {
                    arg_tys.push(self.lower_type(ty));
                }
                self.functions.insert(ext.name.clone(), (ret_ty, arg_tys));
            }
        }

        let mut operations = Vec::new();

        for module_prog in modules.values() {
            for func in &module_prog.functions {
                let ret_ty = self.lower_type(&func.return_type);
                let mut arg_tys = Vec::new();
                for (_, ty) in &func.params {
                    arg_tys.push(self.lower_type(ty));
                }
                self.functions.insert(func.name.clone(), (ret_ty, arg_tys));
            }
        }

        for func in &program.functions {
            let ret_ty = self.lower_type(&func.return_type);
            let mut arg_tys = Vec::new();
            for (_, ty) in &func.params {
                arg_tys.push(self.lower_type(ty));
            }
            self.functions.insert(func.name.clone(), (ret_ty, arg_tys));
        }

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

        let mut all_externs = program.externs.clone();
        for module_prog in modules.values() {
            all_externs.extend(module_prog.externs.clone());
        }

        for ext in &all_externs {
            let name = &ext.name;
            let (ret_ty, arg_tys) = self.functions.get(name).unwrap();
            // FunctionType::new takes arg_tys and ret_tys
            let func_type =
                melior::ir::r#type::FunctionType::new(self.context, arg_tys, &[*ret_ty]);

            // Define the string attribute for the function name
            let name_attr = melior::ir::attribute::StringAttribute::new(self.context, name);
            let type_attr = melior::ir::attribute::TypeAttribute::new(func_type.into());

            let region = melior::ir::Region::new();
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
            .add_regions([region])
            .build()
            .unwrap();

            body.append_operation(func_op);
        }

        for op in operations {
            body.append_operation(op);
        }
    }

    pub(crate) fn generate_function(&mut self, func: &Function) -> melior::ir::Operation<'c> {
        let is_main = func.name == "main";
        let true_ret_ty = self.lower_type(&func.return_type);
        let ret_ty = if is_main {
            Type::parse(self.context, "i32").unwrap()
        } else {
            true_ret_ty
        };

        let mut arg_tys = Vec::new();
        for (_, ty) in &func.params {
            let lowered =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.lower_type(ty)));
            match lowered {
                Ok(t) => arg_tys.push(t),
                Err(_e) => {
                    panic!(
                        "Failed to lower type for function parameter in {}: {:?}",
                        func.name, ty
                    );
                }
            }
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

        self.current_return_type = Some(ret_ty);
        for stmt in &func.body {
            if is_main {
                if let Statement::Return(_) = stmt {
                    continue;
                }
            }
            self.generate_statement(stmt, &block);
        }

        if is_main {
            let i32_ty = Type::parse(self.context, "i32").unwrap();
            let c0_op = block.append_operation(
                melior::ir::operation::OperationBuilder::new(
                    "arith.constant",
                    Location::unknown(self.context),
                )
                .add_results(&[i32_ty])
                .add_attributes(&[(
                    melior::ir::Identifier::new(self.context, "value"),
                    melior::ir::attribute::IntegerAttribute::new(i32_ty, 0).into(),
                )])
                .build()
                .unwrap(),
            );
            let c0 = c0_op.result(0).unwrap().into();
            block.append_operation(
                melior::ir::operation::OperationBuilder::new(
                    "func.return",
                    Location::unknown(self.context),
                )
                .add_operands(&[c0])
                .build()
                .unwrap(),
            );
        }

        self.current_return_type = None;

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

    pub(crate) fn generate_statement(&mut self, stmt: &Statement, block: &melior::ir::Block<'c>) {
        match stmt {
            Statement::Return(s) => s.lower(self, block),
            Statement::LetDecl(s) => s.lower(self, block),
            Statement::Assign(s) => s.lower(self, block),
            Statement::CompoundAssign(s) => s.lower(self, block),
            Statement::ExprStmt(s) => s.lower(self, block),
            Statement::ForLoop(s) => s.lower(self, block),
            Statement::Assert(_) => {
                // TODO: Lower to `scf.if` with panic/abort for runtime checks
            }
            Statement::Loop(_) | Statement::Break(_) => {
                todo!("Phase 2: MLIR Codegen for Loop and Break");
            }
        }
    }

    pub(crate) fn generate_expr(
        &mut self,
        expr: &Expr,
        block: &melior::ir::Block<'c>,
    ) -> (Value<'c, 'c>, Type<'c>) {
        match expr {
            Expr::Identifier(e) => e.lower(self, block),
            Expr::BinaryOp(e) => e.lower(self, block),
            Expr::RelationalOp(e) => e.lower(self, block),
            Expr::LogicalOp(e) => e.lower(self, block),
            Expr::UnaryOp(e) => e.lower(self, block),
            Expr::StructInit(e) => e.lower(self, block),
            Expr::MemberAccess(e) => e.lower(self, block),
            Expr::IndexAccess(e) => e.lower(self, block),
            Expr::FunctionCall(e) => e.lower(self, block),
            Expr::MethodCall(e) => e.lower(self, block),
            Expr::SpawnOn(e) => e.lower(self, block),
            Expr::Array(e) => e.lower(self, block),
            Expr::If(e) => e.lower(self, block),
            Expr::Number(e) => e.lower(self, block),
            Expr::UnsafeBlock(e) => e.lower(self, block),
            Expr::Grad(e) => e.lower(self, block),
            Expr::Vjp(e) => e.lower(self, block),
            Expr::Jvp(e) => e.lower(self, block),
            Expr::Transfer(e) => e.lower(self, block),
            Expr::Borrow(e) => e.lower(self, block),
            Expr::StringLiteral(e) => e.lower(self, block),
            Expr::ComptimeBlock(e) => e.lower(self, block),
            Expr::Dereference(e) => e.lower(self, block),
            _ => todo!("{:?}", expr),
        }
    }

    pub(crate) fn lower_type(&self, ty: &crate::ast::Type) -> Type<'c> {
        let ty_str = match ty {
            crate::ast::Type::Tensor(el_ty, dims, top) => {
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
                    ElementType::Generic(_) => {
                        panic!("Generic element type should be instantiated before codegen")
                    }
                };

                let mut shape_str = String::new();
                if dims.is_empty() {
                    shape_str = "?x?".to_string();
                } else {
                    for (i, dim) in dims.iter().enumerate() {
                        if let crate::ast::Expr::Number(NumberExpr {
                            value: n_str,
                            ty: _,
                            span: _,
                        }) = dim
                        {
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

                let addr_space = match top {
                    Some(crate::ast::Topology::NPU(_))
                    | Some(crate::ast::Topology::Slice(_, _, _))
                    | Some(crate::ast::Topology::ANE) => 1,
                    Some(crate::ast::Topology::AccCore(_)) => 2,
                    Some(crate::ast::Topology::Host)
                    | Some(crate::ast::Topology::AMX)
                    | Some(crate::ast::Topology::GPU)
                    | None => 0,
                };

                if addr_space != 0 {
                    format!("memref<{}{}, {}>", shape_str, ty_str, addr_space)
                } else {
                    format!("memref<{}{}>", shape_str, ty_str)
                }
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
                ElementType::Generic(_) => {
                    panic!("Generic element type should be instantiated before codegen")
                }
            }
            .to_string(),
            crate::ast::Type::Matrix => "tensor<?x?xf32>".to_string(),
            crate::ast::Type::Ref(inner, _mem) => {
                return self.lower_type(inner);
            }
            crate::ast::Type::Verified(inner) => return self.lower_type(inner),
            crate::ast::Type::Pinned(inner, _top) => {
                let inner_ty_str = self.lower_type(inner).to_string();
                inner_ty_str
            }
            crate::ast::Type::Borrow(_, mem, _, _) | crate::ast::Type::Pointer(_, mem, _) => {
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
                panic!(
                    "Generic types should have been monomorphized before codegen! Got type: {:?}",
                    ty
                );
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
                    ElementType::Generic(_) => {
                        panic!("Generic element type should be instantiated before codegen")
                    }
                };
                format!("vector<{}x{}>", n, ty_str)
            }
            crate::ast::Type::Enum(_, _) => "i32".to_string(),
            crate::ast::Type::Module(..) => "none".to_string(),
        };

        Type::parse(self.context, &ty_str)
            .unwrap_or_else(|| panic!("Failed to parse MLIR type: {}", ty_str))
    }

    pub(crate) fn lower_type_str(&self, ty: &crate::ast::Type) -> String {
        let t = self.lower_type(ty);
        t.to_string()
    }

    pub fn flatten_indices(
        &mut self,
        expr: &Expr,
        block: &melior::ir::Block<'c>,
    ) -> Option<(Value<'c, 'c>, Type<'c>, Vec<Value<'c, 'c>>)> {
        match expr {
            Expr::IndexAccess(crate::ast::IndexAccessExpr {
                base,
                index: idx,
                span: _,
            }) => {
                let (base_val, base_ty, mut indices) = self.flatten_indices(base, block)?;
                let (idx_val, _) = self.generate_expr(idx, block);

                let idx_ty_str = idx_val.r#type().to_string();
                let actual_idx = if idx_ty_str != "index" {
                    let cast_op = melior::ir::operation::OperationBuilder::new(
                        "arith.index_cast",
                        Location::unknown(self.context),
                    )
                    .add_operands(&[idx_val])
                    .add_results(&[Type::parse(self.context, "index").unwrap()])
                    .build()
                    .unwrap();
                    block.append_operation(cast_op).result(0).unwrap().into()
                } else {
                    idx_val
                };

                indices.push(actual_idx);
                Some((base_val, base_ty, indices))
            }
            _ => {
                let (val, ty) = self.generate_expr(expr, block);
                Some((val, ty, Vec::new()))
            }
        }
    }
}
