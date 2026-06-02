use super::*;

pub struct MeliorGenerator<'c> {
    pub(crate) context: &'c Context,
    pub(crate) module: Module<'c>,
    pub(crate) env: HashMap<String, (Value<'c, 'c>, Type<'c>)>,
    pub(crate) ast_env: HashMap<String, crate::ast::Type>,
    pub(crate) structs: HashMap<String, crate::ast::StructDecl>,
    #[allow(clippy::type_complexity)]
    pub(crate) enums: HashMap<String, Vec<(String, Option<Vec<crate::ast::Type>>)>>,
    pub(crate) functions: HashMap<String, (Type<'c>, Vec<Type<'c>>)>,
    pub(crate) ast_functions: HashMap<String, crate::ast::Function>,
    pub(crate) enzyme_decls: std::collections::HashSet<String>,
    pub string_counter: usize,
    pub current_return_type: Option<Type<'c>>,
    pub expected_type: Option<Type<'c>>,
    pub in_spawn: bool,
    pub break_flags: Vec<melior::ir::Value<'c, 'c>>,
    pub continue_flags: Vec<melior::ir::Value<'c, 'c>>,
    pub allocs: std::collections::HashSet<String>,
    pub is_lvalue_context: bool,
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

        if val.r#type() == to_ty {
            return val;
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
            ast_env: HashMap::new(),
            structs: HashMap::new(),
            enums: HashMap::new(),
            functions: HashMap::new(),
            ast_functions: HashMap::new(),
            enzyme_decls: std::collections::HashSet::new(),
            string_counter: 0,
            current_return_type: None,
            expected_type: None,
            in_spawn: false,
            break_flags: Vec::new(),
            continue_flags: Vec::new(),
            allocs: std::collections::HashSet::new(),
            is_lvalue_context: false,
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
        std::fs::write("mlir_dump.mlir", &s).unwrap();
        println!("[CODEGEN] Formatted MLIR string to mlir_dump.mlir.");
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
        for module in modules.values() {
            for s in &module.structs {
                self.structs.insert(s.name.clone(), s.clone());
            }
            for e in &module.enums {
                self.enums.insert(e.name.clone(), e.variants.clone());
            }
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
                self.ast_functions.insert(func.name.clone(), func.clone());
            }
        }

        for func in &program.functions {
            let ret_ty = self.lower_type(&func.return_type);
            let mut arg_tys = Vec::new();
            for (_, ty) in &func.params {
                arg_tys.push(self.lower_type(ty));
            }
            self.functions.insert(func.name.clone(), (ret_ty, arg_tys));
            self.ast_functions.insert(func.name.clone(), func.clone());
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

        let mut seen_externs = std::collections::HashSet::new();
        let mut unique_externs = Vec::new();
        for ext in all_externs {
            let sig = format!("{}: {:?}", ext.name, ext.params);
            if seen_externs.insert(sig) {
                unique_externs.push(ext);
            }
        }

        for ext in &unique_externs {
            let name = &ext.name;
            if name == "printf" || name == "vx_internal_printf" {
                continue;
            }
            let (ret_ty, arg_tys) = self.functions.get(name).unwrap();

            // FunctionType::new takes arg_tys and ret_tys
            let func_type =
                melior::ir::r#type::FunctionType::new(self.context, arg_tys, &[*ret_ty]);

            // Define the string attribute for the function name
            let name_attr = melior::ir::attribute::StringAttribute::new(self.context, name);
            let type_attr = melior::ir::attribute::TypeAttribute::new(func_type.into());

            let builder = melior::ir::operation::OperationBuilder::new(
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
            ]);

            let region = melior::ir::Region::new();
            let func_op = builder.add_regions([region]).build().unwrap();

            body.append_operation(func_op);
        }

        for op in operations {
            body.append_operation(op);
        }
    }

    pub(crate) fn generate_function(&mut self, func: &Function) -> melior::ir::Operation<'c> {
        self.env.clear();
        self.allocs.clear();
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
        for (i, (name, ast_ty)) in func.params.iter().enumerate() {
            let arg_val = block.argument(i).unwrap().into();
            self.env.insert(name.clone(), (arg_val, arg_tys[i]));
            self.ast_env.insert(name.clone(), ast_ty.clone());
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
            Statement::Loop(s) => s.lower(self, block),
            Statement::Break(s) => s.lower(self, block),
            Statement::Continue(s) => s.lower(self, block),
            Statement::MacroCall(_) => panic!("Macros should be expanded before codegen"),
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
            Expr::EnumVariant(e) => e.lower(self, block),
            Expr::Match(e) => e.lower(self, block),
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
        println!("lower_type: {:?}", ty);
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
                if let Some(enum_def) = self.enums.get(name) {
                    if name.starts_with("Option<") {
                        let mut payload_ty_str = "none".to_string();
                        for (v_name, payload) in enum_def {
                            if v_name == "Some" {
                                if let Some(types) = payload {
                                    if !types.is_empty() {
                                        let mut lowered = self.lower_type_str(&types[0]);
                                        if lowered.starts_with("memref<") {
                                            lowered = "!llvm.ptr".to_string();
                                        }
                                        payload_ty_str = lowered;
                                    }
                                }
                            }
                        }
                        return Type::parse(
                            self.context,
                            &format!("!llvm.struct<\"{}\", (i32, {})>", name, payload_ty_str),
                        )
                        .unwrap();
                    }
                    return Type::parse(self.context, "i32").unwrap();
                }
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
            crate::ast::Type::GenericInstance(base, args) => {
                if let crate::ast::Type::Struct(name, _) = &**base {
                    if let Some(decl) = self.structs.get(name).cloned() {
                        let mut field_types = Vec::new();
                        let mut mapping = std::collections::HashMap::new();
                        for (i, param) in decl.generics.iter().enumerate() {
                            if i >= args.len() {
                                panic!("Not enough arguments for generic instance {} (expected {}, got {})", name, decl.generics.len(), args.len());
                            }
                            mapping.insert(param.0.clone(), args[i].clone());
                        }
                        for (_, ty) in &decl.fields {
                            let sub_ty = ty.substitute(&mapping);
                            let mut lowered = self.lower_type_str(&sub_ty);
                            if lowered.starts_with("memref<") {
                                lowered = "!llvm.ptr".to_string();
                            }
                            field_types.push(lowered);
                        }
                        let args_str: Vec<String> = args
                            .iter()
                            .map(|a| {
                                let lowered = self.lower_type_str(a);
                                lowered
                                    .replace("!", "")
                                    .replace("<", "_")
                                    .replace(">", "_")
                                    .replace(" ", "_")
                                    .replace(",", "_")
                                    .replace("\"", "")
                                    .replace("(", "_")
                                    .replace(")", "_")
                                    .replace(".", "_")
                            })
                            .collect();
                        format!(
                            "!llvm.struct<\"{}_{}\", ({})>",
                            name,
                            args_str.join("_"),
                            field_types.join(", ")
                        )
                    } else if let Some(enum_def) = self.enums.get(name).cloned() {
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

                        let args_str: Vec<String> = args
                            .iter()
                            .map(|a| {
                                let lowered = self.lower_type_str(a);
                                lowered
                                    .replace("!", "")
                                    .replace("<", "_")
                                    .replace(">", "_")
                                    .replace(" ", "_")
                                    .replace(",", "_")
                                    .replace("\"", "")
                                    .replace("(", "_")
                                    .replace(")", "_")
                                    .replace(".", "_")
                            })
                            .collect();

                        format!(
                            "!llvm.struct<\"{}_{}\", (i32, {})>",
                            name,
                            args_str.join("_"),
                            payload_ty_str
                        )
                    } else {
                        println!("Keys in structs: {:?}", self.structs.keys());
                        println!("Keys in enums: {:?}", self.enums.keys());
                        panic!("Generic struct/enum {} not found", name);
                    }
                } else {
                    panic!("GenericInstance base is not a Struct!");
                }
            }
            crate::ast::Type::Generic(_, _) => {
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
            crate::ast::Type::Enum(name, _) => {
                if let Some(enum_def) = self.enums.get(name) {
                    if name.starts_with("Option<") {
                        let mut payload_ty_str = "none".to_string();
                        for (v_name, payload) in enum_def {
                            if v_name == "Some" {
                                if let Some(types) = payload {
                                    if !types.is_empty() {
                                        let mut lowered = self.lower_type_str(&types[0]);
                                        if lowered.starts_with("memref<") {
                                            lowered = "!llvm.ptr".to_string();
                                        }
                                        payload_ty_str = lowered;
                                    }
                                }
                            }
                        }
                        return Type::parse(
                            self.context,
                            &format!("!llvm.struct<\"{}\", (i32, {})>", name, payload_ty_str),
                        )
                        .unwrap();
                    }
                }
                "i32".to_string()
            }
            crate::ast::Type::Function(_, _) => {
                return Type::parse(self.context, "!llvm.ptr").unwrap();
            }
            crate::ast::Type::Module(..) => "none".to_string(),
            crate::ast::Type::Unknown => "unknown".to_string(),
        };

        Type::parse(self.context, &ty_str)
            .unwrap_or_else(|| panic!("Failed to parse MLIR type: {}", ty_str))
    }

    pub(crate) fn lower_type_str(&self, ty: &crate::ast::Type) -> String {
        if let crate::ast::Type::Function(_, _) = ty {
            return "!llvm.ptr".to_string();
        }
        let t = self.lower_type(ty);
        t.to_string()
    }

    pub fn infer_ast_type(&self, expr: &Expr) -> Option<crate::ast::Type> {
        match expr {
            Expr::Identifier(id) => self.ast_env.get(&id.name).cloned(),
            Expr::MemberAccess(ma) => {
                let mut base_ty = self.infer_ast_type(&ma.base)?;
                if let crate::ast::Type::Borrow(inner, _, _, _) = base_ty {
                    base_ty = *inner;
                }
                let mut generic_args = None;
                if let crate::ast::Type::GenericInstance(inner, args) = base_ty {
                    base_ty = *inner;
                    generic_args = Some(args);
                }
                if let crate::ast::Type::Struct(s_name, _) = base_ty {
                    if let Some(decl) = self.structs.get(&s_name) {
                        for (n, t) in &decl.fields {
                            if n == &ma.member {
                                let mut resolved_ty = t.clone();
                                if let Some(args) = generic_args {
                                    let mut mapping = std::collections::HashMap::new();
                                    for (i, param) in decl.generics.iter().enumerate() {
                                        if i < args.len() {
                                            mapping.insert(param.0.clone(), args[i].clone());
                                        }
                                    }
                                    resolved_ty = resolved_ty.substitute(&mapping);
                                }
                                return Some(resolved_ty);
                            }
                        }
                    }
                }
                None
            }
            Expr::FunctionCall(fc) => self
                .ast_functions
                .get(&fc.name)
                .map(|decl| decl.return_type.clone()),
            Expr::MethodCall(mc) => {
                let mut base_ty = self.infer_ast_type(&mc.base)?;
                if let crate::ast::Type::Borrow(inner, _, _, _) = base_ty {
                    base_ty = *inner;
                }
                if let crate::ast::Type::GenericInstance(inner, _) = base_ty {
                    base_ty = *inner;
                }
                if let crate::ast::Type::Struct(s_name, _) = base_ty {
                    let mangled = format!("{}_{}", s_name, mc.method_name);
                    self.ast_functions
                        .get(&mangled)
                        .map(|decl| decl.return_type.clone())
                } else {
                    None
                }
            }
            _ => None,
        }
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
