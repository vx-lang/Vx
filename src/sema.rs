use crate::ast::*;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Bool(bool),
    Number(f64),
}

pub struct GlobalAstEnv<'a> {
    pub structs: HashMap<String, &'a StructDecl>,
    pub enums: HashMap<String, &'a Vec<String>>,
    pub traits: HashMap<String, &'a TraitDecl>,
    pub impls: HashMap<String, Vec<&'a ImplBlock>>,
    pub functions: HashMap<String, (Type, bool)>,
    pub ast_functions: HashMap<String, &'a Function>,
    pub generic_functions: HashMap<String, (&'a Function, u64)>, // (func, origin_module_hash)
}

impl<'a> GlobalAstEnv<'a> {
    pub fn build(modules: &'a [Program]) -> Self {
        let mut env = Self {
            structs: HashMap::new(),
            enums: HashMap::new(),
            traits: HashMap::new(),
            impls: HashMap::new(),
            functions: HashMap::new(),
            ast_functions: HashMap::new(),
            generic_functions: HashMap::new(),
        };

        for module in modules {
            for s in &module.structs {
                env.structs.insert(s.name.clone(), s);
            }
            for e in &module.enums {
                env.enums.insert(e.name.clone(), &e.variants);
            }
            for t in &module.traits {
                env.traits.insert(t.name.clone(), t);
            }
            for i in &module.impls {
                let trait_name = match &i.trait_name {
                    Some(name) => name.clone(),
                    None => "_inherent".to_string(),
                };
                env.impls.entry(trait_name).or_default().push(i);
            }
            for ext in &module.externs {
                env.functions
                    .insert(ext.name.clone(), (ext.return_type.clone(), !ext.is_safe));
            }
            for func in &module.functions {
                if !func.generics.is_empty() {
                    let module_hash = crate::hash::compute_module_hash(&module.module_path);
                    env.generic_functions
                        .insert(func.name.clone(), (func, module_hash));
                } else {
                    env.functions.insert(
                        func.name.clone(),
                        (func.return_type.clone(), false /* func.is_unsafe */),
                    );
                    env.ast_functions.insert(func.name.clone(), func);
                }
            }
        }
        env
    }
}

pub struct TypeChecker<'a> {
    pub worker: &'a mut crate::session::LocalWorkerState,
    pub env: &'a GlobalAstEnv<'a>,
    scopes: Vec<HashMap<String, (Type, Topology)>>,
    pub monomorphized_functions: Vec<(Function, u64)>,
    pub errors: Vec<String>,
    in_unsafe_block: bool,
    active_topology: Topology,
    active_memory: MemorySpace,
    next_reg: u32,
    var_regs: Vec<HashMap<String, u32>>,
}

impl<'a> TypeChecker<'a> {
    pub fn new(
        env: &'a GlobalAstEnv<'a>,
        worker: &'a mut crate::session::LocalWorkerState,
    ) -> Self {
        Self {
            env,
            worker,
            scopes: vec![HashMap::new()],
            monomorphized_functions: Vec::new(),
            errors: Vec::new(),
            in_unsafe_block: false,
            active_topology: Topology::Host,
            active_memory: MemorySpace::HostDRAM,
            next_reg: 0,
            var_regs: vec![HashMap::new()],
        }
    }

    pub fn emit_type(&mut self, _ty: &Type) -> u32 {
        // Dummy conversion for now: Create a synthetic TypeId and push it.
        // In reality, this would hash the struct name, etc.
        let mut tid = crate::gid::TypeId::new(0, 0, 0, 0);
        tid.words[3] |= crate::gid::LOCAL_DEFERRED_BIT;
        let idx = self.worker.local_type_stream.len() as u32;
        self.worker.local_type_stream.push(tid);
        idx
    }

    pub fn emit_inst(&mut self, opcode: u32, operand1: u32, operand2: u32, type_idx: u32) -> u32 {
        let inst = crate::hir::HirInstruction::new(opcode, operand1, operand2, type_idx);
        self.worker.local_hir_stream.push(inst);
        let reg = self.next_reg;
        self.next_reg += 1;
        reg
    }

    pub fn push_reg_scope(&mut self) {
        self.var_regs.push(HashMap::new());
    }

    pub fn pop_reg_scope(&mut self) {
        self.var_regs.pop();
    }
    pub fn push_scope(&mut self) {
        self.scopes.push(std::collections::HashMap::new());
    }

    pub fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    pub fn insert(&mut self, name: String, ty: Type) {
        self.scopes
            .last_mut()
            .unwrap()
            .insert(name, (ty, self.active_topology.clone()));
    }

    pub fn lookup(&self, name: &str) -> Option<&(Type, Topology)> {
        for scope in self.scopes.iter().rev() {
            if let Some(val) = scope.get(name) {
                return Some(val);
            }
        }
        None
    }

    pub fn unify_types(
        &mut self,
        generic_ty: &Type,
        concrete_ty: &Type,
        mapping: &mut std::collections::HashMap<String, Type>,
    ) -> bool {
        match (generic_ty, concrete_ty) {
            (Type::Generic(name, _), _) => {
                if let Some(existing) = mapping.get(name) {
                    existing == concrete_ty
                } else {
                    mapping.insert(name.clone(), concrete_ty.clone());
                    true
                }
            }
            (Type::Tensor(e1, d1, t1), Type::Tensor(e2, d2, t2)) => {
                e1 == e2 && d1 == d2 && t1 == t2
            }
            (Type::Pointer(t1, m1, mut1), Type::Pointer(t2, m2, mut2)) => {
                m1 == m2 && mut1 == mut2 && self.unify_types(t1, t2, mapping)
            }
            (Type::Borrow(t1, m1, mut1), Type::Borrow(t2, m2, mut2)) => {
                m1 == m2 && mut1 == mut2 && self.unify_types(t1, t2, mapping)
            }
            (Type::Ref(t1, m1), Type::Ref(t2, m2)) => m1 == m2 && self.unify_types(t1, t2, mapping),
            (Type::GenericInstance(b1, args1), Type::GenericInstance(b2, args2)) => {
                if args1.len() != args2.len() {
                    return false;
                }
                if !self.unify_types(b1, b2, mapping) {
                    return false;
                }
                for (a1, a2) in args1.iter().zip(args2.iter()) {
                    if !self.unify_types(a1, a2, mapping) {
                        return false;
                    }
                }
                true
            }
            (Type::Struct(n1, _), Type::Struct(n2, _)) => n1 == n2,
            (t1, t2) => t1 == t2,
        }
    }

    pub fn instantiate_function(
        &mut self,
        generic_func: &Function,
        mapping: &std::collections::HashMap<String, Type>,
    ) -> Function {
        let mut mangled_name = generic_func.name.clone();
        for (g_name, _) in &generic_func.generics {
            if let Some(ty) = mapping.get(g_name) {
                let mut type_str = format!("_{:?}", ty)
                    .replace("(", "")
                    .replace(")", "")
                    .replace(" ", "")
                    .replace("[", "")
                    .replace("]", "")
                    .replace(",", "_")
                    .replace("_None", "")
                    .replace("\"", "");
                while type_str.contains("__") {
                    type_str = type_str.replace("__", "_");
                }
                mangled_name.push_str(&type_str);
            }
        }

        let new_params = generic_func
            .params
            .iter()
            .map(|(n, t)| (n.clone(), t.substitute(mapping)))
            .collect();
        let new_ret = generic_func.return_type.substitute(mapping);
        let new_body = generic_func
            .body
            .iter()
            .map(|s| s.substitute(mapping))
            .collect();

        Function {
            name: mangled_name,
            generics: Vec::new(),
            params: new_params,
            return_type: new_ret,
            body: new_body,
        }
    }

    pub fn mangle_path(path: &str) -> String {
        path.replace("/", "_").replace(".", "_")
    }

    pub fn check_function(&mut self, func: &mut Function) {
        if !func.generics.is_empty() {
            return;
        }

        self.push_scope();
        for (name, ty) in &func.params {
            self.insert(name.clone(), ty.clone());
        }

        for stmt in &mut func.body {
            self.check_statement(stmt, &func.return_type.clone());
        }

        self.pop_scope();
    }

    fn check_statement(&mut self, stmt: &mut Statement, return_type: &Type) {
        // Intercept for HIR lowering
        match stmt {
            Statement::Assign(_lhs, rhs, _) => {
                let (ty, rhs_reg) = self.check_expr(rhs);
                let type_idx = self.emit_type(&ty);
                self.emit_inst(crate::hir::OP_STORE, rhs_reg, 0, type_idx);
                // Fallthrough to standard semantic checks
            }
            Statement::Return(expr, _) => {
                let (ty, ret_reg) = self.check_expr(expr);
                let type_idx = self.emit_type(&ty);
                self.emit_inst(crate::hir::OP_RET, ret_reg, 0, type_idx);
                // Fallthrough to standard semantic checks
            }
            Statement::LetDecl(_, _, _, expr, _) => {
                let (ty, val_reg) = self.check_expr(expr);
                let type_idx = self.emit_type(&ty);
                self.emit_inst(crate::hir::OP_STORE, val_reg, 0, type_idx);
            }
            _ => {}
        }

        match stmt {
            Statement::LetDecl(name, _is_mut, ty_ann, expr, _) => {
                let ty = self.check_expr_type(expr);

                if let Some(ann) = ty_ann {
                    if !self.is_assignable(ann, &ty) {
                        self.errors
                            .push(format!("Type mismatch in variable declaration '{}'", name));
                    }
                    self.insert(name.clone(), ann.clone());
                } else {
                    self.insert(name.clone(), ty);
                }
            }
            Statement::ForLoop(iter, start, end, body, _) => {
                self.check_expr_type(start);
                self.check_expr_type(end);
                self.push_scope();
                self.insert(iter.clone(), Type::Scalar(ElementType::I64));
                for s in body {
                    self.check_statement(s, return_type);
                }
                self.pop_scope();
            }
            Statement::If(cond, then_block, else_block, _) => {
                let cond_ty = self.check_expr_type(cond);
                if cond_ty != Type::Scalar(ElementType::Bool) {
                    self.errors
                        .push("Condition in if statement must be of type bool (i1)".to_string());
                }
                self.push_scope();
                for s in then_block {
                    self.check_statement(s, return_type);
                }
                self.pop_scope();

                if let Some(else_b) = else_block {
                    self.push_scope();
                    for s in else_b {
                        self.check_statement(s, return_type);
                    }
                    self.pop_scope();
                }
            }
            Statement::Assign(lhs, rhs, _) | Statement::CompoundAssign(lhs, _, rhs, _) => {
                let lhs_ty = self.check_expr_type(lhs);
                let rhs_ty = self.check_expr_type(rhs);
                if !self.is_assignable(&lhs_ty, &rhs_ty) {
                    self.errors.push("Type mismatch in assignment".to_string());
                }
            }
            Statement::Return(expr, _) => {
                let ty = self.check_expr_type(expr);
                if !self.is_assignable(return_type, &ty) {
                    self.errors.push(format!(
                        "Type mismatch on return. Expected {:?}, got {:?}",
                        return_type, ty
                    ));
                }
            }
            Statement::SpawnOn(top, stmts, _) => {
                let prev_top = self.active_topology.clone();
                let prev_mem = self.active_memory.clone();
                self.active_topology = top.clone();
                self.active_memory = match top {
                    Topology::NPU(_) => MemorySpace::NPUHBM,
                    Topology::AccCore(_) => MemorySpace::LocalSRAM,
                    Topology::Host => MemorySpace::HostDRAM,
                    Topology::AMX => MemorySpace::HostDRAM,
                    Topology::ANE => MemorySpace::NPUHBM,
                    Topology::GPU => MemorySpace::HostDRAM,
                    Topology::Slice(_, _, _) => MemorySpace::NPUHBM,
                };

                self.push_scope();

                // Validate topology expression if it contains one
                match top {
                    Topology::NPU(expr) | Topology::AccCore(expr) => {
                        let _ty = self.check_expr_type(expr);
                    }
                    Topology::Slice(_, start, end) => {
                        let _t1 = self.check_expr_type(start);
                        let _t2 = self.check_expr_type(end);
                    }
                    Topology::Host | Topology::AMX | Topology::ANE | Topology::GPU => {}
                }

                for s in stmts {
                    self.check_statement(s, return_type);
                }

                self.pop_scope();

                self.active_topology = prev_top;
                self.active_memory = prev_mem;
            }
            Statement::ExprStmt(expr, _) => {
                self.check_expr_type(expr);
            }
            Statement::Assert(expr, _msg, _) => {
                let ty = self.check_expr_type(expr);
                if ty != Type::Scalar(ElementType::Bool) {
                    self.errors
                        .push("Assertion condition must be boolean".to_string());
                }
            }
            Statement::Comptime(stmts, _) => {
                for s in stmts {
                    if let Statement::Assert(expr, msg, _) = s {
                        let ty = self.check_expr_type(expr);
                        if ty != Type::Scalar(ElementType::Bool) {
                            self.errors
                                .push("Assertion condition must be boolean".to_string());
                        }
                        let empty_env = HashMap::new();
                        if let Some(Value::Bool(b)) = self.eval_expr(expr, &empty_env) {
                            if !b {
                                let m = msg
                                    .clone()
                                    .unwrap_or_else(|| "Comptime assertion failed".to_string());
                                self.errors.push(format!("Comptime assert failed: {}", m));
                            }
                        } else {
                            self.errors.push(
                                "Could not evaluate comptime assert condition statically"
                                    .to_string(),
                            );
                        }
                    } else {
                        if !self.in_unsafe_block {
                            self.check_statement(&mut *s, return_type);
                        }
                    }
                }
            }
        }
    }

    fn eval_expr(&self, expr: &Expr, env: &HashMap<String, Value>) -> Option<Value> {
        match expr {
            Expr::Number(n_str, _, _) => {
                if let Ok(n) = n_str.parse::<f64>() {
                    Some(Value::Number(n))
                } else {
                    None
                }
            }
            Expr::Identifier(n, _) if n == "true" => Some(Value::Bool(true)),
            Expr::Identifier(n, _) if n == "false" => Some(Value::Bool(false)),
            Expr::Identifier(n, _) => env.get(n).cloned(),
            Expr::BinaryOp(lhs, op, rhs, _) => {
                let l = self.eval_expr(lhs, env)?;
                let r = self.eval_expr(rhs, env)?;
                match (l, r, op) {
                    (Value::Number(a), Value::Number(b), BinaryOp::Eq) => Some(Value::Bool(a == b)),
                    (Value::Number(a), Value::Number(b), BinaryOp::NotEq) => {
                        Some(Value::Bool(a != b))
                    }
                    (Value::Number(a), Value::Number(b), BinaryOp::Lt) => Some(Value::Bool(a < b)),
                    (Value::Number(a), Value::Number(b), BinaryOp::Gt) => Some(Value::Bool(a > b)),
                    (Value::Number(a), Value::Number(b), BinaryOp::Le) => Some(Value::Bool(a <= b)),
                    (Value::Number(a), Value::Number(b), BinaryOp::Ge) => Some(Value::Bool(a >= b)),
                    (Value::Bool(a), Value::Bool(b), BinaryOp::Eq) => Some(Value::Bool(a == b)),
                    (Value::Bool(a), Value::Bool(b), BinaryOp::NotEq) => Some(Value::Bool(a != b)),
                    (Value::Bool(a), Value::Bool(b), BinaryOp::And) => Some(Value::Bool(a && b)),
                    (Value::Bool(a), Value::Bool(b), BinaryOp::Or) => Some(Value::Bool(a || b)),
                    _ => None,
                }
            }
            Expr::UnaryOp(UnaryOp::Not, inner, _) => {
                if let Value::Bool(b) = self.eval_expr(inner, env)? {
                    Some(Value::Bool(!b))
                } else {
                    None
                }
            }
            Expr::FunctionCall(name, args, _) => {
                let func = self.env.ast_functions.get(name)?;
                let mut local_env = HashMap::new();
                for (i, arg_expr) in args.iter().enumerate() {
                    let arg_val = self.eval_expr(arg_expr, env)?;
                    local_env.insert(func.params[i].0.clone(), arg_val);
                }
                for stmt in &func.body {
                    if let Some(ret_val) = self.eval_statement(stmt, &mut local_env) {
                        return Some(ret_val);
                    }
                }
                None
            }
            _ => None,
        }
    }

    fn eval_statement(&self, stmt: &Statement, env: &mut HashMap<String, Value>) -> Option<Value> {
        match stmt {
            Statement::LetDecl(name, _, _, expr, _) => {
                if let Some(val) = self.eval_expr(expr, env) {
                    env.insert(name.clone(), val);
                }
                None
            }
            Statement::Assign(Expr::Identifier(name, _), rhs, _) => {
                if let Some(val) = self.eval_expr(rhs, env) {
                    env.insert(name.clone(), val);
                }
                None
            }
            Statement::Return(expr, _) => self.eval_expr(expr, env),
            _ => None,
        }
    }

    fn check_expr(&mut self, expr: &mut Expr) -> (Type, u32) {
        // First perform semantic validation
        let ty = self.check_expr_type(expr);

        // Then perform AST to HIR lowering
        let type_idx = self.emit_type(&ty);

        // Determine opcode based on the AST expression
        let opcode = match expr {
            Expr::Number(..) => crate::hir::OP_CONST,
            Expr::Identifier(_, _) => crate::hir::OP_LOAD,
            Expr::BinaryOp(_, op, _, _) => match op {
                BinaryOp::Add => crate::hir::OP_ADD,
                BinaryOp::Sub => crate::hir::OP_SUB,
                BinaryOp::Mul => crate::hir::OP_MUL,
                BinaryOp::Div => crate::hir::OP_DIV,
                _ => crate::hir::OP_NOP,
            },
            Expr::FunctionCall(_, _, _) => crate::hir::OP_CALL,
            _ => crate::hir::OP_NOP,
        };

        // In a full implementation, we would recursively call check_expr here
        // to get operand registers. For this bridge proof-of-concept, we emit
        // dummy operands and assign the result register.
        let reg = self.emit_inst(opcode, 0, 0, type_idx);
        (ty, reg)
    }

    fn check_expr_type(&mut self, expr: &mut Expr) -> Type {
        match expr {
            Expr::Identifier(name, _) => {
                if name == "true" || name == "false" {
                    return Type::Scalar(ElementType::Bool);
                }
                let lookup_res = self.lookup(name).cloned();
                match lookup_res {
                    Some((ty, top)) => {
                        // Enforce Topology Boundaries!
                        let mut is_valid = top == self.active_topology
                            || (top == Topology::Host
                                && matches!(
                                    self.active_topology,
                                    Topology::AMX | Topology::ANE | Topology::GPU
                                ));
                        if let Type::Pinned(_, pinned_top) = &ty {
                            if *pinned_top == self.active_topology {
                                is_valid = true;
                            }
                        }
                        if let Type::Ref(_, MemorySpace::NPUHBM) = &ty {
                            if matches!(
                                self.active_topology,
                                Topology::NPU(_) | Topology::Slice(_, _, _) | Topology::ANE
                            ) {
                                is_valid = true;
                            }
                        }
                        if let Type::Ref(_, MemorySpace::HostDRAM) = &ty {
                            if matches!(
                                self.active_topology,
                                Topology::Host | Topology::AMX | Topology::GPU | Topology::ANE
                            ) {
                                is_valid = true;
                            }
                        }

                        if !is_valid {
                            self.errors.push(format!(
                                "Cross-topology access error: Variable '{}' belongs to {:?} (type: {:?}), but accessed from {:?}",
                                name, top, ty, self.active_topology
                            ));
                        }
                        ty.clone()
                    }
                    None => {
                        self.errors.push(format!("Undefined variable '{}'", name));
                        Type::Tensor(ElementType::F32, vec![], None) // Default placeholder on error
                    }
                }
            }
            Expr::EnumVariant(enum_name, variant, _) => {
                if let Some(variants) = self.env.enums.get(enum_name) {
                    if !variants.contains(variant) {
                        self.errors.push(format!(
                            "Enum {} does not have variant {}",
                            enum_name, variant
                        ));
                    }
                } else {
                    self.errors.push(format!("Unknown enum {}", enum_name));
                }
                Type::Enum(enum_name.clone(), None)
            }
            Expr::Number(_, Some(el_ty), _) => Type::Scalar(el_ty.clone()),
            Expr::Number(_, None, _) => Type::Scalar(ElementType::F32),
            Expr::StringLiteral(..) => Type::Pointer(
                Box::new(Type::Scalar(ElementType::I8)),
                None,
                false, // const
            ),
            Expr::Transfer(inner_expr, target_mem, _) => {
                let inner_ty = self.check_expr_type(inner_expr);
                match inner_ty {
                    Type::Ref(base_ty, _) => Type::Ref(base_ty, target_mem.clone()),
                    Type::Tensor(_, _, _) => Type::Pinned(
                        Box::new(inner_ty.clone()),
                        Topology::NPU(Box::new(Expr::Number("0".to_string(), Some(crate::ast::ElementType::I32), Span::default()))),
                    ),
                    Type::Pinned(base, top) => Type::Pinned(base, top),
                    _ => {
                        self.errors.push(format!(
                            "Cannot transfer non-reference type: {:?}",
                            inner_ty
                        ));
                        Type::Tensor(ElementType::F32, vec![], None)
                    }
                }
            }
            Expr::Import(path, _) => {
                // Imports are resolved in Phase 1
                Type::Module(path.clone(), std::collections::HashMap::new())
            }
            Expr::ComptimeBlock(stmts, ret, _) => {
                self.push_scope();
                for stmt in stmts {
                    if let Statement::ExprStmt(ref mut expr, _) = stmt {
                        self.check_expr_type(expr);
                    } else {
                        self.check_statement(stmt, &Type::Tensor(ElementType::F32, vec![], None));
                    }
                }
                let mut ret_ty = Type::Tensor(ElementType::F32, vec![], None);
                if let Some(r) = ret {
                    ret_ty = self.check_expr_type(r);
                }
                self.pop_scope();
                ret_ty
            }
            Expr::FunctionCall(name, args, _) => {
                // If this is a local call to a function that was mangled, update the name.

                let resolved_name = name.clone();

                // Mocking built-ins
                for arg in args.iter_mut() {
                    self.check_expr_type(arg);
                }

                if resolved_name == "Verified" {
                    if args.len() != 1 {
                        self.errors.push(format!(
                            "Function 'Verified' expects 1 argument, got {}",
                            args.len()
                        ));
                    }
                    let inner_ty = self.check_expr_type(&mut args[0]);
                    Type::Verified(Box::new(inner_ty))
                } else if resolved_name.starts_with("Tensor") && resolved_name.ends_with("::from") {
                    if args.len() != 2 {
                        self.errors.push(format!(
                            "Function '{}' expects 2 arguments (pointer, shape), got {}",
                            resolved_name,
                            args.len()
                        ));
                    }
                    if !self.in_unsafe_block {
                        self.errors.push(format!("Call to '{}' is unsafe because it interprets raw memory. Requires unsafe block.", resolved_name));
                    }
                    self.check_expr_type(&mut args[0]);
                    self.check_expr_type(&mut args[1]);

                    let mut el_ty = ElementType::F32;
                    if resolved_name.contains("_i32") {
                        el_ty = ElementType::I32;
                    } else if resolved_name.contains("_i64") {
                        el_ty = ElementType::I64;
                    } else if resolved_name.contains("_f64") {
                        el_ty = ElementType::F64;
                    }

                    Type::Tensor(el_ty, vec![], None)
                } else if resolved_name.starts_with("Tensor") {
                    let el_ty = match resolved_name.as_str() {
                        "Tensor_f64" => ElementType::F64,
                        "Tensor_bf16" => ElementType::BF16,
                        "Tensor_i32" => ElementType::I32,
                        "Tensor_i64" => ElementType::I64,
                        _ => ElementType::F32,
                    };
                    Type::Tensor(el_ty, vec![], None)
                } else if resolved_name.starts_with("Math::") {
                    if args.len() != 1 {
                        self.errors.push(format!(
                            "Function '{}' expects 1 argument, got {}",
                            resolved_name,
                            args.len()
                        ));
                    }
                    let inner_ty = self.check_expr_type(&mut args[0]);
                    if inner_ty != Type::Scalar(ElementType::F32) {
                        self.errors.push(format!(
                            "Function '{}' expects f32 argument, got {:?}",
                            resolved_name, inner_ty
                        ));
                    }
                    Type::Scalar(ElementType::F32)
                } else if resolved_name == "print" {
                    if args.len() != 1 {
                        self.errors
                            .push("Function 'print' expects 1 argument".to_string());
                    }
                    Type::Tensor(ElementType::F32, vec![], None)
                } else if let Some((ret_ty, is_unsafe)) = self.env.functions.get(&resolved_name) {
                    if *is_unsafe && !self.in_unsafe_block {
                        self.errors.push(format!("Call to unsafe function '{}' is unsafe and requires unsafe function or block", resolved_name));
                    }
                    ret_ty.clone()
                } else if let Some(func) = self
                    .monomorphized_functions
                    .iter()
                    .find(|f| f.0.name == resolved_name)
                {
                    func.0.return_type.clone()
                } else if let Some((generic_func, origin_hash)) =
                    self.env.generic_functions.get(&resolved_name).cloned()
                {
                    // Type deduction
                    let mut mapping = HashMap::new();
                    let mut success = true;
                    if args.len() != generic_func.params.len() {
                        self.errors.push(format!(
                            "Generic function '{}' expects {} arguments, got {}",
                            resolved_name,
                            generic_func.params.len(),
                            args.len()
                        ));
                        success = false;
                    } else {
                        for (i, arg) in args.iter_mut().enumerate() {
                            let arg_ty = self.check_expr_type(arg);
                            let param_ty = &generic_func.params[i].1;
                            if !self.unify_types(param_ty, &arg_ty, &mut mapping) {
                                self.errors.push(format!("Failed to deduce types for generic function '{}': Expected {:?}, got {:?}", name, param_ty, arg_ty));
                                success = false;
                            }
                        }
                    }

                    if success {
                        // Trait Bounds Checking
                        for (g_name, bound_opt) in &generic_func.generics {
                            if let Some(bound_name) = bound_opt {
                                if let Some(concrete_ty) = mapping.get(g_name) {
                                    let mut implements_trait = false;
                                    if let Some(impl_blocks) = self.env.impls.get(bound_name) {
                                        for ib in impl_blocks {
                                            if self.unify_types(
                                                &ib.target_type,
                                                concrete_ty,
                                                &mut HashMap::new(),
                                            ) {
                                                implements_trait = true;
                                                break;
                                            }
                                        }
                                    }
                                    if !implements_trait {
                                        self.errors.push(format!(
                                            "Type '{:?}' does not implement trait '{}' required by parameter '{}'",
                                            concrete_ty, bound_name, g_name
                                        ));
                                        success = false;
                                    }
                                }
                            }
                        }
                    }

                    if success {
                        // Instantiate
                        let mut inst_func = self.instantiate_function(generic_func, &mapping);
                        let inst_ret = inst_func.return_type.clone();
                        let inst_name = inst_func.name.clone();

                        // Rewrite AST name
                        *name = inst_name.clone();

                        if !self.env.functions.contains_key(&inst_name) {
                            // self.env is immutable, monomorphization tracks functions internally
                            self.check_function(&mut inst_func);
                            self.monomorphized_functions.push((inst_func, origin_hash));
                        }
                        inst_ret
                    } else {
                        Type::Tensor(ElementType::F32, vec![], None)
                    }
                } else {
                    let mono_names: Vec<String> = self
                        .monomorphized_functions
                        .iter()
                        .map(|(f, _)| f.name.clone())
                        .collect();
                    self.errors.push(format!(
                        "Undefined function '{}'. Available monos: {:?}",
                        resolved_name, mono_names
                    ));
                    Type::Tensor(ElementType::F32, vec![], None)
                }
            }
            Expr::Array(elements, _) => {
                for el in elements {
                    self.check_expr_type(el);
                }
                Type::Tensor(ElementType::F32, vec![], None)
            }
            Expr::MemberAccess(obj, member, _) => {
                let obj_ty = self.check_expr_type(obj);
                let mut base_ty = obj_ty.clone();
                if let Type::Borrow(t, _, _) | Type::Pointer(t, _, _) = base_ty {
                    base_ty = *t;
                }

                if let Type::Struct(struct_name, _) = &base_ty {
                    if let Some(decl) = self.env.structs.get(struct_name).cloned() {
                        for (f_name, f_type) in &decl.fields {
                            if f_name == member {
                                return f_type.clone();
                            }
                        }
                        self.errors.push(format!(
                            "Struct '{}' has no field '{}'",
                            struct_name, member
                        ));
                    } else {
                        self.errors
                            .push(format!("Unknown struct '{}'", struct_name));
                    }
                } else if let Type::Module(ref path, ref exports) = base_ty {
                    if let Some(exported_ty) = exports.get(member) {
                        return exported_ty.clone();
                    } else {
                        self.errors
                            .push(format!("Module '{}' does not export '{}'", path, member));
                    }
                } else if member != "shape" {
                    // default behavior for Tensor.shape
                    self.errors
                        .push("Member access on non-struct type".to_string());
                }
                Type::Tensor(ElementType::F32, vec![], None)
            }
            Expr::IndexAccess(obj, idx, _) => {
                let obj_ty = self.check_expr_type(obj);
                self.check_expr_type(idx);
                if let Type::Pointer(inner, _, _) = obj_ty {
                    *inner
                } else if let Type::Borrow(inner, _, _) = obj_ty {
                    *inner
                } else if let Type::Tensor(el_ty, _, _) = obj_ty {
                    Type::Scalar(el_ty)
                } else {
                    Type::Scalar(ElementType::F32)
                }
            }
            Expr::MethodCall(obj, _method, args, _) => {
                let mut base_ty = self.check_expr_type(obj);
                for arg in args.iter_mut() {
                    self.check_expr_type(arg);
                }

                if let Type::Module(ref path, ref exports) = base_ty {
                    if let Some(exported_ty) = exports.get(_method) {
                        let prefix = TypeChecker::mangle_path(path);
                        let mangled_name = format!("{}_{}", prefix, _method);
                        let func_call =
                            Expr::FunctionCall(mangled_name, args.clone(), Span::default());
                        *expr = func_call;
                        return exported_ty.clone();
                    } else {
                        self.errors.push(format!(
                            "Module '{}' does not export function '{}'",
                            path, _method
                        ));
                        return Type::Tensor(ElementType::F32, vec![], None);
                    }
                }

                // --- COMPILER INTRINSICS ---
                if let Type::Tensor(el_ty, dims, top) = &base_ty {
                    if _method == "reshape" {
                        if args.is_empty() || args.len() > 3 {
                            self.errors
                                .push("reshape requires 1 to 3 arguments".to_string());
                            return base_ty;
                        }

                        let mut is_exact = true;
                        if args.len() >= 2 {
                            if let Expr::EnumVariant(enum_name, variant, _) = &args[1] {
                                if enum_name == "PadMode" && (variant == "Pad" || variant == "Trim")
                                {
                                    is_exact = false;
                                } else {
                                    self.errors.push(
                                        "reshape mode must be PadMode::Pad or PadMode::Trim"
                                            .to_string(),
                                    );
                                }
                            } else {
                                self.errors.push(
                                    "reshape mode must be an enum variant (e.g. PadMode::Pad)"
                                        .to_string(),
                                );
                            }
                        }

                        if let Expr::Array(new_dims, _) = &args[0] {
                            let empty_env = HashMap::new();
                            let mut src_elements = 1.0;
                            for d in dims {
                                if let Some(Value::Number(v)) = self.eval_expr(d, &empty_env) {
                                    src_elements *= v;
                                } else {
                                    self.errors.push(
                                        "Cannot statically evaluate source dimension for reshape"
                                            .to_string(),
                                    );
                                    return base_ty;
                                }
                            }

                            let mut target_elements = 1.0;
                            for d in new_dims {
                                if let Some(Value::Number(v)) = self.eval_expr(d, &empty_env) {
                                    target_elements *= v;
                                } else {
                                    self.errors.push(
                                        "Cannot statically evaluate target dimension for reshape"
                                            .to_string(),
                                    );
                                    return base_ty;
                                }
                            }

                            if is_exact && (src_elements - target_elements).abs() > 1e-6 {
                                self.errors.push(format!("reshape arithmetic mismatch: source has {} elements, target has {}", src_elements, target_elements));
                                return base_ty;
                            }

                            return Type::Tensor(*el_ty, new_dims.clone(), top.clone());
                        } else {
                            self.errors.push(
                                "reshape requires an array of dimensions as the first argument"
                                    .to_string(),
                            );
                            return base_ty;
                        }
                    } else if _method == "transpose" {
                        if args.len() != 1 {
                            self.errors.push("transpose requires exactly 1 argument (an array of permutation indices)".to_string());
                            return base_ty;
                        }
                        if let Expr::Array(perm, _) = &args[0] {
                            let empty_env = HashMap::new();
                            let mut new_dims = vec![Expr::Number("0".to_string(), Some(crate::ast::ElementType::I32), Span::default()); dims.len()];
                            if perm.len() != dims.len() {
                                self.errors.push(
                                    "transpose permutation map length must match tensor rank"
                                        .to_string(),
                                );
                                return base_ty;
                            }
                            let mut seen = vec![false; dims.len()];
                            for (i, p) in perm.iter().enumerate() {
                                if let Some(Value::Number(v)) = self.eval_expr(p, &empty_env) {
                                    let v = v as usize;
                                    if v >= dims.len() {
                                        self.errors
                                            .push("transpose index out of bounds".to_string());
                                        return base_ty;
                                    }
                                    if seen[v] {
                                        self.errors.push(
                                            "transpose permutation map must not contain duplicates"
                                                .to_string(),
                                        );
                                        return base_ty;
                                    }
                                    seen[v] = true;
                                    new_dims[i] = dims[v].clone();
                                } else {
                                    self.errors.push(
                                        "Cannot statically evaluate transpose permutation index"
                                            .to_string(),
                                    );
                                    return base_ty;
                                }
                            }
                            return Type::Tensor(*el_ty, new_dims, top.clone());
                        } else {
                            self.errors.push(
                                "transpose requires an array of permutation indices".to_string(),
                            );
                            return base_ty;
                        }
                    }
                }
                // --- END INTRINSICS ---

                // Dynamic Method Resolution
                let mut found_method = None;
                for impl_blocks in self.env.impls.values() {
                    for ib in impl_blocks {
                        if self.unify_types(&ib.target_type, &base_ty, &mut HashMap::new()) {
                            for m in &ib.methods {
                                if m.name == *_method {
                                    found_method = Some((m.clone(), (*ib).clone()));
                                    break;
                                }
                            }
                        }
                    }
                }

                if let Some((mut method_func, ib)) = found_method {
                    // Create a unique mangled name for the method based on the target type
                    let mut mangled_name = format!("{:?}_{}", ib.target_type, _method)
                        .replace("(", "_")
                        .replace(")", "")
                        .replace(" ", "")
                        .replace("[", "")
                        .replace("]", "")
                        .replace(",", "_")
                        .replace("_None", "");
                    // Clean up multiple underscores
                    while mangled_name.contains("__") {
                        mangled_name = mangled_name.replace("__", "_");
                    }
                    mangled_name = mangled_name
                        .replace("\"", "")
                        .replace("Tensor", "Tensor_")
                        .replace("Generic", "Gen_");

                    method_func.name = mangled_name.clone();

                    // Register the method if it doesn't exist
                    if !method_func.generics.is_empty() {
                        /* self.env.generic_functions.insert is mock */
                    } else if !self.env.functions.contains_key(&mangled_name) {
                        // Since it's not generic, we must type check it once!
                        let mut func_to_check = method_func.clone();
                        self.check_function(&mut func_to_check);
                        self.monomorphized_functions.push((func_to_check, 0)); // 0 will fall back to caller_module_idx
                    }

                    // Rewrite AST from MethodCall to FunctionCall
                    let mut call_args = vec![(**obj).clone()];
                    for a in args.iter() {
                        call_args.push(a.clone());
                    }

                    let mut func_call =
                        Expr::FunctionCall(mangled_name, call_args, Span::default());
                    let ret_ty = self.check_expr_type(&mut func_call);

                    // Replace the AST node in-place!
                    *expr = func_call;
                    return ret_ty;
                }

                // Fallback for hardcoded mock methods
                if _method == "with_memory" {
                    base_ty = Type::Ref(Box::new(base_ty), MemorySpace::NPUHBM);
                } else if _method == "to_device" {
                    let target_mem = MemorySpace::NPUHBM; // Can be enhanced later to parse arg
                    base_ty = Type::Pinned(
                        Box::new(base_ty),
                        Topology::NPU(Box::new(Expr::Number("0".to_string(), Some(crate::ast::ElementType::I32), Span::default()))),
                    ); // Default to NPU[0]
                    *expr = Expr::Transfer(obj.clone(), target_mem, Span::default());
                } else if _method == "to_host" {
                    let target_mem = MemorySpace::HostDRAM;
                    base_ty = Type::Pinned(Box::new(base_ty), Topology::Host);
                    *expr = Expr::Transfer(obj.clone(), target_mem, Span::default());
                } else if _method == "as_ptr" || _method == "as_mut_ptr" {
                    let is_mut = _method == "as_mut_ptr";
                    match &base_ty {
                        Type::Tensor(el_ty, dims, top) => {
                            base_ty = Type::Pointer(
                                Box::new(Type::Tensor(*el_ty, dims.clone(), top.clone())),
                                None,
                                is_mut,
                            );
                        }
                        Type::Borrow(inner, mem, mutability) => {
                            if is_mut && !mutability {
                                self.errors.push(
                                    "Cannot get mutable pointer from immutable borrow".to_string(),
                                );
                            }
                            base_ty = Type::Pointer(inner.clone(), mem.clone(), is_mut);
                        }
                        Type::Pointer(_, _, _) => {
                            self.errors.push("Already a pointer".to_string());
                        }
                        _ => {
                            self.errors
                                .push(format!("Cannot call {} on {:?}", _method, base_ty));
                        }
                    }
                } else if _method == "len" {
                    match &base_ty {
                        Type::Tensor(_, _, _) | Type::Borrow(_, _, _) | Type::Pointer(_, _, _) => {
                            base_ty = Type::Tensor(ElementType::I64, vec![], None);
                        }
                        _ => {
                            self.errors
                                .push(format!("Cannot call len on {:?}", base_ty));
                        }
                    }
                } else {
                    self.errors.push(format!(
                        "Method '{}' not found on type {:?}",
                        _method, base_ty
                    ));
                }
                base_ty
            }
            Expr::BinaryOp(lhs, op, rhs, _) => {
                let lhs_ty = self.check_expr_type(lhs);
                let rhs_ty = self.check_expr_type(rhs);
                if !self.is_assignable(&lhs_ty, &rhs_ty) {
                    self.errors
                        .push("Type mismatch in binary operation".to_string());
                }
                match op {
                    BinaryOp::Eq
                    | BinaryOp::NotEq
                    | BinaryOp::Lt
                    | BinaryOp::Gt
                    | BinaryOp::Le
                    | BinaryOp::Ge
                    | BinaryOp::And
                    | BinaryOp::Or => Type::Scalar(ElementType::Bool),
                    _ => lhs_ty,
                }
            }
            Expr::MemorySpace(..) | Expr::Topology(..) => {
                Type::Tensor(ElementType::F32, vec![], None)
            }
            Expr::UnaryOp(op, inner, _) => {
                self.check_expr_type(inner);
                match op {
                    UnaryOp::Not => Type::Scalar(ElementType::Bool),
                }
            }
            Expr::Borrow(inner, is_mut, _) => {
                let inner_ty = self.check_expr_type(inner);
                Type::Borrow(Box::new(inner_ty), None, *is_mut)
            }
            Expr::Dereference(inner, _) => {
                if !self.in_unsafe_block {
                    self.errors
                        .push("Dereference of raw pointer outside of unsafe block!".to_string());
                }
                let inner_ty = self.check_expr_type(inner);
                match inner_ty {
                    Type::Pointer(t, _, _) | Type::Borrow(t, _, _) => *t,
                    _ => {
                        self.errors
                            .push("Cannot dereference non-pointer type".to_string());
                        inner_ty
                    }
                }
            }
            Expr::UnsafeBlock(stmts, _ret_expr, _) => {
                let prev_unsafe = self.in_unsafe_block;
                self.in_unsafe_block = true;
                self.push_scope();
                let mut last_type = Type::Tensor(ElementType::F32, vec![], None);
                for s in stmts.iter_mut() {
                    if let Statement::ExprStmt(ref mut expr, _) = s {
                        last_type = self.check_expr_type(expr);
                    } else {
                        self.check_statement(
                            &mut *s,
                            &Type::Tensor(ElementType::F32, vec![], None),
                        );
                    }
                }
                self.pop_scope();
                self.in_unsafe_block = prev_unsafe;
                last_type
            }
            Expr::StructInit(name, fields, _) => {
                let resolved_name = name.clone();
                if false {
                    /* resolved_name = mangled.clone(); */
                    *name = resolved_name.clone();
                }

                if !self.env.structs.contains_key(&resolved_name) {
                    self.errors
                        .push(format!("Unknown struct {}", resolved_name));
                }
                for (_, f_expr) in fields {
                    self.check_expr_type(f_expr);
                }
                Type::Struct(resolved_name, None)
            }
        }
    }

    fn is_assignable(&self, target: &Type, source: &Type) -> bool {
        if target == source {
            return true;
        }

        // Allow assigning Ref<T> to T (implicit unwrap of ref wrapper if target wants base type)
        if let Type::Ref(inner_source, _) = &source {
            if self.is_assignable(target, inner_source) {
                return true;
            }
        }

        // Allow assigning Pinned<T> to T
        if let Type::Pinned(inner_source, _) = source {
            if self.is_assignable(target, inner_source) {
                return true;
            }
        }

        // Allow numeric coercions for scalar literals (mock behavior for now)
        if let Type::Tensor(t_target, dims_target, top_target) = target {
            if let Type::Tensor(t_source, dims_source, top_source) = &source {
                let mut el_match = false;
                if *t_target == *t_source {
                    el_match = true;
                } else if *t_source == ElementType::F32 && t_target != &ElementType::Bool {
                    // Literals currently parse as f32, so we allow f32 to coerce
                    el_match = true;
                }

                if !el_match {
                    return false;
                }

                if top_target.is_some() && top_target != top_source {
                    return false;
                }

                if !dims_target.is_empty() && !dims_source.is_empty() {
                    if dims_target.len() != dims_source.len() {
                        return false;
                    }
                    let empty_env = std::collections::HashMap::new();
                    for (dt, ds) in dims_target.iter().zip(dims_source.iter()) {
                        let vt = self.eval_expr(dt, &empty_env);
                        let vs = self.eval_expr(ds, &empty_env);
                        if vt.is_some() && vs.is_some() {
                            if vt != vs {
                                return false;
                            }
                        } else if dt != ds {
                            return false;
                        }
                    }
                }
                return true;
            }
        }

        if let Type::Scalar(t_target) = target {
            if let Type::Scalar(t_source) = &source {
                if *t_target == *t_source {
                    return true;
                }
                // Allow numeric coercions
                if *t_target != ElementType::Bool && t_source != &ElementType::Bool {
                    return true;
                }
            }
        }

        // Allow coercing Scalar to Tensor (e.g. 0.0 to Tensor<f32>) for backwards compatibility with tests
        if let Type::Tensor(t_target, _, _) = target {
            if let Type::Scalar(t_source) = &source {
                if *t_target == *t_source {
                    return true;
                }
                if *t_source == ElementType::F32 && t_target != &ElementType::Bool {
                    return true;
                }
            }
        }

        // Semantic coercion rule: Ref<T, HostDRAM> can be assigned to Verified<T>
        // Also allow returning Verified(Ref(T, Memory)) as Verified(T)
        if let Type::Verified(inner_target) = target {
            if let Type::Verified(inner_source) = source {
                if let Type::Ref(base_source, _) = &**inner_source {
                    return *inner_target.as_ref() == **base_source;
                }
            }
            if let Type::Ref(inner_source, MemorySpace::HostDRAM) = source {
                return inner_target.as_ref() == inner_source.as_ref();
            }
        }

        // Allow coercing Borrow to Pointer (e.g. &mut T to *mut T)
        if let Type::Pointer(target_inner, target_mem, target_mut) = target {
            if let Type::Borrow(source_inner, source_mem, source_mut) = source {
                if target_inner.as_ref() == source_inner.as_ref()
                    && target_mem == source_mem
                    && (!*target_mut || *source_mut)
                {
                    return true;
                }
            }
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;

    #[test]
    fn test_sema_distributed_matmul() {
        let input = r#"
fn custom_matmul(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32> {
    return a;
}

fn distributed_matmul(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32> {
    let local_a = a.to_device();
    let local_b = b.to_device();
    spawn on(Topology::NPU[0]) {
        let result = custom_matmul(local_a, local_b);
        return result;
    }
}
        "#;
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(tokens, input);
        let mut program = parser.parse().unwrap();

        let program_arr = [program.clone()];
        let env = GlobalAstEnv::build(&program_arr);
        let mut worker = crate::session::LocalWorkerState::new(std::sync::Arc::new(
            crate::session::GlobalSession::new(1),
        ));
        let mut checker = TypeChecker::new(&env, &mut worker);
        let success = {
            for f in &mut program.functions {
                checker.check_function(f);
            }
            checker.errors.is_empty()
        };

        for err in &checker.errors {
            println!("Error: {}", err);
        }
        assert!(success);
        assert!(checker.errors.is_empty());
    }

    #[test]
    fn test_sema_type_mismatch() {
        let input = r#"
fn bad_matmul() -> Tensor {
    return undefined_variable;
}
        "#;
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(tokens, input);
        let mut program = parser.parse().unwrap();

        let program_arr = [program.clone()];
        let env = GlobalAstEnv::build(&program_arr);
        let mut worker = crate::session::LocalWorkerState::new(std::sync::Arc::new(
            crate::session::GlobalSession::new(1),
        ));
        let mut checker = TypeChecker::new(&env, &mut worker);
        let success = {
            for f in &mut program.functions {
                checker.check_function(f);
            }
            checker.errors.is_empty()
        };
        assert!(!success);
        assert!(!checker.errors.is_empty());
    }

    #[test]
    fn test_sema_struct_and_pointers() {
        let input = r#"
        struct Config {
            value: Tensor<f32>
        }

        fn test_pointers(c: &mut Config) -> Tensor<Bool> {
            unsafe {
                let ptr: *mut Config = c;
                let val = *ptr;
            }
            return c.value < 20.0f32;
        }
        "#;
        let mut lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer.tokenize(), input);
        let mut program = parser.parse().unwrap();
        let program_arr = [program.clone()];
        let env = GlobalAstEnv::build(&program_arr);
        let mut worker = crate::session::LocalWorkerState::new(std::sync::Arc::new(
            crate::session::GlobalSession::new(1),
        ));
        let mut checker = TypeChecker::new(&env, &mut worker);
        assert!(
            {
                for f in &mut program.functions {
                    checker.check_function(f);
                }
                checker.errors.is_empty()
            },
            "Semantic checking failed: {:?}",
            checker.errors
        );
    }

    #[test]
    fn test_sema_extern_unsafe() {
        let input = r#"
        extern "C" {
            fn malloc(size: Tensor<f32>) -> *mut Tensor<f32>;
        }

        fn safe_wrapper() -> *mut Tensor<f32> {
            return malloc(1024); // ERROR: unsafe function call
        }

        fn safe_wrapper_fixed() -> *mut Tensor<f32> {
            unsafe {
                return malloc(1024);
            }
        }
        "#;
        let mut lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer.tokenize(), input);
        let mut program = parser.parse().unwrap();
        let program_arr = [program.clone()];
        let env = GlobalAstEnv::build(&program_arr);
        let mut worker = crate::session::LocalWorkerState::new(std::sync::Arc::new(
            crate::session::GlobalSession::new(1),
        ));
        let mut checker = TypeChecker::new(&env, &mut worker);

        let success = {
            for f in &mut program.functions {
                checker.check_function(f);
            }
            checker.errors.is_empty()
        };
        assert!(!success);
        assert!(checker
            .errors
            .iter()
            .any(|e| e.contains("Call to unsafe function 'malloc' is unsafe")));
    }

    #[test]
    fn test_sema_as_ptr_and_len() {
        let input = r#"
        fn test_methods(t: Tensor<f32>) -> Tensor<i64> {
            let ptr: *const Tensor<f32> = t.as_ptr();
            let mut_ptr: *mut Tensor<f32> = t.as_mut_ptr();
            let length: Tensor<i64> = t.len();
            return length;
        }
        "#;
        let mut lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer.tokenize(), input);
        let mut program = parser.parse().unwrap();
        let program_arr = [program.clone()];
        let env = GlobalAstEnv::build(&program_arr);
        let mut worker = crate::session::LocalWorkerState::new(std::sync::Arc::new(
            crate::session::GlobalSession::new(1),
        ));
        let mut checker = TypeChecker::new(&env, &mut worker);

        assert!(
            {
                for f in &mut program.functions {
                    checker.check_function(f);
                }
                checker.errors.is_empty()
            },
            "Semantic checking failed for methods: {:?}",
            checker.errors
        );
    }
}
