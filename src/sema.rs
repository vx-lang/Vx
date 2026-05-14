use crate::ast::*;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Bool(bool),
    Number(f64),
}

pub struct TypeChecker {
    scopes: Vec<HashMap<String, (Type, Topology)>>,
    functions: HashMap<String, (Type, bool)>, // Maps function name to (return type, is_unsafe)
    ast_functions: HashMap<String, Function>, // For constexpr evaluation
    generic_functions: HashMap<String, Function>,
    pub monomorphized_functions: Vec<Function>,
    structs: HashMap<String, StructDecl>,
    traits: HashMap<String, TraitDecl>,
    impls: HashMap<String, Vec<ImplBlock>>, // Maps trait name to list of implementations
    pub errors: Vec<String>,
    in_unsafe_block: bool,
    active_topology: Topology,
    active_memory: MemorySpace,
}

impl Default for TypeChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl TypeChecker {
    pub fn new() -> Self {
        Self {
            scopes: vec![HashMap::new()],
            functions: HashMap::new(),
            ast_functions: HashMap::new(),
            generic_functions: HashMap::new(),
            monomorphized_functions: Vec::new(),
            structs: HashMap::new(),
            traits: HashMap::new(),
            impls: HashMap::new(),
            errors: Vec::new(),
            in_unsafe_block: false,
            active_topology: Topology::Host,
            active_memory: MemorySpace::HostDRAM,
        }
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn insert(&mut self, name: String, ty: Type) {
        let top = self.active_topology.clone();
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name, (ty, top));
        }
    }

    fn lookup(&self, name: &str) -> Option<(Type, Topology)> {
        for scope in self.scopes.iter().rev() {
            if let Some((ty, top)) = scope.get(name) {
                return Some((ty.clone(), top.clone()));
            }
        }
        None
    }

    pub fn check_program(&mut self, program: &mut Program) -> Result<Program, Vec<String>> {
        // Collect structs
        for s in &program.structs {
            self.structs.insert(s.name.clone(), s.clone());
        }

        // Collect traits
        for t in &program.traits {
            self.traits.insert(t.name.clone(), t.clone());
        }

        // Collect impls
        for i in &program.impls {
            let trait_name = match &i.trait_name {
                Some(name) => name.clone(),
                None => "_inherent".to_string(), // For `impl Type { ... }` blocks
            };
            self.impls.entry(trait_name).or_default().push(i.clone());
        }

        // Collect externs (unsafe by default)
        for ext in &program.externs {
            self.functions
                .insert(ext.name.clone(), (ext.return_type.clone(), true));
        }

        let mut functions_to_check = Vec::new();

        // First pass: collect function signatures
        for func in &program.functions {
            if !func.generics.is_empty() {
                self.generic_functions
                    .insert(func.name.clone(), func.clone());
            } else {
                self.functions
                    .insert(func.name.clone(), (func.return_type.clone(), false));
                self.ast_functions.insert(func.name.clone(), func.clone());
                functions_to_check.push(func.clone());
            }
        }

        // Second pass: check non-generic function bodies
        // (This will recursively trigger monomorphization if they call generic functions)
        for mut func in functions_to_check {
            self.check_function(&mut func);
            self.monomorphized_functions.push(func);
        }

        if self.errors.is_empty() {
            let mut new_program = program.clone();
            new_program.functions = self.monomorphized_functions.clone();
            Ok(new_program)
        } else {
            Err(self.errors.clone())
        }
    }

    fn unify_types(
        &self,
        generic_ty: &Type,
        concrete_ty: &Type,
        mapping: &mut HashMap<String, Type>,
    ) -> bool {
        match (generic_ty, concrete_ty) {
            (Type::Generic(name), _) => {
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
            (Type::Struct(n1), Type::Struct(n2)) => n1 == n2,
            // Fallback for simple equality (e.g. Matrix)
            (t1, t2) => t1 == t2,
        }
    }

    fn instantiate_function(
        &mut self,
        generic_func: &Function,
        type_args: &HashMap<String, Type>,
    ) -> Function {
        // Create mangled name
        let mut mangled_name = generic_func.name.clone();
        for (g_name, _) in &generic_func.generics {
            if let Some(ty) = type_args.get(g_name) {
                // simple mangling
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
            .map(|(n, t)| (n.clone(), t.substitute(type_args)))
            .collect();
        let new_ret = generic_func.return_type.substitute(type_args);
        let new_body = generic_func
            .body
            .iter()
            .map(|s| s.substitute(type_args))
            .collect();

        Function {
            name: mangled_name,
            generics: Vec::new(),
            params: new_params,
            return_type: new_ret,
            body: new_body,
        }
    }

    fn check_function(&mut self, func: &mut Function) {
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
        match stmt {
            Statement::LetDecl(name, _is_mut, ty_ann, expr) => {
                let ty = self.check_expr(expr);

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
            Statement::ForLoop(iter, start, end, body) => {
                self.check_expr(start);
                self.check_expr(end);
                self.push_scope();
                self.insert(iter.clone(), Type::Scalar(ElementType::I64));
                for s in body {
                    self.check_statement(s, return_type);
                }
                self.pop_scope();
            }
            Statement::If(cond, then_block, else_block) => {
                let cond_ty = self.check_expr(cond);
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
            Statement::Assign(lhs, rhs) | Statement::CompoundAssign(lhs, _, rhs) => {
                let lhs_ty = self.check_expr(lhs);
                let rhs_ty = self.check_expr(rhs);
                if !self.is_assignable(&lhs_ty, &rhs_ty) {
                    self.errors.push("Type mismatch in assignment".to_string());
                }
            }
            Statement::Return(expr) => {
                let ty = self.check_expr(expr);
                if !self.is_assignable(return_type, &ty) {
                    self.errors.push(format!(
                        "Type mismatch on return. Expected {:?}, got {:?}",
                        return_type, ty
                    ));
                }
            }
            Statement::SpawnOn(top, stmts) => {
                let prev_top = self.active_topology.clone();
                let prev_mem = self.active_memory.clone();
                self.active_topology = top.clone();
                self.active_memory = match top {
                    Topology::NPU(_) => MemorySpace::NPUHBM,
                    Topology::AccCore(_) => MemorySpace::LocalSRAM,
                    Topology::Host => MemorySpace::HostDRAM,
                    Topology::AMX => MemorySpace::HostDRAM,
                    Topology::ANE => MemorySpace::NPUHBM,
                    Topology::Slice(_, _, _) => MemorySpace::NPUHBM,
                };

                self.push_scope();

                // Validate topology expression if it contains one
                match top {
                    Topology::NPU(expr) | Topology::AccCore(expr) => {
                        let _ty = self.check_expr(expr);
                    }
                    Topology::Slice(_, start, end) => {
                        let _t1 = self.check_expr(start);
                        let _t2 = self.check_expr(end);
                    }
                    Topology::Host | Topology::AMX | Topology::ANE => {}
                }

                for s in stmts {
                    self.check_statement(s, return_type);
                }

                self.pop_scope();

                self.active_topology = prev_top;
                self.active_memory = prev_mem;
            }
            Statement::ExprStmt(expr) => {
                self.check_expr(expr);
            }
            Statement::Assert(expr, _msg) => {
                let ty = self.check_expr(expr);
                if ty != Type::Scalar(ElementType::Bool) {
                    self.errors
                        .push("Assertion condition must be boolean".to_string());
                }
            }
            Statement::Comptime(stmts) => {
                for s in stmts {
                    if let Statement::Assert(expr, msg) = s {
                        let ty = self.check_expr(expr);
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
                        self.check_statement(s, return_type);
                    }
                }
            }
        }
    }

    fn eval_expr(&self, expr: &Expr, env: &HashMap<String, Value>) -> Option<Value> {
        match expr {
            Expr::Number(n) => Some(Value::Number(*n)),
            Expr::Identifier(n) if n == "true" => Some(Value::Bool(true)),
            Expr::Identifier(n) if n == "false" => Some(Value::Bool(false)),
            Expr::Identifier(n) => env.get(n).cloned(),
            Expr::BinaryOp(lhs, op, rhs) => {
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
            Expr::UnaryOp(UnaryOp::Not, inner) => {
                if let Value::Bool(b) = self.eval_expr(inner, env)? {
                    Some(Value::Bool(!b))
                } else {
                    None
                }
            }
            Expr::FunctionCall(name, args) => {
                let func = self.ast_functions.get(name)?;
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
            Statement::LetDecl(name, _, _, expr) => {
                if let Some(val) = self.eval_expr(expr, env) {
                    env.insert(name.clone(), val);
                }
                None
            }
            Statement::Assign(Expr::Identifier(name), rhs) => {
                if let Some(val) = self.eval_expr(rhs, env) {
                    env.insert(name.clone(), val);
                }
                None
            }
            Statement::Return(expr) => self.eval_expr(expr, env),
            _ => None,
        }
    }

    fn check_expr(&mut self, expr: &mut Expr) -> Type {
        match expr {
            Expr::Identifier(name) => {
                if name == "true" || name == "false" {
                    return Type::Scalar(ElementType::Bool);
                }
                match self.lookup(name) {
                    Some((ty, top)) => {
                        // Enforce Topology Boundaries!
                        let mut is_valid = top == self.active_topology
                            || (top == Topology::Host
                                && matches!(self.active_topology, Topology::AMX | Topology::ANE));
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
                            if matches!(self.active_topology, Topology::Host | Topology::AMX) {
                                is_valid = true;
                            }
                        }

                        if !is_valid {
                            self.errors.push(format!(
                                "Cross-topology access error: Variable '{}' belongs to {:?} (type: {:?}), but accessed from {:?}",
                                name, top, ty, self.active_topology
                            ));
                        }
                        ty
                    }
                    None => {
                        self.errors.push(format!("Undefined variable '{}'", name));
                        Type::Tensor(ElementType::F32, vec![], None) // Default placeholder on error
                    }
                }
            }
            Expr::EnumVariant(_, _) => Type::Scalar(ElementType::I32), // Mock Enum Type
            Expr::Number(_) => Type::Scalar(ElementType::F32),
            Expr::StringLiteral(_) => Type::Pointer(
                Box::new(Type::Scalar(ElementType::I8)),
                None,
                false, // const
            ),
            Expr::Transfer(inner_expr, target_mem) => {
                let inner_ty = self.check_expr(inner_expr);
                match inner_ty {
                    Type::Ref(base_ty, _) => Type::Ref(base_ty, target_mem.clone()),
                    _ => {
                        self.errors.push(format!(
                            "Cannot transfer non-reference type: {:?}",
                            inner_ty
                        ));
                        Type::Tensor(ElementType::F32, vec![], None)
                    }
                }
            }
            Expr::FunctionCall(name, args) => {
                // Mocking built-ins
                for arg in args.iter_mut() {
                    self.check_expr(arg);
                }

                if name.starts_with("Tensor") {
                    let el_ty = match name.as_str() {
                        "Tensor_f64" => ElementType::F64,
                        "Tensor_bf16" => ElementType::BF16,
                        "Tensor_i32" => ElementType::I32,
                        "Tensor_i64" => ElementType::I64,
                        _ => ElementType::F32,
                    };
                    Type::Tensor(el_ty, vec![], None)
                } else if name == "Verified" {
                    if args.len() != 1 {
                        self.errors.push(format!(
                            "Function 'Verified' expects 1 argument, got {}",
                            args.len()
                        ));
                    }
                    let inner_ty = self.check_expr(&mut args[0]);
                    Type::Verified(Box::new(inner_ty))
                } else if name == "print" {
                    if args.len() != 1 {
                        self.errors
                            .push("Function 'print' expects 1 argument".to_string());
                    }
                    Type::Tensor(ElementType::F32, vec![], None)
                } else if let Some((ret_ty, is_unsafe)) = self.functions.get(name) {
                    if *is_unsafe && !self.in_unsafe_block {
                        self.errors.push(format!("Call to unsafe function '{}' is unsafe and requires unsafe function or block", name));
                    }
                    ret_ty.clone()
                } else if let Some(generic_func) = self.generic_functions.get(name).cloned() {
                    // Type deduction
                    let mut mapping = HashMap::new();
                    let mut success = true;
                    if args.len() != generic_func.params.len() {
                        self.errors.push(format!(
                            "Generic function '{}' expects {} arguments, got {}",
                            name,
                            generic_func.params.len(),
                            args.len()
                        ));
                        success = false;
                    } else {
                        for (i, arg) in args.iter_mut().enumerate() {
                            let arg_ty = self.check_expr(arg);
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
                                    if let Some(impl_blocks) = self.impls.get(bound_name) {
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
                        let mut inst_func = self.instantiate_function(&generic_func, &mapping);
                        let inst_ret = inst_func.return_type.clone();
                        let inst_name = inst_func.name.clone();

                        // Rewrite AST name
                        *name = inst_name.clone();

                        // Check if we already instantiated it
                        if !self.functions.contains_key(&inst_name) {
                            self.functions
                                .insert(inst_name.clone(), (inst_ret.clone(), false));
                            self.check_function(&mut inst_func);
                            self.monomorphized_functions.push(inst_func);
                        }
                        inst_ret
                    } else {
                        Type::Tensor(ElementType::F32, vec![], None)
                    }
                } else {
                    self.errors.push(format!("Undefined function '{}'", name));
                    Type::Tensor(ElementType::F32, vec![], None)
                }
            }
            Expr::Array(elements) => {
                for el in elements {
                    self.check_expr(el);
                }
                Type::Tensor(ElementType::F32, vec![], None)
            }
            Expr::MemberAccess(obj, member) => {
                let obj_ty = self.check_expr(obj);
                let mut base_ty = obj_ty.clone();
                if let Type::Borrow(t, _, _) | Type::Pointer(t, _, _) = base_ty {
                    base_ty = *t;
                }

                if let Type::Struct(struct_name) = &base_ty {
                    if let Some(decl) = self.structs.get(struct_name).cloned() {
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
                } else if member != "shape" {
                    // default behavior for Tensor.shape
                    self.errors
                        .push("Member access on non-struct type".to_string());
                }
                Type::Tensor(ElementType::F32, vec![], None)
            }
            Expr::IndexAccess(obj, idx) => {
                let obj_ty = self.check_expr(obj);
                self.check_expr(idx);
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
            Expr::MethodCall(obj, _method, args) => {
                let mut base_ty = self.check_expr(obj);
                for arg in args.iter_mut() {
                    self.check_expr(arg);
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
                            if let Expr::EnumVariant(enum_name, variant) = &args[1] {
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

                        if let Expr::Array(new_dims) = &args[0] {
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
                        if let Expr::Array(perm) = &args[0] {
                            let empty_env = HashMap::new();
                            let mut new_dims = vec![Expr::Number(0.0); dims.len()];
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
                for impl_blocks in self.impls.values() {
                    for ib in impl_blocks {
                        if self.unify_types(&ib.target_type, &base_ty, &mut HashMap::new()) {
                            for m in &ib.methods {
                                if m.name == *_method {
                                    found_method = Some((m.clone(), ib.clone()));
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
                        self.generic_functions
                            .insert(mangled_name.clone(), method_func.clone());
                    } else if !self.functions.contains_key(&mangled_name) {
                        self.functions.insert(
                            mangled_name.clone(),
                            (method_func.return_type.clone(), false),
                        );
                        // Since it's not generic, we must type check it once!
                        let mut func_to_check = method_func.clone();
                        self.check_function(&mut func_to_check);
                        self.monomorphized_functions.push(func_to_check);
                    }

                    // Rewrite AST from MethodCall to FunctionCall
                    let mut call_args = vec![(**obj).clone()];
                    for a in args.iter() {
                        call_args.push(a.clone());
                    }

                    let mut func_call = Expr::FunctionCall(mangled_name, call_args);
                    let ret_ty = self.check_expr(&mut func_call);

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
                        Topology::NPU(Box::new(Expr::Number(0.0))),
                    ); // Default to NPU[0]
                    *expr = Expr::Transfer(obj.clone(), target_mem);
                } else if _method == "to_host" {
                    let target_mem = MemorySpace::HostDRAM;
                    base_ty = Type::Pinned(Box::new(base_ty), Topology::Host);
                    *expr = Expr::Transfer(obj.clone(), target_mem);
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
            Expr::BinaryOp(lhs, op, rhs) => {
                let lhs_ty = self.check_expr(lhs);
                let rhs_ty = self.check_expr(rhs);
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
            Expr::MemorySpace(_) | Expr::Topology(_) => {
                Type::Tensor(ElementType::F32, vec![], None)
            }
            Expr::UnaryOp(op, inner) => {
                self.check_expr(inner);
                match op {
                    UnaryOp::Not => Type::Scalar(ElementType::Bool),
                }
            }
            Expr::Borrow(inner, is_mut) => {
                let inner_ty = self.check_expr(inner);
                Type::Borrow(Box::new(inner_ty), None, *is_mut)
            }
            Expr::Dereference(inner) => {
                if !self.in_unsafe_block {
                    self.errors
                        .push("Dereference of raw pointer outside of unsafe block!".to_string());
                }
                let inner_ty = self.check_expr(inner);
                match inner_ty {
                    Type::Pointer(t, _, _) | Type::Borrow(t, _, _) => *t,
                    _ => {
                        self.errors
                            .push("Cannot dereference non-pointer type".to_string());
                        inner_ty
                    }
                }
            }
            Expr::UnsafeBlock(stmts, _ret_expr) => {
                let prev_unsafe = self.in_unsafe_block;
                self.in_unsafe_block = true;
                self.push_scope();
                let mut last_type = Type::Tensor(ElementType::F32, vec![], None);
                for s in stmts.iter_mut() {
                    if let Statement::ExprStmt(ref mut expr) = s {
                        last_type = self.check_expr(expr);
                    } else {
                        self.check_statement(s, &Type::Tensor(ElementType::F32, vec![], None));
                    }
                }
                self.pop_scope();
                self.in_unsafe_block = prev_unsafe;
                last_type
            }
            Expr::StructInit(name, fields) => {
                if !self.structs.contains_key(name) {
                    self.errors.push(format!("Unknown struct {}", name));
                }
                for (_, f_expr) in fields {
                    self.check_expr(f_expr);
                }
                Type::Struct(name.clone())
            }
        }
    }

    fn is_assignable(&self, target: &Type, source: &Type) -> bool {
        if target == source {
            return true;
        }

        // Allow assigning Ref<T> to T (implicit unwrap of ref wrapper if target wants base type)
        if let Type::Ref(inner_source, _) = source {
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
            if let Type::Tensor(t_source, dims_source, top_source) = source {
                let mut el_match = false;
                if t_target == t_source {
                    el_match = true;
                } else if t_source == &ElementType::F32 && t_target != &ElementType::Bool {
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
            if let Type::Scalar(t_source) = source {
                if t_target == t_source {
                    return true;
                }
                // Allow numeric coercions
                if t_target != &ElementType::Bool && t_source != &ElementType::Bool {
                    return true;
                }
            }
        }

        // Allow coercing Scalar to Tensor (e.g. 0.0 to Tensor<f32>) for backwards compatibility with tests
        if let Type::Tensor(t_target, _, _) = target {
            if let Type::Scalar(t_source) = source {
                if t_target == t_source {
                    return true;
                }
                if t_source == &ElementType::F32 && t_target != &ElementType::Bool {
                    return true;
                }
            }
        }

        // Semantic coercion rule: Ref<T, HostDRAM> can be assigned to Verified<T>
        // Also allow returning Verified(Ref(T, Memory)) as Verified(T)
        if let Type::Verified(inner_target) = target {
            if let Type::Verified(inner_source) = source {
                if let Type::Ref(base_source, _) = &**inner_source {
                    return inner_target == base_source;
                }
            }
            if let Type::Ref(inner_source, MemorySpace::HostDRAM) = source {
                return inner_target == inner_source;
            }
        }

        // Allow coercing Borrow to Pointer (e.g. &mut T to *mut T)
        if let Type::Pointer(target_inner, target_mem, target_mut) = target {
            if let Type::Borrow(source_inner, source_mem, source_mut) = source {
                if target_inner == source_inner
                    && target_mem == source_mem
                    && (!target_mut || *source_mut)
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
        let mut parser = Parser::new(tokens);
        let mut program = parser.parse().unwrap();

        let mut checker = TypeChecker::new();
        let success = checker.check_program(&mut program).is_ok();

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
        let mut parser = Parser::new(tokens);
        let mut program = parser.parse().unwrap();

        let mut checker = TypeChecker::new();
        let success = checker.check_program(&mut program).is_ok();
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
            return c.value < 20.0;
        }
        "#;
        let mut lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer.tokenize());
        let mut program = parser.parse().unwrap();
        let mut checker = TypeChecker::new();
        assert!(
            checker.check_program(&mut program).is_ok(),
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
        let mut parser = Parser::new(lexer.tokenize());
        let mut program = parser.parse().unwrap();
        let mut checker = TypeChecker::new();

        let success = checker.check_program(&mut program).is_ok();
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
        let mut parser = Parser::new(lexer.tokenize());
        let mut program = parser.parse().unwrap();
        let mut checker = TypeChecker::new();

        assert!(
            checker.check_program(&mut program).is_ok(),
            "Semantic checking failed for methods: {:?}",
            checker.errors
        );
    }
}
