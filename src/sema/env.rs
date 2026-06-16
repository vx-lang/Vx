//===- sema.rs - Vx Compiler -----------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// This file implements the Semantic Analyzer for the Vx compiler.
// It is responsible for type checking, resolving operator overloading (e.g., tensor
// matrix multiplication), verifying memory topology constraints, and constructing
// the global AST environment for subsequent lowering phases.
//
// The Semantic Analyzer also includes the Lexical Borrow Checker, which handles local variable
// lifetimes and Strict Aliasing (Shared XOR Mutable).
// For a comprehensive overview of the Borrow Checker architecture (and how it interacts with
// the FastPath in borrow.rs), see: `docs/discussions/borrow_checker_architecture.md`.
//
// DESIGN NOTE: The `silent` parameter (used in `check_expr_type_flag` and others)
// prevents duplicate compiler errors. Because AST nodes are often traversed multiple
// times (once for initial type validation, and again later when lowering to HIR),
// the `silent` flag is set to `true` on subsequent passes to suppress redundant
// error emissions.
//
//===----------------------------------------------------------------------===//
use crate::ast::*;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Bool(bool),
    Number(f64),
    Topology(Topology),
}

pub struct GlobalAstEnv<'a> {
    pub structs: HashMap<crate::symbol::Symbol, &'a StructDecl>,
    #[allow(clippy::type_complexity)]
    pub enums: HashMap<crate::symbol::Symbol, &'a EnumDecl>,
    pub traits: HashMap<crate::symbol::Symbol, &'a TraitDecl>,
    pub impls: HashMap<crate::symbol::Symbol, Vec<&'a ImplBlock>>,
    #[allow(clippy::type_complexity)]
    pub functions:
        HashMap<crate::symbol::Symbol, (Type, bool, Vec<Type>, Topology, Vec<Expr>, Vec<Expr>)>,
    pub ast_functions: HashMap<crate::symbol::Symbol, &'a Function>,
    pub generic_functions: HashMap<crate::symbol::Symbol, (&'a Function, u64)>, // (func, origin_module_hash)
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
                env.enums.insert(e.name.clone(), e);
            }
            for t in &module.traits {
                env.traits.insert(t.name.clone(), t);
            }
            for i in &module.impls {
                let trait_name = match &i.trait_name {
                    Some(name) => name.clone(),
                    None => "_inherent".to_string().into(),
                };
                env.impls.entry(trait_name).or_default().push(i);
            }
            for ext in &module.externs {
                let param_types: Vec<Type> = ext.params.iter().map(|(_, t)| t.clone()).collect();
                env.functions.insert(
                    ext.name.clone(),
                    (
                        ext.return_type.clone(),
                        !ext.is_safe,
                        param_types,
                        Topology::CPU,
                        Vec::new(),
                        Vec::new(),
                    ),
                );
            }
            for func in &module.functions {
                if !func.generics.is_empty() {
                    let module_hash = crate::hash::compute_module_hash(&module.module_path);
                    env.generic_functions
                        .insert(func.name.clone(), (func, module_hash));
                } else {
                    let param_types: Vec<Type> =
                        func.params.iter().map(|(_, t)| t.clone()).collect();
                    env.functions.insert(
                        func.name.clone(),
                        (
                            func.return_type.clone(),
                            false, /* func.is_unsafe */
                            param_types,
                            func.topology.clone(),
                            func.requires.clone(),
                            func.ensures.clone(),
                        ),
                    );
                    env.ast_functions.insert(func.name.clone(), func);
                }
            }
        }
        env
    }
}

#[derive(Debug, Clone)]
pub struct BorrowRecord {
    pub is_mut: bool,
    pub scope_depth: usize,
    pub borrower_name: Option<String>,
    pub path: Vec<String>,
}

pub struct TypeChecker<'a> {
    pub worker: &'a mut crate::session::LocalWorkerState,
    pub env: &'a GlobalAstEnv<'a>,
    pub(crate) scopes: Vec<HashMap<crate::symbol::Symbol, (Type, Topology)>>,
    pub monomorphized_functions: Vec<(Function, u64)>,
    pub errors: crate::diagnostic::DiagnosticsVec,
    pub(crate) in_unsafe_block: bool,
    pub(crate) active_topology: Topology,
    pub(crate) active_memory: MemorySpace,
    pub transfer_cost_graph: crate::arch::TransferCostGraph,
    pub active_borrows: HashMap<crate::symbol::Symbol, Vec<BorrowRecord>>,
    pub constraints: Vec<Expr>,
    pub(crate) next_id: u32,
    pub(crate) moved_vars: Vec<std::collections::HashSet<String>>,
    pub eval_env: Vec<HashMap<crate::symbol::Symbol, Value>>,
    pub current_return_type: Option<Type>,
    #[allow(dead_code)]
    pub(crate) closure_depths: Vec<usize>,
    #[allow(dead_code)]
    pub(crate) closure_captures_stack: Vec<HashMap<crate::symbol::Symbol, Type>>,
    pub generated_structs: Vec<StructDecl>,
    pub(crate) current_assignment_target: Option<String>,
    pub(crate) block_liveness: Vec<HashMap<crate::symbol::Symbol, usize>>,
    pub(crate) current_stmt_idx: Vec<usize>,
    pub skip_borrow_check: bool,
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
            errors: crate::diagnostic::DiagnosticsVec::new(),
            in_unsafe_block: false,
            active_topology: Topology::CPU,
            active_memory: crate::arch::TransferCostGraph::default_memory_for(&Topology::CPU),
            transfer_cost_graph: crate::arch::TransferCostGraph::default(),
            active_borrows: HashMap::new(),
            constraints: Vec::new(),
            next_id: 1,
            moved_vars: vec![std::collections::HashSet::new()],
            eval_env: vec![HashMap::new()],
            current_return_type: None,
            closure_depths: Vec::new(),
            closure_captures_stack: Vec::new(),
            generated_structs: Vec::new(),
            current_assignment_target: None,
            block_liveness: Vec::new(),
            current_stmt_idx: Vec::new(),
            skip_borrow_check: false,
        }
    }

    pub fn push_scope(&mut self) {
        self.scopes.push(std::collections::HashMap::new());
        self.moved_vars.push(std::collections::HashSet::new());
        self.eval_env.push(std::collections::HashMap::new());
    }

    pub fn pop_scope(&mut self) {
        let depth = self.scopes.len();
        self.scopes.pop();
        self.moved_vars.pop();
        self.eval_env.pop();

        // Lexical Lifetime cleanup: Remove borrows originating in this scope
        for (_, borrows) in self.active_borrows.iter_mut() {
            borrows.retain(|b| b.scope_depth < depth);
        }
    }

    pub fn insert(&mut self, name: String, ty: Type) {
        let current_top = self.active_topology.clone();
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.into(), (ty, current_top));
        }
    }

    pub fn is_variable_used_after(&self, name: &str) -> bool {
        if let (Some(liveness), Some(&current_idx)) =
            (self.block_liveness.last(), self.current_stmt_idx.last())
        {
            return liveness
                .get(name)
                .map(|&u| u > current_idx)
                .unwrap_or(false);
        }
        false
    }

    pub fn extract_uses_stmt(stmt: &Statement, uses: &mut std::collections::HashSet<String>) {
        match stmt {
            Statement::ExprStmt(e) => Self::extract_uses_expr(&e.expr, uses),
            Statement::Return(r) => Self::extract_uses_expr(&r.expr, uses),
            Statement::Assign(a) => {
                Self::extract_uses_expr(&a.lhs, uses);
                Self::extract_uses_expr(&a.rhs, uses);
            }
            Statement::CompoundAssign(ca) => {
                Self::extract_uses_expr(&ca.lhs, uses);
                Self::extract_uses_expr(&ca.rhs, uses);
            }
            Statement::LetDecl(l) => Self::extract_uses_expr(&l.expr, uses),
            Statement::ForLoop(f) => {
                Self::extract_uses_expr(&f.iterable, uses);
                for inv in &f.invariants {
                    Self::extract_uses_expr(inv, uses);
                }
                for s in &f.body {
                    Self::extract_uses_stmt(s, uses);
                }
            }
            Statement::Loop(l) => {
                for inv in &l.invariants {
                    Self::extract_uses_expr(inv, uses);
                }
                for s in &l.body {
                    Self::extract_uses_stmt(s, uses);
                }
            }
            Statement::Assert(a) => Self::extract_uses_expr(&a.expr, uses),
            _ => {}
        }
    }

    pub fn extract_uses_expr(expr: &Expr, uses: &mut std::collections::HashSet<String>) {
        match expr {
            Expr::Identifier(id) => {
                uses.insert(id.name.to_string());
            }
            Expr::MemberAccess(m) => Self::extract_uses_expr(&m.base, uses),
            Expr::MethodCall(m) => {
                Self::extract_uses_expr(&m.base, uses);
                for a in &m.args {
                    Self::extract_uses_expr(a, uses);
                }
            }
            Expr::FunctionCall(f) => {
                uses.insert(f.name.to_string());
                for a in &f.args {
                    Self::extract_uses_expr(a, uses);
                }
            }
            Expr::BinaryOp(b) => {
                Self::extract_uses_expr(&b.lhs, uses);
                Self::extract_uses_expr(&b.rhs, uses);
            }
            Expr::RelationalOp(r) => {
                Self::extract_uses_expr(&r.lhs, uses);
                Self::extract_uses_expr(&r.rhs, uses);
            }
            Expr::LogicalOp(l) => {
                Self::extract_uses_expr(&l.lhs, uses);
                Self::extract_uses_expr(&l.rhs, uses);
            }
            Expr::UnaryOp(u) => Self::extract_uses_expr(&u.expr, uses),
            Expr::IndexAccess(i) => {
                Self::extract_uses_expr(&i.base, uses);
                Self::extract_uses_expr(&i.index, uses);
            }
            Expr::Borrow(b) => Self::extract_uses_expr(&b.expr, uses),
            Expr::Dereference(d) => Self::extract_uses_expr(&d.expr, uses),
            Expr::StructInit(s) => {
                for f in &s.fields {
                    Self::extract_uses_expr(&f.1, uses);
                }
            }
            Expr::Array(a) => {
                for e in &a.elements {
                    Self::extract_uses_expr(e, uses);
                }
            }
            Expr::If(i) => {
                Self::extract_uses_expr(&i.cond, uses);
                for s in &i.then_block {
                    Self::extract_uses_stmt(s, uses);
                }
                if let Some(eb) = &i.else_block {
                    for s in eb {
                        Self::extract_uses_stmt(s, uses);
                    }
                }
            }
            Expr::UnsafeBlock(u) => {
                for s in &u.stmts {
                    Self::extract_uses_stmt(s, uses);
                }
            }
            Expr::AsCast(c) => Self::extract_uses_expr(&c.expr, uses),
            Expr::Closure(c) => {
                Self::extract_uses_expr(&c.body, uses);
            }
            _ => {}
        }
    }

    pub fn consume(&mut self, name: &str) {
        // Find the most recent block where it's defined
        for scope in self.scopes.iter_mut().rev() {
            if scope.contains_key(name) {
                scope.remove(name);
                self.moved_vars.last_mut().unwrap().insert(name.to_string());
                return;
            }
        }
    }

    pub fn is_moved(&self, name: &str) -> bool {
        for moved in self.moved_vars.iter().rev() {
            if moved.contains(name) {
                return true;
            }
        }
        false
    }

    pub fn lookup(&self, name: &str) -> Option<&(Type, Topology)> {
        for scope in self.scopes.iter().rev() {
            if let Some(ty) = scope.get(name) {
                return Some(ty);
            }
        }
        None
    }

    pub fn lookup_with_depth(&self, name: &str) -> Option<(&Type, &Topology, usize)> {
        for (i, scope) in self.scopes.iter().enumerate().rev() {
            if let Some((ty, top)) = scope.get(name) {
                return Some((ty, top, i));
            }
        }
        None
    }

    pub fn unify_types(
        &mut self,
        generic_ty: &Type,
        concrete_ty: &Type,
        mapping: &mut std::collections::HashMap<crate::symbol::Symbol, Type>,
    ) -> bool {
        let mut temp_mapping = mapping.clone();
        if self.unify_types_internal(generic_ty, concrete_ty, &mut temp_mapping) {
            *mapping = temp_mapping;
            true
        } else {
            false
        }
    }

    fn unify_types_internal(
        &mut self,
        generic_ty: &Type,
        concrete_ty: &Type,
        mapping: &mut std::collections::HashMap<crate::symbol::Symbol, Type>,
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
                let e1_match = if let ElementType::Generic(ref name) = e1 {
                    if let Some(existing) = mapping.get(name) {
                        existing == &Type::Scalar(e2.clone())
                    } else {
                        mapping.insert(name.clone(), Type::Scalar(e2.clone()));
                        true
                    }
                } else {
                    e1 == e2
                };
                if !e1_match || d1.len() != d2.len() || t1 != t2 {
                    return false;
                }
                for (dim1, dim2) in d1.iter().zip(d2.iter()) {
                    if let Expr::Identifier(id) = dim1 {
                        if let Expr::Number(n) = dim2 {
                            mapping.insert(
                                id.name.clone(),
                                Type::Generic(n.value.clone().into(), None),
                            );
                        } else if let Expr::Identifier(id2) = dim2 {
                            mapping.insert(id.name.clone(), Type::Generic(id2.name.clone(), None));
                        } else if dim1 != dim2 {
                            return false;
                        }
                    } else if dim1 != dim2 {
                        return false;
                    }
                }
                true
            }
            (Type::Pointer(t1, m1, mut1), Type::Pointer(t2, m2, mut2)) => {
                m1 == m2 && mut1 == mut2 && self.unify_types_internal(t1, t2, mapping)
            }
            (
                Type::Borrow {
                    inner: t1,
                    mem_space: m1,
                    is_mut: mut1,
                    ..
                },
                Type::Borrow {
                    inner: t2,
                    mem_space: m2,
                    is_mut: mut2,
                    ..
                },
            ) => m1 == m2 && mut1 == mut2 && self.unify_types_internal(t1, t2, mapping),
            (Type::Ref(t1, m1), Type::Ref(t2, m2)) => {
                m1 == m2 && self.unify_types_internal(t1, t2, mapping)
            }
            (Type::GenericInstance(b1, args1), Type::GenericInstance(b2, args2)) => {
                if args1.len() != args2.len() {
                    return false;
                }
                if !self.unify_types_internal(b1, b2, mapping) {
                    return false;
                }
                for (a1, a2) in args1.iter().zip(args2.iter()) {
                    if !self.unify_types_internal(a1, a2, mapping) {
                        return false;
                    }
                }
                true
            }
            (Type::Function(p1, r1), Type::Function(p2, r2)) => {
                if p1.len() != p2.len() {
                    return false;
                }
                if !self.unify_types_internal(r1, r2, mapping) {
                    return false;
                }
                for (a1, a2) in p1.iter().zip(p2.iter()) {
                    if !self.unify_types_internal(a1, a2, mapping) {
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
        mapping: &std::collections::HashMap<crate::symbol::Symbol, Type>,
    ) -> Function {
        let mut mangled_name = generic_func.name.to_string();
        let mut sorted_keys: Vec<&crate::symbol::Symbol> = mapping.keys().collect();
        sorted_keys.sort();
        for g_name in sorted_keys {
            if let Some(ty) = mapping.get(g_name) {
                mangled_name.push_str(&format!("${}", ty.mangle()));
            }
        }

        let new_params = generic_func
            .params
            .iter()
            .map(|(n, t)| {
                let substituted = t.substitute(mapping);

                (n.clone(), substituted)
            })
            .collect();
        let new_ret = generic_func.return_type.substitute(mapping);

        let new_body = generic_func
            .body
            .iter()
            .map(|s| s.substitute(mapping))
            .collect();

        Function {
            name: mangled_name.into(),
            generics: Vec::new(),
            params: new_params,
            topology: generic_func.topology.clone(),
            return_type: new_ret,
            requires: generic_func
                .requires
                .iter()
                .map(|e| e.substitute(mapping))
                .collect(),
            ensures: generic_func
                .ensures
                .iter()
                .map(|e| e.substitute(mapping))
                .collect(),
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

        let prev_constraints = self.constraints.clone();
        let prev_ret_ty = self.current_return_type.clone();
        self.current_return_type = Some(func.return_type.clone());
        self.push_scope();

        let prev_top = self.active_topology.clone();
        let prev_mem = self.active_memory.clone();
        self.active_topology = func.topology.clone();
        self.active_memory =
            crate::arch::TransferCostGraph::default_memory_for(&self.active_topology);

        for (name, ty) in &func.params {
            self.insert(name.to_string(), ty.clone());
        }

        // Add preconditions (requires) to our constraints
        for req in &func.requires {
            self.constraints.push(req.clone());
        }

        self.check_block(&mut func.body, &func.return_type.clone());

        // Verify postconditions (ensures)
        for ens in &func.ensures {
            if !self.prove_expr(ens) {
                self.errors.push(format!(
                    "Function '{}' cannot prove postcondition (ensures) at compile time",
                    func.name
                ));
            }
        }

        self.pop_scope();
        self.current_return_type = prev_ret_ty;
        self.constraints = prev_constraints;
        self.active_topology = prev_top;
        self.active_memory = prev_mem;
    }

    pub fn parse_ty_str(&self, s: &str) -> Type {
        let mut lexer = crate::lexer::Lexer::new(s);
        let tokens = lexer.tokenize();

        let mut parser = crate::parser::Parser::new(&tokens, s);
        if let Ok(ty) = parser.parse_type() {
            if parser.check(&crate::lexer::TokenType::Eof) {
                return self.resolve_parsed_type(ty);
            }
        }

        let mut expr_parser = crate::parser::Parser::new(&tokens, s);
        if let Ok(expr) = expr_parser.parse_primary_expr() {
            return Type::Const(Box::new(expr));
        }

        Type::Unknown
    }

    fn resolve_parsed_type(&self, ty: Type) -> Type {
        match ty {
            Type::Struct(name, id) => {
                if self.env.structs.contains_key(&name)
                    || self.generated_structs.iter().any(|s| s.name == name)
                {
                    Type::Struct(name, id)
                } else {
                    Type::Generic(name, id)
                }
            }
            Type::GenericInstance(base, args) => {
                let resolved_base = Box::new(self.resolve_parsed_type(*base));
                let resolved_args = args
                    .into_iter()
                    .map(|a| self.resolve_parsed_type(a))
                    .collect();
                Type::GenericInstance(resolved_base, resolved_args)
            }
            Type::Pointer(inner, mem, mut_flag) => {
                Type::Pointer(Box::new(self.resolve_parsed_type(*inner)), mem, mut_flag)
            }
            Type::Ref(inner, mem) => Type::Ref(Box::new(self.resolve_parsed_type(*inner)), mem),
            Type::Borrow {
                inner,
                mem_space: mem,
                is_mut: mut_flag,
                region_id: r,
            } => Type::Borrow {
                inner: Box::new(self.resolve_parsed_type(*inner)),
                mem_space: mem.clone(),
                is_mut: mut_flag,
                region_id: r,
            },
            Type::Pinned(inner, top) => {
                Type::Pinned(Box::new(self.resolve_parsed_type(*inner)), top)
            }
            Type::Function(args, ret) => {
                let resolved_args = args
                    .into_iter()
                    .map(|a| self.resolve_parsed_type(a))
                    .collect();
                Type::Function(resolved_args, Box::new(self.resolve_parsed_type(*ret)))
            }
            Type::Closure(args, ret) => {
                let resolved_args = args
                    .into_iter()
                    .map(|a| self.resolve_parsed_type(a))
                    .collect();
                Type::Closure(resolved_args, Box::new(self.resolve_parsed_type(*ret)))
            }
            _ => ty,
        }
    }
}
