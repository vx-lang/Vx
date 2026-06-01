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
}

pub struct GlobalAstEnv<'a> {
    pub structs: HashMap<String, &'a StructDecl>,
    #[allow(clippy::type_complexity)]
    pub enums: HashMap<String, &'a EnumDecl>,
    pub traits: HashMap<String, &'a TraitDecl>,
    pub impls: HashMap<String, Vec<&'a ImplBlock>>,
    pub functions: HashMap<String, (Type, bool, Vec<Type>, Topology)>,
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
                env.enums.insert(e.name.clone(), e);
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
                let param_types: Vec<Type> = ext.params.iter().map(|(_, t)| t.clone()).collect();
                env.functions.insert(
                    ext.name.clone(),
                    (
                        ext.return_type.clone(),
                        !ext.is_safe,
                        param_types,
                        Topology::Host,
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
}

pub struct TypeChecker<'a> {
    pub worker: &'a mut crate::session::LocalWorkerState,
    pub env: &'a GlobalAstEnv<'a>,
    pub(crate) scopes: Vec<HashMap<String, (Type, Topology)>>,
    pub monomorphized_functions: Vec<(Function, u64)>,
    pub errors: crate::diagnostic::DiagnosticsVec,
    pub(crate) in_unsafe_block: bool,
    pub(crate) active_topology: Topology,
    pub(crate) active_memory: MemorySpace,
    pub hardware_graph: crate::arch::HardwareGraph,
    pub active_borrows: HashMap<String, Vec<BorrowRecord>>,
    pub constraints: Vec<Expr>,
    pub(crate) next_reg: u32,
    pub(crate) var_regs: Vec<HashMap<String, u32>>,
    pub(crate) moved_vars: Vec<std::collections::HashSet<String>>,
    pub eval_env: Vec<HashMap<String, Value>>,
    pub current_return_type: Option<Type>,
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
            active_topology: Topology::Host,
            active_memory: crate::arch::HardwareGraph::default_memory_for(&Topology::Host),
            hardware_graph: crate::arch::HardwareGraph::default(),
            active_borrows: HashMap::new(),
            constraints: Vec::new(),
            next_reg: 1,
            var_regs: vec![HashMap::new()],
            moved_vars: vec![std::collections::HashSet::new()],
            eval_env: vec![HashMap::new()],
            current_return_type: None,
        }
    }

    pub fn emit_type(&mut self, _ty: &Type) -> u32 {
        // Dummy conversion for now: Create a synthetic TypeId and push it.
        // In reality, this would hash the struct name, etc.
        let tid = crate::gid::TypeId::new(0, 0, 0, 0);
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
        self.moved_vars.push(std::collections::HashSet::new());
        self.eval_env.push(std::collections::HashMap::new());
    }

    pub fn pop_scope(&mut self) {
        let depth = self.scopes.len();
        self.scopes.pop();
        self.var_regs.pop();
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
            scope.insert(name, (ty, current_top));
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
                if name == "iter" {
                    println!("lookup('{}') = {:?}", name, ty);
                }
                return Some(ty);
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
                let e1_match = if let crate::ast::ElementType::Generic(ref name) = e1 {
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
                    if let crate::ast::Expr::Identifier(id) = dim1 {
                        if let crate::ast::Expr::Number(n) = dim2 {
                            mapping.insert(id.name.clone(), Type::Generic(n.value.clone(), None));
                        } else if let crate::ast::Expr::Identifier(id2) = dim2 {
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
                m1 == m2 && mut1 == mut2 && self.unify_types(t1, t2, mapping)
            }
            (Type::Borrow(t1, m1, mut1, _r1), Type::Borrow(t2, m2, mut2, _r2)) => {
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
            (Type::Function(p1, r1), Type::Function(p2, r2)) => {
                if p1.len() != p2.len() {
                    return false;
                }
                if !self.unify_types(r1, r2, mapping) {
                    return false;
                }
                for (a1, a2) in p1.iter().zip(p2.iter()) {
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
        let mut sorted_keys: Vec<&String> = mapping.keys().collect();
        sorted_keys.sort();
        for g_name in sorted_keys {
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
            .map(|(n, t)| {
                let substituted = t.substitute(mapping);

                (n.clone(), substituted)
            })
            .collect();
        let new_ret = generic_func.return_type.substitute(mapping);
        // Add a print to see the substituted return type!
        println!(
            "instantiate_function: func={}, new_ret={:?}",
            mangled_name, new_ret
        );
        let new_body = generic_func
            .body
            .iter()
            .map(|s| s.substitute(mapping))
            .collect();

        Function {
            name: mangled_name,
            generics: Vec::new(),
            params: new_params,
            topology: generic_func.topology.clone(),
            return_type: new_ret,
            body: new_body,
        }
    }

    pub fn mangle_path(path: &str) -> String {
        path.replace("/", "_").replace(".", "_")
    }

    pub fn check_function(&mut self, func: &mut Function) {
        println!("check_function({})", func.name);
        if !func.generics.is_empty() {
            return;
        }

        let prev_constraints = self.constraints.clone();
        let prev_ret_ty = self.current_return_type.clone();
        self.current_return_type = Some(func.return_type.clone());
        self.push_scope();
        for (name, ty) in &func.params {
            self.insert(name.clone(), ty.clone());
        }

        self.check_block(&mut func.body, &func.return_type.clone());

        self.pop_scope();
        self.current_return_type = prev_ret_ty;
        self.constraints = prev_constraints;
    }
}
