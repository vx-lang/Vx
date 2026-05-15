use std::collections::HashMap;
use crate::ast::VxModule;
use crate::gid::TypeId;
use crate::hash::{compute_module_hash, DefPath};

/// A global read-only map generated during Phase 1.25.
/// Maps module_path -> (symbol_name -> TypeId)
pub type SymbolMap = HashMap<String, HashMap<String, TypeId>>;

/// Iterates through all parsed modules sequentially and computes their deterministic
/// 256-bit TypeId for every top-level struct, enum, and trait.
pub fn build_symbol_map(modules: &[VxModule]) -> SymbolMap {
    let mut map: SymbolMap = HashMap::new();

    for module in modules {
        let mut module_symbols = HashMap::new();
        let module_hash = compute_module_hash(&module.module_path);

        for struct_decl in &module.structs {
            let sym_hash = DefPath::Named(struct_decl.name.clone()).compute_symbol_hash();
            // Flag parsing can happen later, for now we just use 0
            let tid = TypeId::new(module_hash, sym_hash, 0, 0);
            module_symbols.insert(struct_decl.name.clone(), tid);
        }

        for enum_decl in &module.enums {
            let sym_hash = DefPath::Named(enum_decl.name.clone()).compute_symbol_hash();
            let tid = TypeId::new(module_hash, sym_hash, 0, 0);
            module_symbols.insert(enum_decl.name.clone(), tid);
        }
        
        for trait_decl in &module.traits {
            let sym_hash = DefPath::Named(trait_decl.name.clone()).compute_symbol_hash();
            let tid = TypeId::new(module_hash, sym_hash, 0, 0);
            module_symbols.insert(trait_decl.name.clone(), tid);
        }

        map.insert(module.module_path.clone(), module_symbols);
    }

    map
}
