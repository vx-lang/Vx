use std::hash::{Hash, Hasher};
use rustc_hash::FxHasher;

/// Computes a fast 64-bit cryptographic-like hash for a given module path.
/// In a production environment, this might use `SipHash` or `xxHash`, but `FxHasher`
/// provides sufficient dispersion and speed for this compiler architecture.
pub fn compute_module_hash(module_path: &str) -> u64 {
    let mut hasher = FxHasher::default();
    module_path.hash(&mut hasher);
    hasher.finish()
}

/// Represents the stable topological path to a definition (struct, enum, closure).
/// This prevents incremental compilation breakage when anonymous types are re-ordered.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DefPath {
    /// A named, top-level symbol (e.g., `MyStruct`).
    Named(String),
    /// An anonymous closure or comptime block, defined by its structural layout
    /// or its position within a specific parent item, rather than file line number.
    Anonymous {
        parent_hash: u64,
        structural_hash: u64,
    },
}

impl DefPath {
    /// Computes the deterministic 64-bit Word 1 for the TypeId.
    pub fn compute_symbol_hash(&self) -> u64 {
        let mut hasher = FxHasher::default();
        match self {
            DefPath::Named(name) => {
                // 0 is the discriminator for Named
                0u8.hash(&mut hasher);
                name.hash(&mut hasher);
            }
            DefPath::Anonymous { parent_hash, structural_hash } => {
                // 1 is the discriminator for Anonymous
                1u8.hash(&mut hasher);
                parent_hash.hash(&mut hasher);
                structural_hash.hash(&mut hasher);
            }
        }
        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_named_defpath_stability() {
        let path1 = DefPath::Named("MyStruct".to_string());
        let path2 = DefPath::Named("MyStruct".to_string());
        assert_eq!(path1.compute_symbol_hash(), path2.compute_symbol_hash());
        
        let path3 = DefPath::Named("OtherStruct".to_string());
        assert_ne!(path1.compute_symbol_hash(), path3.compute_symbol_hash());
    }

    #[test]
    fn test_anonymous_defpath_stability() {
        let parent = DefPath::Named("ParentFn".to_string()).compute_symbol_hash();
        
        let anon1 = DefPath::Anonymous {
            parent_hash: parent,
            structural_hash: 0x12345678,
        };
        
        let anon2 = DefPath::Anonymous {
            parent_hash: parent,
            structural_hash: 0x12345678,
        };
        
        assert_eq!(anon1.compute_symbol_hash(), anon2.compute_symbol_hash());
        
        let anon_different_structure = DefPath::Anonymous {
            parent_hash: parent,
            structural_hash: 0x87654321,
        };
        
        assert_ne!(anon1.compute_symbol_hash(), anon_different_structure.compute_symbol_hash());
    }
}
