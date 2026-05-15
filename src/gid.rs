use bytemuck::{Pod, Zeroable};

// Architectural Masks
const ESCAPE_HATCH_MASK: u64 = 1 << 63;
const INDEX_MASK: u64 = !ESCAPE_HATCH_MASK;

// Bitmask Constants for Word 3
const VISIBILITY_MASK: u64 = 0xF000_0000_0000_0000;
const ATTRIBUTE_MASK: u64 = 0x0FF0_0000_0000_0000;

// Specific High-Frequency Attribute Flags
pub const ATTR_INLINE: u64 = 1 << 52;
pub const ATTR_INLINE_ALWAYS: u64 = 1 << 53;
pub const ATTR_COLD: u64 = 1 << 54;
pub const ATTR_MUST_USE: u64 = 1 << 55;

pub const TYPE_IS_POD: u64 = 1 << 44;
pub const TYPE_NEEDS_DROP: u64 = 1 << 45;

/// The 256-bit Global Identifier
#[derive(Clone, Copy, PartialEq, Eq, Hash, Pod, Zeroable, Debug)]
#[repr(C)]
pub struct TypeId {
    pub words: [u64; 4],
}

impl TypeId {
    pub fn new(module_hash: u64, symbol_hash: u64, generic_hash: u64, flags: u64) -> Self {
        Self {
            words: [module_hash, symbol_hash, generic_hash, flags],
        }
    }

    pub fn module_id(&self) -> u64 {
        self.words[0]
    }
    pub fn symbol_id(&self) -> u64 {
        self.words[1]
    }
    pub fn generic_hash(&self) -> u64 {
        self.words[2]
    }
}

// Mock structure for complex, unbounded parameter layouts (Slow Path)
#[derive(Debug, Clone)]
pub struct UnboundedFunctionMetadata {
    pub type_arguments: Vec<TypeId>,
    pub lifetime_regions: Vec<u32>,
    pub trait_vtables: Vec<u64>, // Architectural Fix: Stores resolved trait implementation IDs
}

// A global, pre-allocated, thread-safe arena for slow-path storage
pub static GLOBAL_SLOW_PATH_ARENA: once_cell::sync::Lazy<std::sync::RwLock<Vec<UnboundedFunctionMetadata>>> =
    once_cell::sync::Lazy::new(|| std::sync::RwLock::new(Vec::new()));

#[derive(Debug)]
pub enum LifetimeSignature<'a> {
    /// 99% Common Case: Raw bitmask payload contained entirely within CPU registers.
    FastPath(u64),
    /// < 1% Outlier Case: Immutable reference to un-sharded, deep heap metadata.
    SlowPath(std::sync::RwLockReadGuard<'a, Vec<UnboundedFunctionMetadata>>, usize),
}

impl TypeId {
    /// Inspects the status of the escape hatch to determine parameter layout strategy
    #[inline(always)]
    pub fn lifetime_context(&self) -> LifetimeSignature {
        let word_2 = self.words[2];

        if (word_2 & ESCAPE_HATCH_MASK) != 0 {
            // SLOW PATH: Extract the clean 63-bit index
            let index = (word_2 & INDEX_MASK) as usize;
            
            // Direct array bounds fetch from our read-only global arena
            let arena = GLOBAL_SLOW_PATH_ARENA.read().unwrap();
            LifetimeSignature::SlowPath(arena, index)
        } else {
            // FAST PATH: Return the register contents for bitwise analysis
            LifetimeSignature::FastPath(word_2)
        }
    }

    /// Helper to extract a specific parameter's 16-bit payload on the fast path
    #[inline(always)]
    pub fn extract_fast_param(&self, param_index: usize) -> u16 {
        debug_assert!(param_index < 4, "Fast path only supports up to 4 parameters");
        let word_2 = self.words[2];
        ((word_2 >> (param_index * 16)) & 0xFFFF) as u16
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum Visibility {
    Private,
    CratePublic,
    FullyPublic,
}

impl TypeId {
    /// Extracts the visibility enum directly from the register byte stream
    #[inline(always)]
    pub fn visibility(&self) -> Visibility {
        let vis_bits = (self.words[3] & VISIBILITY_MASK) >> 60;
        match vis_bits {
            0 => Visibility::Private,
            1 => Visibility::CratePublic,
            2 => Visibility::FullyPublic,
            _ => Visibility::Private, // Safe default fallback
        }
    }

    /// High-performance check for backends to see if they can bypass destructor code paths
    #[inline(always)]
    pub fn is_trivially_copyable(&self) -> bool {
        // Evaluate POD bit status with zero memory dereferencing
        (self.words[3] & TYPE_IS_POD) != 0
    }

    /// Instant verification to assist inline optimizations
    #[inline(always)]
    pub fn should_inline(&self) -> bool {
        (self.words[3] & (ATTR_INLINE | ATTR_INLINE_ALWAYS)) != 0
    }

    /// Mutation helper used during the parsing/declaration phase to bake flags in
    #[inline(always)]
    pub fn with_flags(&mut self, flags_mask: u64) {
        self.words[3] |= flags_mask;
    }
    
    /// Mutation helper used to set visibility
    #[inline(always)]
    pub fn with_visibility(&mut self, vis: Visibility) {
        // Clear existing visibility bits
        self.words[3] &= !VISIBILITY_MASK;
        let vis_bits = match vis {
            Visibility::Private => 0,
            Visibility::CratePublic => 1,
            Visibility::FullyPublic => 2,
        };
        self.words[3] |= vis_bits << 60;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_id_initialization() {
        let tid = TypeId::new(0x1122, 0x3344, 0x5566, 0x7788);
        assert_eq!(tid.module_id(), 0x1122);
        assert_eq!(tid.symbol_id(), 0x3344);
        assert_eq!(tid.generic_hash(), 0x5566);
    }

    #[test]
    fn test_visibility_extraction() {
        let mut tid = TypeId::new(0, 0, 0, 0);
        assert_eq!(tid.visibility(), Visibility::Private);

        tid.with_visibility(Visibility::CratePublic);
        assert_eq!(tid.visibility(), Visibility::CratePublic);

        tid.with_visibility(Visibility::FullyPublic);
        assert_eq!(tid.visibility(), Visibility::FullyPublic);
    }

    #[test]
    fn test_compiler_flags() {
        let mut tid = TypeId::new(0, 0, 0, 0);
        assert!(!tid.is_trivially_copyable());
        assert!(!tid.should_inline());

        tid.with_flags(TYPE_IS_POD);
        assert!(tid.is_trivially_copyable());

        tid.with_flags(ATTR_INLINE);
        assert!(tid.should_inline());
    }

    #[test]
    fn test_fast_path_extraction() {
        // Setup Word 2: Param 0 = 0xAAAA, Param 1 = 0xBBBB, Param 2 = 0xCCCC, Param 3 = 0xDDDD
        let word_2 = 0xDDDD_CCCC_BBBB_AAAA;
        let tid = TypeId::new(0, 0, word_2, 0);

        if let LifetimeSignature::FastPath(val) = tid.lifetime_context() {
            assert_eq!(val, word_2);
        } else {
            panic!("Expected FastPath");
        }

        assert_eq!(tid.extract_fast_param(0), 0xAAAA);
        assert_eq!(tid.extract_fast_param(1), 0xBBBB);
        assert_eq!(tid.extract_fast_param(2), 0xCCCC);
        assert_eq!(tid.extract_fast_param(3), 0xDDDD);
    }

    #[test]
    fn test_slow_path_arena() {
        // Insert a dummy item into the global arena
        let index = {
            let mut arena = GLOBAL_SLOW_PATH_ARENA.write().unwrap();
            let idx = arena.len();
            arena.push(UnboundedFunctionMetadata {
                type_arguments: vec![],
                lifetime_regions: vec![42],
                trait_vtables: vec![100],
            });
            idx as u64
        };

        // Construct a TypeId with the escape hatch bit set
        let word_2 = ESCAPE_HATCH_MASK | index;
        let tid = TypeId::new(0, 0, word_2, 0);

        if let LifetimeSignature::SlowPath(guard, idx) = tid.lifetime_context() {
            assert_eq!(idx, index as usize);
            let meta = &guard[idx];
            assert_eq!(meta.lifetime_regions[0], 42);
            assert_eq!(meta.trait_vtables[0], 100);
        } else {
            panic!("Expected SlowPath");
        }
    }
}
