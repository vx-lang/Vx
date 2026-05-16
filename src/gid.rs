use bytemuck::{Pod, Zeroable};

// Architectural Masks
pub const ESCAPE_HATCH_MASK: u64 = 1 << 63;
pub const INDEX_MASK: u64 = !ESCAPE_HATCH_MASK;

// Bitmask Constants for Word 3
const VISIBILITY_MASK: u64 = 0xF000_0000_0000_0000;

// Specific High-Frequency Attribute Flags
pub const ATTR_INLINE: u64 = 1 << 52;
pub const ATTR_INLINE_ALWAYS: u64 = 1 << 53;
pub const ATTR_COLD: u64 = 1 << 54;
pub const ATTR_MUST_USE: u64 = 1 << 55;

pub const TYPE_IS_POD: u64 = 1 << 44;
pub const TYPE_NEEDS_DROP: u64 = 1 << 45;
pub const LOCAL_DEFERRED_BIT: u64 = 1 << 43;
pub const SYNTHETIC_MONO_FLAG: u64 = 1 << 42;
pub const IS_GENERIC_INST_FLAG: u64 = 1 << 41;

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

    #[inline(always)]
    pub fn is_local_deferred(&self) -> bool {
        (self.words[3] & LOCAL_DEFERRED_BIT) != 0
    }
}

// Mock structure for complex, unbounded parameter layouts (Slow Path)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UnboundedFunctionMetadata {
    pub type_arguments: Vec<TypeId>,
    pub lifetime_regions: Vec<u32>,
    pub trait_vtables: Vec<u64>, // Architectural Fix: Stores resolved trait implementation IDs
}

#[derive(Debug)]
pub enum LifetimeSignature<'a> {
    /// 99% Common Case: Raw bitmask payload contained entirely within CPU registers.
    FastPath(u64),
    /// < 1% Outlier Case: Immutable reference to un-sharded, deep heap metadata.
    SlowPath(&'a UnboundedFunctionMetadata),
}

impl TypeId {
    /// Inspects the status of the escape hatch to determine parameter layout strategy
    #[inline(always)]
    pub fn lifetime_context<'sess>(
        &self,
        arena: &'sess [UnboundedFunctionMetadata],
    ) -> LifetimeSignature<'sess> {
        let word_2 = self.words[2];

        if (word_2 & ESCAPE_HATCH_MASK) != 0 {
            // SLOW PATH: Extract the clean 63-bit index
            let index = (word_2 & INDEX_MASK) as usize;
            LifetimeSignature::SlowPath(&arena[index])
        } else {
            // FAST PATH: Return the register contents for bitwise analysis
            LifetimeSignature::FastPath(word_2)
        }
    }

    /// Helper to extract a specific parameter's 16-bit payload on the fast path
    #[inline(always)]
    pub fn extract_fast_param(&self, param_index: usize) -> u16 {
        debug_assert!(
            param_index < 4,
            "Fast path only supports up to 4 parameters"
        );
        let word_2 = self.words[2];
        ((word_2 >> (param_index * 16)) & 0xFFFF) as u16
    }

    /// Sets a 12-bit lifetime region and 4-bit variance into a specific parameter index.
    /// If the region exceeds 4095, it returns an error to indicate the caller MUST flip the escape hatch and use the Slow Path.
    #[inline(always)]
    pub fn try_set_fast_param(
        &mut self,
        param_index: usize,
        region_id: u16,
        variance_flags: u8,
    ) -> Result<(), &'static str> {
        debug_assert!(
            param_index < 4,
            "Fast path only supports up to 4 parameters"
        );
        if region_id > 4095 {
            return Err("Lifetime region overflowed 12 bits. Must use Slow Path.");
        }
        let payload = ((variance_flags as u16 & 0x0F) << 12) | (region_id & 0x0FFF);
        let shift = param_index * 16;
        let mask = !(0xFFFF_u64 << shift);
        self.words[2] = (self.words[2] & mask) | ((payload as u64) << shift);
        Ok(())
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

/// Serializes an entire dictionary of `TypeId` structures into a zero-copy byte stream.
/// This enables lightning-fast saving of incremental compilation metadata.
pub fn serialize_metadata_symbols(unique_types: &[TypeId], output: &mut Vec<u8>) {
    let len = unique_types.len() as u64;
    output.extend_from_slice(&len.to_le_bytes());

    // Cast the entire slice to raw bytes instantly (Zero-allocation)
    let bytes: &[u8] = bytemuck::cast_slice(unique_types);
    output.extend_from_slice(bytes);
}

/// Instantly maps a byte array back into a slice of `TypeId`s without allocating
/// new memory or chasing pointers.
pub fn deserialize_metadata_symbols(bytes: &[u8]) -> (&[TypeId], &[u8]) {
    let (len_bytes, remaining) = bytes.split_at(8);
    let len = u64::from_le_bytes(len_bytes.try_into().unwrap()) as usize;

    let byte_len = len * std::mem::size_of::<TypeId>();
    let (dict_bytes, rest) = remaining.split_at(byte_len);

    // Cast raw file bytes straight into a Rust slice with zero overhead
    let types: &[TypeId] = bytemuck::cast_slice(dict_bytes);
    (types, rest)
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

        // Reset and test other flags
        let mut tid2 = TypeId::new(0, 0, 0, 0);
        tid2.with_flags(ATTR_INLINE_ALWAYS);
        assert!(tid2.should_inline());

        let mut tid3 = TypeId::new(0, 0, 0, 0);
        tid3.with_flags(
            ATTR_COLD | ATTR_MUST_USE | TYPE_NEEDS_DROP | SYNTHETIC_MONO_FLAG | LOCAL_DEFERRED_BIT,
        );
        assert_eq!((tid3.words[3] & ATTR_COLD), ATTR_COLD);
        assert_eq!((tid3.words[3] & ATTR_MUST_USE), ATTR_MUST_USE);
        assert_eq!((tid3.words[3] & TYPE_NEEDS_DROP), TYPE_NEEDS_DROP);
        assert_eq!((tid3.words[3] & SYNTHETIC_MONO_FLAG), SYNTHETIC_MONO_FLAG);
        assert!(tid3.is_local_deferred());
    }

    #[test]
    fn test_fast_path_extraction() {
        use crate::session::{GlobalSession, LocalWorkerState};
        let global = std::sync::Arc::new(GlobalSession::new(1));
        let worker = LocalWorkerState::new(global);
        // Setup Word 2: Param 0 = 0xAAAA, Param 1 = 0xBBBB, Param 2 = 0xCCCC, Param 3 = 0x5DDD (to keep bit 63 unset)
        let word_2 = 0x5DDD_CCCC_BBBB_AAAA;
        let tid = TypeId::new(0, 0, word_2, 0);

        if let LifetimeSignature::FastPath(val) = worker.resolve_lifetime(&tid) {
            assert_eq!(val, word_2);
        } else {
            panic!("Expected FastPath");
        }

        assert_eq!(tid.extract_fast_param(0), 0xAAAA);
        assert_eq!(tid.extract_fast_param(1), 0xBBBB);
        assert_eq!(tid.extract_fast_param(2), 0xCCCC);
        assert_eq!(tid.extract_fast_param(3), 0x5DDD);
    }

    #[test]
    fn test_fast_path_lifetime_overflow() {
        let mut tid = TypeId::new(0, 0, 0, 0);

        // Setting a valid lifetime (4095 is the max 12-bit value)
        assert!(tid.try_set_fast_param(0, 4095, 0).is_ok());

        // Setting an invalid lifetime (4096) should return an Err
        assert!(tid.try_set_fast_param(1, 4096, 0).is_err());
    }

    #[test]
    fn test_slow_path_arena() {
        use crate::session::{GlobalSession, LocalWorkerState};
        let global = std::sync::Arc::new(GlobalSession::new(1));
        let mut worker = LocalWorkerState::new(global);
        // Insert a dummy item into the local arena
        let index = {
            let idx = worker.local_slow_path_arena.len();
            worker
                .local_slow_path_arena
                .push(UnboundedFunctionMetadata {
                    type_arguments: vec![],
                    lifetime_regions: vec![42],
                    trait_vtables: vec![100],
                });
            idx as u64
        };

        // Construct a TypeId with the escape hatch bit and local deferred bit set
        let word_2 = ESCAPE_HATCH_MASK | index;
        let mut tid = TypeId::new(0, 0, word_2, 0);
        tid.words[3] |= LOCAL_DEFERRED_BIT;

        if let LifetimeSignature::SlowPath(meta) = worker.resolve_lifetime(&tid) {
            assert_eq!(meta.lifetime_regions[0], 42);
            assert_eq!(meta.trait_vtables[0], 100);
        } else {
            panic!("Expected SlowPath");
        }
    }

    #[test]
    fn test_zero_copy_serialization() {
        let mut t1 = TypeId::new(1, 2, 3, 0);
        t1.with_flags(TYPE_IS_POD);
        let mut t2 = TypeId::new(4, 5, 6, 0);
        t2.with_visibility(Visibility::FullyPublic);

        let dict = vec![t1, t2];
        let mut buffer = Vec::new();
        serialize_metadata_symbols(&dict, &mut buffer);

        // The length header is 8 bytes, plus 2 TypeIds (32 bytes each) = 72 bytes total.
        assert_eq!(buffer.len(), 72);

        let (deserialized, remaining) = deserialize_metadata_symbols(&buffer);
        assert_eq!(remaining.len(), 0);
        assert_eq!(deserialized.len(), 2);

        assert_eq!(deserialized[0], t1);
        assert!(deserialized[0].is_trivially_copyable());

        assert_eq!(deserialized[1], t2);
        assert_eq!(deserialized[1].visibility(), Visibility::FullyPublic);
    }
}
