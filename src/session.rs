use std::sync::Arc;
use crate::gid::{TypeId, UnboundedFunctionMetadata, ESCAPE_HATCH_MASK, INDEX_MASK};
use crate::hir::HirInstruction;

// Mock ImmutableGlobalRegistry
pub struct ImmutableGlobalRegistry {}

/// Represents the frozen past of the compilation process.
/// It contains everything compiled before the current phase.
pub struct GlobalSession {
    pub epoch: u64,
    pub registry: Arc<ImmutableGlobalRegistry>,
    pub slow_path_arena: Arc<Vec<UnboundedFunctionMetadata>>,
}

impl GlobalSession {
    pub fn new(epoch: u64) -> Self {
        Self {
            epoch,
            registry: Arc::new(ImmutableGlobalRegistry {}),
            slow_path_arena: Arc::new(Vec::new()),
        }
    }
}

/// Represents the mutable present of a single worker thread during Phase 1.
pub struct LocalWorkerState {
    pub global: Arc<GlobalSession>,
    
    // Completely lock-free, thread-local mutation
    pub local_slow_path_arena: Vec<UnboundedFunctionMetadata>,
    
    // The flat arrays replacing the AST
    pub local_type_stream: Vec<TypeId>, // array of 256-bit GIDs
    pub local_hir_stream: Vec<HirInstruction>,
}

use crate::gid::LifetimeSignature;

impl LocalWorkerState {
    pub fn new(global: Arc<GlobalSession>) -> Self {
        Self {
            global,
            local_slow_path_arena: Vec::new(),
            local_type_stream: Vec::new(),
            local_hir_stream: Vec::new(),
        }
    }

    #[inline(always)]
    pub fn resolve_lifetime<'a>(&'a self, type_id: &TypeId) -> LifetimeSignature<'a> {
        let word_2 = type_id.words[2];
        
        if (word_2 & ESCAPE_HATCH_MASK) != 0 {
            let index = (word_2 & INDEX_MASK) as usize;
            
            // Check the bit to determine routing
            // Bit 43 in Word 3 is LOCAL_DEFERRED_BIT
            const LOCAL_DEFERRED_BIT: u64 = 1 << 43;
            if (type_id.words[3] & LOCAL_DEFERRED_BIT) != 0 {
                LifetimeSignature::SlowPath(&self.local_slow_path_arena[index])
            } else {
                LifetimeSignature::SlowPath(&self.global.slow_path_arena[index])
            }
        } else {
            LifetimeSignature::FastPath(word_2)
        }
    }
}
