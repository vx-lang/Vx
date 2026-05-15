use std::sync::Arc;
use crate::gid::{TypeId, UnboundedFunctionMetadata};

/// Represents a single compilation epoch. 
/// Resolves the "Daemon Mode Memory Leak" by tying unbounded metadata and generic arenas 
/// to a short-lived session, rather than static singletons.
pub struct CompilationSession {
    pub epoch: u64,
    pub slow_path_arena: Arc<Vec<UnboundedFunctionMetadata>>,
    pub generics_arena: Arc<Vec<Vec<TypeId>>>,
}

impl CompilationSession {
    pub fn new(epoch: u64) -> Self {
        Self {
            epoch,
            slow_path_arena: Arc::new(Vec::new()),
            generics_arena: Arc::new(Vec::new()),
        }
    }
}
