use crate::gid::{LifetimeSignature, TypeId, UnboundedFunctionMetadata};
use crate::session::LocalWorkerState;

/// High-performance verification check for variance and lifetime compatibility.
/// Encodes borrow checker math directly into the 256-bit registers.
pub fn verify_subtyping_bounds(type_a: &TypeId, type_b: &TypeId, worker: &LocalWorkerState) -> bool {
    match (worker.resolve_lifetime(type_a), worker.resolve_lifetime(type_b)) {
        (LifetimeSignature::FastPath(bits_a), LifetimeSignature::FastPath(bits_b)) => {
            // FAST PATH: Check lifetime compatibility using register operations
            if bits_a == bits_b {
                return true; // Exact structural match, exit instantly
            }

            // Evaluate individual variance rules for Parameter 0
            let param_a = bits_a & 0xFFFF;
            let param_b = bits_b & 0xFFFF;

            let variance_a = param_a >> 12;
            let variance_b = param_b >> 12;

            if variance_a == variance_b {
                let region_a = param_a & 0x0FFF;
                let region_b = param_b & 0x0FFF;
                // Evaluate relationship between Region A and Region B
                // Region 0 is 'static. Smaller regions outlive larger regions.
                // A type with a larger region can be coerced to a type with a smaller region? No,
                // A longer lifetime (smaller ID) can be coerced to a shorter lifetime (larger ID).
                // So region_a (source) <= region_b (target).
                return region_a <= region_b;
            }
            false
        }
        (LifetimeSignature::SlowPath(meta_a), LifetimeSignature::SlowPath(meta_b)) => {
            // SLOW PATH: Iterate through deep vector elements sequentially
            evaluate_slow_path_variance(meta_a, meta_b)
        }
        _ => false, // Incompatible layout paths
    }
}

fn evaluate_slow_path_variance(a: &UnboundedFunctionMetadata, b: &UnboundedFunctionMetadata) -> bool {
    // Unbounded processing logic for complex signatures
    // For now, require exact structural match
    a.lifetime_regions == b.lifetime_regions && a.trait_vtables == b.trait_vtables
}
