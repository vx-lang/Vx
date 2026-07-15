//===- hash.rs - Vx Compiler -----------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// This file provides specialized, stable hashing utilities for the compiler.
// It implements robust hash generation for type IDs, module signatures, and
// anonymous structural types, ensuring consistent symbol mangling and metadata
// serialization across incremental builds.
//
//===----------------------------------------------------------------------===//
/// 64-bit FNV-1a parameters (standard offset basis + prime). FNV is **project-controlled and stable
/// across toolchains** — unlike `FxHasher`/`DefaultHasher`, whose algorithms may change between
/// library/compiler releases. GID word 0/1 are content hashes that must reproduce byte-for-byte
/// across builds and machines (reproducible builds, cross-crate identity), so the algorithm cannot
/// be an unpinned dependency. This mirrors the FNV scheme `arch.rs` already hand-rolls for dispatch
/// ids, for the same reason. See #195. FNV-1a is *not* cryptographic: accidental collisions are
/// negligible but not impossible, so `ImmutableGlobalRegistry::build_and_validate` additionally
/// rejects any two distinct symbols that land on the same GID.
const FNV_OFFSET_BASIS_64: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME_64: u64 = 0x0000_0100_0000_01b3;

/// Streaming 64-bit FNV-1a: fold `bytes` into a running `hash` (seeded with the offset basis on the
/// first call, then chained) so multi-field def paths hash deterministically as one stream.
#[inline]
fn fnv1a_64(hash: u64, bytes: &[u8]) -> u64 {
    let mut hash = hash;
    for b in bytes {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(FNV_PRIME_64);
    }
    hash
}

/// Computes the stable 64-bit content hash (GID word 0) for a given module path.
pub fn compute_module_hash(module_path: &str) -> u64 {
    fnv1a_64(FNV_OFFSET_BASIS_64, module_path.as_bytes())
}

/// Represents the stable topological path to a definition (struct, enum, closure).
/// This prevents incremental compilation breakage when anonymous types are re-ordered.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DefPath<'a> {
    /// A named, top-level symbol (e.g., `MyStruct`).
    Named(&'a str),
    /// An anonymous closure or comptime block, defined by its structural layout
    /// or its position within a specific parent item, rather than file line number.
    Anonymous {
        parent_hash: u64,
        structural_hash: u64,
    },
}

impl<'a> DefPath<'a> {
    /// Computes the deterministic 64-bit Word 1 for the TypeId (stable FNV-1a; see [`fnv1a_64`]).
    pub fn compute_symbol_hash(&self) -> u64 {
        match self {
            DefPath::Named(name) => {
                // 0 is the discriminator for Named.
                let h = fnv1a_64(FNV_OFFSET_BASIS_64, &[0u8]);
                fnv1a_64(h, name.as_bytes())
            }
            DefPath::Anonymous {
                parent_hash,
                structural_hash,
            } => {
                // 1 is the discriminator for Anonymous; fold the two u64 fields in a fixed byte
                // order so the hash is architecture-independent.
                let mut h = fnv1a_64(FNV_OFFSET_BASIS_64, &[1u8]);
                h = fnv1a_64(h, &parent_hash.to_le_bytes());
                fnv1a_64(h, &structural_hash.to_le_bytes())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pins the exact algorithm + constants (#195). These are FNV-1a-64 golden values computed
    /// independently; if the hash ever silently changes (e.g. someone swaps back to a std hasher),
    /// every previously emitted GID / serialized metadata would break, and this catches it.
    #[test]
    fn hash_is_stable_fnv1a_golden_values() {
        assert_eq!(compute_module_hash("crate::a"), 0xdbc1_8e8e_2c15_a207);
        assert_eq!(
            DefPath::Named("Foo").compute_symbol_hash(),
            0xed6f_617e_4534_5949
        );
    }

    /// Empty input hashes to the bare offset basis (the FNV-1a identity) — a sanity anchor on the
    /// seeding.
    #[test]
    fn empty_module_path_is_offset_basis() {
        assert_eq!(compute_module_hash(""), FNV_OFFSET_BASIS_64);
    }

    #[test]
    fn test_named_defpath_stability() {
        let path1 = DefPath::Named("MyStruct");
        let path2 = DefPath::Named("MyStruct");
        assert_eq!(path1.compute_symbol_hash(), path2.compute_symbol_hash());

        let path3 = DefPath::Named("OtherStruct");
        assert_ne!(path1.compute_symbol_hash(), path3.compute_symbol_hash());
    }

    #[test]
    fn test_anonymous_defpath_stability() {
        let parent = DefPath::Named("ParentFn").compute_symbol_hash();

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

        assert_ne!(
            anon1.compute_symbol_hash(),
            anon_different_structure.compute_symbol_hash()
        );
    }
}
