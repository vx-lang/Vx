//===- api_diff.rs - Vx Compiler -------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Compare the interfaces of two versions of a library, read from their `.vxlib` files, and find
// the smallest safe version bump. An added item needs a minor release. A removed item, or a
// change that breaks a caller, needs a major one. Anything else is a patch.
//
// Items are matched by name. An item's GID includes a hash of the path the library was compiled
// from, so one library built in two directories has two sets of GIDs.
//
// Every item in a `.vxlib` counts as public API: a program can call a function through a
// `.vxlib` whether or not the function is `pub`.
//
//===----------------------------------------------------------------------===//

use crate::gid::TypeId;
use crate::registry::{FnSig, ImmutableGlobalRegistry, StructFields};
use crate::symbol::Symbol;
use crate::syntax::ElementType;
use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;

/// The part of a version number that a release must raise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Bump {
    Patch,
    Minor,
    Major,
}

impl fmt::Display for Bump {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(match self {
            Bump::Patch => "patch",
            Bump::Minor => "minor",
            Bump::Major => "major",
        })
    }
}

/// One difference between two interfaces, and the bump it needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Change {
    pub bump: Bump,
    pub description: String,
}

/// Every difference between two interfaces.
#[derive(Debug, Default)]
pub struct ApiDiff {
    pub changes: Vec<Change>,
}

impl ApiDiff {
    /// The bump of the release: the largest bump of its changes, and at least a patch.
    pub fn bump(&self) -> Bump {
        self.changes
            .iter()
            .map(|c| c.bump)
            .max()
            .unwrap_or(Bump::Patch)
    }

    fn push(&mut self, bump: Bump, description: String) {
        self.changes.push(Change { bump, description });
    }
}

/// What a `.vxlib` does not hold, so a comparison cannot see a change to it.
pub const NOT_COMPARED: &str =
    "traits, enums whose variants carry data, constants, and the bounds of generic parameters";

/// Read the interface that `vxc --emit-interface` wrote to `path`.
pub fn read_interface(path: &Path) -> Result<ImmutableGlobalRegistry, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    // The file starts with a count of 32-byte entries. Check it here, because the decoder
    // trusts it and would panic on a short file.
    let fits = bytes.len() >= 8 && {
        let count = u64::from_le_bytes(bytes[..8].try_into().unwrap());
        count
            .checked_mul(std::mem::size_of::<TypeId>() as u64)
            .is_some_and(|size| size <= (bytes.len() - 8) as u64)
    };
    if !fits {
        return Err(format!("{} is not a .vxlib file", path.display()));
    }
    let metadata = crate::metadata::VxMetadata::load_from_buffer(&bytes);
    crate::metadata::deserialize_registry_interface(metadata.interface_data)
        .map_err(|e| format!("{}: {e}", path.display()))
}

/// Compare the interface of an old version of a library with the interface of a new one.
pub fn diff(old: &ImmutableGlobalRegistry, new: &ImmutableGlobalRegistry) -> ApiDiff {
    let mut diff = ApiDiff::default();
    compare_functions(&mut diff, "function", &functions(old), &functions(new));
    compare_functions(&mut diff, "method", &methods(old), &methods(new));
    compare_structs(&mut diff, &structs(old), &structs(new));
    compare_enums(&mut diff, &enums(old), &enums(new));
    diff
}

fn functions(r: &ImmutableGlobalRegistry) -> BTreeMap<String, &FnSig> {
    r.fn_sigs
        .iter()
        .map(|(name, sig)| (name.to_string(), sig))
        .collect()
}

fn methods(r: &ImmutableGlobalRegistry) -> BTreeMap<String, &FnSig> {
    r.methods
        .iter()
        .map(|((receiver, name), sig)| (format!("{}::{name}", type_name(r, *receiver)), sig))
        .collect()
}

fn structs(r: &ImmutableGlobalRegistry) -> BTreeMap<String, &StructFields> {
    r.structs
        .iter()
        .map(|(id, fields)| (type_name(r, *id), fields))
        .collect()
}

fn enums(r: &ImmutableGlobalRegistry) -> BTreeMap<String, &Vec<Symbol>> {
    r.enum_variants
        .iter()
        .map(|(name, variants)| (name.to_string(), variants))
        .collect()
}

/// The name of type `id` in the interface `r`. A struct or an enum has a layout or an index
/// entry that carries its name. A scalar such as `f32` has a fixed GID.
fn type_name(r: &ImmutableGlobalRegistry, id: TypeId) -> String {
    if let Some(def) = r.layouts.get(&id) {
        return def.name.clone();
    }
    let indexed = r
        .module_indices
        .values()
        .flat_map(|names| names.iter())
        .find(|(_, gid)| **gid == id);
    if let Some((name, _)) = indexed {
        return name.to_string();
    }
    scalars()
        .into_iter()
        .find(|e| crate::hir::flatten::scalar_gid(e) == id)
        .map(|e| e.to_string())
        .unwrap_or_else(|| format!("{id:?}"))
}

/// The scalar types a method can be declared on. A scalar missing here prints as its GID.
fn scalars() -> [ElementType; 21] {
    use ElementType::*;
    [
        F16, F32, F64, BF16, F8E4M3, F8E5M2, F4E2M1, I4, U4, I8, U8, I16, U16, I32, U32, I64, U64,
        I128, U128, USize, Bool,
    ]
}

/// The parameter types and the return type, which a caller depends on.
fn shape(sig: &FnSig) -> (Vec<String>, String) {
    let params = sig.params.iter().map(|t| t.to_string()).collect();
    (params, sig.ret_ty.to_string())
}

fn signature(sig: &FnSig) -> String {
    let (params, ret) = shape(sig);
    let unsafe_ = if sig.is_unsafe { "unsafe " } else { "" };
    format!("{unsafe_}fn({}) -> {ret}", params.join(", "))
}

fn compare_functions(
    diff: &mut ApiDiff,
    kind: &str,
    old: &BTreeMap<String, &FnSig>,
    new: &BTreeMap<String, &FnSig>,
) {
    for (name, old_sig) in old {
        let Some(new_sig) = new.get(name) else {
            diff.push(Bump::Major, format!("{kind} `{name}` was removed"));
            continue;
        };
        if shape(old_sig) != shape(new_sig) {
            diff.push(
                Bump::Major,
                format!(
                    "{kind} `{name}` changed from `{}` to `{}`",
                    signature(old_sig),
                    signature(new_sig)
                ),
            );
        } else if old_sig.ret_prov != new_sig.ret_prov {
            // A caller's borrow check depends on which parameter a returned reference comes
            // from. A new signature already counts as major, so this is checked only without one.
            diff.push(
                Bump::Major,
                format!(
                    "the reference that {kind} `{name}` returns comes from a different parameter"
                ),
            );
        }
        match (old_sig.is_unsafe, new_sig.is_unsafe) {
            (false, true) => diff.push(Bump::Major, format!("{kind} `{name}` became unsafe")),
            (true, false) => diff.push(Bump::Minor, format!("{kind} `{name}` is no longer unsafe")),
            _ => {}
        }
    }
    for name in new.keys().filter(|name| !old.contains_key(*name)) {
        diff.push(Bump::Minor, format!("{kind} `{name}` was added"));
    }
}

fn compare_structs(
    diff: &mut ApiDiff,
    old: &BTreeMap<String, &StructFields>,
    new: &BTreeMap<String, &StructFields>,
) {
    let fields = |s: &StructFields| -> BTreeMap<String, String> {
        s.fields
            .iter()
            .map(|(name, ty)| (name.to_string(), ty.to_string()))
            .collect()
    };
    for (name, old_struct) in old {
        let Some(new_struct) = new.get(name) else {
            diff.push(Bump::Major, format!("struct `{name}` was removed"));
            continue;
        };
        if old_struct.generics != new_struct.generics {
            let list = |s: &StructFields| {
                s.generics
                    .iter()
                    .map(|g| g.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            diff.push(
                Bump::Major,
                format!(
                    "struct `{name}` changed its type parameters from `<{}>` to `<{}>`",
                    list(old_struct),
                    list(new_struct)
                ),
            );
        }
        let (old_fields, new_fields) = (fields(old_struct), fields(new_struct));
        for (field, old_ty) in &old_fields {
            match new_fields.get(field) {
                None => diff.push(
                    Bump::Major,
                    format!("field `{field}` of struct `{name}` was removed"),
                ),
                Some(new_ty) if new_ty != old_ty => diff.push(
                    Bump::Major,
                    format!(
                        "field `{field}` of struct `{name}` changed from `{old_ty}` to `{new_ty}`"
                    ),
                ),
                Some(_) => {}
            }
        }
        // A struct literal must give every field, so a new field breaks the code that builds one.
        for field in new_fields.keys().filter(|f| !old_fields.contains_key(*f)) {
            diff.push(
                Bump::Major,
                format!("struct `{name}` has a new field `{field}`, which code that builds a `{name}` must now give"),
            );
        }
    }
    for name in new.keys().filter(|name| !old.contains_key(*name)) {
        diff.push(Bump::Minor, format!("struct `{name}` was added"));
    }
}

fn compare_enums(
    diff: &mut ApiDiff,
    old: &BTreeMap<String, &Vec<Symbol>>,
    new: &BTreeMap<String, &Vec<Symbol>>,
) {
    for (name, old_variants) in old {
        let Some(new_variants) = new.get(name) else {
            diff.push(Bump::Major, format!("enum `{name}` was removed"));
            continue;
        };
        for variant in old_variants.iter().filter(|v| !new_variants.contains(v)) {
            diff.push(
                Bump::Major,
                format!("variant `{name}::{variant}` was removed"),
            );
        }
        // A `match` must cover every variant, so a new variant breaks a match that names each one.
        for variant in new_variants.iter().filter(|v| !old_variants.contains(v)) {
            diff.push(
                Bump::Major,
                format!("enum `{name}` has a new variant `{variant}`, which a `match` on `{name}` must now cover"),
            );
        }
        let kept = |a: &[Symbol], b: &[Symbol]| -> Vec<Symbol> {
            a.iter().filter(|v| b.contains(v)).cloned().collect()
        };
        if kept(old_variants, new_variants) != kept(new_variants, old_variants) {
            diff.push(
                Bump::Major,
                format!("the variants of enum `{name}` changed their order"),
            );
        }
    }
    for name in new.keys().filter(|name| !old.contains_key(*name)) {
        diff.push(Bump::Minor, format!("enum `{name}` was added"));
    }
}
