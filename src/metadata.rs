//===- metadata.rs - Vx Compiler -------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// This file handles the serialization and deserialization of compiler metadata.
// It defines the zero-copy structures used to encode type information, function
// signatures, and ABI details into compiled artifacts, enabling robust cross-module
// linking and FFI interoperability.
//
//===----------------------------------------------------------------------===//
use crate::gid::{deserialize_metadata_symbols, serialize_metadata_symbols, TypeId};
use crate::layout::{FieldLayout, FieldTy};
use crate::registry::{ImmutableGlobalRegistry, TypeDefinition};
use crate::syntax::ElementType;
use rustc_hash::FxHashMap;
use std::fs;
use std::io;
use std::io::Write;
use std::path::Path;

/// Represents the serialized metadata dictionary for a specific module/crate.
pub struct VxMetadata<'a> {
    /// The zero-copy slice of 256-bit TypeIds directly mapped from the file buffer
    pub type_dictionary: &'a [TypeId],
    /// The serialized **module interface** that trails the type dictionary: the registry-backed
    /// import oracle (`serialize_registry_interface`), not AST. A downstream compile resolves against
    /// this instead of re-parsing the module's source (stdlib decoupling, protocol doc §5b/§6, #220).
    /// Empty when only the dictionary was saved.
    pub interface_data: &'a [u8],
}

impl<'a> VxMetadata<'a> {
    /// Save the fully deduplicated Type Dictionary to disk (no interface section).
    pub fn save_to_file(type_dictionary: &[TypeId], path: &Path) -> io::Result<()> {
        let file = fs::File::create(path)?;
        let mut writer = io::BufWriter::new(file);

        // Zero-copy serialization via bytemuck
        serialize_metadata_symbols(type_dictionary, &mut writer)?;
        Ok(())
    }

    /// Save the type dictionary followed by a serialized module interface (`interface_bytes`, from
    /// [`serialize_registry_interface`]). `load_from_buffer` returns the interface bytes as
    /// `interface_data`, which [`deserialize_registry_interface`] turns back into a queryable registry.
    pub fn save_with_interface(
        type_dictionary: &[TypeId],
        interface_bytes: &[u8],
        path: &Path,
    ) -> io::Result<()> {
        let file = fs::File::create(path)?;
        let mut writer = io::BufWriter::new(file);
        serialize_metadata_symbols(type_dictionary, &mut writer)?;
        writer.write_all(interface_bytes)?;
        Ok(())
    }

    /// Load the Type Dictionary directly from disk using a Zero-Copy memory cast.
    /// This requires the caller to own the `[u8]` backing buffer, returning
    /// a struct containing slice references mapped over that buffer.
    pub fn load_from_buffer(buffer: &'a [u8]) -> Self {
        let (type_dictionary, interface_data) = deserialize_metadata_symbols(buffer);
        Self {
            type_dictionary,
            interface_data,
        }
    }
}

// ---- Module interface serialization (the `.vxlib` payload) ---------------------------------------
//
// A hand-rolled little-endian binary codec for the frozen registry's import-oracle surface. There is
// no serde in the tree (only `bytemuck` for the POD GID arrays), so the structured tables are encoded
// field by field. Keys are emitted in a deterministic order (sorted) so the artifact is reproducible.
//
// This stage covers the fully *closed* part of the interface -- `module_indices` (identity) and
// `layouts` (structural layout), i.e. the `resolve_type` / `layout_of` queries. `fn_sigs` / `methods`
// carry a `ret_ty: syntax::Type` (which embeds `Expr` dimension trees) and the per-function flat HIR
// bodies are added in later stages; both need their own encoders (#220).

/// Magic bytes identifying a serialized Vx module interface.
const VXLIB_MAGIC: &[u8; 4] = b"VXLB";
/// Format tag folded into an FNV-1a stamp (`src/hash.rs`) written after the magic. A codec change
/// bumps this string, so a stale artifact is *detected* (version mismatch on load) rather than misread.
const VXLIB_FORMAT_TAG: &str = "vxlib-interface-v1";

/// Append-only little-endian byte writer for the interface codec.
struct Writer {
    buf: Vec<u8>,
}

impl Writer {
    fn new() -> Self {
        Self { buf: Vec::new() }
    }
    fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }
    fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    /// Length-prefixed raw bytes.
    fn bytes(&mut self, b: &[u8]) {
        self.u64(b.len() as u64);
        self.buf.extend_from_slice(b);
    }
    fn sym(&mut self, s: &str) {
        self.bytes(s.as_bytes());
    }
    fn typeid(&mut self, t: &TypeId) {
        for w in t.words {
            self.u64(w);
        }
    }
}

/// Cursor over a byte buffer; every read is bounds-checked and returns `Err` on a short/invalid buffer.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.buf.len())
            .ok_or_else(|| format!("vxlib: unexpected end of buffer (need {n} at {})", self.pos))?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn bytes(&mut self) -> Result<&'a [u8], String> {
        let n = self.u64()? as usize;
        self.take(n)
    }
    fn sym(&mut self) -> Result<crate::symbol::Symbol, String> {
        let b = self.bytes()?;
        let s = std::str::from_utf8(b).map_err(|e| format!("vxlib: bad utf-8 symbol: {e}"))?;
        Ok(crate::symbol::Symbol::from(s))
    }
    fn typeid(&mut self) -> Result<TypeId, String> {
        let mut words = [0u64; 4];
        for w in &mut words {
            *w = self.u64()?;
        }
        Ok(TypeId { words })
    }
}

fn write_element_type(w: &mut Writer, e: &ElementType) {
    use ElementType::*;
    let tag: u8 = match e {
        F16 => 0,
        F32 => 1,
        F64 => 2,
        BF16 => 3,
        I4 => 4,
        U4 => 5,
        I8 => 6,
        U8 => 7,
        I16 => 8,
        U16 => 9,
        I32 => 10,
        U32 => 11,
        I64 => 12,
        U64 => 13,
        I128 => 14,
        U128 => 15,
        Bool => 16,
        Generic(_) => 17,
    };
    w.u8(tag);
    if let Generic(s) = e {
        w.sym(s);
    }
}

fn read_element_type(r: &mut Reader) -> Result<ElementType, String> {
    use ElementType::*;
    Ok(match r.u8()? {
        0 => F16,
        1 => F32,
        2 => F64,
        3 => BF16,
        4 => I4,
        5 => U4,
        6 => I8,
        7 => U8,
        8 => I16,
        9 => U16,
        10 => I32,
        11 => U32,
        12 => I64,
        13 => U64,
        14 => I128,
        15 => U128,
        16 => Bool,
        17 => Generic(r.sym()?),
        t => return Err(format!("vxlib: bad ElementType tag {t}")),
    })
}

fn write_field_ty(w: &mut Writer, ty: &FieldTy) {
    match ty {
        FieldTy::Scalar(e) => {
            w.u8(0);
            write_element_type(w, e);
        }
        FieldTy::Nominal(id) => {
            w.u8(1);
            w.typeid(id);
        }
        FieldTy::Opaque => w.u8(2),
    }
}

fn read_field_ty(r: &mut Reader) -> Result<FieldTy, String> {
    Ok(match r.u8()? {
        0 => FieldTy::Scalar(read_element_type(r)?),
        1 => FieldTy::Nominal(r.typeid()?),
        2 => FieldTy::Opaque,
        t => return Err(format!("vxlib: bad FieldTy tag {t}")),
    })
}

fn write_type_definition(w: &mut Writer, def: &TypeDefinition) {
    w.typeid(&def.id);
    w.sym(&def.name);
    w.u64(def.size_bytes as u64);
    w.u64(def.align_bytes as u64);
    w.u64(def.fields.len() as u64);
    for f in &def.fields {
        w.sym(&f.name);
        w.u64(f.offset as u64);
        w.u64(f.size as u64);
        write_field_ty(w, &f.ty);
    }
    w.u64(def.by_value_dependencies.len() as u64);
    for d in &def.by_value_dependencies {
        w.typeid(d);
    }
}

fn read_type_definition(r: &mut Reader) -> Result<TypeDefinition, String> {
    let id = r.typeid()?;
    let name = r.sym()?.to_string();
    let size_bytes = r.u64()? as usize;
    let align_bytes = r.u64()? as usize;
    let n_fields = r.u64()? as usize;
    let mut fields = Vec::with_capacity(n_fields);
    for _ in 0..n_fields {
        let fname = r.sym()?;
        let offset = r.u64()? as usize;
        let size = r.u64()? as usize;
        let ty = read_field_ty(r)?;
        fields.push(FieldLayout {
            name: fname,
            offset,
            size,
            ty,
        });
    }
    let n_deps = r.u64()? as usize;
    let mut by_value_dependencies = Vec::with_capacity(n_deps);
    for _ in 0..n_deps {
        by_value_dependencies.push(r.typeid()?);
    }
    Ok(TypeDefinition {
        id,
        name,
        size_bytes,
        align_bytes,
        fields,
        by_value_dependencies,
    })
}

/// Serialize the frozen registry's import-oracle interface to a versioned byte buffer. Covers the
/// identity (`module_indices`) and structural-layout (`layouts`) tables -- the `resolve_type` /
/// `layout_of` surface. Keys are sorted so the output is byte-reproducible for the same registry.
pub fn serialize_registry_interface(reg: &ImmutableGlobalRegistry) -> Vec<u8> {
    let mut w = Writer::new();
    w.buf.extend_from_slice(VXLIB_MAGIC);
    w.u64(crate::hash::compute_module_hash(VXLIB_FORMAT_TAG));

    // module_indices: sorted by module hash, then by symbol name, for a deterministic artifact.
    let mut modules: Vec<(&u64, &FxHashMap<crate::symbol::Symbol, TypeId>)> =
        reg.module_indices.iter().collect();
    modules.sort_by_key(|(h, _)| **h);
    w.u64(modules.len() as u64);
    for (mod_hash, by_name) in modules {
        w.u64(*mod_hash);
        let mut entries: Vec<(&crate::symbol::Symbol, &TypeId)> = by_name.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));
        w.u64(entries.len() as u64);
        for (name, gid) in entries {
            w.sym(name);
            w.typeid(gid);
        }
    }

    // layouts: sorted by GID words.
    let mut defs: Vec<&TypeDefinition> = reg.layouts.values().collect();
    defs.sort_by_key(|d| d.id.words);
    w.u64(defs.len() as u64);
    for def in defs {
        write_type_definition(&mut w, def);
    }

    w.buf
}

/// Rebuild a queryable [`ImmutableGlobalRegistry`] from bytes produced by
/// [`serialize_registry_interface`]. `fn_sigs` / `methods` are left empty until their encoders land
/// (#220). Returns `Err` on a bad magic, a format-version mismatch (stale artifact), or a truncated
/// buffer -- never a silent misread.
pub fn deserialize_registry_interface(bytes: &[u8]) -> Result<ImmutableGlobalRegistry, String> {
    let mut r = Reader::new(bytes);
    if r.take(4)? != VXLIB_MAGIC {
        return Err("vxlib: bad magic (not a Vx module interface)".to_string());
    }
    let stamp = r.u64()?;
    if stamp != crate::hash::compute_module_hash(VXLIB_FORMAT_TAG) {
        return Err("vxlib: format version mismatch (stale or incompatible artifact)".to_string());
    }

    let mut module_indices: FxHashMap<u64, FxHashMap<crate::symbol::Symbol, TypeId>> =
        FxHashMap::default();
    let n_modules = r.u64()?;
    for _ in 0..n_modules {
        let mod_hash = r.u64()?;
        let n_entries = r.u64()?;
        let mut by_name = FxHashMap::default();
        for _ in 0..n_entries {
            let name = r.sym()?;
            let gid = r.typeid()?;
            by_name.insert(name, gid);
        }
        module_indices.insert(mod_hash, by_name);
    }

    let mut layouts: FxHashMap<TypeId, TypeDefinition> = FxHashMap::default();
    let n_defs = r.u64()?;
    for _ in 0..n_defs {
        let def = read_type_definition(&mut r)?;
        layouts.insert(def.id, def);
    }

    Ok(ImmutableGlobalRegistry {
        layouts,
        module_indices,
        fn_sigs: FxHashMap::default(),
        methods: FxHashMap::default(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::ModuleInterface;
    use crate::symbol::Symbol;

    fn parse_and_resolve(path: &str, src: &str) -> crate::syntax::VxModule {
        let mut lexer = crate::lexer::Lexer::new(src);
        let tokens = lexer.tokenize();
        let mut parser = crate::parser::Parser::new(&tokens, src);
        let mut prog = parser.parse().expect("parse failed");
        prog.module_path = path.into();
        let mut mods = vec![prog];
        let symbol_map = crate::resolver::build_symbol_map(&mods);
        mods[0].resolve_names(&symbol_map);
        mods.pop().unwrap()
    }

    /// The registry interface survives a serialize -> file -> load -> deserialize round trip: the
    /// rebuilt registry answers `resolve_type` / `layout_of` identically to the original (#220,
    /// acceptance: "serialize a registry, deserialize, assert resolve_* returns identical results").
    #[test]
    fn registry_interface_round_trips_through_a_file() {
        let m = parse_and_resolve(
            "crate::m",
            "struct Pair { a: i8, b: i32 }\n\
             struct Wrap { flag: i8, inner: Pair }\n\
             enum Color { Red, Green, Blue }",
        );
        let reg =
            crate::pipeline::build_frozen_registry(std::slice::from_ref(&m)).expect("acyclic");

        // Serialize is deterministic (sorted keys) -- the same registry yields byte-identical output.
        let bytes = serialize_registry_interface(&reg);
        assert_eq!(
            bytes,
            serialize_registry_interface(&reg),
            "serialization is stable"
        );

        // Round-trip through the VxMetadata container on disk (dictionary + interface section).
        let dict: Vec<TypeId> = reg.layouts.keys().copied().collect();
        let path = std::env::temp_dir().join(format!("vx_iface_{}.vxlib", std::process::id()));
        VxMetadata::save_with_interface(&dict, &bytes, &path).expect("save");
        let buffer = fs::read(&path).expect("read");
        let loaded = VxMetadata::load_from_buffer(&buffer);
        assert_eq!(loaded.interface_data, bytes.as_slice());
        let round = deserialize_registry_interface(loaded.interface_data).expect("deserialize");
        let _ = fs::remove_file(&path);

        // resolve_type + layout_of parity for every nominal the module defines.
        let hash = crate::hash::compute_module_hash("crate::m");
        for name in ["Pair", "Wrap", "Color"] {
            let sym = Symbol::from(name);
            let orig_gid = reg.resolve_type(hash, &sym).expect("original resolves");
            let round_gid = round.resolve_type(hash, &sym).expect("round-trip resolves");
            assert_eq!(orig_gid, round_gid, "GID parity for {name}");

            let o = reg.layout_of(orig_gid).unwrap();
            let r = round.layout_of(round_gid).unwrap();
            assert_eq!(o.name, r.name);
            assert_eq!((o.size_bytes, o.align_bytes), (r.size_bytes, r.align_bytes));
            assert_eq!(o.fields.len(), r.fields.len(), "field count for {name}");
            for (of, rf) in o.fields.iter().zip(r.fields.iter()) {
                assert_eq!(of.name, rf.name);
                assert_eq!((of.offset, of.size), (rf.offset, rf.size));
                assert_eq!(of.ty, rf.ty, "field type for {}.{}", name, of.name);
            }
            assert_eq!(o.by_value_dependencies, r.by_value_dependencies);
        }
        // The whole table came back.
        assert_eq!(reg.layouts.len(), round.layouts.len());
    }

    /// A bad magic, a stale format stamp, and a truncated buffer are all *detected* -- never misread.
    #[test]
    fn deserialize_rejects_corrupt_or_stale_buffers() {
        assert!(deserialize_registry_interface(b"not a vxlib").is_err());

        let m = parse_and_resolve("crate::m", "struct S { x: i32 }");
        let reg =
            crate::pipeline::build_frozen_registry(std::slice::from_ref(&m)).expect("acyclic");
        let good = serialize_registry_interface(&reg);

        // Corrupt the format stamp (bytes 4..12): version mismatch, not a misread.
        let mut stale = good.clone();
        stale[4] ^= 0xFF;
        assert!(deserialize_registry_interface(&stale).is_err());

        // Truncated mid-stream: a bounds-checked read fails rather than panicking.
        assert!(deserialize_registry_interface(&good[..good.len() - 4]).is_err());
    }
}
