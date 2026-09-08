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
use crate::bytecode::{HirInstruction, Opcode, Register, TypeIdx};
use crate::gid::{deserialize_metadata_symbols, serialize_metadata_symbols, TypeId};
use crate::layout::{FieldLayout, FieldTy};
use crate::registry::{FnBody, FnSig, ImmutableGlobalRegistry, StructFields, TypeDefinition};
use crate::symbol::Symbol;
use crate::syntax::{Dim, ElementType, Expr, MemorySpace, Placement, Topology, Type};
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
// Covers `module_indices` (identity), `layouts` (structural layout), `fn_sigs` / `methods`
// (signatures), and `bodies` (flat-HIR function bodies) -- the `resolve_type` / `layout_of` /
// `resolve_fn` / `resolve_method` / `body_of` queries. A signature's `ret_ty` is a recursive
// `syntax::Type`; its closed variants round-trip faithfully, but the `Expr`-bearing paths (symbolic
// tensor dimensions, `Const`, a topology carrying a count) are not yet encodable -- a signature whose
// return type reaches one is *skipped* (fail closed: `resolve_*` then declines, the same policy the
// registry uses for ambiguous names), never misencoded. A body whose type stream still holds a
// per-compilation *deferred* GID (a generic instantiation) is likewise skipped -- only fully-global
// (non-generic) bodies are portable across a compile boundary.

/// Magic bytes identifying a serialized Vx module interface.
const VXLIB_MAGIC: &[u8; 4] = b"VXLB";
/// Format tag folded into an FNV-1a stamp (`src/hash.rs`) written after the magic. A codec change
/// bumps this string, so a stale artifact is *detected* (version mismatch on load) rather than misread.
const VXLIB_FORMAT_TAG: &str = "vxlib-interface-v12";

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
        // Appended after Generic: tags are stable on disk, never renumber.
        F8E4M3 => 18,
        F8E5M2 => 19,
        F4E2M1 => 20,
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
        18 => F8E4M3,
        19 => F8E5M2,
        20 => F4E2M1,
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
        FieldTy::Tensor(e, rank) => {
            w.u8(3);
            write_element_type(w, e);
            w.u64(*rank as u64);
        }
        FieldTy::Vector(e, lanes) => {
            w.u8(4);
            write_element_type(w, e);
            w.u64(*lanes as u64);
        }
    }
}

fn read_field_ty(r: &mut Reader) -> Result<FieldTy, String> {
    Ok(match r.u8()? {
        0 => FieldTy::Scalar(read_element_type(r)?),
        1 => FieldTy::Nominal(r.typeid()?),
        2 => FieldTy::Opaque,
        3 => {
            let e = read_element_type(r)?;
            FieldTy::Tensor(e, r.u64()? as usize)
        }
        4 => {
            let e = read_element_type(r)?;
            FieldTy::Vector(e, r.u64()? as usize)
        }
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

fn write_opt_typeid(w: &mut Writer, id: &Option<TypeId>) {
    match id {
        None => w.u8(0),
        Some(t) => {
            w.u8(1);
            w.typeid(t);
        }
    }
}

fn read_opt_typeid(r: &mut Reader) -> Result<Option<TypeId>, String> {
    Ok(match r.u8()? {
        0 => None,
        1 => Some(r.typeid()?),
        t => return Err(format!("vxlib: bad Option<TypeId> tag {t}")),
    })
}

fn write_memory_space(w: &mut Writer, m: &MemorySpace) {
    use MemorySpace::*;
    match m {
        CPUDRAM => w.u8(0),
        NPUHBM => w.u8(1),
        GpuHbm => w.u8(2),
        LocalSRAM => w.u8(3),
        NicRam => w.u8(4),
        RemoteHbm => w.u8(5),
        Custom(s) => {
            w.u8(6);
            w.sym(s);
        }
    }
}

fn read_memory_space(r: &mut Reader) -> Result<MemorySpace, String> {
    use MemorySpace::*;
    Ok(match r.u8()? {
        0 => CPUDRAM,
        1 => NPUHBM,
        2 => GpuHbm,
        3 => LocalSRAM,
        4 => NicRam,
        5 => RemoteHbm,
        6 => Custom(r.sym()?),
        t => return Err(format!("vxlib: bad MemorySpace tag {t}")),
    })
}

fn write_opt_memory_space(w: &mut Writer, o: &Option<MemorySpace>) {
    match o {
        None => w.u8(0),
        Some(m) => {
            w.u8(1);
            write_memory_space(w, m);
        }
    }
}

fn read_opt_memory_space(r: &mut Reader) -> Result<Option<MemorySpace>, String> {
    Ok(match r.u8()? {
        0 => None,
        1 => Some(read_memory_space(r)?),
        t => return Err(format!("vxlib: bad Option<MemorySpace> tag {t}")),
    })
}

/// Only the data-free topology variants (plus the named `Custom` and an indexed `GPU`) are
/// encodable so far; the ones carrying a dimension `Expr` (`NPU`/`AccCore`/`Slice`) return `Err`,
/// which fails the whole enclosing type closed rather than dropping the count.
///
/// `GPU` gained a device index, so tag 3 now carries one. A literal index is written; anything
/// else joins the `Err` group, since a device chosen at run time is not an interface fact. The
/// format tag is bumped alongside, so an artifact written before this is detected as stale rather
/// than read as `GPU[0]` followed by whatever came next.
fn write_topology(w: &mut Writer, t: &Topology) -> Result<(), String> {
    use Topology::*;
    match t {
        CPU => w.u8(0),
        AMX => w.u8(1),
        ANE => w.u8(2),
        GPU(e) => {
            let Expr::Number(n) = &**e else {
                return Err("vxlib: GPU device index is not a literal".into());
            };
            let idx: i64 = n
                .value
                .parse()
                .map_err(|_| "vxlib: GPU device index is not an integer".to_string())?;
            w.u8(3);
            w.u64(idx as u64);
        }
        CpuAvx512 => w.u8(4),
        CpuNeon => w.u8(5),
        Current => w.u8(6),
        Custom(s) => {
            w.u8(7);
            w.sym(s);
        }
        NPU(_) | AccCore(_) | Slice(..) => {
            return Err(
                "vxlib: topology carrying a dimension expression not yet serializable".into(),
            )
        }
    }
    Ok(())
}

fn read_topology(r: &mut Reader) -> Result<Topology, String> {
    use Topology::*;
    Ok(match r.u8()? {
        0 => CPU,
        1 => AMX,
        2 => ANE,
        3 => Topology::gpu(r.u64()? as i64),
        4 => CpuAvx512,
        5 => CpuNeon,
        6 => Current,
        7 => Custom(r.sym()?),
        t => return Err(format!("vxlib: bad Topology tag {t}")),
    })
}

fn write_opt_placement(w: &mut Writer, o: &Option<Placement>) -> Result<(), String> {
    match o {
        None => w.u8(0),
        Some(p) => {
            w.u8(1);
            write_topology(w, &p.topology)?;
            write_memory_space(w, &p.space);
        }
    }
    Ok(())
}

fn read_opt_placement(r: &mut Reader) -> Result<Option<Placement>, String> {
    Ok(match r.u8()? {
        0 => None,
        1 => {
            // Written topology-first; the space follows, so a declared space survives the
            // round trip rather than being re-derived from the device.
            let topology = read_topology(r)?;
            let space = read_memory_space(r)?;
            Some(Placement::in_space(space, topology))
        }
        t => return Err(format!("vxlib: bad Option<Placement> tag {t}")),
    })
}

/// Encode a `syntax::Type`. The closed variants round-trip faithfully; the `Expr`-bearing paths
/// (a tensor with symbolic dimensions, `Const`, `Module`) return `Err` so the caller can skip the
/// enclosing signature rather than write a lossy type. See the module header.
fn write_type(w: &mut Writer, ty: &Type) -> Result<(), String> {
    use Type::*;
    match ty {
        Scalar(e) => {
            w.u8(0);
            write_element_type(w, e);
        }
        Struct(name, id) => {
            w.u8(1);
            w.sym(name);
            write_opt_typeid(w, id);
        }
        Enum(name, id) => {
            w.u8(2);
            w.sym(name);
            write_opt_typeid(w, id);
        }
        Generic(name, id) => {
            w.u8(3);
            w.sym(name);
            write_opt_typeid(w, id);
        }
        Ref(inner, mem) => {
            w.u8(4);
            write_type(w, inner)?;
            write_memory_space(w, mem);
        }
        Pointer(inner, mem, is_mut) => {
            w.u8(5);
            write_type(w, inner)?;
            write_opt_memory_space(w, mem);
            w.u8(*is_mut as u8);
        }
        Borrow {
            inner,
            mem_space,
            is_mut,
            region_id,
        } => {
            w.u8(6);
            write_type(w, inner)?;
            write_opt_memory_space(w, mem_space);
            w.u8(*is_mut as u8);
            w.u64(*region_id as u64);
        }
        Pinned(inner, top) => {
            w.u8(7);
            write_type(w, inner)?;
            write_topology(w, top)?;
        }
        Verified(inner) => {
            w.u8(8);
            write_type(w, inner)?;
        }
        GenericInstance(base, args) => {
            w.u8(9);
            write_type(w, base)?;
            w.u64(args.len() as u64);
            for a in args {
                write_type(w, a)?;
            }
        }
        Function(params, ret, unsafe_fn) => {
            w.u8(10);
            w.u8(u8::from(*unsafe_fn));
            w.u64(params.len() as u64);
            for p in params {
                write_type(w, p)?;
            }
            write_type(w, ret)?;
        }
        Closure(params, ret) => {
            w.u8(11);
            w.u64(params.len() as u64);
            for p in params {
                write_type(w, p)?;
            }
            write_type(w, ret)?;
        }
        Simd(e, n) => {
            w.u8(12);
            write_element_type(w, e);
            w.u64(*n as u64);
        }
        Matrix => w.u8(13),
        Unknown => w.u8(14),
        // A dimension is `?` or a literal. A const-generic name or an arithmetic dimension
        // belongs to a template, and a template is not an interface.
        Tensor(e, dims, top) => {
            w.u8(15);
            write_element_type(w, e);
            write_opt_placement(w, top)?;
            w.u64(dims.len() as u64);
            for d in dims {
                match d {
                    Dim::Dyn => w.u8(0),
                    Dim::Static(_) => {
                        let Some(v) = d.literal() else {
                            return Err("vxlib: tensor type with dimension expressions not \
                                        serializable"
                                .into());
                        };
                        w.u8(1);
                        w.sym(v);
                    }
                }
            }
        }
        Const(_) => return Err("vxlib: const-expression type not yet serializable".into()),
        Module(_, _) => return Err("vxlib: module type not serializable".into()),
    }
    Ok(())
}

fn read_type(r: &mut Reader) -> Result<Type, String> {
    use Type::*;
    Ok(match r.u8()? {
        0 => Scalar(read_element_type(r)?),
        1 => Struct(r.sym()?, read_opt_typeid(r)?),
        2 => Enum(r.sym()?, read_opt_typeid(r)?),
        3 => Generic(r.sym()?, read_opt_typeid(r)?),
        4 => {
            let inner = Box::new(read_type(r)?);
            Ref(inner, read_memory_space(r)?)
        }
        5 => {
            let inner = Box::new(read_type(r)?);
            let mem = read_opt_memory_space(r)?;
            let is_mut = r.u8()? != 0;
            Pointer(inner, mem, is_mut)
        }
        6 => {
            let inner = Box::new(read_type(r)?);
            let mem_space = read_opt_memory_space(r)?;
            let is_mut = r.u8()? != 0;
            let region_id = r.u64()? as usize;
            Borrow {
                inner,
                mem_space,
                is_mut,
                region_id,
            }
        }
        7 => {
            let inner = Box::new(read_type(r)?);
            Pinned(inner, read_topology(r)?)
        }
        8 => Verified(Box::new(read_type(r)?)),
        9 => {
            let base = Box::new(read_type(r)?);
            let n = r.u64()? as usize;
            let mut args = Vec::with_capacity(n);
            for _ in 0..n {
                args.push(read_type(r)?);
            }
            GenericInstance(base, args)
        }
        10 => {
            let unsafe_fn = r.u8()? != 0;
            let n = r.u64()? as usize;
            let mut params = Vec::with_capacity(n);
            for _ in 0..n {
                params.push(read_type(r)?);
            }
            let ret = Box::new(read_type(r)?);
            Function(params, ret, unsafe_fn)
        }
        11 => {
            let n = r.u64()? as usize;
            let mut params = Vec::with_capacity(n);
            for _ in 0..n {
                params.push(read_type(r)?);
            }
            let ret = Box::new(read_type(r)?);
            Closure(params, ret)
        }
        12 => Simd(read_element_type(r)?, r.u64()? as usize),
        13 => Matrix,
        14 => Unknown,
        15 => {
            let e = read_element_type(r)?;
            let top = read_opt_placement(r)?;
            let n = r.u64()? as usize;
            let mut dims = Vec::with_capacity(n);
            for _ in 0..n {
                dims.push(match r.u8()? {
                    0 => Dim::Dyn,
                    1 => Dim::Static(crate::syntax::Expr::Number(crate::syntax::NumberExpr::new(
                        r.sym()?.as_ref().to_string(),
                        Some(ElementType::I32),
                        crate::syntax::Span::default(),
                    ))),
                    t => return Err(format!("vxlib: bad Dim tag {t}")),
                });
            }
            Tensor(e, dims, top)
        }
        t => return Err(format!("vxlib: bad Type tag {t}")),
    })
}

fn write_hir_instruction(w: &mut Writer, ins: &HirInstruction) {
    w.u64(ins.opcode as u32 as u64);
    w.u64(ins.operand1.0 as u64);
    w.u64(ins.operand2.0 as u64);
    w.u64(ins.type_idx.0 as u64);
    w.u64(ins.imm);
}

fn read_hir_instruction(r: &mut Reader) -> Result<HirInstruction, String> {
    let opcode_raw = r.u64()?;
    let opcode = u32::try_from(opcode_raw)
        .ok()
        .and_then(Opcode::from_u32)
        .ok_or_else(|| format!("vxlib: bad opcode discriminant {opcode_raw}"))?;
    let operand1 = Register(r.u64()? as u32);
    let operand2 = Register(r.u64()? as u32);
    let type_idx = TypeIdx(r.u64()? as u32);
    let imm = r.u64()?;
    Ok(HirInstruction {
        opcode,
        operand1,
        operand2,
        type_idx,
        imm,
    })
}

/// Encode a function body. Returns `Err` if the signature or any type-stream GID isn't portable
/// (an unencodable return/param type, or a per-compilation deferred generic GID) -- the caller then
/// skips it, so `body_of` declines rather than linking a body it can't resolve.
fn write_fn_body(w: &mut Writer, gid: &TypeId, body: &FnBody) -> Result<(), String> {
    if body.types.iter().any(|t| t.is_local_deferred()) {
        return Err("vxlib: body carries a deferred (generic-instantiation) GID".into());
    }
    let mut sig = Writer::new();
    write_type(&mut sig, &body.ret_ty)?;
    let mut params = Writer::new();
    for p in &body.params {
        write_type(&mut params, p)?;
    }
    // Signature encoded cleanly -- now commit the whole record.
    w.typeid(gid);
    w.sym(&body.name);
    w.u64(body.params.len() as u64);
    w.buf.extend_from_slice(&params.buf);
    w.buf.extend_from_slice(&sig.buf);
    w.u64(body.hir.len() as u64);
    for ins in &body.hir {
        write_hir_instruction(w, ins);
    }
    w.u64(body.types.len() as u64);
    for t in &body.types {
        w.typeid(t);
    }
    Ok(())
}

fn read_fn_body(r: &mut Reader) -> Result<(TypeId, FnBody), String> {
    let gid = r.typeid()?;
    let name = r.sym()?;
    let n_params = r.u64()? as usize;
    let mut params = Vec::with_capacity(n_params);
    for _ in 0..n_params {
        params.push(read_type(r)?);
    }
    let ret_ty = read_type(r)?;
    let n_hir = r.u64()? as usize;
    let mut hir = Vec::with_capacity(n_hir);
    for _ in 0..n_hir {
        hir.push(read_hir_instruction(r)?);
    }
    let n_types = r.u64()? as usize;
    let mut types = Vec::with_capacity(n_types);
    for _ in 0..n_types {
        types.push(r.typeid()?);
    }
    Ok((
        gid,
        FnBody {
            name,
            params,
            ret_ty,
            hir,
            types,
        },
    ))
}

/// Encode a `FnSig`'s payload (everything after its name/gid/receiver): the param count, each param
/// type, the return type, then the 1-byte return-provenance code (#265 step 7). `Err` (with the codec's
/// reason) if any param or the return type is not encodable yet, so the caller skips the whole
/// signature — fail-closed, matching the body codec — and can report *why* it was dropped (#292).
fn encode_sig_record(sig: &FnSig) -> Result<Vec<u8>, String> {
    let mut rec = Writer::new();
    rec.u64(sig.params.len() as u64);
    for p in &sig.params {
        write_type(&mut rec, p)?;
    }
    write_type(&mut rec, &sig.ret_ty)?;
    rec.u8(sig.ret_prov);
    rec.u8(u8::from(sig.is_unsafe));
    Ok(rec.buf)
}

/// Decode the payload written by [`encode_sig_record`] — params, return type, provenance code. The
/// caller reads the name/gid/receiver that precede it.
fn read_sig_record(
    r: &mut Reader,
) -> Result<(Vec<crate::syntax::Type>, crate::syntax::Type, u8, bool), String> {
    let n_params = r.u64()? as usize;
    let mut params = Vec::new();
    for _ in 0..n_params {
        params.push(read_type(r)?);
    }
    let ret_ty = read_type(r)?;
    let ret_prov = r.u8()?;
    let is_unsafe = r.u8()? != 0;
    Ok((params, ret_ty, ret_prov, is_unsafe))
}

/// One table's encode/skip accounting for a `serialize_registry_interface` run (#292).
#[derive(Debug)]
pub struct TableReport {
    pub table: &'static str,
    pub encoded: usize,
    /// Each fail-closed skip: the entry's display name and the codec error that excluded it.
    pub skipped: Vec<(String, String)>,
}

/// What one interface emit encoded vs skipped, per skipping-capable table (#292). The skips are
/// fail-closed and correct — a half-encoded signature would be worse — but they used to be
/// *silent*: the producer printed only a byte count, and the omission surfaced later, in a
/// different compilation, as an unrelated-looking "not found". `Display` renders the summary the
/// driver prints under the "Wrote module interface" line.
#[derive(Debug)]
pub struct InterfaceEmitReport {
    pub tables: Vec<TableReport>,
}

impl InterfaceEmitReport {
    pub fn total_skipped(&self) -> usize {
        self.tables.iter().map(|t| t.skipped.len()).sum()
    }
}

impl std::fmt::Display for InterfaceEmitReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let counts = self
            .tables
            .iter()
            .map(|t| format!("{} {}/{}", t.table, t.encoded, t.encoded + t.skipped.len()))
            .collect::<Vec<_>>()
            .join("  ");
        write!(f, "  {}", counts)?;
        match self.total_skipped() {
            0 => write!(f, "  (nothing skipped)")?,
            n => write!(
                f,
                "  ({} entr{} skipped)",
                n,
                if n == 1 { "y" } else { "ies" }
            )?,
        }
        for t in &self.tables {
            for (entry, reason) in &t.skipped {
                write!(f, "\n  skipped {} '{}': {}", t.table, entry, reason)?;
            }
        }
        Ok(())
    }
}

/// Serialize the frozen registry's import-oracle interface to a versioned byte buffer. Covers the
/// identity (`module_indices`), structural-layout (`layouts`), signature (`fn_sigs` / `methods`), and
/// flat-HIR-body (`bodies`) tables. Keys are sorted so the output is byte-reproducible for the same
/// registry.
pub fn serialize_registry_interface(reg: &ImmutableGlobalRegistry) -> Vec<u8> {
    serialize_registry_interface_reporting(reg).0
}

/// [`serialize_registry_interface`] plus the per-table encoded/skipped accounting, so the emit path
/// can surface an incomplete artifact at produce time instead of letting it fail later, elsewhere,
/// as a "not found" (#292).
pub fn serialize_registry_interface_reporting(
    reg: &ImmutableGlobalRegistry,
) -> (Vec<u8>, InterfaceEmitReport) {
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

    // fn_sigs: sorted by name. A signature whose return type isn't encodable yet is skipped, so the
    // count is written after the entries are built (see the module header).
    let mut fns: Vec<(&Symbol, &FnSig)> = reg.fn_sigs.iter().collect();
    fns.sort_by(|a, b| a.0.cmp(b.0));
    let mut sub = Writer::new();
    let mut n = 0u64;
    let mut fn_report = TableReport {
        table: "fn_sigs",
        encoded: 0,
        skipped: Vec::new(),
    };
    for (name, sig) in fns {
        // Encode params + return type + provenance code into a record; include the entry only if
        // every type is encodable (an unencodable param/return skips the whole signature).
        match encode_sig_record(sig) {
            Ok(rec) => {
                sub.sym(name);
                sub.typeid(&sig.gid);
                sub.buf.extend_from_slice(&rec);
                n += 1;
                fn_report.encoded += 1;
            }
            Err(reason) => fn_report.skipped.push((name.to_string(), reason)),
        }
    }
    w.u64(n);
    w.buf.extend_from_slice(&sub.buf);

    // methods: sorted by (receiver GID, method name). Same skip-on-unencodable-return policy.
    let mut meths: Vec<(&(TypeId, Symbol), &FnSig)> = reg.methods.iter().collect();
    meths.sort_by(|a, b| a.0 .0.words.cmp(&b.0 .0.words).then(a.0 .1.cmp(&b.0 .1)));
    let mut sub = Writer::new();
    let mut n = 0u64;
    let mut meth_report = TableReport {
        table: "methods",
        encoded: 0,
        skipped: Vec::new(),
    };
    for ((recv, name), sig) in meths {
        match encode_sig_record(sig) {
            Ok(rec) => {
                sub.typeid(recv);
                sub.sym(name);
                sub.typeid(&sig.gid);
                sub.buf.extend_from_slice(&rec);
                n += 1;
                meth_report.encoded += 1;
            }
            Err(reason) => meth_report
                .skipped
                .push((format!("{}.{}", nominal_name(reg, recv), name), reason)),
        }
    }
    w.u64(n);
    w.buf.extend_from_slice(&sub.buf);

    // bodies: sorted by GID words. A non-portable body (unencodable signature or a deferred generic
    // GID) is skipped, so the count is written after the encodable entries are built.
    let mut bodies: Vec<(&TypeId, &FnBody)> = reg.bodies.iter().collect();
    bodies.sort_by_key(|(gid, _)| gid.words);
    let mut sub = Writer::new();
    let mut n = 0u64;
    let mut body_report = TableReport {
        table: "bodies",
        encoded: 0,
        skipped: Vec::new(),
    };
    for (gid, body) in bodies {
        let mut one = Writer::new();
        match write_fn_body(&mut one, gid, body) {
            Ok(()) => {
                sub.buf.extend_from_slice(&one.buf);
                n += 1;
                body_report.encoded += 1;
            }
            Err(reason) => body_report.skipped.push((body.name.to_string(), reason)),
        }
    }
    w.u64(n);
    w.buf.extend_from_slice(&sub.buf);

    // structs: each struct's declared (un-erased) field AST `Type`s + generic parameters (#219) — the
    // `StructFields` a downstream compile needs to type member access on an imported struct without
    // its AST. Keyed by the struct's GID (#291), sorted by GID words; a struct with an un-encodable
    // field type is skipped (fail-closed).
    let mut structs: Vec<(&TypeId, &StructFields)> = reg.structs.iter().collect();
    structs.sort_by_key(|(g, _)| g.words);
    let mut sub = Writer::new();
    let mut n = 0u64;
    let mut struct_report = TableReport {
        table: "structs",
        encoded: 0,
        skipped: Vec::new(),
    };
    for (gid, sf) in structs {
        let mut rec = Writer::new();
        rec.u64(sf.generics.len() as u64);
        for g in &sf.generics {
            rec.sym(g);
        }
        rec.u64(sf.fields.len() as u64);
        let mut failed: Option<(String, String)> = None;
        for (fname, fty) in &sf.fields {
            rec.sym(fname);
            if let Err(reason) = write_type(&mut rec, fty) {
                failed = Some((
                    format!("{} (field '{}')", nominal_name(reg, gid), fname),
                    reason,
                ));
                break;
            }
        }
        match failed {
            None => {
                sub.typeid(gid);
                sub.buf.extend_from_slice(&rec.buf);
                n += 1;
                struct_report.encoded += 1;
            }
            Some(entry) => struct_report.skipped.push(entry),
        }
    }
    w.u64(n);
    w.buf.extend_from_slice(&sub.buf);

    let report = InterfaceEmitReport {
        tables: vec![fn_report, meth_report, body_report, struct_report],
    };
    (w.buf, report)
}

/// A nominal type's display name for skip reporting: the layout's name when the registry has one,
/// else the GID's debug form (better an opaque-but-unique identity than nothing).
fn nominal_name(reg: &ImmutableGlobalRegistry, gid: &TypeId) -> String {
    reg.layouts
        .get(gid)
        .map(|d| d.name.clone())
        .unwrap_or_else(|| format!("{:?}", gid.words))
}

/// Rebuild a queryable [`ImmutableGlobalRegistry`] from bytes produced by
/// [`serialize_registry_interface`]. Each `fn_sigs` / `methods` entry carries its params, return type,
/// and return-provenance code (#265 step 7). Returns `Err` on a bad magic, a format-version mismatch
/// (stale artifact), or a truncated buffer -- never a silent misread.
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

    let mut fn_sigs: FxHashMap<Symbol, FnSig> = FxHashMap::default();
    let n_fns = r.u64()?;
    for _ in 0..n_fns {
        let name = r.sym()?;
        let gid = r.typeid()?;
        let (params, ret_ty, ret_prov, is_unsafe) = read_sig_record(&mut r)?;
        fn_sigs.insert(
            name,
            FnSig {
                gid,
                params,
                ret_ty,
                ret_prov,
                is_unsafe,
            },
        );
    }

    let mut methods: FxHashMap<(TypeId, Symbol), FnSig> = FxHashMap::default();
    let n_meths = r.u64()?;
    for _ in 0..n_meths {
        let recv = r.typeid()?;
        let name = r.sym()?;
        let gid = r.typeid()?;
        let (params, ret_ty, ret_prov, is_unsafe) = read_sig_record(&mut r)?;
        methods.insert(
            (recv, name),
            FnSig {
                gid,
                params,
                ret_ty,
                ret_prov,
                is_unsafe,
            },
        );
    }

    let mut bodies: FxHashMap<TypeId, FnBody> = FxHashMap::default();
    let n_bodies = r.u64()?;
    for _ in 0..n_bodies {
        let (gid, body) = read_fn_body(&mut r)?;
        bodies.insert(gid, body);
    }

    // structs: declared field AST types + generics, GID-keyed (#291), so imported struct member
    // access types the same as a local struct (#219). Mirrors the serializer's per-struct record.
    let mut structs: FxHashMap<TypeId, StructFields> = FxHashMap::default();
    let n_structs = r.u64()?;
    for _ in 0..n_structs {
        let gid = r.typeid()?;
        let n_generics = r.u64()? as usize;
        let mut generics = Vec::new();
        for _ in 0..n_generics {
            generics.push(r.sym()?);
        }
        let n_fields = r.u64()? as usize;
        let mut fields = Vec::new();
        for _ in 0..n_fields {
            let fname = r.sym()?;
            let fty = read_type(&mut r)?;
            fields.push((fname, fty));
        }
        structs.insert(gid, StructFields { generics, fields });
    }

    Ok(ImmutableGlobalRegistry {
        layouts,
        module_indices,
        fn_sigs,
        methods,
        bodies,
        // Enum-variant ordinals are not serialized into a `.vxlib` yet; a downstream compile that
        // constructs/matches an imported enum falls back to the AST path (#227).
        enum_variants: FxHashMap::default(),
        structs,
        // Data-carrying enum decls are not serialized into a `.vxlib` yet; constructing/matching a
        // monomorphized imported enum then falls back to the AST path (#242).
        enum_data: FxHashMap::default(),
        merge_state: Default::default(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::ModuleInterface;

    fn parse_and_resolve(path: &str, src: &str) -> crate::syntax::VxModule {
        let mut lexer = crate::lexer::Lexer::new(src);
        let tokens = lexer.tokenize();
        let mut parser = crate::parser::Parser::new(&tokens, src);
        let mut prog = parser.parse().expect("parse failed");
        prog.module_path = path.into();
        let mut mods = vec![prog];
        let symbol_map = crate::resolver::build_symbol_map(&mods);
        mods[0].resolve_names(&symbol_map, &[]);
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

    /// The signature tables round-trip too: `resolve_fn` / `resolve_method` return the same GID and
    /// return type after a serialize -> deserialize cycle, across the return-type shapes the stdlib
    /// actually uses -- scalar, nominal struct, and a (dimensionless) tensor (#220 stage 2).
    #[test]
    fn registry_interface_round_trips_functions_and_methods() {
        let m = parse_and_resolve(
            "crate::m",
            "struct Point { x: i32, y: i32 }\n\
             fn origin() -> Point { return Point { x: 0i32, y: 0i32 }; }\n\
             fn scale() -> f32 { return 2.0f32; }\n\
             fn zeros() -> Tensor<f32, [?, ?]> { return zeros(); }\n\
             fn pick(a: &Point, b: &Point) -> &i32 { return &b.x; }\n\
             unsafe fn peek(p: *mut i32) -> i32 { unsafe { return *p; } }\n\
             impl Point { fn sum(self: Point) -> i32 { return self.x + self.y; } }\n\
             trait Sq { fn sq(self: Self) -> f32; }\n\
             impl Sq for f32 { fn sq(self: f32) -> f32 { return self * self; } }\n",
        );
        let reg =
            crate::pipeline::build_frozen_registry(std::slice::from_ref(&m)).expect("acyclic");
        let bytes = serialize_registry_interface(&reg);
        let round = deserialize_registry_interface(&bytes).expect("deserialize");

        // Every fn_sig the original holds comes back with an identical GID + return type.
        assert!(!reg.fn_sigs.is_empty());
        assert_eq!(reg.fn_sigs.len(), round.fn_sigs.len());
        for (name, sig) in &reg.fn_sigs {
            let got = round
                .resolve_fn(name)
                .expect("fn resolves after round-trip");
            assert_eq!(got.gid, sig.gid, "fn GID parity for {name}");
            assert_eq!(got.ret_ty, sig.ret_ty, "fn return type parity for {name}");
            assert_eq!(got.params, sig.params, "fn param types parity for {name}");
            assert_eq!(got.ret_prov, sig.ret_prov, "fn ret_prov parity for {name}");
            assert_eq!(
                got.is_unsafe, sig.is_unsafe,
                "fn unsafe-ness parity for {name}"
            );
        }
        // The unsafe flag carries a real value across the boundary, so an importing module can
        // refuse a call from safe code without seeing the body.
        assert!(
            round.resolve_fn(&Symbol::from("peek")).unwrap().is_unsafe,
            "`unsafe fn peek` comes back unsafe"
        );
        assert!(
            !round.resolve_fn(&Symbol::from("scale")).unwrap().is_unsafe,
            "a safe fn comes back safe"
        );
        // The provenance code carries a real value across the boundary: `pick(a, b) -> &b.x` derives
        // from parameter slot 1, encoded as `2` (#265 step 7).
        assert_eq!(
            round.resolve_fn(&Symbol::from("pick")).unwrap().ret_prov,
            2,
            "pick's return derives from parameter slot 1"
        );

        // Same for methods, keyed by (receiver GID, method name).
        assert!(!reg.methods.is_empty());
        assert_eq!(reg.methods.len(), round.methods.len());
        for ((recv, name), sig) in &reg.methods {
            let got = round
                .resolve_method(*recv, name)
                .expect("method resolves after round-trip");
            assert_eq!(got.gid, sig.gid, "method GID parity for {name}");
            assert_eq!(
                got.ret_ty, sig.ret_ty,
                "method return type parity for {name}"
            );
            assert_eq!(
                got.params, sig.params,
                "method param types parity for {name}"
            );
            assert_eq!(
                got.ret_prov, sig.ret_prov,
                "method ret_prov parity for {name}"
            );
        }

        // structs round-trip: the declared field AST types come back, so a downstream compile can
        // type member access on an imported struct (#219).
        assert!(!reg.structs.is_empty());
        assert_eq!(reg.structs.len(), round.structs.len());
        for (gid, sf) in &reg.structs {
            let got = round.structs.get(gid).expect("struct round-trips");
            assert_eq!(
                got.generics, sf.generics,
                "struct generics parity for {gid:?}"
            );
            assert_eq!(
                got.fields, sf.fields,
                "struct field-type parity for {gid:?}"
            );
        }
        // `Point.x` specifically survives with its exact declared type — the field an imported `p.x`
        // reads back from the interface. The table is GID-keyed (#291), so resolve the name first.
        let point_gid = round
            .resolve_unique_nominal(&Symbol::from("Point"))
            .expect("Point resolves to its GID");
        let point = round.structs.get(&point_gid).expect("Point round-trips");
        let point_orig = reg.structs.get(&point_gid).unwrap();
        assert_eq!(point.fields[0].0, Symbol::from("x"));
        assert_eq!(
            point.fields[0], point_orig.fields[0],
            "Point.x field type survives"
        );
    }

    /// A real function's flat-HIR body round-trips: lower `fn add(..)` to flat HIR, stash it as a
    /// `FnBody` keyed by the fn GID, serialize -> deserialize, and assert `body_of` returns the
    /// identical instruction stream, type stream, and signature (#220 stage 3).
    #[test]
    fn registry_interface_round_trips_a_flat_hir_body() {
        use std::sync::Arc;
        let m = parse_and_resolve(
            "crate::m",
            "fn add(a: i32, b: i32) -> i32 { return a + b; }",
        );
        let mut reg =
            crate::pipeline::build_frozen_registry(std::slice::from_ref(&m)).expect("acyclic");

        // Lower `add` to flat HIR (a scalar body needs no populated registry).
        let func = m
            .functions
            .iter()
            .find(|f| f.name.as_ref() == "add")
            .unwrap();
        let mut worker =
            crate::session::LocalWorkerState::new(Arc::new(crate::session::GlobalSession::new(1)));
        assert!(
            crate::hir::flatten::lower_function_to_hir(func, &mut worker).is_ok(),
            "add lowers to flat HIR"
        );
        assert!(!worker.local_hir_stream.is_empty());

        let gid = reg.fn_sigs.get(&func.name).expect("add in fn_sigs").gid;
        let body = FnBody {
            name: func.name.clone(),
            params: func.params.iter().map(|(_, t)| t.clone()).collect(),
            ret_ty: func.return_type.clone(),
            hir: worker.local_hir_stream.clone(),
            types: worker.local_type_stream.clone(),
        };
        reg.bodies.insert(gid, body.clone());

        let bytes = serialize_registry_interface(&reg);
        let round = deserialize_registry_interface(&bytes).expect("deserialize");
        let got = round.body_of(gid).expect("body_of after round-trip");
        assert_eq!(got.name, body.name);
        assert_eq!(got.params, body.params);
        assert_eq!(got.ret_ty, body.ret_ty);
        assert_eq!(got.hir, body.hir, "instruction stream parity");
        assert_eq!(got.types, body.types, "type stream parity");
    }

    /// The portability gate: a body whose type stream still holds a per-compilation *deferred* GID (a
    /// generic instantiation) is not portable, so it is *skipped* on serialize -- `body_of` then
    /// declines it rather than the artifact linking a body it can't resolve (#220, fail-closed).
    #[test]
    fn non_portable_generic_body_is_skipped() {
        let m = parse_and_resolve(
            "crate::m",
            "fn add(a: i32, b: i32) -> i32 { return a + b; }",
        );
        let mut reg =
            crate::pipeline::build_frozen_registry(std::slice::from_ref(&m)).expect("acyclic");
        let gid = reg.fn_sigs.get(&Symbol::from("add")).unwrap().gid;

        // A body whose type stream carries a deferred (word-3 escape-hatch) GID is non-portable.
        let deferred = TypeId::new(0xAAAA, 0xBBBB, 0, crate::gid::LOCAL_DEFERRED_BIT);
        assert!(deferred.is_local_deferred());
        reg.bodies.insert(
            gid,
            FnBody {
                name: Symbol::from("add"),
                params: vec![Type::Scalar(ElementType::I32)],
                ret_ty: Type::Scalar(ElementType::I32),
                hir: Vec::new(),
                types: vec![deferred],
            },
        );

        let round = deserialize_registry_interface(&serialize_registry_interface(&reg))
            .expect("deserialize");
        assert!(
            round.body_of(gid).is_none(),
            "a body with a deferred GID must not be serialized"
        );
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

    /// The #292 acceptance: a deliberately unencodable signature is skipped from the artifact AND
    /// reported — named, counted, with the codec's reason — instead of silently dropped.
    #[test]
    fn emit_reports_skipped_entries_per_table() {
        let mut reg = ImmutableGlobalRegistry::build_and_validate(vec![]).unwrap();
        reg.fn_sigs.insert(
            Symbol::from("good"),
            FnSig {
                gid: TypeId::new(1, 1, 0, 0),
                params: Vec::new(),
                ret_ty: Type::Scalar(ElementType::I32),
                ret_prov: 0,
                is_unsafe: false,
            },
        );
        // A tensor return whose dimension is a name belongs to a template, and `write_type`
        // rejects it; a literal or `?` would serialize.
        let dim = crate::syntax::Expr::Identifier(crate::syntax::IdentifierExpr {
            name: "N".into(),
            span: crate::syntax::Span::default(),
        });
        reg.fn_sigs.insert(
            Symbol::from("bad"),
            FnSig {
                gid: TypeId::new(1, 2, 0, 0),
                params: Vec::new(),
                ret_ty: Type::Tensor(
                    ElementType::F32,
                    vec![crate::syntax::Dim::Static(dim)],
                    None,
                ),
                ret_prov: 0,
                is_unsafe: false,
            },
        );

        let (bytes, report) = serialize_registry_interface_reporting(&reg);
        let fns = report
            .tables
            .iter()
            .find(|t| t.table == "fn_sigs")
            .expect("fn_sigs table reported");
        assert_eq!(fns.encoded, 1);
        assert_eq!(fns.skipped.len(), 1);
        assert_eq!(fns.skipped[0].0, "bad");
        assert!(
            fns.skipped[0].1.contains("tensor"),
            "reason names the offending type: {}",
            fns.skipped[0].1
        );
        assert_eq!(report.total_skipped(), 1);
        // The rendered summary carries the counts and the named skip.
        let rendered = report.to_string();
        assert!(rendered.contains("fn_sigs 1/2"), "{rendered}");
        assert!(rendered.contains("skipped fn_sigs 'bad'"), "{rendered}");

        // The artifact itself stays fail-closed: `good` crosses, `bad` is absent.
        let round = deserialize_registry_interface(&bytes).expect("deserialize");
        assert!(round.resolve_fn(&Symbol::from("good")).is_some());
        assert!(round.resolve_fn(&Symbol::from("bad")).is_none());
    }
}
