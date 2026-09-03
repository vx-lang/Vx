//===- layout.rs - Vx Compiler ---------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// Structural layout computation for nominal types (#199).
//
// Computes the byte size, alignment, and per-field offsets of structs (and the
// size/align of C-like enums) so the frozen registry carries real layouts
// instead of the earlier `size = 0 / align = 0` stub. The flat HIR uses
// these to size `Alloca` and resolve field offsets identically to the tree-AST
// codegen path.
//
// Layout matches the AST codegen's LLVM struct lowering: fields are laid out in
// declaration order with natural alignment, each at the next offset that is a
// multiple of its alignment; the struct's alignment is the max field alignment
// and its size is rounded up to that alignment (C-like, non-packed). Scalar
// sizes mirror `SizeOfExpr` lowering in `codegen/lower/expr.rs`.
//
//===----------------------------------------------------------------------===//
use rustc_hash::{FxHashMap, FxHashSet};

use crate::gid::TypeId;
use crate::symbol::Symbol;
use crate::syntax::{ElementType, EnumDecl, StructDecl, Type};

/// The type of a struct field, enough to recover its value type at a field access. A pointer/ref
/// field is `Opaque`: modelled for size/offset but not decomposed into a loadable scalar.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldTy {
    Scalar(ElementType),
    Nominal(TypeId),
    Opaque,
}

/// The byte offset, size, and type of a single struct field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldLayout {
    pub name: Symbol,
    pub offset: usize,
    pub size: usize,
    pub ty: FieldTy,
}

/// The computed layout of a nominal type: total size, alignment, and (for
/// structs) the offset/size of each field in declaration order. `fields` is
/// empty for enums.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeLayout {
    pub size: usize,
    pub align: usize,
    pub fields: Vec<FieldLayout>,
}

/// Round `n` up to the next multiple of `align` (a power of two, `>= 1`).
pub fn align_up(n: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two(), "alignment must be a power of two");
    (n + align - 1) & !(align - 1)
}

/// Size and alignment (bytes) of a scalar element type, with natural alignment
/// (`align == size`). Returns `None` for a `Generic` type variable — it has no
/// concrete layout until the enclosing type is monomorphized.
pub fn scalar_size_align(et: &ElementType) -> Option<(usize, usize)> {
    // Byte size = the dense bit width rounded up to a whole byte (padded storage: a sub-byte `I4`
    // field occupies 1 byte). Derived from the single width source `ElementType::bits`, so this can no
    // longer drift from `hir::memory::element_bits` (they disagreed on `I4` before — see `bits`). A
    // scalar's alignment equals its size here (natural alignment for the modelled widths). (P1-4a)
    let bytes = (et.bits()? as usize).div_ceil(8);
    Some((bytes, bytes))
}

/// Computes nominal-type layouts over a fixed set of struct/enum declarations,
/// resolving nested by-value nominals by their GID. Memoized; a by-value cycle
/// (an infinite-sized type) resolves to `None` here and is reported separately
/// by the registry's cycle detection.
pub struct LayoutComputer<'a> {
    structs: FxHashMap<TypeId, &'a StructDecl>,
    enums: FxHashMap<TypeId, &'a EnumDecl>,
    memo: FxHashMap<TypeId, Option<TypeLayout>>,
    visiting: FxHashSet<TypeId>,
}

impl<'a> LayoutComputer<'a> {
    pub fn new(
        structs: FxHashMap<TypeId, &'a StructDecl>,
        enums: FxHashMap<TypeId, &'a EnumDecl>,
    ) -> Self {
        Self {
            structs,
            enums,
            memo: FxHashMap::default(),
            visiting: FxHashSet::default(),
        }
    }

    /// The layout of the nominal type identified by `id`, or `None` if it is not
    /// a known nominal, is generic, contains a not-yet-modelled field type
    /// (tensor, closure, …), or sits on a by-value cycle.
    pub fn layout_of(&mut self, id: TypeId) -> Option<TypeLayout> {
        if let Some(cached) = self.memo.get(&id) {
            return cached.clone();
        }
        // A back-edge means an infinite-sized by-value cycle; leave it to the
        // registry's cycle detection to report and treat as incomputable here.
        if self.visiting.contains(&id) {
            return None;
        }
        self.visiting.insert(id);
        let result = self.compute(id);
        self.visiting.remove(&id);
        self.memo.insert(id, result.clone());
        result
    }

    fn compute(&mut self, id: TypeId) -> Option<TypeLayout> {
        if let Some(&s) = self.structs.get(&id) {
            return self.struct_layout(s);
        }
        if let Some(&e) = self.enums.get(&id) {
            return self.enum_layout(e);
        }
        None
    }

    fn struct_layout(&mut self, decl: &'a StructDecl) -> Option<TypeLayout> {
        let mut offset = 0usize;
        let mut align = 1usize;
        let mut fields = Vec::with_capacity(decl.fields.len());
        for (name, ty) in &decl.fields {
            let (fsize, falign, fty) = self.field_info(ty)?;
            let foffset = align_up(offset, falign);
            fields.push(FieldLayout {
                name: name.clone(),
                offset: foffset,
                size: fsize,
                ty: fty,
            });
            offset = foffset + fsize;
            align = align.max(falign);
        }
        Some(TypeLayout {
            size: align_up(offset, align),
            align,
            fields,
        })
    }

    fn enum_layout(&mut self, decl: &'a EnumDecl) -> Option<TypeLayout> {
        // Payload-carrying (tagged-union) enums need a layout that matches the
        // codegen representation exactly; deferred. A C-like enum (no payloads)
        // is a bare `i32` discriminant, matching the enum tag in codegen.
        let has_payload = decl
            .variants
            .iter()
            .any(|(_, payload)| payload.as_ref().is_some_and(|tys| !tys.is_empty()));
        if has_payload {
            return None;
        }
        Some(TypeLayout {
            size: 4,
            align: 4,
            fields: Vec::new(),
        })
    }

    /// Size/align/type of a field. Nominals recurse (by GID); pointer-like wrappers
    /// are pointer-sized and `Opaque`; location/proof wrappers pass through to the
    /// inner type. Anything not yet modelled (tensor, generic instance, closure,
    /// unresolved nominal, …) returns `None`, making the enclosing layout
    /// incomputable for now.
    fn field_info(&mut self, ty: &Type) -> Option<(usize, usize, FieldTy)> {
        match ty {
            Type::Scalar(et) => {
                let (size, align) = scalar_size_align(et)?;
                Some((size, align, FieldTy::Scalar(et.clone())))
            }
            // Pointer-like fields, all a pointer-sized opaque word: raw pointers, borrows, and a
            // function/closure type (a function pointer, e.g. `Closure1`'s `func` field). (#242)
            Type::Pointer(..)
            | Type::Ref(..)
            | Type::Borrow { .. }
            | Type::Function(..)
            | Type::Closure(..) => Some((8, 8, FieldTy::Opaque)),
            Type::Pinned(inner, _) | Type::Verified(inner) => self.field_info(inner),
            Type::Struct(_, Some(id)) | Type::Enum(_, Some(id)) => {
                let layout = self.layout_of(*id)?;
                Some((layout.size, layout.align, FieldTy::Nominal(*id)))
            }
            // A by-value generic-instance aggregate field (`VecMap { iter: VecIter<T>, f: Closure1<..> }`):
            // its layout is the base nominal's when that layout is instance-independent — every generic
            // parameter appears only behind a pointer, so the size is the same for any instantiation
            // (`VecIter<T>` = `{ *const Vec<T>, i32 }`). Resolve the base GID (the attached one, else by
            // name — a cross-module generic instance may leave the base's GID unattached, `Struct(
            // "Closure1", None)`). An instance-dependent base (`Box<T> { value: T }`) leaves the base
            // layout a stub, so `layout_of` returns `None` and the field declines. (#242)
            Type::GenericInstance(base, _) => {
                let id = match base.as_ref() {
                    Type::Struct(_, Some(id)) | Type::Enum(_, Some(id)) => *id,
                    Type::Struct(name, None) | Type::Enum(name, None) => *self
                        .structs
                        .iter()
                        .find(|(_, d)| d.name == *name)
                        .map(|(g, _)| g)?,
                    _ => return None,
                };
                let layout = self.layout_of(id)?;
                Some((layout.size, layout.align, FieldTy::Nominal(id)))
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sym(s: &str) -> Symbol {
        s.into()
    }

    fn gid(w1: u64) -> TypeId {
        TypeId::new(1, w1, 0, 0)
    }

    fn strukt(name: &str, fields: Vec<(&str, Type)>) -> StructDecl {
        StructDecl {
            name: sym(name),
            generics: Vec::new(),
            fields: fields.into_iter().map(|(n, t)| (sym(n), t)).collect(),
            doc_comment: None,
        }
    }

    fn scalar(et: ElementType) -> Type {
        Type::Scalar(et)
    }

    #[test]
    fn align_up_rounds_to_power_of_two() {
        assert_eq!(align_up(0, 4), 0);
        assert_eq!(align_up(1, 4), 4);
        assert_eq!(align_up(4, 4), 4);
        assert_eq!(align_up(5, 8), 8);
        assert_eq!(align_up(9, 1), 9);
    }

    #[test]
    fn scalar_sizes_match_codegen() {
        assert_eq!(scalar_size_align(&ElementType::Bool), Some((1, 1)));
        assert_eq!(scalar_size_align(&ElementType::I8), Some((1, 1)));
        assert_eq!(scalar_size_align(&ElementType::F16), Some((2, 2)));
        assert_eq!(scalar_size_align(&ElementType::I32), Some((4, 4)));
        assert_eq!(scalar_size_align(&ElementType::F32), Some((4, 4)));
        assert_eq!(scalar_size_align(&ElementType::I64), Some((8, 8)));
        assert_eq!(scalar_size_align(&ElementType::F64), Some((8, 8)));
        assert_eq!(scalar_size_align(&ElementType::I128), Some((16, 16)));
        // A generic type variable has no concrete layout.
        assert_eq!(scalar_size_align(&ElementType::Generic(sym("T"))), None);
    }

    #[test]
    fn width_tables_derive_from_one_source() {
        // P1-4a invariant: `element_bits` (dense) and `scalar_size_align` (padded bytes) both derive
        // from `ElementType::bits`, so they can never drift again. In particular the sub-byte case that
        // used to be maintained independently: I4 is 4 dense bits but occupies a padded byte.
        for et in [
            ElementType::Bool,
            ElementType::I4,
            ElementType::U4,
            ElementType::I8,
            ElementType::F8E4M3,
            ElementType::F8E5M2,
            ElementType::F4E2M1,
            ElementType::F16,
            ElementType::BF16,
            ElementType::F32,
            ElementType::I64,
            ElementType::I128,
        ] {
            let bits = et.bits().unwrap() as usize;
            assert_eq!(
                crate::hir::memory::element_bits(&et),
                Some(bits as u64),
                "element_bits must equal bits() for {et:?}"
            );
            assert_eq!(
                scalar_size_align(&et),
                Some((bits.div_ceil(8), bits.div_ceil(8))),
                "scalar_size_align must be ceil(bits/8) for {et:?}"
            );
        }
        assert_eq!(ElementType::I4.bits(), Some(4));
        assert_eq!(scalar_size_align(&ElementType::I4), Some((1, 1)));
        assert_eq!(crate::hir::memory::element_bits(&ElementType::I4), Some(4));
    }

    #[test]
    fn struct_layout_pads_and_aligns() {
        // struct S { a: i8, b: i32, c: i8 }
        //   a @ 0 (1B), 3B pad, b @ 4 (4B), c @ 8 (1B) -> size rounded to align 4 = 12.
        let s = strukt(
            "S",
            vec![
                ("a", scalar(ElementType::I8)),
                ("b", scalar(ElementType::I32)),
                ("c", scalar(ElementType::I8)),
            ],
        );
        let mut structs = FxHashMap::default();
        structs.insert(gid(1), &s);
        let mut lc = LayoutComputer::new(structs, FxHashMap::default());
        let l = lc.layout_of(gid(1)).unwrap();
        assert_eq!(l.align, 4);
        assert_eq!(l.size, 12);
        assert_eq!(l.fields[0].offset, 0);
        assert_eq!(l.fields[1].offset, 4);
        assert_eq!(l.fields[2].offset, 8);
        assert_eq!(l.fields[2].size, 1);
    }

    #[test]
    fn nested_struct_layout() {
        // Inner { x: i32, y: i32 } -> size 8, align 4.
        // Outer { flag: i8, inner: Inner } -> flag @ 0, inner @ 4 (aligned) -> size 12.
        let inner = strukt(
            "Inner",
            vec![
                ("x", scalar(ElementType::I32)),
                ("y", scalar(ElementType::I32)),
            ],
        );
        let outer = strukt(
            "Outer",
            vec![
                ("flag", scalar(ElementType::I8)),
                ("inner", Type::Struct(sym("Inner"), Some(gid(1)))),
            ],
        );
        let mut structs = FxHashMap::default();
        structs.insert(gid(1), &inner);
        structs.insert(gid(2), &outer);
        let mut lc = LayoutComputer::new(structs, FxHashMap::default());
        let li = lc.layout_of(gid(1)).unwrap();
        assert_eq!((li.size, li.align), (8, 4));
        let lo = lc.layout_of(gid(2)).unwrap();
        assert_eq!((lo.size, lo.align), (12, 4));
        assert_eq!(lo.fields[1].offset, 4);
        assert_eq!(lo.fields[1].size, 8);
    }

    #[test]
    fn pointer_field_is_word_sized() {
        let s = strukt(
            "P",
            vec![(
                "p",
                Type::Pointer(Box::new(scalar(ElementType::I32)), None, false),
            )],
        );
        let mut structs = FxHashMap::default();
        structs.insert(gid(1), &s);
        let mut lc = LayoutComputer::new(structs, FxHashMap::default());
        let l = lc.layout_of(gid(1)).unwrap();
        assert_eq!((l.size, l.align), (8, 8));
    }

    #[test]
    fn c_like_enum_is_i32() {
        let e = EnumDecl {
            name: sym("Color"),
            generics: Vec::new(),
            variants: vec![
                (sym("Red"), None),
                (sym("Green"), None),
                (sym("Blue"), None),
            ],
            doc_comment: None,
        };
        let mut enums = FxHashMap::default();
        enums.insert(gid(1), &e);
        let mut lc = LayoutComputer::new(FxHashMap::default(), enums);
        let l = lc.layout_of(gid(1)).unwrap();
        assert_eq!((l.size, l.align), (4, 4));
        assert!(l.fields.is_empty());
    }

    #[test]
    fn payload_enum_is_incomputable_for_now() {
        // Tagged-union layout is deferred (must match codegen exactly).
        let e = EnumDecl {
            name: sym("Opt"),
            generics: Vec::new(),
            variants: vec![
                (sym("None"), None),
                (sym("Some"), Some(vec![scalar(ElementType::I32)])),
            ],
            doc_comment: None,
        };
        let mut enums = FxHashMap::default();
        enums.insert(gid(1), &e);
        let mut lc = LayoutComputer::new(FxHashMap::default(), enums);
        assert_eq!(lc.layout_of(gid(1)), None);
    }

    #[test]
    fn generic_field_makes_struct_incomputable() {
        let s = strukt("G", vec![("v", scalar(ElementType::Generic(sym("T"))))]);
        let mut structs = FxHashMap::default();
        structs.insert(gid(1), &s);
        let mut lc = LayoutComputer::new(structs, FxHashMap::default());
        assert_eq!(lc.layout_of(gid(1)), None);
    }

    #[test]
    fn by_value_cycle_is_incomputable() {
        // struct L { next: L } is infinite-sized: no layout (the registry reports the cycle).
        let l = strukt("L", vec![("next", Type::Struct(sym("L"), Some(gid(1))))]);
        let mut structs = FxHashMap::default();
        structs.insert(gid(1), &l);
        let mut lc = LayoutComputer::new(structs, FxHashMap::default());
        assert_eq!(lc.layout_of(gid(1)), None);
    }

    #[test]
    fn pointer_breaks_the_cycle() {
        // struct L { next: *L } is finite: the pointer is word-sized, no recursion into L.
        let l = strukt(
            "L",
            vec![(
                "next",
                Type::Pointer(Box::new(Type::Struct(sym("L"), Some(gid(1)))), None, false),
            )],
        );
        let mut structs = FxHashMap::default();
        structs.insert(gid(1), &l);
        let mut lc = LayoutComputer::new(structs, FxHashMap::default());
        let computed = lc.layout_of(gid(1)).unwrap();
        assert_eq!((computed.size, computed.align), (8, 8));
    }
}
