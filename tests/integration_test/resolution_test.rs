//===- resolution_test.rs - Vx Compiler ------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// This file contains integration tests for the Name Resolution pass.
// It verifies that the compiler correctly links identifiers to their declarations
// across complex nested scopes, module boundaries, and shadowing scenarios.
//
//===----------------------------------------------------------------------===//
use vxc::resolver::build_symbol_map;
use vxc::syntax::{Function, Span, StructDecl, Type, VxModule};

/// Parse a module from source and stamp its module path (as the driver does after parsing).
fn parse_module(path: &str, src: &str) -> VxModule {
    let mut lexer = vxc::lexer::Lexer::new(src);
    let tokens = lexer.tokenize();
    let mut parser = vxc::parser::Parser::new(&tokens, src);
    let mut program = parser.parse().expect("parse failed");
    program.module_path = path.into();
    program
}

fn param_ty(m: &VxModule, fn_idx: usize) -> Type {
    m.functions[fn_idx].params[0].1.clone()
}

fn sym(s: &str) -> vxc::symbol::Symbol {
    s.into()
}

/// The type of struct `struct_idx`'s field `field_idx`.
fn field_ty(m: &VxModule, struct_idx: usize, field_idx: usize) -> Type {
    m.structs[struct_idx].fields[field_idx].1.clone()
}

/// Strip location wrappers (`&T`, `*T`, `Verified<T>`, …) down to the nominal underneath.
fn peel(mut t: Type) -> Type {
    loop {
        t = match t {
            Type::Ref(inner, _)
            | Type::Pointer(inner, _, _)
            | Type::Verified(inner)
            | Type::Pinned(inner, _) => *inner,
            Type::Borrow { inner, .. } => *inner,
            other => return other,
        };
    }
}

fn resolve1(path: &str, src: &str) -> (VxModule, vxc::resolver::SymbolMap) {
    let mut m = parse_module(path, src);
    let map = build_symbol_map(std::slice::from_ref(&m));
    m.resolve_names(&map);
    (m, map)
}

/// #194 acceptance: a *qualified* cross-module reference `A::Foo` (Foo defined in module A, used in
/// module B) is attached the **defining** module's GID -- word 0 = A's module hash -- and matches
/// A's own `Foo` GID exactly. Exercises the full path: the parser producing a `::`-qualified nominal
/// and `resolve_names` routing it to the defining module via the cross-module symbol map.
#[test]
fn cross_module_qualified_reference_resolves_to_defining_module() {
    let module_a = parse_module("A", "struct Foo { x: i32 }");
    let mut module_b = parse_module("B", "fn use_foo(f: A::Foo) -> i32 { return 0; }");

    let symbol_map = build_symbol_map(&[module_a.clone(), module_b.clone()]);
    module_b.resolve_names(&symbol_map);

    let a_foo = symbol_map[&sym("A")][&sym("Foo")];
    match param_ty(&module_b, 0) {
        Type::Struct(name, Some(id)) => {
            assert_eq!(
                name.as_ref(),
                "A::Foo",
                "qualified path kept as the nominal name"
            );
            assert_eq!(id, a_foo, "resolves to A's Foo GID, not None or a local");
            assert_eq!(
                id.module_id(),
                vxc::hash::compute_module_hash("A"),
                "word 0 is the *defining* module's hash"
            );
        }
        other => panic!("expected resolved Struct, got {other:?}"),
    }
}

/// A qualified cross-module reference must **not** be shadowed by a same-named local type. Module B
/// defines its own `Foo`, but `A::Foo` still resolves to A's `Foo` (distinct GID).
#[test]
fn qualified_reference_is_not_shadowed_by_local_same_name() {
    let module_a = parse_module("A", "struct Foo { x: i32 }");
    let mut module_b = parse_module(
        "B",
        "struct Foo { y: f32 }\nfn use_foo(f: A::Foo) -> i32 { return 0; }",
    );

    let symbol_map = build_symbol_map(&[module_a.clone(), module_b.clone()]);
    module_b.resolve_names(&symbol_map);

    let a_foo = symbol_map[&sym("A")][&sym("Foo")];
    let b_foo = symbol_map[&sym("B")][&sym("Foo")];
    assert_ne!(a_foo, b_foo, "distinct modules -> distinct GIDs");

    match param_ty(&module_b, 0) {
        Type::Struct(_, Some(id)) => {
            assert_eq!(id, a_foo, "A::Foo resolves to A's Foo, not B's local Foo");
            assert_ne!(id, b_foo);
        }
        other => panic!("expected resolved Struct, got {other:?}"),
    }
}

/// A qualified reference to the *current* module (`A::Foo` inside module A) resolves through the
/// local symbol table (the fast path) to the same GID as the unqualified `Foo` — module-local
/// references are the common case and must stay both cheap and correct.
#[test]
fn qualified_self_reference_resolves_via_local_table() {
    let mut module_a = parse_module(
        "A",
        "struct Foo { x: i32 }\nfn use_foo(f: A::Foo) -> i32 { return 0; }",
    );
    let symbol_map = build_symbol_map(&[module_a.clone()]);
    module_a.resolve_names(&symbol_map);

    let a_foo = symbol_map[&sym("A")][&sym("Foo")];
    match param_ty(&module_a, 0) {
        Type::Struct(name, Some(id)) => {
            assert_eq!(name.as_ref(), "A::Foo");
            assert_eq!(
                id, a_foo,
                "self-qualified ref resolves to the module's own Foo"
            );
        }
        other => panic!("expected resolved Struct, got {other:?}"),
    }
}

/// An `import a::Foo;` brings the *unqualified* name `Foo` into scope from module `a`; a plain `Foo`
/// reference then resolves to `a`'s GID (when the current module has no local `Foo`).
#[test]
fn imported_unqualified_name_resolves_cross_module() {
    let module_a = parse_module("crate::a", "struct Foo { x: i32 }");
    let mut module_b = parse_module(
        "crate::b",
        "import crate::a::Foo;\nfn use_foo(f: Foo) -> i32 { return 0; }",
    );

    let symbol_map = build_symbol_map(&[module_a.clone(), module_b.clone()]);
    module_b.resolve_names(&symbol_map);

    let a_foo = symbol_map[&sym("crate::a")][&sym("Foo")];
    match param_ty(&module_b, 0) {
        Type::Struct(name, Some(id)) => {
            assert_eq!(name.as_ref(), "Foo");
            assert_eq!(id, a_foo, "imported Foo resolves to crate::a's Foo");
            assert_eq!(id.module_id(), vxc::hash::compute_module_hash("crate::a"));
        }
        other => panic!("expected resolved Struct, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Recursion & corner cases: resolution walks the AST (finite), never *follows*
// a type's definition, so recursive types can't loop it — but they do stress
// that self-references and cross-module cycles attach the right GID. (Actual
// infinite-*size* detection is the registry's job; see the pipeline tests.)
// ---------------------------------------------------------------------------

/// A self-recursive struct: `List`'s own field of type `List` resolves to `List`'s GID.
#[test]
fn self_recursive_struct_field_resolves_to_own_gid() {
    let (m, map) = resolve1("m", "struct List { next: List, val: i32 }");
    let list = map[&sym("m")][&sym("List")];
    match field_ty(&m, 0, 0) {
        Type::Struct(_, Some(id)) => assert_eq!(id, list, "self-reference resolves to own GID"),
        other => panic!("expected resolved Struct, got {other:?}"),
    }
}

/// Mutually recursive structs in one module: each field resolves to the *sibling's* GID.
#[test]
fn mutually_recursive_structs_same_module_resolve() {
    let (m, map) = resolve1("m", "struct A { b: B }\nstruct B { a: A }");
    let a = map[&sym("m")][&sym("A")];
    let b = map[&sym("m")][&sym("B")];
    assert_ne!(a, b);
    // A.b : B
    match field_ty(&m, 0, 0) {
        Type::Struct(_, Some(id)) => assert_eq!(id, b),
        other => panic!("A.b expected B, got {other:?}"),
    }
    // B.a : A
    match field_ty(&m, 1, 0) {
        Type::Struct(_, Some(id)) => assert_eq!(id, a),
        other => panic!("B.a expected A, got {other:?}"),
    }
}

/// A type recursive *through a generic instance* (`children: List<Node>`): the recursive `Node`
/// argument resolves even though the container `List` may be undefined.
#[test]
fn recursive_type_inside_generic_instance_resolves() {
    let (m, map) = resolve1("m", "struct Node { children: List<Node> }");
    let node = map[&sym("m")][&sym("Node")];
    match field_ty(&m, 0, 0) {
        Type::GenericInstance(_, args) => match &args[0] {
            Type::Struct(_, Some(id)) => assert_eq!(*id, node, "recursive generic arg resolves"),
            other => panic!("expected resolved Node arg, got {other:?}"),
        },
        other => panic!("expected GenericInstance, got {other:?}"),
    }
}

/// Recursion behind indirection wrappers (`next: &List`): the nominal under the `&` still resolves.
#[test]
fn recursive_ref_under_wrapper_resolves() {
    let (m, map) = resolve1("m", "struct List { next: &List, val: i32 }");
    let list = map[&sym("m")][&sym("List")];
    match peel(field_ty(&m, 0, 0)) {
        Type::Struct(_, Some(id)) => assert_eq!(id, list),
        other => panic!("expected resolved Struct under &, got {other:?}"),
    }
}

/// A *qualified* self-recursion (`next: m::List` inside module `m`) resolves through the local
/// fast-path to `List`'s own GID.
#[test]
fn qualified_self_recursion_resolves_via_local_table() {
    let (m, map) = resolve1("m", "struct List { next: m::List, val: i32 }");
    let list = map[&sym("m")][&sym("List")];
    match field_ty(&m, 0, 0) {
        Type::Struct(name, Some(id)) => {
            assert_eq!(name.as_ref(), "m::List");
            assert_eq!(id, list);
        }
        other => panic!("expected resolved Struct, got {other:?}"),
    }
}

/// Cross-module *mutual* recursion: A::Node references B::Other and vice-versa; each qualified
/// reference resolves to its *defining* module's GID (no infinite loop — resolution never follows
/// the definitions).
#[test]
fn cross_module_mutual_recursion_resolves_each_side() {
    let mut a = parse_module("A", "struct Node { other: B::Other }");
    let mut b = parse_module("B", "struct Other { back: A::Node }");
    let map = build_symbol_map(&[a.clone(), b.clone()]);
    a.resolve_names(&map);
    b.resolve_names(&map);

    let a_node = map[&sym("A")][&sym("Node")];
    let b_other = map[&sym("B")][&sym("Other")];
    match field_ty(&a, 0, 0) {
        Type::Struct(_, Some(id)) => assert_eq!(id, b_other, "A::Node.other -> B::Other"),
        other => panic!("{other:?}"),
    }
    match field_ty(&b, 0, 0) {
        Type::Struct(_, Some(id)) => assert_eq!(id, a_node, "B::Other.back -> A::Node"),
        other => panic!("{other:?}"),
    }
}

/// A qualified reference into a module that does not exist must resolve to `None` (stay unresolved),
/// never panic.
#[test]
fn unresolved_qualified_module_stays_none() {
    let (m, _map) = resolve1("m", "struct S { x: Ghost::Foo }");
    match field_ty(&m, 0, 0) {
        Type::Struct(name, id) => {
            assert_eq!(name.as_ref(), "Ghost::Foo");
            assert!(id.is_none(), "unknown module -> unresolved, not a crash");
        }
        other => panic!("expected Struct, got {other:?}"),
    }
}

/// A multi-segment module path (`a::b::c::Foo`) splits at the *last* `::` and resolves against the
/// full module key.
#[test]
fn multi_segment_module_path_resolves() {
    let a = parse_module("a::b::c", "struct Foo { x: i32 }");
    let mut user = parse_module("user", "struct Use { f: a::b::c::Foo }");
    let map = build_symbol_map(&[a.clone(), user.clone()]);
    user.resolve_names(&map);

    let foo = map[&sym("a::b::c")][&sym("Foo")];
    match field_ty(&user, 0, 0) {
        Type::Struct(name, Some(id)) => {
            assert_eq!(name.as_ref(), "a::b::c::Foo");
            assert_eq!(id, foo);
        }
        other => panic!("expected resolved Struct, got {other:?}"),
    }
}

/// A recursive *function* (calls itself) whose signature uses a recursive data type: the parameter
/// type resolves; the recursive call in the body does not perturb type resolution.
#[test]
fn recursive_function_signature_types_resolve() {
    let (m, map) = resolve1(
        "m",
        "struct List { next: &List, val: i32 }\nfn len(l: List) -> i32 { return len(l); }",
    );
    let list = map[&sym("m")][&sym("List")];
    match param_ty(&m, 0) {
        Type::Struct(_, Some(id)) => assert_eq!(id, list, "recursive fn's param type resolves"),
        other => panic!("expected resolved List param, got {other:?}"),
    }
}

/// A generic *parameter* is a type variable, not a nominal — it stays unresolved even when a
/// same-named type exists in the module. (The `id` is unused downstream; correctness is that we
/// don't conflate the type variable `T` with a concrete `struct T`.)
#[test]
fn generic_param_does_not_bind_to_same_named_type() {
    // No same-named type: obviously unresolved.
    let (m1, _) = resolve1("m", "fn identity<T>(x: T) -> T { return x; }");
    match param_ty(&m1, 0) {
        Type::Generic(name, id) => {
            assert_eq!(name.as_ref(), "T");
            assert!(id.is_none(), "generic param has no nominal GID");
        }
        other => panic!("expected Generic param, got {other:?}"),
    }

    // A same-named `struct T` must NOT shadow-capture the generic parameter `T`.
    let (m2, _) = resolve1(
        "m",
        "struct T { x: i32 }\nfn identity<T>(x: T) -> T { return x; }",
    );
    match param_ty(&m2, 0) {
        Type::Generic(_, id) => assert!(
            id.is_none(),
            "generic param T must not bind to the same-named struct T"
        ),
        other => panic!("expected Generic param, got {other:?}"),
    }
}

/// The type inside enum variant `variant_idx`'s payload position `pos`.
fn variant_payload_ty(m: &VxModule, enum_idx: usize, variant_idx: usize, pos: usize) -> Type {
    m.enums[enum_idx].variants[variant_idx].1.as_ref().unwrap()[pos].clone()
}

/// A recursive *enum*: `Node(Tree)`'s payload type `Tree` resolves to the enum's own GID. (Enum
/// variant payloads were previously left unresolved; the flat pipeline needs them.)
#[test]
fn recursive_enum_payload_resolves_to_own_gid() {
    let (m, map) = resolve1("m", "enum Tree { Leaf, Node(Tree) }");
    let tree = map[&sym("m")][&sym("Tree")];
    match variant_payload_ty(&m, 0, 1, 0) {
        Type::Struct(_, Some(id)) => assert_eq!(id, tree, "recursive enum payload resolves"),
        other => panic!("expected resolved payload, got {other:?}"),
    }
}

/// An enum variant payload referencing a *cross-module* type resolves to the defining module's GID.
#[test]
fn enum_payload_cross_module_reference_resolves() {
    let a = parse_module("A", "struct Foo { x: i32 }");
    let mut b = parse_module("B", "enum Wrap { None, Some(A::Foo) }");
    let map = build_symbol_map(&[a.clone(), b.clone()]);
    b.resolve_names(&map);

    let a_foo = map[&sym("A")][&sym("Foo")];
    match variant_payload_ty(&b, 0, 1, 0) {
        Type::Struct(_, Some(id)) => assert_eq!(id, a_foo, "enum payload A::Foo -> A's GID"),
        other => panic!("expected resolved payload, got {other:?}"),
    }
}

#[test]
fn test_local_name_resolution() -> Result<(), String> {
    let mut module = VxModule {
        imports: Vec::new(),
        module_path: "core::math".into(),
        topologies: vec![],
        memories: vec![],
        transfer_impls: Vec::new(),
        externs: vec![],
        structs: vec![StructDecl {
            name: "Vector".into(),
            generics: vec![],
            fields: vec![],
            doc_comment: None,
        }],
        enums: vec![],
        traits: vec![],
        impls: vec![],
        macros: vec![],
        functions: vec![Function {
            name: "get_vector".into(),
            generics: vec![],
            params: vec![],
            topology: vxc::syntax::Topology::CPU,
            return_type: Type::Struct("Vector".into(), None),
            requires: Vec::new(),
            ensures: Vec::new(),
            where_transfers: Vec::new(),
            body: vec![],
            doc_comment: None,
        }],
    };

    // Phase 1.25: Build the SymbolMap from parsed modules
    let symbol_map = build_symbol_map(&[module.clone()]);

    // Phase 1.5: Resolve the AST
    module.resolve_names(&symbol_map);

    // Verify that `Vector` was mapped to a deterministic `TypeId`
    if let Type::Struct(name, id) = &module.functions[0].return_type {
        if name.as_ref() != "Vector" {
            return Err(format!("Assertion failed: {} != {}", name, "Vector"));
        }
        if !(id.is_some()) {
            return Err("Assertion failed: id.is_some()".to_string());
        } // Successfully resolved to a TypeId!

        let tid = id.unwrap();
        // The Module Hash and Symbol Hash should be populated
        if tid.module_id() == 0 {
            return Err(format!("Assertion failed: {} == {}", tid.module_id(), 0));
        }
        if tid.symbol_id() == 0 {
            return Err(format!("Assertion failed: {} == {}", tid.symbol_id(), 0));
        }
    } else {
        return Err("Expected Struct type".to_string());
    }

    Ok(())
}

use vxc::syntax::{Expr, LetDeclStmt, MemorySpace, NumberExpr, Statement};

#[test]
fn test_unresolved_symbol_remains_none() -> Result<(), String> {
    let mut module = VxModule {
        imports: Vec::new(),
        module_path: "core::bad".into(),
        topologies: vec![],
        memories: vec![],
        transfer_impls: Vec::new(),
        externs: vec![],
        structs: vec![], // Empty structs, "Vector" does not exist!
        enums: vec![],
        traits: vec![],
        impls: vec![],
        macros: vec![],
        functions: vec![Function {
            name: "get_vector".into(),
            generics: vec![],
            params: vec![],
            topology: vxc::syntax::Topology::CPU,
            return_type: Type::Struct("Vector".into(), None),
            requires: Vec::new(),
            ensures: Vec::new(),
            where_transfers: Vec::new(),
            body: vec![],
            doc_comment: None,
        }],
    };

    let symbol_map = build_symbol_map(&[module.clone()]);
    module.resolve_names(&symbol_map);

    if let Type::Struct(name, id) = &module.functions[0].return_type {
        if name.as_ref() != "Vector" {
            return Err(format!("Assertion failed: {} != {}", name, "Vector"));
        }
        if !(id.is_none()) {
            return Err("Assertion failed: id.is_none()".to_string());
        } // Should remain unresolved!
    } else {
        return Err("Expected Struct type".to_string());
    }

    Ok(())
}

#[test]
fn test_nested_type_resolution() -> Result<(), String> {
    let mut module = VxModule {
        imports: Vec::new(),
        module_path: "core::math".into(),
        topologies: vec![],
        memories: vec![],
        transfer_impls: Vec::new(),
        externs: vec![],
        structs: vec![StructDecl {
            name: "Matrix".into(),
            generics: vec![],
            fields: vec![],
            doc_comment: None,
        }],
        enums: vec![],
        traits: vec![],
        impls: vec![],
        macros: vec![],
        functions: vec![Function {
            name: "compute".into(),
            generics: vec![],
            topology: vxc::syntax::Topology::CPU,
            params: vec![(
                "m".into(),
                // &mut Matrix
                Type::Borrow {
                    inner: Box::new(Type::Struct("Matrix".into(), None)),
                    mem_space: Some(MemorySpace::CPUDRAM),
                    is_mut: true,
                    region_id: 0,
                },
            )],
            return_type: Type::Scalar(vxc::syntax::ElementType::Bool),
            requires: Vec::new(),
            ensures: Vec::new(),
            where_transfers: Vec::new(),
            body: vec![],
            doc_comment: None,
        }],
    };

    let symbol_map = build_symbol_map(&[module.clone()]);
    module.resolve_names(&symbol_map);

    if let Type::Borrow { inner, .. } = &module.functions[0].params[0].1 {
        if let Type::Struct(name, id) = &**inner {
            if name.as_ref() != "Matrix" {
                return Err(format!("Assertion failed: {} != {}", name, "Matrix"));
            }
            if !(id.is_some()) {
                return Err("Assertion failed: id.is_some()".to_string());
            } // Deeply nested type must be resolved!
        } else {
            return Err("Expected inner Struct type".to_string());
        }
    } else {
        return Err("Expected Borrow type".to_string());
    }

    Ok(())
}

#[test]
fn test_expr_and_stmt_resolution() -> Result<(), String> {
    let mut module = VxModule {
        imports: Vec::new(),
        module_path: "core::app".into(),
        topologies: vec![],
        memories: vec![],
        transfer_impls: Vec::new(),
        externs: vec![],
        structs: vec![StructDecl {
            name: "Config".into(),
            generics: vec![],
            fields: vec![],
            doc_comment: None,
        }],
        enums: vec![],
        traits: vec![],
        impls: vec![],
        macros: vec![],
        functions: vec![Function {
            name: "setup".into(),
            generics: vec![],
            params: vec![],
            topology: vxc::syntax::Topology::CPU,
            return_type: Type::Scalar(vxc::syntax::ElementType::Bool),
            requires: Vec::new(),
            ensures: Vec::new(),
            where_transfers: Vec::new(),
            // let c: Config = ...;
            body: vec![Statement::LetDecl(LetDeclStmt {
                name: "c".into(),
                is_mut: false,
                ty_ann: Some(Type::Struct("Config".into(), None)),
                expr: Expr::Number(NumberExpr {
                    value: "0.0".to_string().into(),
                    ty: Some(vxc::syntax::ElementType::F64),
                    span: Span::default(),
                }),
                span: Span::default(),
            })],
            doc_comment: None,
        }],
    };

    let symbol_map = build_symbol_map(&[module.clone()]);
    module.resolve_names(&symbol_map);

    if let Statement::LetDecl(LetDeclStmt {
        ty_ann: Some(Type::Struct(name, id)),
        ..
    }) = &module.functions[0].body[0]
    {
        if name.as_ref() != "Config" {
            return Err(format!("Assertion failed: {} != {}", name, "Config"));
        }
        if !(id.is_some()) {
            return Err("Assertion failed: id.is_some()".to_string());
        } // The Type annotation deep within the LetDecl Statement was resolved!
    } else {
        return Err("Expected LetDecl with Config Struct type".to_string());
    }

    Ok(())
}
