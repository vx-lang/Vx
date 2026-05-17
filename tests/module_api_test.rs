use vxc::parse_module;

#[test]
fn test_parse_module_api() {
    let source = "
        fn hello_world() -> Tensor {
            let x = 42;
        }
    ";
    let module = parse_module(source);
    assert!(module.is_ok(), "Parse failed: {:?}", module.err());
    let module = module.unwrap();
    assert_eq!(module.functions.len(), 1);
    assert_eq!(module.functions[0].name, "hello_world");
}

use vxc::ast::{VxFunction, VxModule};

#[test]
fn test_ak_module_add_function() {
    let mut module = VxModule {
        imports: Vec::new(),
        module_path: "core::test".to_string(),
        externs: vec![],
        structs: vec![],
        enums: vec![],
        traits: vec![],
        impls: vec![],
        functions: vec![],
    };

    // The 'pub' keyword is automatically stripped by our From<&str> implementation
    module.add(VxFunction::from("pub fn foo() -> i64 { return 10; }"));

    assert_eq!(module.functions.len(), 1);
    assert_eq!(module.functions[0].name, "foo");
}
