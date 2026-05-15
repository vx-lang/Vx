use vxc::parse_module;

#[test]
fn test_parse_module_api() {
    let source = "
        fn hello_world() {
            let x = 42;
        }
    ";
    let module = parse_module(source);
    assert!(module.is_ok());
    let module = module.unwrap();
    assert_eq!(module.functions.len(), 1);
    assert_eq!(module.functions[0].name, "hello_world");
}
